"""Proxy service that fronts vLLM using Flask request objects."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Awaitable, Callable, Iterable
from urllib.parse import urljoin
from uuid import UUID

import httpx
from flask import Request, Response, stream_with_context

from app.cache.service import SemanticCacheService
from app.cache.types import (
    CacheDecisionStatus,
    CacheEntry,
    CacheLookupResult,
    CacheQuery,
)
from app.config import Settings
from app.logging import REQUEST_ID_VAR, log_cache_event, log_policy_event
from app.policy import (
    ExtractionContext,
    FeatureExtractor,
    FeatureVector,
    build_default_entity_recognizer,
)

LOG = logging.getLogger("app.proxy")

POLICY_NAME = "linucb"
POLICY_VERSION = "0.1.0"

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


class ProxyError(Exception):
    """Error raised for user-facing proxy issues."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(slots=True)
class CacheContext:
    """Mutable telemetry collected while servicing a request."""

    request_path: str
    request_method: str
    enabled: bool
    status: CacheDecisionStatus
    query: CacheQuery | None = None
    result: CacheLookupResult | None = None
    reasons: list[str] = field(default_factory=list)
    stored: bool | None = None
    backend_status_code: int | None = None
    ttl_bucket: int | None = None
    feature_vector: FeatureVector | None = None
    policy_propensity: float | None = None
    cache_entry_id: UUID | None = None
    ttl_decision_id: UUID | None = None

    def add_reason(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def extend_reasons(self, new_reasons: Iterable[str]) -> None:
        for reason in new_reasons:
            self.add_reason(reason)


class ProxyService:
    """Forward OpenAI-compatible requests to the configured backend."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: httpx.Client,
        semantic_cache: SemanticCacheService | None = None,
        feature_extractor: FeatureExtractor | None = None,
        async_runner: Callable[[Awaitable[Any]], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._semantic_cache = semantic_cache
        self._feature_extractor = feature_extractor or FeatureExtractor(
            entity_recognizer=build_default_entity_recognizer()
        )
        self._run_async = async_runner or (lambda coro: asyncio.run(coro))

    def forward(self, path: str, flask_request: Request) -> Response:
        """Forward a Flask request to the backend and return its response."""
        try:
            return self._forward(path, flask_request)
        except ProxyError as exc:
            payload = json.dumps({"error": exc.message}).encode("utf-8")
            headers = {"content-type": "application/json"}
            return Response(payload, status=exc.status_code, headers=headers)
        except httpx.RequestError as exc:
            LOG.error(
                "proxy.backend_error",
                extra={
                    "kv": {
                        "event": "proxy.backend_error",
                        "error": str(exc),
                        "request_id": REQUEST_ID_VAR.get("-"),
                    }
                },
            )
            payload = json.dumps({"error": "Upstream service unavailable"}).encode(
                "utf-8"
            )
            return Response(payload, status=502, headers={"content-type": "application/json"})

    def _forward(self, path: str, flask_request: Request) -> Response:
        inbound_token = self._authorize(flask_request)
        prepared_headers = self._prepare_headers(flask_request, inbound_token)
        request_id = REQUEST_ID_VAR.get("-")
        if request_id and request_id != "-":
            prepared_headers["X-Request-ID"] = request_id

        query_params = list(flask_request.args.lists())
        body = flask_request.get_data(cache=True)
        backend_url = self._build_backend_url(path)

        cache_context = CacheContext(
            request_path=path,
            request_method=flask_request.method,
            enabled=self._semantic_cache is not None,
            status=CacheDecisionStatus.MISS,
        )
        start_time = perf_counter()

        if self._semantic_cache:
            cache_query = self._maybe_prepare_cache_query(path, flask_request, body)
            if cache_query is None:
                cache_context.add_reason("not_eligible")
            else:
                cache_context.query = cache_query
                cache_result = self._run_async(
                    self._semantic_cache.lookup(cache_query)  # type: ignore[arg-type]
                )
                cache_context.result = cache_result
                cache_context.status = cache_result.status
                cache_context.extend_reasons(cache_result.reasons)
                if cache_result.status is CacheDecisionStatus.MISS:
                    feature_context = ExtractionContext(
                        prompt_hash=cache_query.prompt_hash,
                        route=self._normalize_route(cache_query.request_path),
                        nearest_neighbor_stale_rate=cache_result.neighbor_stale_rate,
                    )
                    feature_vector = self._feature_extractor.extract(
                        cache_query.prompt_text,
                        context=feature_context,
                    )
                    cache_context.feature_vector = feature_vector
                    bucket, propensity = self._run_async(
                        self._semantic_cache.select_ttl_bucket(  # type: ignore[attr-defined]
                            feature_vector.as_dict()
                        )
                    )
                    cache_context.ttl_bucket = bucket
                    cache_context.policy_propensity = propensity
                    log_policy_event(
                        "policy.ttl_select",
                        bucket=bucket,
                        propensity=propensity,
                        request_id=request_id,
                    )
                if cache_result.hit is not None:
                    cache_context.ttl_bucket = cache_result.hit.entry.ttl_bucket
                elif cache_result.stale_entry is not None:
                    cache_context.ttl_bucket = cache_result.stale_entry.ttl_bucket
                if (
                    cache_result.status is CacheDecisionStatus.HIT
                    and cache_result.hit
                ):
                    cache_context.cache_entry_id = cache_result.hit.entry.id
                    self._run_async(
                        self._log_feedback_event(
                            entry_id=cache_result.hit.entry.id,
                            event_type="cache_hit_served",
                            score=1.0,
                            details={
                                "request_fingerprint": cache_query.request_fingerprint,
                                "prompt_hash": cache_query.prompt_hash,
                                "similarity": cache_result.hit.similarity,
                            },
                            ttl_decision_id=cache_context.ttl_decision_id,
                        )
                    )
                    response = self._as_cache_response(cache_result, cache_context)
                    latency_ms = (perf_counter() - start_time) * 1000
                    self._log_cache_decision(cache_context, latency_ms)
                    return response
                if (
                    cache_result.status is CacheDecisionStatus.STALE_HIT
                    and cache_result.stale_entry is not None
                ):
                    cache_context.cache_entry_id = cache_result.stale_entry.id
                    self._run_async(
                        self._log_feedback_event(
                            entry_id=cache_result.stale_entry.id,
                            event_type="cache_stale_hit",
                            score=0.0,
                            details={
                                "request_fingerprint": cache_query.request_fingerprint,
                                "prompt_hash": cache_query.prompt_hash,
                                "similarity": cache_result.stale_similarity,
                            },
                            ttl_decision_id=cache_context.ttl_decision_id,
                        )
                    )
        else:
            cache_context.add_reason("semantic_cache_disabled")

        stream_ctx = self._client.stream(
            method=flask_request.method,
            url=backend_url,
            params=query_params,
            headers=prepared_headers,
            content=body if body else None,
        )
        backend_response = stream_ctx.__enter__()
        try:
            cache_context.backend_status_code = backend_response.status_code
            if _is_streaming_response(backend_response):
                response = self._as_streaming_response(
                    backend_response, stream_ctx, cache_context
                )
                latency_ms = (perf_counter() - start_time) * 1000
                self._log_cache_decision(cache_context, latency_ms)
                return response
            response = self._as_standard_response(
                backend_response,
                cache_context=cache_context,
            )
            latency_ms = (perf_counter() - start_time) * 1000
            self._log_cache_decision(cache_context, latency_ms)
            return response
        finally:
            if not _is_streaming_response(backend_response):
                stream_ctx.__exit__(None, None, None)

    def _authorize(self, flask_request: Request) -> str | None:
        """Validate inbound API key if configured."""
        if not self._settings.inbound_api_keys:
            return None
        auth_header = flask_request.headers.get("Authorization")
        if not auth_header or not auth_header.lower().startswith("bearer "):
            raise ProxyError(401, "Missing bearer token.")
        token = auth_header.split(" ", 1)[1].strip()
        if token not in self._settings.inbound_api_keys:
            raise ProxyError(403, "Invalid API key.")
        return token

    def _prepare_headers(
        self,
        flask_request: Request,
        inbound_token: str | None,
    ) -> dict[str, str]:
        """Prepare the headers to be forwarded to the backend service."""
        headers: dict[str, str] = {}
        for name, value in flask_request.headers.items():
            lower = name.lower()
            if lower in HOP_BY_HOP_HEADERS:
                continue
            if lower == "authorization" and self._settings.inbound_api_keys:
                continue
            headers[name] = value
        if self._settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.vllm_api_key}"
        elif inbound_token and not self._settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {inbound_token}"
        return headers

    def _build_backend_url(self, path: str) -> str:
        base_url = str(self._settings.vllm_base_url).rstrip("/") + "/"
        cleaned_path = path.lstrip("/")
        return urljoin(base_url, cleaned_path)

    def _as_standard_response(
        self,
        backend_response: httpx.Response,
        *,
        cache_context: CacheContext,
    ) -> Response:
        content = backend_response.read()
        headers = self._filtered_response_headers(backend_response.headers)
        cache_header = self._infer_cache_header(cache_context)
        headers.setdefault("X-Cache", cache_header)
        headers["X-Cache-Status"] = cache_context.status.value

        raw_payload: dict[str, Any] | None = None
        client_payload: dict[str, Any] | None = None
        content_type = backend_response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                parsed = json.loads(content.decode("utf-8"))
            except json.JSONDecodeError:
                cache_context.add_reason("response_json_decode_error")
            else:
                if isinstance(parsed, dict):
                    raw_payload = parsed
                    client_payload = copy.deepcopy(parsed)
                else:
                    cache_context.add_reason("response_not_object")
        else:
            cache_context.add_reason("response_not_json")

        stored_entry: CacheEntry | None = None
        if (
            self._semantic_cache
            and cache_context.query is not None
            and raw_payload is not None
            and backend_response.status_code == 200
        ):
            stored_entry = self._run_async(
                self._maybe_store_response(
                    cache_context.query,
                    backend_response,
                    raw_payload,
                    cache_context=cache_context,
                )
            )
        elif backend_response.status_code != 200:
            cache_context.add_reason("non_success_status")
        cache_context.stored = (
            (stored_entry is not None) if self._semantic_cache else None
        )
        if stored_entry is not None:
            cache_context.ttl_bucket = stored_entry.ttl_bucket

        if client_payload is not None:
            metadata = self._build_metadata(cache_context)
            client_payload["ttl_proxy"] = metadata
            content = json.dumps(client_payload).encode("utf-8")
            headers["content-type"] = "application/json"

        return Response(
            content,
            status=backend_response.status_code,
            headers=headers,
            mimetype=headers.get("content-type"),
        )

    def _maybe_prepare_cache_query(
        self,
        path: str,
        flask_request: Request,
        body: bytes,
    ) -> CacheQuery | None:
        if flask_request.method.upper() != "POST":
            return None
        content_type = flask_request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return None
        if not body:
            return None
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return self._semantic_cache.normalize_request(path, payload)  # type: ignore[union-attr]

    def _as_cache_response(
        self,
        decision: CacheLookupResult,
        cache_context: CacheContext,
    ) -> Response:
        if decision.hit is None:
            raise ValueError("Cache hit expected for cache response.")
        payload = copy.deepcopy(decision.hit.entry.response_payload)
        payload["ttl_proxy"] = self._build_metadata(cache_context)
        content = json.dumps(payload).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "X-Cache": self._infer_cache_header(cache_context),
            "X-Cache-Status": cache_context.status.value,
            "X-Cache-Similarity": f"{decision.hit.similarity:.4f}",
            "X-Cache-TTL-Bucket": str(decision.hit.entry.ttl_bucket),
        }
        return Response(content, status=200, headers=headers)

    async def _maybe_store_response(
        self,
        query: CacheQuery,
        backend_response: httpx.Response,
        payload: dict[str, Any] | None,
        *,
        cache_context: CacheContext,
    ) -> CacheEntry | None:
        if backend_response.status_code != 200:
            return None
        if payload is None:
            return None
        try:
            stored_entry = await self._semantic_cache.store(  # type: ignore[union-attr]
                query,
                payload,
                ttl_bucket=cache_context.ttl_bucket,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOG.warning(
                "proxy.cache_store_error",
                extra={
                    "kv": {
                        "event": "proxy.cache_store_error",
                        "error": str(exc),
                    }
                },
            )
            return None
        neighbor_rate = (
            cache_context.result.neighbor_stale_rate if cache_context.result else None
        )
        cache_context.cache_entry_id = stored_entry.id
        decision_id = await self._log_ttl_decision(
            query=query,
            entry=stored_entry,
            neighbor_stale_rate=neighbor_rate,
            feature_vector=cache_context.feature_vector,
            propensity=cache_context.policy_propensity,
        )
        cache_context.ttl_decision_id = decision_id
        return stored_entry

    def _filtered_response_headers(
        self,
        backend_headers: httpx.Headers,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        for name, value in backend_headers.items():
            if name.lower() in HOP_BY_HOP_HEADERS:
                continue
            headers[name] = value
        return headers

    def _as_streaming_response(
        self,
        backend_response: httpx.Response,
        stream_ctx: AbstractContextManager[httpx.Response],
        cache_context: CacheContext,
    ) -> Response:
        headers = self._filtered_response_headers(backend_response.headers)
        headers.setdefault("X-Cache", self._infer_cache_header(cache_context))
        headers["X-Cache-Status"] = cache_context.status.value

        def generate() -> Iterable[bytes]:
            try:
                for chunk in backend_response.iter_raw():
                    yield chunk
            finally:
                stream_ctx.__exit__(None, None, None)

        return Response(
            stream_with_context(generate()),
            status=backend_response.status_code,
            headers=headers,
        )

    def _infer_cache_header(self, cache_context: CacheContext) -> str:
        if cache_context.status is CacheDecisionStatus.HIT:
            return "HIT"
        if cache_context.status is CacheDecisionStatus.STALE_HIT:
            return "STALE"
        if cache_context.status is CacheDecisionStatus.MISS and cache_context.stored:
            return "MISS-STORE"
        return "MISS"

    def _build_metadata(self, cache_context: CacheContext) -> dict[str, Any]:
        metadata = {
            "status": cache_context.status.value,
            "enabled": cache_context.enabled,
            "reasons": cache_context.reasons,
        }
        if cache_context.cache_entry_id is not None:
            metadata["cache_entry_id"] = str(cache_context.cache_entry_id)
        if cache_context.ttl_bucket is not None:
            metadata["ttl_bucket"] = cache_context.ttl_bucket
        if cache_context.policy_propensity is not None:
            metadata["policy_propensity"] = cache_context.policy_propensity
        if cache_context.ttl_decision_id is not None:
            metadata["ttl_decision_id"] = str(cache_context.ttl_decision_id)
        return metadata

    def _log_cache_decision(
        self,
        cache_context: CacheContext,
        latency_ms: float,
    ) -> None:
        request_id = REQUEST_ID_VAR.get("-")
        log_cache_event(
            "proxy.cache_decision",
            request_id=request_id,
            path=cache_context.request_path,
            method=cache_context.request_method,
            status=cache_context.status.value,
            stored=cache_context.stored,
            ttl_bucket=cache_context.ttl_bucket,
            latency_ms=latency_ms,
            reasons=",".join(cache_context.reasons),
        )

    def _normalize_route(self, route: str) -> str:
        lowered = route.lower().strip("/")
        if lowered.endswith("chat/completions"):
            return "/v1/chat/completions"
        if lowered.endswith("completions"):
            return "/v1/completions"
        return route

    async def _log_ttl_decision(
        self,
        *,
        query: CacheQuery,
        entry: CacheEntry,
        neighbor_stale_rate: float | None = None,
        feature_vector: FeatureVector | None = None,
        propensity: float | None = None,
    ) -> UUID | None:
        if self._semantic_cache is None:
            return None

        vector = feature_vector
        if vector is None:
            context = ExtractionContext(
                prompt_hash=query.prompt_hash,
                route=self._normalize_route(query.request_path),
                nearest_neighbor_stale_rate=neighbor_stale_rate,
            )
            vector = self._feature_extractor.extract(
                query.prompt_text,
                context=context,
            )

        decision_id = await self._semantic_cache.log_ttl_decision(
            query=query,
            entry=entry,
            policy_name=POLICY_NAME,
            policy_version=POLICY_VERSION,
            ttl_bucket=entry.ttl_bucket,
            features=vector.as_dict(),
            propensity=propensity,
        )
        log_policy_event(
            "policy.ttl_decision",
            decision_id=str(decision_id),
            bucket=entry.ttl_bucket,
            propensity=propensity,
        )
        return decision_id

    async def _log_feedback_event(
        self,
        *,
        entry_id: UUID,
        event_type: str,
        score: float,
        details: dict[str, Any],
        ttl_decision_id: UUID | None = None,
    ) -> None:
        if self._semantic_cache is None:
            return
        await self._semantic_cache.log_feedback_event(
            cache_entry_id=entry_id,
            event_type=event_type,
            score=score,
            details=details,
            ttl_decision_id=ttl_decision_id,
        )


def _is_streaming_response(backend_response: httpx.Response) -> bool:
    content_type = backend_response.headers.get("content-type", "")
    return "text/event-stream" in content_type.lower()
