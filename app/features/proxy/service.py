"""Core proxy logic for forwarding OpenAI-compatible requests to vLLM."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from urllib.parse import urljoin
from uuid import UUID

import httpx
from asyncpg.exceptions import UniqueViolationError
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.features.semantic_cache.policy import (
    ExtractionContext,
    FeatureExtractor,
    FeatureVector,
    build_default_entity_recognizer,
)
from app.features.semantic_cache.service import SemanticCacheService
from app.features.semantic_cache.types import (
    CacheDecisionStatus,
    CacheEntry,
    CacheLookupResult,
    CacheQuery,
)

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

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
    """Forward requests to the configured OpenAI-compatible backend."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        *,
        semantic_cache: SemanticCacheService | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._semantic_cache = semantic_cache
        self._feature_extractor = feature_extractor or FeatureExtractor(
            entity_recognizer=build_default_entity_recognizer()
        )

    async def forward(self, path: str, request: Request) -> Response:
        """Proxy the incoming request to the backend and return its response."""
        inbound_token = self._authorize(request)
        prepared_headers = self._prepare_headers(request, inbound_token)
        query_params = list(request.query_params.multi_items())
        body = await request.body()
        backend_url = self._build_backend_url(path)

        cache_context = CacheContext(
            request_path=path,
            request_method=request.method,
            enabled=self._semantic_cache is not None,
            status=CacheDecisionStatus.MISS,
        )
        setattr(request.state, "cache_context", cache_context)
        start_time = perf_counter()

        try:
            if self._semantic_cache:
                cache_query = self._maybe_prepare_cache_query(path, request, body)
                if cache_query is None:
                    cache_context.add_reason("not_eligible")
                else:
                    cache_context.query = cache_query
                    cache_result = await self._semantic_cache.lookup(cache_query)
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
                        bucket, propensity = await self._semantic_cache.select_ttl_bucket(  # type: ignore[union-attr]
                            feature_vector.as_dict()
                        )
                        cache_context.ttl_bucket = bucket
                        cache_context.policy_propensity = propensity
                    if cache_result.hit is not None:
                        cache_context.ttl_bucket = cache_result.hit.entry.ttl_bucket
                    elif cache_result.stale_entry is not None:
                        cache_context.ttl_bucket = cache_result.stale_entry.ttl_bucket
                    if (
                        cache_result.status is CacheDecisionStatus.HIT
                        and cache_result.hit
                    ):
                        cache_context.cache_entry_id = cache_result.hit.entry.id
                        response = self._as_cache_response(cache_result, cache_context)
                        await self._log_feedback_event(
                            entry_id=cache_result.hit.entry.id,
                            query=cache_query,
                            event_type="cache_hit_served",
                            score=1.0,
                            details={
                                "request_fingerprint": cache_query.request_fingerprint,
                                "prompt_hash": cache_query.prompt_hash,
                                "similarity": cache_result.hit.similarity,
                            },
                        )
                        cache_context.backend_status_code = status.HTTP_200_OK
                        latency_ms = (perf_counter() - start_time) * 1000
                        self._log_cache_decision(cache_context, latency_ms)
                        return response
                    if (
                        cache_result.status is CacheDecisionStatus.STALE_HIT
                        and cache_result.stale_entry is not None
                    ):
                        cache_context.cache_entry_id = cache_result.stale_entry.id
                        await self._log_feedback_event(
                            entry_id=cache_result.stale_entry.id,
                            query=cache_query,
                            event_type="cache_stale_hit",
                            score=0.0,
                            details={
                                "request_fingerprint": cache_query.request_fingerprint,
                                "prompt_hash": cache_query.prompt_hash,
                                "similarity": cache_result.stale_similarity,
                            },
                        )
            else:
                cache_context.add_reason("semantic_cache_disabled")

            httpx_request = self._client.build_request(
                method=request.method,
                url=backend_url,
                params=query_params,
                headers=prepared_headers,
                content=body if body else None,
            )

            backend_response = await self._client.send(httpx_request, stream=True)
            try:
                cache_context.backend_status_code = backend_response.status_code
                if _is_streaming_response(backend_response):
                    response = self._as_streaming_response(backend_response)
                    cache_header = self._infer_cache_header(cache_context)
                    response.headers.setdefault("X-Cache", cache_header)
                    response.headers["X-Cache-Status"] = cache_context.status.value
                    latency_ms = (perf_counter() - start_time) * 1000
                    self._log_cache_decision(cache_context, latency_ms)
                    return response
                response = await self._as_standard_response(
                    backend_response,
                    cache_context=cache_context,
                )
                latency_ms = (perf_counter() - start_time) * 1000
                self._log_cache_decision(cache_context, latency_ms)
                return response
            except Exception:
                await backend_response.aclose()
                raise
        finally:
            if hasattr(request.state, "cache_context"):
                delattr(request.state, "cache_context")

    def _authorize(self, request: Request) -> str | None:
        """Validate inbound API key if configured."""
        if not self._settings.inbound_api_keys:
            return None

        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token.",
            )

        token = auth_header.split(" ", 1)[1].strip()
        if token not in self._settings.inbound_api_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key.",
            )
        return token

    def _prepare_headers(
        self, request: Request, inbound_token: str | None
    ) -> dict[str, str]:
        """Prepare the headers to be forwarded to the backend service."""
        headers: dict[str, str] = {}
        for name, value in request.headers.items():
            lower_name = name.lower()
            if lower_name in HOP_BY_HOP_HEADERS:
                continue
            if lower_name == "authorization":
                if self._settings.inbound_api_keys:
                    # Replace client auth with backend auth if configured.
                    continue
            headers[name] = value

        if self._settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.vllm_api_key}"
        elif inbound_token and not self._settings.vllm_api_key:
            # We already validated the inbound token; propagate it if we are not
            # overriding with a backend key.
            headers["Authorization"] = f"Bearer {inbound_token}"

        return headers

    def _build_backend_url(self, path: str) -> str:
        """Construct the full backend URL for the given path."""
        base_url = str(self._settings.vllm_base_url).rstrip("/") + "/"
        cleaned_path = path.lstrip("/")
        return urljoin(base_url, cleaned_path)

    async def _as_standard_response(
        self,
        backend_response: httpx.Response,
        *,
        cache_context: CacheContext,
    ) -> Response:
        """Materialize a non-streaming response."""
        content = await backend_response.aread()
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
            and backend_response.status_code == status.HTTP_200_OK
        ):
            stored_entry = await self._maybe_store_response(
                cache_context.query,
                backend_response,
                raw_payload,
                cache_context=cache_context,
            )
        elif backend_response.status_code != status.HTTP_200_OK:
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

        await backend_response.aclose()

        return Response(
            content=content,
            status_code=backend_response.status_code,
            headers=headers,
            media_type=headers.get("content-type"),
        )

    def _maybe_prepare_cache_query(
        self, path: str, request: Request, body: bytes
    ) -> CacheQuery | None:
        """Build a cache query when request conditions are satisfied."""
        if request.method.upper() != "POST":
            return None
        content_type = request.headers.get("content-type", "")
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
        self, decision: CacheLookupResult, cache_context: CacheContext
    ) -> Response:
        """Return a FastAPI response using cached payload."""
        if decision.hit is None:
            raise ValueError("Cache hit expected for cache response.")

        payload = copy.deepcopy(decision.hit.entry.response_payload)
        payload["ttl_proxy"] = self._build_metadata(cache_context)

        response = JSONResponse(content=payload)
        cache_header = self._infer_cache_header(cache_context)
        response.headers["X-Cache"] = cache_header
        response.headers["X-Cache-Status"] = cache_context.status.value
        response.headers["X-Cache-Similarity"] = f"{decision.hit.similarity:.4f}"
        response.headers["X-Cache-TTL-Bucket"] = str(decision.hit.entry.ttl_bucket)
        return response

    async def _maybe_store_response(
        self,
        query: CacheQuery,
        backend_response: httpx.Response,
        payload: dict[str, Any] | None,
        *,
        cache_context: CacheContext,
    ) -> CacheEntry | None:
        """Store a backend response in the cache when eligible."""
        if backend_response.status_code != status.HTTP_200_OK:
            return None
        if payload is None:
            return None
        try:
            stored_entry = await self._semantic_cache.store(  # type: ignore[union-attr]
                query,
                payload,
                ttl_bucket=cache_context.ttl_bucket,
            )
        except IntegrityError as exc:  # pragma: no cover - defensive fallback
            if isinstance(getattr(exc, "orig", None), UniqueViolationError):
                cache_context.add_reason("duplicate_request_fingerprint")
                logger.info(
                    "Cache store skipped duplicate request fingerprint=%s",
                    query.request_fingerprint,
                )
            else:
                cache_context.add_reason("store_integrity_error")
                logger.error(
                    "Cache store integrity error (%s): %s",
                    exc.__class__.__name__,
                    exc,
                )
            return None
        except Exception as exc:  # pragma: no cover - defensive fallback
            cache_context.add_reason("store_error")
            logger.error(
                "Cache store failed (%s): %s",
                exc.__class__.__name__,
                exc,
            )
            return None
        neighbor_rate = (
            cache_context.result.neighbor_stale_rate if cache_context.result else None
        )
        cache_context.ttl_bucket = stored_entry.ttl_bucket
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

    def _as_streaming_response(
        self, backend_response: httpx.Response
    ) -> StreamingResponse:
        """Return a streaming response that relays backend chunks."""
        headers = self._filtered_response_headers(backend_response.headers)
        iterator = self._stream_iterator(backend_response)
        return StreamingResponse(
            iterator,
            status_code=backend_response.status_code,
            headers=headers,
            media_type=headers.get("content-type"),
        )

    async def _log_ttl_decision(
        self,
        *,
        query: CacheQuery,
        entry: CacheEntry,
        neighbor_stale_rate: float | None = None,
        feature_vector: FeatureVector | None = None,
        propensity: float | None = None,
    ) -> UUID | None:
        """Persist the TTL decision generated for a cache miss."""
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
        return decision_id

    async def _log_feedback_event(
        self,
        *,
        entry_id: UUID,
        query: CacheQuery,
        event_type: str,
        score: float,
        details: dict[str, Any],
    ) -> None:
        """Record cache feedback signals for policy learning."""
        if self._semantic_cache is None:
            return
        await self._semantic_cache.log_feedback_event(
            cache_entry_id=entry_id,
            event_type=event_type,
            score=score,
            details=details,
            ttl_decision_id=None,
        )

    @staticmethod
    def _normalize_route(path: str) -> str:
        """Normalize request paths into feature-friendly tokens."""
        cleaned = path.strip().lower().lstrip("/")
        return cleaned.replace("/", "_") or "root"

    def _infer_cache_header(self, context: CacheContext) -> str:
        """Translate cache status into an HTTP header value."""
        if context.status is CacheDecisionStatus.HIT:
            return "HIT"
        if context.status is CacheDecisionStatus.STALE_HIT:
            return "STALE"
        if not context.enabled or context.query is None:
            return "BYPASS"
        return "MISS"

    def _build_metadata(self, context: CacheContext) -> dict[str, Any]:
        """Construct the metadata block attached to JSON responses."""
        metadata: dict[str, Any] = {
            "cache_status": context.status.value,
            "semantic_cache_enabled": context.enabled,
        }

        if context.query is not None:
            metadata["request"] = {
                "path": context.query.request_path,
                "model": context.query.model,
                "request_fingerprint": context.query.request_fingerprint,
                "params_fingerprint": context.query.params_fingerprint,
                "prompt_hash": context.query.prompt_hash,
            }
        else:
            metadata["request"] = {
                "path": context.request_path,
                "cacheable": False,
            }

        if context.result and context.result.hit is not None:
            entry = context.result.hit.entry
            metadata["cache"] = {
                "source": "cache",
                "entry_id": str(entry.id),
                "similarity": context.result.hit.similarity,
                "ttl_bucket": entry.ttl_bucket,
                "ttl_seconds": entry.ttl_seconds,
                "hit_count": entry.hit_count,
                "created_at": entry.created_at.isoformat(),
                "updated_at": entry.updated_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "embedding_model": entry.embedding_model,
                "embedding_mode": entry.embedding_mode,
                "embedding_dimension": entry.embedding_dimension,
            }
        elif (
            context.result
            and context.status is CacheDecisionStatus.STALE_HIT
            and context.result.stale_entry is not None
        ):
            entry = context.result.stale_entry
            metadata["cache"] = {
                "source": "stale_candidate",
                "entry_id": str(entry.id),
                "ttl_bucket": entry.ttl_bucket,
                "ttl_seconds": entry.ttl_seconds,
                "expired_at": entry.expires_at.isoformat(),
                "created_at": entry.created_at.isoformat(),
                "stale_similarity": context.result.stale_similarity,
            }
        else:
            metadata["cache"] = {
                "source": "backend",
                "stored": bool(context.stored) if context.stored is not None else None,
                "entry_id": (
                    str(context.cache_entry_id) if context.cache_entry_id else None
                ),
            }

        if context.reasons:
            metadata["reasons"] = list(context.reasons)

        metadata["decision"] = {
            "ttl_bucket": context.ttl_bucket,
            "policy_name": POLICY_NAME
            if context.policy_propensity is not None
            else None,
            "policy_version": POLICY_VERSION
            if context.policy_propensity is not None
            else None,
            "propensity": context.policy_propensity,
            "ttl_decision_id": (
                str(context.ttl_decision_id) if context.ttl_decision_id else None
            ),
        }

        return metadata

    def _log_cache_decision(self, context: CacheContext, latency_ms: float) -> None:
        """Emit a structured log message for cache decisions."""
        path = context.query.request_path if context.query else context.request_path
        model = context.query.model if context.query else "unknown"

        bucket_value = context.ttl_bucket if context.ttl_bucket is not None else "n/a"

        parts: list[str] = [
            "proxy decision",
            f"method={context.request_method}",
            f"path={path or '/'}",
            f"model={model}",
            f"decision={context.status.value}",
            f"ttl_bucket={bucket_value}",
            f"latency_ms={latency_ms:.2f}",
        ]

        if context.backend_status_code is not None:
            parts.append(f"status_code={context.backend_status_code}")
        if context.stored is not None:
            parts.append(f"stored={context.stored}")
        parts.append(f"cache_enabled={context.enabled}")
        if context.cache_entry_id is not None:
            parts.append(f"cache_entry_id={context.cache_entry_id}")
        if context.ttl_decision_id is not None:
            parts.append(f"ttl_decision_id={context.ttl_decision_id}")
        if context.policy_propensity is not None:
            parts.append(f"propensity={context.policy_propensity:.4f}")

        if (
            context.status is CacheDecisionStatus.HIT
            and context.result
            and context.result.hit is not None
        ):
            parts.append(f"similarity={context.result.hit.similarity:.4f}")
            parts.append(f"hit_count={context.result.hit.entry.hit_count}")
        elif (
            context.status is CacheDecisionStatus.STALE_HIT
            and context.result
            and context.result.stale_entry is not None
        ):
            parts.append(
                f"stale_similarity={context.result.stale_similarity or 0.0:.4f}"
            )
            parts.append(
                f"stale_expires_at={context.result.stale_entry.expires_at.isoformat()}"
            )

        if context.reasons:
            parts.append(f"reasons={','.join(context.reasons)}")

        logger.info(" ".join(parts))

    def _filtered_response_headers(self, headers: httpx.Headers) -> dict[str, str]:
        """Filter hop-by-hop headers from backend responses."""
        filtered: dict[str, str] = {}
        for name, value in headers.items():
            if name.lower() in HOP_BY_HOP_HEADERS:
                continue
            filtered[name] = value
        return filtered

    def _stream_iterator(
        self, backend_response: httpx.Response
    ) -> AsyncIterator[bytes]:
        """Yield raw bytes from the backend streaming response."""

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in backend_response.aiter_raw():
                    yield chunk
            finally:
                await backend_response.aclose()

        return iterator()


def _is_streaming_response(response: httpx.Response) -> bool:
    """Detect if the backend response should be streamed."""
    content_type = response.headers.get("content-type", "")
    return content_type.startswith("text/event-stream")
