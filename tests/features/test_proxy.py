"""Tests for the Flask proxy that fronts the vLLM backend."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Awaitable, Callable
from uuid import uuid4

import httpx
import pytest
from flask import Flask, request

from app.cache.service import SemanticCacheService, SemanticCacheSettings
from app.cache.types import (
    CacheDecisionStatus,
    CacheEntry,
    CacheEntryCreate,
    CacheHit,
    CacheLookupResult,
    CacheQuery,
    FeedbackEventCreate,
    TTLDecisionCreate,
)
from app.config import Settings
from app.proxy.service import CacheContext, ProxyService
from app.runtime import BackgroundAsyncRunner


class _FakeEmbeddingService:
    """Deterministic embedding stub for semantic cache tests."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        return [[1.0] * self._dimension for _ in inputs]


class _InMemoryVectorStore:
    """Minimal vector store stub that satisfies the async interface."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}

    async def ensure_collection(self) -> None:  # pragma: no cover - no-op
        return None

    async def upsert_point(
        self,
        point_id: Any,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        self._points[str(point_id)] = dict(payload)

    async def search(
        self,
        *,
        vector: list[float],
        model: str,
        params_fingerprint: str,
        not_before,
        limit: int,
    ) -> list[Any]:
        results: list[Any] = []
        for payload in self._points.values():
            if (
                payload.get("model") == model
                and payload.get("params_fingerprint") == params_fingerprint
                and payload.get("expires_at_ts", 0.0) >= not_before.timestamp()
            ):
                results.append(
                    SimpleNamespace(
                        id=payload["cache_entry_id"],
                        score=0.99,
                        payload=payload,
                    )
                )
        return results[:limit]


class _InMemoryRepository:
    """In-memory repository stub for semantic cache persistence."""

    def __init__(self) -> None:
        self.entries: dict[str, CacheEntry] = {}
        self.ttl_decisions: list[TTLDecisionCreate] = []
        self.feedback_events: list[FeedbackEventCreate] = []

    async def create_schema(self) -> None:  # pragma: no cover - no-op
        return None

    async def insert_entry(self, payload: CacheEntryCreate) -> None:
        entry = CacheEntry(
            id=payload.id,
            model=payload.model,
            parameters=dict(payload.parameters),
            prompt_text=payload.prompt_text,
            response_payload=dict(payload.response_payload),
            ttl_bucket=payload.ttl_bucket,
            ttl_seconds=payload.ttl_seconds,
            created_at=payload.created_at,
            expires_at=payload.expires_at,
            updated_at=payload.created_at,
            hit_count=0,
            embedding_model=payload.embedding_model,
            embedding_mode=payload.embedding_mode,
            embedding_dimension=payload.embedding_dimension,
            similarity_threshold=payload.similarity_threshold,
        )
        self.entries[str(entry.id)] = entry

    async def get_entry(self, entry_id):  # pragma: no cover - unused
        return self.entries.get(str(entry_id))

    async def mark_hit(self, entry_id, accessed_at: datetime) -> None:
        entry = self.entries[str(entry_id)]
        self.entries[str(entry_id)] = replace(
            entry,
            hit_count=entry.hit_count + 1,
            updated_at=accessed_at,
        )

    async def insert_ttl_decision(self, payload: TTLDecisionCreate) -> None:
        self.ttl_decisions.append(payload)

    async def update_entry_ttl(
        self,
        entry_id,
        *,
        ttl_bucket: int,
        ttl_seconds: int,
        expires_at: datetime,
    ) -> None:  # pragma: no cover - unused
        entry = self.entries[str(entry_id)]
        self.entries[str(entry_id)] = replace(
            entry,
            ttl_bucket=ttl_bucket,
            ttl_seconds=ttl_seconds,
            expires_at=expires_at,
            updated_at=expires_at,
        )

    async def insert_feedback_event(self, payload: FeedbackEventCreate) -> None:
        self.feedback_events.append(payload)

    async def insert_policy_reward(self, payload) -> None:  # pragma: no cover - unused
        return None

    async def attach_entry_to_decision(self, decision_id, cache_entry_id) -> None:  # pragma: no cover - unused
        return None

    async def get_ttl_decision(self, decision_id):  # pragma: no cover - unused
        for record in self.ttl_decisions:
            if record.id == decision_id:
                return record
        return None

    async def fetch_prompt_stale_rates(self, prompt_hashes, *, lookback=timedelta(days=14)):
        return {}


class _Stream(httpx.SyncByteStream):
    """Synchronous byte stream for testing SSE responses."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None


def _make_proxy(
    *,
    settings: Settings,
    backend: Callable[[httpx.Request], httpx.Response],
    semantic_cache: SemanticCacheService | None = None,
    async_runner: Callable[[Awaitable[Any]], Any] | None = None,
) -> ProxyService:
    client = httpx.Client(transport=httpx.MockTransport(backend))
    return ProxyService(
        settings=settings,
        client=client,
        semantic_cache=semantic_cache,
        async_runner=async_runner,
    )


def _invoke_proxy(
    service: ProxyService,
    *,
    path: str,
    method: str = "POST",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    app = Flask(__name__)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    content_type = "application/json" if payload is not None else None
    with app.test_request_context(
        f"/v1/{path}",
        method=method,
        data=data,
        content_type=content_type,
        headers=headers,
    ):
        response = service.forward(path=path, flask_request=request)
    return response


def test_proxy_forwards_json_request() -> None:
    recorded: dict[str, Any] = {}

    def backend(request: httpx.Request) -> httpx.Response:
        recorded["method"] = request.method
        recorded["url"] = str(request.url)
        recorded["headers"] = dict(request.headers)
        recorded["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"id": "req-123", "object": "chat.completion"}, headers={"x-backend": "1"})

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        vllm_api_key="backend-secret",
        inbound_api_keys=["client-secret"],
        semantic_cache_enabled=False,
    )
    service = _make_proxy(settings=settings, backend=backend)

    response = _invoke_proxy(
        service,
        path="chat/completions",
        payload={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "ping"}],
        },
        headers={"Authorization": "Bearer client-secret"},
    )

    assert response.status_code == 200
    assert response.headers["x-backend"] == "1"
    payload = json.loads(response.get_data(as_text=True))
    assert payload["id"] == "req-123"
    assert response.headers["X-Cache"] == "MISS"
    assert payload["ttl_proxy"]["status"] == "miss"
    assert "semantic_cache_disabled" in payload["ttl_proxy"]["reasons"]
    assert recorded["method"] == "POST"
    assert recorded["url"] == "http://backend.local/v1/chat/completions"
    assert recorded["headers"]["authorization"] == "Bearer backend-secret"
    assert recorded["body"]["model"] == "gpt-test"


def test_proxy_streams_event_data() -> None:
    def backend(request: httpx.Request) -> httpx.Response:
        stream = _Stream([b"data: chunk-1\n\n", b"data: chunk-2\n\n", b"data: [DONE]\n\n"])
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    settings = Settings(vllm_base_url="http://backend.local/v1/", semantic_cache_enabled=False)
    service = _make_proxy(settings=settings, backend=backend)

    app = Flask(__name__)
    with app.test_request_context(
        "/v1/chat/completions",
        method="POST",
        data=json.dumps({"model": "gpt-test", "stream": True}),
        content_type="application/json",
    ):
        response = service.forward(path="chat/completions", flask_request=request)
        chunks = list(response.response)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["X-Cache"] == "MISS"
    assert b"".join(chunks) == b"data: chunk-1\n\ndata: chunk-2\n\ndata: [DONE]\n\n"


def test_missing_authentication_rejected() -> None:
    def backend(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Backend should not be called on auth failure.")

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        inbound_api_keys=["client-secret"],
        semantic_cache_enabled=False,
    )
    service = _make_proxy(settings=settings, backend=backend)

    response = _invoke_proxy(
        service,
        path="chat/completions",
        payload={"model": "gpt-test"},
    )

    assert response.status_code == 401
    payload = json.loads(response.get_data(as_text=True))
    assert payload["error"] == "Missing bearer token."


def test_proxy_caches_and_logs_decision() -> None:
    repository = _InMemoryRepository()
    vector_store = _InMemoryVectorStore()
    embedder = _FakeEmbeddingService(dimension=4)
    cache_settings = SemanticCacheSettings(
        similarity_threshold=0.0,
        search_limit=5,
        ttl_seconds=(60, 120, 300),
        default_ttl_bucket=1,
        policy_feature_dimension=16,
        policy_autosave_interval=1,
    )
    semantic_cache = SemanticCacheService(
        repository=repository,
        vector_store=vector_store,
        embedder=embedder,
        settings=cache_settings,
        embedding_model_name="test-embed",
        embedding_mode="clustering",
        embedding_dimension=4,
    )
    runner = BackgroundAsyncRunner()
    decision = None
    try:
        runner.run(semantic_cache.bootstrap())

        backend_calls = {"count": 0}

        def backend(request: httpx.Request) -> httpx.Response:
            backend_calls["count"] += 1
            return httpx.Response(
                200,
                json={
                    "id": "resp-001",
                    "object": "chat.completion",
                    "choices": [
                        {"message": {"role": "assistant", "content": "Hello"}}
                    ],
                },
                request=request,
            )

        settings = Settings(
            vllm_base_url="http://backend.local/v1/",
            inbound_api_keys=[],
            semantic_cache_enabled=True,
        )
        service = _make_proxy(
            settings=settings,
            backend=backend,
            semantic_cache=semantic_cache,
            async_runner=runner.run,
        )

        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "Breaking news about markets"}
            ],
        }

        first_response = _invoke_proxy(service, path="chat/completions", payload=payload)
        assert first_response.status_code == 200
        assert backend_calls["count"] == 1

        second_response = _invoke_proxy(service, path="chat/completions", payload=payload)
        assert second_response.status_code == 200
        assert second_response.headers["X-Cache"] == "HIT"
        assert backend_calls["count"] == 1

        assert repository.ttl_decisions, "expected TTL decision to be recorded"
        decision = repository.ttl_decisions[0]
        assert decision.policy_name == "linucb"
        assert decision.ttl_bucket in {0, 1, 2}
    finally:
        runner.close()
    assert decision is not None
    assert decision.propensity is not None and 0.0 < decision.propensity <= 1.0

    assert repository.feedback_events
    assert repository.feedback_events[0].event_type == "cache_hit_served"


def test_proxy_logs_cache_decision(caplog) -> None:
    settings = Settings(vllm_base_url="http://backend.local/v1/", semantic_cache_enabled=False)
    service = ProxyService(settings=settings, client=httpx.Client())

    query = CacheQuery(
        request_path="/v1/chat/completions",
        model="gpt-test",
        prompt_text="Hello world",
        embedding_input="Hello world",
        request_fingerprint="rfp",
        params_fingerprint="pfp",
        prompt_hash="hash",
        parameters={},
    )

    now = datetime.now(UTC)
    entry = CacheEntry(
        id=uuid4(),
        model="gpt-test",
        parameters={},
        prompt_text="Hello world",
        response_payload={"choices": []},
        ttl_bucket=1,
        ttl_seconds=60,
        created_at=now,
        expires_at=now + timedelta(seconds=60),
        updated_at=now,
        hit_count=3,
        embedding_model="test-embed",
        embedding_mode="clustering",
        embedding_dimension=4,
        similarity_threshold=0.5,
    )

    cache_context = CacheContext(
        request_path="/v1/chat/completions",
        request_method="POST",
        enabled=True,
        status=CacheDecisionStatus.HIT,
    )
    cache_context.query = query
    cache_context.ttl_bucket = entry.ttl_bucket
    cache_context.policy_propensity = 0.4321
    cache_context.cache_entry_id = entry.id
    cache_context.ttl_decision_id = uuid4()
    cache_context.reasons.append("cache_hit_served")
    cache_context.backend_status_code = 200
    cache_context.result = CacheLookupResult(
        status=CacheDecisionStatus.HIT,
        query=query,
        neighbor_stale_rate=0.1234,
        hit=CacheHit(entry=entry, similarity=0.9876),
        reasons=("cache_hit_served",),
    )

    with caplog.at_level(logging.INFO, logger="app.cache"):
        service._log_cache_decision(cache_context, latency_ms=12.34)

    records = [
        record
        for record in caplog.records
        if getattr(record, "kv", {}).get("event") == "proxy.cache_decision"
    ]
    assert records
    kv = records[0].kv
    assert kv["method"] == "POST"
    assert kv["path"] == "/v1/chat/completions"
    assert kv["status"] == "hit"
    assert kv["reasons"] == "cache_hit_served"
