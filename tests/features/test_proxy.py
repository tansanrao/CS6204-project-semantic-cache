"""Tests for the FastAPI proxy against the vLLM backend."""

import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from qdrant_client.http.models import ScoredPoint

from app.core.config import Settings
from app.features.proxy.service import CacheContext, ProxyService
from app.features.semantic_cache.service import (
    SemanticCacheService,
    SemanticCacheSettings,
)
from app.features.semantic_cache.types import (
    CacheDecisionStatus,
    CacheEntry,
    CacheEntryCreate,
    CacheHit,
    CacheLookupResult,
    CacheQuery,
    FeedbackEventCreate,
    TTLDecisionCreate,
)
from app.main import create_app


class _FakeEmbeddingService:
    """Deterministic embedding stub for tests."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        return [[1.0] * self._dimension for _ in inputs]


class _InMemoryVectorStore:
    """Minimal vector store stub that satisfies the interface."""

    def __init__(self) -> None:
        # Bypass parent constructor.
        self._points: dict[str, dict[str, Any]] = {}

    async def ensure_collection(self) -> None:
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
    ) -> list[ScoredPoint]:
        results: list[ScoredPoint] = []
        for payload in self._points.values():
            if (
                payload.get("model") == model
                and payload.get("params_fingerprint") == params_fingerprint
                and payload.get("expires_at_ts", 0.0) >= not_before.timestamp()
            ):
                results.append(
                    ScoredPoint(
                        id=payload["cache_entry_id"],
                        version=1,
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

    async def create_schema(self) -> None:
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

    async def get_entry(self, entry_id):
        return self.entries.get(str(entry_id))

    async def mark_hit(self, entry_id, accessed_at: datetime) -> None:
        entry = self.entries[str(entry_id)]
        updated = replace(
            entry,
            hit_count=entry.hit_count + 1,
            updated_at=accessed_at,
        )
        self.entries[str(entry_id)] = updated

    async def insert_ttl_decision(self, payload: TTLDecisionCreate) -> None:
        self.ttl_decisions.append(payload)

    async def update_entry_ttl(
        self,
        entry_id,
        *,
        ttl_bucket: int,
        ttl_seconds: int,
        expires_at: datetime,
    ) -> None:
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

    async def insert_policy_reward(
        self, payload
    ) -> None:  # pragma: no cover - unused stub
        return None

    async def attach_entry_to_decision(
        self, decision_id, cache_entry_id
    ) -> None:  # pragma: no cover - unused stub
        return None

    async def get_ttl_decision(self, decision_id):  # pragma: no cover - unused stub
        for record in self.ttl_decisions:
            if record.id == decision_id:
                return record
        return None

    async def fetch_prompt_stale_rates(
        self, prompt_hashes, *, lookback=timedelta(days=14)
    ):
        return {}


@asynccontextmanager
async def _build_test_app(
    backend_handler: Callable[[httpx.Request], httpx.Response],
    settings: Settings,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an httpx.AsyncClient bound to the FastAPI app with a mock backend."""

    def client_factory() -> httpx.AsyncClient:
        transport = httpx.MockTransport(backend_handler)
        return httpx.AsyncClient(transport=transport)

    app = create_app(settings=settings, client_factory=client_factory)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


@pytest.mark.anyio
async def test_proxy_forwards_json_request() -> None:
    """The proxy should forward JSON payloads and pass through backend responses."""
    recorded: dict[str, Any] = {}

    def backend(request: httpx.Request) -> httpx.Response:
        recorded["method"] = request.method
        recorded["url"] = str(request.url)
        recorded["headers"] = dict(request.headers)
        recorded["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"id": "req-123", "object": "chat.completion"},
            headers={"x-backend": "1"},
        )

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        vllm_api_key="backend-secret",
        inbound_api_keys=["client-secret"],
        semantic_cache_enabled=False,
    )

    async with _build_test_app(backend, settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "ping"}],
            },
            headers={"Authorization": "Bearer client-secret"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["id"] == "req-123"
    assert response.headers["x-backend"] == "1"
    assert response.headers["x-cache"] == "BYPASS"
    assert response.headers["x-cache-status"] == "miss"
    assert recorded["method"] == "POST"
    assert recorded["url"] == "http://backend.local/v1/chat/completions"
    assert recorded["headers"]["authorization"] == "Bearer backend-secret"
    assert recorded["body"]["model"] == "gpt-test"
    metadata = payload["ttl_proxy"]
    assert metadata["cache_status"] == "miss"
    assert metadata["semantic_cache_enabled"] is False
    assert "semantic_cache_disabled" in metadata.get("reasons", [])


class _EventStream(httpx.AsyncByteStream):
    """Simple async byte stream for testing streaming responses."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.aiter_bytes()

    async def aclose(self) -> None:
        return

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.anyio
async def test_proxy_streams_event_data() -> None:
    """Streaming responses should be relayed without buffering."""

    def backend(request: httpx.Request) -> httpx.Response:
        stream = _EventStream(
            [b"data: chunk-1\n\n", b"data: chunk-2\n\n", b"data: [DONE]\n\n"],
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        semantic_cache_enabled=False,
    )

    async with _build_test_app(backend, settings) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "gpt-test", "stream": True},
        ) as response:
            chunks = [chunk async for chunk in response.aiter_raw()]
            payload = b"".join(chunks)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["x-cache"] == "BYPASS"
    assert response.headers["x-cache-status"] == "miss"
    assert payload == b"data: chunk-1\n\ndata: chunk-2\n\ndata: [DONE]\n\n"
    assert len(chunks) >= 1


@pytest.mark.anyio
async def test_missing_authentication_rejected() -> None:
    """Requests must include a bearer token when inbound auth is configured."""

    def backend(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Backend should not be called on auth failure.")

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        inbound_api_keys=["client-secret"],
        semantic_cache_enabled=False,
    )

    async with _build_test_app(backend, settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing bearer token."


@pytest.mark.anyio
async def test_proxy_logs_decision_and_feedback_events() -> None:
    """Caching a miss should record TTL decisions and cache hits emit feedback."""

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
    await semantic_cache.bootstrap()

    backend_calls = {"count": 0}

    def backend(request: httpx.Request) -> httpx.Response:
        backend_calls["count"] += 1
        return httpx.Response(
            200,
            json={
                "id": "resp-001",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello world",
                        }
                    }
                ],
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    settings = SimpleNamespace(
        vllm_base_url="http://backend.local/v1/",
        vllm_api_key=None,
        inbound_api_keys=[],
        request_timeout_seconds=30.0,
    )
    proxy = ProxyService(
        settings=settings,
        client=client,
        semantic_cache=semantic_cache,
    )

    def _build_request() -> Request:
        payload = {
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "Breaking news about markets"},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "query_string": b"",
            "client": ("test", 1234),
            "server": ("testserver", 80),
        }

        state = {"sent": False}

        async def receive() -> dict[str, Any]:
            if state["sent"]:
                return {"type": "http.disconnect"}
            state["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}

        return Request(scope, receive)

    try:
        first_response = await proxy.forward("/v1/chat/completions", _build_request())
        assert first_response.status_code == 200
        assert backend_calls["count"] == 1

        first_payload = json.loads(first_response.body)
        first_decision_meta = first_payload["ttl_proxy"]["decision"]
        assert first_decision_meta["policy_name"] == "linucb"
        assert 0.0 < first_decision_meta["propensity"] <= 1.0

        second_response = await proxy.forward("/v1/chat/completions", _build_request())
        assert second_response.status_code == 200
        assert second_response.headers["X-Cache"].upper() == "HIT"
        assert backend_calls["count"] == 1, "Second request should hit cache"

        metadata = second_response.headers["X-Cache-Status"].lower()
        assert metadata == "hit"

        assert len(repository.ttl_decisions) == 1
        decision = repository.ttl_decisions[0]
        assert decision.policy_name == "linucb"
        assert decision.features["recency.keyword_hits"] >= 1
        assert decision.propensity is not None
        assert 0.0 < decision.propensity <= 1.0
        assert decision.ttl_bucket == first_decision_meta["ttl_bucket"]
        assert decision.propensity == pytest.approx(first_decision_meta["propensity"])

        assert len(repository.feedback_events) == 1
        assert repository.feedback_events[0].event_type == "cache_hit_served"
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_proxy_logs_cache_decision(caplog) -> None:
    """ProxyService should emit cache decision details via standard logging."""

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        semantic_cache_enabled=False,
    )

    entry_id = uuid4()
    decision_id = uuid4()

    async with httpx.AsyncClient(base_url="http://backend.local") as client:
        service = ProxyService(settings=settings, client=client)

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

        cache_context = CacheContext(
            request_path="/v1/chat/completions",
            request_method="POST",
            enabled=True,
            status=CacheDecisionStatus.HIT,
        )
        cache_context.query = query

        now = datetime.now(UTC)
        entry = CacheEntry(
            id=entry_id,
            model="gpt-test",
            parameters={},
            prompt_text="Hello world",
            response_payload={"choices": []},
            ttl_bucket=1,
            ttl_seconds=60,
            created_at=now,
            expires_at=now + timedelta(seconds=60),
            updated_at=now,
            hit_count=2,
            embedding_model="test-embed",
            embedding_mode="clustering",
            embedding_dimension=4,
            similarity_threshold=0.5,
        )

        cache_context.ttl_bucket = entry.ttl_bucket
        cache_context.policy_propensity = 0.4321
        cache_context.cache_entry_id = entry_id
        cache_context.ttl_decision_id = decision_id
        cache_context.reasons.append("cache_hit_served")
        cache_context.backend_status_code = 200
        cache_context.result = CacheLookupResult(
            status=CacheDecisionStatus.HIT,
            query=query,
            neighbor_stale_rate=0.1234,
            hit=CacheHit(entry=entry, similarity=0.9876),
            reasons=("cache_hit_served",),
        )

        with caplog.at_level(logging.INFO, logger="app.features.proxy.service"):
            service._log_cache_decision(cache_context, latency_ms=12.34)

    message = caplog.text
    assert "proxy decision" in message
    assert f"cache_entry_id={entry_id}" in message
    assert "similarity=0.9876" in message
    assert "reasons=cache_hit_served" in message
