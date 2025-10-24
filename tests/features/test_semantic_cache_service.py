"""Unit tests for the semantic cache service."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from app.cache.service import (
    SemanticCacheService,
    SemanticCacheSettings,
)
from app.cache.types import CacheDecisionStatus, CacheEntry


class _StaticEmbedder:
    """Deterministic embedding stub to keep tests predictable."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class _InMemoryRepository:
    """Minimal in-memory repository compatible with the real interface."""

    def __init__(self) -> None:
        self._entries: dict[UUID, CacheEntry] = {}
        self._prompt_stale_rates: dict[str, float] = {}

    async def create_schema(self) -> None:  # pragma: no cover - nothing to do
        return

    async def insert_entry(self, payload) -> None:
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
        self._entries[payload.id] = entry

    async def get_entry(self, entry_id: UUID) -> CacheEntry | None:
        return self._entries.get(entry_id)

    async def mark_hit(self, entry_id: UUID, accessed_at: datetime) -> None:
        entry = self._entries[entry_id]
        self._entries[entry_id] = replace(
            entry,
            hit_count=entry.hit_count + 1,
            updated_at=accessed_at,
        )

    def set_prompt_stale_rate(self, prompt_hash: str, rate: float) -> None:
        self._prompt_stale_rates[prompt_hash] = rate

    async def fetch_prompt_stale_rates(
        self, prompt_hashes, *, lookback=timedelta(days=14)
    ):
        return {
            prompt_hash: self._prompt_stale_rates[prompt_hash]
            for prompt_hash in prompt_hashes
            if prompt_hash in self._prompt_stale_rates
        }

    async def update_entry_ttl(
        self,
        entry_id: UUID,
        *,
        ttl_bucket: int,
        ttl_seconds: int,
        expires_at: datetime,
    ) -> None:
        entry = self._entries[entry_id]
        self._entries[entry_id] = replace(
            entry,
            ttl_bucket=ttl_bucket,
            ttl_seconds=ttl_seconds,
            expires_at=expires_at,
            updated_at=expires_at,
        )


class _Point:
    """Simple payload holder mimicking qdrant.ScoredPoint."""

    def __init__(self, payload: dict[str, Any], score: float) -> None:
        self.payload = payload
        self.score = score


class _InMemoryVectorStore:
    """Track points that would otherwise live inside Qdrant."""

    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.ready = False

    async def ensure_collection(self) -> None:
        self.ready = True

    async def upsert_point(
        self,
        point_id,
        vector,
        payload,
    ) -> None:
        self.points[str(point_id)] = {
            "vector": list(vector),
            "payload": dict(payload),
        }

    async def search(
        self,
        *,
        vector,
        model: str,
        params_fingerprint: str,
        not_before: datetime,
        limit: int,
    ):
        matches: list[_Point] = []
        for point in self.points.values():
            payload = point["payload"]
            if payload["model"] != model:
                continue
            if payload["params_fingerprint"] != params_fingerprint:
                continue
            if payload["expires_at_ts"] < not_before.timestamp():
                continue
            matches.append(_Point(payload=payload, score=0.95))
        return matches[:limit]


def _build_service(
    embed_dim: int = 3, *, policy_enabled: bool = True
) -> SemanticCacheService:
    repository = _InMemoryRepository()
    vector_store = _InMemoryVectorStore()
    embedder = _StaticEmbedder([0.1] * embed_dim)
    settings = SemanticCacheSettings(
        similarity_threshold=0.5,
        search_limit=3,
        ttl_seconds=(30, 60, 120),
        default_ttl_bucket=1,
        policy_enabled=policy_enabled,
        policy_feature_dimension=16,
        policy_autosave_interval=1,
    )
    return SemanticCacheService(
        repository=repository,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        settings=settings,
        embedding_model_name="nomic-ai/nomic-embed-text-v1.5",
        embedding_mode="clustering",
        embedding_dimension=embed_dim,
    )


@pytest.mark.anyio
async def test_store_and_lookup_round_trip() -> None:
    """Responses stored in the cache should be retrievable via similarity lookup."""
    service = _build_service()
    payload = {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "ping"},
        ],
    }
    query = service.normalize_request("v1/chat/completions", payload)
    assert query is not None

    miss = await service.lookup(query)
    assert miss.status is CacheDecisionStatus.MISS
    assert miss.hit is None

    response_payload = {
        "id": "completion-1",
        "choices": [{"message": {"content": "pong"}}],
    }
    stored = await service.store(query, response_payload)
    assert stored.response_payload == response_payload

    hit = await service.lookup(query)
    assert hit.status is CacheDecisionStatus.HIT
    assert hit.hit is not None
    assert hit.hit.entry.response_payload == response_payload
    assert hit.hit.entry.hit_count == 1
    assert hit.hit.similarity >= 0.5


@pytest.mark.anyio
async def test_lookup_includes_neighbor_stale_rate() -> None:
    """Nearest-neighbor stale-rate estimates should surface on lookup results."""
    service = _build_service()
    payload = {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "keep it short"},
            {"role": "user", "content": "latest earnings for ACME"},
        ],
    }
    query = service.normalize_request("v1/chat/completions", payload)
    assert query is not None

    response_payload = {
        "id": "completion-3",
        "choices": [{"message": {"content": "ACME earnings summary"}}],
    }
    await service.store(query, response_payload)

    repository = service._repository  # type: ignore[attr-defined]
    repository.set_prompt_stale_rate(query.prompt_hash, 0.6)  # type: ignore[attr-defined]

    result = await service.lookup(query)
    assert result.status is CacheDecisionStatus.HIT
    assert result.neighbor_stale_rate == pytest.approx(0.6)


@pytest.mark.anyio
async def test_select_ttl_bucket_returns_propensity() -> None:
    service = _build_service()
    bucket, propensity = await service.select_ttl_bucket({"bias": 1.0})
    assert 0 <= bucket < len(service._settings.ttl_seconds)
    assert 0.0 < propensity <= 1.0


@pytest.mark.anyio
async def test_lookup_skips_expired_entries() -> None:
    """Expired entries should be filtered out during lookup."""
    service = _build_service()
    payload = {
        "model": "gpt-test",
        "prompt": "hello world",
    }
    query = service.normalize_request("v1/completions", payload)
    assert query is not None

    stored = await service.store(query, {"id": "completion-2"})

    # Force the entry to appear expired in both stores.
    repo = service._repository  # type: ignore[attr-defined]
    vector_store = service._vector_store  # type: ignore[attr-defined]
    repo._entries[stored.id] = replace(  # type: ignore[attr-defined]
        repo._entries[stored.id],
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    point_payload = vector_store.points[str(stored.id)]["payload"]  # type: ignore[index]
    point_payload["expires_at_ts"] = (
        datetime.now(UTC) - timedelta(seconds=1)
    ).timestamp()

    result = await service.lookup(query)
    assert result.status is CacheDecisionStatus.MISS
    assert result.hit is None


def test_normalize_chat_completion_handles_structured_content() -> None:
    """Chat payloads with structured content should normalize correctly."""
    service = _build_service()
    payload = {
        "model": "gpt-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "value is"},
                    {"type": "text", "text": "42"},
                ],
            }
        ],
        "temperature": 0.5,
    }
    query = service.normalize_request("v1/chat/completions", payload)
    assert query is not None
    assert query.model == "gpt-test"
    assert "user:" in query.prompt_text
    assert query.parameters["temperature"] == 0.5


def test_semantic_cache_settings_bucket_validation() -> None:
    """Invalid bucket selections should raise a clear error."""
    settings = SemanticCacheSettings(ttl_seconds=(5, 10), default_ttl_bucket=0)
    assert settings.ttl_for_bucket(None) == (0, 5)
    with pytest.raises(ValueError):
        settings.ttl_for_bucket(5)
