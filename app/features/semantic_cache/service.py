"""High-level semantic cache orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4

from qdrant_client.http.models import ScoredPoint

from .embedding import EmbeddingService
from .qdrant import QdrantVectorStore
from .repository import CacheRepository
from .types import (
    CacheDecisionStatus,
    CacheEntry,
    CacheEntryCreate,
    CacheHit,
    CacheLookupResult,
    CacheQuery,
)


@dataclass(frozen=True, slots=True)
class SemanticCacheSettings:
    """Configuration knobs for the semantic cache service."""

    similarity_threshold: float = 0.86
    search_limit: int = 5
    ttl_seconds: tuple[int, ...] = (60, 300, 900, 1800, 3600, 7200)
    default_ttl_bucket: int = 2

    def ttl_for_bucket(self, bucket: int | None) -> tuple[int, int]:
        """Return (bucket_index, ttl_seconds) with validation."""
        if bucket is None:
            bucket = self.default_ttl_bucket
        if not 0 <= bucket < len(self.ttl_seconds):
            raise ValueError(f"Invalid TTL bucket {bucket}.")
        return bucket, self.ttl_seconds[bucket]


class SemanticCacheService:
    """Provide lookup and storage operations backed by Postgres and Qdrant."""

    def __init__(
        self,
        repository: CacheRepository,
        vector_store: QdrantVectorStore,
        embedder: EmbeddingService,
        *,
        settings: SemanticCacheSettings | None = None,
        embedding_model_name: str,
        embedding_mode: str,
        embedding_dimension: int,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._embedder = embedder
        self._settings = settings or SemanticCacheSettings()
        self._embedding_model_name = embedding_model_name
        self._embedding_mode = embedding_mode
        self._embedding_dimension = embedding_dimension

    async def bootstrap(self) -> None:
        """Ensure storage backends are ready for use."""
        await self._repository.create_schema()
        await self._vector_store.ensure_collection()

    def normalize_request(
        self, path: str, payload: dict[str, Any]
    ) -> CacheQuery | None:
        """Normalize a JSON payload into a cache query."""
        lowered = path.lower().strip("/")
        if lowered.endswith("chat/completions"):
            return _normalize_chat_completion(path, payload)
        if lowered.endswith("completions"):
            return _normalize_legacy_completion(path, payload)
        return None

    async def lookup(self, query: CacheQuery) -> CacheLookupResult:
        """Search Qdrant and return the best matching cache entry."""
        vector = await self._embed_prompt(query)
        points = await self._vector_store.search(
            vector=vector,
            model=query.model,
            params_fingerprint=query.params_fingerprint,
            not_before=datetime.now(UTC),
            limit=self._settings.search_limit,
        )
        return await self._materialize_hit(query, points)

    async def store(
        self,
        query: CacheQuery,
        response_payload: dict[str, Any],
        *,
        ttl_bucket: int | None = None,
    ) -> CacheEntry:
        """Persist a cache miss response into Postgres and Qdrant."""
        vector = await self._embed_prompt(query)
        bucket_index, ttl_seconds = self._settings.ttl_for_bucket(ttl_bucket)

        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(seconds=ttl_seconds)
        entry_id = uuid4()

        payload = CacheEntryCreate(
            id=entry_id,
            request_fingerprint=query.request_fingerprint,
            prompt_hash=query.prompt_hash,
            model=query.model,
            parameters=query.parameters,
            prompt_text=query.prompt_text,
            response_payload=response_payload,
            ttl_bucket=bucket_index,
            ttl_seconds=ttl_seconds,
            created_at=created_at,
            expires_at=expires_at,
            embedding_model=self._embedding_model_name,
            embedding_mode=self._embedding_mode,
            embedding_dimension=self._embedding_dimension,
            similarity_threshold=self._settings.similarity_threshold,
        )

        await self._repository.insert_entry(payload)
        await self._vector_store.upsert_point(
            entry_id,
            vector,
            {
                "cache_entry_id": str(entry_id),
                "model": query.model,
                "params_fingerprint": query.params_fingerprint,
                "request_fingerprint": query.request_fingerprint,
                "prompt_hash": query.prompt_hash,
                "created_at_ts": created_at.timestamp(),
                "expires_at_ts": expires_at.timestamp(),
                "ttl_bucket": bucket_index,
                "ttl_seconds": ttl_seconds,
            },
        )
        stored = await self._repository.get_entry(entry_id)
        if stored is None:
            raise RuntimeError("Inserted cache entry could not be reloaded.")
        return stored

    async def record_hit(self, entry_id: UUID) -> None:
        """Increment hit metrics for the supplied entry."""
        await self._repository.mark_hit(entry_id, datetime.now(UTC))

    async def _embed_prompt(self, query: CacheQuery) -> list[float]:
        """Compute an embedding vector for the cache query."""
        vectors = await self._embedder.embed([query.embedding_input])
        if not vectors:
            raise ValueError("Embedding service returned no vectors.")
        if len(vectors[0]) != self._embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: expected "
                f"{self._embedding_dimension}, received {len(vectors[0])}"
            )
        return vectors[0]

    async def _materialize_hit(
        self, query: CacheQuery, points: list[ScoredPoint]
    ) -> CacheLookupResult:
        """Convert Qdrant scored points into cache entries."""
        stale_candidate: tuple[CacheEntry, float] | None = None
        reasons: list[str] = []
        for point in points:
            payload = point.payload or {}
            entry_id_raw = payload.get("cache_entry_id")
            if not entry_id_raw:
                reasons.append("missing_entry_id")
                continue
            try:
                entry_id = UUID(entry_id_raw)
            except ValueError:
                reasons.append("invalid_entry_id")
                continue

            entry = await self._repository.get_entry(entry_id)
            if entry is None:
                reasons.append("entry_not_found")
                continue
            if entry.expires_at <= datetime.now(UTC):
                reasons.append("expired_entry")
                stale_candidate = stale_candidate or (entry, point.score)
                continue
            await self.record_hit(entry_id)
            refreshed = await self._repository.get_entry(entry_id)
            if refreshed is None:
                reasons.append("entry_not_found_after_hit")
                continue
            return CacheLookupResult(
                status=CacheDecisionStatus.HIT,
                query=query,
                hit=CacheHit(entry=refreshed, similarity=point.score),
                reasons=tuple(reasons),
            )
        if stale_candidate is not None:
            entry, similarity = stale_candidate
            return CacheLookupResult(
                status=CacheDecisionStatus.STALE_HIT,
                query=query,
                stale_entry=entry,
                stale_similarity=similarity,
                reasons=tuple(reasons or ("expired_entry",)),
            )
        return CacheLookupResult(
            status=CacheDecisionStatus.MISS,
            query=query,
            reasons=tuple(reasons or ("no_matches",)),
        )


def _normalize_chat_completion(path: str, payload: dict[str, Any]) -> CacheQuery | None:
    """Flatten a Chat Completions request into a cache query."""
    if payload.get("stream"):
        return None
    model = payload.get("model")
    messages = payload.get("messages")
    if not model or not isinstance(messages, list):
        return None

    prompt_segments = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prompt_segments.append(f"{role}:{_stringify_content(content)}")

    prompt_text = "\n".join(prompt_segments)
    parameters = {k: v for k, v in payload.items() if k not in {"messages"}}
    return _build_query(path, model, prompt_text, parameters)


def _normalize_legacy_completion(
    path: str, payload: dict[str, Any]
) -> CacheQuery | None:
    """Flatten legacy completion payloads that use a prompt field."""
    if payload.get("stream"):
        return None
    model = payload.get("model")
    prompt = payload.get("prompt")
    if not model or prompt is None:
        return None
    if isinstance(prompt, list):
        prompt_text = "\n".join(str(item) for item in prompt)
    else:
        prompt_text = str(prompt)
    parameters = {k: v for k, v in payload.items() if k not in {"prompt"}}
    return _build_query(path, model, prompt_text, parameters)


def _build_query(
    path: str,
    model: str,
    prompt_text: str,
    parameters: dict[str, Any],
) -> CacheQuery:
    """Construct the CacheQuery dataclass with consistent hashing."""
    normalized_prompt = prompt_text.strip()
    prompt_hash = sha256(normalized_prompt.encode("utf-8")).hexdigest()
    params_fingerprint = _stable_hash(parameters)
    request_fingerprint = _stable_hash(
        {
            "path": path.lower(),
            "model": model,
            "params_fingerprint": params_fingerprint,
            "prompt_hash": prompt_hash,
        }
    )
    return CacheQuery(
        request_path=path,
        model=model,
        prompt_text=normalized_prompt,
        embedding_input=normalized_prompt,
        request_fingerprint=request_fingerprint,
        params_fingerprint=params_fingerprint,
        prompt_hash=prompt_hash,
        parameters=parameters,
    )


def _stable_hash(payload: Any) -> str:
    """Serialize the payload deterministically and return its SHA-256 hash."""
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _stringify_content(content: Any) -> str:
    """Convert OpenAI content structures into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(
                    item.get("text")
                    or item.get("content")
                    or json.dumps(item, sort_keys=True)
                )
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, sort_keys=True)
    return str(content)
