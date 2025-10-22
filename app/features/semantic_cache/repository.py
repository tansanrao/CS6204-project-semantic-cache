"""Database access layer for the semantic cache."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from .models import cache_entry, metadata
from .types import CacheEntry, CacheEntryCreate


class CacheRepository:
    """Persist and retrieve cache entries from Postgres."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_schema(self) -> None:
        """Ensure all required tables are present."""
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def insert_entry(self, payload: CacheEntryCreate) -> None:
        """Insert a new cache entry."""
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(cache_entry).values(
                    id=payload.id,
                    request_fingerprint=payload.request_fingerprint,
                    prompt_hash=payload.prompt_hash,
                    model=payload.model,
                    parameters=payload.parameters,
                    prompt_text=payload.prompt_text,
                    response_payload=payload.response_payload,
                    ttl_bucket=payload.ttl_bucket,
                    ttl_seconds=payload.ttl_seconds,
                    created_at=payload.created_at,
                    expires_at=payload.expires_at,
                    embedding_model=payload.embedding_model,
                    embedding_mode=payload.embedding_mode,
                    embedding_dimension=payload.embedding_dimension,
                    similarity_threshold=payload.similarity_threshold,
                )
            )

    async def get_entry(self, entry_id: UUID) -> CacheEntry | None:
        """Load a cache entry by its identifier."""
        async with self._engine.connect() as connection:
            result = await connection.execute(
                select(cache_entry).where(cache_entry.c.id == entry_id)
            )
            row = result.first()
        if row is None:
            return None
        return _row_to_entry(row._mapping)

    async def mark_hit(self, entry_id: UUID, accessed_at: datetime) -> None:
        """Increment the hit counter and update last access timestamp."""
        async with self._engine.begin() as connection:
            await connection.execute(
                update(cache_entry)
                .where(cache_entry.c.id == entry_id)
                .values(
                    hit_count=cache_entry.c.hit_count + 1,
                    last_accessed_at=accessed_at,
                    updated_at=accessed_at,
                )
            )


def _row_to_entry(mapping: Any) -> CacheEntry:
    """Convert a SQLAlchemy row mapping into a CacheEntry."""
    return CacheEntry(
        id=mapping["id"],
        model=mapping["model"],
        parameters=dict(mapping["parameters"]),
        prompt_text=mapping["prompt_text"],
        response_payload=dict(mapping["response_payload"]),
        ttl_bucket=mapping["ttl_bucket"],
        ttl_seconds=mapping["ttl_seconds"],
        created_at=mapping["created_at"],
        expires_at=mapping["expires_at"],
        updated_at=mapping["updated_at"],
        hit_count=mapping["hit_count"],
        embedding_model=mapping["embedding_model"],
        embedding_mode=mapping["embedding_mode"],
        embedding_dimension=mapping["embedding_dimension"],
        similarity_threshold=mapping["similarity_threshold"],
    )
