"""Typed structures shared across semantic cache components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class CacheQuery:
    """Normalized request data used to query the semantic cache."""

    request_path: str
    model: str
    prompt_text: str
    embedding_input: str
    request_fingerprint: str
    params_fingerprint: str
    prompt_hash: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CacheEntryCreate:
    """Payload required to insert a new cache entry."""

    id: UUID
    request_fingerprint: str
    prompt_hash: str
    model: str
    parameters: dict[str, Any]
    prompt_text: str
    response_payload: dict[str, Any]
    ttl_bucket: int
    ttl_seconds: int
    created_at: datetime
    expires_at: datetime
    embedding_model: str
    embedding_mode: str
    embedding_dimension: int
    similarity_threshold: float


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Database record that can be returned to callers."""

    id: UUID
    model: str
    parameters: dict[str, Any]
    prompt_text: str
    response_payload: dict[str, Any]
    ttl_bucket: int
    ttl_seconds: int
    created_at: datetime
    expires_at: datetime
    updated_at: datetime
    hit_count: int
    embedding_model: str
    embedding_mode: str
    embedding_dimension: int
    similarity_threshold: float


@dataclass(frozen=True, slots=True)
class CacheHit:
    """Semantic cache hit returned to the proxy."""

    entry: CacheEntry
    similarity: float


class CacheDecisionStatus(StrEnum):
    """Outcome categories for semantic cache lookups."""

    HIT = "hit"
    STALE_HIT = "stale_hit"
    MISS = "miss"


@dataclass(frozen=True, slots=True)
class CacheLookupResult:
    """Detailed result describing a cache lookup decision."""

    status: CacheDecisionStatus
    query: CacheQuery
    hit: CacheHit | None = None
    stale_entry: CacheEntry | None = None
    stale_similarity: float | None = None
    reasons: tuple[str, ...] = ()
