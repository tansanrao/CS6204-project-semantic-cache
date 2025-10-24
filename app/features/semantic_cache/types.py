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
    neighbor_stale_rate: float | None = None
    hit: CacheHit | None = None
    stale_entry: CacheEntry | None = None
    stale_similarity: float | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TTLDecisionCreate:
    """Payload for recording a TTL bucket decision."""

    id: UUID
    cache_entry_id: UUID | None
    request_fingerprint: str
    prompt_hash: str
    policy_name: str
    policy_version: str
    ttl_bucket: int
    features: dict[str, Any]
    propensity: float | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TTLDecision:
    """Persisted TTL decision record."""

    id: UUID
    cache_entry_id: UUID | None
    request_fingerprint: str
    prompt_hash: str
    policy_name: str
    policy_version: str
    ttl_bucket: int
    features: dict[str, Any]
    propensity: float | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class FeedbackEventCreate:
    """Record user or evaluator feedback for a cache entry."""

    id: UUID
    cache_entry_id: UUID
    ttl_decision_id: UUID | None
    event_type: str
    score: float
    details: dict[str, Any] | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PolicyRewardCreate:
    """Persist reward attribution for a TTL decision."""

    id: UUID
    ttl_decision_id: UUID
    reward: float
    attribution_rule: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RefreshCandidate:
    """Cache entry scheduled for a freshness check."""

    id: UUID
    prompt_text: str
    response_payload: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    ttl_bucket: int
    ttl_seconds: int
    hit_count: int


@dataclass(frozen=True, slots=True)
class BucketAggregate:
    """Aggregated metrics for TTL decisions grouped by bucket."""

    ttl_bucket: int
    decision_count: int
    feedback_count: int
    avg_reward: float | None


class RefreshEventType(StrEnum):
    """Enumerate feedback event types for refresh evaluations."""

    STALE_CONFIRMED = "stale_detected"
    FRESH_CONFIRMED = "stale_disproved"
    ERROR = "refresh_failed"


@dataclass(frozen=True, slots=True)
class RefreshOutcome:
    """Result from evaluating a cache entry's freshness."""

    cache_entry_id: UUID
    ttl_decision_id: UUID | None
    stale: bool
    elapsed_seconds: int
    ttl_seconds: int
    ttl_bucket: int
    semantic_delta: float
    fact_delta: float
    cost_savings: float
    latency_savings: float
    feedback_event: RefreshEventType
    feedback_score: float
    feedback_details: dict[str, Any] | None
    obtained_at: datetime
    bonus_applicable: bool = False
