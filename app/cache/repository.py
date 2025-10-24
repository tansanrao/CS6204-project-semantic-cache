"""Database access layer for the semantic cache."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import case, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from .migrations import upgrade_head
from .models import (
    cache_entry,
    feedback_event,
    metadata,
    policy_reward,
    ttl_decision,
)
from .types import (
    BucketAggregate,
    CacheEntry,
    CacheEntryCreate,
    FeedbackEventCreate,
    PolicyRewardCreate,
    RefreshCandidate,
    RefreshEventType,
    TTLDecision,
    TTLDecisionCreate,
)

LOG = logging.getLogger("app.semantic_cache.repository")


class CacheRepository:
    """Persist and retrieve cache entries from Postgres."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_schema(self) -> None:
        """Ensure all required tables are present."""
        backend = self._engine.url.get_backend_name()
        LOG.info(
            "semantic_cache.repository.bootstrap_start",
            extra={"extra": {"backend": backend}},
        )

        async with self._engine.begin() as connection:
            if backend == "postgresql":
                await connection.run_sync(
                    lambda conn: metadata.create_all(
                        bind=conn,
                        tables=[cache_entry],
                    )
                )
            else:
                await connection.run_sync(metadata.create_all)

        if backend == "postgresql":
            await asyncio.to_thread(
                upgrade_head,
                str(self._engine.url),
            )
        LOG.info(
            "semantic_cache.repository.bootstrap_complete",
            extra={"extra": {"backend": backend}},
        )

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

    async def insert_ttl_decision(self, payload: TTLDecisionCreate) -> None:
        """Insert a TTL decision row for policy logging."""
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(ttl_decision).values(
                    id=payload.id,
                    cache_entry_id=payload.cache_entry_id,
                    request_fingerprint=payload.request_fingerprint,
                    prompt_hash=payload.prompt_hash,
                    policy_name=payload.policy_name,
                    policy_version=payload.policy_version,
                    ttl_bucket=payload.ttl_bucket,
                    propensity=payload.propensity,
                    features=payload.features,
                    created_at=payload.created_at,
                )
            )

    async def attach_entry_to_decision(
        self, decision_id: UUID, cache_entry_id: UUID
    ) -> None:
        """Associate a persisted cache entry with an earlier decision."""
        async with self._engine.begin() as connection:
            await connection.execute(
                update(ttl_decision)
                .where(ttl_decision.c.id == decision_id)
                .values(cache_entry_id=cache_entry_id)
            )

    async def insert_feedback_event(self, payload: FeedbackEventCreate) -> None:
        """Persist evaluator or user feedback about cache freshness."""
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(feedback_event).values(
                    id=payload.id,
                    cache_entry_id=payload.cache_entry_id,
                    ttl_decision_id=payload.ttl_decision_id,
                    event_type=payload.event_type,
                    score=payload.score,
                    details=payload.details,
                    created_at=payload.created_at,
                )
            )

    async def insert_policy_reward(self, payload: PolicyRewardCreate) -> None:
        """Log reward attribution for a TTL decision."""
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(policy_reward).values(
                    id=payload.id,
                    ttl_decision_id=payload.ttl_decision_id,
                    reward=payload.reward,
                    attribution_rule=payload.attribution_rule,
                    created_at=payload.created_at,
                )
            )

    async def update_entry_ttl(
        self,
        entry_id: UUID,
        *,
        ttl_bucket: int,
        ttl_seconds: int,
        expires_at: datetime,
    ) -> None:
        """Adjust TTL configuration for an existing cache entry."""
        now = datetime.now(UTC)
        async with self._engine.begin() as connection:
            await connection.execute(
                update(cache_entry)
                .where(cache_entry.c.id == entry_id)
                .values(
                    ttl_bucket=ttl_bucket,
                    ttl_seconds=ttl_seconds,
                    expires_at=expires_at,
                    updated_at=now,
                )
            )

    async def get_ttl_decision(self, decision_id: UUID) -> TTLDecision | None:
        """Load a TTL decision row for auditing or reward attribution."""
        async with self._engine.connect() as connection:
            result = await connection.execute(
                select(ttl_decision).where(ttl_decision.c.id == decision_id)
            )
            row = result.first()
        if row is None:
            return None
        return _row_to_ttl_decision(row._mapping)

    async def list_refresh_candidates(
        self,
        *,
        limit: int = 500,
        lookback: timedelta = timedelta(days=7),
    ) -> list[RefreshCandidate]:
        """Return cache entries that should be evaluated for freshness."""
        now = datetime.now(UTC)
        window_start = now - lookback

        stmt = (
            select(
                cache_entry.c.id,
                cache_entry.c.prompt_text,
                cache_entry.c.response_payload,
                cache_entry.c.created_at,
                cache_entry.c.expires_at,
                cache_entry.c.ttl_bucket,
                cache_entry.c.ttl_seconds,
                cache_entry.c.hit_count,
            )
            .where(cache_entry.c.created_at >= window_start)
            .where(cache_entry.c.expires_at >= now)
            .order_by(cache_entry.c.created_at.desc())
            .limit(limit)
        )

        async with self._engine.connect() as connection:
            result = await connection.execute(stmt)
            rows = result.fetchall()

        LOG.info(
            "semantic_cache.repository.refresh_candidates",
            extra={
                "extra": {
                    "count": len(rows),
                    "limit": limit,
                    "lookback_hours": round(lookback.total_seconds() / 3600.0, 2),
                }
            },
        )

        return [_row_to_refresh_candidate(row._mapping) for row in rows]

    async def aggregate_bucket_metrics(
        self,
        *,
        lookback: timedelta = timedelta(days=7),
    ) -> list[BucketAggregate]:
        """Compute per-bucket decision, feedback, and reward summaries."""
        window_start = datetime.now(UTC) - lookback

        decision_stmt = (
            select(
                ttl_decision.c.ttl_bucket,
                func.count(ttl_decision.c.id).label("decision_count"),
            )
            .where(ttl_decision.c.created_at >= window_start)
            .group_by(ttl_decision.c.ttl_bucket)
        )
        feedback_stmt = (
            select(
                ttl_decision.c.ttl_bucket,
                func.count(feedback_event.c.id).label("feedback_count"),
            )
            .join(
                feedback_event,
                feedback_event.c.ttl_decision_id == ttl_decision.c.id,
            )
            .where(feedback_event.c.created_at >= window_start)
            .group_by(ttl_decision.c.ttl_bucket)
        )
        reward_stmt = (
            select(
                ttl_decision.c.ttl_bucket,
                func.avg(policy_reward.c.reward).label("avg_reward"),
            )
            .join(
                policy_reward,
                policy_reward.c.ttl_decision_id == ttl_decision.c.id,
            )
            .where(policy_reward.c.created_at >= window_start)
            .group_by(ttl_decision.c.ttl_bucket)
        )

        async with self._engine.connect() as connection:
            decision_rows = (await connection.execute(decision_stmt)).all()
            feedback_rows = (await connection.execute(feedback_stmt)).all()
            reward_rows = (await connection.execute(reward_stmt)).all()

        decision_map: dict[int, int] = {
            row._mapping["ttl_bucket"]: row._mapping["decision_count"]
            for row in decision_rows
        }
        feedback_map: dict[int, int] = defaultdict(int)
        for row in feedback_rows:
            feedback_map[row._mapping["ttl_bucket"]] = row._mapping["feedback_count"]
        reward_map: dict[int, float | None] = {}
        for row in reward_rows:
            reward_map[row._mapping["ttl_bucket"]] = row._mapping["avg_reward"]

        aggregates: list[BucketAggregate] = []
        for bucket, decision_count in sorted(decision_map.items()):
            aggregates.append(
                BucketAggregate(
                    ttl_bucket=bucket,
                    decision_count=decision_count,
                    feedback_count=feedback_map.get(bucket, 0),
                    avg_reward=reward_map.get(bucket),
                )
            )
        LOG.info(
            "semantic_cache.repository.bucket_aggregates",
            extra={
                "extra": {
                    "bucket_count": len(aggregates),
                    "lookback_hours": round(lookback.total_seconds() / 3600.0, 2),
                }
            },
        )
        return aggregates

    async def fetch_prompt_stale_rates(
        self,
        prompt_hashes: Sequence[str],
        *,
        lookback: timedelta = timedelta(days=14),
    ) -> dict[str, float]:
        """Return recent stale-rate estimates keyed by prompt hash."""
        unique_hashes = tuple({value for value in prompt_hashes if value})
        if not unique_hashes:
            return {}

        allowed_events = (
            RefreshEventType.STALE_CONFIRMED.value,
            RefreshEventType.FRESH_CONFIRMED.value,
        )
        stale_indicator = case(
            (feedback_event.c.event_type == RefreshEventType.STALE_CONFIRMED.value, 1),
            else_=0,
        )
        stmt = (
            select(
                cache_entry.c.prompt_hash,
                func.sum(stale_indicator).label("stale_count"),
                func.count(feedback_event.c.id).label("total_count"),
            )
            .join(
                feedback_event,
                feedback_event.c.cache_entry_id == cache_entry.c.id,
            )
            .where(cache_entry.c.prompt_hash.in_(unique_hashes))
            .where(feedback_event.c.event_type.in_(allowed_events))
        )
        if lookback is not None:
            cutoff = datetime.now(UTC) - lookback
            stmt = stmt.where(feedback_event.c.created_at >= cutoff)
        stmt = stmt.group_by(cache_entry.c.prompt_hash)

        async with self._engine.connect() as connection:
            rows = (await connection.execute(stmt)).all()

        LOG.info(
            "semantic_cache.repository.stale_rate_fetch",
            extra={
                "extra": {
                    "prompt_count": len(unique_hashes),
                    "result_rows": len(rows),
                    "lookback_hours": None
                    if lookback is None
                    else round(lookback.total_seconds() / 3600.0, 2),
                }
            },
        )

        rates: dict[str, float] = {}
        for row in rows:
            mapping = row._mapping
            total = float(mapping["total_count"])
            if total <= 0:
                continue
            stale_count = float(mapping["stale_count"])
            rates[mapping["prompt_hash"]] = max(0.0, min(stale_count / total, 1.0))
        return rates


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


def _row_to_ttl_decision(mapping: Any) -> TTLDecision:
    """Convert a row mapping into a TTLDecision."""
    return TTLDecision(
        id=mapping["id"],
        cache_entry_id=mapping.get("cache_entry_id"),
        request_fingerprint=mapping["request_fingerprint"],
        prompt_hash=mapping["prompt_hash"],
        policy_name=mapping["policy_name"],
        policy_version=mapping["policy_version"],
        ttl_bucket=mapping["ttl_bucket"],
        features=dict(mapping["features"]),
        propensity=mapping.get("propensity"),
        created_at=mapping["created_at"],
    )


def _row_to_refresh_candidate(mapping: Any) -> RefreshCandidate:
    """Convert a row mapping into a RefreshCandidate."""
    return RefreshCandidate(
        id=mapping["id"],
        prompt_text=mapping["prompt_text"],
        response_payload=dict(mapping["response_payload"]),
        created_at=mapping["created_at"],
        expires_at=mapping["expires_at"],
        ttl_bucket=mapping["ttl_bucket"],
        ttl_seconds=mapping["ttl_seconds"],
        hit_count=mapping["hit_count"],
    )
