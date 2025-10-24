"""Batch job helpers for the TTL refresh pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

from app.cache.repository import CacheRepository
from app.cache.types import BucketAggregate, RefreshCandidate


@dataclass(frozen=True, slots=True)
class RefreshBatch:
    """Container for refresh planning output."""

    candidates: tuple[RefreshCandidate, ...]
    bucket_metrics: tuple[BucketAggregate, ...]


async def build_refresh_batch(
    repository: CacheRepository,
    *,
    limit: int = 200,
    lookback: timedelta = timedelta(days=7),
) -> RefreshBatch:
    """Collect refresh candidates and recent metrics for downstream workers."""
    candidates: Sequence[RefreshCandidate] = await repository.list_refresh_candidates(
        limit=limit,
        lookback=lookback,
    )
    bucket_metrics: Sequence[
        BucketAggregate
    ] = await repository.aggregate_bucket_metrics(
        lookback=lookback,
    )
    return RefreshBatch(
        candidates=tuple(candidates),
        bucket_metrics=tuple(bucket_metrics),
    )
