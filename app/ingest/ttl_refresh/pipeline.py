"""Interfaces for refresh evaluation pipelines."""

from __future__ import annotations

from typing import Protocol, Sequence

from app.features.semantic_cache.types import (
    BucketAggregate,
    RefreshCandidate,
    RefreshOutcome,
)


class RefreshPipeline(Protocol):
    """Evaluate cache entries to determine freshness and outcomes."""

    async def evaluate(
        self,
        candidate: RefreshCandidate,
        *,
        bucket_metrics: Sequence[BucketAggregate],
    ) -> RefreshOutcome:
        """Return the freshness outcome for the supplied cache entry."""
        raise NotImplementedError
