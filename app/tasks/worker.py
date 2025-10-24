"""Refresh worker orchestrating batch scheduling and result handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Sequence

from app.cache.repository import CacheRepository
from app.cache.types import BucketAggregate, RefreshOutcome

from .jobs import build_refresh_batch
from .pipeline import RefreshPipeline


@dataclass(frozen=True, slots=True)
class RefreshWorkerReport:
    """Summary of a refresh worker run."""

    processed: int
    skipped: int
    bucket_metrics: tuple[BucketAggregate, ...]


class RefreshWorker:
    """Coordinate pull-based refresh evaluation cycles."""

    def __init__(
        self,
        repository: CacheRepository,
        pipeline: RefreshPipeline,
        result_handler: Callable[[RefreshOutcome], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._pipeline = pipeline
        self._result_handler = result_handler

    async def run_once(
        self,
        *,
        limit: int = 200,
        lookback: timedelta = timedelta(days=7),
    ) -> RefreshWorkerReport:
        """Execute a single refresh cycle."""
        batch = await build_refresh_batch(
            self._repository,
            limit=limit,
            lookback=lookback,
        )
        processed = 0
        skipped = 0
        metrics: Sequence[BucketAggregate] = batch.bucket_metrics

        for candidate in batch.candidates:
            outcome = await self._pipeline.evaluate(
                candidate,
                bucket_metrics=metrics,
            )
            if outcome is None:
                skipped += 1
                continue
            await self._result_handler(outcome)
            processed += 1

        return RefreshWorkerReport(
            processed=processed,
            skipped=skipped,
            bucket_metrics=batch.bucket_metrics,
        )
