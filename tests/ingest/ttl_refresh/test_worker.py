from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.cache.models import feedback_event, policy_reward
from app.cache.repository import CacheRepository
from app.cache.service import SemanticCacheService
from app.cache.types import (
    CacheEntryCreate,
    RefreshEventType,
    RefreshOutcome,
    TTLDecisionCreate,
)
from app.tasks.worker import RefreshWorker


class DummyVectorStore:
    async def ensure_collection(self) -> None:  # pragma: no cover - no-op
        return

    async def upsert_point(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError

    async def search(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class DummyEmbedder:
    async def embed(self, texts):
        return [[0.0] for _ in texts]


@pytest_asyncio.fixture
async def repository(tmp_path_factory: pytest.TempPathFactory) -> CacheRepository:
    tmp_dir = tmp_path_factory.mktemp("refresh-worker")
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_dir / 'db.sqlite'}",
        future=True,
    )
    repo = CacheRepository(engine)
    await repo.create_schema()
    yield repo
    await engine.dispose()


def build_service(repository: CacheRepository) -> SemanticCacheService:
    return SemanticCacheService(
        repository=repository,
        vector_store=DummyVectorStore(),  # type: ignore[arg-type]
        embedder=DummyEmbedder(),  # type: ignore[arg-type]
        embedding_model_name="dummy",
        embedding_mode="cls",
        embedding_dimension=1,
    )


class StubPipeline:
    def __init__(self, decision_id: UUID, obtained_at: datetime) -> None:
        self.decision_id = decision_id
        self.obtained_at = obtained_at
        self.calls: list[UUID] = []

    async def evaluate(self, candidate, *, bucket_metrics):
        created_at = candidate.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        elapsed = int((self.obtained_at - created_at).total_seconds())
        self.calls.append(candidate.id)
        return RefreshOutcome(
            cache_entry_id=candidate.id,
            ttl_decision_id=self.decision_id,
            stale=False,
            elapsed_seconds=elapsed,
            ttl_seconds=candidate.ttl_seconds,
            ttl_bucket=candidate.ttl_bucket,
            semantic_delta=0.1,
            fact_delta=0.05,
            cost_savings=0.6,
            latency_savings=0.4,
            feedback_event=RefreshEventType.FRESH_CONFIRMED,
            feedback_score=0.9,
            feedback_details={"semantic_delta": 0.1},
            obtained_at=self.obtained_at,
            bonus_applicable=True,
        )


@pytest.mark.asyncio
async def test_refresh_worker_runs_pipeline_and_logs_results(
    repository: CacheRepository,
) -> None:
    service = build_service(repository)
    now = datetime.now(timezone.utc)

    entry_id = uuid4()
    await repository.insert_entry(
        CacheEntryCreate(
            id=entry_id,
            request_fingerprint="worker-fp",
            prompt_hash="worker-hash",
            model="gpt-4o",
            parameters={"temperature": 0.2},
            prompt_text="prompt",
            response_payload={"answer": "cached"},
            ttl_bucket=2,
            ttl_seconds=7200,
            created_at=now - timedelta(hours=2),
            expires_at=now + timedelta(hours=2),
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
            embedding_mode="cls",
            embedding_dimension=768,
            similarity_threshold=0.86,
        )
    )

    decision_id = uuid4()
    await repository.insert_ttl_decision(
        TTLDecisionCreate(
            id=decision_id,
            cache_entry_id=entry_id,
            request_fingerprint="worker-fp",
            prompt_hash="worker-hash",
            policy_name="linucb",
            policy_version="0.1.0",
            ttl_bucket=2,
            features={"length": 42},
            propensity=0.3,
            created_at=now - timedelta(minutes=5),
        )
    )

    pipeline = StubPipeline(decision_id, obtained_at=now)
    handled: list[RefreshOutcome] = []

    async def handler(outcome: RefreshOutcome) -> None:
        handled.append(outcome)
        await service.log_refresh_outcome(outcome)

    worker = RefreshWorker(repository, pipeline, handler)

    report = await worker.run_once(limit=10, lookback=timedelta(days=7))

    assert report.processed == 1
    assert report.skipped == 0
    assert handled[0].cache_entry_id == entry_id
    assert pipeline.calls == [entry_id]

    async with repository._engine.connect() as connection:  # type: ignore[attr-defined]
        feedback_rows = (await connection.execute(select(feedback_event))).all()
        reward_rows = (await connection.execute(select(policy_reward))).all()

    assert len(feedback_rows) == 1
    assert len(reward_rows) == 1
    assert (
        feedback_rows[0]._mapping["event_type"]
        == RefreshEventType.FRESH_CONFIRMED.value
    )
    assert reward_rows[0]._mapping["ttl_decision_id"] == decision_id
