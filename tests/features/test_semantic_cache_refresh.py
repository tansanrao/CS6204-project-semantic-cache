from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.features.semantic_cache.models import feedback_event, policy_reward
from app.features.semantic_cache.repository import CacheRepository
from app.features.semantic_cache.service import SemanticCacheService
from app.features.semantic_cache.types import (
    CacheEntryCreate,
    RefreshEventType,
    RefreshOutcome,
    TTLDecisionCreate,
)


class DummyVectorStore:
    """Minimal vector-store stub for tests."""

    async def ensure_collection(self) -> None:  # pragma: no cover - no-op for tests
        return

    async def upsert_point(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError

    async def search(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class DummyEmbedder:
    """Minimal embedding stub to satisfy service ctor."""

    async def embed(self, texts):
        return [[0.0] for _ in texts]


@pytest_asyncio.fixture
async def repository(tmp_path_factory: pytest.TempPathFactory) -> CacheRepository:
    tmp_dir = tmp_path_factory.mktemp("refresh-service")
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


@pytest.mark.asyncio
async def test_log_refresh_outcome_persists_feedback_and_reward(
    repository: CacheRepository,
) -> None:
    service = build_service(repository)
    now = datetime.now(timezone.utc)

    entry_id = uuid4()
    await repository.insert_entry(
        CacheEntryCreate(
            id=entry_id,
            request_fingerprint="fp",
            prompt_hash="hash",
            model="gpt-4o",
            parameters={"temperature": 0.2},
            prompt_text="prompt",
            response_payload={"output": "cached"},
            ttl_bucket=4,
            ttl_seconds=3600,
            created_at=now - timedelta(hours=1),
            expires_at=now + timedelta(hours=1),
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
            request_fingerprint="fp",
            prompt_hash="hash",
            policy_name="linucb",
            policy_version="0.1.0",
            ttl_bucket=4,
            features={"length": 128},
            propensity=0.5,
            created_at=now - timedelta(minutes=30),
        )
    )

    outcome = RefreshOutcome(
        cache_entry_id=entry_id,
        ttl_decision_id=decision_id,
        stale=True,
        elapsed_seconds=900,
        ttl_seconds=3600,
        ttl_bucket=4,
        semantic_delta=0.7,
        fact_delta=0.6,
        cost_savings=0.4,
        latency_savings=0.3,
        feedback_event=RefreshEventType.STALE_CONFIRMED,
        feedback_score=-1.0,
        feedback_details={"semantic_delta": 0.7},
        obtained_at=now,
        bonus_applicable=False,
    )

    await service.log_refresh_outcome(outcome)

    async with repository._engine.connect() as connection:  # type: ignore[attr-defined]
        feedback_rows = (await connection.execute(select(feedback_event))).all()
        reward_rows = (await connection.execute(select(policy_reward))).all()

    assert len(feedback_rows) == 1
    feedback_record = feedback_rows[0]._mapping
    assert feedback_record["cache_entry_id"] == entry_id
    assert feedback_record["ttl_decision_id"] == decision_id
    assert feedback_record["event_type"] == RefreshEventType.STALE_CONFIRMED.value
    assert feedback_record["score"] == pytest.approx(-1.0)
    assert feedback_record["details"]["semantic_delta"] == pytest.approx(0.7)

    assert len(reward_rows) == 1
    reward_record = reward_rows[0]._mapping
    expected_reward = SemanticCacheService._compute_reward(outcome)
    assert reward_record["ttl_decision_id"] == decision_id
    assert reward_record["reward"] == pytest.approx(expected_reward)


@pytest.mark.asyncio
async def test_log_refresh_outcome_without_decision_skips_reward(
    repository: CacheRepository,
) -> None:
    service = build_service(repository)
    now = datetime.now(timezone.utc)

    entry_id = uuid4()
    await repository.insert_entry(
        CacheEntryCreate(
            id=entry_id,
            request_fingerprint="fp-none",
            prompt_hash="hash-none",
            model="gpt-4o",
            parameters={"temperature": 0.2},
            prompt_text="prompt",
            response_payload={"output": "cached"},
            ttl_bucket=5,
            ttl_seconds=7200,
            created_at=now - timedelta(hours=2),
            expires_at=now + timedelta(hours=4),
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
            embedding_mode="cls",
            embedding_dimension=768,
            similarity_threshold=0.86,
        )
    )

    outcome = RefreshOutcome(
        cache_entry_id=entry_id,
        ttl_decision_id=None,
        stale=False,
        elapsed_seconds=1800,
        ttl_seconds=7200,
        ttl_bucket=5,
        semantic_delta=0.1,
        fact_delta=0.0,
        cost_savings=0.5,
        latency_savings=0.2,
        feedback_event=RefreshEventType.FRESH_CONFIRMED,
        feedback_score=0.8,
        feedback_details={"semantic_delta": 0.1},
        obtained_at=now,
        bonus_applicable=True,
    )

    await service.log_refresh_outcome(outcome)

    async with repository._engine.connect() as connection:  # type: ignore[attr-defined]
        feedback_rows = (await connection.execute(select(feedback_event))).all()
        reward_rows = (await connection.execute(select(policy_reward))).all()

    assert len(feedback_rows) == 1
    assert reward_rows == []


@pytest.mark.asyncio
async def test_log_refresh_outcome_extends_ttl_on_bonus(
    repository: CacheRepository,
) -> None:
    service = build_service(repository)
    now = datetime.now(timezone.utc)

    entry_id = uuid4()
    await repository.insert_entry(
        CacheEntryCreate(
            id=entry_id,
            request_fingerprint="fp-extend",
            prompt_hash="hash-extend",
            model="gpt-4o",
            parameters={"temperature": 0.2},
            prompt_text="prompt",
            response_payload={"output": "cached"},
            ttl_bucket=0,
            ttl_seconds=60,
            created_at=now - timedelta(seconds=30),
            expires_at=now + timedelta(seconds=30),
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
            request_fingerprint="fp-extend",
            prompt_hash="hash-extend",
            policy_name="linucb",
            policy_version="0.1.0",
            ttl_bucket=0,
            features={"length": 128},
            propensity=0.4,
            created_at=now - timedelta(seconds=10),
        )
    )

    outcome = RefreshOutcome(
        cache_entry_id=entry_id,
        ttl_decision_id=decision_id,
        stale=False,
        elapsed_seconds=45,
        ttl_seconds=60,
        ttl_bucket=0,
        semantic_delta=0.05,
        fact_delta=0.05,
        cost_savings=0.6,
        latency_savings=0.4,
        feedback_event=RefreshEventType.FRESH_CONFIRMED,
        feedback_score=0.9,
        feedback_details={},
        obtained_at=now,
        bonus_applicable=True,
    )

    await service.log_refresh_outcome(outcome)

    extended_entry = await repository.get_entry(entry_id)
    assert extended_entry is not None
    assert extended_entry.ttl_bucket == 1
    assert extended_entry.ttl_seconds == 300

    async with repository._engine.connect() as connection:  # type: ignore[attr-defined]
        feedback_rows = (await connection.execute(select(feedback_event))).all()

    assert feedback_rows
    details = feedback_rows[0]._mapping["details"]
    assert "ttl_adjustments" in details
    assert details["ttl_adjustments"][0]["new_bucket"] == 1


@pytest.mark.asyncio
async def test_log_refresh_outcome_guardrail_shrinks_bucket(
    repository: CacheRepository,
) -> None:
    service = build_service(repository)
    now = datetime.now(timezone.utc)

    entry_id = uuid4()
    await repository.insert_entry(
        CacheEntryCreate(
            id=entry_id,
            request_fingerprint="fp-guard",
            prompt_hash="hash-guard",
            model="gpt-4o",
            parameters={"temperature": 0.2},
            prompt_text="prompt",
            response_payload={"output": "cached"},
            ttl_bucket=3,
            ttl_seconds=1800,
            created_at=now - timedelta(minutes=10),
            expires_at=now + timedelta(minutes=20),
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
            request_fingerprint="fp-guard",
            prompt_hash="hash-guard",
            policy_name="linucb",
            policy_version="0.1.0",
            ttl_bucket=3,
            features={"length": 64},
            propensity=0.3,
            created_at=now - timedelta(minutes=5),
        )
    )

    outcome = RefreshOutcome(
        cache_entry_id=entry_id,
        ttl_decision_id=decision_id,
        stale=True,
        elapsed_seconds=120,
        ttl_seconds=1800,
        ttl_bucket=3,
        semantic_delta=0.8,
        fact_delta=0.7,
        cost_savings=0.2,
        latency_savings=0.1,
        feedback_event=RefreshEventType.STALE_CONFIRMED,
        feedback_score=-0.8,
        feedback_details={},
        obtained_at=now,
        bonus_applicable=False,
    )

    await service.log_refresh_outcome(outcome)

    adjusted_entry = await repository.get_entry(entry_id)
    assert adjusted_entry is not None
    assert adjusted_entry.ttl_bucket == 2
    assert adjusted_entry.ttl_seconds == 900

    async with repository._engine.connect() as connection:  # type: ignore[attr-defined]
        feedback_rows = (await connection.execute(select(feedback_event))).all()

    guardrail_info = feedback_rows[0]._mapping["details"].get("guardrails")
    assert guardrail_info
    assert guardrail_info[0]["new_bucket"] == 2
