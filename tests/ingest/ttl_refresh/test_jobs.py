from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from app.cache.repository import CacheRepository
from app.cache.types import (
    BucketAggregate,
    CacheEntryCreate,
    FeedbackEventCreate,
    PolicyRewardCreate,
    TTLDecisionCreate,
)
from app.tasks.jobs import RefreshBatch, build_refresh_batch


@pytest_asyncio.fixture
async def repository(tmp_path_factory: pytest.TempPathFactory) -> CacheRepository:
    """Provide a CacheRepository backed by a temporary SQLite database."""
    tmp_dir = tmp_path_factory.mktemp("db")
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_dir / 'semantic_cache.db'}",
        future=True,
    )
    repo = CacheRepository(engine)
    await repo.create_schema()
    yield repo
    await engine.dispose()


@pytest.mark.asyncio
async def test_list_refresh_candidates_filters_outdated(
    repository: CacheRepository,
) -> None:
    now = datetime.now(timezone.utc)

    recent_entry = CacheEntryCreate(
        id=uuid4(),
        request_fingerprint="recent-fp",
        prompt_hash="recent-hash",
        model="gpt-4o",
        parameters={"temperature": 0.2},
        prompt_text="recent prompt",
        response_payload={"answer": "cached"},
        ttl_bucket=1,
        ttl_seconds=3600,
        created_at=now - timedelta(days=1),
        expires_at=now + timedelta(hours=1),
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_mode="cls",
        embedding_dimension=768,
        similarity_threshold=0.86,
    )
    expired_entry = CacheEntryCreate(
        id=uuid4(),
        request_fingerprint="expired-fp",
        prompt_hash="expired-hash",
        model="gpt-4o",
        parameters={"temperature": 0.2},
        prompt_text="expired prompt",
        response_payload={"answer": "expired"},
        ttl_bucket=2,
        ttl_seconds=7200,
        created_at=now - timedelta(days=2),
        expires_at=now - timedelta(hours=1),
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_mode="cls",
        embedding_dimension=768,
        similarity_threshold=0.86,
    )
    stale_entry = CacheEntryCreate(
        id=uuid4(),
        request_fingerprint="old-fp",
        prompt_hash="old-hash",
        model="gpt-4o",
        parameters={"temperature": 0.2},
        prompt_text="stale prompt",
        response_payload={"answer": "stale"},
        ttl_bucket=3,
        ttl_seconds=10800,
        created_at=now - timedelta(days=10),
        expires_at=now + timedelta(days=1),
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_mode="cls",
        embedding_dimension=768,
        similarity_threshold=0.86,
    )

    await repository.insert_entry(recent_entry)
    await repository.insert_entry(expired_entry)
    await repository.insert_entry(stale_entry)

    candidates = await repository.list_refresh_candidates(
        limit=10,
        lookback=timedelta(days=7),
    )
    assert [candidate.id for candidate in candidates] == [recent_entry.id]


@pytest.mark.asyncio
async def test_aggregate_bucket_metrics_returns_counts(
    repository: CacheRepository,
) -> None:
    now = datetime.now(timezone.utc)

    # Cache entry referenced by feedback events.
    entry_id = uuid4()
    await repository.insert_entry(
        CacheEntryCreate(
            id=entry_id,
            request_fingerprint="fp",
            prompt_hash="hash",
            model="gpt-4o",
            parameters={"temperature": 0.2},
            prompt_text="prompt",
            response_payload={"answer": "cached"},
            ttl_bucket=1,
            ttl_seconds=3600,
            created_at=now - timedelta(days=1),
            expires_at=now + timedelta(days=1),
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
            embedding_mode="cls",
            embedding_dimension=768,
            similarity_threshold=0.86,
        )
    )

    decision_bucket_one = uuid4()
    decision_bucket_one_b = uuid4()
    decision_bucket_three = uuid4()

    await repository.insert_ttl_decision(
        TTLDecisionCreate(
            id=decision_bucket_one,
            cache_entry_id=None,
            request_fingerprint="fp-1",
            prompt_hash="hash-1",
            policy_name="linucb",
            policy_version="0.1.0",
            ttl_bucket=1,
            features={"length": 128},
            propensity=0.4,
            created_at=now - timedelta(hours=2),
        )
    )
    await repository.insert_ttl_decision(
        TTLDecisionCreate(
            id=decision_bucket_one_b,
            cache_entry_id=None,
            request_fingerprint="fp-2",
            prompt_hash="hash-2",
            policy_name="linucb",
            policy_version="0.1.0",
            ttl_bucket=1,
            features={"length": 256},
            propensity=0.5,
            created_at=now - timedelta(hours=1),
        )
    )
    await repository.insert_ttl_decision(
        TTLDecisionCreate(
            id=decision_bucket_three,
            cache_entry_id=None,
            request_fingerprint="fp-3",
            prompt_hash="hash-3",
            policy_name="linucb",
            policy_version="0.1.0",
            ttl_bucket=3,
            features={"length": 64},
            propensity=None,
            created_at=now - timedelta(minutes=30),
        )
    )

    await repository.insert_feedback_event(
        FeedbackEventCreate(
            id=uuid4(),
            cache_entry_id=entry_id,
            ttl_decision_id=decision_bucket_one,
            event_type="stale_detected",
            score=-1.0,
            details={"semantic_delta": 0.7},
            created_at=now - timedelta(minutes=15),
        )
    )

    await repository.insert_policy_reward(
        PolicyRewardCreate(
            id=uuid4(),
            ttl_decision_id=decision_bucket_one,
            reward=0.5,
            attribution_rule="stale_check_v1",
            created_at=now - timedelta(minutes=10),
        )
    )

    metrics = await repository.aggregate_bucket_metrics(
        lookback=timedelta(days=30),
    )

    assert metrics == [
        BucketAggregate(
            ttl_bucket=1,
            decision_count=2,
            feedback_count=1,
            avg_reward=pytest.approx(0.5),
        ),
        BucketAggregate(
            ttl_bucket=3,
            decision_count=1,
            feedback_count=0,
            avg_reward=None,
        ),
    ]


@pytest.mark.asyncio
async def test_build_refresh_batch_combines_results(
    repository: CacheRepository,
) -> None:
    now = datetime.now(timezone.utc)
    entry = CacheEntryCreate(
        id=uuid4(),
        request_fingerprint="batch-fp",
        prompt_hash="batch-hash",
        model="gpt-4o",
        parameters={"temperature": 0.2},
        prompt_text="batch prompt",
        response_payload={"answer": "cached"},
        ttl_bucket=2,
        ttl_seconds=7200,
        created_at=now - timedelta(hours=3),
        expires_at=now + timedelta(hours=2),
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_mode="cls",
        embedding_dimension=768,
        similarity_threshold=0.86,
    )
    await repository.insert_entry(entry)
    await repository.insert_ttl_decision(
        TTLDecisionCreate(
            id=uuid4(),
            cache_entry_id=None,
            request_fingerprint="batch-fp",
            prompt_hash="batch-hash",
            policy_name="linucb",
            policy_version="0.1.0",
            ttl_bucket=2,
            features={"length": 42},
            propensity=0.3,
            created_at=now - timedelta(minutes=5),
        )
    )

    batch = await build_refresh_batch(
        repository,
        limit=5,
        lookback=timedelta(days=7),
    )
    assert isinstance(batch, RefreshBatch)
    assert tuple(candidate.id for candidate in batch.candidates) == (entry.id,)
    assert batch.bucket_metrics[0].ttl_bucket == 2
