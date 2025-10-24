"""Startup regression tests for the FastAPI application."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.core.config import Settings
import app.main as main_module


class _StubEngine:
    """Async engine stub that records disposal."""

    def __init__(self) -> None:
        self.disposed = False
        self.url = SimpleNamespace(
            get_backend_name=lambda: "postgresql",
        )

    async def dispose(self) -> None:
        self.disposed = True

    def __str__(self) -> str:
        return "postgresql+asyncpg://postgres:postgres@localhost:5432/testdb"


class _StubRepository:
    """Minimal repository stub."""

    def __init__(self, engine) -> None:  # noqa: ANN001 - signature parity
        self.engine = engine
        self.schema_created = False

    async def create_schema(self) -> None:
        self.schema_created = True


class _StubQdrantClient:
    """Qdrant client stub that notes when it has been closed."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _StubVectorStore:
    """Vector store stub."""

    def __init__(
        self,
        client,
        collection_name,
        dimension,
        similarity_threshold,
    ) -> None:  # noqa: ANN001
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.ready = False

    async def ensure_collection(self) -> None:
        self.ready = True


class _TimeoutSemanticCacheService:
    """Semantic cache service stub that simulates a long bootstrap."""

    def __init__(
        self,
        repository,
        vector_store,
        embedder,
        *,
        settings,
        embedding_model_name,
        embedding_mode,
        embedding_dimension,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.embedder = embedder
        self.settings = settings
        self.embedding_model_name = embedding_model_name
        self.embedding_mode = embedding_mode
        self.embedding_dimension = embedding_dimension
        self.bootstrap_invocations = 0

    async def bootstrap(self) -> None:
        self.bootstrap_invocations += 1
        await asyncio.sleep(0.05)


@pytest.mark.anyio
async def test_initialize_semantic_cache_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap timeout should disable the semantic cache and clean up resources."""
    engine_holder: dict[str, _StubEngine] = {}
    client_holder: dict[str, _StubQdrantClient] = {}

    def fake_create_async_engine(url: str, **kwargs) -> _StubEngine:  # noqa: ANN001
        engine = _StubEngine()
        engine_holder["engine"] = engine
        return engine

    def fake_load_embedding_service(*args, **kwargs):  # noqa: ANN001
        return object()

    def fake_qdrant_client(*args, **kwargs) -> _StubQdrantClient:  # noqa: ANN001
        client = _StubQdrantClient(*args, **kwargs)
        client_holder["client"] = client
        return client

    monkeypatch.setattr(main_module, "create_async_engine", fake_create_async_engine)
    monkeypatch.setattr(main_module, "CacheRepository", _StubRepository)
    monkeypatch.setattr(
        main_module, "load_embedding_service", fake_load_embedding_service
    )
    monkeypatch.setattr(
        main_module, "SemanticCacheService", _TimeoutSemanticCacheService
    )
    monkeypatch.setattr(main_module, "QdrantClient", fake_qdrant_client)
    monkeypatch.setattr(main_module, "QdrantVectorStore", _StubVectorStore)

    settings = Settings(
        semantic_cache_enabled=True,
        semantic_cache_bootstrap=True,
        semantic_cache_bootstrap_timeout_seconds=0.01,
        database_dsn="postgresql+asyncpg://postgres:postgres@localhost:5432/testdb",
        qdrant_url="http://localhost:6333",
    )

    service, engine, client = await main_module._initialize_semantic_cache(settings)  # noqa: SLF001

    assert service is None
    assert engine is None
    assert client is None
    stub_engine = engine_holder["engine"]
    assert stub_engine.disposed
    stub_client = client_holder["client"]
    assert stub_client.closed
