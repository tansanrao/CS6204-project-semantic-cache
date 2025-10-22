"""FastAPI application factory for the OpenAI-compatible proxy."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings, get_settings
from app.features.proxy.router import build_router
from app.features.proxy.service import ProxyService
from app.features.semantic_cache.embedding import load_embedding_service
from app.features.semantic_cache.qdrant import QdrantVectorStore
from app.features.semantic_cache.repository import CacheRepository
from app.features.semantic_cache.service import (
    SemanticCacheService,
    SemanticCacheSettings,
)

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> FastAPI:
    """Instantiate the FastAPI application."""
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        semantic_cache: SemanticCacheService | None = None
        cache_engine: AsyncEngine | None = None
        qdrant_client: QdrantClient | None = None

        if resolved_settings.semantic_cache_enabled:
            (
                semantic_cache,
                cache_engine,
                qdrant_client,
            ) = await _initialize_semantic_cache(resolved_settings)
            if semantic_cache:
                app.state.semantic_cache_service = semantic_cache

        if client_factory is not None:
            client = client_factory()
            app.state.proxy_service = ProxyService(
                settings=resolved_settings,
                client=client,
                semantic_cache=semantic_cache,
            )
            try:
                yield
            finally:
                await client.aclose()
        else:
            async with httpx.AsyncClient(
                timeout=resolved_settings.request_timeout_seconds
            ) as client:
                app.state.proxy_service = ProxyService(
                    settings=resolved_settings,
                    client=client,
                    semantic_cache=semantic_cache,
                )
                yield
        if cache_engine is not None:
            await cache_engine.dispose()
        if qdrant_client is not None:
            qdrant_client.close()
        if hasattr(app.state, "semantic_cache_service"):
            delattr(app.state, "semantic_cache_service")

    application = FastAPI(
        title="TTL Proxy Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(build_router())

    @application.get("/healthz", tags=["health"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()


async def _initialize_semantic_cache(
    settings: Settings,
) -> tuple[
    SemanticCacheService | None,
    AsyncEngine | None,
    QdrantClient | None,
]:
    """Instantiate the semantic cache stack if configuration is present."""
    if settings.database_dsn is None:
        logger.warning("Semantic cache enabled but PROXY_DATABASE_DSN is not set.")
        return None, None, None
    if settings.qdrant_url is None:
        logger.warning("Semantic cache enabled but PROXY_QDRANT_URL is not set.")
        return None, None, None

    engine = create_async_engine(
        str(settings.database_dsn),
        pool_pre_ping=True,
        future=True,
    )
    repository = CacheRepository(engine)
    qdrant_client = QdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key or None,
    )
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection_name,
        dimension=settings.embedding_dimension,
        similarity_threshold=settings.semantic_cache_similarity_threshold,
    )
    embedder = load_embedding_service(
        model_name=settings.embedding_model_name,
        mode=settings.embedding_model_mode,
        dimension=settings.embedding_dimension,
        device=settings.embedding_device,
    )
    cache_settings = SemanticCacheSettings(
        similarity_threshold=settings.semantic_cache_similarity_threshold,
        search_limit=settings.semantic_cache_search_limit,
        ttl_seconds=tuple(settings.semantic_cache_bucket_seconds),
        default_ttl_bucket=settings.semantic_cache_default_ttl_bucket,
    )
    service = SemanticCacheService(
        repository=repository,
        vector_store=vector_store,
        embedder=embedder,
        settings=cache_settings,
        embedding_model_name=settings.embedding_model_name,
        embedding_mode=settings.embedding_model_mode,
        embedding_dimension=settings.embedding_dimension,
    )

    if settings.semantic_cache_bootstrap:
        try:
            await service.bootstrap()
        except Exception as exc:  # pragma: no cover - startup guard
            logger.error("Semantic cache bootstrap failed: %s", exc, exc_info=True)
            await engine.dispose()
            qdrant_client.close()
            return None, None, None

    return service, engine, qdrant_client
