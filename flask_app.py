"""Flask application factory wiring the proxy, cache, and logging."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Callable
from uuid import uuid4

import atexit
import httpx
from flask import Flask, Response, g, request
from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.cache import SemanticCacheService, SemanticCacheSettings, load_embedding_service
from app.cache.qdrant import QdrantVectorStore
from app.cache.repository import CacheRepository
from app.config import Settings, get_settings
from app.logging import (
    REQUEST_ID_VAR,
    log_request_complete,
    log_request_start,
    setup_logging,
)
from app.proxy.router import create_blueprint
from app.proxy.service import ProxyService

LOG = logging.getLogger("app")


class _ResourceBundle:
    """Container tracking lifecycle-managed resources."""

    def __init__(
        self,
        *,
        http_client: httpx.Client,
        cache_service: SemanticCacheService | None,
        cache_engine: AsyncEngine | None,
        qdrant_client: QdrantClient | None,
    ) -> None:
        self.http_client = http_client
        self.cache_service = cache_service
        self.cache_engine = cache_engine
        self.qdrant_client = qdrant_client

    def close(self) -> None:
        """Dispose managed resources."""
        self.http_client.close()
        if self.cache_engine is not None:
            asyncio.run(self.cache_engine.dispose())
        if self.qdrant_client is not None:
            self.qdrant_client.close()


def create_app(
    settings: Settings | None = None,
    *,
    client_factory: Callable[[], httpx.Client] | None = None,
) -> Flask:
    """Construct the Flask application."""
    resolved_settings = settings or get_settings()
    setup_logging(level=resolved_settings.log_level)

    app = Flask(__name__)

    cache_service: SemanticCacheService | None = None
    cache_engine: AsyncEngine | None = None
    qdrant_client: QdrantClient | None = None

    if resolved_settings.semantic_cache_enabled:
        (
            cache_service,
            cache_engine,
            qdrant_client,
        ) = _initialize_semantic_cache(resolved_settings)
        if cache_service is not None:
            LOG.info(
                "semantic_cache.initialized",
                extra={
                    "kv": {
                        "event": "semantic_cache.initialized",
                        "policy": resolved_settings.semantic_cache_policy_type
                        if resolved_settings.semantic_cache_policy_enabled
                        else "disabled",
                    }
                },
            )

    http_client = client_factory() if client_factory else httpx.Client(
        timeout=resolved_settings.request_timeout_seconds
    )
    proxy_service = ProxyService(
        settings=resolved_settings,
        client=http_client,
        semantic_cache=cache_service,
    )

    bundle = _ResourceBundle(
        http_client=http_client,
        cache_service=cache_service,
        cache_engine=cache_engine,
        qdrant_client=qdrant_client,
    )
    app.config["proxy_service"] = proxy_service
    app.config["_resource_bundle"] = bundle
    atexit.register(bundle.close)

    app.register_blueprint(create_blueprint())

    @app.route("/healthz", methods=["GET"])
    def healthcheck() -> Response:
        return Response(
            response='{"status":"ok"}',
            status=200,
            mimetype="application/json",
        )

    @app.before_request
    def _before_request() -> None:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        g.request_id = request_id
        REQUEST_ID_VAR.set(request_id)
        g.request_start = perf_counter()
        log_request_start(
            request_id=request_id,
            method=request.method,
            path=request.path,
            client_ip=request.headers.get("X-Forwarded-For", request.remote_addr or "-"),
            ua=request.headers.get("User-Agent", "-"),
        )

    @app.after_request
    def _after_request(response: Response) -> Response:
        latency_ms = (perf_counter() - g.get("request_start", perf_counter())) * 1000
        request_id = g.get("request_id", "-")
        log_request_complete(
            request_id=request_id,
            method=request.method,
            path=request.path,
            status=response.status_code,
            latency_ms=latency_ms,
            bytes_out=response.calculate_content_length() or 0,
        )
        if request_id:
            response.headers.setdefault("X-Request-ID", request_id)
        REQUEST_ID_VAR.set("-")
        return response

    @app.teardown_appcontext
    def _teardown(_exc: Exception | None) -> None:
        if REQUEST_ID_VAR.get("-") != "-":
            REQUEST_ID_VAR.set("-")

    return app


def _initialize_semantic_cache(
    settings: Settings,
) -> tuple[
    SemanticCacheService | None,
    AsyncEngine | None,
    QdrantClient | None,
]:
    """Instantiate the semantic cache components synchronously."""
    if settings.database_dsn is None:
        LOG.warning(
            "semantic_cache.missing_database_dsn",
            extra={"kv": {"event": "semantic_cache.missing_database_dsn"}},
        )
        return None, None, None
    if settings.qdrant_url is None:
        LOG.warning(
            "semantic_cache.missing_qdrant_url",
            extra={"kv": {"event": "semantic_cache.missing_qdrant_url"}},
        )
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
        policy_enabled=settings.semantic_cache_policy_enabled,
        policy_type=settings.semantic_cache_policy_type,
        policy_feature_dimension=settings.semantic_cache_policy_feature_dimension,
        policy_allow_feature_growth=settings.semantic_cache_policy_allow_feature_growth,
        policy_snapshot_path=settings.semantic_cache_policy_snapshot_path,
        policy_autosave_interval=settings.semantic_cache_policy_autosave_interval,
        linucb_alpha=settings.semantic_cache_linucb_alpha,
        linucb_regularization=settings.semantic_cache_linucb_regularization,
        linucb_min_propensity=settings.semantic_cache_linucb_min_propensity,
        ts_regularization=settings.semantic_cache_ts_regularization,
        ts_sampling_variance=settings.semantic_cache_ts_sampling_variance,
        ts_min_propensity=settings.semantic_cache_ts_min_propensity,
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
            asyncio.run(
                asyncio.wait_for(
                    service.bootstrap(),
                    timeout=settings.semantic_cache_bootstrap_timeout_seconds,
                )
            )
        except TimeoutError:
            LOG.error(
                "semantic_cache.bootstrap_timeout",
                extra={
                    "kv": {
                        "event": "semantic_cache.bootstrap_timeout",
                        "timeout_s": settings.semantic_cache_bootstrap_timeout_seconds,
                    }
                },
            )
            asyncio.run(engine.dispose())
            qdrant_client.close()
            return None, None, None
        except Exception as exc:  # pragma: no cover - defensive guard
            LOG.error(
                "semantic_cache.bootstrap_error",
                extra={
                    "kv": {
                        "event": "semantic_cache.bootstrap_error",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                },
            )
            asyncio.run(engine.dispose())
            qdrant_client.close()
            return None, None, None

    return service, engine, qdrant_client


app = create_app()
