"""Dataclass-based runtime settings loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Mapping


def _lookup_env(
    env: Mapping[str, str],
    name: str,
    *,
    prefix: str = "PROXY_",
) -> str | None:
    """Return an environment value using either upper- or lower-case keys."""
    candidates = (
        f"{prefix}{name}",
        f"{prefix}{name}".lower(),
    )
    for candidate in candidates:
        if candidate in env:
            return env[candidate]
    return None


def _parse_bool(value: str | None, default: bool) -> bool:
    """Convert a string to a boolean flag."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_float(value: str | None, default: float) -> float:
    """Parse a floating point configuration value."""
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_int(value: str | None, default: int) -> int:
    """Parse an integer configuration value."""
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_list(value: str | None) -> list[str]:
    """Parse a comma-separated list of values."""
    if value is None or value.strip() == "":
        return []
    return [
        segment.strip()
        for segment in value.split(",")
        if segment.strip()
    ]


def _parse_int_list(value: str | None, default: Iterable[int]) -> list[int]:
    """Parse a comma-separated list of integers."""
    if value is None or value.strip() == "":
        return list(default)
    items: list[int] = []
    for segment in value.split(","):
        segment = segment.strip()
        if not segment:
            continue
        try:
            items.append(int(segment))
        except ValueError:
            continue
    return items or list(default)


@dataclass(slots=True)
class Settings:
    """Runtime configuration for the Flask proxy."""

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    inbound_api_keys: list[str] = field(default_factory=list)
    request_timeout_seconds: float = 30.0
    log_level: str = "INFO"

    database_dsn: str | None = None
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "cache_entries"

    semantic_cache_enabled: bool = False
    semantic_cache_bootstrap: bool = True
    semantic_cache_bootstrap_timeout_seconds: float = 20.0

    embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_model_mode: str = "clustering"
    embedding_dimension: int = 768
    embedding_device: str = "cpu"

    semantic_cache_similarity_threshold: float = 0.86
    semantic_cache_search_limit: int = 5
    semantic_cache_default_ttl_bucket: int = 2
    semantic_cache_bucket_seconds: list[int] = field(
        default_factory=lambda: [60, 300, 900, 1800, 3600, 7200]
    )

    semantic_cache_policy_enabled: bool = True
    semantic_cache_policy_type: str = "linucb"
    semantic_cache_policy_feature_dimension: int = 256
    semantic_cache_policy_allow_feature_growth: bool = True
    semantic_cache_policy_snapshot_path: str | None = None
    semantic_cache_policy_autosave_interval: int = 50

    semantic_cache_linucb_alpha: float = 0.6
    semantic_cache_linucb_regularization: float = 1.0
    semantic_cache_linucb_min_propensity: float = 1e-3

    semantic_cache_ts_regularization: float = 1.0
    semantic_cache_ts_sampling_variance: float = 1.0
    semantic_cache_ts_min_propensity: float = 1e-3

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "Settings":
        """Construct settings from environment variables."""
        env_map: Mapping[str, str] = env or os.environ

        kwargs: MutableMapping[str, object] = {}

        base_url = _lookup_env(env_map, "VLLM_BASE_URL")
        if base_url:
            kwargs["vllm_base_url"] = base_url

        vllm_api_key = _lookup_env(env_map, "VLLM_API_KEY")
        if vllm_api_key:
            kwargs["vllm_api_key"] = vllm_api_key

        inbound_keys = _lookup_env(env_map, "INBOUND_API_KEYS")
        if inbound_keys is not None:
            kwargs["inbound_api_keys"] = _parse_list(inbound_keys)

        timeout = _lookup_env(env_map, "REQUEST_TIMEOUT_SECONDS")
        if timeout is not None:
            kwargs["request_timeout_seconds"] = _parse_float(
                timeout,
                cls.request_timeout_seconds,
            )

        log_level = _lookup_env(env_map, "LOG_LEVEL")
        if log_level:
            kwargs["log_level"] = log_level.upper()

        database_dsn = _lookup_env(env_map, "DATABASE_DSN")
        if database_dsn:
            kwargs["database_dsn"] = database_dsn

        qdrant_url = _lookup_env(env_map, "QDRANT_URL")
        if qdrant_url:
            kwargs["qdrant_url"] = qdrant_url

        qdrant_api_key = _lookup_env(env_map, "QDRANT_API_KEY")
        if qdrant_api_key:
            kwargs["qdrant_api_key"] = qdrant_api_key

        collection_name = _lookup_env(env_map, "QDRANT_COLLECTION_NAME")
        if collection_name:
            kwargs["qdrant_collection_name"] = collection_name

        semantic_cache_enabled = _lookup_env(env_map, "SEMANTIC_CACHE_ENABLED")
        if semantic_cache_enabled is not None:
            kwargs["semantic_cache_enabled"] = _parse_bool(
                semantic_cache_enabled,
                cls.semantic_cache_enabled,
            )

        semantic_cache_bootstrap = _lookup_env(env_map, "SEMANTIC_CACHE_BOOTSTRAP")
        if semantic_cache_bootstrap is not None:
            kwargs["semantic_cache_bootstrap"] = _parse_bool(
                semantic_cache_bootstrap,
                cls.semantic_cache_bootstrap,
            )

        bootstrap_timeout = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_BOOTSTRAP_TIMEOUT_SECONDS",
        )
        if bootstrap_timeout is not None:
            kwargs["semantic_cache_bootstrap_timeout_seconds"] = _parse_float(
                bootstrap_timeout,
                cls.semantic_cache_bootstrap_timeout_seconds,
            )

        embedding_model_name = _lookup_env(env_map, "EMBEDDING_MODEL_NAME")
        if embedding_model_name:
            kwargs["embedding_model_name"] = embedding_model_name

        embedding_model_mode = _lookup_env(env_map, "EMBEDDING_MODEL_MODE")
        if embedding_model_mode:
            kwargs["embedding_model_mode"] = embedding_model_mode

        embedding_dimension = _lookup_env(env_map, "EMBEDDING_DIMENSION")
        if embedding_dimension is not None:
            kwargs["embedding_dimension"] = _parse_int(
                embedding_dimension,
                cls.embedding_dimension,
            )

        embedding_device = _lookup_env(env_map, "EMBEDDING_DEVICE")
        if embedding_device:
            kwargs["embedding_device"] = embedding_device

        similarity_threshold = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_SIMILARITY_THRESHOLD",
        )
        if similarity_threshold is not None:
            kwargs["semantic_cache_similarity_threshold"] = _parse_float(
                similarity_threshold,
                cls.semantic_cache_similarity_threshold,
            )

        search_limit = _lookup_env(env_map, "SEMANTIC_CACHE_SEARCH_LIMIT")
        if search_limit is not None:
            kwargs["semantic_cache_search_limit"] = _parse_int(
                search_limit,
                cls.semantic_cache_search_limit,
            )

        default_bucket = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_DEFAULT_TTL_BUCKET",
        )
        if default_bucket is not None:
            kwargs["semantic_cache_default_ttl_bucket"] = _parse_int(
                default_bucket,
                cls.semantic_cache_default_ttl_bucket,
            )

        bucket_seconds = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_BUCKET_SECONDS",
        )
        if bucket_seconds is not None:
            kwargs["semantic_cache_bucket_seconds"] = _parse_int_list(
                bucket_seconds,
                cls().semantic_cache_bucket_seconds,
            )

        policy_enabled = _lookup_env(env_map, "SEMANTIC_CACHE_POLICY_ENABLED")
        if policy_enabled is not None:
            kwargs["semantic_cache_policy_enabled"] = _parse_bool(
                policy_enabled,
                cls.semantic_cache_policy_enabled,
            )

        policy_type = _lookup_env(env_map, "SEMANTIC_CACHE_POLICY_TYPE")
        if policy_type:
            kwargs["semantic_cache_policy_type"] = policy_type

        policy_dim = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_POLICY_FEATURE_DIMENSION",
        )
        if policy_dim is not None:
            kwargs["semantic_cache_policy_feature_dimension"] = _parse_int(
                policy_dim,
                cls.semantic_cache_policy_feature_dimension,
            )

        policy_growth = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_POLICY_ALLOW_FEATURE_GROWTH",
        )
        if policy_growth is not None:
            kwargs["semantic_cache_policy_allow_feature_growth"] = _parse_bool(
                policy_growth,
                cls.semantic_cache_policy_allow_feature_growth,
            )

        policy_snapshot = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_POLICY_SNAPSHOT_PATH",
        )
        if policy_snapshot:
            kwargs["semantic_cache_policy_snapshot_path"] = policy_snapshot

        policy_autosave = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_POLICY_AUTOSAVE_INTERVAL",
        )
        if policy_autosave is not None:
            kwargs["semantic_cache_policy_autosave_interval"] = _parse_int(
                policy_autosave,
                cls.semantic_cache_policy_autosave_interval,
            )

        linucb_alpha = _lookup_env(env_map, "SEMANTIC_CACHE_LINUCB_ALPHA")
        if linucb_alpha is not None:
            kwargs["semantic_cache_linucb_alpha"] = _parse_float(
                linucb_alpha,
                cls.semantic_cache_linucb_alpha,
            )

        linucb_reg = _lookup_env(env_map, "SEMANTIC_CACHE_LINUCB_REGULARIZATION")
        if linucb_reg is not None:
            kwargs["semantic_cache_linucb_regularization"] = _parse_float(
                linucb_reg,
                cls.semantic_cache_linucb_regularization,
            )

        linucb_min_prop = _lookup_env(
            env_map,
            "SEMANTIC_CACHE_LINUCB_MIN_PROPENSITY",
        )
        if linucb_min_prop is not None:
            kwargs["semantic_cache_linucb_min_propensity"] = _parse_float(
                linucb_min_prop,
                cls.semantic_cache_linucb_min_propensity,
            )

        ts_reg = _lookup_env(env_map, "SEMANTIC_CACHE_TS_REGULARIZATION")
        if ts_reg is not None:
            kwargs["semantic_cache_ts_regularization"] = _parse_float(
                ts_reg,
                cls.semantic_cache_ts_regularization,
            )

        ts_variance = _lookup_env(env_map, "SEMANTIC_CACHE_TS_SAMPLING_VARIANCE")
        if ts_variance is not None:
            kwargs["semantic_cache_ts_sampling_variance"] = _parse_float(
                ts_variance,
                cls.semantic_cache_ts_sampling_variance,
            )

        ts_min_prop = _lookup_env(env_map, "SEMANTIC_CACHE_TS_MIN_PROPENSITY")
        if ts_min_prop is not None:
            kwargs["semantic_cache_ts_min_propensity"] = _parse_float(
                ts_min_prop,
                cls.semantic_cache_ts_min_propensity,
            )

        return cls(**kwargs)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings built from the current environment."""
    return Settings.from_env()
