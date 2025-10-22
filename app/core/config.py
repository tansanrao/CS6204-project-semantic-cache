"""Application configuration loaded from the environment."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, HttpUrl, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource, EnvSettingsSource


class LenientEnvSettingsSource(EnvSettingsSource):
    """Environment source that tolerates non-JSON complex values."""

    def decode_complex_value(self, field_name, field, value):  # type: ignore[override]
        try:
            return super().decode_complex_value(field_name, field, value)
        except ValueError:
            return value


class LenientDotEnvSettingsSource(DotEnvSettingsSource):
    """Dotenv source that tolerates non-JSON complex values."""

    def decode_complex_value(self, field_name, field, value):  # type: ignore[override]
        try:
            return super().decode_complex_value(field_name, field, value)
        except ValueError:
            return value


class Settings(BaseSettings):
    """Runtime settings for the proxy service."""

    model_config = SettingsConfigDict(
        env_prefix="proxy_",
        env_file=".env",
        extra="ignore",
    )

    vllm_base_url: HttpUrl = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    inbound_api_keys: list[str] = Field(default_factory=list)
    request_timeout_seconds: float = 30.0
    database_dsn: PostgresDsn | None = None
    qdrant_url: AnyHttpUrl | None = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "cache_entries"
    semantic_cache_enabled: bool = False
    semantic_cache_bootstrap: bool = True
    embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_model_mode: Literal["clustering", "search"] = "clustering"
    embedding_dimension: int = 768
    embedding_device: str = "cpu"
    semantic_cache_similarity_threshold: float = 0.86
    semantic_cache_search_limit: int = 5
    semantic_cache_default_ttl_bucket: int = 2
    semantic_cache_bucket_seconds: list[int] = Field(
        default_factory=lambda: [60, 300, 900, 1800, 3600, 7200]
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        env_source = LenientEnvSettingsSource(
            settings_cls,
            case_sensitive=getattr(env_settings, "case_sensitive", None),
            env_prefix=getattr(env_settings, "env_prefix", None),
            env_nested_delimiter=getattr(env_settings, "env_nested_delimiter", None),
            env_nested_max_split=getattr(env_settings, "env_nested_max_split", None),
            env_ignore_empty=getattr(env_settings, "env_ignore_empty", None),
            env_parse_none_str=getattr(env_settings, "env_parse_none_str", None),
            env_parse_enums=getattr(env_settings, "env_parse_enums", None),
        )

        dotenv_source = LenientDotEnvSettingsSource(
            settings_cls,
            env_file=getattr(dotenv_settings, "env_file", None)
            or cls.model_config.get("env_file"),
            env_file_encoding=getattr(dotenv_settings, "env_file_encoding", None),
            case_sensitive=getattr(dotenv_settings, "case_sensitive", None),
            env_prefix=getattr(dotenv_settings, "env_prefix", None),
            env_nested_delimiter=getattr(dotenv_settings, "env_nested_delimiter", None),
            env_nested_max_split=getattr(dotenv_settings, "env_nested_max_split", None),
            env_ignore_empty=getattr(dotenv_settings, "env_ignore_empty", None),
            env_parse_none_str=getattr(dotenv_settings, "env_parse_none_str", None),
            env_parse_enums=getattr(dotenv_settings, "env_parse_enums", None),
        )

        return (
            init_settings,
            env_source,
            dotenv_source,
            file_secret_settings,
        )

    @field_validator("inbound_api_keys", mode="before")
    @classmethod
    def _split_inbound_keys(
        cls, value: Annotated[str | list[str] | None, "Inbound API keys"]
    ) -> list[str]:
        """Allow either CSV string or list input for inbound API keys."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [segment.strip() for segment in value.split(",") if segment.strip()]
        return list(value)

    @field_validator("semantic_cache_bucket_seconds", mode="before")
    @classmethod
    def _parse_bucket_seconds(
        cls, value: Annotated[str | list[int], "TTL bucket seconds"]
    ) -> list[int]:
        """Allow either CSV strings or lists for TTL bucket configuration."""
        if isinstance(value, str):
            parts = [segment.strip() for segment in value.split(",") if segment.strip()]
            return [int(part) for part in parts]
        return list(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
