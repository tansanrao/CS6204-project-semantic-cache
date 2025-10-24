"""Settings loading tests."""

from dotenv import load_dotenv

from app.config import Settings


def test_settings_from_env_mapping() -> None:
    """Settings should parse CSV lists and numeric values from mappings."""

    env = {
        "PROXY_VLLM_BASE_URL": "http://example.com/api/",
        "PROXY_INBOUND_API_KEYS": "key1,key2",
        "PROXY_REQUEST_TIMEOUT_SECONDS": "15",
    }

    settings = Settings.from_env(env)

    assert settings.vllm_base_url == "http://example.com/api/"
    assert settings.inbound_api_keys == ["key1", "key2"]
    assert settings.request_timeout_seconds == 15


def test_settings_from_env_file(monkeypatch, tmp_path) -> None:
    """Settings should respect variables loaded via python-dotenv."""

    env_file = tmp_path / "test.env"
    env_file.write_text(
        "\n".join(
            [
                "# sample config",
                "PROXY_SEMANTIC_CACHE_ENABLED=true",
                "PROXY_DATABASE_DSN=postgresql+asyncpg://postgres:postgres@localhost:5432/cache",
            ]
        ),
        encoding="utf-8",
    )

    load_dotenv(env_file, override=True)

    settings = Settings.from_env()

    assert settings.semantic_cache_enabled is True
    assert (
        settings.database_dsn
        == "postgresql+asyncpg://postgres:postgres@localhost:5432/cache"
    )

    monkeypatch.delenv("PROXY_SEMANTIC_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("PROXY_DATABASE_DSN", raising=False)
