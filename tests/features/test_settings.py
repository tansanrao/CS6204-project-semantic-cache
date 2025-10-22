"""Settings loading tests."""

from pathlib import Path

from app.core.config import Settings


def test_settings_load_env_file(tmp_path: Path) -> None:
    """Settings should load values from a standard .env file."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "PROXY_VLLM_BASE_URL=http://example.com/api/",
                "PROXY_INBOUND_API_KEYS=key1,key2",
                "PROXY_REQUEST_TIMEOUT_SECONDS=15",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_path)

    assert str(settings.vllm_base_url) == "http://example.com/api/"
    assert settings.inbound_api_keys == ["key1", "key2"]
    assert settings.request_timeout_seconds == 15
