"""Settings loading tests."""

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
