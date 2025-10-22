"""Tests for the FastAPI proxy against the vLLM backend."""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.main import create_app


@asynccontextmanager
async def _build_test_app(
    backend_handler: Callable[[httpx.Request], httpx.Response],
    settings: Settings,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an httpx.AsyncClient bound to the FastAPI app with a mock backend."""

    def client_factory() -> httpx.AsyncClient:
        transport = httpx.MockTransport(backend_handler)
        return httpx.AsyncClient(transport=transport)

    app = create_app(settings=settings, client_factory=client_factory)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


@pytest.mark.anyio
async def test_proxy_forwards_json_request() -> None:
    """The proxy should forward JSON payloads and pass through backend responses."""
    recorded: dict[str, Any] = {}

    def backend(request: httpx.Request) -> httpx.Response:
        recorded["method"] = request.method
        recorded["url"] = str(request.url)
        recorded["headers"] = dict(request.headers)
        recorded["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"id": "req-123", "object": "chat.completion"},
            headers={"x-backend": "1"},
        )

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        vllm_api_key="backend-secret",
        inbound_api_keys=["client-secret"],
        semantic_cache_enabled=False,
    )

    async with _build_test_app(backend, settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "ping"}],
            },
            headers={"Authorization": "Bearer client-secret"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["id"] == "req-123"
    assert response.headers["x-backend"] == "1"
    assert response.headers["x-cache"] == "BYPASS"
    assert response.headers["x-cache-status"] == "miss"
    assert recorded["method"] == "POST"
    assert recorded["url"] == "http://backend.local/v1/chat/completions"
    assert recorded["headers"]["authorization"] == "Bearer backend-secret"
    assert recorded["body"]["model"] == "gpt-test"
    metadata = payload["ttl_proxy"]
    assert metadata["cache_status"] == "miss"
    assert metadata["semantic_cache_enabled"] is False
    assert "semantic_cache_disabled" in metadata.get("reasons", [])


class _EventStream(httpx.AsyncByteStream):
    """Simple async byte stream for testing streaming responses."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self.aiter_bytes()

    async def aclose(self) -> None:
        return

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.anyio
async def test_proxy_streams_event_data() -> None:
    """Streaming responses should be relayed without buffering."""

    def backend(request: httpx.Request) -> httpx.Response:
        stream = _EventStream(
            [b"data: chunk-1\n\n", b"data: chunk-2\n\n", b"data: [DONE]\n\n"],
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        semantic_cache_enabled=False,
    )

    async with _build_test_app(backend, settings) as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "gpt-test", "stream": True},
        ) as response:
            chunks = [chunk async for chunk in response.aiter_raw()]
            payload = b"".join(chunks)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.headers["x-cache"] == "BYPASS"
    assert response.headers["x-cache-status"] == "miss"
    assert payload == b"data: chunk-1\n\ndata: chunk-2\n\ndata: [DONE]\n\n"
    assert len(chunks) >= 1


@pytest.mark.anyio
async def test_missing_authentication_rejected() -> None:
    """Requests must include a bearer token when inbound auth is configured."""

    def backend(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Backend should not be called on auth failure.")

    settings = Settings(
        vllm_base_url="http://backend.local/v1/",
        inbound_api_keys=["client-secret"],
        semantic_cache_enabled=False,
    )

    async with _build_test_app(backend, settings) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing bearer token."
