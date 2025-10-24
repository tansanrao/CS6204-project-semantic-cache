"""Tests for the request logging hooks in the Flask app."""

from __future__ import annotations

import logging

from app.config import Settings
from flask_app import create_app


def test_request_log_emits_request_id(caplog) -> None:
    """Request lifecycle logs should include the propagated request id."""
    app = create_app(settings=Settings(semantic_cache_enabled=False))

    with caplog.at_level(logging.INFO, logger="app.proxy"):
        with app.test_client() as client:
            response = client.get(
                "/healthz",
                headers={"X-Request-ID": "abc123"},
            )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "abc123"

    matching = [
        record
        for record in caplog.records
        if getattr(record, "kv", {}).get("event") == "proxy.request_complete"
    ]
    assert matching, "expected proxy completion log"
    record = matching[0]
    assert record.kv["request_id"] == "abc123"
    assert record.kv["status"] == 200
    assert record.kv["method"] == "GET"
    assert record.kv["path"] == "/healthz"
