"""Validate the semantic cache database schema."""

from __future__ import annotations

from sqlalchemy import UniqueConstraint

from app.cache import models


def test_cache_entry_table_has_expected_columns() -> None:
    """Ensure the cache_entry table includes critical columns and constraints."""
    table = models.cache_entry
    column_names = {column.name for column in table.columns}
    assert {
        "request_fingerprint",
        "prompt_hash",
        "model",
        "response_payload",
    } <= column_names
    assert table.c.similarity_threshold is not None
    assert any(
        isinstance(constraint, UniqueConstraint)
        and constraint.name == "uq_cache_entry_request_fingerprint"
        for constraint in table.constraints
    )
