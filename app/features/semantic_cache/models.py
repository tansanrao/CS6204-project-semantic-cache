"""SQLAlchemy metadata and table definitions for the semantic cache."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.sqltypes import JSON, TypeEngine, Uuid
from sqlalchemy.types import TypeDecorator


class JSONLike(TypeDecorator[Any]):
    """JSON type that compiles to JSONB on Postgres and JSON elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB(astext_type=Text()))
        return dialect.type_descriptor(JSON())


def _json_type(_: Dialect) -> TypeEngine:
    """Return a portable JSON column type."""
    return JSONLike()


naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=naming_convention)


def cache_entry_table(dialect: Dialect | None = None) -> Table:
    """Return the cache_entry table for the provided SQL dialect."""
    json_type: TypeEngine
    if dialect is not None:
        json_type = _json_type(dialect)
    else:
        json_type = JSONLike()

    uuid_type: TypeEngine = (
        PG_UUID(as_uuid=True)
        if dialect is None or dialect.name == "postgresql"
        else Uuid(native=True)
    )

    table = Table(
        "cache_entry",
        metadata,
        Column("id", uuid_type, primary_key=True),
        Column("request_fingerprint", String(length=64), nullable=False),
        Column("prompt_hash", String(length=64), nullable=False),
        Column("model", String(length=128), nullable=False),
        Column("parameters", json_type, nullable=False),
        Column("prompt_text", Text, nullable=False),
        Column("response_payload", json_type, nullable=False),
        Column("ttl_bucket", SmallInteger, nullable=False),
        Column("ttl_seconds", Integer, nullable=False),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
        Column("expires_at", DateTime(timezone=True), nullable=False),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
        Column("last_accessed_at", DateTime(timezone=True), nullable=True),
        Column("hit_count", Integer, nullable=False, server_default="0"),
        Column("embedding_model", String(length=128), nullable=False),
        Column("embedding_mode", String(length=32), nullable=False),
        Column("embedding_dimension", SmallInteger, nullable=False),
        Column("similarity_threshold", Float, nullable=False),
        UniqueConstraint(
            "request_fingerprint", name="uq_cache_entry_request_fingerprint"
        ),
    )
    Index("ix_cache_entry_expires_at", table.c.expires_at)
    Index("ix_cache_entry_model", table.c.model)
    return table


cache_entry = cache_entry_table()


def ttl_decision_table(dialect: Dialect | None = None) -> Table:
    """Return the ttl_decision table definition."""
    json_type: TypeEngine
    if dialect is not None:
        json_type = _json_type(dialect)
    else:
        json_type = JSONLike()

    uuid_type: TypeEngine = (
        PG_UUID(as_uuid=True)
        if dialect is None or dialect.name == "postgresql"
        else Uuid(native=True)
    )

    table = Table(
        "ttl_decision",
        metadata,
        Column("id", uuid_type, primary_key=True),
        Column(
            "cache_entry_id", uuid_type, ForeignKey("cache_entry.id"), nullable=True
        ),
        Column("request_fingerprint", String(length=64), nullable=False),
        Column("prompt_hash", String(length=64), nullable=False),
        Column("policy_name", String(length=64), nullable=False),
        Column("policy_version", String(length=32), nullable=False),
        Column("ttl_bucket", SmallInteger, nullable=False),
        Column("propensity", Float, nullable=True),
        Column("features", json_type, nullable=False),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
    Index("ix_ttl_decision_created_at", table.c.created_at)
    Index("ix_ttl_decision_request_fingerprint", table.c.request_fingerprint)
    return table


ttl_decision = ttl_decision_table()


def feedback_event_table(dialect: Dialect | None = None) -> Table:
    """Return the feedback_event table definition."""
    json_type: TypeEngine
    if dialect is not None:
        json_type = _json_type(dialect)
    else:
        json_type = JSONLike()

    uuid_type: TypeEngine = (
        PG_UUID(as_uuid=True)
        if dialect is None or dialect.name == "postgresql"
        else Uuid(native=True)
    )

    table = Table(
        "feedback_event",
        metadata,
        Column("id", uuid_type, primary_key=True),
        Column(
            "cache_entry_id", uuid_type, ForeignKey("cache_entry.id"), nullable=False
        ),
        Column(
            "ttl_decision_id", uuid_type, ForeignKey("ttl_decision.id"), nullable=True
        ),
        Column("event_type", String(length=32), nullable=False),
        Column("score", Float, nullable=False),
        Column("details", json_type, nullable=True),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
    Index("ix_feedback_event_created_at", table.c.created_at)
    Index("ix_feedback_event_event_type", table.c.event_type)
    return table


feedback_event = feedback_event_table()


def policy_reward_table(dialect: Dialect | None = None) -> Table:
    """Return the policy_reward table definition."""
    uuid_type: TypeEngine = (
        PG_UUID(as_uuid=True)
        if dialect is None or dialect.name == "postgresql"
        else Uuid(native=True)
    )

    table = Table(
        "policy_reward",
        metadata,
        Column("id", uuid_type, primary_key=True),
        Column(
            "ttl_decision_id", uuid_type, ForeignKey("ttl_decision.id"), nullable=False
        ),
        Column("reward", Float, nullable=False),
        Column("attribution_rule", String(length=64), nullable=False),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
    Index("ix_policy_reward_created_at", table.c.created_at)
    return table


policy_reward = policy_reward_table()
