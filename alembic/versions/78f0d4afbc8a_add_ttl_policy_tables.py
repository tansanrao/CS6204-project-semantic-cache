"""Add TTL policy logging tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "78f0d4afbc8a"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid = postgresql.UUID(as_uuid=True)
    jsonb = postgresql.JSONB()

    op.create_table(
        "ttl_decision",
        sa.Column("id", uuid, primary_key=True),
        sa.Column(
            "cache_entry_id", uuid, sa.ForeignKey("cache_entry.id"), nullable=True
        ),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("policy_name", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("ttl_bucket", sa.SmallInteger(), nullable=False),
        sa.Column("propensity", sa.Float(), nullable=True),
        sa.Column("features", jsonb, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_ttl_decision_created_at", "ttl_decision", ["created_at"], unique=False
    )
    op.create_index(
        "ix_ttl_decision_request_fingerprint",
        "ttl_decision",
        ["request_fingerprint"],
        unique=False,
    )

    op.create_table(
        "feedback_event",
        sa.Column("id", uuid, primary_key=True),
        sa.Column(
            "cache_entry_id", uuid, sa.ForeignKey("cache_entry.id"), nullable=False
        ),
        sa.Column(
            "ttl_decision_id", uuid, sa.ForeignKey("ttl_decision.id"), nullable=True
        ),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("details", jsonb, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_feedback_event_created_at", "feedback_event", ["created_at"], unique=False
    )
    op.create_index(
        "ix_feedback_event_event_type", "feedback_event", ["event_type"], unique=False
    )

    op.create_table(
        "policy_reward",
        sa.Column("id", uuid, primary_key=True),
        sa.Column(
            "ttl_decision_id", uuid, sa.ForeignKey("ttl_decision.id"), nullable=False
        ),
        sa.Column("reward", sa.Float(), nullable=False),
        sa.Column("attribution_rule", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_policy_reward_created_at", "policy_reward", ["created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_policy_reward_created_at", table_name="policy_reward")
    op.drop_table("policy_reward")

    op.drop_index("ix_feedback_event_event_type", table_name="feedback_event")
    op.drop_index("ix_feedback_event_created_at", table_name="feedback_event")
    op.drop_table("feedback_event")

    op.drop_index("ix_ttl_decision_request_fingerprint", table_name="ttl_decision")
    op.drop_index("ix_ttl_decision_created_at", table_name="ttl_decision")
    op.drop_table("ttl_decision")
