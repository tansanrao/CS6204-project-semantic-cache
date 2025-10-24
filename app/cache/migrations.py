"""Utilities for applying Alembic migrations programmatically."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def upgrade_head(database_url: str) -> None:
    """Run Alembic migrations up to the head revision for the given database."""
    project_root = Path(__file__).resolve().parents[3]
    alembic_ini = project_root / "alembic.ini"
    if not alembic_ini.exists():  # Defensive guard for test environments.
        raise FileNotFoundError(f"Alembic configuration not found at {alembic_ini}")

    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
