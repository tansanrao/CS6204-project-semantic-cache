"""Utilities for applying Alembic migrations programmatically."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def upgrade_head(database_url: str) -> None:
    """Run Alembic migrations up to the head revision for the given database."""
    alembic_ini = _locate_alembic_ini()

    project_root = alembic_ini.parent
    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(project_root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


def _locate_alembic_ini() -> Path:
    """Walk parent directories until an alembic.ini file is located."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        candidate = current / "alembic.ini"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        f"Alembic configuration not found when traversing from {Path(__file__).resolve()}"
    )
