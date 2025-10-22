# Repository Guidelines

## Project Structure & Module Organization
- Use `DESIGN.md` as the authoritative product and systems brief; keep it updated when architecture shifts.
- Infrastructure manifests live in `db-stack/` (Postgres, Qdrant) and `llm-stack/` (vLLM gateway plus Open WebUI). Update compose files before altering shared ports or service names.
- Place Python application packages under a top-level module (e.g., `app/`) with subpackages per domain (`features/`, `policies/`, `ingest/`). Mirror business logic tests under `tests/` using the same package structure.
- Store large assets or notebooks outside the repository and reference them via documentation links; keep checked-in data under 1 MB files in a dedicated `data/` subfolder if unavoidable.

## Build, Test, and Development Commands
- `uv sync --frozen --python 3.11` — install dependencies into `.venv/` using the pinned `pyproject.toml`.
- `uv run python -m app.main` — run the primary service entry point; adjust the module path for new executables.
- `docker compose -f db-stack/docker-compose.yaml up -d` — start Postgres and Qdrant for local development; pair with `down` when finished.
- `docker compose -f llm-stack/docker-compose.yaml up -d` — launch the local LLM gateway and Open WebUI for manual evaluation.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, 88-character lines, and comprehensive type hints; prefer `dataclass`es for structured payloads.
- Name Python modules and functions in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`.
- Run formatters and linters through uv: `uv run ruff format` then `uv run ruff check`. Add new rules to `pyproject.toml` instead of per-file ignores.

## Testing Guidelines
- Tests live in `tests/` and use `pytest` with feature-focused modules such as `tests/features/test_bandit.py`.
- Execute the suite with `uv run pytest`; include `-k` filters for targeted debugging and `--maxfail=1` in CI.
- Maintain ≥80 % statement coverage (`uv run pytest --cov=app --cov-report=term-missing`) and add regression cases when fixing bugs.

## Commit & Pull Request Guidelines
- Write conventional commit messages (`type(scope): subject`) to replace the vague history currently in `git log`. Example: `feat(policy): add linucb initializer`.
- Reference related issues or design sections in the PR description, list test evidence (`uv run pytest`, manual stack checks), and include screenshots for UI-facing changes (Open WebUI flows).
- Seek at least one review for behavioral changes; re-request review after addressing feedback and rebasing onto the latest main.
