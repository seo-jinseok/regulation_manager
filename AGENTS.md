# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core pipeline (HWP conversion, preprocessing, parsing/formatting, LLM client, caching).
- `scripts/`: one-off utilities (verification, patching, inspection).
- `tests/`: pytest suite; `tests/debug/` contains diagnostic tests that should still be deterministic.
- `data/input/`: source HWP files; `data/output/`: generated JSON/MD/HTML artifacts.
- `docs/` and `SCHEMA_REFERENCE.md`: schema and design references; `conductor/` holds planning artifacts.

## Build, Test, and Development Commands
- `uv venv` / `uv pip install -r requirements.txt`: create env and install deps.
- `uv run python -m src.main "data/input/규정집.hwp"`: run pipeline on a file.
- `regulation-manager "data/input/규정집.hwp"`: installed CLI entry point.
- `uv run pytest`: run all tests (including `tests/debug`).
- `uv lock`: refresh `uv.lock` after dependency changes.

## Coding Style & Naming Conventions
- Python 3.11+, 4-space indentation, `snake_case` for functions/vars, `CamelCase` for classes.
- Prefer `pathlib.Path` and relative imports inside `src/`.
- Keep output schema stable: use node fields `type`, `display_no`, `sort_no`, `children`, and `metadata` keys.
- Avoid non-ASCII in code/comments unless required by domain data (Korean regulation text).

## Testing Guidelines
- Framework: `pytest`. Test files follow `tests/test_*.py`.
- Add focused tests when changing parsing logic (`src/formatter.py`, `src/preprocessor.py`).
- Debug tests should not depend on external services or sleep-based timing.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits (e.g., `feat: ...`, `fix: ...`, `chore: ...`, optional scopes like `feat(parser): ...`).
- PRs should include a brief summary, relevant test command(s), and note any data files or schema changes.

## Security & Configuration
- Keep secrets in `.env` (never commit); use `.env.example` as a template.
- Optional cache controls: `LLM_CACHE_TTL_DAYS`, `LLM_CACHE_MAX_ENTRIES`.
