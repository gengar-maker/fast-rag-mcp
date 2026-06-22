#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m compileall -q src tests
uv run ruff check src tests
uv run mypy src/blazing_rag_mcp
uv run bandit -q -r src/blazing_rag_mcp
uv run pytest -q
uv build
