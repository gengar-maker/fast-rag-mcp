#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m compileall -q src tests
uv run --extra documents --extra code --extra dev ruff check src tests
uv run --extra documents --extra code --extra dev mypy src/blazing_rag_mcp
uv run --extra documents --extra code --extra dev bandit -q -r src/blazing_rag_mcp
uv run --extra documents --extra code --extra dev pytest -q
uv build
