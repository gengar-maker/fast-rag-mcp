#!/usr/bin/env bash
set -euo pipefail

export BRAG_ROOTS="${BRAG_ROOTS:-$PWD}"
export BRAG_DB_DIR="${BRAG_DB_DIR:-$PWD/.brag}"
uv run brag index
uv run brag status
uv run brag search "where is the main server implemented" --top-k 5
uv run brag symbol main --limit 5
