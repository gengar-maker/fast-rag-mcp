#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${1:-$ROOT_DIR}"
DB_DIR="$(mktemp -d -t brag-smoke.XXXXXX)"
trap 'rm -rf "$DB_DIR"' EXIT

export BRAG_ROOTS="$REPO_DIR"
export BRAG_DB_DIR="$DB_DIR"
export BRAG_DEVICE=cpu
export BRAG_VECTOR_BACKEND=numpy
export BRAG_EMBEDDING_ALLOW_HASH_FALLBACK=true
export BRAG_EMBEDDING_MODEL=__smoke_hash_fallback__
export BRAG_READ_ONLY=true

"$ROOT_DIR/.venv/bin/brag" scan
"$ROOT_DIR/.venv/bin/brag" index --force
"$ROOT_DIR/.venv/bin/brag" doctor
"$ROOT_DIR/.venv/bin/brag" search "Application reindex" --top-k 3
