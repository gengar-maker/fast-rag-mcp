#!/usr/bin/env bash
set -euo pipefail

SERVER_DIR="/ABS/PATH/TO/blazing-rag-mcp"
REPO_DIR="${1:-$PWD}"

claude mcp add \
  --transport stdio \
  --env BRAG_ROOTS="$REPO_DIR" \
  --env BRAG_DB_DIR="$REPO_DIR/.brag" \
  --env BRAG_DEVICE="auto" \
  --env BRAG_FAISS_GPU="true" \
  --env BRAG_VECTOR_BACKEND="auto" \
  --env BRAG_TORCH_VECTOR_DEVICE="auto" \
  --env BRAG_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B" \
  --env BRAG_ALLOW_REINDEX_TOOL="false" \
  blazing-code-rag -- uv --directory "$SERVER_DIR" run brag-mcp
