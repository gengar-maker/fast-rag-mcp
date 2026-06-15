#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/absolute/path/to/blazing-rag-mcp-code-fastm4}"
REPO_DIR="${REPO_DIR:-/absolute/path/to/your/repo}"

claude mcp add \
  --transport stdio \
  --env BRAG_ROOTS="$REPO_DIR" \
  --env BRAG_DB_DIR="$REPO_DIR/.brag" \
  --env BRAG_DEVICE="mps" \
  --env BRAG_VECTOR_BACKEND="torch" \
  --env BRAG_TORCH_VECTOR_DEVICE="mps" \
  --env BRAG_FAISS_GPU="false" \
  --env BRAG_EMBEDDING_MODEL="nomic-ai/CodeRankEmbed" \
  --env BRAG_EMBEDDING_TRUST_REMOTE_CODE="true" \
  --env BRAG_EMBEDDING_ALLOW_HASH_FALLBACK="false" \
  --env BRAG_EMBEDDING_BATCH_SIZE="16" \
  --env BRAG_EMBEDDING_FLUSH_CHUNKS="512" \
  --env BRAG_EMBEDDING_MAX_SEQ_LENGTH="384" \
  --env BRAG_EMBEDDING_EMPTY_CACHE_AFTER_ENCODE="false" \
  --env BRAG_ADD_FILE_SUMMARY_CHUNKS="false" \
  blazing-rag -- uv --directory "$PROJECT_DIR" run brag-mcp
