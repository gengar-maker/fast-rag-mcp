# blazing-rag-mcp 0.5 — M3 Pro 36 GB edition

Local code-intelligence RAG MCP server for Claude Code and opencode, optimized for Apple Silicon.

## What is accelerated

- **Git-aware scanning:** uses `git ls-files -co --exclude-standard` instead of recursively walking every ignored dependency/cache directory.
- **Stat-only incremental skip:** unchanged files are not opened or hashed.
- **Chunk-level embedding reuse:** changing one function reuses vectors for unchanged functions in the same file.
- **Cross-file GPU batching:** many files are embedded in one MPS workload.
- **Duplicate-text elimination:** identical chunks in one batch are embedded once.
- **Fast staging rebuilds:** forced rebuilds use a disposable SQLite database with fsync/WAL disabled, then atomically swap it into place.
- **FP16 vector storage:** half the vector disk and load bandwidth with negligible retrieval-quality impact.
- **Adaptive vector backend:** small indexes use NumPy/Accelerate; larger indexes use Torch MPS FP16. This avoids Metal launch overhead for a 6k-chunk repository.
- **Exact-symbol query routing:** identifier queries can skip neural query embedding entirely.
- **Lazy MCP runtime:** Claude completes MCP initialization before Torch/model/vector loading.

Retrieval combines Tree-sitter symbols, paths, shallow references, SQLite FTS5/BM25, and dense code embeddings.

## Install

```bash
cd /absolute/path/to/blazing-rag-mcp-code-m3pro
uv sync --extra mac-metal --extra code
```

## M3 Pro 36 GB profile

Copy `.env.example` or use the following environment:

```bash
export BRAG_ROOTS=/absolute/path/to/repository
export BRAG_DB_DIR=/absolute/path/to/.brag

export BRAG_DEVICE=mps
export BRAG_EMBEDDING_MODEL=nomic-ai/CodeRankEmbed
export BRAG_EMBEDDING_TRUST_REMOTE_CODE=true
export BRAG_EMBEDDING_ALLOW_HASH_FALLBACK=false
export BRAG_EMBEDDING_PRECISION=float16
export BRAG_EMBEDDING_BATCH_SIZE=32
export BRAG_EMBEDDING_FLUSH_CHUNKS=1024
export BRAG_EMBEDDING_MAX_SEQ_LENGTH=384
export BRAG_EMBEDDING_EMPTY_CACHE_AFTER_ENCODE=false
export BRAG_EMBEDDING_REUSE_UNCHANGED_CHUNKS=true

export BRAG_TOKENIZER_PARALLELISM=true
export BRAG_TOKENIZER_THREADS=8
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8

export BRAG_SCAN_BACKEND=auto
export BRAG_FAST_STAT_SKIP=true
export BRAG_CHUNK_TOKENS=384
export BRAG_CHUNK_OVERLAP=32
export BRAG_ADD_FILE_SUMMARY_CHUNKS=false

export BRAG_VECTOR_BACKEND=auto
export BRAG_VECTOR_STORAGE_DTYPE=float16
export BRAG_TORCH_VECTOR_DEVICE=mps
export BRAG_TORCH_VECTOR_DTYPE=float16
export BRAG_MPS_VECTOR_MIN_VECTORS=30000
export BRAG_FAISS_GPU=false
export BRAG_KEEP_CPU_VECTOR_COPY=false
```

Do not set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` as an optimization.

## First rebuild

Version 0.5 adds embedding fingerprints and FP16 vector storage. Build a fresh index once:

```bash
.venv/bin/brag scan
.venv/bin/brag index --force
```

Subsequent point changes use incremental indexing:

```bash
.venv/bin/brag index
```

Expected behavior after changing one function:

```text
files scanned:            repository file count
files opened/hashed:      one or a few
chunks_written:           chunks in changed files
chunks_reused:            unchanged functions/chunks
chunks_embedded:          only genuinely changed chunks
embedding_calls:          usually 1
```

A no-change run should report `docs_changed=0` and `chunks_embedded=0`.

## Tune the actual M3 Pro

M3 Pro variants and thermals differ. Benchmark real repository chunks:

```bash
.venv/bin/brag tune \
  --sample-chunks 128 \
  --batch-sizes 8,16,24,32,48,64
```

Set `BRAG_EMBEDDING_BATCH_SIZE` to the reported recommendation. A prepared turbo profile is in `configs/m3pro-36gb-turbo.env`; fall back to batch `32` if memory pressure or swap grows.

## Why `BRAG_VECTOR_BACKEND=auto`

For a repository around 6,000–20,000 vectors, CPU Accelerate exact dot products usually have lower query latency than dispatching a tiny matrix-vector operation to Metal and synchronizing it. Above `BRAG_MPS_VECTOR_MIN_VECTORS`, the server automatically switches to an FP16 Torch MPS matrix.

To force Metal vector search:

```bash
export BRAG_VECTOR_BACKEND=torch
```

This affects retrieval only; document/query embeddings still run on MPS.

## Claude Code

Use the already-installed executable directly, not `uv run`:

```json
{
  "mcpServers": {
    "blazing-code-rag": {
      "type": "stdio",
      "command": "/absolute/path/to/blazing-rag-mcp-code-m3pro/.venv/bin/brag-mcp",
      "args": [],
      "env": {
        "BRAG_ROOTS": "/absolute/path/to/repository",
        "BRAG_DB_DIR": "/absolute/path/to/.brag",
        "BRAG_DEVICE": "mps",
        "BRAG_EMBEDDING_MODEL": "nomic-ai/CodeRankEmbed",
        "BRAG_EMBEDDING_TRUST_REMOTE_CODE": "true",
        "BRAG_EMBEDDING_ALLOW_HASH_FALLBACK": "false",
        "BRAG_EMBEDDING_PRECISION": "float16",
        "BRAG_EMBEDDING_BATCH_SIZE": "32",
        "BRAG_EMBEDDING_FLUSH_CHUNKS": "1024",
        "BRAG_EMBEDDING_MAX_SEQ_LENGTH": "384",
        "BRAG_EMBEDDING_REUSE_UNCHANGED_CHUNKS": "true",
        "BRAG_TOKENIZER_PARALLELISM": "true",
        "BRAG_TOKENIZER_THREADS": "8",
        "BRAG_SCAN_BACKEND": "auto",
        "BRAG_FAST_STAT_SKIP": "true",
        "BRAG_VECTOR_BACKEND": "auto",
        "BRAG_VECTOR_STORAGE_DTYPE": "float16",
        "BRAG_TORCH_VECTOR_DEVICE": "mps",
        "BRAG_TORCH_VECTOR_DTYPE": "float16",
        "BRAG_MPS_VECTOR_MIN_VECTORS": "30000",
        "BRAG_FAISS_GPU": "false",
        "BRAG_KEEP_CPU_VECTOR_COPY": "false",
        "BRAG_ALLOW_REINDEX_TOOL": "false"
      }
    }
  }
}
```

A complete template is at `configs/claude-project.m3pro-36gb.json`.

Claude can connect before the model loads. Use `rag_status(load_runtime=false)` for lightweight status and `rag_warmup()` to explicitly load/warm the model after connection.

## opencode

Use `configs/opencode.m3pro-36gb.jsonc`. It also invokes `.venv/bin/brag-mcp` directly.

## Tools

- `code_search` — smart symbol/path/FTS/dense retrieval; defaults to exact-symbol fast routing where applicable.
- `code_find_symbol`
- `code_references`
- `code_neighbors`
- `code_repo_map`
- `code_fetch`
- compatibility tools: `rag_search`, `rag_fetch`, `rag_status`, `rag_warmup`, `rag_reindex`

Keep `BRAG_ALLOW_REINDEX_TOOL=false` and run indexing from a terminal. This prevents an agent from accidentally starting a full rebuild.

## Diagnostics

```bash
.venv/bin/brag status
.venv/bin/brag status --runtime
.venv/bin/brag search "where is attention backward implemented" --top-k 8
```

Index output separates:

```text
scan_and_hash
prepare
embedding
storage
other
chunks_embedded
chunks_reused
new_embeddings_per_second
```

If `embedding` dominates, tune batch size. If `prepare` dominates, inspect generated/large source files. If `storage` dominates, keep `.brag` on a local non-synced SSD.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The implementation remains a retrieval index, not a full compiler/LSP semantic database. References are intentionally shallow; exact type resolution should still come from an LSP/compiler index.
