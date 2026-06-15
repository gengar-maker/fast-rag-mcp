# blazing-rag-mcp

Local code-intelligence RAG MCP server for Claude Code and opencode.

This build is optimized for Apple Silicon and fixes the original indexing hot path:

- changed chunks are accumulated across files before embedding;
- one bounded GPU embedding call handles many documents;
- MPS cache is not flushed after every file;
- SQLite writes use `executemany` and one transaction per bounded batch;
- indexing reports separate parsing, embedding, and storage timings;
- unchanged files remain incremental and are skipped.

## Retrieval architecture

```text
Tree-sitter symbols + paths + references
        + SQLite FTS5/BM25
        + dense embeddings
        -> symbol-first hybrid ranking
        -> compact MCP results
```

Dense retrieval is only one signal. Exact symbols, paths, code tokens, imports, and references are handled by CPU indexes, while neural embedding and optional exact dense search can use CUDA or Metal/MPS.

## MCP tools

- `code_search`
- `code_find_symbol`
- `code_references`
- `code_neighbors`
- `code_repo_map`
- `code_fetch`
- `rag_search`, `rag_fetch`, `rag_status`, `rag_reindex`

`rag_reindex` is disabled by default. Prefer the CLI for indexing.

## Install

```bash
cd /path/to/blazing-rag-mcp-code-fastm4
uv sync --extra mac-metal --extra code
```

For Linux/NVIDIA:

```bash
uv sync --extra embeddings --extra code --extra faiss-cpu
```

## Recommended M4 configuration

```bash
export BRAG_ROOTS=/absolute/path/to/repo
export BRAG_DB_DIR=/absolute/path/to/repo/.brag

export BRAG_DEVICE=mps
export BRAG_VECTOR_BACKEND=torch
export BRAG_TORCH_VECTOR_DEVICE=mps
export BRAG_FAISS_GPU=false

export BRAG_EMBEDDING_MODEL=nomic-ai/CodeRankEmbed
export BRAG_EMBEDDING_TRUST_REMOTE_CODE=true
export BRAG_EMBEDDING_BATCH_SIZE=16
export BRAG_EMBEDDING_FLUSH_CHUNKS=512
export BRAG_EMBEDDING_MAX_SEQ_LENGTH=384
export BRAG_EMBEDDING_EMPTY_CACHE_AFTER_ENCODE=false

export BRAG_CHUNK_TOKENS=384
export BRAG_CHUNK_OVERLAP=32
export BRAG_ADD_FILE_SUMMARY_CHUNKS=false
export BRAG_KEEP_CPU_VECTOR_COPY=false
```

Do not set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` as a normal optimization. It disables the MPS allocator hard limit and can let the process consume unsafe amounts of unified memory.

The M4 template uses `nomic-ai/CodeRankEmbed`, a 137M code-retrieval bi-encoder, instead of the much heavier 0.6B general embedding model. It requires `trust_remote_code=True`; pin the model revision in security-sensitive deployments. The server automatically applies its required query instruction prefix.

## Diagnose the repository before indexing

```bash
uv run brag scan
```

This does not load the embedding model. Confirm that `.venv`, `.git`, `node_modules`, `.brag`, build outputs, model caches, and dependency trees are excluded.

## Index

```bash
uv run brag index --force
```

The result includes:

```json
{
  "throughput": {
    "chunks_per_second": 0,
    "embedding_calls": 0,
    "embedding_flush_chunks": 512
  },
  "timings_ms": {
    "prepare": 0,
    "embedding": 0,
    "storage": 0,
    "other": 0
  }
}
```

Use these timings to identify the bottleneck:

- `embedding` dominates: tune model, sequence length, or MPS batch size;
- `storage` dominates: place `.brag` on a local SSD and avoid network/synced folders;
- `prepare` dominates: inspect generated files, very large source files, or reference extraction;
- excessive `embedding_calls`: increase `BRAG_EMBEDDING_FLUSH_CHUNKS` within your memory budget.

The index command does not load the in-memory vector search index unless explicitly requested:

```bash
uv run brag index --build-vector-index
```

## M4 tuning profiles

### Safe profile

```bash
export BRAG_EMBEDDING_MODEL=nomic-ai/CodeRankEmbed
export BRAG_EMBEDDING_TRUST_REMOTE_CODE=true
export BRAG_EMBEDDING_BATCH_SIZE=4
export BRAG_EMBEDDING_FLUSH_CHUNKS=128
export BRAG_EMBEDDING_MAX_SEQ_LENGTH=320
```

### Balanced profile

```bash
export BRAG_EMBEDDING_TRUST_REMOTE_CODE=true
export BRAG_EMBEDDING_BATCH_SIZE=16
export BRAG_EMBEDDING_FLUSH_CHUNKS=512
export BRAG_EMBEDDING_MAX_SEQ_LENGTH=384
```

### Throughput profile

Use only on M4 Pro/Max or after observing stable memory use:

```bash
export BRAG_EMBEDDING_MODEL=nomic-ai/CodeRankEmbed
export BRAG_EMBEDDING_TRUST_REMOTE_CODE=true
export BRAG_EMBEDDING_BATCH_SIZE=32
export BRAG_EMBEDDING_FLUSH_CHUNKS=1024
export BRAG_EMBEDDING_MAX_SEQ_LENGTH=384
```

The embedding wrapper automatically retries with a smaller internal batch after an MPS/CUDA allocation failure.

## Why this version is faster

The old implementation effectively did this for every changed file:

```text
parse one file
embed roughly 1-20 chunks
copy results from MPS to CPU
empty MPS cache
open/commit a SQLite transaction
repeat hundreds of times
```

This version does:

```text
prepare many files
flatten up to N chunks
embed them in one bounded call
write all associated documents in one transaction
repeat
```

For a run with 6429 chunks and 391 changed documents, `BRAG_EMBEDDING_FLUSH_CHUNKS=512` reduces top-level embedding calls from approximately 391 to approximately 13. Sentence Transformers still performs internal mini-batches according to `BRAG_EMBEDDING_BATCH_SIZE`, but allocator, synchronization, tokenization setup, Python, and transaction overhead are substantially lower.

## Search

```bash
uv run brag search "where is authentication implemented" --top-k 8
uv run brag symbol AuthProvider --limit 8
uv run brag refs validateToken --limit 20
uv run brag map --path-prefix src/auth
```

## Claude Code

Edit and run:

```bash
configs/claude-code-add-mac-metal.sh
```

The generated command uses an stdio MCP server.

## opencode

Copy and edit:

```text
configs/opencode.mac-metal.jsonc
```

## Important settings

| Variable | Default | Purpose |
|---|---:|---|
| `BRAG_EMBEDDING_BATCH_SIZE` | `8` | Neural mini-batch size. Controls activation memory. |
| `BRAG_EMBEDDING_FLUSH_CHUNKS` | `256` | Cross-file chunks per top-level embedding/write batch. |
| `BRAG_EMBEDDING_MAX_SEQ_LENGTH` | `512` | Hard transformer token limit. M4 template uses `384`. |
| `BRAG_EMBEDDING_EMPTY_CACHE_AFTER_ENCODE` | `false` | Avoids expensive cache flush/synchronization after every call. |
| `BRAG_CHUNK_TOKENS` | `384` | Target code/text chunk size. |
| `BRAG_MAX_FILE_BYTES` | `750000` | Skip unexpectedly large files. |
| `BRAG_KEEP_CPU_VECTOR_COPY` | `false` | Avoid duplicate CPU plus MPS vector matrices during serving. |

## Incremental behavior

A normal second run should skip unchanged documents:

```bash
uv run brag index
```

A full `--force` run intentionally recomputes all embeddings. It builds a staging SQLite index and atomically swaps it in after success, avoiding slow per-document FTS deletion. Use it only when changing the embedding model, chunking rules, or schema.

## Tests

```bash
PYTHONPATH=src pytest -q
```

## Limits

The reference index is intentionally shallow. It detects common imports and call-like references but is not a complete LSP/type-resolution engine. Tree-sitter improves structure extraction; exact semantic resolution still belongs to an LSP or compiler index.

## Claude Code init failures

This build uses lazy runtime initialization: MCP `initialize` and `tools/list` complete before
PyTorch, the embedding model, and the vector matrix are loaded. The first semantic tool call (or
`rag_warmup`) initializes the runtime.

For reliable Claude Code startup, synchronize once and invoke the venv executable directly:

```bash
uv sync --extra mac-metal --extra code
realpath .venv/bin/brag-mcp
```

Use that absolute path as `.mcp.json`'s `command`; do not put `uv run` on the MCP startup path.
`uv run` verifies/synchronizes the project before execution and can exceed the MCP startup budget.
See `configs/claude-project.mac-metal.lazy.json`.

Terminal diagnostics with the same environment:

```bash
/path/to/.venv/bin/brag status
/path/to/.venv/bin/brag status --runtime
```

The first command is lightweight. The second deliberately loads CodeRankEmbed and the MPS vector
index and prints the actual startup exception if model loading fails.
