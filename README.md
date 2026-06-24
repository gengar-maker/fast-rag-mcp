# blazing-rag-mcp 1.1

Production-oriented local code and PDF-documentation MCP server for Claude Code and opencode. CUDA `.cu` / `.cuh` files are indexed structurally as C++, while `contrib` directories are excluded by default. It combines structural code indexing, page-aware PDF extraction, exact symbol/path lookup, SQLite FTS5, shallow reference extraction, and optional dense retrieval accelerated by Apple Metal/MPS or CUDA.

The server is optimized for a persistent local MCP process. Heavy resources are lazy: Claude can complete the MCP handshake before PyTorch, the embedding model, or the vector matrix are loaded.

## Key properties

- Offline Tree-sitter symbol extraction with a pinned bundled-parser wheel and regex fallback.
- Exact and fuzzy symbol lookup without loading the embedding model.
- Git-aware scanning with safe filesystem fallback.
- Targeted one-file or one-directory updates.
- Stat-only no-op detection for unchanged files.
- Chunk-level embedding reuse inside changed files.
- Cross-file embedding batches for full builds.
- Stable symbol/chunk IDs across unrelated line shifts.
- SQLite WAL mode for normal operation and disposable staging DBs for atomic full rebuilds.
- Lazy vector invalidation: incremental updates do not synchronously rebuild the complete in-memory vector matrix.
- True SQLite read-only query connection by default; writer connections exist only during locked index mutations.
- Page-aware PDF chunks with citations, layout metadata, repeated header/footer removal, optional OCR, and optional table extraction.
- Dedicated PDF search, outline, fetch, and targeted-update tools.
- Integrity, compatibility, lifecycle, and timing diagnostics.

## Architecture

```text
Claude Code / opencode
        │ stdio MCP
        ▼
FastMCP transport
        ▼
Application lifecycle
  ├── lazy SQLite Store
  ├── lazy EmbeddingModel
  ├── lazy VectorIndex
  ├── Retriever
  └── serialized Indexer mutations
        │
        ├── symbols / paths / references / FTS5
        ├── PDF pages / sections / citations / optional OCR
        └── dense vectors: NumPy, Torch MPS/CUDA, or FAISS
```

The MCP transport does not own indexing logic. `Application` owns process-lifetime resources and separates lightweight exact retrieval from expensive neural retrieval.

## Installation on M3 Pro 36 GB

```bash
cd /absolute/path/to/blazing-rag-mcp-production
uv sync --extra mac-metal --extra code --extra documents
```

Use the installed executable directly from MCP clients:

```text
/absolute/path/to/blazing-rag-mcp-production/.venv/bin/brag-mcp
```

Do not put `uv run` on the MCP startup path unless necessary. Direct execution avoids dependency resolution and PATH differences during client initialization.

## PDF documentation indexing

Install the optional document extractor:

```bash
uv sync --extra mac-metal --extra code --extra documents
```

PDF files under `BRAG_ROOTS` are discovered automatically. Text is extracted page-by-page with PyMuPDF, sorted into reading order, and chunked without crossing page boundaries. Each result includes `page`, `page_label`, bounding box metadata and a stable citation such as:

```text
docs/reference-manual.pdf#page=42
```

Index a documentation directory or one changed PDF:

```bash
.venv/bin/brag index docs/
.venv/bin/brag index docs/reference-manual.pdf
```

Search from the CLI:

```bash
.venv/bin/brag docs "how is session token refresh configured" --top-k 8
.venv/bin/brag outline docs/reference-manual.pdf
```

MCP tools:

```text
document_search(query, top_k?, mode?, path_prefix?, include_text?)
document_outline(path_or_doc_id, limit?)
document_fetch(resource_uri? | chunk_id?, context_chunks=1)
document_update_index(paths, refresh_vectors=false)
```

The existing `code_update_index` and `rag_reindex(paths=[...])` also accept PDF paths.

Recommended PDF settings:

```bash
BRAG_MAX_PDF_BYTES=200000000
BRAG_PDF_MAX_PAGES=2000
BRAG_PDF_MAX_CHARS=20000000
BRAG_PDF_CHUNK_TOKENS=512
BRAG_PDF_CHUNK_OVERLAP=48
BRAG_PDF_DENSE_CANDIDATE_MULTIPLIER=8
BRAG_PDF_QUERY_PREFIX="Represent this query for retrieving relevant technical documentation:"
BRAG_PDF_OCR_MODE=off
BRAG_PDF_EXTRACT_TABLES=false
```

### Scanned PDFs and OCR

OCR is opt-in because it is much slower than native text extraction. PyMuPDF delegates OCR to a separately installed Tesseract executable and language data:

```bash
brew install tesseract
export BRAG_PDF_OCR_MODE=auto
export BRAG_PDF_OCR_LANGUAGE=eng
```

`auto` OCR runs only on low-text pages that contain images. Use `always` only for known scanned corpora. OCR failures are recorded in chunk metadata and do not abort extraction of normal text pages.

### Tables

Set `BRAG_PDF_EXTRACT_TABLES=true` to add Markdown-like `pdf_table` chunks. Table detection is disabled by default because it can dominate indexing time on long manuals. Normal text blocks remain available even when table extraction is disabled.

### Embedding-model choice

The default CodeRankEmbed model is optimized for code retrieval. It remains useful for technical manuals that contain APIs and code, but a PDF-heavy natural-language corpus may benefit from a general multilingual embedding model. Changing the embedding model requires `brag index --force` because the persisted vector space must remain consistent.

PyMuPDF is an optional dependency with its own AGPL/commercial licensing terms; review them for your deployment.

## Initial configuration

Copy `.env.example`, then set at least:

```bash
BRAG_ROOTS=/absolute/path/to/repository
BRAG_DB_DIR=/absolute/path/to/repository/.brag
```

Recommended M3 Pro 36 GB profile:

```bash
BRAG_DEVICE=mps
BRAG_EMBEDDING_MODEL=nomic-ai/CodeRankEmbed
BRAG_EMBEDDING_TRUST_REMOTE_CODE=true
BRAG_EMBEDDING_ALLOW_HASH_FALLBACK=false
BRAG_EMBEDDING_PRECISION=float16
BRAG_EMBEDDING_BATCH_SIZE=32
BRAG_EMBEDDING_FLUSH_CHUNKS=1024
BRAG_EMBEDDING_MAX_SEQ_LENGTH=384
BRAG_EMBEDDING_EMPTY_CACHE_AFTER_ENCODE=false
BRAG_EMBEDDING_REUSE_UNCHANGED_CHUNKS=true
BRAG_MPS_ENABLE_FALLBACK=false

BRAG_VECTOR_BACKEND=auto
BRAG_VECTOR_STORAGE_DTYPE=float16
BRAG_TORCH_VECTOR_DEVICE=mps
BRAG_TORCH_VECTOR_DTYPE=float16
BRAG_MPS_VECTOR_MIN_VECTORS=30000
BRAG_FAISS_GPU=false
BRAG_KEEP_CPU_VECTOR_COPY=false
```

Do not use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` as a performance setting. The server also no longer forces `PYTORCH_ENABLE_MPS_FALLBACK=1`; CPU fallback is an explicit diagnostic/reliability opt-in because it can create severe, hard-to-see latency spikes.

## Fixing CodeRankEmbed load failures

Version 1.1 pins a Transformers 4.x runtime and an ABI3 tokenizer wheel instead of allowing Transformers 5.x:

```text
sentence-transformers==3.4.1
transformers==4.47.1
tokenizers==0.21.4
huggingface-hub<1
einops>=0.8,<1
```

`CodeRankEmbed` ships custom model code written for Transformers 4.x. The previous lock file could
resolve Transformers 5.x, where that custom code may fail during import. Recreate or resync the
environment after upgrading:

```bash
rm -rf .venv
uv sync --extra mac-metal --extra code --extra documents
.venv/bin/python -c 'import sentence_transformers, transformers; print(sentence_transformers.__version__, transformers.__version__)'
```

Expected versions are `3.4.1`, `4.47.1`, and `0.21.4` for sentence-transformers, Transformers, and tokenizers. The tokenizers wheel uses CPython ABI3, avoiding a local Rust build on Python 3.14. Use one explicit cache location for both terminal
prefetch and Claude MCP:

```bash
export HF_HOME="$HOME/.cache/blazing-rag/huggingface"
export SENTENCE_TRANSFORMERS_HOME="$HOME/.cache/blazing-rag/sentence-transformers"
export BRAG_EMBEDDING_CACHE_DIR="$SENTENCE_TRANSFORMERS_HOME"
export BRAG_EMBEDDING_REVISION=3c4b60807d71f79b43f3c4363786d9493691f8b1
export BRAG_EMBEDDING_TRUST_REMOTE_CODE=true
export BRAG_EMBEDDING_LOCAL_FILES_ONLY=false
export BRAG_DEVICE=cpu
export BRAG_EMBEDDING_PRECISION=float32

.venv/bin/brag warmup --no-vectors
```

After the successful prefetch, restore the MPS profile and set
`BRAG_EMBEDDING_LOCAL_FILES_ONLY=true`. The loader now includes the underlying exception, installed
package versions and cache settings in its error instead of returning only the generic fallback
message.

## Prefetch the model and build the index

The shipped profiles pin the reviewed `CodeRankEmbed` commit `3c4b60807d71f79b43f3c4363786d9493691f8b1`. Fetch it once from a trusted network:

```bash
BRAG_EMBEDDING_LOCAL_FILES_ONLY=false .venv/bin/brag warmup --no-vectors
```

Production MCP templates set `BRAG_EMBEDDING_LOCAL_FILES_ONLY=true`, so the running server never downloads model files. Version 1.1 uses index format 3, which adds page-aware PDF chunks. Rebuild once after upgrading:

```bash
.venv/bin/brag scan
.venv/bin/brag index --force
.venv/bin/brag doctor --model
```

A forced rebuild is written to a staging SQLite database and replaces the active database only after a successful build.

## Fast incremental updates

### Preferred: target the changed file

```bash
.venv/bin/brag index src/package/module.py
```

Or through MCP:

```text
code_update_index(paths=["src/package/module.py"])
```

`rag_reindex(paths=[...])` supports the same targeted path for compatibility.

A targeted update:

- resolves only the requested path;
- does not walk the repository;
- skips model loading when file metadata/content is unchanged;
- embeds only changed chunks;
- reuses embeddings for unchanged symbols in the same changed file;
- invalidates an already-loaded vector matrix only when corpus data changed;
- defers vector reload until the next dense search.

### Repository consistency scan

```bash
.venv/bin/brag index
```

This scans Git-tracked/untracked non-ignored files where possible, but still embeds only changed content.

### Full rebuild

```bash
.venv/bin/brag index --force
```

Use a full rebuild after changing embedding model, embedding revision, chunking settings, vector storage dtype, or index format.

## Why a first one-file update can still be slower

In a new CLI process, the first changed file may require loading the embedding model. The result now separates this from indexing:

```json
{
  "timings_ms": {
    "scan_and_hash": 0,
    "prepare": 0,
    "model_load": 0,
    "embedding": 0,
    "storage": 0,
    "other": 0
  }
}
```

Interpretation:

- high `model_load`: process started cold; use persistent MCP and `rag_warmup(load_vectors=false)`;
- high `embedding`: tune MPS batch size or reduce sequence length;
- high `storage`: keep `.brag` on a local SSD, outside synced/network storage;
- high `scan_and_hash`: pass a specific path instead of a full scan;
- high next-search latency: the dense vector matrix was stale and loaded lazily after an update.

In a persistent MCP process, the model remains loaded after warmup or the first dense search/update.

The code parser is also fully offline. Version 1.0 pins the bundled `tree-sitter-language-pack==0.7.2`; do not loosen that constraint without adding an explicit parser prefetch/install phase, because newer on-demand releases can perform network discovery during cold parser creation.

## Claude Code configuration

Use `configs/claude-project.m3pro-36gb.json` as the safe, read-only default. Replace both absolute paths.

```json
{
  "mcpServers": {
    "blazing-code-rag": {
      "type": "stdio",
      "command": "/absolute/path/to/blazing-rag-mcp-production/.venv/bin/brag-mcp",
      "args": [],
      "env": {
        "BRAG_ROOTS": "/absolute/path/to/repository",
        "BRAG_DB_DIR": "/absolute/path/to/repository/.brag",
        "BRAG_DEVICE": "mps",
        "BRAG_READ_ONLY": "true",
        "BRAG_ALLOW_REINDEX_TOOL": "false"
      }
    }
  }
}
```

To allow targeted updates through Claude, use the write-enabled template and keep the repository and index directory restricted:

```json
{
  "BRAG_READ_ONLY": "false",
  "BRAG_ALLOW_REINDEX_TOOL": "true"
}
```

A CLI process may index regardless of MCP write policy, so the safer production setup is read-only MCP plus terminal/editor-triggered targeted indexing.

## opencode configuration

Use `configs/opencode.m3pro-36gb.jsonc`. It also invokes the installed executable directly and starts read-only.

## MCP tools

| Tool | Purpose | Loads neural resources? |
|---|---|---:|
| `code_find_symbol` | Exact/fuzzy symbol definitions | No |
| `code_references` | Indexed calls/import/reference mentions | No |
| `code_neighbors` | Same-file symbols and callers | No |
| `code_repo_map` | Compact structural map | No |
| `code_fetch` | Fetch a chunk/symbol | No |
| `code_search` | Symbol/path/FTS/dense hybrid retrieval | Only when dense route is needed |
| `code_update_index` | Update selected code or document paths | Only if new embeddings are needed |
| `document_search` | PDF-only hybrid/semantic/keyword search with page citations | Only for semantic/hybrid mode |
| `document_outline` | PDF section/page outline | No |
| `document_fetch` | Fetch a PDF hit with adjacent document context | No |
| `document_update_index` | Targeted PDF update | Only if new embeddings are needed |
| `rag_status` | Lifecycle/index status | Optional |
| `rag_doctor` | Database/fingerprint checks | Optional |
| `rag_warmup` | Load model and optionally vectors | Yes |
| `rag_reindex` | Targeted, incremental, or forced indexing | As needed |

## Operational commands

```bash
# Lightweight; does not load the model
.venv/bin/brag status

# Validate SQLite and persisted-index compatibility
.venv/bin/brag doctor

# Also load and validate the configured model
.venv/bin/brag doctor --model

# Explicitly warm/inspect heavy resources
.venv/bin/brag warmup --no-vectors
.venv/bin/brag status --model
.venv/bin/brag status --vectors

# Search
.venv/bin/brag search "where is backward attention implemented" --top-k 8
.venv/bin/brag symbol flash_attn_backward
.venv/bin/brag refs flash_attn_backward
.venv/bin/brag docs "how is authentication configured" --top-k 8
.venv/bin/brag outline docs/reference-manual.pdf
```

## Tuning M3 Pro

Benchmark the real model against chunks from the actual repository:

```bash
.venv/bin/brag tune \
  --sample-chunks 128 \
  --batch-sizes 8,16,24,32,48,64
```

Start at batch `32`, flush `1024`. Use `24` or `16` if macOS memory pressure or swap increases. Use the turbo profile only after measuring.

For repositories below roughly the configured `BRAG_MPS_VECTOR_MIN_VECTORS`, `vector_backend=auto` keeps exact matrix search on NumPy/Apple Accelerate to avoid Metal dispatch overhead. Document/query embeddings still use MPS.

## Safety and failure behavior

- The long-lived query connection uses SQLite `mode=ro`/`query_only`; writer connections are short-lived and exist only inside an index mutation.
- MCP writes are disabled unless both `BRAG_READ_ONLY=false` and `BRAG_ALLOW_REINDEX_TOOL=true`.
- Requested paths must resolve inside configured roots.
- Symlink escapes and excluded/cache/vendor directories are rejected.
- Common secret files (`.env`, private keys, credentials/secrets files) are denied by default.
- Tree-sitter grammars are pinned to a bundled offline wheel; indexing never downloads parser manifests on the hot path.
- Invalid roots fail closed instead of silently indexing the server working directory.
- Only one mutation can run at a time across MCP and CLI processes for the same index directory.
- Incomplete scans do not delete apparently missing documents by default.
- Forced rebuilds preserve the previous database until atomic replacement.
- SQLite foreign keys, WAL, busy timeout, integrity checks, and compatibility fingerprints are enabled.
- Logs go to stderr; stdout remains reserved for stdio JSON-RPC.

## Verification

```bash
uv sync --extra code --extra documents --extra dev
uv run ruff check src tests
uv run mypy src/blazing_rag_mcp
uv run bandit -q -r src/blazing_rag_mcp
uv run pytest -q
uv build
```

See `PRODUCTION_AUDIT.md` for the review findings, measured control-path benchmark, fixed bottlenecks, and remaining limitations.

## Scope

This is a fast retrieval/code-navigation index, not a compiler or full LSP database. Reference extraction is intentionally shallow. Exact type resolution, macro expansion, dynamic dispatch, and complete call graphs should still come from language-specific compilers/LSPs.
