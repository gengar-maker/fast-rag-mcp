# Production audit - PDF documentation release 1.1.1

## Added PDF path

The PDF pipeline is isolated in `pdf_ingest.py` and is optional at install time. It uses PyMuPDF block extraction with reading-order sorting, page-local chunking, bounded resources, streaming SHA-256 hashing, repeated margin suppression, and optional OCR/table extraction. PDF extraction does not invoke Tree-sitter or write symbols/references.

## Retrieval isolation

`document_search` uses PDF-only FTS filtering and filters oversampled dense candidates to `pdf_*` chunks. Code retrieval remains symbol/path-first and is not slowed by PDF-specific extraction. PDF results expose page number, page label, bounding box, OCR status, and a stable `path#page=N` citation.

## Operational constraints

- OCR requires external Tesseract and is disabled by default.
- Table detection is disabled by default because it can be expensive.
- The configured embedding model is shared by code and PDFs; changing it requires a full rebuild.
- PyMuPDF is optional and has AGPL/commercial licensing terms.
- Index format 3 requires a one-time atomic forced rebuild.

## Verification

- Python compileall passed.
- PDF extraction/chunking/search/outline/fetch tests passed.
- Full test suite passed with MCP, PyMuPDF, and pinned Tree-sitter dependencies installed.
- Ruff, mypy, and Bandit passed.


---

# Production audit

## Executive result

The slow one-file reindex was not primarily caused by parsing or by the number of changed chunks. The old lifecycle performed full dense-runtime work around a small mutation:

1. `rag_reindex` initialized the full search runtime before indexing.
2. The runtime loaded the embedding model and existing vectors.
3. After the update, all vectors were read from SQLite again.
4. The complete in-memory vector index was rebuilt synchronously.
5. Even a no-op update invalidated the vector state, so the next search repeated the load.
6. The unconstrained Tree-sitter language-pack dependency resolved to a network-on-demand 1.x release; cold parser creation could block on a parser-manifest DNS timeout for roughly five seconds.

The production path now treats indexing and retrieval as separate lifecycle concerns.

## Corrected one-file path

```text
code_update_index([changed_file])
  → validate path is inside configured root
  → stat/content comparison
  → parse only changed file
  → reuse unchanged chunk embeddings
  → embed only new/changed chunks
  → one bounded SQLite transaction
  → increment corpus version only on real change
  → clear retrieval cache
  → release stale vector matrix if it had been loaded
  → return immediately

next dense search
  → load current vectors lazily once
```

No-op path:

```text
code_update_index([unchanged_file])
  → stat skip
  → no model load
  → no vector invalidation
  → no vector reload
```

## Measured control-plane benchmark

A synthetic repository with 1,000 tiny Python files and deterministic fake embeddings was used to isolate scanning, parsing, SQLite, and lifecycle overhead from model speed.

| Operation | Time |
|---|---:|
| Initial full index | ~0.68–0.75 s |
| Full no-change consistency scan | ~0.24–0.27 s |
| Targeted one-file update | ~14–16 ms |

Targeted update breakdown in that environment:

| Stage | Time |
|---|---:|
| Scan/hash | ~0.3 ms |
| Prepare/parse/chunk | ~0.7 ms |
| Fake embedding | ~0.1 ms |
| SQLite storage | ~4–6 ms |

These figures are not an M3 Pro neural benchmark. On real hardware, cold `CodeRankEmbed` model initialization and MPS inference add their own time. The server reports `model_load` and `embedding` separately so they can be measured directly.


## Cold parser stall reproduced and removed

The originally unconstrained parser dependency resolved to `tree-sitter-language-pack` 1.x. In an offline test, cold `get_parser("python")` attempted manifest discovery and alternated between an immediate failure and an approximately five-second DNS timeout. That delay was charged to the indexer's `prepare` stage, making a one-file update look slow even with fake embeddings.

Measured in isolated fresh CLI processes against one changed Python file:

| Parser dependency | Index operation time | `prepare` time |
|---|---:|---:|
| Unconstrained 1.x, parser manifest unavailable | ~5.03 s on affected runs | ~5.02 s |
| Pinned bundled 0.7.2 wheel | ~23–26 ms | ~9–11 ms |

The remaining ~0.6–0.7 s fresh-process wall time in that synthetic test was Python/CLI process startup. A persistent MCP process avoids that startup cost. Real neural embedding time is additional and is reported separately.

## Structural changes

### Lifecycle

- Added `Application` as the process-lifetime owner.
- Store, embedding model, vector matrix, and retriever are independently lazy.
- MCP initialization no longer waits for heavy resources.
- Removed the old unconditional `PYTORCH_ENABLE_MPS_FALLBACK=1`; MPS-to-CPU fallback is now explicit because it can dominate point-update latency.
- Mutation is serialized with both an in-process guard and an OS advisory lock shared by MCP/CLI processes.
- Dense state is invalidated only after a real corpus change.
- Removed unconditional global garbage collection from point updates; it added roughly 20–30 ms of stop-the-world overhead in the synthetic benchmark.

### Indexer

- Pinned the bundled offline Tree-sitter parser wheel (`tree-sitter-language-pack==0.7.2`) so parser creation never performs network discovery.
- Added targeted files/directories.
- Preserved full repository consistency scans.
- Added lazy embedding model factory.
- Added cross-file bounded embedding batches.
- Added embedding deduplication and unchanged-chunk reuse.
- Split timings into scan/hash, prepare, model load, embedding, storage, and residual overhead.
- Prevented stale-document deletion after incomplete scans.

### Persistence

- The long-lived retrieval connection opens SQLite in `mode=ro` with `query_only=ON`; index mutations use separate short-lived writer connections.
- SQLite WAL for active index.
- Disposable unsafe-fast pragmas only for staging rebuilds.
- Atomic database replacement after successful full build.
- Foreign keys, busy timeout, cache/mmap tuning, integrity checks.
- Persisted schema version, index state, corpus version, embedding compatibility, and index fingerprint.
- Batched writes and deletes.

### Retrieval

- Exact symbol/path/FTS routes do not load embeddings or vectors.
- Exact identifier routing can bypass dense retrieval.
- Search cache is keyed by corpus version.
- Vector matrix reload is lazy after mutations.
- Small-index exact search can remain on CPU/Accelerate; larger matrices can use MPS.

### Stable identities

Index format 2 introduced stable identities that are retained in format 3. It removes full-file hash and byte offsets from symbol/chunk identities. An unrelated line insertion no longer changes IDs for unchanged functions/chunks. This improves cached references and agent navigation stability.

## Security hardening

- Common secret files are excluded before reading.
- Invalid configured roots raise an error instead of falling back to the current directory.
- Git is invoked through a resolved executable with fixed argv and no shell.
- Dynamic SQL fragments are limited to generated placeholder lists or static clauses; values remain bound parameters.
- MCP mutation is fail-closed and requires two explicit settings.
- Remote model code produces a doctor warning until a model revision is pinned.

## Production defaults

- MCP read-only: enabled, including a true SQLite read-only retrieval connection.
- MCP reindex tools: disabled.
- Hash embedding fallback: disabled.
- FAISS GPU on macOS: disabled.
- File-summary chunks: disabled.
- FP16 vector storage: enabled in M3 profile.
- Incomplete-scan deletion: disabled.
- Model loading: lazy.
- Vector loading: lazy.

## Upgrade requirement

The current release uses index format `3` for page-aware PDF metadata. Run once:

```bash
.venv/bin/brag index --force
```

A compatibility mismatch is rejected rather than silently mixing vectors/chunks from different configurations.

## Remaining limitations

- The first changed-file update in a fresh CLI process must load the embedding model if a new embedding is needed. Persistent MCP plus `rag_warmup(load_vectors=false)` avoids repeated cold starts.
- After a real update, the next dense search loads the current vector matrix. This is deferred rather than included in update latency.
- References include bare and member calls such as `self.method()` and `client.fetch()`, but remain lexical/structural rather than compiler-resolved.
- Multi-process writers for the same index directory are rejected by an OS advisory lock; network filesystems with unreliable advisory locking are unsupported.
- The repository root is part of document identity; moving a repository to a new absolute root requires a rebuild.
- No filesystem watcher is bundled. Editors/agents should call targeted update tools, with occasional full consistency scans.

## Release validation

- Python bytecode compilation.
- Ruff static checks.
- Ruff, mypy, Bandit, and Python bytecode checks.
- Unit/integration tests for no-op model avoidance, one-file targeting, deferred vector reload, exact-search lazy behavior, no-op vector validity, chunk reuse, cross-file batching, stable IDs, offline parser startup, true read-only SQLite access, interprocess writer exclusion, and member-call extraction.
- Package build to wheel and source distribution.
- MCP initialize/tools-list handshake test after dependency installation.
