# Changelog

## 1.0.1

- Pin `sentence-transformers==3.4.1` and `transformers==4.45.1` for CodeRankEmbed compatibility.
- Add direct `einops` and compatible `huggingface-hub` dependencies required by remote model code.
- Regenerate `uv.lock`; the previous lock could install Transformers 5.x.
- Add explicit embedding cache directory support through `BRAG_EMBEDDING_CACHE_DIR`.
- Surface the original model-load exception, package versions and cache diagnostics.
- Fail early with a targeted error when CodeRankEmbed is used with Transformers 5.x.

## 1.0.0

- Pinned bundled offline Tree-sitter grammars to eliminate cold-path network/DNS stalls.
- Pinned the reviewed CodeRankEmbed model revision in production profiles.

- Added a true SQLite read-only retrieval connection; writes use separate locked writer connections.
- Added interprocess writer exclusion for concurrent CLI/MCP mutations.
- Added member-call reference extraction for `self.method()`, `client.fetch()`, and module-qualified calls.
- Introduced production `Application` lifecycle with independent lazy resources.
- Added targeted `code_update_index` and path-aware `rag_reindex`.
- Removed synchronous full vector-index rebuild from incremental updates.
- Fixed no-op updates invalidating loaded vectors.
- Added atomic staging rebuilds and SQLite integrity/state metadata.
- Added stable symbol/chunk IDs with index format 2.
- Added model-load timing and production diagnostics.
- Removed unconditional global GC from incremental vector invalidation.
- Added read-only MCP policy by default.
- Added M3 Pro 36 GB deployment profiles for Claude Code and opencode.
