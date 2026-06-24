# Changelog

## 1.1.0

- Added production PDF ingestion through the optional `documents` extra (`PyMuPDF`).
- Added page-aware `pdf_text`, `pdf_code`, and optional `pdf_table` chunks.
- Added repeated header/footer suppression, heading and monospace-block detection, page labels, bounding boxes, encrypted-PDF handling, streaming file hashes, and bounded page/character limits.
- Added opt-in Tesseract OCR modes: `off`, `auto`, and `always`.
- Added `document_search`, `document_outline`, `document_fetch`, and `document_update_index` MCP tools plus `brag docs` and `brag outline` CLI commands.
- Added PDF-only FTS filtering and dense-result oversampling with document-specific query prefixing.
- Added contextual adjacent-chunk fetch and page citations.
- Bumped persisted index format to 3; a one-time forced rebuild is required.
- Added PDF extraction/index/search tests.
- Updated the Python 3.14 installation path to `transformers==4.47.1` and `tokenizers==0.21.4`; the latter ships an ABI3 wheel and avoids a local Rust build.

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
