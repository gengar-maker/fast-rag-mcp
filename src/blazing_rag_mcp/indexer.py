from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import chunk_document
from .code_index import extract_references, extract_symbols
from .config import Settings
from .embeddings import EmbeddingModel
from .io import ScanStats, iter_candidate_files, read_document
from .store import Store
from .types import Chunk, CodeReference, CodeSymbol, Document
from .vector import VectorIndex

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _PreparedDocument:
    doc: Document
    chunks: list[Chunk]
    symbols: list[CodeSymbol]
    references: list[CodeReference]


class Indexer:
    def __init__(self, settings: Settings, store: Store, embeddings: EmbeddingModel):
        self.settings = settings
        self.store = store
        self.embeddings = embeddings

    def index_all(self, *, force: bool = False) -> dict:
        started = time.perf_counter()
        roots = self.settings.resolved_roots()
        root_by_prefix = sorted(roots, key=lambda p: len(p.as_posix()), reverse=True)
        scan_stats = ScanStats()
        paths = iter_candidate_files(self.settings, scan_stats)

        files_seen = 0
        live_doc_ids: set[str] = set()
        changed_docs = 0
        skipped_current = 0
        chunks_written = 0
        symbols_written = 0
        references_written = 0
        errors: list[str] = []

        prepare_seconds = 0.0
        embedding_seconds = 0.0
        storage_seconds = 0.0
        embedding_calls = 0

        pending: list[_PreparedDocument] = []
        pending_chunks = 0
        flush_limit = max(1, int(self.settings.embedding_flush_chunks))

        def flush_pending() -> None:
            nonlocal pending, pending_chunks
            nonlocal changed_docs, chunks_written, symbols_written, references_written
            nonlocal embedding_seconds, storage_seconds, embedding_calls
            if not pending:
                return

            flat_chunks = [chunk for item in pending for chunk in item.chunks]
            texts = [_embedding_text(chunk) for chunk in flat_chunks]

            embed_started = time.perf_counter()
            vectors = self.embeddings.embed_texts(texts)
            embedding_seconds += time.perf_counter() - embed_started
            embedding_calls += 1

            payloads: list[tuple[Document, list[Chunk], np.ndarray, list[CodeSymbol], list[CodeReference]]] = []
            offset = 0
            for item in pending:
                count = len(item.chunks)
                payloads.append(
                    (
                        item.doc,
                        item.chunks,
                        vectors[offset : offset + count],
                        item.symbols,
                        item.references,
                    )
                )
                offset += count

            store_started = time.perf_counter()
            try:
                self.store.upsert_documents(payloads)
            except Exception:
                # Keep one malformed document from rolling back the entire GPU batch.
                log.exception("batch store failed; retrying documents individually")
                for payload in payloads:
                    try:
                        self.store.upsert_document(*payload)
                    except Exception as exc:
                        doc = payload[0]
                        msg = f"{doc.path.as_posix()}: {exc}"
                        log.exception("failed to store %s", doc.path)
                        errors.append(msg)
                        continue
                    _count_payload(payload)
            else:
                for payload in payloads:
                    _count_payload(payload)
            storage_seconds += time.perf_counter() - store_started

            pending = []
            pending_chunks = 0

        def _count_payload(payload) -> None:
            nonlocal changed_docs, chunks_written, symbols_written, references_written
            _, chunks, _, symbols, references = payload
            changed_docs += 1
            chunks_written += len(chunks)
            symbols_written += len(symbols)
            references_written += len(references)

        for path in paths:
            files_seen += 1
            root = _find_root(path, root_by_prefix)
            if root is None:
                continue
            doc = read_document(root, path, self.settings)
            if doc is None:
                continue
            live_doc_ids.add(doc.doc_id)
            if not force and self.store.doc_is_current(doc):
                skipped_current += 1
                continue

            prep_started = time.perf_counter()
            try:
                symbols = extract_symbols(doc)
                references = extract_references(doc, symbols)
                chunks = chunk_document(doc, self.settings, symbols=symbols)
                if not chunks:
                    continue
                pending.append(
                    _PreparedDocument(
                        doc=doc,
                        chunks=chunks,
                        symbols=symbols,
                        references=references,
                    )
                )
                pending_chunks += len(chunks)
            except Exception as exc:  # keep indexing robust across weird files
                msg = f"{path.as_posix()}: {exc}"
                log.exception("failed to prepare %s", path)
                errors.append(msg)
            finally:
                prepare_seconds += time.perf_counter() - prep_started

            if pending_chunks >= flush_limit:
                try:
                    flush_pending()
                except Exception as exc:
                    log.exception("embedding batch failed")
                    errors.append(f"embedding batch: {exc}")
                    pending = []
                    pending_chunks = 0

        if pending:
            try:
                flush_pending()
            except Exception as exc:
                log.exception("final embedding batch failed")
                errors.append(f"final embedding batch: {exc}")

        stale_deleted = self.store.delete_missing_docs(live_doc_ids)
        if changed_docs or stale_deleted or force:
            version = self.store.mark_corpus_changed()
        else:
            version = self.store.corpus_version()

        elapsed_seconds = time.perf_counter() - started
        return {
            "roots": [p.as_posix() for p in roots],
            "files_seen": files_seen,
            "scan": scan_stats.as_dict(),
            "docs_changed": changed_docs,
            "docs_skipped_current": skipped_current,
            "stale_docs_deleted": stale_deleted,
            "chunks_written": chunks_written,
            "symbols_written": symbols_written,
            "references_written": references_written,
            "errors": errors[:20],
            "error_count": len(errors),
            "corpus_version": version,
            "elapsed_ms": round(elapsed_seconds * 1000, 3),
            "throughput": {
                "chunks_per_second": round(chunks_written / elapsed_seconds, 3) if elapsed_seconds else 0.0,
                "embedding_calls": embedding_calls,
                "embedding_flush_chunks": flush_limit,
            },
            "timings_ms": {
                "prepare": round(prepare_seconds * 1000, 3),
                "embedding": round(embedding_seconds * 1000, 3),
                "storage": round(storage_seconds * 1000, 3),
                "other": round(max(0.0, elapsed_seconds - prepare_seconds - embedding_seconds - storage_seconds) * 1000, 3),
            },
            "embedding": asdict(self.embeddings.info),
        }

    def rebuild_vector_index(self, vector_index: VectorIndex) -> dict:
        started = time.perf_counter()
        ids, vectors = self.store.all_vectors()
        vector_index.build(ids, vectors)
        return {
            "vectors": len(ids),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "index": asdict(vector_index.info()),
        }


def _embedding_text(chunk: Chunk) -> str:
    meta = chunk.metadata or {}
    parts = [
        f"path: {chunk.path}",
        f"language: {meta.get('language', '')}",
        f"chunk_type: {meta.get('chunk_type', '')}",
    ]
    if meta.get("qualified_name"):
        parts.append(f"symbol: {meta.get('qualified_name')}")
    if meta.get("symbol_kind"):
        parts.append(f"kind: {meta.get('symbol_kind')}")
    if meta.get("signature"):
        parts.append(f"signature: {meta.get('signature')}")
    if meta.get("docstring"):
        parts.append(f"docs: {meta.get('docstring')}")
    parts.append(chunk.text)
    return "\n".join(str(p) for p in parts if p)


def _find_root(path: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        try:
            path.relative_to(root)
            return root
        except ValueError:
            continue
    return None
