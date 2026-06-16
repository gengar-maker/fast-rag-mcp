from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .chunking import chunk_document
from .code_index import extract_references, extract_symbols
from .config import Settings
from .embeddings import EmbeddingModel
from .io import ScanStats, document_id, iter_candidate_files_with_roots, read_document
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
    embedding_texts: list[str]
    embedding_hashes: list[str]
    vector_slots: list[np.ndarray | None]


class Indexer:
    def __init__(self, settings: Settings, store: Store, embeddings: EmbeddingModel):
        self.settings = settings
        self.store = store
        self.embeddings = embeddings

    def index_all(self, *, force: bool = False) -> dict:
        started = time.perf_counter()
        roots = self.settings.resolved_roots()
        scan_stats = ScanStats()
        paths = iter_candidate_files_with_roots(self.settings, scan_stats)
        manifest = {} if force else self.store.document_manifest()

        files_seen = 0
        live_doc_ids: set[str] = set()
        changed_docs = 0
        skipped_current = 0
        skipped_same_content = 0
        chunks_written = 0
        chunks_embedded = 0
        chunks_reused = 0
        symbols_written = 0
        references_written = 0
        errors: list[str] = []

        prepare_seconds = 0.0
        embedding_seconds = 0.0
        storage_seconds = 0.0
        scan_hash_seconds = 0.0
        embedding_calls = 0

        pending: list[_PreparedDocument] = []
        pending_chunks = 0
        flush_limit = max(1, int(self.settings.embedding_flush_chunks))
        touched_docs: list[Document] = []

        def flush_pending() -> None:
            nonlocal pending, pending_chunks
            nonlocal changed_docs, chunks_written, chunks_embedded, chunks_reused
            nonlocal symbols_written, references_written
            nonlocal embedding_seconds, storage_seconds, embedding_calls
            if not pending:
                return

            # Deduplicate missing embedding texts across files in the same flush. Generated repos
            # often contain repeated boilerplate; one GPU inference can serve every identical chunk.
            missing_by_hash: dict[str, str] = {}
            for item in pending:
                for text, emb_hash, slot in zip(
                    item.embedding_texts,
                    item.embedding_hashes,
                    item.vector_slots,
                    strict=True,
                ):
                    if slot is None:
                        missing_by_hash.setdefault(emb_hash, text)

            embedded_by_hash: dict[str, np.ndarray] = {}
            if missing_by_hash:
                hashes = list(missing_by_hash)
                texts = [missing_by_hash[h] for h in hashes]
                embed_started = time.perf_counter()
                vectors = self.embeddings.embed_texts(texts)
                embedding_seconds += time.perf_counter() - embed_started
                embedding_calls += 1
                embedded_by_hash = {h: vectors[i] for i, h in enumerate(hashes)}
                chunks_embedded += len(hashes)

            payloads: list[tuple[Document, list[Chunk], np.ndarray, list[CodeSymbol], list[CodeReference]]] = []
            for item in pending:
                rows: list[np.ndarray] = []
                for emb_hash, slot in zip(item.embedding_hashes, item.vector_slots, strict=True):
                    if slot is not None:
                        rows.append(slot)
                        chunks_reused += 1
                    else:
                        rows.append(embedded_by_hash[emb_hash])
                matrix = np.stack(rows).astype("float32", copy=False) if rows else np.zeros((0, self.embeddings.info.dim), dtype="float32")
                payloads.append((item.doc, item.chunks, matrix, item.symbols, item.references))

            store_started = time.perf_counter()
            try:
                self.store.upsert_documents(payloads)
            except Exception:
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

        for root, path in paths:
            files_seen += 1
            doc_id = document_id(root, path)
            old = manifest.get(doc_id)

            # The common incremental path does only stat() and never opens/hashes unchanged files.
            if not force and self.settings.fast_stat_skip and old is not None:
                try:
                    st = path.stat()
                    if int(old["mtime_ns"]) == int(st.st_mtime_ns) and int(old["size_bytes"]) == int(st.st_size):
                        live_doc_ids.add(doc_id)
                        skipped_current += 1
                        continue
                except OSError:
                    continue

            read_started = time.perf_counter()
            doc = read_document(root, path, self.settings)
            scan_hash_seconds += time.perf_counter() - read_started
            if doc is None:
                continue
            live_doc_ids.add(doc.doc_id)

            # mtime changed but bytes did not: update metadata only, preserving chunks/vectors.
            if not force and old is not None and str(old["content_hash"]) == doc.content_hash:
                touched_docs.append(doc)
                skipped_same_content += 1
                continue

            prep_started = time.perf_counter()
            try:
                symbols = extract_symbols(doc)
                references = extract_references(doc, symbols)
                raw_chunks = chunk_document(doc, self.settings, symbols=symbols)
                if not raw_chunks:
                    continue

                texts: list[str] = []
                hashes: list[str] = []
                chunks: list[Chunk] = []
                for chunk in raw_chunks:
                    text = _embedding_text(chunk)
                    emb_hash = _embedding_hash(self.embeddings, text)
                    meta = dict(chunk.metadata or {})
                    meta["embedding_hash"] = emb_hash
                    chunks.append(replace(chunk, metadata=meta))
                    texts.append(text)
                    hashes.append(emb_hash)

                old_vectors: dict[str, np.ndarray] = {}
                if (
                    not force
                    and old is not None
                    and self.settings.embedding_reuse_unchanged_chunks
                ):
                    old_vectors = self.store.document_vector_cache(doc.doc_id)
                slots = [old_vectors.get(h) for h in hashes]

                pending.append(
                    _PreparedDocument(
                        doc=doc,
                        chunks=chunks,
                        symbols=symbols,
                        references=references,
                        embedding_texts=texts,
                        embedding_hashes=hashes,
                        vector_slots=slots,
                    )
                )
                pending_chunks += len(chunks)
            except Exception as exc:
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

        if touched_docs:
            store_started = time.perf_counter()
            self.store.touch_documents(touched_docs)
            storage_seconds += time.perf_counter() - store_started

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
            "docs_skipped_same_content": skipped_same_content,
            "stale_docs_deleted": stale_deleted,
            "chunks_written": chunks_written,
            "chunks_embedded": chunks_embedded,
            "chunks_reused": chunks_reused,
            "symbols_written": symbols_written,
            "references_written": references_written,
            "errors": errors[:20],
            "error_count": len(errors),
            "corpus_version": version,
            "elapsed_ms": round(elapsed_seconds * 1000, 3),
            "throughput": {
                "chunks_per_second": round(chunks_written / elapsed_seconds, 3) if elapsed_seconds else 0.0,
                "new_embeddings_per_second": round(chunks_embedded / embedding_seconds, 3) if embedding_seconds else 0.0,
                "embedding_calls": embedding_calls,
                "embedding_flush_chunks": flush_limit,
            },
            "timings_ms": {
                "scan_and_hash": round(scan_hash_seconds * 1000, 3),
                "prepare": round(prepare_seconds * 1000, 3),
                "embedding": round(embedding_seconds * 1000, 3),
                "storage": round(storage_seconds * 1000, 3),
                "other": round(max(0.0, elapsed_seconds - scan_hash_seconds - prepare_seconds - embedding_seconds - storage_seconds) * 1000, 3),
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


def _embedding_hash(embeddings: object, text: str) -> str:
    method = getattr(embeddings, "embedding_hash", None)
    if callable(method):
        return str(method(text))
    import hashlib
    info = getattr(embeddings, "info", None)
    model = getattr(info, "model", "unknown")
    return hashlib.blake2b(f"{model}\x1f{text}".encode("utf-8", "ignore"), digest_size=20).hexdigest()
