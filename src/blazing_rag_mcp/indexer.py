from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .chunking import chunk_document
from .code_index import extract_references, extract_symbols
from .config import Settings
from .embeddings import EmbeddingModel
from .io import (
    RequestedPaths,
    ScanStats,
    document_id,
    iter_candidate_files_with_roots,
    read_document,
    resolve_requested_paths,
)
from .store import Store
from .types import Chunk, CodeReference, CodeSymbol, Document
from .vector import VectorIndex

log = logging.getLogger(__name__)
EmbeddingSource = EmbeddingModel | Callable[[], EmbeddingModel]


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
    """Incremental indexer with lazy model loading and bounded cross-file batching."""

    def __init__(self, settings: Settings, store: Store, embeddings: EmbeddingSource):
        self.settings = settings
        self.store = store
        self._embedding_source = embeddings
        self._embeddings: EmbeddingModel | None = (
            embeddings if isinstance(embeddings, EmbeddingModel) else None
        )

    @property
    def embeddings_loaded(self) -> bool:
        return self._embeddings is not None

    def _get_embeddings(self) -> EmbeddingModel:
        if self._embeddings is None:
            source = self._embedding_source
            self._embeddings = source() if callable(source) else source
        return self._embeddings

    def index_all(
        self,
        *,
        force: bool = False,
        paths: Sequence[str] | None = None,
    ) -> dict:
        started = time.perf_counter()
        roots = self.settings.resolved_roots()
        scan_stats = ScanStats(roots=[path.as_posix() for path in roots])
        targeted = paths is not None
        requested: RequestedPaths | None = None
        if targeted:
            requested = resolve_requested_paths(self.settings, list(paths or []))
            if not requested.existing and not requested.missing:
                rejected = ", ".join(requested.rejected) or "no paths"
                raise ValueError(f"no indexable paths were resolved: {rejected}")
            iterator: Iterable[tuple[Path, Path]] = requested.existing
        else:
            iterator = iter_candidate_files_with_roots(self.settings, scan_stats)

        fingerprint = self.settings.index_fingerprint()
        if not force:
            self.store.validate_index_fingerprint(fingerprint)
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
        changed_paths: list[str] = []

        prepare_seconds = 0.0
        model_load_seconds = 0.0
        embedding_seconds = 0.0
        storage_seconds = 0.0
        scan_hash_seconds = 0.0
        embedding_calls = 0

        pending: list[_PreparedDocument] = []
        pending_chunks = 0
        flush_limit = max(1, int(self.settings.embedding_flush_chunks))
        touched_docs: list[Document] = []
        missing_doc_ids = [
            document_id(root, path) for root, path in (requested.missing if requested else [])
        ]

        self.store.begin_index_run()

        def count_payload(payload) -> None:
            nonlocal changed_docs, chunks_written, symbols_written, references_written
            doc, chunks, _, symbols, references = payload
            changed_docs += 1
            changed_paths.append(doc.rel_path)
            chunks_written += len(chunks)
            symbols_written += len(symbols)
            references_written += len(references)

        def flush_pending() -> None:
            nonlocal pending, pending_chunks, chunks_embedded, chunks_reused
            nonlocal model_load_seconds, embedding_seconds, storage_seconds, embedding_calls
            if not pending:
                return

            missing_by_hash: dict[str, str] = {}
            for item in pending:
                for text, embedding_hash, slot in zip(
                    item.embedding_texts,
                    item.embedding_hashes,
                    item.vector_slots,
                    strict=True,
                ):
                    if slot is None:
                        missing_by_hash.setdefault(embedding_hash, text)

            embedded_by_hash: dict[str, np.ndarray] = {}
            if missing_by_hash:
                model_started = time.perf_counter()
                embeddings = self._get_embeddings()
                model_load_seconds += time.perf_counter() - model_started
                hashes = list(missing_by_hash)
                texts = [missing_by_hash[value] for value in hashes]
                embed_started = time.perf_counter()
                vectors = embeddings.embed_texts(texts)
                embedding_seconds += time.perf_counter() - embed_started
                embedding_calls += 1
                embedded_by_hash = {value: vectors[index] for index, value in enumerate(hashes)}
                chunks_embedded += len(hashes)

            payloads: list[
                tuple[Document, list[Chunk], np.ndarray, list[CodeSymbol], list[CodeReference]]
            ] = []
            for item in pending:
                rows: list[np.ndarray] = []
                for embedding_hash, slot in zip(
                    item.embedding_hashes, item.vector_slots, strict=True
                ):
                    if slot is not None:
                        rows.append(slot)
                        chunks_reused += 1
                    else:
                        rows.append(embedded_by_hash[embedding_hash])
                if rows:
                    matrix = np.stack(rows).astype("float32", copy=False)
                else:
                    dim = self._get_embeddings().info.dim
                    matrix = np.zeros((0, dim), dtype="float32")
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
                        errors.append(f"{doc.path.as_posix()}: {exc}")
                        log.exception("failed to store %s", doc.path)
                        continue
                    count_payload(payload)
            else:
                for payload in payloads:
                    count_payload(payload)
            storage_seconds += time.perf_counter() - store_started
            pending = []
            pending_chunks = 0

        try:
            for root, path in iterator:
                files_seen += 1
                doc_id = document_id(root, path)
                old = manifest.get(doc_id)
                # The scanner proved the file exists. Preserve its previous record on transient read errors.
                live_doc_ids.add(doc_id)

                if not force and self.settings.fast_stat_skip and old is not None:
                    try:
                        stat = path.stat()
                        if int(old["mtime_ns"]) == int(stat.st_mtime_ns) and int(
                            old["size_bytes"]
                        ) == int(stat.st_size):
                            skipped_current += 1
                            continue
                    except OSError:
                        continue

                read_started = time.perf_counter()
                doc = read_document(root, path, self.settings)
                scan_hash_seconds += time.perf_counter() - read_started
                if doc is None:
                    errors.append(f"{path.as_posix()}: unreadable or unsupported file")
                    continue

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
                        embedding_hash = _embedding_hash(fingerprint, text)
                        metadata = dict(chunk.metadata or {})
                        metadata["embedding_hash"] = embedding_hash
                        chunks.append(replace(chunk, metadata=metadata))
                        texts.append(text)
                        hashes.append(embedding_hash)

                    old_vectors: dict[str, np.ndarray] = {}
                    if (
                        not force
                        and old is not None
                        and self.settings.embedding_reuse_unchanged_chunks
                    ):
                        old_vectors = self.store.document_vector_cache(doc.doc_id)
                    slots = [old_vectors.get(value) for value in hashes]
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
                    errors.append(f"{path.as_posix()}: {exc}")
                    log.exception("failed to prepare %s", path)
                finally:
                    prepare_seconds += time.perf_counter() - prep_started

                if pending_chunks >= flush_limit:
                    flush_pending()

            flush_pending()

            if touched_docs:
                store_started = time.perf_counter()
                self.store.touch_documents(touched_docs)
                storage_seconds += time.perf_counter() - store_started

            deleted = self.store.delete_docs(missing_doc_ids)
            if not targeted:
                safe_to_delete = (
                    scan_stats.complete or self.settings.delete_missing_on_incomplete_scan
                )
                if safe_to_delete:
                    deleted += self.store.delete_missing_docs(live_doc_ids)
                elif not scan_stats.complete:
                    errors.append(
                        "scan was incomplete; stale-document deletion was skipped for safety"
                    )

            if changed_docs or deleted or force:
                version = self.store.mark_corpus_changed()
            else:
                version = self.store.corpus_version()

            embedding_key = self.store.get_meta("embedding_key", "")
            embedding_dim = int(self.store.get_meta("embedding_dim", "0") or 0)
            if self._embeddings is not None:
                embedding_key = str(
                    getattr(
                        self._embeddings,
                        "index_key",
                        f"{self._embeddings.info.model}|{self._embeddings.info.dim}|"
                        f"{self._embeddings.info.normalized}",
                    )
                )
                embedding_dim = self._embeddings.info.dim
            if changed_docs or deleted or force:
                self.store.record_index_metadata(
                    fingerprint=fingerprint,
                    embedding_key=embedding_key,
                    embedding_dim=embedding_dim,
                )
            self.store.finish_index_run(degraded=bool(errors))
        except Exception as exc:
            self.store.fail_index_run(f"{type(exc).__name__}: {exc}")
            raise

        elapsed_seconds = time.perf_counter() - started
        embedding_info = (
            {"loaded": True, **asdict(self._embeddings.info)}
            if self._embeddings is not None
            else {
                "loaded": False,
                "model": self.settings.embedding_model,
                "reason": "no new embeddings were required",
            }
        )
        return {
            "roots": [path.as_posix() for path in roots],
            "targeted": targeted,
            "requested_paths": list(paths or []),
            "rejected_paths": requested.rejected if requested else [],
            "files_seen": files_seen,
            "scan": scan_stats.as_dict(),
            "docs_changed": changed_docs,
            "changed_paths": changed_paths[:100],
            "docs_skipped_current": skipped_current,
            "docs_skipped_same_content": skipped_same_content,
            "stale_docs_deleted": deleted,
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
                "chunks_per_second": (
                    round(chunks_written / elapsed_seconds, 3) if elapsed_seconds else 0.0
                ),
                "new_embeddings_per_second": (
                    round(chunks_embedded / embedding_seconds, 3) if embedding_seconds else 0.0
                ),
                "embedding_calls": embedding_calls,
                "embedding_flush_chunks": flush_limit,
            },
            "timings_ms": {
                "scan_and_hash": round(scan_hash_seconds * 1000, 3),
                "prepare": round(prepare_seconds * 1000, 3),
                "model_load": round(model_load_seconds * 1000, 3),
                "embedding": round(embedding_seconds * 1000, 3),
                "storage": round(storage_seconds * 1000, 3),
                "other": round(
                    max(
                        0.0,
                        elapsed_seconds
                        - scan_hash_seconds
                        - prepare_seconds
                        - model_load_seconds
                        - embedding_seconds
                        - storage_seconds,
                    )
                    * 1000,
                    3,
                ),
            },
            "embedding": embedding_info,
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
    metadata = chunk.metadata or {}
    parts = [
        f"path: {chunk.path}",
        f"language: {metadata.get('language', '')}",
        f"chunk_type: {metadata.get('chunk_type', '')}",
    ]
    if metadata.get("qualified_name"):
        parts.append(f"symbol: {metadata.get('qualified_name')}")
    if metadata.get("symbol_kind"):
        parts.append(f"kind: {metadata.get('symbol_kind')}")
    if metadata.get("signature"):
        parts.append(f"signature: {metadata.get('signature')}")
    if metadata.get("docstring"):
        parts.append(f"docs: {metadata.get('docstring')}")
    parts.append(chunk.text)
    return "\n".join(str(part) for part in parts if part)


def _embedding_hash(index_fingerprint: str, text: str) -> str:
    return hashlib.blake2b(
        f"{index_fingerprint}\x1f{text}".encode("utf-8", "ignore"), digest_size=20
    ).hexdigest()
