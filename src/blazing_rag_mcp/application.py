from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings
from .embeddings import EmbeddingModel, embedding_environment
from .indexer import Indexer
from .locking import IndexMutationLock
from .retrieval import Retriever
from .store import Store, replace_index_database
from .vector import VectorIndex

log = logging.getLogger(__name__)


class Application:
    """Process-lifetime application service.

    Heavy components are lazy and independently managed:
    - SQLite metadata/FTS can answer exact queries without loading a model.
    - The embedding model loads only when new embeddings or dense search are required.
    - The vector matrix loads only on the first dense search and is invalidated, not rebuilt,
      after an incremental reindex.

    A single re-entrant lock protects the local SQLite connection and lazy resources. MCP stdio is
    normally single-client; serializing mutations makes failure behavior deterministic.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._lock = threading.RLock()
        self._index_lock = threading.Lock()
        self._store: Store | None = None
        self._embeddings: EmbeddingModel | None = None
        self._vector_index: VectorIndex | None = None
        self._vector_version: str | None = None
        self._retriever: Retriever | None = None
        self._model_init_ms: float | None = None
        self._vector_init_ms: float | None = None

    def _new_store(
        self,
        db_dir: Path | None = None,
        *,
        bulk_build: bool = False,
        read_only: bool = False,
    ) -> Store:
        return Store(
            db_dir or self.settings.resolved_db_dir(),
            vector_storage_dtype=self.settings.vector_storage_dtype,
            bulk_build=bulk_build,
            timeout_seconds=self.settings.sqlite_timeout_seconds,
            busy_timeout_ms=self.settings.sqlite_busy_timeout_ms,
            cache_kib=self.settings.sqlite_cache_kib,
            mmap_bytes=self.settings.sqlite_mmap_bytes,
            vector_load_batch_size=self.settings.vector_load_batch_size,
            read_only=read_only,
        )

    def _get_store(self) -> Store:
        if self._store is None:
            # Retrieval never needs a write-capable connection. Index mutations use a separate
            # writer Store under the inter-process lock.
            self._store = self._new_store(read_only=True)
        return self._store

    def _get_embeddings(self) -> EmbeddingModel:
        if self._embeddings is None:
            started = time.perf_counter()
            self._embeddings = EmbeddingModel(self.settings)
            self._model_init_ms = round((time.perf_counter() - started) * 1000, 3)
            log.info("embedding model initialized in %.3f ms", self._model_init_ms)
        return self._embeddings

    def _validate_dense_compatibility(self, store: Store, embeddings: EmbeddingModel) -> None:
        store.validate_index_fingerprint(self.settings.index_fingerprint())
        stored_key = store.get_meta("embedding_key", "")
        stored_dim = int(store.get_meta("embedding_dim", "0") or 0)
        if stored_key and stored_key != embeddings.index_key:
            raise RuntimeError(
                "configured embedding runtime does not match the persisted index; "
                "run `brag index --force` with the same environment"
            )
        if stored_dim and stored_dim != embeddings.info.dim:
            raise RuntimeError(
                f"embedding dimension mismatch: index={stored_dim}, runtime={embeddings.info.dim}"
            )

    def _get_vector_index(self) -> VectorIndex:
        store = self._get_store()
        current_version = store.corpus_version()
        if self._vector_index is not None and self._vector_version == current_version:
            return self._vector_index

        embeddings = self._get_embeddings()
        self._validate_dense_compatibility(store, embeddings)
        started = time.perf_counter()
        ids, vectors = store.all_vectors()
        vector_index = VectorIndex(self.settings)
        vector_index.build(ids, vectors)
        self._vector_index = vector_index
        self._vector_version = current_version
        self._vector_init_ms = round((time.perf_counter() - started) * 1000, 3)
        log.info(
            "vector index loaded version=%s vectors=%s in %.3f ms",
            current_version,
            len(ids),
            self._vector_init_ms,
        )
        return vector_index

    def _assert_queryable(self) -> None:
        state = self._get_store().get_meta("index_state", "unknown")
        if state in {"building", "failed"}:
            raise RuntimeError(
                f"index is not queryable (state={state}); run `brag doctor` and reindex"
            )

    def _get_retriever(self) -> Retriever:
        self._assert_queryable()
        if self._retriever is None:
            self._retriever = Retriever(
                self.settings,
                self._get_store(),
                self._get_embeddings,
                self._get_vector_index,
            )
        return self._retriever

    def _invalidate_vectors(self, *, release: bool = True) -> None:
        self._vector_version = None
        if release:
            # Dropping the final reference is immediate in CPython. A global gc.collect() here
            # adds stop-the-world latency to every point update and is unnecessary for tensors.
            self._vector_index = None
        if self._retriever is not None:
            self._retriever.clear_cache()

    def _reset_store_after_swap(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store = None
        self._vector_index = None
        self._vector_version = None
        self._retriever = None

    def _configuration_warnings(self) -> list[str]:
        warnings: list[str] = []
        fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower()
        if fallback_env in {"1", "true", "yes", "on"}:
            warnings.append(
                "PYTORCH_ENABLE_MPS_FALLBACK is enabled; unsupported Metal operations may run "
                "on CPU and create large latency spikes"
            )
        if self.settings.embedding_trust_remote_code and not self.settings.embedding_revision:
            warnings.append(
                "embedding remote code is enabled without BRAG_EMBEDDING_REVISION; pin a reviewed "
                "model commit for reproducible production deployments"
            )
        if not self.settings.read_only and self.settings.allow_reindex_tool:
            warnings.append(
                "MCP index mutation is enabled; restrict BRAG_ROOTS/BRAG_DB_DIR and use targeted "
                "updates instead of force rebuilds"
            )
        return warnings

    def code_search(self, **kwargs: Any) -> dict:
        with self._lock:
            return self._get_retriever().code_search(**kwargs)

    def document_search(self, **kwargs: Any) -> dict:
        with self._lock:
            return self._get_retriever().document_search(**kwargs)

    def document_outline(self, **kwargs: Any) -> dict:
        with self._lock:
            return self._get_retriever().document_outline(**kwargs)

    def find_symbol(self, **kwargs: Any) -> dict:
        with self._lock:
            return self._get_retriever().find_symbol(**kwargs)

    def references(self, **kwargs: Any) -> dict:
        with self._lock:
            return self._get_retriever().references(**kwargs)

    def neighbors(self, **kwargs: Any) -> dict:
        with self._lock:
            return self._get_retriever().neighbors(**kwargs)

    def repo_map(self, **kwargs: Any) -> dict:
        with self._lock:
            return self._get_retriever().repo_map(**kwargs)

    def fetch(self, **kwargs: Any) -> dict:
        with self._lock:
            return self._get_retriever().fetch(**kwargs)

    def reindex(
        self,
        *,
        force: bool = False,
        paths: list[str] | None = None,
        refresh_vectors: bool = False,
        enforce_mcp_policy: bool = False,
    ) -> dict:
        operation_started = time.perf_counter()
        if enforce_mcp_policy:
            if self.settings.read_only:
                raise PermissionError(
                    "server is read-only; set BRAG_READ_ONLY=false to mutate the index"
                )
            if not self.settings.allow_reindex_tool:
                raise PermissionError(
                    "reindex tools are disabled; set BRAG_ALLOW_REINDEX_TOOL=true"
                )
        if paths is not None and not paths:
            raise ValueError(
                "paths must be omitted for a repository scan or contain at least one path"
            )
        if not self._index_lock.acquire(blocking=False):
            raise RuntimeError("an indexing operation is already running in this process")
        mutation_lock = IndexMutationLock(self.settings.resolved_db_dir())
        if not mutation_lock.acquire():
            self._index_lock.release()
            raise RuntimeError("another process is already mutating this index")
        try:
            with self._lock:
                if force and paths is not None:
                    raise ValueError("force rebuild and targeted paths are mutually exclusive")
                if force:
                    result = self._force_rebuild()
                else:
                    index_store = self._new_store()
                    try:
                        indexer = Indexer(self.settings, index_store, self._get_embeddings)
                        result = indexer.index_all(paths=paths)
                    finally:
                        index_store.close()
                    vector_loaded_before = self._vector_index is not None
                    corpus_changed = bool(
                        result.get("docs_changed") or result.get("stale_docs_deleted")
                    )
                    if corpus_changed:
                        self._invalidate_vectors(release=True)
                if force:
                    vector_loaded_before = False
                    corpus_changed = True

                if refresh_vectors and corpus_changed:
                    vector = self._get_vector_index()
                    result["vector_refresh"] = {
                        "deferred": False,
                        "index": asdict(vector.info()),
                        "elapsed_ms": self._vector_init_ms,
                    }
                else:
                    result["vector_refresh"] = {
                        "deferred": True,
                        "reason": (
                            "vector state is invalidated and will reload on the next dense search"
                            if corpus_changed and vector_loaded_before
                            else (
                                "corpus did not change; current vector index remains valid"
                                if not corpus_changed
                                else "vector index was not loaded"
                            )
                        ),
                    }
                result["operation_elapsed_ms"] = round(
                    (time.perf_counter() - operation_started) * 1000, 3
                )
                return result
        finally:
            mutation_lock.release()
            self._index_lock.release()

    def _force_rebuild(self) -> dict:
        target_dir = self.settings.resolved_db_dir()
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{target_dir.name}.rebuild-", dir=target_dir.parent)
        )
        staging_settings = self.settings.model_copy(update={"db_dir": staging_dir})
        staging_store = self._new_store(staging_dir, bulk_build=True)
        try:
            indexer = Indexer(staging_settings, staging_store, self._get_embeddings)
            result = indexer.index_all(force=True)
            staging_store.optimize()
            staging_store.checkpoint()
            staging_store.close()
            self._reset_store_after_swap()
            replace_index_database(staging_dir, target_dir)
            result["full_rebuild"] = True
            result["db_swap"] = "completed"
            return result
        finally:
            with suppress(Exception):
                staging_store.close()
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def warmup(self, *, load_vectors: bool = True) -> dict:
        with self._lock:
            embeddings = self._get_embeddings()
            embeddings.warmup()
            vector_info = None
            if load_vectors:
                vector_info = asdict(self._get_vector_index().info())
            return {
                "ok": True,
                "model_init_ms": self._model_init_ms,
                "vector_init_ms": self._vector_init_ms,
                "embedding": asdict(embeddings.info),
                "index": vector_info or {"loaded": False},
            }

    def status(self, *, load_model: bool = False, load_vectors: bool = False) -> dict:
        with self._lock:
            store = self._get_store()
            if load_vectors:
                self._get_vector_index()
            elif load_model:
                self._get_embeddings()
            return {
                "ok": True,
                "version": __version__,
                "settings": {
                    "db_dir": self.settings.resolved_db_dir().as_posix(),
                    "roots": [path.as_posix() for path in self.settings.resolved_roots()],
                    "embedding_model": self.settings.embedding_model,
                    "embedding_cache_dir": (
                        self.settings.embedding_cache_dir.expanduser().resolve().as_posix()
                        if self.settings.embedding_cache_dir is not None
                        else "default"
                    ),
                    "device": self.settings.device,
                    "vector_backend": self.settings.vector_backend,
                    "read_only": self.settings.read_only,
                    "allow_reindex_tool": self.settings.allow_reindex_tool,
                    "mps_enable_fallback": self.settings.mps_enable_fallback,
                    "pdf": {
                        "max_pdf_bytes": self.settings.max_pdf_bytes,
                        "max_pages": self.settings.pdf_max_pages,
                        "chunk_tokens": self.settings.pdf_chunk_tokens,
                        "ocr_mode": self.settings.pdf_ocr_mode,
                        "extract_tables": self.settings.pdf_extract_tables,
                    },
                    "index_fingerprint": self.settings.index_fingerprint(),
                },
                "embedding_environment": embedding_environment(),
                "environment": {
                    "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get(
                        "PYTORCH_ENABLE_MPS_FALLBACK", ""
                    ),
                    "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", ""),
                    "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS", ""),
                },
                "warnings": self._configuration_warnings(),
                "store": store.stats(),
                "resources": {
                    "embedding_loaded": self._embeddings is not None,
                    "vector_loaded": self._vector_index is not None,
                    "vector_version": self._vector_version,
                    "model_init_ms": self._model_init_ms,
                    "vector_init_ms": self._vector_init_ms,
                },
                "embedding": asdict(self._embeddings.info) if self._embeddings else None,
                "index": asdict(self._vector_index.info()) if self._vector_index else None,
            }

    def doctor(self, *, load_model: bool = False) -> dict:
        with self._lock:
            store = self._get_store()
            checks: dict[str, Any] = {
                "embedding_environment": embedding_environment(),
                "database": store.integrity_check(),
                "fingerprint_matches": (
                    store.document_count() == 0
                    or store.get_meta("index_fingerprint", "") == self.settings.index_fingerprint()
                ),
                "index_state": store.get_meta("index_state", "unknown"),
            }
            if load_model:
                embeddings = self._get_embeddings()
                try:
                    self._validate_dense_compatibility(store, embeddings)
                    checks["embedding_compatible"] = True
                except Exception as exc:
                    checks["embedding_compatible"] = False
                    checks["embedding_error"] = f"{type(exc).__name__}: {exc}"
            database = checks["database"]
            database_ok = (
                database.get("quick_check") == "ok"
                and int(database.get("foreign_key_violations", 0)) == 0
            )
            state_ok = checks["index_state"] in {"ready", "degraded", "unknown"}
            embedding_ok = checks.get("embedding_compatible", True) is not False
            checks["database_ok"] = database_ok
            checks["index_state_ok"] = state_ok
            return {
                "ok": bool(
                    database_ok and checks["fingerprint_matches"] and state_ok and embedding_ok
                ),
                "checks": checks,
                "warnings": self._configuration_warnings(),
            }

    def close(self) -> None:
        with self._lock:
            if self._store is not None:
                self._store.close()
            self._store = None
            self._retriever = None
            self._vector_index = None
            self._vector_version = None
