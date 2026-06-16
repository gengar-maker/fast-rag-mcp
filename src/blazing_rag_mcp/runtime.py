from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .embeddings import EmbeddingModel
from .indexer import Indexer
from .retrieval import Retriever
from .store import Store, replace_index_database
from .vector import VectorIndex

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.store = Store(self.settings.db_dir, vector_storage_dtype=self.settings.vector_storage_dtype)
        self.embeddings = EmbeddingModel(self.settings)
        self.vector_index = VectorIndex(self.settings)
        self.indexer = Indexer(self.settings, self.store, self.embeddings)
        self.reload_vector_index()
        self.retriever = Retriever(self.settings, self.store, self.embeddings, self.vector_index)

    def reload_vector_index(self) -> dict:
        ids, vectors = self.store.all_vectors()
        self.vector_index.build(ids, vectors)
        return asdict(self.vector_index.info())

    def reindex(self, *, force: bool = False) -> dict:
        if not force:
            result = self.indexer.index_all(force=False)
            result["full_rebuild"] = False
            result["vector_reload"] = self.reload_vector_index()
            return result

        target_dir = self.settings.resolved_db_dir()
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{target_dir.name}.rebuild-",
                dir=target_dir.parent,
            )
        )
        staging_settings = self.settings.model_copy(update={"db_dir": staging_dir})
        staging_store = Store(staging_dir, vector_storage_dtype=self.settings.vector_storage_dtype, bulk_build=True)
        try:
            staging_indexer = Indexer(staging_settings, staging_store, self.embeddings)
            result = staging_indexer.index_all(force=False)
            staging_store.optimize()
            staging_store.checkpoint()
            staging_store.close()

            # Drop old in-memory vectors before replacing the backing database. This is
            # particularly important on MPS because GPU allocations use unified memory.
            self.vector_index = VectorIndex(self.settings)
            self.store.close()
            replace_index_database(staging_dir, target_dir)

            self.store = Store(target_dir, vector_storage_dtype=self.settings.vector_storage_dtype)
            self.indexer = Indexer(self.settings, self.store, self.embeddings)
            self.vector_index = VectorIndex(self.settings)
            vector_reload = self.reload_vector_index()
            self.retriever = Retriever(
                self.settings,
                self.store,
                self.embeddings,
                self.vector_index,
            )
            result["full_rebuild"] = True
            result["db_swap"] = "completed"
            result["vector_reload"] = vector_reload
            return result
        except Exception:
            # Reopen the previous target if it was closed before a failed swap.
            try:
                self.store.stats()
            except Exception:
                self.store = Store(target_dir, vector_storage_dtype=self.settings.vector_storage_dtype)
                self.indexer = Indexer(self.settings, self.store, self.embeddings)
                self.vector_index = VectorIndex(self.settings)
                self.reload_vector_index()
                self.retriever = Retriever(
                    self.settings,
                    self.store,
                    self.embeddings,
                    self.vector_index,
                )
            raise
        finally:
            try:
                staging_store.close()
            except Exception:
                pass
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def status(self) -> dict:
        return {
            "settings": {
                "db_dir": self.settings.db_dir.as_posix(),
                "roots": [p.as_posix() for p in self.settings.resolved_roots()],
                "embedding_model": self.settings.embedding_model,
                "device": self.settings.device,
                "vector_backend": self.settings.vector_backend,
                "faiss_gpu": self.settings.faiss_gpu,
                "read_only": self.settings.read_only,
                "allow_reindex_tool": self.settings.allow_reindex_tool,
                "chunk_tokens": self.settings.chunk_tokens,
                "add_file_summary_chunks": self.settings.add_file_summary_chunks,
                "code_symbol_weight": self.settings.code_symbol_weight,
                "code_path_weight": self.settings.code_path_weight,
            },
            "store": self.store.stats(),
            "embedding": asdict(self.embeddings.info),
            "index": asdict(self.vector_index.info()),
        }
