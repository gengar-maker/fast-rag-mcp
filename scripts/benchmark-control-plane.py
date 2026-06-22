#!/usr/bin/env python3
"""Benchmark scan/parse/SQLite/lifecycle overhead without downloading a neural model."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from blazing_rag_mcp.application import Application
from blazing_rag_mcp.config import Settings
from blazing_rag_mcp.embeddings import EmbeddingInfo


class FakeEmbeddings:
    def __init__(self, dim: int = 32):
        self.info = EmbeddingInfo(
            model="benchmark-fake",
            device="cpu",
            dim=dim,
            normalized=True,
            provider="fake",
            max_seq_length=384,
            precision="float32",
            batch_size=64,
        )
        self.index_key = f"benchmark-fake|{dim}"

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.blake2b(text.encode(), digest_size=self.info.dim).digest()
            row = np.frombuffer(digest, dtype=np.uint8).astype("float32")
            row /= max(float(np.linalg.norm(row)), 1e-12)
            rows.append(row)
        return np.stack(rows) if rows else np.zeros((0, self.info.dim), dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text], is_query=True)[0]

    def warmup(self) -> None:
        self.embed_texts(["warmup"])


def elapsed_ms(fn):
    started = time.perf_counter()
    value = fn()
    return round((time.perf_counter() - started) * 1000, 3), value


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="brag-control-plane-") as tmp:
        base = Path(tmp)
        repo = base / "repo"
        db = base / "db"
        repo.mkdir()
        for index in range(1000):
            (repo / f"module_{index}.py").write_text(
                f"def function_{index}(value):\n    return value + {index}\n",
                encoding="utf-8",
            )

        settings = Settings(
            roots=[repo.as_posix()],
            db_dir=db,
            scan_backend="walk",
            min_chunk_chars=10,
            embedding_flush_chunks=2048,
            vector_backend="numpy",
            vector_storage_dtype="float16",
            read_only=False,
            allow_reindex_tool=True,
        )
        app = Application(settings)
        app._embeddings = FakeEmbeddings()
        try:
            initial_ms, initial = elapsed_ms(lambda: app.reindex(force=True))
            noop_ms, noop = elapsed_ms(app.reindex)
            changed = repo / "module_777.py"
            changed.write_text(
                "def function_777(value):\n    return value + 777000\n",
                encoding="utf-8",
            )
            targeted_ms, targeted = elapsed_ms(
                lambda: app.reindex(paths=[changed.as_posix()])
            )
        finally:
            app.close()

        print(
            json.dumps(
                {
                    "files": 1000,
                    "initial_ms": initial_ms,
                    "full_noop_ms": noop_ms,
                    "targeted_one_file_ms": targeted_ms,
                    "targeted_timings_ms": targeted["timings_ms"],
                    "targeted_docs_changed": targeted["docs_changed"],
                    "targeted_chunks_embedded": targeted["chunks_embedded"],
                    "noop_docs_changed": noop["docs_changed"],
                    "initial_chunks": initial["chunks_written"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
