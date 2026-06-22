from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from blazing_rag_mcp.application import Application
from blazing_rag_mcp.config import Settings
from blazing_rag_mcp.embeddings import EmbeddingInfo
from blazing_rag_mcp.indexer import Indexer
from blazing_rag_mcp.store import Store


class FakeEmbeddings:
    def __init__(self, dim: int = 16):
        self.calls = 0
        self.texts = 0
        self.info = EmbeddingInfo(
            model="fake-code-model",
            device="cpu",
            dim=dim,
            normalized=True,
            provider="fake",
            max_seq_length=384,
            precision="float32",
            batch_size=32,
        )
        self.index_key = f"fake-code-model|fake|{dim}|True|384|float32"

    def embed_texts(self, texts, *, is_query=False):
        self.calls += 1
        self.texts += len(texts)
        rows = []
        for text in texts:
            digest = hashlib.blake2b(text.encode(), digest_size=self.info.dim).digest()
            row = np.frombuffer(digest, dtype=np.uint8).astype("float32")
            row /= max(float(np.linalg.norm(row)), 1e-12)
            rows.append(row)
        return np.stack(rows) if rows else np.zeros((0, self.info.dim), dtype="float32")

    def embed_query(self, text: str):
        return self.embed_texts([text], is_query=True)[0]

    def warmup(self):
        self.embed_texts(["warmup"])


def _settings(repo: Path, db: Path) -> Settings:
    return Settings(
        roots=[repo.as_posix()],
        db_dir=db,
        scan_backend="walk",
        min_chunk_chars=10,
        vector_backend="numpy",
        vector_storage_dtype="float16",
        embedding_allow_hash_fallback=True,
        read_only=False,
        allow_reindex_tool=True,
    )


def _write_repo(repo: Path, count: int = 40) -> None:
    repo.mkdir(parents=True)
    for index in range(count):
        (repo / f"module_{index}.py").write_text(
            f"def function_{index}(value):\n    return value + {index}\n",
            encoding="utf-8",
        )


def test_noop_targeted_reindex_does_not_load_model(tmp_path):
    repo, db = tmp_path / "repo", tmp_path / "db"
    _write_repo(repo, 4)
    settings = _settings(repo, db)

    store = Store(db, vector_storage_dtype="float16")
    initial_embeddings = FakeEmbeddings()
    Indexer(settings, store, initial_embeddings).index_all(force=True)
    store.close()

    app = Application(settings)
    result = app.reindex(paths=["module_1.py"])
    assert result["files_seen"] == 1
    assert result["docs_changed"] == 0
    assert result["chunks_embedded"] == 0
    assert result["embedding"]["loaded"] is False
    assert app._embeddings is None
    app.close()


def test_one_file_update_is_targeted_and_vector_reload_is_deferred(tmp_path):
    repo, db = tmp_path / "repo", tmp_path / "db"
    _write_repo(repo, 80)
    settings = _settings(repo, db)
    app = Application(settings)
    fake = FakeEmbeddings()
    app._embeddings = fake

    initial = app.reindex(force=True)
    assert initial["docs_changed"] == 80
    vector = app._get_vector_index()
    old_version = app._vector_version
    assert vector.info().vectors > 0

    changed = repo / "module_37.py"
    changed.write_text(
        "def function_37(value):\n    return value + 3700\n",
        encoding="utf-8",
    )
    os.utime(changed, None)

    result = app.reindex(paths=[changed.as_posix()])
    assert result["targeted"] is True
    assert result["files_seen"] == 1
    assert result["docs_changed"] == 1
    assert result["vector_refresh"]["deferred"] is True
    assert app._vector_index is None
    assert app._vector_version is None
    assert old_version != result["corpus_version"]
    app.close()


def test_exact_symbol_lookup_does_not_load_dense_resources(tmp_path):
    repo, db = tmp_path / "repo", tmp_path / "db"
    _write_repo(repo, 2)
    settings = _settings(repo, db)
    store = Store(db, vector_storage_dtype="float16")
    Indexer(settings, store, FakeEmbeddings()).index_all(force=True)
    store.close()

    app = Application(settings)
    result = app.find_symbol(name="function_1")
    assert result["results"][0]["name"] == "function_1"
    assert app._embeddings is None
    assert app._vector_index is None
    app.close()


def test_noop_reindex_keeps_loaded_vector_valid(tmp_path):
    repo, db = tmp_path / "repo", tmp_path / "db"
    _write_repo(repo, 8)
    settings = _settings(repo, db)
    app = Application(settings)
    app._embeddings = FakeEmbeddings()
    app.reindex(force=True)
    vector = app._get_vector_index()
    version = app._vector_version

    result = app.reindex(paths=["module_2.py"])
    assert result["docs_changed"] == 0
    assert app._vector_index is vector
    assert app._vector_version == version
    assert "remains valid" in result["vector_refresh"]["reason"]
    app.close()


def test_symbol_and_chunk_ids_survive_unrelated_prefix_edit():
    from blazing_rag_mcp.chunking import chunk_document
    from blazing_rag_mcp.code_index import extract_symbols
    from blazing_rag_mcp.types import Document

    settings = Settings(roots=["."], min_chunk_chars=10, add_file_summary_chunks=False)

    def build(text: str, content_hash: str):
        doc = Document(
            doc_id="stable-doc",
            root=Path("."),
            path=Path("src/module.py"),
            rel_path="src/module.py",
            title="module.py",
            text=text,
            content_hash=content_hash,
            mtime_ns=1,
            size_bytes=len(text),
        )
        symbols = extract_symbols(doc)
        chunks = chunk_document(doc, settings, symbols=symbols)
        return symbols, chunks

    before = (
        "def stable(value):\n    return value + 1\n\n"
        "def second(value):\n    return value * 2\n"
    )
    after = "# unrelated header\n\n" + before
    before_symbols, before_chunks = build(before, "before")
    after_symbols, after_chunks = build(after, "after")

    before_symbol = next(symbol for symbol in before_symbols if symbol.name == "stable")
    after_symbol = next(symbol for symbol in after_symbols if symbol.name == "stable")
    assert before_symbol.id == after_symbol.id

    before_chunk = next(chunk for chunk in before_chunks if chunk.symbol_name == "stable")
    after_chunk = next(chunk for chunk in after_chunks if chunk.symbol_name == "stable")
    assert before_chunk.id == after_chunk.id


def test_scanner_rejects_secret_files_and_invalid_roots(tmp_path):
    import pytest

    from blazing_rag_mcp.io import ScanStats, iter_candidate_files

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("value = 1\n", encoding="utf-8")
    (repo / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (repo / "private.pem").write_text("secret\n", encoding="utf-8")

    settings = Settings(roots=[repo.as_posix()], scan_backend="walk")
    stats = ScanStats()
    names = {path.name for path in iter_candidate_files(settings, stats)}
    assert "module.py" in names
    assert ".env" not in names
    assert "private.pem" not in names

    invalid = Settings(roots=[(tmp_path / "missing").as_posix()])
    with pytest.raises(ValueError, match="no configured BRAG_ROOTS"):
        invalid.resolved_roots()


def test_settings_reject_unsafe_or_inconsistent_limits(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        Settings(roots=[tmp_path.as_posix()], embedding_batch_size=32, embedding_flush_chunks=16)
    with pytest.raises(ValueError):
        Settings(roots=[tmp_path.as_posix()], default_top_k=16, max_top_k=8)
    settings = Settings(roots=[tmp_path.as_posix()], db_dir=tmp_path)
    with pytest.raises(ValueError, match="must not be identical"):
        settings.resolved_roots()


def test_mcp_write_policy_is_fail_closed(tmp_path):
    repo, db = tmp_path / "repo", tmp_path / "db"
    _write_repo(repo, 1)
    app = Application(_settings(repo, db).model_copy(update={"read_only": True}))
    try:
        import pytest

        with pytest.raises(PermissionError, match="read-only"):
            app.reindex(paths=["module_0.py"], enforce_mcp_policy=True)
    finally:
        app.close()


def test_failed_force_rebuild_preserves_active_database(tmp_path):
    import pytest

    repo, db = tmp_path / "repo", tmp_path / "db"
    _write_repo(repo, 3)
    settings = _settings(repo, db)

    original = Application(settings)
    original._embeddings = FakeEmbeddings()
    original.reindex(force=True)
    original.close()

    class FailingEmbeddings(FakeEmbeddings):
        def embed_texts(self, texts, *, is_query=False):
            raise RuntimeError("synthetic embedding failure")

    failing = Application(settings)
    failing._embeddings = FailingEmbeddings()
    with pytest.raises(RuntimeError, match="synthetic embedding failure"):
        failing.reindex(force=True)
    failing.close()

    store = Store(db, vector_storage_dtype="float16")
    try:
        assert store.stats()["docs"] == 3
        assert store.integrity_check()["quick_check"] == "ok"
    finally:
        store.close()


def test_targeted_deleted_file_removes_only_that_document(tmp_path):
    repo, db = tmp_path / "repo", tmp_path / "db"
    _write_repo(repo, 3)
    settings = _settings(repo, db)
    app = Application(settings)
    app._embeddings = FakeEmbeddings()
    try:
        app.reindex(force=True)
        deleted = repo / "module_1.py"
        deleted.unlink()
        result = app.reindex(paths=[deleted.as_posix()])
        assert result["stale_docs_deleted"] == 1
        assert result["docs_changed"] == 0
        assert app.status()["store"]["docs"] == 2
    finally:
        app.close()


def test_interprocess_index_lock_rejects_second_writer(tmp_path):
    from blazing_rag_mcp.locking import IndexMutationLock

    first = IndexMutationLock(tmp_path)
    second = IndexMutationLock(tmp_path)
    assert first.acquire() is True
    try:
        assert second.acquire() is False
    finally:
        first.release()
    assert second.acquire() is True
    second.release()


def test_tree_sitter_uses_pinned_bundled_parser_release():
    from importlib.metadata import version

    from blazing_rag_mcp.code_index import _get_parser

    assert version("tree-sitter-language-pack") == "0.7.2"
    assert _get_parser("python") is not None


def test_application_query_store_is_sqlite_read_only(tmp_path):
    import sqlite3

    repo, db = tmp_path / "repo", tmp_path / "db"
    _write_repo(repo, 2)
    settings = _settings(repo, db)
    app = Application(settings)
    app._embeddings = FakeEmbeddings()
    app.reindex(force=True)

    store = app._get_store()
    assert store.read_only is True
    assert int(store.conn.execute("PRAGMA query_only").fetchone()[0]) == 1
    with pytest.raises(sqlite3.OperationalError):
        store.conn.execute("INSERT INTO meta(key, value) VALUES ('forbidden', '1')")
    app.close()
