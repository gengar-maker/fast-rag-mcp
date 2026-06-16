from pathlib import Path

from blazing_rag_mcp.chunking import chunk_document
from blazing_rag_mcp.code_index import extract_references, extract_symbols
from blazing_rag_mcp.config import Settings
from blazing_rag_mcp.embeddings import EmbeddingModel
from blazing_rag_mcp.indexer import Indexer
from blazing_rag_mcp.retrieval import Retriever
from blazing_rag_mcp.store import Store
from blazing_rag_mcp.types import Document
from blazing_rag_mcp.vector import VectorIndex


def test_chunk_document_smoke():
    settings = Settings(roots=["."], chunk_tokens=64, min_chunk_chars=10)
    doc = Document(
        doc_id="doc",
        root=Path("."),
        path=Path("README.md"),
        rel_path="README.md",
        title="README.md",
        text="# Title\n\nhello world\n\n## Part\n\n" + "some text " * 100,
        content_hash="abc",
        mtime_ns=1,
        size_bytes=100,
    )
    chunks = chunk_document(doc, settings)
    assert chunks
    assert chunks[0].path == "README.md"


def test_code_symbol_extraction_regex_fallback():
    doc = Document(
        doc_id="doc",
        root=Path("."),
        path=Path("src/auth.py"),
        rel_path="src/auth.py",
        title="auth.py",
        text="""
class SessionManager:
    def refresh(self, token: str) -> str:
        return validate_token(token)

def validate_token(token: str) -> str:
    return token
""".strip(),
        content_hash="abc",
        mtime_ns=1,
        size_bytes=100,
    )
    symbols = extract_symbols(doc)
    names = {s.name for s in symbols}
    assert "SessionManager" in names
    assert "refresh" in names
    assert "validate_token" in names
    chunks = chunk_document(doc, Settings(roots=["."], min_chunk_chars=10), symbols=symbols)
    assert any(c.metadata.get("chunk_type") == "symbol" for c in chunks)
    refs = extract_references(doc, symbols)
    assert any(r.target_name == "validate_token" and r.ref_kind == "call" for r in refs)


def test_index_and_code_search(tmp_path):
    repo = tmp_path / "repo"
    db = tmp_path / "db"
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "auth.py").write_text(
        """
class SessionManager:
    def refresh(self, token: str) -> str:
        return validate_token(token)

def validate_token(token: str) -> str:
    return token
""".strip(),
        encoding="utf-8",
    )
    settings = Settings(roots=[repo.as_posix()], db_dir=db, min_chunk_chars=10, vector_backend="numpy")
    store = Store(db)
    embeddings = EmbeddingModel(settings)
    indexer = Indexer(settings, store, embeddings)
    result = indexer.index_all(force=True)
    assert result["symbols_written"] >= 2
    vector = VectorIndex(settings)
    ids, matrix = store.all_vectors()
    vector.build(ids, matrix)
    retriever = Retriever(settings, store, embeddings, vector)
    found = retriever.find_symbol("SessionManager")
    assert found["results"][0]["name"] == "SessionManager"
    search = retriever.code_search("validate_token", top_k=3)
    assert any(r.get("symbol_name") == "validate_token" for r in search["results"])


def test_cross_file_embedding_batching(tmp_path):
    import numpy as np

    class FakeEmbeddings:
        def __init__(self):
            from blazing_rag_mcp.embeddings import EmbeddingInfo

            self.calls = 0
            self.info = EmbeddingInfo(
                model="fake",
                device="cpu",
                dim=8,
                normalized=True,
                provider="fake",
                max_seq_length=64,
                precision="float32",
            )

        def embed_texts(self, texts, *, is_query=False):
            self.calls += 1
            arr = np.ones((len(texts), 8), dtype="float32")
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
            return arr

    repo = tmp_path / "repo"
    db = tmp_path / "db"
    repo.mkdir()
    for i in range(12):
        (repo / f"module_{i}.py").write_text(
            f"def function_{i}(value):\n    return value + {i}\n",
            encoding="utf-8",
        )

    settings = Settings(
        roots=[repo.as_posix()],
        db_dir=db,
        min_chunk_chars=10,
        embedding_flush_chunks=128,
        vector_backend="numpy",
    )
    store = Store(db)
    embeddings = FakeEmbeddings()
    result = Indexer(settings, store, embeddings).index_all(force=True)
    assert result["docs_changed"] == 12
    assert result["throughput"]["embedding_calls"] == 1
    assert embeddings.calls == 1


def test_incremental_reuses_unchanged_chunk_embeddings(tmp_path):
    import hashlib
    import os
    import numpy as np

    class FakeEmbeddings:
        def __init__(self):
            from blazing_rag_mcp.embeddings import EmbeddingInfo
            self.calls = 0
            self.texts = 0
            self.info = EmbeddingInfo(
                model="fake", device="cpu", dim=8, normalized=True,
                provider="fake", max_seq_length=64, precision="float32", batch_size=32,
            )

        def embedding_hash(self, text):
            return hashlib.blake2b(text.encode(), digest_size=20).hexdigest()

        def embed_texts(self, texts, *, is_query=False):
            self.calls += 1
            self.texts += len(texts)
            rows = []
            for text in texts:
                digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
                row = np.frombuffer(digest, dtype=np.uint8).astype("float32")
                row /= max(np.linalg.norm(row), 1e-12)
                rows.append(row)
            return np.stack(rows)

    repo = tmp_path / "repo"
    db = tmp_path / "db"
    repo.mkdir()
    source = repo / "module.py"
    source.write_text(
        "def unchanged(value):\n    return value + 1\n\n"
        "def changed(value):\n    return value + 2\n",
        encoding="utf-8",
    )
    settings = Settings(
        roots=[repo.as_posix()], db_dir=db, min_chunk_chars=10,
        embedding_flush_chunks=128, vector_backend="numpy",
        vector_storage_dtype="float16", scan_backend="walk",
    )
    store = Store(db, vector_storage_dtype="float16")
    embeddings = FakeEmbeddings()
    first = Indexer(settings, store, embeddings).index_all()
    assert first["chunks_embedded"] > 0

    second = Indexer(settings, store, embeddings).index_all()
    assert second["docs_changed"] == 0
    assert second["chunks_embedded"] == 0

    old_texts = embeddings.texts
    source.write_text(
        "def unchanged(value):\n    return value + 1\n\n"
        "def changed(value):\n    return value + 200\n",
        encoding="utf-8",
    )
    os.utime(source, None)
    third = Indexer(settings, store, embeddings).index_all()
    assert third["docs_changed"] == 1
    assert third["chunks_reused"] >= 1
    assert third["chunks_embedded"] < third["chunks_written"]
    assert embeddings.texts > old_texts

    ids, matrix = store.all_vectors()
    assert ids
    assert matrix.dtype == np.float16
