from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blazing_rag_mcp.chunking import chunk_document
from blazing_rag_mcp.config import Settings
from blazing_rag_mcp.indexer import Indexer
from blazing_rag_mcp.io import ScanStats, iter_candidate_files, read_document
from blazing_rag_mcp.retrieval import Retriever
from blazing_rag_mcp.store import Store
from blazing_rag_mcp.vector import VectorIndex

pymupdf = pytest.importorskip("pymupdf")


class FakeEmbeddings:
    def __init__(self):
        from blazing_rag_mcp.embeddings import EmbeddingInfo

        self.info = EmbeddingInfo(
            model="fake",
            device="cpu",
            dim=16,
            normalized=True,
            provider="fake",
            max_seq_length=512,
            precision="float32",
        )
        self.index_key = "fake|16"

    def embed_texts(self, texts, *, is_query=False, query_prefix=None):
        rows = []
        for text in texts:
            row = np.zeros(16, dtype="float32")
            for token in text.lower().split():
                row[hash(token) % 16] += 1.0
            norm = np.linalg.norm(row)
            rows.append(row / norm if norm else row)
        return np.stack(rows)

    def embed_query(self, query, *, query_prefix=None):
        return self.embed_texts([query], is_query=True, query_prefix=query_prefix)[0]


def _make_pdf(path: Path) -> None:
    pdf = pymupdf.open()
    for page_no, (heading, body) in enumerate(
        [
            ("Installation", "Install the package with uv sync. Configure the local MCP server."),
            ("Authentication", "Access tokens are refreshed by the session manager before expiry."),
        ],
        start=1,
    ):
        page = pdf.new_page()
        page.insert_text((72, 35), "Example Product Documentation", fontsize=9)
        page.insert_text((72, 90), heading, fontsize=20)
        page.insert_textbox(pymupdf.Rect(72, 120, 520, 300), body, fontsize=12)
        page.insert_text((280, 800), str(page_no), fontsize=9)
    pdf.set_metadata({"title": "Example Product Manual"})
    pdf.save(path)
    pdf.close()


def test_pdf_read_chunk_and_citations(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    pdf_path = repo / "manual.pdf"
    _make_pdf(pdf_path)
    settings = Settings(
        roots=[repo.as_posix()],
        db_dir=tmp_path / "db",
        min_chunk_chars=10,
        pdf_chunk_tokens=80,
        pdf_chunk_overlap=0,
    )
    document = read_document(repo, pdf_path, settings)
    assert document is not None
    assert document.metadata["document_type"] == "pdf"
    assert document.metadata["page_count"] == 2
    chunks = chunk_document(document, settings)
    assert chunks
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2}
    assert all(chunk.metadata["document_type"] == "pdf" for chunk in chunks)
    # Repeated running header is removed from page content.
    assert all("Example Product Documentation" not in chunk.text for chunk in chunks)


def test_pdf_index_search_outline_and_fetch(tmp_path):
    repo = tmp_path / "repo"
    db = tmp_path / "db"
    repo.mkdir()
    pdf_path = repo / "manual.pdf"
    _make_pdf(pdf_path)
    settings = Settings(
        roots=[repo.as_posix()],
        db_dir=db,
        min_chunk_chars=10,
        pdf_chunk_tokens=80,
        vector_backend="numpy",
        vector_storage_dtype="float32",
    )
    stats = ScanStats()
    assert pdf_path in list(iter_candidate_files(settings, stats))

    store = Store(db, vector_storage_dtype="float32")
    embeddings = FakeEmbeddings()
    result = Indexer(settings, store, embeddings).index_all(force=True)
    assert result["docs_changed"] == 1
    assert result["chunks_written"] >= 2

    vector = VectorIndex(settings)
    ids, matrix = store.all_vectors()
    vector.build(ids, matrix)
    retriever = Retriever(settings, store, embeddings, vector)
    search = retriever.document_search("session manager access tokens", mode="keyword", top_k=3)
    assert search["results"]
    hit = search["results"][0]
    assert hit["document_type"] == "pdf"
    assert hit["page"] == 2
    assert hit["citation"].endswith("#page=2")

    outline = retriever.document_outline("manual.pdf")
    assert outline["page_count"] == 2
    fetched = retriever.fetch(resource_uri=hit["resource_uri"], context_chunks=1)
    assert fetched["metadata"]["page"] == 2
    assert fetched["citation"].endswith("#page=2")
    assert fetched["context"]
