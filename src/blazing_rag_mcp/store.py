from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .types import Chunk, CodeReference, CodeSymbol, Document

SCHEMA_VERSION = 3


def adapt_array(arr: np.ndarray, dtype: str = "float32") -> bytes:
    np_dtype = np.float16 if dtype == "float16" else np.float32
    return np.asarray(arr, dtype=np_dtype).tobytes(order="C")


def decode_array(blob: bytes, dim: int, dtype: str | None = None) -> np.ndarray:
    if dtype not in {"float16", "float32"}:
        dtype = "float16" if len(blob) == dim * 2 else "float32"
    np_dtype = np.float16 if dtype == "float16" else np.float32
    return np.frombuffer(blob, dtype=np_dtype, count=dim).reshape(dim)


class Store:
    def __init__(self, db_dir: Path, *, vector_storage_dtype: str = "float32", bulk_build: bool = False):
        self.db_dir = db_dir.expanduser().resolve()
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.db_dir / "rag.sqlite3"
        self.vector_storage_dtype = vector_storage_dtype if vector_storage_dtype in {"float16", "float32"} else "float32"
        self.bulk_build = bulk_build
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if bulk_build:
            # A forced rebuild writes to a disposable staging DB. Durability comes from the final
            # atomic swap, so disabling WAL/fsync here substantially accelerates FTS/vector writes.
            self.conn.execute("PRAGMA journal_mode=OFF")
            self.conn.execute("PRAGMA synchronous=OFF")
            self.conn.execute("PRAGMA locking_mode=EXCLUSIVE")
            self.conn.execute("PRAGMA temp_store=MEMORY")
            self.conn.execute("PRAGMA cache_size=-262144")
            self.conn.execute("PRAGMA mmap_size=268435456")
        else:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA temp_store=DEFAULT")
            self.conn.execute("PRAGMA mmap_size=134217728")
        self._init_schema()

    def checkpoint(self) -> None:
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.conn.execute("BEGIN")
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS docs (
              doc_id TEXT PRIMARY KEY,
              root TEXT NOT NULL,
              path TEXT NOT NULL,
              title TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              mtime_ns INTEGER NOT NULL,
              size_bytes INTEGER NOT NULL,
              indexed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL,
              root TEXT NOT NULL,
              path TEXT NOT NULL,
              title TEXT NOT NULL,
              section TEXT NOT NULL,
              text TEXT NOT NULL,
              line_start INTEGER NOT NULL,
              line_end INTEGER NOT NULL,
              byte_start INTEGER NOT NULL,
              byte_end INTEGER NOT NULL,
              content_hash TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              ord INTEGER NOT NULL,
              language TEXT NOT NULL DEFAULT '',
              chunk_type TEXT NOT NULL DEFAULT 'text',
              symbol_id TEXT NOT NULL DEFAULT '',
              symbol_name TEXT NOT NULL DEFAULT '',
              qualified_name TEXT NOT NULL DEFAULT '',
              embedding_hash TEXT NOT NULL DEFAULT '',
              FOREIGN KEY(doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS symbols (
              id TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL,
              root TEXT NOT NULL,
              path TEXT NOT NULL,
              language TEXT NOT NULL,
              kind TEXT NOT NULL,
              name TEXT NOT NULL,
              qualified_name TEXT NOT NULL,
              parent TEXT NOT NULL,
              signature TEXT NOT NULL,
              docstring TEXT NOT NULL,
              text TEXT NOT NULL,
              line_start INTEGER NOT NULL,
              line_end INTEGER NOT NULL,
              byte_start INTEGER NOT NULL,
              byte_end INTEGER NOT NULL,
              metadata_json TEXT NOT NULL,
              FOREIGN KEY(doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS refs (
              id TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL,
              root TEXT NOT NULL,
              path TEXT NOT NULL,
              language TEXT NOT NULL,
              source_symbol_id TEXT NOT NULL,
              target_name TEXT NOT NULL,
              target_name_lc TEXT NOT NULL,
              ref_kind TEXT NOT NULL,
              line INTEGER NOT NULL,
              snippet TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              FOREIGN KEY(doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS vectors (
              chunk_id TEXT PRIMARY KEY,
              dim INTEGER NOT NULL,
              dtype TEXT NOT NULL DEFAULT 'float32',
              vector BLOB NOT NULL,
              FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
              chunk_id UNINDEXED,
              title,
              section,
              path,
              text,
              tokenize='unicode61 remove_diacritics 2'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
              symbol_id UNINDEXED,
              name,
              qualified_name,
              kind,
              path,
              signature,
              docstring,
              text,
              tokenize='unicode61 remove_diacritics 2'
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_qualified ON chunks(qualified_name);
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hash ON chunks(embedding_hash);
            CREATE INDEX IF NOT EXISTS idx_docs_path ON docs(path);
            CREATE INDEX IF NOT EXISTS idx_symbols_doc ON symbols(doc_id);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_qname ON symbols(qualified_name);
            CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);
            CREATE INDEX IF NOT EXISTS idx_refs_target ON refs(target_name_lc);
            CREATE INDEX IF NOT EXISTS idx_refs_source ON refs(source_symbol_id);
            CREATE INDEX IF NOT EXISTS idx_refs_path ON refs(path);
            """
        )
        # Best-effort migration for databases created by earlier scaffold versions.
        for name, ddl in {
            "language": "TEXT NOT NULL DEFAULT ''",
            "chunk_type": "TEXT NOT NULL DEFAULT 'text'",
            "symbol_id": "TEXT NOT NULL DEFAULT ''",
            "symbol_name": "TEXT NOT NULL DEFAULT ''",
            "qualified_name": "TEXT NOT NULL DEFAULT ''",
            "embedding_hash": "TEXT NOT NULL DEFAULT ''",
        }.items():
            self._ensure_column("chunks", name, ddl)
        self._ensure_column("vectors", "dtype", "TEXT NOT NULL DEFAULT 'float32'")
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def corpus_version(self) -> str:
        return self.get_meta("corpus_version", "empty")

    def mark_corpus_changed(self) -> str:
        version = f"{int(time.time() * 1000)}"
        self.set_meta("corpus_version", version)
        return version

    def document_manifest(self) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT doc_id, root, path, content_hash, mtime_ns, size_bytes FROM docs"
        ).fetchall()
        return {str(r["doc_id"]): dict(r) for r in rows}

    def touch_document(self, doc: Document) -> None:
        self.touch_documents([doc])

    def touch_documents(self, docs: Sequence[Document]) -> None:
        if not docs:
            return
        now = time.time()
        with self.transaction() as conn:
            conn.executemany(
                "UPDATE docs SET content_hash=?, mtime_ns=?, size_bytes=?, indexed_at=? WHERE doc_id=?",
                [(d.content_hash, d.mtime_ns, d.size_bytes, now, d.doc_id) for d in docs],
            )

    def doc_is_current(self, doc: Document) -> bool:
        row = self.conn.execute(
            "SELECT content_hash, mtime_ns, size_bytes FROM docs WHERE doc_id=?", (doc.doc_id,)
        ).fetchone()
        return bool(
            row
            and row["content_hash"] == doc.content_hash
            and int(row["mtime_ns"]) == int(doc.mtime_ns)
            and int(row["size_bytes"]) == int(doc.size_bytes)
        )

    def document_vector_cache(self, doc_id: str) -> dict[str, np.ndarray]:
        """Return old vectors keyed by semantic embedding hash for one document.

        This lets a one-line edit reuse embeddings for every unchanged function/chunk in the file.
        """
        rows = self.conn.execute(
            """
            SELECT c.embedding_hash, v.dim, v.dtype, v.vector
            FROM chunks c JOIN vectors v ON v.chunk_id=c.id
            WHERE c.doc_id=? AND c.embedding_hash<>''
            """,
            (doc_id,),
        ).fetchall()
        out: dict[str, np.ndarray] = {}
        for row in rows:
            key = str(row["embedding_hash"])
            if key and key not in out:
                out[key] = decode_array(row["vector"], int(row["dim"]), str(row["dtype"])).astype("float32", copy=True)
        return out

    def upsert_document(
        self,
        doc: Document,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        symbols: Sequence[CodeSymbol] | None = None,
        references: Sequence[CodeReference] | None = None,
    ) -> None:
        with self.transaction() as conn:
            self._upsert_document_on_conn(
                conn,
                doc,
                chunks,
                vectors,
                symbols or [],
                references or [],
            )

    def upsert_documents(
        self,
        payloads: Sequence[
            tuple[
                Document,
                Sequence[Chunk],
                np.ndarray,
                Sequence[CodeSymbol],
                Sequence[CodeReference],
            ]
        ],
    ) -> None:
        """Write a bounded indexing batch in one transaction.

        The indexer already caps batches by chunk count, so this reduces fsync/transaction
        overhead without holding the whole corpus in memory.
        """
        if not payloads:
            return
        with self.transaction() as conn:
            doc_ids = [doc.doc_id for doc, *_ in payloads]
            existing: list[str] = []
            if not self.bulk_build:
                # Stay below SQLite's host-parameter limit even for batches of tiny files.
                for start in range(0, len(doc_ids), 800):
                    part = doc_ids[start : start + 800]
                    placeholders = ",".join("?" for _ in part)
                    existing.extend(
                        str(row["doc_id"])
                        for row in conn.execute(
                            f"SELECT doc_id FROM docs WHERE doc_id IN ({placeholders})",
                            part,
                        ).fetchall()
                    )
            if existing:
                for start in range(0, len(existing), 800):
                    self._delete_docs_payload(conn, existing[start : start + 800])
            for doc, chunks, vectors, symbols, references in payloads:
                self._upsert_document_on_conn(
                    conn,
                    doc,
                    chunks,
                    vectors,
                    symbols,
                    references,
                    delete_existing=False,
                )

    def _upsert_document_on_conn(
        self,
        conn: sqlite3.Connection,
        doc: Document,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        symbols: Sequence[CodeSymbol],
        references: Sequence[CodeReference],
        *,
        delete_existing: bool = True,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}")

        now = time.time()
        dim = int(vectors.shape[1]) if vectors.size else 0
        if delete_existing:
            # Avoid expensive FTS delete scans for documents that are new to the index.
            exists = conn.execute("SELECT 1 FROM docs WHERE doc_id=?", (doc.doc_id,)).fetchone()
            if exists:
                self._delete_doc_payload(conn, doc.doc_id)
        conn.execute(
            """
            INSERT OR REPLACE INTO docs(doc_id, root, path, title, content_hash, mtime_ns, size_bytes, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc.doc_id, doc.root.as_posix(), doc.rel_path, doc.title, doc.content_hash, doc.mtime_ns, doc.size_bytes, now),
        )

        chunk_rows = []
        vector_rows = []
        fts_rows = []
        for ord_, chunk in enumerate(chunks):
            meta = chunk.metadata or {}
            language = str(meta.get("language", ""))
            chunk_type = str(meta.get("chunk_type", "text"))
            symbol_id = str(meta.get("symbol_id", ""))
            symbol_name = str(meta.get("symbol_name", ""))
            qualified_name = str(meta.get("qualified_name", ""))
            embedding_hash = str(meta.get("embedding_hash", ""))
            chunk_rows.append(
                (
                    chunk.id,
                    chunk.doc_id,
                    chunk.root,
                    chunk.path,
                    chunk.title,
                    chunk.section,
                    chunk.text,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.byte_start,
                    chunk.byte_end,
                    chunk.content_hash,
                    json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
                    ord_,
                    language,
                    chunk_type,
                    symbol_id,
                    symbol_name,
                    qualified_name,
                    embedding_hash,
                )
            )
            vector_rows.append((chunk.id, dim, self.vector_storage_dtype, adapt_array(vectors[ord_], self.vector_storage_dtype)))
            fts_text = "\n".join(x for x in [qualified_name, symbol_name, chunk.text] if x)
            fts_rows.append((chunk.id, chunk.title, chunk.section, chunk.path, fts_text))

        conn.executemany(
            """
            INSERT INTO chunks(
              id, doc_id, root, path, title, section, text, line_start, line_end,
              byte_start, byte_end, content_hash, metadata_json, ord,
              language, chunk_type, symbol_id, symbol_name, qualified_name, embedding_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )
        conn.executemany(
            "INSERT INTO vectors(chunk_id, dim, dtype, vector) VALUES (?, ?, ?, ?)",
            vector_rows,
        )
        conn.executemany(
            "INSERT INTO chunks_fts(chunk_id, title, section, path, text) VALUES (?, ?, ?, ?, ?)",
            fts_rows,
        )

        symbol_rows = [
            (
                sym.id,
                sym.doc_id,
                sym.root,
                sym.path,
                sym.language,
                sym.kind,
                sym.name,
                sym.qualified_name,
                sym.parent,
                sym.signature,
                sym.docstring,
                sym.text,
                sym.line_start,
                sym.line_end,
                sym.byte_start,
                sym.byte_end,
                json.dumps(sym.metadata, ensure_ascii=False, separators=(",", ":")),
            )
            for sym in symbols
        ]
        symbol_fts_rows = [
            (
                sym.id,
                sym.name,
                sym.qualified_name,
                sym.kind,
                sym.path,
                sym.signature,
                sym.docstring,
                sym.text[:8000],
            )
            for sym in symbols
        ]
        conn.executemany(
            """
            INSERT INTO symbols(
              id, doc_id, root, path, language, kind, name, qualified_name, parent,
              signature, docstring, text, line_start, line_end, byte_start, byte_end, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            symbol_rows,
        )
        conn.executemany(
            """
            INSERT INTO symbols_fts(symbol_id, name, qualified_name, kind, path, signature, docstring, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            symbol_fts_rows,
        )

        ref_rows = [
            (
                ref.id,
                ref.doc_id,
                ref.root,
                ref.path,
                ref.language,
                ref.source_symbol_id,
                ref.target_name,
                ref.target_name.lower(),
                ref.ref_kind,
                ref.line,
                ref.snippet,
                json.dumps(ref.metadata, ensure_ascii=False, separators=(",", ":")),
            )
            for ref in references
        ]
        conn.executemany(
            """
            INSERT OR IGNORE INTO refs(
              id, doc_id, root, path, language, source_symbol_id, target_name,
              target_name_lc, ref_kind, line, snippet, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ref_rows,
        )

    def reset_index_payload(self) -> None:
        """Replace the SQLite index for a true full rebuild.

        FTS5 `DELETE FROM` can be surprisingly expensive on a populated index. Recreating the
        local index database is both faster and leaves less fragmentation. Call this only for an
        explicit full rebuild.
        """
        self.conn.close()
        for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if self.bulk_build:
            self.conn.execute("PRAGMA journal_mode=OFF")
            self.conn.execute("PRAGMA synchronous=OFF")
            self.conn.execute("PRAGMA locking_mode=EXCLUSIVE")
            self.conn.execute("PRAGMA temp_store=MEMORY")
            self.conn.execute("PRAGMA cache_size=-262144")
            self.conn.execute("PRAGMA mmap_size=268435456")
        else:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA temp_store=DEFAULT")
            self.conn.execute("PRAGMA mmap_size=134217728")
        self._init_schema()

    def _delete_docs_payload(self, conn: sqlite3.Connection, doc_ids: Sequence[str]) -> None:
        if not doc_ids:
            return
        placeholders = ",".join("?" for _ in doc_ids)
        params = tuple(doc_ids)
        conn.execute(
            f"DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id IN ({placeholders}))",
            params,
        )
        conn.execute(
            f"DELETE FROM symbols_fts WHERE symbol_id IN (SELECT id FROM symbols WHERE doc_id IN ({placeholders}))",
            params,
        )
        conn.execute(
            f"DELETE FROM vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id IN ({placeholders}))",
            params,
        )
        conn.execute(f"DELETE FROM refs WHERE doc_id IN ({placeholders})", params)
        conn.execute(f"DELETE FROM chunks WHERE doc_id IN ({placeholders})", params)
        conn.execute(f"DELETE FROM symbols WHERE doc_id IN ({placeholders})", params)

    def _delete_doc_payload(self, conn: sqlite3.Connection, doc_id: str) -> None:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,))
        conn.execute("DELETE FROM symbols_fts WHERE symbol_id IN (SELECT id FROM symbols WHERE doc_id=?)", (doc_id,))
        conn.execute("DELETE FROM vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,))
        conn.execute("DELETE FROM refs WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM symbols WHERE doc_id=?", (doc_id,))

    def delete_missing_docs(self, live_doc_ids: set[str]) -> int:
        rows = self.conn.execute("SELECT doc_id FROM docs").fetchall()
        stale = [r["doc_id"] for r in rows if r["doc_id"] not in live_doc_ids]
        if not stale:
            return 0
        with self.transaction() as conn:
            for doc_id in stale:
                self._delete_doc_payload(conn, doc_id)
                conn.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))
        return len(stale)

    def all_vectors(self) -> tuple[list[str], np.ndarray]:
        meta = self.conn.execute(
            "SELECT COUNT(*) AS n, MIN(dim) AS min_dim, MAX(dim) AS max_dim FROM vectors"
        ).fetchone()
        n = int(meta["n"] or 0) if meta else 0
        if n <= 0:
            return [], np.zeros((0, 0), dtype="float32")
        min_dim = int(meta["min_dim"] or 0)
        max_dim = int(meta["max_dim"] or 0)
        if min_dim <= 0 or min_dim != max_dim:
            raise ValueError(f"inconsistent vector dimensions: min={min_dim} max={max_dim}")

        dim = min_dim
        target_dtype = np.float16 if self.vector_storage_dtype == "float16" else np.float32
        ids: list[str] = []
        matrix = np.empty((n, dim), dtype=target_dtype)
        cur = self.conn.execute("SELECT chunk_id, dim, dtype, vector FROM vectors ORDER BY chunk_id")
        row_idx = 0
        while True:
            batch = cur.fetchmany(8192)
            if not batch:
                break
            for row in batch:
                ids.append(str(row["chunk_id"]))
                decoded = decode_array(row["vector"], dim, str(row["dtype"]))
                matrix[row_idx, :] = decoded.astype(target_dtype, copy=False)
                row_idx += 1
        if row_idx != n:
            ids = ids[:row_idx]
            matrix = matrix[:row_idx]
        return ids, matrix

    def get_chunk(self, chunk_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[dict]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", tuple(chunk_ids)).fetchall()
        by_id = {r["id"]: dict(r) for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def chunks_for_symbols(self, symbol_ids: Sequence[str], *, limit_per_symbol: int = 1) -> list[dict]:
        out: list[dict] = []
        for sid in symbol_ids:
            rows = self.conn.execute(
                """
                SELECT * FROM chunks
                WHERE symbol_id=?
                ORDER BY CASE chunk_type WHEN 'symbol' THEN 0 WHEN 'symbol_part' THEN 1 ELSE 2 END, ord
                LIMIT ?
                """,
                (sid, limit_per_symbol),
            ).fetchall()
            out.extend(dict(r) for r in rows)
        return out

    def chunks_for_docs(self, doc_ids: Sequence[str], *, limit_per_doc: int = 1) -> list[dict]:
        out: list[dict] = []
        for doc_id in doc_ids:
            rows = self.conn.execute(
                """
                SELECT * FROM chunks
                WHERE doc_id=?
                ORDER BY CASE chunk_type WHEN 'file_summary' THEN 0 WHEN 'imports' THEN 1 ELSE 2 END, ord
                LIMIT ?
                """,
                (doc_id, limit_per_doc),
            ).fetchall()
            out.extend(dict(r) for r in rows)
        return out

    def fts_search(self, query: str, limit: int, path_prefix: str | None = None) -> list[tuple[str, float]]:
        sql = """
          SELECT chunk_id, bm25(chunks_fts, 1.4, 1.2, 1.8, 1.0) AS rank
          FROM chunks_fts
          WHERE chunks_fts MATCH ?
        """
        params: list[object] = [self._fts_query(query)]
        if path_prefix:
            sql += " AND path LIKE ?"
            params.append(f"{path_prefix}%")
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(r["chunk_id"]), float(-r["rank"])) for r in rows]

    def symbol_search(self, query: str, limit: int, path_prefix: str | None = None, kind: str | None = None) -> list[dict]:
        q = query.strip()
        if not q:
            return []
        by_id: dict[str, dict] = {}

        exact_sql = "SELECT *, 100.0 AS score, 'exact' AS match_type FROM symbols WHERE (lower(name)=? OR lower(qualified_name)=?)"
        params: list[object] = [q.lower(), q.lower()]
        if path_prefix:
            exact_sql += " AND path LIKE ?"
            params.append(f"{path_prefix}%")
        if kind:
            exact_sql += " AND kind=?"
            params.append(kind)
        exact_sql += " LIMIT ?"
        params.append(limit)
        for row in self.conn.execute(exact_sql, params).fetchall():
            d = dict(row)
            by_id[d["id"]] = d

        suffix_sql = "SELECT *, 75.0 AS score, 'qualified_suffix' AS match_type FROM symbols WHERE lower(qualified_name) LIKE ?"
        params = [f"%.{q.lower()}"]
        if path_prefix:
            suffix_sql += " AND path LIKE ?"
            params.append(f"{path_prefix}%")
        if kind:
            suffix_sql += " AND kind=?"
            params.append(kind)
        suffix_sql += " LIMIT ?"
        params.append(limit)
        for row in self.conn.execute(suffix_sql, params).fetchall():
            d = dict(row)
            by_id.setdefault(d["id"], d)

        like = "%" + "%".join(_split_query_terms(q)[:4]) + "%"
        like_sql = "SELECT *, 40.0 AS score, 'like' AS match_type FROM symbols WHERE (name LIKE ? OR qualified_name LIKE ? OR path LIKE ?)"
        params = [like, like, like]
        if path_prefix:
            like_sql += " AND path LIKE ?"
            params.append(f"{path_prefix}%")
        if kind:
            like_sql += " AND kind=?"
            params.append(kind)
        like_sql += " LIMIT ?"
        params.append(limit)
        for row in self.conn.execute(like_sql, params).fetchall():
            d = dict(row)
            by_id.setdefault(d["id"], d)

        fts_sql = """
          SELECT s.*, -bm25(symbols_fts, 4.0, 5.0, 1.0, 2.5, 2.5, 1.5, 0.4) AS score, 'fts' AS match_type
          FROM symbols_fts f JOIN symbols s ON s.id = f.symbol_id
          WHERE symbols_fts MATCH ?
        """
        params = [self._fts_query(q)]
        if path_prefix:
            fts_sql += " AND s.path LIKE ?"
            params.append(f"{path_prefix}%")
        if kind:
            fts_sql += " AND s.kind=?"
            params.append(kind)
        fts_sql += " ORDER BY score DESC LIMIT ?"
        params.append(limit)
        try:
            for row in self.conn.execute(fts_sql, params).fetchall():
                d = dict(row)
                if d["id"] in by_id:
                    by_id[d["id"]]["score"] = max(float(by_id[d["id"]]["score"]), float(d["score"]))
                else:
                    by_id[d["id"]] = d
        except sqlite3.OperationalError:
            pass
        out = list(by_id.values())
        out.sort(key=lambda d: (float(d.get("score", 0)), -int(d.get("line_start", 0))), reverse=True)
        return out[:limit]

    def path_search(self, query: str, limit: int, path_prefix: str | None = None) -> list[dict]:
        terms = _split_query_terms(query)
        if not terms:
            return []
        clauses = []
        params: list[object] = []
        for term in terms[:5]:
            clauses.append("lower(path) LIKE ?")
            params.append(f"%{term.lower()}%")
        sql = "SELECT *, 1.0 AS score FROM docs WHERE " + " AND ".join(clauses)
        if path_prefix:
            sql += " AND path LIKE ?"
            params.append(f"{path_prefix}%")
        sql += " ORDER BY length(path), path LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_symbol(self, symbol_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM symbols WHERE id=?", (symbol_id,)).fetchone()
        return dict(row) if row else None

    def get_symbol_by_name(self, name: str, path_prefix: str | None = None) -> dict | None:
        rows = self.symbol_search(name, 1, path_prefix=path_prefix)
        return rows[0] if rows else None

    def get_symbols(self, symbol_ids: Sequence[str]) -> list[dict]:
        if not symbol_ids:
            return []
        placeholders = ",".join("?" for _ in symbol_ids)
        rows = self.conn.execute(f"SELECT * FROM symbols WHERE id IN ({placeholders})", tuple(symbol_ids)).fetchall()
        by_id = {r["id"]: dict(r) for r in rows}
        return [by_id[sid] for sid in symbol_ids if sid in by_id]

    def symbols_in_doc(self, doc_id: str, *, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM symbols WHERE doc_id=? ORDER BY line_start, line_end LIMIT ?", (doc_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def find_references(self, target_name: str, limit: int, path_prefix: str | None = None) -> list[dict]:
        q = target_name.strip().split(".")[-1].lower()
        if not q:
            return []
        sql = "SELECT * FROM refs WHERE target_name_lc=?"
        params: list[object] = [q]
        if path_prefix:
            sql += " AND path LIKE ?"
            params.append(f"{path_prefix}%")
        sql += " ORDER BY path, line LIMIT ?"
        params.append(limit)
        rows = [dict(r) for r in self.conn.execute(sql, params).fetchall()]
        if rows:
            return rows
        # Fuzzy fallback for case-sensitive or member references.
        sql = "SELECT * FROM refs WHERE target_name_lc LIKE ?"
        params = [f"%{q}%"]
        if path_prefix:
            sql += " AND path LIKE ?"
            params.append(f"{path_prefix}%")
        sql += " ORDER BY path, line LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def repo_map(self, path_prefix: str | None = None, limit: int = 240) -> dict:
        params: list[object] = []
        where = ""
        if path_prefix:
            where = "WHERE path LIKE ?"
            params.append(f"{path_prefix}%")
        docs = [dict(r) for r in self.conn.execute(
            f"SELECT doc_id, path, title, size_bytes FROM docs {where} ORDER BY path LIMIT ?", [*params, limit]
        ).fetchall()]
        symbols = [dict(r) for r in self.conn.execute(
            f"""
            SELECT id, doc_id, path, language, kind, name, qualified_name, line_start, line_end
            FROM symbols {where}
            ORDER BY path, line_start LIMIT ?
            """,
            [*params, limit * 3],
        ).fetchall()]
        languages = [dict(r) for r in self.conn.execute(
            f"SELECT language, COUNT(*) AS symbols FROM symbols {where} GROUP BY language ORDER BY symbols DESC",
            params,
        ).fetchall()]
        by_doc: dict[str, list[dict]] = {}
        for sym in symbols:
            by_doc.setdefault(sym["doc_id"], []).append(sym)
        files = []
        for doc in docs:
            syms = by_doc.get(doc["doc_id"], [])[:25]
            files.append({
                "path": doc["path"],
                "size_bytes": doc["size_bytes"],
                "symbols": [
                    {
                        "id": s["id"],
                        "kind": s["kind"],
                        "name": s["name"],
                        "qualified_name": s["qualified_name"],
                        "line_start": s["line_start"],
                        "line_end": s["line_end"],
                    }
                    for s in syms
                ],
            })
        return {"files": files, "languages": languages, "file_count": len(docs), "symbol_count_sampled": len(symbols)}

    def _fts_query(self, query: str) -> str:
        terms = []
        for tok in _split_query_terms(query):
            tok = tok.strip().replace('"', "")
            if len(tok) >= 2:
                terms.append(tok)
        if not terms:
            return '""'
        # OR improves recall; exact paths/symbols still score well with bm25.
        return " OR ".join(f'"{t}"' for t in terms[:32])

    def optimize(self) -> None:
        """Finalize an index after a full/staging build."""
        try:
            self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            self.conn.execute("INSERT INTO symbols_fts(symbols_fts) VALUES('optimize')")
            self.conn.execute("ANALYZE")
            self.conn.commit()
        except sqlite3.Error:
            self.conn.rollback()

    def stats(self) -> dict:
        docs = self.conn.execute("SELECT COUNT(*) AS c FROM docs").fetchone()["c"]
        chunks = self.conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        symbols = self.conn.execute("SELECT COUNT(*) AS c FROM symbols").fetchone()["c"]
        refs = self.conn.execute("SELECT COUNT(*) AS c FROM refs").fetchone()["c"]
        vecs = self.conn.execute("SELECT COUNT(*) AS c FROM vectors").fetchone()["c"]
        return {
            "db": self.path.as_posix(),
            "docs": int(docs),
            "chunks": int(chunks),
            "symbols": int(symbols),
            "references": int(refs),
            "vectors": int(vecs),
            "corpus_version": self.corpus_version(),
        }


def _split_query_terms(query: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9_.$/@:-]+", query) if t]


def replace_index_database(source_db_dir: Path, target_db_dir: Path) -> None:
    """Atomically replace the target SQLite index with a completed staging index."""
    source_dir = source_db_dir.expanduser().resolve()
    target_dir = target_db_dir.expanduser().resolve()
    source = source_dir / "rag.sqlite3"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "rag.sqlite3"
    backup = target_dir / "rag.sqlite3.previous"

    if not source.exists():
        raise FileNotFoundError(f"staging database not found: {source}")

    for suffix in ("-wal", "-shm"):
        try:
            Path(str(target) + suffix).unlink()
        except FileNotFoundError:
            pass
    try:
        backup.unlink()
    except FileNotFoundError:
        pass

    had_target = target.exists()
    if had_target:
        os.replace(target, backup)
    try:
        os.replace(source, target)
    except Exception:
        if had_target and backup.exists():
            os.replace(backup, target)
        raise
    else:
        try:
            backup.unlink()
        except FileNotFoundError:
            pass
        shutil.rmtree(source_dir, ignore_errors=True)
