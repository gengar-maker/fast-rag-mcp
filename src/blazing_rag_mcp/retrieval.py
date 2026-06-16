from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from typing import Literal

from cachetools import LRUCache

from .config import Settings
from .embeddings import EmbeddingModel
from .store import Store
from .vector import VectorIndex

Mode = Literal["auto", "code", "hybrid", "semantic", "keyword", "symbol"]


class Retriever:
    def __init__(self, settings: Settings, store: Store, embeddings: EmbeddingModel, vector_index: VectorIndex):
        self.settings = settings
        self.store = store
        self.embeddings = embeddings
        self.vector_index = vector_index
        self._search_cache: LRUCache[str, dict] = LRUCache(maxsize=1024)
        self._loaded_version = store.corpus_version()

    def ensure_fresh_index(self) -> None:
        version = self.store.corpus_version()
        if version != self._loaded_version:
            ids, vectors = self.store.all_vectors()
            self.vector_index.build(ids, vectors)
            self._loaded_version = version
            self._search_cache.clear()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: Mode = "auto",
        path_prefix: str | None = None,
        include_text: bool = False,
    ) -> dict:
        # Backwards-compatible generic RAG endpoint. In auto mode, use code-aware search because
        # it dominates generic retrieval on repositories and still falls back to dense+FTS.
        return self.code_search(
            query=query,
            top_k=top_k,
            mode=mode,
            path_prefix=path_prefix,
            include_text=include_text,
        )

    def code_search(
        self,
        query: str,
        top_k: int | None = None,
        mode: Mode = "code",
        path_prefix: str | None = None,
        include_text: bool = False,
    ) -> dict:
        self.ensure_fresh_index()
        started = time.perf_counter()
        top_k = min(max(1, top_k or self.settings.default_top_k), self.settings.max_top_k)
        requested_mode = mode
        mode = "code" if mode == "auto" else mode
        cache_key = json.dumps(
            {
                "v": self.store.corpus_version(),
                "q": query,
                "k": top_k,
                "mode": requested_mode,
                "path_prefix": path_prefix,
                "include_text": include_text,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            result = dict(cached)
            result["cached"] = True
            return result

        timings: dict[str, float] = {}
        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []
        symbol_chunks: list[tuple[str, float]] = []
        path_chunks: list[tuple[str, float]] = []
        symbol_hits: list[dict] = []
        path_hits: list[dict] = []

        if mode in ("code", "hybrid", "symbol"):
            t0 = time.perf_counter()
            symbol_hits = self.store.symbol_search(query, self.settings.candidate_k, path_prefix=path_prefix)
            symbol_chunks = self._symbol_hits_to_chunks(symbol_hits)
            path_hits = self.store.path_search(query, min(self.settings.candidate_k, 50), path_prefix=path_prefix)
            path_chunks = self._path_hits_to_chunks(path_hits)
            timings["symbol_path_ms"] = _ms(t0)

        exact_symbol_hit = bool(symbol_hits and float(symbol_hits[0].get("score", 0.0)) >= 100.0)
        identifier_query = bool(re.fullmatch(r"[A-Za-z_$][\w$]*(?:[.:][A-Za-z_$][\w$]*)*", query.strip()))
        skip_dense = bool(
            requested_mode == "auto"
            and self.settings.exact_symbol_fast_path
            and exact_symbol_hit
            and identifier_query
        )
        if skip_dense:
            timings["route"] = "exact-symbol-fast-path"

        if mode in ("code", "hybrid", "semantic") and not skip_dense:
            t0 = time.perf_counter()
            qvec = self.embeddings.embed_query(query)
            timings["embed_ms"] = _ms(t0)
            t0 = time.perf_counter()
            dense = self.vector_index.search(qvec, self.settings.candidate_k)
            timings["dense_ms"] = _ms(t0)

        if mode in ("code", "hybrid", "keyword"):
            t0 = time.perf_counter()
            sparse = self.store.fts_search(query, self.settings.candidate_k, path_prefix=path_prefix)
            timings["fts_ms"] = _ms(t0)

        t0 = time.perf_counter()
        fused = self._fuse_code(symbol_chunks, path_chunks, dense, sparse, mode)
        chunk_ids = [cid for cid, _ in fused[:top_k]]
        chunks = self.store.get_chunks(chunk_ids)
        timings["pack_ms"] = _ms(t0)

        by_score = {cid: score for cid, score in fused}
        results = [self._pack_chunk(c, by_score.get(c["id"], 0.0), include_text) for c in chunks]
        result = {
            "query": query,
            "corpus_version": self.store.corpus_version(),
            "mode": "exact-symbol-fast-path" if skip_dense else mode,
            "cached": False,
            "results": results,
            "symbol_hits_preview": [self._pack_symbol(s, compact=True) for s in symbol_hits[: min(8, top_k)]],
            "path_hits_preview": [{"path": p["path"], "doc_id": p["doc_id"]} for p in path_hits[: min(8, top_k)]],
            "timings_ms": {**timings, "total_ms": _ms(started)},
            "index": asdict(self.vector_index.info()),
            "embedding": asdict(self.embeddings.info),
        }
        self._search_cache[cache_key] = result
        return result

    def find_symbol(
        self,
        name: str,
        kind: str | None = None,
        path_prefix: str | None = None,
        limit: int | None = None,
    ) -> dict:
        self.ensure_fresh_index()
        started = time.perf_counter()
        limit = min(max(1, limit or self.settings.default_top_k), self.settings.max_top_k)
        rows = self.store.symbol_search(name, limit, path_prefix=path_prefix, kind=kind)
        return {
            "query": name,
            "kind": kind,
            "results": [self._pack_symbol(r) for r in rows],
            "timings_ms": {"total_ms": _ms(started)},
        }

    def references(self, symbol: str, path_prefix: str | None = None, limit: int | None = None) -> dict:
        self.ensure_fresh_index()
        started = time.perf_counter()
        limit = min(max(1, limit or self.settings.code_reference_limit), self.settings.code_reference_limit * 4)
        sym = self.store.get_symbol(symbol) or self.store.get_symbol_by_name(symbol, path_prefix=path_prefix)
        target = sym["name"] if sym else symbol.split(".")[-1]
        refs = self.store.find_references(target, limit, path_prefix=path_prefix)
        fallback_chunks: list[dict] = []
        if not refs:
            fts = self.store.fts_search(target, min(limit, self.settings.candidate_k), path_prefix=path_prefix)
            fallback_chunks = self.store.get_chunks([cid for cid, _ in fts[:limit]])
        return {
            "symbol": self._pack_symbol(sym) if sym else {"query": symbol, "resolved": False},
            "target_name": target,
            "references": [self._pack_reference(r) for r in refs],
            "fallback_search_results": [self._pack_chunk(c, 0.0, include_text=False) for c in fallback_chunks[:limit]],
            "timings_ms": {"total_ms": _ms(started)},
        }

    def neighbors(self, symbol: str, path_prefix: str | None = None, limit: int = 40) -> dict:
        self.ensure_fresh_index()
        started = time.perf_counter()
        sym = self.store.get_symbol(symbol) or self.store.get_symbol_by_name(symbol, path_prefix=path_prefix)
        if not sym:
            return {"symbol": {"query": symbol, "resolved": False}, "neighbors": {}, "timings_ms": {"total_ms": _ms(started)}}
        siblings = [s for s in self.store.symbols_in_doc(sym["doc_id"], limit=limit * 3) if s["id"] != sym["id"]]
        outgoing_refs = self.store.find_references(sym["name"], limit, path_prefix=path_prefix)
        # Incoming callers: any reference to this symbol name; source_symbol_id lets the agent jump to caller.
        callers = []
        for ref in outgoing_refs:
            if ref.get("source_symbol_id"):
                caller = self.store.get_symbol(ref["source_symbol_id"])
                if caller:
                    callers.append(caller)
        return {
            "symbol": self._pack_symbol(sym),
            "neighbors": {
                "same_file_symbols": [self._pack_symbol(s, compact=True) for s in siblings[:limit]],
                "references_to_symbol": [self._pack_reference(r) for r in outgoing_refs[:limit]],
                "callers_or_containing_symbols": [self._pack_symbol(c, compact=True) for c in callers[:limit]],
            },
            "timings_ms": {"total_ms": _ms(started)},
        }

    def repo_map(self, path_prefix: str | None = None, limit: int | None = None) -> dict:
        self.ensure_fresh_index()
        limit = min(max(1, limit or self.settings.code_map_limit), self.settings.code_map_limit * 3)
        started = time.perf_counter()
        out = self.store.repo_map(path_prefix=path_prefix, limit=limit)
        out["timings_ms"] = {"total_ms": _ms(started)}
        out["corpus_version"] = self.store.corpus_version()
        return out

    def fetch(
        self,
        resource_uri: str | None = None,
        chunk_id: str | None = None,
        symbol_id: str | None = None,
        context_chunks: int = 0,
    ) -> dict:
        self.ensure_fresh_index()
        sid = symbol_id or _symbol_id_from_uri(resource_uri or "")
        if sid:
            sym = self.store.get_symbol(sid)
            if sym is None:
                raise KeyError(f"Symbol not found: {sid}")
            text = sym["text"]
            if len(text) > self.settings.max_fetch_chars:
                text = text[: self.settings.max_fetch_chars] + "\n...[truncated]"
            return {**self._pack_symbol(sym), "text": text, "metadata": _loads(sym.get("metadata_json", "{}"))}

        cid = chunk_id or _chunk_id_from_uri(resource_uri or "")
        if not cid:
            raise ValueError("Provide chunk_id, symbol_id, rag://...#chunk=<id>, or symbol://<id>")
        chunk = self.store.get_chunk(cid)
        if chunk is None:
            raise KeyError(f"Chunk not found: {cid}")
        text = chunk["text"]
        if len(text) > self.settings.max_fetch_chars:
            text = text[: self.settings.max_fetch_chars] + "\n...[truncated]"
        return {
            "resource_uri": f"rag://{chunk['doc_id']}#chunk={chunk['id']}",
            "id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "path": chunk["path"],
            "section": chunk["section"],
            "language": chunk.get("language", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "symbol_id": chunk.get("symbol_id", ""),
            "symbol_name": chunk.get("symbol_name", ""),
            "qualified_name": chunk.get("qualified_name", ""),
            "line_start": chunk["line_start"],
            "line_end": chunk["line_end"],
            "text": text,
            "metadata": _loads(chunk["metadata_json"] or "{}"),
        }

    def _symbol_hits_to_chunks(self, symbol_hits: list[dict]) -> list[tuple[str, float]]:
        symbol_ids = [s["id"] for s in symbol_hits]
        chunks = self.store.chunks_for_symbols(symbol_ids, limit_per_symbol=1)
        score_by_symbol = {s["id"]: float(s.get("score", 1.0)) for s in symbol_hits}
        out = []
        for c in chunks:
            out.append((c["id"], score_by_symbol.get(c.get("symbol_id", ""), 1.0)))
        return out

    def _path_hits_to_chunks(self, path_hits: list[dict]) -> list[tuple[str, float]]:
        chunks = self.store.chunks_for_docs([p["doc_id"] for p in path_hits], limit_per_doc=1)
        rank_by_doc = {p["doc_id"]: idx + 1 for idx, p in enumerate(path_hits)}
        out = []
        for c in chunks:
            out.append((c["id"], 1.0 / rank_by_doc.get(c.get("doc_id", ""), 1)))
        return out

    def _fuse_code(
        self,
        symbol_chunks: list[tuple[str, float]],
        path_chunks: list[tuple[str, float]],
        dense: list[tuple[str, float]],
        sparse: list[tuple[str, float]],
        mode: str,
    ) -> list[tuple[str, float]]:
        if mode == "semantic":
            return dense
        if mode == "keyword":
            return sparse
        if mode == "symbol":
            return symbol_chunks or path_chunks
        scores: dict[str, float] = {}
        rrf_k = self.settings.rrf_k

        def add(results: list[tuple[str, float]], weight: float, raw_scale: float = 0.0) -> None:
            for rank, (cid, raw) in enumerate(results, start=1):
                scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + rank) + raw_scale * max(0.0, float(raw))

        add(symbol_chunks, self.settings.code_symbol_weight, raw_scale=0.0005)
        add(path_chunks, self.settings.code_path_weight)
        add(dense, self.settings.dense_weight)
        add(sparse, self.settings.fts_weight)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def _pack_chunk(self, chunk: dict, score: float, include_text: bool) -> dict:
        text = chunk["text"]
        snippet = text[: self.settings.max_snippet_chars]
        if len(text) > self.settings.max_snippet_chars:
            snippet += "…"
        out = {
            "id": chunk["id"],
            "resource_uri": f"rag://{chunk['doc_id']}#chunk={chunk['id']}",
            "score": round(float(score), 6),
            "title": chunk["title"],
            "path": chunk["path"],
            "section": chunk["section"],
            "language": chunk.get("language", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "symbol_id": chunk.get("symbol_id", ""),
            "symbol_name": chunk.get("symbol_name", ""),
            "qualified_name": chunk.get("qualified_name", ""),
            "line_start": chunk["line_start"],
            "line_end": chunk["line_end"],
            "snippet": snippet,
        }
        if out["symbol_id"]:
            out["symbol_uri"] = f"symbol://{out['symbol_id']}"
        if include_text:
            out["text"] = text
        return out

    def _pack_symbol(self, sym: dict | None, compact: bool = False) -> dict:
        if not sym:
            return {}
        out = {
            "id": sym["id"],
            "resource_uri": f"symbol://{sym['id']}",
            "score": round(float(sym.get("score", 0.0)), 6),
            "match_type": sym.get("match_type", ""),
            "kind": sym["kind"],
            "name": sym["name"],
            "qualified_name": sym["qualified_name"],
            "path": sym["path"],
            "language": sym["language"],
            "line_start": sym["line_start"],
            "line_end": sym["line_end"],
            "signature": sym["signature"],
        }
        if not compact:
            out["docstring"] = sym.get("docstring", "")
            out["parent"] = sym.get("parent", "")
        return out

    def _pack_reference(self, ref: dict) -> dict:
        return {
            "id": ref["id"],
            "path": ref["path"],
            "language": ref["language"],
            "source_symbol_id": ref.get("source_symbol_id", ""),
            "source_symbol_uri": f"symbol://{ref['source_symbol_id']}" if ref.get("source_symbol_id") else "",
            "target_name": ref["target_name"],
            "ref_kind": ref["ref_kind"],
            "line": ref["line"],
            "snippet": ref["snippet"],
        }


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _chunk_id_from_uri(uri: str) -> str:
    if "#chunk=" not in uri:
        return ""
    return uri.rsplit("#chunk=", 1)[-1].strip()


def _symbol_id_from_uri(uri: str) -> str:
    if not uri.startswith("symbol://"):
        return ""
    return uri.replace("symbol://", "", 1).strip()


def _loads(value: str) -> dict:
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}
