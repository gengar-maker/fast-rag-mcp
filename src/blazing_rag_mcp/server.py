from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Literal

from mcp.server.fastmcp import FastMCP

# Never log to stdout for stdio MCP: stdout is the JSON-RPC transport.
logging.basicConfig(
    level=getattr(logging, os.environ.get("BRAG_LOG_LEVEL", "INFO").upper(), logging.INFO),
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# IMPORTANT: do not construct Runtime at module import time. Loading Torch, a Hugging Face model,
# SQLite vectors, and an MPS tensor before FastMCP starts blocks the initialize handshake and can
# make Claude Code report an init failure. Runtime is loaded on the first tool that needs it.
_runtime = None
_runtime_lock = threading.Lock()
_runtime_init_ms: float | None = None

mcp = FastMCP("blazing-code-rag", json_response=True)


def _get_runtime():
    global _runtime, _runtime_init_ms
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        started = time.perf_counter()
        log.warning("initializing code RAG runtime lazily")
        from .runtime import Runtime

        _runtime = Runtime()
        _runtime_init_ms = round((time.perf_counter() - started) * 1000, 2)
        log.warning("code RAG runtime initialized in %.2f ms", _runtime_init_ms)
        return _runtime


def _lightweight_status() -> dict:
    """Return index/config status without importing Torch or loading the embedding model."""
    from .config import Settings
    from .store import Store

    settings = Settings()
    store = Store(settings.db_dir, vector_storage_dtype=settings.vector_storage_dtype)
    try:
        return {
            "ok": True,
            "runtime_loaded": _runtime is not None,
            "runtime_init_ms": _runtime_init_ms,
            "settings": {
                "db_dir": settings.resolved_db_dir().as_posix(),
                "roots": [p.as_posix() for p in settings.resolved_roots()],
                "embedding_model": settings.embedding_model,
                "device": settings.device,
                "vector_backend": settings.vector_backend,
                "torch_vector_device": settings.torch_vector_device,
                "read_only": settings.read_only,
                "allow_reindex_tool": settings.allow_reindex_tool,
            },
            "store": store.stats(),
        }
    finally:
        store.close()


@mcp.tool()
def code_search(
    query: str,
    top_k: int | None = None,
    mode: Literal["auto", "code", "hybrid", "semantic", "keyword", "symbol"] = "auto",
    path_prefix: str | None = None,
    include_text: bool = False,
) -> dict:
    """Search code with symbol-first hybrid retrieval.

    Best first tool for coding agents. It combines exact/fuzzy symbol lookup, path lookup,
    BM25/FTS, and GPU-accelerated dense retrieval. Returns compact snippets plus rag:// and
    symbol:// URIs; call code_fetch for full context.
    """
    runtime = _get_runtime()
    return runtime.retriever.code_search(
        query=query,
        top_k=top_k,
        mode=mode,
        path_prefix=path_prefix,
        include_text=include_text,
    )


@mcp.tool()
def code_find_symbol(
    name: str,
    kind: str | None = None,
    path_prefix: str | None = None,
    limit: int | None = None,
) -> dict:
    """Find definitions by exact/fuzzy symbol name or qualified name."""
    runtime = _get_runtime()
    return runtime.retriever.find_symbol(name=name, kind=kind, path_prefix=path_prefix, limit=limit)


@mcp.tool()
def code_references(symbol: str, path_prefix: str | None = None, limit: int | None = None) -> dict:
    """Find indexed references/call sites/import mentions for a symbol name or symbol:// id."""
    runtime = _get_runtime()
    return runtime.retriever.references(symbol=symbol, path_prefix=path_prefix, limit=limit)


@mcp.tool()
def code_neighbors(symbol: str, path_prefix: str | None = None, limit: int = 40) -> dict:
    """Return nearby code-intelligence context for a symbol."""
    runtime = _get_runtime()
    return runtime.retriever.neighbors(symbol=symbol, path_prefix=path_prefix, limit=limit)


@mcp.tool()
def code_repo_map(path_prefix: str | None = None, limit: int | None = None) -> dict:
    """Return a compact repository/module map: files, languages, and top symbols per file."""
    runtime = _get_runtime()
    return runtime.retriever.repo_map(path_prefix=path_prefix, limit=limit)


@mcp.tool()
def code_fetch(
    resource_uri: str | None = None,
    chunk_id: str | None = None,
    symbol_id: str | None = None,
) -> dict:
    """Fetch full text for a rag:// chunk or symbol:// symbol."""
    runtime = _get_runtime()
    return runtime.retriever.fetch(
        resource_uri=resource_uri,
        chunk_id=chunk_id,
        symbol_id=symbol_id,
    )


@mcp.tool()
def rag_search(
    query: str,
    top_k: int | None = None,
    mode: Literal["auto", "code", "hybrid", "semantic", "keyword", "symbol"] = "auto",
    path_prefix: str | None = None,
    include_text: bool = False,
) -> dict:
    """Backward-compatible alias for code_search/generic hybrid retrieval."""
    runtime = _get_runtime()
    return runtime.retriever.search(
        query=query,
        top_k=top_k,
        mode=mode,
        path_prefix=path_prefix,
        include_text=include_text,
    )


@mcp.tool()
def rag_fetch(
    resource_uri: str | None = None,
    chunk_id: str | None = None,
    symbol_id: str | None = None,
) -> dict:
    """Backward-compatible fetch for rag:// chunks and symbol:// symbols."""
    runtime = _get_runtime()
    return runtime.retriever.fetch(
        resource_uri=resource_uri,
        chunk_id=chunk_id,
        symbol_id=symbol_id,
    )


@mcp.tool()
def rag_status(load_runtime: bool = False) -> dict:
    """Return index status. Set load_runtime=true to initialize model and vector backend."""
    if not load_runtime:
        return _lightweight_status()
    runtime = _get_runtime()
    return {"ok": True, "runtime_loaded": True, "runtime_init_ms": _runtime_init_ms, **runtime.status()}


@mcp.tool()
def rag_warmup() -> dict:
    """Explicitly initialize the embedding model and vector index after MCP has connected."""
    try:
        runtime = _get_runtime()
        runtime.embeddings.warmup()
        return {
            "ok": True,
            "runtime_init_ms": _runtime_init_ms,
            "runtime": runtime.status(),
        }
    except Exception as exc:
        log.exception("runtime warmup failed")
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Run `brag status --runtime` in a terminal with the same BRAG_* environment to see the full traceback.",
        }


@mcp.tool()
def rag_reindex(force: bool = False) -> dict:
    """Reindex configured local roots. Disabled unless BRAG_ALLOW_REINDEX_TOOL=true."""
    runtime = _get_runtime()
    if not runtime.settings.allow_reindex_tool:
        return {
            "ok": False,
            "error": "rag_reindex is disabled. Set BRAG_ALLOW_REINDEX_TOOL=true or run `brag index` manually.",
        }
    result = runtime.reindex(force=force)
    return {"ok": True, **result}


def main() -> None:
    # At this point only the lightweight MCP SDK and tool schemas are loaded. Claude can complete
    # initialize/listTools immediately; Torch/model/vector initialization happens on first use.
    log.warning("starting blazing-code-rag MCP over stdio (lazy runtime enabled)")
    mcp.run()


if __name__ == "__main__":
    main()
