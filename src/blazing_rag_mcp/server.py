from __future__ import annotations

import atexit
import logging
import os
import sys
from typing import Literal

from mcp.server.fastmcp import FastMCP

from .application import Application

# stdout is the stdio JSON-RPC transport. Production logs must stay on stderr.
logging.basicConfig(
    level=getattr(logging, os.environ.get("BRAG_LOG_LEVEL", "INFO").upper(), logging.INFO),
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

mcp = FastMCP("blazing-code-rag", json_response=True)
_app = Application()
atexit.register(_app.close)


def _error(exc: Exception) -> dict:
    log.exception("MCP operation failed")
    return {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


@mcp.tool()
def code_search(
    query: str,
    top_k: int | None = None,
    mode: Literal["auto", "code", "hybrid", "semantic", "keyword", "symbol"] = "auto",
    path_prefix: str | None = None,
    include_text: bool = False,
) -> dict:
    """Search code using exact symbols, paths, FTS and lazy dense retrieval.

    Exact symbol and keyword-only routes do not load the embedding model or vector matrix.
    """
    return _app.code_search(
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
    """Find definitions by exact/fuzzy symbol name without loading dense retrieval."""
    return _app.find_symbol(name=name, kind=kind, path_prefix=path_prefix, limit=limit)


@mcp.tool()
def code_references(
    symbol: str,
    path_prefix: str | None = None,
    limit: int | None = None,
) -> dict:
    """Find indexed references, calls and import mentions for a symbol."""
    return _app.references(symbol=symbol, path_prefix=path_prefix, limit=limit)


@mcp.tool()
def code_neighbors(symbol: str, path_prefix: str | None = None, limit: int = 40) -> dict:
    """Return same-file symbols and callers/references around a symbol."""
    return _app.neighbors(symbol=symbol, path_prefix=path_prefix, limit=limit)


@mcp.tool()
def code_repo_map(path_prefix: str | None = None, limit: int | None = None) -> dict:
    """Return a compact repository map without loading dense retrieval."""
    return _app.repo_map(path_prefix=path_prefix, limit=limit)


@mcp.tool()
def code_fetch(
    resource_uri: str | None = None,
    chunk_id: str | None = None,
    symbol_id: str | None = None,
) -> dict:
    """Fetch full text for a rag:// chunk or symbol:// symbol."""
    return _app.fetch(resource_uri=resource_uri, chunk_id=chunk_id, symbol_id=symbol_id)


@mcp.tool()
def code_update_index(paths: list[str], refresh_vectors: bool = False) -> dict:
    """Incrementally update only the provided files/directories.

    This is the preferred tool after an editor change. It does not scan the whole repository and
    does not rebuild the vector matrix. Dense vectors reload lazily on the next semantic search.
    Requires BRAG_READ_ONLY=false and BRAG_ALLOW_REINDEX_TOOL=true.
    """
    try:
        return {
            "ok": True,
            **_app.reindex(
                paths=paths,
                refresh_vectors=refresh_vectors,
                enforce_mcp_policy=True,
            ),
        }
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def rag_search(
    query: str,
    top_k: int | None = None,
    mode: Literal["auto", "code", "hybrid", "semantic", "keyword", "symbol"] = "auto",
    path_prefix: str | None = None,
    include_text: bool = False,
) -> dict:
    """Backward-compatible alias for code_search."""
    return code_search(query, top_k, mode, path_prefix, include_text)


@mcp.tool()
def rag_fetch(
    resource_uri: str | None = None,
    chunk_id: str | None = None,
    symbol_id: str | None = None,
) -> dict:
    """Backward-compatible alias for code_fetch."""
    return code_fetch(resource_uri, chunk_id, symbol_id)


@mcp.tool()
def rag_status(load_model: bool = False, load_vectors: bool = False) -> dict:
    """Return health/config/index status; heavy components stay lazy unless requested."""
    try:
        return _app.status(load_model=load_model, load_vectors=load_vectors)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def rag_doctor(load_model: bool = False) -> dict:
    """Run SQLite integrity and index compatibility checks."""
    try:
        return _app.doctor(load_model=load_model)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def rag_warmup(load_vectors: bool = True) -> dict:
    """Warm the embedding model and optionally load the vector matrix after MCP connects."""
    try:
        return _app.warmup(load_vectors=load_vectors)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def rag_reindex(
    force: bool = False,
    paths: list[str] | None = None,
    refresh_vectors: bool = False,
) -> dict:
    """Reindex configured roots, selected paths, or atomically rebuild with force=true.

    Pass paths for point updates. Omitting paths performs a repository consistency scan. Vector
    reload is deferred by default so index updates are not dominated by rebuilding the full search
    matrix. force=true and paths are mutually exclusive.
    """
    try:
        return {
            "ok": True,
            **_app.reindex(
                force=force,
                paths=paths,
                refresh_vectors=refresh_vectors,
                enforce_mcp_policy=True,
            ),
        }
    except Exception as exc:
        return _error(exc)


def main() -> None:
    log.info("starting blazing-code-rag MCP over stdio (production lazy lifecycle)")
    mcp.run()


if __name__ == "__main__":
    main()
