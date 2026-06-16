from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from dataclasses import asdict

from .config import Settings
from .embeddings import EmbeddingModel
from .indexer import Indexer
from .store import Store, replace_index_database
from .vector import VectorIndex

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _index_command(*, force: bool = False, build_vector_index: bool = False) -> None:
    """Index without constructing the search runtime.

    A forced rebuild is created in a staging SQLite database and swapped in only after success.
    This avoids the very slow FTS delete/update path and keeps the previous index valid if the
    rebuild fails.
    """
    settings = Settings()
    embeddings = EmbeddingModel(settings)
    staging_dir: Path | None = None
    store: Store | None = None

    try:
        index_settings = settings
        if force:
            target_dir = settings.resolved_db_dir()
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            staging_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{target_dir.name}.rebuild-",
                    dir=target_dir.parent,
                )
            )
            index_settings = settings.model_copy(update={"db_dir": staging_dir})

        store = Store(index_settings.db_dir, vector_storage_dtype=index_settings.vector_storage_dtype, bulk_build=force)
        indexer = Indexer(index_settings, store, embeddings)
        result = indexer.index_all(force=False)

        if force:
            store.optimize()
            store.checkpoint()
            store.close()
            store = None
            replace_index_database(staging_dir, settings.resolved_db_dir())
            result["full_rebuild"] = True
            result["db_swap"] = "completed"
        else:
            result["full_rebuild"] = False

        if build_vector_index:
            final_store = Store(settings.db_dir, vector_storage_dtype=settings.vector_storage_dtype)
            try:
                vector_index = VectorIndex(settings)
                final_indexer = Indexer(settings, final_store, embeddings)
                result["vector_reload"] = final_indexer.rebuild_vector_index(vector_index)
            finally:
                final_store.close()
        else:
            result["vector_reload"] = {
                "skipped": True,
                "reason": "CLI indexing does not load/build the in-memory search index by default.",
            }
        _print_json(result)
    finally:
        if store is not None:
            store.close()
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _status_command(*, runtime: bool = False) -> None:
    """Lightweight status by default.

    Runtime status loads the embedding model and full vector index, which is useful for diagnostics
    but surprising when the user only wants doc/chunk counts.
    """
    if runtime:
        from .runtime import Runtime

        rt = Runtime()
        _print_json(rt.status())
        return

    settings = Settings()
    store = Store(settings.db_dir, vector_storage_dtype=settings.vector_storage_dtype)
    try:
        _print_json({
            "settings": {
                "db_dir": settings.db_dir.as_posix(),
                "roots": [p.as_posix() for p in settings.resolved_roots()],
                "embedding_model": settings.embedding_model,
                "device": settings.device,
                "vector_backend": settings.vector_backend,
                "faiss_gpu": settings.faiss_gpu,
                "torch_vector_device": settings.torch_vector_device,
                "keep_cpu_vector_copy": settings.keep_cpu_vector_copy,
                "read_only": settings.read_only,
                "allow_reindex_tool": settings.allow_reindex_tool,
                "chunk_tokens": settings.chunk_tokens,
                "add_file_summary_chunks": settings.add_file_summary_chunks,
            },
            "store": store.stats(),
            "runtime_loaded": False,
            "hint": "Use `brag status --runtime` to load the embedding model and vector index for full diagnostics.",
        })
    finally:
        store.close()



def _scan_command() -> None:
    """Scan candidate files without loading the embedding model. Useful for diagnosing memory blowups."""
    from .io import ScanStats, iter_candidate_files

    settings = Settings()
    stats = ScanStats()
    # Consume the generator to populate counts, but do not read file contents or load models.
    for _ in iter_candidate_files(settings, stats):
        pass
    _print_json({
        "settings": {
            "roots": [p.as_posix() for p in settings.resolved_roots()],
            "db_dir": settings.resolved_db_dir().as_posix(),
            "max_file_bytes": settings.max_file_bytes,
            "max_files": settings.max_files,
            "exclude_hidden_dirs": settings.exclude_hidden_dirs,
        },
        "scan": stats.as_dict(),
    })



def _tune_command(sample_chunks: int, batch_sizes: str) -> None:
    """Benchmark embedding batch sizes on real chunks without modifying the index."""
    from .chunking import chunk_document
    from .code_index import extract_symbols
    from .indexer import _embedding_text
    from .io import ScanStats, iter_candidate_files_with_roots, read_document

    settings = Settings()
    embeddings = EmbeddingModel(settings)
    stats = ScanStats()
    texts: list[str] = []
    for root, path in iter_candidate_files_with_roots(settings, stats):
        doc = read_document(root, path, settings)
        if doc is None:
            continue
        symbols = extract_symbols(doc)
        chunks = chunk_document(doc, settings, symbols=symbols)
        for chunk in chunks:
            texts.append(_embedding_text(chunk))
            if len(texts) >= sample_chunks:
                break
        if len(texts) >= sample_chunks:
            break
    candidates = [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]
    result = embeddings.benchmark_batch_sizes(texts, candidates)
    result["scan"] = stats.as_dict()
    result["device"] = embeddings.info.device
    result["model"] = embeddings.info.model
    _print_json(result)


def main() -> None:
    parser = argparse.ArgumentParser(prog="brag", description="Blazing local code/RAG MCP server")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index configured BRAG_ROOTS")
    p_index.add_argument("--force", action="store_true", help="Re-embed and rewrite all documents")
    p_index.add_argument(
        "--build-vector-index",
        action="store_true",
        help="Also load/build the in-memory vector index after indexing. Off by default to keep indexing memory low.",
    )

    sub.add_parser("scan", help="Scan files that would be indexed without loading the embedding model")

    p_tune = sub.add_parser("tune", help="Benchmark embedding batch sizes on representative code chunks")
    p_tune.add_argument("--sample-chunks", type=int, default=128)
    p_tune.add_argument("--batch-sizes", default="8,16,24,32,48,64")

    p_status = sub.add_parser("status", help="Show index/runtime status")
    p_status.add_argument(
        "--runtime",
        action="store_true",
        help="Load embedding model and vector index before printing status. Higher memory, but shows active backend.",
    )

    p_search = sub.add_parser("search", help="Code-aware hybrid search without MCP")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=None)
    p_search.add_argument("--mode", choices=["auto", "code", "hybrid", "semantic", "keyword", "symbol"], default="auto")
    p_search.add_argument("--path-prefix", default=None)
    p_search.add_argument("--include-text", action="store_true")

    p_symbol = sub.add_parser("symbol", help="Find symbols by name/qualified name")
    p_symbol.add_argument("name")
    p_symbol.add_argument("--kind", default=None)
    p_symbol.add_argument("--path-prefix", default=None)
    p_symbol.add_argument("--limit", type=int, default=None)

    p_refs = sub.add_parser("refs", help="Find references/calls/imports for a symbol")
    p_refs.add_argument("symbol")
    p_refs.add_argument("--path-prefix", default=None)
    p_refs.add_argument("--limit", type=int, default=None)

    p_fetch = sub.add_parser("fetch", help="Fetch rag:// chunk or symbol:// symbol")
    p_fetch.add_argument("uri_or_id")

    p_map = sub.add_parser("map", help="Print compact repository map")
    p_map.add_argument("--path-prefix", default=None)
    p_map.add_argument("--limit", type=int, default=None)

    sub.add_parser("serve", help="Run MCP server over stdio")

    args = parser.parse_args()

    if args.cmd == "serve":
        from .server import main as server_main

        server_main()
        return

    if args.cmd == "index":
        _index_command(force=args.force, build_vector_index=args.build_vector_index)
        return

    if args.cmd == "status":
        _status_command(runtime=args.runtime)
        return

    if args.cmd == "scan":
        _scan_command()
        return

    if args.cmd == "tune":
        _tune_command(args.sample_chunks, args.batch_sizes)
        return

    from .runtime import Runtime

    rt = Runtime()
    if args.cmd == "search":
        _print_json(rt.retriever.code_search(
            args.query,
            top_k=args.top_k,
            mode=args.mode,
            path_prefix=args.path_prefix,
            include_text=args.include_text,
        ))
    elif args.cmd == "symbol":
        _print_json(rt.retriever.find_symbol(args.name, kind=args.kind, path_prefix=args.path_prefix, limit=args.limit))
    elif args.cmd == "refs":
        _print_json(rt.retriever.references(args.symbol, path_prefix=args.path_prefix, limit=args.limit))
    elif args.cmd == "fetch":
        value = args.uri_or_id
        if value.startswith("rag://") or value.startswith("symbol://"):
            _print_json(rt.retriever.fetch(resource_uri=value))
        elif len(value) >= 20:
            try:
                _print_json(rt.retriever.fetch(symbol_id=value))
            except Exception:
                _print_json(rt.retriever.fetch(chunk_id=value))
        else:
            _print_json({"error": "provide a rag:// URI, symbol:// URI, chunk id, or symbol id"})
    elif args.cmd == "map":
        _print_json(rt.retriever.repo_map(path_prefix=args.path_prefix, limit=args.limit))


if __name__ == "__main__":
    main()
