from __future__ import annotations

import argparse
import json
from typing import Any

from .application import Application
from .config import Settings


def _print_json(value: Any) -> None:
    try:
        import orjson

        print(orjson.dumps(value, option=orjson.OPT_INDENT_2).decode("utf-8"))
    except Exception:
        print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _scan_command() -> None:
    from .io import ScanStats, iter_candidate_files

    settings = Settings()
    stats = ScanStats()
    for _ in iter_candidate_files(settings, stats):
        pass
    _print_json(
        {
            "settings": {
                "roots": [path.as_posix() for path in settings.resolved_roots()],
                "db_dir": settings.resolved_db_dir().as_posix(),
                "max_file_bytes": settings.max_file_bytes,
                "max_files": settings.max_files,
                "scan_backend": settings.scan_backend,
            },
            "scan": stats.as_dict(),
        }
    )


def _tune_command(sample_chunks: int, batch_sizes: str) -> None:
    from .chunking import chunk_document
    from .code_index import extract_symbols
    from .embeddings import EmbeddingModel
    from .indexer import _embedding_text
    from .io import ScanStats, iter_candidate_files_with_roots, read_document

    settings = Settings()
    embeddings = EmbeddingModel(settings)
    stats = ScanStats()
    texts: list[str] = []
    for root, path in iter_candidate_files_with_roots(settings, stats):
        document = read_document(root, path, settings)
        if document is None:
            continue
        symbols = extract_symbols(document)
        for chunk in chunk_document(document, settings, symbols=symbols):
            texts.append(_embedding_text(chunk))
            if len(texts) >= sample_chunks:
                break
        if len(texts) >= sample_chunks:
            break
    candidates = [int(value.strip()) for value in batch_sizes.split(",") if value.strip()]
    result = embeddings.benchmark_batch_sizes(texts, candidates)
    result["scan"] = stats.as_dict()
    result["device"] = embeddings.info.device
    result["model"] = embeddings.info.model
    _print_json(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="brag", description="Production local code-intelligence RAG MCP server"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Incrementally index configured roots")
    index_parser.add_argument(
        "paths",
        nargs="*",
        help="Optional files/directories to update without scanning the repository",
    )
    index_parser.add_argument("--force", action="store_true", help="Atomic full rebuild")
    index_parser.add_argument(
        "--refresh-vectors",
        action="store_true",
        help="Reload vectors in this process after indexing (normally unnecessary)",
    )

    subparsers.add_parser("scan", help="List scan statistics without loading the model")

    tune_parser = subparsers.add_parser("tune", help="Benchmark MPS/CUDA embedding batches")
    tune_parser.add_argument("--sample-chunks", type=int, default=128)
    tune_parser.add_argument("--batch-sizes", default="8,16,24,32,48,64")

    status_parser = subparsers.add_parser("status", help="Show lightweight status")
    status_parser.add_argument("--model", action="store_true", help="Load embedding model")
    status_parser.add_argument("--vectors", action="store_true", help="Load model and vectors")

    doctor_parser = subparsers.add_parser("doctor", help="Run integrity/compatibility checks")
    doctor_parser.add_argument(
        "--model", action="store_true", help="Also validate model compatibility"
    )

    warmup_parser = subparsers.add_parser(
        "warmup", help="Load/warm the embedding model and optionally the vector matrix"
    )
    warmup_parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="Warm only the embedding model; useful for model prefetch and MCP startup",
    )

    search_parser = subparsers.add_parser("search", help="Code-aware hybrid search")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=None)
    search_parser.add_argument(
        "--mode",
        choices=["auto", "code", "hybrid", "semantic", "keyword", "symbol"],
        default="auto",
    )
    search_parser.add_argument("--path-prefix", default=None)
    search_parser.add_argument("--include-text", action="store_true")

    symbol_parser = subparsers.add_parser("symbol", help="Find symbols")
    symbol_parser.add_argument("name")
    symbol_parser.add_argument("--kind", default=None)
    symbol_parser.add_argument("--path-prefix", default=None)
    symbol_parser.add_argument("--limit", type=int, default=None)

    refs_parser = subparsers.add_parser("refs", help="Find symbol references")
    refs_parser.add_argument("symbol")
    refs_parser.add_argument("--path-prefix", default=None)
    refs_parser.add_argument("--limit", type=int, default=None)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch a chunk or symbol")
    fetch_parser.add_argument("uri_or_id")

    map_parser = subparsers.add_parser("map", help="Print a compact repository map")
    map_parser.add_argument("--path-prefix", default=None)
    map_parser.add_argument("--limit", type=int, default=None)

    subparsers.add_parser("serve", help="Run MCP over stdio")

    args = parser.parse_args()
    if args.command == "serve":
        from .server import main as server_main

        server_main()
        return
    if args.command == "scan":
        _scan_command()
        return
    if args.command == "tune":
        _tune_command(args.sample_chunks, args.batch_sizes)
        return

    app = Application()
    try:
        if args.command == "index":
            paths = args.paths or None
            _print_json(
                app.reindex(
                    force=args.force,
                    paths=paths,
                    refresh_vectors=args.refresh_vectors,
                    enforce_mcp_policy=False,
                )
            )
        elif args.command == "status":
            _print_json(app.status(load_model=args.model, load_vectors=args.vectors))
        elif args.command == "doctor":
            _print_json(app.doctor(load_model=args.model))
        elif args.command == "warmup":
            _print_json(app.warmup(load_vectors=not args.no_vectors))
        elif args.command == "search":
            _print_json(
                app.code_search(
                    query=args.query,
                    top_k=args.top_k,
                    mode=args.mode,
                    path_prefix=args.path_prefix,
                    include_text=args.include_text,
                )
            )
        elif args.command == "symbol":
            _print_json(
                app.find_symbol(
                    name=args.name,
                    kind=args.kind,
                    path_prefix=args.path_prefix,
                    limit=args.limit,
                )
            )
        elif args.command == "refs":
            _print_json(
                app.references(
                    symbol=args.symbol,
                    path_prefix=args.path_prefix,
                    limit=args.limit,
                )
            )
        elif args.command == "fetch":
            value = args.uri_or_id
            if value.startswith("rag://") or value.startswith("symbol://"):
                _print_json(app.fetch(resource_uri=value))
            elif len(value) >= 20:
                try:
                    _print_json(app.fetch(symbol_id=value))
                except Exception:
                    _print_json(app.fetch(chunk_id=value))
            else:
                raise ValueError("provide a rag:// URI, symbol:// URI, chunk id or symbol id")
        elif args.command == "map":
            _print_json(app.repo_map(path_prefix=args.path_prefix, limit=args.limit))
    finally:
        app.close()


if __name__ == "__main__":
    main()
