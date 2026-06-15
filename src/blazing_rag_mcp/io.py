from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator

from .config import Settings
from .types import Document

TEXT_EXTENSIONS = {
    ".md", ".mdx", ".txt", ".rst", ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs",
    ".java", ".kt", ".kts", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".json", ".yaml", ".yml",
    ".toml", ".sql", ".sh", ".bash", ".zsh", ".dockerfile", ".env", ".ini", ".cfg",
}


@dataclass(slots=True)
class ScanStats:
    roots: list[str] = field(default_factory=list)
    files_seen: int = 0
    files_yielded: int = 0
    dirs_skipped: int = 0
    files_skipped_size: int = 0
    files_skipped_pattern: int = 0
    errors: int = 0
    largest_files: list[tuple[int, str]] = field(default_factory=list)

    def add_largest(self, size: int, path: Path) -> None:
        self.largest_files.append((size, path.as_posix()))
        self.largest_files.sort(key=lambda x: x[0], reverse=True)
        del self.largest_files[20:]

    def as_dict(self) -> dict:
        return {
            "roots": self.roots,
            "files_seen": self.files_seen,
            "files_yielded": self.files_yielded,
            "dirs_skipped": self.dirs_skipped,
            "files_skipped_size": self.files_skipped_size,
            "files_skipped_pattern": self.files_skipped_pattern,
            "errors": self.errors,
            "largest_files": [{"bytes": n, "path": p} for n, p in self.largest_files],
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(*parts: str, n: int = 20) -> str:
    h = hashlib.blake2b("\x1f".join(parts).encode("utf-8", "ignore"), digest_size=16).hexdigest()
    return h[:n]


def is_binary_sample(data: bytes) -> bool:
    if b"\0" in data:
        return True
    if not data:
        return False
    non_text = sum(1 for b in data[:4096] if b < 9 or (13 < b < 32))
    return non_text / max(1, min(len(data), 4096)) > 0.08


def _matches_any(name: str, patterns: set[str]) -> bool:
    return name in patterns or any(("*" in pat or "?" in pat or "[" in pat) and fnmatch(name, pat) for pat in patterns)


def should_skip_dir(path: Path, settings: Settings) -> bool:
    parts = path.parts
    db_dir = settings.resolved_db_dir()
    try:
        path.resolve().relative_to(db_dir)
        return True
    except Exception:
        pass
    for part in parts:
        if _matches_any(part, settings.exclude_dirs):
            return True
        if settings.exclude_hidden_dirs and part.startswith(".") and part not in settings.include_hidden_dir_names:
            return True
    return False


def matches_include(rel_path: str, settings: Settings) -> bool:
    return any(fnmatch(rel_path, pat) for pat in settings.include_globs)


def iter_candidate_files(settings: Settings, stats: ScanStats | None = None) -> Iterator[Path]:
    """Yield candidate files without materializing the whole repo in memory.

    This also refuses to descend into the configured DB directory even if the user names it
    something other than `.brag`.
    """
    roots = settings.resolved_roots()
    if stats is not None:
        stats.roots = [p.as_posix() for p in roots]
    for root in roots:
        root = root.resolve()
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dpath = Path(dirpath)
            before = len(dirnames)
            kept_dirs: list[str] = []
            for d in dirnames:
                child = dpath / d
                if should_skip_dir(child, settings):
                    if stats is not None:
                        stats.dirs_skipped += 1
                    continue
                kept_dirs.append(d)
            dirnames[:] = kept_dirs
            if stats is not None:
                stats.dirs_skipped += max(0, before - len(kept_dirs))
            if should_skip_dir(dpath, settings):
                continue
            for name in sorted(filenames):
                path = dpath / name
                if stats is not None:
                    stats.files_seen += 1
                try:
                    rel = path.relative_to(root).as_posix()
                    st = path.stat()
                    if st.st_size > settings.max_file_bytes:
                        if stats is not None:
                            stats.files_skipped_size += 1
                            stats.add_largest(st.st_size, path)
                        continue
                    if not matches_include(rel, settings) and path.suffix.lower() not in TEXT_EXTENSIONS:
                        if stats is not None:
                            stats.files_skipped_pattern += 1
                        continue
                    if stats is not None:
                        stats.files_yielded += 1
                        stats.add_largest(st.st_size, path)
                    yield path
                    if stats is not None and stats.files_yielded >= settings.max_files:
                        return
                except OSError:
                    if stats is not None:
                        stats.errors += 1
                    continue


def read_document(root: Path, path: Path, settings: Settings) -> Document | None:
    try:
        stat = path.stat()
        if stat.st_size > settings.max_file_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if is_binary_sample(data):
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    content_hash = sha256_bytes(data)
    rel_path = path.relative_to(root).as_posix()
    doc_id = stable_id(root.as_posix(), rel_path)
    return Document(
        doc_id=doc_id,
        root=root,
        path=path,
        rel_path=rel_path,
        title=path.name,
        text=text,
        content_hash=content_hash,
        mtime_ns=stat.st_mtime_ns,
        size_bytes=stat.st_size,
    )
