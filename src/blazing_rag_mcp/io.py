from __future__ import annotations

import hashlib
import os
import subprocess
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
    backends: dict[str, str] = field(default_factory=dict)
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
            "backends": self.backends,
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


def document_id(root: Path, path: Path) -> str:
    return stable_id(root.resolve().as_posix(), path.resolve().relative_to(root.resolve()).as_posix())


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
    db_dir = settings.resolved_db_dir()
    try:
        path.resolve().relative_to(db_dir)
        return True
    except Exception:
        pass
    name = path.name
    if _matches_any(name, settings.exclude_dirs):
        return True
    if settings.exclude_hidden_dirs and name.startswith(".") and name not in settings.include_hidden_dir_names:
        return True
    return False


def matches_include(rel_path: str, settings: Settings) -> bool:
    return any(fnmatch(rel_path, pat) for pat in settings.include_globs)


def _accept_file(root: Path, path: Path, settings: Settings, stats: ScanStats | None) -> bool:
    if stats is not None:
        stats.files_seen += 1
    try:
        rel = path.relative_to(root).as_posix()
        if any(_matches_any(part, settings.exclude_dirs) for part in Path(rel).parts[:-1]):
            if stats is not None:
                stats.files_skipped_pattern += 1
            return False
        st = path.stat()
        if not path.is_file() or st.st_size > settings.max_file_bytes:
            if stats is not None and st.st_size > settings.max_file_bytes:
                stats.files_skipped_size += 1
                stats.add_largest(st.st_size, path)
            return False
        if not matches_include(rel, settings) and path.suffix.lower() not in TEXT_EXTENSIONS:
            if stats is not None:
                stats.files_skipped_pattern += 1
            return False
        if stats is not None:
            stats.files_yielded += 1
            stats.add_largest(st.st_size, path)
        return True
    except OSError:
        if stats is not None:
            stats.errors += 1
        return False


def _git_paths(root: Path) -> Iterator[Path]:
    proc = subprocess.run(
        ["git", "-C", root.as_posix(), "ls-files", "-co", "--exclude-standard", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=30,
    )
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", "surrogateescape")
        yield root / rel


def _can_use_git(root: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", root.as_posix(), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
            text=True,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def iter_candidate_files_with_roots(
    settings: Settings,
    stats: ScanStats | None = None,
) -> Iterator[tuple[Path, Path]]:
    """Yield ``(root, path)`` pairs without materializing the repository.

    Git repositories use ``git ls-files -co --exclude-standard`` by default. This is much faster
    than a recursive Python walk for large repositories and automatically respects .gitignore.
    """
    roots = settings.resolved_roots()
    if stats is not None:
        stats.roots = [p.as_posix() for p in roots]
    yielded = 0
    seen: set[str] = set()

    for root in roots:
        root = root.resolve()
        use_git = settings.scan_backend == "git" or (
            settings.scan_backend == "auto" and _can_use_git(root)
        )
        if use_git:
            try:
                iterator = _git_paths(root)
                if stats is not None:
                    stats.backends[root.as_posix()] = "git"
                for path in iterator:
                    key = path.resolve().as_posix()
                    if key in seen:
                        continue
                    seen.add(key)
                    if _accept_file(root, path, settings, stats):
                        yield root, path
                        yielded += 1
                        if yielded >= settings.max_files:
                            return
                continue
            except (OSError, subprocess.SubprocessError):
                if settings.scan_backend == "git":
                    raise

        if stats is not None:
            stats.backends[root.as_posix()] = "walk"
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dpath = Path(dirpath)
            kept_dirs: list[str] = []
            for dirname in dirnames:
                child = dpath / dirname
                if should_skip_dir(child, settings):
                    if stats is not None:
                        stats.dirs_skipped += 1
                    continue
                kept_dirs.append(dirname)
            dirnames[:] = kept_dirs
            if should_skip_dir(dpath, settings):
                continue
            for name in sorted(filenames):
                path = dpath / name
                key = path.resolve().as_posix()
                if key in seen:
                    continue
                seen.add(key)
                if _accept_file(root, path, settings, stats):
                    yield root, path
                    yielded += 1
                    if yielded >= settings.max_files:
                        return


def iter_candidate_files(settings: Settings, stats: ScanStats | None = None) -> Iterator[Path]:
    for _, path in iter_candidate_files_with_roots(settings, stats):
        yield path


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
    return Document(
        doc_id=document_id(root, path),
        root=root,
        path=path,
        rel_path=rel_path,
        title=path.name,
        text=text,
        content_hash=content_hash,
        mtime_ns=stat.st_mtime_ns,
        size_bytes=stat.st_size,
    )
