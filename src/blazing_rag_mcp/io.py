from __future__ import annotations

import hashlib
import os
import shutil
import subprocess  # nosec B404
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from .config import Settings
from .types import Document

GIT_EXECUTABLE = shutil.which("git")


TEXT_EXTENSIONS = {
    ".md",
    ".mdx",
    ".txt",
    ".rst",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".kts",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".dockerfile",
    ".ini",
    ".cfg",
    ".pdf",
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
    truncated: bool = False
    largest_files: list[tuple[int, str]] = field(default_factory=list)

    def add_largest(self, size: int, path: Path) -> None:
        self.largest_files.append((size, path.as_posix()))
        self.largest_files.sort(key=lambda item: item[0], reverse=True)
        del self.largest_files[20:]

    @property
    def complete(self) -> bool:
        return not self.truncated and self.errors == 0

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
            "truncated": self.truncated,
            "complete": self.complete,
            "largest_files": [{"bytes": size, "path": path} for size, path in self.largest_files],
        }


@dataclass(slots=True)
class RequestedPaths:
    existing: list[tuple[Path, Path]] = field(default_factory=list)
    missing: list[tuple[Path, Path]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(*parts: str, n: int = 20) -> str:
    digest = hashlib.blake2b(
        "\x1f".join(parts).encode("utf-8", "ignore"), digest_size=16
    ).hexdigest()
    return digest[:n]


def document_id(root: Path, path: Path) -> str:
    root_resolved = root.resolve()
    path_resolved = path.resolve(strict=False)
    relative = path_resolved.relative_to(root_resolved).as_posix()
    return stable_id(root_resolved.as_posix(), relative)


def is_binary_sample(data: bytes) -> bool:
    if b"\0" in data:
        return True
    if not data:
        return False
    sample = data[:4096]
    non_text = sum(1 for byte in sample if byte < 9 or 13 < byte < 32)
    return non_text / max(1, len(sample)) > 0.08


def _matches_any(name: str, patterns: set[str]) -> bool:
    return name in patterns or any(
        ("*" in pattern or "?" in pattern or "[" in pattern) and fnmatch(name, pattern)
        for pattern in patterns
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def should_skip_dir(path: Path, settings: Settings) -> bool:
    db_dir = settings.resolved_db_dir()
    if _is_within(path, db_dir):
        return True
    name = path.name
    if _matches_any(name, settings.exclude_dirs):
        return True
    return bool(
        settings.exclude_hidden_dirs
        and name.startswith(".")
        and name not in settings.include_hidden_dir_names
    )


def matches_include(relative_path: str, settings: Settings) -> bool:
    return any(fnmatch(relative_path, pattern) for pattern in settings.include_globs)


def _accept_file(root: Path, path: Path, settings: Settings, stats: ScanStats | None) -> bool:
    if stats is not None:
        stats.files_seen += 1
    try:
        if not _is_within(path, root):
            if stats is not None:
                stats.files_skipped_pattern += 1
            return False
        relative = path.resolve(strict=False).relative_to(root.resolve()).as_posix()
        if any(
            fnmatch(path.name, pattern) or fnmatch(relative, pattern)
            for pattern in settings.exclude_file_globs
        ):
            if stats is not None:
                stats.files_skipped_pattern += 1
            return False
        parts = Path(relative).parts
        if any(_matches_any(part, settings.exclude_dirs) for part in parts[:-1]):
            if stats is not None:
                stats.files_skipped_pattern += 1
            return False
        stat = path.stat()
        size_limit = (
            settings.max_pdf_bytes if path.suffix.lower() == ".pdf" else settings.max_file_bytes
        )
        if not path.is_file() or stat.st_size > size_limit:
            if stats is not None and stat.st_size > size_limit:
                stats.files_skipped_size += 1
                stats.add_largest(stat.st_size, path)
            return False
        if not matches_include(relative, settings) and path.suffix.lower() not in TEXT_EXTENSIONS:
            if stats is not None:
                stats.files_skipped_pattern += 1
            return False
        if stats is not None:
            stats.files_yielded += 1
            stats.add_largest(stat.st_size, path)
        return True
    except OSError:
        if stats is not None:
            stats.errors += 1
        return False


def _iter_null_delimited(stream) -> Iterator[bytes]:
    buffer = b""
    while True:
        chunk = stream.read(65_536)
        if not chunk:
            break
        buffer += chunk
        parts = buffer.split(b"\0")
        buffer = parts.pop()
        yield from (part for part in parts if part)
    if buffer:
        yield buffer


def _git_paths(root: Path) -> Iterator[Path]:
    if not GIT_EXECUTABLE:
        raise FileNotFoundError("git executable was not found")
    process = subprocess.Popen(  # nosec B603
        [GIT_EXECUTABLE, "-C", root.as_posix(), "ls-files", "-co", "--exclude-standard", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("failed to capture git ls-files output")
    try:
        for raw in _iter_null_delimited(process.stdout):
            relative = raw.decode("utf-8", "surrogateescape")
            yield root / relative
    finally:
        process.stdout.close()
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            raise
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)


def _can_use_git(root: Path) -> bool:
    if not GIT_EXECUTABLE:
        return False
    try:
        process = subprocess.run(  # nosec B603
            [GIT_EXECUTABLE, "-C", root.as_posix(), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
            text=True,
        )
        return process.returncode == 0 and process.stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def _walk_root(root: Path, settings: Settings, stats: ScanStats | None) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        kept: list[str] = []
        for dirname in dirnames:
            child = directory / dirname
            if should_skip_dir(child, settings):
                if stats is not None:
                    stats.dirs_skipped += 1
                continue
            kept.append(dirname)
        dirnames[:] = kept
        if should_skip_dir(directory, settings):
            continue
        for name in sorted(filenames):
            yield directory / name


def iter_candidate_files_with_roots(
    settings: Settings,
    stats: ScanStats | None = None,
) -> Iterator[tuple[Path, Path]]:
    """Yield candidate ``(root, path)`` pairs without materializing the repository."""
    roots = settings.resolved_roots()
    if stats is not None:
        stats.roots = [path.as_posix() for path in roots]
    yielded = 0
    seen: set[str] = set()

    for root in roots:
        root = root.resolve()
        use_git = settings.scan_backend == "git" or (
            settings.scan_backend == "auto" and _can_use_git(root)
        )
        iterator: Iterator[Path]
        if use_git:
            try:
                iterator = _git_paths(root)
                if stats is not None:
                    stats.backends[root.as_posix()] = "git"
                for path in iterator:
                    key = path.absolute().as_posix()
                    if key in seen:
                        continue
                    seen.add(key)
                    if _accept_file(root, path, settings, stats):
                        yield root, path
                        yielded += 1
                        if yielded >= settings.max_files:
                            if stats is not None:
                                stats.truncated = True
                            return
                continue
            except (OSError, subprocess.SubprocessError):
                if settings.scan_backend == "git":
                    raise

        if stats is not None:
            stats.backends[root.as_posix()] = "walk"
        for path in _walk_root(root, settings, stats):
            key = path.absolute().as_posix()
            if key in seen:
                continue
            seen.add(key)
            if _accept_file(root, path, settings, stats):
                yield root, path
                yielded += 1
                if yielded >= settings.max_files:
                    if stats is not None:
                        stats.truncated = True
                    return


def iter_candidate_files(settings: Settings, stats: ScanStats | None = None) -> Iterator[Path]:
    for _, path in iter_candidate_files_with_roots(settings, stats):
        yield path


def resolve_requested_paths(settings: Settings, requested: Sequence[str]) -> RequestedPaths:
    """Resolve user-provided files/directories under configured roots.

    Missing files are retained so their existing index records can be removed. Paths outside all
    configured roots are rejected. Directories are expanded using the same include/exclude policy.
    """
    if len(requested) > settings.max_targeted_paths:
        raise ValueError(
            f"too many targeted paths: {len(requested)} > {settings.max_targeted_paths}"
        )
    roots = settings.resolved_roots()
    result = RequestedPaths()
    seen: set[str] = set()

    for raw in requested:
        raw_path = Path(raw).expanduser()
        candidates = [raw_path] if raw_path.is_absolute() else [root / raw_path for root in roots]
        chosen: tuple[Path, Path] | None = None
        for candidate in candidates:
            for root in roots:
                if _is_within(candidate, root):
                    chosen = (root.resolve(), candidate.resolve(strict=False))
                    if candidate.exists():
                        break
            if chosen is not None and candidate.exists():
                break
        if chosen is None:
            result.rejected.append(raw)
            continue
        root, path = chosen
        key = f"{root}\x1f{path}"
        if key in seen:
            continue
        seen.add(key)

        if not path.exists():
            result.missing.append((root, path))
            continue
        if path.is_dir():
            for child in _walk_root(path, settings, None):
                child_key = f"{root}\x1f{child.absolute()}"
                if child_key in seen:
                    continue
                seen.add(child_key)
                if _accept_file(root, child, settings, None):
                    result.existing.append((root, child))
            continue
        if _accept_file(root, path, settings, None):
            result.existing.append((root, path))
        else:
            result.rejected.append(raw)
    return result


def read_document(root: Path, path: Path, settings: Settings) -> Document | None:
    if path.suffix.lower() == ".pdf":
        from .pdf_ingest import read_pdf_document

        return read_pdf_document(root, path, settings)
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
    relative_path = path.resolve(strict=False).relative_to(root.resolve()).as_posix()
    return Document(
        doc_id=document_id(root, path),
        root=root,
        path=path,
        rel_path=relative_path,
        title=path.name,
        text=text,
        content_hash=content_hash,
        mtime_ns=stat.st_mtime_ns,
        size_bytes=stat.st_size,
    )
