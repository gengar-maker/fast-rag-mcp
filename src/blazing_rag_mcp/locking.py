from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO


class IndexMutationLock:
    """Non-blocking inter-process lock for index writers.

    SQLite serializes individual transactions, but an index update consists of many transactions
    plus corpus/version state changes. This lock prevents a CLI indexer and MCP indexer from
    interleaving one logical mutation. OS advisory locks are released automatically on process exit.
    """

    def __init__(self, db_dir: Path):
        self.path = db_dir.expanduser().resolve() / ".index.lock"
        self._file: BinaryIO | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == "nt":
                import msvcrt

                handle.seek(0)
                if handle.read(1) == b"":
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    handle.fileno(), msvcrt.LK_NBLCK, 1  # type: ignore[attr-defined]
                )
            else:
                handle.close()
                raise RuntimeError(f"unsupported platform for index lock: {os.name}")
        except (BlockingIOError, OSError):
            handle.close()
            return False
        self._file = handle
        return True

    def release(self) -> None:
        handle = self._file
        if handle is None:
            return
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(  # type: ignore[attr-defined]
                    handle.fileno(), msvcrt.LK_UNLCK, 1  # type: ignore[attr-defined]
                )
        finally:
            handle.close()
            self._file = None

    def __enter__(self) -> IndexMutationLock:
        if not self.acquire():
            raise RuntimeError(f"another index writer holds {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
