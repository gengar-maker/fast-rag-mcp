from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    root: str
    path: str
    title: str
    section: str
    text: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    content_hash: str
    metadata: dict[str, Any]

    @property
    def resource_uri(self) -> str:
        return f"rag://{self.doc_id}#chunk={self.id}"

    @property
    def language(self) -> str:
        return str(self.metadata.get("language", ""))

    @property
    def chunk_type(self) -> str:
        return str(self.metadata.get("chunk_type", "text"))

    @property
    def symbol_id(self) -> str:
        return str(self.metadata.get("symbol_id", ""))

    @property
    def symbol_name(self) -> str:
        return str(self.metadata.get("symbol_name", ""))

    @property
    def qualified_name(self) -> str:
        return str(self.metadata.get("qualified_name", ""))


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    root: Path
    path: Path
    rel_path: str
    title: str
    text: str
    content_hash: str
    mtime_ns: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CodeSymbol:
    id: str
    doc_id: str
    root: str
    path: str
    language: str
    kind: str
    name: str
    qualified_name: str
    parent: str
    signature: str
    docstring: str
    text: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    metadata: dict[str, Any]

    @property
    def resource_uri(self) -> str:
        return f"symbol://{self.id}"


@dataclass(frozen=True, slots=True)
class CodeReference:
    id: str
    doc_id: str
    root: str
    path: str
    language: str
    source_symbol_id: str
    target_name: str
    ref_kind: str
    line: int
    snippet: str
    metadata: dict[str, Any]
