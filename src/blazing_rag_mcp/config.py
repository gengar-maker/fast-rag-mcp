from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

INDEX_FORMAT_VERSION = 3


def _split_paths(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    raw = value.replace("\n", os.pathsep).replace(",", os.pathsep)
    return [p.strip() for p in raw.split(os.pathsep) if p.strip()]


class Settings(BaseSettings):
    """Runtime configuration, primarily via BRAG_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="BRAG_", env_file=".env", extra="ignore")

    db_dir: Path = Field(default=Path(".brag"), description="Index/docstore directory")
    roots: Annotated[list[str], NoDecode] = Field(default_factory=lambda: [os.getcwd()])
    include_globs: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "**/*.md",
            "**/*.mdx",
            "**/*.txt",
            "**/*.rst",
            "**/*.pdf",
            "**/*.py",
            "**/*.ts",
            "**/*.tsx",
            "**/*.js",
            "**/*.jsx",
            "**/*.go",
            "**/*.rs",
            "**/*.java",
            "**/*.kt",
            "**/*.kts",
            "**/*.c",
            "**/*.cc",
            "**/*.cpp",
            "**/*.cxx",
            "**/*.cu",
            "**/*.cuh",
            "**/*.h",
            "**/*.hpp",
            "**/*.hh",
            "**/*.cs",
            "**/*.rb",
            "**/*.php",
            "**/*.swift",
            "**/*.scala",
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/*.toml",
            "**/*.sql",
            "**/*.sh",
            "**/*.bash",
            "**/*.zsh",
            "**/*.dockerfile",
            "**/Dockerfile",
        ]
    )
    exclude_file_globs: set[str] = Field(
        default_factory=lambda: {
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "id_rsa",
            "id_rsa.*",
            "id_ed25519",
            "id_ed25519.*",
            ".npmrc",
            ".pypirc",
            "credentials",
            "credentials.*",
            "secrets.*",
        }
    )
    exclude_dirs: set[str] = Field(
        default_factory=lambda: {
            ".git",
            ".hg",
            ".svn",
            ".brag",
            ".rag",
            ".index",
            ".indexes",
            ".venv",
            "venv",
            "env",
            ".env",
            "virtualenv",
            "node_modules",
            "bower_components",
            "dist",
            "build",
            "out",
            "target",
            "coverage",
            "htmlcov",
            ".next",
            ".nuxt",
            ".svelte-kit",
            ".turbo",
            ".parcel-cache",
            ".cache",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".nox",
            ".uv",
            ".eggs",
            "*.egg-info",
            "__pycache__",
            ".ipynb_checkpoints",
            ".idea",
            ".vscode",
            "vendor",
            "contrib",
            "third_party",
            "external",
            "site-packages",
            "dist-packages",
            "models",
            ".models",
            "checkpoints",
            ".checkpoints",
            "huggingface",
            ".huggingface",
            "transformers",
            ".transformers",
        }
    )
    exclude_hidden_dirs: bool = True
    include_hidden_dir_names: set[str] = Field(default_factory=lambda: {".github"})
    max_file_bytes: int = 750_000
    max_pdf_bytes: int = 200_000_000
    pdf_max_pages: int = 2_000
    pdf_max_chars: int = 20_000_000
    pdf_chunk_tokens: int = 512
    pdf_chunk_overlap: int = 48
    pdf_ocr_mode: Literal["off", "auto", "always"] = "off"
    pdf_ocr_language: str = "eng"
    pdf_ocr_dpi: int = 200
    pdf_ocr_min_chars: int = 48
    pdf_extract_tables: bool = False
    pdf_password: str = ""
    pdf_dense_candidate_multiplier: int = 8
    pdf_query_prefix: str = "Represent this query for retrieving relevant technical documentation:"
    max_files: int = 250_000
    max_targeted_paths: int = 256
    scan_backend: Literal["auto", "git", "walk"] = "auto"
    fast_stat_skip: bool = True
    delete_missing_on_incomplete_scan: bool = False

    embedding_model: str = "nomic-ai/CodeRankEmbed"
    embedding_cache_dir: Path | None = None
    embedding_revision: str = ""
    embedding_trust_remote_code: bool = True
    embedding_local_files_only: bool = False
    embedding_query_prefix: str = ""
    embedding_allow_hash_fallback: bool = False
    device: str = "auto"  # auto | cuda | mps | cpu
    normalize_embeddings: bool = True
    embedding_batch_size: int = 16
    embedding_flush_chunks: int = 512
    embedding_max_seq_length: int = 384
    embedding_precision: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    embedding_empty_cache_after_encode: bool = False
    embedding_reuse_unchanged_chunks: bool = True
    mps_enable_fallback: bool = False
    tokenizer_parallelism: bool = True
    tokenizer_threads: int = 8
    query_cache_size: int = 4096

    chunk_tokens: int = 384
    chunk_overlap: int = 32
    min_chunk_chars: int = 80
    add_file_summary_chunks: bool = False
    code_symbol_weight: float = 0.82
    code_path_weight: float = 0.55
    code_reference_limit: int = 80
    code_map_limit: int = 240

    vector_backend: Literal["auto", "faiss", "torch", "numpy"] = "auto"
    vector_storage_dtype: Literal["float16", "float32"] = "float16"
    faiss_gpu: bool = False
    torch_vector_device: str = "auto"  # auto | cuda | mps | cpu
    torch_vector_dtype: Literal["auto", "float16", "float32"] = "auto"
    torch_vector_max_vectors: int = 1_000_000
    mps_vector_min_vectors: int = 30_000
    keep_cpu_vector_copy: bool = False
    vector_load_batch_size: int = 8192
    exact_search_threshold: int = 80_000
    defer_vector_reload_after_reindex: bool = True
    fts_weight: float = 0.38
    dense_weight: float = 0.50
    rrf_k: int = 60

    sqlite_timeout_seconds: float = 30.0
    sqlite_busy_timeout_ms: int = 30_000
    sqlite_cache_kib: int = 131_072
    sqlite_mmap_bytes: int = 268_435_456

    default_top_k: int = 8
    max_top_k: int = 32
    candidate_k: int = 120
    max_snippet_chars: int = 700
    max_fetch_chars: int = 16_000
    exact_symbol_fast_path: bool = True

    read_only: bool = True
    allow_reindex_tool: bool = False

    @field_validator(
        "max_file_bytes",
        "max_pdf_bytes",
        "pdf_max_pages",
        "pdf_max_chars",
        "pdf_chunk_tokens",
        "pdf_ocr_dpi",
        "pdf_ocr_min_chars",
        "pdf_dense_candidate_multiplier",
        "max_files",
        "max_targeted_paths",
        "embedding_batch_size",
        "embedding_flush_chunks",
        "embedding_max_seq_length",
        "tokenizer_threads",
        "query_cache_size",
        "chunk_tokens",
        "min_chunk_chars",
        "code_reference_limit",
        "code_map_limit",
        "torch_vector_max_vectors",
        "mps_vector_min_vectors",
        "vector_load_batch_size",
        "exact_search_threshold",
        "sqlite_busy_timeout_ms",
        "sqlite_cache_kib",
        "default_top_k",
        "max_top_k",
        "candidate_k",
        "max_snippet_chars",
        "max_fetch_chars",
    )
    @classmethod
    def positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value

    @field_validator("chunk_overlap", "pdf_chunk_overlap")
    @classmethod
    def non_negative_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_relationships(self) -> Settings:
        if self.max_top_k < self.default_top_k:
            raise ValueError("max_top_k must be greater than or equal to default_top_k")
        if self.candidate_k < self.max_top_k:
            raise ValueError("candidate_k must be greater than or equal to max_top_k")
        if self.embedding_flush_chunks < self.embedding_batch_size:
            raise ValueError(
                "embedding_flush_chunks must be greater than or equal to embedding_batch_size"
            )
        return self

    @field_validator("roots", mode="before")
    @classmethod
    def parse_roots(cls, value: str | list[str]) -> list[str]:
        return _split_paths(value)

    @field_validator("include_globs", mode="before")
    @classmethod
    def parse_globs(cls, value: str | list[str]) -> list[str]:
        return _split_paths(value)

    def resolved_roots(self) -> list[Path]:
        roots: list[Path] = []
        for item in self.roots:
            path = Path(item).expanduser().resolve()
            if path.exists() and path.is_dir():
                roots.append(path)
        if not roots:
            configured = ", ".join(self.roots) or "<empty>"
            raise ValueError(f"no configured BRAG_ROOTS exist or are directories: {configured}")
        db_dir = self.resolved_db_dir()
        if any(root == db_dir for root in roots):
            raise ValueError("BRAG_DB_DIR must not be identical to a BRAG_ROOTS directory")
        return roots

    def resolved_db_dir(self) -> Path:
        return self.db_dir.expanduser().resolve()

    def index_fingerprint(self) -> str:
        """Fingerprint settings that affect persisted chunks or embedding compatibility."""
        payload = {
            "format": INDEX_FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "embedding_trust_remote_code": self.embedding_trust_remote_code,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_max_seq_length": self.embedding_max_seq_length,
            "embedding_precision": self.embedding_precision,
            "mps_enable_fallback": self.mps_enable_fallback,
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_chars": self.min_chunk_chars,
            "add_file_summary_chunks": self.add_file_summary_chunks,
            "vector_storage_dtype": self.vector_storage_dtype,
            "pdf_extractor": "pymupdf-v1",
            "pdf_chunk_tokens": self.pdf_chunk_tokens,
            "pdf_chunk_overlap": self.pdf_chunk_overlap,
            "pdf_ocr_mode": self.pdf_ocr_mode,
            "pdf_ocr_language": self.pdf_ocr_language,
            "pdf_ocr_dpi": self.pdf_ocr_dpi,
            "pdf_extract_tables": self.pdf_extract_tables,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.blake2b(raw, digest_size=20).hexdigest()
