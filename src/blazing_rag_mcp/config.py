from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    include_globs: Annotated[list[str], NoDecode] = Field(default_factory=lambda: [
        "**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst",
        "**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx",
        "**/*.go", "**/*.rs", "**/*.java", "**/*.kt", "**/*.kts",
        "**/*.c", "**/*.cc", "**/*.cpp", "**/*.cxx", "**/*.h", "**/*.hpp", "**/*.hh",
        "**/*.cs", "**/*.rb", "**/*.php", "**/*.swift", "**/*.scala",
        "**/*.json", "**/*.yaml", "**/*.yml", "**/*.toml",
        "**/*.sql", "**/*.sh", "**/*.bash", "**/*.zsh", "**/*.dockerfile", "**/Dockerfile",
    ])
    exclude_dirs: set[str] = Field(default_factory=lambda: {
        ".git", ".hg", ".svn",
        ".brag", ".rag", ".index", ".indexes",
        ".venv", "venv", "env", ".env", "virtualenv",
        "node_modules", "bower_components",
        "dist", "build", "out", "target", "coverage", "htmlcov",
        ".next", ".nuxt", ".svelte-kit", ".turbo", ".parcel-cache",
        ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox",
        ".uv", ".eggs", "*.egg-info",
        "__pycache__", ".ipynb_checkpoints",
        ".idea", ".vscode",
        "vendor", "third_party", "external",
        "site-packages", "dist-packages",
        "models", ".models", "checkpoints", ".checkpoints",
        "huggingface", ".huggingface", "transformers", ".transformers",
    })
    exclude_hidden_dirs: bool = True
    include_hidden_dir_names: set[str] = Field(default_factory=lambda: {".github"})
    max_file_bytes: int = 750_000
    max_files: int = 250_000
    scan_backend: Literal["auto", "git", "walk"] = "auto"
    fast_stat_skip: bool = True

    embedding_model: str = "nomic-ai/CodeRankEmbed"
    embedding_trust_remote_code: bool = True
    embedding_query_prefix: str = ""
    embedding_allow_hash_fallback: bool = True
    device: str = "auto"  # auto | cuda | mps | cpu
    normalize_embeddings: bool = True
    embedding_batch_size: int = 16
    embedding_flush_chunks: int = 512
    embedding_max_seq_length: int = 384
    embedding_precision: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    embedding_empty_cache_after_encode: bool = False
    embedding_reuse_unchanged_chunks: bool = True
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
    faiss_gpu: bool = True
    torch_vector_device: str = "auto"  # auto | cuda | mps | cpu
    torch_vector_dtype: Literal["auto", "float16", "float32"] = "auto"
    torch_vector_max_vectors: int = 1_000_000
    mps_vector_min_vectors: int = 30_000
    keep_cpu_vector_copy: bool = False
    vector_load_batch_size: int = 8192
    exact_search_threshold: int = 80_000
    fts_weight: float = 0.38
    dense_weight: float = 0.50
    rrf_k: int = 60

    default_top_k: int = 8
    max_top_k: int = 32
    candidate_k: int = 120
    max_snippet_chars: int = 700
    max_fetch_chars: int = 16_000
    exact_symbol_fast_path: bool = True

    read_only: bool = True
    allow_reindex_tool: bool = False

    @field_validator("roots", mode="before")
    @classmethod
    def parse_roots(cls, value: str | list[str]) -> list[str]:
        return _split_paths(value)

    @field_validator("include_globs", mode="before")
    @classmethod
    def parse_globs(cls, value: str | list[str]) -> list[str]:
        return _split_paths(value)

    def resolved_roots(self) -> list[Path]:
        roots = []
        for item in self.roots:
            p = Path(item).expanduser().resolve()
            if p.exists() and p.is_dir():
                roots.append(p)
        if not roots:
            roots.append(Path.cwd().resolve())
        return roots

    def resolved_db_dir(self) -> Path:
        return self.db_dir.expanduser().resolve()
