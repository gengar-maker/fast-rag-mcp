from __future__ import annotations

import hashlib
import logging
import os
import sys
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from importlib import metadata

import numpy as np
from cachetools import LRUCache

from .config import Settings

log = logging.getLogger(__name__)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def embedding_environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": _package_version("torch"),
        "sentence_transformers": _package_version("sentence-transformers"),
        "transformers": _package_version("transformers"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "einops": _package_version("einops"),
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "SENTENCE_TRANSFORMERS_HOME": os.environ.get("SENTENCE_TRANSFORMERS_HOME", ""),
    }


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception as exc:
        log.debug("torch device detection failed: %s", exc)
    return "cpu"


@dataclass(slots=True)
class EmbeddingInfo:
    model: str
    device: str
    dim: int
    normalized: bool
    provider: str
    max_seq_length: int
    precision: str
    batch_size: int = 0


class EmbeddingModel:
    """GPU-aware embedding wrapper with bounded memory and persistent-compatible fingerprints."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.mps_enable_fallback:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.device = resolve_device(settings.device)
        self._model = None
        self._provider = "hash-fallback"
        self._dim = 384
        self._precision = "none"
        self._max_seq_length = int(settings.embedding_max_seq_length)
        self._effective_batch_size = max(1, int(settings.embedding_batch_size))
        self._cache: LRUCache[str, np.ndarray] = LRUCache(maxsize=settings.query_cache_size)

        os.environ["TOKENIZERS_PARALLELISM"] = "true" if settings.tokenizer_parallelism else "false"
        if settings.tokenizer_threads > 0:
            os.environ.setdefault("RAYON_NUM_THREADS", str(settings.tokenizer_threads))
        self._load_sentence_transformer()

    def _desired_torch_dtype(self):
        precision = self.settings.embedding_precision
        if precision == "float32":
            return None, "float32"
        try:
            import torch
        except Exception:
            return None, "none"
        if precision == "float16" or (precision == "auto" and self.device in {"cuda", "mps"}):
            return torch.float16, "float16"
        if precision == "bfloat16":
            return torch.bfloat16, "bfloat16"
        return None, "float32"

    def _load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            dtype, precision_name = self._desired_torch_dtype()
            kwargs = {}
            if dtype is not None:
                kwargs["model_kwargs"] = {"torch_dtype": dtype}
            cache_folder = None
            if self.settings.embedding_cache_dir is not None:
                cache_path = self.settings.embedding_cache_dir.expanduser().resolve()
                if self.settings.embedding_local_files_only and not cache_path.exists():
                    raise FileNotFoundError(
                        f"embedding cache directory does not exist: {cache_path}"
                    )
                cache_path.mkdir(parents=True, exist_ok=True)
                cache_folder = str(cache_path)

            installed_transformers = _package_version("transformers")
            if (
                self.settings.embedding_model.lower().endswith("coderankembed")
                and installed_transformers != "not-installed"
                and int(installed_transformers.split(".", 1)[0]) >= 5
            ):
                raise RuntimeError(
                    "CodeRankEmbed custom model code requires Transformers 4.x; "
                    f"installed transformers={installed_transformers}"
                )

            model = SentenceTransformer(
                self.settings.embedding_model,
                revision=self.settings.embedding_revision or None,
                device=self.device,
                trust_remote_code=self.settings.embedding_trust_remote_code,
                local_files_only=self.settings.embedding_local_files_only,
                cache_folder=cache_folder,
                **kwargs,
            )
            self._model = model
            try:
                current = int(
                    getattr(model, "max_seq_length", self._max_seq_length) or self._max_seq_length
                )
                model.max_seq_length = min(current, self._max_seq_length)
            except Exception as exc:
                log.debug("could not cap model max_seq_length: %s", exc)
            with suppress(Exception):
                model.eval()
            dim = model.get_sentence_embedding_dimension()
            if dim:
                self._dim = int(dim)
            self._provider = "sentence-transformers"
            self._precision = precision_name
            self._max_seq_length = int(
                getattr(model, "max_seq_length", self._max_seq_length) or self._max_seq_length
            )
            log.info(
                "loaded embedding model=%s device=%s dim=%s precision=%s max_seq_length=%s batch_size=%s tokenizer_threads=%s",
                self.settings.embedding_model,
                self.device,
                self._dim,
                self._precision,
                self._max_seq_length,
                self._effective_batch_size,
                self.settings.tokenizer_threads,
            )
        except Exception as exc:
            if not self.settings.embedding_allow_hash_fallback:
                env = embedding_environment()
                hints: list[str] = []
                if env["sentence_transformers"] == "not-installed":
                    hints.append(
                        "install embedding dependencies with `uv sync --extra mac-metal --extra code`"
                    )
                if self.settings.embedding_local_files_only:
                    hints.append(
                        "the server is in local-files-only mode; prefetch with "
                        "`BRAG_EMBEDDING_LOCAL_FILES_ONLY=false brag warmup --no-vectors` "
                        "using the same HF_HOME/cache and model revision"
                    )
                if env["transformers"].split(".", 1)[0] == "5":
                    hints.append(
                        "CodeRankEmbed is incompatible with Transformers 5.x; install the pinned 4.47.1 release"
                    )
                details = "; ".join(f"{key}={value}" for key, value in env.items())
                hint_text = f" Hints: {'; '.join(hints)}." if hints else ""
                raise RuntimeError(
                    f"failed to load embedding model {self.settings.embedding_model!r}: "
                    f"{type(exc).__name__}: {exc}.{hint_text} Environment: {details}"
                ) from exc
            log.warning("sentence-transformers unavailable; using hash fallback: %s", exc)

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            model=self.settings.embedding_model,
            device=self.device,
            dim=self._dim,
            normalized=self.settings.normalize_embeddings,
            provider=self._provider,
            max_seq_length=self._max_seq_length,
            precision=self._precision,
            batch_size=self._effective_batch_size,
        )

    @property
    def index_key(self) -> str:
        return "|".join(
            [
                self.settings.embedding_model,
                self.settings.embedding_revision or "default",
                self._provider,
                self.device,
                str(self._dim),
                str(self.settings.normalize_embeddings),
                str(self._max_seq_length),
                self._precision,
            ]
        )

    def embedding_hash(self, text: str) -> str:
        return hashlib.blake2b(
            f"{self.index_key}\x1f{text}".encode("utf-8", "ignore"),
            digest_size=20,
        ).hexdigest()

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if not self.settings.normalize_embeddings:
            return arr.astype("float32", copy=False)
        denom = np.linalg.norm(arr, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        return (arr / denom).astype("float32", copy=False)

    def _empty_device_cache(self, *, force: bool = False) -> None:
        if not force and not self.settings.embedding_empty_cache_after_encode:
            return
        try:
            import torch

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception as exc:
            log.debug("device cache cleanup failed: %s", exc)

    def embed_texts(
        self, texts: list[str], *, is_query: bool = False, query_prefix: str | None = None
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        if is_query:
            prefix = (
                query_prefix if query_prefix is not None else self.settings.embedding_query_prefix
            ).strip()
            if not prefix and self.settings.embedding_model.lower().endswith("coderankembed"):
                prefix = "Represent this query for searching relevant code:"
            if prefix:
                texts = [f"{prefix} {text}" for text in texts]

        if self._model is not None:
            try:
                import torch

                context_factory = torch.inference_mode
            except Exception:
                context_factory = _NullContext

            batch_size = self._effective_batch_size
            while True:
                try:
                    with context_factory():
                        if is_query:
                            encoder = getattr(self._model, "encode_query", self._model.encode)
                        else:
                            encoder = getattr(self._model, "encode_document", self._model.encode)
                        arr = encoder(
                            texts,
                            batch_size=batch_size,
                            normalize_embeddings=self.settings.normalize_embeddings,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )
                    self._effective_batch_size = batch_size
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    allocation_failure = any(
                        token in message
                        for token in (
                            "out of memory",
                            "mps backend out of memory",
                            "failed to allocate",
                            "allocation failed",
                        )
                    )
                    if batch_size <= 1 or not allocation_failure:
                        raise
                    batch_size = max(1, batch_size // 2)
                    self._effective_batch_size = batch_size
                    log.warning(
                        "embedding allocation failure; retrying with batch_size=%s", batch_size
                    )
                    self._empty_device_cache(force=True)
            self._empty_device_cache()
            return np.asarray(arr, dtype="float32")
        return self._hash_embed(texts)

    def embed_query(self, query: str, *, query_prefix: str | None = None) -> np.ndarray:
        key = hashlib.blake2b(
            f"{self.index_key}\x1fquery\x1f{query_prefix or ''}\x1f{query}".encode(),
            digest_size=16,
        ).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        arr = self.embed_texts([query], is_query=True, query_prefix=query_prefix)[0]
        self._cache[key] = arr
        return arr

    def warmup(self) -> None:
        self.embed_texts(["def warmup(value):\n    return value"], is_query=False)
        self.embed_query("warmup function")

    def benchmark_batch_sizes(self, texts: list[str], candidates: list[int]) -> dict:
        import time

        if not texts:
            raise ValueError("no benchmark texts")
        original = self._effective_batch_size
        results: list[dict] = []
        best_batch = original
        best_rate = 0.0
        # Warm kernels/tokenizer once before measuring.
        self.embed_texts(texts[: min(8, len(texts))])
        for candidate in candidates:
            if candidate <= 0:
                continue
            self._effective_batch_size = candidate
            started = time.perf_counter()
            try:
                self.embed_texts(texts)
                elapsed = time.perf_counter() - started
                rate = len(texts) / elapsed if elapsed else 0.0
                results.append(
                    {
                        "batch_size": candidate,
                        "elapsed_ms": round(elapsed * 1000, 3),
                        "texts_per_second": round(rate, 3),
                        "ok": True,
                    }
                )
                if rate > best_rate:
                    best_rate = rate
                    best_batch = self._effective_batch_size
            except Exception as exc:
                results.append(
                    {
                        "batch_size": candidate,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                self._empty_device_cache(force=True)
        self._effective_batch_size = best_batch
        return {
            "sample_texts": len(texts),
            "results": results,
            "recommended_batch_size": best_batch,
            "recommended_rate": round(best_rate, 3),
            "previous_batch_size": original,
        }

    def _hash_embed(self, texts: Iterable[str]) -> np.ndarray:
        if not isinstance(texts, list):
            texts = list(texts)
        vectors = np.zeros((len(texts), self._dim), dtype="float32")
        for row, text in enumerate(texts):
            padded = f"  {text.lower()}  "
            for i in range(max(1, len(padded) - 2)):
                gram = padded[i : i + 3]
                h = int.from_bytes(
                    hashlib.blake2b(gram.encode("utf-8", "ignore"), digest_size=8).digest(),
                    "little",
                )
                idx = h % self._dim
                sign = 1.0 if (h >> 63) == 0 else -1.0
                vectors[row, idx] += sign
        return self._normalize(vectors)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
