from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from cachetools import LRUCache

from .config import Settings

log = logging.getLogger(__name__)

# These must be set before torch initializes MPS/CUDA to be maximally effective. Users can
# override them in their shell. They are intentionally conservative for indexing workflows.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _has_torch_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
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


class EmbeddingModel:
    """GPU-aware embedding wrapper with a deterministic fallback.

    Memory defaults are intentionally conservative. On Apple Silicon, Activity Monitor reports
    PyTorch MPS unified-memory allocations under the Python process, so large embedding batches can
    look like Python RSS explosions. Keep batch size and max sequence length bounded.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.device)
        self._model = None
        self._provider = "hash-fallback"
        self._dim = 384
        self._precision = "none"
        self._max_seq_length = int(settings.embedding_max_seq_length)
        self._cache: LRUCache[str, np.ndarray] = LRUCache(maxsize=settings.query_cache_size)
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
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.device,
                trust_remote_code=self.settings.embedding_trust_remote_code,
                **kwargs,
            )
            # Hard cap the transformer sequence length. This is the main protection against MPS/CUDA
            # activation-memory spikes during indexing.
            try:
                current = int(getattr(self._model, "max_seq_length", self._max_seq_length) or self._max_seq_length)
                self._model.max_seq_length = min(current, self._max_seq_length)
            except Exception:
                pass
            try:
                self._model.eval()
            except Exception:
                pass
            dim = self._model.get_sentence_embedding_dimension()
            if dim:
                self._dim = int(dim)
            self._provider = "sentence-transformers"
            self._precision = precision_name
            self._max_seq_length = int(getattr(self._model, "max_seq_length", self._max_seq_length) or self._max_seq_length)
            log.warning(
                "loaded embedding model=%s device=%s dim=%s precision=%s max_seq_length=%s batch_size=%s",
                self.settings.embedding_model,
                self.device,
                self._dim,
                self._precision,
                self._max_seq_length,
                self.settings.embedding_batch_size,
            )
        except Exception as exc:
            if not self.settings.embedding_allow_hash_fallback:
                raise RuntimeError(
                    f"failed to load embedding model {self.settings.embedding_model!r}; "
                    "hash fallback is disabled"
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
        )

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
        except Exception:
            pass

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        if is_query:
            prefix = self.settings.embedding_query_prefix.strip()
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
            batch_size = max(1, int(self.settings.embedding_batch_size))
            while True:
                try:
                    with context_factory():
                        encoder = getattr(self._model, "encode_document", self._model.encode)
                        arr = encoder(
                            texts,
                            batch_size=batch_size,
                            normalize_embeddings=self.settings.normalize_embeddings,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if batch_size <= 1 or not any(token in message for token in ("out of memory", "mps backend", "allocation")):
                        raise
                    batch_size = max(1, batch_size // 2)
                    log.warning("embedding OOM/allocation failure; retrying with batch_size=%s", batch_size)
                    self._empty_device_cache(force=True)
            self._empty_device_cache()
            return np.asarray(arr, dtype="float32")
        return self._hash_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        key = hashlib.blake2b(
            f"{self.settings.embedding_model}\x1f{self.settings.normalize_embeddings}\x1f{query}".encode(),
            digest_size=16,
        ).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        arr = self.embed_texts([query], is_query=True)[0]
        self._cache[key] = arr
        return arr

    def _hash_embed(self, texts: Iterable[str]) -> np.ndarray:
        if not isinstance(texts, list):
            texts = list(texts)
        vectors = np.zeros((len(texts), self._dim), dtype="float32")
        for row, text in enumerate(texts):
            padded = f"  {text.lower()}  "
            for i in range(max(1, len(padded) - 2)):
                gram = padded[i : i + 3]
                h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8", "ignore"), digest_size=8).digest(), "little")
                idx = h % self._dim
                sign = 1.0 if (h >> 63) == 0 else -1.0
                vectors[row, idx] += sign
        return self._normalize(vectors)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
