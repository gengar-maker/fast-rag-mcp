from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorIndexInfo:
    backend: str
    gpu: bool
    vectors: int
    dim: int
    exact: bool


class VectorIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ids: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype="float32")
        self.index = None
        self.backend = "numpy"
        self.gpu = False
        self.dim = 0
        self.exact = True
        self._torch = None
        self._torch_matrix = None
        self._torch_device = "cpu"

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        self.ids = ids
        self.matrix = np.asarray(vectors, dtype="float32")
        self.dim = int(self.matrix.shape[1]) if self.matrix.size else 0
        self.index = None
        self.backend = "numpy"
        self.gpu = False
        self.exact = True
        self._torch = None
        self._torch_matrix = None
        self._torch_device = "cpu"
        if not ids or self.dim == 0:
            return

        backend = self.settings.vector_backend

        if backend == "torch":
            if self._build_torch():
                return
            log.warning("torch vector backend requested but unavailable; falling back")
        elif backend == "faiss":
            if self._build_faiss():
                return
            log.warning("faiss vector backend requested but unavailable; falling back")
        elif backend == "auto":
            # Apple Silicon has no FAISS Metal GPU path. Exact Torch/MPS matmul is usually the
            # fastest local option for small-to-medium code indexes and avoids slow CPU-only FAISS.
            if _resolve_torch_vector_device(self.settings.torch_vector_device, self.settings.device) == "mps":
                if self._build_torch():
                    return
            # CUDA/ROCm FAISS remains the best ANN path where available.
            if self._build_faiss():
                return
            # If FAISS is not installed, a Torch CUDA/MPS exact index is still useful.
            if self._build_torch():
                return

        log.warning("using numpy exact vector search vectors=%s dim=%s", len(self.ids), self.dim)

    def _release_cpu_matrix_if_allowed(self) -> None:
        # FAISS and Torch backends own their own copy of vectors. Keeping the original numpy
        # matrix is useful as a debug fallback but expensive for large repos, especially on
        # Apple Silicon where Torch/MPS also keeps a GPU-resident copy.
        if not self.settings.keep_cpu_vector_copy:
            self.matrix = np.zeros((0, self.dim), dtype="float32")

    def _build_torch(self) -> bool:
        n = len(self.ids)
        if n > self.settings.torch_vector_max_vectors:
            log.warning(
                "torch exact vector index skipped: vectors=%s exceeds BRAG_TORCH_VECTOR_MAX_VECTORS=%s",
                n,
                self.settings.torch_vector_max_vectors,
            )
            return False
        device = _resolve_torch_vector_device(self.settings.torch_vector_device, self.settings.device)
        if device == "cpu":
            return False
        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                return False
            if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                return False
            mat = torch.as_tensor(self.matrix, dtype=torch.float32, device=device).contiguous()
            # One tiny matmul/topk validates that the backend supports the hot path.
            probe = torch.as_tensor(self.matrix[:1], dtype=torch.float32, device=device)
            _ = torch.topk(mat @ probe[0], k=1)
            self._torch = torch
            self._torch_matrix = mat
            self._torch_device = device
            self.backend = f"torch-{device}"
            self.gpu = device in {"cuda", "mps"}
            self.exact = True
            self._release_cpu_matrix_if_allowed()
            log.warning("built Torch %s exact vector index vectors=%s dim=%s keep_cpu_copy=%s", device.upper(), n, self.dim, self.settings.keep_cpu_vector_copy)
            return True
        except Exception as exc:
            log.warning("Torch vector backend unavailable for device=%s: %s", device, exc)
            self._torch = None
            self._torch_matrix = None
            return False

    def _build_faiss(self) -> bool:
        try:
            import faiss  # type: ignore
        except Exception as exc:
            if self.settings.vector_backend == "faiss":
                log.warning("faiss requested but unavailable: %s", exc)
            return False

        n = len(self.ids)
        dim = self.dim
        try:
            # For normalized embeddings, inner product equals cosine similarity.
            if self.settings.faiss_gpu and _faiss_gpu_available(faiss):
                try:
                    res = faiss.StandardGpuResources()
                    if n <= self.settings.exact_search_threshold:
                        cpu_index = faiss.IndexFlatIP(dim)
                        self.exact = True
                    else:
                        # GPU-friendly ANN. HNSW is great on CPU, but GPU FAISS reliably supports IVF variants.
                        nlist = _choose_nlist(n)
                        quantizer = faiss.IndexFlatIP(dim)
                        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                        cpu_index.nprobe = min(64, max(8, nlist // 16))
                        train_x = _training_sample(self.matrix, max_train=min(n, nlist * 256))
                        cpu_index.train(train_x)
                        self.exact = False
                    self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    self.index.add(self.matrix)
                    self.backend = "faiss-gpu"
                    self.gpu = True
                    self._release_cpu_matrix_if_allowed()
                    log.warning("built FAISS GPU index vectors=%s dim=%s exact=%s keep_cpu_copy=%s", n, dim, self.exact, self.settings.keep_cpu_vector_copy)
                    return True
                except Exception as exc:
                    log.warning("FAISS GPU unavailable for this index/wheel; falling back to CPU FAISS: %s", exc)

            if n <= self.settings.exact_search_threshold:
                cpu_index = faiss.IndexFlatIP(dim)
                self.exact = True
            else:
                # CPU fallback optimized for changing local workspaces: no IVF training required.
                cpu_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                cpu_index.hnsw.efConstruction = 96
                cpu_index.hnsw.efSearch = 96
                self.exact = False

            cpu_index.add(self.matrix)
            self.index = cpu_index
            self.backend = "faiss-cpu"
            self.gpu = False
            self._release_cpu_matrix_if_allowed()
            log.warning("built FAISS CPU index vectors=%s dim=%s exact=%s keep_cpu_copy=%s", n, dim, self.exact, self.settings.keep_cpu_vector_copy)
            return True
        except Exception as exc:
            log.warning("failed to build FAISS index: %s", exc)
            return False

    def search(self, query_vec: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if not self.ids or self.dim == 0:
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        if self._torch_matrix is not None and self._torch is not None:
            try:
                torch = self._torch
                tq = torch.as_tensor(q[0], dtype=torch.float32, device=self._torch_device)
                scores = self._torch_matrix @ tq
                k = min(limit, len(self.ids))
                vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)
                vals_l = vals.detach().cpu().numpy().tolist()
                idxs_l = idxs.detach().cpu().numpy().tolist()
                return [(self.ids[int(i)], float(s)) for s, i in zip(vals_l, idxs_l, strict=False)]
            except Exception as exc:
                log.warning("Torch vector search failed; falling back to CPU path: %s", exc)
        if self.index is not None:
            scores, idxs = self.index.search(q, min(limit, len(self.ids)))
            out: list[tuple[str, float]] = []
            for score, idx in zip(scores[0].tolist(), idxs[0].tolist(), strict=False):
                if idx < 0:
                    continue
                out.append((self.ids[idx], float(score)))
            return out
        sims = self.matrix @ q[0]
        k = min(limit, sims.shape[0])
        if k <= 0:
            return []
        # argpartition is much faster than full sort for large arrays.
        idxs = np.argpartition(-sims, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]
        return [(self.ids[int(i)], float(sims[int(i)])) for i in idxs]

    def info(self) -> VectorIndexInfo:
        return VectorIndexInfo(
            backend=self.backend,
            gpu=self.gpu,
            vectors=len(self.ids),
            dim=self.dim,
            exact=self.exact,
        )


def _faiss_gpu_available(faiss_module: object) -> bool:
    # FAISS does not expose a Metal/MPS GPU backend; Python GPU wheels are CUDA/ROCm-oriented.
    return hasattr(faiss_module, "StandardGpuResources") and hasattr(faiss_module, "index_cpu_to_gpu")


def _resolve_torch_vector_device(torch_vector_device: str, embedding_device: str) -> str:
    requested = torch_vector_device
    if requested == "auto":
        requested = embedding_device
    if requested == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            return "cpu"
        return "cpu"
    return requested


def _choose_nlist(n: int) -> int:
    # FAISS IVF rule of thumb: sqrt-ish bucket count, constrained to practical powers.
    raw = int(max(64, min(8192, round(n ** 0.5 * 4))))
    # Round to a multiple of 32 for nicer GPU behavior.
    return max(64, (raw // 32) * 32)


def _training_sample(matrix: np.ndarray, max_train: int) -> np.ndarray:
    if matrix.shape[0] <= max_train:
        return matrix
    # Deterministic strided sample avoids importing RNG and is reproducible.
    step = max(1, matrix.shape[0] // max_train)
    return matrix[::step][:max_train].astype("float32", copy=False)
