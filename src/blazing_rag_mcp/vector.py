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
    dtype: str = "float32"


class VectorIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ids: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype="float32")
        self.index = None
        self.backend = "numpy-accelerate"
        self.gpu = False
        self.dim = 0
        self.exact = True
        self.dtype = "float32"
        self._torch = None
        self._torch_matrix = None
        self._torch_device = "cpu"
        self._torch_dtype = None

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        self.ids = ids
        raw = np.asarray(vectors)
        self.dim = int(raw.shape[1]) if raw.size else 0
        self.index = None
        self.backend = "numpy-accelerate"
        self.gpu = False
        self.exact = True
        self._torch = None
        self._torch_matrix = None
        self._torch_device = "cpu"
        self._torch_dtype = None
        if not ids or self.dim == 0:
            self.matrix = np.zeros((0, 0), dtype="float32")
            self.dtype = "float32"
            return

        backend = self.settings.vector_backend
        torch_device = _resolve_torch_vector_device(self.settings.torch_vector_device, self.settings.device)

        # For small/medium local code indexes, Accelerate-backed float32 GEMV is usually faster
        # than paying an MPS command-buffer launch + synchronization for every query.
        if backend == "auto" and torch_device == "mps" and len(ids) < self.settings.mps_vector_min_vectors:
            self.matrix = np.asarray(raw, dtype="float32", order="C")
            self.dtype = "float32"
            log.warning(
                "using NumPy/Accelerate exact vector search vectors=%s dim=%s (< MPS threshold %s)",
                len(ids),
                self.dim,
                self.settings.mps_vector_min_vectors,
            )
            return

        # Preserve float16 for MPS to halve unified-memory traffic. FAISS receives float32 below.
        self.matrix = np.asarray(raw, order="C")
        self.dtype = str(self.matrix.dtype)

        if backend == "torch":
            if self._build_torch():
                return
            log.warning("torch vector backend requested but unavailable; falling back")
        elif backend == "faiss":
            if self._build_faiss():
                return
            log.warning("faiss vector backend requested but unavailable; falling back")
        elif backend == "auto":
            if torch_device == "mps" and self._build_torch():
                return
            if self._build_faiss():
                return
            if self._build_torch():
                return

        self.matrix = np.asarray(raw, dtype="float32", order="C")
        self.dtype = "float32"
        log.warning("using numpy exact vector search vectors=%s dim=%s", len(self.ids), self.dim)

    def _release_cpu_matrix_if_allowed(self) -> None:
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
            dtype_name = self.settings.torch_vector_dtype
            if dtype_name == "auto":
                dtype_name = "float16" if device in {"mps", "cuda"} else "float32"
            torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
            cpu_source = np.asarray(self.matrix, dtype=np.float16 if torch_dtype == torch.float16 else np.float32, order="C")
            mat = torch.as_tensor(cpu_source, dtype=torch_dtype, device=device).contiguous()
            probe = torch.as_tensor(cpu_source[:1], dtype=torch_dtype, device=device)
            _ = torch.topk(mat @ probe[0], k=1)
            self._torch = torch
            self._torch_matrix = mat
            self._torch_device = device
            self._torch_dtype = torch_dtype
            self.dtype = dtype_name
            self.backend = f"torch-{device}"
            self.gpu = device in {"cuda", "mps"}
            self.exact = True
            self._release_cpu_matrix_if_allowed()
            log.warning(
                "built Torch %s exact vector index vectors=%s dim=%s dtype=%s keep_cpu_copy=%s",
                device.upper(), n, self.dim, dtype_name, self.settings.keep_cpu_vector_copy,
            )
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
        matrix = np.asarray(self.matrix, dtype="float32", order="C")
        try:
            if self.settings.faiss_gpu and _faiss_gpu_available(faiss):
                try:
                    res = faiss.StandardGpuResources()
                    if n <= self.settings.exact_search_threshold:
                        cpu_index = faiss.IndexFlatIP(dim)
                        self.exact = True
                    else:
                        nlist = _choose_nlist(n)
                        quantizer = faiss.IndexFlatIP(dim)
                        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                        cpu_index.nprobe = min(64, max(8, nlist // 16))
                        cpu_index.train(_training_sample(matrix, max_train=min(n, nlist * 256)))
                        self.exact = False
                    self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    self.index.add(matrix)
                    self.backend = "faiss-gpu"
                    self.gpu = True
                    self.dtype = "float32"
                    self._release_cpu_matrix_if_allowed()
                    return True
                except Exception as exc:
                    log.warning("FAISS GPU unavailable; falling back to CPU FAISS: %s", exc)

            if n <= self.settings.exact_search_threshold:
                cpu_index = faiss.IndexFlatIP(dim)
                self.exact = True
            else:
                cpu_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                cpu_index.hnsw.efConstruction = 96
                cpu_index.hnsw.efSearch = 96
                self.exact = False
            cpu_index.add(matrix)
            self.index = cpu_index
            self.backend = "faiss-cpu"
            self.gpu = False
            self.dtype = "float32"
            self._release_cpu_matrix_if_allowed()
            return True
        except Exception as exc:
            log.warning("failed to build FAISS index: %s", exc)
            return False

    def search(self, query_vec: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if not self.ids or self.dim == 0:
            return []
        q32 = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        if self._torch_matrix is not None and self._torch is not None:
            try:
                torch = self._torch
                tq = torch.as_tensor(q32[0], dtype=self._torch_dtype, device=self._torch_device)
                scores = self._torch_matrix @ tq
                k = min(limit, len(self.ids))
                vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)
                vals_l = vals.float().detach().cpu().numpy().tolist()
                idxs_l = idxs.detach().cpu().numpy().tolist()
                return [(self.ids[int(i)], float(s)) for s, i in zip(vals_l, idxs_l, strict=False)]
            except Exception as exc:
                log.warning("Torch vector search failed; falling back to CPU path: %s", exc)
        if self.index is not None:
            scores, idxs = self.index.search(q32, min(limit, len(self.ids)))
            return [
                (self.ids[idx], float(score))
                for score, idx in zip(scores[0].tolist(), idxs[0].tolist(), strict=False)
                if idx >= 0
            ]
        matrix = np.asarray(self.matrix, dtype="float32")
        sims = matrix @ q32[0]
        k = min(limit, sims.shape[0])
        if k <= 0:
            return []
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
            dtype=self.dtype,
        )


def _faiss_gpu_available(faiss_module: object) -> bool:
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
    raw = int(max(64, min(8192, round(n ** 0.5 * 4))))
    return max(64, (raw // 32) * 32)


def _training_sample(matrix: np.ndarray, max_train: int) -> np.ndarray:
    if matrix.shape[0] <= max_train:
        return matrix
    step = max(1, matrix.shape[0] // max_train)
    return matrix[::step][:max_train].astype("float32", copy=False)
