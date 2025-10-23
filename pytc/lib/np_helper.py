"""NumPy helper utilities for determinant evaluation."""

import numpy as np

try:
    from numba import njit, prange  # type: ignore
except ImportError:  # pragma: no cover - numba is optional
    _NUMBA_AVAILABLE = False
    njit = None  # type: ignore
    prange = range  # type: ignore
else:  # pragma: no cover - exercised only when numba is installed
    _NUMBA_AVAILABLE = True


if _NUMBA_AVAILABLE:  # pragma: no cover - depends on optional dependency

    @njit(parallel=True, cache=True)
    def _det_batch_numba(mats: np.ndarray) -> np.ndarray:
        """Compute determinants for a stack of matrices using Numba."""
        n = mats.shape[0]
        out = np.empty(n, dtype=mats.dtype)
        for i in prange(n):
            out[i] = np.linalg.det(mats[i])
        return out
else:
    # Provide a pure-NumPy fallback implementation when numba is not installed.
    def _det_batch_numba(mats: np.ndarray) -> np.ndarray:
        """Compute determinants for a stack of matrices using NumPy.

        This matches the signature of the numba implementation so callers
        (e.g. batched_det) can call it unconditionally when parallel
        processing is requested.
        """
        # Use a simple Python loop to compute determinants per matrix. We
        # keep this straightforward and correct; for large batches users
        # should install numba for better parallel performance.
        n = mats.shape[0]
        out = np.empty(n, dtype=mats.dtype)
        for i in range(n):
            out[i] = np.linalg.det(mats[i])
        return out


def _as_array(mats: np.ndarray) -> np.ndarray:
    """Ensure we have a contiguous ndarray without unnecessary copies."""
    return np.ascontiguousarray(mats)



def batched_det(
    mats: np.ndarray,
    parallel: bool = True,
) -> np.ndarray:
    """Compute determinants for a single matrix or a batch of matrices."""

    mats = _as_array(mats)
    if mats.ndim == 2:
        return np.linalg.det(mats)
    if mats.ndim != 3:
        raise ValueError("Expected a 2D matrix or a 3D batch of matrices")

    n = mats.shape[0]
    if n == 0:
        return np.empty((0,), dtype=mats.dtype)

    if parallel and n > 1:
        return _det_batch_numba(mats)

    return np.linalg.det(mats)
