"""Canonical, test-only array provenance for the physical H10 diagnostics."""

from __future__ import annotations

import hashlib

import numpy as np


def canonical_array_fingerprint(value: object) -> dict[str, object]:
    """Fingerprint a real numeric array in a stable, explicitly typed form.

    Floating inputs canonicalize to contiguous ``float64`` bytes; integral
    inputs (the ISDF pivots) canonicalize to contiguous ``int64`` bytes.
    The reported norm is calculated from the canonical representation in
    float64 so the record is comparable across numeric backends.  This helper
    is test-only provenance, not a cache key or production solver route.
    """

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError("canonical H10 fingerprints require real arrays")
    if np.issubdtype(array.dtype, np.floating):
        canonical_dtype = np.dtype(np.float64)
    elif np.issubdtype(array.dtype, np.integer):
        canonical_dtype = np.dtype(np.int64)
    else:
        raise ValueError(
            "canonical H10 fingerprints require floating or integral arrays; "
            f"got {array.dtype}"
        )

    canonical = np.ascontiguousarray(array, dtype=canonical_dtype)
    return {
        "shape": list(canonical.shape),
        "canonical_dtype": canonical_dtype.name,
        "frobenius_norm": float(
            np.linalg.norm(np.asarray(canonical, dtype=np.float64))
        ),
        "sha256_c_contiguous_canonical_bytes": hashlib.sha256(
            canonical.tobytes(order="C")
        ).hexdigest(),
    }
