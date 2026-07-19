"""Periodic ISDF fit machinery: matrix-free Hermitian-PSD pivoted Cholesky
selector, Pi^q/eta^q builders, kernel-apply-and-solve (NumPy oracle and
device/KernelProvider paths), and staging policy. See design doc §3-§7.
Deliberately independent of pytc.df.pivots (see design doc §3).
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


# The exact E1 selector keeps the complete Bloch AO cache on the JAX device.
# This is intentionally a bounded, fail-closed baseline.  Any distinct
# localized backend is separately held until measured capacity evidence and
# owner direction justify it; no fallback changes this selector's physical
# candidate set.
DEFAULT_JAX_CACHED_SELECTOR_PEAK_MAX_BYTES = 24 * 2**30
DEFAULT_JAX_CACHED_SELECTOR_PEAK_SAFETY_FACTOR = 1.10


class JAXCachedMatrixFreeCapacityError(RuntimeError):
    """Raised when predicted selector peak exceeds the declared policy."""

    condition = "JAX_CACHED_MATRIX_FREE_SELECTION_PEAK_EXCEEDS_POLICY"


class JAXTranslationMatrixFreeCapacityError(RuntimeError):
    """Raised when the exact translation-cache selector exceeds policy."""

    condition = "JAX_TRANSLATION_MATRIX_FREE_SELECTION_PEAK_EXCEEDS_POLICY"


class TranslationAORepresentationError(RuntimeError):
    """Raised when k-points do not admit the exact real translation cache."""

    condition = "TRANSLATION_AO_REQUIRES_GAMMA_ORTHOGONAL_PHASE_CLASSES"


@dataclasses.dataclass(frozen=True)
class TranslationAORepresentation:
    """Exact lattice-image phase classes for a gamma-containing k mesh."""

    lattice_vectors: np.ndarray
    groups: tuple[np.ndarray, ...]
    unitary_phase: np.ndarray
    kpts: np.ndarray
    gamma_kpt: np.ndarray
    phase_orthogonality_residual: float

    @property
    def n_classes(self):
        return len(self.groups)


def build_translation_ao_representation(cell, kpts, *, phase_tol=1e-10):
    """Group lattice images whose Bloch phases agree on the full k mesh.

    For a complete gamma-containing Monkhorst-Pack mesh, lattice images fall
    into at most ``Nk`` phase classes.  If ``B_c`` is the gamma AO sum over
    class ``c`` and ``P[k,c]`` its phase, then ``P.conj().T @ P = Nk I`` and
    the scaled real factors ``sqrt(Nk) B_c`` have exactly the same grid Gram
    matrix as the materialized complex Bloch factors.  For a complete mesh
    the class rank is still ``Nk``: this is an exact complex-to-real 2x cache
    reduction, not a sparse channel reduction.  This function verifies those
    identities before any AO evaluation.
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    if kpts.ndim != 2 or kpts.shape[1] != 3 or kpts.shape[0] == 0:
        raise ValueError(f"kpts must have nonempty shape (Nk,3), got {kpts.shape}.")
    if not isinstance(phase_tol, (int, float)) or phase_tol <= 0:
        raise ValueError("phase_tol must be positive.")

    gamma_index = int(np.argmin(np.linalg.norm(kpts, axis=1)))
    gamma_kpt = kpts[gamma_index]
    if np.linalg.norm(gamma_kpt) > phase_tol:
        raise TranslationAORepresentationError(
            f"{TranslationAORepresentationError.condition}: k mesh has no gamma point."
        )

    lattice_vectors = np.asarray(cell.get_lattice_Ls(), dtype=np.float64)
    if lattice_vectors.ndim != 2 or lattice_vectors.shape[1] != 3:
        raise ValueError(
            "cell.get_lattice_Ls() must return shape (Nimages,3), got "
            f"{lattice_vectors.shape}."
        )
    phases = np.exp(1j * (kpts @ lattice_vectors.T))
    representatives = []
    groups = []
    for image in range(lattice_vectors.shape[0]):
        for class_index, representative in enumerate(representatives):
            if np.max(np.abs(phases[:, image] - phases[:, representative])) <= phase_tol:
                groups[class_index].append(image)
                break
        else:
            representatives.append(image)
            groups.append([image])

    phase = phases[:, representatives]
    n_kpts = kpts.shape[0]
    gram = phase.conj().T @ phase / n_kpts
    residual = float(np.max(np.abs(gram - np.eye(len(groups)))))
    if len(groups) > n_kpts or residual > 10 * phase_tol:
        raise TranslationAORepresentationError(
            f"{TranslationAORepresentationError.condition}: {len(groups)} phase classes "
            f"for {n_kpts} k-points, normalized phase-Gram residual={residual:.3e}."
        )
    return TranslationAORepresentation(
        lattice_vectors=lattice_vectors,
        groups=tuple(np.asarray(group, dtype=np.int64) for group in groups),
        unitary_phase=np.asarray(phase / np.sqrt(n_kpts), dtype=np.complex128),
        kpts=np.asarray(kpts, dtype=np.float64),
        gamma_kpt=np.asarray(gamma_kpt, dtype=np.float64),
        phase_orthogonality_residual=residual,
    )


def jax_translation_matrix_free_byte_model(
    n_kpts, n_classes, n_grid, n_ao, rank, *,
    ao_block_size,
    selection_peak_max_bytes=DEFAULT_JAX_CACHED_SELECTOR_PEAK_MAX_BYTES,
    peak_safety_factor=DEFAULT_JAX_CACHED_SELECTOR_PEAK_SAFETY_FACTOR,
):
    """Return bounded host/device residency for translation-class selection.

    The real class cache persists on the host while it is copied to the JAX
    device.  During construction, one full-k complex AO block and its
    complex class-transform block coexist with that cache.  The declared
    host peak therefore covers cache construction, while the device peak
    covers the compiled pivot kernel.
    """
    values = {
        "n_kpts": n_kpts,
        "n_classes": n_classes,
        "n_grid": n_grid,
        "n_ao": n_ao,
        "rank": rank,
        "ao_block_size": ao_block_size,
        "selection_peak_max_bytes": selection_peak_max_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if n_classes > n_kpts:
        raise ValueError("n_classes cannot exceed n_kpts for the orthogonal representation.")
    if not isinstance(peak_safety_factor, (int, float)) or peak_safety_factor < 1.0:
        raise ValueError("peak_safety_factor must be a real number >= 1.0.")

    ao_block_size = min(int(ao_block_size), int(n_grid))
    cache_bytes = int(n_classes) * int(n_ao) * int(n_grid) * np.dtype(np.float64).itemsize
    complex_cache_bytes = (
        int(n_kpts) * int(n_ao) * int(n_grid) * np.dtype(np.complex128).itemsize
    )
    factor_bytes = int(n_grid) * int(rank) * np.dtype(np.float64).itemsize
    gram_work_bytes = int(n_grid) * np.dtype(np.float64).itemsize
    real_work_bytes = 2 * int(n_grid) * np.dtype(np.float64).itemsize
    selector_bytes = int(n_grid) * np.dtype(np.bool_).itemsize
    pivot_bytes = int(rank) * np.dtype(np.int64).itemsize
    phase_bytes = int(n_kpts) * int(n_classes) * np.dtype(np.complex128).itemsize
    ao_block_bytes = (
        int(n_kpts) * int(ao_block_size) * int(n_ao) * np.dtype(np.complex128).itemsize
    )
    transform_block_bytes = (
        int(n_classes) * int(ao_block_size) * int(n_ao)
        * np.dtype(np.complex128).itemsize
    )
    work_bytes = gram_work_bytes + real_work_bytes + selector_bytes + pivot_bytes
    selection_peak_device_bytes = cache_bytes + factor_bytes + work_bytes
    selection_peak_host_bytes = (
        cache_bytes + phase_bytes + ao_block_bytes + transform_block_bytes
    )
    device_with_safety = int(
        np.ceil(selection_peak_device_bytes * float(peak_safety_factor))
    )
    host_with_safety = int(
        np.ceil(selection_peak_host_bytes * float(peak_safety_factor))
    )
    device_condition = (
        None
        if device_with_safety <= int(selection_peak_max_bytes)
        else JAXTranslationMatrixFreeCapacityError.condition
    )
    host_condition = (
        None
        if host_with_safety <= int(selection_peak_max_bytes)
        else JAXTranslationMatrixFreeCapacityError.condition
    )
    condition = device_condition or host_condition
    return {
        "mode": "jax_translation_matrix_free",
        "translation_cache_layout": "sqrt(Nk)*B[Nclass,Nao,Ng]",
        "translation_cache_real_float64_bytes": cache_bytes,
        "complex_bloch_cache_equivalent_bytes": complex_cache_bytes,
        "translation_cache_to_complex_cache_ratio": cache_bytes / complex_cache_bytes,
        "unitary_phase_complex128_bytes": phase_bytes,
        "bounded_ao_block_complex128_bytes": ao_block_bytes,
        "bounded_transform_block_complex128_bytes": transform_block_bytes,
        "ao_block_size": ao_block_size,
        "cholesky_real_float64_bytes": factor_bytes,
        "pivot_work_bytes": work_bytes,
        "selection_peak_device_bytes": selection_peak_device_bytes,
        "selection_peak_host_bytes": selection_peak_host_bytes,
        "selection_peak_safety_factor": float(peak_safety_factor),
        "selection_peak_device_with_safety_bytes": device_with_safety,
        "selection_peak_host_with_safety_bytes": host_with_safety,
        "selection_peak_required_with_safety_bytes": max(
            device_with_safety, host_with_safety,
        ),
        "selection_peak_max_bytes": int(selection_peak_max_bytes),
        "device_capacity_condition": device_condition,
        "host_capacity_condition": host_condition,
        "capacity_condition": condition,
        "within_cache_policy": condition is None,
    }


def jax_cached_matrix_free_byte_model(
    n_kpts, n_grid, n_ao, rank, *,
    selection_peak_max_bytes=DEFAULT_JAX_CACHED_SELECTOR_PEAK_MAX_BYTES,
    peak_safety_factor=DEFAULT_JAX_CACHED_SELECTOR_PEAK_SAFETY_FACTOR,
):
    """Return exact selector residency terms without allocating an AO cache.

    ``F`` is the complex128 matrix with shape ``(Nk*Nao, Ng)`` used by the
    device pivot loop.  The Cholesky factor and residual are real float64;
    the metric column is formed transiently from ``F.conj().T @ F[:, pivot]``.
    A cache-policy failure is named in the returned record so callers can
    refuse before evaluating the complete AO grid.
    """
    values = {
        "n_kpts": n_kpts,
        "n_grid": n_grid,
        "n_ao": n_ao,
        "rank": rank,
        "selection_peak_max_bytes": selection_peak_max_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if not isinstance(peak_safety_factor, (int, float)) or peak_safety_factor < 1.0:
        raise ValueError("peak_safety_factor must be a real number >= 1.0.")

    cache_bytes = int(n_kpts) * int(n_ao) * int(n_grid) * np.dtype(np.complex128).itemsize
    factor_bytes = int(n_grid) * int(rank) * np.dtype(np.float64).itemsize
    gram_work_bytes = int(n_grid) * np.dtype(np.complex128).itemsize
    real_work_bytes = 2 * int(n_grid) * np.dtype(np.float64).itemsize
    selector_bytes = int(n_grid) * np.dtype(np.bool_).itemsize
    pivot_bytes = int(rank) * np.dtype(np.int64).itemsize
    work_bytes = gram_work_bytes + real_work_bytes + selector_bytes + pivot_bytes
    selection_peak_device_bytes = cache_bytes + factor_bytes + work_bytes
    selection_peak_device_with_safety_bytes = int(
        np.ceil(selection_peak_device_bytes * float(peak_safety_factor))
    )
    capacity_condition = (
        None
        if selection_peak_device_with_safety_bytes <= int(selection_peak_max_bytes)
        else JAXCachedMatrixFreeCapacityError.condition
    )
    return {
        "mode": "jax_cached_matrix_free",
        "ao_cache_layout": "F[Nk*Nao,Ng]",
        "ao_cache_complex128_bytes": cache_bytes,
        "host_ao_cache_complex128_bytes": cache_bytes,
        "cholesky_real_float64_bytes": factor_bytes,
        "pivot_work_bytes": work_bytes,
        "selection_peak_device_bytes": selection_peak_device_bytes,
        "selection_peak_host_bytes": cache_bytes,
        "selection_peak_safety_factor": float(peak_safety_factor),
        "selection_peak_device_with_safety_bytes": selection_peak_device_with_safety_bytes,
        "selection_peak_max_bytes": int(selection_peak_max_bytes),
        "capacity_condition": capacity_condition,
        "within_cache_policy": capacity_condition is None,
    }


@partial(jax.jit, static_argnames=("rank", "n_kpts"))
def _select_jax_cached_matrix_free_kernel(
    f, *, rank, n_kpts, rcond=1e-12, ramp_scale=1e-12,
):
    """Stable compiled kernel for the exact full-grid JAX selector."""
    _, n_grid = f.shape
    diagonal = jnp.sum(jnp.abs(f) ** 2, axis=0) ** 2 / n_kpts
    max_diagonal = jnp.max(diagonal)
    ramp = ramp_scale * jnp.arange(n_grid, dtype=jnp.float64) * max_diagonal
    threshold = rcond * max_diagonal
    initial = (
        diagonal.astype(jnp.float64),
        jnp.zeros((n_grid, rank), dtype=jnp.float64),
        jnp.full((rank,), -1, dtype=jnp.int64),
        jnp.zeros((n_grid,), dtype=jnp.bool_),
        jnp.array(0, dtype=jnp.int64),
    )

    def body(t, state):
        residual, factor, pivots, selected, count = state
        pivot = jnp.argmax(jnp.where(selected, -jnp.inf, residual + ramp))
        active = residual[pivot] > threshold
        gram = f.conj().T @ f[:, pivot]
        metric_column = (jnp.abs(gram) ** 2 / n_kpts).astype(jnp.float64)
        previous = factor @ factor[pivot, :]
        denominator = jnp.sqrt(jnp.maximum(residual[pivot], jnp.finfo(jnp.float64).tiny))
        new_column = (metric_column - previous) / denominator
        new_column = jnp.where(active, new_column, jnp.zeros_like(new_column))
        factor = factor.at[:, t].set(new_column)
        residual = jnp.where(
            active, jnp.maximum(residual - new_column ** 2, 0.0), residual,
        )
        pivots = pivots.at[t].set(jnp.where(active, pivot, -1))
        selected = jnp.where(active, selected.at[pivot].set(True), selected)
        return residual, factor, pivots, selected, count + active.astype(jnp.int64)

    return jax.lax.fori_loop(0, rank, body, initial)


def select_jax_cached_matrix_free(
    ao_cache, rank, *, rcond=1e-12, ramp_scale=1e-12,
    selection_peak_max_bytes=DEFAULT_JAX_CACHED_SELECTOR_PEAK_MAX_BYTES,
    peak_safety_factor=DEFAULT_JAX_CACHED_SELECTOR_PEAK_SAFETY_FACTOR,
    return_factor=False,
):
    """Select exact full-grid pivots in one JIT/device control-flow loop.

    ``ao_cache`` has shape ``(Nk, Ng, Nao)``.  The returned pivots use the
    same metric, residual update, threshold, and high-index tie convention as
    :func:`pivoted_cholesky_hermitian`; the only change is that every pivot
    column is formed from the already cached AO matrix on the JAX device.
    Only the compact pivot vector/count is copied to host in production.
    ``return_factor=True`` is test-only and explicitly requests a host factor.
    """
    ao_cache = np.asarray(ao_cache, dtype=np.complex128)
    if ao_cache.ndim != 3 or any(size <= 0 for size in ao_cache.shape):
        raise ValueError("ao_cache must have nonempty shape (Nk,Ng,Nao).")
    n_kpts, n_grid, n_ao = ao_cache.shape
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError(f"rank must be an integer, got {rank!r}.")
    rank = int(rank)
    if rank <= 0 or rank > n_grid:
        raise ValueError(f"rank must be in [1,{n_grid}], got {rank}.")
    byte_model = jax_cached_matrix_free_byte_model(
        n_kpts, n_grid, n_ao, rank,
        selection_peak_max_bytes=selection_peak_max_bytes,
        peak_safety_factor=peak_safety_factor,
    )
    if byte_model["capacity_condition"] is not None:
        raise JAXCachedMatrixFreeCapacityError(
            f"{byte_model['capacity_condition']}: selector peak with safety requires "
            f"{byte_model['selection_peak_device_with_safety_bytes']} bytes, policy allows "
            f"{byte_model['selection_peak_max_bytes']} bytes."
        )
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "jax_cached_matrix_free requires jax_enable_x64=True for real float64 "
            "residual/L and complex128 AO cache."
        )

    # F is the only AO object used by the pivot loop: rows are flattened
    # (k, AO) channels and columns are physical-grid points.
    f_cache = jnp.asarray(
        np.ascontiguousarray(ao_cache.transpose(0, 2, 1).reshape(n_kpts * n_ao, n_grid)),
        dtype=jnp.complex128,
    )

    _, factor, pivots, _, count = _select_jax_cached_matrix_free_kernel(
        f_cache, rank=rank, n_kpts=n_kpts, rcond=rcond, ramp_scale=ramp_scale,
    )
    n_selected = int(np.asarray(count))
    pivots_host = np.asarray(pivots)[:n_selected]
    factor_host = np.asarray(factor)[:, :n_selected] if return_factor else None
    provenance = {
        **byte_model,
        "pivot_executor": "jax.jit/lax.fori_loop",
        "pivot_loop_device_resident": True,
        "ao_cache_dtype": str(f_cache.dtype),
        "factor_dtype": str(factor.dtype),
        "device_platforms": sorted({device.platform for device in f_cache.devices()}),
    }
    return pivots_host, factor_host, n_selected, provenance


def build_translation_ao_cache(
    cell, grid_coords, block_size, representation, *, stats=None, imag_tol=1e-12,
):
    """Build ``sqrt(Nk) B[Nclass,Nao,Ng]`` without materializing k-AOs.

    Each full-k AO block is evaluated once and immediately transformed by the
    verified unitary phase matrix.  Thus all-k AOs exist only for one bounded
    block; the persistent cache is the real translation-class coefficient.
    A non-negligible imaginary component fails closed instead of silently
    changing the representation.
    """
    if not isinstance(representation, TranslationAORepresentation):
        raise TypeError("representation must be a TranslationAORepresentation.")
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3 or grid_coords.shape[0] == 0:
        raise ValueError(
            f"grid_coords must have nonempty shape (Ng,3), got {grid_coords.shape}."
        )
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if not isinstance(imag_tol, (int, float)) or imag_tol < 0:
        raise ValueError("imag_tol must be nonnegative.")

    n_grid = grid_coords.shape[0]
    n_ao = int(cell.nao_nr())
    cache = np.empty((representation.n_classes, n_ao, n_grid), dtype=np.float64)
    for g0 in range(0, n_grid, block_size):
        g1 = min(g0 + block_size, n_grid)
        ao = np.asarray(cell.pbc_eval_gto(
            "GTOval", grid_coords[g0:g1], kpts=list(representation.kpts),
        ), dtype=np.complex128)
        expected = (representation.kpts.shape[0], g1 - g0, n_ao)
        if ao.shape != expected:
            raise ValueError(f"full-k AO block must have shape {expected}, got {ao.shape}.")
        coefficients = np.einsum(
            "ck,kba->cba", representation.unitary_phase.conj().T, ao, optimize=True,
        )
        imag_max = float(np.max(np.abs(coefficients.imag)))
        real_scale = max(1.0, float(np.max(np.abs(coefficients.real))))
        if imag_max > imag_tol * real_scale:
            raise TranslationAORepresentationError(
                "translation-class coefficients are not real within "
                f"imag_tol={imag_tol:.1e}: max imaginary={imag_max:.3e}."
            )
        cache[:, :, g0:g1] = coefficients.real.transpose(0, 2, 1)
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + (g1 - g0)
    return cache


def stream_ao_blocks_from_translation_cache(
    translation_cache, unitary_phase, block_size, *, stats=None,
):
    """Reconstruct exact ``(Nk,block,Nao)`` AO blocks from the real cache."""
    cache = np.asarray(translation_cache, dtype=np.float64)
    phase = np.asarray(unitary_phase, dtype=np.complex128)
    if cache.ndim != 3 or any(size <= 0 for size in cache.shape):
        raise ValueError("translation_cache must have shape (Nclass,Nao,Ng).")
    if phase.ndim != 2 or phase.shape[1] != cache.shape[0] or phase.shape[0] == 0:
        raise ValueError(
            "unitary_phase must have shape (Nk,Nclass) matching translation_cache."
        )
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    n_grid = cache.shape[2]
    for g0 in range(0, n_grid, block_size):
        g1 = min(g0 + block_size, n_grid)
        ao_block = np.einsum(
            "kc,cab->kba", phase, cache[:, :, g0:g1], optimize=True,
        ).astype(np.complex128, copy=False)
        if stats is not None:
            stats["translation_reconstruction_calls"] = (
                stats.get("translation_reconstruction_calls", 0) + 1
            )
            stats["translation_reconstruction_grid_points"] = (
                stats.get("translation_reconstruction_grid_points", 0) + (g1 - g0)
            )
        yield g0, g1, ao_block


def select_jax_translation_matrix_free(
    translation_cache, n_kpts, rank, *, rcond=1e-12, ramp_scale=1e-12,
    ao_block_size,
    selection_peak_max_bytes=DEFAULT_JAX_CACHED_SELECTOR_PEAK_MAX_BYTES,
    peak_safety_factor=DEFAULT_JAX_CACHED_SELECTOR_PEAK_SAFETY_FACTOR,
    return_factor=False,
):
    """Select exact pivots from the scaled real translation-class factors."""
    cache = np.asarray(translation_cache)
    if cache.ndim != 3 or any(size <= 0 for size in cache.shape):
        raise ValueError("translation_cache must have nonempty shape (Nclass,Nao,Ng).")
    if np.iscomplexobj(cache):
        raise ValueError("translation_cache must be real float64.")
    cache = np.asarray(cache, dtype=np.float64)
    n_classes, n_ao, n_grid = cache.shape
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError(f"rank must be an integer, got {rank!r}.")
    rank = int(rank)
    if rank <= 0 or rank > n_grid:
        raise ValueError(f"rank must be in [1,{n_grid}], got {rank}.")
    byte_model = jax_translation_matrix_free_byte_model(
        n_kpts, n_classes, n_grid, n_ao, rank,
        ao_block_size=ao_block_size,
        selection_peak_max_bytes=selection_peak_max_bytes,
        peak_safety_factor=peak_safety_factor,
    )
    if byte_model["capacity_condition"] is not None:
        raise JAXTranslationMatrixFreeCapacityError(
            f"{byte_model['capacity_condition']}: selector peak with safety requires "
            f"{byte_model['selection_peak_required_with_safety_bytes']} bytes, policy allows "
            f"{byte_model['selection_peak_max_bytes']} bytes."
        )
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "jax_translation_matrix_free requires jax_enable_x64=True for the real "
            "float64 translation cache and residual/L."
        )

    f_cache = jnp.asarray(cache.reshape(n_classes * n_ao, n_grid), dtype=jnp.float64)
    _, factor, pivots, _, count = _select_jax_cached_matrix_free_kernel(
        f_cache, rank=rank, n_kpts=n_kpts, rcond=rcond, ramp_scale=ramp_scale,
    )
    n_selected = int(np.asarray(count))
    pivots_host = np.asarray(pivots)[:n_selected]
    factor_host = np.asarray(factor)[:, :n_selected] if return_factor else None
    provenance = {
        **byte_model,
        "pivot_executor": "jax.jit/lax.fori_loop",
        "pivot_loop_device_resident": True,
        "translation_cache_dtype": str(f_cache.dtype),
        "factor_dtype": str(factor.dtype),
        "device_platforms": sorted({device.platform for device in f_cache.devices()}),
    }
    return pivots_host, factor_host, n_selected, provenance


def pivoted_cholesky_hermitian(diag, col_eval, rank, *, rcond=1e-12, ramp_scale=1e-12):
    """Matrix-free greedy pivoted (partial) Cholesky for an implicit N x N
    Hermitian PSD matrix M given diag(M) and a column oracle
    col_eval(j) -> M[:, j] of the ORIGINAL M (shape (N,), complex128).

    Tie-break: a tiny increasing ramp `ramp_scale * arange(n) * max(diag)`
    is added to the argmax score, biasing exact ties toward the higher
    index (matches pytc.df.pivots's convention; copied, not imported).

    Returns:
        (pivots, L, n_selected): pivots (n_selected,) int64; L
        (n, n_selected) complex128 partial Cholesky factor; n_selected
        <= rank (fewer if the Schur diagonal exhausts below
        rcond*max(diag) first).
    """
    diag = np.asarray(diag, dtype=np.float64)
    if diag.ndim != 1:
        raise ValueError(f"diag must be 1-D, got shape {diag.shape}.")
    n = diag.shape[0]
    if n == 0:
        raise ValueError("diag must be nonempty.")
    if not np.all(np.isfinite(diag)):
        raise ValueError("diag must be finite.")

    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError(f"rank must be an integer, got {rank!r}.")
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    if rank > n:
        raise ValueError(f"rank={rank} exceeds n={n} -- cannot select more pivots than rows.")

    max_diag = float(np.max(diag)) if n > 0 else 0.0
    if max_diag <= 0.0:
        raise ValueError("diag is entirely non-positive -- M appears to be the zero matrix.")

    neg_floor = -rcond * max_diag
    if np.any(diag < neg_floor):
        raise ValueError(
            f"diag contains an entry below -{rcond:.1e}*max(diag)={neg_floor:.3e} -- "
            f"M does not appear to be PSD within the declared rcond."
        )
    diag = np.maximum(diag, 0.0)

    ramp = ramp_scale * np.arange(n, dtype=np.float64) * max_diag
    threshold = rcond * max_diag

    L = np.zeros((n, rank), dtype=np.complex128)
    pivots = np.zeros(rank, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)

    n_selected = 0
    for t in range(rank):
        score = np.where(selected, -np.inf, diag + ramp)
        j = int(np.argmax(score))
        if diag[j] <= threshold:
            break

        col = np.asarray(col_eval(j))
        if col.shape != (n,):
            raise ValueError(f"col_eval({j}) must return shape ({n},), got {col.shape}.")
        col = col.astype(np.complex128)

        if t > 0:
            update = L[:, :t] @ L[j, :t].conj()
        else:
            update = 0.0
        l_t = (col - update) / np.sqrt(diag[j])
        L[:, t] = l_t

        diag = diag - np.abs(l_t) ** 2
        diag = np.maximum(diag, 0.0)

        pivots[t] = j
        selected[j] = True
        n_selected += 1

    return pivots[:n_selected], L[:, :n_selected], n_selected


def _batch_candidates(diag, selected, batch_size, mesh, min_separation, ramp):
    """Choose a deterministic, minimum-image-separated stale-diagonal batch."""
    score = np.where(selected, -np.inf, diag + ramp)
    order = np.argsort(score, kind="stable")[::-1]
    accepted = []
    coordinates = []
    mesh = np.asarray(mesh, dtype=np.int64)
    for index in order:
        if not np.isfinite(score[index]):
            break
        coordinate = np.asarray(np.unravel_index(int(index), tuple(mesh)), dtype=np.int64)
        if coordinates:
            delta = np.abs(np.asarray(coordinates) - coordinate)
            delta = np.minimum(delta, mesh - delta)
            if np.any(np.linalg.norm(delta, axis=1) < min_separation):
                continue
        accepted.append(int(index))
        coordinates.append(coordinate)
        if len(accepted) == batch_size:
            break
    return np.asarray(accepted, dtype=np.int64)


def pivoted_cholesky_batched_hermitian(
    diag, col_batch_eval, rank, *, mesh, batch_size, min_separation=2.0,
    candidate_oversampling=1, n_topup=0, rcond=1e-12, ramp_scale=1e-12,
    stage_stats=None, blocked_projection=False,
):
    """Approximate greedy selection with exact batched columns and updates.

    A stale-diagonal candidate pool of ``candidate_oversampling * batch_size``
    members is chosen, but only ``batch_size`` members are retained after exact
    within-pool re-pivoting.  Columns are exact, so oversampling changes only
    the cheap per-round column GEMM, not the streamed AO-sweep count.
    Optionally, the final ``n_topup`` pivots are selected as exact greedy
    singleton batches; this repairs final-residual order-statistic error
    without changing the preceding batched rounds.
    """
    diag = np.asarray(diag, dtype=np.float64)
    mesh = tuple(int(value) for value in mesh)
    if diag.ndim != 1 or np.prod(mesh) != diag.size:
        raise ValueError("mesh must be a positive grid shape matching diag.")
    if (
        rank <= 0 or rank > diag.size or batch_size <= 0 or min_separation < 0
        or isinstance(candidate_oversampling, bool)
        or not isinstance(candidate_oversampling, (int, np.integer))
        or candidate_oversampling <= 0
        or isinstance(n_topup, bool) or not isinstance(n_topup, (int, np.integer))
        or n_topup < 0 or n_topup > rank
    ):
        raise ValueError(
            "invalid rank, batch_size, min_separation, candidate_oversampling, or n_topup."
        )
    if stage_stats is not None and not isinstance(stage_stats, list):
        raise ValueError("stage_stats must be a list when supplied.")
    candidate_oversampling = int(candidate_oversampling)
    n_topup = int(n_topup)
    initial_max = float(np.max(diag))
    if initial_max <= 0:
        raise ValueError("diag is entirely non-positive.")
    threshold = rcond * initial_max
    ramp = ramp_scale * np.arange(diag.size, dtype=np.float64) * initial_max
    # factor L follows the metric's dtype (real-f64-L): the periodic selection
    # metric M = |S|^2/Nk is real, so its factor is float64 and this [Ng x rank]
    # array -- the dominant selection-phase memory term -- halves; a general
    # complex Hermitian metric keeps a complex128 factor unchanged. Allocated
    # lazily on the first column batch, once columns.dtype is known.
    factor = None
    selected = np.zeros(diag.size, dtype=bool)
    pivots = []
    rounds = []
    batched_rank = rank - n_topup
    while len(pivots) < batched_rank:
        retain_count = min(batch_size, batched_rank - len(pivots))
        requested = min(candidate_oversampling * retain_count, diag.size - len(pivots))
        candidates = _batch_candidates(
            diag, selected, requested, mesh, min_separation, ramp,
        )
        if candidates.size == 0 or diag[candidates[0]] <= threshold:
            break
        candidate_started = time.perf_counter()
        columns = np.asarray(col_batch_eval(candidates))
        candidate_seconds = time.perf_counter() - candidate_started
        if columns.shape != (diag.size, candidates.size):
            raise ValueError("col_batch_eval must return shape (n_grid, n_batch).")
        if factor is None:
            factor = np.zeros((diag.size, rank), dtype=columns.dtype)
        existing = factor[:, :len(pivots)]
        projection_started = time.perf_counter()
        residual_batch = columns[candidates]
        if pivots:
            residual_batch = residual_batch - existing[candidates] @ existing[candidates].conj().T
        projection_seconds = time.perf_counter() - projection_started
        local_diag = np.maximum(np.real(np.diag(residual_batch)), 0.0)
        within_batch_started = time.perf_counter()
        local_pivots, _, local_count = pivoted_cholesky_hermitian(
            local_diag, lambda index: residual_batch[:, index],
            rank=min(candidates.size, retain_count), rcond=rcond,
            ramp_scale=ramp_scale,
        )
        within_batch_seconds = time.perf_counter() - within_batch_started
        retained = []
        factor_update_seconds = 0.0
        if blocked_projection:
            # The sequential per-pivot arithmetic below, as three level-3 ops:
            #   (a) pre-round-L correction: block = columns - L @ conj(L[idx]).T
            #   (b) round-mate orthogonalization: block = V R => V = block R^-1,
            #       R being the within-batch Cholesky (R^H R = the retained
            #       sub-Gram, the factorization the sequential loop performs
            #       implicitly);
            #   (c) blocked diagonal update, diag -= sum(|V|^2, axis=1), summing
            #       retained columns in the sequential loop's order.
            # A reordering of identical arithmetic: local_pivots and the
            # candidate residual Gram are byte-identical, and the only fp delta
            # is R's Gram recursion vs sequential coefficient accumulation.
            from scipy.linalg import solve_triangular
            retained_local = []
            for local_index in local_pivots[:local_count]:
                index = int(candidates[local_index])
                if selected[index] or diag[index] <= threshold:
                    continue
                retained_local.append(int(local_index))
                selected[index] = True
                pivots.append(index)
                retained.append(index)
                if len(pivots) == rank:
                    break
            m = len(retained_local)
            if m:
                p0 = len(pivots) - m
                retained_local = np.asarray(retained_local, dtype=np.int64)
                retained_idx = candidates[retained_local]
                projection_started = time.perf_counter()
                block = np.array(columns[:, retained_local])
                if p0:
                    factor_L = factor[:, :p0]
                    block -= factor_L @ factor_L[retained_idx].conj().T
                projection_seconds += time.perf_counter() - projection_started
                factor_update_started = time.perf_counter()
                gram = residual_batch[np.ix_(retained_local, retained_local)]
                upper_R = np.linalg.cholesky(gram).conj().T
                block_factor = solve_triangular(
                    upper_R.T, block.T, lower=True,
                ).T
                factor[:, p0:p0 + m] = block_factor
                diag = np.maximum(
                    diag - np.sum(np.abs(block_factor) ** 2, axis=1), 0.0,
                )
                factor_update_seconds += time.perf_counter() - factor_update_started
            existing = factor[:, :len(pivots)]
        else:
            for local_index in local_pivots[:local_count]:
                index = int(candidates[local_index])
                if selected[index] or diag[index] <= threshold:
                    continue
                projection_started = time.perf_counter()
                correction = existing @ existing[index].conj() if pivots else 0.0
                projection_seconds += time.perf_counter() - projection_started
                factor_update_started = time.perf_counter()
                vector = (columns[:, local_index] - correction) / np.sqrt(diag[index])
                factor[:, len(pivots)] = vector
                diag = np.maximum(diag - np.abs(vector) ** 2, 0.0)
                factor_update_seconds += time.perf_counter() - factor_update_started
                selected[index] = True
                pivots.append(index)
                retained.append(index)
                existing = factor[:, :len(pivots)]
                if len(pivots) == rank:
                    break
        rounds.append({
            "requested_candidates": candidates.tolist(),
            "within_batch_pivots": [int(candidates[index]) for index in local_pivots[:local_count]],
            "retained_pivots": retained,
        })
        if stage_stats is not None:
            stage_stats.append({
                "stage": "batched", "round_index": len(rounds) - 1,
                "candidate_count": int(candidates.size), "retained_count": len(retained),
                "candidate_eval_seconds": candidate_seconds,
                "projection_seconds": projection_seconds,
                "within_batch_pivot_seconds": within_batch_seconds,
                "factor_update_seconds": factor_update_seconds,
            })
        if not retained:
            break
    topup_start_max_index = None
    topup_start_was_last_round_rejected = None
    if len(pivots) < rank and n_topup:
        topup_start_max_index = int(np.argmax(np.where(selected, -np.inf, diag + ramp)))
        if rounds:
            last_round = rounds[-1]
            topup_start_was_last_round_rejected = (
                topup_start_max_index in last_round["requested_candidates"]
                and topup_start_max_index not in last_round["retained_pivots"]
            )
    topup_pivots = []
    while len(pivots) < rank:
        score = np.where(selected, -np.inf, diag + ramp)
        index = int(np.argmax(score))
        if diag[index] <= threshold:
            break
        candidate_started = time.perf_counter()
        column = np.asarray(col_batch_eval(np.asarray([index], dtype=np.int64)))
        candidate_seconds = time.perf_counter() - candidate_started
        if column.shape != (diag.size, 1):
            raise ValueError("col_batch_eval singleton must return shape (n_grid, 1).")
        if factor is None:
            factor = np.zeros((diag.size, rank), dtype=column.dtype)
        existing = factor[:, :len(pivots)]
        projection_started = time.perf_counter()
        correction = existing @ existing[index].conj() if pivots else 0.0
        projection_seconds = time.perf_counter() - projection_started
        factor_update_started = time.perf_counter()
        vector = (column[:, 0] - correction) / np.sqrt(diag[index])
        factor[:, len(pivots)] = vector
        diag = np.maximum(diag - np.abs(vector) ** 2, 0.0)
        factor_update_seconds = time.perf_counter() - factor_update_started
        selected[index] = True
        pivots.append(index)
        topup_pivots.append(index)
        if stage_stats is not None:
            stage_stats.append({
                "stage": "topup", "round_index": len(topup_pivots) - 1,
                "candidate_count": 1, "retained_count": 1,
                "candidate_eval_seconds": candidate_seconds,
                "projection_seconds": projection_seconds,
                "within_batch_pivot_seconds": 0.0,
                "factor_update_seconds": factor_update_seconds,
            })
    if n_topup:
        rounds.append({
            "mode": "exact_topup",
            "n_requested": n_topup,
            "topup_pivots": topup_pivots,
            "pre_topup_max_index": topup_start_max_index,
            "pre_topup_max_was_last_round_rejected": topup_start_was_last_round_rejected,
        })
    if factor is None:
        # No column was ever evaluated -> nothing selected (empty first batch /
        # exhausted diag). No metric dtype was observed, so don't guess one:
        # return an empty (Ng, 0) factor, matching the pre-real-f64-L empty
        # return exactly. len(pivots) is 0 here.
        return (np.asarray(pivots, dtype=np.int64),
                np.zeros((diag.size, 0), dtype=np.complex128), len(pivots), rounds)
    return np.asarray(pivots, dtype=np.int64), factor[:, :len(pivots)], len(pivots), rounds


def build_pi_eta(X, ao_blocks, phase, neg, *, imag_tol=1e-10):
    """Build Pi^q = pair_convolve(X, X)[q] and eta^q = pair_convolve(X, AO)[q].
    eta is accumulated block-by-block so one pair_convolve call holds only
    one block of AO data. See design doc §4-§5.

    Alg. 1's convolution and Eq. 4/5's defining equations for Pi/eta agree
    only up to a q<->-q relabeling, so both outputs' q-axis is relabeled with
    neg once after construction. Every consumer then receives the physical
    Pi^q/eta^q and needs no convention logic of its own. The relabeling is
    deliberately not inside pair_convolve, which is shared and correct as is.
    The offset is invisible at self-paired q, and invisible in Pi against a
    transpose-based oracle check, so it must be preserved by construction
    rather than by test (task #25).

    Args:
        X: (Nk, Nip, Nao) complex128 across the canonical k-mesh.
        ao_blocks: (Nk, Ng, Nao) complex128 array, or iterable of
            (Nk, blk_i, Nao) blocks on the SAME canonical k-mesh.
        phase: (Nk, Nk) unitary matrix (KptsMesh.phase).
        neg: (Nk,) int array (KptsMesh.neg), used to relabel both
            outputs' q-axis.

    Returns:
        (Pi, eta): (Nk, Nip, Nip) and (Nk, Nip, Ng) complex128.
    """
    # Local import: kpts.py stays a leaf.
    from pytc.pbc.df.kpts import pair_convolve

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (Nk, Nip, Nao), got shape {X.shape}.")
    neg = np.asarray(neg)
    if neg.shape != (X.shape[0],):
        raise ValueError(f"neg must have shape ({X.shape[0]},), got {neg.shape}.")

    Pi = pair_convolve(X, X, phase, imag_tol=imag_tol)[neg]

    if isinstance(ao_blocks, np.ndarray):
        ao_blocks = [ao_blocks]
    else:
        ao_blocks = list(ao_blocks)
    if not ao_blocks:
        raise ValueError("ao_blocks must be nonempty.")

    # Preallocate and fill rather than concatenate. The previous form,
    #   np.concatenate([...], axis=2)[neg]
    # held the whole chunk list, the concatenate's full copy, and the
    # fancy-index's full copy simultaneously -- a 3x eta transient that OOM'd
    # a 333 build at 622 GiB before it could reach the third copy. Filling in
    # place holds one eta plus one block.
    #
    # [neg] is applied per block because it commutes with the concatenation:
    # it permutes axis 0 (k) while blocks concatenate along axis 2 (grid).
    # Same reasoning build_pi_eta_staged relies on.
    n_grid_total = sum(int(np.asarray(block).shape[1]) for block in ao_blocks)
    eta = None
    col = 0
    for block in ao_blocks:
        Z = pair_convolve(X, np.asarray(block), phase, imag_tol=imag_tol)[neg]
        if eta is None:
            eta = np.empty((Z.shape[0], Z.shape[1], n_grid_total), dtype=Z.dtype)
        eta[:, :, col:col + Z.shape[2]] = Z
        col += int(Z.shape[2])
        del Z
    if col != n_grid_total:
        raise ValueError(
            f"eta covered {col} grid points, expected {n_grid_total} -- the AO "
            f"block stream did not span the grid."
        )
    return Pi, eta


def build_pi_eta_staged(X, ao_blocks, phase, neg, *, staging_path, n_grid,
                        staging_block=4096, imag_tol=1e-10,
                        free_bytes_safety=1.25, additional_reserve_bytes=0):
    """build_pi_eta with eta written to a (Nk, Nip, Ng) C-order memmap rather
    than held in RAM, so only one q's contiguous slab need be resident.

    The q axis is not separable (pair_convolve couples all k), so eta is still
    produced all-q-per-grid-block; this stages the assembled array without
    restructuring the math. Writes are buffered to `staging_block` grid points,
    decoupling the write run length from the AO block size.

    The caller owns the staged file and must unlink it. Returns
    (Pi, eta_memmap, stats). Costs and design:
    docs/isdf-periodic/task46_phaseB_eta_staging_spec.md.
    """
    from pytc.pbc.df.kpts import pair_convolve

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (Nk, Nip, Nao), got shape {X.shape}.")
    neg = np.asarray(neg)
    if neg.shape != (X.shape[0],):
        raise ValueError(f"neg must have shape ({X.shape[0]},), got {neg.shape}.")
    n_kpts, n_ip = int(X.shape[0]), int(X.shape[1])
    n_grid = int(n_grid)
    if n_grid <= 0 or int(staging_block) <= 0:
        raise ValueError("n_grid and staging_block must be positive.")

    Pi = pair_convolve(X, X, phase, imag_tol=imag_tol)[neg]

    # Fail closed before writing: never start a stage we cannot finish.
    # Reserve the concurrent peak so the refusal precedes the large write.
    predicted = n_kpts * n_ip * n_grid * np.dtype(np.complex128).itemsize
    required = predicted + int(additional_reserve_bytes)
    staging_dir = os.path.dirname(os.path.abspath(staging_path)) or "."
    stat = os.statvfs(staging_dir)
    free = stat.f_bavail * stat.f_frsize
    if free < required * float(free_bytes_safety):
        raise OSError(
            f"eta staging REFUSED: {staging_dir} has {free / 2**30:.1f} GiB free, "
            f"needs {required * float(free_bytes_safety) / 2**30:.1f} GiB "
            f"(eta {predicted / 2**30:.1f} + reserved "
            f"{int(additional_reserve_bytes) / 2**30:.1f} GiB, x{free_bytes_safety})."
        )

    # One 3-D array is a single block; an iterable is streamed, never list()-ed.
    if isinstance(ao_blocks, np.ndarray):
        ao_blocks = [ao_blocks]

    eta = np.memmap(staging_path, dtype=np.complex128, mode="w+",
                    shape=(n_kpts, n_ip, n_grid))
    pending, pending_cols, col0 = [], 0, 0
    write_seconds, bytes_written = 0.0, 0

    def _flush(pending, pending_cols, col0):
        if not pending:
            return col0, 0.0, 0
        chunk = pending[0] if len(pending) == 1 else np.concatenate(pending, axis=2)
        started = time.perf_counter()
        eta[:, :, col0:col0 + pending_cols] = chunk
        return col0 + pending_cols, time.perf_counter() - started, chunk.nbytes

    for block in ao_blocks:
        # [neg] commutes with the concatenation: it permutes axis 0.
        Z = pair_convolve(X, np.asarray(block), phase, imag_tol=imag_tol)[neg]
        pending.append(Z)
        pending_cols += int(Z.shape[2])
        if pending_cols >= int(staging_block):
            col0, dt, nb = _flush(pending, pending_cols, col0)
            write_seconds += dt
            bytes_written += nb
            pending, pending_cols = [], 0
    col0, dt, nb = _flush(pending, pending_cols, col0)
    write_seconds += dt
    bytes_written += nb

    if col0 != n_grid:
        raise ValueError(
            f"staged eta covered {col0} grid points, expected {n_grid} -- the AO "
            f"block stream did not span the grid."
        )
    started = time.perf_counter()
    eta.flush()
    write_seconds += time.perf_counter() - started

    stats = {
        "staging_path": str(staging_path),
        "staged_bytes": int(bytes_written),
        "staging_block": int(staging_block),
        "write_run_bytes": int(staging_block) * np.dtype(np.complex128).itemsize,
        "write_seconds": float(write_seconds),
        "write_gb_per_s": (float(bytes_written) / 1e9 / write_seconds
                           if write_seconds > 0 else None),
        "free_bytes_before": int(free),
    }
    return Pi, eta, stats


def apply_raw_kernel_and_solve(
    Pi_q, eta_q, *, cell, q_kpt, grid_coords, grid_mesh, rtol=1e-4, self_paired=False
):
    """Apply the "raw" (bare 4pi/G^2, exx=False) periodic Coulomb kernel to
    eta^q over the spatial grid, contract to (Nip, Nip), and solve the
    Hermitian sandwich for W^q. Plain NumPy, single q -- the CPU oracle for
    the device KernelProvider path. See design doc §5, §7.

        lq     = eta_q * exp(-1j * grid_coords @ q_kpt)   # Bloch phase
        wq     = FFT(lq, grid_mesh)
        vq     = coulG(q, exx=False) * vol / Ng
        rq     = conj(IFFT(wq * vq, grid_mesh))
        kern_q = lq @ rq.T / sqrt(Ng)
        W_q    = sqrt(Ng) * hermitian_sandwich_solve(Pi_q, kern_q)[0]

    The final sqrt(Ng) rescale exactly cancels kern_q's 1/sqrt(Ng)
    (paper Eq. 10 factor placements; verified in the V2 reference-replay
    test). exxdiv is NEVER applied here -- it is owned by a later get_k
    post-processing step.

    Args:
        Pi_q: (Nip, Nip) complex128 metric.
        eta_q: (Nip, Ng) complex128 RHS.
        q_kpt: (3,) absolute k-vector for this q.
        grid_coords: (Ng, 3), same flattened order as eta_q's grid axis.
        grid_mesh: (3,) positive ints, real-space integration mesh
            (distinct from the k-point mesh); prod must equal Ng.
        rtol: forwarded to hermitian_sandwich_solve.
        self_paired: True when neg[q]==q. Physics requires both Pi_q and
            kern_q real for such q, but the complex intermediates leave
            floating-point imaginary noise that the near-singular solve
            amplifies; when True, Pi_q.real and kern_q.real are taken
            BEFORE the solve (noise projection, not a loosened gate). See
            design doc §5.

    Returns:
        (W_q, kern_q, solve_info): W_q (Nip, Nip) complex128; kern_q is
        the raw contracted kernel before the solve; solve_info is
        hermitian_sandwich_solve's info dict.
    """
    from pyscf.pbc import tools as pbctools

    from pytc.df.solvers import hermitian_sandwich_solve

    Pi_q = np.asarray(Pi_q)
    eta_q = np.asarray(eta_q)
    n_ip = Pi_q.shape[0]
    if Pi_q.shape != (n_ip, n_ip):
        raise ValueError(f"Pi_q must be square, got shape {Pi_q.shape}.")
    if eta_q.ndim != 2 or eta_q.shape[0] != n_ip:
        raise ValueError(f"eta_q must have shape ({n_ip},Ng), got {eta_q.shape}.")
    n_grid = eta_q.shape[1]
    if self_paired:
        Pi_q = Pi_q.real.astype(np.complex128)

    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords.shape != (n_grid, 3):
        raise ValueError(
            f"grid_coords must have shape ({n_grid},3) matching eta_q's grid axis, "
            f"got {grid_coords.shape}."
        )
    grid_mesh_t = tuple(int(x) for x in grid_mesh)
    if len(grid_mesh_t) != 3 or any(m <= 0 for m in grid_mesh_t):
        raise ValueError(f"grid_mesh must be 3 positive ints, got {grid_mesh_t}.")
    if int(np.prod(grid_mesh_t)) != n_grid:
        raise ValueError(
            f"prod(grid_mesh)={int(np.prod(grid_mesh_t))} != eta_q's grid size {n_grid}."
        )

    q_kpt = np.asarray(q_kpt, dtype=np.float64)
    if q_kpt.shape != (3,):
        raise ValueError(f"q_kpt must have shape (3,), got {q_kpt.shape}.")

    phase = np.exp(-1j * (grid_coords @ q_kpt))
    lq = eta_q * phase[None, :]

    lq_mesh = lq.reshape((n_ip,) + grid_mesh_t)
    wq_mesh = np.fft.fftn(lq_mesh, axes=(1, 2, 3), norm="backward")

    Gv = cell.get_Gv(list(grid_mesh_t))
    vq = pbctools.get_coulG(cell, k=q_kpt, exx=False, Gv=Gv, mesh=list(grid_mesh_t))
    vq = vq * (cell.vol / n_grid)
    vq_mesh = vq.reshape(grid_mesh_t)

    rq_mesh = np.fft.ifftn(wq_mesh * vq_mesh[None, :, :, :], axes=(1, 2, 3), norm="backward")
    rq = rq_mesh.reshape(n_ip, n_grid).conj()

    kern_q = (lq @ rq.T) / np.sqrt(n_grid)
    kern_q = np.asarray(kern_q, dtype=np.complex128)
    if self_paired:
        kern_q = kern_q.real.astype(np.complex128)

    W_q_unscaled, solve_info = hermitian_sandwich_solve(Pi_q, kern_q, rtol=rtol)
    # sqrt(Ng) rescale cancels kern_q's own 1/sqrt(Ng) (Eq. 10 factor placement).
    W_q = np.sqrt(n_grid) * W_q_unscaled
    return W_q, kern_q, solve_info


# ---------------------------------------------------------------------------
# Device path, KernelProvider protocol (design doc §7).
#
# KernelProvider.apply(q_index, lq) is a LINEAR q-momentum kernel operator on
# a PRE-PHASED (Nip, Ng) slab, returning v_q BEFORE the final conjugate; the
# Bloch-phase multiply and outer conjugate are pipeline glue, not part of the
# contract. vol/Ng normalization stays INSIDE the provider; exxdiv stays
# OUTSIDE (owned by get_k post-processing).
#
# Dagger law at this seam: apply(neg[q], conj(l_q)) == conj(apply(q, l_q)),
# given l_q[neg[q]] = conj(l_q[q]). Verified in test_raw_kernel_apply_dagger_law.


@partial(jax.jit, static_argnames=("grid_mesh",))
def _raw_kernel_apply_core(lq, coulG_scaled, grid_mesh):
    """Jitted core of the "raw" provider: v_q = IFFT(coulG_scaled * FFT(lq)).
    No validation (host wrapper's job), no phase multiply, no outer
    conjugate."""
    n_ip = lq.shape[0]
    lq_mesh = lq.reshape((n_ip,) + grid_mesh)
    wq_mesh = jnp.fft.fftn(lq_mesh, axes=(1, 2, 3))
    vq_mesh = jnp.asarray(coulG_scaled, dtype=lq.dtype).reshape(grid_mesh)
    vq_mesh = wq_mesh * vq_mesh[None, :, :, :]
    rq_mesh = jnp.fft.ifftn(vq_mesh, axes=(1, 2, 3))
    return rq_mesh.reshape(n_ip, -1)


def raw_kernel_apply(lq, *, cell, q_kpt, grid_mesh):
    """Host-side wrapper: validate, compute coulG(q)*vol/Ng via pyscf (not
    jittable), dispatch to the jitted core.

    Args:
        lq: (Nip, Ng) complex128, ALREADY Bloch-phase-corrected.
        grid_mesh: (3,) positive ints; prod must equal Ng.

    Returns:
        v_q: (Nip, Ng) complex128 jax array, BEFORE the outer conjugate.
    """
    from pyscf.pbc import tools as pbctools

    lq_np = np.asarray(lq)
    if lq_np.ndim != 2:
        raise ValueError(f"lq must be 2-D (Nip, Ng), got shape {lq_np.shape}.")
    n_ip, n_grid = lq_np.shape

    grid_mesh_t = tuple(int(x) for x in grid_mesh)
    if len(grid_mesh_t) != 3 or any(m <= 0 for m in grid_mesh_t):
        raise ValueError(f"grid_mesh must be 3 positive ints, got {grid_mesh_t}.")
    if int(np.prod(grid_mesh_t)) != n_grid:
        raise ValueError(f"prod(grid_mesh)={int(np.prod(grid_mesh_t))} != lq's grid size {n_grid}.")

    q_kpt_np = np.asarray(q_kpt, dtype=np.float64)
    if q_kpt_np.shape != (3,):
        raise ValueError(f"q_kpt must have shape (3,), got {q_kpt_np.shape}.")

    Gv = cell.get_Gv(list(grid_mesh_t))
    coulG = pbctools.get_coulG(cell, k=q_kpt_np, exx=False, Gv=Gv, mesh=list(grid_mesh_t))
    coulG_scaled = np.asarray(coulG, dtype=np.float64) * (cell.vol / n_grid)

    lq_jnp = jnp.asarray(lq_np, dtype=jnp.complex128)
    if lq_jnp.dtype != jnp.complex128:
        logger.warning(
            f"raw_kernel_apply: resolved dtype is {lq_jnp.dtype}, not complex128 -- JAX "
            f"defaults to complex64 SILENTLY unless the caller has enabled "
            f"jax.config.update('jax_enable_x64', True). This device path is defined only "
            f"at the c128 parity tier (design v2.1 section 1); verify x64 is enabled "
            f"before trusting production numbers from this path."
        )
    return _raw_kernel_apply_core(lq_jnp, jnp.asarray(coulG_scaled), grid_mesh_t)


def precompute_coulG_all_q(cell, canonical_kpts, grid_mesh):
    """Precompute coulG(q)*vol/Ng for every q at once (host-only pyscf
    calls are not jittable; the result is a per-q constant that can then
    be threaded into a jitted core as a traced argument).

    Returns:
        coulG_all: (Nk, Ng) float64 jax array.
    """
    from pyscf.pbc import tools as pbctools

    canonical_kpts_np = np.asarray(canonical_kpts, dtype=np.float64)
    if canonical_kpts_np.ndim != 2 or canonical_kpts_np.shape[1] != 3:
        raise ValueError(
            f"canonical_kpts must have shape (Nk,3), got {canonical_kpts_np.shape}."
        )
    grid_mesh_t = tuple(int(x) for x in grid_mesh)
    if len(grid_mesh_t) != 3 or any(m <= 0 for m in grid_mesh_t):
        raise ValueError(f"grid_mesh must be 3 positive ints, got {grid_mesh_t}.")

    n_grid = int(np.prod(grid_mesh_t))
    Gv = cell.get_Gv(list(grid_mesh_t))
    n_kpts = canonical_kpts_np.shape[0]
    coulG_all = np.empty((n_kpts, n_grid), dtype=np.float64)
    for q in range(n_kpts):
        coulG = pbctools.get_coulG(
            cell, k=canonical_kpts_np[q], exx=False, Gv=Gv, mesh=list(grid_mesh_t)
        )
        coulG_all[q] = np.asarray(coulG, dtype=np.float64) * (cell.vol / n_grid)
    return jnp.asarray(coulG_all)


@partial(jax.jit, static_argnames=("grid_mesh", "self_paired", "retention_mode"))
def _fused_apply_kernel_and_solve_core(
    Pi_q, eta_q, phase_q, coulG_scaled_q, grid_mesh, rtol, self_paired, retention_mode="single"
):
    """Fully fused single-jax.jit per-q hot path: phase-multiply -> raw
    kernel apply -> conjugate -> ZGEMM -> Hermitian sandwich solve, one XLA
    graph with no host round trips. Reachable only via a provider exposing
    fused_apply_and_solve. See design doc §6."""
    n_grid = eta_q.shape[1]
    lq = eta_q * phase_q[None, :]
    v_q = _raw_kernel_apply_core(lq, coulG_scaled_q, grid_mesh)
    rq = jnp.conj(v_q)
    kern_q = (lq @ rq.T) / jnp.sqrt(n_grid)
    if self_paired:
        kern_q = kern_q.real.astype(jnp.complex128)

    from pytc.df.solvers import _hermitian_sandwich_solve_core

    (
        W, n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
        v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
    ) = _hermitian_sandwich_solve_core(Pi_q, kern_q, rtol, retention_mode)

    return (
        W, kern_q, n_retained, s_max, s_min_retained,
        pi_anti_hermitian_residual, v_anti_hermitian_residual,
        retained_solve_residual, truncation_residual,
    )


@dataclasses.dataclass(frozen=True)
class RawKernelProvider:
    """The "raw" (bare 4pi/G^2, exx=False) KernelProvider (design doc §7):
    apply(q_index, lq) -> v_q plus provenance(). q_index resolves the
    absolute k-vector from canonical_kpts internally.

    fused_apply_and_solve is an OPTIONAL fast-path hook that
    apply_kernel_and_solve_device prefers when present; providers without
    it fall back to the eager per-stage path.

    Args:
        canonical_kpts: (Nk, 3) float64, e.g. KptsMesh.canonical_kpts.
        grid_mesh: (3,) positive ints, real-space integration mesh.
    """
    cell: object
    canonical_kpts: object
    grid_mesh: tuple

    def __post_init__(self):
        canonical_kpts = np.asarray(self.canonical_kpts, dtype=np.float64)
        if canonical_kpts.ndim != 2 or canonical_kpts.shape[1] != 3:
            raise ValueError(
                f"canonical_kpts must have shape (Nk,3), got {canonical_kpts.shape}."
            )
        grid_mesh = tuple(int(x) for x in self.grid_mesh)
        if len(grid_mesh) != 3 or any(m <= 0 for m in grid_mesh):
            raise ValueError(f"grid_mesh must be 3 positive ints, got {grid_mesh}.")
        object.__setattr__(self, "canonical_kpts", canonical_kpts)
        object.__setattr__(self, "grid_mesh", grid_mesh)
        object.__setattr__(
            self, "coulG_all", precompute_coulG_all_q(self.cell, canonical_kpts, grid_mesh)
        )

    def apply(self, q_index, lq):
        n_kpts = self.canonical_kpts.shape[0]
        if not (0 <= q_index < n_kpts):
            raise ValueError(f"q_index={q_index} out of range for {n_kpts} k-points.")
        return raw_kernel_apply(
            lq, cell=self.cell, q_kpt=self.canonical_kpts[q_index], grid_mesh=self.grid_mesh
        )

    def fused_apply_and_solve(
        self, q_index, Pi_q, eta_q, phase_q, rtol, self_paired, retention_mode="single"
    ):
        n_kpts = self.canonical_kpts.shape[0]
        if not (0 <= q_index < n_kpts):
            raise ValueError(f"q_index={q_index} out of range for {n_kpts} k-points.")
        return _fused_apply_kernel_and_solve_core(
            Pi_q, eta_q, phase_q, self.coulG_all[q_index], self.grid_mesh, rtol, self_paired,
            retention_mode,
        )

    def provenance(self):
        return {
            "kernel_name": "raw",
            "kernel_version": 1,
            "g0_convention": "pyscf_get_coulG_exx_false",
            "grid_mesh": self.grid_mesh,
            "normalization": "vol_over_ng_inside_provider",
            "exxdiv": "owned_by_get_k_postprocessing_not_this_provider",
        }


@jax.jit
def _precompute_phase_all_q_core(grid_coords, canonical_kpts):
    """Jitted batched core: per-q Bloch phase exp(-1j * grid_coords @ q_kpt)
    for every q from a single grid_coords upload."""
    return jnp.exp(-1j * (grid_coords @ canonical_kpts.T)).T


def precompute_phase_all_q(grid_coords, canonical_kpts):
    """Compute the per-q Bloch phase for every q in one batched jitted
    call; slice per-q into apply_kernel_and_solve_device's phase_q.

    Returns:
        phase_all: (Nk, Ng) complex128 jax array.
    """
    grid_coords_np = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords_np.ndim != 2 or grid_coords_np.shape[1] != 3:
        raise ValueError(f"grid_coords must have shape (Ng,3), got {grid_coords_np.shape}.")
    canonical_kpts_np = np.asarray(canonical_kpts, dtype=np.float64)
    if canonical_kpts_np.ndim != 2 or canonical_kpts_np.shape[1] != 3:
        raise ValueError(
            f"canonical_kpts must have shape (Nk,3), got {canonical_kpts_np.shape}."
        )
    return _precompute_phase_all_q_core(
        jnp.asarray(grid_coords_np), jnp.asarray(canonical_kpts_np)
    )


def apply_kernel_and_solve_device(
    provider, q_index, Pi_q, eta_q, *, grid_coords=None, phase_q=None, rtol=1e-4,
    retained_solve_residual_gate=1e-10, self_paired=False, retention_mode="single",
    kern_blocking=None,
):
    """S4 pipeline glue, device-resident, provider-agnostic: phase multiply
    -> provider.apply -> conjugate -> ZGEMM -> device Hermitian sandwich
    solve. Matches apply_raw_kernel_and_solve's math for RawKernelProvider.
    See design doc §6.

    Args:
        provider: object exposing .apply(q_index, lq) -> v_q (Nip, Ng).
        Pi_q: (Nip, Nip) complex128 metric.
        eta_q: (Nip, Ng) complex128 RHS, NOT yet phase-corrected.
        grid_coords: (Ng, 3); required only when phase_q is not given.
        phase_q: (Ng,) complex128 precomputed
            exp(-1j * grid_coords @ canonical_kpts[q_index]); callers
            looping over q should precompute via precompute_phase_all_q.
            Exactly one of grid_coords/phase_q must be given.
        rtol: forwarded to the device sandwich solve.
        retained_solve_residual_gate: HARD host-side gate on
            solve_info["retained_solve_residual"]; also hard-fails on
            n_retained == 0 (the jitted solve cannot raise on traced
            values, so degradation is turned into an error here).
        self_paired: True when neg[q]==q; Pi_q.real and kern_q.real are
            taken before the solve (see apply_raw_kernel_and_solve).
        retention_mode: "single" (default) or "pairwise" -- forwarded to
            hermitian_sandwich_solve_device / the fused core. See
            hermitian_sandwich_solve's docstring for the two modes.

    Returns:
        (W_q, kern_q, solve_info): W_q (Nip, Nip) complex128 jax array;
        kern_q the raw contracted kernel; solve_info the solve info dict.
    """
    from pytc.df.solvers import _solve_info_from_core_output, hermitian_sandwich_solve_device

    n_ip, n_grid = eta_q.shape

    if phase_q is None and grid_coords is None:
        raise ValueError("apply_kernel_and_solve_device: give one of grid_coords/phase_q.")
    if phase_q is not None:
        phase = jnp.asarray(phase_q, dtype=jnp.complex128)
        if phase.shape != (n_grid,):
            raise ValueError(
                f"phase_q must have shape ({n_grid},) matching eta_q's grid axis, "
                f"got {phase.shape}."
            )
    else:
        q_kpt = provider.canonical_kpts[q_index]
        grid_coords_np = np.asarray(grid_coords, dtype=np.float64)
        if grid_coords_np.shape != (n_grid, 3):
            raise ValueError(
                f"grid_coords must have shape ({n_grid},3) matching eta_q's grid axis, "
                f"got {grid_coords_np.shape}."
            )
        phase = jnp.exp(-1j * (jnp.asarray(grid_coords_np) @ jnp.asarray(q_kpt)))

    eta_q_jnp = jnp.asarray(eta_q, dtype=jnp.complex128)
    Pi_q_jnp = jnp.asarray(Pi_q, dtype=jnp.complex128)
    # Fail closed at the boundary common to BOTH the fused and eager solve
    # paths: with jax_enable_x64 off, JAX silently downcasts the
    # complex128 casts above to complex64, the device solve runs in single
    # precision (~1e-5 accuracy), and W is silently corrupted -- surfacing only
    # as an opaque trip of the 1e-10 machine-tier retained-solve gate below.
    # The fused path never reaches hermitian_sandwich_solve_device's guard, so
    # the check must live here.
    if eta_q_jnp.dtype != jnp.complex128 or Pi_q_jnp.dtype != jnp.complex128:
        raise ValueError(
            f"apply_kernel_and_solve_device: q_index={q_index} resolved to dtype "
            f"{Pi_q_jnp.dtype}, not complex128 -- jax_enable_x64 is off, so JAX "
            f"silently downcast to complex64 and the device solve would run in single "
            f"precision (~1e-5 accuracy), corrupting W and tripping the 1e-10 "
            f"machine-tier gate downstream. Call jax.config.update('jax_enable_x64', "
            f"True) before building (design v2.1 section 1 fixes c128 as the only tier "
            f"with defined 1e-6-class gates)."
        )
    if self_paired:
        Pi_q_jnp = Pi_q_jnp.real.astype(jnp.complex128)

    # Providers with a fused fast path run the whole chain as one jax.jit
    # graph; others fall back to the eager per-stage path below.
    if kern_blocking is not None:
        # Bypasses the fused graph, which assumes a resident (Nip, Ng).
        kern_q = jnp.asarray(
            build_kern_q_blocked(
                provider, q_index, np.asarray(eta_q), np.asarray(phase),
                self_paired=self_paired, **kern_blocking),
            dtype=jnp.complex128)
        W_q_unscaled, solve_info = hermitian_sandwich_solve_device(
            Pi_q_jnp, kern_q, rtol=rtol, retention_mode=retention_mode)
    elif (fused := getattr(provider, "fused_apply_and_solve", None)) is not None:
        (
            W_q_unscaled, kern_q, n_retained, s_max, s_min_retained,
            pi_anti_hermitian_residual, v_anti_hermitian_residual,
            retained_solve_residual, truncation_residual,
        ) = fused(q_index, Pi_q_jnp, eta_q_jnp, phase, rtol, self_paired, retention_mode)

        solve_info = _solve_info_from_core_output(
            n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
            v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
            n_ip, W_q_unscaled.dtype, rtol, caller="apply_kernel_and_solve_device[fused]",
            retention_mode=retention_mode,
        )
    else:
        lq = eta_q_jnp * phase[None, :]

        v_q = provider.apply(q_index, lq)
        rq = jnp.conj(v_q)

        kern_q = (lq @ rq.T) / jnp.sqrt(n_grid)
        if self_paired:
            kern_q = kern_q.real.astype(jnp.complex128)

        W_q_unscaled, solve_info = hermitian_sandwich_solve_device(
            Pi_q_jnp, kern_q, rtol=rtol, retention_mode=retention_mode
        )

    # Host-side gate: the jitted solve cannot raise on a traced value, so
    # degeneracy becomes a precise, q-indexed error here (not a silent W_q=0).
    if solve_info["n_retained"] == 0:
        raise ValueError(
            f"apply_kernel_and_solve_device: q_index={q_index} retained ZERO modes of "
            f"Pi_q in the device sandwich solve (Pi_q is non-PSD, the zero matrix, or "
            f"rtol={rtol} is too large) -- W_q would be silently zero; refusing to "
            f"proceed. Validate Pi_q against the NumPy oracle (hermitian_sandwich_solve) "
            f"for a precise diagnosis."
        )
    if solve_info["retained_solve_residual"] > retained_solve_residual_gate:
        raise ValueError(
            f"apply_kernel_and_solve_device: q_index={q_index} retained-space solve "
            f"residual {solve_info['retained_solve_residual']:.3e} exceeds the hard "
            f"machine-tier gate {retained_solve_residual_gate:.1e} (design v2.1 section "
            f"5) -- this is a numerical sanity check on the eigendecomposition/solve "
            f"arithmetic itself, not the (separately reported, ungated here) truncation "
            f"residual; something is wrong with this q's Pi_q/kern_q inputs or dtype."
        )

    W_q = jnp.sqrt(n_grid) * W_q_unscaled
    return W_q, kern_q, solve_info


def build_kern_q_blocked(provider, q_index, eta_q, phase, *, staging_root,
                         row_block=2048, grid_chunk=4096, self_paired=False):
    """kern_q without holding a full (Nip, Ng) array.

    Pass 1 row-blocks the transform (the FFT's axis 0 is a batch axis) and stages
    rq; pass 2 accumulates kern_q over grid chunks. Peak is
    max(2*row_block*Ng, 2*Nip*grid_chunk) + Nip^2. Design and costs:
    docs/isdf-periodic/task46_phaseB_solve_blocking_spec.md.
    """
    n_ip, n_grid = int(eta_q.shape[0]), int(eta_q.shape[1])
    phase = np.asarray(phase, dtype=np.complex128)
    itemsize = np.dtype(np.complex128).itemsize
    predicted = n_ip * n_grid * itemsize
    stat = os.statvfs(staging_root)
    free = stat.f_bavail * stat.f_frsize
    if free < predicted * 1.25:
        raise OSError(
            f"rq staging REFUSED for q={int(q_index)}: {staging_root} has "
            f"{free / 2**30:.1f} GiB free, needs {predicted * 1.25 / 2**30:.1f} GiB."
        )

    rq_path = os.path.join(
        staging_root, f"isdf_rq_q{int(q_index)}_{os.getpid()}.dat")
    rq = None
    try:
        rq = np.memmap(rq_path, dtype=np.complex128, mode="w+",
                       shape=(n_ip, n_grid))
        for r0 in range(0, n_ip, int(row_block)):
            r1 = min(r0 + int(row_block), n_ip)
            lq_rows = np.asarray(eta_q[r0:r1], dtype=np.complex128) * phase[None, :]
            rq[r0:r1] = np.conj(np.asarray(provider.apply(q_index, lq_rows)))
        rq.flush()

        kern = np.zeros((n_ip, n_ip), dtype=np.complex128)
        for g0 in range(0, n_grid, int(grid_chunk)):
            g1 = min(g0 + int(grid_chunk), n_grid)
            lq_c = (np.asarray(eta_q[:, g0:g1], dtype=np.complex128)
                    * phase[None, g0:g1])
            kern += lq_c @ np.asarray(rq[:, g0:g1]).T
        kern /= np.sqrt(n_grid)
        if self_paired:
            kern = kern.real.astype(np.complex128)
        return kern
    finally:
        rq = None
        gc.collect()
        try:
            os.unlink(rq_path)
        except FileNotFoundError:
            pass


def build_coul_kpt_device(provider, Pi, eta, grid_coords, mesh_obj, *, rtol=1e-4,
                           retained_solve_residual_gate=1e-10, retention_mode="single",
                           kern_blocking=None):
    """S4 orchestration: build coul_kpt (Nk, Nip, Nip) with one
    apply_kernel_and_solve_device call per unique {q, neg[q]} pair; the
    partner is set by exact conjugation (W[neg[q]] = conj(W[q]),
    kern[neg[q]] = conj(kern[q])). See design doc §6; verified against
    independent neg[q] builds in
    test_build_coul_kpt_device_conjugate_shortcut_matches_independent_build.

    Args:
        provider: KernelProvider built against mesh_obj.canonical_kpts.
        Pi: (Nk, Nip, Nip) complex128.
        eta: (Nk, Nip, Ng) complex128.
        grid_coords: (Ng, 3).
        mesh_obj: KptsMesh (uses .neg, .n_kpts).

    Returns:
        (coul_kpt, kern_kpt, infos, n_pipeline_calls): (Nk, Nip, Nip) jax
        arrays; length-Nk info list (a conjugated q shares its partner's
        dict); number of q's that actually ran the pipeline.
    """
    n_kpts = mesh_obj.n_kpts
    Pi = np.asarray(Pi)
    eta = np.asarray(eta)
    if Pi.shape[0] != n_kpts:
        raise ValueError(f"Pi.shape[0]={Pi.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")
    if eta.shape[0] != n_kpts:
        raise ValueError(f"eta.shape[0]={eta.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")

    # Precompute every q's Bloch phase once: per-q constant within one build.
    phase_all = precompute_phase_all_q(grid_coords, mesh_obj.canonical_kpts)

    neg = mesh_obj.neg
    coul_kpt = [None] * n_kpts
    kern_kpt = [None] * n_kpts
    infos = [None] * n_kpts
    done = [False] * n_kpts
    n_pipeline_calls = 0

    for q in range(n_kpts):
        if done[q]:
            continue
        nq = int(neg[q])
        W_q, kern_q, info_q = apply_kernel_and_solve_device(
            provider, q, Pi[q], eta[q], phase_q=phase_all[q], rtol=rtol,
            retained_solve_residual_gate=retained_solve_residual_gate,
            self_paired=(nq == q), retention_mode=retention_mode,
            kern_blocking=kern_blocking,
        )
        coul_kpt[q] = W_q
        kern_kpt[q] = kern_q
        infos[q] = info_q
        done[q] = True
        n_pipeline_calls += 1

        if nq != q and not done[nq]:
            coul_kpt[nq] = jnp.conj(W_q)
            kern_kpt[nq] = jnp.conj(kern_q)
            infos[nq] = info_q
            done[nq] = True

    return jnp.stack(coul_kpt, axis=0), jnp.stack(kern_kpt, axis=0), infos, n_pipeline_calls


# ---------------------------------------------------------------------------
# S1/S2 streaming (design v2.1 section 6): AO evaluation and the periodic
# pivot-selection metric oracle, both grid-block-streamed so host memory for
# either stays bounded by one block regardless of the full grid size Ng.


def stream_ao_blocks(cell, kpts, grid_coords, block_size, *, stats=None):
    """S1: stream AO values at kpts over grid_coords in blocks of
    block_size grid points; host memory stays bounded by one block.

    Yields:
        (g0, g1, ao_block): grid-index bounds [g0,g1) and the
        (Nk, g1-g0, Nao) complex128 block.
    """
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError(f"grid_coords must have shape (Ng,3), got {grid_coords.shape}.")
    n_grid = grid_coords.shape[0]
    if n_grid == 0:
        raise ValueError("grid_coords must be nonempty.")

    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    kpts_list = list(np.asarray(kpts, dtype=np.float64))

    for g0 in range(0, n_grid, block_size):
        g1 = min(g0 + block_size, n_grid)
        ao_block = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[g0:g1], kpts=kpts_list), dtype=np.complex128,
        )
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + (g1 - g0)
        yield g0, g1, ao_block


def build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """S2 periodic pivot-selection metric oracle (design doc §3): a
    (diag, col_eval) pair for the reference-cell pair-density Gram matrix
    M[r,r'] = |sum_{k,mu} conj(AO_k(r,mu)) AO_k(r',mu)|^2 / Nk, never
    materialized. Each col_eval(j) costs a full streamed AO-grid sweep,
    so selecting `rank` pivots costs `rank` sweeps -- callers should
    account for this traffic.

    Returns:
        (diag, col_eval): diag (Ng,) float64; col_eval(j) -> (Ng,)
        complex128 (M is real-valued; complex128 only to match
        pivoted_cholesky_hermitian's contract).
    """
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    n_grid = grid_coords.shape[0]
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_kpts = kpts_np.shape[0]

    diag = np.empty(n_grid, dtype=np.float64)
    for g0, g1, ao_block in stream_ao_blocks(
        cell, kpts_np, grid_coords, block_size, stats=stats,
    ):
        pooled = np.sum(np.abs(ao_block) ** 2, axis=(0, 2))  # (blk,), sum_{k,mu} |AO_k(r,mu)|^2
        diag[g0:g1] = pooled ** 2 / n_kpts

    def col_eval(j):
        ao_j_block = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[j:j + 1], kpts=list(kpts_np)),
            dtype=np.complex128,
        )
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + 1
        ao_j = ao_j_block[:, 0, :]  # (Nk, Nao)

        col = np.empty(n_grid, dtype=np.complex128)
        for g0, g1, ao_block in stream_ao_blocks(
            cell, kpts_np, grid_coords, block_size, stats=stats,
        ):
            gram = np.einsum("km,krm->r", ao_j.conj(), ao_block, optimize=True)
            col[g0:g1] = (np.abs(gram) ** 2 / n_kpts).astype(np.complex128)
        return col

    return diag, col_eval


def build_periodic_batched_pivot_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """Exact streamed periodic metric with one full AO sweep per column batch."""
    diagonal, _ = build_periodic_pivot_oracle(
        cell, kpts, grid_coords, block_size, stats=stats,
    )
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_grid = len(grid_coords)
    n_kpts = len(kpts_np)

    def col_batch_eval(indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= n_grid):
            raise ValueError("indices must be a nonempty in-range integer vector.")
        pivot_ao = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[indices], kpts=list(kpts_np)),
            dtype=np.complex128,
        )
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + int(indices.size)
        columns = np.empty((n_grid, indices.size), dtype=np.complex128)
        for g0, g1, ao_block in stream_ao_blocks(
            cell, kpts_np, grid_coords, block_size, stats=stats,
        ):
            gram = np.einsum("kbm,krm->br", pivot_ao.conj(), ao_block, optimize=True)
            columns[g0:g1] = (np.abs(gram) ** 2 / n_kpts).T
        return columns

    return diagonal, col_batch_eval


def periodic_metric_from_ao(ao):
    """Materialize the periodic metric for an explicit, bounded AO panel."""
    ao = np.asarray(ao, dtype=np.complex128)
    if ao.ndim != 3:
        raise ValueError(f"ao must have shape (Nk,Npanel,Nao), got {ao.shape}.")
    n_kpts, n_panel, _ = ao.shape
    if n_kpts == 0 or n_panel == 0:
        raise ValueError("ao must have nonempty k and panel axes.")
    flat = ao.transpose(1, 0, 2).reshape(n_panel, -1)
    return (np.abs(flat.conj() @ flat.T) ** 2 / n_kpts).astype(np.complex128)


def periodic_metric_column_from_ao(ao, index):
    """Return one periodic-metric column without materializing the metric."""
    ao = np.asarray(ao, dtype=np.complex128)
    if ao.ndim != 3:
        raise ValueError(f"ao must have shape (Nk,Npanel,Nao), got {ao.shape}.")
    n_kpts, n_panel, _ = ao.shape
    if n_kpts == 0 or n_panel == 0:
        raise ValueError("ao must have nonempty k and panel axes.")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer.")
    if not 0 <= index < n_panel:
        raise ValueError(f"index={index} is outside panel size {n_panel}.")
    gram = np.einsum("km,krm->r", ao[:, index, :].conj(), ao, optimize=True)
    return (np.abs(gram) ** 2 / n_kpts).astype(np.complex128)


def candidate_panel_indices(diag, rank, *, panel_factor=4, ramp_scale=1e-12):
    """Deterministically combine high-score and spatially stratified candidates."""
    diag = np.asarray(diag, dtype=np.float64)
    if diag.ndim != 1 or diag.size == 0 or not np.all(np.isfinite(diag)):
        raise ValueError("diag must be a nonempty 1-D array.")
    if np.any(diag < 0) or np.max(diag) <= 0:
        raise ValueError("diag must be nonnegative with a positive maximum.")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer.")
    if isinstance(panel_factor, bool) or not isinstance(panel_factor, (int, np.integer)):
        raise ValueError("panel_factor must be an integer.")
    if rank <= 0 or panel_factor < 2:
        raise ValueError("rank must be positive and panel_factor must be at least 2.")
    n_grid = diag.size
    panel_size = min(n_grid, panel_factor * rank)
    score = diag + ramp_scale * np.arange(n_grid) * float(np.max(diag))
    order = np.argsort(score, kind="stable")[::-1]
    high_count = min(panel_size // 2, n_grid)
    selected = list(order[:high_count])
    bins = np.array_split(np.arange(n_grid), panel_size - high_count)
    for indices in bins:
        selected.append(indices[np.argmax(score[indices])])
    unique = []
    seen = set()
    for index in selected + list(order):
        index = int(index)
        if index not in seen:
            unique.append(index)
            seen.add(index)
        if len(unique) == panel_size:
            break
    return np.asarray(unique, dtype=np.int64)


def full_grid_candidate_identity(n_grid):
    """Return a compact identity for the complete grid candidate set."""
    if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)) or n_grid <= 0:
        raise ValueError("n_grid must be a positive integer.")
    return {"kind": "range", "start": 0, "stop": int(n_grid), "step": 1}


def explicit_candidate_identity(indices):
    """Return the persisted identity for a bounded explicit candidate panel."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array.")
    return {"kind": "explicit_indices", "indices": indices.tolist()}


def build_cached_periodic_pivot_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """Experimental full-cache oracle with the same metric as the streamed path."""
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_grid = len(grid_coords)
    n_kpts = len(kpts_np)
    cache = None
    for g0, g1, ao_block in stream_ao_blocks(
        cell, kpts_np, grid_coords, block_size, stats=stats,
    ):
        if cache is None:
            cache = np.empty((n_kpts, n_grid, ao_block.shape[2]), dtype=np.complex128)
        cache[:, g0:g1] = ao_block
    pooled = np.sum(np.abs(cache) ** 2, axis=(0, 2))
    diag = pooled ** 2 / n_kpts
    return diag, lambda j: periodic_metric_column_from_ao(cache, j), cache


def periodic_metric_columns_from_ao(ao, indices):
    """Exact periodic-metric columns for a bounded batch from a cached AO tensor."""
    ao = np.asarray(ao, dtype=np.complex128)
    indices = np.asarray(indices, dtype=np.int64)
    if ao.ndim != 3 or indices.ndim != 1 or indices.size == 0:
        raise ValueError("ao must be (Nk,Ng,Nao) and indices must be nonempty 1-D.")
    if np.any(indices < 0) or np.any(indices >= ao.shape[1]):
        raise ValueError("indices are outside the AO grid.")
    pivot_ao = ao[:, indices, :]
    gram = np.einsum("kbm,krm->br", pivot_ao.conj(), ao, optimize=True)
    # M = |gram|^2/Nk is real, nonnegative; return float64 (not the historical
    # interface-convenience complex128) so the pivoted-Cholesky factor L it feeds
    # is stored real.
    return (np.abs(gram) ** 2 / ao.shape[0]).T.astype(np.float64)


def build_cached_periodic_bpc_gemm_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """Opt-in contiguous feature cache for BPC's threaded candidate GEMM."""
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_grid = len(grid_coords)
    n_kpts = len(kpts_np)
    features = None
    for g0, g1, ao_block in stream_ao_blocks(cell, kpts_np, grid_coords, block_size, stats=stats):
        if features is None:
            features = np.empty((n_grid, n_kpts * ao_block.shape[2]), dtype=np.complex128)
        features[g0:g1] = ao_block.transpose(1, 0, 2).reshape(g1 - g0, -1)
    pooled = np.sum(np.abs(features) ** 2, axis=1)
    diag = pooled ** 2 / n_kpts

    def col_batch_eval(indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= n_grid):
            raise ValueError("indices must be a nonempty in-range integer vector.")
        gram = features[indices].conj() @ features.T
        # real M -> float64 columns so the BPC factor L is stored real (real-f64-L).
        return (np.abs(gram) ** 2 / n_kpts).T.astype(np.float64)

    return diag, col_batch_eval, features


# ---------------------------------------------------------------------------
# Staging-policy layer (design doc §6): predicted-byte-model-driven selection
# among ram/memmap/recompute eta-store policies, plus the mechanics.
# Byte model is PREDICTED ONLY; observed fields exist in the schema as
# None/"unmeasured" so a later calibration pass backfills without a schema
# change. Real resource queries live only in query_host_resources.


def predicted_byte_model(n_kpts, n_ip, n_grid, n_ao, block_size, *, itemsize=16):
    """Closed-form predicted byte counts for one build (c128, itemsize=16);
    nothing here is measured. Returns a dict of the per-component byte
    terms plus total_predicted_bytes (eta_store + selection_traffic).
    See design doc §6."""
    for name, value in (
        ("n_kpts", n_kpts), ("n_ip", n_ip), ("n_grid", n_grid),
        ("n_ao", n_ao), ("block_size", block_size),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    ao_grid_block_bytes = n_kpts * block_size * n_ao * itemsize
    eta_store_bytes = n_kpts * n_ip * n_grid * itemsize
    double_buffer_bytes = 2 * n_ip * block_size * itemsize
    fft_workspace_bytes = 2 * n_ip * n_grid * itemsize
    pi_v_w_workspace_bytes = (n_kpts * n_ip * n_ip + n_ip * n_ip) * itemsize
    selection_traffic_bytes = n_ip * n_kpts * n_grid * n_ao * itemsize
    total_predicted_bytes = eta_store_bytes + selection_traffic_bytes

    return {
        "ao_grid_block_bytes": ao_grid_block_bytes,
        "eta_store_bytes": eta_store_bytes,
        "double_buffer_bytes": double_buffer_bytes,
        "fft_workspace_bytes": fft_workspace_bytes,
        "pi_v_w_workspace_bytes": pi_v_w_workspace_bytes,
        "selection_traffic_bytes": selection_traffic_bytes,
        "total_predicted_bytes": total_predicted_bytes,
    }


def choose_staging_policy(byte_model, *, available_host_bytes, available_disk_bytes,
                           ram_headroom_fraction=0.5, disk_headroom_fraction=0.9):
    """Select the eta-store staging policy from a predicted byte model and
    caller-supplied resource numbers (never queried internally, so this
    stays synthetic-input testable).

    Rule: "ram" if eta_store_bytes <= ram_headroom_fraction *
    available_host_bytes; else "memmap" if it fits the disk headroom;
    else "recompute".

    Returns a provenance dict; observed_peak_host_bytes/observed_status
    are always None/"unmeasured" here so a later calibration pass
    backfills the SAME schema.
    """
    eta_bytes = byte_model["eta_store_bytes"]
    if eta_bytes < 0:
        raise ValueError(f"byte_model['eta_store_bytes'] must be non-negative, got {eta_bytes}.")
    for name, value in (
        ("available_host_bytes", available_host_bytes),
        ("available_disk_bytes", available_disk_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    for name, value in (
        ("ram_headroom_fraction", ram_headroom_fraction),
        ("disk_headroom_fraction", disk_headroom_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < value <= 1.0):
            raise ValueError(f"{name} must be in (0,1], got {value!r}.")

    ram_headroom_bytes = int(ram_headroom_fraction * available_host_bytes)
    disk_headroom_bytes = int(disk_headroom_fraction * available_disk_bytes)

    if eta_bytes <= ram_headroom_bytes:
        policy = "ram"
    elif eta_bytes <= disk_headroom_bytes:
        policy = "memmap"
    else:
        policy = "recompute"

    return {
        "policy": policy,
        "eta_store_bytes": eta_bytes,
        "ram_headroom_bytes": ram_headroom_bytes,
        "disk_headroom_bytes": disk_headroom_bytes,
        "ram_headroom_fraction": ram_headroom_fraction,
        "disk_headroom_fraction": disk_headroom_fraction,
        "available_host_bytes": available_host_bytes,
        "available_disk_bytes": available_disk_bytes,
        "observed_peak_host_bytes": None,
        "observed_status": "unmeasured",
    }


def query_host_resources(scratch_dir="."):
    """The one place this module reads real host/disk resource numbers
    (psutil / shutil.disk_usage); pass the result to choose_staging_policy.

    Returns:
        (available_host_bytes, available_disk_bytes): both int.
    """
    import shutil

    import psutil

    available_host_bytes = int(psutil.virtual_memory().available)
    available_disk_bytes = int(shutil.disk_usage(scratch_dir).free)
    return available_host_bytes, available_disk_bytes


def jit_memory_analysis_smoke(jitted_fn, *args):
    """Compile jitted_fn against args (shapes/dtypes only; not executed)
    and return its jax CompiledMemoryStats as a dict, labeled by the
    actual backend (CPU stats are a structural smoke, not HBM data).
    See design doc §6."""
    backend = jax.default_backend()
    label = (
        "cpu_backend_structural_smoke_not_hbm"
        if backend == "cpu"
        else "gpu_backend_compiled_memory_stats"
    )
    stats = jitted_fn.lower(*args).compile().memory_analysis()
    return {
        "backend": backend,
        "label": label,
        "generated_code_size_in_bytes": stats.generated_code_size_in_bytes,
        "argument_size_in_bytes": stats.argument_size_in_bytes,
        "output_size_in_bytes": stats.output_size_in_bytes,
        "alias_size_in_bytes": stats.alias_size_in_bytes,
        "temp_size_in_bytes": stats.temp_size_in_bytes,
        "host_generated_code_size_in_bytes": stats.host_generated_code_size_in_bytes,
        "host_argument_size_in_bytes": stats.host_argument_size_in_bytes,
        "host_output_size_in_bytes": stats.host_output_size_in_bytes,
        "host_alias_size_in_bytes": stats.host_alias_size_in_bytes,
        "host_temp_size_in_bytes": stats.host_temp_size_in_bytes,
    }


def stage_eta_memmap(eta_chunks_iter, shape, memmap_path):
    """memmap staging (policy 2): write streamed eta chunks into an
    np.memmap at memmap_path without holding the full (Nk,Nip,Ng) eta in
    RAM.

    Args:
        eta_chunks_iter: iterable of (g0, g1, chunk), chunk
            (Nk,Nip,g1-g0) complex128, covering [0,Ng) in order.
        shape: (Nk,Nip,Ng).

    Returns:
        np.memmap, dtype complex128, flushed to disk.
    """
    shape_t = tuple(int(x) for x in shape)
    if len(shape_t) != 3 or any(s <= 0 for s in shape_t):
        raise ValueError(f"shape must be 3 positive ints, got {shape_t}.")

    mm = np.memmap(memmap_path, dtype=np.complex128, mode="w+", shape=shape_t)
    for g0, g1, chunk in eta_chunks_iter:
        mm[:, :, g0:g1] = chunk
    mm.flush()
    return mm


def stage_eta_recompute_tile(X, ao_block_source, phase, neg, q_slice=None):
    """recompute staging (policy 3): rebuild eta on demand via build_pi_eta,
    with ao_block_source() returning a FRESH iterable of (Nk,blk,Nao)
    blocks on every call. q_slice is applied to the leading (Nk) axis
    AFTER the full build (the complete pass still runs each call).

    Returns:
        (Pi, eta): same as build_pi_eta, optionally sliced by q_slice.
    """
    Pi, eta = build_pi_eta(X, ao_block_source(), phase, neg)
    if q_slice is not None:
        return Pi[q_slice], eta[q_slice]
    return Pi, eta
