"""CPU-only feasibility helpers for reciprocal primitive-AO translation.

These helpers are deliberately standalone: they are not used by the periodic
selector or any production build path.  They preserve the sampled-grid
approximation explicitly so a pilot can compare it with direct AO values.
"""

from __future__ import annotations

import numpy as np
from pyscf.pbc import tools as pbc_tools

from pytc.pbc.df.isdf import pivoted_cholesky_hermitian


def periodic_part_from_bloch(bloch_ao, grid_coords, kpt):
    """Return sampled ``u(r)`` for ``phi(r) = exp(i k.r) u(r)``."""
    bloch_ao = np.asarray(bloch_ao, dtype=np.complex128)
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    kpt = np.asarray(kpt, dtype=np.float64)
    if bloch_ao.ndim != 2 or bloch_ao.shape[0] != grid_coords.shape[0]:
        raise ValueError("bloch_ao must have shape (Ng,Nao) for grid_coords.")
    return np.exp(-1j * (grid_coords @ kpt))[:, None] * bloch_ao


def reciprocal_translate_bloch_ao(
    seed_bloch_ao, grid_coords, mesh, kpt, translation, g_vectors,
):
    """Generate translated sampled Bloch AOs from a periodic-part FFT.

    This is a same-grid spectral translation.  It is not exact unless the
    periodic part is represented without aliasing on ``mesh``.
    """
    mesh = np.asarray(mesh, dtype=np.int64)
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    kpt = np.asarray(kpt, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    g_vectors = np.asarray(g_vectors, dtype=np.float64)
    u_seed = periodic_part_from_bloch(seed_bloch_ao, grid_coords, kpt)
    if g_vectors.shape != grid_coords.shape:
        raise ValueError("g_vectors must have the same shape as grid_coords.")
    u_g = pbc_tools.fft(u_seed.T, mesh)
    translated_u = pbc_tools.ifft(
        u_g * np.exp(-1j * (g_vectors @ translation)), mesh,
    ).T
    return (
        np.exp(-1j * (kpt @ translation))
        * np.exp(1j * (grid_coords @ kpt))[:, None]
        * translated_u
    )


def relative_frobenius_error(reference, candidate):
    """Return max-abs and relative Frobenius error for matching arrays."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes.")
    difference = reference - candidate
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_frobenius": float(np.linalg.norm(difference) / denominator),
    }


def metric_column_from_ao_groups(groups, pivot, n_kpts):
    """Form one exact metric column after accumulating every AO group."""
    accumulator = None
    for group in groups:
        group = np.asarray(group, dtype=np.complex128)
        if group.ndim != 3:
            raise ValueError("each AO group must have shape (Nk,Ng,Nao).")
        contribution = np.einsum(
            "ka,kga->g", group[:, pivot, :].conj(), group, optimize=True,
        )
        accumulator = contribution if accumulator is None else accumulator + contribution
    if accumulator is None:
        raise ValueError("at least one AO group is required.")
    return np.abs(accumulator) ** 2 / n_kpts


def metric_diagonal_from_ao_groups(groups, n_kpts):
    """Form the exact metric diagonal while holding one AO group at a time."""
    density = None
    for group in groups:
        group = np.asarray(group, dtype=np.complex128)
        if group.ndim != 3:
            raise ValueError("each AO group must have shape (Nk,Ng,Nao).")
        contribution = np.sum(np.abs(group) ** 2, axis=(0, 2))
        density = contribution if density is None else density + contribution
    if density is None:
        raise ValueError("at least one AO group is required.")
    return density ** 2 / n_kpts


def pivot_prefix_from_ao_groups(group_factory, n_grid, n_kpts, rank):
    """Select a short exact prefix through streamed AO-group contributions."""
    diagonal = metric_diagonal_from_ao_groups(group_factory(), n_kpts)

    def column(pivot):
        return metric_column_from_ao_groups(group_factory(), pivot, n_kpts)

    return pivoted_cholesky_hermitian(diagonal, column, rank=rank)
