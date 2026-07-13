"""Direct irregular-grid single-IBP Coulomb primitives (atom-centered
grids), moved here from ``pytc/utils/atom_centered_single_ibp_benchmark.py``
so the canonical implementation lives in ``pytc/df/`` alongside the other
model-agnostic DF/ISDF machinery (task #2, #proj-isdf-ibp-coulomb).
``kernel`` is the main-entrance single-IBP primitive (PySCF-style naming,
per Ke); ``naive_coulomb_kernel`` is the diagnostic comparison quadrature.

Unlike a uniform grid, a PySCF/TC atom-centered Becke grid is not
translationally invariant, so the ``rhat`` action here is evaluated by
blocked real-space summation, not FFT:

    (ia|jb) = -1/2 sum_g w_g grad(phi_i phi_a)_g .
                         sum_h w_h rhat_(g-h) (phi_j phi_b)_h.

Coincident left/right points are assigned the centered-point convention
``rhat=0`` explicitly (not skipped or NaN-guarded implicitly) --
``coincident_pairs`` is returned by both primitives so a caller cannot
silently forget which diagonal convention was used.
"""

from __future__ import annotations

import numpy as np


def _validate_grid_inputs(density, coords, weights, name):
    density = np.asarray(density)
    coords = np.asarray(coords)
    weights = np.asarray(weights)
    if density.ndim != 2:
        raise ValueError(f"{name}_density must have shape (n_function,n_grid)")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{name}_coords must have shape (n_grid,3)")
    if weights.ndim != 1 or weights.shape[0] != coords.shape[0]:
        raise ValueError(f"{name}_weights must have shape (n_grid,)")
    if density.shape[1] != coords.shape[0]:
        raise ValueError(f"{name}_density grid axis does not match {name}_coords")
    if not np.issubdtype(density.dtype, np.inexact):
        raise ValueError(f"{name}_density must use a floating or complex dtype")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} coordinates and weights must be finite")
    return density, coords, weights


def kernel(
    gradient_density_left,
    density_right,
    coords_left,
    weights_left,
    *,
    coords_right=None,
    weights_right=None,
    eval_block_size=128,
    source_block_size=4096,
):
    """Blocked one-sided single-IBP core on one or two irregular grids.

    The returned matrix is deliberately unsymmetrized.  Coincident left/right
    points receive ``rhat=0``; ``coincident_pairs`` is returned so a caller
    cannot silently forget which diagonal convention was used.
    """
    gradient = np.asarray(gradient_density_left)
    density_right = np.asarray(density_right)
    if gradient.ndim != 3 or gradient.shape[1] != 3:
        raise ValueError("gradient_density_left must have shape (n_left,3,n_grid_left)")
    dummy_left = np.empty((gradient.shape[0], gradient.shape[2]), dtype=gradient.dtype)
    _, coords_left, weights_left = _validate_grid_inputs(
        dummy_left, coords_left, weights_left, "left"
    )
    if coords_right is None:
        coords_right = coords_left
    if weights_right is None:
        weights_right = weights_left
    density_right, coords_right, weights_right = _validate_grid_inputs(
        density_right, coords_right, weights_right, "right"
    )
    if gradient.dtype != density_right.dtype:
        raise ValueError("gradient_density_left and density_right dtype must match")
    eval_block_size = int(eval_block_size)
    source_block_size = int(source_block_size)
    if eval_block_size <= 0 or source_block_size <= 0:
        raise ValueError("eval_block_size and source_block_size must be positive")

    n_left = gradient.shape[0]
    n_right = density_right.shape[0]
    result = np.zeros((n_left, n_right), dtype=density_right.dtype)
    coincident_pairs = 0
    for i0 in range(0, coords_left.shape[0], eval_block_size):
        i1 = min(i0 + eval_block_size, coords_left.shape[0])
        n_eval = i1 - i0
        vector = np.zeros((n_right, 3, n_eval), dtype=density_right.dtype)
        for j0 in range(0, coords_right.shape[0], source_block_size):
            j1 = min(j0 + source_block_size, coords_right.shape[0])
            diff = coords_left[i0:i1, None, :] - coords_right[None, j0:j1, :]
            radius = np.linalg.norm(diff, axis=-1)
            coincident_pairs += int(np.count_nonzero(radius == 0.0))
            rhat = np.divide(
                diff, radius[..., None], out=np.zeros_like(diff),
                where=radius[..., None] != 0.0,
            )
            weighted_density = density_right[:, j0:j1] * weights_right[j0:j1]
            vector += np.einsum("nj,ijc->nci", weighted_density, rhat, optimize=True)
        result += -0.5 * np.einsum(
            "mci,nci,i->mn", gradient[:, :, i0:i1].conj(), vector,
            weights_left[i0:i1], optimize=True,
        )
    return result, coincident_pairs


def naive_coulomb_kernel(
    density_left,
    density_right,
    coords_left,
    weights_left,
    *,
    coords_right=None,
    weights_right=None,
    eval_block_size=128,
    source_block_size=4096,
):
    """Diagnostic ``1/r`` quadrature with coincident terms set to zero.

    This is *not* a controlled production self-cell prescription.  It exists
    to quantify how much the bounded single-IBP kernel helps relative to the
    naive singular grid sum on the same atom-centered points.
    """
    density_left, coords_left, weights_left = _validate_grid_inputs(
        density_left, coords_left, weights_left, "left"
    )
    if coords_right is None:
        coords_right = coords_left
    if weights_right is None:
        weights_right = weights_left
    density_right, coords_right, weights_right = _validate_grid_inputs(
        density_right, coords_right, weights_right, "right"
    )
    if density_left.dtype != density_right.dtype:
        raise ValueError("density_left and density_right dtype must match")
    eval_block_size = int(eval_block_size)
    source_block_size = int(source_block_size)
    if eval_block_size <= 0 or source_block_size <= 0:
        raise ValueError("eval_block_size and source_block_size must be positive")

    result = np.zeros((density_left.shape[0], density_right.shape[0]),
                      dtype=density_right.dtype)
    coincident_pairs = 0
    for i0 in range(0, coords_left.shape[0], eval_block_size):
        i1 = min(i0 + eval_block_size, coords_left.shape[0])
        potential = np.zeros((density_right.shape[0], i1 - i0), dtype=density_right.dtype)
        for j0 in range(0, coords_right.shape[0], source_block_size):
            j1 = min(j0 + source_block_size, coords_right.shape[0])
            diff = coords_left[i0:i1, None, :] - coords_right[None, j0:j1, :]
            radius = np.linalg.norm(diff, axis=-1)
            coincident_pairs += int(np.count_nonzero(radius == 0.0))
            inv_r = np.divide(
                1.0, radius, out=np.zeros_like(radius), where=radius != 0.0
            )
            weighted_density = density_right[:, j0:j1] * weights_right[j0:j1]
            potential += weighted_density @ inv_r.T
        result += (density_left[:, i0:i1].conj() * weights_left[i0:i1]) @ potential.T
    return result, coincident_pairs
