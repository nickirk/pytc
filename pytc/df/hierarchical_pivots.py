"""Small-grid diagnostic for geometry-hierarchical ISDF pivot screening.

This is deliberately a prototype, not the production ISDF selector.  It
uses the existing geometry tree to obtain local pivot candidates, then runs
the same residual-diagonal Cholesky update used by a global selector.  The
optional refinement mode adds each leaf's exact residual maximum before the
global candidate choice; this recovers the global pivot sequence exactly and
is a correctness control, not a speedup.  Replacing those exact maxima with
certified block bounds is the remaining algorithmic work needed for scale.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from pytc.df.hmatrix import build_cluster_tree


def _validate_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("features must have shape (n_feature, n_grid)")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    return values


def _kernel_diagonal(features: np.ndarray) -> np.ndarray:
    return np.sum(features * features, axis=0) ** 2


def _kernel_column(features: np.ndarray, pivot: int) -> np.ndarray:
    overlap = features.T @ features[:, pivot]
    return overlap * overlap


def _validate_block_indices(
    values: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    row_indices = np.asarray(rows, dtype=int)
    col_indices = np.asarray(cols, dtype=int)
    if row_indices.ndim != 1 or col_indices.ndim != 1:
        raise ValueError("rows and cols must be one-dimensional")
    if not len(row_indices) or not len(col_indices):
        raise ValueError("rows and cols must be nonempty")
    n_grid = values.shape[1]
    if (
        np.any(row_indices < 0)
        or np.any(col_indices < 0)
        or np.any(row_indices >= n_grid)
        or np.any(col_indices >= n_grid)
    ):
        raise ValueError("rows and cols must index the grid")
    return row_indices, col_indices


def orbital_product_kernel_block(
    features: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Return the exact orbital-product Gram block ``K[rows, cols]``."""
    values = _validate_features(features)
    row_indices, col_indices = _validate_block_indices(values, rows, cols)
    overlap = values[:, row_indices].T @ values[:, col_indices]
    return overlap * overlap


def gradient_orbital_product_kernel_block(
    features: np.ndarray,
    gradient_features: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Return the exact production gradient Gram block ``K_grad[rows, cols]``."""
    values = _validate_features(features)
    gradients = np.asarray(gradient_features, dtype=float)
    if gradients.shape != (values.shape[0], values.shape[1], 3):
        raise ValueError("gradient_features must have shape (n_feature, n_grid, 3)")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("gradient_features must contain only finite values")
    row_indices, col_indices = _validate_block_indices(values, rows, cols)
    orbital_overlap = values[:, row_indices].T @ values[:, col_indices]
    gradient_overlap = sum(
        gradients[:, row_indices, component].T
        @ gradients[:, col_indices, component]
        for component in range(3)
    )
    return orbital_overlap * gradient_overlap


def orbital_product_feature_sketch(
    features: np.ndarray, *, dimension: int = 16, seed: int = 701
) -> np.ndarray:
    """Build deterministic randomized coordinates for orbital-product clustering.

    The ISDF kernel is the inner product of symmetrized orbital-pair features.
    This routine projects those exact pair features to a small dimension solely
    to partition the grid; it does not replace the kernel used for rank/error
    measurements or pivot selection.
    """
    values = _validate_features(features)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    left, right = np.triu_indices(values.shape[0])
    pair_features = values[left].T * values[right].T
    pair_features[:, left != right] *= np.sqrt(2.0)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((pair_features.shape[1], dimension))
    sketch = pair_features @ projection / np.sqrt(dimension)
    sketch -= np.mean(sketch, axis=0, keepdims=True)
    scale = np.std(sketch, axis=0, keepdims=True)
    return sketch / np.maximum(scale, np.finfo(float).tiny)


def relative_block_rank_profile(
    features: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    tolerances: Iterable[float],
) -> dict:
    """Measure exact-SVD ranks needed for relative Frobenius block error."""
    block = orbital_product_kernel_block(features, rows, cols)
    return _relative_rank_profile(block, tolerances)


def relative_gradient_block_rank_profile(
    features: np.ndarray,
    gradient_features: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    tolerances: Iterable[float],
) -> dict:
    """Measure exact-SVD ranks of the production ISDF gradient kernel."""
    block = gradient_orbital_product_kernel_block(
        features, gradient_features, rows, cols
    )
    return _relative_rank_profile(block, tolerances)


def _relative_rank_profile(block: np.ndarray, tolerances: Iterable[float]) -> dict:
    values = tuple(float(tolerance) for tolerance in tolerances)
    if any(tolerance < 0.0 for tolerance in values):
        raise ValueError("tolerances must be nonnegative")
    singular_values = np.linalg.svd(block, compute_uv=False)
    tail_squared = np.concatenate(
        (np.cumsum(singular_values[::-1] ** 2)[::-1], np.zeros(1))
    )
    norm_squared = float(tail_squared[0])
    scale = max(norm_squared, np.finfo(float).tiny)
    ranks: dict[str, int] = {}
    errors: dict[str, float] = {}
    for tolerance in values:
        rank = int(np.flatnonzero(np.sqrt(tail_squared / scale) <= tolerance)[0])
        key = f"{tolerance:.0e}"
        ranks[key] = rank
        errors[key] = float(np.sqrt(tail_squared[rank] / scale))
    return {
        "shape": [int(block.shape[0]), int(block.shape[1])],
        "frobenius_norm": float(np.sqrt(norm_squared)),
        "ranks": ranks,
        "relative_errors": errors,
    }


def _argmax_with_tiebreak(values: np.ndarray, indices: np.ndarray) -> int:
    """Choose the largest global index among exact ties, deterministically."""
    maximum = np.max(values)
    return int(indices[np.flatnonzero(values == maximum)[-1]])


def _pivoted_cholesky_select(
    features: np.ndarray,
    rank: int,
    candidates: np.ndarray,
    *,
    leaf_indices: Iterable[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run a global residual update while selecting only from ``candidates``.

    ``leaf_indices`` activates the exact-control refinement: before each
    selection it contributes every leaf's current residual maximizer to the
    candidate pool.  Consequently the global maximum is present and the
    resulting pivots equal uncompressed pivoted Cholesky.
    """
    n_grid = features.shape[1]
    if rank < 1 or rank > n_grid:
        raise ValueError("rank must lie in [1, n_grid]")
    candidate_mask = np.zeros(n_grid, dtype=bool)
    candidate_mask[np.asarray(candidates, dtype=int)] = True
    if not np.any(candidate_mask):
        raise ValueError("at least one candidate is required")

    diagonal = _kernel_diagonal(features)
    factor = np.zeros((n_grid, rank), dtype=float)
    selected = np.zeros(n_grid, dtype=bool)
    pivots: list[int] = []
    refinement_additions = 0

    for step in range(rank):
        if leaf_indices is not None:
            for leaf in leaf_indices:
                available = leaf[~selected[leaf]]
                if not len(available):
                    continue
                leaf_pivot = _argmax_with_tiebreak(diagonal[available], available)
                if not candidate_mask[leaf_pivot]:
                    candidate_mask[leaf_pivot] = True
                    refinement_additions += 1

        available = np.flatnonzero(candidate_mask & ~selected)
        if not len(available):
            raise RuntimeError("candidate pool exhausted before requested rank")
        pivot = _argmax_with_tiebreak(diagonal[available], available)
        pivot_value = float(diagonal[pivot])
        if pivot_value <= np.finfo(float).eps * max(1.0, float(np.max(diagonal))):
            break

        column = _kernel_column(features, pivot)
        if step:
            column -= factor[:, :step] @ factor[pivot, :step]
        new_column = column / np.sqrt(pivot_value)
        factor[:, step] = new_column
        diagonal = np.maximum(diagonal - new_column * new_column, 0.0)
        diagonal[pivot] = 0.0
        selected[pivot] = True
        pivots.append(pivot)

    return np.asarray(pivots, dtype=int), diagonal, {
        "initial_candidate_count": int(len(np.unique(candidates))),
        "final_candidate_count": int(np.count_nonzero(candidate_mask)),
        "refinement_additions": int(refinement_additions),
        "global_kernel_columns": int(len(pivots)),
        "global_residual_updates": int(len(pivots) * n_grid),
    }


def global_pivoted_cholesky(features: np.ndarray, rank: int) -> dict:
    """Reference global pivoted Cholesky for a small orbital-product kernel."""
    values = _validate_features(features)
    pivots, residual_diagonal, metadata = _pivoted_cholesky_select(
        values, rank, np.arange(values.shape[1])
    )
    metadata.update({"n_grid": int(values.shape[1]), "mode": "global"})
    return {
        "pivots": pivots,
        "residual_diagonal": residual_diagonal,
        "metadata": metadata,
    }


def local_pivot_candidates(
    features: np.ndarray,
    points: np.ndarray,
    *,
    leaf_size: int,
    local_rank: int,
) -> tuple[list[np.ndarray], dict]:
    """Select up to ``local_rank`` Cholesky pivots independently per leaf."""
    values = _validate_features(features)
    points = np.asarray(points, dtype=float)
    if points.shape != (values.shape[1], 3):
        raise ValueError("points must have shape (n_grid, 3)")
    if local_rank < 1:
        raise ValueError("local_rank must be positive")

    nodes, _ = build_cluster_tree(points, leaf_size)
    leaves = [node.indices for node in nodes if node.left is None]
    candidates: list[np.ndarray] = []
    local_kernel_work = 0
    for leaf in leaves:
        local_values = values[:, leaf]
        local_rank_actual = min(local_rank, len(leaf))
        local = global_pivoted_cholesky(local_values, local_rank_actual)["pivots"]
        candidates.append(leaf[local])
        local_kernel_work += len(leaf) * len(leaf)
    return candidates, {
        "n_leaf": int(len(leaves)),
        "local_rank": int(local_rank),
        "local_kernel_entries": int(local_kernel_work),
        "candidate_count": int(sum(len(candidate) for candidate in candidates)),
    }


def hierarchical_pivoted_cholesky(
    features: np.ndarray,
    points: np.ndarray,
    rank: int,
    *,
    leaf_size: int = 16,
    local_rank: int = 2,
    refine_with_leaf_maxima: bool = False,
) -> dict:
    """Screen global pivot selection with local tree-leaf candidates.

    With ``refine_with_leaf_maxima=False`` this is the cheap screened proposal.
    With it enabled, exact residual maxima from all leaves are admitted before
    each choice; that mode is an algebraic control for the proposed future
    bound-based refinement, and should reproduce global pivoted Cholesky.
    """
    values = _validate_features(features)
    candidates_by_leaf, local_metadata = local_pivot_candidates(
        values, points, leaf_size=leaf_size, local_rank=local_rank
    )
    candidates = np.concatenate(candidates_by_leaf)
    leaves = candidates_by_leaf if refine_with_leaf_maxima else None

    if refine_with_leaf_maxima:
        nodes, _ = build_cluster_tree(np.asarray(points, dtype=float), leaf_size)
        leaves = [node.indices for node in nodes if node.left is None]
    pivots, residual_diagonal, select_metadata = _pivoted_cholesky_select(
        values, rank, candidates, leaf_indices=leaves
    )
    metadata = {
        **local_metadata,
        **select_metadata,
        "n_grid": int(values.shape[1]),
        "leaf_size": int(leaf_size),
        "mode": "leaf-refined" if refine_with_leaf_maxima else "leaf-screened",
        "argmax_fraction_initial": local_metadata["candidate_count"] / values.shape[1],
        "argmax_fraction_final": select_metadata["final_candidate_count"] / values.shape[1],
    }
    return {
        "pivots": pivots,
        "residual_diagonal": residual_diagonal,
        "metadata": metadata,
    }


def orbital_product_projection_error(features: np.ndarray, pivots: np.ndarray) -> float:
    """Small-grid ISDF column-space diagnostic; not suitable for production grids."""
    values = _validate_features(features)
    products = np.einsum("pg,qg->pqg", values, values).reshape(
        values.shape[0] * values.shape[0], values.shape[1]
    )
    basis = products[:, np.asarray(pivots, dtype=int)]
    coefficients, *_ = np.linalg.lstsq(basis, products, rcond=None)
    residual = products - basis @ coefficients
    return float(np.linalg.norm(residual) / max(np.linalg.norm(products), np.finfo(float).tiny))
