"""Geometry-hierarchical application of a pair-gradient kernel.

This opt-in operator approximates the L_aux residual.  It does not
form a full pair matrix: near blocks are evaluated directly and far blocks are
queried through sampled cross interpolation (CUR).  The vector gradient and
its squared-norm companion are factored independently because approximating a
gradient and then squaring it would change H_aux.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable

import numpy as np


GradientEvaluator = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ClusterNode:
    """One axis-aligned geometry cluster."""

    indices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    left: int | None = None
    right: int | None = None


@dataclass(frozen=True)
class LauxHMatrixConfig:
    """Explicit controls for the opt-in L_aux hierarchy.

    No production default is implied by this object: callers must construct it
    and pass it to the ISDF build.  The direct path remains the default.
    """

    leaf_size: int
    eta: float
    tolerance: float
    max_rank: int
    heldout_size: int
    direct_fallback: bool = True

    def __post_init__(self) -> None:
        integer_controls = (self.leaf_size, self.max_rank, self.heldout_size)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in integer_controls
        ):
            raise TypeError("leaf_size, max_rank, and heldout_size must be integers")
        real_controls = (self.eta, self.tolerance)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            for value in real_controls
        ):
            raise TypeError("eta and tolerance must be finite real numbers")
        if not isinstance(self.direct_fallback, bool):
            raise TypeError("direct_fallback must be a boolean")
        if self.leaf_size < 2:
            raise ValueError("leaf_size must be at least two")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        if self.max_rank < 1:
            raise ValueError("max_rank must be positive")
        if self.heldout_size < 1:
            raise ValueError("heldout_size must be positive")

    @property
    def cache_tag(self) -> str:
        return (
            "hmatrix"
            f"[leaf={self.leaf_size},eta={self.eta:.8g},"
            f"tol={self.tolerance:.8g},rank={self.max_rank},"
            f"heldout={self.heldout_size},fallback={int(self.direct_fallback)}]"
        )


def build_cluster_tree(points: np.ndarray, leaf_size: int) -> tuple[list[ClusterNode], int]:
    """Build a deterministic longest-axis binary cluster tree."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_point, 3)")
    if leaf_size < 2:
        raise ValueError("leaf_size must be at least two")

    nodes: list[ClusterNode] = []

    def add(indices: np.ndarray) -> int:
        lower = points[indices].min(axis=0)
        upper = points[indices].max(axis=0)
        node_id = len(nodes)
        nodes.append(ClusterNode(indices=indices, lower=lower, upper=upper))
        if len(indices) > leaf_size:
            axis = int(np.argmax(upper - lower))
            ordered = indices[np.argsort(points[indices, axis], kind="stable")]
            middle = len(ordered) // 2
            left = add(ordered[:middle])
            right = add(ordered[middle:])
            nodes[node_id] = ClusterNode(indices, lower, upper, left, right)
        return node_id

    return nodes, add(np.arange(len(points), dtype=int))


def _box_distance(left: ClusterNode, right: ClusterNode) -> float:
    gap = np.maximum(
        0.0, np.maximum(left.lower - right.upper, right.lower - left.upper)
    )
    return float(np.linalg.norm(gap))


def _box_diameter(node: ClusterNode) -> float:
    return float(np.linalg.norm(node.upper - node.lower))


def admissible_blocks(
    nodes: list[ClusterNode], left_id: int, right_id: int, eta: float
) -> list[tuple[int, int, bool]]:
    """Exactly partition a pair matrix into direct and admissible far blocks."""
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    left = nodes[left_id]
    right = nodes[right_id]
    far = (
        left_id != right_id
        and _box_distance(left, right)
        > eta * max(_box_diameter(left), _box_diameter(right))
    )
    if far:
        return [(left_id, right_id, True)]
    if left.left is None and right.left is None:
        return [(left_id, right_id, False)]
    if left_id == right_id:
        assert left.left is not None and left.right is not None
        return (
            admissible_blocks(nodes, left.left, left.left, eta)
            + admissible_blocks(nodes, left.left, left.right, eta)
            + admissible_blocks(nodes, left.right, left.left, eta)
            + admissible_blocks(nodes, left.right, left.right, eta)
        )
    if (
        len(left.indices) >= len(right.indices) and left.left is not None
    ) or right.left is None:
        assert left.left is not None and left.right is not None
        return admissible_blocks(nodes, left.left, right_id, eta) + admissible_blocks(
            nodes, left.right, right_id, eta
        )
    assert right.left is not None and right.right is not None
    return admissible_blocks(nodes, left_id, right.left, eta) + admissible_blocks(
        nodes, left_id, right.right, eta
    )


def _heldout_indices(length: int, count: int) -> np.ndarray:
    """Return deterministic, evenly spread validation indices."""
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=int))


def _greedy_cross_pivots(
    matrix: np.ndarray, rank_limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a stable cross from a small, already sampled proxy matrix."""
    residual = np.asarray(matrix, dtype=float).copy()
    initial_scale = max(float(np.linalg.norm(residual, ord=np.inf)), 1.0)
    rows: list[int] = []
    cols: list[int] = []
    for _ in range(min(rank_limit, *residual.shape)):
        row, col = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        pivot = residual[row, col]
        if abs(pivot) <= np.finfo(float).eps * initial_scale:
            break
        rows.append(int(row))
        cols.append(int(col))
        residual -= np.outer(residual[:, col], residual[row, :]) / pivot
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _cross_factor(
    left: np.ndarray,
    right: np.ndarray,
    candidate_rows: np.ndarray,
    candidate_cols: np.ndarray,
    rank_limit: int,
    held_rows: np.ndarray,
    held_cols: np.ndarray,
    heldout: np.ndarray,
    tolerance: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a CUR factor from two sampled kernel strips.

    The pivot search is confined to the small candidate cross.  The returned
    error is measured on fresh, deterministic held-out entries, so it can
    select a rank but is not a global certificate.
    """
    proxy = left[candidate_rows, :]
    pivot_rows, pivot_cols = _greedy_cross_pivots(proxy, rank_limit)
    if not len(pivot_rows):
        error = 0.0 if np.linalg.norm(heldout) == 0.0 else 1.0
        return (
            np.empty((left.shape[0], 0)),
            np.empty((0, right.shape[1])),
            error,
        )

    scale = max(float(np.linalg.norm(heldout)), np.finfo(float).tiny)
    selected_error = float("inf")
    selected_left = None
    selected_right = None
    for rank in range(1, len(pivot_rows) + 1):
        local_rows = pivot_rows[:rank]
        local_cols = pivot_cols[:rank]
        # ``left`` already uses the candidate-column coordinate and ``right``
        # already uses the candidate-row coordinate; only the left *rows*
        # retain the block-local grid indexing.
        core = left[candidate_rows[local_rows]][:, local_cols]
        factor_left = left[:, local_cols] @ np.linalg.pinv(core, rcond=1e-12)
        factor_right = right[local_rows, :]
        error = float(
            np.linalg.norm(
                heldout
                - factor_left[held_rows] @ factor_right[:, held_cols]
            )
            / scale
        )
        selected_left = factor_left
        selected_right = factor_right
        selected_error = error
        if tolerance is not None and error <= tolerance:
            break
    assert selected_left is not None and selected_right is not None
    return selected_left, selected_right, selected_error


def _validation_indices(length: int, count: int, excluded: np.ndarray) -> np.ndarray:
    """Return deterministic points outside cross candidates when possible."""
    excluded_set = set(np.asarray(excluded, dtype=int).tolist())
    candidates = np.unique(
        np.floor(
            (np.arange(max(count * 3, count)) + 0.5)
            * length
            / max(count * 3, count)
        ).astype(int)
    )
    chosen = [item for item in candidates if item not in excluded_set]
    if not chosen:
        chosen = list(range(length))
    return np.asarray(chosen[: min(count, len(chosen))], dtype=int)


def apply_pair_gradient_interpolative_hmatrix(
    points: np.ndarray,
    weights: np.ndarray,
    xi_phi: np.ndarray,
    gradient: GradientEvaluator,
    *,
    config: LauxHMatrixConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply a sampled-CUR hierarchical approximation to a pair gradient.

    Each admissible far block evaluates only two kernel strips of width at
    most ``max_rank`` plus a small held-out check.  Direct near blocks are
    exact.  The gradient components and the scalar ``|gradient|^2`` each get
    their own CUR factor; this preserves the required independent H_aux
    approximation.  A block whose held-out error exceeds ``tolerance`` falls
    back to its exact direct contraction when ``direct_fallback`` is enabled.

    This is a streamed construction: it never forms a global pair matrix.
    The tolerance is a rank-selection/fallback gate, not a global error
    certificate.  Callers must validate L_aux, H_aux, K1/K3, two-body, and
    energy against the direct operator before using it in chemistry results.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    xi_phi = np.asarray(xi_phi, dtype=float)
    n_grid = len(points)
    if n_grid == 0:
        raise ValueError("points must be nonempty")
    if weights.shape != (n_grid,) or xi_phi.ndim != 2 or xi_phi.shape[1] != n_grid:
        raise ValueError("incompatible points, weights, and xi_phi shapes")
    if not all(np.all(np.isfinite(array)) for array in (points, weights, xi_phi)):
        raise ValueError("points, weights, and xi_phi must be finite")

    nodes, root = build_cluster_tree(points, config.leaf_size)
    blocks = admissible_blocks(nodes, root, root, config.eta)
    row_pair_counts = np.zeros(n_grid, dtype=np.int64)
    l_aux = np.zeros((xi_phi.shape[0], n_grid, 3))
    h_aux = np.zeros((xi_phi.shape[0], n_grid))
    weighted_xi = xi_phi * weights[None, :]
    metadata = {
        "mode": "interpolative-cur-v1",
        "leaf_size": config.leaf_size,
        "eta": config.eta,
        "tolerance": config.tolerance,
        "max_rank": config.max_rank,
        "direct_fallback": config.direct_fallback,
        "near_blocks": 0,
        "far_blocks": 0,
        "far_direct_fallbacks": 0,
        "near_pair_count": 0,
        "far_pair_count": 0,
        "far_factor_storage": 0,
        "gradient_ranks": [],
        "h_aux_ranks": [],
        "heldout_errors": [],
    }

    def accumulate_direct(rows: np.ndarray, cols: np.ndarray) -> None:
        value = np.asarray(gradient(points[rows], points[cols]), dtype=float)
        if value.shape != (len(rows), len(cols), 3):
            raise ValueError("gradient evaluator returned an incompatible block")
        if not np.all(np.isfinite(value)):
            raise ValueError("gradient evaluator returned non-finite values")
        xi_weighted = weighted_xi[:, cols]
        l_aux[:, rows, :] += np.einsum("ah,ihk->aik", xi_weighted, value)
        h_aux[:, rows] += xi_weighted @ np.sum(value * value, axis=2).T

    for left_id, right_id, is_far in blocks:
        rows = nodes[left_id].indices
        cols = nodes[right_id].indices
        row_pair_counts[rows] += len(cols)
        if not is_far:
            accumulate_direct(rows, cols)
            metadata["near_blocks"] += 1
            metadata["near_pair_count"] += len(rows) * len(cols)
            continue

        metadata["far_blocks"] += 1
        metadata["far_pair_count"] += len(rows) * len(cols)
        candidate_count = min(config.max_rank, len(rows), len(cols))
        candidate_rows = _heldout_indices(len(rows), candidate_count)
        candidate_cols = _heldout_indices(len(cols), candidate_count)
        held_rows = _validation_indices(
            len(rows), config.heldout_size, candidate_rows
        )
        held_cols = _validation_indices(
            len(cols), config.heldout_size, candidate_cols
        )

        # Three JAX kernel calls irrespective of factor rank: all-row/cross,
        # cross/all-column, then an independent validation patch.
        left_values = np.asarray(
            gradient(points[rows], points[cols[candidate_cols]]), dtype=float
        )
        right_values = np.asarray(
            gradient(points[rows[candidate_rows]], points[cols]), dtype=float
        )
        held_values = np.asarray(
            gradient(points[rows[held_rows]], points[cols[held_cols]]), dtype=float
        )
        if (
            left_values.shape != (len(rows), candidate_count, 3)
            or right_values.shape != (candidate_count, len(cols), 3)
            or held_values.shape != (len(held_rows), len(held_cols), 3)
        ):
            raise ValueError("gradient evaluator returned an incompatible sampled block")
        if not all(
            np.all(np.isfinite(array))
            for array in (left_values, right_values, held_values)
        ):
            raise ValueError("gradient evaluator returned non-finite values")

        factors: list[tuple[np.ndarray, np.ndarray]] = []
        errors: list[float] = []
        for component in range(3):
            factor = _cross_factor(
                left_values[:, :, component],
                right_values[:, :, component],
                candidate_rows,
                candidate_cols,
                config.max_rank,
                held_rows,
                held_cols,
                held_values[:, :, component],
                config.tolerance,
            )
            factors.append(factor[:2])
            errors.append(factor[2])

        h_left = np.sum(left_values * left_values, axis=2)
        h_right = np.sum(right_values * right_values, axis=2)
        h_held = np.sum(held_values * held_values, axis=2)
        h_factor = _cross_factor(
            h_left,
            h_right,
            candidate_rows,
            candidate_cols,
            config.max_rank,
            held_rows,
            held_cols,
            h_held,
            config.tolerance,
        )
        errors.append(h_factor[2])

        if config.direct_fallback and max(errors) > config.tolerance:
            accumulate_direct(rows, cols)
            metadata["far_direct_fallbacks"] += 1
            metadata["heldout_errors"].append(max(errors))
            continue

        xi_weighted = weighted_xi[:, cols]
        for component, (factor_left, factor_right) in enumerate(factors):
            l_aux[:, rows, component] += (
                xi_weighted @ factor_right.T
            ) @ factor_left.T
            metadata["gradient_ranks"].append(factor_left.shape[1])
            metadata["far_factor_storage"] += factor_left.size + factor_right.size
        h_left_factor, h_right_factor = h_factor[:2]
        h_aux[:, rows] += (xi_weighted @ h_right_factor.T) @ h_left_factor.T
        metadata["h_aux_ranks"].append(h_left_factor.shape[1])
        metadata["far_factor_storage"] += h_left_factor.size + h_right_factor.size
        metadata["heldout_errors"].append(max(errors))

    if not np.all(row_pair_counts == n_grid):
        raise AssertionError("hierarchical block partition does not cover every pair")
    ranks = metadata["gradient_ranks"] + metadata["h_aux_ranks"]
    metadata["far_rank_max"] = max(ranks, default=0)
    metadata["far_rank_mean"] = float(np.mean(ranks)) if ranks else 0.0
    metadata["heldout_error_max"] = max(metadata["heldout_errors"], default=0.0)
    return l_aux, h_aux, metadata
