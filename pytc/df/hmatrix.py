"""Geometry-hierarchical application of a pair-gradient kernel.

This module is an experimental operator for the L_aux residual.  It does not
form a full pair matrix: near blocks are evaluated directly and far blocks are
queried through adaptive cross approximation (ACA).  The vector gradient and
its squared-norm companion are factored independently because approximating a
gradient and then squaring it would change H_aux.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class ACAFactor:
    """Low-rank representation ``left @ right`` of one far block."""

    left: np.ndarray
    right: np.ndarray
    estimated_relative_error: float

    @property
    def rank(self) -> int:
        return self.left.shape[1]


def build_cluster_tree(points: np.ndarray, leaf_size: int) -> tuple[list[ClusterNode], int]:
    """Build a deterministic longest-axis binary tree over point features.

    Physical three-dimensional coordinates are the production use, but the
    routine is deliberately dimension-agnostic so diagnostics can partition
    grid points in a compact orbital-product feature space as well.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] < 1:
        raise ValueError("points must have shape (n_point, n_dimension)")
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
    gap = np.maximum(0.0, np.maximum(left.lower - right.upper, right.lower - left.upper))
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
        and _box_distance(left, right) > eta * max(_box_diameter(left), _box_diameter(right))
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
    if (len(left.indices) >= len(right.indices) and left.left is not None) or right.left is None:
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


def adaptive_cross_approximation(
    row_query: Callable[[int], np.ndarray],
    col_query: Callable[[int], np.ndarray],
    heldout_query: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_row: int,
    n_col: int,
    tolerance: float,
    max_rank: int | None = None,
    heldout_size: int = 8,
) -> ACAFactor:
    """Factor one matrix block from row/column kernel queries.

    A nonzero ``tolerance`` is a held-out stopping criterion, not a proof of
    global error.  ``tolerance=0`` drives ACA to its algebraic rank limit and
    is the exact-control mode used by the H2 test.
    """
    if n_row < 1 or n_col < 1:
        raise ValueError("ACA blocks must be nonempty")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    rank_limit = min(n_row, n_col) if max_rank is None else min(max_rank, n_row, n_col)
    if rank_limit < 1:
        return ACAFactor(np.empty((n_row, 0)), np.empty((0, n_col)), 0.0)

    held_rows = _heldout_indices(n_row, heldout_size)
    held_cols = _heldout_indices(n_col, heldout_size)
    heldout = np.asarray(heldout_query(held_rows, held_cols), dtype=float)
    heldout_norm = float(np.linalg.norm(heldout))
    scale = max(heldout_norm, np.finfo(float).tiny)

    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    used_rows: set[int] = set()
    pivot_row = 0
    error = float("inf")

    def residual_row(index: int) -> np.ndarray:
        row = np.asarray(row_query(index), dtype=float).copy()
        if left_parts:
            row -= np.column_stack(left_parts)[index] @ np.vstack(right_parts)
        return row

    def residual_col(index: int) -> np.ndarray:
        col = np.asarray(col_query(index), dtype=float).copy()
        if left_parts:
            col -= np.column_stack(left_parts) @ np.vstack(right_parts)[:, index]
        return col

    while len(left_parts) < rank_limit:
        row = residual_row(pivot_row)
        pivot_col = int(np.argmax(np.abs(row)))
        col = residual_col(pivot_col)
        pivot = float(col[pivot_row])
        threshold = np.finfo(float).eps * max(1.0, np.linalg.norm(row), np.linalg.norm(col))

        if abs(pivot) <= threshold:
            used_rows.add(pivot_row)
            available = [index for index in range(n_row) if index not in used_rows]
            if not available:
                break
            norms = [np.linalg.norm(residual_row(index), ord=np.inf) for index in available]
            best = int(np.argmax(norms))
            if norms[best] <= threshold:
                break
            pivot_row = available[best]
            continue

        left_parts.append(col)
        right_parts.append(row / pivot)
        used_rows.add(pivot_row)

        left = np.column_stack(left_parts)
        right = np.vstack(right_parts)
        error = float(np.linalg.norm(heldout - left[held_rows] @ right[:, held_cols]) / scale)
        if tolerance > 0.0 and error <= tolerance:
            break

        available = [index for index in range(n_row) if index not in used_rows]
        if not available:
            break
        pivot_row = max(available, key=lambda index: abs(col[index]))

    if left_parts:
        left = np.column_stack(left_parts)
        right = np.vstack(right_parts)
    else:
        left = np.empty((n_row, 0))
        right = np.empty((0, n_col))
        error = 0.0 if heldout_norm == 0.0 else 1.0
    return ACAFactor(left, right, error)


def apply_pair_gradient_hmatrix(
    points: np.ndarray,
    weights: np.ndarray,
    xi_phi: np.ndarray,
    gradient: GradientEvaluator,
    *,
    leaf_size: int = 16,
    eta: float = 0.5,
    tolerance: float = 0.0,
    max_rank: int | None = None,
    heldout_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply a pair-gradient and its H_aux kernel through an H-matrix.

    ``gradient(rows, cols)`` must return ``(len(rows), len(cols), 3)``.  The
    operator itself requests only near blocks and ACA rows/columns; it never
    materializes a full pair matrix.  The returned ``L_aux`` and ``H_aux``
    approximate the supplied pair kernel alone (e.g. the Boys--Handy
    residual), so exact one-grid contributions can be restored separately.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    xi_phi = np.asarray(xi_phi, dtype=float)
    n_grid = len(points)
    if weights.shape != (n_grid,) or xi_phi.ndim != 2 or xi_phi.shape[1] != n_grid:
        raise ValueError("incompatible points, weights, and xi_phi shapes")

    nodes, root = build_cluster_tree(points, leaf_size)
    blocks = admissible_blocks(nodes, root, root, eta)
    # Keep only a per-row pair count.  A full boolean coverage matrix would
    # reintroduce the O(G^2) allocation that the operator is intended to
    # avoid; the recursive block construction itself supplies disjointness.
    row_pair_counts = np.zeros(n_grid, dtype=np.int64)
    l_aux = np.zeros((xi_phi.shape[0], n_grid, 3))
    h_aux = np.zeros((xi_phi.shape[0], n_grid))
    weighted_xi = xi_phi * weights[None, :]
    metadata = {
        "leaf_size": leaf_size,
        "eta": eta,
        "tolerance": tolerance,
        "max_rank": max_rank,
        "near_blocks": 0,
        "far_blocks": 0,
        "near_pair_count": 0,
        "far_factor_storage": 0,
        "gradient_ranks": [],
        "h_aux_ranks": [],
        "gradient_heldout_errors": [],
        "h_aux_heldout_errors": [],
    }

    for left_id, right_id, is_far in blocks:
        rows = nodes[left_id].indices
        cols = nodes[right_id].indices
        row_pair_counts[rows] += len(cols)
        xi_weighted_block = weighted_xi[:, cols]

        if not is_far:
            value = np.asarray(gradient(points[rows], points[cols]), dtype=float)
            if value.shape != (len(rows), len(cols), 3):
                raise ValueError("gradient evaluator returned an incompatible block")
            l_aux[:, rows, :] += np.einsum("ah,ihk->aik", xi_weighted_block, value)
            h_aux[:, rows] += xi_weighted_block @ np.sum(value * value, axis=2).T
            metadata["near_blocks"] += 1
            metadata["near_pair_count"] += len(rows) * len(cols)
            continue

        metadata["far_blocks"] += 1

        def value_block(row_local: np.ndarray, col_local: np.ndarray) -> np.ndarray:
            return np.asarray(
                gradient(points[rows[row_local]], points[cols[col_local]]),
                dtype=float,
            )

        for component in range(3):
            factor = adaptive_cross_approximation(
                lambda index, c=component: value_block(
                    np.asarray([index]), np.arange(len(cols))
                )[0, :, c],
                lambda index, c=component: value_block(
                    np.arange(len(rows)), np.asarray([index])
                )[:, 0, c],
                lambda row_ids, col_ids, c=component: value_block(row_ids, col_ids)[:, :, c],
                len(rows),
                len(cols),
                tolerance,
                max_rank,
                heldout_size,
            )
            l_aux[:, rows, component] += (
                xi_weighted_block @ factor.right.T
            ) @ factor.left.T
            metadata["gradient_ranks"].append(factor.rank)
            metadata["gradient_heldout_errors"].append(factor.estimated_relative_error)
            metadata["far_factor_storage"] += factor.rank * (len(rows) + len(cols))

        h_factor = adaptive_cross_approximation(
            lambda index: np.sum(
                value_block(np.asarray([index]), np.arange(len(cols)))[0] ** 2,
                axis=1,
            ),
            lambda index: np.sum(
                value_block(np.arange(len(rows)), np.asarray([index]))[:, 0] ** 2,
                axis=1,
            ),
            lambda row_ids, col_ids: np.sum(value_block(row_ids, col_ids) ** 2, axis=2),
            len(rows),
            len(cols),
            tolerance,
            max_rank,
            heldout_size,
        )
        h_aux[:, rows] += (xi_weighted_block @ h_factor.right.T) @ h_factor.left.T
        metadata["h_aux_ranks"].append(h_factor.rank)
        metadata["h_aux_heldout_errors"].append(h_factor.estimated_relative_error)
        metadata["far_factor_storage"] += h_factor.rank * (len(rows) + len(cols))

    if not np.all(row_pair_counts == n_grid):
        raise AssertionError("hierarchical block partition does not cover every pair")
    ranks = metadata["gradient_ranks"] + metadata["h_aux_ranks"]
    metadata["far_rank_max"] = max(ranks, default=0)
    metadata["far_rank_mean"] = float(np.mean(ranks)) if ranks else 0.0
    return l_aux, h_aux, metadata


def _greedy_cross_pivots(matrix: np.ndarray, rank_limit: int) -> tuple[np.ndarray, np.ndarray]:
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
        return np.empty((left.shape[0], 0)), np.empty((0, right.shape[1])), 0.0

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
        selected_left, selected_right, selected_error = factor_left, factor_right, error
        if tolerance is not None and error <= tolerance:
            break
    assert selected_left is not None and selected_right is not None
    return selected_left, selected_right, selected_error


def _validation_indices(length: int, count: int, excluded: np.ndarray) -> np.ndarray:
    """Return deterministic validation points distinct from cross candidates when possible."""
    excluded_set = set(np.asarray(excluded, dtype=int).tolist())
    candidates = np.unique(
        np.floor((np.arange(max(count * 3, count)) + 0.5) * length / max(count * 3, count)).astype(int)
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
    leaf_size: int = 128,
    eta: float = 0.5,
    tolerance: float | None = 1e-4,
    max_rank: int = 16,
    heldout_size: int = 16,
    direct_fallback: bool = True,
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
    if weights.shape != (n_grid,) or xi_phi.ndim != 2 or xi_phi.shape[1] != n_grid:
        raise ValueError("incompatible points, weights, and xi_phi shapes")
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    if tolerance is not None and tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative or None")

    nodes, root = build_cluster_tree(points, leaf_size)
    blocks = admissible_blocks(nodes, root, root, eta)
    row_pair_counts = np.zeros(n_grid, dtype=np.int64)
    l_aux = np.zeros((xi_phi.shape[0], n_grid, 3))
    h_aux = np.zeros((xi_phi.shape[0], n_grid))
    weighted_xi = xi_phi * weights[None, :]
    metadata = {
        "mode": "interpolative-cur-v1",
        "leaf_size": leaf_size,
        "eta": eta,
        "tolerance": tolerance,
        "max_rank": max_rank,
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
        candidate_count = min(max_rank, len(rows), len(cols))
        candidate_rows = _heldout_indices(len(rows), candidate_count)
        candidate_cols = _heldout_indices(len(cols), candidate_count)
        held_rows = _validation_indices(len(rows), heldout_size, candidate_rows)
        held_cols = _validation_indices(len(cols), heldout_size, candidate_cols)

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

        factors: list[tuple[np.ndarray, np.ndarray]] = []
        errors: list[float] = []
        for component in range(3):
            factor = _cross_factor(
                left_values[:, :, component],
                right_values[:, :, component],
                candidate_rows,
                candidate_cols,
                max_rank,
                held_rows,
                held_cols,
                held_values[:, :, component],
                tolerance,
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
            max_rank,
            held_rows,
            held_cols,
            h_held,
            tolerance,
        )
        errors.append(h_factor[2])

        if direct_fallback and tolerance is not None and max(errors) > tolerance:
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
