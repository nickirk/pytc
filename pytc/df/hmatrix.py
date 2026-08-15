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
