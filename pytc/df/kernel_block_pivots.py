"""Experimental geometry-blocked orbital-product pivot selection.

The block matrix evaluates near blocks directly and stores held-out-validated
low-rank far blocks. Rejected far blocks fall back to direct evaluation. This
changes only how kernel columns are supplied; the Cholesky factor update
remains dense.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from pytc.df.hmatrix import adaptive_cross_approximation, admissible_blocks, build_cluster_tree


@dataclass(frozen=True)
class KernelBlock:
    """One symmetric block evaluated directly or stored as ``left @ right``."""

    rows: np.ndarray
    cols: np.ndarray
    left: np.ndarray | None
    right: np.ndarray | None
    relative_error: float

    @property
    def rank(self) -> int:
        return 0 if self.left is None else int(self.left.shape[1])

    @property
    def is_compressed(self) -> bool:
        return self.left is not None


class KernelBlockMatrix:
    """Symmetric hierarchical kernel with exact column-coverage bookkeeping."""

    def __init__(
        self,
        diagonal: np.ndarray,
        blocks: list[KernelBlock],
        evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
        metadata: dict,
    ):
        self.diagonal = np.asarray(diagonal, dtype=float)
        self.blocks = tuple(blocks)
        self._evaluator = evaluator
        self.metadata = dict(metadata)
        self._column_blocks: list[list[tuple[int, bool, int]]] = [
            [] for _ in range(len(self.diagonal))
        ]
        for block_index, block in enumerate(self.blocks):
            for local_col, global_col in enumerate(block.cols):
                self._column_blocks[int(global_col)].append(
                    (block_index, False, local_col)
                )
            if not np.array_equal(block.rows, block.cols):
                for local_row, global_row in enumerate(block.rows):
                    self._column_blocks[int(global_row)].append(
                        (block_index, True, local_row)
                    )

    @property
    def shape(self) -> tuple[int, int]:
        size = len(self.diagonal)
        return size, size

    def column(self, pivot: int) -> np.ndarray:
        """Assemble one column without materializing the complete kernel."""
        if pivot < 0 or pivot >= len(self.diagonal):
            raise IndexError("pivot lies outside the kernel")
        result = np.empty(len(self.diagonal), dtype=float)
        coverage = np.zeros(len(self.diagonal), dtype=np.int8)
        for block_index, transposed, local_index in self._column_blocks[pivot]:
            block = self.blocks[block_index]
            if not transposed:
                indices = block.rows
                if block.is_compressed:
                    assert block.left is not None and block.right is not None
                    values = block.left @ block.right[:, local_index]
                else:
                    values = self._evaluator(
                        block.rows, np.asarray([pivot], dtype=int)
                    )[:, 0]
            else:
                indices = block.cols
                if block.is_compressed:
                    assert block.left is not None and block.right is not None
                    values = block.right.T @ block.left[local_index, :]
                else:
                    values = self._evaluator(
                        block.cols, np.asarray([pivot], dtype=int)
                    )[:, 0]
            result[indices] = values
            coverage[indices] += 1
        if not np.all(coverage == 1):
            raise AssertionError("kernel block partition does not cover one full column")
        return result


def _validate_inputs(
    features: np.ndarray,
    points: np.ndarray,
    gradient_features: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    values = np.asarray(features, dtype=float)
    coordinates = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("features must have shape (n_feature, n_grid)")
    if coordinates.shape != (values.shape[1], 3):
        raise ValueError("points must have shape (n_grid, 3)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coordinates)):
        raise ValueError("features and points must be finite")
    if gradient_features is None:
        gradients = None
    else:
        gradients = np.asarray(gradient_features, dtype=float)
        if gradients.shape != (values.shape[0], values.shape[1], 3):
            raise ValueError(
                "gradient_features must have shape (n_feature, n_grid, 3)"
            )
        if not np.all(np.isfinite(gradients)):
            raise ValueError("gradient_features must be finite")
    return values, coordinates, gradients


def _kernel_evaluator(
    features: np.ndarray, gradient_features: np.ndarray | None
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], np.ndarray, str]:
    if gradient_features is None:

        def evaluate(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
            overlap = features[:, rows].T @ features[:, cols]
            return overlap * overlap

        diagonal = np.sum(features * features, axis=0) ** 2
        return evaluate, diagonal, "phi"

    def evaluate(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        orbital = features[:, rows].T @ features[:, cols]
        gradient = sum(
            gradient_features[:, rows, component].T
            @ gradient_features[:, cols, component]
            for component in range(3)
        )
        return orbital * gradient

    diagonal = np.sum(features * features, axis=0) * np.sum(
        gradient_features * gradient_features, axis=(0, 2)
    )
    return evaluate, diagonal, "gradient"


def build_kernel_block_matrix(
    features: np.ndarray,
    points: np.ndarray,
    *,
    gradient_features: np.ndarray | None = None,
    leaf_size: int = 64,
    eta: float = 0.5,
    tolerance: float = 1e-6,
    max_rank: int | None = None,
    heldout_size: int = 8,
    direct_fallback: bool = True,
) -> KernelBlockMatrix:
    """Build a symmetric exact-near/validated-low-rank-far kernel matrix."""
    values, coordinates, gradients = _validate_inputs(
        features, points, gradient_features
    )
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank must be positive")
    if heldout_size < 1:
        raise ValueError("heldout_size must be positive")

    evaluate, diagonal, kernel = _kernel_evaluator(values, gradients)
    started = time.perf_counter()
    nodes, root = build_cluster_tree(coordinates, leaf_size)
    partition = admissible_blocks(nodes, root, root, eta)
    blocks: list[KernelBlock] = []
    metadata = {
        "kernel": kernel,
        "n_grid": int(values.shape[1]),
        "leaf_size": int(leaf_size),
        "eta": float(eta),
        "tolerance": float(tolerance),
        "max_rank": max_rank,
        "heldout_size": int(heldout_size),
        "near_blocks": 0,
        "far_blocks": 0,
        "compressed_far_blocks": 0,
        "dense_far_fallbacks": 0,
        "partition_entries": 0,
        "block_index_storage": 0,
        "low_rank_storage": 0,
        "compressed_far_dense_entries": 0,
        "far_ranks": [],
        "far_heldout_errors": [],
    }

    for left_index, right_index, is_far in partition:
        if left_index > right_index:
            continue
        rows = nodes[left_index].indices
        cols = nodes[right_index].indices
        copies = 1 if left_index == right_index else 2
        covered_entries = copies * len(rows) * len(cols)
        stored_dense_entries = len(rows) * len(cols)
        metadata["partition_entries"] += covered_entries
        metadata["block_index_storage"] += len(rows) + len(cols)

        if not is_far:
            blocks.append(KernelBlock(rows, cols, None, None, 0.0))
            metadata["near_blocks"] += 1
            continue

        metadata["far_blocks"] += 1

        def query(row_local: np.ndarray, col_local: np.ndarray) -> np.ndarray:
            return evaluate(rows[row_local], cols[col_local])

        factor = adaptive_cross_approximation(
            lambda index: query(
                np.asarray([index]), np.arange(len(cols), dtype=int)
            )[0],
            lambda index: query(
                np.arange(len(rows), dtype=int), np.asarray([index])
            )[:, 0],
            query,
            len(rows),
            len(cols),
            tolerance,
            max_rank,
            heldout_size,
        )
        factor_entries = factor.rank * (len(rows) + len(cols))
        accepted = (
            factor.estimated_relative_error <= tolerance
            and factor_entries < stored_dense_entries
        )
        if accepted or not direct_fallback:
            blocks.append(
                KernelBlock(
                    rows,
                    cols,
                    factor.left,
                    factor.right,
                    factor.estimated_relative_error,
                )
            )
            metadata["compressed_far_blocks"] += 1
            metadata["low_rank_storage"] += factor_entries
            metadata["compressed_far_dense_entries"] += covered_entries
        else:
            blocks.append(
                KernelBlock(
                    rows,
                    cols,
                    None,
                    None,
                    factor.estimated_relative_error,
                )
            )
            metadata["dense_far_fallbacks"] += 1
        metadata["far_ranks"].append(factor.rank)
        metadata["far_heldout_errors"].append(factor.estimated_relative_error)

    if metadata["partition_entries"] != values.shape[1] ** 2:
        raise AssertionError("kernel block partition does not cover the full matrix")
    packed_dense_entries = values.shape[1] * (values.shape[1] + 1) // 2
    metadata["packed_dense_entries"] = int(packed_dense_entries)
    metadata["auxiliary_storage"] = (
        metadata["block_index_storage"] + metadata["low_rank_storage"]
    )
    metadata["auxiliary_storage_ratio_to_packed_dense"] = (
        metadata["auxiliary_storage"] / packed_dense_entries
    )
    metadata["far_rank_max"] = max(metadata["far_ranks"], default=0)
    metadata["far_rank_mean"] = float(
        np.mean(metadata["far_ranks"])
    ) if metadata["far_ranks"] else 0.0
    metadata["heldout_error_max"] = max(
        metadata["far_heldout_errors"], default=0.0
    )
    metadata["build_wall_s"] = time.perf_counter() - started
    return KernelBlockMatrix(diagonal, blocks, evaluate, metadata)


def kernel_block_pivoted_cholesky(
    matrix: KernelBlockMatrix,
    rank: int,
    *,
    shift_scale: float = 1e-12,
    tie_break_scale: float = 1e-12,
    stopping_tolerance: float = 1e-12,
) -> dict:
    """Select pivots from hierarchical kernel columns with a dense factor update."""
    n_grid = matrix.shape[0]
    if rank < 1 or rank > n_grid:
        raise ValueError("rank must lie in [1, n_grid]")
    if min(shift_scale, tie_break_scale, stopping_tolerance) < 0.0:
        raise ValueError(
            "shift, tie break, and stopping tolerance must be nonnegative"
        )

    scale = float(np.max(np.abs(matrix.diagonal)))
    shift = shift_scale * scale
    tie_break = tie_break_scale * np.arange(n_grid, dtype=float) * scale
    diagonal = matrix.diagonal.copy() + shift + tie_break
    factor = np.zeros((n_grid, rank), dtype=float)
    selected = np.zeros(n_grid, dtype=bool)
    pivots: list[int] = []
    column_wall_s = 0.0
    factor_wall_s = 0.0
    residual_wall_s = 0.0

    started = time.perf_counter()
    for step in range(rank):
        pivot = int(np.argmax(np.where(selected, -np.inf, diagonal)))
        pivot_value = float(diagonal[pivot])
        if pivot_value < stopping_tolerance:
            break

        phase = time.perf_counter()
        column = matrix.column(pivot)
        column[pivot] += shift + tie_break[pivot]
        column_wall_s += time.perf_counter() - phase

        phase = time.perf_counter()
        column -= factor @ factor[pivot]
        factor_wall_s += time.perf_counter() - phase

        phase = time.perf_counter()
        new_column = column / np.sqrt(pivot_value)
        factor[:, step] = new_column
        diagonal = np.maximum(diagonal - new_column * new_column, 0.0)
        diagonal[pivot] = 0.0
        selected[pivot] = True
        pivots.append(pivot)
        residual_wall_s += time.perf_counter() - phase

    total_wall_s = time.perf_counter() - started
    return {
        "pivots": np.asarray(pivots, dtype=int),
        "residual_diagonal": diagonal,
        "metadata": {
            **matrix.metadata,
            "rank_requested": int(rank),
            "rank_selected": int(len(pivots)),
            "column_wall_s": column_wall_s,
            "factor_wall_s": factor_wall_s,
            "residual_wall_s": residual_wall_s,
            "selection_wall_s": total_wall_s,
            "factor_bytes": int(factor.nbytes),
        },
    }


def kernel_block_pivoted_cholesky_jax(
    matrix: KernelBlockMatrix,
    rank: int,
    *,
    shift_scale: float = 1e-12,
    tie_break_scale: float = 1e-12,
    stopping_tolerance: float = 1e-12,
) -> dict:
    """Run the dense factor update on JAX while supplying block columns on host.

    This private experimental path preserves the production selector's
    fixed-shape dense ``L @ L[pivot]`` update on an accelerator.  Kernel-block
    assembly remains host-side and is timed separately, so a large-system run
    exposes transfer/dispatch overhead instead of hiding it in a monolithic
    selector timer.
    """
    import jax
    import jax.numpy as jnp

    n_grid = matrix.shape[0]
    if rank < 1 or rank > n_grid:
        raise ValueError("rank must lie in [1, n_grid]")
    if min(shift_scale, tie_break_scale, stopping_tolerance) < 0.0:
        raise ValueError(
            "shift, tie break, and stopping tolerance must be nonnegative"
        )

    scale = float(np.max(np.abs(matrix.diagonal)))
    shift = shift_scale * scale
    tie_break = tie_break_scale * np.arange(n_grid, dtype=float) * scale
    diagonal = jnp.asarray(matrix.diagonal + shift + tie_break)
    factor = jnp.zeros((n_grid, rank), dtype=diagonal.dtype)
    selected = jnp.zeros(n_grid, dtype=bool)

    @jax.jit
    def choose_pivot(current_diagonal, current_selected):
        return jnp.argmax(jnp.where(current_selected, -jnp.inf, current_diagonal))

    @jax.jit
    def update(
        current_diagonal,
        current_factor,
        current_selected,
        column,
        pivot,
        step,
    ):
        pivot_value = current_diagonal[pivot]
        residual_column = column - jnp.dot(current_factor, current_factor[pivot])
        is_small = pivot_value < stopping_tolerance
        safe_pivot = jnp.where(is_small, 1.0, pivot_value)
        new_column = residual_column * jax.lax.rsqrt(safe_pivot)
        new_column = jnp.where(is_small, 0.0, new_column)
        current_factor = current_factor.at[:, step].set(new_column)
        current_diagonal = jnp.maximum(
            current_diagonal - new_column * new_column, 0.0
        )
        current_diagonal = current_diagonal.at[pivot].set(0.0)
        current_selected = current_selected.at[pivot].set(True)
        return current_diagonal, current_factor, current_selected

    pivots: list[int] = []
    column_wall_s = 0.0
    device_wall_s = 0.0
    started = time.perf_counter()
    for step in range(rank):
        phase = time.perf_counter()
        pivot_device = choose_pivot(diagonal, selected)
        pivot = int(pivot_device)
        device_wall_s += time.perf_counter() - phase

        phase = time.perf_counter()
        column_host = matrix.column(pivot)
        column_host[pivot] += shift + tie_break[pivot]
        column_wall_s += time.perf_counter() - phase

        phase = time.perf_counter()
        diagonal, factor, selected = update(
            diagonal,
            factor,
            selected,
            jnp.asarray(column_host, dtype=diagonal.dtype),
            pivot_device,
            step,
        )
        pivots.append(pivot)
        device_wall_s += time.perf_counter() - phase

    residual_diagonal = np.asarray(diagonal)
    total_wall_s = time.perf_counter() - started
    return {
        "pivots": np.asarray(pivots, dtype=int),
        "residual_diagonal": residual_diagonal,
        "metadata": {
            **matrix.metadata,
            "backend": jax.default_backend(),
            "rank_requested": int(rank),
            "rank_selected": int(len(pivots)),
            "column_wall_s": column_wall_s,
            "device_wall_s": device_wall_s,
            "selection_wall_s": total_wall_s,
            "factor_bytes": int(n_grid * rank * diagonal.dtype.itemsize),
        },
    }
