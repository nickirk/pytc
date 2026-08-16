"""Streaming exact-column pivot selection for molecular ISDF.

The selector in this module deliberately owns no dense grid-by-grid matrix.
It consumes an exact metric diagonal and a callable that evaluates requested
columns.  Molecular ISDF backs that interface with already-resident weighted
orbital factors; periodic ISDF has a separate implementation and a streamed
AO provider, but uses the same batch controls and schedule.
"""

from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def validate_pivot_controls(
    n_grid, n_rank, batch_size, candidate_oversampling, n_topup
):
    values = (n_rank, batch_size, candidate_oversampling, n_topup)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in values
    ):
        raise ValueError(
            "n_rank, batch_size, candidate_oversampling, and n_topup must be integers"
        )
    if not 0 < n_rank <= n_grid:
        raise ValueError(f"n_rank must be in [1, {n_grid}]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if candidate_oversampling <= 0:
        raise ValueError("candidate_oversampling must be positive")
    if not 0 <= n_topup <= n_rank:
        raise ValueError("n_topup must be in [0, n_rank]")


def _diverse_local_pivots(matrix, already_selected, count):
    """Select an exact greedy subset from a stale-diagonal candidate pool."""
    size = matrix.shape[0]
    diagonal = jnp.maximum(jnp.real(jnp.diag(matrix)), 0.0)
    factor = jnp.zeros((size, count), dtype=matrix.dtype)
    pivots = jnp.zeros(count, dtype=jnp.int32)

    def body(step, state):
        current_diagonal, current_factor, selected, current_pivots = state
        pivot = jnp.argmax(
            jnp.where(selected, -jnp.inf, current_diagonal)
        ).astype(jnp.int32)
        pivot_value = current_diagonal[pivot]
        column = matrix[:, pivot] - (
            current_factor @ jnp.conj(current_factor[pivot])
        )
        safe_value = jnp.where(pivot_value < 1e-14, 1.0, pivot_value)
        new_column = column * jax.lax.rsqrt(safe_value)
        new_column = jnp.where(pivot_value < 1e-14, 0.0, new_column)
        current_factor = current_factor.at[:, step].set(new_column)
        current_diagonal = jnp.maximum(
            current_diagonal - jnp.abs(new_column) ** 2, 0.0
        )
        current_diagonal = current_diagonal.at[pivot].set(0.0)
        selected = selected.at[pivot].set(True)
        current_pivots = current_pivots.at[step].set(pivot)
        return current_diagonal, current_factor, selected, current_pivots

    return jax.lax.fori_loop(
        0, count, body, (diagonal, factor, already_selected, pivots)
    )[3]


def _sequential_round(state, step, column_evaluator, shift, tie_break):
    diagonal, factor, pivots, selected = state
    pivot = jnp.argmax(jnp.where(selected, -jnp.inf, diagonal)).astype(jnp.int32)
    pivot_value = diagonal[pivot]
    column = column_evaluator(jnp.asarray([pivot], dtype=jnp.int32))[:, 0]
    column = column.at[pivot].add(shift + tie_break[pivot])
    projection = factor @ jnp.conj(factor[pivot])
    is_small = pivot_value < 1e-12
    safe_value = jnp.where(is_small, 1.0, pivot_value)
    new_column = (column - projection) * jax.lax.rsqrt(safe_value)
    new_column = jnp.where(is_small, 0.0, new_column)
    factor = factor.at[:, step].set(new_column)
    diagonal = jnp.maximum(diagonal - jnp.abs(new_column) ** 2, 0.0)
    diagonal = diagonal.at[pivot].set(0.0)
    pivots = pivots.at[step].set(pivot)
    selected = selected.at[pivot].set(True)
    return diagonal, factor, pivots, selected


def _blocked_round(
    state,
    start,
    retain_count,
    candidate_count,
    column_evaluator,
    shift,
    tie_break,
):
    diagonal, factor, pivots, selected = state
    scores = jnp.where(selected, -jnp.inf, diagonal)
    _, candidates = jax.lax.top_k(scores, candidate_count)
    candidates = candidates.astype(jnp.int32)

    # The provider returns exact columns of the unperturbed metric.  The
    # selector owns the diagonal shift/ramp, so all providers share one scale
    # convention and cannot silently disagree about regularization.
    candidate_columns = column_evaluator(candidates)
    candidate_range = jnp.arange(candidate_count)
    candidate_columns = candidate_columns.at[candidates, candidate_range].add(
        shift + tie_break[candidates]
    )
    candidate_rows = factor[candidates]
    candidate_residual = candidate_columns[candidates] - (
        candidate_rows @ jnp.conj(candidate_rows).T
    )
    candidate_residual = 0.5 * (
        candidate_residual + jnp.conj(candidate_residual).T
    )
    local_pivots = _diverse_local_pivots(
        candidate_residual, selected[candidates], retain_count
    )
    block_pivots = candidates[local_pivots]

    columns = candidate_columns[:, local_pivots]
    residual_columns = columns - factor @ jnp.conj(factor[block_pivots]).T
    pivot_block = residual_columns[block_pivots]
    pivot_block = 0.5 * (pivot_block + jnp.conj(pivot_block).T)
    block_range = jnp.arange(retain_count)
    pivot_block = pivot_block.at[block_range, block_range].set(
        diagonal[block_pivots]
    )
    block_scale = jnp.maximum(
        jnp.max(jnp.abs(jnp.diag(pivot_block))),
        jnp.finfo(diagonal.dtype).tiny,
    )
    pivot_block = pivot_block + (
        1e-14
        * block_scale
        * jnp.eye(retain_count, dtype=pivot_block.dtype)
    )
    chol = jnp.linalg.cholesky(pivot_block)
    new_factor = jsp_linalg.solve_triangular(
        jnp.conj(chol), residual_columns.T, lower=True
    ).T

    factor = jax.lax.dynamic_update_slice(factor, new_factor, (0, start))
    diagonal = jnp.maximum(
        diagonal - jnp.sum(jnp.abs(new_factor) ** 2, axis=1), 0.0
    )
    diagonal = diagonal.at[block_pivots].set(0.0)
    pivots = jax.lax.dynamic_update_slice(pivots, block_pivots, (start,))
    selected = selected.at[block_pivots].set(True)
    return diagonal, factor, pivots, selected


@partial(
    jax.jit,
    static_argnames=("n_rank", "batch_size", "candidate_oversampling", "n_topup"),
)
def _pivoted_cholesky_streaming(
    diagonal,
    column_evaluator,
    shift,
    *,
    n_rank,
    batch_size,
    candidate_oversampling,
    n_topup,
):
    """Compiled schedule shared by all molecular exact-column providers."""
    n_grid = diagonal.shape[0]
    tie_break = (
        1e-12
        * jnp.arange(n_grid, dtype=diagonal.dtype)
        * jnp.max(jnp.abs(diagonal + shift))
    )
    diagonal = diagonal + shift + tie_break
    factor = jnp.zeros((n_grid, n_rank), dtype=diagonal.dtype)
    pivots = jnp.zeros(n_rank, dtype=jnp.int32)
    selected = jnp.zeros(n_grid, dtype=bool)
    state = (diagonal, factor, pivots, selected)

    # This is a contract, not an optimization shortcut: batch size one uses
    # the same arithmetic as the historical greedy selector at every pivot.
    if batch_size == 1:
        return jax.lax.fori_loop(
            0,
            n_rank,
            lambda step, current: _sequential_round(
                current, step, column_evaluator, shift, tie_break
            ),
            state,
        )[2]

    batched_rank = n_rank - n_topup
    full_rounds = batched_rank // batch_size
    remainder = batched_rank % batch_size
    full_candidate_count = min(candidate_oversampling * batch_size, n_grid)

    def full_round(block, current):
        return _blocked_round(
            current,
            block * batch_size,
            batch_size,
            full_candidate_count,
            column_evaluator,
            shift,
            tie_break,
        )

    state = jax.lax.fori_loop(0, full_rounds, full_round, state)
    next_step = full_rounds * batch_size
    if remainder:
        state = _blocked_round(
            state,
            next_step,
            remainder,
            min(candidate_oversampling * remainder, n_grid),
            column_evaluator,
            shift,
            tie_break,
        )
        next_step += remainder

    def topup_round(offset, current):
        return _sequential_round(
            current, next_step + offset, column_evaluator, shift, tie_break
        )

    return jax.lax.fori_loop(0, n_topup, topup_round, state)[2]


def pivoted_cholesky_streaming(
    diagonal,
    column_evaluator,
    shift,
    *,
    n_rank,
    batch_size=1,
    candidate_oversampling=1,
    n_topup=0,
):
    """Select pivots from an exact diagonal and exact batched-column oracle.

    ``candidate_oversampling`` chooses a stale-diagonal pool of
    ``candidate_oversampling * batch_size`` columns.  Exact greedy re-pivoting
    retains only ``batch_size`` columns, followed by one blocked factor update.
    The final ``n_topup`` pivots use exact greedy singleton updates.
    """
    n_grid = int(diagonal.shape[0])
    validate_pivot_controls(
        n_grid, n_rank, batch_size, candidate_oversampling, n_topup
    )
    if diagonal.ndim != 1:
        raise ValueError("diagonal must be one-dimensional")
    if not callable(column_evaluator):
        raise ValueError("column_evaluator must be callable")
    n_rank = int(n_rank)
    batch_size = int(batch_size)
    candidate_oversampling = int(candidate_oversampling)
    n_topup = int(n_topup)
    return _pivoted_cholesky_streaming(
        diagonal,
        column_evaluator,
        shift,
        n_rank=n_rank,
        batch_size=batch_size,
        candidate_oversampling=candidate_oversampling,
        n_topup=n_topup,
    )


def phi_diagonal(phi_weighted):
    """Diagonal of ``K_rs = (phi[:, r]^T phi[:, s])**2``."""
    orbital = jnp.sum(phi_weighted**2, axis=0)
    return orbital**2


def phi_columns(phi_weighted, indices):
    """Exact requested columns of the weighted molecular density metric."""
    orbital = phi_weighted.T @ phi_weighted[:, indices]
    return orbital**2


def grad_diagonal(phi_weighted, grad_phi_weighted):
    """Diagonal of the weighted molecular density-gradient metric."""
    orbital = jnp.sum(phi_weighted**2, axis=0)
    gradient = jnp.sum(jnp.sum(grad_phi_weighted**2, axis=2), axis=0)
    return orbital * gradient


def grad_columns(phi_weighted, grad_phi_weighted, indices):
    """Exact requested columns of the weighted density-gradient metric.

    The metric convention is
    ``K_rs = (phi_r.T phi_s) * sum_c(grad_phi_r,c.T grad_phi_s,c)``.
    There is no normalization beyond the weights already included in the two
    factors.  The component-wise sum order matches the historical selector.
    """
    orbital = phi_weighted.T @ phi_weighted[:, indices]
    gradient = jnp.zeros_like(orbital)
    for component in range(3):
        values = grad_phi_weighted[:, :, component]
        gradient = gradient + values.T @ values[:, indices]
    return orbital * gradient
