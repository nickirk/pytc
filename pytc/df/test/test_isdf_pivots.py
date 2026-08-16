"""Regression controls for production ISDF pivot selection."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial as JaxPartial

jax.config.update("jax_enable_x64", True)

from pytc.df.isdf import _pivoted_cholesky_grad, _pivoted_cholesky_phi
from pytc.df import pivots as molecular_pivots
from pytc.df.pivots import (
    grad_columns,
    grad_diagonal,
    phi_columns,
    phi_diagonal,
    pivoted_cholesky_streaming,
)
from pytc.xtc import ISDFXTC


@partial(jax.jit, static_argnames=("n_rank",))
def _legacy_phi(phi, shift, *, n_rank):
    """Frozen pre-blocking selector used for full-sequence compatibility."""
    n_grid = phi.shape[1]
    diagonal = jnp.sum(phi**2, axis=0) ** 2 + shift
    tie = 1e-12 * jnp.arange(n_grid, dtype=diagonal.dtype) * jnp.max(
        jnp.abs(diagonal)
    )
    diagonal = diagonal + tie
    factor = jnp.zeros((n_grid, n_rank), dtype=phi.dtype)
    pivots = jnp.zeros(n_rank, dtype=jnp.int32)
    selected = jnp.zeros(n_grid, dtype=bool)

    def body(step, state):
        current_diagonal, current_factor, current_pivots, current_selected = state
        pivot = jnp.argmax(
            jnp.where(current_selected, -jnp.inf, current_diagonal)
        ).astype(jnp.int32)
        current_pivots = current_pivots.at[step].set(pivot)
        value = current_diagonal[pivot]
        column = (phi.T @ phi[:, pivot]) ** 2
        column = column.at[pivot].add(shift + tie[pivot])
        safe_value = jnp.where(value < 1e-12, 1.0, value)
        new_column = (column - current_factor @ current_factor[pivot]) * (
            jax.lax.rsqrt(safe_value)
        )
        new_column = jnp.where(value < 1e-12, 0.0, new_column)
        current_factor = current_factor.at[:, step].set(new_column)
        current_diagonal = jnp.maximum(
            current_diagonal - new_column**2, 0.0
        )
        current_diagonal = current_diagonal.at[pivot].set(0.0)
        current_selected = current_selected.at[pivot].set(True)
        return current_diagonal, current_factor, current_pivots, current_selected

    return jax.lax.fori_loop(
        0, n_rank, body, (diagonal, factor, pivots, selected)
    )[2]


@partial(jax.jit, static_argnames=("n_rank",))
def _legacy_grad(phi, grad, shift, *, n_rank):
    n_grid = phi.shape[1]
    orbital_diagonal = jnp.sum(phi**2, axis=0)
    gradient_diagonal = jnp.sum(jnp.sum(grad**2, axis=2), axis=0)
    diagonal = orbital_diagonal * gradient_diagonal + shift
    tie = 1e-12 * jnp.arange(n_grid, dtype=diagonal.dtype) * jnp.max(
        jnp.abs(diagonal)
    )
    diagonal = diagonal + tie
    factor = jnp.zeros((n_grid, n_rank), dtype=phi.dtype)
    pivots = jnp.zeros(n_rank, dtype=jnp.int32)
    selected = jnp.zeros(n_grid, dtype=bool)

    def body(step, state):
        current_diagonal, current_factor, current_pivots, current_selected = state
        pivot = jnp.argmax(
            jnp.where(current_selected, -jnp.inf, current_diagonal)
        ).astype(jnp.int32)
        current_pivots = current_pivots.at[step].set(pivot)
        value = current_diagonal[pivot]
        orbital = phi.T @ phi[:, pivot]
        gradient = jnp.zeros(n_grid, dtype=phi.dtype)
        for component in range(3):
            values = grad[:, :, component]
            gradient = gradient + values.T @ values[:, pivot]
        column = orbital * gradient
        column = column.at[pivot].add(shift + tie[pivot])
        safe_value = jnp.where(value < 1e-12, 1.0, value)
        new_column = (column - current_factor @ current_factor[pivot]) * (
            jax.lax.rsqrt(safe_value)
        )
        new_column = jnp.where(value < 1e-12, 0.0, new_column)
        current_factor = current_factor.at[:, step].set(new_column)
        current_diagonal = jnp.maximum(
            current_diagonal - new_column**2, 0.0
        )
        current_diagonal = current_diagonal.at[pivot].set(0.0)
        current_selected = current_selected.at[pivot].set(True)
        return current_diagonal, current_factor, current_pivots, current_selected

    return jax.lax.fori_loop(
        0, n_rank, body, (diagonal, factor, pivots, selected)
    )[2]


def _dense_columns(matrix, indices):
    return matrix[:, indices]


def _mutated_sequential_round(
    state, step, column_evaluator, shift, tie_break
):
    """Projection mutation injected into the compiled production schedule."""
    updated = _ORIGINAL_SEQUENTIAL_ROUND(
        state, step, column_evaluator, shift, tie_break
    )
    diagonal, factor, pivots, selected = updated
    factor = factor.at[:, step].set(0.99 * factor[:, step])
    return diagonal, factor, pivots, selected


_ORIGINAL_SEQUENTIAL_ROUND = molecular_pivots._sequential_round


def _has_square_grid_array(closed_jaxpr, n_grid):
    """Inspect nested JAX IR structurally for an ``n_grid x n_grid`` value."""
    seen = set()

    def visit(value):
        value_id = id(value)
        if value_id in seen:
            return False
        seen.add(value_id)
        aval = getattr(value, "aval", None)
        if tuple(getattr(aval, "shape", ())) == (n_grid, n_grid):
            return True
        if isinstance(value, dict):
            return any(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(visit(item) for item in value)
        for attribute in (
            "jaxpr",
            "eqns",
            "invars",
            "outvars",
            "constvars",
            "params",
        ):
            nested = getattr(value, attribute, None)
            if nested is not None and visit(nested):
                return True
        return False

    return visit(closed_jaxpr)


class TestISDFPivotSelection(unittest.TestCase):
    def setUp(self):
        self.phi = jnp.array([[1.0, 1e-3, 1e-3, 1e-3]])
        self.grad = jnp.repeat(self.phi.T, 3, axis=1)[None, :, :]

    def test_phi_tie_break_does_not_reselect_a_live_pivot(self):
        pivots = np.asarray(_pivoted_cholesky_phi(self.phi, 4, jnp.array(1e-12)))
        self.assertEqual(len(np.unique(pivots)), 4)

    def test_gradient_tie_break_does_not_reselect_a_live_pivot(self):
        pivots = np.asarray(
            _pivoted_cholesky_grad(self.phi, self.grad, 4, jnp.array(3e-12))
        )
        self.assertEqual(len(np.unique(pivots)), 4)

    def test_batch_one_matches_full_legacy_sequences(self):
        rng = np.random.default_rng(11)
        phi = jnp.asarray(rng.normal(size=(7, 48)))
        grad = jnp.asarray(rng.normal(size=(7, 48, 3)))
        phi_shift = 1e-12 * jnp.max(jnp.abs(phi_diagonal(phi)))
        grad_shift = 1e-12 * jnp.max(jnp.abs(grad_diagonal(phi, grad)))

        np.testing.assert_array_equal(
            np.asarray(
                pivoted_cholesky_streaming(
                    phi_diagonal(phi),
                    JaxPartial(phi_columns, phi),
                    phi_shift,
                    n_rank=48,
                    batch_size=1,
                    candidate_oversampling=1,
                    n_topup=0,
                )
            ),
            np.asarray(_legacy_phi(phi, phi_shift, n_rank=48)),
        )
        np.testing.assert_array_equal(
            np.asarray(
                pivoted_cholesky_streaming(
                    grad_diagonal(phi, grad),
                    JaxPartial(grad_columns, phi, grad),
                    grad_shift,
                    n_rank=48,
                    batch_size=1,
                    candidate_oversampling=1,
                    n_topup=0,
                )
            ),
            np.asarray(_legacy_grad(phi, grad, grad_shift, n_rank=48)),
        )

    def test_full_sequence_gate_rejects_new_projection_mutation(self):
        rng = np.random.default_rng(19)
        phi = jnp.asarray(rng.normal(size=(6, 32)))
        shift = 1e-12 * jnp.max(jnp.abs(phi_diagonal(phi)))
        expected = np.asarray(_legacy_phi(phi, shift, n_rank=32))
        molecular_pivots._pivoted_cholesky_streaming.clear_cache()
        try:
            with patch.object(
                molecular_pivots,
                "_sequential_round",
                _mutated_sequential_round,
            ):
                mutated = np.asarray(
                    pivoted_cholesky_streaming(
                        phi_diagonal(phi),
                        JaxPartial(phi_columns, phi),
                        shift,
                        n_rank=32,
                        batch_size=1,
                        candidate_oversampling=1,
                        n_topup=0,
                    )
                )
        finally:
            molecular_pivots._pivoted_cholesky_streaming.clear_cache()
        self.assertFalse(
            np.array_equal(expected, mutated),
            "the full-sequence parity gate must reject a new-path projection mutation",
        )

    def test_dense_and_streaming_providers_are_interchangeable(self):
        rng = np.random.default_rng(23)
        phi = jnp.asarray(rng.normal(size=(6, 40)))
        diagonal = phi_diagonal(phi)
        shift = 1e-12 * jnp.max(jnp.abs(diagonal))
        streaming = JaxPartial(phi_columns, phi)
        dense_matrix = phi_columns(phi, jnp.arange(phi.shape[1]))
        dense = JaxPartial(_dense_columns, dense_matrix)
        controls = dict(
            n_rank=24,
            batch_size=4,
            candidate_oversampling=3,
            n_topup=3,
        )
        np.testing.assert_array_equal(
            np.asarray(
                pivoted_cholesky_streaming(
                    diagonal, streaming, shift, **controls
                )
            ),
            np.asarray(
                pivoted_cholesky_streaming(diagonal, dense, shift, **controls)
            ),
        )

    def test_streaming_trace_has_no_grid_square_array(self):
        rng = np.random.default_rng(31)
        phi = jnp.asarray(rng.normal(size=(5, 64)))
        diagonal = phi_diagonal(phi)
        shift = 1e-12 * jnp.max(jnp.abs(diagonal))

        traced = jax.make_jaxpr(
            lambda current_diagonal, current_phi: pivoted_cholesky_streaming(
                current_diagonal,
                JaxPartial(phi_columns, current_phi),
                shift,
                n_rank=16,
                batch_size=4,
                candidate_oversampling=2,
                n_topup=2,
            )
        )(diagonal, phi)
        self.assertFalse(_has_square_grid_array(traced, 64))

        dense_control = jax.make_jaxpr(
            lambda values: (values.T @ values).reshape(-1)
        )(phi)
        self.assertTrue(
            _has_square_grid_array(dense_control, 64),
            "the memory invariant must detect a deliberately materialized Gram",
        )

    def test_all_topup_is_exact_greedy(self):
        rng = np.random.default_rng(47)
        phi = jnp.asarray(rng.normal(size=(5, 32)))
        diagonal = phi_diagonal(phi)
        shift = 1e-12 * jnp.max(jnp.abs(diagonal))
        provider = JaxPartial(phi_columns, phi)
        exact = pivoted_cholesky_streaming(
            diagonal,
            provider,
            shift,
            n_rank=20,
            batch_size=1,
            candidate_oversampling=1,
            n_topup=0,
        )
        topup = pivoted_cholesky_streaming(
            diagonal,
            provider,
            shift,
            n_rank=20,
            batch_size=8,
            candidate_oversampling=4,
            n_topup=20,
        )
        np.testing.assert_array_equal(np.asarray(topup), np.asarray(exact))

    def test_invalid_block_controls_fail_closed(self):
        diagonal = phi_diagonal(self.phi)
        provider = JaxPartial(phi_columns, self.phi)
        for controls in (
            dict(batch_size=0, candidate_oversampling=1, n_topup=0),
            dict(batch_size=2, candidate_oversampling=0, n_topup=0),
            dict(batch_size=2, candidate_oversampling=1, n_topup=5),
        ):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                pivoted_cholesky_streaming(
                    diagonal,
                    provider,
                    jnp.array(1e-12),
                    n_rank=4,
                    **controls,
                )

    def test_from_xtc_forwards_aligned_pivot_controls(self):
        phi = jnp.ones((2, 6))
        grad = jnp.ones((2, 6, 3))
        source = SimpleNamespace(
            grid_points=jnp.zeros((6, 3)),
            weights=jnp.ones(6),
            phi=phi,
            grad_phi=grad,
            n_orb=2,
            grid_lvl=0,
            jastrow_factor=None,
            mo_coeff=jnp.eye(2),
            mo_occ=jnp.asarray([2.0, 0.0]),
            nocc=1,
            energy_nuc=0.0,
        )
        decomposition = (
            phi[:, :2],
            jnp.ones((2, 6)),
            grad[:, :2],
            jnp.ones((2, 6, 3)),
            jnp.asarray([0, 1]),
            None,
        )
        with patch("pytc.df.isdf_decompose", return_value=decomposition) as mocked:
            ISDFXTC.from_xtc(
                source,
                n_rank=2,
                is_incore=True,
                batch_size=8,
                candidate_oversampling=4,
                n_topup=1,
            )
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["batch_size"], 8)
        self.assertEqual(kwargs["candidate_oversampling"], 4)
        self.assertEqual(kwargs["n_topup"], 1)


if __name__ == "__main__":
    unittest.main()
