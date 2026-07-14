"""Direct tests for pytc.df.ibp's canonical single-IBP primitives and the
typed numerical artifacts (grid, operator plan, interpolation sector, Z core,
PSD factorization).

Molecular end-to-end regression coverage (H2O atom-centered grid vs exact
4-center/analytic-DF references) stays in
pytc/test/test_atom_centered_single_ibp.py.
"""

import subprocess
import sys
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pytc.df.ibp import (
    IBPCoreArtifact,
    IBPGrid,
    IBPInterpolationSector,
    IBPOperatorPlan,
    PSDFactorization,
    _validate_grid_inputs,
    build_ibp_grid,
    build_ibp_interpolation_sector,
    build_ibp_operator_plan,
    ibp_core,
    kernel,
    psd_factorize,
)


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
    """Diagnostic ``1/r`` quadrature oracle with coincident terms set to zero.

    This is *not* a controlled production self-cell prescription; it exists as
    a test oracle to quantify how much the bounded single-IBP ``kernel`` helps
    relative to the naive singular grid sum on the same atom-centered points.
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


def _normalized_frobenius_residual_ref(a, b):
    """Independent reference for ||a-b||_F/||a||_F (0/0 -> 0) so the tests do
    not verify the module against its own helper."""
    num = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    den = float(np.linalg.norm(np.asarray(a)))
    return 0.0 if den == 0.0 else num / den


class TestAtomCenteredSingleIBPCore(unittest.TestCase):
    def _case(self, dtype):
        rng = np.random.default_rng(21)
        coords = rng.normal(size=(9, 3))
        weights = rng.normal(size=9)
        density = rng.normal(size=(4, 9))
        gradient = rng.normal(size=(3, 3, 9))
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            density = density + 1j * rng.normal(size=density.shape)
            gradient = gradient + 1j * rng.normal(size=gradient.shape)
        density = density.astype(dtype)
        gradient = gradient.astype(dtype)
        return coords, weights, density, gradient

    def test_single_ibp_matches_explicit_dense_real_and_complex(self):
        for dtype in (np.float64, np.complex128):
            coords, weights, density, gradient = self._case(dtype)
            diff = coords[:, None, :] - coords[None, :, :]
            radius = np.linalg.norm(diff, axis=-1)
            rhat = np.divide(
                diff, radius[..., None], out=np.zeros_like(diff),
                where=radius[..., None] != 0.0,
            )
            vector = np.einsum("nj,j,ijc->nci", density, weights, rhat)
            expected = -0.5 * np.einsum(
                "mci,nci,i->mn", gradient.conj(), vector, weights
            )
            actual, coincident = kernel(
                gradient, density, coords, weights,
                eval_block_size=4, source_block_size=3,
            )
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
            self.assertEqual(coincident, len(weights))

    def test_direct_offdiagonal_matches_explicit_dense(self):
        coords, weights, density, _ = self._case(np.complex128)
        diff = coords[:, None, :] - coords[None, :, :]
        radius = np.linalg.norm(diff, axis=-1)
        inv_r = np.divide(1.0, radius, out=np.zeros_like(radius), where=radius != 0.0)
        expected = np.einsum(
            "mi,i,ij,nj,j->mn", density.conj(), weights, inv_r, density, weights
        )
        actual, coincident = naive_coulomb_kernel(
            density, density, coords, weights,
            eval_block_size=4, source_block_size=3,
        )
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
        self.assertEqual(coincident, len(weights))

    def test_cross_grid_matches_dense_and_has_no_coincident_points(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(7, 3))
        y = rng.normal(size=(8, 3)) + 0.123
        wx, wy = rng.normal(size=7), rng.normal(size=8)
        grad = rng.normal(size=(2, 3, 7))
        rho = rng.normal(size=(3, 8))
        diff = x[:, None, :] - y[None, :, :]
        radius = np.linalg.norm(diff, axis=-1)
        rhat = diff / radius[..., None]
        vector = np.einsum("nj,j,ijc->nci", rho, wy, rhat)
        expected = -0.5 * np.einsum("mci,nci,i->mn", grad, vector, wx)
        actual, coincident = kernel(
            grad, rho, x, wx, coords_right=y, weights_right=wy,
            eval_block_size=3, source_block_size=5,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        self.assertEqual(coincident, 0)

    def test_block_sizes_do_not_change_result(self):
        coords, weights, density, gradient = self._case(np.float64)
        small, _ = kernel(
            gradient, density, coords, weights,
            eval_block_size=2, source_block_size=2,
        )
        full, _ = kernel(
            gradient, density, coords, weights,
            eval_block_size=100, source_block_size=100,
        )
        np.testing.assert_allclose(small, full, atol=3e-12, rtol=3e-12)

    def test_block_sizes_do_not_change_result_offdiagonal(self):
        coords, weights, density, _ = self._case(np.complex128)
        small, _ = naive_coulomb_kernel(
            density, density, coords, weights,
            eval_block_size=2, source_block_size=2,
        )
        full, _ = naive_coulomb_kernel(
            density, density, coords, weights,
            eval_block_size=100, source_block_size=100,
        )
        np.testing.assert_allclose(small, full, atol=3e-12, rtol=3e-12)

    def test_rejects_malformed_inputs(self):
        coords, weights, density, gradient = self._case(np.float64)
        with self.assertRaises(ValueError):
            kernel(gradient[:, :2], density, coords, weights)
        with self.assertRaises(ValueError):
            kernel(
                gradient.astype(np.float32), density, coords, weights
            )
        with self.assertRaises(ValueError):
            naive_coulomb_kernel(
                density, density, coords, weights, eval_block_size=0
            )
        with self.assertRaises(ValueError):
            # coords shape mismatch: wrong number of grid points
            kernel(
                gradient, density, coords[:-1], weights
            )
        with self.assertRaises(ValueError):
            # non-finite coordinates
            bad_coords = coords.copy()
            bad_coords[0, 0] = np.nan
            kernel(gradient, density, bad_coords, weights)
        with self.assertRaises(ValueError):
            # dtype mismatch between gradient and density_right
            kernel(
                gradient.astype(np.complex128), density, coords, weights
            )

    def test_explicit_coincident_point_rhat_zero_convention(self):
        """A single point coincident with itself must contribute rhat=0,
        not NaN/inf -- and coincident_pairs must report it."""
        coords = np.zeros((1, 3))
        weights = np.ones(1)
        density = np.ones((1, 1))
        gradient = np.ones((1, 3, 1))
        result, coincident = kernel(
            gradient, density, coords, weights,
        )
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, np.zeros((1, 1)))
        self.assertEqual(coincident, 1)


class TestIBPGrid(unittest.TestCase):
    def _random_grid_arrays(self, n=6, seed=1):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, 3)), rng.normal(size=n)

    def test_numpy_grid_round_trip(self):
        coords, weights = self._random_grid_arrays()
        grid = build_ibp_grid(coords, weights)
        self.assertEqual(grid.n_grid, 6)
        self.assertEqual(grid.backend, "numpy")
        self.assertEqual(grid.device, "cpu")
        self.assertEqual(grid.dtype, "float64")
        self.assertEqual(grid.coincident_point_policy, "centered_zero")
        np.testing.assert_allclose(grid.coords, coords)
        np.testing.assert_allclose(grid.weights, weights)

    def test_numpy_grid_defensive_copy_not_caller_array(self):
        coords, weights = self._random_grid_arrays()
        grid = build_ibp_grid(coords, weights)
        self.assertIsNot(grid.coords, coords)
        self.assertIsNot(grid.weights, weights)
        self.assertFalse(grid.coords.flags.writeable)
        self.assertFalse(grid.weights.flags.writeable)
        coords[0, 0] = 999.0
        weights[0] = 999.0
        self.assertNotEqual(grid.coords[0, 0], 999.0)
        self.assertNotEqual(grid.weights[0], 999.0)

    def test_jax_grid_round_trip(self):
        coords_np, weights_np = self._random_grid_arrays()
        coords = jnp.asarray(coords_np)
        weights = jnp.asarray(weights_np)
        grid = build_ibp_grid(coords, weights, backend="jax")
        self.assertEqual(grid.backend, "jax")
        self.assertEqual(grid.device, str(coords.device))
        self.assertIs(grid.coords, coords)
        self.assertIs(grid.weights, weights)

    def test_jax_grid_requires_jax_array(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(TypeError):
            build_ibp_grid(coords, weights, backend="jax")

    def test_rejects_malformed_shapes(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords[:, :2], weights)
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, weights[:-1])
        with self.assertRaises(ValueError):
            build_ibp_grid(np.empty((0, 3)), np.empty(0))

    def test_rejects_non_finite(self):
        coords, weights = self._random_grid_arrays()
        bad_coords = coords.copy()
        bad_coords[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_ibp_grid(bad_coords, weights)
        bad_weights = weights.copy()
        bad_weights[0] = np.inf
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, bad_weights)

    def test_rejects_dtype_mismatch(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords.astype(np.float32), weights.astype(np.float64))

    def test_rejects_complex_dtype(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords.astype(np.complex128), weights.astype(np.complex128))

    def test_rejects_unsupported_backend(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, weights, backend="numba")

    def test_rejects_unsupported_coincident_point_policy(self):
        coords, weights = self._random_grid_arrays()
        with self.assertRaises(ValueError):
            build_ibp_grid(coords, weights, coincident_point_policy="skip")


class TestIBPOperatorPlan(unittest.TestCase):
    def _grid(self, n=6, seed=3):
        rng = np.random.default_rng(seed)
        return build_ibp_grid(rng.normal(size=(n, 3)), rng.normal(size=n))

    def test_plan_round_trip_numpy(self):
        grid = self._grid()
        plan = build_ibp_operator_plan(grid, eval_block_size=16, source_block_size=64)
        self.assertIs(plan.grid, grid)
        self.assertEqual(plan.method, "direct")
        self.assertEqual(plan.eval_block_size, 16)
        self.assertEqual(plan.source_block_size, 64)
        self.assertIsNone(plan.tolerance)
        self.assertEqual(plan.backend, grid.backend)
        self.assertEqual(plan.device, grid.device)
        self.assertEqual(plan.dtype, grid.dtype)

    def test_plan_jax_backend_round_trip(self):
        rng = np.random.default_rng(4)
        coords = jnp.asarray(rng.normal(size=(5, 3)))
        weights = jnp.asarray(rng.normal(size=5))
        grid = build_ibp_grid(coords, weights, backend="jax")
        plan = build_ibp_operator_plan(grid)
        self.assertEqual(plan.backend, "jax")
        self.assertEqual(plan.device, grid.device)
        self.assertEqual(plan.dtype, grid.dtype)

    def test_plan_requires_ibp_grid_type(self):
        with self.assertRaises(TypeError):
            build_ibp_operator_plan("not-a-grid")

    def test_plan_rejects_unsupported_method(self):
        grid = self._grid()
        for method in ("hierarchical", "nufft", "fft"):
            with self.assertRaises(ValueError):
                build_ibp_operator_plan(grid, method=method)

    def test_plan_rejects_tolerance_for_direct(self):
        grid = self._grid()
        with self.assertRaises(ValueError):
            build_ibp_operator_plan(grid, tolerance=1e-6)

    def test_plan_rejects_bad_block_sizes(self):
        grid = self._grid()
        for kwargs in (
            {"eval_block_size": 0}, {"eval_block_size": -4},
            {"source_block_size": 0}, {"eval_block_size": True},
            {"eval_block_size": 3.5},
        ):
            with self.assertRaises(ValueError):
                build_ibp_operator_plan(grid, **kwargs)

    def test_plan_block_size_accepts_integral_float(self):
        grid = self._grid()
        plan = build_ibp_operator_plan(grid, eval_block_size=8.0, source_block_size=32.0)
        self.assertEqual(plan.eval_block_size, 8)
        self.assertIsInstance(plan.eval_block_size, int)

    def test_plan_holds_no_array_data_of_its_own(self):
        import dataclasses
        grid = self._grid()
        plan = build_ibp_operator_plan(grid)
        for field in dataclasses.fields(plan):
            if field.name == "grid":
                continue
            value = getattr(plan, field.name)
            self.assertNotIsInstance(value, np.ndarray)
            self.assertNotIsInstance(value, jax.Array)

    def test_plan_scales_to_large_grid_without_materializing_pairs(self):
        rng = np.random.default_rng(5)
        n = 20000
        grid = build_ibp_grid(rng.normal(size=(n, 3)), rng.normal(size=n))
        # Must be cheap -- a dense (3, n, n) kernel here would be ~9.6GB at
        # float64, and building this plan does none of that work.
        plan = build_ibp_operator_plan(grid, eval_block_size=128, source_block_size=4096)
        self.assertEqual(plan.grid.n_grid, n)


class TestIBPInterpolationSector(unittest.TestCase):
    def _case(self, dtype=np.float64, *, same_factor=False, seed=30):
        rng = np.random.default_rng(seed)
        n_grid = 13
        n_p = 3 if same_factor else 2
        n_q = n_p if same_factor else 3
        factor_p = rng.normal(size=(n_p, n_grid))
        factor_q = factor_p.copy() if same_factor else rng.normal(size=(n_q, n_grid))
        gradient_p = rng.normal(size=(3, n_p, n_grid))
        gradient_q = gradient_p.copy() if same_factor else rng.normal(size=(3, n_q, n_grid))
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            factor_p = factor_p + 1j * rng.normal(size=factor_p.shape)
            factor_q = (
                factor_p.copy()
                if same_factor
                else factor_q + 1j * rng.normal(size=factor_q.shape)
            )
            gradient_p = gradient_p + 1j * rng.normal(size=gradient_p.shape)
            gradient_q = (
                gradient_p.copy()
                if same_factor
                else gradient_q + 1j * rng.normal(size=gradient_q.shape)
            )
        factor_p = factor_p.astype(dtype)
        factor_q = factor_q.astype(dtype)
        gradient_p = gradient_p.astype(dtype)
        gradient_q = gradient_q.astype(dtype)
        real_dtype = np.empty((), dtype=dtype).real.dtype
        coords = rng.normal(size=(n_grid, 3)).astype(real_dtype)
        weights = rng.random(n_grid).astype(real_dtype)
        grid = build_ibp_grid(coords, weights)
        return grid, factor_p, factor_q, gradient_p, gradient_q

    def _rank_record(self, n_p, n_q, n_pivots, *, same_factor=False,
                     requested_rank=None, exhausted=False):
        analytic = n_p * (n_p + 1) // 2 if same_factor else n_p * n_q
        requested = n_pivots if requested_rank is None else requested_rank
        capped = min(requested, analytic)
        return {
            "requested_rank": requested,
            "analytic_rank_bound": analytic,
            "n_rank_capped": capped,
            "rank_exhausted": exhausted,
            "numerical_rank": n_pivots if exhausted else None,
            "numerical_rank_lower_bound": n_pivots,
            "n_pivots": n_pivots,
        }

    def _build(self, dtype=np.float64, *, same_factor=False, batch=4, seed=30):
        case = self._case(dtype, same_factor=same_factor, seed=seed)
        grid, factor_p, factor_q, gradient_p, gradient_q = case
        n_pivots = min(5, factor_p.shape[0] * factor_q.shape[0])
        pivots = np.array([0, 2, 5, 8, 11][:n_pivots])
        record = self._rank_record(
            factor_p.shape[0], factor_q.shape[0], n_pivots,
            same_factor=same_factor,
        )
        sector = build_ibp_interpolation_sector(
            factor_p, factor_q, gradient_p, gradient_q, pivots, grid,
            pivot_provenance=record, same_factor=same_factor,
            grid_batch_size=batch, rcond=1e-14,
        )
        return sector, case, pivots, record

    def test_real_and_complex_match_independent_dense_lstsq(self):
        for dtype in (np.float64, np.complex128):
            sector, case, pivots, _ = self._build(dtype)
            _, factor_p, factor_q, gradient_p, gradient_q = case
            collocation = np.einsum(
                "pu,qu->pqu", factor_p[:, pivots], factor_q[:, pivots]
            ).reshape(factor_p.shape[0] * factor_q.shape[0], len(pivots))
            pair_grid = np.einsum("pg,qg->pqg", factor_p, factor_q).reshape(
                factor_p.shape[0] * factor_q.shape[0], factor_p.shape[1]
            )
            theta_expected = np.linalg.lstsq(collocation, pair_grid, rcond=None)[0]
            gradients_expected = []
            for axis in range(3):
                derivative = (
                    np.einsum("pg,qg->pqg", gradient_p[axis], factor_q)
                    + np.einsum("pg,qg->pqg", factor_p, gradient_q[axis])
                ).reshape(factor_p.shape[0] * factor_q.shape[0], factor_p.shape[1])
                gradients_expected.append(
                    np.linalg.lstsq(collocation, derivative, rcond=None)[0]
                )
            gradient_expected = np.stack(gradients_expected, axis=1)
            np.testing.assert_allclose(sector.P, collocation.T, atol=3e-11, rtol=3e-11)
            np.testing.assert_allclose(
                sector.Theta, theta_expected, atol=3e-11, rtol=3e-11
            )
            np.testing.assert_allclose(
                sector.grad_Theta, gradient_expected, atol=5e-11, rtol=5e-11
            )
            self.assertEqual(sector.pair_layout, "full_row_major")

    def test_float32_and_complex64_are_explicitly_gated_against_dense_lstsq(self):
        for dtype in (np.float32, np.complex64):
            sector, case, pivots, _ = self._build(dtype, seed=35)
            _, factor_p, factor_q, gradient_p, gradient_q = case
            collocation = np.einsum(
                "pu,qu->pqu", factor_p[:, pivots], factor_q[:, pivots]
            ).reshape(factor_p.shape[0] * factor_q.shape[0], len(pivots))
            pair_grid = np.einsum("pg,qg->pqg", factor_p, factor_q).reshape(
                factor_p.shape[0] * factor_q.shape[0], factor_p.shape[1]
            )
            theta_expected = np.linalg.lstsq(collocation, pair_grid, rcond=None)[0]
            gradients_expected = []
            for axis in range(3):
                derivative = (
                    np.einsum("pg,qg->pqg", gradient_p[axis], factor_q)
                    + np.einsum("pg,qg->pqg", factor_p, gradient_q[axis])
                ).reshape(factor_p.shape[0] * factor_q.shape[0], factor_p.shape[1])
                gradients_expected.append(
                    np.linalg.lstsq(collocation, derivative, rcond=None)[0]
                )
            gradient_expected = np.stack(gradients_expected, axis=1)
            self.assertEqual(sector.realized_dtype, np.dtype(dtype).name)
            np.testing.assert_allclose(
                sector.Theta, theta_expected, atol=3e-4, rtol=3e-4
            )
            np.testing.assert_allclose(
                sector.grad_Theta, gradient_expected, atol=5e-4, rtol=5e-4
            )

    def test_numpy_float64_rejects_cleanly_when_jax_x64_is_disabled(self):
        code = "\n".join((
            "import jax",
            "jax.config.update('jax_enable_x64', False)",
            "import numpy as np",
            "from pytc.df.ibp import build_ibp_grid, build_ibp_interpolation_sector",
            "coords = np.arange(18, dtype=np.float64).reshape(6, 3)",
            "weights = np.ones(6, dtype=np.float64)",
            "grid = build_ibp_grid(coords, weights)",
            "fp = np.arange(12, dtype=np.float64).reshape(2, 6) + 1",
            "fq = np.arange(18, dtype=np.float64).reshape(3, 6) + 2",
            "gp = np.ones((3, 2, 6), dtype=np.float64)",
            "gq = np.ones((3, 3, 6), dtype=np.float64)",
            "record = {'requested_rank': 3, 'analytic_rank_bound': 6, "
            "'n_rank_capped': 3, 'rank_exhausted': False, "
            "'numerical_rank': None, 'numerical_rank_lower_bound': 3, "
            "'n_pivots': 3}",
            "try:",
            "    build_ibp_interpolation_sector(fp, fq, gp, gq, "
            "np.array([0, 2, 4]), grid, pivot_provenance=record)",
            "except ValueError as exc:",
            "    assert 'jax_enable_x64' in str(exc)",
            "    print('EXPECTED')",
            "else:",
            "    raise AssertionError('float64 sector silently downcast')",
        ))
        result = subprocess.run(
            [sys.executable, "-W", "error", "-c", code],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.strip().splitlines()[-1], "EXPECTED")

    def test_accepts_exact_canonical_selector_provenance(self):
        from pytc.integrals.coulomb import select_sector_pivots, weight_mo_values

        grid, factor_p, factor_q, gradient_p, gradient_q = self._case(seed=36)
        weighted_p = weight_mo_values(factor_p, grid.weights)
        weighted_q = weight_mo_values(factor_q, grid.weights)
        pivots, record = select_sector_pivots(
            weighted_p, weighted_q, 5, return_provenance=True
        )
        sector = build_ibp_interpolation_sector(
            factor_p, factor_q, gradient_p, gradient_q, pivots, grid,
            pivot_provenance=record,
        )
        self.assertEqual(sector.requested_rank, record["requested_rank"])
        self.assertEqual(sector.selected_rank, record["n_pivots"])

    def test_same_factor_uses_packed_lower_pair_layout(self):
        sector, case, pivots, _ = self._build(np.float64, same_factor=True)
        _, factor, _, _, _ = case
        pair_p, pair_q = np.tril_indices(factor.shape[0])
        expected = (
            factor[pair_p[:, None], pivots[None, :]]
            * factor[pair_q[:, None], pivots[None, :]]
        ).T
        np.testing.assert_allclose(sector.P, expected)
        self.assertEqual(sector.pair_layout, "packed_lower")
        self.assertEqual(sector.n_pair, factor.shape[0] * (factor.shape[0] + 1) // 2)

    def test_grid_batching_is_numerically_invariant(self):
        small, _, _, _ = self._build(np.complex128, batch=2, seed=31)
        full, _, _, _ = self._build(np.complex128, batch=1000, seed=31)
        np.testing.assert_allclose(small.Theta, full.Theta, atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(
            small.grad_Theta, full.grad_Theta, atol=3e-12, rtol=3e-12
        )

    def test_prepared_solver_is_built_once_and_reused_for_all_gradients(self):
        case = self._case()
        grid, factor_p, factor_q, gradient_p, gradient_q = case
        pivots = np.array([0, 2, 5, 8, 11])
        record = self._rank_record(2, 3, len(pivots))
        import pytc.df.ibp as ibp_module
        original_prepare = ibp_module.prepare_normal_equations_solver
        original_solve = ibp_module.solve_normal_equations_batch_prepared
        with (
            mock.patch.object(
                ibp_module, "prepare_normal_equations_solver", wraps=original_prepare
            ) as prepare_spy,
            mock.patch.object(
                ibp_module, "solve_normal_equations_batch_prepared", wraps=original_solve
            ) as solve_spy,
        ):
            build_ibp_interpolation_sector(
                factor_p, factor_q, gradient_p, gradient_q, pivots, grid,
                pivot_provenance=record, grid_batch_size=4,
            )
        self.assertEqual(prepare_spy.call_count, 1)
        n_batches = 4
        self.assertEqual(solve_spy.call_count, n_batches * 7)

    def test_rank_exhaustion_record_is_preserved_without_conflation(self):
        case = self._case(seed=32)
        grid, factor_p, factor_q, gradient_p, gradient_q = case
        pivots = np.array([0, 2, 5])
        record = self._rank_record(2, 3, 3, requested_rank=5, exhausted=True)
        sector = build_ibp_interpolation_sector(
            factor_p, factor_q, gradient_p, gradient_q, pivots, grid,
            pivot_provenance=record,
        )
        self.assertEqual(sector.requested_rank, 5)
        self.assertEqual(sector.selected_rank, 3)
        self.assertEqual(sector.numerical_rank, 3)

    def test_rejects_inconsistent_rank_records(self):
        case = self._case()
        grid, factor_p, factor_q, gradient_p, gradient_q = case
        pivots = np.array([0, 2, 5, 8, 11])
        good = self._rank_record(2, 3, len(pivots))
        for update in (
            {"n_pivots": 4},
            {"analytic_rank_bound": 99},
            {"numerical_rank": 5},
            {"rank_exhausted": True, "numerical_rank": 5},
        ):
            bad = {**good, **update}
            with self.assertRaises((TypeError, ValueError)):
                build_ibp_interpolation_sector(
                    factor_p, factor_q, gradient_p, gradient_q, pivots, grid,
                    pivot_provenance=bad,
                )
        with self.assertRaises(ValueError):
            build_ibp_interpolation_sector(
                factor_p, factor_q, gradient_p, gradient_q, pivots, grid,
                pivot_provenance={**good, "extra": 1},
            )

    def test_numpy_outputs_are_immutable(self):
        sector, _, _, _ = self._build()
        self.assertIsInstance(sector, IBPInterpolationSector)
        for array in (sector.P, sector.Theta, sector.grad_Theta, sector.pivots):
            self.assertFalse(array.flags.writeable)

    def test_rejects_malformed_inputs_and_same_factor_lies(self):
        case = self._case()
        grid, factor_p, factor_q, gradient_p, gradient_q = case
        pivots = np.array([0, 2, 5, 8, 11])
        record = self._rank_record(2, 3, len(pivots))
        bad_gradient = gradient_p.copy()
        bad_gradient[0, 0, 0] = np.nan
        for args in (
            (factor_p[:, :-1], factor_q, gradient_p[:, :, :-1], gradient_q),
            (factor_p, factor_q, bad_gradient, gradient_q),
            (factor_p.astype(np.float32), factor_q, gradient_p, gradient_q),
        ):
            with self.assertRaises(ValueError):
                build_ibp_interpolation_sector(
                    *args, pivots, grid, pivot_provenance=record,
                )
        same_grid, fp, fq, gp, gq = self._case(same_factor=True)
        fq = fq.copy()
        fq[0, 0] += 1.0
        same_record = self._rank_record(3, 3, 5, same_factor=True)
        with self.assertRaises(ValueError):
            build_ibp_interpolation_sector(
                fp, fq, gp, gq, pivots, same_grid,
                pivot_provenance=same_record, same_factor=True,
            )

    def test_jax_backend_preserves_device(self):
        numpy_case = self._case(np.float64, seed=33)
        np_grid, fp_np, fq_np, gp_np, gq_np = numpy_case
        coords = jnp.asarray(np_grid.coords)
        weights = jnp.asarray(np_grid.weights)
        grid = build_ibp_grid(coords, weights, backend="jax")
        inputs = tuple(jnp.asarray(a) for a in (fp_np, fq_np, gp_np, gq_np))
        pivots = np.array([0, 2, 5, 8, 11])
        record = self._rank_record(2, 3, len(pivots))
        sector = build_ibp_interpolation_sector(
            *inputs, pivots, grid, pivot_provenance=record,
        )
        self.assertEqual(sector.backend, "jax")
        self.assertEqual(sector.device, grid.device)
        for array in (sector.P, sector.Theta, sector.grad_Theta, sector.pivots):
            self.assertIsInstance(array, jax.Array)
            self.assertEqual(str(array.device), grid.device)

    def test_numpy_and_jax_agree_for_all_supported_precisions(self):
        for dtype in (np.float32, np.complex64, np.float64, np.complex128):
            case = self._case(dtype, seed=37)
            np_grid, fp, fq, gp, gq = case
            pivots = np.array([0, 2, 5, 8, 11])
            record = self._rank_record(2, 3, len(pivots))
            numpy_sector = build_ibp_interpolation_sector(
                fp, fq, gp, gq, pivots, np_grid,
                pivot_provenance=record, grid_batch_size=3,
            )
            jax_grid = build_ibp_grid(
                jnp.asarray(np_grid.coords), jnp.asarray(np_grid.weights), backend="jax",
            )
            jax_sector = build_ibp_interpolation_sector(
                *(jnp.asarray(a) for a in (fp, fq, gp, gq)),
                pivots, jax_grid, pivot_provenance=record, grid_batch_size=3,
            )
            real_itemsize = np.empty((), dtype=dtype).real.dtype.itemsize
            tolerance = 5e-4 if real_itemsize == 4 else 5e-11
            for numpy_value, jax_value in (
                (numpy_sector.P, jax_sector.P),
                (numpy_sector.Theta, jax_sector.Theta),
                (numpy_sector.grad_Theta, jax_sector.grad_Theta),
            ):
                np.testing.assert_allclose(
                    numpy_value, np.asarray(jax_value), atol=tolerance, rtol=tolerance
                )


class TestPSDFactorize(unittest.TestCase):
    """Direct unit tests for psd_factorize -- the reusable Hermitian PSD gate."""

    def test_positive_semidefinite_exact_reconstruction(self):
        # Full-rank PSD (square generator, well-conditioned) so every mode is
        # genuinely positive -- no numerically-ambiguous null space.
        for dtype in (np.float64, np.complex128):
            rng = np.random.default_rng(7)
            a = rng.normal(size=(6, 6))
            if np.issubdtype(np.dtype(dtype), np.complexfloating):
                a = a + 1j * rng.normal(size=(6, 6))
            z = a @ a.conj().T + 0.5 * np.eye(6)  # PSD, full rank, conditioned
            result = psd_factorize(z.astype(dtype))
            self.assertIsInstance(result, PSDFactorization)
            self.assertEqual(result.negative_mode_count, 0)
            self.assertEqual(result.clipped_mode_count, 0)
            self.assertEqual(result.clipped_absolute_weight, 0.0)
            self.assertEqual(result.retained_rank, 6)
            self.assertEqual(result.factor.shape, (6, 6))
            self.assertFalse(result.factor.flags.writeable)
            recon = result.factor @ result.factor.conj().T
            np.testing.assert_allclose(recon, z, rtol=1e-10, atol=1e-10)
            self.assertLess(result.reconstruction_residual, 1e-10)

    def test_roundoff_negative_modes_are_clipped(self):
        rng = np.random.default_rng(11)
        u, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        scale = 1.0
        eigs = np.array([scale, 0.6 * scale, 0.3 * scale, -1e-13 * scale, -3e-13 * scale])
        z = (u * eigs) @ u.conj().T
        z = (z + z.conj().T) / 2
        result = psd_factorize(z, rtol=1e-10)
        self.assertEqual(result.negative_mode_count, 2)
        self.assertEqual(result.clipped_mode_count, 2)
        self.assertEqual(result.retained_rank, 3)
        self.assertGreater(result.clipped_absolute_weight, 0.0)
        self.assertLess(result.clipped_absolute_weight, 1e-12)
        self.assertEqual(result.factor.shape, (5, 3))

    def test_material_negative_mode_hard_fails(self):
        rng = np.random.default_rng(13)
        u, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        eigs = np.array([1.0, 0.5, 0.2, -1e-4])  # far beyond the roundoff band
        z = (u * eigs) @ u.conj().T
        z = (z + z.conj().T) / 2
        with self.assertRaisesRegex(ValueError, "material negative"):
            psd_factorize(z, rtol=1e-10)

    def test_zero_core_normalizes_to_zero_rank(self):
        z = np.zeros((4, 4))
        result = psd_factorize(z)
        self.assertEqual(result.spectral_scale, 0.0)
        self.assertEqual(result.raw_min_eigenvalue, 0.0)
        self.assertEqual(result.retained_rank, 0)
        self.assertEqual(result.reconstruction_residual, 0.0)
        self.assertEqual(result.factor.shape, (4, 0))

    def test_rejects_nonsquare_and_nonfinite(self):
        with self.assertRaises(ValueError):
            psd_factorize(np.zeros((3, 4)))
        z = np.eye(3)
        z[0, 0] = np.nan
        with self.assertRaises(ValueError):
            psd_factorize(z)

    def test_rejects_negative_rtol(self):
        with self.assertRaises(ValueError):
            psd_factorize(np.eye(3), rtol=-1e-10)

    def test_rejects_non_hermitian_input(self):
        # eigh reads one triangle, so a non-Hermitian input would otherwise be
        # "factorized" against its Hermitian completion with a large residual.
        z = np.array([[1.0, 100.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "not Hermitian"):
            psd_factorize(z)

    def test_rejects_bool_and_nonfinite_rtol(self):
        z = np.eye(3)
        for bad in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                psd_factorize(z, rtol=bad)


class TestIBPCoreArtifact(unittest.TestCase):
    def _record(self, n_pivots, analytic_bound):
        return {
            "requested_rank": n_pivots, "analytic_rank_bound": analytic_bound,
            "n_rank_capped": min(n_pivots, analytic_bound), "rank_exhausted": False,
            "numerical_rank": None, "numerical_rank_lower_bound": n_pivots,
            "n_pivots": n_pivots,
        }

    def _setup(self, dtype=np.float64, seed=40, n_grid=14, eval_block_size=5,
              source_block_size=6):
        rng = np.random.default_rng(seed)
        coords = rng.normal(size=(n_grid, 3))
        weights = rng.random(n_grid)
        grid = build_ibp_grid(coords, weights)
        plan = build_ibp_operator_plan(
            grid, eval_block_size=eval_block_size, source_block_size=source_block_size
        )

        n_a, n_b = 2, 3
        factor_a = rng.normal(size=(n_a, n_grid))
        factor_b = rng.normal(size=(n_b, n_grid))
        grad_a = rng.normal(size=(3, n_a, n_grid))
        grad_b = rng.normal(size=(3, n_b, n_grid))
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            factor_a = factor_a + 1j * rng.normal(size=factor_a.shape)
            factor_b = factor_b + 1j * rng.normal(size=factor_b.shape)
            grad_a = grad_a + 1j * rng.normal(size=grad_a.shape)
            grad_b = grad_b + 1j * rng.normal(size=grad_b.shape)
        factor_a, factor_b = factor_a.astype(dtype), factor_b.astype(dtype)
        grad_a, grad_b = grad_a.astype(dtype), grad_b.astype(dtype)

        n_same = n_a * (n_a + 1) // 2
        pivots_same = np.arange(n_grid)[:n_same]
        sector_a = build_ibp_interpolation_sector(
            factor_a, factor_a, grad_a, grad_a, pivots_same, grid,
            pivot_provenance=self._record(n_same, n_same), same_factor=True,
        )
        pivots_cross = np.arange(n_grid)[:5]
        sector_b = build_ibp_interpolation_sector(
            factor_a, factor_b, grad_a, grad_b, pivots_cross, grid,
            pivot_provenance=self._record(5, n_a * n_b),
        )
        return grid, plan, sector_a, sector_b

    def test_same_sector_matches_independent_kernel_call(self):
        # one_sided: the raw quadrature is what this verifies against kernel().
        # A random nonphysical same-sector core is materially indefinite, so
        # the production two_sided PSD gate rejects it by design; that path is
        # covered separately with an injected PSD/indefinite Z.
        for dtype in (np.float64, np.complex128):
            grid, plan, sector_a, _ = self._setup(dtype)
            core = ibp_core(sector_a, operator=plan, symmetry_mode="one_sided")
            expected_forward, expected_coincident = kernel(
                sector_a.grad_Theta, sector_a.Theta, grid.coords, grid.weights,
                eval_block_size=plan.eval_block_size, source_block_size=plan.source_block_size,
            )
            np.testing.assert_allclose(core.z_forward_one_sided, expected_forward, atol=1e-12)
            np.testing.assert_allclose(core.Z, expected_forward, atol=1e-12)
            self.assertEqual(core.coincident_pairs, expected_coincident)
            self.assertTrue(core.same_sector)
            self.assertIsNone(core.z_reverse_one_sided)
            self.assertEqual(core.psd_status, "not_applicable")
            expected_residual = _normalized_frobenius_residual_ref(
                expected_forward, expected_forward.conj().T
            )
            self.assertAlmostEqual(core.raw_dagger_residual, expected_residual, places=10)

    def test_cross_sector_matches_independent_kernel_calls_both_orientations(self):
        grid, plan, sector_a, sector_b = self._setup(np.complex128)
        core = ibp_core(sector_a, sector_b, operator=plan)
        expected_forward, _ = kernel(
            sector_a.grad_Theta, sector_b.Theta, grid.coords, grid.weights,
            eval_block_size=plan.eval_block_size, source_block_size=plan.source_block_size,
        )
        expected_reverse, _ = kernel(
            sector_b.grad_Theta, sector_a.Theta, grid.coords, grid.weights,
            eval_block_size=plan.eval_block_size, source_block_size=plan.source_block_size,
        )
        np.testing.assert_allclose(core.z_forward_one_sided, expected_forward, atol=1e-12)
        np.testing.assert_allclose(core.z_reverse_one_sided, expected_reverse, atol=1e-12)
        expected_Z = (expected_forward + expected_reverse.conj().T) / 2
        np.testing.assert_allclose(core.Z, expected_Z, atol=1e-12)
        self.assertFalse(core.same_sector)
        self.assertEqual(core.Z.shape, (sector_a.selected_rank, sector_b.selected_rank))

    def test_one_sided_mode_is_unaveraged(self):
        grid, plan, sector_a, sector_b = self._setup(np.float64)
        core = ibp_core(sector_a, sector_b, operator=plan, symmetry_mode="one_sided")
        np.testing.assert_array_equal(core.Z, core.z_forward_one_sided)
        # The raw residual is still computed and retained even in diagnostic mode --
        # quadrature bias is never hidden regardless of which Z ships as production.
        self.assertGreater(core.raw_dagger_residual, 0.0)

    def test_rejects_unsupported_symmetry_mode(self):
        _, plan, sector_a, _ = self._setup()
        with self.assertRaises(ValueError):
            ibp_core(sector_a, operator=plan, symmetry_mode="nonsense")

    def test_rejects_non_sector_left_and_non_plan_operator(self):
        _, plan, sector_a, _ = self._setup()
        with self.assertRaises(TypeError):
            ibp_core("not-a-sector", operator=plan)
        with self.assertRaises(TypeError):
            ibp_core(sector_a, operator="not-a-plan")
        with self.assertRaises(TypeError):
            ibp_core(sector_a, "not-a-sector", operator=plan)

    def test_rejects_sector_operator_grid_mismatch(self):
        _, plan, sector_a, _ = self._setup()
        other_rng = np.random.default_rng(999)
        other_grid = build_ibp_grid(
            other_rng.normal(size=(14, 3)), other_rng.random(14)
        )
        other_plan = build_ibp_operator_plan(other_grid)
        with self.assertRaises(ValueError):
            ibp_core(sector_a, operator=other_plan)

    def test_mu_nu_block_size_does_not_change_result(self):
        _, plan, sector_a, sector_b = self._setup(np.complex128)
        full = ibp_core(sector_a, sector_b, operator=plan)
        blocked = ibp_core(sector_a, sector_b, operator=plan, mu_block_size=1, nu_block_size=2)
        np.testing.assert_allclose(blocked.Z, full.Z, atol=1e-12)
        np.testing.assert_allclose(
            blocked.z_forward_one_sided, full.z_forward_one_sided, atol=1e-12
        )
        self.assertEqual(blocked.mu_block_size, 1)
        self.assertEqual(blocked.nu_block_size, 2)
        self.assertEqual(full.mu_block_size, sector_a.selected_rank)
        self.assertEqual(full.nu_block_size, sector_b.selected_rank)

    def test_mixed_backend_operator_and_sector_rejected(self):
        # A core cannot MIX backends: a JAX operator with a NumPy sector (or
        # vice versa) is a structural error and must be rejected.
        rng = np.random.default_rng(41)
        coords = jnp.asarray(rng.normal(size=(6, 3)))
        weights = jnp.asarray(rng.random(6))
        grid = build_ibp_grid(coords, weights, backend="jax")
        plan = build_ibp_operator_plan(grid)
        _, _, sector_a, _ = self._setup()   # numpy sector
        with self.assertRaises(ValueError):
            ibp_core(sector_a, operator=plan)

    def _hermitian_psd(self, n, seed=3):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(n, n))
        z = a @ a.conj().T + 0.5 * np.eye(n)  # Hermitian, PSD, full rank
        return (z + z.conj().T) / 2

    def test_same_sector_two_sided_psd_factorization_is_wired(self):
        # Inject a known Hermitian PSD Z so the ibp_core two-sided path is
        # exercised end to end without a full physical grid. Non-circular:
        # psd_factorize has its own eigensystem tests and the raw-kernel path
        # is checked in one-sided mode.
        _, plan, sector_a, _ = self._setup()
        n = sector_a.selected_rank
        z_psd = self._hermitian_psd(n)
        with mock.patch("pytc.df.ibp._ibp_one_sided_block", return_value=(z_psd, 2)):
            core = ibp_core(sector_a, operator=plan)  # two_sided_average default
        self.assertEqual(core.psd_status, "factorized")
        self.assertEqual(core.psd_rtol, 1e-10)
        self.assertEqual(core.psd_negative_mode_count, 0)
        self.assertEqual(core.psd_retained_rank, n)
        self.assertEqual(core.psd_factor.shape, (n, n))
        self.assertFalse(core.psd_factor.flags.writeable)
        recon = core.psd_factor @ core.psd_factor.conj().T
        np.testing.assert_allclose(recon, core.Z, atol=1e-10)
        self.assertLess(core.psd_reconstruction_residual, 1e-10)
        self.assertIsNotNone(core.raw_packed_pair_metric_dagger_residual)
        self.assertGreaterEqual(core.raw_packed_pair_metric_dagger_residual, 0.0)

    def test_same_sector_material_indefinite_hard_fails_in_builder(self):
        _, plan, sector_a, _ = self._setup()
        n = sector_a.selected_rank
        rng = np.random.default_rng(5)
        u, _ = np.linalg.qr(rng.normal(size=(n, n)))
        eigs = np.linspace(1.0, 0.2, n)
        eigs[-1] = -1e-3  # far outside the roundoff band
        z_bad = (u * eigs) @ u.conj().T
        z_bad = (z_bad + z_bad.conj().T) / 2
        with mock.patch("pytc.df.ibp._ibp_one_sided_block", return_value=(z_bad, 0)):
            with self.assertRaisesRegex(ValueError, "material negative"):
                ibp_core(sector_a, operator=plan)

    def test_cross_and_one_sided_have_psd_not_applicable(self):
        _, plan, sector_a, sector_b = self._setup()
        cross = ibp_core(sector_a, sector_b, operator=plan)
        self.assertEqual(cross.psd_status, "not_applicable")
        self.assertIsNone(cross.psd_factor)
        self.assertIsNone(cross.psd_rtol)
        self.assertIsNone(cross.raw_packed_pair_metric_dagger_residual)
        one_sided = ibp_core(sector_a, operator=plan, symmetry_mode="one_sided")
        self.assertEqual(one_sided.psd_status, "not_applicable")
        self.assertIsNone(one_sided.psd_factor)

    def test_zero_forward_nonzero_reverse_residual_is_rejected(self):
        _, plan, sector_a, sector_b = self._setup()
        n_mu = sector_a.selected_rank
        n_nu = sector_b.selected_rank
        z_zero = np.zeros((n_mu, n_nu))
        z_nonzero = np.ones((n_nu, n_mu))
        with mock.patch("pytc.df.ibp._ibp_one_sided_block",
                        side_effect=[(z_zero, 0), (z_nonzero, 0)]):
            with self.assertRaises(ValueError):
                ibp_core(sector_a, sector_b, operator=plan)

    def test_overlarge_block_size_is_clamped_to_rank(self):
        _, plan, sector_a, _ = self._setup()
        n = sector_a.selected_rank
        z_psd = self._hermitian_psd(n)
        with mock.patch("pytc.df.ibp._ibp_one_sided_block", return_value=(z_psd, 0)):
            core = ibp_core(sector_a, operator=plan, mu_block_size=999, nu_block_size=999)
        self.assertEqual(core.mu_block_size, n)
        self.assertEqual(core.nu_block_size, n)

    def test_nonfinite_core_is_rejected(self):
        _, plan, sector_a, _ = self._setup()
        n = sector_a.selected_rank
        z_bad = np.array(self._hermitian_psd(n))
        z_bad[0, 0] = np.inf
        with mock.patch("pytc.df.ibp._ibp_one_sided_block", return_value=(z_bad, 0)):
            with self.assertRaises(ValueError):
                ibp_core(sector_a, operator=plan, symmetry_mode="one_sided")


if __name__ == "__main__":
    unittest.main()
