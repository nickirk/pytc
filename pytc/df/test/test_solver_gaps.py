"""Regression tests for Alice's 4 solver-implementation gaps found on
commit b59c6ce (task #8 commit 3, isdf-coulomb-cuda, 2026-07-12):
1) jitter floor not scale-relative, 2) adaptive acceptance rule not
implemented (regularized backward-error gate + unregularized-bias ->
TSVD fallback), 3) O(n^3) residual not production-scalable (sampled
mode), 4) O(n^2) same-sector detection via dense array_equal.
"""
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pytc.df.solvers import prepare_spd_cholesky
from pytc.df.fit import compute_Z, compute_Z_cross


class TestJitterFloorScaleRelative(unittest.TestCase):
    """Gap 1: eps_scale must be PURELY matrix-relative -- no absolute
    eps*max(diag_mean, 1.0) floor, which previously injected an
    absolute ~2.22e-16 jitter regardless of the matrix's own scale
    (2.22x the matrix itself for a diag_mean=1e-16 system)."""

    def test_jitter_floor_is_matrix_relative_not_absolute(self):
        mat = jnp.array([[1e-16]], dtype=jnp.float64)
        chol, lower, jitter_used, n_tries = prepare_spd_cholesky(mat)
        self.assertEqual(n_tries, 1)
        self.assertLess(jitter_used, 1e-16)

    def test_jitter_scales_with_matrix_diagonal(self):
        small = jnp.array([[1e-10]], dtype=jnp.float64)
        large = jnp.array([[1e10]], dtype=jnp.float64)
        _, _, jitter_small, _ = prepare_spd_cholesky(small)
        _, _, jitter_large, _ = prepare_spd_cholesky(large)
        # Both should stay proportional to their own matrix's scale --
        # the ratio of jitters should track the ratio of diag values,
        # not collapse to a shared absolute floor.
        self.assertLess(jitter_small, 1e-10)
        self.assertLess(jitter_large, 1e10)
        self.assertGreater(jitter_large / jitter_small, 1e10)

    def test_nonpositive_diag_mean_raises_linalg_error(self):
        mat = jnp.array([[-1.0]], dtype=jnp.float64)
        with self.assertRaises(np.linalg.LinAlgError):
            prepare_spd_cholesky(mat)

    def test_nonfinite_diag_mean_raises_value_error(self):
        mat = jnp.array([[float("nan")]], dtype=jnp.float64)
        with self.assertRaises(ValueError):
            prepare_spd_cholesky(mat)


class TestUnregularizedBiasFallback(unittest.TestCase):
    """Gap 2: an unregularized-bias fit residual over threshold must
    trigger an automatic solver='tsvd' fallback (never more jitter,
    which only increases the bias) -- previously this was warn-only
    with no corrective action."""

    def setUp(self):
        rng = np.random.default_rng(0)
        # P is (n_pivots, n_pair); n_pair > n_pivots so S = P P^T is
        # generically full rank (well-conditioned baseline, matching
        # the design doc's "pivot budgets are analytically pre-capped"
        # assumption -- production P always has n_pivots < n_pair).
        self.P = rng.standard_normal((6, 8))
        self.C = rng.standard_normal((6, 5))

    def test_large_rcond_forces_tsvd_fallback(self):
        # rcond=1.0 -> base jitter ~= the matrix's own diagonal scale,
        # enough regularization bias to blow well past the 1e-10
        # acceptance threshold regardless of the random data drawn.
        Z, prov = compute_Z(self.P, self.C, rcond=1.0, solver="cholesky_jitter")
        self.assertTrue(prov["fallback_triggered"])
        self.assertEqual(prov["solver"], "tsvd")
        self.assertIn("fallback_reason", prov)
        self.assertGreater(prov["preceding_cholesky_fit_residual"], 1e-10)

    def test_well_conditioned_default_does_not_fall_back(self):
        Z, prov = compute_Z(self.P, self.C, solver="cholesky_jitter")
        self.assertFalse(prov["fallback_triggered"])
        self.assertEqual(prov["solver"], "unscaled_cholesky_jitter")


class TestResidualModes(unittest.TestCase):
    """Gap 3: residual_mode='sampled' must be an available, provenance-
    documented O(n^2 * n_probes) alternative to the exact O(n^3)
    two-sided residual, without changing the returned Z."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.P = rng.standard_normal((7, 10))
        self.C = rng.standard_normal((7, 4))

    def test_sampled_mode_provenance_fields(self):
        Z_exact, prov_exact = compute_Z(self.P, self.C, residual_mode="exact")
        Z_sampled, prov_sampled = compute_Z(
            self.P, self.C, residual_mode="sampled",
            residual_n_probes=64, residual_seed=42)
        self.assertEqual(prov_exact["residual_mode"], "exact")
        self.assertEqual(prov_sampled["residual_mode"], "sampled")
        self.assertEqual(prov_sampled["residual_n_probes"], 64)
        self.assertEqual(prov_sampled["residual_seed"], 42)
        # residual_mode only changes how the diagnostic is measured, not
        # the solve itself.
        np.testing.assert_allclose(Z_sampled, Z_exact, atol=1e-10)
        self.assertGreaterEqual(prov_sampled["fit_residual"], 0.0)
        self.assertLess(prov_sampled["fit_residual"], 1e-3)

    def test_unknown_residual_mode_rejected(self):
        with self.assertRaises(ValueError):
            compute_Z(self.P, self.C, residual_mode="not-a-real-mode")


class TestExplicitSameSector(unittest.TestCase):
    """Gap 4: same_sector must be an explicit caller-supplied fact, not
    inferred via dense jnp.array_equal(S_A, S_B) -- compute_Z_cross's
    default (same_sector=None) falls back to a cheap `P_A is P_B`
    identity check, which does NOT catch value-equal-but-distinct array
    objects (unlike the old array_equal inference)."""

    def setUp(self):
        rng = np.random.default_rng(2)
        self.P = rng.standard_normal((5, 6))
        self.C = rng.standard_normal((5, 4))

    def test_explicit_same_sector_true_matches_compute_Z(self):
        Z_direct, prov_direct = compute_Z(self.P, self.C)
        Z_cross, prov_cross = compute_Z_cross(
            self.P, self.C, self.P, self.C, same_sector=True)
        np.testing.assert_allclose(Z_cross, Z_direct, atol=1e-12)
        self.assertEqual(prov_cross["jitter_used"], prov_direct["jitter_used"])

    def test_identity_default_still_matches_object_identity(self):
        Z_direct, _ = compute_Z(self.P, self.C)
        Z_cross, _ = compute_Z_cross(self.P, self.C, self.P, self.C)
        np.testing.assert_allclose(Z_cross, Z_direct, atol=1e-12)

    def test_value_equal_distinct_object_takes_cross_sector_path(self):
        P_copy = self.P.copy()
        Z_direct, _ = compute_Z(self.P, self.C)
        # Default same_sector=None -> `P_A is P_B` is False here (distinct
        # object), so this takes the independent-factorization cross-
        # sector path even though the values are numerically identical --
        # result must still agree numerically, but via a different code
        # path than the same_sector=True fast path.
        Z_cross, _ = compute_Z_cross(self.P, self.C, P_copy, self.C)
        np.testing.assert_allclose(Z_cross, Z_direct, atol=1e-10)

    def test_explicit_same_sector_false_forces_independent_solve(self):
        Z_direct, _ = compute_Z(self.P, self.C)
        Z_cross, _ = compute_Z_cross(
            self.P, self.C, self.P, self.C, same_sector=False)
        np.testing.assert_allclose(Z_cross, Z_direct, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
