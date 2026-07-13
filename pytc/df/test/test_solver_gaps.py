"""Regression tests for Alice's 4 solver-implementation gaps, TWO
review rounds (task #8 commit 3, isdf-coulomb-cuda, 2026-07-12):
1) jitter floor not scale-relative, 2) adaptive acceptance rule not
implemented (regularized backward-error gate + unregularized-bias ->
TSVD fallback), 3) O(n^3) residual not production-scalable (sampled
mode), 4) O(n^2) same-sector detection via dense array_equal.

Round-2 findings on the first fix (commit 7064a7b), all covered here:
1b) the automatic TSVD fallback reused the Cholesky jitter_rcond as the
    TSVD singular-value cutoff -- a real correctness bug (a caller-
    forced jitter_rcond=1.0 made TSVD retain ZERO modes, Z=0, residual
    1.0 -- the fallback made things WORSE, not better).
2b) the backward-error gate's dense LL^dagger reconstruction ran
    unconditionally, recreating gap 3's O(n^3)-per-retry cost problem
    at the generic solver layer.
3b) the sampled residual called np.asarray on the full dense matrices
    (multi-GB device->host transfer at production N_mu) and could
    auto-trigger the hard fallback off an uncalibrated estimate.
4b) same-sector identity inference ran AFTER np.asarray conversion,
    silently broken for JAX/CuPy inputs (device arrays get a NEW host
    object on every np.asarray call).
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


class TestBackwardErrorMode(unittest.TestCase):
    """Gap 2 (regularized half): backward_error_mode must default to
    "finite_only" (O(1), production-safe) and only do the dense O(n^3)
    LL^dagger reconstruction when "exact" is explicitly requested --
    an earlier version ran the exact check unconditionally, recreating
    gap 3's cost problem at the generic solver layer."""

    def setUp(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal((6, 6))
        self.mat = jnp.array(a @ a.T + 1e-3 * np.eye(6))

    def test_finite_only_is_default_and_succeeds(self):
        chol, lower, jitter_used, n_tries = prepare_spd_cholesky(self.mat)
        self.assertTrue(np.all(np.isfinite(np.asarray(chol))))

    def test_exact_mode_also_succeeds_on_well_conditioned_matrix(self):
        chol, lower, jitter_used, n_tries = prepare_spd_cholesky(
            self.mat, backward_error_mode="exact")
        self.assertTrue(np.all(np.isfinite(np.asarray(chol))))

    def test_unknown_backward_error_mode_rejected(self):
        with self.assertRaises(ValueError):
            prepare_spd_cholesky(self.mat, backward_error_mode="not-a-real-mode")


class TestUnregularizedBiasFallback(unittest.TestCase):
    """Gap 2 (unregularized half): an unregularized-bias fit residual
    over threshold must trigger an automatic solver='tsvd' fallback
    (never more jitter, which only increases the bias) -- previously
    this was warn-only with no corrective action. Round-2 fix: the
    fallback must use its OWN independent tsvd_rcond, not the Cholesky
    jitter_rcond -- reusing it made TSVD retain zero modes (Z=0) for a
    caller-forced jitter_rcond=1.0, i.e. the fallback made the result
    WORSE than the failing Cholesky attempt, not better."""

    def setUp(self):
        rng = np.random.default_rng(0)
        # P is (n_pivots, n_pair); n_pair > n_pivots so S = P P^T is
        # generically full rank (well-conditioned baseline, matching
        # the design doc's "pivot budgets are analytically pre-capped"
        # assumption -- production P always has n_pivots < n_pair).
        self.P = rng.standard_normal((6, 8))
        self.C = rng.standard_normal((6, 5))

    def test_large_rcond_forces_tsvd_fallback_that_actually_helps(self):
        # jitter_rcond=1.0 -> base jitter ~= the matrix's own diagonal
        # scale, enough regularization bias to blow well past the
        # 1e-10 acceptance threshold regardless of the random data.
        Z, prov = compute_Z(self.P, self.C, rcond=1.0, solver="cholesky_jitter")
        self.assertTrue(prov["fallback_triggered"])
        self.assertEqual(prov["solver"], "tsvd")
        self.assertIn("fallback_reason", prov)
        preceding = prov["preceding_cholesky_fit_residual"]
        self.assertGreater(preceding, 1e-10)
        # The fallback must actually IMPROVE on the failing Cholesky
        # attempt, not silently return a useless Z=0 by inheriting
        # jitter_rcond=1.0 as a TSVD singular-value cutoff (round-2 bug:
        # cutoff = 1.0 * s_max means NO singular value survives
        # `s > cutoff`, n_retained=0, Z=0, residual=1.0).
        self.assertLess(prov["fit_residual"], preceding)
        self.assertLess(prov["fit_residual"], 1e-6)
        n_retained_A, n_retained_B = prov["n_retained"]
        self.assertGreater(n_retained_A, 0)
        self.assertGreater(n_retained_B, 0)

    def test_fallback_uses_independent_tsvd_rcond_not_jitter_rcond(self):
        # Explicit tsvd_rcond must be honored on fallback, independent
        # of whatever jitter_rcond triggered it.
        Z, prov = compute_Z(
            self.P, self.C, rcond=1.0, solver="cholesky_jitter", tsvd_rcond=1e-10)
        self.assertTrue(prov["fallback_triggered"])
        self.assertEqual(prov["cutoff"], 1e-10)
        n_retained_A, _ = prov["n_retained"]
        self.assertGreater(n_retained_A, 0)

    def test_well_conditioned_default_does_not_fall_back(self):
        Z, prov = compute_Z(self.P, self.C, solver="cholesky_jitter")
        self.assertFalse(prov["fallback_triggered"])
        self.assertEqual(prov["solver"], "unscaled_cholesky_jitter")
        self.assertEqual(prov["backward_error_mode"], "finite_only")
        self.assertIsNone(prov["backward_error_tol"])


class TestResidualModes(unittest.TestCase):
    """Gap 3: residual_mode='sampled' must be an available, provenance-
    documented O(n^2 * n_probes) on-device alternative to the exact
    O(n^3) two-sided residual, without changing the returned Z, and
    must NEVER drive the automatic TSVD fallback (diagnostic-only,
    uncalibrated) -- round-2 fix: only residual_mode='exact' gates the
    fallback."""

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
        self.assertIn("residual_relative_standard_error", prov_sampled)
        # residual_mode only changes how the diagnostic is measured, not
        # the solve itself.
        np.testing.assert_allclose(Z_sampled, Z_exact, atol=1e-10)
        self.assertGreaterEqual(prov_sampled["fit_residual"], 0.0)
        self.assertLess(prov_sampled["fit_residual"], 1e-3)

    def test_sampled_mode_never_triggers_fallback_even_when_elevated(self):
        # jitter_rcond=1.0 -> the TRUE unregularized bias is huge, so
        # the sampled estimate should register a large value too (same
        # underlying quantity, just noisier) -- but residual_mode=
        # 'sampled' must never auto-trigger the TSVD fallback, unlike
        # 'exact' on the identical inputs.
        Z_exact, prov_exact = compute_Z(
            self.P, self.C, rcond=1.0, residual_mode="exact")
        Z_sampled, prov_sampled = compute_Z(
            self.P, self.C, rcond=1.0, residual_mode="sampled")
        self.assertTrue(prov_exact["fallback_triggered"])
        self.assertFalse(prov_sampled["fallback_triggered"])
        self.assertEqual(prov_sampled["solver"], "unscaled_cholesky_jitter")
        self.assertFalse(prov_sampled["fallback_gating_applicable"])
        self.assertGreater(prov_sampled["fit_residual"], 1e-10)

    def test_unknown_residual_mode_rejected(self):
        with self.assertRaises(ValueError):
            compute_Z(self.P, self.C, residual_mode="not-a-real-mode")


class TestExplicitSameSector(unittest.TestCase):
    """Gap 4: same_sector must be an explicit caller-supplied fact, not
    inferred via any array comparison. Round-2 fix: compute_Z_cross now
    defaults to same_sector=False (always-correct, conservative)
    instead of a `P_A is P_B` identity check performed AFTER
    np.asarray conversion, which silently broke for JAX/CuPy inputs
    (device arrays get a NEW host object on every np.asarray call, so
    two calls on the SAME underlying device array never satisfied
    `is` post-conversion)."""

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

    def test_explicit_same_sector_true_works_for_jax_array_inputs(self):
        # The whole point of dropping identity-after-conversion
        # inference: correctness must not depend on the input array
        # backend. Pass the SAME jnp array object for both sides.
        P_jax = jnp.asarray(self.P)
        C_jax = jnp.asarray(self.C)
        Z_direct, _ = compute_Z(self.P, self.C)
        Z_cross, prov_cross = compute_Z_cross(
            P_jax, C_jax, P_jax, C_jax, same_sector=True)
        np.testing.assert_allclose(np.asarray(Z_cross), Z_direct, atol=1e-10)
        self.assertEqual(prov_cross["n_tries"][0], prov_cross["n_tries"][1])

    def test_default_is_conservative_cross_sector_path(self):
        # No same_sector passed -> default False -> independent
        # factorization path, even for the literal same array object
        # passed twice. Result must still agree numerically.
        Z_direct, _ = compute_Z(self.P, self.C)
        Z_cross, _ = compute_Z_cross(self.P, self.C, self.P, self.C)
        np.testing.assert_allclose(Z_cross, Z_direct, atol=1e-10)

    def test_value_equal_distinct_object_also_takes_cross_sector_path(self):
        P_copy = self.P.copy()
        Z_direct, _ = compute_Z(self.P, self.C)
        Z_cross, _ = compute_Z_cross(self.P, self.C, P_copy, self.C)
        np.testing.assert_allclose(Z_cross, Z_direct, atol=1e-10)

    def test_explicit_same_sector_false_forces_independent_solve(self):
        Z_direct, _ = compute_Z(self.P, self.C)
        Z_cross, _ = compute_Z_cross(
            self.P, self.C, self.P, self.C, same_sector=False)
        np.testing.assert_allclose(Z_cross, Z_direct, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
