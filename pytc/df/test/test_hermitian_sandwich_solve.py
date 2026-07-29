"""V1 oracle tests for pytc.df.solvers.hermitian_sandwich_solve
(task #21, #proj-isdf-periodic, design v2.1 section 5/8)."""

import unittest

import numpy as np

from pytc.df.solvers import hermitian_sandwich_solve, hermitian_sandwich_solve_device


def _random_hermitian_psd(rng, n, rank=None, dtype=np.complex128):
    rank = n if rank is None else rank
    A = (rng.normal(size=(n, rank)) + 1j * rng.normal(size=(n, rank))).astype(dtype)
    return A @ A.conj().T


class TestHermitianSandwichSolve(unittest.TestCase):
    def test_full_rank_solves_to_machine_precision(self):
        rng = np.random.default_rng(40)
        n = 6
        Pi = _random_hermitian_psd(rng, n) + np.eye(n) * 0.5
        V = _random_hermitian_psd(rng, n)
        W, info = hermitian_sandwich_solve(Pi, V, rtol=1e-10)
        self.assertEqual(info["n_retained"], n)
        self.assertEqual(info["n_discarded"], 0)
        self.assertLess(info["retained_solve_residual"], 1e-10)
        self.assertLess(info["truncation_residual"], 1e-10)
        recon_err = np.linalg.norm(Pi @ W @ Pi - V) / np.linalg.norm(V)
        self.assertLess(recon_err, 1e-10)

    def test_w_is_hermitian(self):
        rng = np.random.default_rng(41)
        n = 5
        Pi = _random_hermitian_psd(rng, n) + np.eye(n) * 0.1
        V = _random_hermitian_psd(rng, n)
        W, _ = hermitian_sandwich_solve(Pi, V, rtol=1e-8)
        np.testing.assert_allclose(W, W.conj().T, atol=1e-12)

    def test_rank_deficient_pi_retained_residual_stays_tiny_regardless_of_truncation(self):
        rng = np.random.default_rng(42)
        n, rank = 8, 3
        Pi = _random_hermitian_psd(rng, n, rank=rank)
        V = _random_hermitian_psd(rng, n)  # generic V, not aligned with Pi's row space
        W, info = hermitian_sandwich_solve(Pi, V, rtol=1e-8)
        self.assertEqual(info["n_retained"], rank)
        # Retained-space residual is a numerical sanity check, essentially
        # zero BY CONSTRUCTION regardless of how much of V is discarded.
        self.assertLess(info["retained_solve_residual"], 1e-9)
        # Truncation residual should be substantial here -- V has real
        # support outside Pi's 3-dimensional retained subspace.
        self.assertGreater(info["truncation_residual"], 0.1)

    def test_split_residual_identity_matches_direct_projector_computation(self):
        # Independently recompute both residuals via a completely separate
        # code path (direct eigh + explicit projector construction) and
        # compare against the function's own reported values.
        rng = np.random.default_rng(43)
        n, rank = 7, 4
        Pi = _random_hermitian_psd(rng, n, rank=rank)
        V = _random_hermitian_psd(rng, n)
        W, info = hermitian_sandwich_solve(Pi, V, rtol=1e-8)

        eigvals, eigvecs = np.linalg.eigh(Pi)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        s_max = eigvals[0]
        retained = eigvals > 1e-8 * s_max
        U_r = eigvecs[:, retained]
        proj = U_r @ U_r.conj().T

        expected_retained_residual = np.linalg.norm(
            proj @ (Pi @ W @ Pi - V) @ proj
        ) / np.linalg.norm(proj @ V @ proj)
        expected_truncation_residual = np.linalg.norm(
            V - proj @ V @ proj
        ) / np.linalg.norm(V)

        self.assertAlmostEqual(
            info["retained_solve_residual"], expected_retained_residual, places=8
        )
        self.assertAlmostEqual(
            info["truncation_residual"], expected_truncation_residual, places=8
        )

    def test_adaptive_retention_expands_until_target_met(self):
        rng = np.random.default_rng(44)
        n, rank = 6, 2
        Pi = _random_hermitian_psd(rng, n, rank=rank)
        V = _random_hermitian_psd(rng, n)
        W_base, info_base = hermitian_sandwich_solve(Pi, V, rtol=1e-8)
        self.assertFalse(info_base["adaptive_retention_used"])

        W_adaptive, info_adaptive = hermitian_sandwich_solve(
            Pi, V, rtol=1e-8, target_truncation_residual=1e-2
        )
        self.assertTrue(info_adaptive["adaptive_retention_used"])
        self.assertGreaterEqual(info_adaptive["n_retained"], info_base["n_retained"])
        self.assertLessEqual(info_adaptive["truncation_residual"], 1e-2 + 1e-9)

    def test_anti_hermitian_input_residuals_are_recorded_not_discarded(self):
        rng = np.random.default_rng(45)
        n = 5
        Pi_herm = _random_hermitian_psd(rng, n) + np.eye(n) * 0.5
        noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Pi_noisy = Pi_herm + 0.01 * (noise - noise.conj().T)  # add anti-Hermitian noise
        V = _random_hermitian_psd(rng, n)
        _, info = hermitian_sandwich_solve(Pi_noisy, V, rtol=1e-8)
        self.assertGreater(info["pi_anti_hermitian_residual"], 1e-6)

        V_hermitian = _random_hermitian_psd(rng, n)
        _, info2 = hermitian_sandwich_solve(Pi_herm, V_hermitian, rtol=1e-8)
        self.assertLess(info2["v_anti_hermitian_residual"], 1e-12)

    def test_scale_invariance_of_retention(self):
        rng = np.random.default_rng(46)
        n = 6
        Pi = _random_hermitian_psd(rng, n, rank=4)
        V = _random_hermitian_psd(rng, n)
        _, info1 = hermitian_sandwich_solve(Pi, V, rtol=1e-8)
        _, info2 = hermitian_sandwich_solve(Pi * 1e8, V, rtol=1e-8)
        self.assertEqual(info1["n_retained"], info2["n_retained"])

    def test_rejects_malformed_shapes(self):
        rng = np.random.default_rng(47)
        Pi = _random_hermitian_psd(rng, 4)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve(Pi[:, :3], Pi, rtol=1e-8)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve(Pi, Pi[:3, :3], rtol=1e-8)

    def test_rejects_bad_rtol_and_target(self):
        rng = np.random.default_rng(48)
        Pi = _random_hermitian_psd(rng, 4) + np.eye(4)
        V = _random_hermitian_psd(rng, 4)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve(Pi, V, rtol=0.0)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve(Pi, V, rtol=-1e-8)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve(Pi, V, target_truncation_residual=-0.1)

    def test_rejects_non_psd_pi(self):
        Pi = -np.eye(3, dtype=np.complex128)
        V = np.eye(3, dtype=np.complex128)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve(Pi, V, rtol=1e-8)

    def test_default_rtol_is_1e_4(self):
        # The retention-policy fix's headline change: the old 1e-8
        # default was 4-5 orders too loose and silently blew up on
        # over-complete rank (design v2.1 section 5). Verified by
        # constructing a Pi where a stale rtol=1e-8 default would
        # retain a near-singular mode the new rtol=1e-4 default drops.
        rng = np.random.default_rng(60)
        n = 6

        def _pi_with_spectrum(eigvals):
            A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            Q, _ = np.linalg.qr(A)
            return Q @ np.diag(eigvals) @ Q.conj().T

        eigvals = np.array([1.0, 0.8, 0.6, 0.4, 1e-6, 1e-9])
        Pi = _pi_with_spectrum(eigvals)
        V = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        V = V @ V.conj().T
        _, info_default = hermitian_sandwich_solve(Pi, V)
        _, info_explicit_old = hermitian_sandwich_solve(Pi, V, rtol=1e-8)
        self.assertEqual(info_default["rtol"], 1e-4)
        self.assertEqual(info_default["n_retained"], 4)  # drops both 1e-6 and 1e-9 modes
        self.assertEqual(info_explicit_old["n_retained"], 5)  # old default only drops 1e-9

    def test_retention_marginal_flags_near_cutoff_high_condition_case(self):
        rng = np.random.default_rng(61)
        n = 6

        def _pi_with_spectrum(eigvals):
            A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            Q, _ = np.linalg.qr(A)
            return Q @ np.diag(eigvals) @ Q.conj().T

        V = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        V = V @ V.conj().T

        # Marginal: smallest retained eigenvalue (5e-4) is within 10x of
        # the rtol=1e-4 cutoff (1e-4) and cond=2000 > 1e3.
        eigvals_marginal = np.array([1.0, 0.8, 0.6, 0.4, 5e-4, 1e-5])
        Pi_marginal = _pi_with_spectrum(eigvals_marginal)
        _, info_marginal = hermitian_sandwich_solve(Pi_marginal, V, rtol=1e-4)
        self.assertTrue(info_marginal["retention_marginal"])
        self.assertAlmostEqual(info_marginal["cond_pi_retained"], 2000.0, places=3)

        # Healthy: well-separated spectrum, low condition number.
        eigvals_healthy = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2])
        Pi_healthy = _pi_with_spectrum(eigvals_healthy)
        _, info_healthy = hermitian_sandwich_solve(Pi_healthy, V, rtol=1e-4)
        self.assertFalse(info_healthy["retention_marginal"])

    def test_retention_marginal_is_false_when_nothing_retained(self):
        rng = np.random.default_rng(62)
        n = 4
        Pi = np.eye(n, dtype=np.complex128) * 1e-20
        V = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        V = V @ V.conj().T
        # rtol threshold is relative to s_max, so a uniformly tiny Pi
        # still retains all modes (all eigvals == s_max) -- use a
        # decisively tighter rtol to force n_retained=0 instead.
        _, info = hermitian_sandwich_solve(Pi, V, rtol=2.0)
        self.assertEqual(info["n_retained"], 0)
        self.assertFalse(info["retention_marginal"])
        self.assertIsNone(info["cond_pi_retained"])


if __name__ == "__main__":
    unittest.main()


class TestCholeskyJitterFailsClosed(unittest.TestCase):
    """A second in-module implementation of the two-sided Cholesky sandwich
    produced a parameter conflation (the spectral rtol used as the jitter scale,
    ten orders off), an explicit inverse with no condition certificate, and -- on
    the device path -- eig executed while the returned info was labelled Cholesky.
    The original guard here refused the mode outright "until the shared helper is
    routed through". The host path now IS routed through _cholesky_jitter_sandwich,
    so the guard becomes: it must delegate rather than reimplement, and the device
    path (not yet routed) must still refuse. Deleting the class instead would drop
    the only check that the device path has not quietly grown a second copy."""

    def test_host_delegates_to_the_shared_helper(self):
        pi = np.eye(8, dtype=np.complex128)
        _, info = hermitian_sandwich_solve(pi, pi.copy(),
                                           retention_mode="cholesky_jitter")
        # The helper's own provenance label. A reimplementation would have to
        # forge this string to pass, which is the point.
        self.assertEqual(info["solver"], "unscaled_cholesky_jitter")
        self.assertIn("jitter_used", info)

    def test_rtol_is_still_rejected_by_name(self):
        # The conflation that motivated the original refusal: rtol must not reach
        # the jitter scale. It is now rejected explicitly rather than by refusing
        # the whole mode.
        pi = np.eye(8, dtype=np.complex128)
        with self.assertRaises(ValueError) as ctx:
            hermitian_sandwich_solve(pi, pi.copy(), rtol=1e-6,
                                     retention_mode="cholesky_jitter")
        self.assertIn("jitter_rcond", str(ctx.exception))

    def test_device_path_still_refuses(self):
        pi = np.eye(8, dtype=np.complex128)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve_device(pi, pi.copy(), rtol=1e-6,
                                            retention_mode="cholesky_jitter")

    def test_supported_modes_are_unaffected(self):
        pi = np.eye(8, dtype=np.complex128)
        for mode in ("single", "pairwise"):
            with self.subTest(mode=mode):
                _, info = hermitian_sandwich_solve(pi, pi.copy(), rtol=1e-6,
                                                   retention_mode=mode)
                self.assertEqual(info["retention_mode"], mode)
