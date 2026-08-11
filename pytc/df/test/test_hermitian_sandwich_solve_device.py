"""Device (JAX, jitted) parity tests for
pytc.df.solvers.hermitian_sandwich_solve_device against the NumPy oracle
hermitian_sandwich_solve (design v2.1 sections 5+6/7, V3's "JAX device
path vs NumPy oracle bit-tier <=1e-12" sub-gate)."""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from pytc.df.solvers import hermitian_sandwich_solve, hermitian_sandwich_solve_device


def _random_hermitian_psd(rng, n, rank=None, dtype=np.complex128):
    rank = n if rank is None else rank
    A = (rng.normal(size=(n, rank)) + 1j * rng.normal(size=(n, rank))).astype(dtype)
    return A @ A.conj().T


class TestHermitianSandwichSolveDevice(unittest.TestCase):
    def test_full_rank_matches_numpy_oracle(self):
        rng = np.random.default_rng(60)
        n = 6
        Pi = _random_hermitian_psd(rng, n) + np.eye(n) * 0.5
        V = _random_hermitian_psd(rng, n)
        W_np, info_np = hermitian_sandwich_solve(Pi, V, rtol=1e-10)
        W_jax, info_jax = hermitian_sandwich_solve_device(Pi, V, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(W_jax), W_np, atol=1e-12)
        self.assertEqual(info_jax["n_retained"], info_np["n_retained"])
        self.assertEqual(info_jax["backend"], "jax")
        self.assertLess(abs(info_jax["retained_solve_residual"] - info_np["retained_solve_residual"]), 1e-10)
        self.assertLess(abs(info_jax["truncation_residual"] - info_np["truncation_residual"]), 1e-10)

    def test_rank_deficient_matches_numpy_oracle(self):
        rng = np.random.default_rng(61)
        n, rank = 8, 3
        Pi = _random_hermitian_psd(rng, n, rank=rank)
        V = _random_hermitian_psd(rng, n)
        W_np, info_np = hermitian_sandwich_solve(Pi, V, rtol=1e-8)
        W_jax, info_jax = hermitian_sandwich_solve_device(Pi, V, rtol=1e-8)
        np.testing.assert_allclose(np.asarray(W_jax), W_np, atol=1e-9)
        self.assertEqual(info_jax["n_retained"], rank)
        self.assertEqual(info_jax["n_retained"], info_np["n_retained"])
        self.assertEqual(info_jax["s_min_retained"], None if rank == 0 else info_jax["s_min_retained"])

    def test_w_is_hermitian(self):
        rng = np.random.default_rng(62)
        n = 5
        Pi = _random_hermitian_psd(rng, n) + np.eye(n) * 0.1
        V = _random_hermitian_psd(rng, n)
        W, _ = hermitian_sandwich_solve_device(Pi, V, rtol=1e-8)
        np.testing.assert_allclose(np.asarray(W), np.asarray(W).conj().T, atol=1e-12)

    def test_zero_pi_degrades_to_zero_retained_not_a_raise(self):
        # Deliberate device-path simplification (see docstring): unlike
        # the NumPy oracle's explicit "Pi not PSD" ValueError, a
        # non-positive Pi degrades to n_retained=0 / W=0 on the device
        # path, since raising from inside a jax.jit graph on a traced
        # value is not available the way host-side code can do it.
        n = 3
        Pi = -np.eye(n, dtype=np.complex128)
        V = np.eye(n, dtype=np.complex128)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve(Pi, V, rtol=1e-8)
        W, info = hermitian_sandwich_solve_device(Pi, V, rtol=1e-8)
        self.assertEqual(info["n_retained"], 0)
        self.assertIsNone(info["s_min_retained"])
        np.testing.assert_allclose(np.asarray(W), np.zeros((n, n)), atol=1e-14)

    def test_adaptive_retention_not_supported_on_device(self):
        rng = np.random.default_rng(63)
        Pi = _random_hermitian_psd(rng, 4, rank=2) + np.eye(4) * 1e-3
        V = _random_hermitian_psd(rng, 4)
        _, info = hermitian_sandwich_solve_device(Pi, V, rtol=1e-8)
        self.assertFalse(info["adaptive_retention_used"])
        self.assertIsNone(info["target_truncation_residual"])

    def test_rejects_malformed_shapes(self):
        rng = np.random.default_rng(64)
        Pi = _random_hermitian_psd(rng, 4)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve_device(Pi[:, :3], Pi, rtol=1e-8)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve_device(Pi, Pi[:3, :3], rtol=1e-8)

    def test_rejects_bad_rtol(self):
        rng = np.random.default_rng(65)
        Pi = _random_hermitian_psd(rng, 4) + np.eye(4)
        V = _random_hermitian_psd(rng, 4)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve_device(Pi, V, rtol=0.0)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve_device(Pi, V, rtol=-1e-8)

    def test_default_rtol_is_1e_4(self):
        rng = np.random.default_rng(66)
        Pi = _random_hermitian_psd(rng, 4) + np.eye(4)
        V = _random_hermitian_psd(rng, 4)
        _, info = hermitian_sandwich_solve_device(Pi, V)
        self.assertEqual(info["rtol"], 1e-4)

    def test_retention_marginal_matches_numpy_oracle(self):
        rng = np.random.default_rng(67)
        n = 6

        def _pi_with_spectrum(eigvals):
            A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            Q, _ = np.linalg.qr(A)
            return Q @ np.diag(eigvals) @ Q.conj().T

        V = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        V = V @ V.conj().T

        eigvals_marginal = np.array([1.0, 0.8, 0.6, 0.4, 5e-4, 1e-5])
        Pi_marginal = _pi_with_spectrum(eigvals_marginal)
        _, info_np = hermitian_sandwich_solve(Pi_marginal, V, rtol=1e-4)
        _, info_jax = hermitian_sandwich_solve_device(Pi_marginal, V, rtol=1e-4)
        self.assertTrue(info_np["retention_marginal"])
        self.assertTrue(info_jax["retention_marginal"])
        self.assertAlmostEqual(
            info_jax["cond_pi_retained"], info_np["cond_pi_retained"], places=6
        )

        eigvals_healthy = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2])
        Pi_healthy = _pi_with_spectrum(eigvals_healthy)
        _, info_healthy = hermitian_sandwich_solve_device(Pi_healthy, V, rtol=1e-4)
        self.assertFalse(info_healthy["retention_marginal"])


if __name__ == "__main__":
    unittest.main()


class TestDeviceCholeskyJitterFailsClosed(unittest.TestCase):
    """Regression: the device path ACCEPTED retention_mode='cholesky_jitter',
    fell through to the eig branch, and returned info labelled Cholesky -- a
    false claim inside a data structure, and worse than a missing feature
    because it is invisible to a caller who trusts the label."""

    def test_device_refuses_rather_than_mislabelling(self):
        pi = np.eye(8, dtype=np.complex128)
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve_device(pi, pi.copy(), rtol=1e-6,
                                            retention_mode="cholesky_jitter")
