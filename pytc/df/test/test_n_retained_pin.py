"""n_retained_pin (fixed effective rank) tests for
pytc.df.solvers.hermitian_sandwich_solve and
hermitian_sandwich_solve_device: pin-at-threshold-count equals the rtol
selection on both oracles, misuse combinations hard-raise, and the
retention_marginal band is the s_K/s_{K+1} spectral gap when pinned."""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from pytc.df.solvers import hermitian_sandwich_solve, hermitian_sandwich_solve_device


def _random_hermitian_psd(rng, n, rank=None, dtype=np.complex128):
    rank = n if rank is None else rank
    A = (rng.normal(size=(n, rank)) + 1j * rng.normal(size=(n, rank))).astype(dtype)
    return A @ A.conj().T


def _pi_with_spectrum(rng, eigvals):
    n = len(eigvals)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    return Q @ np.diag(np.asarray(eigvals, dtype=np.float64)) @ Q.conj().T


class TestPinEquivalence(unittest.TestCase):
    def test_pin_at_threshold_count_matches_rtol_numpy(self):
        rng = np.random.default_rng(90)
        n = 8
        Pi = _random_hermitian_psd(rng, n, rank=5) + np.eye(n) * 1e-3
        V = _random_hermitian_psd(rng, n)
        W_rtol, info_rtol = hermitian_sandwich_solve(Pi, V, rtol=1e-8)
        m = info_rtol["n_retained"]
        self.assertGreater(m, 0)
        W_pin, info_pin = hermitian_sandwich_solve(Pi, V, n_retained_pin=m)
        np.testing.assert_array_equal(W_pin, W_rtol)
        self.assertEqual(info_pin["n_retained"], m)
        self.assertEqual(info_pin["n_retained_pin"], m)
        self.assertIsNone(info_pin["rtol"])

    def test_pin_at_threshold_count_matches_rtol_device(self):
        rng = np.random.default_rng(91)
        n = 8
        Pi = _random_hermitian_psd(rng, n, rank=5) + np.eye(n) * 1e-3
        V = _random_hermitian_psd(rng, n)
        W_rtol, info_rtol = hermitian_sandwich_solve_device(Pi, V, rtol=1e-8)
        m = info_rtol["n_retained"]
        self.assertGreater(m, 0)
        W_pin, info_pin = hermitian_sandwich_solve_device(Pi, V, n_retained_pin=m)
        np.testing.assert_array_equal(np.asarray(W_pin), np.asarray(W_rtol))
        self.assertEqual(info_pin["n_retained"], m)
        self.assertEqual(info_pin["n_retained_pin"], m)
        self.assertIsNone(info_pin["rtol"])

    def test_device_pin_matches_numpy_pin(self):
        rng = np.random.default_rng(92)
        n = 8
        Pi = _random_hermitian_psd(rng, n, rank=6) + np.eye(n) * 1e-3
        V = _random_hermitian_psd(rng, n)
        for K in (1, 3, n):
            W_np, info_np = hermitian_sandwich_solve(Pi, V, n_retained_pin=K)
            W_jax, info_jax = hermitian_sandwich_solve_device(Pi, V, n_retained_pin=K)
            np.testing.assert_allclose(np.asarray(W_jax), W_np, atol=1e-12)
            self.assertEqual(info_jax["n_retained"], K)
            self.assertEqual(info_np["n_retained"], K)

    def test_pin_full_retention_drives_truncation_residual_to_zero(self):
        rng = np.random.default_rng(93)
        n = 6
        Pi = _random_hermitian_psd(rng, n) + np.eye(n) * 1e-6
        V = _random_hermitian_psd(rng, n)
        _, info = hermitian_sandwich_solve(Pi, V, n_retained_pin=n)
        self.assertEqual(info["n_retained"], n)
        self.assertLess(info["truncation_residual"], 1e-12)
        self.assertFalse(info["retention_marginal"])


class TestPinMisuse(unittest.TestCase):
    def _inputs(self, seed=94, n=5):
        rng = np.random.default_rng(seed)
        Pi = _random_hermitian_psd(rng, n) + np.eye(n) * 0.1
        V = _random_hermitian_psd(rng, n)
        return Pi, V

    def test_pin_plus_rtol_raises(self):
        Pi, V = self._inputs()
        for solve in (hermitian_sandwich_solve, hermitian_sandwich_solve_device):
            with self.assertRaises(ValueError):
                solve(Pi, V, rtol=1e-8, n_retained_pin=2)

    def test_pin_plus_target_truncation_residual_raises(self):
        Pi, V = self._inputs()
        with self.assertRaises(ValueError):
            hermitian_sandwich_solve(Pi, V, n_retained_pin=2, target_truncation_residual=1e-6)

    def test_pin_non_single_mode_raises(self):
        Pi, V = self._inputs()
        for mode in ("pairwise", "svd_lstsq"):
            for solve in (hermitian_sandwich_solve, hermitian_sandwich_solve_device):
                with self.assertRaises(ValueError):
                    solve(Pi, V, retention_mode=mode, n_retained_pin=2)

    def test_pin_out_of_range_raises(self):
        Pi, V = self._inputs()
        n = Pi.shape[0]
        for bad in (0, -1, n + 1):
            for solve in (hermitian_sandwich_solve, hermitian_sandwich_solve_device):
                with self.assertRaises(ValueError):
                    solve(Pi, V, n_retained_pin=bad)

    def test_pin_non_integer_raises(self):
        Pi, V = self._inputs()
        for bad in (2.5, "2", True, np.float64(2.0)):
            for solve in (hermitian_sandwich_solve, hermitian_sandwich_solve_device):
                with self.assertRaises(ValueError):
                    solve(Pi, V, n_retained_pin=bad)


class TestPinMarginalBand(unittest.TestCase):
    def test_marginal_uses_spectral_gap_at_pin(self):
        rng = np.random.default_rng(95)
        # K=2: s_K=1e-4 within 10x of s_3=9e-5 AND cond=1e4 -> marginal.
        # K=3: s_K=9e-5 vs s_4=1e-8 gap >> 10x -> not marginal.
        # K=4: nothing discarded -> no edge, never marginal.
        eigvals = [1.0, 1e-4, 9e-5, 1e-8]
        Pi = _pi_with_spectrum(rng, eigvals)
        V = _random_hermitian_psd(rng, len(eigvals))
        expected = {2: True, 3: False, 4: False}
        for K, want in expected.items():
            _, info_np = hermitian_sandwich_solve(Pi, V, n_retained_pin=K)
            _, info_jax = hermitian_sandwich_solve_device(Pi, V, n_retained_pin=K)
            self.assertEqual(info_np["retention_marginal"], want, f"numpy K={K}")
            self.assertEqual(info_jax["retention_marginal"], want, f"device K={K}")


if __name__ == "__main__":
    unittest.main()
