"""V1 oracle tests for pytc.pbc.df.isdf.pivoted_cholesky_hermitian
(task #21, design v2.1 section 3/8).
"""

import unittest

import numpy as np
from scipy.linalg.lapack import zpstrf

from pytc.pbc.df.isdf import pivoted_cholesky_hermitian


def _random_psd(rng, n, true_rank, dtype=np.complex128):
    A = rng.normal(size=(n, true_rank)) + 1j * rng.normal(size=(n, true_rank))
    A = A.astype(dtype)
    return A @ A.conj().T


class TestPivotedCholeskyHermitian(unittest.TestCase):
    def test_reconstructs_exactly_at_true_rank(self):
        rng = np.random.default_rng(20)
        n, true_rank = 10, 6
        M = _random_psd(rng, n, true_rank)
        diag = np.real(np.diag(M)).copy()
        pivots, L, n_sel = pivoted_cholesky_hermitian(diag, lambda j: M[:, j], rank=true_rank)
        self.assertEqual(n_sel, true_rank)
        recon = L @ L.conj().T
        np.testing.assert_allclose(recon, M, atol=1e-10)

    def test_over_requesting_rank_stops_early_at_true_rank(self):
        rng = np.random.default_rng(21)
        n, true_rank = 10, 4
        M = _random_psd(rng, n, true_rank)
        diag = np.real(np.diag(M)).copy()
        pivots, L, n_sel = pivoted_cholesky_hermitian(diag, lambda j: M[:, j], rank=n)
        self.assertEqual(n_sel, true_rank)
        recon = L @ L.conj().T
        np.testing.assert_allclose(recon, M, atol=1e-10)

    def test_duplicate_pivot_never_selected_twice(self):
        rng = np.random.default_rng(22)
        n, true_rank = 12, 5
        M = _random_psd(rng, n, true_rank)
        diag = np.real(np.diag(M)).copy()
        pivots, L, n_sel = pivoted_cholesky_hermitian(diag, lambda j: M[:, j], rank=n)
        self.assertEqual(len(set(pivots.tolist())), len(pivots))

    def test_matches_lapack_zpstrf_rank(self):
        # Cross-validate the numerical RANK-DETECTION threshold against an
        # independent LAPACK implementation -- a permutation-independent
        # invariant (LAPACK's own pivot convention is not replicated here;
        # our own reconstruction correctness against M is already checked
        # separately in test_reconstructs_exactly_at_true_rank and
        # test_over_requesting_rank_stops_early_at_true_rank).
        rng = np.random.default_rng(23)
        n, true_rank = 9, 5
        M = _random_psd(rng, n, true_rank)
        diag = np.real(np.diag(M)).copy()

        pivots, L, n_sel = pivoted_cholesky_hermitian(diag, lambda j: M[:, j], rank=n)

        _, _, rank_lapack, info = zpstrf(M.copy(), lower=1)
        # LAPACK ?pstrf: info=0 means full rank, info=k>0 means rank-
        # deficient with the computed rank still valid in rank_lapack --
        # only info<0 is a real argument error. M is intentionally rank-
        # deficient here (true_rank < n), so info=1 is the EXPECTED,
        # correct outcome, not a failure.
        self.assertGreaterEqual(info, 0)
        self.assertEqual(n_sel, rank_lapack)

    def test_scale_invariance(self):
        rng = np.random.default_rng(24)
        n, true_rank = 8, 4
        M = _random_psd(rng, n, true_rank)
        diag = np.real(np.diag(M)).copy()
        pivots1, L1, n1 = pivoted_cholesky_hermitian(diag, lambda j: M[:, j], rank=n)

        scale = 1e6
        M2 = M * scale
        diag2 = np.real(np.diag(M2)).copy()
        pivots2, L2, n2 = pivoted_cholesky_hermitian(diag2, lambda j: M2[:, j], rank=n)

        self.assertEqual(n1, n2)
        np.testing.assert_array_equal(pivots1, pivots2)
        recon2 = L2 @ L2.conj().T
        np.testing.assert_allclose(recon2, M2, atol=1e-9 * scale)

    def test_dtype_invariance_real_vs_complex_with_zero_imag(self):
        rng = np.random.default_rng(25)
        n, true_rank = 7, 3
        A = rng.normal(size=(n, true_rank))
        M_real_valued = (A @ A.T).astype(np.complex128)  # real-valued but complex dtype
        diag = np.real(np.diag(M_real_valued)).copy()
        pivots, L, n_sel = pivoted_cholesky_hermitian(
            diag, lambda j: M_real_valued[:, j], rank=n
        )
        recon = L @ L.conj().T
        np.testing.assert_allclose(recon, M_real_valued, atol=1e-10)
        np.testing.assert_allclose(recon.imag, 0.0, atol=1e-10)

    def test_determinism_across_repeated_calls(self):
        rng = np.random.default_rng(26)
        n, true_rank = 10, 5
        M = _random_psd(rng, n, true_rank)
        diag = np.real(np.diag(M)).copy()
        pivots1, L1, n1 = pivoted_cholesky_hermitian(diag, lambda j: M[:, j], rank=n)
        pivots2, L2, n2 = pivoted_cholesky_hermitian(diag, lambda j: M[:, j], rank=n)
        np.testing.assert_array_equal(pivots1, pivots2)
        np.testing.assert_array_equal(L1, L2)

    def test_direct_dense_periodic_metric_tiny_cell(self):
        # A tiny "periodic-style" metric: Pi = X^dagger X for a random
        # (Nao, Nip) factor X, exactly the structure Pi^q takes in the
        # real periodic builder (V2) -- exercised here as a plain dense
        # PSD oracle, independent of any k-point machinery.
        rng = np.random.default_rng(27)
        n_ao, n_ip = 6, 4
        X = rng.normal(size=(n_ao, n_ip)) + 1j * rng.normal(size=(n_ao, n_ip))
        Pi = X.conj().T @ X  # (n_ip, n_ip) Hermitian PSD
        diag = np.real(np.diag(Pi)).copy()
        pivots, L, n_sel = pivoted_cholesky_hermitian(diag, lambda j: Pi[:, j], rank=n_ip)
        recon = L @ L.conj().T
        np.testing.assert_allclose(recon, Pi, atol=1e-10)

    def test_rejects_rank_exceeding_n(self):
        rng = np.random.default_rng(28)
        M = _random_psd(rng, 5, 3)
        diag = np.real(np.diag(M)).copy()
        with self.assertRaises(ValueError):
            pivoted_cholesky_hermitian(diag, lambda j: M[:, j], rank=6)

    def test_rejects_non_psd_diag(self):
        diag = np.array([1.0, -5.0, 2.0])
        with self.assertRaises(ValueError):
            pivoted_cholesky_hermitian(diag, lambda j: np.zeros(3), rank=2)

    def test_rejects_malformed_col_eval_output(self):
        diag = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            pivoted_cholesky_hermitian(diag, lambda j: np.zeros(2), rank=2)

    def test_rejects_bad_rank_type(self):
        diag = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            pivoted_cholesky_hermitian(diag, lambda j: np.zeros(2), rank=0)
        with self.assertRaises(ValueError):
            pivoted_cholesky_hermitian(diag, lambda j: np.zeros(2), rank=True)


if __name__ == "__main__":
    unittest.main()
