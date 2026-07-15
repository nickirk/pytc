"""V1 oracle tests for pytc.pbc.df.isdf.pivoted_cholesky_hermitian
(task #21, design v2.1 section 3/8).
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy.linalg.lapack import zpstrf

from pytc.pbc.df.isdf import (
    build_cached_periodic_pivot_oracle,
    build_periodic_pivot_oracle,
    candidate_panel_indices,
    explicit_candidate_identity,
    full_grid_candidate_identity,
    JAXCachedMatrixFreeCapacityError,
    jax_cached_matrix_free_byte_model,
    periodic_metric_column_from_ao,
    periodic_metric_from_ao,
    pivoted_cholesky_hermitian,
    select_jax_cached_matrix_free,
)


def _random_psd(rng, n, true_rank, dtype=np.complex128):
    A = rng.normal(size=(n, true_rank)) + 1j * rng.normal(size=(n, true_rank))
    A = A.astype(dtype)
    return A @ A.conj().T


class _SyntheticPeriodicCell:
    def pbc_eval_gto(self, label, coords, kpts):
        self.assert_label(label)
        grid_index = np.asarray(coords[:, 0], dtype=np.int64)
        n_kpts = len(kpts)
        ao_index = np.arange(3)[None, :]
        values = []
        for k_index in range(n_kpts):
            phase = np.exp(1j * (k_index + 1) * (grid_index[:, None] + ao_index))
            values.append((grid_index[:, None] + 1 + ao_index) * phase)
        return np.asarray(values)

    @staticmethod
    def assert_label(label):
        if label != "GTOval":
            raise AssertionError(label)


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


class TestExperimentalSelectionPrimitives(unittest.TestCase):
    def test_full_grid_identity_is_compact_range(self):
        identity = full_grid_candidate_identity(10**9)
        self.assertEqual(
            identity,
            {"kind": "range", "start": 0, "stop": 10**9, "step": 1},
        )
        self.assertNotIn("indices", identity)

    def test_panel_identity_keeps_explicit_indices(self):
        identity = explicit_candidate_identity(np.array([7, 2, 5], dtype=np.int64))
        self.assertEqual(identity, {"kind": "explicit_indices", "indices": [7, 2, 5]})

    def test_cached_full_oracle_matches_streamed_pivots(self):
        cell = _SyntheticPeriodicCell()
        grid_coords = np.column_stack((np.arange(9), np.zeros((9, 2))))
        kpts = np.zeros((2, 3))
        streamed_diag, streamed_col = build_periodic_pivot_oracle(
            cell, kpts, grid_coords, block_size=4,
        )
        cached_stats = {}
        cached_diag, cached_col, _ = build_cached_periodic_pivot_oracle(
            cell, kpts, grid_coords, block_size=4, stats=cached_stats,
        )
        streamed_pivots, _, streamed_count = pivoted_cholesky_hermitian(
            streamed_diag, streamed_col, rank=3,
        )
        cached_pivots, _, cached_count = pivoted_cholesky_hermitian(
            cached_diag, cached_col, rank=3,
        )
        np.testing.assert_allclose(cached_diag, streamed_diag)
        self.assertEqual(cached_count, streamed_count)
        np.testing.assert_array_equal(cached_pivots, streamed_pivots)
        self.assertEqual(cached_stats, {"pbc_eval_calls": 3, "grid_points": 9})

    def test_jax_cached_matrix_free_matches_streamed_pivots_and_rank(self):
        cell = _SyntheticPeriodicCell()
        grid_coords = np.column_stack((np.arange(9), np.zeros((9, 2))))
        kpts = np.zeros((2, 3))
        streamed_diag, streamed_col = build_periodic_pivot_oracle(
            cell, kpts, grid_coords, block_size=4,
        )
        cache_stats = {}
        _, _, ao_cache = build_cached_periodic_pivot_oracle(
            cell, kpts, grid_coords, block_size=4, stats=cache_stats,
        )
        streamed_pivots, _, streamed_count = pivoted_cholesky_hermitian(
            streamed_diag, streamed_col, rank=3,
        )
        pivots, factor, count, provenance = select_jax_cached_matrix_free(
            ao_cache, rank=3, cache_max_bytes=10**9,
        )
        self.assertEqual(count, streamed_count)
        np.testing.assert_array_equal(pivots, streamed_pivots)
        self.assertEqual(factor.dtype, np.float64)
        self.assertEqual(provenance["pivot_executor"], "jax.jit/lax.fori_loop")
        self.assertTrue(provenance["pivot_loop_device_resident"])
        self.assertEqual(cache_stats, {"pbc_eval_calls": 3, "grid_points": 9})

    def test_jax_cached_matrix_free_capacity_is_fail_closed_before_selection(self):
        model = jax_cached_matrix_free_byte_model(2, 9, 3, 3, cache_max_bytes=1)
        self.assertFalse(model["within_cache_policy"])
        self.assertEqual(
            model["capacity_condition"],
            "JAX_CACHED_MATRIX_FREE_AO_CACHE_EXCEEDS_POLICY",
        )
        ao_cache = np.ones((2, 9, 3), dtype=np.complex128)
        with self.assertRaisesRegex(
            JAXCachedMatrixFreeCapacityError,
            "JAX_CACHED_MATRIX_FREE_AO_CACHE_EXCEEDS_POLICY",
        ):
            select_jax_cached_matrix_free(ao_cache, rank=3, cache_max_bytes=1)

    def test_panel_metric_matches_direct_periodic_definition(self):
        rng = np.random.default_rng(29)
        ao = rng.normal(size=(3, 5, 2)) + 1j * rng.normal(size=(3, 5, 2))
        metric = periodic_metric_from_ao(ao)
        direct = np.empty((5, 5), dtype=np.complex128)
        for r in range(5):
            for s in range(5):
                direct[r, s] = abs(np.vdot(ao[:, r, :], ao[:, s, :])) ** 2 / 3
        np.testing.assert_allclose(metric, direct, atol=1e-12)
        for column in range(5):
            np.testing.assert_allclose(periodic_metric_column_from_ao(ao, column), direct[:, column])

    def test_panel_dense_and_oracle_select_identical_pivots(self):
        rng = np.random.default_rng(31)
        ao = rng.normal(size=(3, 8, 3)) + 1j * rng.normal(size=(3, 8, 3))
        metric = periodic_metric_from_ao(ao)
        dense_pivots, _, dense_count = pivoted_cholesky_hermitian(
            metric.real.diagonal(), lambda j: metric[:, j], rank=4,
        )
        oracle_pivots, _, oracle_count = pivoted_cholesky_hermitian(
            metric.real.diagonal(), lambda j: periodic_metric_column_from_ao(ao, j), rank=4,
        )
        self.assertEqual(oracle_count, dense_count)
        np.testing.assert_array_equal(oracle_pivots, dense_pivots)

    def test_candidate_panel_is_unique_and_uses_higher_index_ties(self):
        panel = candidate_panel_indices(np.ones(20), rank=3, panel_factor=4)
        self.assertEqual(panel.size, 12)
        self.assertEqual(len(set(panel.tolist())), panel.size)
        self.assertEqual(panel[0], 19)

    def test_candidate_panel_contract_on_nonuniform_diagonal(self):
        diag = np.array([1., 9., 2., 8., 3., 7., 4., 6., 5., 10.])
        panel = candidate_panel_indices(diag, rank=2, panel_factor=4)
        np.testing.assert_array_equal(panel, np.array([9, 1, 3, 5, 7, 8, 6, 4]))

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
