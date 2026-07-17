"""V1 oracle tests for pytc.pbc.df.isdf.pivoted_cholesky_hermitian
(task #21, design v2.1 section 3/8).
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell
from scipy.linalg.lapack import zpstrf

from pytc.pbc.df.isdf import (
    build_translation_ao_cache,
    build_translation_ao_representation,
    build_cached_periodic_pivot_oracle,
    build_periodic_batched_pivot_oracle,
    build_periodic_pivot_oracle,
    candidate_panel_indices,
    explicit_candidate_identity,
    full_grid_candidate_identity,
    JAXCachedMatrixFreeCapacityError,
    JAXTranslationMatrixFreeCapacityError,
    TranslationAORepresentationError,
    jax_cached_matrix_free_byte_model,
    jax_translation_matrix_free_byte_model,
    periodic_metric_column_from_ao,
    periodic_metric_columns_from_ao,
    periodic_metric_from_ao,
    pivoted_cholesky_hermitian,
    pivoted_cholesky_batched_hermitian,
    select_jax_cached_matrix_free,
    select_jax_translation_matrix_free,
    stream_ao_blocks_from_translation_cache,
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


def _translation_test_cell():
    cell = Cell()
    cell.atom = "He 0.2 0.3 0.4"
    cell.a = np.array([[3.2, 0.0, 0.0], [0.3, 3.0, 0.0], [0.1, 0.2, 2.9]])
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.mesh = [4, 3, 3]
    cell.verbose = 0
    cell.build()
    return cell


def _translation_diamond_cell():
    cell = Cell()
    cell.atom = "C 0.0 0.0 0.0; C 0.8917 0.8917 0.8917"
    cell.a = """0.0 1.7834 1.7834
1.7834 0.0 1.7834
1.7834 1.7834 0.0"""
    cell.unit = "A"
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 60.0
    cell.verbose = 0
    cell.build()
    return cell


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
    def test_batched_selector_is_deterministic_and_uses_exact_batch_columns(self):
        rng = np.random.default_rng(41)
        feature = rng.normal(size=(16, 7)) + 1j * rng.normal(size=(16, 7))
        metric = feature @ feature.conj().T
        diagonal = np.real(np.diag(metric))
        calls = []

        def batch_columns(indices):
            calls.append(np.asarray(indices).copy())
            return metric[:, indices]

        first = pivoted_cholesky_batched_hermitian(
            diagonal, batch_columns, rank=7, mesh=(4, 4, 1), batch_size=4,
        )
        second = pivoted_cholesky_batched_hermitian(
            diagonal, lambda indices: metric[:, indices], rank=7,
            mesh=(4, 4, 1), batch_size=4,
        )
        pivots, factor, count, rounds = first
        self.assertEqual(count, 7)
        np.testing.assert_array_equal(pivots, second[0])
        np.testing.assert_allclose(factor @ factor.conj().T, metric, atol=1e-10)
        self.assertEqual(sum(len(round_["retained_pivots"]) for round_ in rounds), count)
        self.assertTrue(all(1 <= len(indices) <= 4 for indices in calls))

    def test_batched_selector_exact_topup_matches_greedy(self):
        rng = np.random.default_rng(14)
        feature = rng.normal(size=(12, 8)) + 1j * rng.normal(size=(12, 8))
        metric = feature @ feature.conj().T
        diagonal = np.real(np.diag(metric))
        exact, _, _ = pivoted_cholesky_hermitian(
            diagonal, lambda index: metric[:, index], rank=7,
        )
        pivots, _, count, rounds = pivoted_cholesky_batched_hermitian(
            diagonal, lambda indices: metric[:, indices], rank=7, mesh=(3, 4, 1),
            batch_size=4, n_topup=7,
        )
        self.assertEqual(count, 7)
        np.testing.assert_array_equal(pivots, exact)
        self.assertEqual(rounds[-1]["mode"], "exact_topup")
        self.assertEqual(rounds[-1]["n_requested"], 7)

    def test_batched_streamed_and_cached_columns_match(self):
        cell = _SyntheticPeriodicCell()
        grid_coords = np.column_stack((np.arange(9), np.zeros((9, 2))))
        kpts = np.zeros((2, 3))
        _, streamed_batch = build_periodic_batched_pivot_oracle(
            cell, kpts, grid_coords, block_size=4,
        )
        _, _, cache = build_cached_periodic_pivot_oracle(cell, kpts, grid_coords, block_size=4)
        indices = np.array([1, 4, 7], dtype=np.int64)
        np.testing.assert_allclose(
            streamed_batch(indices), periodic_metric_columns_from_ao(cache, indices), atol=1e-12,
        )

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
            ao_cache, rank=3, selection_peak_max_bytes=10**9, return_factor=True,
        )
        self.assertEqual(count, streamed_count)
        np.testing.assert_array_equal(pivots, streamed_pivots)
        self.assertEqual(factor.dtype, np.float64)
        self.assertEqual(provenance["pivot_executor"], "jax.jit/lax.fori_loop")
        self.assertTrue(provenance["pivot_loop_device_resident"])
        self.assertEqual(cache_stats, {"pbc_eval_calls": 3, "grid_points": 9})
        production_pivots, production_factor, production_count, _ = (
            select_jax_cached_matrix_free(
                ao_cache, rank=3, selection_peak_max_bytes=10**9,
            )
        )
        self.assertIsNone(production_factor)
        self.assertEqual(production_count, streamed_count)
        np.testing.assert_array_equal(production_pivots, streamed_pivots)

    def test_jax_cached_matrix_free_capacity_is_fail_closed_before_selection(self):
        model = jax_cached_matrix_free_byte_model(2, 9, 3, 3, selection_peak_max_bytes=1)
        self.assertFalse(model["within_cache_policy"])
        self.assertEqual(
            model["capacity_condition"],
            "JAX_CACHED_MATRIX_FREE_SELECTION_PEAK_EXCEEDS_POLICY",
        )
        ao_cache = np.ones((2, 9, 3), dtype=np.complex128)
        with self.assertRaisesRegex(
            JAXCachedMatrixFreeCapacityError,
            "JAX_CACHED_MATRIX_FREE_SELECTION_PEAK_EXCEEDS_POLICY",
        ):
            select_jax_cached_matrix_free(ao_cache, rank=3, selection_peak_max_bytes=1)

    def test_translation_cache_reconstructs_bloch_aos_and_exact_pivots(self):
        cell = _translation_test_cell()
        kpts = cell.make_kpts([2, 2, 2], wrap_around=False)
        coords = cell.get_uniform_grids(cell.mesh)
        representation = build_translation_ao_representation(cell, kpts)
        self.assertEqual(representation.n_classes, len(kpts))
        self.assertLess(representation.phase_orthogonality_residual, 1e-12)

        stats = {}
        cache = build_translation_ao_cache(
            cell, coords, block_size=7, representation=representation, stats=stats,
        )
        reconstructed = np.concatenate([
            block for _, _, block in stream_ao_blocks_from_translation_cache(
                cache, representation.unitary_phase, block_size=5,
            )
        ], axis=1)
        direct = np.asarray(cell.pbc_eval_gto("GTOval", coords, kpts=list(kpts)))
        np.testing.assert_allclose(reconstructed, direct, atol=2e-12, rtol=2e-12)

        streamed_diag, streamed_col = build_periodic_pivot_oracle(
            cell, kpts, coords, block_size=7,
        )
        streamed_pivots, _, streamed_count = pivoted_cholesky_hermitian(
            streamed_diag, streamed_col, rank=4,
        )
        pivots, factor, count, provenance = select_jax_translation_matrix_free(
            cache, len(kpts), rank=4, selection_peak_max_bytes=10**9,
            ao_block_size=7,
            return_factor=True,
        )
        self.assertEqual(count, streamed_count)
        np.testing.assert_array_equal(pivots, streamed_pivots)
        self.assertEqual(factor.dtype, np.float64)
        self.assertEqual(provenance["mode"], "jax_translation_matrix_free")
        self.assertEqual(provenance["translation_cache_dtype"], "float64")
        self.assertEqual(
            stats["pbc_eval_calls"],
            int(np.ceil(len(coords) / 7)),
        )

    def test_translation_cache_is_half_the_complex_bloch_cache(self):
        full = jax_cached_matrix_free_byte_model(
            4, 27, 5, 6, selection_peak_max_bytes=10**9,
        )
        translated = jax_translation_matrix_free_byte_model(
            4, 4, 27, 5, 6, ao_block_size=7, selection_peak_max_bytes=10**9,
        )
        self.assertEqual(
            2 * translated["translation_cache_real_float64_bytes"],
            full["ao_cache_complex128_bytes"],
        )
        self.assertEqual(translated["translation_cache_to_complex_cache_ratio"], 0.5)
        self.assertEqual(
            translated["bounded_ao_block_complex128_bytes"], 4 * 7 * 5 * 16,
        )
        self.assertEqual(
            translated["bounded_transform_block_complex128_bytes"], 4 * 7 * 5 * 16,
        )
        self.assertGreater(
            translated["selection_peak_host_bytes"],
            translated["translation_cache_real_float64_bytes"],
        )

    def test_translation_selector_capacity_is_fail_closed(self):
        cache = np.ones((2, 3, 9), dtype=np.float64)
        with self.assertRaisesRegex(
            JAXTranslationMatrixFreeCapacityError,
            "JAX_TRANSLATION_MATRIX_FREE_SELECTION_PEAK_EXCEEDS_POLICY",
        ):
            select_jax_translation_matrix_free(
                cache, n_kpts=2, rank=3, ao_block_size=3,
                selection_peak_max_bytes=1,
            )

    def test_translation_representation_rejects_mesh_without_gamma(self):
        cell = _translation_test_cell()
        shifted = cell.make_kpts(
            [3, 1, 1], wrap_around=False, scaled_center=[0.17, 0.0, 0.0],
        )
        with self.assertRaisesRegex(
            TranslationAORepresentationError,
            "TRANSLATION_AO_REQUIRES_GAMMA_ORTHOGONAL_PHASE_CLASSES",
        ):
            build_translation_ao_representation(cell, shifted)

    def test_translation_representation_rejects_nonorthogonal_gamma_mesh(self):
        cell = _translation_test_cell()
        nonorthogonal = np.array([[0.0, 0.0, 0.0], [0.37, 0.0, 0.0]])
        with self.assertRaisesRegex(
            TranslationAORepresentationError,
            "TRANSLATION_AO_REQUIRES_GAMMA_ORTHOGONAL_PHASE_CLASSES",
        ):
            build_translation_ao_representation(cell, nonorthogonal)

    def test_diamond_4x4x4_has_64_orthogonal_translation_classes(self):
        cell = _translation_diamond_cell()
        representation = build_translation_ao_representation(
            cell, cell.make_kpts([4, 4, 4], wrap_around=False),
        )
        self.assertEqual(representation.lattice_vectors.shape[0], 887)
        self.assertEqual(representation.n_classes, 64)
        self.assertLess(representation.phase_orthogonality_residual, 1e-12)

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
