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
    build_cached_periodic_pivot_oracle,
    build_cached_periodic_bpc_gemm_oracle,
    build_periodic_batched_pivot_oracle,
    build_periodic_pivot_oracle,
    explicit_candidate_identity,
    full_grid_candidate_identity,
    periodic_metric_column_from_ao,
    periodic_metric_columns_from_ao,
    pivoted_cholesky_hermitian,
    pivoted_cholesky_batched_hermitian,
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

    def test_batched_selector_stage_stats_are_observational(self):
        rng = np.random.default_rng(113)
        feature = rng.normal(size=(12, 8)) + 1j * rng.normal(size=(12, 8))
        metric = feature @ feature.conj().T
        stats = []
        pivots, _, count, _ = pivoted_cholesky_batched_hermitian(
            np.real(np.diag(metric)), lambda indices: metric[:, indices], rank=7,
            mesh=(3, 4, 1), batch_size=4, n_topup=2, stage_stats=stats,
        )
        self.assertEqual(count, 7)
        self.assertEqual(len(pivots), 7)
        self.assertGreaterEqual(len(stats), 4)
        self.assertEqual([item["stage"] for item in stats[-2:]], ["topup", "topup"])
        for item in stats:
            self.assertGreaterEqual(item["candidate_eval_seconds"], 0.0)
            self.assertGreaterEqual(item["projection_seconds"], 0.0)
            self.assertGreaterEqual(item["within_batch_pivot_seconds"], 0.0)
            self.assertGreaterEqual(item["factor_update_seconds"], 0.0)
            # Required on EVERY record, both stages. Under blocked_projection
            # `projection_seconds` times dispatch only -- the gather and GEMM are
            # async and the host pays for them at the next read. Without this
            # bucket those seconds fall between two perf_counter calls and are
            # attributed to no stage, which is how a wall-clock gap can look
            # unexplained while every named stage looks cheap.
            self.assertIn("materialisation_seconds", item)
            self.assertGreaterEqual(item["materialisation_seconds"], 0.0)

    def test_batched_selector_oversampling_keeps_batch_rank(self):
        rng = np.random.default_rng(52)
        feature = rng.normal(size=(18, 9)) + 1j * rng.normal(size=(18, 9))
        metric = feature @ feature.conj().T
        diagonal = np.real(np.diag(metric))
        calls = []

        def batch_columns(indices):
            calls.append(np.asarray(indices).copy())
            return metric[:, indices]

        pivots, _, count, rounds = pivoted_cholesky_batched_hermitian(
            diagonal, batch_columns, rank=8, mesh=(3, 3, 2), batch_size=4,
            candidate_oversampling=2, min_separation=0.0,
        )
        self.assertEqual(count, 8)
        self.assertEqual(len(pivots), 8)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(len(indices) == 8 for indices in calls))
        self.assertTrue(all(len(round_["retained_pivots"]) == 4 for round_ in rounds))

    def test_blocked_projection_matches_sequential_pivots(self):
        # task #45: the opt-in BLAS-3 blocked projection/update is a reordering
        # of the accepted sequential per-pivot factor update, so it must
        # reproduce the sequential pivots bit-for-bit (residual arbitrates any
        # summation-order tie) across ranks/oversampling/top-up, and its factor
        # must agree to fp noise.
        configs = [
            dict(seed=1, n=64, true_rank=40, rank=30, batch_size=8,
                 candidate_oversampling=2, n_topup=4, min_separation=1.0),
            dict(seed=2, n=64, true_rank=64, rank=48, batch_size=16,
                 candidate_oversampling=1, n_topup=0, min_separation=0.0),
            dict(seed=3, n=81, true_rank=50, rank=40, batch_size=8,
                 candidate_oversampling=3, n_topup=8, min_separation=1.5),
            dict(seed=4, n=100, true_rank=100, rank=60, batch_size=16,
                 candidate_oversampling=2, n_topup=10, min_separation=0.0),
        ]
        for cfg in configs:
            rng = np.random.default_rng(cfg["seed"])
            metric = _random_psd(rng, cfg["n"], cfg["true_rank"])
            diagonal = np.real(np.diag(metric))
            side = int(round(cfg["n"] ** 0.5))
            mesh = (side, cfg["n"] // side, 1)
            self.assertEqual(mesh[0] * mesh[1], cfg["n"])
            kw = dict(
                mesh=mesh, batch_size=cfg["batch_size"],
                min_separation=cfg["min_separation"],
                candidate_oversampling=cfg["candidate_oversampling"],
                n_topup=cfg["n_topup"],
            )

            def columns(indices):
                return metric[:, indices]

            p_seq, f_seq, c_seq, _ = pivoted_cholesky_batched_hermitian(
                diagonal, columns, rank=cfg["rank"],
                blocked_projection=False, **kw)
            p_blk, f_blk, c_blk, _ = pivoted_cholesky_batched_hermitian(
                diagonal, columns, rank=cfg["rank"],
                blocked_projection=True, **kw)
            with self.subTest(**cfg):
                self.assertEqual(c_blk, c_seq)
                np.testing.assert_array_equal(p_blk, p_seq)
                np.testing.assert_allclose(f_blk, f_seq, atol=1e-11, rtol=0)
                np.testing.assert_allclose(
                    f_blk @ f_blk.conj().T, f_seq @ f_seq.conj().T, atol=1e-11)

    def test_blocked_projection_reconstructs_metric(self):
        # Blocked path is a valid partial Cholesky in its own right.
        rng = np.random.default_rng(77)
        metric = _random_psd(rng, 25, 25)
        diagonal = np.real(np.diag(metric))
        pivots, factor, count, _ = pivoted_cholesky_batched_hermitian(
            diagonal, lambda indices: metric[:, indices], rank=25,
            mesh=(5, 5, 1), batch_size=6, candidate_oversampling=2,
            n_topup=3, blocked_projection=True)
        self.assertEqual(count, 25)
        np.testing.assert_allclose(factor @ factor.conj().T, metric, atol=1e-9)

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

    def test_cached_bpc_gemm_columns_match_existing_cache(self):
        cell = _SyntheticPeriodicCell()
        grid_coords = np.column_stack((np.arange(9), np.zeros((9, 2))))
        kpts = np.zeros((2, 3))
        _, _, cache = build_cached_periodic_pivot_oracle(cell, kpts, grid_coords, block_size=4)
        diag, gemm_columns, _ = build_cached_periodic_bpc_gemm_oracle(
            cell, kpts, grid_coords, block_size=4,
        )
        indices = np.array([1, 4, 7], dtype=np.int64)
        np.testing.assert_allclose(diag, np.sum(np.abs(cache) ** 2, axis=(0, 2)) ** 2 / 2)
        np.testing.assert_allclose(gemm_columns(indices), periodic_metric_columns_from_ao(cache, indices))

    def test_real_f64_l_periodic_factor_matches_complex_reference(self):
        # real-f64-L (task #46): the periodic metric M = |gram|^2/Nk is real, so
        # the BPC pivoted-Cholesky factor L is stored float64 -- HALVING the
        # dominant [Ng x rank] selection-phase term. The selection is unchanged:
        # pivots are BIT-IDENTICAL to a complex-recast reference and the factor
        # matches its real part to BLAS precision (real-vs-complex GEMM, dgemm vs
        # zgemm, differs only at ~1e-15 -- the factor is NOT bit-for-bit, the
        # PIVOTS are). The complex reference's imaginary part is identically zero.
        cell = _translation_diamond_cell()
        kpts = cell.make_kpts([2, 2, 2])
        grid_coords = cell.get_uniform_grids(cell.mesh)
        diag, real_columns, _ = build_cached_periodic_bpc_gemm_oracle(
            cell, kpts, grid_coords, block_size=256,
        )
        complex_columns = lambda idx: np.asarray(real_columns(idx)).astype(np.complex128)
        kw = dict(rank=4 * cell.nao_nr(), mesh=cell.mesh, batch_size=64,
                  min_separation=2.0, candidate_oversampling=4, n_topup=16)
        pivots_real, factor_real, n_real, _ = pivoted_cholesky_batched_hermitian(
            diag.copy(), real_columns, **kw)
        pivots_cplx, factor_cplx, n_cplx, _ = pivoted_cholesky_batched_hermitian(
            diag.copy(), complex_columns, **kw)
        # factor follows the metric dtype: real metric -> float64 (halved).
        self.assertEqual(factor_real.dtype, np.float64)
        self.assertEqual(factor_cplx.dtype, np.complex128)
        self.assertEqual(factor_cplx.nbytes, 2 * factor_real.nbytes)
        # selection is IDENTICAL (pivots bit-for-bit); metric was genuinely real.
        self.assertEqual(n_real, n_cplx)
        np.testing.assert_array_equal(pivots_real, pivots_cplx)
        self.assertEqual(float(np.max(np.abs(factor_cplx.imag))), 0.0)
        # factor matches to BLAS precision (dgemm vs zgemm), not bit-for-bit.
        np.testing.assert_allclose(factor_real, factor_cplx.real, rtol=0.0, atol=1e-11)

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

    def test_cached_oracle_matches_streamed_pivots(self):
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


class TestSelectionFactorLifetime(unittest.TestCase):
    """The selection factor must not survive its own selector call.

    `pivoted_cholesky_*` returns the [n_grid x rank] factor in position 1.
    Binding it to `_` at the call site keeps it alive for the rest of
    coulomb.build -- which includes the entire panel loop. At 444/cc-pvtz that
    was measured as one live float64[328509, 37120] = 97.6 GB still resident at
    panel-loop entry (job 60116274), and it is the memory half of the 444
    regression.

    NOTE for anyone extending this: do not patch with `Mock(return_value=...)`.
    The mock retains the returned tuple itself, so the weakref never dies, the
    factor is alive at the callback no matter what coulomb.py does, and the
    assertions below fail even against a correct fix -- a gate that cannot
    pass. Patch with a plain function that builds the tuple per call.
    """

    def _run_and_report_liveness(self, selection_mode):
        import gc
        import weakref
        import jax.numpy as jnp
        from pytc.pbc import coulomb as _coulomb

        cell = Cell()
        cell.atom = "H 0 0 0; H 0 0 0.74"
        cell.a = np.eye(3) * 4.0
        cell.basis = "sto-3g"
        cell.unit = "A"
        cell.ke_cutoff = 8.0
        cell.verbose = 0
        cell.build()

        seen = {}

        def _make(real):
            def patched(*args, **kwargs):
                out = real(*args, **kwargs)
                factor = jnp.asarray(np.asarray(out[1]))
                seen["ref"] = weakref.ref(factor)
                return (out[0], factor) + tuple(out[2:])
            return patched

        target = ("pivoted_cholesky_batched_hermitian"
                  if selection_mode != "streamed" else
                  "pivoted_cholesky_hermitian")
        real = getattr(_coulomb, target)

        def on_selection(pivots, provenance):
            gc.collect()
            seen["alive_at_callback"] = seen["ref"]() is not None

        original = getattr(_coulomb, target)
        setattr(_coulomb, target, _make(real))
        try:
            _coulomb.build(cell, cell.make_kpts([1, 1, 1], wrap_around=False),
                           rank=4, block_size=64,
                           selection_mode=selection_mode,
                           on_selection=on_selection)
        finally:
            setattr(_coulomb, target, original)
        self.assertIn("alive_at_callback", seen,
                      "the patched selector never ran -- the test proves nothing")
        return seen["alive_at_callback"]

    def test_streamed_releases_selection_factor(self):
        self.assertFalse(
            self._run_and_report_liveness("streamed"),
            "streamed branch retains the selection factor past pivot extraction; "
            "remove of `del selection_factor` must fail this test")

    def test_bpc_releases_selection_factor(self):
        self.assertFalse(
            self._run_and_report_liveness("bpc_streamed"),
            "bpc branch retains the selection factor past pivot extraction; "
            "removal of `del selection_factor` must fail this test")


if __name__ == "__main__":
    unittest.main()
