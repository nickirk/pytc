"""FP64 dense-oracle gates for the test-only panelled Phase-C prototype."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pytc.df import thc as robust_df_thc
from pytc.solver.test.robust_df_thc_h10_fingerprint import (
    canonical_array_fingerprint,
)
from pytc.solver.test.robust_df_thc_scalable import (
    NORMAL_EQUATION_RESOLUTION_RCOND,
    direct_df_sandwiches_panelled,
    fit_panelled_lsthc,
    phase_c_shape_flop_memory_ledger,
)
import pytc.solver.test.robust_df_thc_scalable as scalable_module


class TestPanelledRobustDFTHCOracle(unittest.TestCase):
    """Keep the Phase-C algebra paired to the accepted dense FP64 oracle."""

    def setUp(self):
        rng = np.random.default_rng(20260716)
        self.nocc = 2
        self.nvir = 5
        self.naux = 9
        self.rank = 4
        b_raw = rng.normal(size=(self.nvir, self.nvir, self.naux)).astype(np.float64)
        self.b = 0.5 * (b_raw + b_raw.swapaxes(0, 1))
        self.p = rng.normal(size=(self.nvir, self.rank)).astype(np.float64)
        t2_raw = rng.normal(
            size=(self.nocc, self.nocc, self.nvir, self.nvir)
        ).astype(np.float64)
        self.t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))
        self.rcond = 1.0e-12
        self.dense_model = robust_df_thc.build_robust_df_thc_model(
            self.b, self.p, rcond=self.rcond
        )
        self.fit = fit_panelled_lsthc(
            self.p, self.b, rcond=self.rcond, virtual_panel=2
        )
        self.panelled = direct_df_sandwiches_panelled(
            self.b, self.fit, self.t2, rank_panel=2, aux_panel=3
        )

    def test_panelled_fit_matches_dense_weights_rank_and_conditioning(self):
        self.assertEqual(self.fit.effective_rank, self.dense_model.effective_rank)
        self.assertEqual(self.fit.rcond, self.dense_model.rcond)
        self.assertEqual(
            self.fit.resolved_rcond,
            max(self.rcond, NORMAL_EQUATION_RESOLUTION_RCOND),
        )
        np.testing.assert_allclose(
            self.fit.y, self.dense_model.weights, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            self.fit.singular_values,
            self.dense_model.singular_values,
            rtol=1e-11,
            atol=1e-11,
        )
        self.assertAlmostEqual(
            self.fit.condition_number,
            self.dense_model.condition_number,
            delta=1e-11 * self.dense_model.condition_number,
        )

    def test_panelled_exact_cross_full_and_robust_terms_match_dense_oracle(self):
        b_tilde = self.dense_model.b_tilde.reshape(
            self.nvir, self.nvir, self.naux
        )
        expected_exact = robust_df_thc.direct_df_vvvv_t2_sandwich(
            self.b, self.b, self.t2
        )
        expected_fit_left = robust_df_thc.direct_df_vvvv_t2_sandwich(
            b_tilde, self.b, self.t2
        )
        expected_df_left = robust_df_thc.direct_df_vvvv_t2_sandwich(
            self.b, b_tilde, self.t2
        )
        expected_full = robust_df_thc.direct_df_vvvv_t2_sandwich(
            b_tilde, b_tilde, self.t2
        )
        expected_robust = expected_fit_left + expected_df_left - expected_full
        expected_delta_delta = robust_df_thc.direct_df_vvvv_t2_sandwich(
            self.dense_model.delta_b.reshape(self.nvir, self.nvir, self.naux),
            self.dense_model.delta_b.reshape(self.nvir, self.nvir, self.naux),
            self.t2,
        )

        for actual, expected in (
            (self.panelled.exact, expected_exact),
            (self.panelled.fit_left_df_right, expected_fit_left),
            (self.panelled.df_left_fit_right, expected_df_left),
            (self.panelled.full_thc, expected_full),
            (self.panelled.robust, expected_robust),
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

        # The sign is the required robust identity, not an error magnitude.
        np.testing.assert_allclose(
            self.panelled.exact - self.panelled.robust,
            expected_delta_delta,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_panelled_cross_terms_and_results_preserve_rccsd_pair_swaps(self):
        pair_swap = (1, 0, 3, 2)
        np.testing.assert_allclose(
            self.panelled.fit_left_df_right,
            self.panelled.df_left_fit_right.transpose(pair_swap),
            rtol=1e-11,
            atol=1e-11,
        )
        for value in (
            self.panelled.exact,
            self.panelled.full_thc,
            self.panelled.robust,
        ):
            np.testing.assert_allclose(
                value, value.transpose(pair_swap), rtol=1e-11, atol=1e-11
            )

    def test_irregular_rank_and_auxiliary_panels_are_invariant(self):
        """Both choices deliberately leave tails on rank and auxiliary axes."""

        first = direct_df_sandwiches_panelled(
            self.b, self.fit, self.t2, rank_panel=3, aux_panel=4
        )
        second = direct_df_sandwiches_panelled(
            self.b, self.fit, self.t2, rank_panel=3, aux_panel=5
        )
        for name in (
            "exact",
            "fit_left_df_right",
            "df_left_fit_right",
            "full_thc",
            "robust",
        ):
            np.testing.assert_allclose(
                getattr(first, name), getattr(second, name), rtol=1e-11, atol=1e-11
            )

    def test_overcomplete_source_uses_reported_normal_equation_resolution_floor(self):
        """Rank provenance must not retain Gram-roundoff modes as LS modes."""

        rng = np.random.default_rng(20260717)
        nvir, rank, naux, nocc = 3, 8, 5, 2
        p = rng.normal(size=(nvir, rank)).astype(np.float64)
        b_raw = rng.normal(size=(nvir, nvir, naux)).astype(np.float64)
        b = 0.5 * (b_raw + b_raw.swapaxes(0, 1))
        t2_raw = rng.normal(size=(nocc, nocc, nvir, nvir)).astype(np.float64)
        t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))

        fit = fit_panelled_lsthc(p, b, rcond=1.0e-12, virtual_panel=2)
        dense = robust_df_thc.build_robust_df_thc_model(
            b, p, rcond=fit.resolved_rcond
        )
        represented_b_tilde = np.einsum(
            "am,cm,mq->acq", p, p, fit.y, optimize=True
        )
        panelled = direct_df_sandwiches_panelled(
            b, fit, t2, rank_panel=3, aux_panel=2
        )
        dense_sandwiches = robust_df_thc.direct_df_sandwiches(dense, t2)

        self.assertEqual(fit.rcond, 1.0e-12)
        self.assertEqual(fit.normal_equation_resolution_rcond, NORMAL_EQUATION_RESOLUTION_RCOND)
        self.assertEqual(fit.resolved_rcond, NORMAL_EQUATION_RESOLUTION_RCOND)
        self.assertEqual(fit.effective_rank, dense.effective_rank)
        self.assertEqual(fit.effective_rank, nvir * (nvir + 1) // 2)
        np.testing.assert_allclose(
            represented_b_tilde,
            dense.b_tilde.reshape(nvir, nvir, naux),
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            panelled.exact, dense_sandwiches.exact, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            panelled.full_thc, dense_sandwiches.lsthc, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            panelled.robust, dense_sandwiches.robust, rtol=1e-11, atol=1e-11
        )

    def test_seeded_random_inputs_have_canonical_fp64_fingerprints(self):
        fingerprints = {
            "b": canonical_array_fingerprint(self.b),
            "p": canonical_array_fingerprint(self.p),
            "t2": canonical_array_fingerprint(self.t2),
        }
        self.assertEqual(fingerprints["b"]["shape"], [5, 5, 9])
        self.assertEqual(fingerprints["p"]["shape"], [5, 4])
        self.assertEqual(fingerprints["t2"]["shape"], [2, 2, 5, 5])
        self.assertEqual(
            {entry["canonical_dtype"] for entry in fingerprints.values()}, {"float64"}
        )
        self.assertEqual(
            fingerprints["b"]["sha256_c_contiguous_canonical_bytes"],
            "9ac7258759d51b56bae6b56bac36665c9ce73cfabcbcb443cf743abf80eb0fad",
        )
        self.assertEqual(
            fingerprints["p"]["sha256_c_contiguous_canonical_bytes"],
            "1d5892cb89b32a6d86032548f2b9cea3cb4a1e515a7e29b0d2efeb56fcaa939c",
        )
        self.assertEqual(
            fingerprints["t2"]["sha256_c_contiguous_canonical_bytes"],
            "9f51a8cad9c90436913833311146565e5123a00e8f152be9e06acb76c1875286",
        )

    def test_panelled_source_does_not_call_phase_b_pair_or_b_tilde_builders(self):
        """The prototype may use B/P panels, never Phase-B dense helpers."""

        source = Path(scalable_module.__file__).read_text()
        self.assertNotIn("scalar_pair_collocation(", source)
        self.assertNotIn("virtual_pair_df_matrix(", source)
        self.assertNotIn("build_robust_df_thc_model(", source)
        self.assertNotIn("ijmbq", source)
        self.assertNotIn("ijanq", source)


class TestPhaseCShapeFlopMemoryLedger(unittest.TestCase):
    def test_h10_fixed_rank_ledger_keeps_forbidden_dense_shapes_as_metadata_only(self):
        ledger = phase_c_shape_flop_memory_ledger(
            nocc=5,
            nvir=5,
            naux=180,
            n_fused=240,
            rank_panel=17,
            aux_panel=31,
        )
        self.assertEqual(
            ledger["dimensions"],
            {"nocc": 5, "nvir": 5, "naux": 180, "n_fused": 240},
        )
        self.assertEqual(ledger["forbidden_dense_shapes"]["scalar_collocation_c"], [25, 240])
        self.assertEqual(ledger["forbidden_dense_shapes"]["fitted_df_factor_b_tilde"], [25, 180])
        self.assertEqual(ledger["forbidden_dense_shapes"]["vvvv_tensor"], [5, 5, 5, 5])
        self.assertEqual(ledger["fp64_bytes"]["df_source_panel"], 5 * 5 * 31 * 8)
        self.assertEqual(
            ledger["fp64_element_shapes"]["partial_df_y_endpoint_live"], 5 * 5 * 17
        )
        self.assertNotIn("partial_cross_panel_live", ledger["fp64_element_shapes"])
        self.assertGreater(
            ledger["peak_fp64_bytes_estimate"]["audit_api_all_returned_outputs_plus_largest_term"],
            0,
        )
        self.assertEqual(
            ledger["contraction_leading_flops"]["exact_df"],
            4 * 5 * 5 * 5 * 5 * 5 * 180,
        )
        self.assertEqual(
            ledger["contraction_leading_flops"]["both_partial_thc_cross_terms"],
            2 * 5 * 5 * 240 * 180 + 12 * 5 * 5 * 240 * 5 * 5,
        )
