"""FP64 numerical gates for the standalone Phase-B robust DF/THC oracle."""

from __future__ import annotations

import unittest

import numpy as np

from pytc.solver import robust_df_thc


class TestRobustDFTHCOracle(unittest.TestCase):
    """The algebra is explicit because this is not yet a production path."""

    def setUp(self):
        rng = np.random.default_rng(20260715)
        self.nocc = 2
        self.nvir = 5
        self.naux = 9
        self.rank = 4
        self.b = rng.normal(size=(self.nvir, self.nvir, self.naux)).astype(np.float64)
        self.p = rng.normal(size=(self.nvir, self.rank)).astype(np.float64)
        t2_raw = rng.normal(
            size=(self.nocc, self.nocc, self.nvir, self.nvir)
        ).astype(np.float64)
        self.t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))
        self.model = robust_df_thc.build_robust_df_thc_model(self.b, self.p)

    def test_virtual_pair_and_scalar_collocation_orders_are_ac(self):
        b_pair = robust_df_thc.virtual_pair_df_matrix(self.b)
        collocation = robust_df_thc.scalar_pair_collocation(self.p)
        for a, c in ((0, 0), (1, 3), (4, 2)):
            row = a * self.nvir + c
            np.testing.assert_array_equal(b_pair[row], self.b[a, c])
            np.testing.assert_allclose(collocation[row], self.p[a] * self.p[c])

    def test_exact_lsthc_and_robust_pair_metrics(self):
        expected_exact = self.model.b_pair @ self.model.b_pair.T
        expected_lsthc = self.model.b_tilde @ self.model.b_tilde.T
        expected_robust = (
            self.model.b_tilde @ self.model.b_pair.T
            + self.model.b_pair @ self.model.b_tilde.T
            - expected_lsthc
        )
        np.testing.assert_allclose(self.model.exact_metric, expected_exact, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(self.model.lsthc_metric, expected_lsthc, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(self.model.robust_metric, expected_robust, rtol=1e-13, atol=1e-13)

    def test_robust_pair_residual_has_the_signed_delta_delta_order(self):
        expected = self.model.delta_b @ self.model.delta_b.T
        np.testing.assert_allclose(
            self.model.exact_minus_robust, expected, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(self.model.exact_minus_robust, expected.T, rtol=1e-13, atol=1e-13)

    def test_direct_df_vvvv_t2_sandwiches_match_dense_oracle_and_delta_residual(self):
        b_tilde = self.model.b_tilde.reshape(self.nvir, self.nvir, self.naux)
        delta_b = self.model.delta_b.reshape(self.nvir, self.nvir, self.naux)
        dense_exact = np.einsum("acq,bdq,ijcd->ijab", self.b, self.b, self.t2, optimize=True)
        dense_lsthc = np.einsum(
            "acq,bdq,ijcd->ijab", b_tilde, b_tilde, self.t2, optimize=True
        )
        dense_robust = (
            np.einsum("acq,bdq,ijcd->ijab", b_tilde, self.b, self.t2, optimize=True)
            + np.einsum("acq,bdq,ijcd->ijab", self.b, b_tilde, self.t2, optimize=True)
            - dense_lsthc
        )
        expected_residual = np.einsum(
            "acq,bdq,ijcd->ijab", delta_b, delta_b, self.t2, optimize=True
        )
        actual = robust_df_thc.direct_df_sandwiches(self.model, self.t2)
        np.testing.assert_allclose(actual.exact, dense_exact, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.lsthc, dense_lsthc, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.robust, dense_robust, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.exact_minus_robust, expected_residual, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.delta_delta, expected_residual, rtol=1e-12, atol=1e-12)

    def test_rejects_non_fp64_inputs(self):
        with self.assertRaisesRegex(ValueError, "float64"):
            robust_df_thc.build_robust_df_thc_model(self.b.astype(np.float32), self.p)
