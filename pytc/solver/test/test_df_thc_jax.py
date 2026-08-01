import unittest

import numpy as np

from pytc.df.thc import (
    df_sandwiches_jax,
    fit_lsthc_jax,
)
from pytc.solver.test.thc_scalable_oracle import (
    direct_df_sandwiches_panelled,
    fit_panelled_lsthc,
)


class TestJaxFitAndSandwich(unittest.TestCase):

    def test_jax_fit_and_sandwich_match_panelled_oracle(self):
        rng = np.random.default_rng(4)
        p = rng.normal(size=(4, 3))
        b = rng.normal(size=(4, 4, 5))
        t2 = rng.normal(size=(2, 2, 4, 4))
        oracle_fit = fit_panelled_lsthc(p, b, rcond=1e-12, virtual_panel=2)
        jax_fit = fit_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
        self.assertLess(
            np.max(np.abs(oracle_fit.y - np.asarray(jax_fit.y))), 1e-12)
        oracle = direct_df_sandwiches_panelled(
            b, oracle_fit, t2, rank_panel=2, aux_panel=3)
        actual = df_sandwiches_jax(
            b, jax_fit, t2, rank_panel=2, aux_panel=3)
        self.assertLess(
            np.max(np.abs(oracle.robust - np.asarray(actual.robust))), 1e-12)

    def test_jax_fit_rejects_invalid_rcond(self):
        p = np.eye(2)
        b = np.ones((2, 2, 1))
        for rcond in (0.0, -1.0, 1.1, np.nan, np.inf):
            with self.subTest(rcond=rcond):
                with self.assertRaisesRegex(ValueError, "rcond"):
                    fit_lsthc_jax(
                        p, b, rcond=rcond, virtual_panel=1)

    def test_jax_fit_keeps_b_host_resident(self):
        import pytc.df.thc as mod

        uploads = []
        real = mod._as_fp64_jax

        def recording(name, value, ndim):
            uploads.append((name, np.shape(value)))
            return real(name, value, ndim)

        mod._as_fp64_jax = recording
        try:
            rng = np.random.default_rng(5)
            p = rng.normal(size=(5, 3))
            b = rng.normal(size=(5, 5, 4))
            fit_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
        finally:
            mod._as_fp64_jax = real

        self.assertNotIn("b", [name for name, _ in uploads])
        panel_shapes = [shape for name, shape in uploads if name == "b_panel"]
        self.assertEqual(panel_shapes, [(2, 5, 4), (2, 5, 4), (1, 5, 4)])

    def test_sandwich_keeps_b_host_resident(self):
        # The full 3-index B block must never be cast onto the device
        # wholesale -- the panel loops read it one aux slice at a time
        # from the host.
        import pytc.df.thc as mod

        uploaded = []
        real = mod._as_fp64_jax

        def recording(name, value, ndim):
            uploaded.append(name)
            return real(name, value, ndim)

        mod._as_fp64_jax = recording
        try:
            rng = np.random.default_rng(7)
            p = rng.normal(size=(4, 3))
            b = rng.normal(size=(4, 4, 5))
            t2 = rng.normal(size=(2, 2, 4, 4))
            jax_fit = fit_lsthc_jax(
                p, b, rcond=1e-12, virtual_panel=2)
            uploaded.clear()
            df_sandwiches_jax(
                b, jax_fit, t2, rank_panel=2, aux_panel=3)
        finally:
            mod._as_fp64_jax = real
        self.assertNotIn("b", uploaded)
        self.assertTrue({"t2", "p_virtual", "y"} <= set(uploaded))

    def test_exact_panel_cap_preserves_result(self):
        # PYTC_EXACT_PANEL_CAP_GB clamps the (nocc^2, nvir, nvir, q)
        # scratch; forcing q_step=1 must reproduce the wider-panel result.
        import os
        rng = np.random.default_rng(11)
        p = rng.normal(size=(4, 3))
        b = rng.normal(size=(4, 4, 5))
        t2 = rng.normal(size=(2, 2, 4, 4))
        jax_fit = fit_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
        wide = df_sandwiches_jax(
            b, jax_fit, t2, rank_panel=2, aux_panel=3)
        os.environ["PYTC_EXACT_PANEL_CAP_GB"] = str(512 / 1024 ** 3)
        try:
            clamped = df_sandwiches_jax(
                b, jax_fit, t2, rank_panel=2, aux_panel=3)
        finally:
            del os.environ["PYTC_EXACT_PANEL_CAP_GB"]
        self.assertLess(
            np.max(np.abs(np.asarray(wide.robust)
                          - np.asarray(clamped.robust))), 1e-12)


if __name__ == "__main__":
    unittest.main()
