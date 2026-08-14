"""Regression tests for the production JAX DF/THC path."""

from __future__ import annotations

import os
import unittest

import numpy as np

from pytc.df.thc import df_sandwiches_jax, fit_lsthc_jax


class TestJaxFitAndSandwich(unittest.TestCase):
    def test_frozen_fit_and_sandwich_values(self):
        # These values were frozen only after the former independent NumPy
        # implementation and the JAX path agreed.  Keeping the values avoids
        # maintaining a second implementation that could acquire the same bug.
        p = np.array([[1.0, 0.2], [0.3, 1.1]], dtype=np.float64)
        b = np.array(
            [
                [[0.7, -0.2], [0.1, 0.5]],
                [[0.1, 0.5], [-0.4, 0.9]],
            ],
            dtype=np.float64,
        )
        t2 = np.array([[[[0.6, -0.3], [-0.3, 0.8]]]], dtype=np.float64)

        fit = fit_lsthc_jax(p, b, rcond=1e-12, virtual_panel=1)
        np.testing.assert_allclose(
            np.asarray(fit.gram),
            [[1.1881, 0.2809], [0.2809, 1.5625]],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            np.asarray(fit.cross),
            [[0.724, 0.181], [-0.412, 1.301]],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            np.asarray(fit.y),
            [
                [0.7015357467164697, -0.04649132661180869],
                [-0.3897992904017, 0.8409980247329644],
            ],
            rtol=1e-12,
            atol=1e-12,
        )

        actual = df_sandwiches_jax(
            b, fit, t2, rank_panel=1, aux_panel=1
        )
        expected = {
            "exact": [[0.544, 0.37], [0.37, 0.686]],
            "fit_left_df_right": [
                [0.33346951442132167, 0.1769487647559937],
                [0.5217276378240234, 0.7482119995132045],
            ],
            "df_left_fit_right": [
                [0.33346951442132167, 0.5217276378240234],
                [0.1769487647559937, 0.7482119995132045],
            ],
            "full_thc": [
                [0.2682587965604809, 0.22247297741235816],
                [0.22247297741235816, 0.9085674005761541],
            ],
            "robust": [
                [0.39868023228216243, 0.476203425167659],
                [0.476203425167659, 0.5878565984502548],
            ],
        }
        for name, value in expected.items():
            np.testing.assert_allclose(
                np.asarray(getattr(actual, name))[0, 0],
                value,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_jax_fit_rejects_invalid_rcond(self):
        p = np.eye(2)
        b = np.ones((2, 2, 1))
        for rcond in (0.0, -1.0, 1.1, np.nan, np.inf):
            with self.subTest(rcond=rcond):
                with self.assertRaisesRegex(ValueError, "rcond"):
                    fit_lsthc_jax(
                        p, b, rcond=rcond, virtual_panel=1
                    )

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
            fit = fit_lsthc_jax(
                p, b, rcond=1e-12, virtual_panel=2
            )
            uploaded.clear()
            df_sandwiches_jax(
                b, fit, t2, rank_panel=2, aux_panel=3
            )
        finally:
            mod._as_fp64_jax = real
        self.assertNotIn("b", uploaded)
        self.assertTrue({"t2", "p_virtual", "y"} <= set(uploaded))

    def test_exact_panel_cap_preserves_result(self):
        rng = np.random.default_rng(11)
        p = rng.normal(size=(4, 3))
        b = rng.normal(size=(4, 4, 5))
        t2 = rng.normal(size=(2, 2, 4, 4))
        fit = fit_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
        wide = df_sandwiches_jax(
            b, fit, t2, rank_panel=2, aux_panel=3
        )
        os.environ["PYTC_EXACT_PANEL_CAP_GB"] = str(512 / 1024**3)
        try:
            clamped = df_sandwiches_jax(
                b, fit, t2, rank_panel=2, aux_panel=3
            )
        finally:
            del os.environ["PYTC_EXACT_PANEL_CAP_GB"]
        np.testing.assert_allclose(
            np.asarray(wide.robust),
            np.asarray(clamped.robust),
            rtol=1e-12,
            atol=1e-12,
        )
