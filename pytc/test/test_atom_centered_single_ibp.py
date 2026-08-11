"""Reference tests for the TC-style atom-centered single-IBP benchmark."""

import unittest

import numpy as np

from pytc.utils.atom_centered_single_ibp_benchmark import (
    AtomCenteredSingleIBPCase,
    atom_centered_direct_coulomb_core_offdiagonal,
    atom_centered_single_ibp_core,
    run_atom_centered_case,
)


class TestAtomCenteredSingleIBPBlockedOracles(unittest.TestCase):
    def _case(self, dtype):
        rng = np.random.default_rng(21)
        coords = rng.normal(size=(9, 3))
        weights = rng.normal(size=9)
        density = rng.normal(size=(4, 9))
        gradient = rng.normal(size=(3, 3, 9))
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            density = density + 1j * rng.normal(size=density.shape)
            gradient = gradient + 1j * rng.normal(size=gradient.shape)
        density = density.astype(dtype)
        gradient = gradient.astype(dtype)
        return coords, weights, density, gradient

    def test_single_ibp_matches_explicit_dense_real_and_complex(self):
        for dtype in (np.float64, np.complex128):
            coords, weights, density, gradient = self._case(dtype)
            diff = coords[:, None, :] - coords[None, :, :]
            radius = np.linalg.norm(diff, axis=-1)
            rhat = np.divide(
                diff, radius[..., None], out=np.zeros_like(diff),
                where=radius[..., None] != 0.0,
            )
            vector = np.einsum("nj,j,ijc->nci", density, weights, rhat)
            expected = -0.5 * np.einsum(
                "mci,nci,i->mn", gradient.conj(), vector, weights
            )
            actual, coincident = atom_centered_single_ibp_core(
                gradient, density, coords, weights,
                eval_block_size=4, source_block_size=3,
            )
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
            self.assertEqual(coincident, len(weights))

    def test_direct_offdiagonal_matches_explicit_dense(self):
        coords, weights, density, _ = self._case(np.complex128)
        diff = coords[:, None, :] - coords[None, :, :]
        radius = np.linalg.norm(diff, axis=-1)
        kernel = np.divide(1.0, radius, out=np.zeros_like(radius), where=radius != 0.0)
        expected = np.einsum(
            "mi,i,ij,nj,j->mn", density.conj(), weights, kernel, density, weights
        )
        actual, coincident = atom_centered_direct_coulomb_core_offdiagonal(
            density, density, coords, weights,
            eval_block_size=4, source_block_size=3,
        )
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
        self.assertEqual(coincident, len(weights))

    def test_cross_grid_matches_dense_and_has_no_coincident_points(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(7, 3))
        y = rng.normal(size=(8, 3)) + 0.123
        wx, wy = rng.normal(size=7), rng.normal(size=8)
        grad = rng.normal(size=(2, 3, 7))
        rho = rng.normal(size=(3, 8))
        diff = x[:, None, :] - y[None, :, :]
        radius = np.linalg.norm(diff, axis=-1)
        rhat = diff / radius[..., None]
        vector = np.einsum("nj,j,ijc->nci", rho, wy, rhat)
        expected = -0.5 * np.einsum("mci,nci,i->mn", grad, vector, wx)
        actual, coincident = atom_centered_single_ibp_core(
            grad, rho, x, wx, coords_right=y, weights_right=wy,
            eval_block_size=3, source_block_size=5,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        self.assertEqual(coincident, 0)

    def test_block_sizes_do_not_change_result(self):
        coords, weights, density, gradient = self._case(np.float64)
        small, _ = atom_centered_single_ibp_core(
            gradient, density, coords, weights,
            eval_block_size=2, source_block_size=2,
        )
        full, _ = atom_centered_single_ibp_core(
            gradient, density, coords, weights,
            eval_block_size=100, source_block_size=100,
        )
        np.testing.assert_allclose(small, full, atol=3e-12, rtol=3e-12)

    def test_rejects_malformed_inputs(self):
        coords, weights, density, gradient = self._case(np.float64)
        with self.assertRaises(ValueError):
            atom_centered_single_ibp_core(gradient[:, :2], density, coords, weights)
        with self.assertRaises(ValueError):
            atom_centered_single_ibp_core(
                gradient.astype(np.float32), density, coords, weights
            )
        with self.assertRaises(ValueError):
            atom_centered_direct_coulomb_core_offdiagonal(
                density, density, coords, weights, eval_block_size=0
            )


class TestAtomCenteredSingleIBPMolecularRegression(unittest.TestCase):
    def test_h2o_level0_full_ovov_beats_naive_direct_grid_sum(self):
        result = run_atom_centered_case(AtomCenteredSingleIBPCase(
            grid_level=0, eval_block_size=128, source_block_size=1024,
        ))
        self.assertEqual(result["n_grid"], 2328)
        self.assertEqual(result["n_pairs"], 95)
        self.assertLess(result["single_ibp"]["relative_eri_error_exact_4c"], 0.02)
        self.assertGreater(
            result["direct_1_over_r_offdiagonal_only"]["relative_eri_error_exact_4c"], 0.20
        )
        self.assertLess(abs(result["single_ibp"]["mp2_delta_exact_4c_mha"]), 1.0)
        self.assertGreater(
            result["analytic_df_approximation"]["relative_eri_error_vs_exact_4c"], 0.02
        )
        # At least the same-grid diagonal is coincident; atom-centered grids
        # may additionally contain duplicate molecular points from different
        # atomic grids after partitioning.
        self.assertGreaterEqual(result["single_ibp_coincident_pairs"], result["n_grid"])


if __name__ == "__main__":
    unittest.main()
