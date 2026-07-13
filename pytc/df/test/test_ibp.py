"""Direct tests for pytc.df.ibp's canonical single-IBP primitives (task #2,
#proj-isdf-ibp-coulomb). Exercises the primitives at their new home
independently of the atom-centered benchmark's re-export, so a future
change to the benchmark's import path cannot silently stop testing the
canonical implementation.

Molecular end-to-end regression coverage (H2O atom-centered grid vs exact
4-center/analytic-DF references) stays in
pytc/test/test_atom_centered_single_ibp.py, which already re-runs
unchanged against pytc.df.ibp via the benchmark's import.
"""

import unittest

import numpy as np

from pytc.df.ibp import (
    naive_coulomb_kernel,
    kernel,
)


class TestAtomCenteredSingleIBPCore(unittest.TestCase):
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
            actual, coincident = kernel(
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
        actual, coincident = naive_coulomb_kernel(
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
        actual, coincident = kernel(
            grad, rho, x, wx, coords_right=y, weights_right=wy,
            eval_block_size=3, source_block_size=5,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        self.assertEqual(coincident, 0)

    def test_block_sizes_do_not_change_result(self):
        coords, weights, density, gradient = self._case(np.float64)
        small, _ = kernel(
            gradient, density, coords, weights,
            eval_block_size=2, source_block_size=2,
        )
        full, _ = kernel(
            gradient, density, coords, weights,
            eval_block_size=100, source_block_size=100,
        )
        np.testing.assert_allclose(small, full, atol=3e-12, rtol=3e-12)

    def test_block_sizes_do_not_change_result_offdiagonal(self):
        coords, weights, density, _ = self._case(np.complex128)
        small, _ = naive_coulomb_kernel(
            density, density, coords, weights,
            eval_block_size=2, source_block_size=2,
        )
        full, _ = naive_coulomb_kernel(
            density, density, coords, weights,
            eval_block_size=100, source_block_size=100,
        )
        np.testing.assert_allclose(small, full, atol=3e-12, rtol=3e-12)

    def test_rejects_malformed_inputs(self):
        coords, weights, density, gradient = self._case(np.float64)
        with self.assertRaises(ValueError):
            kernel(gradient[:, :2], density, coords, weights)
        with self.assertRaises(ValueError):
            kernel(
                gradient.astype(np.float32), density, coords, weights
            )
        with self.assertRaises(ValueError):
            naive_coulomb_kernel(
                density, density, coords, weights, eval_block_size=0
            )
        with self.assertRaises(ValueError):
            # coords shape mismatch: wrong number of grid points
            kernel(
                gradient, density, coords[:-1], weights
            )
        with self.assertRaises(ValueError):
            # non-finite coordinates
            bad_coords = coords.copy()
            bad_coords[0, 0] = np.nan
            kernel(gradient, density, bad_coords, weights)
        with self.assertRaises(ValueError):
            # dtype mismatch between gradient and density_right
            kernel(
                gradient.astype(np.complex128), density, coords, weights
            )

    def test_explicit_coincident_point_rhat_zero_convention(self):
        """A single point coincident with itself must contribute rhat=0,
        not NaN/inf -- and coincident_pairs must report it."""
        coords = np.zeros((1, 3))
        weights = np.ones(1)
        density = np.ones((1, 1))
        gradient = np.ones((1, 3, 1))
        result, coincident = kernel(
            gradient, density, coords, weights,
        )
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, np.zeros((1, 1)))
        self.assertEqual(coincident, 1)


if __name__ == "__main__":
    unittest.main()
