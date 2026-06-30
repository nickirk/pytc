"""Unit tests for isdf_decompose fixed_pivots override path (pytc/df.py).

Covers the still-live ``fixed_pivots`` kwarg: native pivots fed back as
fixed_pivots must reproduce bit-identical phi_piv/grad_phi_piv columns and
agree to double precision on xi_phi / xi_grad.  CPU-only so CI runs these
without a GPU.
"""
import unittest

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def _make_arrays(n_orb=4, n_grid=40, seed=7):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(n_orb, n_grid)).astype(np.float64)
    grad_phi = rng.normal(size=(n_orb, n_grid, 3)).astype(np.float64)
    return phi, grad_phi


def _isdf(phi, grad_phi, n_rank_phi=6, n_rank_grad=6, fixed_pivots=None):
    from pytc.df import isdf_decompose
    return isdf_decompose(
        phi, grad_phi, n_rank_phi, n_rank_grad,
        is_incore=True,
        fixed_pivots=fixed_pivots,
    )


class TestFixedPivots(unittest.TestCase):

    def test_fixed_pivots_reproduces_native_pivots(self):
        """fixed_pivots=own_pivots must return the same pivot indices."""
        phi, grad_phi = _make_arrays()
        *_, pivots_nat, _ = _isdf(phi, grad_phi)
        native = np.asarray(pivots_nat)

        *_, pivots_fix, _ = _isdf(phi, grad_phi, fixed_pivots=native)
        np.testing.assert_array_equal(np.asarray(pivots_fix), native)

    def test_fixed_pivots_phi_piv_bit_identical(self):
        """phi_piv = phi[:, pivots]: identical pivots -> identical column slices."""
        phi, grad_phi = _make_arrays()
        phi_piv_nat, _, gpiv_nat, _, pivots_nat, _ = _isdf(phi, grad_phi)
        native = np.asarray(pivots_nat)

        phi_piv_fix, _, gpiv_fix, _, _, _ = _isdf(phi, grad_phi, fixed_pivots=native)
        np.testing.assert_array_equal(np.asarray(phi_piv_fix), np.asarray(phi_piv_nat))
        np.testing.assert_array_equal(np.asarray(gpiv_fix), np.asarray(gpiv_nat))

    def test_fixed_pivots_xi_agrees_to_double_precision(self):
        """Same pivots -> same normal-equation system -> xi_phi/xi_grad agree to 1e-12."""
        phi, grad_phi = _make_arrays()
        _, xi_nat, _, xig_nat, pivots_nat, _ = _isdf(phi, grad_phi)
        native = np.asarray(pivots_nat)

        _, xi_fix, _, xig_fix, _, _ = _isdf(phi, grad_phi, fixed_pivots=native)
        np.testing.assert_allclose(np.asarray(xi_fix), np.asarray(xi_nat),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(xig_fix), np.asarray(xig_nat),
                                   rtol=0, atol=1e-12)

    def test_fixed_pivots_shapes_and_finiteness(self):
        """Outputs have expected shapes and are finite when fixed_pivots is used."""
        n_orb, n_grid = 4, 40
        phi, grad_phi = _make_arrays(n_orb=n_orb, n_grid=n_grid)
        n_rank = 5
        *_, pivots_nat, _ = _isdf(phi, grad_phi, n_rank_phi=n_rank, n_rank_grad=n_rank)
        native = np.asarray(pivots_nat)
        n_fused = len(native)

        phi_piv, xi, gpiv, xig, pivots, _ = _isdf(
            phi, grad_phi, n_rank_phi=n_rank, n_rank_grad=n_rank, fixed_pivots=native
        )
        self.assertEqual(np.asarray(phi_piv).shape, (n_orb, n_fused))
        self.assertEqual(np.asarray(xi).shape, (n_fused, n_grid))
        self.assertEqual(np.asarray(gpiv).shape, (n_orb, n_fused, 3))
        self.assertEqual(np.asarray(xig).shape, (n_fused, n_grid, 3))
        self.assertTrue(np.all(np.isfinite(np.asarray(xi))))
        self.assertTrue(np.all(np.isfinite(np.asarray(xig))))

    def test_fixed_pivots_validation_non_1d_raises(self):
        from pytc.df import isdf_decompose
        phi, grad_phi = _make_arrays()
        with self.assertRaises(ValueError):
            isdf_decompose(phi, grad_phi, 4, 4, is_incore=True,
                           fixed_pivots=np.array([[0, 1], [2, 3]]))

    def test_fixed_pivots_validation_duplicates_raise(self):
        from pytc.df import isdf_decompose
        phi, grad_phi = _make_arrays()
        with self.assertRaises(ValueError):
            isdf_decompose(phi, grad_phi, 4, 4, is_incore=True,
                           fixed_pivots=np.array([0, 1, 1, 2]))

    def test_fixed_pivots_validation_out_of_range_raises(self):
        from pytc.df import isdf_decompose
        phi, grad_phi = _make_arrays()
        n_grid = phi.shape[1]
        with self.assertRaises(ValueError):
            isdf_decompose(phi, grad_phi, 4, 4, is_incore=True,
                           fixed_pivots=np.array([0, 1, n_grid]))


if __name__ == "__main__":
    unittest.main()
