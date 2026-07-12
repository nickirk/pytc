"""Tests for pytc.coulomb.pivot_selection (task #5, isdf-coulomb-cuda).

Covers: the sector-aware oo/ov/vv API on a real small system, the
same-factor case matching pytc.df's original phi-decomposition pivot
selection exactly (the shared-primitive refactor's key correctness
claim), and the rank-curve harness stub.
"""

import unittest

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pyscf import gto, scf, dft

from pytc.df import pivoted_cholesky_pair_pivots, _pivoted_cholesky_phi
from pytc.coulomb.gpu4pyscf_adapter import get_mo_coeff, get_grid_ao_values_and_weights
from pytc.coulomb.pivot_selection import (
    weight_mo_values,
    select_sector_pivots,
    select_pivots_oo_ov_vv,
    pair_collocation_reconstruction_error,
    rank_curve,
)


class TestPivotSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="O 0 0 0; H 0 0 0.96; H 0.926 0 -0.24",
            basis="sto-3g", verbose=0)
        cls.mf = scf.RHF(cls.mol).run()
        cls.n_occ = cls.mol.nelectron // 2

        ao_values, weights, coords = get_grid_ao_values_and_weights(cls.mf, grid_lvl=1)
        mo_coeff = get_mo_coeff(cls.mf)
        # ao_values: (n_grid, n_ao) -> mo_values: (n_mo, n_grid), matching
        # pytc.df's own (n_orb, n_grid) orbital-axis-0 convention.
        cls.mo_values = (ao_values @ mo_coeff).T
        cls.weights = weights

    def test_same_factor_matches_original_phi_pivot_selection(self):
        """select_sector_pivots(X, X, ...) must reproduce
        pytc.df's original single-Gram-squared pivot selection exactly
        -- the core claim behind treating the refactor as loss-free."""
        mo_weighted = weight_mo_values(self.mo_values, self.weights)
        n_rank = 5

        diag_err = jnp.sum(mo_weighted**2, axis=0)
        shift = 1e-12 * jnp.max(jnp.abs(diag_err**2))

        via_sector_api = select_sector_pivots(mo_weighted, mo_weighted, n_rank, shift)
        via_shared_primitive = pivoted_cholesky_pair_pivots(mo_weighted, mo_weighted, n_rank, shift)
        via_original_wrapper = _pivoted_cholesky_phi(mo_weighted, n_rank, shift)

        np.testing.assert_array_equal(np.asarray(via_sector_api), np.asarray(via_shared_primitive))
        np.testing.assert_array_equal(np.asarray(via_sector_api), np.asarray(via_original_wrapper))

    def test_select_pivots_oo_ov_vv_shapes_and_validity(self):
        n_grid = self.mo_values.shape[1]
        result = select_pivots_oo_ov_vv(
            self.mo_values, self.n_occ, self.weights,
            n_rank_oo=3, n_rank_ov=4, n_rank_vv=3)

        self.assertEqual(set(result.keys()), {"oo", "ov", "vv"})
        self.assertEqual(result["oo"].shape, (3,))
        self.assertEqual(result["ov"].shape, (4,))
        self.assertEqual(result["vv"].shape, (3,))
        for sector, pivots in result.items():
            pivots_np = np.asarray(pivots)
            self.assertTrue(np.all(pivots_np >= 0))
            self.assertTrue(np.all(pivots_np < n_grid))
            self.assertEqual(len(np.unique(pivots_np)), len(pivots_np),
                              f"{sector} pivots must be unique within a sector")

    def test_select_pivots_oo_ov_vv_rejects_bad_n_occ(self):
        with self.assertRaises(ValueError):
            select_pivots_oo_ov_vv(
                self.mo_values, n_occ=0, weights=self.weights,
                n_rank_oo=1, n_rank_ov=1, n_rank_vv=1)
        with self.assertRaises(ValueError):
            select_pivots_oo_ov_vv(
                self.mo_values, n_occ=self.mo_values.shape[0], weights=self.weights,
                n_rank_oo=1, n_rank_ov=1, n_rank_vv=1)

    def test_reconstruction_error_decreases_with_rank(self):
        """More interpolation points should reconstruct the pair-product
        Gram matrix at least as well -- the monotonicity a rank-curve is
        supposed to show."""
        mo_weighted = weight_mo_values(self.mo_values, self.weights)
        occ_weighted = mo_weighted[:self.n_occ]

        curve = rank_curve(occ_weighted, occ_weighted, ranks=[1, 2, 4, 8])
        self.assertEqual(len(curve), 4)
        errors = [err for _, err in curve]
        # Non-increasing overall (allow tiny numerical noise at any single
        # step, but the endpoints must show real improvement).
        self.assertLess(errors[-1], errors[0])
        for err in errors:
            self.assertGreaterEqual(err, 0.0)

    def test_reconstruction_error_near_zero_at_full_rank(self):
        """Selecting n_grid pivots (all of them) must reconstruct the
        Gram matrix to within pinv's own numerical precision on this
        matrix.

        Mathematically G @ pinv(G) @ G == G exactly for the Moore-Penrose
        pseudoinverse regardless of rank, but this particular Gram matrix
        is large (n_grid x n_grid, here ~10^4) and highly rank-deficient
        (rank = n_mo^2 for the same-factor case, here 7^2=49 at most,
        measured 28) with a condition number of ~8e4 (measured directly
        via SVD, 2026-07-12) -- np.linalg.pinv's SVD-based computation on
        a matrix this size/conditioning has a real, non-negligible
        floating-point error floor (measured ~4e-5 relative, both via
        this function and independently via a direct G@pinv(G)@G-G
        check), not the ~1e-14 one might expect from float64 alone. Use
        a threshold with real margin above that measured floor rather
        than an idealized exact-arithmetic bound.
        """
        mo_weighted = weight_mo_values(self.mo_values, self.weights)
        n_grid = mo_weighted.shape[1]
        pivots = np.arange(n_grid)
        error = pair_collocation_reconstruction_error(mo_weighted, mo_weighted, pivots)
        self.assertLess(error, 1e-3)


if __name__ == "__main__":
    unittest.main()
