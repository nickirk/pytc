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
        via_shared_primitive, _effective_rank = pivoted_cholesky_pair_pivots(mo_weighted, mo_weighted, n_rank, shift)
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
        occ_raw = self.mo_values[:self.n_occ]

        curve = rank_curve(occ_raw, occ_raw, self.weights, ranks=[1, 2, 4, 8])
        self.assertEqual(len(curve), 4)
        errors = [err for _, err, _ in curve]
        cond_S_values = [cond_s for _, _, cond_s in curve]
        for cond_s in cond_S_values:
            self.assertTrue(np.isfinite(cond_s))
            self.assertGreaterEqual(cond_s, 1.0)
        # Non-increasing overall (allow tiny numerical noise at any single
        # step, but the endpoints must show real improvement).
        self.assertLess(errors[-1], errors[0])
        for err in errors:
            self.assertGreaterEqual(err, 0.0)

    def test_reconstruction_error_near_zero_at_full_rank(self):
        """Selecting all pivots (n_grid of them) must reconstruct the
        Gram matrix to within pinv's own numerical precision on that
        matrix -- G @ pinv(G) @ G == G exactly for the Moore-Penrose
        pseudoinverse regardless of rank.

        Uses a small SYNTHETIC factor matrix (n_grid=60, n_feature=4),
        not the real ~10^4-grid-point H2O system: the real-system
        version of this test explicitly materializes an (n_grid,
        n_grid) Gram matrix and pinv's it, which is O(n_grid^3) and
        stalled a combined test run at that size (Alice's task #6
        review, 2026-07-12, item 5 -- CI-safety). This synthetic matrix
        reproduces the same rank-deficient, moderately-ill-conditioned
        shape (n_grid=60 >> rank<=n_feature^2=16) that makes the
        near-zero-but-not-exact-zero assertion meaningful, without the
        cubic cost.
        """
        rng = np.random.default_rng(0)
        n_grid, n_feature = 60, 4
        factor = jnp.asarray(rng.standard_normal((n_feature, n_grid)))
        pivots = np.arange(n_grid)
        error = pair_collocation_reconstruction_error(factor, factor, pivots)
        self.assertLess(error, 1e-8)

    def test_effective_rank_capped_at_true_pair_rank_h2o_ov(self):
        """Regression for Alice's re-review catch (2026-07-12, task #6
        blocker item 3): effective_rank must never exceed the sector's
        TRUE pair-space rank (n_occ*n_vir=95 for this H2O/cc-pVDZ ov
        sector) regardless of how much n_rank overshoots it. The
        original effective-rank accounting (counting diag_err[pivot] >=
        1e-12 against diag_err ITSELF, which includes the Tikhonov
        shift AND the deterministic tie-break ramp) gave an IMPOSSIBLE
        effective_rank=106 for a 95-column matrix at n_rank=120/300 --
        this exact H2O/cc-pVDZ ov reproduction is Alice's own repro
        case."""
        mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.926 0 -0.24", basis="cc-pvdz", verbose=0)
        mf = scf.RHF(mol).density_fit().run()
        n_occ = mol.nelectron // 2
        mo_coeff = get_mo_coeff(mf)
        ao_values, weights, coords = get_grid_ao_values_and_weights(mf, grid_lvl=2)
        mo_values = (ao_values @ mo_coeff).T
        mo_weighted = weight_mo_values(mo_values, weights)
        occ_weighted = mo_weighted[:n_occ]
        vir_weighted = mo_weighted[n_occ:]
        n_vir = vir_weighted.shape[0]
        n_pair = n_occ * n_vir  # 95

        diag_err = jnp.sum(occ_weighted**2, axis=0) * jnp.sum(vir_weighted**2, axis=0)
        shift = 1e-12 * jnp.max(jnp.abs(diag_err))

        effective_ranks = {}
        for n_rank in (90, 95, 120, 300):
            pivots, effective_rank = pivoted_cholesky_pair_pivots(occ_weighted, vir_weighted, n_rank, shift)
            effective_ranks[n_rank] = effective_rank
            self.assertLessEqual(
                effective_rank, n_pair,
                f"effective_rank={effective_rank} exceeds true pair rank "
                f"n_pair={n_pair} at n_rank={n_rank} -- impossible.")
            self.assertLessEqual(effective_rank, n_rank)
            # Still no duplicates regardless of over-rank padding (item 3's
            # first fix, unaffected by this second fix).
            self.assertEqual(len(np.unique(np.asarray(pivots))), n_rank)

        # Over-rank requests (120, 300) must both report the SAME
        # effective_rank -- the pair space's own true rank -- not a value
        # that drifts with how much padding was requested.
        self.assertEqual(effective_ranks[120], effective_ranks[300])
        self.assertEqual(effective_ranks[120], n_pair)
        # At n_rank == n_pair exactly, everything requested should be
        # genuinely effective.
        self.assertEqual(effective_ranks[95], n_pair)

    def test_select_sector_pivots_truncates_over_rank_request(self):
        """select_sector_pivots must not silently return residual-
        exhausted, numerically-arbitrary pivots as if they were
        production-quality interpolation points (Alice's task #6
        re-review, blocker item 3: 'surface it through
        select_sector_pivots and either slice to effective pivots or
        explicitly reject/stop over-rank requests')."""
        mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.926 0 -0.24", basis="cc-pvdz", verbose=0)
        mf = scf.RHF(mol).density_fit().run()
        n_occ = mol.nelectron // 2
        mo_coeff = get_mo_coeff(mf)
        ao_values, weights, coords = get_grid_ao_values_and_weights(mf, grid_lvl=2)
        mo_values = (ao_values @ mo_coeff).T
        mo_weighted = weight_mo_values(mo_values, weights)
        occ_weighted = mo_weighted[:n_occ]
        vir_weighted = mo_weighted[n_occ:]
        n_pair = n_occ * vir_weighted.shape[0]

        pivots = select_sector_pivots(occ_weighted, vir_weighted, 300)
        self.assertEqual(len(np.asarray(pivots)), n_pair,
                          "default on_over_rank='truncate' must slice to effective_rank, not return 300")

        with self.assertRaises(ValueError):
            select_sector_pivots(occ_weighted, vir_weighted, 300, on_over_rank="raise")

        with self.assertRaises(ValueError):
            select_sector_pivots(occ_weighted, vir_weighted, 5, on_over_rank="not-a-real-mode")

        # Not-over-rank requests are returned in full, untouched.
        pivots_small = select_sector_pivots(occ_weighted, vir_weighted, 10)
        self.assertEqual(len(np.asarray(pivots_small)), 10)


if __name__ == "__main__":
    unittest.main()
