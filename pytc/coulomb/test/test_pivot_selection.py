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
        """pivoted_cholesky_pair_pivots's default legacy path
        (track_effective_rank=False) must reproduce pytc.df's original
        single-Gram-squared pivot selection exactly -- the core claim
        behind treating the P1b refactor as loss-free for TC's own
        production call sites.

        select_sector_pivots (the Coulomb path) is NOT compared here
        post-round-3 (Felix's architect ranking, Alice's re-review,
        2026-07-12): it always opts into track_effective_rank=True,
        which uses prefix-forced selection -- a deliberately DIFFERENT
        algorithm branch from the legacy path, not required to be
        bit-identical to it (see test_select_sector_pivots_* for its own
        correctness tests, and the shared primitive's docstring for why
        the two branches diverge)."""
        mo_weighted = weight_mo_values(self.mo_values, self.weights)
        n_rank = 5

        diag_err = jnp.sum(mo_weighted**2, axis=0)
        shift = 1e-12 * jnp.max(jnp.abs(diag_err**2))

        via_shared_primitive, effective_rank = pivoted_cholesky_pair_pivots(mo_weighted, mo_weighted, n_rank, shift)
        via_original_wrapper = _pivoted_cholesky_phi(mo_weighted, n_rank, shift)

        self.assertIsNone(effective_rank, "default track_effective_rank=False must not compute effective_rank")
        np.testing.assert_array_equal(np.asarray(via_shared_primitive), np.asarray(via_original_wrapper))

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
        case.

        Uses the library's DEFAULT effective_rank_rtol (1e-6, not the
        1e-10 used in the small synthetic prefix test) -- this system's
        raw residual beyond the true rank decays SLOWLY, not to a sharp
        zero (see pivoted_cholesky_pair_pivots's docstring for the
        measured rtol sweep); 1e-6 gives a safe, if slightly
        conservative, effective_rank=94 (<=95, a 1-point margin), while
        1e-10 would give an impossible 130 (verified: the ORIGINAL bug's
        failure mode, still reproducible at the wrong rtol).
        """
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
            pivots, effective_rank = pivoted_cholesky_pair_pivots(
                occ_weighted, vir_weighted, n_rank, shift, track_effective_rank=True)
            effective_ranks[n_rank] = effective_rank
            self.assertLessEqual(
                effective_rank, n_pair,
                f"effective_rank={effective_rank} exceeds true pair rank "
                f"n_pair={n_pair} at n_rank={n_rank} -- impossible.")
            self.assertLessEqual(effective_rank, n_rank)
            # Still no duplicates regardless of over-rank padding (item 3's
            # first fix, unaffected by this second fix).
            self.assertEqual(len(np.unique(np.asarray(pivots))), n_rank)

        # Below the true rank (90 < 95), nothing has failed yet -- every
        # requested pivot is genuinely effective.
        self.assertEqual(effective_ranks[90], 90)
        # At and beyond the true rank, effective_rank settles at the SAME
        # value regardless of how much further padding was requested --
        # selection at step k depends only on steps 0..k-1, not on the
        # total n_rank, so the streak-break point is deterministic.
        self.assertEqual(effective_ranks[95], effective_ranks[120])
        self.assertEqual(effective_ranks[120], effective_ranks[300])
        self.assertGreaterEqual(effective_ranks[120], 90)  # safely close to n_pair, not collapsed

    def test_effective_pivots_form_valid_prefix_synthetic(self):
        """Regression for Alice's re-review catch (2026-07-12, task #6
        blocker item 3, round 3): pivots[:effective_rank] must be a
        VALID PREFIX -- selection must not pick a residual-exhausted,
        tie-break-favored candidate BEFORE a genuinely effective one.

        Alice's exact synthetic repro (n_grid=1000): one strong signal
        (raw diagonal=1), one weak-but-effective signal (raw
        diagonal=2e-10, above an explicit 1e-10 rtol threshold -- tight
        on purpose to demonstrate the mechanism; the library DEFAULT is
        1e-6, calibrated instead for a realistic ill-conditioned
        production system's noise floor, see compute_Z's docstring),
        everywhere else ineffective noise (raw diagonal=5e-11, below
        threshold). The pre-round-3 primitive selected [0, 999, 998]
        with effective_rank=1, completely missing the real signal at
        index 1 -- the unnormalized tie-break ramp's span (growing with
        n_grid) exceeded the effective-rank threshold and pulled
        high-index noise candidates ahead of the genuine weak signal.

        Uses an ORTHOGONAL (diagonal-factor) construction --
        factor_p=diag(sqrt(raw_diag)), factor_q=identity -- so each grid
        index's pair-product signal is independent of every other's
        (selecting one pivot doesn't perturb any other index's raw
        residual at all). A naive same-shape-everywhere construction
        (e.g. factor_q all-ones) accidentally correlates every index
        through the shared factor, so selecting the strong pivot
        explains away the "independent" weak one too -- not a valid
        test of independent-signal detection.
        """
        n_grid = 1000
        raw_diag = jnp.full((n_grid,), 5e-11)
        raw_diag = raw_diag.at[0].set(1.0)
        raw_diag = raw_diag.at[1].set(2e-10)
        factor_p = jnp.diag(jnp.sqrt(raw_diag))
        factor_q = jnp.eye(n_grid)

        pivots, effective_rank = pivoted_cholesky_pair_pivots(
            factor_p, factor_q, 3, shift=0.0, track_effective_rank=True,
            effective_rank_rtol=1e-10)
        pivots_np = np.asarray(pivots)

        self.assertEqual(effective_rank, 2)
        self.assertEqual(set(pivots_np[:2].tolist()), {0, 1},
                          "effective prefix must contain both real signals, not noise padding")

    def test_effective_rank_invariant_under_global_rescale(self):
        """Regression for Alice's 3rd re-review catch (2026-07-12, task
        #6 blocker item 3, round 4): the tracked branch's internal
        numerical-safety guards (is_small/is_small_raw) used to be
        ABSOLUTE (``pivot_val < 1e-12``) while effective_rank itself is
        defined by the SCALE-RELATIVE ``effective_rank_rtol * max_diag``
        -- rescaling the factor matrices by a positive constant (same
        mathematical row space, Gram merely scales) shifted max_diag by
        the same factor but left the guards fixed, so a small enough
        global rescale (e.g. 1e-4) could push EVERY pivot_val below the
        absolute 1e-12 floor, permanently disabling the Cholesky
        deflation update -- a duplicate/correlated column was then never
        deflated after its twin was selected, and got counted as a
        SECOND independent effective signal instead of being recognized
        as redundant.

        Alice's exact repro: one-feature rank-1 factors with two
        identical nonzero grid columns (indices 0 and 2) -- analytically
        rank 1 (a single feature can only span a 1-dimensional pair
        space), so effective_rank must be 1 regardless of global scale,
        not 2.
        """
        factor_p = jnp.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        factor_q = jnp.array([[1.0, 0.0, 1.0, 0.0, 0.0]])

        results = {}
        for scale in (1.0, 1e-4):
            fp = factor_p * scale
            fq = factor_q * scale
            diag_err = jnp.sum(fp**2, axis=0) * jnp.sum(fq**2, axis=0)
            shift = 1e-12 * jnp.max(jnp.abs(diag_err))
            pivots, effective_rank = pivoted_cholesky_pair_pivots(
                fp, fq, 2, shift, track_effective_rank=True)
            results[scale] = (np.asarray(pivots).tolist(), effective_rank)
            self.assertEqual(effective_rank, 1,
                              f"scale={scale}: analytically rank-1 problem must report effective_rank=1")

        # Same prefix (up to the effective_rank=1 point) regardless of scale.
        self.assertEqual(results[1.0][0][0], results[1e-4][0][0])

    def test_effective_rank_guard_ordering_invariant_float32(self):
        """Regression for Alice's 4th re-review catch (2026-07-12, task
        #6 blocker item 3, round 5): the numerical smallness cutoff
        (small_eps, guarding rsqrt) must NEVER exceed the scientific
        significance cutoff (eff_tol = effective_rank_rtol * max_diag),
        or a pivot judged scientifically effective gets SKIPPED by the
        official L update (never deflated) -- its duplicate/correlated
        twin then looks like fresh signal, and the prefix latch closes
        too early, silently dropping later genuinely-independent signal.

        In float32, ``100 * eps`` is ~1.2e-5 -- LARGER than the default
        ``effective_rank_rtol=1e-6`` -- so this failure mode is real at
        the library's own default, not just at some exotic rtol choice.
        The test module's module-level ``jax_enable_x64`` does NOT hide
        this (unlike the earlier float64-only bugs): explicit
        ``dtype=jnp.float32`` factor/shift arrays force float32
        computation regardless of the ambient x64 flag.

        Construction: three orthogonal features on 4 grid points -- a
        strong signal (index 0, raw diag 1), a DUPLICATED correlated
        signal split across two indices sharing the SAME feature row
        with the SAME coefficient (indices 1 and 2, raw diag 5e-6 each
        -- genuinely redundant, not independent), and an independent
        weak signal (index 3, raw diag 2e-6). True effective rank is 3
        (strong + one 5e-6 direction + the independent 2e-6 direction);
        the pre-round-5 primitive gave 2, having skipped deflating the
        first 5e-6 pivot (its raw residual sits below the too-loose
        float32 small_eps), so its duplicate looked like fresh signal
        and the independent 2e-6 point was never reached before the
        (prematurely closed) prefix latch shut.
        """
        a = 1.0 ** 0.25
        b = (5e-6) ** 0.25
        c = (2e-6) ** 0.25
        factor_p = jnp.array([
            [a, 0.0, 0.0, 0.0],
            [0.0, b, b, 0.0],
            [0.0, 0.0, 0.0, c],
        ], dtype=jnp.float32)
        factor_q = factor_p

        diag_err = jnp.sum(factor_p**2, axis=0) * jnp.sum(factor_q**2, axis=0)
        self.assertEqual(diag_err.dtype, jnp.float32)
        shift = jnp.array(0.0, dtype=jnp.float32)
        pivots, effective_rank = pivoted_cholesky_pair_pivots(
            factor_p, factor_q, 4, shift, track_effective_rank=True, effective_rank_rtol=1e-6)
        pivots_np = np.asarray(pivots)

        self.assertEqual(effective_rank, 3)
        # Prefix must contain the strong signal (0), the independent
        # weak signal (3), and exactly one of the duplicated pair (1 or
        # 2) -- never both (that would mean deflation still failed to
        # recognize the duplicate as redundant).
        prefix = set(pivots_np[:3].tolist())
        self.assertIn(0, prefix)
        self.assertIn(3, prefix)
        self.assertEqual(len(prefix & {1, 2}), 1)

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
        # Bounded by BOTH the analytic pair-rank cap (95) AND the numeric
        # effective-rank check (measured 94 at the library's default
        # rtol=1e-6, a safe 1-point margin below 95 -- see
        # pivoted_cholesky_pair_pivots's docstring) -- never 300.
        self.assertLessEqual(len(np.asarray(pivots)), n_pair)
        self.assertGreaterEqual(len(np.asarray(pivots)), n_pair - 5)

        with self.assertRaises(ValueError):
            select_sector_pivots(occ_weighted, vir_weighted, 300, on_over_rank="raise")

        with self.assertRaises(ValueError):
            select_sector_pivots(occ_weighted, vir_weighted, 5, on_over_rank="not-a-real-mode")

        # Not-over-rank requests are returned in full, untouched.
        pivots_small = select_sector_pivots(occ_weighted, vir_weighted, 10)
        self.assertEqual(len(np.asarray(pivots_small)), 10)


if __name__ == "__main__":
    unittest.main()
