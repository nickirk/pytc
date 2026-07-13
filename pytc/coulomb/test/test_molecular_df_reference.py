"""Tests for pytc.coulomb.molecular_df_reference (task #6, isdf-coulomb-cuda).

Validation ladder steps 1-2 (decision 001):
  1. ERI-block errors vs exact DF blocks.
  2. ISDF-MP2 (oovv) vs gpu4pyscf/pyscf exact DF-MP2 -- one-number
     acceptance test.

Runs on H2O/cc-pVDZ (CPU, no CUDA available here) -- gpu4pyscf-backend
validation is later work once GPU hardware is available (Grace, per
Ke's sequencing).
"""

import unittest

import numpy as np
from pyscf import gto, scf, mp

from pytc.coulomb.gpu4pyscf_adapter import get_mo_coeff, get_grid_ao_values_and_weights
from pytc.coulomb.pivot_selection import weight_mo_values, select_sector_pivots
from pytc.coulomb.molecular_df_reference import (
    pair_collocation_at_pivots,
    compute_C_streamed,
    compute_Z,
    compute_Z_cross,
    reconstruct_eri_block,
)


class TestMolecularDFReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="O 0 0 0; H 0 0 0.96; H 0.926 0 -0.24",
            basis="cc-pvdz", verbose=0)
        cls.mf = scf.RHF(cls.mol).density_fit().run()
        cls.n_occ = cls.mol.nelectron // 2
        mo_coeff = get_mo_coeff(cls.mf)
        cls.mo_occ = mo_coeff[:, :cls.n_occ]
        cls.mo_vir = mo_coeff[:, cls.n_occ:]
        cls.n_vir = cls.mo_vir.shape[1]

        ao_values, weights, coords = get_grid_ao_values_and_weights(cls.mf, grid_lvl=2)
        cls.weights = weights
        mo_values = (ao_values @ mo_coeff).T
        cls.occ_raw = mo_values[:cls.n_occ]
        cls.vir_raw = mo_values[cls.n_occ:]
        mo_weighted = weight_mo_values(mo_values, weights)
        cls.occ_weighted = mo_weighted[:cls.n_occ]
        cls.vir_weighted = mo_weighted[cls.n_occ:]

        # n_rank=300 requested >> n_pair=95 (n_occ*n_vir) -- deliberately
        # over-asked so select_sector_pivots's over-rank truncation (Alice's
        # task #6 re-review, 2026-07-12, blocker item 3) kicks in and cls.P
        # ends up with exactly n_pair=95 pivots, the sector's true rank --
        # the near-exact regime this fixture is FOR (see
        # test_cross_sector_*_compressed_rank for the genuinely-compressed
        # regime these near-exact fixtures don't cover).
        cls.n_rank = 300
        pivots = np.asarray(select_sector_pivots(
            cls.occ_weighted, cls.vir_weighted, cls.n_rank))
        # RAW (unweighted) values at the pivots for P -- pair_collocation_at_pivots
        # must not see the sqrt(weight) scaling used for pivot selection
        # (Alice's task #6 review, 2026-07-12; see its docstring).
        occ_at_piv = np.asarray(cls.occ_raw)[:, pivots]
        vir_at_piv = np.asarray(cls.vir_raw)[:, pivots]
        cls.P = pair_collocation_at_pivots(occ_at_piv, vir_at_piv)
        cls.C = compute_C_streamed(cls.mf, cls.P, cls.mo_occ, cls.mo_vir, auxbasis="weigend")
        cls.Z, cls.Z_provenance = compute_Z(cls.P, cls.C)

        cls.eri_ovov_exact = cls.mf.with_df.ao2mo(
            (cls.mo_occ, cls.mo_vir, cls.mo_occ, cls.mo_vir), compact=False
        ).reshape(cls.n_occ, cls.n_vir, cls.n_occ, cls.n_vir)

        # oo and vv sectors, independently pivoted (per
        # pivot_selection.select_pivots_oo_ov_vv's convention) -- needed
        # to exercise compute_Z_cross for the oo|vv and ov|vv blocks
        # CCSD needs but MP2's ov|ov-only validation never touches
        # (Alice's task #6 review, 2026-07-12, blocker item 2).
        # oo's pair-product space is SYMMETRIC (phi_p*phi_q == phi_q*phi_p
        # as functions on the grid for p,q both occupied) so its true rank
        # is the triangular number n_occ*(n_occ+1)/2=15, not n_occ**2=25 --
        # 75 requested truncates down to 15 (the sector's true rank).
        cls.n_rank_oo = 75
        pivots_oo = np.asarray(select_sector_pivots(
            cls.occ_weighted, cls.occ_weighted, cls.n_rank_oo))
        occ_at_piv_oo = np.asarray(cls.occ_raw)[:, pivots_oo]
        cls.P_oo = pair_collocation_at_pivots(occ_at_piv_oo, occ_at_piv_oo)
        cls.C_oo = compute_C_streamed(cls.mf, cls.P_oo, cls.mo_occ, cls.mo_occ, auxbasis="weigend")

        # Same symmetric-sector argument as oo: vv's true rank is
        # n_vir*(n_vir+1)/2=190, not n_vir**2=361 -- 380 requested
        # truncates down to 190.
        cls.n_rank_vv = 380
        pivots_vv = np.asarray(select_sector_pivots(
            cls.vir_weighted, cls.vir_weighted, cls.n_rank_vv))
        vir_at_piv_vv = np.asarray(cls.vir_raw)[:, pivots_vv]
        cls.P_vv = pair_collocation_at_pivots(vir_at_piv_vv, vir_at_piv_vv)
        cls.C_vv = compute_C_streamed(cls.mf, cls.P_vv, cls.mo_vir, cls.mo_vir, auxbasis="weigend")

        cls.eri_oovv_exact = cls.mf.with_df.ao2mo(
            (cls.mo_occ, cls.mo_occ, cls.mo_vir, cls.mo_vir), compact=False
        ).reshape(cls.n_occ, cls.n_occ, cls.n_vir, cls.n_vir)
        cls.eri_ovvv_exact = cls.mf.with_df.ao2mo(
            (cls.mo_occ, cls.mo_vir, cls.mo_vir, cls.mo_vir), compact=False
        ).reshape(cls.n_occ, cls.n_vir, cls.n_vir, cls.n_vir)

    def test_C_streamed_matches_direct_PVPdagger(self):
        """C C^dagger must equal P V P^dagger exactly (pure algebra, no
        ISDF approximation involved -- C=PB^dagger, V=B^dagger B implies
        this identically). A mismatch here would mean a real coding bug
        in compute_C_streamed's AO->MO transform or block accumulation,
        not an approximation-quality issue."""
        V_flat = self.eri_ovov_exact.reshape(self.n_occ * self.n_vir, self.n_occ * self.n_vir)
        CCt = self.C @ self.C.conj().T
        PVPt = self.P @ V_flat @ self.P.conj().T
        np.testing.assert_allclose(CCt, PVPt, atol=1e-10)

    def test_eri_block_reconstruction_error_validation_ladder_step1(self):
        """Validation ladder step 1: ERI-block error vs exact DF, at a
        deliberately overcomplete rank (300 pivots for a 95-dim pair
        space) -- should be a small fraction of a percent, not just
        "plausible."""
        eri_isdf = reconstruct_eri_block(self.P, self.Z, self.P).reshape(
            self.n_occ, self.n_vir, self.n_occ, self.n_vir)
        rel_err = (np.linalg.norm(eri_isdf - self.eri_ovov_exact)
                   / np.linalg.norm(self.eri_ovov_exact))
        self.assertLess(rel_err, 0.01)  # measured ~1.2e-6 with raw P + numpy's default rcond

    def test_isdf_mp2_matches_exact_dfmp2_validation_ladder_step2(self):
        """Validation ladder step 2: ISDF-MP2 vs exact DF-MP2 one-number
        acceptance test."""
        eri_isdf = reconstruct_eri_block(self.P, self.Z, self.P).reshape(
            self.n_occ, self.n_vir, self.n_occ, self.n_vir)
        mo_energy = self.mf.mo_energy
        e_occ = mo_energy[:self.n_occ]
        e_vir = mo_energy[self.n_occ:]
        denom = (e_occ[:, None, None, None] + e_occ[None, None, :, None]
                 - e_vir[None, :, None, None] - e_vir[None, None, None, :])
        t2 = eri_isdf / denom
        e_corr_isdf = np.einsum(
            "iajb,iajb->", t2, 2 * eri_isdf - eri_isdf.transpose(0, 3, 2, 1))

        pt = mp.dfmp2.DFMP2(self.mf)
        pt.run()

        # Measured ~0.025 mHa deviation on this system -- 0.5 mHa gives
        # real margin while still being a meaningful acceptance bound
        # (well inside the ~1 mHa/atom manuscript-relevant scale).
        self.assertLess(abs(e_corr_isdf - pt.e_corr), 5e-4)

    def test_cross_sector_oovv_reconstruction_matches_exact_df(self):
        """compute_Z_cross must reconstruct the oo|vv ERI block -- a
        cross-sector block MP2's ov|ov-only validation never exercises,
        but CCSD needs (Alice's task #6 review, blocker item 2)."""
        Z_oovv, _prov = compute_Z_cross(self.P_oo, self.C_oo, self.P_vv, self.C_vv)
        eri_isdf = reconstruct_eri_block(self.P_oo, Z_oovv, self.P_vv).reshape(
            self.n_occ, self.n_occ, self.n_vir, self.n_vir)
        rel_err = (np.linalg.norm(eri_isdf - self.eri_oovv_exact)
                   / np.linalg.norm(self.eri_oovv_exact))
        self.assertLess(rel_err, 0.05)

    def test_cross_sector_ovvv_reconstruction_matches_exact_df(self):
        """compute_Z_cross must reconstruct the ov|vv ERI block -- the
        other cross-sector block CCSD needs (Alice's task #6 review,
        blocker item 2)."""
        Z_ovvv, _prov = compute_Z_cross(self.P, self.C, self.P_vv, self.C_vv)
        eri_isdf = reconstruct_eri_block(self.P, Z_ovvv, self.P_vv).reshape(
            self.n_occ, self.n_vir, self.n_vir, self.n_vir)
        rel_err = (np.linalg.norm(eri_isdf - self.eri_ovvv_exact)
                   / np.linalg.norm(self.eri_ovvv_exact))
        self.assertLess(rel_err, 0.05)

    def test_cross_sector_ovvv_reconstruction_compressed_rank(self):
        """The near-exact setUpClass fixtures (n_rank_oo/n_rank_vv chosen
        deliberately overcomplete, now truncated by select_sector_pivots
        to exactly the sector's true rank -- see blocker item 3) only
        validate the cross-sector interface in the near-exact full-pair
        regime. CCSD production runs deliberately use fewer pivots than
        the full pair rank for efficiency -- this test exercises that
        actually-compressed regime (n_rank_ov=50 < true rank 95,
        n_rank_vv=100 < true rank 190) and checks the reconstruction
        degrades GRACEFULLY (finite, well below the 1.0-relative-error
        nonsense floor), not that it's near-exact (Alice's task #6
        re-review, 2026-07-12, blocker item 4)."""
        n_rank_ov_compressed = 50
        n_rank_vv_compressed = 100
        pivots_ov = np.asarray(select_sector_pivots(
            self.occ_weighted, self.vir_weighted, n_rank_ov_compressed))
        self.assertEqual(len(pivots_ov), n_rank_ov_compressed,
                          "n_rank_ov_compressed must be below the true rank -- no truncation expected")
        P_ov = pair_collocation_at_pivots(
            np.asarray(self.occ_raw)[:, pivots_ov], np.asarray(self.vir_raw)[:, pivots_ov])
        C_ov = compute_C_streamed(self.mf, P_ov, self.mo_occ, self.mo_vir, auxbasis="weigend")

        pivots_vv = np.asarray(select_sector_pivots(
            self.vir_weighted, self.vir_weighted, n_rank_vv_compressed))
        self.assertEqual(len(pivots_vv), n_rank_vv_compressed,
                          "n_rank_vv_compressed must be below the true rank -- no truncation expected")
        P_vv = pair_collocation_at_pivots(
            np.asarray(self.vir_raw)[:, pivots_vv], np.asarray(self.vir_raw)[:, pivots_vv])
        C_vv = compute_C_streamed(self.mf, P_vv, self.mo_vir, self.mo_vir, auxbasis="weigend")

        Z_ovvv, _prov = compute_Z_cross(P_ov, C_ov, P_vv, C_vv)
        eri_isdf = reconstruct_eri_block(P_ov, Z_ovvv, P_vv).reshape(
            self.n_occ, self.n_vir, self.n_vir, self.n_vir)
        rel_err = (np.linalg.norm(eri_isdf - self.eri_ovvv_exact)
                   / np.linalg.norm(self.eri_ovvv_exact))
        self.assertTrue(np.isfinite(rel_err))
        # measured ~0.27 at these ranks -- generous margin, this is a
        # sanity/regime check, not a tight acceptance bound.
        self.assertLess(rel_err, 0.6)

    def test_cross_sector_Z_matches_same_sector_Z_special_case(self):
        """compute_Z_cross(P, C, P, C) must equal compute_Z(P, C) exactly
        -- compute_Z is documented as compute_Z_cross's same-sector
        special case, so this identity must hold bit-for-bit (same
        solver calls, same inputs), not just approximately."""
        Z_cross_same, prov_cross = compute_Z_cross(self.P, self.C, self.P, self.C)
        np.testing.assert_allclose(Z_cross_same, self.Z, atol=1e-12)
        self.assertEqual(prov_cross["solver"], self.Z_provenance["solver"])
        self.assertEqual(prov_cross["jitter_used"], self.Z_provenance["jitter_used"])

    def test_compute_Z_provenance_fields(self):
        """compute_Z's provenance dict must carry the design doc §4
        solver-level fields it can actually observe (2026-07-12):
        solver, jitter/retries, dtype, two-sided fit residual + its
        norm convention + acceptance threshold, row-scaling. Default
        solver is cholesky_jitter (production, per the doc's measured
        decision); tsvd remains available as the diagnostic/fallback
        mode and must report a genuinely different field set (retained
        singular-value range instead of jitter)."""
        prov = self.Z_provenance
        self.assertEqual(prov["solver"], "unscaled_cholesky_jitter")
        self.assertIsInstance(prov["jitter_used"], tuple)
        self.assertIsInstance(prov["n_tries"], tuple)
        self.assertIsNone(prov["retained_singular_value_range"])
        self.assertIn("float", prov["dtype"])
        self.assertGreaterEqual(prov["fit_residual"], 0.0)
        # Two-sided residual on the actual returned Z must be tiny and
        # well within the design doc's acceptance threshold for this
        # well-conditioned (analytically pre-capped, near-full-rank) test.
        self.assertLess(prov["fit_residual"], prov["residual_warn_threshold"])
        self.assertEqual(prov["row_scaling"], "identity")

        Z_tsvd, prov_tsvd = compute_Z(self.P, self.C, solver="tsvd")
        self.assertEqual(prov_tsvd["solver"], "tsvd")
        self.assertIsNotNone(prov_tsvd["retained_singular_value_range"])
        self.assertIn("n_retained", prov_tsvd)
        # Both solvers must agree on the actual ERI reconstruction to
        # within a small tolerance -- different regularization, same
        # underlying physics.
        eri_chol = reconstruct_eri_block(self.P, self.Z, self.P)
        eri_tsvd = reconstruct_eri_block(self.P, Z_tsvd, self.P)
        rel_diff = np.linalg.norm(eri_chol - eri_tsvd) / np.linalg.norm(eri_chol)
        self.assertLess(rel_diff, 0.05)

    def test_compute_Z_rejects_unknown_solver(self):
        with self.assertRaises(ValueError):
            compute_Z(self.P, self.C, solver="not-a-real-solver")
        with self.assertRaises(ValueError):
            compute_Z_cross(self.P, self.C, self.P, self.C, solver="not-a-real-solver")

    def test_P_independent_of_weight_scale(self):
        """Rescaling the integration weights by a positive constant must
        not change which pivots pivot SELECTION picks (the pivoted-
        Cholesky argmax/shift/tie-break-ramp are all scale-covariant)
        or the resulting RAW-value P (which doesn't depend on weights
        at all once selection is done -- Alice's task #6 review,
        blocker item 1: weighting must be confined to pivot selection,
        never touching the actual interpolation factors). A regression
        here would mean weighting leaked back into P's construction.

        Uses a SMALL rank (20, well below the ov pair space's true rank
        of 95) rather than self.n_rank=300: past the true rank, the
        residual is floating-point noise and which noise-dominated
        index argmax picks next is inherently scale-sensitive (verified
        separately -- not a bug, just not what this test is checking).
        Within the true rank, selection order reflects real signal and
        must be scale-invariant.
        """
        scale = 7.0
        n_rank_small = 20
        occ_weighted_scaled = weight_mo_values(self.occ_raw, self.weights * scale)
        vir_weighted_scaled = weight_mo_values(self.vir_raw, self.weights * scale)
        pivots_scaled = np.asarray(select_sector_pivots(
            occ_weighted_scaled, vir_weighted_scaled, n_rank_small))
        pivots_orig = np.asarray(select_sector_pivots(
            self.occ_weighted, self.vir_weighted, n_rank_small))
        np.testing.assert_array_equal(pivots_scaled, pivots_orig)

        occ_at_piv_scaled = np.asarray(self.occ_raw)[:, pivots_scaled]
        vir_at_piv_scaled = np.asarray(self.vir_raw)[:, pivots_scaled]
        P_scaled = pair_collocation_at_pivots(occ_at_piv_scaled, vir_at_piv_scaled)
        occ_at_piv_orig = np.asarray(self.occ_raw)[:, pivots_orig]
        vir_at_piv_orig = np.asarray(self.vir_raw)[:, pivots_orig]
        P_orig = pair_collocation_at_pivots(occ_at_piv_orig, vir_at_piv_orig)
        np.testing.assert_allclose(P_scaled, P_orig, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
