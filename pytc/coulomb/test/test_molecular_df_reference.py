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
        mo_values = (ao_values @ mo_coeff).T
        mo_weighted = weight_mo_values(mo_values, weights)
        cls.occ_weighted = mo_weighted[:cls.n_occ]
        cls.vir_weighted = mo_weighted[cls.n_occ:]

        # n_rank=300 >> n_pair=95 (n_occ*n_vir) -- deliberately overcomplete
        # so the ISDF approximation should be accurate, not just plausible.
        cls.n_rank = 300
        pivots = np.asarray(select_sector_pivots(
            cls.occ_weighted, cls.vir_weighted, cls.n_rank))
        occ_at_piv = np.asarray(cls.occ_weighted)[:, pivots]
        vir_at_piv = np.asarray(cls.vir_weighted)[:, pivots]
        cls.P = pair_collocation_at_pivots(occ_at_piv, vir_at_piv)
        cls.C = compute_C_streamed(cls.mf, cls.P, cls.mo_occ, cls.mo_vir, auxbasis="weigend")
        cls.Z = compute_Z(cls.P, cls.C)

        cls.eri_ovov_exact = cls.mf.with_df.ao2mo(
            (cls.mo_occ, cls.mo_vir, cls.mo_occ, cls.mo_vir), compact=False
        ).reshape(cls.n_occ, cls.n_vir, cls.n_occ, cls.n_vir)

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
        self.assertLess(rel_err, 0.01)  # measured ~0.0014 on this system

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


if __name__ == "__main__":
    unittest.main()
