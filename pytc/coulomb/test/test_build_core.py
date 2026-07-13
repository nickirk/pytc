"""Tests for pytc.coulomb.build_core (task #8 follow-up, isdf-coulomb-cuda,
2026-07-12): build_sector/build_core must reproduce exactly what the
hand-rolled pivot-selection -> pair_collocation_at_pivots ->
compute_C_streamed -> compute_Z/compute_Z_cross pipeline produces
(bit-exact), while additionally assembling the full §4 provenance record
(pivot/grid/mo_coeff hashes, distinct rank fields, kernel policy,
upstream provenance) that pipeline never recorded.
"""

import unittest

import numpy as np
from pyscf import gto, scf

from pytc.coulomb.gpu4pyscf_adapter import get_mo_coeff, get_grid_ao_values_and_weights
from pytc.coulomb.pivot_selection import weight_mo_values, select_sector_pivots
from pytc.coulomb.molecular_df_reference import pair_collocation_at_pivots, compute_C_streamed
from pytc.df.fit import compute_Z, compute_Z_cross
from pytc.coulomb.build_core import build_sector, build_core, SectorFit, CoreArtifact


class TestBuildCore(unittest.TestCase):
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

        cls.ao_values, cls.weights, cls.coords = get_grid_ao_values_and_weights(cls.mf, grid_lvl=2)
        mo_values = (cls.ao_values @ mo_coeff).T
        cls.occ_raw = mo_values[:cls.n_occ]
        cls.vir_raw = mo_values[cls.n_occ:]
        mo_weighted = weight_mo_values(mo_values, cls.weights)
        cls.occ_weighted = mo_weighted[:cls.n_occ]
        cls.vir_weighted = mo_weighted[cls.n_occ:]

        cls.n_rank_ov = 300
        cls.n_rank_oo = 75

    def _reference_ov_pipeline(self):
        pivots = np.asarray(select_sector_pivots(
            self.occ_weighted, self.vir_weighted, self.n_rank_ov))
        occ_at_piv = np.asarray(self.occ_raw)[:, pivots]
        vir_at_piv = np.asarray(self.vir_raw)[:, pivots]
        P = pair_collocation_at_pivots(occ_at_piv, vir_at_piv)
        C = compute_C_streamed(self.mf, P, self.mo_occ, self.mo_vir, auxbasis="weigend")
        return pivots, P, C

    def test_build_sector_matches_hand_rolled_pipeline_bit_exact(self):
        ref_pivots, ref_P, ref_C = self._reference_ov_pipeline()
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
            self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov)
        self.assertIsInstance(sector, SectorFit)
        np.testing.assert_array_equal(np.asarray(sector.pivots), ref_pivots)
        np.testing.assert_allclose(sector.P, ref_P, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(sector.C, ref_C, atol=0.0, rtol=0.0)

    def test_build_core_same_sector_matches_compute_Z_directly(self):
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
            self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov)
        core = build_core(sector)
        self.assertIsInstance(core, CoreArtifact)
        Z_direct, prov_direct = compute_Z(sector.P, sector.C)
        np.testing.assert_allclose(core.Z, Z_direct, atol=1e-12)
        self.assertEqual(core.provenance["solver"], prov_direct["solver"])
        self.assertEqual(core.provenance["jitter_used"], prov_direct["jitter_used"])
        self.assertIn("sector", core.provenance)
        self.assertEqual(core.provenance["sector"], sector.provenance)

    def test_build_core_cross_sector_matches_compute_Z_cross_directly(self):
        oo_pivots = np.asarray(select_sector_pivots(
            self.occ_weighted, self.occ_weighted, self.n_rank_oo, same_factor=True))
        sector_oo = build_sector(
            self.mf, self.occ_raw, self.occ_raw, self.occ_weighted, self.occ_weighted,
            self.mo_occ, self.mo_occ, self.coords, self.weights, self.n_rank_oo,
            same_factor=True)
        sector_ov = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
            self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov)

        core = build_core(sector_oo, sector_ov)
        Z_direct, prov_direct = compute_Z_cross(
            sector_oo.P, sector_oo.C, sector_ov.P, sector_ov.C, same_sector=False)
        np.testing.assert_allclose(core.Z, Z_direct, atol=1e-12)
        self.assertEqual(core.provenance["jitter_used"], prov_direct["jitter_used"])
        self.assertIn("left_sector", core.provenance)
        self.assertIn("right_sector", core.provenance)
        self.assertEqual(core.provenance["left_sector"], sector_oo.provenance)
        self.assertEqual(core.provenance["right_sector"], sector_ov.provenance)

    def test_provenance_rank_fields_are_distinct(self):
        # n_rank_oo=75 requested >> oo's true triangular-number rank
        # (n_occ*(n_occ+1)/2=15 for H2O/cc-pVDZ, n_occ=5) -- so
        # requested_rank, analytic_rank_bound, and numerical_rank/
        # n_pivots must all differ, proving they're tracked as distinct
        # concepts, not one value standing in for all three (Alice's
        # build_core API review, 2026-07-12, point 3).
        sector_oo = build_sector(
            self.mf, self.occ_raw, self.occ_raw, self.occ_weighted, self.occ_weighted,
            self.mo_occ, self.mo_occ, self.coords, self.weights, self.n_rank_oo,
            same_factor=True)
        prov = sector_oo.provenance
        self.assertEqual(prov["requested_rank"], self.n_rank_oo)
        self.assertLess(prov["analytic_rank_bound"], self.n_rank_oo)
        self.assertEqual(prov["numerical_rank"], prov["n_pivots"])
        self.assertLess(prov["n_pivots"], self.n_rank_oo)

    def test_provenance_hashes_present_and_reproducible(self):
        sector_a = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
            self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov)
        sector_b = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
            self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov)
        # Same inputs -> identical hashes (reproducibility).
        self.assertEqual(sector_a.provenance["pivot_indices_sha256"],
                          sector_b.provenance["pivot_indices_sha256"])
        self.assertEqual(sector_a.provenance["grid_sha256"], sector_b.provenance["grid_sha256"])
        self.assertEqual(sector_a.provenance["mo_coeff_p_sha256"],
                          sector_b.provenance["mo_coeff_p_sha256"])
        # Different pivot sets (oo vs ov) -> different pivot hash.
        sector_oo = build_sector(
            self.mf, self.occ_raw, self.occ_raw, self.occ_weighted, self.occ_weighted,
            self.mo_occ, self.mo_occ, self.coords, self.weights, self.n_rank_oo,
            same_factor=True)
        self.assertNotEqual(sector_a.provenance["pivot_indices_sha256"],
                             sector_oo.provenance["pivot_indices_sha256"])
        # Same grid -> same grid hash even across different sectors.
        self.assertEqual(sector_a.provenance["grid_sha256"], sector_oo.provenance["grid_sha256"])
        # Different MO coefficients (occ vs vir) -> different mo_coeff hash.
        self.assertNotEqual(sector_a.provenance["mo_coeff_p_sha256"],
                             sector_a.provenance["mo_coeff_q_sha256"])

    def test_upstream_provenance_passthrough(self):
        upstream = {"pyscf_version": "2.10.0", "gpu4pyscf_version": None, "grid_lvl": 2}
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
            self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov,
            upstream_provenance=upstream)
        self.assertEqual(sector.provenance["upstream_provenance"], upstream)

    def test_upstream_provenance_defaults_to_empty_dict(self):
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
            self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov)
        self.assertEqual(sector.provenance["upstream_provenance"], {})

    def test_unsupported_kernel_policy_rejected(self):
        with self.assertRaises(ValueError):
            build_sector(
                self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
                self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov,
                kernel_policy="MolecularFreeSpacePoisson")

    def test_kernel_policy_recorded_in_provenance(self):
        sector = build_sector(
            self.mf, self.occ_raw, self.vir_raw, self.occ_weighted, self.vir_weighted,
            self.mo_occ, self.mo_vir, self.coords, self.weights, self.n_rank_ov)
        self.assertEqual(sector.provenance["kernel_policy"], "MolecularDFReference")
        self.assertEqual(sector.provenance["kernel_policy_params"]["auxbasis"], "weigend")


if __name__ == "__main__":
    unittest.main()
