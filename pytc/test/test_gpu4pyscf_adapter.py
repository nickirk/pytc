"""Tests for pytc.integrals.coulomb's gpu4pyscf-adapter section (task #4,
isdf-coulomb-cuda).

Runs against plain pyscf RHF only -- no CUDA/gpu4pyscf available on this
host. The adapter's gpu4pyscf code paths (cupy .get() conversion,
gpu4pyscf.df.df.DF dispatch) are exercised by construction whenever
type(mf).__module__ starts with "gpu4pyscf"; that branch is validated on
real hardware separately (Grace, when a later task produces something to
run on GPU) rather than mocked here.
"""

import unittest

import numpy as np
from pyscf import gto, scf, dft, df

from pytc.integrals.coulomb import (
    get_mo_coeff,
    get_grid_ao_values_and_weights,
    get_naux,
    stream_df_cderi_blocks,
)


def _benzene_geom(cc=1.397, ch=1.084):
    """D6h benzene geometry (Angstrom). The formula was shared with
    pytc/utils/perf_baseline.py, deleted upstream in e8c3374; this is now the
    only copy. Kept small (STO-3G) purely for test speed, not physical
    accuracy."""
    atoms = []
    rh = cc + ch
    for i in range(6):
        theta = np.deg2rad(60.0 * i)
        atoms.append(f"C {np.cos(theta):.6f} {np.sin(theta):.6f} 0")
    for i in range(6):
        theta = np.deg2rad(60.0 * i)
        atoms.append(f"H {rh*np.cos(theta):.6f} {rh*np.sin(theta):.6f} 0")
    return "; ".join(atoms)


class TestGpu4PyscfAdapterH2O(unittest.TestCase):
    """H2O/STO-3G -- the smallest of the two required test systems."""

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="O 0 0 0; H 0 0 0.96; H 0.926 0 -0.24",
            basis="sto-3g", verbose=0)
        cls.mf = scf.RHF(cls.mol).run()

    def test_get_mo_coeff_shape_and_identity(self):
        mo_coeff = get_mo_coeff(self.mf)
        self.assertEqual(mo_coeff.shape, (self.mol.nao, self.mol.nao))
        self.assertIsInstance(mo_coeff, np.ndarray)
        # Plain pyscf mf.mo_coeff is already host numpy -- adapter must
        # round-trip it unchanged, not silently transform it.
        np.testing.assert_array_equal(mo_coeff, self.mf.mo_coeff)

    def test_get_mo_coeff_raises_before_kernel(self):
        fresh_mf = scf.RHF(self.mol)  # .kernel() never called
        with self.assertRaises(ValueError):
            get_mo_coeff(fresh_mf)

    def test_grid_ao_values_and_weights_match_pyscf_reference(self):
        ao_values, weights, coords = get_grid_ao_values_and_weights(
            self.mf, grid_lvl=1)

        # Reproduce pytc/tc.py's own from_pyscf convention independently
        # (dft.gen_grid.Grids + dft.numint.eval_ao at the same level) and
        # require exact agreement -- this pins the adapter to the SAME
        # grid pytc's existing dense/DF pipeline already uses.
        ref_grids = dft.gen_grid.Grids(self.mol)
        ref_grids.level = 1
        ref_grids.build()
        ref_ao = dft.numint.eval_ao(self.mol, ref_grids.coords, deriv=0)

        self.assertEqual(coords.shape, ref_grids.coords.shape)
        np.testing.assert_array_equal(coords, ref_grids.coords)
        np.testing.assert_array_equal(weights, ref_grids.weights)
        np.testing.assert_array_equal(ao_values, ref_ao)

        n_grid = coords.shape[0]
        self.assertEqual(ao_values.shape, (n_grid, self.mol.nao))
        self.assertEqual(weights.shape, (n_grid,))
        self.assertTrue(np.all(np.isfinite(weights)))

    def test_grid_ao_values_deriv1_shape(self):
        ao_values, weights, coords = get_grid_ao_values_and_weights(
            self.mf, grid_lvl=1, deriv=1)
        n_grid = coords.shape[0]
        # deriv=1: (value, d/dx, d/dy, d/dz) components (pyscf convention).
        self.assertEqual(ao_values.shape, (4, n_grid, self.mol.nao))

    def test_get_naux_matches_pyscf_get_naoaux(self):
        with_df = df.DF(self.mol, auxbasis="weigend")
        with_df.build()
        self.assertEqual(get_naux(with_df), with_df.get_naoaux())

    def test_stream_df_cderi_blocks_reconstructs_naux_and_matches_reference(self):
        with_df = df.DF(self.mol, auxbasis="weigend")
        with_df.build()
        naux_ref = with_df.get_naoaux()

        blocks = list(stream_df_cderi_blocks(self.mf, auxbasis="weigend"))
        self.assertGreater(len(blocks), 0)
        total_naux = sum(b.shape[0] for b in blocks)
        self.assertEqual(total_naux, naux_ref)

        nao_pair = self.mol.nao * (self.mol.nao + 1) // 2
        for b in blocks:
            self.assertIsInstance(b, np.ndarray)
            self.assertEqual(b.shape[1], nao_pair)

        # Full reconstruction must equal a fresh independent DF build's
        # own concatenated blocks -- correctness, not just shape.
        reconstructed = np.concatenate(blocks, axis=0)
        reference = np.concatenate(list(with_df.loop()), axis=0)
        np.testing.assert_allclose(reconstructed, reference, atol=1e-12)


class TestGpu4PyscfAdapterBenzene(unittest.TestCase):
    """Benzene/STO-3G -- the larger of the two required test systems
    (12 atoms, D6h), kept at a minimal basis purely for CPU test speed."""

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=_benzene_geom(), basis="sto-3g", verbose=0)
        cls.mf = scf.RHF(cls.mol).run()

    def test_get_mo_coeff_shape(self):
        mo_coeff = get_mo_coeff(self.mf)
        self.assertEqual(mo_coeff.shape, (self.mol.nao, self.mol.nao))

    def test_grid_ao_values_and_weights_shapes(self):
        ao_values, weights, coords = get_grid_ao_values_and_weights(
            self.mf, grid_lvl=0)  # level 0: coarsest, keeps the test fast
        n_grid = coords.shape[0]
        self.assertEqual(ao_values.shape, (n_grid, self.mol.nao))
        self.assertEqual(weights.shape, (n_grid,))
        self.assertTrue(np.all(np.isfinite(ao_values)))

    def test_stream_df_cderi_blocks_reconstructs_naux(self):
        with_df = df.DF(self.mol, auxbasis="weigend")
        with_df.build()
        naux_ref = with_df.get_naoaux()

        blocks = list(stream_df_cderi_blocks(self.mf, auxbasis="weigend"))
        total_naux = sum(b.shape[0] for b in blocks)
        self.assertEqual(total_naux, naux_ref)


if __name__ == "__main__":
    unittest.main()
