"""Tests for the PBC TC integral grid factory.

The molecular K-matrix kernels in :mod:`pytc.kmat` are grid-agnostic —
they take ``phi``, ``grad_phi``, the Jastrow, and the grid as opaque
arrays. So once we have a PBC :class:`TC` whose phi/grad_phi values
match the molecular evaluation in the large-cell limit (and whose
Jastrow uses MIC distances), the downstream K1/K3 must agree.

That cross-check is the load-bearing test below.
"""

import unittest

import numpy as np
import jax.numpy as jnp
from pyscf import gto as molgto, scf as molscf
from pyscf.pbc import gto as pbcgto, scf as pbcscf

from pytc.tc import TC as MolTC
from pytc.jastrow import CompositeJastrow
from pytc.jastrow import (
    NuclearCusp as MolNuclearCusp,
    BoysHandy as MolBoysHandy,
)
from pytc import kmat as kmat_jax

from pytc.pbc.tc import create_tc
from pytc.pbc.jastrow import NuclearCusp, BoysHandy


def _build_pbc(L=15.0):
    cell = pbcgto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 1.4'
    cell.basis = 'sto-3g'
    cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    cell.unit = 'B'
    cell.cart = True
    cell.verbose = 0
    cell.build()
    mf = pbcscf.RHF(cell)
    mf.exxdiv = None
    mf.kernel()
    return cell, mf


def _build_mol():
    mol = molgto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1.4'
    mol.basis = 'sto-3g'
    mol.unit = 'B'
    mol.cart = True
    mol.verbose = 0
    mol.build()
    mf = molscf.RHF(mol)
    mf.kernel()
    return mol, mf


class TestPBCTCConstruction(unittest.TestCase):
    def test_construct(self):
        cell, mf = _build_pbc()
        ncusp = NuclearCusp.create(cell, n_radial=300)
        bh = BoysHandy.create(cell)
        jastrow = CompositeJastrow.create([ncusp, bh])
        tc = create_tc(mf, jastrow_factor=jastrow)
        self.assertEqual(tc.phi.shape[0], mf.mo_coeff.shape[1])
        self.assertEqual(tc.grad_phi.shape[-1], 3)
        self.assertEqual(tc.grid_points.shape[1], 3)
        self.assertEqual(tc.weights.shape[0], tc.grid_points.shape[0])

    def test_rejects_spherical_basis(self):
        cell = pbcgto.Cell()
        cell.atom = 'H 0 0 0; H 0 0 1.4'
        cell.basis = 'sto-3g'
        cell.a = [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
        cell.unit = 'B'
        cell.cart = False
        cell.verbose = 0
        cell.build()
        mf = pbcscf.RHF(cell)
        mf.exxdiv = None
        mf.kernel()
        ncusp = NuclearCusp.create(cell, n_radial=200)
        with self.assertRaises(ValueError):
            create_tc(mf, jastrow_factor=ncusp)

    def test_grid_weights_sum_close_to_volume(self):
        """Becke grid integrates 1 to ~ cell volume."""
        cell, mf = _build_pbc(L=8.0)
        ncusp = NuclearCusp.create(cell, n_radial=200)
        tc = create_tc(mf, jastrow_factor=ncusp)
        volume = float(np.abs(np.linalg.det(cell.lattice_vectors())))
        weight_sum = float(jnp.sum(tc.weights))
        # Becke grids on PBC cells centre weight near nuclei; the sum is
        # close to the atomic-decomposed volume rather than exact cell volume.
        # Just sanity-check it's in a reasonable range.
        self.assertGreater(weight_sum, 0.5 * volume)
        self.assertLess(weight_sum, 1.5 * volume)


class TestPBCTCMatchesMolecularInLargeCell(unittest.TestCase):
    """Load-bearing correctness check: in a cell large enough that periodic
    image contributions to the AO grid values are negligible, the PBC TC
    object should produce K1 and K3 matrices that agree with the
    molecular TC built from the same MOs and Jastrow parameters."""

    def setUp(self):
        # Large cell so AO Bloch sum reduces to the bare atomic orbital.
        self.cell, self.mf_pbc = _build_pbc(L=15.0)
        self.mol, _ = _build_mol()

        # Use the PBC mo_coeff for both to avoid sign/phase ambiguity.
        mo = self.mf_pbc.mo_coeff
        self.pbc_ncusp = NuclearCusp.create(self.cell, n_radial=300)
        self.pbc_bh = BoysHandy.create(self.cell)
        self.pbc_jas = CompositeJastrow.create([self.pbc_ncusp, self.pbc_bh])

        self.mol_ncusp = MolNuclearCusp.create(self.mol, n_radial=300)
        self.mol_bh = MolBoysHandy.create(self.mol)
        self.mol_jas = CompositeJastrow.create([self.mol_ncusp, self.mol_bh])

        # Build TCs at the same grid level so weights match
        self.tc_pbc = create_tc(self.mf_pbc, jastrow_factor=self.pbc_jas,
                                mo_coeff=mo, grid_lvl=2)
        self.tc_mol = MolTC.from_pyscf(
            type('MfShim', (), {'mol': self.mol, 'mo_coeff': mo,
                                'mo_occ': self.mf_pbc.mo_occ})(),
            jastrow_factor=self.mol_jas, mo_coeff=mo, grid_lvl=2,
        )

        # init_params is deterministic for same atom data + basis
        self.pbc_params = self.pbc_jas.init_params()
        self.mol_params = self.mol_jas.init_params()

    def test_grid_sizes_close(self):
        """At large L the PBC and molecular Becke grids should be similar in
        size, though boundary handling causes small differences (a few
        dozen points out of thousands). The downstream integrals must
        match regardless."""
        n_pbc = self.tc_pbc.grid_points.shape[0]
        n_mol = self.tc_mol.grid_points.shape[0]
        self.assertLess(abs(n_pbc - n_mol) / n_mol, 0.05)

    def test_K1_matches_molecular(self):
        """K1 from the PBC TC must equal K1 from the molecular TC at large L."""
        k1_pbc = kmat_jax.calc_K1(
            self.tc_pbc.phi, self.tc_pbc.grad_phi,
            self.pbc_jas, self.pbc_params,
            self.tc_pbc.grid_points, self.tc_pbc.weights,
            batch_size=500,
        )
        k1_mol = kmat_jax.calc_K1(
            self.tc_mol.phi, self.tc_mol.grad_phi,
            self.mol_jas, self.mol_params,
            self.tc_mol.grid_points, self.tc_mol.weights,
            batch_size=500,
        )
        np.testing.assert_allclose(np.asarray(k1_pbc), np.asarray(k1_mol), atol=1e-6)

    def test_K3_matches_molecular(self):
        k3_pbc = kmat_jax.calc_K3(
            self.tc_pbc.phi, self.pbc_jas, self.pbc_params,
            self.tc_pbc.grid_points, self.tc_pbc.weights,
            batch_size=500,
        )
        k3_mol = kmat_jax.calc_K3(
            self.tc_mol.phi, self.mol_jas, self.mol_params,
            self.tc_mol.grid_points, self.tc_mol.weights,
            batch_size=500,
        )
        np.testing.assert_allclose(np.asarray(k3_pbc), np.asarray(k3_mol), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
