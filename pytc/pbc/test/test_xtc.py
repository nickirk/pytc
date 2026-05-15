"""Tests for the PBC xTC corrections.

The xTC correction methods (``get_const``, ``get_1b``, ``get_2b``,
``get_delta_U``) are pure JAX operations on the grid + Jastrow data
held on the :class:`XTC` instance. They never read back to the
molecule object or to standard ERIs. So in the large-cell limit, where
the PBC grid + PBC Jastrow reproduce the molecular setup, the
corrections must agree.

That cross-check is the load-bearing test below.
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto as molgto, scf as molscf
from pyscf.pbc import gto as pbcgto, scf as pbcscf

from pytc.xtc import XTC as MolXTC
from pytc.jastrow import (
    CompositeJastrow,
    NuclearCusp as MolNuclearCusp,
    BoysHandy as MolBoysHandy,
)

from pytc.pbc.xtc import create_xtc
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


class TestPBCXTCConstruction(unittest.TestCase):
    def test_construct(self):
        cell, mf = _build_pbc(L=6.0)
        ncusp = NuclearCusp.create(cell, n_radial=300)
        bh = BoysHandy.create(cell)
        jas = CompositeJastrow.create([ncusp, bh])
        xtc = create_xtc(mf, jastrow_factor=jas)
        self.assertIsInstance(xtc, MolXTC)
        np.testing.assert_array_equal(np.asarray(xtc.mo_occ), np.asarray(mf.mo_occ))
        # Periodic energy_nuc must be the Madelung sum (negative for H2 in
        # a small neutral cell), not the bare 1/R = 0.714 Ha.
        self.assertLess(xtc.energy_nuc, 0.0)

    def test_energy_nuc_is_madelung(self):
        cell, mf = _build_pbc(L=6.0)
        ncusp = NuclearCusp.create(cell, n_radial=200)
        xtc = create_xtc(mf, jastrow_factor=ncusp)
        # Should be exactly mf.energy_nuc() = cell.ewald()
        np.testing.assert_allclose(xtc.energy_nuc, mf.energy_nuc(), atol=1e-12)


class TestPBCXTCCorrectionsMatchMolecular(unittest.TestCase):
    """Soft structural check that get_2b / get_const compose correctly.

    The tight cross-check on the underlying integrals lives in
    test_tc.py (K1 and K3 at L=15 to 1e-6). Here we verify at L=8 with
    grid_lvl=1 that get_2b adds up to roughly the molecular value (well
    within finite-cell error) and that get_const carries the right
    Madelung offset on energy_nuc. Anything looser than ~5 mHa would
    be a real structural bug; tighter values are bounded by the
    intrinsic L=8 finite-cell error, not by the code.
    """

    @classmethod
    def setUpClass(cls):
        cls.cell, cls.mf_pbc = _build_pbc(L=8.0)
        cls.mol, _ = _build_mol()
        mo = cls.mf_pbc.mo_coeff

        pbc_ncusp = NuclearCusp.create(cls.cell, n_radial=200)
        pbc_bh = BoysHandy.create(cls.cell)
        cls.pbc_jas = CompositeJastrow.create([pbc_ncusp, pbc_bh])

        mol_ncusp = MolNuclearCusp.create(cls.mol, n_radial=200)
        mol_bh = MolBoysHandy.create(cls.mol)
        cls.mol_jas = CompositeJastrow.create([mol_ncusp, mol_bh])

        cls.xtc_pbc = create_xtc(
            cls.mf_pbc, jastrow_factor=cls.pbc_jas, mo_coeff=mo, grid_lvl=1
        )

        mf_shim = type('MfShim', (), {})()
        mf_shim.mol = cls.mol
        mf_shim.mo_coeff = mo
        mf_shim.mo_occ = cls.mf_pbc.mo_occ
        mf_shim.energy_nuc = lambda: float(cls.mol.energy_nuc())
        cls.xtc_mol = MolXTC.from_pyscf(
            mf_shim, jastrow_factor=cls.mol_jas, mo_coeff=mo, grid_lvl=1,
        )

        pbc_params = cls.pbc_jas.init_params()
        mol_params = cls.mol_jas.init_params()

        cls.h2_pbc = np.asarray(cls.xtc_pbc.get_2b(pbc_params))
        cls.h2_mol = np.asarray(cls.xtc_mol.get_2b(mol_params))
        cls.c_pbc = float(cls.xtc_pbc.get_const(pbc_params))
        cls.c_mol = float(cls.xtc_mol.get_const(mol_params))

    def test_get_2b_close_to_molecular(self):
        """Sanity check: the 2-body xTC correction agrees with the molecular
        one to within the finite-cell error at L=8 (~5 mHa)."""
        np.testing.assert_allclose(self.h2_pbc, self.h2_mol, atol=5e-3)

    def test_get_const_differs_by_madelung_only(self):
        """get_const adds energy_nuc. The PBC value uses cell.ewald(), the
        molecular value uses bare 1/R. The difference must equal that
        Madelung offset (the rest of get_const — the trace of delta_h
        against the density matrix — is computed identically). Tolerance
        bounded by finite-cell error at L=8."""
        expected_diff = float(self.mf_pbc.energy_nuc()) - float(self.mol.energy_nuc())
        np.testing.assert_allclose(self.c_pbc - self.c_mol, expected_diff, atol=5e-3)


if __name__ == '__main__':
    unittest.main()
