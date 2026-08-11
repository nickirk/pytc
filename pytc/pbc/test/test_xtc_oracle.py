import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto as molecular_gto
from pyscf.pbc import gto as periodic_gto
from pyscf.pbc import scf as periodic_scf

from pytc import kmat
from pytc.integrals.tc import TC as MolecularTC
from pytc.jastrow import BoysHandy as MolecularBoysHandy
from pytc.jastrow import CompositeJastrow
from pytc.jastrow import NuclearCusp as MolecularNuclearCusp
from pytc.pbc.jastrow import BoysHandy, NuclearCusp
from pytc.pbc.tc import create_tc
from pytc.pbc.xtc import create_xtc

jax.config.update("jax_enable_x64", True)


def _cell(length=15.0, cart=True):
    cell = periodic_gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.4"
    cell.basis = "sto-3g"
    cell.a = np.eye(3) * length
    cell.unit = "B"
    cell.cart = cart
    cell.verbose = 0
    cell.build()
    return cell


def _mean_field(cell):
    mf = periodic_scf.RHF(cell)
    mf.exxdiv = None
    mf.kernel()
    return mf


def _molecule():
    molecule = molecular_gto.Mole()
    molecule.atom = "H 0 0 0; H 0 0 1.4"
    molecule.basis = "sto-3g"
    molecule.unit = "B"
    molecule.cart = True
    molecule.verbose = 0
    molecule.build()
    return molecule


class TestPeriodicJastrows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _cell(6.0)
        cls.lattice = jnp.asarray(cls.cell.lattice_vectors())

    def test_boys_handy_value_and_gradient_are_periodic(self):
        jastrow = BoysHandy.create(self.cell)
        params = jastrow.init_params()
        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        value = jastrow._compute(r1, r2, params)
        gradient, laplacian = jastrow.get_log_grads_r1(r1, r2, params)
        for translation in self.lattice:
            np.testing.assert_allclose(
                jastrow._compute(r1 + translation, r2, params), value, atol=1e-12
            )
            shifted_gradient, shifted_laplacian = jastrow.get_log_grads_r1(
                r1 + translation, r2, params
            )
            np.testing.assert_allclose(shifted_gradient, gradient, atol=1e-10)
            np.testing.assert_allclose(shifted_laplacian, laplacian, atol=1e-10)

    def test_nuclear_cusp_value_and_gradient_are_periodic(self):
        jastrow = NuclearCusp.create(self.cell, n_radial=200)
        params = jastrow.init_params()
        r1 = jnp.array([0.05, 0.0, 0.0])
        r2 = jnp.zeros(3)
        value = jastrow._compute(r1, r2, params)
        gradient, laplacian = jastrow.get_log_grads_r1(r1, r2, params)
        for translation in self.lattice:
            np.testing.assert_allclose(
                jastrow._compute(r1 + translation, r2, params), value, atol=1e-12
            )
            shifted_gradient, shifted_laplacian = jastrow.get_log_grads_r1(
                r1 + translation, r2, params
            )
            np.testing.assert_allclose(shifted_gradient, gradient, atol=1e-10)
            np.testing.assert_allclose(shifted_laplacian, laplacian, atol=1e-10)


class TestGammaOracle(unittest.TestCase):
    def test_rejects_spherical_basis(self):
        cell = _cell(6.0, cart=False)
        mf = _mean_field(cell)
        with self.assertRaisesRegex(ValueError, "Cartesian"):
            create_tc(mf, NuclearCusp.create(cell, n_radial=100), grid_lvl=0)

    def test_rejects_non_gamma_mean_field(self):
        cell = _cell(6.0)
        mf = periodic_scf.RHF(cell, kpt=np.array([0.1, 0.0, 0.0]))
        mf.kernel()
        with self.assertRaisesRegex(NotImplementedError, "Gamma"):
            create_tc(mf, BoysHandy.create(cell), grid_lvl=0)

    def test_xtc_uses_periodic_nuclear_energy(self):
        cell = _cell(6.0)
        mf = _mean_field(cell)
        xtc = create_xtc(mf, BoysHandy.create(cell), grid_lvl=0)
        np.testing.assert_allclose(xtc.energy_nuc, mf.energy_nuc(), atol=1e-12)
        np.testing.assert_array_equal(np.asarray(xtc.mo_occ), np.asarray(mf.mo_occ))


class TestLargeCellParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _cell(15.0)
        cls.mf = _mean_field(cls.cell)
        cls.molecule = _molecule()
        coefficients = cls.mf.mo_coeff

        periodic_jastrow = CompositeJastrow.create(
            [
                NuclearCusp.create(cls.cell, n_radial=200),
                BoysHandy.create(cls.cell),
            ]
        )
        molecular_jastrow = CompositeJastrow.create(
            [
                MolecularNuclearCusp.create(cls.molecule, n_radial=200),
                MolecularBoysHandy.create(cls.molecule),
            ]
        )
        cls.periodic_jastrow = periodic_jastrow
        cls.molecular_jastrow = molecular_jastrow
        cls.periodic_tc = create_tc(
            cls.mf, periodic_jastrow, mo_coeff=coefficients, grid_lvl=2
        )
        mf_shim = type(
            "MeanFieldShim",
            (),
            {
                "mol": cls.molecule,
                "mo_coeff": coefficients,
                "mo_occ": cls.mf.mo_occ,
            },
        )()
        cls.molecular_tc = MolecularTC.from_pyscf(
            mf_shim,
            molecular_jastrow,
            mo_coeff=coefficients,
            grid_lvl=2,
        )
        cls.periodic_params = periodic_jastrow.init_params()
        cls.molecular_params = molecular_jastrow.init_params()

    def test_k1_matches_molecular_oracle(self):
        periodic = kmat.calc_K1(
            self.periodic_tc.phi,
            self.periodic_tc.grad_phi,
            self.periodic_jastrow,
            self.periodic_params,
            self.periodic_tc.grid_points,
            self.periodic_tc.weights,
            batch_size=500,
        )
        molecular = kmat.calc_K1(
            self.molecular_tc.phi,
            self.molecular_tc.grad_phi,
            self.molecular_jastrow,
            self.molecular_params,
            self.molecular_tc.grid_points,
            self.molecular_tc.weights,
            batch_size=500,
        )
        np.testing.assert_allclose(periodic, molecular, atol=1e-6)

    def test_k3_matches_molecular_oracle(self):
        periodic = kmat.calc_K3(
            self.periodic_tc.phi,
            self.periodic_jastrow,
            self.periodic_params,
            self.periodic_tc.grid_points,
            self.periodic_tc.weights,
            batch_size=500,
        )
        molecular = kmat.calc_K3(
            self.molecular_tc.phi,
            self.molecular_jastrow,
            self.molecular_params,
            self.molecular_tc.grid_points,
            self.molecular_tc.weights,
            batch_size=500,
        )
        np.testing.assert_allclose(periodic, molecular, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
