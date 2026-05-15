import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto as molgto
from pyscf.pbc import gto as pbcgto

from pytc.jastrow import NuclearCusp as MolNuclearCusp
from pytc.pbc.jastrow import NuclearCusp


def _h2_cell(L=6.0):
    cell = pbcgto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 0.7'
    cell.basis = 'sto-3g'
    cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    cell.unit = 'B'
    cell.cart = True
    cell.verbose = 0
    cell.build()
    return cell


def _h2_mol():
    mol = molgto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.7'
    mol.basis = 'sto-3g'
    mol.unit = 'B'
    mol.cart = True
    mol.verbose = 0
    mol.build()
    return mol


class TestPBCNuclearCuspConstruction(unittest.TestCase):
    def test_build_from_cell(self):
        cell = _h2_cell()
        cusp = NuclearCusp.create(cell, n_radial=200)
        self.assertEqual(cusp.n_nuclei, 2)
        self.assertEqual(cusp.nelectron, 2)
        np.testing.assert_allclose(cusp.lattice, cell.lattice_vectors())
        np.testing.assert_allclose(cusp.coords, cell.atom_coords())
        np.testing.assert_allclose(cusp.charges, cell.atom_charges())

    def test_init_params_shapes(self):
        cell = _h2_cell()
        cusp = NuclearCusp.create(cell, n_radial=200)
        params = cusp.init_params()
        self.assertEqual(params['rc'].shape, (cusp.n_types,))
        self.assertEqual(params['X4'].shape, (cusp.n_types,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(params['rc']))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(params['X4']))))


class TestPBCNuclearCuspValues(unittest.TestCase):
    def setUp(self):
        self.cell = _h2_cell(L=6.0)
        self.cusp = NuclearCusp.create(self.cell, n_radial=500)
        self.params = self.cusp.init_params()
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def test_compute_finite_near_nucleus(self):
        # An electron close to atom 0
        r1 = jnp.array([0.05, 0.0, 0.0])
        r2 = jnp.array([1.5, 1.5, 1.5])  # dummy (NuclearCusp ignores r2)
        val = self.cusp._compute(r1, r2, self.params)
        self.assertTrue(bool(jnp.isfinite(val)))

    def test_compute_periodic_in_r1(self):
        """Translating the electron by a lattice vector must leave the
        cusp value invariant (each contribution depends only on the MIC
        distance to the nearest image of each nucleus)."""
        r1 = jnp.array([0.05, 0.0, 0.0])
        r2 = jnp.zeros(3)
        v0 = self.cusp._compute(r1, r2, self.params)
        for T in [self.lattice[0], self.lattice[1], self.lattice[2],
                  self.lattice[0] - 2 * self.lattice[2]]:
            v_shift = self.cusp._compute(r1 + T, r2, self.params)
            np.testing.assert_allclose(v_shift, v0, atol=1e-12)

    def test_uses_minimum_image_nucleus(self):
        """An electron near the +x face of the cell should see the nucleus
        at the origin via its image at +a1, not the bare nucleus far away."""
        # Place electron at (5.95, 0, 0) in a 6.0 Bohr cubic cell.
        # Bare distance to origin nucleus: 5.95; MIC: 0.05.
        r1 = jnp.array([5.95, 0.0, 0.0])
        r2 = jnp.zeros(3)
        val_pbc = self.cusp._compute(r1, r2, self.params)

        # Compare against the same Jastrow evaluated at the wrapped point.
        r1_wrapped = jnp.array([0.05, 0.0, 0.0])
        # The bare displacement is along -a1; MIC folds it to +0.05 along +x.
        val_at_wrap = self.cusp._compute(r1_wrapped, r2, self.params)

        # MIC means both give the same answer.
        np.testing.assert_allclose(val_pbc, val_at_wrap, atol=1e-12)

    def test_matches_molecular_in_large_cell(self):
        """Inside a huge cell, the MIC distance equals the raw distance and
        the periodic Jastrow value must equal the molecular one (for any
        electron well inside the cell, far from boundaries)."""
        big_cell = _h2_cell(L=100.0)
        pbc_cusp = NuclearCusp.create(big_cell, n_radial=500)
        mol = _h2_mol()
        mol_cusp = MolNuclearCusp.create(mol, n_radial=500)

        params_pbc = pbc_cusp.init_params()
        params_mol = mol_cusp.init_params()
        # Same molecular construction → same params
        np.testing.assert_allclose(params_pbc['rc'], params_mol['rc'])
        np.testing.assert_allclose(params_pbc['X4'], params_mol['X4'])

        r1 = jnp.array([0.1, 0.0, 0.0])
        r2 = jnp.array([1.0, 0.0, 0.7])
        np.testing.assert_allclose(
            pbc_cusp._compute(r1, r2, params_pbc),
            mol_cusp._compute(r1, r2, params_mol),
            atol=1e-12,
        )

    def test_far_from_nuclei_returns_zero(self):
        """Outside the cutoff rc (~1/Z, ~1 Bohr for H), the cusp correction
        is exactly zero."""
        # In a 6 Bohr cell with nuclei at 0 and 0.7 along z, the MIC
        # distance from (3, 3, 3) to both nuclei is ~ sqrt(9 + 9 + 9)=5.2,
        # but MIC folds to ~ sqrt(9+9+9) (already minimal) which is > 1.
        # Actually for cubic 6 cell, MIC fold to nearest image: e.g. nucleus
        # at (0,0,0) has nearest image (6,6,6) for r1=(3,3,3): both at 5.2.
        # MIC distance = 5.2 > rc ~1/Z = 1. So contribution should be 0.
        r1 = jnp.array([3.0, 3.0, 3.0])
        r2 = jnp.zeros(3)
        val = self.cusp._compute(r1, r2, self.params)
        np.testing.assert_allclose(val, 0.0, atol=1e-12)


class TestPBCNuclearCuspGrads(unittest.TestCase):
    def setUp(self):
        self.cell = _h2_cell(L=6.0)
        self.cusp = NuclearCusp.create(self.cell, n_radial=500)
        self.params = self.cusp.init_params()
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def test_log_grads_r1_finite(self):
        r1 = jnp.array([0.05, 0.0, 0.0])
        r2 = jnp.array([1.5, 1.5, 1.5])
        grad, lap = self.cusp.get_log_grads_r1(r1, r2, self.params)
        self.assertEqual(grad.shape, (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertTrue(bool(jnp.isfinite(lap)))

    def test_gradient_periodic(self):
        """∇u(r + T) == ∇u(r) under any lattice translation."""
        r1 = jnp.array([0.05, 0.0, 0.0])
        r2 = jnp.zeros(3)
        g0, _ = self.cusp.get_log_grads_r1(r1, r2, self.params)
        for T in [self.lattice[0], self.lattice[1], self.lattice[2]]:
            g_shift, _ = self.cusp.get_log_grads_r1(r1 + T, r2, self.params)
            np.testing.assert_allclose(g_shift, g0, atol=1e-10)

    def test_jit_compatible(self):
        r1 = jnp.array([0.05, 0.0, 0.0])
        r2 = jnp.array([1.5, 1.5, 1.5])

        f = jax.jit(lambda r: self.cusp._compute(r, r2, self.params))
        np.testing.assert_allclose(
            f(r1), self.cusp._compute(r1, r2, self.params), atol=1e-12
        )


if __name__ == '__main__':
    unittest.main()
