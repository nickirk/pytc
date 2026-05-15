import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto as molgto
from pyscf.pbc import gto as pbcgto

from pytc.jastrow import BoysHandy as MolBoysHandy
from pytc.pbc.jastrow import BoysHandy


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


class TestPBCBoysHandyConstruction(unittest.TestCase):
    def test_build_from_cell(self):
        cell = _h2_cell()
        bh = BoysHandy.create(cell)
        np.testing.assert_allclose(bh.lattice, cell.lattice_vectors())
        np.testing.assert_allclose(bh.nuclear_pos, cell.atom_coords())
        np.testing.assert_allclose(bh.nuclear_charges, cell.atom_charges())
        self.assertEqual(bh.nelectron, cell.nelectron)
        self.assertGreater(bh.n_terms, 0)

    def test_init_params_shapes(self):
        cell = _h2_cell()
        bh = BoysHandy.create(cell)
        params = bh.init_params()
        self.assertEqual(params['b_raw'].shape, (bh.n_types,))
        self.assertEqual(params['d_raw'].shape, (bh.n_types,))
        self.assertEqual(params['c_raw'].shape, (bh.n_types, bh.n_terms))


class TestPBCBoysHandyValues(unittest.TestCase):
    def setUp(self):
        self.cell = _h2_cell(L=6.0)
        self.bh = BoysHandy.create(self.cell)
        self.params = self.bh.init_params()
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def test_compute_finite(self):
        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        val = self.bh._compute(r1, r2, self.params)
        self.assertTrue(bool(jnp.isfinite(val)))

    def test_periodic_in_r1(self):
        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        v0 = self.bh._compute(r1, r2, self.params)
        for T in [self.lattice[0], self.lattice[1], self.lattice[2],
                  self.lattice[0] - 2 * self.lattice[2]]:
            v_shift = self.bh._compute(r1 + T, r2, self.params)
            np.testing.assert_allclose(v_shift, v0, atol=1e-12)

    def test_periodic_in_r2(self):
        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        v0 = self.bh._compute(r1, r2, self.params)
        for T in [self.lattice[0], self.lattice[1], self.lattice[2]]:
            v_shift = self.bh._compute(r1, r2 + T, self.params)
            np.testing.assert_allclose(v_shift, v0, atol=1e-12)

    def test_mic_ee_uses_nearest_image(self):
        """Electrons on opposite faces of the cell should be paired by MIC,
        so the Jastrow value matches the case where r2 is wrapped close to r1."""
        r1 = jnp.array([0.05, 0.0, 0.0])
        r2_far = jnp.array([5.95, 0.0, 0.0])     # bare separation 5.9
        r2_near = jnp.array([-0.05, 0.0, 0.0])    # MIC-equivalent (close to r1)
        v_far = self.bh._compute(r1, r2_far, self.params)
        v_near = self.bh._compute(r1, r2_near, self.params)
        # Both should be the same since MIC folds r2_far → r2_near for the e-e term.
        np.testing.assert_allclose(v_far, v_near, atol=1e-12)

    def test_mic_en_uses_nearest_nucleus_image(self):
        """An electron at +x face sees the nucleus at the origin via its +a1
        image. Shifting r1 by a lattice vector and keeping r2 fixed must
        leave the Jastrow value invariant — this is the same as the
        periodicity-in-r1 test but at a cell-boundary configuration."""
        r1 = jnp.array([5.95, 1.0, 1.0])      # near +x face
        r2 = jnp.array([1.0, 1.0, 1.0])        # interior; e-e MIC well-defined
        v_face = self.bh._compute(r1, r2, self.params)

        # Translate r1 by a1: the wrapped position (-0.05, 1, 1) sees the
        # origin nucleus at distance 0.05 (without wrap, the bare distance
        # would be 5.95). MIC should make these two evaluations identical.
        r1_wrapped = r1 - self.lattice[0]
        v_wrapped = self.bh._compute(r1_wrapped, r2, self.params)
        np.testing.assert_allclose(v_face, v_wrapped, atol=1e-12)

    def test_matches_molecular_in_large_cell(self):
        """Inside a huge cell with MIC ≡ bare, periodic Boys-Handy must match
        the molecular one."""
        big_cell = _h2_cell(L=100.0)
        pbc_bh = BoysHandy.create(big_cell)
        mol = _h2_mol()
        mol_bh = MolBoysHandy.create(mol)

        params = pbc_bh.init_params()
        # Same construction → identical molecular params (up to dataclass
        # shape; values should be identical).
        np.testing.assert_allclose(params['b_raw'], mol_bh.init_params()['b_raw'])
        np.testing.assert_allclose(params['c_raw'], mol_bh.init_params()['c_raw'])

        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        np.testing.assert_allclose(
            pbc_bh._compute(r1, r2, params),
            mol_bh._compute(r1, r2, mol_bh.init_params()),
            atol=1e-12,
        )


class TestPBCBoysHandyGrads(unittest.TestCase):
    def setUp(self):
        self.cell = _h2_cell(L=6.0)
        self.bh = BoysHandy.create(self.cell)
        self.params = self.bh.init_params()
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def test_log_grads_r1_finite(self):
        """get_log_grads_r1 (folx forward-Laplacian) returns finite ∇ and Δ.

        This implicitly verifies that mic_displacement plays nicely with
        folx's higher-order autodiff — the previous custom_jvp fix on
        ``_wrap_to_half`` is what keeps this path fast and correct."""
        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        grad, lap = self.bh.get_log_grads_r1(r1, r2, self.params)
        self.assertEqual(grad.shape, (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertTrue(bool(jnp.isfinite(lap)))

    def test_gradient_periodic_in_r1(self):
        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        g0, _ = self.bh.get_log_grads_r1(r1, r2, self.params)
        for T in [self.lattice[0], self.lattice[1], self.lattice[2]]:
            g_shift, _ = self.bh.get_log_grads_r1(r1 + T, r2, self.params)
            np.testing.assert_allclose(g_shift, g0, atol=1e-10)

    def test_laplacian_periodic_in_r1(self):
        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        _, lap0 = self.bh.get_log_grads_r1(r1, r2, self.params)
        for T in [self.lattice[0], self.lattice[1], self.lattice[2]]:
            _, lap_s = self.bh.get_log_grads_r1(r1 + T, r2, self.params)
            np.testing.assert_allclose(lap_s, lap0, atol=1e-10)

    def test_jit_compatible(self):
        r1 = jnp.array([0.5, 0.3, 0.4])
        r2 = jnp.array([1.0, 0.0, 0.7])
        f = jax.jit(lambda a, b: self.bh._compute(a, b, self.params))
        np.testing.assert_allclose(
            f(r1, r2), self.bh._compute(r1, r2, self.params), atol=1e-12
        )


if __name__ == '__main__':
    unittest.main()
