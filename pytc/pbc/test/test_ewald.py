import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf.pbc import gto as pbcgto

from pytc.pbc.vmc.ewald import (
    EwaldParams,
    make_ewald_params,
    ewald_self_energy,
    ewald_cross_energy,
    total_coulomb_energy,
)


def _build_cell(L=6.0, atoms='H 0 0 0; H 0 0 0.7', spin=0):
    cell = pbcgto.Cell()
    cell.atom = atoms
    cell.basis = 'sto-3g'
    cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    cell.unit = 'B'
    cell.cart = True
    cell.spin = spin
    cell.verbose = 0
    cell.build()
    return cell


class TestEwaldConstruction(unittest.TestCase):
    def test_params_shape(self):
        cell = _build_cell()
        ewald = make_ewald_params(cell.lattice_vectors())
        self.assertGreater(ewald.real_images.shape[0], 1)
        self.assertGreater(ewald.recip_vectors.shape[0], 0)
        self.assertGreater(ewald.alpha, 0.0)
        self.assertGreater(ewald.volume, 0.0)

    def test_origin_in_real_images(self):
        cell = _build_cell()
        ewald = make_ewald_params(cell.lattice_vectors())
        np.testing.assert_array_equal(ewald.real_images[0], np.zeros(3))

    def test_origin_not_in_reciprocal(self):
        cell = _build_cell()
        ewald = make_ewald_params(cell.lattice_vectors())
        norms = jnp.linalg.norm(ewald.recip_vectors, axis=-1)
        self.assertTrue(bool(jnp.all(norms > 0)))


class TestSelfEnergyAgainstPyscf(unittest.TestCase):
    """The PySCF reference: cell.ewald() implements Martin's formulation."""

    def _compare(self, cell, atol=1e-9):
        positions = jnp.asarray(cell.atom_coords())
        charges = jnp.asarray(cell.atom_charges(), dtype=jnp.float64)
        ewald = make_ewald_params(cell.lattice_vectors())
        pyscf_val = cell.ewald()
        ours = float(ewald_self_energy(positions, charges, ewald))
        np.testing.assert_allclose(ours, pyscf_val, atol=atol)

    def test_h2_in_cubic_cell(self):
        self._compare(_build_cell(L=6.0))

    def test_h2_in_small_cell(self):
        self._compare(_build_cell(L=3.0))

    def test_h2_in_large_cell(self):
        self._compare(_build_cell(L=20.0))

    def test_single_proton_wigner(self):
        """Single proton in a neutralizing background — should match the
        Wigner crystal Madelung constant for SC."""
        cell = _build_cell(L=3.0, atoms='H 0 0 0', spin=1)
        self._compare(cell)

    def test_h2o(self):
        cell = _build_cell(L=10.0, atoms='O 0 0 0; H 0.96 0 0; H -0.24 0.93 0')
        self._compare(cell, atol=1e-8)


class TestSelfEnergyProperties(unittest.TestCase):
    def setUp(self):
        self.cell = _build_cell(L=6.0)
        self.positions = jnp.asarray(self.cell.atom_coords())
        self.charges = jnp.asarray(self.cell.atom_charges(), dtype=jnp.float64)
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def test_alpha_independence(self):
        """The total Ewald energy must not depend on the splitting parameter
        within the chosen precision."""
        e_vals = []
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
            ewald = make_ewald_params(self.lattice, alpha=alpha, precision=1e-12)
            e_vals.append(float(ewald_self_energy(self.positions, self.charges, ewald)))
        for e in e_vals[1:]:
            np.testing.assert_allclose(e, e_vals[0], atol=1e-9)

    def test_translation_invariance(self):
        """Shifting all charges by a lattice vector leaves the energy invariant."""
        ewald = make_ewald_params(self.lattice)
        e0 = float(ewald_self_energy(self.positions, self.charges, ewald))
        for T in [self.lattice[0], self.lattice[1], self.lattice[0] + self.lattice[2]]:
            e_shift = float(ewald_self_energy(self.positions + T, self.charges, ewald))
            np.testing.assert_allclose(e_shift, e0, atol=1e-10)

    def test_rigid_translation_of_charges(self):
        """Shifting all charges by a non-lattice constant vector leaves the
        energy invariant (Coulomb only depends on differences)."""
        ewald = make_ewald_params(self.lattice)
        e0 = float(ewald_self_energy(self.positions, self.charges, ewald))
        shift = jnp.array([0.3, -0.1, 0.4])
        e_shift = float(ewald_self_energy(self.positions + shift, self.charges, ewald))
        np.testing.assert_allclose(e_shift, e0, atol=1e-10)


class TestCrossEnergy(unittest.TestCase):
    def test_alpha_independence(self):
        """Cross-energy is α-independent within precision (the splitting
        parameter is purely a numerical knob, not physical)."""
        cell = _build_cell(L=6.0)
        lattice = cell.lattice_vectors()
        r_a = jnp.array([[0.5, 0.5, 0.5]])
        r_b = jnp.array([[2.0, 2.0, 2.0]])
        q_a = jnp.array([1.0])
        q_b = jnp.array([-1.0])
        e_vals = []
        for alpha in [0.5, 1.0, 2.0, 3.0]:
            ewald = make_ewald_params(lattice, alpha=alpha, precision=1e-12)
            e_vals.append(float(ewald_cross_energy(r_a, q_a, r_b, q_b, ewald)))
        for e in e_vals[1:]:
            np.testing.assert_allclose(e, e_vals[0], atol=1e-9)

    def test_decomposition_matches_total(self):
        """Total energy via combined self-energy must match the sum of
        the three sub-pieces (V_ee + V_en + V_NN) computed separately."""
        cell = _build_cell(L=6.0, atoms='H 0 0 0; H 0 0 0.7')
        ewald = make_ewald_params(cell.lattice_vectors())

        # Place two electrons inside the cell (neutralising the 2 protons).
        elec = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        nuc = jnp.asarray(cell.atom_coords())
        nuc_q = jnp.asarray(cell.atom_charges(), dtype=jnp.float64)
        elec_q = -jnp.ones(elec.shape[0])

        total = float(total_coulomb_energy(elec, nuc, nuc_q, ewald))

        v_ee = float(ewald_self_energy(elec, elec_q, ewald))
        v_nn = float(ewald_self_energy(nuc, nuc_q, ewald))
        v_en = float(ewald_cross_energy(elec, elec_q, nuc, nuc_q, ewald))
        np.testing.assert_allclose(total, v_ee + v_nn + v_en, atol=1e-10)


class TestEwaldJax(unittest.TestCase):
    def setUp(self):
        self.cell = _build_cell(L=6.0)
        self.positions = jnp.asarray(self.cell.atom_coords())
        self.charges = jnp.asarray(self.cell.atom_charges(), dtype=jnp.float64)
        self.ewald = make_ewald_params(self.cell.lattice_vectors())

    def test_jit_self_energy(self):
        f = jax.jit(ewald_self_energy)
        e_jit = float(f(self.positions, self.charges, self.ewald))
        e_ref = float(ewald_self_energy(self.positions, self.charges, self.ewald))
        np.testing.assert_allclose(e_jit, e_ref, atol=1e-12)

    def test_grad_self_energy(self):
        """Gradient of the Ewald energy w.r.t. positions must be finite —
        this is what enters the local-energy potential matrix."""
        g = jax.grad(lambda r: ewald_self_energy(r, self.charges, self.ewald))(self.positions)
        self.assertEqual(g.shape, self.positions.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    def test_jit_cross_energy(self):
        r_a = jnp.array([[0.0, 0.0, 0.0]])
        r_b = jnp.array([[0.0, 0.0, 1.5]])
        q_a = jnp.array([1.0])
        q_b = jnp.array([-1.0])
        f = jax.jit(ewald_cross_energy)
        e_jit = float(f(r_a, q_a, r_b, q_b, self.ewald))
        e_ref = float(ewald_cross_energy(r_a, q_a, r_b, q_b, self.ewald))
        np.testing.assert_allclose(e_jit, e_ref, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
