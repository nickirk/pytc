import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf.pbc import gto as pbcgto, scf as pbcscf

from pytc.pbc.ansatz import create_slater_det
from pytc.pbc.ansatz.gto import PBCGTO
from pytc.ansatz.det import (
    SlaterDet,
    eval_det_value,
    eval_det_value_and_grad,
    eval_single_electron_ao,
    rank1_update_one_electron,
)
from pytc.vmc.walker import initialize_walker_state


def _h2_pbc_rhf(L=6.0, rcut=None):
    cell = pbcgto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 0.7'
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


class TestCreateSlaterDet(unittest.TestCase):
    def test_construct_from_pbc_rhf(self):
        cell, mf = _h2_pbc_rhf()
        det = create_slater_det(cell, mo_coeff=mf.mo_coeff, rcut=8.0)
        self.assertIsInstance(det, SlaterDet)
        self.assertIsInstance(det.mol_gto, PBCGTO)
        self.assertEqual(det.n_alpha, 1)
        self.assertEqual(det.n_beta, 1)

    def test_atom_data_carried_through(self):
        cell, mf = _h2_pbc_rhf()
        det = create_slater_det(cell, mo_coeff=mf.mo_coeff, rcut=8.0)
        np.testing.assert_allclose(det.atom_coords, cell.atom_coords())
        np.testing.assert_allclose(det.atom_charges, cell.atom_charges())

    def test_rcut_propagates(self):
        cell, mf = _h2_pbc_rhf()
        det_small = create_slater_det(cell, mo_coeff=mf.mo_coeff, rcut=3.0)
        det_large = create_slater_det(cell, mo_coeff=mf.mo_coeff, rcut=10.0)
        self.assertGreater(det_large.mol_gto.images.shape[0],
                           det_small.mol_gto.images.shape[0])


class TestPBCDetValues(unittest.TestCase):
    def setUp(self):
        self.cell, self.mf = _h2_pbc_rhf(L=6.0)
        # Use the default rcut: r_atom + cell_diam puts the periodicity
        # error at machine precision, so periodicity tests below verify the
        # algebra and not the truncation.
        self.det = create_slater_det(self.cell, mo_coeff=self.mf.mo_coeff)
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def _walker(self, positions):
        return initialize_walker_state(self.det, jnp.asarray(positions))

    def test_evaluation_finite(self):
        positions = jnp.array([[[0.1, 0.0, 0.0], [0.0, 0.0, 0.7]]])
        walker = self._walker(positions)
        _, walker = eval_det_value(self.det, walker)
        sign_up, logdet_up = walker.det_up
        sign_dn, logdet_dn = walker.det_down
        self.assertTrue(bool(jnp.all(jnp.isfinite(logdet_up))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logdet_dn))))
        self.assertTrue(bool(jnp.all((sign_up == 1.0) | (sign_up == -1.0))))

    def test_periodic_in_each_electron(self):
        """Shifting any one electron by a lattice vector must leave the
        determinant value invariant."""
        positions = jnp.array([[[0.3, 0.4, 0.2], [0.1, 0.0, 0.7]]])
        _, walker0 = eval_det_value(self.det, self._walker(positions))
        sign0, logd0 = walker0.det_up
        sign0_b, logd0_b = walker0.det_down

        for elec in range(2):
            for T in [self.lattice[0], self.lattice[1], self.lattice[2]]:
                shifted = positions.at[0, elec].add(T)
                _, walker_shift = eval_det_value(self.det, self._walker(shifted))
                np.testing.assert_allclose(walker_shift.det_up[0], sign0, atol=1e-8)
                np.testing.assert_allclose(walker_shift.det_up[1], logd0, atol=1e-8)
                np.testing.assert_allclose(walker_shift.det_down[0], sign0_b, atol=1e-8)
                np.testing.assert_allclose(walker_shift.det_down[1], logd0_b, atol=1e-8)

    def test_reduces_to_molecular_in_large_cell(self):
        """With a cell large enough that the orbital tails do not overlap
        their images, the PBC determinant should equal the molecular one."""
        from pyscf import gto as molgto, scf as molscf
        from pytc.ansatz.det import SlaterDet as MolSlaterDet

        mol = molgto.Mole()
        mol.atom = 'H 0 0 0; H 0 0 0.7'
        mol.basis = 'sto-3g'
        mol.unit = 'B'
        mol.cart = True
        mol.verbose = 0
        mol.build()
        mf_mol = molscf.RHF(mol)
        mf_mol.kernel()

        cell, mf_pbc = _h2_pbc_rhf(L=20.0)
        # Sign of mo_coeff can flip between mol and pbc SCFs; match by
        # using the PBC mo_coeff and constructing both dets from it.
        pbc_det = create_slater_det(cell, mo_coeff=mf_pbc.mo_coeff, rcut=1.0)
        mol_det = MolSlaterDet.create(mol, mo_coeff=mf_pbc.mo_coeff)

        positions = jnp.array([[[0.1, 0.2, 0.3], [0.0, 0.0, 0.7]]])
        _, w_pbc = eval_det_value(pbc_det, initialize_walker_state(pbc_det, positions))
        _, w_mol = eval_det_value(mol_det, initialize_walker_state(mol_det, positions))
        np.testing.assert_allclose(w_pbc.det_up[1], w_mol.det_up[1], atol=1e-10)
        np.testing.assert_allclose(w_pbc.det_down[1], w_mol.det_down[1], atol=1e-10)


class TestPBCRank1Update(unittest.TestCase):
    """rank1_update_one_electron is generic in the orbital evaluator. With a
    PBCGTO it should still produce a Slater determinant ratio consistent
    with a full re-evaluation at the new configuration."""

    def setUp(self):
        self.cell, self.mf = _h2_pbc_rhf(L=6.0)
        # Default rcut for the periodicity-under-wrap test below.
        self.det = create_slater_det(self.cell, mo_coeff=self.mf.mo_coeff)

    def _unbatched_walker(self, positions_2d):
        """Build an unbatched walker from a (n_electrons, 3) array."""
        walker_b = initialize_walker_state(self.det, positions_2d[None, :, :])
        _, walker_b = eval_det_value(self.det, walker_b)
        # Strip the batch dim to make it unbatched
        return jax.tree_util.tree_map(lambda x: x[0], walker_b)

    def test_rank1_ratio_matches_full_recompute(self):
        rng = np.random.default_rng(0)
        positions = jnp.asarray(rng.uniform(0.0, 6.0, size=(2, 3)))
        walker = self._unbatched_walker(positions)

        # Propose a new position for electron 0
        new_pos = positions + 1e-3  # small move, well-defined ratio
        new_positions = positions.at[0].set(new_pos[0])
        proposal = walker.replace(positions=new_positions)

        ratio, logdet_new, sign_new, _ = rank1_update_one_electron(
            self.det, proposal, electron_idx=0
        )

        # Full re-evaluation
        full_walker = self._unbatched_walker(new_positions)
        # Ratio = sign_new * exp(logdet_new) / (sign_old * exp(logdet_old))
        sign_old_up, logd_old_up = walker.det_up
        sign_new_up, logd_new_up = full_walker.det_up
        sign_old_dn, logd_old_dn = walker.det_down
        sign_new_dn, logd_new_dn = full_walker.det_down
        expected_total_ratio = (
            (sign_new_up * sign_new_dn) * jnp.exp(logd_new_up + logd_new_dn)
            / ((sign_old_up * sign_old_dn) * jnp.exp(logd_old_up + logd_old_dn))
        )
        # rank1_update returns the per-spin ratio for the moved electron,
        # which equals the total since the other spin's ratio is 1.
        np.testing.assert_allclose(ratio, expected_total_ratio, atol=1e-10)
        np.testing.assert_allclose(logdet_new, logd_new_up + logd_new_dn, atol=1e-10)

    def test_rank1_periodic_under_wrap(self):
        """A rank-1 update at position p and at position p + T (lattice
        vector) should give the same ratio."""
        rng = np.random.default_rng(1)
        positions = jnp.asarray(rng.uniform(0.0, 6.0, size=(2, 3)))
        walker = self._unbatched_walker(positions)
        lattice = jnp.asarray(self.cell.lattice_vectors())

        new_pos = positions[0] + jnp.array([0.3, 0.2, 0.1])
        walker_p = walker.replace(positions=positions.at[0].set(new_pos))
        ratio_p, _, _, _ = rank1_update_one_electron(
            self.det, walker_p, electron_idx=0
        )

        new_pos_T = new_pos + lattice[0]
        walker_pT = walker.replace(positions=positions.at[0].set(new_pos_T))
        ratio_pT, _, _, _ = rank1_update_one_electron(
            self.det, walker_pT, electron_idx=0
        )
        np.testing.assert_allclose(ratio_p, ratio_pT, atol=1e-7, rtol=1e-7)


class TestPBCDetGrad(unittest.TestCase):
    def test_value_and_grad_periodic(self):
        cell, mf = _h2_pbc_rhf(L=6.0)
        det = create_slater_det(cell, mo_coeff=mf.mo_coeff)
        lattice = jnp.asarray(cell.lattice_vectors())

        rng = np.random.default_rng(0)
        positions = jnp.asarray(rng.uniform(0.0, 6.0, size=(1, 2, 3)))
        walker = initialize_walker_state(det, positions)
        _, walker = eval_det_value_and_grad(det, walker)

        # Translate electron 0 by a1; gradients (for both electrons) must
        # be invariant because each MO is periodic.
        shifted = positions.at[0, 0].add(lattice[0])
        walker_s = initialize_walker_state(det, shifted)
        _, walker_s = eval_det_value_and_grad(det, walker_s)

        np.testing.assert_allclose(walker_s.grad_up, walker.grad_up, atol=1e-8)
        np.testing.assert_allclose(walker_s.grad_down, walker.grad_down, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
