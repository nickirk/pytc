import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto as molgto, scf as molscf
from pyscf.pbc import gto as pbcgto, scf as pbcscf

from pytc.ansatz.det import eval_det_value_and_grad
from pytc.ansatz.sj import SlaterJastrow
from pytc.jastrow import CompositeJastrow

from pytc.pbc.ansatz import create_slater_det
from pytc.pbc.jastrow import NuclearCusp, BoysHandy
from pytc.pbc.vmc import (
    initialize_walker_state,
    make_ewald_params,
    compute_single_walker_energy,
    eval_local_energy,
)


def _build_pbc_sj(L=6.0):
    """Build a PBC SlaterJastrow for H2 at L Bohr cubic cell."""
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

    det = create_slater_det(cell, mo_coeff=mf.mo_coeff)
    ncusp = NuclearCusp.create(cell, name='ncusp', n_radial=300)
    bh = BoysHandy.create(cell, name='bh')
    jastrow = CompositeJastrow.create([ncusp, bh])

    sj = SlaterJastrow.create(cell, jastrow=jastrow, dets=[det])
    ewald = make_ewald_params(cell.lattice_vectors())
    return cell, mf, sj, ewald


def _populate_walker(sj, positions):
    """Build a walker batched on a single config and fill det/grad caches."""
    pos_b = jnp.asarray(positions)[None, :, :]
    walker = initialize_walker_state(sj.dets[0], pos_b)
    _, walker = jax.vmap(lambda w: eval_det_value_and_grad(sj.dets[0], w))(walker)
    return walker


class TestPBCHamiltonianSmoke(unittest.TestCase):
    def setUp(self):
        self.cell, self.mf, self.sj, self.ewald = _build_pbc_sj(L=6.0)
        self.params = self.sj.init_params(jax.random.PRNGKey(0))

    def test_energy_finite(self):
        positions = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.7]])
        walker_b = _populate_walker(self.sj, positions)
        # Unbatch for single-walker function
        walker = jax.tree_util.tree_map(lambda x: x[0], walker_b)
        e = float(compute_single_walker_energy(self.sj, walker, self.params[0], self.ewald))
        self.assertTrue(np.isfinite(e))

    def test_eval_local_energy_returns_pair(self):
        positions = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.7]])
        walker_b = _populate_walker(self.sj, positions)
        walker = jax.tree_util.tree_map(lambda x: x[0], walker_b)
        e, w_out = eval_local_energy(self.sj, walker, self.params, self.ewald)
        self.assertTrue(np.isfinite(float(e)))
        # walker is passed through unchanged
        np.testing.assert_array_equal(w_out.positions, walker.positions)


class TestPBCHamiltonianProperties(unittest.TestCase):
    def setUp(self):
        self.cell, self.mf, self.sj, self.ewald = _build_pbc_sj(L=6.0)
        self.params = self.sj.init_params(jax.random.PRNGKey(0))
        self.lattice = jnp.asarray(self.cell.lattice_vectors())

    def _energy_at(self, positions, ewald=None):
        walker_b = _populate_walker(self.sj, positions)
        walker = jax.tree_util.tree_map(lambda x: x[0], walker_b)
        ewald = self.ewald if ewald is None else ewald
        return float(compute_single_walker_energy(
            self.sj, walker, self.params[0], ewald
        ))

    def test_translation_invariance_single_electron(self):
        """Shifting one electron by a lattice vector leaves the local
        energy invariant."""
        positions = jnp.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.7]])
        e0 = self._energy_at(positions)
        for T in [self.lattice[0], self.lattice[1], self.lattice[2]]:
            shifted = positions.at[0].add(T)
            e_shift = self._energy_at(shifted)
            np.testing.assert_allclose(e_shift, e0, atol=1e-8)

    def test_translation_invariance_whole_config(self):
        """Shifting the entire walker (electrons + an implied wrap) by a
        lattice vector leaves the local energy invariant."""
        positions = jnp.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.7]])
        e0 = self._energy_at(positions)
        for T in [self.lattice[0], self.lattice[1] + self.lattice[2]]:
            shifted = positions + T
            e_shift = self._energy_at(shifted)
            np.testing.assert_allclose(e_shift, e0, atol=1e-8)

    def test_alpha_independence(self):
        """The Ewald α is a numerical knob; the local energy must not
        depend on it within precision."""
        positions = jnp.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.7]])
        e_vals = []
        for alpha in [0.6, 1.0, 1.5, 2.0]:
            ewald = make_ewald_params(self.lattice, alpha=alpha, precision=1e-12)
            e_vals.append(self._energy_at(positions, ewald=ewald))
        for e in e_vals[1:]:
            np.testing.assert_allclose(e, e_vals[0], atol=1e-7)


class TestPBCHamiltonianLargeCellLimit(unittest.TestCase):
    """In a cell much larger than the molecule, the periodic local energy
    should approach the molecular local energy at the same configuration.

    Convergence is dominated by the leading multipole of the molecule's
    image-image interaction. For H2 with a centered electron config the
    dipole vanishes, so the leading correction is quadrupole and decays
    fast — but not infinitely fast at any finite L. We use a generous
    cell and loose tolerance.
    """

    def test_large_cell_matches_molecular(self):
        from pytc.ansatz.det import SlaterDet as MolSlaterDet
        from pytc.jastrow import NuclearCusp as MolNuclearCusp
        from pytc.jastrow import BoysHandy as MolBoysHandy
        from pytc.vmc.hamiltonian import (
            compute_single_walker_energy as mol_energy,
        )

        L = 30.0
        cell, mf_pbc, sj_pbc, ewald = _build_pbc_sj(L=L)

        # Matching molecular system at the SAME mo_coeff so the determinants
        # are identical (avoids spurious sign or normalization differences).
        mol = molgto.Mole()
        mol.atom = 'H 0 0 0; H 0 0 0.7'
        mol.basis = 'sto-3g'
        mol.unit = 'B'
        mol.cart = True
        mol.verbose = 0
        mol.build()

        # Use the PBC mo_coeff so the orbital amplitudes match at L>>0
        mol_det = MolSlaterDet.create(mol, mo_coeff=mf_pbc.mo_coeff)
        mol_ncusp = MolNuclearCusp.create(mol, name='ncusp', n_radial=300)
        mol_bh = MolBoysHandy.create(mol, name='bh')
        mol_jastrow = CompositeJastrow.create([mol_ncusp, mol_bh])
        sj_mol = SlaterJastrow.create(mol, jastrow=mol_jastrow, dets=[mol_det])

        params_pbc = sj_pbc.init_params(jax.random.PRNGKey(0))
        params_mol = sj_mol.init_params(jax.random.PRNGKey(0))

        # Place electrons near each H atom (typical VMC config)
        positions = jnp.array([[0.05, 0.0, 0.0], [0.0, 0.0, 0.65]])

        # PBC energy
        walker_b = _populate_walker(sj_pbc, positions)
        walker_pbc = jax.tree_util.tree_map(lambda x: x[0], walker_b)
        e_pbc = float(compute_single_walker_energy(
            sj_pbc, walker_pbc, params_pbc[0], ewald
        ))

        # Molecular energy at the same config
        from pytc.vmc.walker import initialize_walker_state as init_mol_walker
        walker_mol_b = init_mol_walker(sj_mol.dets[0], positions[None, :, :])
        _, walker_mol_b = jax.vmap(
            lambda w: eval_det_value_and_grad(sj_mol.dets[0], w)
        )(walker_mol_b)
        walker_mol = jax.tree_util.tree_map(lambda x: x[0], walker_mol_b)
        e_mol = float(mol_energy(sj_mol, walker_mol, params_mol[0]))

        # H2 is charge-neutral so periodic image contributions to the total
        # Coulomb decay quickly; at L=30 Bohr the PBC value should be very
        # close to the molecular one.
        np.testing.assert_allclose(e_pbc, e_mol, atol=5e-3)


class TestPBCHamiltonianJax(unittest.TestCase):
    def setUp(self):
        self.cell, self.mf, self.sj, self.ewald = _build_pbc_sj(L=6.0)
        self.params = self.sj.init_params(jax.random.PRNGKey(0))

    def test_jit_compatible(self):
        positions = jnp.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.7]])
        walker_b = _populate_walker(self.sj, positions)
        walker = jax.tree_util.tree_map(lambda x: x[0], walker_b)

        f = jax.jit(lambda w: compute_single_walker_energy(
            self.sj, w, self.params[0], self.ewald
        ))
        e_jit = float(f(walker))
        e_ref = float(compute_single_walker_energy(
            self.sj, walker, self.params[0], self.ewald
        ))
        np.testing.assert_allclose(e_jit, e_ref, atol=1e-10)

    def test_vmap_over_walkers(self):
        """Batched evaluation over walkers must work."""
        rng = jax.random.PRNGKey(1)
        positions = jax.random.uniform(rng, (4, 2, 3)) * 6.0
        walker_b = initialize_walker_state(self.sj.dets[0], positions)
        _, walker_b = jax.vmap(
            lambda w: eval_det_value_and_grad(self.sj.dets[0], w)
        )(walker_b)
        energies = jax.vmap(
            lambda w: compute_single_walker_energy(
                self.sj, w, self.params[0], self.ewald
            )
        )(walker_b)
        self.assertEqual(energies.shape, (4,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(energies))))


if __name__ == '__main__':
    unittest.main()
