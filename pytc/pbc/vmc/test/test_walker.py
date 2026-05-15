import unittest
import numpy as np
import jax.numpy as jnp
from jax import random
from pyscf.pbc import gto as pbcgto

from pytc.pbc.vmc.walker import initialize_walkers
from pytc.vmc.walker import Walker


class _MockAnsatz:
    """Minimal ansatz exposing the attributes required by initialize_walkers."""

    def __init__(self, atom_coords, atom_charges, n_electrons, n_alpha):
        self.atom_coords = np.asarray(atom_coords)
        self.atom_charges = np.asarray(atom_charges)
        self.n_electrons = n_electrons
        self.n_alpha = n_alpha


def _h2_cell():
    cell = pbcgto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 0.7'
    cell.basis = 'sto-3g'
    cell.a = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    cell.unit = 'B'
    cell.verbose = 0
    cell.build()
    return cell


class TestInitializeWalkers(unittest.TestCase):
    def setUp(self):
        self.cell = _h2_cell()
        self.ansatz = _MockAnsatz(
            atom_coords=self.cell.atom_coords(),
            atom_charges=self.cell.atom_charges(),
            n_electrons=self.cell.nelectron,
            n_alpha=1,
        )

    def test_positions_inside_cell(self):
        walker = initialize_walkers(
            self.ansatz, self.cell, n_walkers=64,
            key=random.PRNGKey(0), log_init=False,
        )
        self.assertIsInstance(walker, Walker)
        L = jnp.diag(jnp.asarray(self.cell.lattice_vectors()))
        self.assertTrue(bool(jnp.all(walker.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(walker.positions < L)))

    def test_shape(self):
        n_walkers = 32
        walker = initialize_walkers(
            self.ansatz, self.cell, n_walkers=n_walkers,
            key=random.PRNGKey(1), log_init=False,
        )
        self.assertEqual(walker.positions.shape, (n_walkers, 2, 3))

    def test_wraps_explicit_positions(self):
        """Passing positions outside the cell should fold them back in."""
        n_walkers = 8
        # Build positions in [10, 13)^3 — clearly outside [0, 3)^3
        positions = jnp.asarray(
            np.random.default_rng(0).uniform(10.0, 13.0, size=(n_walkers, 2, 3))
        )
        walker = initialize_walkers(
            self.ansatz, self.cell, n_walkers=n_walkers,
            initial_walkers=positions, key=random.PRNGKey(2), log_init=False,
        )
        L = jnp.diag(jnp.asarray(self.cell.lattice_vectors()))
        self.assertTrue(bool(jnp.all(walker.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(walker.positions < L)))

    def test_wraps_existing_walker(self):
        """A pre-built Walker passed in is re-wrapped on output."""
        walker = initialize_walkers(
            self.ansatz, self.cell, n_walkers=16,
            key=random.PRNGKey(3), log_init=False,
        )
        # Shift positions outside the cell by adding a lattice vector
        L = jnp.diag(jnp.asarray(self.cell.lattice_vectors()))
        shifted = walker.replace(positions=walker.positions + L * 2.0)
        re_wrapped = initialize_walkers(
            self.ansatz, self.cell, n_walkers=16,
            initial_walkers=shifted, log_init=False,
        )
        self.assertTrue(bool(jnp.all(re_wrapped.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(re_wrapped.positions < L)))
        # Wrapped positions equal original (modulo numerical tolerance) since
        # we added an integer number of lattice vectors.
        np.testing.assert_allclose(
            re_wrapped.positions, walker.positions, atol=1e-10
        )

    def test_wraps_atoms_outside_cell(self):
        """Atoms placed outside the cell should not produce out-of-cell electrons."""
        atom_coords = np.array([[15.0, -10.0, 100.0], [0.5, 0.5, 0.5]])
        ansatz = _MockAnsatz(
            atom_coords=atom_coords,
            atom_charges=np.array([1, 1]),
            n_electrons=2,
            n_alpha=1,
        )
        walker = initialize_walkers(
            ansatz, self.cell, n_walkers=32,
            key=random.PRNGKey(4), log_init=False,
        )
        L = jnp.diag(jnp.asarray(self.cell.lattice_vectors()))
        self.assertTrue(bool(jnp.all(walker.positions >= 0.0)))
        self.assertTrue(bool(jnp.all(walker.positions < L)))


if __name__ == '__main__':
    unittest.main()
