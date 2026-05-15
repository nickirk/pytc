import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto as molgto
from pyscf.pbc import gto as pbcgto

from pytc.ansatz.gto import (
    MolGTO,
    eval_gto,
    eval_ao,
    eval_gto_grad,
    eval_gto_lap,
)
from pytc.pbc.ansatz.gto import GTO, default_rcut


def _h2_cell(L=3.0, atom_offset=0.0):
    cell = pbcgto.Cell()
    cell.atom = f'H 0 0 0; H 0 0 {0.7 + atom_offset}'
    cell.basis = 'sto-3g'
    cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    cell.unit = 'B'
    cell.cart = True
    cell.verbose = 0
    cell.build()
    return cell


def _matching_mol():
    """Open-boundary mol with same atoms as the H2 cell — used to check that
    a huge cell GTO reduces to the molecular evaluator."""
    mol = molgto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.7'
    mol.basis = 'sto-3g'
    mol.unit = 'B'
    mol.cart = True
    mol.verbose = 0
    mol.build()
    return mol


class TestGTOConstruction(unittest.TestCase):
    def test_rejects_spherical_basis(self):
        cell = pbcgto.Cell()
        cell.atom = 'H 0 0 0'
        cell.basis = 'sto-3g'
        cell.a = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        cell.unit = 'B'
        cell.cart = False
        cell.verbose = 0
        cell.build()
        with self.assertRaises(ValueError):
            GTO.from_cell(cell)

    def test_origin_image_included(self):
        cell = _h2_cell(L=3.0)
        gto_obj = GTO.from_cell(cell, rcut=0.5)
        images = np.asarray(gto_obj.images)
        # rcut < |a_i| means only the zero image survives
        self.assertEqual(images.shape, (1, 3))
        np.testing.assert_array_equal(images[0], np.zeros(3))

    def test_image_count_grows_with_rcut(self):
        cell = _h2_cell(L=3.0)
        small = GTO.from_cell(cell, rcut=2.5)
        large = GTO.from_cell(cell, rcut=6.0)
        self.assertGreater(large.images.shape[0], small.images.shape[0])

    def test_default_rcut_positive(self):
        cell = _h2_cell(L=3.0)
        rcut = default_rcut(cell, precision=1e-8)
        self.assertGreater(rcut, 0.0)
        # Should be at least the cell diagonal.
        L = jnp.array(cell.lattice_vectors())
        self.assertGreater(rcut, float(jnp.linalg.norm(L.sum(axis=0))))


class TestGTOValues(unittest.TestCase):
    def test_reduces_to_molecular_in_large_cell(self):
        """With a cell large enough that only the origin image contributes,
        GTO values must equal the molecular MolGTO values."""
        cell = _h2_cell(L=30.0)
        mol = _matching_mol()
        pbc_gto = GTO.from_cell(cell, rcut=1.0)  # only zero image
        mol_gto = MolGTO.create(mol)
        # The two should now have the same data
        self.assertEqual(pbc_gto.images.shape[0], 1)
        rng = np.random.default_rng(0)
        for _ in range(5):
            xyz = jnp.asarray(rng.uniform(-1.0, 1.0, size=3))
            np.testing.assert_allclose(
                eval_gto(pbc_gto, xyz), eval_gto(mol_gto, xyz), atol=1e-10
            )

    def test_periodicity(self):
        """The Bloch-summed orbital must satisfy φ(r + T) == φ(r) for any
        lattice vector T. Uses a tight rcut so the truncation error is well
        below the asserted tolerance."""
        cell = _h2_cell(L=3.0)
        gto_obj = GTO.from_cell(cell, rcut=25.0)
        lattice = jnp.asarray(cell.lattice_vectors())

        rng = np.random.default_rng(1)
        for _ in range(5):
            r = jnp.asarray(rng.uniform(0.0, 3.0, size=3))
            v0 = eval_gto(gto_obj, r)
            # Translate by single lattice vectors and by a combination
            for shift in [lattice[0], lattice[1], lattice[2],
                          lattice[0] - 2 * lattice[2]]:
                v_shifted = eval_gto(gto_obj, r + shift)
                np.testing.assert_allclose(v0, v_shifted, atol=1e-10, rtol=1e-10)

    def test_default_rcut_periodic_to_single_shift(self):
        """At the default rcut, single-lattice-vector translations should be
        periodic to better than ~1e-8 — this is the regime that matters for
        Monte Carlo wrap-arounds."""
        cell = _h2_cell(L=3.0)
        gto_obj = GTO.from_cell(cell)
        lattice = jnp.asarray(cell.lattice_vectors())
        rng = np.random.default_rng(10)
        r = jnp.asarray(rng.uniform(0.0, 3.0, size=3))
        v0 = eval_gto(gto_obj, r)
        for shift in [lattice[0], lattice[1], lattice[2]]:
            v_shifted = eval_gto(gto_obj, r + shift)
            np.testing.assert_allclose(v0, v_shifted, atol=1e-8, rtol=1e-8)

    def test_periodicity_of_gradient(self):
        cell = _h2_cell(L=3.0)
        gto_obj = GTO.from_cell(cell)
        lattice = jnp.asarray(cell.lattice_vectors())
        rng = np.random.default_rng(2)
        r = jnp.asarray(rng.uniform(0.0, 3.0, size=3))
        g0 = eval_gto_grad(gto_obj, r)
        g1 = eval_gto_grad(gto_obj, r + lattice[1])
        np.testing.assert_allclose(g0, g1, atol=1e-7, rtol=1e-7)

    def test_bloch_sum_matches_brute_force(self):
        """Compare GTO at a field point against an explicit Python sum
        over the molecular orbital evaluated at translated copies. Both
        sums use the same cutoff so they agree exactly (up to floating
        point)."""
        cell = _h2_cell(L=2.5)
        rcut = 15.0
        pbc_gto = GTO.from_cell(cell, rcut=rcut)

        mol = _matching_mol()
        mol_gto = MolGTO.create(mol)
        lattice = np.asarray(cell.lattice_vectors())

        rng = np.random.default_rng(3)
        r = jnp.asarray(rng.uniform(0.0, 2.5, size=3))

        # Match the GTO image sum exactly: iterate over the same
        # image translations and evaluate the single-image molecular orbital
        # at r - T (mol_gto's internal image is the origin only).
        ref = jnp.zeros_like(eval_gto(mol_gto, r))
        for T in np.asarray(pbc_gto.images):
            ref = ref + eval_gto(mol_gto, r - jnp.asarray(T))

        pbc_val = eval_gto(pbc_gto, r)
        np.testing.assert_allclose(pbc_val, ref, atol=1e-12, rtol=1e-12)

    def test_eval_ao_batched(self):
        """The high-level eval_ao with deriv=2 works on a batch of positions."""
        cell = _h2_cell(L=3.0)
        gto_obj = GTO.from_cell(cell)
        rng = np.random.default_rng(4)
        positions = jnp.asarray(rng.uniform(0.0, 3.0, size=(4, 2, 3)))
        vals, grads, laps = eval_ao(gto_obj, positions, deriv=2)
        self.assertEqual(vals.shape[:2], (4, 2))
        self.assertEqual(grads.shape[-1], 3)
        self.assertTrue(bool(jnp.all(jnp.isfinite(vals))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grads))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(laps))))


class TestGTOJit(unittest.TestCase):
    def test_jit_eval(self):
        cell = _h2_cell(L=3.0)
        gto_obj = GTO.from_cell(cell)
        r = jnp.array([0.5, 1.0, 1.5])
        f = jax.jit(eval_gto)
        np.testing.assert_allclose(
            f(gto_obj, r), eval_gto(gto_obj, r), atol=1e-12
        )

    def test_grad_compiles(self):
        cell = _h2_cell(L=3.0)
        gto_obj = GTO.from_cell(cell)
        r = jnp.array([0.5, 1.0, 1.5])
        # Sum reduction to get a scalar for grad.
        f = jax.jit(jax.grad(lambda x: jnp.sum(eval_gto(gto_obj, x))))
        g = f(r)
        self.assertEqual(g.shape, (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))


if __name__ == '__main__':
    unittest.main()
