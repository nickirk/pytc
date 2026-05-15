"""Tests for the k-point Bloch-summed GTO evaluator.

The load-bearing test is the value-by-value cross-check against
``pyscf.pbc.dft.numint.eval_ao_kpts`` at a non-trivial k-mesh. This
pins down both the math and the sign convention.
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from pyscf.pbc import gto as pbcgto, dft as pbcdft

from pytc.pbc.ansatz.gto import GTO
from pytc.pbc.ansatz.kgto import KGTO, eval_ao, eval_gto, eval_gto_grad


def _h2_cell(L=4.0):
    cell = pbcgto.Cell()
    cell.atom = 'H 0 0 0; H 0 0 0.7'
    cell.basis = 'sto-3g'
    cell.a = [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]
    cell.unit = 'B'
    cell.cart = True
    cell.verbose = 0
    cell.build()
    return cell


class TestKGTOConstruction(unittest.TestCase):
    def test_build_at_gamma(self):
        cell = _h2_cell()
        kgto = KGTO.from_cell(cell, kpts=np.zeros((1, 3)))
        self.assertEqual(kgto.kpts.shape, (1, 3))
        self.assertGreater(kgto.images.shape[0], 1)

    def test_build_at_kmesh(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        kgto = KGTO.from_cell(cell, kpts=kpts)
        self.assertEqual(kgto.kpts.shape, (2, 3))

    def test_rejects_spherical_basis(self):
        cell = pbcgto.Cell()
        cell.atom = 'H 0 0 0'
        cell.basis = 'sto-3g'
        cell.a = [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]
        cell.unit = 'B'
        cell.cart = False
        cell.verbose = 0
        cell.build()
        with self.assertRaises(ValueError):
            KGTO.from_cell(cell, kpts=np.zeros((1, 3)))


class TestKGTOReducesToGammaGTO(unittest.TestCase):
    """At a single k-point at the origin, KGTO must return the same values
    as the Gamma-only GTO (modulo the leading k-axis and complex dtype)."""

    def test_values(self):
        cell = _h2_cell()
        rcut = 15.0
        gamma_gto = GTO.from_cell(cell, rcut=rcut)
        kgto = KGTO.from_cell(cell, kpts=np.zeros((1, 3)), rcut=rcut)

        rng = np.random.default_rng(0)
        pts = jnp.asarray(rng.uniform(-1.0, 1.0, size=(5, 3)))
        # Gamma values: (n_pts, n_ao) real
        gamma_vals = jax.vmap(
            lambda x: __import__('pytc.ansatz.gto', fromlist=['eval_gto']).eval_gto(gamma_gto, x)
        )(pts)
        # K values: (n_pts, 1, n_ao) complex; squeeze k-axis
        k_vals = eval_ao(kgto, pts, deriv=0)[:, 0, :]
        np.testing.assert_allclose(
            np.asarray(k_vals).real, np.asarray(gamma_vals), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(k_vals).imag, 0.0, atol=1e-12
        )


class TestKGTOMatchesPyscf(unittest.TestCase):
    """Value-level cross-check against ``pyscf.pbc.dft.numint.eval_ao_kpts``."""

    def _cross_check(self, kpts, rcut=15.0):
        cell = _h2_cell()
        kgto = KGTO.from_cell(cell, kpts=kpts, rcut=rcut)
        rng = np.random.default_rng(123)
        pts = rng.uniform(-1.0, 4.0, size=(6, 3))

        ours = np.asarray(eval_ao(kgto, jnp.asarray(pts), deriv=0))  # (n_pts, Nk, n_ao) complex
        pyscf_kpts = pbcdft.numint.eval_ao_kpts(cell, pts, kpts=kpts, deriv=0)
        # pyscf returns a list of (n_pts, n_ao) per k-point
        pyscf_arr = np.stack(pyscf_kpts, axis=1)  # (n_pts, Nk, n_ao)
        np.testing.assert_allclose(ours, pyscf_arr, atol=1e-9)

    def test_single_nontrivial_k(self):
        self._cross_check(kpts=np.array([[np.pi / 3.0, 0.0, 0.0]]))

    def test_kmesh_2x1x1(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        self._cross_check(kpts=kpts)

    def test_kmesh_2x2x1(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 2, 1])
        self._cross_check(kpts=kpts)


class TestKGTOBlochCondition(unittest.TestCase):
    """χ_k(r + a) = e^{i k·a} χ_k(r)."""

    def test_bloch_phase(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 2, 1])
        kgto = KGTO.from_cell(cell, kpts=kpts, rcut=20.0)
        lattice = jnp.asarray(cell.lattice_vectors())

        rng = np.random.default_rng(7)
        r = jnp.asarray(rng.uniform(0.0, 1.0, size=3))
        v0 = eval_gto(kgto, r)                                    # (Nk, n_ao)
        for a in [lattice[0], lattice[1], lattice[0] + lattice[1]]:
            v_shift = eval_gto(kgto, r + a)
            expected_phase = jnp.exp(1j * (kpts @ a))             # (Nk,)
            v_expected = v0 * expected_phase[:, None]
            np.testing.assert_allclose(
                np.asarray(v_shift), np.asarray(v_expected), atol=1e-7
            )

    def test_density_periodic(self):
        """|χ_k(r + a)|² == |χ_k(r)|² since the phase has unit modulus."""
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        kgto = KGTO.from_cell(cell, kpts=kpts, rcut=20.0)
        lattice = jnp.asarray(cell.lattice_vectors())
        rng = np.random.default_rng(11)
        r = jnp.asarray(rng.uniform(0.0, 1.0, size=3))
        d0 = jnp.abs(eval_gto(kgto, r)) ** 2
        for a in [lattice[0], lattice[2]]:
            d_shift = jnp.abs(eval_gto(kgto, r + a)) ** 2
            np.testing.assert_allclose(np.asarray(d_shift), np.asarray(d0), atol=1e-9)


class TestKGTODerivatives(unittest.TestCase):
    def test_grad_finite_and_shape(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        kgto = KGTO.from_cell(cell, kpts=kpts)
        r = jnp.array([0.3, 0.4, 0.5])
        g = eval_gto_grad(kgto, r)
        self.assertEqual(g.shape[-1], 3)
        self.assertEqual(g.shape[0], 2)              # Nk
        self.assertTrue(bool(jnp.all(jnp.isfinite(g.real))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(g.imag))))

    def test_grad_matches_finite_difference(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        kgto = KGTO.from_cell(cell, kpts=kpts, rcut=20.0)
        r = jnp.array([0.3, 0.4, 0.5])
        g = eval_gto_grad(kgto, r)                                # (Nk, n_ao, 3)
        eps = 1e-5
        for axis in range(3):
            dr = jnp.zeros(3).at[axis].set(eps)
            num = (eval_gto(kgto, r + dr) - eval_gto(kgto, r - dr)) / (2 * eps)
            np.testing.assert_allclose(
                np.asarray(g[..., axis]), np.asarray(num), atol=1e-7
            )


class TestKGTOJit(unittest.TestCase):
    def test_jit_compiles(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        kgto = KGTO.from_cell(cell, kpts=kpts)
        r = jnp.array([0.1, 0.2, 0.3])
        f = jax.jit(lambda x: eval_gto(kgto, x))
        np.testing.assert_allclose(
            np.asarray(f(r)), np.asarray(eval_gto(kgto, r)), atol=1e-12
        )


if __name__ == '__main__':
    unittest.main()
