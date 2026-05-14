"""Tests for spherical-GTO evaluation at high angular momentum.

The previous hand-coded implementation in ``gto_spherical.py`` covered only
l ≤ 3 and silently returned zeros for l ≥ 4 — producing wrong AO values for
any basis set containing g, h, ... shells (cc-pVQZ on TM atoms, cc-pV5Z on
first-row atoms, etc.).  The refactor uses Cartesian monomials + PySCF's
``cart2sph`` transformation, which handles any l by construction.

These tests verify exact agreement with PySCF's own ``GTOval_sph`` for
synthetic single-l shells (l=0..5) and for realistic basis sets that include
g and h shells (H2O / bfd-v5z, Cu / cc-pVTZ).
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from pytc.ansatz.gto_spherical import MolGTO_Spherical, eval_ao_spherical


class TestSingleLShells(unittest.TestCase):
    """Synthetic single-l shells on a single atom, compared against PySCF."""

    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        rng = np.random.default_rng(0)
        # 16 evaluation points: roughly within atomic radii, including
        # near-origin and far-out cases so r^l differences are visible.
        self.coords = rng.uniform(-2.5, 2.5, size=(16, 3))

    def _check_l(self, l, alpha=1.0, atom="H", spin=None):
        """Build a single-shell basis with one primitive of exponent alpha
        and compare pytc's eval_ao_spherical against pyscf's GTOval_sph."""
        basis = {atom: [[l, [alpha, 1.0]]]}
        kwargs = dict(atom=f"{atom} 0 0 0", basis=basis, unit="Bohr", verbose=0)
        if spin is not None:
            kwargs["spin"] = spin
        # We need enough electrons (or accept 'unphysical' spin) — pyscf does
        # not actually require an SCF here, just basis evaluation.
        try:
            mol = gto.M(**kwargs, spin=0)
        except RuntimeError:
            mol = gto.M(**kwargs, spin=1)

        ref = mol.eval_gto("GTOval_sph", self.coords)  # (16, 2l+1)
        mol_gto = MolGTO_Spherical.create(mol)
        got = np.asarray(
            eval_ao_spherical(mol_gto, jnp.asarray(self.coords), deriv=0)
        )
        np.testing.assert_allclose(
            got, ref, atol=1e-12, rtol=1e-10,
            err_msg=f"l={l}: pytc spherical GTO mismatch vs pyscf",
        )

    def test_l0(self): self._check_l(0)
    def test_l1(self): self._check_l(1)
    def test_l2(self): self._check_l(2)
    def test_l3(self): self._check_l(3)
    def test_l4(self): self._check_l(4)  # g — used to be silently zero
    def test_l5(self): self._check_l(5)  # h — used to be silently zero


class TestRealBasesWithHighL(unittest.TestCase):
    """Realistic basis sets that contain g and/or h shells."""

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def _compare_to_pyscf(self, mol, coords):
        ref = mol.eval_gto("GTOval_sph", coords)
        mol_gto = MolGTO_Spherical.create(mol)
        got = np.asarray(
            eval_ao_spherical(mol_gto, jnp.asarray(coords), deriv=0)
        )
        np.testing.assert_allclose(got, ref, atol=1e-10, rtol=1e-10)
        return ref, got

    def test_h2o_bfd_v5z(self):
        """H2O / bfd-v5z (O has max_l = 5: s,p,d,f,g,h).

        Earlier H2O/V5Z VMC numbers (commit 24ee418) used the buggy code that
        silently zeroed g and h shells; this test ensures they're correct
        from now on.
        """
        mol = gto.M(
            atom="O 0 0 0; H 0.7572 0 -0.5215; H -0.7572 0 -0.5215",
            basis={"O": "bfd-v5z", "H": "bfd-v5z"},
            ecp={"O": "bfd"},
            unit="Angstrom",
            spin=0,
            verbose=0,
        )
        # Sanity: confirm the basis actually contains g and h.
        max_l = max(mol.bas_angular(i) for i in range(mol.nbas))
        self.assertGreaterEqual(max_l, 4)
        rng = np.random.default_rng(11)
        coords = rng.uniform(-2.5, 2.5, size=(20, 3))
        self._compare_to_pyscf(mol, coords)

    def test_cu_ccecp_vtz(self):
        """Cu / ccECP / cc-pVTZ (Cu has max_l = 4: s,p,d,f,g).

        Earlier Cu/VTZ VMC numbers would have been wrong for the same reason.
        """
        mol = gto.M(
            atom="Cu 0 0 0",
            basis="ccecp-cc-pvtz",
            ecp="ccecp",
            spin=1,
            unit="Bohr",
            verbose=0,
        )
        max_l = max(mol.bas_angular(i) for i in range(mol.nbas))
        self.assertEqual(max_l, 4)
        rng = np.random.default_rng(23)
        coords = rng.uniform(-3.0, 3.0, size=(20, 3))
        self._compare_to_pyscf(mol, coords)


class TestGradAndLapAtHighL(unittest.TestCase):
    """Gradient and Laplacian paths must also be correct at high l.

    The refactored code uses JAX autograd through ``_eval_shell_group``;
    folx provides the forward-Laplacian. Both should match pyscf's analytic
    derivatives.
    """

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def _check_l_deriv(self, l):
        mol = gto.M(
            atom="H 0 0 0",
            basis={"H": [[l, [1.0, 1.0]]]},
            unit="Bohr",
            verbose=0,
            spin=1,
        )
        rng = np.random.default_rng(0)
        coords = rng.uniform(-2.0, 2.0, size=(8, 3))

        # pyscf reference: value, gradient, Laplacian.
        ref_val_grad = mol.eval_gto("GTOval_sph_deriv1", coords)  # (4, 8, 2l+1)
        ref_val = ref_val_grad[0]
        ref_grad = np.transpose(ref_val_grad[1:4], (1, 2, 0))      # (8, 2l+1, 3)

        ref_lap_arr = mol.eval_gto("GTOval_sph_deriv2", coords)    # (10, ...)
        # deriv2 returns d/dx2, d/dy2, d/dz2 at indices 4, 7, 9 (after diag).
        # The 10 components are [val, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz].
        ref_lap = ref_lap_arr[4] + ref_lap_arr[7] + ref_lap_arr[9]  # (8, 2l+1)

        mol_gto = MolGTO_Spherical.create(mol)
        got_val, got_grad, got_lap = eval_ao_spherical(
            mol_gto, jnp.asarray(coords), deriv=2,
        )
        np.testing.assert_allclose(
            np.asarray(got_val), ref_val, atol=1e-10, rtol=1e-10,
            err_msg=f"value mismatch at l={l}",
        )
        np.testing.assert_allclose(
            np.asarray(got_grad), ref_grad, atol=1e-9, rtol=1e-9,
            err_msg=f"gradient mismatch at l={l}",
        )
        np.testing.assert_allclose(
            np.asarray(got_lap), ref_lap, atol=1e-8, rtol=1e-8,
            err_msg=f"laplacian mismatch at l={l}",
        )

    def test_grad_lap_l2(self): self._check_l_deriv(2)
    def test_grad_lap_l3(self): self._check_l_deriv(3)
    def test_grad_lap_l4(self): self._check_l_deriv(4)
    def test_grad_lap_l5(self): self._check_l_deriv(5)


if __name__ == "__main__":
    unittest.main()
