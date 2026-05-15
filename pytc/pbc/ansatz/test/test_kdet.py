"""Tests for the k-point Slater determinant.

Validation has two pieces:
  * Gamma-reduction: at ``Nk = 1`` the KSlaterDet |det| matches the
    existing :func:`create_slater_det` |det| to machine precision.
  * Supercell equivalence: at general k-mesh the |det| matches the
    |det| of a supercell-Gamma SlaterDet built on the Nk-replicated
    cell. This is the load-bearing correctness check — the two
    approaches use different orbitals but span the same subspace, so
    the absolute determinant must agree.
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from pyscf.pbc import gto as pbcgto, scf as pbcscf, tools as pbctools

from pytc.ansatz.det import eval_det_value
from pytc.vmc.walker import initialize_walker_state

from pytc.pbc.ansatz import (
    create_slater_det,
    create_slater_det_kpts,
    KSlaterDet,
)
from pytc.pbc.ansatz.kdet import eval_kdet_value, eval_slater_matrix


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


class TestConstruction(unittest.TestCase):
    def test_build_from_rhf(self):
        """Gamma-only RHF input should auto-wrap to list form."""
        cell = _h2_cell()
        mf = pbcscf.RHF(cell); mf.exxdiv = None; mf.kernel()
        kdet = create_slater_det_kpts(mf)
        self.assertIsInstance(kdet, KSlaterDet)
        self.assertEqual(kdet.n_kpts, 1)
        # H2 in STO-3G: 1 occupied band per k.
        self.assertEqual(kdet.n_alpha, 1)
        self.assertEqual(kdet.n_beta, 1)
        self.assertEqual(kdet.mo_coeff_kpts_alpha.dtype, jnp.complex128)
        self.assertEqual(kdet.mo_coeff_kpts_alpha.shape, (1, 2, 2))

    def test_build_from_krhf_gamma(self):
        cell = _h2_cell()
        mf = pbcscf.KRHF(cell, kpts=np.zeros((1, 3)))
        mf.exxdiv = None; mf.kernel()
        kdet = create_slater_det_kpts(mf)
        self.assertEqual(kdet.n_kpts, 1)
        self.assertEqual(kdet.n_alpha, 1)

    def test_build_from_krhf_kmesh(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        mf = pbcscf.KRHF(cell, kpts=kpts); mf.exxdiv = None; mf.kernel()
        kdet = create_slater_det_kpts(mf)
        self.assertEqual(kdet.n_kpts, 2)
        # With 1 occupied band per k, total alpha electrons = 2.
        self.assertEqual(kdet.n_alpha, 2)
        self.assertEqual(kdet.n_beta, 2)

    def test_rejects_spherical_basis(self):
        cell = pbcgto.Cell()
        cell.atom = 'H 0 0 0; H 0 0 0.7'
        cell.basis = 'sto-3g'
        cell.a = [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 6.0]]
        cell.unit = 'B'; cell.cart = False; cell.verbose = 0
        cell.build()
        mf = pbcscf.RHF(cell); mf.exxdiv = None; mf.kernel()
        with self.assertRaises(ValueError):
            create_slater_det_kpts(mf)


class TestSlaterMatrixShape(unittest.TestCase):
    def test_alpha_shape(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        mf = pbcscf.KRHF(cell, kpts=kpts); mf.exxdiv = None; mf.kernel()
        kdet = create_slater_det_kpts(mf)
        positions = jnp.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.7]])
        s = eval_slater_matrix(kdet, positions, spin='alpha')
        self.assertEqual(s.shape, (2, 2))
        self.assertEqual(s.dtype, jnp.complex128)


class TestGammaReduction(unittest.TestCase):
    """At Nk = 1, KSlaterDet log|det| must match the existing Gamma-only
    SlaterDet evaluated on the same MOs and configuration."""

    def setUp(self):
        self.cell = _h2_cell()
        self.mf = pbcscf.RHF(self.cell); self.mf.exxdiv = None
        self.mf.kernel()
        # KSlaterDet from KRHF Gamma-only (need complex/list path)
        mf_k = pbcscf.KRHF(self.cell, kpts=np.zeros((1, 3)))
        mf_k.exxdiv = None; mf_k.kernel()
        self.kdet = create_slater_det_kpts(mf_k)
        # Existing Gamma SlaterDet built from RHF
        self.det = create_slater_det(self.cell, mo_coeff=self.mf.mo_coeff)

    def test_logdet_matches(self):
        rng = np.random.default_rng(0)
        n_alpha = self.kdet.n_alpha           # = 1 for H2/STO-3G
        for _ in range(5):
            r_up = jnp.asarray(rng.uniform(-1, 1, size=(n_alpha, 3)))
            r_dn = jnp.asarray(rng.uniform(-1, 1, size=(n_alpha, 3)))
            sign_up_k, logabs_up_k, sign_dn_k, logabs_dn_k = eval_kdet_value(
                self.kdet, r_up, r_dn
            )
            # Build full positions array for the molecular evaluator
            positions = jnp.concatenate([r_up, r_dn], axis=0)[None, :, :]
            walker = initialize_walker_state(self.det, positions)
            _, walker = eval_det_value(self.det, walker)
            _, logabs_mol_up = walker.det_up
            _, logabs_mol_dn = walker.det_down
            np.testing.assert_allclose(
                float(logabs_up_k.real), float(logabs_mol_up[0]), atol=1e-10
            )
            np.testing.assert_allclose(
                float(logabs_dn_k.real), float(logabs_mol_dn[0]), atol=1e-10
            )
            # At Gamma the orbitals are real (apart from numerical noise);
            # the imag part of log|det| must be zero.
            np.testing.assert_allclose(float(logabs_up_k.imag), 0.0, atol=1e-12)


class TestSupercellEquivalence(unittest.TestCase):
    """Load-bearing: the |det| of a primitive-k-mesh KSlaterDet equals the
    |det| of a supercell-Gamma SlaterDet built on the Nk-replicated cell.

    Both span the same one-electron space; they differ at most by a
    unitary rotation, so the absolute determinant must agree.
    """

    def setUp(self):
        # Primitive cell with H2
        prim = pbcgto.Cell()
        prim.atom = 'H 0 0 0; H 0 0 0.7'
        prim.basis = 'sto-3g'
        prim.a = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]]
        prim.unit = 'B'; prim.cart = True; prim.verbose = 0
        prim.build()

        # 2x1x1 supercell
        nk = [2, 1, 1]
        sup = pbctools.super_cell(prim, nk)
        sup.cart = True; sup.verbose = 0; sup.build()

        # Primitive k-mesh KRHF
        kpts = prim.make_kpts(nk)
        mf_k = pbcscf.KRHF(prim, kpts=kpts); mf_k.exxdiv = None
        mf_k.kernel()

        # Supercell Gamma RHF
        mf_sup = pbcscf.RHF(sup); mf_sup.exxdiv = None
        mf_sup.kernel()

        self.kdet = create_slater_det_kpts(mf_k)
        self.det_sup = create_slater_det(sup, mo_coeff=mf_sup.mo_coeff)
        # Number of alpha electrons (= occupied bands * Nk for primitive,
        # = occupied bands of supercell for supercell). These must agree.
        self.assertEqual(self.kdet.n_alpha, self.det_sup.n_alpha)

    def test_logabs_det_matches(self):
        n_alpha = self.kdet.n_alpha
        rng = np.random.default_rng(7)
        for trial in range(3):
            r_up = jnp.asarray(rng.uniform(0, 4, size=(n_alpha, 3)))
            r_dn = jnp.asarray(rng.uniform(0, 4, size=(n_alpha, 3)))
            sign_up_k, logabs_up_k, sign_dn_k, logabs_dn_k = eval_kdet_value(
                self.kdet, r_up, r_dn
            )

            positions = jnp.concatenate([r_up, r_dn], axis=0)[None, :, :]
            walker = initialize_walker_state(self.det_sup, positions)
            _, walker = eval_det_value(self.det_sup, walker)
            _, logabs_sup_up = walker.det_up
            _, logabs_sup_dn = walker.det_down

            np.testing.assert_allclose(
                float(logabs_up_k.real), float(logabs_sup_up[0]),
                atol=1e-7,
                err_msg=f"alpha block mismatch on trial {trial}",
            )
            np.testing.assert_allclose(
                float(logabs_dn_k.real), float(logabs_sup_dn[0]),
                atol=1e-7,
                err_msg=f"beta block mismatch on trial {trial}",
            )


class TestKDetJit(unittest.TestCase):
    def test_eval_jit_compiles(self):
        cell = _h2_cell()
        kpts = cell.make_kpts([2, 1, 1])
        mf = pbcscf.KRHF(cell, kpts=kpts); mf.exxdiv = None; mf.kernel()
        kdet = create_slater_det_kpts(mf)
        r_up = jnp.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.7]])
        r_dn = jnp.array([[0.3, 0.2, 0.1], [0.0, 0.0, 1.2]])
        f = jax.jit(eval_kdet_value)
        out_jit = f(kdet, r_up, r_dn)
        out_ref = eval_kdet_value(kdet, r_up, r_dn)
        for a, b in zip(out_jit, out_ref):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
