"""Tests for *honest* Option B' Phase 3 — ΔU^{NL, Δu} 2-body correction.

The honest delivery keeps the second electron coordinate on the TC grid
(no density contraction) and produces a rank-4 ERI-style tensor that is
added to h2e, not h1e.  See ``pytc/xtc_ecp_du.py`` and §8 of
``_local/design/ecp_xtc_theory.md``.

Tests:
    1. All-electron molecules: get_2b_ecp_du returns zero.
    2. ECP molecule with NuclearCusp-only Jastrow -> Phase 3 zero.
    2b. Composite with only NuclearCusp -> Phase 3 zero.
    3. No-NuclearCusp Jastrow (REXP alone) on Be/ccECP: ΔU finite,
       Phase 1 zero (no NCusp).
    4. REXP + NuclearCusp on Be/ccECP-VDZ: smoke test, finite,
       reasonable magnitude.
    5. (pq|rs) ↔ (rs|pq) symmetrisation works (the kernel transposes
       internally, so the returned tensor is symmetric under
       (pq) ↔ (rs)).
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import CompositeJastrow, NuclearCusp, REXP


def _make_be_ccecp(basis="ccecp-cc-pvdz"):
    mol = gto.M(
        atom="Be 0 0 0",
        basis=basis,
        ecp="ccecp",
        spin=0,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def _make_he_ae():
    mol = gto.M(
        atom="He 0 0 0",
        basis="cc-pvdz",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


class TestXtcEcpDu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    # --- 1. AE -> zero ---
    def test_no_ecp_returns_zero(self):
        mol, mf = _make_he_ae()
        ncusp = NuclearCusp.create(mol)
        rexp = REXP()
        jastrow = CompositeJastrow.create([ncusp, rexp])
        params = jastrow.init_params()
        params[1]["alpha"] = jnp.array([0.5])

        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=1)
        self.assertIsNone(my_xtc.ecp_data)
        delta_U = np.asarray(my_xtc.get_2b_ecp_du(mf, params))
        no = my_xtc.n_orb
        self.assertEqual(delta_U.shape, (no, no, no, no))
        np.testing.assert_allclose(delta_U, 0.0)

    # --- 2. NCusp-only Jastrow on ECP -> zero ---
    def test_ecp_ncusp_only_returns_zero(self):
        mol, mf = _make_be_ccecp()
        ncusp = NuclearCusp.create(mol)
        params = ncusp.init_params()
        my_xtc = xtc.XTC.from_pyscf(mf, ncusp, grid_lvl=1)
        self.assertIsNotNone(my_xtc.ecp_data)
        delta_U = np.asarray(my_xtc.get_2b_ecp_du(mf, params))
        no = my_xtc.n_orb
        self.assertEqual(delta_U.shape, (no, no, no, no))
        np.testing.assert_allclose(delta_U, 0.0)

    # --- 2b. Composite-NCusp-only -> zero ---
    def test_composite_ncusp_only_returns_zero(self):
        mol, mf = _make_be_ccecp()
        ncusp = NuclearCusp.create(mol)
        jastrow = CompositeJastrow.create([ncusp])
        params = jastrow.init_params()
        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=1)
        delta_U = np.asarray(my_xtc.get_2b_ecp_du(mf, params))
        np.testing.assert_allclose(delta_U, 0.0)

    # --- 3. REXP-only on Be/ccECP: ΔU finite, chi piece zero ---
    def test_rexp_only_on_be_ccecp(self):
        mol, mf = _make_be_ccecp()
        jastrow = REXP()
        params = {"alpha": jnp.array([0.5])}
        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=1)

        # Phase 1 -> zero (no NuclearCusp).
        delta_h_chi = np.asarray(my_xtc.get_1b_ecp_chi(mf, params))
        np.testing.assert_allclose(delta_h_chi, 0.0)

        # Phase 3 -> finite, nontrivial.
        delta_U = np.asarray(my_xtc.get_2b_ecp_du(mf, params))
        self.assertTrue(np.all(np.isfinite(delta_U)))
        fro = float(np.linalg.norm(delta_U))
        self.assertGreater(fro, 0.0)
        print(f"\n[Be/ccECP-VDZ, REXP-only] ||ΔU^{{NL,Δu}}||_F = {fro*1e3:.3f} mHa")

    # --- 4. REXP + NCusp on Be/ccECP: smoke test ---
    def test_be_ccecp_rexp_plus_ncusp_smoke(self):
        mol, mf = _make_be_ccecp()
        ncusp = NuclearCusp.create(mol)
        rexp = REXP()
        jastrow = CompositeJastrow.create([ncusp, rexp])
        params = jastrow.init_params()
        params[1]["alpha"] = jnp.array([0.5])

        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=2)
        delta_U = np.asarray(my_xtc.get_2b_ecp_du(mf, params))

        self.assertTrue(np.all(np.isfinite(delta_U)))
        fro_mha = float(np.linalg.norm(delta_U) * 1e3)
        max_mha = float(np.max(np.abs(delta_U)) * 1e3)
        print(f"\n[Be/ccECP-VDZ, REXP+NCusp] ||ΔU^{{NL,Δu}}||_F = {fro_mha:.3f} mHa, "
              f"max|elt| = {max_mha:.3f} mHa")
        self.assertLess(max_mha, 1e3, f"unexpectedly large |ΔU|_max = {max_mha} mHa")

    # --- 5. (pq) ↔ (rs) symmetrisation ---
    def test_pq_rs_symmetry(self):
        mol, mf = _make_be_ccecp()
        jastrow = REXP()
        params = {"alpha": jnp.array([0.5])}
        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=1)
        delta_U = np.asarray(my_xtc.get_2b_ecp_du(mf, params))

        # The kernel symmetrises internally by adding the (rs)<->(pq)
        # transpose.  Check that swap is exact (modulo fp noise).
        sym_residual = delta_U - np.transpose(delta_U, (2, 3, 0, 1))
        fro = float(np.linalg.norm(delta_U))
        if fro > 1e-12:
            rel = float(np.linalg.norm(sym_residual) / fro)
            print(f"\n[Be/ccECP-VDZ, REXP] rel (pq)<->(rs) residual = {rel:.2e}")
            self.assertLess(rel, 1e-10)


if __name__ == "__main__":
    unittest.main()
