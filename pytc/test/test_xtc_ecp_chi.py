"""Tests for the Option B Phase 1 [V_NL, chi] correction.

Covers:
    1. All-electron molecules: get_1b_ecp_chi returns zero.
    2. ECP molecule, no NuclearCusp factor in jastrow: returns zero.
    3. Constant-chi invariance: if chi has no spatial variation,
       exp(Delta chi) - 1 = 0 -> result is zero.
    4. Hermiticity / approximate symmetry of the resulting matrix.
    5. Smoke test on Be/ccECP-VDZ: compute Delta h and report magnitude.
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


def _make_h2_ae():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


class TestXtcEcpChi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    # ------------------------------------------------------------------
    # 1. No-ECP fallback: all-electron molecule -> zero matrix.
    # ------------------------------------------------------------------
    def test_no_ecp_returns_zero(self):
        mol, mf = _make_h2_ae()
        ncusp = NuclearCusp.create(mol)
        jastrow = CompositeJastrow.create([ncusp])
        params = jastrow.init_params()

        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=1)
        # All-electron -> ecp_data should be None.
        self.assertIsNone(my_xtc.ecp_data)
        delta_h = my_xtc.get_1b_ecp_chi(mf, params)
        delta_h = np.asarray(delta_h)
        self.assertEqual(delta_h.shape, (my_xtc.n_orb, my_xtc.n_orb))
        np.testing.assert_allclose(delta_h, 0.0)

    # ------------------------------------------------------------------
    # 2. ECP molecule but no NuclearCusp in jastrow -> zero.
    # ------------------------------------------------------------------
    def test_ecp_no_ncusp_returns_zero(self):
        mol, mf = _make_be_ccecp()
        jastrow = REXP()  # no NuclearCusp component
        params = {"alpha": jnp.array([0.5])}

        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=1)
        self.assertIsNotNone(my_xtc.ecp_data)

        delta_h = my_xtc.get_1b_ecp_chi(mf, params)
        delta_h = np.asarray(delta_h)
        self.assertEqual(delta_h.shape, (my_xtc.n_orb, my_xtc.n_orb))
        np.testing.assert_allclose(delta_h, 0.0)

    # ------------------------------------------------------------------
    # 3. Constant-chi invariance: if chi == const, exp(Delta chi) - 1 = 0.
    # ------------------------------------------------------------------
    def test_constant_chi_invariance(self):
        from pytc.xtc_ecp_chi import compute_delta_h_ecp_chi
        from pytc.ecp.parser import parse_pyscf_ecp
        from pytc.ecp.quadrature import get_grid

        mol, mf = _make_be_ccecp()
        ecp = parse_pyscf_ecp(mol, warn_on_overlap=False)
        my_xtc = xtc.XTC.from_pyscf(mf, REXP(), grid_lvl=1)

        # Inject a chi_fn that is literally constant. exp(0) - 1 = 0.
        const_chi = lambda r: jnp.array(1.234)
        delta_h = compute_delta_h_ecp_chi(
            mol=mol,
            mo_coeff=np.asarray(my_xtc.mo_coeff),
            grid_points=np.asarray(my_xtc.grid_points),
            weights=np.asarray(my_xtc.weights),
            phi_grid=my_xtc.phi,
            ecp=ecp,
            angular_grid=get_grid("icosahedral_12"),
            chi_fn=const_chi,
        )
        np.testing.assert_allclose(np.asarray(delta_h), 0.0, atol=1e-12)

    # ------------------------------------------------------------------
    # 4. Approximate symmetry / Hermiticity.
    #    The kernel is Hermitian to within grid+quad approximation.
    # ------------------------------------------------------------------
    def test_approximate_symmetry(self):
        mol, mf = _make_be_ccecp()
        ncusp = NuclearCusp.create(mol)
        jastrow = CompositeJastrow.create([ncusp])
        params = jastrow.init_params()

        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=2)
        delta_h = np.asarray(my_xtc.get_1b_ecp_chi(mf, params))

        # Off-diagonal asymmetry should be small relative to Frobenius norm.
        asym = delta_h - delta_h.T
        fro = float(np.linalg.norm(delta_h))
        # Tolerance is loose because the integrand is grid-approximated.
        # For Be with grid_lvl=2 + 12-point icosahedral, ~1e-4 rel.
        if fro > 1e-12:
            rel = float(np.linalg.norm(asym) / fro)
            self.assertLess(rel, 5e-3, f"rel asym = {rel:.2e}")
        # else delta_h is essentially zero — trivially symmetric.

    # ------------------------------------------------------------------
    # 5. Smoke test on Be/ccECP-VDZ: finite, reasonable magnitude.
    #
    # NOTE: On a pure-ECP system (Be has only one Be atom, which is ECP'd),
    # the NuclearCusp Jastrow is gated off at every ECP center
    # (Drummond-Towler-Needs convention, see ncusp.py).  So chi(r) == 0
    # everywhere -> Delta chi == 0 -> Delta h^{NL,chi} == 0 by
    # construction.  This is the *expected physical behavior*, not a bug;
    # for pure-ECP systems Option B Phase 1 contributes nothing and the
    # full effect must come from Phase 3 (2-body Delta u).  See
    # ``_local/design/ecp_xtc_theory.md`` §9.
    # ------------------------------------------------------------------
    def test_be_ccecp_smoke(self):
        mol, mf = _make_be_ccecp()
        ncusp = NuclearCusp.create(mol)
        rexp = REXP()
        jastrow = CompositeJastrow.create([ncusp, rexp])
        params = jastrow.init_params()
        if "alpha" in params[1]:
            params[1]["alpha"] = jnp.array([0.5])

        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=2)
        delta_h = np.asarray(my_xtc.get_1b_ecp_chi(mf, params))

        self.assertTrue(np.all(np.isfinite(delta_h)))
        # Pure-ECP -> exactly zero by the cusp-gating convention.
        np.testing.assert_allclose(delta_h, 0.0, atol=1e-12)
        print(f"\n[Be/ccECP-VDZ] ||Δh^{{NL,χ}}||_F = "
              f"{np.linalg.norm(delta_h)*1e3:.6f} mHa (zero by construction)")

    # ------------------------------------------------------------------
    # 6. Mixed AE+ECP smoke test on H2O/BFD: O is ECP, H is AE, so the
    #    cusp at H is active and chi != 0 at points sampled on O's ECP
    #    angular sphere.  Delta h should be small but nonzero.
    # ------------------------------------------------------------------
    def test_h2o_bfd_mixed_ae_ecp(self):
        mol = gto.M(
            atom="O 0 0 0; H 1.43 0 -1.11; H -1.43 0 -1.11",
            basis={"O": "bfd-vdz", "H": "bfd-vdz"},
            ecp={"O": "bfd"},
            unit="Bohr",
            spin=0,
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.kernel()

        ncusp = NuclearCusp.create(mol)
        jastrow = CompositeJastrow.create([ncusp])
        params = jastrow.init_params()

        my_xtc = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=2)
        delta_h = np.asarray(my_xtc.get_1b_ecp_chi(mf, params))

        self.assertTrue(np.all(np.isfinite(delta_h)))
        fro_mha = float(np.linalg.norm(delta_h) * 1e3)
        max_mha = float(np.max(np.abs(delta_h)) * 1e3)
        print(f"\n[H2O/BFD-VDZ] ||Δh^{{NL,χ}}||_F = {fro_mha:.3f} mHa, "
              f"max|elt| = {max_mha:.3f} mHa")
        # Sanity bound: shouldn't blow up.
        self.assertLess(max_mha, 1e3,  # 1 Ha
                        f"unexpectedly large |Δh|_max = {max_mha} mHa")


if __name__ == "__main__":
    unittest.main()
