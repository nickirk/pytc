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

    # --- 6. Independent integration cross-check (regression guard) ---
    def test_kernel_matches_independent_integration(self):
        """The batched/padded scan kernel must agree, to machine precision,
        with a clean single-batch hand-coded integration that uses the same
        (e^{Δu} - 1) integrand.

        This guards against any future regression in the scan structure
        (batching, padding, einsum index orders, symmetrisation, etc.).
        The reference computes the (0,0,0,0) and (0,0,1,1) tensor entries
        explicitly on the full (small) grid via einsums that mirror the
        formula in `_local/design/ecp_xtc_theory.md` §8.1 line-for-line:

            ΔU_{pqrs} = ∫dr_1 dr_2 w_1 w_2 φ_p(r_1) Σ_{A,l,q_quad}
                        (2l+1) V_l^A P_l w_{q_quad}
                        (e^{u(r'_{1,A,q}, r_2) - u(r_1, r_2)} - 1)
                        φ_q(r'_{1,A,q}) φ_r(r_2) φ_s(r_2)

            (plus the (p,q) <-> (r,s) symmetrisation partner).

        Use grid_lvl=0 to keep N_grid small (~1000) so the O(N_grid^2)
        reference is cheap.
        """
        from pyscf.dft import numint
        from pytc.ecp.parser import parse_pyscf_ecp
        from pytc.ecp.quadrature import get_grid
        from pytc.ecp.radial import eval_v_nl
        from pytc.ecp.energy import _legendre_p_stack

        mol, mf = _make_be_ccecp()
        rexp = REXP()
        # Use REXP only so there is no NuclearCusp (chi_fn is None inside
        # the kernel, which removes the e^{Δχ} factor and isolates the
        # 2-body Δu integrand we want to validate).
        params = {"alpha": jnp.array([0.5])}

        my_xtc = xtc.XTC.from_pyscf(mf, rexp, grid_lvl=0)
        pair_fn = my_xtc._extract_pair_jastrow_fn(params)
        self.assertIsNotNone(pair_fn)

        ecp = parse_pyscf_ecp(mol, warn_on_overlap=False)
        ang = get_grid("icosahedral_12")

        # Run the production kernel.
        delta_U_kernel = np.asarray(my_xtc.get_2b_ecp_du(mf, params))

        # Build the same intermediates on the host and contract them by
        # hand on the (small) full grid in a single shot.
        mo_coeff = np.asarray(my_xtc.mo_coeff)
        grid_points = np.asarray(my_xtc.grid_points)
        weights = np.asarray(my_xtc.weights)
        phi_grid = my_xtc.phi
        n_orb, n_grid = phi_grid.shape

        atom_coords = np.asarray(mol.atom_coords())
        n_atoms = atom_coords.shape[0]
        n_quad = ang.n_points
        l_plus_1 = int(ecp.nl_n.shape[1])

        omega_q = np.asarray(ang.directions)
        rel = grid_points[:, None, :] - atom_coords[None, :, :]
        r_gA = np.linalg.norm(rel, axis=-1)
        omega_g = rel / np.maximum(r_gA, 1e-12)[..., None]
        displaced = (
            atom_coords[None, :, None, :]
            + r_gA[..., None, None] * omega_q[None, None, :, :]
        )  # (n_grid, n_atoms, n_quad, 3)

        ao_disp = numint.eval_ao(mol, displaced.reshape(-1, 3), deriv=0)
        phi_disp = jnp.asarray((mo_coeff.T @ ao_disp.T).reshape(
            n_orb, n_grid, n_atoms, n_quad
        ))

        r_gA_j = jnp.asarray(r_gA)
        omega_g_j = jnp.asarray(omega_g)
        omega_q_j = jnp.asarray(omega_q)
        w_q_j = jnp.asarray(ang.weights)
        v_l = eval_v_nl(r_gA_j, ecp.nl_n, ecp.nl_zeta, ecp.nl_c)
        v_l = v_l * jnp.asarray(ecp.has_ecp).astype(v_l.dtype)[None, :, None]
        cos_theta = jnp.einsum('gad,qd->gaq', omega_g_j, omega_q_j)
        P_l = _legendre_p_stack(l_plus_1, cos_theta)
        two_l_plus_1 = (2 * jnp.arange(l_plus_1) + 1).astype(phi_grid.dtype)
        v_lga = jnp.transpose(v_l, (2, 0, 1))
        alpha_w = jnp.einsum('l,lga,lgaq->gaq', two_l_plus_1, v_lga, P_l)
        alpha_w = alpha_w * w_q_j[None, None, :]

        n_m = n_atoms * n_quad
        pref = (phi_disp * alpha_w[None, ...]).reshape(n_orb, n_grid, n_m)

        gp = jnp.asarray(grid_points)
        gw = jnp.asarray(weights)

        u_inner = jax.vmap(pair_fn, in_axes=(None, 0))
        u_r1r2 = jax.vmap(u_inner, in_axes=(0, None))(gp, gp)
        disp_flat = jnp.asarray(displaced.reshape(-1, 3))
        u_disp_flat = jax.vmap(u_inner, in_axes=(0, None))(disp_flat, gp)
        u_disp = u_disp_flat.reshape(n_grid, n_m, n_grid)
        E = jnp.expm1(u_disp - u_r1r2[:, None, :])

        weighted = phi_grid * gw[None, :]                       # (n_orb, n_grid)
        wphi_rs = phi_grid * gw[None, :]                        # (n_orb, n_grid) -- same axis
        angK = jnp.einsum('qim,imj->qij', pref, E)              # (n_orb, n_grid, n_grid)
        tmp = jnp.einsum('pi,qij->pqj', weighted, angK)         # (n_orb, n_orb, n_grid)
        delta_U_ref_one = jnp.einsum(
            'pqj,rj,sj->pqrs', tmp, wphi_rs, phi_grid
        )
        # The kernel symmetrises by adding the (p,q)<->(r,s) transpose,
        # which corresponds to placing V_NL on electron 2.  Mirror that.
        delta_U_ref = np.asarray(
            delta_U_ref_one + jnp.transpose(delta_U_ref_one, (2, 3, 0, 1))
        )

        # Check a couple of (p,q,r,s) entries to machine precision.
        for idx in [(0, 0, 0, 0), (0, 0, 1, 1), (1, 0, 0, 1), (2, 1, 0, 0)]:
            ref = float(delta_U_ref[idx])
            ker = float(delta_U_kernel[idx])
            if abs(ref) > 1e-14:
                rel = abs(ker - ref) / abs(ref)
                self.assertLess(
                    rel, 1e-6,
                    f"kernel vs reference at {idx}: ker={ker:.6e}, "
                    f"ref={ref:.6e}, rel={rel:.2e}",
                )
            else:
                self.assertLess(abs(ker - ref), 1e-12)
        print(
            f"\n[Be/ccECP-VDZ, REXP] independent-integration cross-check OK "
            f"(N_grid={n_grid}, max|kernel-ref|={float(np.max(np.abs(delta_U_kernel - delta_U_ref))):.2e})"
        )


if __name__ == "__main__":
    unittest.main()
