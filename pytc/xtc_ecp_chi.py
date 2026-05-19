"""Phase 1 of Option B for xTC + ECP — exact 1-body Jastrow resummation
inside the non-local ECP angular quadrature.

Adds to the 1-body MO Hamiltonian the correction

    Delta h^{NL,chi}_{pq} = sum_g w_g phi_p(r_g)
                          * sum_A sum_l (2l+1) V_l^A(r_{gA})
                          * sum_q w_q P_l(cos theta_q)
                          * [exp(chi(r'_{g,A,q}) - chi(r_g)) - 1]
                          * phi_q(r'_{g,A,q})

with r'_{g,A,q} = R_A + r_{gA} * Omega_q the angular-displaced point
on the sphere around atom A.  The "-1" subtracts the bare non-local
contribution that is already inside ``mf.get_hcore()``; what remains is
the chi-dressed correction (an operator-identity resummation of all BCH
orders of the 1-body Jastrow piece — see ``_local/design/ecp_xtc_theory.md``
§5-§8 / Option B).

Phase 1 deliberately drops the 2-body Delta-u piece (Phase 3).  At
acceptance, ``compute_delta_h_ecp_chi`` is callable in isolation and only
needs (1) the MO basis on the radial grid, (2) the ECP table and angular
quadrature, (3) an evaluable ``chi(r)`` for the 1-body Jastrow factor.

The implementation deliberately uses host-side (numpy) AO evaluation
through ``pyscf.dft.numint.eval_ao`` at displaced points — the grid +
quadrature points are deterministic given ``(grid_points, atom_coords)``,
so we build them once and push to device.  The angular integration and
matrix accumulation are pure JAX.
"""

from __future__ import annotations

import logging
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from pyscf.dft import numint as _dft_numint

from pytc.ecp.energy import _legendre_p_stack
from pytc.ecp.parser import EcpData
from pytc.ecp.quadrature import AngularGrid
from pytc.ecp.radial import eval_v_nl


logger = logging.getLogger(__name__)


def _eval_phi_at_points(mol, mo_coeff: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Evaluate MO basis values at arbitrary spatial points.

    Args:
        mol: pyscf Mole.
        mo_coeff: (N_ao, N_orb) AO -> MO transform (numpy).
        coords: (N_points, 3) point list (numpy).

    Returns:
        (N_orb, N_points) array of MO values at each point.
    """
    ao = _dft_numint.eval_ao(mol, coords, deriv=0)  # (N_points, N_ao)
    # MO values: (N_orb, N_points) = mo_coeff.T @ ao.T
    return np.asarray(mo_coeff.T @ ao.T)


def compute_delta_h_ecp_chi(
    *,
    mol,
    mo_coeff: np.ndarray,
    grid_points: np.ndarray,
    weights: np.ndarray,
    phi_grid: np.ndarray,
    ecp: EcpData,
    angular_grid: AngularGrid,
    chi_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Compute Delta h^{NL,chi}_{pq} on the MO basis.

    Args:
        mol: pyscf Mole, used only to evaluate AOs at displaced points.
        mo_coeff: (N_ao, N_orb) AO -> MO matrix (numpy).
        grid_points: (N_grid, 3) radial+angular DFT grid (numpy).
        weights: (N_grid,) grid weights (numpy or jnp).
        phi_grid: (N_orb, N_grid) MO values on grid_points (jnp), as held
            by the XTC dataclass.
        ecp: parsed EcpData dataclass.
        angular_grid: AngularGrid for the non-local angular integral.
        chi_fn: jax-traceable function mapping (3,) -> scalar; returns
            chi(r) = sum_A chi_A(|r - R_A|) at any electron position.

    Returns:
        (N_orb, N_orb) jnp.ndarray.  Zero matrix if no atom carries an
        ECP (caller may also short-circuit before calling).
    """
    n_orb = phi_grid.shape[0]
    if not bool(np.any(np.asarray(ecp.has_ecp))):
        return jnp.zeros((n_orb, n_orb), dtype=phi_grid.dtype)

    atom_coords_np = np.asarray(mol.atom_coords())            # (N_atoms, 3)
    n_grid = grid_points.shape[0]
    n_atoms = atom_coords_np.shape[0]
    n_quad = angular_grid.n_points
    l_plus_1 = int(ecp.nl_n.shape[1])

    # --- Geometry on host ---
    # rel[g, A, :] = r_g - R_A
    rel_np = grid_points[:, None, :] - atom_coords_np[None, :, :]   # (n_grid, n_atoms, 3)
    r_gA_np = np.linalg.norm(rel_np, axis=-1)                        # (n_grid, n_atoms)
    r_gA_safe = np.maximum(r_gA_np, 1e-12)
    omega_g_np = rel_np / r_gA_safe[..., None]                       # (n_grid, n_atoms, 3)

    omega_q_np = np.asarray(angular_grid.directions)                 # (n_quad, 3)

    # Displaced points r'_{g, A, q} = R_A + r_gA * Omega_q
    # Shape: (n_grid, n_atoms, n_quad, 3)
    displaced_np = (
        atom_coords_np[None, :, None, :]
        + r_gA_np[..., None, None] * omega_q_np[None, None, :, :]
    )
    displaced_flat = displaced_np.reshape(-1, 3)

    # --- AO evaluation at displaced points (host) ---
    # phi at displaced points: (n_orb, n_grid * n_atoms * n_quad)
    phi_disp_flat_np = _eval_phi_at_points(mol, np.asarray(mo_coeff), displaced_flat)
    phi_disp = jnp.asarray(
        phi_disp_flat_np.reshape(n_orb, n_grid, n_atoms, n_quad)
    )

    # --- chi evaluation at all displaced points and grid points ---
    # We vmap chi_fn over (n_grid * n_atoms * n_quad) points.
    displaced_flat_j = jnp.asarray(displaced_flat)
    chi_disp_flat = jax.vmap(chi_fn)(displaced_flat_j)
    chi_disp = chi_disp_flat.reshape(n_grid, n_atoms, n_quad)

    grid_points_j = jnp.asarray(grid_points)
    chi_grid = jax.vmap(chi_fn)(grid_points_j)                       # (n_grid,)

    # --- JAX-side angular integration ---
    weights_j = jnp.asarray(weights)
    r_gA_j = jnp.asarray(r_gA_np)
    omega_g_j = jnp.asarray(omega_g_np)
    omega_q_j = jnp.asarray(omega_q_np)
    w_q_j = jnp.asarray(angular_grid.weights)

    # V_l(r_gA): (n_grid, n_atoms, l_plus_1)
    v_l = eval_v_nl(r_gA_j, ecp.nl_n, ecp.nl_zeta, ecp.nl_c)

    # cos theta: (n_grid, n_atoms, n_quad)
    cos_theta = jnp.einsum('gad,qd->gaq', omega_g_j, omega_q_j)

    # P_l(cos theta): (l_plus_1, n_grid, n_atoms, n_quad)
    P_l = _legendre_p_stack(l_plus_1, cos_theta)

    # exp(Delta chi) - 1, broadcast across quad axis:
    delta_chi = chi_disp - chi_grid[:, None, None]                   # (n_grid, n_atoms, n_quad)
    expm1_chi = jnp.expm1(delta_chi)                                  # exp - 1, accurate near 0

    # Mask non-ECP atoms (V_l is already ~0 outside r_cut, but
    # has_ecp[A] = False atoms have ecp.nl_c = 0 so V_l = 0 already;
    # explicit mask keeps gradients clean).
    has_ecp_j = ecp.has_ecp.astype(v_l.dtype)
    # v_l shape (n_grid, n_atoms, l_plus_1); broadcast has_ecp on atom axis.
    v_l = v_l * has_ecp_j[None, :, None]

    # Angular accumulation: for each (g, A), sum_l (2l+1) V_l * sum_q w_q P_l(cos) * (exp(Dchi) - 1) * phi_q(r')
    # First: contract over q (inner angular integral) per l.
    two_l_plus_1 = (2 * jnp.arange(l_plus_1) + 1).astype(phi_grid.dtype)

    # Build the q-integrand: w_q * (exp(Dchi) - 1) * phi_q(r')
    # phi_disp: (n_orb, n_grid, n_atoms, n_quad)
    # expm1_chi: (n_grid, n_atoms, n_quad)
    # Combined integrand (over q) has axes (n_orb, n_grid, n_atoms, n_quad).
    q_integrand = phi_disp * (expm1_chi[None, ...] * w_q_j[None, None, None, :])

    # Contract over the angular index q only:
    # K_{l, orb_q, g, A} = sum_q P_l(g, A, q) * w_q * (exp(Dchi)-1)(g, A, q) * phi_q(g, A, q)
    # P_l: (l_plus_1, n_grid, n_atoms, n_quad); q_integrand: (n_orb, n_grid, n_atoms, n_quad)
    K = jnp.einsum('lgaq,Qgaq->lQga', P_l, q_integrand)

    # Multiply by (2l+1) V_l and sum over l, A:
    # v_l[g, a, l] -> permute to (l, g, a)
    v_lga = jnp.transpose(v_l, (2, 0, 1))                            # (l_plus_1, n_grid, n_atoms)

    # angular_kernel[orb_q, g] = sum_{l, a} (2l+1) v_l(g, a) K_{l, orb_q, g, a}
    angular_kernel = jnp.einsum('l,lga,lQga->Qg', two_l_plus_1, v_lga, K)

    # Final contraction with phi_p(r_g) * w_g:
    # phi_grid: (n_orb, n_grid); weights: (n_grid,)
    # Delta h[p, q] = sum_g w_g phi_p(r_g) * angular_kernel[q, g]
    weighted_phi = phi_grid * weights_j[None, :]                     # (n_orb, n_grid)
    delta_h = jnp.einsum('pg,Qg->pQ', weighted_phi, angular_kernel)

    return delta_h
