"""Phase 3 of Option B' for xTC + ECP — *honest* 2-body Δu correction.

The Phase 1 chi resummation handles the rank-1 piece e^{Delta chi} of the
dressed V_NL.  The remaining rank-2 piece is the 2-body cross term

    V_NL,1 * e^{Delta chi_1} * (e^{u(r_1', r_2) - u(r_1, r_2)} - 1)

where r_1' is the angularly displaced point on the ECP angular
quadrature sphere around the ECP atom.  This file builds the rank-4 MO
tensor Delta U^{(NL,Du)}_{pqrs} of that operator on the TC grid and
angular quadrature, by mirroring the outer/inner scan structure of
``pytc.kmat.calc_K3``:

    outer scan over r_2 batches  (kets phi_r, phi_s, weights w_{r2})
        inner scan over r_1 batches  (bras phi_p; phi_q comes via the
                                      angularly-displaced AO eval)

The "kernel" inside that nested scan is — instead of K3's |grad u|^2(r_1, r_2)
— the *vector-in-q* angular sum

    angK_{q}(r_1, r_2) =
        sum_A sum_l (2l+1) V_l^A(r_{1A})
        sum_{q_quad} w_{q_quad} P_l(cos theta) e^{Delta chi_{1,A,q_quad}}
        * (exp(u(r'_{1,A,q_quad}, r_2) - u(r_1, r_2)) - 1)
        * phi_q(r'_{1,A,q_quad})

so that

    Delta U^{(NL,Du)}_{pqrs} =
        sum_{r_1} sum_{r_2} w_{r_1} w_{r_2}
            phi_p(r_1) * angK_{q}(r_1, r_2) * phi_r(r_2) phi_s(r_2)

The bare V_NL piece (the "+1" in 1 + (e^{Du}-1)) is absorbed via the
``expm1`` on Du; the K=0 rank already lives in ``mf.get_hcore()``.

Symmetrisation: V_NL on electron 2 is added by transposing the (p,q)
and (r,s) index pairs of the resulting tensor.  Crucially, the
second-electron coordinate r_2 is *not* density-contracted: it is kept
as a free grid axis just like in get_delta_U.
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
    """Evaluate MO basis values at arbitrary spatial points; returns (N_orb, N_points)."""
    ao = _dft_numint.eval_ao(mol, coords, deriv=0)
    return np.asarray(mo_coeff.T @ ao.T)


def _build_outer_geometry(grid_points, atom_coords, omega_q):
    """Build the angularly-displaced points r'_{g,A,q_quad} and helpers."""
    rel = grid_points[:, None, :] - atom_coords[None, :, :]             # (n_grid, n_atoms, 3)
    r_gA = np.linalg.norm(rel, axis=-1)                                  # (n_grid, n_atoms)
    r_gA_safe = np.maximum(r_gA, 1e-12)
    omega_g = rel / r_gA_safe[..., None]                                 # (n_grid, n_atoms, 3)

    displaced = (atom_coords[None, :, None, :]
                 + r_gA[..., None, None] * omega_q[None, None, :, :])    # (n_grid, n_atoms, n_quad, 3)
    return r_gA, omega_g, displaced


def compute_delta_U_ecp_du(
    *,
    mol,
    mo_coeff: np.ndarray,
    grid_points: np.ndarray,
    weights: np.ndarray,
    phi_grid: jnp.ndarray,
    ecp: EcpData,
    angular_grid: AngularGrid,
    chi_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    pair_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    outer_batch: int = 256,
    inner_batch: int = 256,
    symmetrize: bool = True,
) -> jnp.ndarray:
    """Build the honest-B' rank-4 ECP-Δu tensor on the MO basis.

    Mirrors ``pytc.kmat.calc_K3`` in scan layout — outer batches over
    r_2, inner batches over r_1 — but injects the ECP angular quadrature
    inside the inner scan to evaluate the per-(r_1, r_2) "pair kernel"
    angK_{q}(r_1, r_2).

    Args:
        mol: pyscf Mole for AO evaluation at displaced points.
        mo_coeff: (N_ao, N_orb) AO -> MO matrix (numpy).
        grid_points: (N_grid, 3) TC grid (numpy).
        weights: (N_grid,) integration weights.
        phi_grid: (N_orb, N_grid) MO values on grid (jnp).
        ecp: parsed EcpData.
        angular_grid: AngularGrid for the non-local angular quadrature.
        chi_fn: jax-traceable chi(r) -> scalar, or None.
        pair_fn: jax-traceable u(r1, r2) -> scalar (the non-chi pair J).
        outer_batch: batch size for the outer (r_2) scan.
        inner_batch: batch size for the inner (r_1) scan.
        symmetrize: if True (default), add the V_NL-on-electron-2 partner
            by transposing the resulting tensor.

    Returns:
        (N_orb, N_orb, N_orb, N_orb) tensor in chemist's notation pqrs
        for the ECP-Δu correction.  Zero if no ECP atom.
    """
    n_orb = phi_grid.shape[0]
    if not bool(np.any(np.asarray(ecp.has_ecp))):
        return jnp.zeros((n_orb, n_orb, n_orb, n_orb), dtype=phi_grid.dtype)

    # --- Host-side geometry and AO evaluation at displaced points ---
    atom_coords_np = np.asarray(mol.atom_coords())                       # (N_atoms, 3)
    n_grid = grid_points.shape[0]
    n_atoms = atom_coords_np.shape[0]
    n_quad = angular_grid.n_points
    l_plus_1 = int(ecp.nl_n.shape[1])

    omega_q_np = np.asarray(angular_grid.directions)                     # (n_quad, 3)
    r_gA_np, omega_g_np, displaced_np = _build_outer_geometry(
        grid_points, atom_coords_np, omega_q_np
    )                                                                    # (n_grid, n_atoms),(...),(n_grid, n_atoms, n_quad, 3)
    displaced_flat_np = displaced_np.reshape(-1, 3)

    # phi at all displaced points: (n_orb, n_grid, n_atoms, n_quad)
    phi_disp_flat_np = _eval_phi_at_points(mol, np.asarray(mo_coeff), displaced_flat_np)
    phi_disp = jnp.asarray(
        phi_disp_flat_np.reshape(n_orb, n_grid, n_atoms, n_quad)
    )

    # --- Per-(g, A, q_quad) angular-side prefactor (no r_2 dependence) ---
    r_gA_j = jnp.asarray(r_gA_np)
    omega_g_j = jnp.asarray(omega_g_np)
    omega_q_j = jnp.asarray(omega_q_np)
    w_q_j = jnp.asarray(angular_grid.weights)
    grid_points_j = jnp.asarray(grid_points)
    weights_j = jnp.asarray(weights)

    v_l = eval_v_nl(r_gA_j, ecp.nl_n, ecp.nl_zeta, ecp.nl_c)             # (n_grid, n_atoms, l_plus_1)
    has_ecp_j = ecp.has_ecp.astype(v_l.dtype)
    v_l = v_l * has_ecp_j[None, :, None]

    cos_theta = jnp.einsum('gad,qd->gaq', omega_g_j, omega_q_j)
    P_l = _legendre_p_stack(l_plus_1, cos_theta)                         # (l_plus_1, n_grid, n_atoms, n_quad)
    two_l_plus_1 = (2 * jnp.arange(l_plus_1) + 1).astype(phi_grid.dtype)
    v_lga = jnp.transpose(v_l, (2, 0, 1))                                # (l_plus_1, n_grid, n_atoms)

    # alpha[g, A, q_quad] = sum_l (2l+1) V_l(g,A) P_l(g,A,q_quad)
    alpha = jnp.einsum('l,lga,lgaq->gaq', two_l_plus_1, v_lga, P_l)
    alpha = alpha * w_q_j[None, None, :]                                 # absorb angular-quad weights

    # Multiply in exp(Delta chi).  This factor depends only on r_1's
    # source position and the chosen sphere direction, NOT on r_2.
    if chi_fn is not None:
        displaced_flat_j = jnp.asarray(displaced_flat_np)
        chi_disp_flat = jax.vmap(chi_fn)(displaced_flat_j)
        chi_disp = chi_disp_flat.reshape(n_grid, n_atoms, n_quad)
        chi_grid = jax.vmap(chi_fn)(grid_points_j)
        exp_dchi = jnp.exp(chi_disp - chi_grid[:, None, None])           # (n_grid, n_atoms, n_quad)
    else:
        exp_dchi = jnp.ones((n_grid, n_atoms, n_quad), dtype=phi_grid.dtype)

    angK_prefactor = alpha * exp_dchi                                    # (n_grid, n_atoms, n_quad)

    # phi_disp_with_pref[q_orb, g, A, q_quad] = phi_q(r'_{g,A,q_quad}) * angK_prefactor[g,A,q_quad]
    phi_disp_pref = phi_disp * angK_prefactor[None, ...]                  # (n_orb, n_grid, n_atoms, n_quad)

    # Pre-flatten the (A, q_quad) axes for the scan body to consume.
    # We need, for every (g) point: the displaced positions r'_{g, m}
    # (m = (A, q_quad)) and the prefactor-times-phi_q array
    # phi_disp_pref[q_orb, g, m].
    displaced_per_g = jnp.asarray(displaced_np.reshape(n_grid, n_atoms * n_quad, 3))
    phi_disp_pref_per_g = phi_disp_pref.reshape(n_orb, n_grid, n_atoms * n_quad)

    n_m = n_atoms * n_quad

    # --- Outer scan over r_2 batches (mirrors calc_K3 structure) ---
    # Pad to multiples of inner_batch / outer_batch.
    def _pad_axis(arr, axis, total, value=0.0):
        pad_widths = [(0, 0)] * arr.ndim
        pad_widths[axis] = (0, total - arr.shape[axis])
        return jnp.pad(arr, pad_widths, constant_values=value)

    padded_outer = ((n_grid + outer_batch - 1) // outer_batch) * outer_batch
    padded_inner = ((n_grid + inner_batch - 1) // inner_batch) * inner_batch

    r2_padded = _pad_axis(grid_points_j, 0, padded_outer)
    w2_padded = _pad_axis(weights_j, 0, padded_outer)
    phi_padded_r2 = _pad_axis(phi_grid, 1, padded_outer)                 # (n_orb, padded_outer)

    n_outer_batches = padded_outer // outer_batch
    r2_batches = r2_padded.reshape(n_outer_batches, outer_batch, 3)
    w2_batches = w2_padded.reshape(n_outer_batches, outer_batch)
    phi_rs_batches = phi_padded_r2.reshape(n_orb, n_outer_batches, outer_batch).transpose(1, 0, 2)

    r1_padded = _pad_axis(grid_points_j, 0, padded_inner)
    w1_padded = _pad_axis(weights_j, 0, padded_inner)
    phi_padded_r1 = _pad_axis(phi_grid, 1, padded_inner)                  # phi_p(r_1)
    disp_padded = _pad_axis(displaced_per_g, 0, padded_inner)             # (padded_inner, n_m, 3)
    pref_padded = _pad_axis(phi_disp_pref_per_g, 1, padded_inner)         # (n_orb, padded_inner, n_m)

    n_inner_batches = padded_inner // inner_batch
    r1_batches = r1_padded.reshape(n_inner_batches, inner_batch, 3)
    w1_batches = w1_padded.reshape(n_inner_batches, inner_batch)
    phi_p_batches = phi_padded_r1.reshape(n_orb, n_inner_batches, inner_batch).transpose(1, 0, 2)
    disp_batches = disp_padded.reshape(n_inner_batches, inner_batch, n_m, 3)
    pref_batches = pref_padded.reshape(n_orb, n_inner_batches, inner_batch, n_m).transpose(1, 0, 2, 3)

    # vmap helper: u_vec(r_a, r_b) -> u-value array of shape (len(r_b),) for a single r_a.
    u_inner = jax.vmap(pair_fn, in_axes=(None, 0))                       # (3,), (M, 3) -> (M,)

    @jax.checkpoint
    def outer_scan(carry, args):
        r2_batch, w2_batch, phi_rs_batch = args                          # (Nb_outer, 3), (Nb_outer,), (n_orb, Nb_outer)

        @jax.checkpoint
        def inner_scan(inner_carry, inner_args):
            r1_batch, w1_batch, phi_p_batch, disp_batch, pref_batch = inner_args
            # r1_batch: (Nb_inner, 3)
            # disp_batch: (Nb_inner, n_m, 3)
            # pref_batch: (n_orb, Nb_inner, n_m)
            # phi_p_batch: (n_orb, Nb_inner)
            # w1_batch: (Nb_inner,)

            # --- Pair-Jastrow values on the (r_1, r_2) Cartesian product ---
            # u_r1r2[i, j] = u(r_1_i, r_2_j) -- shape (Nb_inner, Nb_outer)
            u_r1r2 = jax.vmap(u_inner, in_axes=(0, None))(r1_batch, r2_batch)

            # --- Displaced-side u values: u(r'_{i, m}, r_2_j), shape (Nb_inner, n_m, Nb_outer) ---
            disp_flat = disp_batch.reshape(-1, 3)                        # (Nb_inner * n_m, 3)
            u_disp_flat = jax.vmap(u_inner, in_axes=(0, None))(disp_flat, r2_batch)
            u_disp = u_disp_flat.reshape(r1_batch.shape[0], n_m, r2_batch.shape[0])

            # Du[i, m, j] = u(r'_{i,m}, r_2_j) - u(r_1_i, r_2_j)
            Du = u_disp - u_r1r2[:, None, :]                              # (Nb_inner, n_m, Nb_outer)
            E = jnp.expm1(Du)                                             # (Nb_inner, n_m, Nb_outer)

            # angK_{q_orb, i, j} = sum_m pref_batch[q_orb, i, m] * E[i, m, j]
            angK = jnp.einsum('qim,imj->qij', pref_batch, E)             # (n_orb, Nb_inner, Nb_outer)

            # contribution to tmp[p, q_orb, j] = sum_i w1_i * phi_p(i) * angK[q_orb, i, j]
            weighted = phi_p_batch * w1_batch[None, :]                    # (n_orb, Nb_inner)
            contrib = jnp.einsum('pi,qij->pqj', weighted, angK)           # (n_orb, n_orb, Nb_outer)

            return inner_carry + contrib, None

        tmp_init = jnp.zeros((n_orb, n_orb, r2_batch.shape[0]), dtype=phi_grid.dtype)
        tmp, _ = jax.lax.scan(
            inner_scan, tmp_init,
            (r1_batches, w1_batches, phi_p_batches, disp_batches, pref_batches),
        )
        # tmp[p, q_orb, j] now holds the r_1-integrated kernel for this r_2 batch.

        # Accumulate the r_2 contribution against the ket-side AO-pair density.
        # phi_rs_batch: (n_orb, Nb_outer); w2_batch: (Nb_outer,)
        # delta_U[p, q, r, s] += sum_j w2_j * phi_r(j) phi_s(j) * tmp[p, q, j]
        wphi_rs = phi_rs_batch * w2_batch[None, :]                        # (n_orb, Nb_outer)
        contrib_outer = jnp.einsum('pqj,rj,sj->pqrs', tmp, wphi_rs, phi_rs_batch)
        return carry + contrib_outer, None

    init = jnp.zeros((n_orb, n_orb, n_orb, n_orb), dtype=phi_grid.dtype)
    delta_U, _ = jax.lax.scan(outer_scan, init, (r2_batches, w2_batches, phi_rs_batches))

    if symmetrize:
        delta_U = delta_U + jnp.transpose(delta_U, (2, 3, 0, 1))

    return delta_U
