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

Tile-aware construction: ``compute_delta_U_ecp_du(ranges=(p,q,r,s))``
builds only the requested orbital sub-block — the displaced-AO
evaluation, the orbital projections, and the scan accumulator all carry
the reduced (|p|, |q|, |r|, |s|) shape, so a ranged query never
materialises the full n_orb^4 tensor.  Pair-exchange symmetrisation of
an off-diagonal tile is handled by computing the raw tile and its
transpose-partner raw tile and summing.
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


def _resolve_slice(s, n: int) -> slice:
    """Normalise an index/slice/None to a concrete slice over [0, n).

    ``None`` means "the full range".  Integer indices become single-element
    slices so the resulting axis keeps ndim == 1 (the scans and einsums
    rely on every orbital axis being a real dimension).
    """
    if s is None or isinstance(s, slice):
        return s if isinstance(s, slice) else slice(None)
    i = int(s)
    if i < 0:
        i += n
    return slice(i, i + 1)


def _slice_len(s, n: int) -> int:
    start, stop, step = s.indices(n)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def _raw_du_tile(
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
    p_sl,
    q_sl,
    r_sl,
    s_sl,
    outer_batch: int = 256,
    inner_batch: int = 256,
) -> jnp.ndarray:
    """Build the RAW (V_NL-on-electron-1) ECP-Δu tile for orbital sub-blocks.

    ``p_sl``/``q_sl`` are the bra / angular-displaced indices (electron 1);
    ``r_sl``/``s_sl`` are the ket indices (electron 2).  Each is an index,
    a slice, or ``None`` (= full [0, n_orb) range).

    The displaced-AO evaluation projects only the requested ``q`` orbitals,
    and the grid projections / scan accumulator carry the reduced
    (|p|, |q|, |r|, |s|) shape — so this never materialises the full
    ``n_orb**4`` tensor for a sub-block query.

    No pair-exchange (V_NL-on-electron-2) symmetrisation is applied here;
    :func:`compute_delta_U_ecp_du` is responsible for adding the transpose
    partner.
    """
    n_orb = phi_grid.shape[0]
    p_sl = _resolve_slice(p_sl, n_orb)
    q_sl = _resolve_slice(q_sl, n_orb)
    r_sl = _resolve_slice(r_sl, n_orb)
    s_sl = _resolve_slice(s_sl, n_orb)
    n_p, n_q, n_r, n_s = (
        _slice_len(p_sl, n_orb), _slice_len(q_sl, n_orb),
        _slice_len(r_sl, n_orb), _slice_len(s_sl, n_orb),
    )
    out_shape = (n_p, n_q, n_r, n_s)
    if 0 in out_shape:
        return jnp.zeros(out_shape, dtype=phi_grid.dtype)

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

    # phi at all displaced points for the REQUESTED q orbitals only:
    # (n_q, n_grid, n_atoms, n_quad)
    phi_disp_flat_np = _eval_phi_at_points(
        mol, np.asarray(mo_coeff[:, q_sl]), displaced_flat_np
    )
    phi_disp = jnp.asarray(
        phi_disp_flat_np.reshape(n_q, n_grid, n_atoms, n_quad)
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
    phi_disp_pref = phi_disp * angK_prefactor[None, ...]                  # (n_q, n_grid, n_atoms, n_quad)

    # Pre-flatten the (A, q_quad) axes for the scan body to consume.
    displaced_per_g = jnp.asarray(displaced_np.reshape(n_grid, n_atoms * n_quad, 3))
    phi_disp_pref_per_g = phi_disp_pref.reshape(n_q, n_grid, n_atoms * n_quad)

    n_m = n_atoms * n_quad

    # --- Orbital-indexed grid values, sliced to the requested tile ---
    phi_p_grid = phi_grid[p_sl, :]                                       # (n_p, n_grid)  [r_1 bra]
    phi_r_grid = phi_grid[r_sl, :]                                       # (n_r, n_grid)  [r_2 ket]
    phi_s_grid = phi_grid[s_sl, :]                                       # (n_s, n_grid)  [r_2 ket]

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
    phi_padded_r = _pad_axis(phi_r_grid, 1, padded_outer)                # (n_r, padded_outer)
    phi_padded_s = _pad_axis(phi_s_grid, 1, padded_outer)                # (n_s, padded_outer)

    n_outer_batches = padded_outer // outer_batch
    r2_batches = r2_padded.reshape(n_outer_batches, outer_batch, 3)
    w2_batches = w2_padded.reshape(n_outer_batches, outer_batch)
    phi_rs_batches_r = phi_padded_r.reshape(n_r, n_outer_batches, outer_batch).transpose(1, 0, 2)
    phi_rs_batches_s = phi_padded_s.reshape(n_s, n_outer_batches, outer_batch).transpose(1, 0, 2)

    r1_padded = _pad_axis(grid_points_j, 0, padded_inner)
    w1_padded = _pad_axis(weights_j, 0, padded_inner)
    phi_padded_p = _pad_axis(phi_p_grid, 1, padded_inner)                # phi_p(r_1) -> (n_p, padded_inner)
    disp_padded = _pad_axis(displaced_per_g, 0, padded_inner)            # (padded_inner, n_m, 3)
    pref_padded = _pad_axis(phi_disp_pref_per_g, 1, padded_inner)        # (n_q, padded_inner, n_m)

    n_inner_batches = padded_inner // inner_batch
    r1_batches = r1_padded.reshape(n_inner_batches, inner_batch, 3)
    w1_batches = w1_padded.reshape(n_inner_batches, inner_batch)
    phi_p_batches = phi_padded_p.reshape(n_p, n_inner_batches, inner_batch).transpose(1, 0, 2)
    disp_batches = disp_padded.reshape(n_inner_batches, inner_batch, n_m, 3)
    pref_batches = pref_padded.reshape(n_q, n_inner_batches, inner_batch, n_m).transpose(1, 0, 2, 3)

    # vmap helper: u_vec(r_a, r_b) -> u-value array of shape (len(r_b),) for a single r_a.
    u_inner = jax.vmap(pair_fn, in_axes=(None, 0))                       # (3,), (M, 3) -> (M,)

    @jax.checkpoint
    def outer_scan(carry, args):
        r2_batch, w2_batch, phi_rs_r, phi_rs_s = args                    # (Nb_outer,3),(Nb_outer,),(n_r,Nb_outer),(n_s,Nb_outer)

        @jax.checkpoint
        def inner_scan(inner_carry, inner_args):
            r1_batch, w1_batch, phi_p_batch, disp_batch, pref_batch = inner_args
            # r1_batch: (Nb_inner, 3)
            # disp_batch: (Nb_inner, n_m, 3)
            # pref_batch: (n_q, Nb_inner, n_m)
            # phi_p_batch: (n_p, Nb_inner)
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
            angK = jnp.einsum('qim,imj->qij', pref_batch, E)             # (n_q, Nb_inner, Nb_outer)

            # contribution to tmp[p, q_orb, j] = sum_i w1_i * phi_p(i) * angK[q_orb, i, j]
            weighted = phi_p_batch * w1_batch[None, :]                    # (n_p, Nb_inner)
            contrib = jnp.einsum('pi,qij->pqj', weighted, angK)           # (n_p, n_q, Nb_outer)

            return inner_carry + contrib, None

        tmp_init = jnp.zeros((n_p, n_q, r2_batch.shape[0]), dtype=phi_grid.dtype)
        tmp, _ = jax.lax.scan(
            inner_scan, tmp_init,
            (r1_batches, w1_batches, phi_p_batches, disp_batches, pref_batches),
        )
        # tmp[p, q_orb, j] now holds the r_1-integrated kernel for this r_2 batch.

        # Accumulate the r_2 contribution against the ket-side AO-pair density.
        # phi_rs_r: (n_r, Nb_outer); phi_rs_s: (n_s, Nb_outer); w2_batch: (Nb_outer,)
        # delta_U[p, q, r, s] += sum_j w2_j * phi_r(j) phi_s(j) * tmp[p, q, j]
        wphi_rs_r = phi_rs_r * w2_batch[None, :]                         # (n_r, Nb_outer)
        contrib_outer = jnp.einsum('pqj,rj,sj->pqrs', tmp, wphi_rs_r, phi_rs_s)
        return carry + contrib_outer, None

    init = jnp.zeros(out_shape, dtype=phi_grid.dtype)
    delta_U, _ = jax.lax.scan(
        outer_scan, init, (r2_batches, w2_batches, phi_rs_batches_r, phi_rs_batches_s)
    )
    return delta_U


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
    ranges=None,
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
        ranges: optional ``(p, q, r, s)`` of indices/slices/None selecting
            the orbital sub-block to return.  When given, only that tile
            is constructed — the displaced-AO eval, orbital projections,
            and scan accumulator all carry the reduced (|p|, |q|, |r|, |s|)
            shape, so the full ``n_orb**4`` tensor is never materialised.
            Pair-exchange symmetrisation is preserved on the tile by
            summing the raw tile with the transposed transpose-partner
            raw tile.  When ``None`` (default) the full tensor is built.

    Returns:
        ``(N_orb, N_orb, N_orb, N_orb)`` tensor in chemist's notation pqrs
        for the ECP-Δu correction (or the requested ``(p, q, r, s)``
        sub-block).  Zero if no ECP atom.
    """
    n_orb = phi_grid.shape[0]

    def _zero_for(ranges_):
        if ranges_ is None:
            return jnp.zeros((n_orb, n_orb, n_orb, n_orb), dtype=phi_grid.dtype)
        shape = tuple(
            _slice_len(_resolve_slice(r, n_orb), n_orb) for r in ranges_
        )
        return jnp.zeros(shape, dtype=phi_grid.dtype)

    if not bool(np.any(np.asarray(ecp.has_ecp))):
        return _zero_for(ranges)

    common = dict(
        mol=mol,
        mo_coeff=mo_coeff,
        grid_points=grid_points,
        weights=weights,
        phi_grid=phi_grid,
        ecp=ecp,
        angular_grid=angular_grid,
        chi_fn=chi_fn,
        pair_fn=pair_fn,
        outer_batch=outer_batch,
        inner_batch=inner_batch,
    )

    if ranges is None:
        # Full tensor: one raw pass + in-place transpose (compute-optimal).
        raw = _raw_du_tile(**common, p_sl=None, q_sl=None, r_sl=None, s_sl=None)
        return (raw + jnp.transpose(raw, (2, 3, 0, 1))) if symmetrize else raw

    P, Q, R, S = ranges
    raw_a = _raw_du_tile(**common, p_sl=P, q_sl=Q, r_sl=R, s_sl=S)        # (|p|,|q|,|r|,|s|)
    if not symmetrize:
        return raw_a
    # Symmetrised tile = raw(p,q,r,s) + raw(r,s,p,q)^T(2,3,0,1).
    raw_b = _raw_du_tile(**common, p_sl=R, q_sl=S, r_sl=P, s_sl=Q)        # (|r|,|s|,|p|,|q|)
    return raw_a + jnp.transpose(raw_b, (2, 3, 0, 1))
