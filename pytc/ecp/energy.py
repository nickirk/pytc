"""Non-local effective-core-potential contribution to the VMC local energy.

Implements the locality-approximation working expression derived in
``docs/design_ecp_vmc.md`` §4:

    (V_NL ψ / ψ)(r_i) = Σ_l (2l+1) V_l(r_iA)
                        Σ_q w_q P_l(cos θ_q) ψ(r'_{i,q})/ψ(r_i),

summed over electrons i and ECP atoms A.  ``r'_{i,q} = R_A + r_iA Ω_q`` is
the electron's position rotated to quadrature direction Ω_q around the
ECP nucleus at R_A.  cos θ_q is the dot product of the original electron
direction with Ω_q.  The wavefunction ratio is evaluated under the trial
wavefunction (locality approximation) via ``sj.psi_ratio_single``.

The kernel is fully ``vmap``-friendly: per-pair masking by
``has_ecp[A] & (r_iA < r_cut[A])`` zeroes the contribution from non-ECP
atoms and from electrons outside the per-atom cutoff radius §4.6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pytc.ecp.quadrature import AngularGrid, icosahedral_12
from pytc.ecp.radial import eval_v_nl


def _legendre_p_stack(l_plus_1: int, x: jnp.ndarray) -> jnp.ndarray:
    """Stack P_0(x), P_1(x), ..., P_{L}(x) along a new leading axis.

    Uses the standard 3-term recurrence
        l P_l = (2l-1) x P_{l-1} - (l-1) P_{l-2}.
    """
    # Build with a Python loop; l_plus_1 is static at trace time.
    polys = []
    if l_plus_1 >= 1:
        polys.append(jnp.ones_like(x))
    if l_plus_1 >= 2:
        polys.append(x)
    for l in range(2, l_plus_1):
        polys.append(((2 * l - 1) * x * polys[l - 1] - (l - 1) * polys[l - 2]) / l)
    return jnp.stack(polys, axis=0)


def compute_nonlocal_ecp_energy(sj, walker, jastrow_params,
                                *, quad_grid: AngularGrid | None = None):
    """Non-local ECP contribution to E_L for a single walker.

    Args:
        sj: SlaterJastrow ansatz (must carry ``sj.ecp``; single-determinant only,
            inherited from the v1 ``psi_ratio_single`` restriction).
        walker: populated walker (unbatched). ``walker.inv_up`` / ``inv_down``
            and ``walker.log_jastrow`` must already be set.
        jastrow_params: Jastrow parameters (linear-det coeff is not needed for
            single-determinant ratios — it cancels).
        quad_grid: angular quadrature.  Defaults to the 12-point icosahedral
            grid (exact through l = 5, the QMC standard).

    Returns:
        Scalar: Σ_{i,A∈ECP} (V_NL ψ / ψ)(r_i) under the locality approximation.
        Returns 0 when no atom carries an ECP.
    """
    ecp = sj.ecp
    if quad_grid is None:
        quad_grid = icosahedral_12()

    positions = walker.positions                       # (n_e, 3)
    atom_coords = sj.atom_coords                       # (n_a, 3)
    n_e = positions.shape[0]
    l_plus_1 = ecp.nl_n.shape[1]

    # Per (electron, atom) pair: separation vector, distance, unit direction.
    rel = positions[:, None, :] - atom_coords[None, :, :]    # (n_e, n_a, 3)
    r_iA = jnp.linalg.norm(rel, axis=-1)                     # (n_e, n_a)
    r_safe = jnp.maximum(r_iA, 1e-12)
    omega_i = rel / r_safe[..., None]                        # (n_e, n_a, 3)

    # Radial channel values V_l(r_iA), padded with zeros for non-existent l's
    # and for non-ECP atoms (where all nl_c = 0 by construction).
    v_l = eval_v_nl(r_iA, ecp.nl_n, ecp.nl_zeta, ecp.nl_c)   # (n_e, n_a, l+1)

    # Angular geometry: cos θ_q = Ω̂_i · Ω_q  and  r'_{i,q}^A = R_A + r_iA Ω_q.
    omega_q = quad_grid.directions                           # (n_q, 3)
    w_q = quad_grid.weights                                  # (n_q,)
    cos_theta = jnp.einsum('iad,qd->iaq', omega_i, omega_q)  # (n_e, n_a, n_q)

    # Displaced electron positions per (i, A, q).
    new_pos = (
        atom_coords[None, :, None, :]                        # (1, n_a, 1, 3)
        + r_iA[..., None, None] * omega_q[None, None, :, :]  # (n_e, n_a, n_q, 3)
    )

    # Wavefunction ratios ψ(r'_{i,q,A}) / ψ(r_i).  Triple-nested vmap:
    # innermost over q, then atoms, then electrons.
    from pytc.ansatz.sj import eval_psi_ratio_single

    def _ratio(i, pos):
        return eval_psi_ratio_single(sj, walker, i, pos, jastrow_params)

    ratios = jax.vmap(                                       # over electrons
        jax.vmap(                                            # over atoms
            jax.vmap(_ratio, in_axes=(None, 0)),             # over q
            in_axes=(None, 0),
        ),
        in_axes=(0, 0),
    )(jnp.arange(n_e), new_pos)                              # (n_e, n_a, n_q)

    # Legendre polynomials at cos θ_q, stacked over l.
    P_l = _legendre_p_stack(l_plus_1, cos_theta)             # (l+1, n_e, n_a, n_q)

    # Angular integral per (i, A, l): Σ_q w_q P_l(cos θ_q) ratio_q.
    angular = jnp.einsum('liaq,q,iaq->lia', P_l, w_q, ratios)  # (l+1, n_e, n_a)

    # Contract with (2l+1) and V_l(r_iA).
    two_l_plus_1 = jnp.arange(l_plus_1) * 2 + 1               # (l+1,)
    v_lia = jnp.transpose(v_l, (2, 0, 1))                     # (l+1, n_e, n_a)
    e_iA = jnp.einsum('l,lia,lia->ia', two_l_plus_1, v_lia, angular)  # (n_e, n_a)

    # Mask: non-ECP atoms contribute 0; electrons beyond r_cut contribute 0.
    pair_mask = ecp.has_ecp[None, :] & (r_iA < ecp.r_cut[None, :])
    e_iA = e_iA * pair_mask.astype(e_iA.dtype)

    return jnp.sum(e_iA)
