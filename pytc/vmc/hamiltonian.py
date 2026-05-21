"""Hamiltonian and local energy computation functions for VMC."""

import jax
import jax.numpy as jnp


def compute_jastrow_terms(sj, elec_coords, jastrow_params):
    """Compute ∇J/J and ∇²J/J with explicit parameters."""
    n_electrons = elec_coords.shape[0]

    # Fully vectorized implementation (O(N^2) parallelism)
    # Optimized for symmetric Jastrow factors (u(r1, r2) = u(r2, r1))

    # 1. Create all pairs (i, j)
    # Broadcast to (N, N, 3)
    r1 = elec_coords[:, None, :]  # (N, 1, 3) - represents i
    r2 = elec_coords[None, :, :]  # (1, N, 3) - represents j

    # 2. Compute gradients and laplacians for all pairs
    # We only compute gradients w.r.t first argument (r_i)
    # Due to symmetry, sum_j grad_1(ri, rj) == sum_i grad_2(ri, rj)
    # And specifically for the total gradient on electron k:
    # grad_k U = sum_{j!=k} grad_1(rk, rj)

    def compute_pair_grads(r_i, r_j):
        g1, l1 = sj.jastrow.get_log_grads_r1(r_i, r_j, jastrow_params)
        return g1, l1

    # vmap over j (inner), then i (outer)
    inner_vmap = jax.vmap(compute_pair_grads, in_axes=(None, 0))
    outer_vmap = jax.vmap(inner_vmap, in_axes=(0, None))

    # Compute for all pairs
    # g1s: (N, N, 3), l1s: (N, N)
    g1s, l1s = outer_vmap(elec_coords, elec_coords)

    # 3. Mask diagonal (i == j)
    mask = 1.0 - jnp.eye(n_electrons)
    # Expand mask for gradients (N, N, 1)
    mask_grad = mask[:, :, None]

    g1s = g1s * mask_grad
    l1s = l1s * mask

    # 4. Sum over j to get values for each electron i
    sum_g1 = jnp.sum(g1s, axis=1)
    sum_l1 = jnp.sum(l1s, axis=1)

    # 5. Result
    # grad_k U = sum_{j!=k} grad_1(rk, rj)
    # The factor of 0.5 from the definition U = 0.5 * sum u(ri, rj) cancels with the
    # fact that we have two identical sums (one for i=k, one for j=k).
    # So we just take the sum over j of grad_1.

    grad_J_over_J = sum_g1
    lap_sum = sum_l1

    grad_squared = jnp.sum(grad_J_over_J**2, axis=1)
    lap_J_over_J = lap_sum + grad_squared

    return grad_J_over_J, lap_J_over_J


# -----------------------------------------------------------------------------
# Single-determinant kinetic + potential helpers (re-used by multi-det path)
# -----------------------------------------------------------------------------


def compute_kinetic_from_slater(
    slater_up, slater_down,
    inv_up, inv_down,
    grad_up, grad_down,
    lap_up, lap_down,
    grad_J_alpha, grad_J_beta,
    lap_J_alpha, lap_J_beta,
):
    """Single-Slater-determinant kinetic local energy T^{(i)} = -½ ∇²(J D)/(J D).

    Filippi/Assaraf/Moroni Eq. (10) with O = -½∇²:
        ∇²(JD)/(JD) = ∇²J/J + 2 (∇J/J)·(∇D/D) + ∇²D/D
                    = ∇²J/J + tr(A⁻¹ B_kin), with
        B_kin_ij    = ∇²φ_j(r_i) + 2 (∇J/J)(r_i)·∇φ_j(r_i) + (∇²J/J)(r_i)·φ_j(r_i)

    The expression returned here is the kinetic-energy contribution to the
    full local energy E_L = T^{(i)} + V_total + V_NN. The Slater-matrix
    arguments must correspond to a SINGLE determinant; ``grad_J_*`` /
    ``lap_J_*`` are the spin-split Jastrow log-gradients and log-Laplacian.

    Args:
        slater_up, slater_down: A_α (n_alpha, n_alpha), A_β (n_beta, n_beta).
        inv_up, inv_down: A_α⁻¹, A_β⁻¹.
        grad_up, grad_down: ∇A_α (n_alpha, n_alpha, 3), ∇A_β.
        lap_up, lap_down: ∇²A_α (n_alpha, n_alpha), ∇²A_β.
        grad_J_alpha, grad_J_beta: ∇J/J per electron, spin-split.
        lap_J_alpha, lap_J_beta: ∇²J/J per electron, spin-split.

    Returns:
        Scalar T^{(i)} for this determinant.
    """
    B_kin_alpha = -0.5 * (
        lap_up
        + 2 * jnp.einsum('ik,ijk->ij', grad_J_alpha, grad_up)
        + jnp.multiply(lap_J_alpha[:, None], slater_up)
    )
    B_kin_beta = -0.5 * (
        lap_down
        + 2 * jnp.einsum('ik,ijk->ij', grad_J_beta, grad_down)
        + jnp.multiply(lap_J_beta[:, None], slater_down)
    )
    T = jnp.trace(inv_up @ B_kin_alpha) + jnp.trace(inv_down @ B_kin_beta)
    return T


def compute_potential_value(sj, elec_coords):
    """Total potential energy V_en + V_ee (scalar, independent of Slater dets).

    Returns ``sum_i V_en(r_i) + sum_{i<j} 1/|r_i-r_j|``. This is the same
    quantity that the previous ``compute_potential_matrix`` produced
    implicitly via ``tr(A⁻¹ V·A) = sum_i V_i``; extracting it directly is
    cheaper (no matrix multiplies) and makes the multi-det code identical
    across determinants since V doesn't depend on which det we use.
    """
    n_electrons = elec_coords.shape[0]
    atom_coords = sj.atom_coords
    atom_charges = sj.atom_charges

    # e-N
    diff_en = elec_coords[:, None, :] - atom_coords[None, :, :]
    dists_en = jnp.linalg.norm(diff_en, axis=-1)
    V_en = jnp.sum(-atom_charges[None, :] / (dists_en + 1e-10))

    # e-e (factor of 0.5 because we double-count pairs)
    diff_ee = elec_coords[:, None, :] - elec_coords[None, :, :]
    dist_ee = jnp.sqrt(jnp.sum(diff_ee * diff_ee, axis=-1) + 1e-10)
    inv_dist = 1.0 / dist_ee
    inv_dist = inv_dist * (1.0 - jnp.eye(n_electrons))
    V_ee = 0.5 * jnp.sum(inv_dist)

    return V_en + V_ee


# -----------------------------------------------------------------------------
# Backward-compatible single-det local energy (refactored to use the helpers)
# -----------------------------------------------------------------------------


def compute_potential_matrix(sj, elec_coords, slater_alpha, slater_beta):
    """Legacy potential B-matrix returner. Kept for any external callers.

    The current ``compute_single_walker_energy`` no longer uses this — it
    calls ``compute_potential_value`` directly — but the function is exported
    by name and may be in use elsewhere, so we preserve it unchanged.
    """
    n_alpha = sj.dets[0].n_alpha
    n_electrons = len(elec_coords)

    atom_coords = sj.atom_coords
    atom_charges = sj.atom_charges

    def e_n_potential(r):
        r_reshaped = r[:, jnp.newaxis, :]
        diff = r_reshaped - atom_coords[jnp.newaxis, :, :]
        dists = jnp.linalg.norm(diff, axis=2)
        potentials = -atom_charges[jnp.newaxis, :] / (dists + 1e-10)
        return jnp.sum(potentials, axis=1)

    alpha_coords = jnp.take(elec_coords, jnp.arange(n_alpha), axis=0)
    beta_coords = jnp.take(elec_coords, jnp.arange(n_alpha, n_electrons), axis=0)

    V_en_alpha = e_n_potential(alpha_coords)[:, None]
    V_en_beta = e_n_potential(beta_coords)[:, None]

    def pairwise_distance(r_i, r_j):
        diff = r_i - r_j
        dist = jnp.sqrt(jnp.sum(diff**2) + 1e-10)
        return 1.0 / dist

    ee_vmap_inner = jax.vmap(pairwise_distance, in_axes=(None, 0))
    ee_vmap_outer = jax.vmap(ee_vmap_inner, in_axes=(0, None))
    all_e_e_pot = jax.jit(ee_vmap_outer)(elec_coords, elec_coords)

    mask = 1.0 - jnp.eye(n_electrons)
    all_e_e_pot = all_e_e_pot * mask
    e_e_pot = 0.5 * jnp.sum(all_e_e_pot, axis=1)

    V_ee_alpha = e_e_pot[:n_alpha, None]
    V_ee_beta = e_e_pot[n_alpha:, None]

    B_alpha = (V_en_alpha + V_ee_alpha) * slater_alpha
    B_beta = (V_en_beta + V_ee_beta) * slater_beta

    return B_alpha, B_beta


def compute_single_walker_energy(sj, walker, jastrow_params):
    """Single-determinant local energy.

    Uses Slater matrix quantities already stored on the walker by a prior
    ``eval_sj`` / ``eval_det_value_and_grad`` call. Assumes the ansatz is a
    single-determinant SlaterJastrow (``len(sj.dets) == 1``). For
    multi-determinant ansatze, see ``compute_multi_det_local_energy`` —
    ``eval_local_energy`` dispatches automatically.
    """
    n_alpha = sj.dets[0].n_alpha

    grad_J_over_J, lap_J_over_J = compute_jastrow_terms(
        sj, walker.positions, jastrow_params
    )
    grad_J_alpha = grad_J_over_J[:n_alpha]
    grad_J_beta = grad_J_over_J[n_alpha:]
    lap_J_alpha = lap_J_over_J[:n_alpha]
    lap_J_beta = lap_J_over_J[n_alpha:]

    T = compute_kinetic_from_slater(
        walker.slater_up, walker.slater_down,
        walker.inv_up, walker.inv_down,
        walker.grad_up, walker.grad_down,
        walker.lap_up, walker.lap_down,
        grad_J_alpha, grad_J_beta,
        lap_J_alpha, lap_J_beta,
    )
    V = compute_potential_value(sj, walker.positions)
    return jnp.real(T + V + sj.ion_ion_potential)


# -----------------------------------------------------------------------------
# Multi-determinant local energy via weighted single-det composition
# -----------------------------------------------------------------------------


def compute_multi_det_local_energy(sj, walker, jastrow_params, linear_coeffs):
    """Local energy E_L = H Psi / Psi for Psi = J · sum_i c_i D_i.

    Modular derivation: writing Phi = sum_i c_i D_i and using
        ∇Phi/Phi  = sum_i w_i ∇D_i/D_i,    w_i = c_i D_i / Phi
        ∇²Phi/Phi = sum_i w_i ∇²D_i/D_i
    (with ``sum_i w_i = 1`` as signed weights), the kinetic local energy
    reduces to a CONFIGURATION-DEPENDENT weighted average of the single-det
    kinetic energies:

        T_multi(R) = sum_i w_i(R) * T^{(i)}(R)

    where T^{(i)} = -½ ∇²(J D_i)/(J D_i) is the single-det kinetic local
    energy of J · D_i. Potentials don't depend on the determinantal part,
    so the total local energy is

        E_L = sum_i w_i T^{(i)} + V_total + V_NN.

    Implementation: call the existing ``eval_det_value_and_grad`` for each
    determinant in ``sj.dets`` to get its Slater matrices, then reuse
    ``compute_kinetic_from_slater`` for each. Signed weights from the
    determinant values.

    Note on nodes: when Phi = sum_i c_i D_i approaches zero (the
    multi-determinant wavefunction's nodal surface), individual weights
    ``w_i = c_i D_i / Phi`` can diverge in magnitude even though they sum
    to 1. Walkers sampled from |Phi|² avoid the nodes (measure zero), but
    can approach them; near nodes E_L can be heavy-tailed unless J is close
    to optimal. This is a property of the multi-det reference and is
    discussed in the optimizer docstring.
    """
    # Lazy import to avoid circular dependency at module load time
    from pytc.ansatz.det import eval_det_value_and_grad

    n_dets = len(sj.dets)
    n_alpha = sj.dets[0].n_alpha

    # ---- Common: Jastrow gradient/Laplacian (same for all dets) ----
    grad_J_over_J, lap_J_over_J = compute_jastrow_terms(
        sj, walker.positions, jastrow_params
    )
    grad_J_alpha = grad_J_over_J[:n_alpha]
    grad_J_beta = grad_J_over_J[n_alpha:]
    lap_J_alpha = lap_J_over_J[:n_alpha]
    lap_J_beta = lap_J_over_J[n_alpha:]

    # ---- Common: total potential (V_en + V_ee), independent of D_i ----
    V = compute_potential_value(sj, walker.positions)

    # ---- Per-det: kinetic contribution T^{(i)} and determinant value D_i ----
    T_list = []
    D_list = []
    for det_i in sj.dets:
        (det_sign, det_logabs), det_walker = eval_det_value_and_grad(det_i, walker)
        T_i = compute_kinetic_from_slater(
            det_walker.slater_up, det_walker.slater_down,
            det_walker.inv_up, det_walker.inv_down,
            det_walker.grad_up, det_walker.grad_down,
            det_walker.lap_up, det_walker.lap_down,
            grad_J_alpha, grad_J_beta,
            lap_J_alpha, lap_J_beta,
        )
        # D_i = sign * exp(logabs). Use the value directly (not log) because
        # we need signed amplitudes for the multi-det mixing.
        D_i = det_sign * jnp.exp(det_logabs)
        T_list.append(T_i)
        D_list.append(D_i)

    # ---- Signed mixing weights w_i = c_i D_i / Phi ----
    weighted_D = jnp.stack(
        [linear_coeffs[i] * D_list[i] for i in range(n_dets)]
    )
    Phi = jnp.sum(weighted_D)
    # Tiny epsilon to avoid 0/0 at exact nodes (measure-zero set; walkers
    # sampled from |Phi|² never land exactly on them, but JAX may evaluate
    # at intermediate configurations during AD).
    weights = weighted_D / (Phi + 1e-300)

    # ---- Combine ----
    T_multi = jnp.sum(weights * jnp.stack(T_list))
    return jnp.real(T_multi + V + sj.ion_ion_potential)


# -----------------------------------------------------------------------------
# Top-level dispatch
# -----------------------------------------------------------------------------


def eval_local_energy(sj, walker, params):
    """Evaluate local energy for a SlaterJastrow ansatz.

    Dispatches to single-det or multi-det kernel based on ``len(sj.dets)``.

    Args:
        sj: SlaterJastrow ansatz object.
        walker: Walker object (positions populated; Slater matrices ignored
            for the multi-det path, which recomputes them per det).
        params: ``[jastrow_params, linear_coeffs]``.

    Returns:
        ``(energy, walker)``. The walker is returned unchanged so that the
        caller's reference matrices (from the most recent ``eval_sj`` call)
        remain available for downstream consumers.
    """
    jastrow_params, linear_coeffs = params
    if len(sj.dets) == 1:
        energy = compute_single_walker_energy(sj, walker, jastrow_params)
    else:
        energy = compute_multi_det_local_energy(
            sj, walker, jastrow_params, linear_coeffs
        )
    return energy, walker
