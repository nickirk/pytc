"""Hamiltonian and local energy computation functions for VMC."""

import jax
import jax.numpy as jnp
from functools import partial
import folx


def compute_jastrow_terms(sj, elec_coords, jastrow_params):
    """Compute ∇J/J and ∇²J/J with explicit parameters."""
    n_electrons = elec_coords.shape[0]
    
    # Memory-efficient implementation of the original averaged strategy
    # We use jax.lax.map to loop over electrons sequentially (O(N) memory)
    # instead of vmapping over all pairs (O(N^2) memory).
    # This preserves the exact math of:
    # grad[i] = 0.5 * (sum_{j!=i} grad_1(ri, rj) + sum_{j!=i} grad_2(rj, ri))
    
    indices = jnp.arange(n_electrons)
    
    def compute_electron_i(carrier):
        idx_i, r_i = carrier
        
        # Inner vmap over all other electrons j
        def compute_pair(idx_j, r_j):
            # Mask self-interaction
            mask = jnp.where(idx_i == idx_j, 0.0, 1.0)
            
            # grad_1(ri, rj) and lap_1(ri, rj)
            g1, l1 = sj.jastrow.get_log_grads_r1(r_i, r_j, jastrow_params)
            
            # grad_2(rj, ri) and lap_2(rj, ri)
            g2, l2 = sj.jastrow.get_log_grads_r2(r_j, r_i, jastrow_params)
            
            return (g1 * mask, l1 * mask, g2 * mask, l2 * mask)
            
        # vmap over j
        g1s, l1s, g2s, l2s = jax.vmap(compute_pair, in_axes=(0, 0))(indices, elec_coords)
        
        # Sum over j
        sum_g1 = jnp.sum(g1s, axis=0)
        sum_l1 = jnp.sum(l1s, axis=0)
        sum_g2 = jnp.sum(g2s, axis=0)
        sum_l2 = jnp.sum(l2s, axis=0)
        
        # Average
        grad_i = 0.5 * (sum_g1 + sum_g2)
        lap_i = 0.5 * (sum_l1 + sum_l2)
        
        return grad_i, lap_i

    # Map over electrons (sequential to save memory)
    grad_J_over_J, lap_sum = jax.lax.map(compute_electron_i, (indices, elec_coords))
    
    grad_squared = jnp.sum(grad_J_over_J**2, axis=1)
    lap_J_over_J = lap_sum + grad_squared
    
    return grad_J_over_J, lap_J_over_J
    
    grad_squared = jnp.sum(grad_J_over_J**2, axis=1)
    lap_J_over_J = lap_sum + grad_squared
    
    return grad_J_over_J, lap_J_over_J


def compute_potential_matrix(sj, elec_coords, slater_alpha, slater_beta):
    """Compute potential energy part of B matrix."""
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
    """Compute energy for a single walker.
    
    Args:
        sj: SlaterJastrow ansatz object
        walker: Walker object containing positions and Slater matrices
        jastrow_params: Jastrow parameters
        
    Returns:
        Local energy value
    """
    n_alpha = sj.dets[0].n_alpha
    
    # Compute Jastrow terms internally
    grad_J_over_J, lap_J_over_J = compute_jastrow_terms(sj, walker.positions, jastrow_params)
    
    grad_J_alpha = grad_J_over_J[:n_alpha]
    grad_J_beta = grad_J_over_J[n_alpha:]
    lap_J_alpha = lap_J_over_J[:n_alpha]
    lap_J_beta = lap_J_over_J[n_alpha:]
    
    B_kin_alpha = -0.5 * (
        walker.lap_up +
        2 * jnp.einsum('ik,ijk->ij', grad_J_alpha, walker.grad_up) +
        jnp.multiply(lap_J_alpha[:, None], walker.slater_up)
    )
    
    B_kin_beta = -0.5 * (
        walker.lap_down + 
        2 * jnp.einsum('ik,ijk->ij', grad_J_beta, walker.grad_down) +
        jnp.multiply(lap_J_beta[:, None], walker.slater_down)
    )
    
    B_pot_alpha, B_pot_beta = compute_potential_matrix(
        sj, walker.positions, walker.slater_up, walker.slater_down
    )
    
    E_L = (jnp.trace(walker.inv_up @ (B_kin_alpha + B_pot_alpha)) + 
           jnp.trace(walker.inv_down @ (B_kin_beta + B_pot_beta)))
    
    E_L = E_L + sj.ion_ion_potential
    
    return jnp.real(E_L)


def eval_local_energy(sj, walker, params):
    """Evaluate local energy for a SlaterJastrow ansatz.
    
    Args:
        sj: SlaterJastrow ansatz object
        walker: Walker object
        params: Tuple of (jastrow_params, linear_coeffs)
        
    Returns:
        Tuple of (energy, walker)
    """
    jastrow_params, linear_coeffs = params
    energy = compute_single_walker_energy(sj, walker, jastrow_params)
    return energy, walker
