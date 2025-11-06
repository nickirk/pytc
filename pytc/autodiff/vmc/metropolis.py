"""Metropolis-Hastings algorithms for VMC sampling.

This module contains the core Metropolis-Hastings sampling algorithms,
including both standard MCMC and importance sampling with drift-diffusion.
"""

import gc
import jax.numpy as jnp
from jax import random

from .moves import _all_electron_move, _one_electron_move, _compute_green_function
from .walker import Walker


def metropolis_hastings(ansatz, walker, step_size, key, params, move_type="one"):
    """Perform one step of Metropolis-Hastings sampling for quantum wavefunction.
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        walker: Walker dataclass with current state
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        params: contains jastrow_params and linear_coeffs
        move_type: "all" to move all electrons at once, "one" to move one electron at a time
    
    Returns:
        Tuple containing:
        - new_walker: New walker state after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Choose move type
    if move_type == "all":
        psi_values, new_psi_values, current_walker, proposals = _all_electron_move(ansatz, walker, step_size, key, params)
    elif move_type == "one":
        psi_values, new_psi_values, current_walker, proposals = _one_electron_move(ansatz, walker, step_size, key, params)
    else:
        raise ValueError("move_type must be either 'all' or 'one'")
    
    # Compute acceptance probabilities
    acceptance_prob = (jnp.abs(new_psi_values) / jnp.abs(psi_values))**2
    
    # Accept or reject
    key, subkey = random.split(key)
    n_walkers = walker.positions.shape[0]
    accept_mask = random.uniform(subkey, shape=(n_walkers,)) < acceptance_prob
    accept_count = jnp.sum(accept_mask)
    
    # Create new walker by selecting accepted proposals or keeping current
    # Reshape accept_mask for broadcasting
    accept_mask_3d = accept_mask[:, None, None]  # For 2D matrices
    accept_mask_4d = accept_mask[:, None, None, None]  # For gradients (3D tensors)

    # Use current_walker (with updated matrices) instead of original walker
    new_walker = current_walker.replace(
        positions=jnp.where(accept_mask_4d[:, :, :, 0], proposals.positions, current_walker.positions),
        psi_values=jnp.where(accept_mask, new_psi_values, current_walker.psi_values),
        slater_up=jnp.where(accept_mask_3d, proposals.slater_up, current_walker.slater_up),
        slater_down=jnp.where(accept_mask_3d, proposals.slater_down, current_walker.slater_down),
        inv_up=jnp.where(accept_mask_3d, proposals.inv_up, current_walker.inv_up),
        inv_down=jnp.where(accept_mask_3d, proposals.inv_down, current_walker.inv_down),
        det_up=jnp.where(accept_mask, proposals.det_up, current_walker.det_up),
        det_down=jnp.where(accept_mask, proposals.det_down, current_walker.det_down),
        grad_up=jnp.where(accept_mask_4d, proposals.grad_up, current_walker.grad_up),
        grad_down=jnp.where(accept_mask_4d, proposals.grad_down, current_walker.grad_down),
        lap_up=jnp.where(accept_mask_3d, proposals.lap_up, current_walker.lap_up),
        lap_down=jnp.where(accept_mask_3d, proposals.lap_down, current_walker.lap_down),
        move_mask=jnp.zeros_like(current_walker.move_mask)  # Reset to all False after accept/reject
    )
    
    # Calculate acceptance rate
    acceptance_rate = accept_count / n_walkers
    
    # Explicitly delete intermediate walkers to help garbage collection
    del walker, psi_values, new_psi_values, current_walker, proposals, accept_mask, accept_mask_3d, accept_mask_4d
    gc.collect()

    return new_walker, acceptance_rate


def metropolis_hastings_importance_sampling(ansatz, walkers, time_step, key, params):
    """Perform one step of Metropolis-Hastings with importance sampling (drift).
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
               Should also have a quantum_force method
        walkers: Array of walker configurations with shape (n_walkers, n_electrons, 3)
        time_step: Time step for the drift-diffusion process
        key: PRNG key
        params: Parameters for the ansatz, including jastrow and linear coefficients
    
    Returns:
        Tuple containing:
        - new_walkers: New walker configurations after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Compute initial wavefunction values and quantum forces with parameters
    psi_values = ansatz(walkers, params)
    
    # Compute quantum force: F = 2∇ψ/ψ (gradient of log wavefunction)
    quantum_forces = ansatz.quantum_force(walkers, params)
    
    # Generate drift-diffusion proposals:
    # R' = R + D*F(R)*τ + √(2D*τ)*χ (D=0.5 in atomic units)
    drift_term = 0.5 * quantum_forces * time_step
    diffusion_coef = jnp.sqrt(time_step)
    
    key, subkey = random.split(key)
    random_term = diffusion_coef * random.normal(subkey, walkers.shape)
    
    # Combine drift and diffusion terms
    proposals = walkers + drift_term + random_term
    
    # Compute new wavefunction values and quantum forces at proposed positions
    new_psi_values = ansatz(proposals, params)
    new_quantum_forces = ansatz.quantum_force(proposals, params)
    
    # Modified acceptance probability for importance sampling
    # G(R→R') = exp(-(R'-R-D*F(R)*τ)²/(2*τ))
    forward_density = _compute_green_function(proposals, walkers, quantum_forces, time_step)
    backward_density = _compute_green_function(walkers, proposals, new_quantum_forces, time_step)
    
    # Compute acceptance probabilities with Green's function ratio
    acceptance_prob = (jnp.abs(new_psi_values) / jnp.abs(psi_values))**2 * (backward_density / forward_density)
    
    # Accept or reject
    key, subkey = random.split(key)
    accept_mask = random.uniform(subkey, shape=(walkers.shape[0],)) < acceptance_prob
    accept_count = jnp.sum(accept_mask)
    
    # Create new walkers without modifying input
    accept_mask_3d = accept_mask[:, jnp.newaxis, jnp.newaxis]
    new_walkers = jnp.where(accept_mask_3d, proposals, walkers)
    
    # Calculate acceptance rate
    acceptance_rate = accept_count / walkers.shape[0]
    
    return new_walkers, acceptance_rate
