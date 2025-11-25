"""Metropolis-Hastings algorithms for VMC sampling.

This module contains the core Metropolis-Hastings sampling algorithms,
including both standard MCMC and importance sampling with drift-diffusion.
"""

import jax
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
    # psi_values and new_psi_values are now (sign, log|psi|) tuples
    # Acceptance probability: |ψ'|²/|ψ|² = exp(2*(log|ψ'| - log|ψ|))
    psi_sign, psi_logabs = psi_values
    new_psi_sign, new_psi_logabs = new_psi_values
    acceptance_prob = jnp.exp(2.0 * (new_psi_logabs - psi_logabs))
    
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
    # Handle det_up and det_down tuples (sign, log|det|) separately
    new_det_up = (
        jnp.where(accept_mask, proposals.det_up[0], current_walker.det_up[0]),  # signs
        jnp.where(accept_mask, proposals.det_up[1], current_walker.det_up[1])   # log|det|
    )
    new_det_down = (
        jnp.where(accept_mask, proposals.det_down[0], current_walker.det_down[0]),  # signs
        jnp.where(accept_mask, proposals.det_down[1], current_walker.det_down[1])   # log|det|
    )
    
    new_walker = current_walker.replace(
        positions=jnp.where(accept_mask_4d[:, :, :, 0], proposals.positions, current_walker.positions),
        slater_up=jnp.where(accept_mask_3d, proposals.slater_up, current_walker.slater_up),
        slater_down=jnp.where(accept_mask_3d, proposals.slater_down, current_walker.slater_down),
        inv_up=jnp.where(accept_mask_3d, proposals.inv_up, current_walker.inv_up),
        inv_down=jnp.where(accept_mask_3d, proposals.inv_down, current_walker.inv_down),
        det_up=new_det_up,
        det_down=new_det_down,
        grad_up=jnp.where(accept_mask_4d, proposals.grad_up, current_walker.grad_up),
        grad_down=jnp.where(accept_mask_4d, proposals.grad_down, current_walker.grad_down),
        lap_up=jnp.where(accept_mask_3d, proposals.lap_up, current_walker.lap_up),
        lap_down=jnp.where(accept_mask_3d, proposals.lap_down, current_walker.lap_down),
        move_mask=jnp.zeros_like(current_walker.move_mask)  # Reset to all False after accept/reject
    )
    
    # Calculate acceptance rate
    acceptance_rate = accept_count / n_walkers

    return new_walker, acceptance_rate


def metropolis_hastings_importance_sampling(ansatz, walkers, time_step, key, params):
    """Perform one step of Metropolis-Hastings with importance sampling (drift).
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
               Should also have a quantum_force method
        walkers: Batched walker configurations with shape (n_walkers, n_electrons, 3)
        time_step: Time step for the drift-diffusion process
        key: PRNG key
        params: Parameters for the ansatz, including jastrow and linear coefficients
    
    Returns:
        Tuple containing:
        - new_walkers: New walker configurations after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Compute initial wavefunction values and quantum forces with parameters
    # ansatz() and quantum_force() now work with single walkers, so vmap over batch
    batch_ansatz = jax.vmap(lambda w, p: ansatz(w, p), in_axes=(0, None))
    batch_quantum_force = jax.vmap(lambda w, p: ansatz.quantum_force(w, p), in_axes=(0, None))
    
    # ansatz returns ((sign, log), updated_walker)
    (psi_sign, psi_logabs), walkers = batch_ansatz(walkers, params)
    
    # Compute quantum force: F = 2∇ψ/ψ (gradient of log wavefunction)
    # Note: We use the updated walkers which might have cached matrices if ansatz updates them
    quantum_forces = batch_quantum_force(walkers, params)
    
    # Generate drift-diffusion proposals:
    # R' = R + D*F(R)*τ + √(2D*τ)*χ (D=0.5 in atomic units)
    drift_term = 0.5 * quantum_forces * time_step
    diffusion_coef = jnp.sqrt(time_step)
    
    key, subkey = random.split(key)
    random_term = diffusion_coef * random.normal(subkey, walkers.positions.shape)
    
    # Combine drift and diffusion terms for proposed positions
    proposed_positions = walkers.positions + drift_term + random_term
    
    # Create proposal walkers with new positions and full recomputation mask
    from .walker import initialize_walker_state
    proposal_walkers = initialize_walker_state(ansatz, proposed_positions)
    
    # Compute new wavefunction values and quantum forces at proposed positions
    (new_psi_sign, new_psi_logabs), proposal_walkers = batch_ansatz(proposal_walkers, params)
    new_quantum_forces = batch_quantum_force(proposal_walkers, params)
    
    # Modified acceptance probability for importance sampling
    # G(R→R') = exp(-(R'-R-D*F(R)*τ)²/(2*τ))
    forward_density = _compute_green_function(proposed_positions, walkers.positions, quantum_forces, time_step)
    backward_density = _compute_green_function(walkers.positions, proposed_positions, new_quantum_forces, time_step)
    
    # Compute acceptance probabilities with Green's function ratio
    # psi_values and new_psi_values are now (sign, log|psi|) tuples
    # Acceptance probability: |ψ'|²/|ψ|² * (G_back/G_fwd) = exp(2*(log|ψ'| - log|ψ|)) * (G_back/G_fwd)
    acceptance_prob = jnp.exp(2.0 * (new_psi_logabs - psi_logabs)) * (backward_density / forward_density)
    
    # Accept or reject
    key, subkey = random.split(key)
    accept_mask = random.uniform(subkey, shape=(walkers.positions.shape[0],)) < acceptance_prob
    accept_count = jnp.sum(accept_mask)
    
    # Update walkers: accepted moves get new walker state, rejected keep old state
    # We need to update all walker fields, not just positions
    accept_mask_3d = accept_mask[:, jnp.newaxis, jnp.newaxis]
    accept_mask_4d = accept_mask[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    accept_mask_1d = accept_mask[:, jnp.newaxis]
    
    # For tuple fields (det_up, det_down), unpack and update each component
    old_det_up_sign, old_det_up_logabs = walkers.det_up
    new_det_up_sign, new_det_up_logabs = proposal_walkers.det_up
    old_det_down_sign, old_det_down_logabs = walkers.det_down
    new_det_down_sign, new_det_down_logabs = proposal_walkers.det_down
    
    new_walkers = walkers.replace(
        positions=jnp.where(accept_mask_3d, proposed_positions, walkers.positions),
        slater_up=jnp.where(accept_mask_3d, proposal_walkers.slater_up, walkers.slater_up),
        slater_down=jnp.where(accept_mask_3d, proposal_walkers.slater_down, walkers.slater_down),
        inv_up=jnp.where(accept_mask_3d, proposal_walkers.inv_up, walkers.inv_up),
        inv_down=jnp.where(accept_mask_3d, proposal_walkers.inv_down, walkers.inv_down),
        det_up=(
            jnp.where(accept_mask, new_det_up_sign, old_det_up_sign),
            jnp.where(accept_mask, new_det_up_logabs, old_det_up_logabs)
        ),
        det_down=(
            jnp.where(accept_mask, new_det_down_sign, old_det_down_sign),
            jnp.where(accept_mask, new_det_down_logabs, old_det_down_logabs)
        ),
        grad_up=jnp.where(accept_mask_4d, proposal_walkers.grad_up, walkers.grad_up),
        grad_down=jnp.where(accept_mask_4d, proposal_walkers.grad_down, walkers.grad_down),
        lap_up=jnp.where(accept_mask_3d, proposal_walkers.lap_up, walkers.lap_up),
        lap_down=jnp.where(accept_mask_3d, proposal_walkers.lap_down, walkers.lap_down),
        move_mask=jnp.ones_like(walkers.move_mask, dtype=bool)  # All electrons moved
    )
    
    # Calculate acceptance rate
    acceptance_rate = accept_count / walkers.positions.shape[0]
    
    return new_walkers, acceptance_rate


def make_mcmc_step(ansatz, step_size, move_type="one"):
    """Factory to create a JIT-compilable MCMC step function.
    
    This function validates move_type at creation time (not JIT time) and returns
    a JIT-compilable step function that performs Metropolis-Hastings sampling.
    
    Args:
        ansatz: Wavefunction object (used for validation, not captured)
        step_size: Standard deviation of Gaussian proposal for MCMC moves
        move_type: "all" to move all electrons at once, "one" to move one electron at a time
    
    Returns:
        A JIT-compiled function with signature:
            mcmc_step(ansatz, walkers, key, params) -> (new_walkers, acceptance_rate)
    
    Raises:
        ValueError: If move_type is not "all" or "one"
    """
    # Validate move_type at factory creation time (not JIT time)
    if move_type not in ["all", "one"]:
        raise ValueError(f"move_type must be either 'all' or 'one', got '{move_type}'")
    
    def mcmc_step(ansatz, walkers, key, params):
        """Single MCMC step - fully JIT-compatible.
        
        Args:
            ansatz: Wavefunction object
            walkers: Walker dataclass with current state
            key: PRNG key for random number generation
            params: Parameters for the ansatz [jastrow_params, linear_coeffs]
        
        Returns:
            new_walkers: Updated walker state after MCMC step
            acceptance_rate: Fraction of proposals that were accepted
        """
        new_walkers, acceptance_rate = metropolis_hastings(
            ansatz, walkers, step_size, key, params, move_type=move_type
        )
        return new_walkers, acceptance_rate
    
    # JIT compile the step function
    return jax.jit(mcmc_step)


def make_mcmc_step_importance(ansatz, time_step):
    """Factory to create a JIT-compilable importance sampling MCMC step.
    
    Args:
        ansatz: Wavefunction object (used for validation, not captured)
        time_step: Time step for the drift-diffusion process
    
    Returns:
        A JIT-compiled function with signature:
            mcmc_step(ansatz, walkers, key, params) -> (new_walkers, acceptance_rate)
    """
    def mcmc_step(ansatz, walkers, key, params):
        """Single importance sampling MCMC step - fully JIT-compatible."""
        new_walkers, acceptance_rate = metropolis_hastings_importance_sampling(
            ansatz, walkers, time_step, key, params
        )
        return new_walkers, acceptance_rate
    
    # JIT compile the step function
    return jax.jit(mcmc_step)
