"""Move generation algorithms for VMC sampling.

This module contains functions for generating different types of moves
in Monte Carlo sampling, including single-electron and multi-electron moves,
as well as importance sampling with drift-diffusion.
"""

import jax
import jax.numpy as jnp
from jax import random


def _all_electron_move(ansatz, walker, step_size, key, params, batch_ansatz=None):
    """Move all electrons at once for each walker.
    
    Args:
        ansatz: Wavefunction object
        walker: Walker dataclass with current state (batched)
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        params: Parameters for the ansatz
        batch_ansatz: Optional pre-vmapped ansatz function
        
    Returns:
        proposals: Walker with proposed new positions and all-True move_mask
        psi_values: Wavefunction values for current walker
        new_psi_values: Wavefunction values for proposals
        current_walker: Updated current walker (if ansatz updates it)
    """
    # Use provided batch_ansatz or create default vmap
    if batch_ansatz is None:
        batch_ansatz = jax.vmap(lambda w, p: ansatz(w, p), in_axes=(0, None))
    
    # Compute initial wavefunction values
    psi_values, current_walker = batch_ansatz(walker, params)
    
    # Generate proposals (all electrons move)
    key, subkey = random.split(key)
    new_positions = walker.positions + random.normal(subkey, walker.positions.shape) * step_size
    
    # Create proposal walker with all-True move_mask (all electrons moved)
    proposals = current_walker.replace(
        positions=new_positions,
        move_mask=jnp.ones_like(walker.move_mask)
    )
    
    # Compute new wavefunction values
    new_psi_values, proposals = batch_ansatz(proposals, params)
    
    # Return updated current_walker
    return psi_values, new_psi_values, current_walker, proposals

def _one_electron_move(ansatz, walker, step_size, key, params, batch_ansatz=None):
    """Move one randomly selected electron for each walker.
    
    Uses cached log_psi/psi_sign from the walker to avoid recomputing the
    current wavefunction value.  Only the *proposal* wavefunction is evaluated
    via batch_ansatz.
    
    Args:
        ansatz: Wavefunction object
        walker: Walker dataclass with current state (batched).
            Must have valid log_psi and psi_sign fields (populated by a
            prior batch_ansatz call).
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        params: Parameters for the ansatz
        batch_ansatz: Optional pre-vmapped ansatz function
        
    Returns:
        psi_values: Wavefunction (sign, log|ψ|) for current walker (from cache)
        new_psi_values: Wavefunction (sign, log|ψ|) for proposals
        walker_updated: Current walker (unchanged)
        proposals: Walker with proposed new positions, move_mask set, and
            all Slater fields recomputed for the proposal geometry.
    """
    # Use provided batch_ansatz or create default vmap
    if batch_ansatz is None:
        batch_ansatz = jax.vmap(lambda w, p: ansatz(w, p), in_axes=(0, None))
    
    # Select electron to move for each walker
    key, subkey = random.split(key)
    n_electrons = walker.positions.shape[1]
    electron_indices = random.randint(subkey, (walker.positions.shape[0],), 0, n_electrons)
    
    # Create move mask indicating which electron moved
    move_mask = (jnp.arange(n_electrons)[None, :] == electron_indices[:, None])
    
    # Generate proposals for selected electrons only
    key, subkey = random.split(key)
    mask_3d = move_mask[:, :, None]
    new_positions = walker.positions + mask_3d * random.normal(subkey, walker.positions.shape) * step_size
    
    # Create proposal walker with new positions and move mask
    proposals = walker.replace(
        positions=new_positions,
        move_mask=move_mask
    )
    
    # ---- Current ψ: use cached values from walker ----
    psi_values = (walker.psi_sign, walker.log_psi)
    
    # ---- Proposal ψ: full recomputation ----
    new_psi_values, proposals = batch_ansatz(proposals, params)
    
    return psi_values, new_psi_values, walker, proposals


@jax.jit
def _compute_green_function(r_target, r_source, quantum_force, time_step):
    """Compute transition probability density for drift-diffusion.
    
    G(R→R') = exp(-(R'-R-D*F(R)*τ)²/(2*τ))
    
    Args:
        r_target: Target position array
        r_source: Source position array
        quantum_force: Quantum force at source
        time_step: Time step
        
    Returns:
        Green's function values for transitions
    """
    # Calculate the expected drift
    drift = 0.5 * quantum_force * time_step
    
    # Calculate the difference between actual and drift-guided movement
    diff = r_target - r_source - drift
    
    # Calculate the exponent of the Green's function
    exponent = -jnp.sum(diff**2, axis=(-2, -1)) / (2.0 * time_step)
    
    # Return the Green's function values
    return jnp.exp(exponent)
