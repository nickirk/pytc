"""Move generation algorithms for VMC sampling.

This module contains functions for generating different types of moves
in Monte Carlo sampling, including single-electron and multi-electron moves,
as well as importance sampling with drift-diffusion.
"""

import jax
import jax.numpy as jnp
from jax import random


def _all_electron_move(ansatz, walker, step_size, key, params):
    """Move all electrons at once for each walker.
    
    Args:
        ansatz: Wavefunction object
        walker: Walker dataclass with current state
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        params: Parameters for the ansatz
        
    Returns:
        proposals: Walker with proposed new positions and all-True move_mask
        psi_values: Wavefunction values for current walker
        new_psi_values: Wavefunction values for proposals
    """
    # Compute initial wavefunction values
    psi_values, current_walker = ansatz(walker, params)
    
    # Generate proposals (all electrons move)
    key, subkey = random.split(key)
    new_positions = walker.positions + random.normal(subkey, walker.positions.shape) * step_size
    
    # Create proposal walker with all-True move_mask (all electrons moved)
    proposals = current_walker.replace(
        positions=new_positions,
        move_mask=jnp.ones_like(walker.move_mask)
    )
    
    # Compute new wavefunction values
    new_psi_values, proposals = ansatz(proposals, params)
    
    # Return updated current_walker
    return psi_values, new_psi_values, current_walker, proposals

def _one_electron_move(ansatz, walker, step_size, key, params):
    """Move one randomly selected electron for each walker.
    
    Args:
        ansatz: Wavefunction object
        walker: Walker dataclass with current state
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        params: Parameters for the ansatz
        
    Returns:
        proposals: Walker with proposed new positions and move_mask set
        psi_values: Wavefunction values for current walker
        new_psi_values: Wavefunction values for proposals
    """
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
    
    # Proposals have move_mask indicating moved electron
    new_psi_values, proposals = ansatz(proposals, params)
    
    return walker.psi_values, new_psi_values, walker, proposals


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
