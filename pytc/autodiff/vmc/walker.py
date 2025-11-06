"""Walker state management for VMC sampling.

This module contains the Walker dataclass and functions for initializing
and managing walker states during Monte Carlo sampling.
"""

import time
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
from typing import Optional

from .mcmc_utils import init_electron_configs


@dataclass
class Walker:
    """Batched walker state for MCMC sampling.
    
    All fields have shape (n_walkers, ...) in the first dimension.
    Memory estimate: For 100k walkers with 42 electrons (benzene):
    - Without grad/lap: ~1.5 GB
    - With grad/lap: ~4.3 GB (acceptable for modern systems)
    """
    positions: jnp.ndarray      # (n_walkers, n_electrons, 3)
    psi_values: jnp.ndarray   # (n_walkers,)
    det_up: jnp.ndarray         # (n_walkers,)
    det_down: jnp.ndarray       # (n_walkers,)
    slater_up: jnp.ndarray      # (n_walkers, n_alpha, n_alpha)
    slater_down: jnp.ndarray    # (n_walkers, n_beta, n_beta)
    inv_up: jnp.ndarray         # (n_walkers, n_alpha, n_alpha)
    inv_down: jnp.ndarray       # (n_walkers, n_beta, n_beta)
    grad_up: jnp.ndarray        # (n_walkers, n_alpha, n_alpha, 3)
    grad_down: jnp.ndarray      # (n_walkers, n_beta, n_beta, 3)
    lap_up: jnp.ndarray         # (n_walkers, n_alpha, n_alpha)
    lap_down: jnp.ndarray       # (n_walkers, n_beta, n_beta)
    move_mask: jnp.ndarray      # (n_walkers, n_electrons) boolean - tracks which electrons moved


def initialize_walker_state(ansatz, positions):
    """Initialize Walker state with positions and all-True move_mask.
    
    Args:
        ansatz: Wavefunction object (contains determinant info)
        positions: Array of initial positions with shape (n_walkers, n_electrons, 3)
        
    Returns:
        Walker: Initialized walker state with:
            - positions: provided positions
            - move_mask: all True (indicates full computation needed)
            - all other fields: zeros (will be computed on first ansatz call)
    """
    n_walkers, n_electrons = positions.shape[0], positions.shape[1]
    n_alpha = ansatz.n_alpha
    n_beta = n_electrons - n_alpha
    
    return Walker(
        positions=positions,
        psi_values=jnp.zeros((n_walkers,)),
        slater_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        slater_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        inv_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        inv_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        det_up=jnp.zeros((n_walkers,)),
        det_down=jnp.zeros((n_walkers,)),
        grad_up=jnp.zeros((n_walkers, n_alpha, n_alpha, 3)),
        grad_down=jnp.zeros((n_walkers, n_beta, n_beta, 3)),
        lap_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        lap_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        move_mask=jnp.ones((n_walkers, n_electrons), dtype=bool)
    )


def initialize_walkers(ansatz, n_walkers, initial_walkers=None, key=None):
    """Initialize walker configurations based on molecular structure.
    
    Args:
        ansatz: Wavefunction object with molecular information
        n_walkers: Number of parallel walkers
        initial_walkers: Optional initial Walker state or positions
        key: PRNG key
        
    Returns:
        Walker: Initialized walker state with all-True move_mask
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))
    
    # If initial_walkers is already a Walker, return it
    if isinstance(initial_walkers, Walker):
        return initial_walkers
    
    # If initial_walkers are positions, use them
    if initial_walkers is not None:
        positions = initial_walkers
    else:
        # Get molecular information needed for initialization
        atom_coords = ansatz.mol.atom_coords()
        atom_charges = ansatz.mol.atom_charges()
        n_electrons = ansatz.n_electrons
        n_alpha = ansatz.n_alpha
        
        # Initialize electron positions based on nuclear positions and spin counts
        key, subkey = random.split(key)
        positions = init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, subkey, n_alpha=n_alpha)
    
    # Create Walker state with all-True move_mask
    return initialize_walker_state(ansatz, positions)
