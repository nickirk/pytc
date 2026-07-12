"""Walker state management for VMC sampling.

This module contains the Walker dataclass and functions for initializing
and managing walker states during Monte Carlo sampling.
"""

import time
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from .mcmc_utils import init_electron_configs


@dataclass
class Walker:
    """Batched walker state for MCMC sampling.
    
    All fields have shape (n_walkers, ...) in the first dimension.
    Memory estimate: For 100k walkers with 42 electrons (benzene):
    - Without grad/lap: ~1.5 GB
    - With grad/lap: ~4.3 GB (acceptable for modern systems)
    
    log_psi and log_jastrow cache the most recent wavefunction values so
    that the current ψ(R) does not need to be recomputed each MCMC step.
    """
    positions: jnp.ndarray      # (n_walkers, n_electrons, 3)
    det_up: jnp.ndarray         # (n_walkers,) or tuple of (sign, log|det|)
    det_down: jnp.ndarray       # (n_walkers,) or tuple of (sign, log|det|)
    slater_up: jnp.ndarray      # (n_walkers, n_alpha, n_alpha)
    slater_down: jnp.ndarray    # (n_walkers, n_beta, n_beta)
    inv_up: jnp.ndarray         # (n_walkers, n_alpha, n_alpha)
    inv_down: jnp.ndarray       # (n_walkers, n_beta, n_beta)
    grad_up: jnp.ndarray        # (n_walkers, n_alpha, n_alpha, 3)
    grad_down: jnp.ndarray      # (n_walkers, n_beta, n_beta, 3)
    lap_up: jnp.ndarray         # (n_walkers, n_alpha, n_alpha)
    lap_down: jnp.ndarray       # (n_walkers, n_beta, n_beta)
    move_mask: jnp.ndarray      # (n_walkers, n_electrons) boolean - tracks which electrons moved
    log_psi: jnp.ndarray        # (n_walkers,) cached log|ψ| value (sign stored in psi_sign)
    psi_sign: jnp.ndarray       # (n_walkers,) cached sign(ψ)
    log_jastrow: jnp.ndarray    # (n_walkers,) cached log(J) (Jastrow is always positive)
    
    @property
    def elec_coords(self):
        """Alias for positions for compatibility."""
        return self.positions
    
    @property
    def shape(self):
        """Return shape of positions for compatibility."""
        return self.positions.shape


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
        slater_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        slater_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        inv_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        inv_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        det_up=(jnp.zeros((n_walkers,)), jnp.zeros((n_walkers,))),  # (sign, log|det|) format
        det_down=(jnp.zeros((n_walkers,)), jnp.zeros((n_walkers,))),  # (sign, log|det|) format
        grad_up=jnp.zeros((n_walkers, n_alpha, n_alpha, 3)),
        grad_down=jnp.zeros((n_walkers, n_beta, n_beta, 3)),
        lap_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        lap_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        move_mask=jnp.ones((n_walkers, n_electrons), dtype=bool),
        log_psi=jnp.zeros((n_walkers,)),
        psi_sign=jnp.zeros((n_walkers,)),
        log_jastrow=jnp.zeros((n_walkers,)),
    )


def initialize_walkers(ansatz, n_walkers, initial_walkers=None, key=None, log_init: bool = True):
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

    # If initial_walkers is already a Walker, return it -- but only if its
    # walker count actually matches n_walkers. Silently returning a
    # checkpoint's Walker with the WRONG count corrupts every downstream
    # W-dependent computation (damping, batch shapes, statistics) with no
    # error signal. Callers that want a different walker count than a
    # saved checkpoint must resample explicitly first
    # (see mcmc_utils.resample_walkers).
    if isinstance(initial_walkers, Walker):
        actual_n = initial_walkers.positions.shape[0]
        if actual_n != n_walkers:
            raise ValueError(
                f"initialize_walkers: initial_walkers is a Walker with "
                f"{actual_n} walkers but n_walkers={n_walkers} was "
                f"requested. Resample to the target size explicitly "
                f"(mcmc_utils.resample_walkers) if you intend a different "
                f"walker count than the checkpoint -- this used to be a "
                f"silent no-op that returned the wrong walker count."
            )
        return initial_walkers

    # If initial_walkers are positions, use them
    if initial_walkers is not None:
        positions = initial_walkers
        if positions.shape[0] != n_walkers:
            raise ValueError(
                f"initialize_walkers: initial_walkers positions have "
                f"{positions.shape[0]} walkers but n_walkers={n_walkers} "
                f"was requested -- same silent-mismatch class as the "
                f"Walker-instance case above."
            )
    else:
        # Get molecular information needed for initialization
        # ansatz here is a SlaterDet, which now has atom_coords and atom_charges as attributes
        atom_coords = ansatz.atom_coords
        atom_charges = ansatz.atom_charges
        n_electrons = ansatz.n_electrons
        n_alpha = ansatz.n_alpha
        
        # Initialize electron positions based on nuclear positions and spin counts
        key, subkey = random.split(key)
        positions = init_electron_configs(
            atom_coords, atom_charges, n_electrons, n_walkers, subkey,
            n_alpha=n_alpha, log_init=log_init
        )
    
    # Create Walker state with all-True move_mask
    return initialize_walker_state(ansatz, positions)
