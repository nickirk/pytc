'''
Metropolis-Hastings sampling
'''

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import time
from functools import partial
from typing import Tuple, Dict, Any, Optional
import concurrent.futures
import os

def init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, key):
    """Initialize electron configurations based on atomic positions.
    
    Args:
        atom_coords: Array of atom coordinates with shape (n_atoms, 3)
        atom_charges: Array of atomic charges with shape (n_atoms,)
        n_electrons: Total number of electrons in the system
        n_walkers: Number of walker configurations to generate
        key: PRNG key for random initialization
        
    Returns:
        Array of shape (n_walkers, n_electrons, 3) with initial positions
    """
    # Initialize electrons near atoms proportional to nuclear charge
    n_atoms = len(atom_charges)
    
    # Calculate electron allocation per atom based on charge
    electron_counts = jnp.round(atom_charges / jnp.sum(atom_charges) * n_electrons).astype(jnp.int32)
    
    # Adjust to ensure correct total
    electron_counts = _adjust_electron_counts(electron_counts, n_electrons, atom_charges)
            
    # Generate positions with randomness around atoms
    positions = []
    
    # Place electrons around each atom with distance ~ 1.0 bohr
    for i in range(n_atoms):
        n_elec_at_atom = electron_counts[i]
        if n_elec_at_atom > 0:
            # Generate random directions
            key, subkey = random.split(key)
            directions = random.normal(subkey, (n_walkers, n_elec_at_atom, 3))
            directions = directions / jnp.linalg.norm(directions, axis=2, keepdims=True)
            
            # Generate distances (peaked around 1.0 bohr)
            key, subkey = random.split(key)
            distances = 1.0 + 0.1 * random.normal(subkey, (n_walkers, n_elec_at_atom, 1))
            
            # Calculate positions
            atom_pos = atom_coords[i]
            new_positions = atom_pos + directions * distances
            positions.append(new_positions)
    
    # Concatenate positions for all atoms
    all_positions = jnp.concatenate(positions, axis=1)
    
    # Ensure we have exactly n_electrons (in case of rounding issues)
    all_positions = all_positions[:, :n_electrons, :]
    
    return all_positions

# Remove the JIT decorator - this function isn't a performance bottleneck
def _adjust_electron_counts(electron_counts, n_electrons, atom_charges):
    """Helper function to adjust electron counts to match total."""
    # Convert to numpy for easier manipulation
    ec = np.array(electron_counts)
    ac = np.array(atom_charges)
    
    # Calculate current deficit
    deficit = int(n_electrons - np.sum(ec))
    
    # Add electrons one by one to atoms with highest remaining charge-to-electron ratio
    for _ in range(deficit):
        # Find atom with highest remaining charge-to-electron ratio
        idx = np.argmax(ac - ec)
        ec[idx] += 1
    
    # Convert back to JAX array
    return jnp.array(ec)

def metropolis_hastings(
    ansatz, 
    n_walkers: int, 
    n_steps: int, 
    step_size: float = 0.1, 
    burn_in: int = 1000,
    thinning: int = 10,
    n_samples: Optional[int] = None,
    initial_walkers=None,
    key=None
) -> Dict[str, Any]:
    """Perform Metropolis-Hastings sampling for quantum wavefunction.
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        n_walkers: Number of parallel walkers
        n_steps: Number of steps for each walker after burn-in
        step_size: Standard deviation of Gaussian proposal
        burn_in: Number of initial steps to discard (equilibration)
        thinning: Keep only every `thinning` steps to reduce autocorrelation
        n_samples: If provided, collect this many uncorrelated samples
        initial_walkers: Optional initial positions, otherwise initialized near nuclei
        key: PRNG key
    
    Returns:
        Dictionary with sampling results and statistics
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))
        
    # Initialize walkers
    if initial_walkers is None:
        key, subkey = random.split(key)
        # Get molecular information needed for initialization
        atom_coords = ansatz.mol.atom_coords()
        atom_charges = ansatz.mol.atom_charges()
        n_electrons = ansatz.n_electrons
        
        # Initialize electron positions based on nuclear positions
        walkers = init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, subkey)
    else:
        walkers = initial_walkers
    
    # Compute initial wavefunction values - use batch capability
    psi_values = ansatz(walkers)
    
    # Separate step sizes for up and down electrons can improve sampling
    step_size_up = step_size 
    step_size_down = step_size
    
    # Track acceptance rates
    accepted_total = 0
    acceptance_history = []
    
    # Burn-in phase
    print(f"Starting burn-in with {burn_in} steps...")
    for step in range(burn_in):
        # Move up-spin electrons
        key, subkey = random.split(key)
        up_walkers = walkers[:, :ansatz.n_up, :]
        up_proposals = up_walkers + random.normal(subkey, up_walkers.shape) * step_size_up
        
        # Create combined walkers with proposed up-spin positions
        proposals = walkers.at[:, :ansatz.n_up, :].set(up_proposals)
        
        # Compute acceptance probabilities directly using batch capability
        new_psi_values = ansatz(proposals)
        acceptance_prob = (jnp.abs(new_psi_values) / jnp.abs(psi_values))**2
        
        # Accept or reject
        key, subkey = random.split(key)
        accept_mask = random.uniform(subkey, shape=(walkers.shape[0],)) < acceptance_prob
        
        # Update walkers and psi values
        accept_mask_3d = accept_mask[:, jnp.newaxis, jnp.newaxis]
        walkers = jnp.where(accept_mask_3d, proposals, walkers)
        psi_values = jnp.where(accept_mask, new_psi_values, psi_values)
        
        # Move down-spin electrons similarly
        key, subkey = random.split(key)
        down_walkers = walkers[:, ansatz.n_up:, :]
        down_proposals = down_walkers + random.normal(subkey, down_walkers.shape) * step_size_down
        
        proposals = walkers.at[:, ansatz.n_up:, :].set(down_proposals)
        
        new_psi_values = ansatz(proposals)
        acceptance_prob = (jnp.abs(new_psi_values) / jnp.abs(psi_values))**2
        
        key, subkey = random.split(key)
        accept_mask = random.uniform(subkey, shape=(walkers.shape[0],)) < acceptance_prob
        
        accept_mask_3d = accept_mask[:, jnp.newaxis, jnp.newaxis]
        walkers = jnp.where(accept_mask_3d, proposals, walkers)
        psi_values = jnp.where(accept_mask, new_psi_values, psi_values)
        
        if step % 100 == 0:
            print(f"Burn-in step {step}/{burn_in}")
    
    print("Burn-in complete. Starting production sampling...")
    
    # Determine how many steps to actually run
    if n_samples is not None:
        # If we need n_samples and are thinning by thinning, calculate required steps
        required_steps = n_samples * thinning
        n_steps = max(n_steps, required_steps)
    
    # Storage for collected samples
    n_samples_to_collect = n_steps // thinning
    collected_samples = []
    collected_energies = []
    step_times = []
    
    # Main sampling loop
    for step in range(n_steps):
        start_time = time.time()
        
        # Move up-spin electrons
        key, subkey = random.split(key)
        up_walkers = walkers[:, :ansatz.n_up, :]
        up_proposals = up_walkers + random.normal(subkey, up_walkers.shape) * step_size_up
        
        # Create combined walkers with proposed up-spin positions
        proposals = walkers.at[:, :ansatz.n_up, :].set(up_proposals)
        
        # Compute acceptance probabilities
        new_psi_values = ansatz(proposals)
        acceptance_prob = (jnp.abs(new_psi_values) / jnp.abs(psi_values))**2
        
        # Accept or reject
        key, subkey = random.split(key)
        accept_mask = random.uniform(subkey, shape=(walkers.shape[0],)) < acceptance_prob
        accept_count = jnp.sum(accept_mask)
        
        # Update walkers and psi values
        accept_mask_3d = accept_mask[:, jnp.newaxis, jnp.newaxis]
        walkers = jnp.where(accept_mask_3d, proposals, walkers)
        psi_values = jnp.where(accept_mask, new_psi_values, psi_values)
        
        # Move down-spin electrons similarly
        key, subkey = random.split(key)
        down_walkers = walkers[:, ansatz.n_up:, :]
        down_proposals = down_walkers + random.normal(subkey, down_walkers.shape) * step_size_down
        
        proposals = walkers.at[:, ansatz.n_up:, :].set(down_proposals)
        
        new_psi_values = ansatz(proposals)
        acceptance_prob = (jnp.abs(new_psi_values) / jnp.abs(psi_values))**2
        
        key, subkey = random.split(key)
        accept_mask = random.uniform(subkey, shape=(walkers.shape[0],)) < acceptance_prob
        accept_count += jnp.sum(accept_mask)
        
        accept_mask_3d = accept_mask[:, jnp.newaxis, jnp.newaxis]
        walkers = jnp.where(accept_mask_3d, proposals, walkers)
        psi_values = jnp.where(accept_mask, new_psi_values, psi_values)
        
        # Track acceptance rate (for both up and down moves)
        acceptance_rate = float(accept_count) / (2 * n_walkers)
        acceptance_history.append(acceptance_rate)
        
        # Store samples at thinning interval
        if step % thinning == 0:
            # Use the new batch capability to compute local energies for all walkers at once
            energies = ansatz.local_energy(walkers)
            
            # Store samples and energies
            collected_samples.append(walkers)
            collected_energies.append(energies)
        
        step_time = time.time() - start_time
        step_times.append(step_time)
        
        # Print progress
        if step % 100 == 0 or step == n_steps - 1:
            avg_acceptance = jnp.mean(jnp.array(acceptance_history[-100:]))
            avg_time = jnp.mean(jnp.array(step_times[-100:]))
            print(f"Step {step}/{n_steps}, Acceptance: {avg_acceptance:.4f}, Time/step: {avg_time*1000:.2f}ms")
    
    # Stack collected samples and energies
    all_samples = jnp.stack(collected_samples) if collected_samples else None
    all_energies = jnp.stack(collected_energies) if collected_energies else None
    
    # Calculate statistics
    energy_mean = jnp.mean(all_energies) if all_energies is not None else None
    energy_std = jnp.std(all_energies) / jnp.sqrt(len(all_energies)) if all_energies is not None else None
    
    results = {
        "samples": all_samples,
        "energies": all_energies,
        "energy_mean": energy_mean,
        "energy_error": energy_std,
        "acceptance_rates": jnp.array(acceptance_history),
        "final_walkers": walkers,
        "step_times": jnp.array(step_times)
    }
    
    return results