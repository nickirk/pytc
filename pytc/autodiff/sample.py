'''
Metropolis-Hastings sampling
'''

import numpy as np
import jax.numpy as jnp
from jax import random
import time
from typing import Dict, Any, Optional

def init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, key, n_up=None):
    """Initialize electron configurations based on atomic positions with proper spin ordering.
    
    Args:
        atom_coords: Array of atom coordinates with shape (n_atoms, 3)
        atom_charges: Array of atomic charges with shape (n_atoms,)
        n_electrons: Total number of electrons in the system
        n_walkers: Number of walker configurations to generate
        key: PRNG key for random initialization
        n_up: Number of up-spin electrons (if None, defaults to n_electrons//2)
        
    Returns:
        Array of shape (n_walkers, n_electrons, 3) with initial positions
        where the first n_up positions are up-spin electrons
    """
    # Initialize electrons near atoms proportional to nuclear charge
    n_atoms = len(atom_charges)
    
    # Set default n_up if not specified
    if n_up is None:
        n_up = n_electrons // 2
        
    n_down = n_electrons - n_up
    
    # Use the pairing-based electron distribution algorithm
    up_counts, down_counts = _distribute_electrons_by_pairing(atom_charges, n_up, n_down)
    
    print(f"Electron distribution by atom:")
    for i in range(len(up_counts)):
        print(f"  Atom {i}: {up_counts[i]} up, {down_counts[i]} down")
    
    # Generate positions for up-spin electrons around each atom
    up_positions = []
    for i in range(n_atoms):
        n_up_at_atom = up_counts[i]
        if n_up_at_atom > 0:
            # Generate random directions
            key, subkey = random.split(key)
            directions = random.normal(subkey, (n_walkers, n_up_at_atom, 3))
            directions = directions / jnp.linalg.norm(directions, axis=2, keepdims=True)
            
            # Generate distances (peaked around 0.5 bohr)
            key, subkey = random.split(key)
            distances = 0.5 + 0.1 * random.normal(subkey, (n_walkers, n_up_at_atom, 1))
            
            # Calculate positions
            atom_pos = atom_coords[i]
            new_positions = atom_pos + directions * distances
            up_positions.append(new_positions)
    
    # Generate positions for down-spin electrons around each atom
    down_positions = []
    for i in range(n_atoms):
        n_down_at_atom = down_counts[i]
        if n_down_at_atom > 0:
            # Generate random directions
            key, subkey = random.split(key)
            directions = random.normal(subkey, (n_walkers, n_down_at_atom, 3))
            directions = directions / jnp.linalg.norm(directions, axis=2, keepdims=True)
            
            # Generate distances (peaked around 0.5 bohr)
            key, subkey = random.split(key)
            distances = 1 + 0.3 * random.normal(subkey, (n_walkers, n_down_at_atom, 1))
            
            # Calculate positions
            atom_pos = atom_coords[i]
            new_positions = atom_pos + directions * distances
            down_positions.append(new_positions)
    
    # Concatenate all up positions and all down positions
    all_up_positions = jnp.concatenate(up_positions, axis=1) if up_positions else jnp.empty((n_walkers, 0, 3))
    all_down_positions = jnp.concatenate(down_positions, axis=1) if down_positions else jnp.empty((n_walkers, 0, 3))
    
    # Ensure we have exactly the right number of electrons
    all_up_positions = all_up_positions[:, :n_up, :]
    all_down_positions = all_down_positions[:, :n_down, :]
    
    # Combine up and down positions in correct order
    all_positions = jnp.concatenate([all_up_positions, all_down_positions], axis=1)
    
    print(f"Initialized {n_up} up-spin and {n_down} down-spin electrons around {n_atoms} atoms")
    
    return all_positions

def _distribute_electrons_by_pairing(atom_charges, n_up, n_down):
    """Distribute electrons across atoms following physical pairing patterns.
    
    This algorithm follows the typical pattern of filling atomic orbitals:
    first up, then down, alternating until the atom is filled or we run out
    of electrons.
    
    Args:
        atom_charges: Array of atomic charges
        n_up: Total number of up-spin electrons to distribute
        n_down: Total number of down-spin electrons to distribute
        
    Returns:
        Tuple of (up_counts, down_counts) arrays showing distribution by atom
    """
    n_atoms = len(atom_charges)
    up_counts = np.zeros(n_atoms, dtype=np.int32)
    down_counts = np.zeros(n_atoms, dtype=np.int32)
    
    remaining_up = n_up
    remaining_down = n_down
    
    # First pass: distribute electrons following alternating up/down pattern
    for i in range(n_atoms):
        atom_charge = int(atom_charges[i])
        atom_electrons = 0
        
        # Fill atom with alternating up/down until reaching charge limit
        while atom_electrons < atom_charge:
            # Try to add an up electron if we're at an even position
            if atom_electrons % 2 == 0 and remaining_up > 0:
                up_counts[i] += 1
                remaining_up -= 1
                atom_electrons += 1
            # Then try to add a down electron
            elif atom_electrons % 2 == 1 and remaining_down > 0:
                down_counts[i] += 1
                remaining_down -= 1
                atom_electrons += 1
            else:
                # No more electrons of needed type or atom is full
                break
    
    # Second pass: handle any remaining electrons by assigned to highest charge atoms
    # (should be rare, but we need to handle it)
    atoms_by_charge = np.argsort(-atom_charges)  # Sort by descending charge
    
    # Distribute remaining up electrons
    for i in atoms_by_charge:
        while (up_counts[i] + down_counts[i] < atom_charges[i]) and remaining_up > 0:
            up_counts[i] += 1
            remaining_up -= 1
    
    # Distribute remaining down electrons
    for i in atoms_by_charge:
        while (up_counts[i] + down_counts[i] < atom_charges[i]) and remaining_down > 0:
            down_counts[i] += 1
            remaining_down -= 1
    
    # If we still have electrons left, add them to the highest charge atoms
    # This could happen if total electrons > sum of charges
    for i in atoms_by_charge:
        while remaining_up > 0:
            up_counts[i] += 1
            remaining_up -= 1
    
    for i in atoms_by_charge:
        while remaining_down > 0:
            down_counts[i] += 1
            remaining_down -= 1
            
    return jnp.array(up_counts), jnp.array(down_counts)


def metropolis_hastings(
    ansatz, 
    n_walkers: int, 
    n_steps: int, 
    step_size: float = 1, 
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
        n_up = ansatz.n_up  # Use n_up from the ansatz
        
        # Initialize electron positions based on nuclear positions and spin counts
        walkers = init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, subkey, n_up=n_up)
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
            # print the current average energy
            energies = jnp.array(collected_energies)
            print(f"  Current energy: {jnp.mean(energies):.6f}")
    
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