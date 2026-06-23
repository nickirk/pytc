"""Utility functions for analyzing quantum Monte Carlo samples."""

import logging
import numpy as np
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__) 

def analyze_energies(sampling_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze energy convergence and statistics from sampling results.
    
    Args:
        sampling_results: Dictionary returned by metropolis_hastings
        
    Returns:
        Dictionary with energy statistics
    """
    energies = sampling_results["energies"]
    
    # Flatten the energies if they have walker dimension
    if len(energies.shape) > 1:
        flat_energies = energies.reshape(-1)
    else:
        flat_energies = energies
    
    # Calculate statistics
    energy_mean = jnp.mean(flat_energies)
    energy_error = jnp.std(flat_energies) / jnp.sqrt(len(flat_energies))
    energy_variance = jnp.var(flat_energies)
    
    # Calculate moving average
    window_size = 10
    cumsum = jnp.cumsum(jnp.insert(flat_energies, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    
    # Calculate autocorrelation
    n = len(flat_energies)
    mean = jnp.mean(flat_energies)
    var = jnp.var(flat_energies)
    
    # Simple autocorrelation calculation for lag-1
    autocorr_1 = jnp.sum((flat_energies[:-1] - mean) * (flat_energies[1:] - mean)) / ((n-1) * var)
    
    stats = {
        "mean": energy_mean,
        "error": energy_error,
        "variance": energy_variance,
        "autocorr_lag1": autocorr_1,
    }
    
    return stats

def prepare_sampling_results(samples, energies, acceptance_rates, walkers, step_times):
    """Prepare standardized sampling results dictionary.
    
    Args:
        samples: Collected walker samples
        energies: Computed energies for samples
        acceptance_rates: History of acceptance rates
        walkers: Final walker positions
        step_times: Time taken for each step
        
    Returns:
        Dictionary with standardized sampling results
    """
    # Stack collected samples and energies if they exist
    all_samples = jnp.stack(samples) if samples else None
    all_energies = jnp.concatenate(energies) if energies else None
    
    # Calculate statistics
    energy_mean = jnp.mean(all_energies) if all_energies is not None else None
    energy_std = jnp.std(all_energies) / jnp.sqrt(len(all_energies)) if all_energies is not None else None
    
    return {
        "samples": all_samples,
        "energies": all_energies,
        "energy_mean": energy_mean,
        "energy_error": energy_std,
        "acceptance_rates": jnp.array(acceptance_rates),
        "final_walkers": walkers,
        "step_times": jnp.array(step_times)
    }

def report_progress(step, total_steps, acceptance_history, step_times, energies=None):
    """Print progress information.
    
    Args:
        step: Current step number
        total_steps: Total number of steps
        acceptance_history: History of acceptance rates
        step_times: Time taken for each step
        energies: Optional collected energies
    """
    recent_acceptance = jnp.mean(jnp.array(acceptance_history[-100:]))
    recent_time = jnp.mean(jnp.array(step_times[-100:]))
    logger.info(f"Step {step}/{total_steps}, Acceptance: {recent_acceptance:.4f}, Time/step: {recent_time:.2f}s")

    if energies:
        recent_energy = jnp.mean(jnp.concatenate(energies))
        logger.info(f"  Current energy: {recent_energy:.6f}")



def init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, key, n_alpha=None, log_init: bool = True):
    """Initialize electron configurations based on atomic positions with proper spin ordering.
    
    Args:
        atom_coords: Array of atom coordinates with shape (n_atoms, 3)
        atom_charges: Array of atomic charges with shape (n_atoms,)
        n_electrons: Total number of electrons in the system
        n_walkers: Number of walker configurations to generate
        key: PRNG key for random initialization
        n_alpha: Number of up-spin electrons (if None, defaults to n_electrons//2)
        
    Returns:
        Array of shape (n_walkers, n_electrons, 3) with initial positions
        where the first n_alpha positions are up-spin electrons
    """
    # Initialize electrons near atoms proportional to nuclear charge
    n_atoms = len(atom_charges)
    
    # Set default n_alpha if not specified
    if n_alpha is None:
        n_alpha = n_electrons // 2
        
    n_beta = n_electrons - n_alpha
    
    # Use the pairing-based electron distribution algorithm
    alpha_counts, beta_counts = _distribute_electrons_by_pairing(atom_charges, n_alpha, n_beta)
    
    if log_init:
        logger.info("Electron distribution by atom:")
        for i in range(len(alpha_counts)):
            logger.info(f"  Atom {i}: {alpha_counts[i]} up, {beta_counts[i]} down")
    
    # Generate positions for up-spin electrons around each atom
    alpha_positions = []
    for i in range(n_atoms):
        n_alpha_at_atom = alpha_counts[i]
        if n_alpha_at_atom > 0:
            # Generate random directions
            key, subkey = random.split(key)
            directions = random.normal(subkey, (n_walkers, n_alpha_at_atom, 3))
            directions = directions / jnp.linalg.norm(directions, axis=2, keepdims=True)
            
            # Generate distances (peaked around 0.5 bohr)
            key, subkey = random.split(key)
            distances = 0.5 + 0.1 * random.normal(subkey, (n_walkers, n_alpha_at_atom, 1))
            
            # Calculate positions
            atom_pos = atom_coords[i]
            new_positions = atom_pos + directions * distances
            alpha_positions.append(new_positions)
    
    # Generate positions for down-spin electrons around each atom
    beta_positions = []
    for i in range(n_atoms):
        n_beta_at_atom = beta_counts[i]
        if n_beta_at_atom > 0:
            # Generate random directions
            key, subkey = random.split(key)
            directions = random.normal(subkey, (n_walkers, n_beta_at_atom, 3))
            directions = directions / jnp.linalg.norm(directions, axis=2, keepdims=True)
            
            # Generate distances (peaked around 0.5 bohr)
            key, subkey = random.split(key)
            distances = 1 + 0.3 * random.normal(subkey, (n_walkers, n_beta_at_atom, 1))
            
            # Calculate positions
            atom_pos = atom_coords[i]
            new_positions = atom_pos + directions * distances
            beta_positions.append(new_positions)
    
    # Concatenate all up positions and all down positions
    all_alpha_positions = jnp.concatenate(alpha_positions, axis=1) if alpha_positions else jnp.empty((n_walkers, 0, 3))
    all_beta_positions = jnp.concatenate(beta_positions, axis=1) if beta_positions else jnp.empty((n_walkers, 0, 3))
    
    # Ensure we have exactly the right number of electrons
    all_alpha_positions = all_alpha_positions[:, :n_alpha, :]
    all_beta_positions = all_beta_positions[:, :n_beta, :]
    
    # Combine up and down positions in correct order
    all_positions = jnp.concatenate([all_alpha_positions, all_beta_positions], axis=1)
    
    if log_init:
        logger.info(f"Initialized {n_alpha} up-spin and {n_beta} down-spin electrons around {n_atoms} atoms")
    
    return all_positions

def _distribute_electrons_by_pairing(atom_charges, n_alpha, n_beta):
    """Distribute electrons across atoms following physical pairing patterns.
    
    This algorithm follows the typical pattern of filling atomic orbitals:
    first up, then down, alternating until the atom is filled or we run out
    of electrons.
    
    Args:
        atom_charges: Array of atomic charges
        n_alpha: Total number of up-spin electrons to distribute
        n_beta: Total number of down-spin electrons to distribute
        
    Returns:
        Tuple of (alpha_counts, beta_counts) arrays showing distribution by atom
    """
    n_atoms = len(atom_charges)
    alpha_counts = np.zeros(n_atoms, dtype=np.int32)
    beta_counts = np.zeros(n_atoms, dtype=np.int32)
    
    remaining_alpha = n_alpha
    remaining_beta = n_beta
    
    # First pass: distribute electrons following alternating up/down pattern
    for i in range(n_atoms):
        atom_charge = int(atom_charges[i])
        atom_electrons = 0
        
        # Fill atom with alternating up/down until reaching charge limit
        while atom_electrons < atom_charge:
            # Try to add an up electron if we're at an even position
            if atom_electrons % 2 == 0 and remaining_alpha > 0:
                alpha_counts[i] += 1
                remaining_alpha -= 1
                atom_electrons += 1
            # Then try to add a down electron
            elif atom_electrons % 2 == 1 and remaining_beta > 0:
                beta_counts[i] += 1
                remaining_beta -= 1
                atom_electrons += 1
            else:
                # No more electrons of needed type or atom is full
                break
    
    # Second pass: handle any remaining electrons by assigned to highest charge atoms
    # (should be rare, but we need to handle it)
    atoms_by_charge = np.argsort(-atom_charges)  # Sort by descending charge
    
    # Distribute remaining up electrons
    for i in atoms_by_charge:
        while (alpha_counts[i] + beta_counts[i] < atom_charges[i]) and remaining_alpha > 0:
            alpha_counts[i] += 1
            remaining_alpha -= 1
    
    # Distribute remaining down electrons
    for i in atoms_by_charge:
        while (alpha_counts[i] + beta_counts[i] < atom_charges[i]) and remaining_beta > 0:
            beta_counts[i] += 1
            remaining_beta -= 1
    
    # If we still have electrons left, add them to the highest charge atoms
    # This could happen if total electrons > sum of charges
    for i in atoms_by_charge:
        while remaining_alpha > 0:
            alpha_counts[i] += 1
            remaining_alpha -= 1
    
    for i in atoms_by_charge:
        while remaining_beta > 0:
            beta_counts[i] += 1
            remaining_beta -= 1
            
    return jnp.array(alpha_counts), jnp.array(beta_counts)

def save_optimization_history(data: Dict[str, Any], filepath: str) -> str:
    """Save the optimization history dictionary to an HDF5 file.
    
    Args:
        data: Dictionary with keys 'cost', 'energies', 'stds', 'acceptance', 'params'
            as outputted by the optimization loop.
        filepath: Path to the HDF5 file (e.g., 'optimization_results.h5').
        
    Returns:
        Path to the saved HDF5 file.
    """
    import h5py
    import jax
    import numpy as np
    
    def _save_element(group, name, item):
        """Recursively saves elements (dicts, lists, arrays) to an HDF5 group."""
        
        # Handle dictionary-like objects (e.g., dict, flax FrozenDict)
        if hasattr(item, 'items') and callable(item.items):
            subgroup = group.create_group(name)
            for k, v in item.items():
                # HDF5 keys must be strings
                _save_element(subgroup, str(k), v)
                
        # Handle lists or tuples containing sub-trees (e.g., layer parameters)
        elif isinstance(item, (list, tuple)):
            subgroup = group.create_group(name)
            if isinstance(item, tuple):
                subgroup.attrs['__is_tuple__'] = True
            for i, v in enumerate(item):
                _save_element(subgroup, str(i), v)
                
        # Base case: leaves of the tree (numpy arrays, scalars)
        else:
            try:
                group.create_dataset(name, data=item)
            except TypeError:
                # If h5py doesn't naturally support the type, cast it to string
                group.create_dataset(name, data=str(item))

    with h5py.File(filepath, 'w') as f:
        for key, value in data.items():
            if key == 'params':
                # Convert list of PyTrees -> PyTree of stacked arrays (axis 0 is the step)
                stacked_params = jax.tree_util.tree_map(
                    lambda *leaves: np.stack(leaves), 
                    *value
                )
                # Now save the single stacked PyTree structure
                _save_element(f, 'params', stacked_params)
            else:
                # Top level metrics (cost, energies, stds, acceptance) are arrays
                f.create_dataset(key, data=value)
                
    logger.info(f"Successfully saved optimization history to {filepath}")
    return filepath

def load_optimization_history(filepath: str) -> Dict[str, Any]:
    """Load the optimization history from an HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file.
        
    Returns:
        Dictionary with 'cost', 'energies', 'stds', 'acceptance', 'params'.
        'params' is a PyTree where each leaf is stacked along axis=0 (the step).
    """
    import h5py
    
    def _load_element(item):
        """Recursively reconstructs the dictionary/list tree from an HDF5 group/dataset."""
        if isinstance(item, h5py.Group):
            result = {}
            for k, v in item.items():
                if k == '__is_tuple__':
                    continue
                result[k] = _load_element(v)
                
            # Check if this group was originally a list/tuple (all keys are digits)
            if all(k.isdigit() for k in result.keys()) and len(result) > 0:
                # Reconstruct list/tuple safely
                max_idx = max(int(k) for k in result.keys())
                sub_list = [None] * (max_idx + 1)
                for k, v in result.items():
                    sub_list[int(k)] = v
                    
                # If we marked it as a tuple, convert it back to a tuple
                if '__is_tuple__' in item.attrs and item.attrs['__is_tuple__']:
                    return tuple(sub_list)
                    
                return sub_list
            return result
        else:
            # It's a dataset, pull the numpy array into memory
            return item[()]
            
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if key == 'params':
                data[key] = _load_element(f[key])
            else:
                data[key] = f[key][()]
                
    return data
