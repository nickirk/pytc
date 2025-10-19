"""Utility functions for analyzing quantum Monte Carlo samples."""

import numpy as np
import jax.numpy as jnp
import optax
import kfac_jax
from jax import random
from jax.lax import stop_gradient
from typing import Dict, Any, List, Optional, Tuple
from jax.tree_util import tree_map 

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

def electron_density(sampling_results: Dict[str, Any], grid_points: int = 50, 
                     range_xyz: Optional[List[Tuple[float, float]]] = None) -> jnp.ndarray:
    """Calculate electron density from sampling results.
    
    Args:
        sampling_results: Dictionary returned by metropolis_hastings
        grid_points: Number of grid points in each dimension
        range_xyz: Optional list of (min, max) tuples for x, y, z dimensions
                   If None, automatically determined from samples
                   
    Returns:
        3D array representing electron density
    """
    samples = sampling_results["samples"]
    
    # Flatten all walker and step dimensions to get list of all electron positions
    all_electrons = samples.reshape(-1, samples.shape[-2], samples.shape[-1])
    all_positions = all_electrons.reshape(-1, all_electrons.shape[-1])
    
    # Determine grid range if not provided
    if range_xyz is None:
        min_xyz = jnp.min(all_positions, axis=0) - 1.0
        max_xyz = jnp.max(all_positions, axis=0) + 1.0
        range_xyz = [(min_xyz[i], max_xyz[i]) for i in range(3)]
    
    # Create grid
    x = jnp.linspace(range_xyz[0][0], range_xyz[0][1], grid_points)
    y = jnp.linspace(range_xyz[1][0], range_xyz[1][1], grid_points)
    z = jnp.linspace(range_xyz[2][0], range_xyz[2][1], grid_points)
    
    # Initialize density array
    density = jnp.zeros((grid_points, grid_points, grid_points))
    
    # Simple histogram approach for electron density
    x_indices = jnp.clip(jnp.floor((all_positions[:, 0] - range_xyz[0][0]) / 
                          (range_xyz[0][1] - range_xyz[0][0]) * grid_points).astype(jnp.int32), 
                          0, grid_points-1)
    y_indices = jnp.clip(jnp.floor((all_positions[:, 1] - range_xyz[1][0]) / 
                          (range_xyz[1][1] - range_xyz[1][0]) * grid_points).astype(jnp.int32), 
                          0, grid_points-1)
    z_indices = jnp.clip(jnp.floor((all_positions[:, 2] - range_xyz[2][0]) / 
                          (range_xyz[2][1] - range_xyz[2][0]) * grid_points).astype(jnp.int32), 
                          0, grid_points-1)
    
    # Use numpy for histogram generation since jax doesn't have an equivalent
    import numpy as np
    density_np = np.zeros((grid_points, grid_points, grid_points))
    
    # Convert JAX arrays to numpy
    x_indices_np = np.array(x_indices)
    y_indices_np = np.array(y_indices)
    z_indices_np = np.array(z_indices)
    
    # Count electrons in each grid cell
    for i in range(len(x_indices_np)):
        density_np[x_indices_np[i], y_indices_np[i], z_indices_np[i]] += 1
    
    # Normalize
    density = jnp.array(density_np) / len(all_positions)
    
    return density, (x, y, z)

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
    print(f"Step {step}/{total_steps}, Acceptance: {recent_acceptance:.4f}, Time/step: {recent_time:.2f}s")
    
    if energies:
        recent_energy = jnp.mean(jnp.concatenate(energies))
        print(f"  Current energy: {recent_energy:.6f}")

def create_optimizer(optimizer_type, learning_rate, opt_kwargs=None):
    """Create an optimizer based on specified type and parameters."""
    if opt_kwargs is None:
        opt_kwargs = {}
    
    base_kwargs = {}
    merged_kwargs = {**base_kwargs, **opt_kwargs}
    def schedule_lr(step):
        return learning_rate / (1.0 + step/100)
    
    if optimizer_type.lower() == "kfac":
        # K-FAC requires a value_and_grad_func, which should be provided in opt_kwargs
        if "value_and_grad_func" not in merged_kwargs:
            raise ValueError("KFAC optimizer requires value_and_grad_func in opt_kwargs")
        
        return kfac_jax.Optimizer(
            value_and_grad_func=merged_kwargs["value_and_grad_func"],
            l2_reg=merged_kwargs.get("l2_reg", 0.0),
            value_func_has_aux=merged_kwargs.get("value_func_has_aux", False),
            value_func_has_state=merged_kwargs.get("value_func_has_state", False),
            value_func_has_rng=merged_kwargs.get("value_func_has_rng", False),
            use_adaptive_learning_rate=merged_kwargs.get("use_adaptive_learning_rate", True),
            use_adaptive_momentum=merged_kwargs.get("use_adaptive_momentum", True),
            use_adaptive_damping=merged_kwargs.get("use_adaptive_damping", True),
            initial_damping=merged_kwargs.get("initial_damping", 1.0),
            num_burnin_steps=merged_kwargs.get("num_burnin_steps", 0),  # Set to 0 by default to avoid requiring data_iterator
            multi_device=merged_kwargs.get("multi_device", False),
        )
    elif optimizer_type.lower() == "adam":
        #return optax.adamw(learning_rate=schedule_lr)
        return optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "sgd":
        return optax.chain(
            optax.sgd(learning_rate=learning_rate),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "rmsprop":
        return optax.chain(
            optax.scale_by_rms(decay=merged_kwargs.get("decay", 0.9), eps=merged_kwargs.get("eps", 1e-8)),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "lion":
        return optax.lion(learning_rate=learning_rate, b1=merged_kwargs.get("b1", 0.9), b2=merged_kwargs.get("b2", 0.99))
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, key, n_alpha=None):
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
    
    print(f"Electron distribution by atom:")
    for i in range(len(alpha_counts)):
        print(f"  Atom {i}: {alpha_counts[i]} up, {beta_counts[i]} down")
    
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
    
    print(f"Initialized {n_alpha} up-spin and {n_beta} down-spin electrons around {n_atoms} atoms")
    
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

def create_gradient_mask(ansatz, params, frozen_params):
    """Create a gradient mask PyTree for the combined params structure.
    
    Assumes params = [jastrow_params, linear_coeffs]. The mask is applied
    only to the jastrow_params part based on frozen_params identifiers.
    The linear_coeffs part of the mask is always True (not frozen).

    Args:
        ansatz: The wavefunction ansatz object.
        params: The combined parameters PyTree [jastrow_params, linear_coeffs].
        frozen_params: A list of identifiers (int index or str name/type)
                       for Jastrow factors whose parameters should be frozen.

    Returns:
        A PyTree with the same structure as params, where frozen parameters
        are wrapped with `jax.lax.stop_gradient`.
    """
    if not frozen_params:
        return params  # No freezing requested, return params unchanged

    if not isinstance(params, (list, tuple)) or len(params) != 2:
        raise ValueError("`params` must be a list or tuple: [jastrow_params, linear_coeffs]")

    jastrow_params = params[0]
    linear_coeffs = params[1]

    print(f"Creating gradient mask for frozen Jastrow parameters: {frozen_params}")
    jastrows = ansatz.jastrow.jastrows
    if not isinstance(jastrow_params, (list, tuple)) or len(jastrow_params) != len(jastrows):
        raise TypeError(f"Jastrow params structure (length {len(jastrow_params)}) does not match jastrows (length {len(jastrows)})")

    # Deep copy the parameters to avoid modifying the input
    masked_jastrow_params = []
    for i, (param_pytree, jastrow) in enumerate(zip(jastrow_params, jastrows)):
        should_freeze = False
        for fp in frozen_params:
            if isinstance(fp, int) and fp == i:
                should_freeze = True
                break
            elif isinstance(fp, str):
                if fp == jastrow.__class__.__name__ or fp == getattr(jastrow, 'name', None):
                    should_freeze = True
                    break
        
        if should_freeze:
            # Apply stop_gradient to all leaves in the frozen parameter PyTree
            param_pytree = tree_map(stop_gradient, param_pytree)
            print(f"  Freezing Jastrow {i}: type={jastrow.__class__.__name__}, name={getattr(jastrow, 'name', None)}")
            print("Warning: Freezing parameters does not work for KFAC yet.")
        masked_jastrow_params.append(param_pytree)

    # Return the masked parameters
    return [masked_jastrow_params, linear_coeffs]

def apply_gradient_mask(grads, mask):
    """Apply gradient mask to gradients to freeze parameters.
    
    Args:
        grads: The gradient PyTree
        mask: The mask PyTree created by create_gradient_mask
        
    Returns:
        A PyTree with the same structure as grads, where gradients for frozen
        parameters are set to zero.
    """
    if mask is None:
        return grads
        
    def _apply_mask(g, m):
        # If m is already stop_gradient'd, zero out the gradient
        if isinstance(m, type(stop_gradient(m))):
            return jnp.zeros_like(g)
        return g
        
    return tree_map(_apply_mask, grads, mask)