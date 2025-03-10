'''
Metropolis-Hastings sampling
'''

import numpy as np
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import optax
import time
from typing import Dict, Any, Optional, Callable, Tuple

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

def metropolis_hastings(ansatz, walkers, step_size, key):
    """Perform one step of Metropolis-Hastings sampling for quantum wavefunction.
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        walkers: Array of walker configurations with shape (n_walkers, n_electrons, 3)
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
    
    Returns:
        Tuple containing:
        - new_walkers: New walker configurations after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Compute initial wavefunction values
    psi_values = ansatz(walkers)
    
    # Generate proposals (one step)
    key, subkey = random.split(key)
    proposals = walkers + random.normal(subkey, walkers.shape) * step_size
    
    # Compute acceptance probabilities
    new_psi_values = ansatz(proposals)
    acceptance_prob = (jnp.abs(new_psi_values) / jnp.abs(psi_values))**2
    
    # Accept or reject
    key, subkey = random.split(key)
    accept_mask = random.uniform(subkey, shape=(walkers.shape[0],)) < acceptance_prob
    accept_count = jnp.sum(accept_mask)
    
    # Create new walkers without modifying input
    accept_mask_3d = accept_mask[:, jnp.newaxis, jnp.newaxis]
    new_walkers = jnp.where(accept_mask_3d, proposals, walkers)
    
    # Calculate acceptance rate
    acceptance_rate = float(accept_count) / walkers.shape[0]
    
    return new_walkers, acceptance_rate

def initialize_walkers(ansatz, n_walkers, initial_walkers=None, key=None):
    """Initialize walker configurations based on molecular structure.
    
    Args:
        ansatz: Wavefunction object with molecular information
        n_walkers: Number of parallel walkers
        initial_walkers: Optional initial positions
        key: PRNG key
        
    Returns:
        Array of initialized walker positions with shape (n_walkers, n_electrons, 3)
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))
        
    if initial_walkers is not None:
        return initial_walkers
        
    # Get molecular information needed for initialization
    atom_coords = ansatz.mol.atom_coords()
    atom_charges = ansatz.mol.atom_charges()
    n_electrons = ansatz.n_electrons
    n_up = ansatz.n_up
    
    # Initialize electron positions based on nuclear positions and spin counts
    key, subkey = random.split(key)
    walkers = init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, subkey, n_up=n_up)
    
    return walkers

def perform_mcmc_step(ansatz, walkers, step_size_up, step_size_down, key):
    """Perform one full MCMC step for both up and down electrons.
    
    Args:
        ansatz: Wavefunction object
        walkers: Current walker configurations
        step_size_up: Step size for up-spin electrons
        step_size_down: Step size for down-spin electrons
        key: PRNG key
    
    Returns:
        Tuple of (new_walkers, avg_acceptance, new_key)
    """
    # Move up-spin electrons with metropolis_hastings
    key, subkey = random.split(key)
    walkers, up_acceptance = metropolis_hastings(ansatz, walkers, step_size_up, subkey)
    
    # Move down-spin electrons with metropolis_hastings
    key, subkey = random.split(key)
    walkers, down_acceptance = metropolis_hastings(ansatz, walkers, step_size_down, subkey)
    
    # Average the acceptance rates from up and down moves
    avg_acceptance = (up_acceptance + down_acceptance) / 2
    
    return walkers, avg_acceptance, key

def burn_in(ansatz, walkers, n_steps, step_size_up, step_size_down, key, report_interval=100):
    """Perform burn-in steps for MCMC sampling.
    
    Args:
        ansatz: Wavefunction object
        walkers: Initial walker configurations
        n_steps: Number of burn-in steps
        step_size_up: Step size for up-spin electrons
        step_size_down: Step size for down-spin electrons
        key: PRNG key
        report_interval: How often to print progress
    
    Returns:
        Tuple of (equilibrated_walkers, acceptance_history, new_key)
    """
    acceptance_history = []
    
    if n_steps <= 0:
        return walkers, acceptance_history, key
        
    print(f"Starting burn-in with {n_steps} steps...")
    for step in range(n_steps):
        walkers, acceptance, key = perform_mcmc_step(
            ansatz, walkers, step_size_up, step_size_down, key)
        acceptance_history.append(acceptance)
        
        if step % report_interval == 0:
            print(f"Burn-in step {step}/{n_steps}")
    
    print("Burn-in complete.")
    return walkers, acceptance_history, key

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
    print(f"Step {step}/{total_steps}, Acceptance: {recent_acceptance:.4f}, Time/step: {recent_time*1000:.2f}ms")
    
    if energies:
        recent_energy = jnp.mean(jnp.concatenate(energies))
        print(f"  Current energy: {recent_energy:.6f}")

def create_optimizer(optimizer_type, learning_rate, opt_kwargs=None):
    """Create an optimizer based on specified type and parameters.
    
    Args:
        optimizer_type: Type of optimizer ("adam", "sgd", etc.)
        learning_rate: Learning rate for optimizer
        opt_kwargs: Additional optimizer parameters
    
    Returns:
        Configured optimizer
    """
    if opt_kwargs is None:
        opt_kwargs = {}
        
    if optimizer_type.lower() == "adam":
        return optax.adam(learning_rate=learning_rate, **opt_kwargs)
    elif optimizer_type.lower() == "sgd":
        return optax.sgd(learning_rate=learning_rate, **opt_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def sample(
    ansatz, 
    n_walkers: int = 100, 
    n_steps: int = 1000, 
    step_size: float = 1.0,
    thinning: int = 10,
    burn_in_steps: int = 1000,
    initial_walkers=None,
    key=None
) -> Dict[str, Any]:
    """Perform MCMC sampling for quantum wavefunction.
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        n_walkers: Number of parallel walkers
        n_steps: Number of MCMC steps for each walker
        step_size: Standard deviation of Gaussian proposal for MCMC
        thinning: Keep only every `thinning` steps to reduce autocorrelation
        burn_in_steps: Number of initial MCMC steps to discard (equilibration)
        initial_walkers: Optional initial positions, otherwise initialized near nuclei
        key: PRNG key
    
    Returns:
        Dictionary with sampling results and statistics
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))
    
    # Initialize walkers
    walkers = initialize_walkers(ansatz, n_walkers, initial_walkers, key)
    
    # Separate step sizes for up and down electrons can improve sampling
    step_size_up = step_size 
    step_size_down = step_size
    
    # Perform burn-in
    walkers, acceptance_history, key = burn_in(
        ansatz, walkers, burn_in_steps, step_size_up, step_size_down, key)
    
    if burn_in_steps > 0:
        print("Starting production sampling...")
    
    # Storage for collected samples
    collected_samples = []
    collected_energies = []
    step_times = []
    
    # Main sampling loop
    for step in range(n_steps):
        start_time = time.time()
        
        # Perform one MCMC step (moves both up and down electrons)
        walkers, acceptance, key = perform_mcmc_step(
            ansatz, walkers, step_size_up, step_size_down, key)
        acceptance_history.append(acceptance)
        
        # Store samples at thinning interval
        if step % thinning == 0:
            # Compute local energies
            energies = ansatz.local_energy(walkers)
            
            # Store samples and energies
            collected_samples.append(walkers)
            collected_energies.append(energies)
        
        step_time = time.time() - start_time
        step_times.append(step_time)
        
        # Print progress occasionally
        if step % 100 == 0 or step == n_steps - 1:
            report_progress(step, n_steps, acceptance_history, step_times, 
                           collected_energies if collected_energies else None)
    
    # Prepare and return results
    return prepare_sampling_results(
        collected_samples, collected_energies, acceptance_history, walkers, step_times)

def optimize(
    ansatz, 
    cost_fn=None,
    n_walkers: int = 100, 
    n_steps: int = 1000, 
    step_size: float = 1.0, 
    burn_in_steps: int = 1000,
    thinning: int = 10,
    n_samples: Optional[int] = None,
    initial_walkers=None,
    key=None,
    # Optimization parameters
    n_opt_steps: int = 100,
    learning_rate: float = 0.01,
    optimizer_type: str = "adam",
    opt_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Perform wavefunction optimization using MCMC sampling.
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        cost_fn: Cost function (defaults to average local energy if None)
        n_walkers: Number of parallel walkers
        n_steps: Number of MCMC steps for each walker in each opt iteration
        step_size: Standard deviation of Gaussian proposal for MCMC
        burn_in_steps: Number of initial MCMC steps to discard (equilibration)
        thinning: Keep only every `thinning` steps to reduce autocorrelation
        n_samples: If provided, collect this many uncorrelated samples
        initial_walkers: Optional initial positions, otherwise initialized near nuclei
        key: PRNG key
        n_opt_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer
        optimizer_type: Type of optimizer ("adam", "sgd", etc.)
        opt_kwargs: Additional optimizer parameters
    
    Returns:
        Dictionary with optimization results and statistics
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))
    
    if opt_kwargs is None:
        opt_kwargs = {}
        
    # Default to average energy as cost function if none provided
    if cost_fn is None:
        def energy_cost_fn(energies):
            return jnp.mean(energies)
        cost_fn = energy_cost_fn
    
    # Initialize walkers
    walkers = initialize_walkers(ansatz, n_walkers, initial_walkers, key)
    
    # Separate step sizes for up and down electrons can improve sampling
    step_size_up = step_size 
    step_size_down = step_size
    
    # Perform burn-in
    walkers, acceptance_history, key = burn_in(
        ansatz, walkers, burn_in_steps, step_size_up, step_size_down, key)
    
    print("Starting optimization...")
    
    # Initialize optimizer
    jastrow_params = ansatz.jastrow.params
    optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
    opt_state = optimizer.init(jastrow_params)
    
    # Track optimization progress
    opt_history = {
        "energy": [],
        "energy_std": [],
        "params": [],
        "gradients": [],
        "steps": []
    }
    
    # Define loss function that computes average energy
    def loss_fn(params):
        # Update ansatz with new parameters
        new_ansatz = ansatz.update_jastrow(params)
        
        # Compute energies for all walkers
        energies = new_ansatz.local_energy(walkers)
        
        # Compute cost (default: mean energy)
        cost = cost_fn(energies)
        
        # Also return energies for statistics
        return cost, energies
    
    # Optimization loop
    best_energy = float('inf')
    best_params = None
    
    print(f"Starting optimization with {n_opt_steps} steps...")
    for opt_step in range(n_opt_steps):
        start_time = time.time()
        
        # Compute loss and gradients
        (loss, energies), grads = value_and_grad(loss_fn, has_aux=True)(jastrow_params)
        
        # Compute energy statistics
        energy_mean = jnp.mean(energies).item()
        energy_std = jnp.std(energies).item() / jnp.sqrt(len(energies))
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        jastrow_params = optax.apply_updates(jastrow_params, updates)
        
        # Update ansatz with new parameters
        ansatz = ansatz.update_jastrow(jastrow_params)
        
        # Track best parameters
        if energy_mean < best_energy:
            best_energy = energy_mean
            best_params = jastrow_params
        
        # Store optimization history
        opt_history["energy"].append(energy_mean)
        opt_history["energy_std"].append(energy_std)
        opt_history["params"].append(jastrow_params)
        opt_history["gradients"].append(grads)
        opt_history["steps"].append(opt_step)
        
        # Print progress
        step_time = time.time() - start_time
        print(f"Opt step {opt_step}/{n_opt_steps}, Energy: {energy_mean:.6f} ± {energy_std:.6f}, Time: {step_time*1000:.2f}ms")
        
        # Resample configurations for next iteration (except for last step)
        if opt_step < n_opt_steps - 1:
            # Perform MCMC steps to get new samples
            acceptance_temp = []
            for mcmc_step in range(n_steps):
                walkers, acceptance, key = perform_mcmc_step(
                    ansatz, walkers, step_size_up, step_size_down, key)
                acceptance_temp.append(acceptance)
            
            # Add acceptances to history
            acceptance_history.extend(acceptance_temp)
    
    print(f"Optimization complete. Best energy: {best_energy:.6f}")
    
    
    # Combine optimization results with final sampling results
    results = {
        "optimization_history": opt_history,
        "best_params": best_params,
        "best_energy": best_energy
    }
    
    return results