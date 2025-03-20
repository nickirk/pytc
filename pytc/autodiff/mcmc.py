'''
Metropolis-Hastings sampling
'''

import numpy as np
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import jax
import optax
import time
from typing import Dict, Any, Optional, Callable, Tuple

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
    n_alpha = ansatz.n_alpha
    
    # Initialize electron positions based on nuclear positions and spin counts
    key, subkey = random.split(key)
    walkers = init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, subkey, n_alpha=n_alpha)
    
    return walkers

def perform_mcmc_step(ansatz, walkers, step_size, key):
    """Perform one full MCMC step for both up and down electrons.
    
    Args:
        ansatz: Wavefunction object
        walkers: Current walker configurations
        step_size: Step size, std dev of Gaussian proposal
        key: PRNG key
    
    Returns:
        Tuple of (new_walkers, avg_acceptance, new_key)
    """
    # Move up-spin electrons with metropolis_hastings
    key, subkey = random.split(key)
    walkers, alpha_acceptance = metropolis_hastings(ansatz, walkers, step_size, subkey)
    
    # Move down-spin electrons with metropolis_hastings
    key, subkey = random.split(key)
    walkers, beta_acceptance = metropolis_hastings(ansatz, walkers, step_size, subkey)
    
    # Average the acceptance rates from up and down moves
    avg_acceptance = (alpha_acceptance + beta_acceptance) / 2
    
    return walkers, avg_acceptance, key

def burn_in(ansatz, walkers, n_steps, step_size, key, report_interval=100):
    """Perform burn-in steps for MCMC sampling.
    
    Args:
        ansatz: Wavefunction object
        walkers: Initial walker configurations
        n_steps: Number of burn-in steps
        step_size: Step size for MCMC proposals, std dev of Gaussian
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
            ansatz, walkers, step_size, key)
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
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=learning_rate, **opt_kwargs))
    elif optimizer_type.lower() == "sgd":
        return optax.sgd(learning_rate=learning_rate, **opt_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

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

def metropolis_hastings_importance_sampling(ansatz, walkers, time_step, key):
    """Perform one step of Metropolis-Hastings with importance sampling (drift).
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
               Should also have a quantum_force method
        walkers: Array of walker configurations with shape (n_walkers, n_electrons, 3)
        time_step: Time step for the drift-diffusion process
        key: PRNG key
    
    Returns:
        Tuple containing:
        - new_walkers: New walker configurations after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Compute initial wavefunction values and quantum forces
    psi_values = ansatz(walkers)
    
    # Compute quantum force: F = 2∇ψ/ψ (gradient of log wavefunction)
    quantum_forces = ansatz.quantum_force(walkers)
    
    # Generate drift-diffusion proposals:
    # R' = R + D*F(R)*τ + √(2D*τ)*χ (D=0.5 in atomic units)
    drift_term = 0.5 * quantum_forces * time_step
    diffusion_coef = jnp.sqrt(time_step)
    
    key, subkey = random.split(key)
    random_term = diffusion_coef * random.normal(subkey, walkers.shape)
    
    # Combine drift and diffusion terms
    proposals = walkers + drift_term + random_term
    
    # Compute new wavefunction values and quantum forces at proposed positions
    new_psi_values = ansatz(proposals)
    new_quantum_forces = ansatz.quantum_force(proposals)
    
    # Modified acceptance probability for importance sampling
    # G(R→R') = exp(-(R'-R-D*F(R)*τ)²/(2*τ))
    forward_density = _compute_green_function(proposals, walkers, quantum_forces, time_step)
    backward_density = _compute_green_function(walkers, proposals, new_quantum_forces, time_step)
    
    # Compute acceptance probabilities with Green's function ratio
    acceptance_prob = (jnp.abs(new_psi_values) / jnp.abs(psi_values))**2 * (backward_density / forward_density)
    
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

def perform_mcmc_step_with_importance(ansatz, walkers, time_step, key):
    """Perform one full MCMC step with importance sampling.
    
    Args:
        ansatz: Wavefunction object
        walkers: Current walker configurations
        time_step: Time step for the drift-diffusion process
        key: PRNG key
    
    Returns:
        Tuple of (new_walkers, acceptance_rate, new_key)
    """
    # Move all electrons at once with importance sampling
    key, subkey = random.split(key)
    walkers, acceptance = metropolis_hastings_importance_sampling(ansatz, walkers, time_step, subkey)
    
    return walkers, acceptance, key

def burn_in_with_importance(ansatz, walkers, n_steps, time_step, key, report_interval=100):
    """Perform burn-in steps for MCMC sampling with importance sampling.
    
    Args:
        ansatz: Wavefunction object
        walkers: Initial walker configurations
        n_steps: Number of burn-in steps
        time_step: Time step for the drift-diffusion process
        key: PRNG key
        report_interval: How often to print progress
    
    Returns:
        Tuple of (equilibrated_walkers, acceptance_history, new_key)
    """
    acceptance_history = []
    
    if n_steps <= 0:
        return walkers, acceptance_history, key
        
    print(f"Starting burn-in with {n_steps} steps using importance sampling...")
    for step in range(n_steps):
        walkers, acceptance, key = perform_mcmc_step_with_importance(
            ansatz, walkers, time_step, key)
        acceptance_history.append(acceptance)
        
        if step % report_interval == 0:
            print(f"Burn-in step {step}/{n_steps}")
    
    print("Burn-in complete.")
    return walkers, acceptance_history, key

# Modified sample function to support importance sampling
def sample(
    ansatz, 
    n_walkers: int = 100, 
    n_steps: int = 1000, 
    step_size: float = 1.0,
    thinning: int = 10,
    burn_in_steps: int = 1000,
    initial_walkers=None,
    use_importance_sampling: bool = False,  # New parameter to toggle importance sampling
    key=None
) -> Dict[str, Any]:
    """Perform MCMC sampling for quantum wavefunction.
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        n_walkers: Number of parallel walkers
        n_steps: Number of MCMC steps for each walker
        step_size: Standard deviation of Gaussian proposal for regular MCMC
                  or time step for importance sampling (typically 0.01-0.05)
        thinning: Keep only every `thinning` steps to reduce autocorrelation
        burn_in_steps: Number of initial MCMC steps to discard (equilibration)
        initial_walkers: Optional initial positions, otherwise initialized near nuclei
        use_importance_sampling: Whether to use importance sampling with drift
        key: PRNG key
    
    Returns:
        Dictionary with sampling results and statistics
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))
    
    # Initialize walkers
    walkers = initialize_walkers(ansatz, n_walkers, initial_walkers, key)
    
    # Perform burn-in with appropriate method
    if use_importance_sampling:
        walkers, acceptance_history, key = burn_in_with_importance(
            ansatz, walkers, burn_in_steps, step_size, key)
    else:
        walkers, acceptance_history, key = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key)
    
    
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
        if use_importance_sampling:
            walkers, acceptance, key = perform_mcmc_step_with_importance(
                ansatz, walkers, step_size, key)
        else:
            walkers, acceptance, key = perform_mcmc_step(
                ansatz, walkers, step_size, key)
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
    use_importance_sampling: bool = True,
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
        use_importance_sampling: Whether to use importance sampling with drift
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
    
    if use_importance_sampling:
        walkers, acceptance_history, key = burn_in_with_importance(
            ansatz, walkers, burn_in_steps, step_size, key)
    else:
        walkers, acceptance_history, key = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key)
    
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
    
    # Define loss function that computes energy
    def loss_fn(params, walkers_batch):
        # Update ansatz with new parameters
        new_ansatz = ansatz.update_jastrow(params)
        
        # Compute energies for all walkers
        energies = new_ansatz.local_energy(walkers_batch)

        # clip energies to avoid numerical instability
        #energies = jnp.clip(energies, -50., 50.)
        
        # Compute cost (default: mean energy)
        cost = cost_fn(energies)
        
        # Also return energies for statistics
        return cost, energies
    
    # Vectorized gradient function
    value_and_grad_fn = jax.jit(value_and_grad(loss_fn, has_aux=True))
    
    # Optimization loop
    best_energy = float('inf')
    best_params = None
    step_times = []
    
    print(f"Starting optimization with {n_opt_steps} steps...")
    for opt_step in range(n_opt_steps):
        start_time = time.time()
        
        # Initialize numpy-based accumulator for gradients (mutable)
        accumulated_grads = None
        all_energies = []
        acceptance_temp = []
        
        # Perform n_steps MCMC steps, accumulating gradients
        # for mcmc_step in range(n_steps):
        if opt_step % n_steps == 0:
            # Update walker positions using MCMC
            if use_importance_sampling:
                walkers, acceptance, key = perform_mcmc_step_with_importance(
                    ansatz, walkers, step_size, key)
            else:
                walkers, acceptance, key = perform_mcmc_step(
                    ansatz, walkers, step_size, key)
            acceptance_temp.append(acceptance)
            
            # Compute loss and gradients for current walker configurations
        (loss, energies), grads = value_and_grad_fn(jastrow_params, walkers)
        all_energies.append(energies)
            
        # Accumulate gradients, creating the structure only once
        #if accumulated_grads is None:
        #    accumulated_grads = grads
        #else:
        #    accumulated_grads = jax.tree_util.tree_map(
        #        lambda acc, g: acc + g,
        #        accumulated_grads, grads)
        accumulated_grads = -grads
        
        
        # Flatten energy arrays for statistics
        all_energies = jnp.concatenate(all_energies)
        energy_mean = jnp.mean(all_energies).item()
        energy_std = jnp.std(all_energies).item() / jnp.sqrt(len(all_energies))
        
        # Update parameters using accumulated gradients
        updates, opt_state = optimizer.update(accumulated_grads, opt_state)
        print(f"Loss: {loss}, Accumulated Gradients: {accumulated_grads}, Updates: {updates}")
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
        opt_history["gradients"].append(accumulated_grads)
        opt_history["steps"].append(opt_step)
        
        # Add acceptances to history
        acceptance_history.extend(acceptance_temp)
        
        # Print progress
        step_time = time.time() - start_time
        step_times.append(step_time)
        hist_energies = jnp.array(opt_history["energy"])[-200:]
        print(f"Opt step {opt_step}/{n_opt_steps}, Inst Energy: {energy_mean:.6f} ± {energy_std:.6f}, ") 
        print(f"Mean Energy: {jnp.mean(hist_energies):.6f} ± {jnp.std(hist_energies)/jnp.sqrt(len(hist_energies))},")
        print(f"Params: {jastrow_params}, Time: {step_time*1000:.2f}ms")
    
    print(f"Optimization complete. Best energy: {best_energy:.6f}")
    
    # Combine optimization results with final sampling results
    results = {
        "optimization_history": opt_history,
        "best_params": best_params,
        "best_energy": best_energy
    }
    
    return results