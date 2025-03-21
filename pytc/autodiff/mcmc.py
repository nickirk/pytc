'''
Metropolis-Hastings sampling
'''
import gc
import numpy as np
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import jax
import optax
import time
from typing import Dict, Any, Optional, Callable, Tuple

from pytc.autodiff.mcmc_utils import (
    prepare_sampling_results, report_progress, create_optimizer,
    init_electron_configs
)


def metropolis_hastings(ansatz, walkers, step_size, key, jastrow_params, linear_coeffs):
    """Perform one step of Metropolis-Hastings sampling for quantum wavefunction.
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        walkers: Array of walker configurations with shape (n_walkers, n_electrons, 3)
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        jastrow_params: Jastrow parameters
        linear_coeffs: Linear coefficients
    
    Returns:
        Tuple containing:
        - new_walkers: New walker configurations after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Compute initial wavefunction values with parameters
    psi_values = ansatz(walkers, jastrow_params, linear_coeffs)
    
    # Generate proposals (one step)
    key, subkey = random.split(key)
    proposals = walkers + random.normal(subkey, walkers.shape) * step_size
    
    # Compute acceptance probabilities with parameters
    new_psi_values = ansatz(proposals, jastrow_params, linear_coeffs)
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

def burn_in(ansatz, walkers, n_steps, step_size, key, jastrow_params, linear_coeffs, report_interval=100):
    """Perform burn-in steps for MCMC sampling.
    
    Args:
        ansatz: Wavefunction object
        walkers: Initial walker configurations
        n_steps: Number of burn-in steps
        step_size: Step size for MCMC proposals, std dev of Gaussian
        key: PRNG key
        jastrow_params: Jastrow parameters
        linear_coeffs: Linear coefficients
        report_interval: How often to print progress
    
    Returns:
        Tuple of (equilibrated_walkers, acceptance_history, new_key)
    """
    acceptance_history = []
    
    if n_steps <= 0:
        return walkers, acceptance_history, key
        
    print(f"Starting burn-in with {n_steps} steps...")
    for step in range(n_steps):
        key, subkey = random.split(key)
        walkers, alpha_acceptance = metropolis_hastings(
            ansatz, walkers, step_size, subkey, jastrow_params, linear_coeffs)
        
        key, subkey = random.split(key)
        walkers, beta_acceptance = metropolis_hastings(
            ansatz, walkers, step_size, subkey, jastrow_params, linear_coeffs)
        
        acceptance_history.append((alpha_acceptance + beta_acceptance) / 2)
        
        if step % report_interval == 0:
            print(f"Burn-in step {step}/{n_steps}")
    
    print("Burn-in complete.")
    return walkers, acceptance_history, key


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

def metropolis_hastings_importance_sampling(ansatz, walkers, time_step, key, jastrow_params, linear_coeffs):
    """Perform one step of Metropolis-Hastings with importance sampling (drift).
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
               Should also have a quantum_force method
        walkers: Array of walker configurations with shape (n_walkers, n_electrons, 3)
        time_step: Time step for the drift-diffusion process
        key: PRNG key
        jastrow_params: Jastrow parameters
        linear_coeffs: Linear coefficients
    
    Returns:
        Tuple containing:
        - new_walkers: New walker configurations after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Compute initial wavefunction values and quantum forces with parameters
    psi_values = ansatz(walkers, jastrow_params, linear_coeffs)
    
    # Compute quantum force: F = 2∇ψ/ψ (gradient of log wavefunction)
    quantum_forces = ansatz.quantum_force(walkers, jastrow_params, linear_coeffs)
    
    # Generate drift-diffusion proposals:
    # R' = R + D*F(R)*τ + √(2D*τ)*χ (D=0.5 in atomic units)
    drift_term = 0.5 * quantum_forces * time_step
    diffusion_coef = jnp.sqrt(time_step)
    
    key, subkey = random.split(key)
    random_term = diffusion_coef * random.normal(subkey, walkers.shape)
    
    # Combine drift and diffusion terms
    proposals = walkers + drift_term + random_term
    
    # Compute new wavefunction values and quantum forces at proposed positions
    new_psi_values = ansatz(proposals, jastrow_params, linear_coeffs)
    new_quantum_forces = ansatz.quantum_force(proposals, jastrow_params, linear_coeffs)
    
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

def burn_in_with_importance(ansatz, walkers, n_steps, time_step, key, jastrow_params, linear_coeffs, report_interval=100):
    """Perform burn-in steps for MCMC sampling with importance sampling.
    
    Args:
        ansatz: Wavefunction object
        walkers: Initial walker configurations
        n_steps: Number of burn-in steps
        time_step: Time step for the drift-diffusion process
        key: PRNG key
        jastrow_params: Jastrow parameters
        linear_coeffs: Linear coefficients
        report_interval: How often to print progress
    
    Returns:
        Tuple of (equilibrated_walkers, acceptance_history, new_key)
    """
    acceptance_history = []
    
    if n_steps <= 0:
        return walkers, acceptance_history, key
        
    print(f"Starting burn-in with {n_steps} steps using importance sampling...")
    for step in range(n_steps):
        key, subkey = random.split(key)
        walkers, acceptance = metropolis_hastings_importance_sampling(
            ansatz, walkers, time_step, subkey, jastrow_params, linear_coeffs)
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
    jastrow_params=None,  # New parameter
    linear_coeffs=None,  # New parameter
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
        jastrow_params: Jastrow parameters
        linear_coeffs: Linear coefficients
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
            ansatz, walkers, burn_in_steps, step_size, key, jastrow_params, linear_coeffs)
    else:
        walkers, acceptance_history, key = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key, jastrow_params, linear_coeffs)
    
    
    if burn_in_steps > 0:
        print("Starting production sampling...")
    
    # Storage for collected samples
    collected_samples = []
    collected_energies = []
    step_times = []
    
    # Main sampling loop
    for step in range(n_steps):
        start_time = time.time()
        
        # Call sampling functions directly instead of through wrapper functions
        key, subkey = random.split(key)
        if use_importance_sampling:
            walkers, acceptance = metropolis_hastings_importance_sampling(
                ansatz, walkers, step_size, subkey, jastrow_params, linear_coeffs)
        else:
            # Do both up and down spin moves
            walkers, alpha_acceptance = metropolis_hastings(
                ansatz, walkers, step_size, subkey, jastrow_params, linear_coeffs)
            key, subkey = random.split(key)
            walkers, beta_acceptance = metropolis_hastings(
                ansatz, walkers, step_size, subkey, jastrow_params, linear_coeffs)
            acceptance = (alpha_acceptance + beta_acceptance) / 2
            
        acceptance_history.append(acceptance)
        
        # Store samples at thinning interval
        if step % thinning == 0:
            # Compute local energies with parameters
            energies = ansatz.local_energy(walkers, jastrow_params, linear_coeffs)
            
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
    initial_walkers=None,
    key=None,
    # Optimization parameters
    n_opt_steps: int = 100,
    learning_rate: float = 0.01,
    optimizer_type: str = "adam",
    opt_kwargs: Optional[Dict[str, Any]] = None,
    jastrow_params=None,  # Added parameter
    linear_coeffs=None    # Added parameter
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
        jastrow_params: Initial Jastrow parameters
        linear_coeffs: Initial linear coefficients
    
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
            ansatz, walkers, burn_in_steps, step_size, key, jastrow_params, linear_coeffs)
    else:
        walkers, acceptance_history, key = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key, jastrow_params, linear_coeffs)
    
    print("Starting optimization...")
    
    # Initialize optimizer with provided or default parameters
    if jastrow_params is None:
        jastrow_params = ansatz.jastrow.params
    
    if linear_coeffs is None:
        linear_coeffs = jnp.ones(len(ansatz.dets))  # Default to equal weights
        
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
    
    # Define loss function that computes energy with explicit parameters
    def loss_fn(params, walkers_batch):
        # Compute energies for all walkers with current parameters
        energies = ansatz.local_energy(walkers_batch, params, linear_coeffs)
        
        # Compute cost (default: mean energy)
        cost = cost_fn(energies)
        
        return cost, energies
    
    # Vectorized gradient function
    value_and_grad_fn = jax.jit(value_and_grad(loss_fn, has_aux=True))
    
    # Optimization loop
    best_energy = float('inf')
    best_params = None
    step_times = []
    accumulated_grads = []
    losses = []
    
    print(f"Starting optimization with {n_opt_steps} steps...")
    for opt_step in range(n_opt_steps):
        start_time = time.time()
        
        # Initialize numpy-based accumulator for gradients (mutable)
        acceptance_temp = []
        
        # Perform n_steps MCMC steps, accumulating gradients
        # for mcmc_step in range(n_steps):
        # Update walker positions using MCMC directly
        key, subkey = random.split(key)
        if use_importance_sampling:
            walkers, acceptance = metropolis_hastings_importance_sampling(
                ansatz, walkers, step_size, subkey, jastrow_params, linear_coeffs)
        else:
            walkers, alpha_acceptance = metropolis_hastings(
                ansatz, walkers, step_size, subkey, jastrow_params, linear_coeffs)
            key, subkey = random.split(key)
            walkers, beta_acceptance = metropolis_hastings(
                ansatz, walkers, step_size, subkey, jastrow_params, linear_coeffs)
            acceptance = (alpha_acceptance + beta_acceptance) / 2
        acceptance_temp.append(acceptance)
            
        if opt_step % n_steps == 0:
            # Compute loss and gradients for current walker configurations
            (loss, energies), grads = value_and_grad_fn(jastrow_params, walkers)
            losses.append(loss) 
            accumulated_grads.append(grads)
        
        
            # Update parameters using accumulated gradients
            updates, opt_state = optimizer.update(grads, opt_state)
            jastrow_params = optax.apply_updates(jastrow_params, updates)
        
            # Store optimization history
            opt_history["energy"].append(loss)
            opt_history["params"].append(jastrow_params)
            opt_history["gradients"].append(accumulated_grads)
            opt_history["steps"].append(opt_step)
            # Convert arrays to scalars for printing
            loss_val = float(loss)
            if len(accumulated_grads) > 0:
                grad_mean = float(jnp.mean(jnp.asarray(accumulated_grads)))
        
            # Print progress with proper scalar conversions
            step_time = time.time() - start_time
            step_times.append(step_time)

            print(f"Step: {opt_step}, Loss: {loss_val:.6f}, "
                  f"Mean loss: {jnp.mean(jnp.asarray(losses[-500:])):.6f}, "
                  f"Ave Gradients: {grad_mean:.6f}, "
                  f"Params: {jastrow_params[0]:.6f}, "
                  f"Time: {step_time*1000:.2f}ms")
        
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