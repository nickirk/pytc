"""Metropolis-Hastings sampling
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import kfac_jax
import time
from jax import random, value_and_grad
from jax.lax import stop_gradient
from typing import Dict, Any, Optional

from pytc.autodiff.mcmc_utils import (
    prepare_sampling_results, report_progress, create_optimizer,
    init_electron_configs, create_gradient_mask
)


def metropolis_hastings(ansatz, walkers, step_size, key, params):
    """Perform one step of Metropolis-Hastings sampling for quantum wavefunction.
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        walkers: Array of walker configurations with shape (n_walkers, n_electrons, 3)
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        params: contians jastrow_params and linear_coeffs
    
    Returns:
        Tuple containing:
        - new_walkers: New walker configurations after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Compute initial wavefunction values with parameters
    psi_values = ansatz(walkers, params)
    
    # Generate proposals (one step)
    key, subkey = random.split(key)
    proposals = walkers + random.normal(subkey, walkers.shape) * step_size
    
    # Compute acceptance probabilities with parameters
    new_psi_values = ansatz(proposals, params)
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

def burn_in(ansatz, walkers, n_steps, step_size, key, params, report_interval=100):
    """Perform burn-in steps for MCMC sampling.
    
    Args:
        ansatz: Wavefunction object
        walkers: Initial walker configurations
        n_steps: Number of burn-in steps
        step_size: Step size for MCMC proposals, std dev of Gaussian
        key: PRNG key
        params: Parameters for the ansatz, including jastrow and linear coefficients
        report_interval: How often to print progress
    
    Returns:
        Tuple of (equilibrated_walkers, acceptance_history, new_key)
    """
    acceptance_history = []
    
    if n_steps <= 0:
        return walkers, acceptance_history, key, step_size
        
    print(f"Starting burn-in with {n_steps} steps...")
    start_time = time.time()
    for step in range(n_steps):
        key, subkey = random.split(key)
        walkers, alpha_acceptance = metropolis_hastings(
            ansatz, walkers, step_size, subkey, params)
        
        key, subkey = random.split(key)
        walkers, beta_acceptance = metropolis_hastings(
            ansatz, walkers, step_size, subkey, params)
        
        acceptance_history.append((alpha_acceptance + beta_acceptance) / 2)
        
        if step % report_interval == 0:
            print(f"Burn-in step {step}/{n_steps}, acceptance: {acceptance_history[-1]:.3f}, time: {time.time() - start_time:.2f}s")
            step_size *= acceptance_history[-1]/0.5
            start_time = time.time()
    
    print("Burn-in complete.")
    return walkers, acceptance_history, key, step_size


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

def metropolis_hastings_importance_sampling(ansatz, walkers, time_step, key, params):
    """Perform one step of Metropolis-Hastings with importance sampling (drift).
    
    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
               Should also have a quantum_force method
        walkers: Array of walker configurations with shape (n_walkers, n_electrons, 3)
        time_step: Time step for the drift-diffusion process
        key: PRNG key
        params: Parameters for the ansatz, including jastrow and linear coefficients
    
    Returns:
        Tuple containing:
        - new_walkers: New walker configurations after one sampling step
        - acceptance_rate: Fraction of proposals that were accepted
    """
    # Compute initial wavefunction values and quantum forces with parameters
    psi_values = ansatz(walkers, params)
    
    # Compute quantum force: F = 2∇ψ/ψ (gradient of log wavefunction)
    quantum_forces = ansatz.quantum_force(walkers, params)
    
    # Generate drift-diffusion proposals:
    # R' = R + D*F(R)*τ + √(2D*τ)*χ (D=0.5 in atomic units)
    drift_term = 0.5 * quantum_forces * time_step
    diffusion_coef = jnp.sqrt(time_step)
    
    key, subkey = random.split(key)
    random_term = diffusion_coef * random.normal(subkey, walkers.shape)
    
    # Combine drift and diffusion terms
    proposals = walkers + drift_term + random_term
    
    # Compute new wavefunction values and quantum forces at proposed positions
    new_psi_values = ansatz(proposals, params)
    new_quantum_forces = ansatz.quantum_force(proposals, params)
    
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

def burn_in_with_importance(ansatz, walkers, n_steps, time_step, key, params, report_interval=100):
    """Perform burn-in steps for MCMC sampling with importance sampling.
    
    Args:
        ansatz: Wavefunction object
        walkers: Initial walker configurations
        n_steps: Number of burn-in steps
        time_step: Time step for the drift-diffusion process
        key: PRNG key
        params: Parameters for the ansatz, including jastrow and linear coefficients
        report_interval: How often to print progress
    
    Returns:
        Tuple of (equilibrated_walkers, acceptance_history, new_key)
    """
    acceptance_history = []
    if n_steps <= 0:
        return walkers, acceptance_history, key
        
    print(f"Starting burn-in with {n_steps} steps using importance sampling...")
    time_start = time.time()
    for step in range(n_steps):
        key, subkey = random.split(key)
        walkers, acceptance = metropolis_hastings_importance_sampling(
            ansatz, walkers, time_step, subkey, params)
        acceptance_history.append(acceptance)
        
        if step % report_interval == 0:
            print(f"Burn-in step {step}/{n_steps}, acceptance: {acceptance_history[-1]}, time: {time.time() - time_start:.2f}s")
            time_step *= acceptance_history[-1]/0.5
            time_start = time.time()
    
    print("Burn-in complete.")
    return walkers, acceptance_history, key, time_step

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
    params=None,
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
        params: Parameters for the ansatz, including jastrow and linear coefficients
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
        walkers, acceptance_history, key, step_size = burn_in_with_importance(
            ansatz, walkers, burn_in_steps, step_size, key, params)
    else:
        walkers, acceptance_history, key, step_size = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key, params)
    
    
    if burn_in_steps > 0:
        print("Starting production sampling...")
    
    # Storage for collected samples
    collected_samples = []
    collected_energies = []
    step_times = []
    
    # Main sampling loop
    start_time = time.time()
    for step in range(n_steps):
        
        key, subkey = random.split(key)
        if use_importance_sampling:
            walkers, acceptance = metropolis_hastings_importance_sampling(
                ansatz, walkers, step_size, subkey, params)
        else:
            walkers, acceptance = metropolis_hastings(
                ansatz, walkers, step_size, subkey, params)
            
        acceptance_history.append(acceptance)
        
        if step % thinning == 0:
            # Compute local energies with parameters
            energies = ansatz.local_energy(walkers, params)
            
            collected_samples.append(walkers)
            collected_energies.append(energies)
        
        
        # Print progress occasionally
        if step % 100 == 0 or step == n_steps - 1:
            step_time = time.time() - start_time
            step_times.append(step_time)
            report_progress(step, n_steps, acceptance_history, step_times, 
                           collected_energies if collected_energies else None)
            start_time = time.time()
    
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
    params=None, # Combined params: [jastrow_params, linear_coeffs]
    frozen_params=None  # Parameter freezing identifiers for Jastrow part
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
    # This outer cost_fn is what the user provides or the default jnp.mean
    user_or_default_cost_fn = cost_fn
    if user_or_default_cost_fn is None:
        def energy_cost_fn(energies_for_cost): # Renamed to avoid conflict
            return jnp.mean(energies_for_cost)
        user_or_default_cost_fn = energy_cost_fn
    
    # Initialize walkers
    walkers = initialize_walkers(ansatz, n_walkers, initial_walkers, key)
    
    # Perform burn-in with appropriate method
    if use_importance_sampling:
        walkers, acceptance_history, key, step_size = burn_in_with_importance(
            ansatz, walkers, burn_in_steps, step_size, key, params)
    else:
        walkers, acceptance_history, key, step_size = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key, params)
    
    print("Starting optimization...")
    
    # Initialize parameters if not provided
    if params is None:
        jastrow_params = ansatz.jastrow.init_params()
        linear_coeffs = jnp.ones(len(ansatz.dets))
        params = [jastrow_params, linear_coeffs]
    else:
        if not isinstance(params, (list, tuple)) or len(params) != 2:
             raise ValueError("`params` must be a list or tuple: [jastrow_params, linear_coeffs]")

    # --- Define internal loss function for optimization ---
    # This function will be passed to jax.value_and_grad
    def internal_loss_fn(current_params_for_loss, batch_data_for_loss):
        # When KFAC is used, batch_data_for_loss is (walkers_array, None)
        # Otherwise, for Optax etc., it's directly walkers_array
        if optimizer_type.lower() == "kfac":
            actual_walkers_for_loss = batch_data_for_loss[0]
        else:
            actual_walkers_for_loss = batch_data_for_loss

        # Compute energies for all walkers with current parameters
        energies_val = ansatz.local_energy(actual_walkers_for_loss, current_params_for_loss)
        
        # clip energies around the mean energy to avoid numerical instability
        median_energy_val = jnp.median(energies_val)
        mean_energy_val = jnp.mean(energies_val)
        var_e_val = jnp.mean(jnp.abs(energies_val - median_energy_val))
        clip_multiplier = 5
        clipped_energies = jnp.clip(energies_val, median_energy_val - clip_multiplier * var_e_val, median_energy_val + clip_multiplier * var_e_val)
        
        # Compute cost using the user-provided or default cost function
        cost = user_or_default_cost_fn(clipped_energies) # Use clipped energies for cost

        if optimizer_type.lower() == "kfac":
            # Register local energies as the "logits" for KFAC
            # KFAC will implicitly assume a squared error loss on these for Fisher approximation
            kfac_jax.register_normal_predictive_distribution(energies_val[:, None])
        
        # Return cost and auxiliary data (mean_energy of original energies, var_e of original energies)
        return cost, (mean_energy_val, var_e_val)

    # Initialize history lists and gradient mask before optimizer setup
    losses = []
    params_history = []
    acceptances = []
    opt_history = {}
    # Create mask for parameter freezing
    gradient_mask = create_gradient_mask(ansatz, params, frozen_params)

    # --- Initialize Optimizer for combined params ---
    if optimizer_type.lower() == "kfac":
        # KFAC specific setup
        opt_kwargs["value_and_grad_func"] = jax.value_and_grad(internal_loss_fn, argnums=0, has_aux=True)
        opt_kwargs["value_func_has_aux"] = True
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        key, subkey_init = random.split(key)
        
        # For KFAC, the mask does not work yet.
        opt_state = optimizer.init(params, subkey_init, (walkers, None)) # Use original params
        # If params were wrapped for init, ensure the main 'params' var reflects this structure
        # if frozen_params:
        #     params = params_for_init
            
    else:
        # Standard Optax or other optimizers
        value_and_grad_fn = jax.jit(value_and_grad(internal_loss_fn, argnums=0, has_aux=True))
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        opt_state = optimizer.init(params)
    

    print(f"Starting optimization with {n_opt_steps} steps...")

    start_time = time.time()
    for opt_step in range(n_opt_steps):
        
        # Perform MCMC step to update walkers
        key, subkey = random.split(key)
        if use_importance_sampling:
            walkers, acceptance = metropolis_hastings_importance_sampling(
                ansatz, walkers, step_size, subkey, params)
        else:
            walkers, acceptance = metropolis_hastings(
                ansatz, walkers, step_size, subkey, params)

        if opt_step % n_steps == 0:    
            # Update walkers using KFAC if applicable
            if optimizer_type.lower() == "kfac":
                key, subkey_step = random.split(key)
                # KFAC's step function computes gradients internally and updates params
                params, opt_state, stats = optimizer.step(
                    params, opt_state, subkey_step, batch=(walkers, None), global_step_int=opt_step
                )

                current_batch_cost = stats['loss']
                current_batch_mean_energy, current_batch_energy_variance = stats['aux']
            else:
                (cost_val, (mean_energy_val, var_e_val)), grads = value_and_grad_fn(params, walkers)
                current_batch_cost = cost_val
                current_batch_mean_energy = mean_energy_val
                current_batch_energy_variance = var_e_val

                if gradient_mask is not None:
                    # Zero out gradients for frozen parameters
                    grads = jax.tree_map(lambda g, m: jnp.zeros_like(g) if not m else g, 
                                       grads, gradient_mask)

                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

            # Create materialized copies of parameters for history storage
            params_copy = jax.tree_map(lambda x: jax.device_get(x), params)
            params_history.append(params_copy)

            losses.append(float(current_batch_mean_energy))  # Convert to Python float
            acceptances.append(float(acceptance))  # Convert to Python float
        
            step_time = time.time() - start_time
            start_time = time.time()
            print(f"Step: {opt_step}, Cost: {float(current_batch_cost):.6f}, "
                  f"Mean E (hist): {jnp.mean(jnp.asarray(losses[-100:])):.6f}, "
                  f"Batch Var E: {current_batch_energy_variance:.6f}, "
                  f"Acceptance: {acceptance:.3f}, "
                  f"Time: {step_time:.2f}s")
        
    opt_history["energies"] = jnp.asarray(losses)
    opt_history["params"] = params_history
    opt_history["acceptance"] = jnp.asarray(acceptances)
    opt_history["steps"] = jnp.arange(n_opt_steps)

    print(f"Optimization complete.")
    
    return opt_history

def optimize_ref_var(
    ansatz,
    cost_fn=None, # User can provide a custom variance-like cost function
    n_walkers: int = 100,
    n_steps: int = 1000,
    step_size: float = 1.0,
    burn_in_steps: int = 1000,
    initial_walkers=None,
    key=None,
    # Optimization parameters
    n_opt_steps: int = 100,
    learning_rate: float = 0.01,
    optimizer_type: str = "adam",
    opt_kwargs: Optional[Dict[str, Any]] = None,
    params=None, # Combined params: [jastrow_params, linear_coeffs]
    frozen_params=None  # Parameter freezing identifiers for Jastrow part
):
    """Perform reference variance optimization using MCMC sampling.

    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        cost_fn: Cost function (defaults to reference variance if None).
                 Should accept (params, walkers_batch) and return (cost, aux_data).
        n_walkers: Number of parallel walkers
        n_steps: Number of MCMC steps per optimization step (used for walker updates)
        step_size: Standard deviation of Gaussian proposal for MCMC
        burn_in_steps: Number of initial MCMC steps to discard (equilibration)
        initial_walkers: Optional initial positions, otherwise initialized near nuclei
        key: PRNG key
        n_opt_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer
        optimizer_type: Type of optimizer ("adam", "sgd", etc.)
        opt_kwargs: Additional optimizer parameters
        params: Initial combined parameters [jastrow_params, linear_coeffs].
        frozen_params: List of identifiers (int index or str name/type) for Jastrow factors to freeze.

    Returns:
        Dictionary with optimization results and statistics
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))

    if opt_kwargs is None:
        opt_kwargs = {}

    # Initialize parameters if not provided
    if params is None:
        jastrow_params = ansatz.jastrow.init_params()
        linear_coeffs = jnp.ones(len(ansatz.dets))
        params = [jastrow_params, linear_coeffs]
    else:
        if not isinstance(params, (list, tuple)) or len(params) != 2:
             raise ValueError("`params` must be a list or tuple: [jastrow_params, linear_coeffs]")

    # Default cost function (ref variance) if user does not provide one
    user_or_default_cost_fn = cost_fn
    if user_or_default_cost_fn is None:
        def ref_var_cost_fn(current_params_for_loss, batch_data_for_loss):
            # When KFAC is used, batch_data_for_loss is (walkers_array, None)
            # Otherwise, for Optax etc., it's directly walkers_array
            if optimizer_type.lower() == "kfac":
                actual_walkers_for_loss = batch_data_for_loss[0]
            else:
                actual_walkers_for_loss = batch_data_for_loss

            # This is the function that will be differentiated or used by KFAC
            energies_val = ansatz.local_energy(actual_walkers_for_loss, current_params_for_loss)
            e_ref_val = jnp.mean(energies_val)
            e_std_val = jnp.std(energies_val)
            var_ref_val = jnp.sum((energies_val - e_ref_val)**2) / (energies_val.shape[0] - 1) if energies_val.shape[0] > 1 else 0.0
            
            if optimizer_type.lower() == "kfac":
                # Register local energies as the "logits" for KFAC
                # The variance loss is a form of squared error on these energies
                kfac_jax.register_normal_predictive_distribution(energies_val[:, None])

            # Return variance as cost, and (mean_energy, std_dev_energy) as auxiliary
            return var_ref_val, (e_ref_val, e_std_val)
        user_or_default_cost_fn = ref_var_cost_fn
    # else, the user_or_default_cost_fn is the one provided by the user.
    # It must have the signature: (params, batch_data) -> (cost, aux_data)
    # and if KFAC is used, it must internally call a KFAC registration function.

    # Initialize walkers using the reference determinant's info
    ref_det = ansatz.dets[0]
    walkers = initialize_walkers(ref_det, n_walkers, initial_walkers, key)

    # Burn-in walkers using the initial combined parameters
    walkers, acceptance_history, key, step_size = burn_in(
        ref_det, walkers, burn_in_steps, step_size, key, params)

    print("Starting optimization...")
    params_history = []
    losses = []
    energies = []
    acceptances = []
    opt_history = {}
    # --- Initialize Optimizer for combined params ---
    if optimizer_type.lower() == "kfac":
        opt_kwargs["value_and_grad_func"] = jax.value_and_grad(user_or_default_cost_fn, argnums=0, has_aux=True)
        opt_kwargs["value_func_has_aux"] = True
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        key, subkey_init = random.split(key)
        
        opt_state = optimizer.init(params, subkey_init, (walkers, None))
    else:
        value_and_grad_fn = jax.jit(value_and_grad(user_or_default_cost_fn, argnums=0, has_aux=True))
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        
        # Apply gradient masking to initial parameters
        if frozen_params:
            params = create_gradient_mask(ansatz, params, frozen_params)
            
        opt_state = optimizer.init(params)

    print(f"Starting optimization with {n_opt_steps} steps...")

    start_time = time.time()
    for opt_step in range(n_opt_steps):

        current_batch_cost = None
        current_batch_ref_e = None
        current_batch_std_e = None

        if optimizer_type.lower() == "kfac":
            key, subkey_step = random.split(key)
            
            # KFAC's step function computes gradients internally and updates params
            params, opt_state, stats = optimizer.step(
                params, opt_state, subkey_step, batch=(walkers, None), global_step_int=opt_step
            )
            
            current_batch_cost = stats['loss']
            current_batch_ref_e, current_batch_std_e = stats['aux']
        else:  # Optax-style
            # For Optax, value_and_grad_fn is called with (params, walkers)
            # So, user_or_default_cost_fn will receive `walkers` directly as batch_data_for_loss
            (cost_val, (ref_e_val, std_e_val)), grads = value_and_grad_fn(params, walkers)
            current_batch_cost = cost_val
            current_batch_ref_e = ref_e_val
            current_batch_std_e = std_e_val
            
            
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)


        if opt_step % n_steps == 0:
            # Perform MCMC step to update walkers
            key, subkey_mcmc = random.split(key)
            current_acceptance_rate = 0.0
            walkers, current_acceptance_rate = metropolis_hastings(
                ref_det, walkers, step_size, subkey_mcmc, params)
            acceptances.append(current_acceptance_rate)

            # Create materialized copies of parameters for history storage
            # First materialize all parameters by copying them to ensure they don't get deleted
            def materialize_and_get(x):
                # Explicitly copy array and move to host
                if isinstance(x, (jnp.ndarray, np.ndarray)):
                    return np.array(jax.device_get(x))
                return x
                
            params_copy = jax.tree_map(materialize_and_get, params)
            
            params_history.append(params_copy)
            losses.append(float(current_batch_cost))
            energies.append(float(current_batch_ref_e))

            step_time_val = time.time() - start_time # Corrected variable name
            print(f"Step: {opt_step}, Var: {float(current_batch_cost):.6f}, E_mean: {float(current_batch_ref_e):.6f}+\-{float(current_batch_std_e):.6f}, "
                  f"Acceptance: {current_acceptance_rate:.3f}, Time: {step_time_val:.2f}s, "
                  f"atom 0: rc: {float(params[0][0]['rc'][0]):.6f}, X4: {float(params[0][0]['X4'][0]):.6f},")

            start_time = time.time()

    opt_history["cost"] = jnp.asarray(losses)
    opt_history["energies"] = jnp.asarray(energies)
    opt_history["params"] = params_history
    opt_history["acceptance"] = jnp.asarray(acceptances)
    opt_history["steps"] = jnp.arange(n_opt_steps)

    print(f"Optimization complete.")

    return opt_history
