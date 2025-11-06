"""Optimization algorithms for VMC parameter optimization.

This module contains functions for optimizing wavefunction parameters
using various optimization strategies and cost functions.
"""

import time
import numpy as np
import gc
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
from jax.tree_util import tree_map
import optax
import kfac_jax
from typing import Dict, Any, Optional

from .metropolis import metropolis_hastings, metropolis_hastings_importance_sampling
from .walker import initialize_walkers, Walker
from .sampling import burn_in, burn_in_with_importance
from .mcmc_utils import create_gradient_mask, create_optimizer


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
    params=None,
    frozen_params=None,
    move_type: str = "one"
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
    user_or_default_cost_fn = cost_fn
    if user_or_default_cost_fn is None:
        def energy_cost_fn(energies_for_cost):
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
            ansatz, walkers, burn_in_steps, step_size, key, params, move_type=move_type)
    
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
    def internal_loss_fn(current_params_for_loss, batch_data_for_loss):
        if optimizer_type.lower() == "kfac":
            actual_walkers_for_loss = batch_data_for_loss[0]
        else:
            actual_walkers_for_loss = batch_data_for_loss

        # Compute energies for all walkers with current parameters
        energies_val = ansatz.local_energy(actual_walkers_for_loss, current_params_for_loss)
        
        # clip energies around the mean energy to avoid numerical instability
        median_energy_val = jnp.median(energies_val)
        mean_energy_val = jnp.mean(energies_val)
        var_e_val = jnp.mean(jnp.abs(energies_val - mean_energy_val))
        clip_multiplier = 5
        clipped_energies = jnp.clip(energies_val, mean_energy_val - clip_multiplier * var_e_val, mean_energy_val + clip_multiplier * var_e_val)
        
        # Compute cost using the user-provided or default cost function
        cost = user_or_default_cost_fn(clipped_energies)

        if optimizer_type.lower() == "kfac":
            kfac_jax.register_normal_predictive_distribution(energies_val[:, None])
        
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
        opt_state = optimizer.init(params, subkey_init, (walkers, None))
            
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
                ansatz, walkers, step_size, subkey, params, move_type=move_type)

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
                    grads = tree_map(lambda g, m: jnp.where(m, g, jnp.zeros_like(g)), 
                                       grads, gradient_mask)

                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

            # Create materialized copies of parameters for history storage
            params_copy = tree_map(lambda x: np.array(jax.device_get(x)), params)
            params_history.append(params_copy)
            print("Optimization step parameters:", params_copy)

            losses.append(float(current_batch_mean_energy))
            acceptances.append(float(acceptance))
        
            step_time = time.time() - start_time
            start_time = time.time()
            print(f"Step: {opt_step}, Cost: {float(current_batch_cost):.6f}, "
                  f"Mean E (hist): {jnp.mean(jnp.asarray(losses[-500:])):.6f}, "
                  f"Batch Var E: {current_batch_energy_variance:.6f}, "
                  f"Acceptance: {float(acceptance):.3f}, "
                  f"Time: {step_time:.2f}s")
            gc.collect()
        
    opt_history["energies"] = jnp.asarray(losses)
    opt_history["params"] = params_history
    opt_history["acceptance"] = jnp.asarray(acceptances)
    opt_history["steps"] = jnp.arange(n_opt_steps)

    print(f"Optimization complete.")
    
    return opt_history


def optimize_ref_var(
    ansatz,
    cost_fn=None,
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
    move_type: str = "one",
    opt_kwargs: Optional[Dict[str, Any]] = None,
    params=None,
    frozen_params=None
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
            if optimizer_type.lower() == "kfac":
                actual_walkers_for_loss = batch_data_for_loss[0]
            else:
                actual_walkers_for_loss = batch_data_for_loss

            energies_val = ansatz.local_energy(actual_walkers_for_loss, current_params_for_loss)
            e_ref_val = jnp.mean(energies_val)
            e_std_val = jnp.std(energies_val)
            var_ref_val = jnp.sum((energies_val - e_ref_val)**2) / (energies_val.shape[0] - 1) if energies_val.shape[0] > 1 else 0.0
            
            if optimizer_type.lower() == "kfac":
                kfac_jax.register_normal_predictive_distribution(energies_val[:, None])

            return var_ref_val, (e_ref_val, e_std_val)
        user_or_default_cost_fn = ref_var_cost_fn

    # Initialize walkers using the reference determinant's info
    ref_det = ansatz.dets[0]
    walkers = initialize_walkers(ref_det, n_walkers, initial_walkers, key)

    # Burn-in walkers using the initial combined parameters
    walkers, acceptance_history, key, step_size = burn_in(
        ref_det, walkers, burn_in_steps, step_size, key, params=params, move_type=move_type)

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
                ref_det, walkers, step_size, subkey_mcmc, params, move_type=move_type)
            acceptances.append(current_acceptance_rate)

            # Create materialized copies of parameters for history storage
            def materialize_and_get(x):
                if isinstance(x, (jnp.ndarray, np.ndarray)):
                    return np.array(jax.device_get(x))
                return x
                
            params_copy = tree_map(materialize_and_get, params)
            
            params_history.append(params_copy)
            losses.append(float(current_batch_cost))
            energies.append(float(current_batch_ref_e))

            step_time_val = time.time() - start_time
            print(f"Step: {opt_step}, Var: {float(current_batch_cost):.6f}, E_mean: {float(current_batch_ref_e):.6f}+\-{float(current_batch_std_e):.6f}, "
                  f"Acceptance: {float(current_acceptance_rate):.3f}, Time: {step_time_val:.2f}s ")

            start_time = time.time()
        gc.collect()

    opt_history["cost"] = jnp.asarray(losses)
    opt_history["energies"] = jnp.asarray(energies)
    opt_history["params"] = params_history
    opt_history["acceptance"] = jnp.asarray(acceptances)
    opt_history["steps"] = jnp.arange(n_opt_steps)

    print(f"Optimization complete.")

    return opt_history
