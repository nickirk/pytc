"""Optimization algorithms for VMC parameter optimization.

This module contains functions for optimizing wavefunction parameters
using various optimization strategies and cost functions.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
from jax.tree_util import tree_map
import optax
import kfac_jax
from typing import Dict, Any, Optional

from .metropolis import metropolis_hastings, metropolis_hastings_importance_sampling, make_mcmc_step, make_mcmc_step_importance
from .walker import initialize_walkers, Walker
from .sampling import burn_in, burn_in_with_importance
from .mcmc_utils import create_gradient_mask, create_optimizer


def make_opt_update_step(loss_fn, optimizer):
    """Factory to create a JIT-compilable optimizer step for Optax optimizers.
    
    Args:
        loss_fn: Loss function with signature (params, walkers) -> (loss, aux_data)
                 where aux_data is a tuple of auxiliary outputs
        optimizer: Optax optimizer (e.g., optax.adam)
    
    Returns:
        A JIT-compiled function with signature:
            opt_step(params, walkers, opt_state, key) -> (params, opt_state, loss, aux_data)
    """
    # Create value_and_grad function
    loss_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    
    def opt_step(params, walkers, opt_state, key):
        """Single optimizer step - fully JIT-compatible.
        
        Args:
            params: Current parameters [jastrow_params, linear_coeffs]
            walkers: Walker dataclass with current MCMC configurations
            opt_state: Optimizer internal state
            key: PRNG key (for potential stochastic operations)
        
        Returns:
            params: Updated parameters
            opt_state: Updated optimizer state
            loss: Scalar loss value
            aux_data: Auxiliary data from loss function (e.g., energy, variance)
        """
        # Compute loss and gradients
        (loss, aux_data), grads = loss_and_grad(params, walkers)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, aux_data
    
    # JIT compile the step function
    return jax.jit(opt_step)


def make_training_step(mcmc_step, opt_update_step, n_opt_per_mcmc=1):
    """Factory to create unified training step combining optimization and MCMC.
    
    This creates a single JIT-compiled function that performs multiple optimization updates
    followed by one MCMC step, which is common in variance optimization.
    
    Args:
        mcmc_step: JIT-compiled MCMC step function from make_mcmc_step()
        opt_update_step: JIT-compiled optimizer step from make_opt_update_step()
        n_opt_per_mcmc: Number of optimization steps per MCMC update
    
    Returns:
        A JIT-compiled function with signature:
            training_step(walkers, params, opt_state, key) ->
                (walkers, params, opt_state, loss, aux_data, pmove)
    """
    def training_step(walkers, params, opt_state, key):
        """One full training iteration: optimization + MCMC.
        
        This function is fully JIT-compilable and contains no side effects.
        All values are returned as JAX arrays - materialization happens outside.
        
        Args:
            walkers: Walker dataclass with current MCMC configurations
            params: Current parameters [jastrow_params, linear_coeffs]
            opt_state: Optimizer internal state
            key: PRNG key for random number generation
        
        Returns:
            walkers: Updated walker configurations
            params: Updated parameters
            opt_state: Updated optimizer state
            loss: Scalar loss value from last optimization step
            aux_data: Auxiliary data from loss function
            pmove: Acceptance probability from MCMC step
        """
        # Optimization loop using jax.lax.scan for JIT compatibility
        def opt_scan_fn(carry, _):
            params_carry, opt_state_carry, key_carry = carry
            key_carry, subkey = random.split(key_carry)
            params_carry, opt_state_carry, loss_carry, aux_data_carry = opt_update_step(
                params_carry, walkers, opt_state_carry, subkey
            )
            return (params_carry, opt_state_carry, key_carry), (loss_carry, aux_data_carry)
        
        # Run optimization loop
        (params, opt_state, key), (losses, aux_data_list) = jax.lax.scan(
            opt_scan_fn,
            (params, opt_state, key),
            None,
            length=n_opt_per_mcmc
        )
        
        # Use the last loss and aux_data from the optimization loop
        loss = losses[-1] if n_opt_per_mcmc > 1 else losses
        aux_data = tree_map(lambda x: x[-1] if n_opt_per_mcmc > 1 else x, aux_data_list)
        
        # Single MCMC step
        key, subkey = random.split(key)
        walkers, pmove = mcmc_step(walkers, subkey, params)
        
        return walkers, params, opt_state, loss, aux_data, pmove
    
    return training_step


def make_kfac_training_step(mcmc_step, optimizer, n_opt_per_mcmc=1):
    """Factory to create unified training step for KFAC optimizer.
    
    KFAC optimizer handles gradient computation internally via its step() method,
    so this is slightly different from the standard Optax version.
    
    Args:
        mcmc_step: JIT-compiled MCMC step function from make_mcmc_step()
        optimizer: KFAC optimizer instance
        n_opt_per_mcmc: Number of optimization steps per MCMC update
    
    Returns:
        A function (not JIT-compiled yet) with signature:
            training_step(walkers, params, opt_state, key, global_step) ->
                (walkers, params, opt_state, loss, aux_data, pmove)
    """
    def training_step(walkers, params, opt_state, key, global_step):
        """One full training iteration: optimization + MCMC.
        
        Note: This is NOT JIT-compiled because KFAC handles JIT internally.
        
        Args:
            walkers: Walker dataclass with current MCMC configurations
            params: Current parameters [jastrow_params, linear_coeffs]
            opt_state: KFAC optimizer internal state
            key: PRNG key for random number generation
            global_step: Current optimization step (required by KFAC)
        
        Returns:
            walkers: Updated walker configurations
            params: Updated parameters
            opt_state: Updated optimizer state
            loss: Scalar loss value from last optimization step
            aux_data: Auxiliary data from loss function
            pmove: Acceptance probability from MCMC step
        """
        # Optimization loop
        losses_list = []
        aux_data_list = []
        for _ in range(n_opt_per_mcmc):
            key, subkey = random.split(key)
            params, opt_state, stats = optimizer.step(
                params=params,
                state=opt_state,
                rng=subkey,
                batch=(walkers, None),
                global_step_int=global_step
            )
            losses_list.append(stats['loss'])
            aux_data_list.append(stats['aux'])
        
        # Use the last loss and aux_data from the optimization loop
        loss = losses_list[-1]
        aux_data = aux_data_list[-1]
        
        # Single MCMC step
        key, subkey = random.split(key)
        walkers, pmove = mcmc_step(walkers, subkey, params)
        
        return walkers, params, opt_state, loss, aux_data, pmove
    
    # KFAC handles JIT internally, so we don't JIT compile here
    return training_step


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
            else:
                (cost_val, (mean_energy_val, var_e_val)), grads = value_and_grad_fn(params, walkers)
                
                current_batch_cost = float(jax.device_get(cost_val))
                current_batch_mean_energy = float(jax.device_get(mean_energy_val))
                current_batch_energy_variance = float(jax.device_get(var_e_val))

                if gradient_mask is not None:
                    # Zero out gradients for frozen parameters
                    grads = tree_map(lambda g, m: jnp.where(m, g, jnp.zeros_like(g)), 
                                       grads, gradient_mask)

                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                
                # Materialize params to prevent accumulation across iterations
                params = tree_map(lambda x: jax.device_get(x) if isinstance(x, jnp.ndarray) else x, params)

            # Materialize acceptance rate
            acceptance_float = float(jax.device_get(acceptance))

            # Store a copy of the already-materialized params
            params_copy = tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, params)
            params_history.append(params_copy)
            print("Optimization step parameters:", params_copy)

            losses.append(current_batch_mean_energy)
            acceptances.append(acceptance_float)
        
            step_time = time.time() - start_time
            start_time = time.time()
            print(f"Step: {opt_step}, Cost: {current_batch_cost:.6f}, "
                  f"Mean E (hist): {jnp.mean(jnp.asarray(losses[-500:])):.6f}, "
                  f"Batch Var E: {current_batch_energy_variance:.6f}, "
                  f"Acceptance: {acceptance_float:.3f}, "
                  f"Time: {step_time:.2f}s")
        
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
    n_steps: int = 20,  # MCMC steps per optimization update
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
):
    """Perform reference variance optimization using MCMC sampling.
    

    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        cost_fn: Cost function (defaults to reference variance if None).
                 Should accept (params, walkers) and return (cost, aux_data).
        n_walkers: Number of parallel walkers
        n_steps: Number of MCMC steps per optimization update
        step_size: Standard deviation of Gaussian proposal for MCMC
        burn_in_steps: Number of initial MCMC steps to discard (equilibration)
        initial_walkers: Optional initial positions, otherwise initialized near nuclei
        key: PRNG key
        n_opt_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer
        optimizer_type: Type of optimizer ("adam", "sgd", "kfac", etc.)
        move_type: "one" or "all" for MCMC electron moves
        opt_kwargs: Additional optimizer parameters
        params: Initial combined parameters [jastrow_params, linear_coeffs].
        frozen_params: List of identifiers for Jastrow factors to freeze (not yet supported with new structure).

    Returns:
        Dictionary with optimization results and statistics
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))

    if opt_kwargs is None:
        opt_kwargs = {}

    if params is None:
        jastrow_params = ansatz.jastrow.init_params()
        linear_coeffs = jnp.ones(len(ansatz.dets))
        params = [jastrow_params, linear_coeffs]
    else:
        if not isinstance(params, (list, tuple)) or len(params) != 2:
             raise ValueError("`params` must be a list or tuple: [jastrow_params, linear_coeffs]")

    # Initialize walkers using the reference determinant's info
    ref_det = ansatz.dets[0]
    walkers = initialize_walkers(ref_det, n_walkers, initial_walkers, key)

    # Burn-in walkers using the initial combined parameters
    print("Performing burn-in...")
    walkers, acceptance_history, key, step_size = burn_in(
        ref_det, walkers, burn_in_steps, step_size, key, params=params, move_type=move_type)
    print(f"Burn-in complete. Final step size: {step_size:.4f}")


    # Create loss function (default to reference variance)
    if cost_fn is None:
        def loss_fn(params_inner, batch_data):
            """Default reference variance loss function."""
            # KFAC passes (walkers, None) as batch; Optax just passes walkers
            if isinstance(batch_data, tuple):
                walkers_inner = batch_data[0]
            else:
                walkers_inner = batch_data
                
            energies = ansatz.local_energy(walkers_inner, params_inner)
            e_mean = jnp.mean(energies)
            e_std = jnp.std(energies)
            # Reference variance: sum of squared deviations / (n-1)
            variance = jnp.sum((energies - e_mean)**2) / (energies.shape[0] - 1) if energies.shape[0] > 1 else 0.0
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac":
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return variance, (e_mean, e_std)
    else:
        loss_fn = cost_fn

    # Create MCMC step function
    mcmc_step = make_mcmc_step(ref_det, step_size, move_type)

    # Create optimizer and training step
    if optimizer_type.lower() == "kfac":
        # KFAC setup
        opt_kwargs["value_and_grad_func"] = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        opt_kwargs["value_func_has_aux"] = True
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        
        key, subkey = random.split(key)
        opt_state = optimizer.init(params, subkey, (walkers, None))
        
        # Create KFAC training step
        training_step = make_kfac_training_step(mcmc_step, optimizer, n_steps)
    else:
        # Optax setup
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        opt_state = optimizer.init(params)
        
        # Create Optax training step
        opt_update_step = make_opt_update_step(loss_fn, optimizer)
        training_step = make_training_step(mcmc_step, opt_update_step, n_steps)

    # ========== MAIN LOOP (Uses JIT-compiled step) ==========
    print(f"Starting optimization with {n_opt_steps} steps...")
    
    losses = []
    energies = []
    stds = []
    acceptances = []
    params_history = []
    
    start_time = time.time()
    
    for opt_step in range(n_opt_steps):
        key, subkey = random.split(key)
        
        if optimizer_type.lower() == "kfac":
            walkers, params, opt_state, loss, aux_data, pmove = training_step(
                walkers, params, opt_state, subkey, opt_step
            )
        else:
            walkers, params, opt_state, loss, aux_data, pmove = training_step(
                walkers, params, opt_state, subkey
            )
        
        variance_val = float(jax.device_get(loss))
        energy_val, std_val = jax.device_get(aux_data)
        energy_val = float(energy_val)
        std_val = float(std_val)
        pmove_val = float(jax.device_get(pmove))

        
        log_frequency = max(1, n_opt_steps // 100)  # Log ~100 times
        if opt_step % log_frequency == 0 or opt_step == n_opt_steps - 1:
            # Store history
            losses.append(variance_val)
            energies.append(energy_val)
            stds.append(std_val)
            acceptances.append(pmove_val)
            
            # Store params (materialize to numpy)
            params_copy = tree_map(
                lambda x: np.array(jax.device_get(x)) if isinstance(x, jnp.ndarray) else x,
                params
            )
            params_history.append(params_copy)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Step {opt_step:5d} | Var: {variance_val:.6f} | "
                  f"E: {energy_val:.6f}±{std_val:.6f} | "
                  f"Accept: {pmove_val:.3f} | Time: {elapsed:.2f}s")
            start_time = time.time()
    
    print("Optimization complete!")
    
    
    return {
        "cost": np.array(losses),
        "energies": np.array(energies),
        "stds": np.array(stds),
        "acceptance": np.array(acceptances),
        "params": params_history
    }
