"""Optimization algorithms for VMC parameter optimization.

This module contains functions for optimizing wavefunction parameters
using various optimization strategies and cost functions.

Design Patterns
===============

The optimization module supports two complementary training patterns:

1. **Energy Minimization** (optimize function):
   - Pattern: Multiple MCMC steps → Single optimization step
   - Parameter: n_mcmc_per_opt (default: n_steps in optimize)
   - Rationale: Decorrelate walkers before computing gradients
   - Use case: Minimizing ground state energy
   - Example: n_mcmc_per_opt=20 means 20 MCMC steps, then 1 parameter update

2. **Variance Minimization** (optimize_ref_var function):
   - Pattern: Multiple optimization steps → Single MCMC step
   - Parameter: n_opt_per_mcmc (default: n_steps in optimize_ref_var)
   - Rationale: Multiple gradient steps on same walker configuration
   - Use case: Reducing variance for fixed reference determinant
   - Example: n_opt_per_mcmc=20 means 20 parameter updates, then 1 MCMC step

The factory functions make_training_step() and make_kfac_training_step() support
both patterns through their n_mcmc_per_opt and n_opt_per_mcmc parameters.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as spla
from jax import random, value_and_grad
from jax.tree_util import tree_map
import optax
import kfac_jax
from typing import Dict, Any, Optional


from .mcmc_utils import create_gradient_mask, create_optimizer
from .loss import make_variance_loss


def make_opt_update_step(loss_fn, optimizer):
    """Factory to create a JIT-compilable optimizer step for Optax optimizers.
    
    Args:
        loss_fn: Loss function with signature (params, walkers) -> (loss, aux_data)
                 where aux_data is a tuple of auxiliary outputs
        optimizer: Optax optimizer (e.g., optax.adam)
    
    Returns:
        A JIT-compiled function with signature:
            opt_step(ansatz, params, walkers, opt_state, key) -> (params, opt_state, loss, aux_data)
    """
    # Create value_and_grad function
    loss_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    
    def opt_step(ansatz, params, walkers, opt_state, key):
        """Single optimizer step - fully JIT-compatible.
        
        Args:
            ansatz: Wavefunction object
            params: Parameters to optimize
            walkers: Walker dataclass with current MCMC configurations
            opt_state: Optimizer internal state
            key: PRNG key (for potential stochastic operations)
        
        Returns:
            params: Updated parameters
            opt_state: Updated optimizer state
            loss: Loss value
            aux_data: Auxiliary data from loss function (e.g., energy, variance)
        """
        # Compute loss and gradients
        # Note: loss_fn expects (params, walkers), ansatz is baked in or handled via wrapper
        (loss, aux_data), grads = loss_and_grad(params, walkers)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, opt_state, loss, aux_data
    
    # JIT compile the step function
    return jax.jit(opt_step)


def make_training_step(mcmc_step, opt_update_step, n_mcmc_per_opt=1, n_opt_per_mcmc=1):
    """Factory to create unified training step combining optimization and MCMC.
    
    This creates a single JIT-compiled function that can perform either:
    1. Multiple MCMC steps → single optimization (for energy minimization)
    2. Multiple optimization steps → single MCMC step (for variance minimization)
    3. Any combination of the above
    
    Args:
        mcmc_step: JIT-compiled MCMC step function from make_mcmc_step()
                   Signature: (ansatz, walkers, key, params) -> (walkers, pmove)
        opt_update_step: JIT-compiled optimizer step from make_opt_update_step()
                         Signature: (ansatz, params, walkers, opt_state, key) -> (params, opt_state, loss, aux_data)
        n_mcmc_per_opt: Number of MCMC steps before each optimization update (default: 1)
        n_opt_per_mcmc: Number of optimization steps per MCMC update (default: 1)
                        Note: If n_mcmc_per_opt > 1, this should typically be 1.
    
    Returns:
        A JIT-compiled function with signature:
            training_step(ansatz, walkers, params, opt_state, key) ->
                (walkers, params, opt_state, loss, aux_data, pmove)
    
    Design patterns:
        - Energy minimization: n_mcmc_per_opt=10-100, n_opt_per_mcmc=1
          (decorrelate walkers before each parameter update)
        - Variance minimization: n_mcmc_per_opt=1, n_opt_per_mcmc=5-20
          (multiple gradient steps on same walker configuration)
    """
    def training_step(ansatz, walkers, params, opt_state, key):
        """One full training iteration: MCMC + optimization.
        
        This function is fully JIT-compilable and contains no side effects.
        All values are returned as JAX arrays - materialization happens outside.
        
        Args:
            ansatz: Wavefunction object
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
            pmove: Acceptance probability from last MCMC step
        """
        # Pattern 1: Multiple MCMC steps before optimization (energy minimization)
        if n_mcmc_per_opt > 1:
            def mcmc_scan_fn(carry, _):
                walkers_carry, key_carry, params_carry = carry
                key_carry, subkey = random.split(key_carry)
                walkers_carry, pmove_carry = mcmc_step(ansatz, walkers_carry, subkey, params_carry)
                return (walkers_carry, key_carry, params_carry), pmove_carry
            
            # Run MCMC loop
            (walkers, key, _), pmoves = jax.lax.scan(
                mcmc_scan_fn,
                (walkers, key, params),
                None,
                length=n_mcmc_per_opt
            )
            pmove = pmoves[-1]  # Use last acceptance rate
            
            # Single optimization step after MCMC
            key, subkey = random.split(key)
            params, opt_state, loss, aux_data = opt_update_step(
                ansatz, params, walkers, opt_state, subkey
            )
        
        # Pattern 2: Multiple optimization steps per MCMC (variance minimization)
        elif n_opt_per_mcmc > 1:
            def opt_scan_fn(carry, _):
                params_carry, opt_state_carry, key_carry = carry
                key_carry, subkey = random.split(key_carry)
                params_carry, opt_state_carry, loss_carry, aux_data_carry = opt_update_step(
                    ansatz, params_carry, walkers, opt_state_carry, subkey
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
            loss = losses[-1]
            aux_data = tree_map(lambda x: x[-1], aux_data_list)
            
            # Single MCMC step after optimization
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
        
        # Pattern 3: Balanced (1 MCMC, 1 opt) - default simple case
        else:
            # Single MCMC step
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
            
            # Single optimization step
            key, subkey = random.split(key)
            params, opt_state, loss, aux_data = opt_update_step(
                ansatz, params, walkers, opt_state, subkey
            )
        
        return walkers, params, opt_state, loss, aux_data, pmove
    
    return jax.jit(training_step)


def make_kfac_training_step(mcmc_step, optimizer, n_mcmc_per_opt=1, n_opt_per_mcmc=1):
    """Factory to create unified training step for KFAC optimizer.
    
    KFAC optimizer handles gradient computation internally via its step() method,
    so this is slightly different from the standard Optax version.
    
    Args:
        mcmc_step: JIT-compiled MCMC step function from make_mcmc_step()
                   Signature: (ansatz, walkers, key, params) -> (walkers, pmove)
        optimizer: KFAC optimizer instance
        n_mcmc_per_opt: Number of MCMC steps before each optimization update (default: 1)
        n_opt_per_mcmc: Number of optimization steps per MCMC update (default: 1)
    
    Returns:
        A function (not JIT-compiled yet) with signature:
            training_step(ansatz, walkers, params, opt_state, key, global_step) ->
                (walkers, params, opt_state, loss, aux_data, pmove)
    """
    def training_step(ansatz, walkers, params, opt_state, key, global_step):
        """One full training iteration: MCMC + optimization.
        
        Note: This is NOT JIT-compiled because KFAC handles JIT internally.
        
        Args:
            ansatz: Wavefunction object
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
            pmove: Acceptance probability from last MCMC step
        """
        # Pattern 1: Multiple MCMC steps before optimization (energy minimization)
        if n_mcmc_per_opt > 1:
            pmove_list = []
            for _ in range(n_mcmc_per_opt):
                key, subkey = random.split(key)
                walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
                pmove_list.append(pmove)
            pmove = pmove_list[-1]  # Use last acceptance rate
            
            # Single optimization step
            key, subkey = random.split(key)
            params, opt_state, stats = optimizer.step(
                params=params,
                state=opt_state,
                rng=subkey,
                batch=(walkers, ansatz), # Pass ansatz in batch
                global_step_int=global_step
            )
            loss = stats['loss']
            aux_data = stats['aux']
        
        # Pattern 2: Multiple optimization steps per MCMC (variance minimization)
        elif n_opt_per_mcmc > 1:
            # Use jax.lax.scan for the optimization loop to avoid unrolling overhead
            def opt_scan_body(carry, _):
                p, s, k = carry
                k, sk = random.split(k)
                new_p, new_s, stats = optimizer.step(
                    params=p,
                    state=s,
                    rng=sk,
                    batch=(walkers, ansatz), # Pass ansatz in batch
                    global_step_int=global_step
                )
                return (new_p, new_s, k), stats

            (params, opt_state, key), stats_history = jax.lax.scan(
                opt_scan_body,
                (params, opt_state, key),
                None,
                length=n_opt_per_mcmc
            )
            
            # Use the last loss and aux_data from the optimization loop
            # stats_history contains stacked results from all steps
            # We take the last element (index -1)
            loss = jax.tree_util.tree_map(lambda x: x[-1], stats_history['loss'])
            aux_data = jax.tree_util.tree_map(lambda x: x[-1], stats_history['aux'])
            
            # Single MCMC step after optimization
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
        
        # Pattern 3: Balanced (1 MCMC, 1 opt)
        else:
            # Single MCMC step
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
            
            # Single optimization step
            key, subkey = random.split(key)
            params, opt_state, stats = optimizer.step(
                params=params,
                state=opt_state,
                rng=subkey,
                batch=(walkers, ansatz), # Pass ansatz in batch
                global_step_int=global_step
            )
            loss = stats['loss']
            aux_data = stats['aux']
        
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
    use_importance_sampling: bool = False,
    initial_walkers=None,
    key=None,
    # Optimization parameters
    n_opt_steps: int = 100,
    max_vmap_batch_size: int = 0,
    learning_rate: float = 0.01,
    optimizer_type: str = "adam",
    opt_kwargs: Optional[Dict[str, Any]] = None,
    params=None,
    frozen_params=None,
    move_type: str = "one",
    use_custom_jvp: bool = True,
    adaptive_step_size: bool = True,
    step_size_adjust_interval: int = 10,
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
        max_vmap_batch_size: If 0, use standard vmap. If >0, use folx.batched_vmap with
                            the given batch size for memory efficiency. Recommended: 10-50
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

    # Create loss function using modular factory
    # If user provides custom cost_fn, use it; otherwise use default mean energy
    # internal_loss_fn = make_energy_loss(
    #     ansatz=ansatz,
    #     optimizer_type=optimizer_type,
    #     cost_fn=user_or_default_cost_fn,
    #     clip_multiplier=5.0,
    #     use_custom_jvp=use_custom_jvp,
    #     max_vmap_batch_size=max_vmap_batch_size
    # )
    raise NotImplementedError("make_energy_loss has been removed. Use ferminet.loss.make_loss directly or update this function.")

    # Create mask for parameter freezing
    gradient_mask = create_gradient_mask(ansatz, params, frozen_params)

    # Create MCMC step function using factory
    if use_importance_sampling:
        mcmc_step = make_mcmc_step_importance(ansatz, step_size)
    else:
        mcmc_step = make_mcmc_step(ansatz, step_size, move_type)

    # Create optimizer and training step using factory functions
    if optimizer_type.lower() == "kfac":
        # KFAC specific setup
        opt_kwargs["value_and_grad_func"] = jax.value_and_grad(internal_loss_fn, argnums=0, has_aux=True)
        opt_kwargs["value_func_has_aux"] = True
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        
        key, subkey = random.split(key)
        opt_state = optimizer.init(params, subkey, (walkers, ansatz))
        
        training_step = make_kfac_training_step(
            mcmc_step, optimizer, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
        )
    elif optimizer_type.lower() == "mfgn":
        # MFGN setup
        opt_kwargs["value_and_grad_func"] = jax.value_and_grad(internal_loss_fn, argnums=0, has_aux=True)
        opt_kwargs["curvature"] = "fisher" # Energy minimization uses SR
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        
        key, subkey = random.split(key)
        opt_state = optimizer.init(params, subkey, (walkers, ansatz))
        
        # Use KFAC training step factory as it supports the step() interface
        training_step = make_kfac_training_step(
            mcmc_step, optimizer, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
        )
        # MFGN needs explicit JIT since it doesn't handle it internally like KFAC
        training_step = jax.jit(training_step)
    else:
        # Optax setup
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        opt_state = optimizer.init(params)
        
        # Create Optax training step - apply gradient mask in the loss function wrapper
        if gradient_mask is not None:
            # Wrap internal_loss_fn to apply gradient masking
            original_loss_fn = internal_loss_fn
            def masked_loss_fn(params_inner, batch_data):
                return original_loss_fn(params_inner, batch_data)
            # Note: gradient masking will be applied via custom_jvp, which respects the mask
            internal_loss_fn = masked_loss_fn
        
        opt_update_step = make_opt_update_step(internal_loss_fn, optimizer)
        # Use n_mcmc_per_opt pattern for energy optimization
        training_step = make_training_step(
            mcmc_step, opt_update_step, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
        )

    # ========== MAIN LOOP (Uses JIT-compiled step) ==========
    print(f"Starting optimization with {n_opt_steps} steps...")
    if adaptive_step_size:
        print(f"Adaptive step-size enabled (target accept=0.5, adjust every {step_size_adjust_interval} steps)")
    
    losses = []
    energies = []
    stds = []
    acceptances = []
    params_history = []
    step_sizes = [step_size]  # Track step_size history
    
    start_time = time.time()
    
    for opt_step in range(n_opt_steps):
        key, subkey = random.split(key)
        
        if optimizer_type.lower() in ["kfac", "mfgn"]:
            walkers, params, opt_state, loss, aux_data, pmove = training_step(
                ansatz, walkers, params, opt_state, subkey, opt_step
            )
        else:
            walkers, params, opt_state, loss, aux_data, pmove = training_step(
                ansatz, walkers, params, opt_state, subkey
            )
        
        # Materialize values
        cost_val = float(jax.device_get(loss))
        # aux_data is a namedtuple with (mean_energy, energy_std, clipped_energies, diff)
        # Extract only the first two for backward compatibility
        aux_data_materialized = jax.device_get(aux_data)
        energy_val = float(aux_data_materialized[0])  # mean_energy
        std_val = float(aux_data_materialized[1])     # energy_std
        pmove_val = float(jax.device_get(pmove))
        
        # Store history
        losses.append(cost_val)
        energies.append(energy_val)
        stds.append(std_val)
        acceptances.append(pmove_val)
        
        # Store params (materialize to numpy)
        params_copy = tree_map(
            lambda x: np.array(jax.device_get(x)) if isinstance(x, jnp.ndarray) else x,
            params
        )
        params_history.append(params_copy)
        
        # Adaptive step-size adjustment (similar to burn-in)
        if adaptive_step_size and (opt_step + 1) % step_size_adjust_interval == 0 and opt_step < 5*step_size_adjust_interval:
            # Calculate mean acceptance over last interval
            recent_accept = np.mean(acceptances[-step_size_adjust_interval:])
            # Adjust step_size to target 0.5 acceptance rate
            step_size *= recent_accept / 0.5
            step_sizes.append(step_size)
            
            # Recreate mcmc_step with new step_size
            if use_importance_sampling:
                mcmc_step = make_mcmc_step_importance(ansatz, step_size)
            else:
                mcmc_step = make_mcmc_step(ansatz, step_size, move_type)
            
            # Recreate training_step with new mcmc_step
            if optimizer_type.lower() in ["kfac", "mfgn"]:
                training_step = make_kfac_training_step(
                    mcmc_step, optimizer, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
                )
                if optimizer_type.lower() == "mfgn":
                    training_step = jax.jit(training_step)
            else:
                training_step = make_training_step(
                    mcmc_step, opt_update_step, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
                )
        
        # Print progress
        log_frequency = 1  # Log ~100 times
        if opt_step % log_frequency == 0 or opt_step == n_opt_steps - 1:
            elapsed = time.time() - start_time
            step_size_str = f" | StepSize: {step_size:.4f}" if adaptive_step_size else ""
            print(f"Step {opt_step:5d} | Cost: {cost_val:.6f} | "
                  f"E: {energy_val:.6f}±{std_val:.6f} | "
                  f"Accept: {pmove_val:.3f}{step_size_str} | Time: {elapsed:.2f}s")
            start_time = time.time()
    
    print("Optimization complete!")
    if adaptive_step_size:
        print(f"Final step size: {step_size:.4f}")
    
    return {
        "cost": np.array(losses),
        "energies": np.array(energies),
        "stds": np.array(stds),
        "acceptance": np.array(acceptances),
        "params": params_history,
        "step_sizes": np.array(step_sizes) if adaptive_step_size else None
    }


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
    max_vmap_batch_size: int = 0,
    learning_rate: float = 0.01,
    optimizer_type: str = "adam",
    move_type: str = "one",
    opt_kwargs: Optional[Dict[str, Any]] = None,
    params=None,
    adaptive_step_size: bool = True,
    step_size_adjust_interval: int = 10,
):
    """Perform variational Monte Carlo optimization using MCMC sampling.

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
        max_vmap_batch_size: If 0, use standard vmap. If >0, use folx.batched_vmap with
                            the given batch size for memory efficiency. Recommended: 10-50
        learning_rate: Learning rate for optimizer
        optimizer_type: Type of optimizer ("adam", "sgd", "kfac", etc.)
        move_type: "one" or "all" for MCMC electron moves
        opt_kwargs: Additional optimizer parameters
        params: Initial combined parameters [jastrow_params, linear_coeffs].

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


    # Create loss function using modular factory
    if cost_fn is None:
        # Use modular variance loss factory
        loss_fn = make_variance_loss(
            ansatz=ansatz,
            optimizer_type=optimizer_type,
            use_custom_jvp=True,
            max_vmap_batch_size=max_vmap_batch_size
        )
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
        opt_state = optimizer.init(params, subkey, (walkers, ansatz))
        
        # Create KFAC training step with n_opt_per_mcmc pattern
        training_step = make_kfac_training_step(
            mcmc_step, optimizer, n_mcmc_per_opt=1, n_opt_per_mcmc=n_steps
        )
    elif optimizer_type.lower() == "mfgn":
        # MFGN setup
        opt_kwargs["value_and_grad_func"] = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        opt_kwargs["curvature"] = "gauss_newton" # Variance minimization uses GN
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        
        key, subkey = random.split(key)
        opt_state = optimizer.init(params, subkey, (walkers, ansatz))
        
        training_step = make_kfac_training_step(
            mcmc_step, optimizer, n_mcmc_per_opt=1, n_opt_per_mcmc=n_steps
        )
        # MFGN needs explicit JIT since it doesn't handle it internally like KFAC
        training_step = jax.jit(training_step)
    else:
        # Optax setup
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        opt_state = optimizer.init(params)
        
        # Create Optax training step - use n_opt_per_mcmc pattern for variance optimization
        opt_update_step = make_opt_update_step(loss_fn, optimizer)
        training_step = make_training_step(
            mcmc_step, opt_update_step, n_mcmc_per_opt=1, n_opt_per_mcmc=n_steps
        )

    # ========== MAIN LOOP (Uses JIT-compiled step) ==========
    print(f"Starting optimization with {n_opt_steps} steps...")
    
    losses = []
    energies = []
    stds = []
    acceptances = []
    params_history = []
    
    start_time = time.time()
    
    # Run first step separately to measure compilation time
    print("Compiling training step...")
    compilation_start = time.time()
    
    key, subkey = random.split(key)
    if optimizer_type.lower() in ["kfac", "mfgn"]:
        walkers, params, opt_state, loss, aux_data, pmove = training_step(
            ansatz, walkers, params, opt_state, subkey, 0
        )
    else:
        walkers, params, opt_state, loss, aux_data, pmove = training_step(
            ansatz, walkers, params, opt_state, subkey
        )
    compilation_end = time.time()
    print(f"Compilation finished in {compilation_end - compilation_start:.2f}s")
    
    # Process first step results
    variance_val = float(jax.device_get(loss))
    energy_val, std_val = jax.device_get(aux_data)
    energy_val = float(energy_val)
    std_val = float(std_val)
    pmove_val = float(jax.device_get(pmove))
    
    losses.append(variance_val)
    energies.append(energy_val)
    stds.append(std_val)
    acceptances.append(pmove_val)
    
    params_copy = tree_map(
        lambda x: np.array(jax.device_get(x)) if isinstance(x, jnp.ndarray) else x,
        params
    )
    params_history.append(params_copy)
    
    print(f"Step     0 | Var: {variance_val:.6f} | "
          f"E: {energy_val:.6f}±{std_val:.6f} | "
          f"Accept: {pmove_val:.3f} | Time: {compilation_end - start_time:.2f}s")

    for opt_step in range(1, n_opt_steps):
        key, subkey = random.split(key)
        
        if optimizer_type.lower() in ["kfac", "mfgn"]:
            walkers, params, opt_state, loss, aux_data, pmove = training_step(
                ansatz, walkers, params, opt_state, subkey, opt_step
            )
        else:
            walkers, params, opt_state, loss, aux_data, pmove = training_step(
                ansatz, walkers, params, opt_state, subkey
            )
        
        variance_val = float(jax.device_get(loss))
        energy_val, std_val = jax.device_get(aux_data)
        energy_val = float(energy_val)
        std_val = float(std_val)
        pmove_val = float(jax.device_get(pmove))

        
        log_frequency = 1 #max(1, n_opt_steps // 10)  # Log ~100 times
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
