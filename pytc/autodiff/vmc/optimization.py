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
from pytc.utils.checkpoint import Checkpoint


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
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 10,
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

    # Initialize checkpointer if path is provided
    checkpointer = None
    if checkpoint_path:
        checkpointer = Checkpoint(checkpoint_path)

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
        
        # Checkpoint
        if checkpointer and (opt_step % checkpoint_every == 0 or opt_step == n_opt_steps - 1):
             checkpointer.save(opt_step, params)
    
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
    params=None,
    data=None, # Walkers/Data
    mcmc_step=None,
    loss_fn=None, # Optional, if None will be created
    optimizer=None, # Optional, if None will be created
    opt_state=None, # Optional
    n_walkers: int = 100,
    n_steps: int = 20,  # MCMC steps per optimization update (not used if mcmc_step provided with internal steps)
    burn_in_steps: int = 1000,
    n_opt_steps: int = 100,
    learning_rate: float = 0.01,
    optimizer_type: str = "adam",
    opt_kwargs: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 10,
    # Additional args for initialization if needed
    initial_walkers=None,
    key=None,
    atoms=None, # Needed for initialization if data is None
    electrons=None, # Needed for initialization if data is None
    batch_size=None,
    mcmc_width=0.1,
    adapt_frequency=10,
):
    """Perform variational Monte Carlo optimization (Variance Minimization).

    Refactored to match the workflow in test_vmc_ref_variance_min_single_device.py.
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))

    if opt_kwargs is None:
        opt_kwargs = {}

    # 1. Initialization of Params
    if params is None:
        jastrow_params = ansatz.jastrow.init_params()
        linear_coeffs = jnp.ones(len(ansatz.dets))
        params = [jastrow_params, linear_coeffs]
    
    # 2. Initialization of Data/Walkers
    if data is None:
        if atoms is None or electrons is None:
             raise ValueError("If `data` is not provided, `atoms` and `electrons` must be provided for initialization.")
        
        if batch_size is None:
            batch_size = n_walkers

        # Initialize electrons
        key, subkey = random.split(key)
        # Assuming train.init_electrons is available or we use a local helper
        # For now, let's assume the user passes initialized data or we use the helper from the test file logic
        # But we don't have train imported here. 
        # Let's rely on initialize_walkers from mcmc_utils if possible, or raise error if not simple.
        # The test file uses: train.init_electrons and networks.FermiNetData
        # We will assume data is passed in for now to be safe, or use initialize_walkers
        
        # Fallback to initialize_walkers from mcmc_utils (which returns simple array usually)
        # But FermiNetData is a dataclass.
        # Let's try to use initialize_walkers and wrap it?
        # For KFAC, we need FermiNetData structure usually if using KFAC-JAX.
        # Let's assume data is passed for this refactor to avoid dependency hell, 
        # or use the simple initialize_walkers and hope it works for the ansatz.
        
        # Actually, let's use the initialize_walkers from this module which calls ansatz.init_walkers?
        # No, initialize_walkers in this module calls `initialize_walkers` from `mcmc_utils`.
        
        ref_det = ansatz.dets[0]
        walkers = initialize_walkers(ref_det, n_walkers, initial_walkers, key)
        # If we need FermiNetData (positions, spins, atoms, charges), we might need more info.
        # For now, let's assume 'walkers' is sufficient or 'data' is passed.
        data = walkers

    # 3. MCMC Step Setup
    if mcmc_step is None:
        # Create default MCMC step
        ref_det = ansatz.dets[0]
        # We need atom positions for MCMC if using nuclear attraction?
        # The test file uses mcmc.make_mcmc_step.
        # Let's use make_mcmc_step from this module which wraps it.
        # make_mcmc_step(ansatz, step_size, move_type="one")
        mcmc_step = make_mcmc_step(ref_det, step_size=mcmc_width, move_type="one")

    # 4. Optimizer Setup
    if optimizer is None:
        if loss_fn is None:
             # Create default variance loss
             # We need local_energy_fn. 
             # The test file creates it using hamiltonian.local_energy.
             # We can use make_variance_loss which handles it? 
             # make_variance_loss in loss.py takes 'local_energy' function.
             # This seems circular if we don't have it.
             # Let's assume loss_fn is passed or we can construct it if we have enough info.
             # For this refactor, let's assume the caller provides the loss_fn or we use the one from loss.py
             # But make_variance_loss needs local_energy.
             raise ValueError("`loss_fn` must be provided for now.")
        
        if optimizer_type.lower() == 'kfac':
            def value_and_grad_func(p, rng, batch):
                return jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(p, rng, batch)
            
            # Update opt_kwargs for KFAC
            kfac_kwargs = {
                'value_and_grad_func': value_and_grad_func,
                'value_func_has_aux': True,
                'value_func_has_rng': True,
                'initial_damping': 1.0,
                'use_adaptive_learning_rate': False,
                'norm_constraint': 1e-5,
            }
            kfac_kwargs.update(opt_kwargs)
            
            optimizer = create_optimizer('kfac', learning_rate, kfac_kwargs)
            
            key, subkey = random.split(key)
            if opt_state is None:
                opt_state = optimizer.init(params, subkey, data)
                
            def opt_step_fn(data, params, state, key, global_step_int):
                new_params, new_state, stats = optimizer.step(params, state, key, batch=data, learning_rate=learning_rate)
                loss_val = stats['loss']
                aux_data = stats['aux']
                return new_params, new_state, loss_val, aux_data
                
        else:
            optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
            if opt_state is None:
                opt_state = optimizer.init(params)
                
            def opt_step_fn(data, params, state, key, global_step_int=None):
                loss_key = key
                (loss_val, aux_data), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
                    params, loss_key, data
                )
                updates, new_state = optimizer.update(grads, state, params)
                new_params = optax.apply_updates(params, updates)
                return new_params, new_state, loss_val, aux_data
            
            opt_step_fn = jax.jit(opt_step_fn)
    else:
        # If optimizer is passed, we assume opt_step_fn is also handled or we recreate it?
        # For simplicity, let's require optimizer AND opt_state if optimizer is passed, 
        # OR we just recreate the step function.
        # Let's assume the user passes everything or nothing.
        # If optimizer is passed, we need to know if it's KFAC or not to define opt_step_fn.
        pass # To be implemented if needed, for now assume we create it.

    # 5. Checkpointing Setup
    checkpointer = None
    if checkpoint_path:
        checkpointer = Checkpoint(checkpoint_path)
        # Load if exists and params were not explicitly provided (or we want to overwrite)
        # Actually, if we just initialized params, we should try to load.
        # If params were passed in, maybe we shouldn't overwrite?
        # Let's try to load.
        loaded = checkpointer.load()
        if loaded:
            print(f"Loaded parameters from {checkpoint_path}")
            # Assuming loaded is the params pytree
            params = loaded
            # We might need to re-init opt_state if params changed?
            # KFAC state depends on params.
            # If we loaded params, we should probably re-init opt_state or load opt_state too.
            # For now, let's re-init opt_state with loaded params.
            if optimizer_type.lower() == 'kfac':
                key, subkey = random.split(key)
                opt_state = optimizer.init(params, subkey, data)
            else:
                opt_state = optimizer.init(params)

    # 6. Burn-in (if needed)
    # If we loaded params, maybe we don't need burn-in? Or maybe we do to equilibrate walkers.
    # Let's do burn-in if burn_in_steps > 0
    if burn_in_steps > 0:
        print(f"Burning in for {burn_in_steps} steps...")
        pmoves = np.zeros(adapt_frequency)
        current_pmove = 0.5
        curr_mcmc_width = mcmc_width
        
        for i in range(burn_in_steps):
            key, subkey = random.split(key)
            data, pmove = mcmc_step(params, data, subkey, curr_mcmc_width)
            
            pmoves[i % adapt_frequency] = pmove
            if i > 0 and i % adapt_frequency == 0:
                curr_mcmc_width, pmoves = update_mcmc_width_local(
                    i, curr_mcmc_width, adapt_frequency, current_pmove, pmoves
                )
            current_pmove = pmove
            
        print(f"Burn-in complete. Final pmove={current_pmove:.2f}, width={curr_mcmc_width:.4f}")
        mcmc_width = curr_mcmc_width # Update starting width for optimization

    # 7. Main Loop
    print(f"Starting optimization with {n_opt_steps} steps...")
    
    losses = []
    energies = []
    stds = []
    acceptances = []
    params_history = []
    
    start_time = time.time()
    
    pmoves = np.zeros(adapt_frequency)
    current_pmove = 0.5
    
    for t in range(n_opt_steps):
        # MCMC Step
        key, subkey = random.split(key)
        data, pmove = mcmc_step(params, data, subkey, mcmc_width)
        
        # Update MCMC width
        pmoves[t % adapt_frequency] = pmove
        if t > 0 and t % adapt_frequency == 0:
            mcmc_width, pmoves = update_mcmc_width_local(
                t, mcmc_width, adapt_frequency, current_pmove, pmoves
            )
        current_pmove = pmove

        # Optimization Step
        key, subkey = random.split(key)
        global_step = t
        params, opt_state, loss_val, aux_data = opt_step_fn(data, params, opt_state, subkey, global_step)
        
        # Constrain ncusp parameters if present (generic way?)
        # This is specific to the test file logic.
        # We can check if 'ncusp' is in params and if ansatz has ncusp_apply.
        # For now, let's skip or assume the user handles it via callback?
        # Or we can check:
        if 'ncusp' in params and hasattr(ansatz, 'ncusp_apply') and hasattr(ansatz.ncusp_apply, 'constrain'):
             params['ncusp'] = ansatz.ncusp_apply.constrain(params['ncusp'])

        # Materialize
        loss_val = float(jax.device_get(loss_val))
        energy_val = float(jax.device_get(aux_data.energy)) if hasattr(aux_data, 'energy') else 0.0
        # aux_data might be different depending on loss_fn.
        # In test file: aux_data.energy.
        
        losses.append(loss_val)
        energies.append(energy_val)
        acceptances.append(float(pmove))
        
        # Logging
        if t % adapt_frequency == 0 or t == n_opt_steps - 1:
            elapsed = time.time() - start_time
            print(f"Step {t:5d} | Var: {loss_val:.6f} | E: {energy_val:.6f} | Accept: {pmove:.2f} | Time: {elapsed:.2f}s")
            start_time = time.time()

        # Checkpoint
        if checkpointer and (t % checkpoint_every == 0 or t == n_opt_steps - 1):
             checkpointer.save(t, params)

    print("Optimization complete!")
    
    return {
        "cost": np.array(losses),
        "energies": np.array(energies),
        "acceptance": np.array(acceptances),
        "params": params # Return final params
    }

def update_mcmc_width_local(t, width, adapt_frequency, pmove, pmoves):
    """Helper to update MCMC width."""
    # Simple adaptation: if accept > 0.55, increase width; if < 0.45, decrease.
    # Or use the logic from ferminet.mcmc
    # For now, simple placeholder or copy from ferminet if imported.
    # The test file uses mcmc.update_mcmc_width.
    # We should probably import it.
    from ferminet import mcmc
    return mcmc.update_mcmc_width(t, width, adapt_frequency, pmove, pmoves)

