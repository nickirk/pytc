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

The factory function make_training_step() supports both patterns.
"""

import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as spla
from jax import random, value_and_grad
from jax.tree_util import tree_map
import optax
from typing import Dict, Any, Optional

from .metropolis import make_mcmc_step, make_mcmc_step_importance
from .walker import initialize_walkers
from .sampling import burn_in, burn_in_with_importance
from .optimizer import create_optimizer, create_gradient_mask
from .loss import (
    make_energy_loss,
    make_variance_loss,
    make_state_averaged_variance_loss,
)
from .mcmc_utils import save_optimization_history

logger = logging.getLogger(__name__)


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
        
        # Try to get learning rate from opt_update_step's auxiliary output if possible
        # but for Optax it's cleaner to just return it from here if we want to log it.
        # However, optax.scale_by_learning_rate usually handles it within opt_state.
        
        return walkers, params, opt_state, loss, aux_data, pmove
    
    return jax.jit(training_step)


def make_second_order_training_step(mcmc_step, optimizer, n_mcmc_per_opt=1, n_opt_per_mcmc=1):
    """Factory to create unified training step for optimizers with a .step() method.
    
    This supports Newton-style optimizers which handle gradient computation 
    internally via their step() method.
    """
    def training_step(ansatz, walkers, params, opt_state, key, global_step):
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
            pmove = pmoves[-1]
            
            # Single optimization step
            key, subkey = random.split(key)
            params, opt_state, stats = optimizer.step(
                params=params,
                state=opt_state,
                rng=subkey,
                batch=(walkers, ansatz),
                global_step_int=global_step
            )
            loss = stats['loss']
            aux_data = stats['aux']
        
        # Pattern 2: Multiple optimization steps per MCMC (variance minimization)
        elif n_opt_per_mcmc > 1:
            def opt_scan_body(carry, _):
                p, s, k = carry
                k, sk = random.split(k)
                new_p, new_s, stats = optimizer.step(
                    params=p,
                    state=s,
                    rng=sk,
                    batch=(walkers, ansatz),
                    global_step_int=global_step
                )
                return (new_p, new_s, k), stats

            (params, opt_state, key), stats_history = jax.lax.scan(
                opt_scan_body,
                (params, opt_state, key),
                None,
                length=n_opt_per_mcmc
            )

            # Collapse the per-iteration stack to the trailing entry so the
            # `stats` variable downstream sees the same shape as patterns 1/3.
            stats = jax.tree_util.tree_map(lambda x: x[-1], stats_history)
            loss = stats['loss']
            aux_data = stats['aux']
            
            # Single MCMC step after optimization
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
        
        # Pattern 3: Balanced
        else:
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
            
            key, subkey = random.split(key)
            params, opt_state, stats = optimizer.step(
                params=params,
                state=opt_state,
                rng=subkey,
                batch=(walkers, ansatz),
                global_step_int=global_step
            )
            loss = stats['loss']
            aux_data = stats['aux']
        
        lr = stats.get('lr', None)
        return walkers, params, opt_state, loss, aux_data, pmove, lr
    
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
    save_frequency: int = 100,
    save_path: Optional[str] = None,
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
    
    # ---- Multi-GPU setup ----
    from .sharding import (
        create_mesh, replicate, initialize_walkers_sharded,
        pad_n_walkers, pad_walker, n_devices as get_n_devices,
        is_multi_gpu as check_multi_gpu
    )
    
    multi_gpu = check_multi_gpu()
    mesh = None
    if multi_gpu:
        num_devices = get_n_devices()
        mesh = create_mesh()
        padded_n = pad_n_walkers(n_walkers, num_devices)
        if padded_n != n_walkers:
            logger.info(f"Padding n_walkers from {n_walkers} to {padded_n} "
                  f"(divisible by {num_devices} devices)")
            n_walkers = padded_n
        
        logger.info(f"Multi-GPU auto-detected: {num_devices} devices, "
              f"{n_walkers // num_devices} walkers/device")
    
    # Initialize walkers
    if multi_gpu and mesh is not None:
        walkers = initialize_walkers_sharded(
            ansatz, n_walkers, mesh, initial_walkers=initial_walkers, key=key
        )
        if params is not None:
            params = replicate(params, mesh)
        key = replicate(key, mesh)
    else:
        walkers = initialize_walkers(ansatz, n_walkers, initial_walkers, key)

    # Perform burn-in with appropriate method
    if use_importance_sampling:
        walkers, acceptance_history, key, step_size = burn_in_with_importance(
            ansatz, walkers, burn_in_steps, step_size, key, params, mesh=mesh)
    else:
        walkers, acceptance_history, key, step_size = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key, params, 
            move_type=move_type, max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)
    
    logger.info("Starting optimization...")
    
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
    internal_loss_fn = make_energy_loss(
        ansatz=ansatz,
        optimizer_type=optimizer_type,
        cost_fn=user_or_default_cost_fn,
        clip_multiplier=5.0,
        use_custom_jvp=use_custom_jvp,
        max_vmap_batch_size=max_vmap_batch_size,
        mesh=mesh
    )

    # Create mask for parameter freezing
    gradient_mask = create_gradient_mask(ansatz, params, frozen_params)

    # Create MCMC step function using factory
    if use_importance_sampling:
        mcmc_step = make_mcmc_step_importance(ansatz, step_size, mesh=mesh)
    else:
        mcmc_step = make_mcmc_step(
            ansatz, step_size, move_type,
            max_vmap_batch_size=max_vmap_batch_size, mesh=mesh
        )

    # Define loss function JVP for KFAC and Newton
    loss_fn_jvp = jax.value_and_grad(internal_loss_fn, argnums=0, has_aux=True)

    # Create optimizer and training step using factory functions
    if optimizer_type.lower() == "newton":
        # Newton setup
        opt_kwargs["value_and_grad_func"] = loss_fn_jvp
        opt_kwargs["curvature"] = "fisher" # Energy minimization uses Fisher
        opt_kwargs["max_vmap_batch_size"] = max_vmap_batch_size
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        
        key, subkey = random.split(key)
        opt_state = optimizer.init(params, subkey, (walkers, ansatz))
        
        # Use second order training step factory as it supports the step() interface
        training_step = make_second_order_training_step(
            mcmc_step, optimizer, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
        )
        # Newton needs explicit JIT since it doesn't handle it internally
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
    logger.info(f"Starting optimization with {n_opt_steps} steps...")
    if adaptive_step_size:
        logger.info(f"Adaptive step-size enabled (target accept=0.5, adjust every {step_size_adjust_interval} steps)")
    
    losses = []
    energies = []
    stds = []
    acceptances = []
    params_history = []
    step_sizes = [step_size]  # Track step_size history
    
    start_time = time.time()
    
    for opt_step in range(n_opt_steps):
        key, subkey = random.split(key)
        
        if optimizer_type.lower() in ["newton"]:
            walkers, params, opt_state, loss, aux_data, pmove, current_lr = training_step(
                ansatz, walkers, params, opt_state, subkey, opt_step
            )
        else:
            walkers, params, opt_state, loss, aux_data, pmove = training_step(
                ansatz, walkers, params, opt_state, subkey
            )
            current_lr = None # Optax handles internally, could extract from opt_state if needed
        
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
                mcmc_step = make_mcmc_step_importance(ansatz, step_size, mesh=mesh)
            else:
                mcmc_step = make_mcmc_step(
                    ansatz, step_size, move_type,
                    max_vmap_batch_size=max_vmap_batch_size, mesh=mesh
                )
            
            # Recreate training_step with new mcmc_step
            if optimizer_type.lower() in ["newton"]:
                training_step = make_second_order_training_step(
                    mcmc_step, optimizer, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
                )
                if optimizer_type.lower() == "newton":
                    training_step = jax.jit(training_step)
            else:
                training_step = make_training_step(
                    mcmc_step, opt_update_step, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
                )
        
        # Print progress
        log_frequency = 1  # Log ~100 times
        if opt_step % log_frequency == 0 or opt_step == n_opt_steps - 1:
            lr_str = f" | LR: {current_lr:.4f}" if current_lr is not None else ""
            elapsed = time.time() - start_time
            step_size_str = f" | StepSize: {step_size:.4f}" if adaptive_step_size else ""
            logger.info(f"Step {opt_step:5d} | Cost: {cost_val:.6f} | "
                  f"E: {energy_val:.6f}±{std_val:.6f} | "
                  f"Accept: {pmove_val:.3f}{step_size_str}{lr_str} | Time: {elapsed:.2f}s")
            start_time = time.time()

        # Periodic save to disk
        if save_path and (opt_step + 1) % save_frequency == 0:
            current_history = {
                "cost": np.array(losses),
                "energies": np.array(energies),
                "stds": np.array(stds),
                "acceptance": np.array(acceptances),
                "params": params_history,
                "step_sizes": np.array(step_sizes) if adaptive_step_size else None
            }
            save_optimization_history(current_history, save_path)
            logger.info(f"Saved intermediate optimization history to {save_path}")
    
    logger.info("Optimization complete!")
    if adaptive_step_size:
        logger.info(f"Final step size: {step_size:.4f}")
    
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
    n_steps: int = 20,
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
    jacobian_sample_size: Optional[int] = None,
    save_frequency: int = 100,
    save_path: Optional[str] = None,
    n_mcmc_per_opt: Optional[int] = None,
    n_opt_per_mcmc: Optional[int] = None,
):
    """Perform variational Monte Carlo optimization using MCMC sampling.

    Args:
        ansatz: Wavefunction object with __call__ method that returns ψ(R)
        cost_fn: Cost function (defaults to reference variance if None).
                 Should accept (params, walkers) and return (cost, aux_data).
        n_walkers: Number of parallel walkers
        n_steps: Legacy training cadence parameter. If neither
                 ``n_mcmc_per_opt`` nor ``n_opt_per_mcmc`` is provided,
                 ``optimize_ref_var`` preserves its historical behavior and
                 uses ``n_opt_per_mcmc=n_steps``.
        step_size: Standard deviation of Gaussian proposal for MCMC
        burn_in_steps: Number of initial MCMC steps to discard (equilibration)
        initial_walkers: Optional initial positions, otherwise initialized near nuclei
        key: PRNG key
        n_opt_steps: Number of optimization steps
        max_vmap_batch_size: If 0, use standard vmap. If >0, use folx.batched_vmap with
                            the given batch size for memory efficiency. Recommended: 10-50
        learning_rate: Learning rate for optimizer
        optimizer_type: Type of optimizer ("adam", "sgd", etc.)
        move_type: "one" or "all" for MCMC electron moves
        opt_kwargs: Additional optimizer parameters
        params: Initial combined parameters [jastrow_params, linear_coeffs].
        jacobian_sample_size: Optional[int]. If provided and using Newton optimizer,
                             subsample this many walkers for Jacobian computation
                             (curvature matrix approximation). Speeds up Newton steps
                             when n_walkers is large. Typical: 500-2000 for 100k walkers.
        n_mcmc_per_opt: Optional explicit number of MCMC steps before each
                        optimization update.
        n_opt_per_mcmc: Optional explicit number of optimization steps before
                        each MCMC refresh.

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

    if n_mcmc_per_opt is None and n_opt_per_mcmc is None:
        n_mcmc_per_opt = 1
        n_opt_per_mcmc = n_steps
    elif n_mcmc_per_opt is None:
        n_mcmc_per_opt = 1
    elif n_opt_per_mcmc is None:
        n_opt_per_mcmc = 1

    if n_mcmc_per_opt < 1 or n_opt_per_mcmc < 1:
        raise ValueError("`n_mcmc_per_opt` and `n_opt_per_mcmc` must both be >= 1.")

    if n_mcmc_per_opt > 1 and n_opt_per_mcmc > 1:
        raise ValueError(
            "`optimize_ref_var` supports either multiple MCMC steps per update "
            "or multiple optimization steps per MCMC refresh, not both at once."
        )

    # ---- Multi-GPU setup ----
    from .sharding import (
        create_mesh, replicate, initialize_walkers_sharded,
        pad_n_walkers, pad_walker, n_devices as get_n_devices,
        is_multi_gpu as check_multi_gpu
    )
    
    multi_gpu = check_multi_gpu()
    mesh = None
    if multi_gpu:
        num_devices = get_n_devices()
        mesh = create_mesh()
        padded_n = pad_n_walkers(n_walkers, num_devices)
        if padded_n != n_walkers:
            logger.info(f"Padding n_walkers from {n_walkers} to {padded_n} "
                  f"(divisible by {num_devices} devices)")
            n_walkers = padded_n
        logger.info(f"Multi-GPU auto-detected: {num_devices} devices, "
              f"{n_walkers // num_devices} walkers/device")

    # Initialize walkers using the reference determinant's info
    ref_det = ansatz.dets[0]
    if multi_gpu and mesh is not None:
        walkers = initialize_walkers_sharded(
            ref_det, n_walkers, mesh, initial_walkers=initial_walkers, key=key
        )
        params = replicate(params, mesh)
        key = replicate(key, mesh)
        logger.info("Walkers initialized and sharded across devices.")
    else:
        walkers = initialize_walkers(ref_det, n_walkers, initial_walkers, key)

    # Burn-in walkers using the initial combined parameters
    logger.info("Performing burn-in...")
    walkers, acceptance_history, key, step_size = burn_in(
        ref_det, walkers, burn_in_steps, step_size, key, params=params, 
        move_type=move_type, max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)
    logger.info(f"Burn-in complete. Final step size: {step_size:.4f}")

    # Create loss function using modular factory
    if cost_fn is None:
        # Use modular variance loss factory
        loss_fn = make_variance_loss(
            ansatz=ansatz,
            optimizer_type=optimizer_type,
            use_custom_jvp=True,
            max_vmap_batch_size=max_vmap_batch_size,
            mesh=mesh
        )
    else:
        loss_fn = cost_fn

    # Create MCMC step function
    mcmc_step = make_mcmc_step(ref_det, step_size, move_type,
                               max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)

    # Create optimizer and training step
    # Define loss function JVP for KFAC and Newton
    loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    if optimizer_type.lower() == "newton":
        # Newton setup
        opt_kwargs["value_and_grad_func"] = loss_fn_jvp
        opt_kwargs["curvature"] = "gauss_newton" # Variance minimization uses GN
        opt_kwargs["max_vmap_batch_size"] = max_vmap_batch_size
        
        # Add jacobian_sample_size if provided
        if jacobian_sample_size is not None:
            opt_kwargs["jacobian_sample_size"] = jacobian_sample_size
        
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        
        key, subkey = random.split(key)
        opt_state = optimizer.init(params, subkey, (walkers, ansatz))
        
        training_step = make_second_order_training_step(
            mcmc_step,
            optimizer,
            n_mcmc_per_opt=n_mcmc_per_opt,
            n_opt_per_mcmc=n_opt_per_mcmc,
        )
        # Newton needs explicit JIT
        training_step = jax.jit(training_step)
    else:
        # Optax setup
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        opt_state = optimizer.init(params)
        
        # Create Optax training step with configurable cadence.
        opt_update_step = make_opt_update_step(loss_fn, optimizer)
        training_step = make_training_step(
            mcmc_step,
            opt_update_step,
            n_mcmc_per_opt=n_mcmc_per_opt,
            n_opt_per_mcmc=n_opt_per_mcmc,
        )

    # ========== MAIN LOOP (Uses JIT-compiled step) ==========
    logger.info(f"Starting optimization with {n_opt_steps} steps...")
    
    losses = []
    energies = []
    stds = []
    acceptances = []
    params_history = []
    
    start_time = time.time()
    
    # Run first step separately to measure compilation time
    logger.info("Compiling training step...")
    compilation_start = time.time()
    
    key, subkey = random.split(key)
    if optimizer_type.lower() in ["newton"]:
        walkers, params, opt_state, loss, aux_data, pmove, current_lr = training_step(
            ansatz, walkers, params, opt_state, subkey, 0
        )
    else:
        walkers, params, opt_state, loss, aux_data, pmove = training_step(
            ansatz, walkers, params, opt_state, subkey
        )
        current_lr = None
    compilation_end = time.time()
    logger.info(f"Compilation + First Step finished in {compilation_end - compilation_start:.2f}s")
    
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
    
    lr_str = f" | LR: {current_lr:.4f}" if current_lr is not None else ""
    logger.info(f"Step     0 | Var: {variance_val:.6f} | "
          f"E: {energy_val:.6f}±{std_val:.6f} | "
          f"Accept: {pmove_val:.3f}{lr_str} | Time: {compilation_end - start_time:.2f}s")

    for opt_step in range(1, n_opt_steps):
        key, subkey = random.split(key)
        
        if optimizer_type.lower() in ["newton"]:
            walkers, params, opt_state, loss, aux_data, pmove, current_lr = training_step(
                ansatz, walkers, params, opt_state, subkey, opt_step
            )
        else:
            walkers, params, opt_state, loss, aux_data, pmove = training_step(
                ansatz, walkers, params, opt_state, subkey
            )
            current_lr = None
        
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
            lr_str = f" | LR: {current_lr:.4f}" if current_lr is not None else ""
            elapsed = time.time() - start_time
            logger.info(f"Step {opt_step:5d} | Var: {variance_val:.6f} | "
                  f"E: {energy_val:.6f}±{std_val:.6f} | "
                  f"Accept: {pmove_val:.3f}{lr_str} | Time: {elapsed:.2f}s")
            start_time = time.time()

        # Periodic save to disk
        if save_path and (opt_step + 1) % save_frequency == 0:
            current_history = {
                "cost": np.array(losses),
                "energies": np.array(energies),
                "stds": np.array(stds),
                "acceptance": np.array(acceptances),
                "params": params_history
            }
            save_optimization_history(current_history, save_path)
            logger.info(f"Saved intermediate optimization history to {save_path}")
    
    logger.info("Optimization complete!")


    return {
        "cost": np.array(losses),
        "energies": np.array(energies),
        "stds": np.array(stds),
        "acceptance": np.array(acceptances),
        "params": params_history
    }


# =============================================================================
# State-averaged variance optimization (multi-state, shared Jastrow)
# =============================================================================


def optimize_ref_var_multistate(
    ansatze,
    linear_coeffs_list,
    jastrow_params=None,
    weights=None,
    n_walkers: int = 30000,
    n_mcmc_per_opt: int = 20,
    step_size: float = 0.3,
    burn_in_steps: int = 2000,
    n_opt_steps: int = 1000,
    optimizer_type: str = "adam",
    learning_rate: float = 0.01,
    max_vmap_batch_size: int = 0,
    move_type: str = "all",
    opt_kwargs: Optional[Dict[str, Any]] = None,
    key=None,
    save_frequency: int = 100,
    save_path: Optional[str] = None,
    clip_multiplier: float = 5.0,
    grad_clip_norm: Optional[float] = 1.0,
):
    """State-averaged variance optimization of a SHARED Jastrow.

    Minimizes ``L[J] = sum_n w_n Var_{Psi_n}[E_L]``, where each
    ``Psi_n = J * Phi_n`` shares the same Jastrow ``J`` and Phi_n is a
    FIXED linear combination of Slater determinants (e.g. a singlet-CIS
    CSF whose coefficients are constrained by spin symmetry). Only the
    Jastrow is optimized; per-state linear coefficients are constants.

    The reduced parameter space (a single ``jastrow_params`` PyTree, no
    nested per-state coefficient list) means this routine is a drop-in
    replacement for single-state ``optimize_ref_var`` modulo the
    multi-walker bookkeeping.

    Each state n is sampled from its own walker population, distributed
    as ``|Phi_n|^2 = |sum_i c_i^{(n)} D_i^{(n)}|^2`` (multi-det, no
    Jastrow). At every optimization step we run ``n_mcmc_per_opt`` MCMC
    sweeps per state, then compute the combined loss + gradients on the
    JAX side and apply one optimizer update.

    Args:
        ansatze: list of ``SlaterJastrow``. All must share the same Jastrow
            (Python identity).
        linear_coeffs_list: list of arrays giving the fixed linear
            coefficients for each state's determinant expansion. Must
            satisfy ``len(linear_coeffs_list[n]) == len(ansatze[n].dets)``.
            For a singlet-CIS CSF, this is ``[1/sqrt(2), 1/sqrt(2)]``.
        jastrow_params: optional initial Jastrow parameter tree. If None,
            uses ``ansatze[0].jastrow.init_params()``.
        weights: optional per-state weights (default uniform).
        n_walkers: TOTAL walker budget; split evenly across states.
        n_mcmc_per_opt: MCMC sweeps between optimization updates (per state).
        step_size: initial MCMC step size.
        burn_in_steps: burn-in length per state.
        n_opt_steps: number of optimization steps.
        optimizer_type: Optax-style optimizer ("adam", "sgd", ...). Newton is
            NOT yet supported in this multi-state path.
        learning_rate: learning rate.
        max_vmap_batch_size: forwarded to per-state variance losses.
        move_type: "all" (default) or "one". The single-electron rank-1
            update path in ``_one_electron_move`` only updates the first
            determinant and computes the proposal psi from ``c_0 * D_0``
            instead of ``sum_i c_i D_i`` — so it is INCORRECT for any
            multi-determinant reference (e.g. CIS singlet CSFs). The
            default ``"all"`` move type goes through the full multi-det
            ansatz call and is statistically correct. Only override to
            ``"one"`` if you know every ansatz is single-det (then the
            fast rank-1 path is exact).
        opt_kwargs: extra optimizer kwargs.
        key: PRNG key.
        save_frequency: history-save cadence (in opt steps).
        save_path: optional HDF5 path for periodic history dumps.
        clip_multiplier: per-state energy clipping (passed to variance loss).
        grad_clip_norm: global-norm clip for gradients before the optimizer
            step. Set to ``None`` to disable. Recommended ~1.0 for Adam to
            tame occasional spikes from outlier walkers.

    Returns:
        Dictionary with arrays:
            cost: combined loss per opt step
            energies: shape (n_steps, n_states) — per-state mean E_L
            stds: shape (n_steps, n_states) — per-state std of E_L
            acceptance: shape (n_steps, n_states) — per-state pmove
            params: list of jastrow_params snapshots (PyTree, not the
                    nested list — linear_coeffs_list is fixed)
    """
    n_states = len(ansatze)
    if n_states < 2:
        raise ValueError(
            f"optimize_ref_var_multistate needs >=2 ansatze; got {n_states}. "
            "Use optimize_ref_var for the single-state case."
        )

    # ---- Validate inputs ----
    j0 = ansatze[0].jastrow
    for k, a in enumerate(ansatze[1:], start=1):
        if a.jastrow is not j0:
            raise ValueError(
                f"ansatze[{k}].jastrow is a different Python object from "
                f"ansatze[0].jastrow. The state-averaged optimizer requires "
                f"all ansatze to share the SAME Jastrow instance so that "
                f"jastrow_params is meaningful as a shared parameter."
            )
    if len(linear_coeffs_list) != n_states:
        raise ValueError(
            f"linear_coeffs_list has length {len(linear_coeffs_list)}, "
            f"but n_states = {n_states}."
        )
    for n, (a, lc) in enumerate(zip(ansatze, linear_coeffs_list)):
        if len(lc) != len(a.dets):
            raise ValueError(
                f"state {n}: linear_coeffs_list has length {len(lc)} but "
                f"ansatze[{n}].dets has length {len(a.dets)}."
            )

    if key is None:
        key = random.PRNGKey(int(time.time()))
    if opt_kwargs is None:
        opt_kwargs = {}

    use_newton = optimizer_type.lower() == "newton"
    newton_damping = float(opt_kwargs.get("damping", 1e-3)) if use_newton else None

    # ---- Walker budget per state ----
    n_walkers_per_state = n_walkers // n_states
    if n_walkers_per_state * n_states != n_walkers:
        logger.info(
            f"n_walkers={n_walkers} not divisible by n_states={n_states}; "
            f"using {n_walkers_per_state} walkers per state "
            f"(total {n_walkers_per_state * n_states})."
        )

    # ---- Default Jastrow params ----
    if jastrow_params is None:
        jastrow_params = j0.init_params()

    # ---- Convert linear_coeffs_list to jnp constants ----
    linear_coeffs_list = [jnp.asarray(lc) for lc in linear_coeffs_list]

    # ---- Build per-state multi-det references (sampling distribution) ----
    # For multi-det CSFs (e.g. singlet CIS), sampling from |D_0|^2 alone
    # misses the inter-determinant interference and uses the wrong nodal
    # surface. The correct reference distribution is the multi-det
    # |sum_i c_i D_i|^2 — implemented by MultiSlaterRef, which ignores the
    # Jastrow params. For single-det ground states this reduces to |D_0|^2.
    from pytc.ansatz.sj import MultiSlaterRef
    refs = [MultiSlaterRef.from_dets(ans.dets) for ans in ansatze]

    # ---- Initialize walker populations (one per state) ----
    logger.info(
        f"Initializing {n_states} walker populations × "
        f"{n_walkers_per_state} walkers each."
    )
    walkers_list = []
    step_sizes = []
    for n, ans in enumerate(ansatze):
        key, subkey = random.split(key)
        ref_n = refs[n]
        w_n = initialize_walkers(ref_n, n_walkers_per_state, None, subkey)
        # Burn-in: MultiSlaterRef.__call__ takes [jastrow_params, linear_coeffs]
        # in its `params` argument and reads ONLY the linear coeffs (Jastrow
        # is ignored by the reference). We supply jastrow_params here just to
        # match the expected PyTree signature.
        params_n = [jastrow_params, linear_coeffs_list[n]]
        logger.info(f"  burn-in state {n} ({burn_in_steps} steps)...")
        w_n, _, key, sz_n = burn_in(
            ref_n, w_n, burn_in_steps, step_size, key,
            params=params_n,
            move_type=move_type,
            max_vmap_batch_size=max_vmap_batch_size,
            mesh=None,
        )
        walkers_list.append(w_n)
        step_sizes.append(sz_n)

    logger.info(
        "Burn-in done. Step sizes per state: "
        + ", ".join(f"{float(s):.3f}" for s in step_sizes)
    )

    # ---- Build per-state MCMC steps ----
    # Captured ansatz = MultiSlaterRef (multi-det, no Jastrow).
    mcmc_steps = []
    for n, ans in enumerate(ansatze):
        mcmc_n = make_mcmc_step(
            refs[n], step_sizes[n], move_type,
            max_vmap_batch_size=max_vmap_batch_size, mesh=None,
        )
        mcmc_steps.append(mcmc_n)

    # ---- Build combined loss + gradient ----
    # Loss signature: (jastrow_params, walkers_list) -> (scalar, aux)
    loss_fn = make_state_averaged_variance_loss(
        ansatze=ansatze,
        linear_coeffs_list=linear_coeffs_list,
        weights=weights,
        optimizer_type=optimizer_type,
        use_custom_jvp=True,
        max_vmap_batch_size=max_vmap_batch_size,
        clip_multiplier=clip_multiplier,
    )
    loss_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    if use_newton:
        # ---- Multistate Gauss-Newton step ----
        # Combined curvature G = sum_n w_n * G_n and gradient g = sum_n w_n
        # * g_n on the same parameter space (jastrow_params). One solve per
        # opt step. Per-state Jacobian / local-energy computations are JITted
        # in build_per_state_gn().
        if weights is None:
            w_arr = np.ones(n_states) / n_states
        else:
            w_arr_raw = np.asarray(weights, dtype=np.float64)
            w_arr = w_arr_raw / w_arr_raw.sum()

        # Pre-flatten the Jastrow PyTree to get the unravel_fn (static).
        _, unravel_fn = jax.flatten_util.ravel_pytree(jastrow_params)
        n_params_flat = jax.flatten_util.ravel_pytree(jastrow_params)[0].shape[0]

        def make_per_state_gn(ansatz_n, linear_coeffs_n):
            """Returns a JIT'd function computing (G_n, g_n, e_mean_n, e_std_n)
            for state n on its walker batch.

            Math (per state):
                E_i  = E_L(w_i; J, c_n)
                J_ip = d E_L(w_i; J, c_n) / d J_p              (Jacobian)
                E_clipped = clip(E_i, mean ± clip_mul * MAD)
                E_diff = E_clipped - mean(E_clipped)
                J_centered = J - mean_walker(J)
                g_n = 2/(N-1) * J^T @ E_diff
                G_n = 2/N * J_centered^T @ J_centered
            """
            def single_le_and_grad(w, jp):
                params_n = [jp, linear_coeffs_n]
                def le_of_j(j_inner):
                    p_inner = [j_inner, linear_coeffs_n]
                    return ansatz_n.local_energy(w, p_inner)[0]
                return jax.value_and_grad(le_of_j)(jp)

            def per_state(jp, walkers_n):
                vmap_fn = jax.vmap(single_le_and_grad, in_axes=(0, None))
                e_vec, j_tree = vmap_fn(walkers_n, jp)
                # Flatten Jacobian into (N, P)
                j_flat_leaves, _ = jax.tree_util.tree_flatten(j_tree)
                jac_mat = jnp.concatenate(
                    [jnp.reshape(leaf, (e_vec.shape[0], -1))
                     for leaf in j_flat_leaves],
                    axis=1,
                )
                # Clip energies (same MAD-based scheme as single-state)
                if clip_multiplier > 0:
                    e_mean_raw = jnp.mean(e_vec)
                    e_mad = jnp.mean(jnp.abs(e_vec - e_mean_raw))
                    e_lo = e_mean_raw - clip_multiplier * e_mad
                    e_hi = e_mean_raw + clip_multiplier * e_mad
                    e_vec = jnp.clip(e_vec, e_lo, e_hi)
                n_w = e_vec.shape[0]
                e_mean = jnp.mean(e_vec)
                e_std = jnp.std(e_vec)
                e_diff = e_vec - e_mean
                variance_n = jnp.sum(e_diff * e_diff) / (n_w - 1)
                g_n_vec = (2.0 / (n_w - 1)) * (jac_mat.T @ e_diff)
                jac_centered = jac_mat - jnp.mean(jac_mat, axis=0, keepdims=True)
                G_n_mat = (2.0 / n_w) * (jac_centered.T @ jac_centered)
                return variance_n, g_n_vec, G_n_mat, e_mean, e_std

            return jax.jit(per_state)

        per_state_gn_fns = [
            make_per_state_gn(ansatze[n], linear_coeffs_list[n])
            for n in range(n_states)
        ]

        def newton_opt_step(jastrow_params, walkers_list_in):
            # Per-state contributions
            total_loss = 0.0
            g_total = jnp.zeros(n_params_flat, dtype=jnp.float64)
            G_total = jnp.zeros(
                (n_params_flat, n_params_flat), dtype=jnp.float64
            )
            e_means = []
            e_stds = []
            for n in range(n_states):
                var_n, g_n_vec, G_n_mat, em, es = per_state_gn_fns[n](
                    jastrow_params, walkers_list_in[n]
                )
                w_n = float(w_arr[n])
                total_loss = total_loss + w_n * var_n
                g_total = g_total + w_n * g_n_vec
                G_total = G_total + w_n * G_n_mat
                e_means.append(em)
                e_stds.append(es)
            # Damped solve
            G_damped = G_total + newton_damping * jnp.eye(n_params_flat)
            delta_vec = jax.scipy.linalg.solve(
                G_damped, -g_total, assume_a="pos"
            )
            # Diagnostic: raw gradient norm BEFORE damping (no clip applied)
            grad_norm = jnp.linalg.norm(g_total)
            # Apply update
            lr = float(learning_rate)
            delta_pytree = unravel_fn(delta_vec)
            new_jastrow_params = jax.tree_util.tree_map(
                lambda p, d: p + lr * d, jastrow_params, delta_pytree
            )
            aux = (jnp.stack(e_means), jnp.stack(e_stds))
            return new_jastrow_params, total_loss, aux, grad_norm

        opt_update_step_jit = newton_opt_step  # already JIT'd internally
        opt_state = None  # unused for Newton
    else:
        # Optax path (Adam, SGD, RMSprop, ...)
        base_optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            # Compose global-norm gradient clipping BEFORE the base optimizer
            # so that catastrophic single-step spikes in the noisy variance
            # gradient cannot derail Adam (which has no inherent step-size
            # regulation, unlike Newton/Gauss-Newton).
            optimizer = optax.chain(
                optax.clip_by_global_norm(grad_clip_norm),
                base_optimizer,
            )
        else:
            optimizer = base_optimizer
        opt_state = optimizer.init(jastrow_params)

        # ---- Combined opt step (JIT'd) ----
        def opt_update_step(jastrow_params, walkers_list_in, opt_state):
            (loss, aux_data), grads = loss_and_grad(
                jastrow_params, walkers_list_in
            )
            # Diagnostic: raw gradient norm BEFORE the clip-by-global-norm
            # transformation. Actual clipping is applied inside `optimizer`.
            grad_leaves = jax.tree_util.tree_leaves(grads)
            grad_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in grad_leaves))
            updates, opt_state = optimizer.update(
                grads, opt_state, jastrow_params
            )
            new_jastrow_params = optax.apply_updates(jastrow_params, updates)
            return new_jastrow_params, opt_state, loss, aux_data, grad_norm

        opt_update_step_jit = jax.jit(opt_update_step)

    # ---- Per-state MCMC sweep (JIT'd separately per state) ----
    # The mcmc_step has the MultiSlaterRef captured at factory time and
    # ignores its runtime ansatz argument, but pass `refs[n]` for clarity.
    def make_mcmc_sweep(mcmc_step_n, ref_n, n_sweeps):
        def sweep(walkers, key_in, params_n):
            def scan_body(carry, _):
                w_c, k_c = carry
                k_c, sk = random.split(k_c)
                w_c, pmv = mcmc_step_n(ref_n, w_c, sk, params_n)
                return (w_c, k_c), pmv
            (walkers, key_in), pmoves = jax.lax.scan(
                scan_body, (walkers, key_in), None, length=n_sweeps
            )
            return walkers, key_in, jnp.mean(pmoves)
        return jax.jit(sweep)

    mcmc_sweeps = [
        make_mcmc_sweep(mcmc_steps[n], refs[n], n_mcmc_per_opt)
        for n in range(n_states)
    ]

    # ---- History ----
    losses = []
    energies_history = []      # list of arrays shape (n_states,)
    stds_history = []          # list of arrays shape (n_states,)
    acceptance_history = []    # list of arrays shape (n_states,)
    params_history = []

    logger.info(f"Starting state-averaged optimization "
                f"({n_states} states, lr={learning_rate}, "
                f"{n_opt_steps} steps)...")

    t_start = time.time()
    for opt_step in range(n_opt_steps):
        # MCMC: refresh each state's walkers (sampled from |Phi_n|^2 via
        # MultiSlaterRef; the linear coeffs come from linear_coeffs_list).
        pmoves_step = []
        for n in range(n_states):
            key, subkey = random.split(key)
            params_n = [jastrow_params, linear_coeffs_list[n]]
            walkers_list[n], _, pmv = mcmc_sweeps[n](
                walkers_list[n], subkey, params_n
            )
            pmoves_step.append(pmv)

        # Optimization step on combined loss (Newton vs Adam dispatch)
        if use_newton:
            jastrow_params, loss_val, aux, grad_norm = opt_update_step_jit(
                jastrow_params, walkers_list
            )
        else:
            jastrow_params, opt_state, loss_val, aux, grad_norm = opt_update_step_jit(
                jastrow_params, walkers_list, opt_state
            )

        # Record
        loss_f = float(jax.device_get(loss_val))
        e_means, e_stds = aux
        e_means_np = np.array(jax.device_get(e_means))
        e_stds_np = np.array(jax.device_get(e_stds))
        pmoves_np = np.array([float(jax.device_get(p)) for p in pmoves_step])
        gnorm_f = float(jax.device_get(grad_norm))

        losses.append(loss_f)
        energies_history.append(e_means_np)
        stds_history.append(e_stds_np)
        acceptance_history.append(pmoves_np)

        # Only the Jastrow params evolve; record those.
        jastrow_copy = tree_map(
            lambda x: np.array(jax.device_get(x)) if isinstance(x, jnp.ndarray) else x,
            jastrow_params,
        )
        params_history.append(jastrow_copy)

        # Logging cadence
        if (opt_step % max(1, n_opt_steps // 200) == 0
                or opt_step == n_opt_steps - 1):
            e_str = " ".join(f"{e:+.4f}" for e in e_means_np)
            acc_str = " ".join(f"{p:.2f}" for p in pmoves_np)
            elapsed = time.time() - t_start
            logger.info(
                f"Step {opt_step:5d} | L: {loss_f:.5f} | "
                f"E: [{e_str}] | Acc: [{acc_str}] | "
                f"|g|: {gnorm_f:.3f} | t: {elapsed:.1f}s"
            )

        # Periodic save
        if save_path and (opt_step + 1) % save_frequency == 0:
            current = {
                "cost": np.array(losses),
                "energies": np.stack(energies_history),       # (steps, n_states)
                "stds": np.stack(stds_history),
                "acceptance": np.stack(acceptance_history),
                "params": params_history,
            }
            save_optimization_history(current, save_path)
            logger.info(f"Saved multistate history to {save_path}")

    logger.info("State-averaged optimization complete.")

    return {
        "cost": np.array(losses),
        "energies": np.stack(energies_history),
        "stds": np.stack(stds_history),
        "acceptance": np.stack(acceptance_history),
        "params": params_history,            # jastrow_params snapshots (PyTree)
        "linear_coeffs": [np.array(lc) for lc in linear_coeffs_list],
        "n_states": n_states,
    }
