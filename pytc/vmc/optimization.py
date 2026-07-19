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
from .loss import make_energy_loss, make_variance_loss
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
        # Note: loss_fn expects (params, walkers), ansatz is baked in or handled via wrapper
        (loss, aux_data), grads = loss_and_grad(params, walkers)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, opt_state, loss, aux_data
    
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
            
            (walkers, key, _), pmoves = jax.lax.scan(
                mcmc_scan_fn,
                (walkers, key, params),
                None,
                length=n_mcmc_per_opt
            )
            pmove = pmoves[-1]
            
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
            
            (params, opt_state, key), (losses, aux_data_list) = jax.lax.scan(
                opt_scan_fn,
                (params, opt_state, key),
                None,
                length=n_opt_per_mcmc
            )
            
            loss = losses[-1]
            aux_data = tree_map(lambda x: x[-1], aux_data_list)
            
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
        
        # Pattern 3: Balanced (1 MCMC, 1 opt) - default simple case
        else:
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ansatz, walkers, subkey, params)
            
            key, subkey = random.split(key)
            params, opt_state, loss, aux_data = opt_update_step(
                ansatz, params, walkers, opt_state, subkey
            )
        
        
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
            
            (walkers, key, _), pmoves = jax.lax.scan(
                mcmc_scan_fn,
                (walkers, key, params),
                None,
                length=n_mcmc_per_opt
            )
            pmove = pmoves[-1]
            
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
        
    user_or_default_cost_fn = cost_fn
    if user_or_default_cost_fn is None:
        def energy_cost_fn(energies_for_cost):
            return jnp.mean(energies_for_cost)
        user_or_default_cost_fn = energy_cost_fn
    
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
    
    if multi_gpu and mesh is not None:
        walkers = initialize_walkers_sharded(
            ansatz, n_walkers, mesh, initial_walkers=initial_walkers, key=key
        )
        if params is not None:
            params = replicate(params, mesh)
        key = replicate(key, mesh)
    else:
        walkers = initialize_walkers(ansatz, n_walkers, initial_walkers, key)

    if use_importance_sampling:
        walkers, acceptance_history, key, step_size = burn_in_with_importance(
            ansatz, walkers, burn_in_steps, step_size, key, params, mesh=mesh)
    else:
        walkers, acceptance_history, key, step_size = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key, params, 
            move_type=move_type, max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)
    
    logger.info("Starting optimization...")
    
    if params is None:
        jastrow_params = ansatz.jastrow.init_params()
        linear_coeffs = jnp.ones(len(ansatz.dets))
        params = [jastrow_params, linear_coeffs]
    else:
        if not isinstance(params, (list, tuple)) or len(params) != 2:
             raise ValueError("`params` must be a list or tuple: [jastrow_params, linear_coeffs]")

    internal_loss_fn = make_energy_loss(
        ansatz=ansatz,
        optimizer_type=optimizer_type,
        cost_fn=user_or_default_cost_fn,
        clip_multiplier=5.0,
        use_custom_jvp=use_custom_jvp,
        max_vmap_batch_size=max_vmap_batch_size,
        mesh=mesh
    )

    gradient_mask = create_gradient_mask(ansatz, params, frozen_params)

    if use_importance_sampling:
        mcmc_step = make_mcmc_step_importance(ansatz, step_size, mesh=mesh)
    else:
        mcmc_step = make_mcmc_step(
            ansatz, step_size, move_type,
            max_vmap_batch_size=max_vmap_batch_size, mesh=mesh
        )

    loss_fn_jvp = jax.value_and_grad(internal_loss_fn, argnums=0, has_aux=True)

    if optimizer_type.lower() == "newton":
        opt_kwargs["value_and_grad_func"] = loss_fn_jvp
        opt_kwargs["curvature"] = "fisher" # Energy minimization uses Fisher
        opt_kwargs["max_vmap_batch_size"] = max_vmap_batch_size
        opt_kwargs["mesh"] = mesh
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
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        opt_state = optimizer.init(params)
        
        # Create Optax training step
        if gradient_mask is not None:
            # No-op passthrough wrapper; gradient masking is not applied here
            original_loss_fn = internal_loss_fn
            def masked_loss_fn(params_inner, batch_data):
                return original_loss_fn(params_inner, batch_data)
            internal_loss_fn = masked_loss_fn
        
        opt_update_step = make_opt_update_step(internal_loss_fn, optimizer)
        training_step = make_training_step(
            mcmc_step, opt_update_step, n_mcmc_per_opt=n_steps, n_opt_per_mcmc=1
        )

    logger.info(f"Starting optimization with {n_opt_steps} steps...")
    if adaptive_step_size:
        logger.info(f"Adaptive step-size enabled (target accept=0.5, adjust every {step_size_adjust_interval} steps)")
    
    losses = []
    energies = []
    stds = []
    acceptances = []
    params_history = []
    step_sizes = [step_size]
    
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
            current_lr = None
        
        cost_val = float(jax.device_get(loss))
        # aux_data is a namedtuple with (mean_energy, energy_std, clipped_energies, diff)
        # Extract only the first two for backward compatibility
        aux_data_materialized = jax.device_get(aux_data)
        energy_val = float(aux_data_materialized[0])
        std_val = float(aux_data_materialized[1])
        pmove_val = float(jax.device_get(pmove))
        
        losses.append(cost_val)
        energies.append(energy_val)
        stds.append(std_val)
        acceptances.append(pmove_val)
        
        params_copy = tree_map(
            lambda x: np.array(jax.device_get(x)) if isinstance(x, jnp.ndarray) else x,
            params
        )
        params_history.append(params_copy)
        
        if adaptive_step_size and (opt_step + 1) % step_size_adjust_interval == 0 and opt_step < 5*step_size_adjust_interval:
            recent_accept = np.mean(acceptances[-step_size_adjust_interval:])
            step_size *= recent_accept / 0.5
            step_sizes.append(step_size)
            
            if use_importance_sampling:
                mcmc_step = make_mcmc_step_importance(ansatz, step_size, mesh=mesh)
            else:
                mcmc_step = make_mcmc_step(
                    ansatz, step_size, move_type,
                    max_vmap_batch_size=max_vmap_batch_size, mesh=mesh
                )
            
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
        
        log_frequency = 1  # Log every step
        if opt_step % log_frequency == 0 or opt_step == n_opt_steps - 1:
            lr_str = f" | LR: {current_lr:.4f}" if current_lr is not None else ""
            elapsed = time.time() - start_time
            step_size_str = f" | StepSize: {step_size:.4f}" if adaptive_step_size else ""
            logger.info(f"Step {opt_step:5d} | Cost: {cost_val:.6f} | "
                  f"E: {energy_val:.6f}±{std_val:.6f} | "
                  f"Accept: {pmove_val:.3f}{step_size_str}{lr_str} | Time: {elapsed:.2f}s")
            start_time = time.time()

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
    initial_opt_state: Optional[int] = None,
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
        initial_opt_state: Newton optimizer only. If provided, seeds the
                        optimizer's internal step counter (which drives the
                        learning-rate decay schedule, see create_optimizer's
                        `schedule_lr`) at this value instead of 0 -- lets a
                        warm-started run (`params` loaded from a prior run's
                        history) continue that run's LR decay instead of
                        silently restarting it at full `learning_rate` at
                        each wall-clock chunk boundary.

    Returns:
        Dictionary with optimization results and statistics. Includes
        "final_opt_state" (int, Newton only) -- the optimizer's step counter
        after the last update, for chaining into a subsequent warm-started
        run's `initial_opt_state`.
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))

    if initial_opt_state is not None and optimizer_type.lower() != "newton":
        raise ValueError(
            "`initial_opt_state` is only supported for optimizer_type='newton' "
            f"(got {optimizer_type!r}) -- it seeds NewtonOptimizer's internal "
            "step counter, which other optimizer types don't expose this way."
        )

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

    logger.info("Performing burn-in...")
    walkers, acceptance_history, key, step_size = burn_in(
        ref_det, walkers, burn_in_steps, step_size, key, params=params, 
        move_type=move_type, max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)
    logger.info(f"Burn-in complete. Final step size: {step_size:.4f}")

    if cost_fn is None:
        loss_fn = make_variance_loss(
            ansatz=ansatz,
            optimizer_type=optimizer_type,
            use_custom_jvp=True,
            max_vmap_batch_size=max_vmap_batch_size,
            mesh=mesh
        )
    else:
        loss_fn = cost_fn

    mcmc_step = make_mcmc_step(ref_det, step_size, move_type,
                               max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)

    loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    if optimizer_type.lower() == "newton":
        opt_kwargs["value_and_grad_func"] = loss_fn_jvp
        opt_kwargs["curvature"] = "gauss_newton" # Variance minimization uses GN
        opt_kwargs["max_vmap_batch_size"] = max_vmap_batch_size
        opt_kwargs["mesh"] = mesh

        if jacobian_sample_size is not None:
            opt_kwargs["jacobian_sample_size"] = jacobian_sample_size
        
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)

        key, subkey = random.split(key)
        if initial_opt_state is not None:
            opt_state = jnp.array(initial_opt_state, dtype=jnp.int32)
        else:
            opt_state = optimizer.init(params, subkey, (walkers, ansatz))

        training_step = make_second_order_training_step(
            mcmc_step,
            optimizer,
            n_mcmc_per_opt=n_mcmc_per_opt,
            n_opt_per_mcmc=n_opt_per_mcmc,
        )
        training_step = jax.jit(training_step)
    else:
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        opt_state = optimizer.init(params)
        
        opt_update_step = make_opt_update_step(loss_fn, optimizer)
        training_step = make_training_step(
            mcmc_step,
            opt_update_step,
            n_mcmc_per_opt=n_mcmc_per_opt,
            n_opt_per_mcmc=n_opt_per_mcmc,
        )

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

        
        log_frequency = 1
        if opt_step % log_frequency == 0 or opt_step == n_opt_steps - 1:
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
            elapsed = time.time() - start_time
            logger.info(f"Step {opt_step:5d} | Var: {variance_val:.6f} | "
                  f"E: {energy_val:.6f}±{std_val:.6f} | "
                  f"Accept: {pmove_val:.3f}{lr_str} | Time: {elapsed:.2f}s")
            start_time = time.time()

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
    
    
    final_opt_state = (
        int(jax.device_get(opt_state)) if optimizer_type.lower() == "newton" else None
    )

    return {
        "cost": np.array(losses),
        "energies": np.array(energies),
        "stds": np.array(stds),
        "acceptance": np.array(acceptances),
        "params": params_history,
        "final_opt_state": final_opt_state,
        "final_walkers": jax.device_get(walkers),
    }


def evaluate_ref_var(
    ansatz,
    params,
    n_walkers: int = 100,
    step_size: float = 1.0,
    burn_in_steps: int = 1000,
    initial_walkers=None,
    key=None,
    move_type: str = "one",
    max_vmap_batch_size: int = 0,
    n_eval_batches: int = 1,
    n_mcmc_per_eval: int = 1,
    clip_multiplier: float = 5.0,
):
    """Evaluate reference-variance / local-energy statistics at FIXED params.

    Unlike ``optimize_ref_var``, this never updates ``params`` -- it exists
    for controlled experiments where the params must be held
    bit-identical across runs to isolate a single variable (walker count,
    burn-in, warm-start convention). It also surfaces the raw local-energy
    tail (clipped fraction, max|E_L|) that the production loss function
    discards after clipping, since that tail is the object of the W-scaling
    hypothesis under test.

    Args:
        ansatz: Wavefunction object (SlaterJastrow).
        params: Frozen [jastrow_params, linear_coeffs] -- never updated.
        n_walkers: Number of parallel walkers.
        step_size: MCMC proposal std dev.
        burn_in_steps: Burn-in steps before the first eval batch. Pass 0
                       when ``initial_walkers`` is an already-equilibrated
                       checkpoint (the fresh-vs-continued-walkers experiment).
        initial_walkers: Optional Walker state (e.g. from
                       ``mcmc_utils.load_walkers``) or raw positions.
        key: PRNG key.
        move_type: "one" or "all" for MCMC electron moves.
        max_vmap_batch_size: If >0, use folx.batched_vmap for memory efficiency.
        n_eval_batches: Number of independent stat batches to record.
        n_mcmc_per_eval: MCMC steps to decorrelate walkers between batches.
        clip_multiplier: Same clipping window as the production loss
                       (mean +/- multiplier * MAD); only used to report
                       clipped_fraction/variance, never to modify walkers.

    Returns:
        Dictionary with:
            "batches": list of per-batch dicts (cost, mean_energy,
                energy_mad, clipped_fraction, max_abs_local_energy,
                acceptance).
            "final_walkers": Walker state after the last batch, host-local
                (pass to ``mcmc_utils.save_walkers`` to checkpoint).
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))

    if not isinstance(params, (list, tuple)) or len(params) != 2:
        raise ValueError("`params` must be a list or tuple: [jastrow_params, linear_coeffs]")

    from .sharding import (
        create_mesh, replicate, initialize_walkers_sharded,
        pad_n_walkers, n_devices as get_n_devices,
        is_multi_gpu as check_multi_gpu, get_vmap_fn,
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
        params = replicate(params, mesh)

    ref_det = ansatz.dets[0]
    if multi_gpu and mesh is not None:
        walkers = initialize_walkers_sharded(
            ref_det, n_walkers, mesh, initial_walkers=initial_walkers, key=key
        )
    else:
        walkers = initialize_walkers(ref_det, n_walkers, initial_walkers, key)

    if burn_in_steps > 0:
        logger.info("Performing burn-in...")
        walkers, _, key, step_size = burn_in(
            ref_det, walkers, burn_in_steps, step_size, key, params=params,
            move_type=move_type, max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)
        logger.info(f"Burn-in complete. Final step size: {step_size:.4f}")
    else:
        logger.info("burn_in_steps=0: using walkers as-provided (continued-walkers mode).")

    mcmc_step = make_mcmc_step(ref_det, step_size, move_type,
                                max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)

    vmap_impl = get_vmap_fn(max_vmap_batch_size, mesh)
    batch_local_energy = jax.jit(vmap_impl(
        lambda w, p: ansatz.local_energy(w, p)[0],
        in_axes=(0, None),
        out_axes=0,
    ))

    batches = []
    for b in range(n_eval_batches):
        pmove_val = None
        for _ in range(n_mcmc_per_eval):
            key, subkey = random.split(key)
            walkers, pmove = mcmc_step(ref_det, walkers, subkey, params)
            pmove_val = float(jax.device_get(pmove))

        energies = np.asarray(jax.device_get(batch_local_energy(walkers, params))).reshape(-1)
        n = energies.shape[0]
        e_mean = float(np.mean(energies))
        e_mad = float(np.mean(np.abs(energies - e_mean)))

        if clip_multiplier > 0 and e_mad > 0:
            lo, hi = e_mean - clip_multiplier * e_mad, e_mean + clip_multiplier * e_mad
            clipped_fraction = float(np.mean((energies < lo) | (energies > hi)))
            clipped_energies = np.clip(energies, lo, hi)
            clipped_mean = float(np.mean(clipped_energies))
            variance = float(np.sum((clipped_energies - clipped_mean) ** 2) / (n - 1)) if n > 1 else 0.0
        else:
            clipped_fraction = 0.0
            variance = float(np.sum((energies - e_mean) ** 2) / (n - 1)) if n > 1 else 0.0

        batch_stats = {
            "cost": variance,
            "mean_energy": e_mean,
            "energy_mad": e_mad,
            "clipped_fraction": clipped_fraction,
            "max_abs_local_energy": float(np.max(np.abs(energies))),
            "acceptance": pmove_val,
        }
        batches.append(batch_stats)
        accept_str = f"{pmove_val:.3f}" if pmove_val is not None else "n/a"
        logger.info(
            f"Eval batch {b:3d} | Var: {variance:.6f} | E: {e_mean:.6f} | "
            f"clipped_frac: {clipped_fraction:.4f} | max|E_L|: {batch_stats['max_abs_local_energy']:.4f} | "
            f"Accept: {accept_str}"
        )

    return {
        "batches": batches,
        "final_walkers": jax.device_get(walkers),
        "n_walkers": n_walkers,
    }
