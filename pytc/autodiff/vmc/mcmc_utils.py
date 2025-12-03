"""Utility functions for analyzing quantum Monte Carlo samples."""

import numpy as np
import jax.numpy as jnp
import optax
import kfac_jax
from jax import random
from jax.lax import stop_gradient
from typing import Dict, Any, List, Optional, Tuple
from jax import tree_util
import jax
import jax.scipy.sparse.linalg as spla 

def analyze_energies(sampling_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze energy convergence and statistics from sampling results.
    
    Args:
        sampling_results: Dictionary returned by metropolis_hastings
        
    Returns:
        Dictionary with energy statistics
    """
    energies = sampling_results["energies"]
    
    # Flatten the energies if they have walker dimension
    if len(energies.shape) > 1:
        flat_energies = energies.reshape(-1)
    else:
        flat_energies = energies
    
    # Calculate statistics
    energy_mean = jnp.mean(flat_energies)
    energy_error = jnp.std(flat_energies) / jnp.sqrt(len(flat_energies))
    energy_variance = jnp.var(flat_energies)
    
    # Calculate moving average
    window_size = 10
    cumsum = jnp.cumsum(jnp.insert(flat_energies, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    
    # Calculate autocorrelation
    n = len(flat_energies)
    mean = jnp.mean(flat_energies)
    var = jnp.var(flat_energies)
    
    # Simple autocorrelation calculation for lag-1
    autocorr_1 = jnp.sum((flat_energies[:-1] - mean) * (flat_energies[1:] - mean)) / ((n-1) * var)
    
    stats = {
        "mean": energy_mean,
        "error": energy_error,
        "variance": energy_variance,
        "autocorr_lag1": autocorr_1,
    }
    
    return stats



def create_optimizer(optimizer_type, learning_rate, opt_kwargs=None):
    """Create an optimizer based on specified type and parameters."""
    if opt_kwargs is None:
        opt_kwargs = {}
    
    base_kwargs = {}
    merged_kwargs = {**base_kwargs, **opt_kwargs}
    def schedule_lr(step):
        return learning_rate / (1.0 + step/100)
    
    if optimizer_type.lower() == "kfac":
        # K-FAC requires a value_and_grad_func, which should be provided in opt_kwargs
        if "value_and_grad_func" not in merged_kwargs:
            raise ValueError("KFAC optimizer requires value_and_grad_func in opt_kwargs")
        
        return kfac_jax.Optimizer(
            value_and_grad_func=merged_kwargs["value_and_grad_func"],
            l2_reg=merged_kwargs.get("l2_reg", 0.0),
            value_func_has_aux=merged_kwargs.get("value_func_has_aux", False),
            value_func_has_state=merged_kwargs.get("value_func_has_state", False),
            value_func_has_rng=merged_kwargs.get("value_func_has_rng", False),
            learning_rate_schedule=merged_kwargs.get("learning_rate_schedule", None),
            use_adaptive_learning_rate=merged_kwargs.get("use_adaptive_learning_rate", True),
            use_adaptive_momentum=merged_kwargs.get("use_adaptive_momentum", True),
            use_adaptive_damping=merged_kwargs.get("use_adaptive_damping", True),
            initial_damping=merged_kwargs.get("initial_damping", 1.0),
            num_burnin_steps=merged_kwargs.get("num_burnin_steps", 0),  # Set to 0 by default to avoid requiring data_iterator
            multi_device=merged_kwargs.get("multi_device", False),
            norm_constraint=merged_kwargs.get("norm_constraint", None),
        )
    elif optimizer_type.lower() == "adam":
        #return optax.adamw(learning_rate=schedule_lr)
        return optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "sgd":
        return optax.chain(
            optax.sgd(learning_rate=learning_rate),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "rmsprop":
        return optax.chain(
            optax.scale_by_rms(decay=merged_kwargs.get("decay", 0.9), eps=merged_kwargs.get("eps", 1e-8)),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "lion":
        return optax.lion(learning_rate=learning_rate, b1=merged_kwargs.get("b1", 0.9), b2=merged_kwargs.get("b2", 0.99))
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_gradient_mask(ansatz, params, frozen_params):
    """Create a gradient mask PyTree for the combined params structure.
    
    Assumes params = [jastrow_params, linear_coeffs]. The mask is applied
    only to the jastrow_params part based on frozen_params identifiers.
    The linear_coeffs part of the mask is always True (not frozen).

    Args:
        ansatz: The wavefunction ansatz object.
        params: The combined parameters PyTree [jastrow_params, linear_coeffs].
        frozen_params: A list of identifiers (int index or str name/type)
                       for Jastrow factors whose parameters should be frozen.

    Returns:
        A PyTree with the same structure as params, where frozen parameters
        are wrapped with `jax.lax.stop_gradient`.
    """
    if not frozen_params:
        return params  # No freezing requested, return params unchanged

    if not isinstance(params, (list, tuple)) or len(params) != 2:
        raise ValueError("`params` must be a list or tuple: [jastrow_params, linear_coeffs]")

    jastrow_params = params[0]
    linear_coeffs = params[1]

    print(f"Creating gradient mask for frozen Jastrow parameters: {frozen_params}")
    jastrows = ansatz.jastrow.jastrows
    if not isinstance(jastrow_params, (list, tuple)) or len(jastrow_params) != len(jastrows):
        raise TypeError(f"Jastrow params structure (length {len(jastrow_params)}) does not match jastrows (length {len(jastrows)})")

    # Deep copy the parameters to avoid modifying the input
    masked_jastrow_params = []
    for i, (param_pytree, jastrow) in enumerate(zip(jastrow_params, jastrows)):
        should_freeze = False
        for fp in frozen_params:
            if isinstance(fp, int) and fp == i:
                should_freeze = True
                break
            elif isinstance(fp, str):
                if fp == jastrow.__class__.__name__ or fp == getattr(jastrow, 'name', None):
                    should_freeze = True
                    break
        
        if should_freeze:
            # Apply stop_gradient to all leaves in the frozen parameter PyTree
            param_pytree = tree_map(stop_gradient, param_pytree)
            print(f"  Freezing Jastrow {i}: type={jastrow.__class__.__name__}, name={getattr(jastrow, 'name', None)}")
            print("Warning: Freezing parameters does not work for KFAC yet.")
        masked_jastrow_params.append(param_pytree)

    # Return the masked parameters
    return [masked_jastrow_params, linear_coeffs]

def apply_gradient_mask(grads, mask):
    """Apply gradient mask to gradients to freeze parameters.
    
    Args:
        grads: The gradient PyTree
        mask: The mask PyTree created by create_gradient_mask
        
    Returns:
        A PyTree with the same structure as grads, where gradients for frozen
        parameters are set to zero.
    """
    if mask is None:
        return grads
        
    def _apply_mask(g, m):
        # If m is already stop_gradient'd, zero out the gradient
        if isinstance(m, type(stop_gradient(m))):
            return jnp.zeros_like(g)
        return g
        
    return tree_map(_apply_mask, grads, mask)