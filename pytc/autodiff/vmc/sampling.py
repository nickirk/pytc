"""Sampling procedures for VMC simulation.

This module contains functions for burn-in procedures and main sampling loops,
including both standard MCMC and importance sampling variants.
"""

import gc
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any

from .metropolis import metropolis_hastings, metropolis_hastings_importance_sampling
from .walker import initialize_walkers
from .mcmc_utils import prepare_sampling_results, report_progress
from functools import partial
import folx


def burn_in(ansatz, 
            walkers, 
            n_steps=2000, 
            step_size=0.01,
            key=None, 
            params=None, 
            report_interval=100, 
            move_type="one",
            max_vmap_batch_size=0):
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
    
    # Warm up walker cache (populate log_psi, psi_sign) so that
    # _one_electron_move can reuse cached values instead of recomputing.
    if max_vmap_batch_size > 0:
        _warmup_ansatz = folx.batched_vmap(
            lambda w, p: ansatz(w, p),
            in_axes=(0, None),
            max_batch_size=max_vmap_batch_size
        )
    else:
        _warmup_ansatz = jax.vmap(lambda w, p: ansatz(w, p), in_axes=(0, None))
    _, walkers = _warmup_ansatz(walkers, params)

    # JIT-compile the MCMC step function to speed up the loop
    # We partial out move_type since it's a static string argument
    # step_size is passed as argument so it can vary without recompilation
    mcmc_step = jax.jit(
        partial(metropolis_hastings, move_type=move_type, batch_ansatz=folx.batched_vmap(
            lambda w, p: ansatz(w, p), 
            in_axes=(0, None), 
            max_batch_size=max_vmap_batch_size
        ) if max_vmap_batch_size > 0 else None)
    )
    
    start_time = time.time()
    for step in range(n_steps):
        key, subkey = random.split(key)
        walkers, acceptance = mcmc_step(
            ansatz, walkers, step_size, subkey, params)
        
        # Convert acceptance to Python float for history
        acceptance_float = float(acceptance)
        acceptance_history.append(acceptance_float)
        
        if step % report_interval == 0:
            print(f"Burn-in step {step}/{n_steps}, acceptance: {acceptance_float:.3f}, time: {time.time() - start_time:.2f}s")
            step_size *= acceptance_float / 0.5
            start_time = time.time()
            # Periodic garbage collection
            gc.collect()
    
    print("Burn-in complete.")
    return walkers, acceptance_history, key, step_size


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
    
    # JIT-compile the MCMC step
    mcmc_step = jax.jit(metropolis_hastings_importance_sampling)
    
    time_start = time.time()
    for step in range(n_steps):
        key, subkey = random.split(key)
        walkers, acceptance = mcmc_step(
            ansatz, walkers, time_step, subkey, params)
        acceptance_history.append(acceptance)
        
        if step % report_interval == 0:
            print(f"Burn-in step {step}/{n_steps}, acceptance: {acceptance_history[-1]}, time: {time.time() - time_start:.2f}s")
            time_step *= acceptance_history[-1]/0.5
            time_start = time.time()
    
    print("Burn-in complete.")
    return walkers, acceptance_history, key, time_step


def sample(
    ansatz, 
    n_walkers: int = 100, 
    n_steps: int = 1000, 
    step_size: float = 1.0,
    thinning: int = 10,
    burn_in_steps: int = 1000,
    initial_walkers=None,
    use_importance_sampling: bool = False,
    params=None,
    key=None,
    move_type: str = "one",
    report_interval: int = 100,
    max_vmap_batch_size: int = 0
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
    
    print("Starting production sampling...")
    print(f"Burn-in steps = {burn_in_steps}")
    print(f"Number of walkers = {n_walkers}")
    print(f"Number of steps = {n_steps}")
    print(f"Thinning factor = {thinning}")
    print(f"Step size = {step_size:.4f}")
    print(f"Using importance sampling: {use_importance_sampling}")
    print(f"Move type: {move_type}")
    
    # Perform burn-in with appropriate method
    if use_importance_sampling:
        walkers, acceptance_history, key, step_size = burn_in_with_importance(
            ansatz, walkers, burn_in_steps, step_size, key, params)
    else:
        walkers, acceptance_history, key, step_size = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key=key, params=params, 
            move_type=move_type, max_vmap_batch_size=max_vmap_batch_size)
    
    # Storage for collected samples
    collected_samples = []
    collected_energies = []
    step_times = []
    
    # JIT-compile MCMC step for production run
    if use_importance_sampling:
        mcmc_step = jax.jit(metropolis_hastings_importance_sampling)
    else:
        mcmc_step = jax.jit(
            partial(metropolis_hastings, move_type=move_type, batch_ansatz=folx.batched_vmap(
            lambda w, p: ansatz(w, p), 
            in_axes=(0, None), 
            max_batch_size=max_vmap_batch_size
        ) if max_vmap_batch_size > 0 else None)
        )
        
    # JIT-compile energy evaluation
    batch_local_energy = jax.jit(jax.vmap(
        lambda w, p: ansatz.local_energy(w, p)[0],
        in_axes=(0, None)
    ))
    
    # Main sampling loop
    start_time = time.time()
    for step in range(n_steps):
        
        key, subkey = random.split(key)
        walkers, acceptance = mcmc_step(
            ansatz, walkers, step_size, subkey, params)
            
        acceptance_history.append(acceptance)
        
        if step % thinning == 0:
            # Compute local energies with parameters
            energies = batch_local_energy(walkers, params)
            
            # Convert to numpy to avoid holding JAX device references
            collected_samples.append(np.array(walkers.positions))
            collected_energies.append(np.array(energies))
        
        
        # Print progress occasionally
        if step % report_interval == 0 or step == n_steps - 1:
            step_time = time.time() - start_time
            step_times.append(step_time)
            # Use latest computed energies if available, otherwise compute for display
            if not collected_energies and step == 0:
                 energies = batch_local_energy(walkers, params)
            
            print(f"Batch mean energy: {jnp.mean(energies):.6f}")
            report_progress(step, n_steps, acceptance_history, step_times, 
                           collected_energies if collected_energies else None)
            start_time = time.time()
            gc.collect()
    
    # Prepare and return results
    return prepare_sampling_results(
        collected_samples, collected_energies, acceptance_history, walkers, step_times)
