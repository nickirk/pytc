"""Sampling procedures for VMC simulation.

This module contains functions for burn-in procedures and main sampling loops,
including both standard MCMC and importance sampling variants.
"""

import gc
import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
import folx
from typing import Dict, Any

from .metropolis import (
    metropolis_hastings, metropolis_hastings_importance_sampling,
    make_mcmc_step, make_mcmc_step_importance
)
from .walker import initialize_walkers
from .mcmc_utils import prepare_sampling_results, report_progress
from .hamiltonian import eval_local_energy
from functools import partial
from .sharding import (
    get_vmap_fn, shard_map_wrap,
    create_mesh, pad_n_walkers, n_devices, is_multi_gpu,
    initialize_walkers_sharded, replicate
)

logger = logging.getLogger(__name__)


def burn_in(ansatz, 
            walkers, 
            n_steps=2000, 
            step_size=0.01,
            key=None, 
            params=None, 
            report_interval=100, 
            move_type="one",
            max_vmap_batch_size=0,
            mesh=None):
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
        
    logger.info(f"Starting burn-in with {n_steps} steps...")
    
    # Warm up walker cache (populate log_psi, psi_sign) so that
    # _one_electron_move can reuse cached values instead of recomputing.
    vmap_fn = get_vmap_fn(max_vmap_batch_size=max_vmap_batch_size)
    _warmup_ansatz = vmap_fn(lambda w, p: ansatz(w, p), in_axes=(0, None))
    _, walkers = _warmup_ansatz(walkers, params)

    if mesh is not None:
        axis_name = "walkers"
        local_batch_ansatz = (
            folx.batched_vmap(
                lambda w, p: ansatz(w, p),
                in_axes=(0, None),
                max_batch_size=max_vmap_batch_size,
            ) if max_vmap_batch_size > 0 else
            jax.vmap(lambda w, p: ansatz(w, p), in_axes=(0, None))
        )

        def _mcmc_step_sharded(walkers, key, params, step_size):
            key = random.fold_in(key, jax.lax.axis_index(axis_name))
            walkers_out, acceptance = metropolis_hastings(
                ansatz, walkers, step_size, key, params,
                move_type=move_type, batch_ansatz=local_batch_ansatz
            )
            acceptance = (
                jax.lax.psum(acceptance, axis_name)
                / jax.lax.psum(jnp.array(1.0, dtype=acceptance.dtype), axis_name)
            )
            return walkers_out, acceptance

        sharded_step = shard_map_wrap(
            _mcmc_step_sharded,
            mesh=mesh,
            in_specs=(P(axis_name), P(), P(), P()),
            out_specs=(P(axis_name), P()),
        )

        def mcmc_step(_ansatz, walkers, step_size, key, params):
            return sharded_step(walkers, key, params, step_size)
    else:
        # JIT-compile the MCMC step function to speed up the loop.
        # We partial out move_type since it's a static string argument.
        # step_size is passed as argument so it can vary without recompilation.
        mcmc_step = partial(
            metropolis_hastings, move_type=move_type, batch_ansatz=vmap_fn(
                lambda w, p: ansatz(w, p),
                in_axes=(0, None)
            )
        )
    mcmc_step = jax.jit(mcmc_step)
    
    start_time = time.time()
    for step in range(n_steps):
        key, subkey = random.split(key)
        walkers, acceptance = mcmc_step(
            ansatz, walkers, step_size, subkey, params)
        
        # Convert acceptance to Python float for history
        acceptance_float = float(acceptance)
        acceptance_history.append(acceptance_float)
        
        if step % report_interval == 0:
            logger.info(f"Burn-in step {step}/{n_steps}, acceptance: {acceptance_float:.3f}, time: {time.time() - start_time:.2f}s")
            step_size *= acceptance_float / 0.5
            start_time = time.time()
            # Periodic garbage collection
            gc.collect()
    
    logger.info("Burn-in complete.")
    return walkers, acceptance_history, key, step_size


def adaptive_burn_in(
    ref_det,
    full_ansatz,
    walkers,
    params,
    step_size=0.01,
    key=None,
    move_type="one",
    max_vmap_batch_size=0,
    mesh=None,
    chunk_size=500,
    max_steps=50000,
    acceptance_target=0.5,
    acceptance_tol=0.02,
    stability_window=3,
    energy_stability_atol=0.05,
    variance_stability_rtol=0.02,
):
    """Burn in until the ensemble's E_L/Var estimates stabilize, instead of
    a fixed step count.

    Tier-2 (safety net) of the two-tier burn-in-deficit fix (task #12,
    2026-07-12): a fixed burn-in count that's tuned for one system size
    silently under-provisions a larger one -- Grace's H40/W=30000 sweep
    showed the production default (500) leaves Var 21x too high, and even
    10,000 sweeps hadn't plateaued (true plateau: ~15,000-20,000 sweeps).
    Equilibration time should grow with system size, so this replaces the
    guess with a real termination criterion: run in chunks of
    `chunk_size` sweeps (reusing burn_in() unmodified -- this function
    only orchestrates repeated calls to it, it does not change burn_in's
    own behavior), and after each chunk:

    1. PRE-GATE on acceptance being within `acceptance_tol` of
       `acceptance_target`. Acceptance reflects the MCMC step size's own
       adaptive calibration reaching target, NOT global |Psi|^2 mixing --
       Grace's sweep is a direct counter-example to using it as a
       sufficient criterion alone (acceptance was already ~0.50 by
       burn-in=2000 while E kept moving substantially through
       burn-in=10000), so this is a necessary cheap pre-check, not the
       stopping decision itself.
    2. Once the pre-gate passes, compute batch-mean E_L and Var via
       `full_ansatz` (the actual physical trial wavefunction, i.e. the
       jastrow-dressed ansatz -- NOT `ref_det`, which only defines the
       MCMC proposal/target distribution the walkers are drawn from) and
       append to a sliding window of the last `stability_window` chunks.
       Terminate once E has stayed within `energy_stability_atol` (an
       ABSOLUTE tolerance, not relative -- E crosses zero over this
       campaign's trajectories, so a relative tolerance is either
       trivially satisfied or meaninglessly strict near the crossing,
       Felix's refinement 2026-07-12) and Var has stayed within
       `variance_stability_rtol` (relative -- Var is strictly >= 0, no
       zero-crossing issue) across the full window.
    3. Hard-capped at `max_steps` total sweeps regardless, so a system
       that never stabilizes (or a badly-set tolerance) can't hang.

    If `walkers` came from `mcmc_utils.resample_walkers`, this function's
    own sweep counter IS "sweeps since resample" by construction (it has
    no knowledge of any burn-in the walkers may have already had before
    being passed in) -- exactly the accounting Felix's resample-design
    refinement asked for, so the stability window can't read "stable" off
    still-correlated bootstrap duplicates.

    Args:
        ref_det: The determinant (or other) ansatz that defines the MCMC
                proposal/target distribution -- same role as `ansatz` in
                `burn_in`.
        full_ansatz: The physical trial wavefunction (e.g. SlaterJastrow)
                whose local_energy is the actual quantity of interest for
                the stability check.
        walkers: Initial walker configurations.
        params: Full [jastrow_params, linear_coeffs] for `full_ansatz`.
        step_size: Initial MCMC proposal step size.
        key: PRNG key.
        move_type, max_vmap_batch_size, mesh: forwarded to burn_in.
        chunk_size: Sweeps per chunk (same role as burn_in's
                report_interval -- one step-size adaptation per chunk).
                Default 500: Grace's H40/W=30000 plateau sweep found the
                real equilibration timescale is ~15,000-20,000 sweeps
                (2026-07-12), so a 100-sweep chunk would mean ~150-200
                separate E_L-batch evaluations before termination -- 500
                trades some termination-point precision for 5x fewer
                evals at that scale.
        max_steps: Hard cap on total sweeps. Default 50000: comfortably
                above Grace's measured H40/W=30000 plateau point
                (20,000-40,000 sweeps) so the stability criterion, not
                this cap, is what normally terminates -- a cap close to
                the expected equilibration time would mask whether the
                mechanism actually works. Systems slower than H40 may
                still need this raised.
        acceptance_target: Pre-gate center. Default 0.5, matching
                burn_in's own step-size adaptation target.
        acceptance_tol: Pre-gate band. Default 0.02: Grace's equilibrated
                H40 readings cluster 0.499-0.513 (max deviation ~0.013
                from target) vs 0.672 unequilibrated -- 0.02 covers the
                observed equilibrated spread with a little margin while
                staying far below the unequilibrated value (Felix
                suggested ~0.01 as a starting point, 2026-07-12; widened
                slightly so the pre-gate doesn't reject the 0.513 reading
                actually observed at the validated plateau).
        stability_window: Number of consecutive chunks required stable.
        energy_stability_atol: Absolute energy tolerance (Ha) for the
                window range. Default 0.05 Ha: Felix's framing from
                Grace's H40 plateau residual, parameterized as an
                absolute (not per-atom) Ha value here -- callers on
                larger systems should scale this up (e.g. tolerance-per-
                atom * n_atoms) since equilibrium energy fluctuations
                grow with system size, pending a real H80/H160
                calibration point (2026-07-12).
        variance_stability_rtol: Relative tolerance for Var's window
                range. Default 0.02 (2%): Grace's plateau data shows a
                clean separation -- pre-plateau relative changes are
                20-130% (5000->10000->20000 sweeps), while the actual
                plateau (20,000->40,000 sweeps) shows Var changing only
                0.83%. 2% sits comfortably above that plateau noise floor
                and well below the transition region.

    Returns:
        Tuple of (equilibrated_walkers, chunk_history, new_key, step_size,
        total_steps_run). chunk_history is a list of per-chunk dicts with
        keys: steps_so_far, acceptance, mean_energy, variance (the latter
        two are None for chunks skipped by the acceptance pre-gate).
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))

    vmap_fn = get_vmap_fn(max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)
    batch_local_energy = jax.jit(vmap_fn(
        lambda w, p: full_ansatz.local_energy(w, p)[0],
        in_axes=(0, None),
        out_axes=0,
    ))

    chunk_history = []
    e_window = []
    var_window = []
    total_steps = 0

    while total_steps < max_steps:
        this_chunk = min(chunk_size, max_steps - total_steps)
        walkers, acc_hist, key, step_size = burn_in(
            ref_det, walkers, this_chunk, step_size, key, params=params,
            report_interval=chunk_size, move_type=move_type,
            max_vmap_batch_size=max_vmap_batch_size, mesh=mesh,
        )
        total_steps += this_chunk
        chunk_acceptance = float(np.mean(acc_hist)) if acc_hist else None

        record = {"steps_so_far": total_steps, "acceptance": chunk_acceptance,
                   "mean_energy": None, "variance": None}

        if chunk_acceptance is not None and abs(chunk_acceptance - acceptance_target) <= acceptance_tol:
            energies = np.asarray(jax.device_get(batch_local_energy(walkers, params))).reshape(-1)
            e_mean = float(np.mean(energies))
            var = float(np.mean((energies - e_mean) ** 2))
            record["mean_energy"] = e_mean
            record["variance"] = var

            e_window.append(e_mean)
            var_window.append(var)
            e_window = e_window[-stability_window:]
            var_window = var_window[-stability_window:]

            if len(e_window) == stability_window:
                e_range = max(e_window) - min(e_window)
                var_range = max(var_window) - min(var_window)
                var_scale = max(abs(np.mean(var_window)), 1e-12)
                if e_range <= energy_stability_atol and var_range / var_scale <= variance_stability_rtol:
                    chunk_history.append(record)
                    logger.info(
                        f"adaptive_burn_in converged after {total_steps} sweeps "
                        f"(E window {e_window}, Var window {var_window})."
                    )
                    return walkers, chunk_history, key, step_size, total_steps
        else:
            # Pre-gate not yet passed -- acceptance still settling.
            # Reset the stability window: a chunk that skipped the E/Var
            # check contributes no evidence either way, and letting a
            # stale window from before a pre-gate dip carry over risks
            # false "stable" on a window that isn't contiguous.
            e_window = []
            var_window = []

        chunk_history.append(record)

    logger.info(
        f"adaptive_burn_in hit max_steps={max_steps} without meeting the "
        f"stability criterion -- returning current state; consider "
        f"raising max_steps or loosening the stability tolerances."
    )
    return walkers, chunk_history, key, step_size, total_steps


def burn_in_with_importance(ansatz, walkers, n_steps, time_step, key, params, report_interval=100, mesh=None):
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
        
    logger.info(f"Starting burn-in with {n_steps} steps using importance sampling...")
    
    if mesh is not None:
        axis_name = "walkers"

        def _mcmc_step_sharded(walkers, key, params, time_step):
            key = random.fold_in(key, jax.lax.axis_index(axis_name))
            walkers_out, acceptance = metropolis_hastings_importance_sampling(
                ansatz, walkers, time_step, key, params
            )
            acceptance = (
                jax.lax.psum(acceptance, axis_name)
                / jax.lax.psum(jnp.array(1.0, dtype=acceptance.dtype), axis_name)
            )
            return walkers_out, acceptance

        sharded_step = shard_map_wrap(
            _mcmc_step_sharded,
            mesh=mesh,
            in_specs=(P(axis_name), P(), P(), P()),
            out_specs=(P(axis_name), P()),
        )

        def mcmc_step(_ansatz, walkers, time_step, key, params):
            return sharded_step(walkers, key, params, time_step)
    else:
        mcmc_step = metropolis_hastings_importance_sampling
    mcmc_step = jax.jit(mcmc_step)
    
    time_start = time.time()
    for step in range(n_steps):
        key, subkey = random.split(key)
        walkers, acceptance = mcmc_step(
            ansatz, walkers, time_step, subkey, params)
        acceptance_history.append(acceptance)
        
        if step % report_interval == 0:
            logger.info(f"Burn-in step {step}/{n_steps}, acceptance: {acceptance_history[-1]}, time: {time.time() - time_start:.2f}s")
            time_step *= acceptance_history[-1]/0.5
            time_start = time.time()
    
    logger.info("Burn-in complete.")
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
    
    # ---- Multi-GPU setup ----
    mesh = None
    if is_multi_gpu():
        num_devices = n_devices()
        mesh = create_mesh()
        padded_n = pad_n_walkers(n_walkers, num_devices)
        if padded_n != n_walkers:
            print(f"Padding n_walkers from {n_walkers} to {padded_n} "
                  f"(divisible by {num_devices} devices)")
            n_walkers = padded_n
        print(f"Multi-GPU auto-detected: {num_devices} devices, "
              f"{n_walkers // num_devices} walkers/device")
    
    # Initialize walkers
    if mesh is not None:
        walkers = initialize_walkers_sharded(
            ansatz, n_walkers, mesh, initial_walkers=initial_walkers, key=key
        )
        if params is not None:
            params = replicate(params, mesh)
        key = replicate(key, mesh)
    else:
        walkers = initialize_walkers(ansatz, n_walkers, initial_walkers, key)
    
    logger.info("Starting production sampling...")
    logger.info(f"Burn-in steps = {burn_in_steps}")
    logger.info(f"Number of walkers = {n_walkers}")
    logger.info(f"Number of steps = {n_steps}")
    logger.info(f"Thinning factor = {thinning}")
    logger.info(f"Step size = {step_size:.4f}")
    logger.info(f"Using importance sampling: {use_importance_sampling}")
    logger.info(f"Move type: {move_type}")
    
    # Perform burn-in with appropriate method
    if use_importance_sampling:
        walkers, acceptance_history, key, step_size = burn_in_with_importance(
            ansatz, walkers, burn_in_steps, step_size, key, params, mesh=mesh)
    else:
        walkers, acceptance_history, key, step_size = burn_in(
            ansatz, walkers, burn_in_steps, step_size, key=key, params=params, 
            move_type=move_type, max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)
    
    # Storage for collected samples
    collected_samples = []
    collected_energies = []
    step_times = []
    
    # JIT-compile MCMC step for production run
    vmap_fn = get_vmap_fn(max_vmap_batch_size=max_vmap_batch_size, mesh=mesh)
    if use_importance_sampling:
        mcmc_step = make_mcmc_step_importance(ansatz, step_size, mesh=mesh)
    else:
        mcmc_step = make_mcmc_step(
            ansatz, step_size, move_type,
            max_vmap_batch_size=max_vmap_batch_size, mesh=mesh
        )
        
    # JIT-compile energy evaluation
    batch_local_energy = jax.jit(vmap_fn(
        lambda w, p: ansatz.local_energy(w, p)[0],
        in_axes=(0, None)
    ))
    
    # Main sampling loop
    start_time = time.time()
    for step in range(n_steps):
        
        key, subkey = random.split(key)
        walkers, acceptance = mcmc_step(
            ansatz, walkers, subkey, params)
            
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
            
            logger.info(f"Batch mean energy: {jnp.mean(energies):.6f}")
            report_progress(step, n_steps, acceptance_history, step_times, 
                           collected_energies if collected_energies else None)
            start_time = time.time()
            gc.collect()
    
    # Prepare and return results
    return prepare_sampling_results(
        collected_samples, collected_energies, acceptance_history, walkers, step_times)
