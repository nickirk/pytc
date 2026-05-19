"""Sampling drivers for PBC VMC.

Minimal :func:`burn_in` and :func:`sample` that compose the PBC MCMC step
and the PBC Ewald-summed local energy. Multi-GPU sharding and importance
sampling are not implemented in this initial version — they are pure
re-exports away from the molecular code once needed.
"""

import logging
import time
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from .metropolis import make_mcmc_step
from .hamiltonian import compute_single_walker_energy, compute_single_walker_energy_kpts
from .walker import initialize_walkers

logger = logging.getLogger(__name__)


def burn_in(
    sj,
    cell,
    n_walkers: int,
    n_steps: int,
    step_size: float,
    params,
    key,
    move_type: str = "one",
    log: bool = True,
    adaptive_step_size: bool = True,
    step_size_adjust_interval: int = 100,
    target_acceptance: float = 0.5,
):
    """Run an MCMC burn-in pass and return the equilibrated walker batch.

    With ``adaptive_step_size=True`` (default), the step size is rescaled
    every ``step_size_adjust_interval`` steps toward ``target_acceptance``
    via ``step_size *= acc / target_acceptance`` — matching the molecular
    :func:`pytc.vmc.burn_in` behavior. The adapted final step size is
    returned so production samplers can use it.

    Args:
        sj: PBC :class:`SlaterJastrow`.
        cell: ``pyscf.pbc.gto.Cell``.
        n_walkers: Number of walkers.
        n_steps: Number of MCMC steps to discard.
        step_size: Initial Gaussian proposal width.
        params: Ansatz parameters.
        key: PRNG key.
        move_type: ``"one"`` or ``"all"``.
        log: Emit info-level progress at each adjustment.
        adaptive_step_size: If True, rescale step_size toward target.
        step_size_adjust_interval: Steps between rescaling.
        target_acceptance: Target acceptance rate (~0.5 is standard).

    Returns:
        ``(walker, mean_acceptance_rate, key, final_step_size)``.
    """
    lattice = jnp.asarray(cell.lattice_vectors())

    walker = initialize_walkers(
        sj.dets[0], cell, n_walkers=n_walkers, key=key, log_init=False
    )
    # Prime the walker cache so the rank-1 move sees valid log_psi.
    batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
    _, walker = batch_ansatz(walker, params)

    acceptance_sum = 0.0
    interval_acc_sum = 0.0
    interval_count = 0
    current_step = make_mcmc_step(
        sj, step_size=step_size, lattice=lattice, move_type=move_type)

    for i in range(n_steps):
        key, sub = random.split(key)
        walker, acc = current_step(sj, walker, sub, params)
        a = float(acc)
        acceptance_sum += a
        interval_acc_sum += a
        interval_count += 1

        last_step = (i == n_steps - 1)
        do_adjust = adaptive_step_size and (
            interval_count >= step_size_adjust_interval or last_step
        )
        if do_adjust:
            interval_acc = interval_acc_sum / interval_count
            new_step_size = step_size * interval_acc / max(target_acceptance, 1e-6)
            # Clamp to prevent runaway.
            new_step_size = max(new_step_size, 1e-4)
            if log:
                logger.info(
                    "Burn-in step %d / %d  acc(interval)=%.3f  step_size %.4g -> %.4g",
                    i + 1, n_steps, interval_acc, step_size, new_step_size,
                )
            if not last_step:
                step_size = new_step_size
                current_step = make_mcmc_step(
                    sj, step_size=step_size, lattice=lattice, move_type=move_type)
            interval_acc_sum = 0.0
            interval_count = 0

    return walker, acceptance_sum / max(n_steps, 1), key, step_size


def sample(
    sj,
    cell,
    ewald,
    n_walkers: int = 64,
    n_steps: int = 500,
    step_size: float = 0.4,
    burn_in_steps: int = 200,
    thinning: int = 1,
    params=None,
    key=None,
    move_type: str = "one",
    log: bool = True,
) -> Dict[str, Any]:
    """Sample local energies from a PBC trial wavefunction.

    Args:
        sj: PBC :class:`SlaterJastrow`.
        cell: ``pyscf.pbc.gto.Cell``.
        ewald: Cached :class:`EwaldParams` for the cell.
        n_walkers: Number of walkers.
        n_steps: Number of production steps (after burn-in).
        step_size: Gaussian proposal width.
        burn_in_steps: Number of steps to discard before measuring.
        thinning: Keep every ``thinning``-th sample. Set > 1 to reduce
            autocorrelation in the recorded series.
        params: Ansatz parameters; defaults to ``sj.init_params(key)``.
        key: PRNG key; defaults to a wall-clock seed.
        move_type: ``"one"`` or ``"all"``.
        log: Emit progress messages.

    Returns:
        Dict with keys:
          - ``energies``: 1D NumPy array of all recorded local energies
            (``n_walkers * n_recorded_steps`` entries).
          - ``mean``: Sample mean.
          - ``stderr``: Standard error of the mean (naïve, ignores
            autocorrelation; use blocking for an honest estimate).
          - ``acceptance``: Mean MCMC acceptance rate during production.
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))
    if params is None:
        params = sj.init_params(key)

    lattice = jnp.asarray(cell.lattice_vectors())

    walker, burn_acc, key, step_size = burn_in(
        sj, cell, n_walkers, burn_in_steps, step_size, params, key,
        move_type=move_type, log=log,
    )
    if log:
        logger.info(
            "Burn-in complete (acc %.3f, adapted step_size=%.4g). Beginning production.",
            burn_acc, step_size,
        )

    step = make_mcmc_step(sj, step_size=step_size, lattice=lattice, move_type=move_type)
    batch_local_energy = jax.jit(jax.vmap(
        lambda w: compute_single_walker_energy(sj, w, params[0], ewald)
    ))

    all_energies = []
    acc_sum = 0.0
    n_recorded = 0
    for step_i in range(n_steps):
        key, sub = random.split(key)
        walker, acc = step(sj, walker, sub, params)
        acc_sum += float(acc)
        if step_i % thinning == 0:
            es = np.asarray(batch_local_energy(walker))
            all_energies.append(es)
            n_recorded += 1

    energies = np.concatenate(all_energies)
    return {
        'energies': energies,
        'mean': float(np.mean(energies)),
        'stderr': float(np.std(energies) / np.sqrt(len(energies))),
        'acceptance': acc_sum / max(n_steps, 1),
    }


def sample_bare(
    det,
    cell,
    ewald,
    n_walkers: int = 64,
    n_steps: int = 500,
    step_size: float = 0.4,
    burn_in_steps: int = 200,
    thinning: int = 1,
    key=None,
    log: bool = True,
) -> Dict[str, Any]:
    """Sample local energies from a bare (no-Jastrow) PBC determinant.

    Works on either ``SlaterDet`` (Gamma-only) or ``KSlaterDet`` (k-mesh)
    — the metropolis step dispatches on the determinant type, and the
    energy function is duck-typed on ``atom_coords`` / ``atom_charges``
    and the cached complex Slater quantities.

    Args:
        det: A bare PBC SlaterDet or KSlaterDet.
        cell: The supercell (for walker init + lattice).
        ewald: Cached :class:`EwaldParams` for the supercell.
        Other kwargs analogous to :func:`sample`.

    Returns:
        Dict with ``energies``, ``mean``, ``stderr``, ``acceptance``.
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))

    lattice = jnp.asarray(cell.lattice_vectors())

    walker = initialize_walkers(
        det, cell, n_walkers=n_walkers, key=key, log_init=False,
    )
    # Prime walker cache. The det's __call__ runs the right eval path
    # (eval_det_value_and_grad for SlaterDet, eval_kdet_value_and_grad
    # for KSlaterDet).
    batch_call = jax.vmap(lambda w: det(w, None))
    _, walker = batch_call(walker)

    # Burn-in with adaptive step_size (matches the SlaterJastrow burn_in
    # behavior: rescale toward acceptance 0.5 every 100 steps).
    step = make_mcmc_step(det, step_size=step_size, lattice=lattice)
    acc_sum_burn = 0.0
    interval_acc = 0.0
    interval_count = 0
    adjust_every = 100
    target_acc = 0.5
    for i in range(burn_in_steps):
        key, sub = random.split(key)
        walker, acc = step(det, walker, sub, None)
        a = float(acc)
        acc_sum_burn += a
        interval_acc += a
        interval_count += 1
        last = (i == burn_in_steps - 1)
        if interval_count >= adjust_every or last:
            mean_interval_acc = interval_acc / interval_count
            new_step = max(step_size * mean_interval_acc / max(target_acc, 1e-6), 1e-4)
            if log:
                logger.info(
                    "Burn-in step %d / %d  acc(interval)=%.3f  step_size %.4g -> %.4g",
                    i + 1, burn_in_steps, mean_interval_acc, step_size, new_step,
                )
            if not last:
                step_size = new_step
                step = make_mcmc_step(det, step_size=step_size, lattice=lattice)
            interval_acc = 0.0
            interval_count = 0

    if log:
        logger.info(
            "Burn-in complete (acc %.3f, adapted step_size=%.4g). Beginning production.",
            acc_sum_burn / max(burn_in_steps, 1), step_size,
        )

    batch_local_energy = jax.jit(jax.vmap(
        lambda w: compute_single_walker_energy_kpts(det, w, ewald)
    ))

    all_energies = []
    acc_sum = 0.0
    for step_i in range(n_steps):
        key, sub = random.split(key)
        walker, acc = step(det, walker, sub, None)
        acc_sum += float(acc)
        if step_i % thinning == 0:
            es = np.asarray(batch_local_energy(walker))
            all_energies.append(es)

    energies = np.concatenate(all_energies)
    return {
        'energies': energies,
        'mean': float(np.mean(energies)),
        'stderr': float(np.std(energies) / np.sqrt(len(energies))),
        'acceptance': acc_sum / max(n_steps, 1),
    }
