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
):
    """Run an MCMC burn-in pass and return the equilibrated walker batch.

    Args:
        sj: PBC :class:`SlaterJastrow`.
        cell: ``pyscf.pbc.gto.Cell`` (used for walker initialization and to
            obtain the lattice).
        n_walkers: Number of walkers.
        n_steps: Number of MCMC steps to discard.
        step_size: Gaussian proposal width.
        params: Ansatz parameters.
        key: PRNG key.
        move_type: ``"one"`` (rank-1) or ``"all"`` (full-batch proposal).
        log: Emit info-level progress every ``n_steps // 5`` steps.

    Returns:
        ``(walker, mean_acceptance_rate, key)``.
    """
    lattice = jnp.asarray(cell.lattice_vectors())

    walker = initialize_walkers(
        sj.dets[0], cell, n_walkers=n_walkers, key=key, log_init=False
    )
    # Prime the walker cache so the rank-1 move sees valid log_psi.
    batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
    _, walker = batch_ansatz(walker, params)

    step = make_mcmc_step(sj, step_size=step_size, lattice=lattice, move_type=move_type)

    acceptance_sum = 0.0
    report_every = max(1, n_steps // 5)
    for i in range(n_steps):
        key, sub = random.split(key)
        walker, acc = step(sj, walker, sub, params)
        acceptance_sum += float(acc)
        if log and (i % report_every == 0 or i == n_steps - 1):
            logger.info(
                "Burn-in step %d / %d, running acceptance = %.3f",
                i, n_steps, acceptance_sum / (i + 1),
            )

    return walker, acceptance_sum / max(n_steps, 1), key


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

    walker, burn_acc, key = burn_in(
        sj, cell, n_walkers, burn_in_steps, step_size, params, key,
        move_type=move_type, log=log,
    )
    if log:
        logger.info("Burn-in complete (acceptance %.3f). Beginning production.", burn_acc)

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

    step = make_mcmc_step(det, step_size=step_size, lattice=lattice)

    # Burn-in
    acc_sum_burn = 0.0
    report_every = max(1, burn_in_steps // 5)
    for i in range(burn_in_steps):
        key, sub = random.split(key)
        walker, acc = step(det, walker, sub, None)
        acc_sum_burn += float(acc)
        if log and (i % report_every == 0 or i == burn_in_steps - 1):
            logger.info(
                "Burn-in step %d / %d, running acceptance = %.3f",
                i, burn_in_steps, acc_sum_burn / (i + 1),
            )

    if log:
        logger.info(
            "Burn-in complete (acceptance %.3f). Beginning production.",
            acc_sum_burn / max(burn_in_steps, 1),
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
