"""PBC VMC optimization driver.

A periodic-boundary-condition counterpart to
:func:`pytc.vmc.optimize_ref_var`. Reuses the molecular Newton optimizer
and variance-loss factory by injecting two PBC-specific pieces:

  * :func:`pytc.pbc.vmc.metropolis.make_mcmc_step` (lattice-aware proposal)
  * :func:`pytc.pbc.vmc.hamiltonian.compute_single_walker_energy`
    (Ewald-summed Coulomb instead of bare 1/r)

The Ewald total enters via a closure-based local-energy override on
the SlaterJastrow instance: we wrap ``ansatz.local_energy`` so the
molecular variance loss factory transparently uses the PBC energy.

Multi-GPU sharding is not yet wired up — single-device only for now.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import jax
from jax import random
from jax.tree_util import tree_map
from flax import struct

from pytc.ansatz.sj import SlaterJastrow
from pytc.vmc.optimizer import create_optimizer
from pytc.vmc.loss import make_variance_loss
from pytc.vmc.optimization import (
    make_second_order_training_step,
    make_training_step,
    make_opt_update_step,
)

from ..ansatz.ksj import KSlaterJastrow
from .ewald import EwaldParams
from .metropolis import make_mcmc_step
from .sampling import burn_in
from .walker import initialize_walkers
from .hamiltonian import compute_single_walker_energy


@struct.dataclass
class PBCSlaterJastrow(SlaterJastrow):
    """Γ-only SlaterJastrow with Ewald-aware ``local_energy``.

    Adds an :class:`EwaldParams` field and overrides ``local_energy`` to
    use :func:`pytc.pbc.vmc.hamiltonian.compute_single_walker_energy`
    (periodic Coulomb) instead of the molecular bare 1/r path.
    """
    ewald: EwaldParams = None

    @classmethod
    def from_base(cls, sj: SlaterJastrow, ewald: EwaldParams):
        return cls(
            dets=sj.dets,
            atom_coords=sj.atom_coords,
            atom_charges=sj.atom_charges,
            ion_ion_potential=sj.ion_ion_potential,
            jastrow=sj.jastrow,
            ewald=ewald,
        )

    def local_energy(self, walker, params):
        jastrow_params, _ = params
        e = compute_single_walker_energy(self, walker, jastrow_params, self.ewald)
        return e, walker


@struct.dataclass
class PBCKSlaterJastrow(KSlaterJastrow):
    """k-point SlaterJastrow with Ewald-aware ``local_energy``.

    Same idea as :class:`PBCSlaterJastrow` but inherits :class:`KSlaterJastrow`
    so ``__call__`` routes through ``eval_ksj`` (complex Bloch path)
    instead of the real-valued molecular evaluator.
    """
    ewald: EwaldParams = None

    @classmethod
    def from_base(cls, ksj: KSlaterJastrow, ewald: EwaldParams):
        return cls(
            dets=ksj.dets,
            atom_coords=ksj.atom_coords,
            atom_charges=ksj.atom_charges,
            ion_ion_potential=ksj.ion_ion_potential,
            jastrow=ksj.jastrow,
            ewald=ewald,
        )

    def local_energy(self, walker, params):
        jastrow_params, _ = params
        e = compute_single_walker_energy(self, walker, jastrow_params, self.ewald)
        return e, walker


def _wrap_with_ewald(sj, ewald):
    """Dispatch on the SJ type to attach the Ewald-aware ``local_energy``."""
    if isinstance(sj, KSlaterJastrow):
        return PBCKSlaterJastrow.from_base(sj, ewald)
    return PBCSlaterJastrow.from_base(sj, ewald)

logger = logging.getLogger(__name__)


def optimize_ref_var(
    sj,
    cell,
    ewald,
    *,
    n_walkers: int = 1024,
    step_size: float = 0.5,
    burn_in_steps: int = 1000,
    n_opt_steps: int = 100,
    n_steps: int = 20,
    optimizer_type: str = "newton",
    learning_rate: float = 0.1,
    opt_kwargs: Optional[Dict[str, Any]] = None,
    params=None,
    key=None,
    move_type: str = "one",
    n_mcmc_per_opt: Optional[int] = None,
    n_opt_per_mcmc: Optional[int] = None,
    max_vmap_batch_size: int = 0,
) -> Dict[str, Any]:
    """Variance-minimization optimization for a PBC SlaterJastrow.

    Args:
        sj: Periodic :class:`SlaterJastrow` built via
            ``SlaterJastrow.create(cell, jastrow, dets=[create_slater_det(cell, ...)])``.
        cell: ``pyscf.pbc.gto.Cell`` (supercell at Γ for now).
        ewald: Cached :class:`EwaldParams` for ``cell``.
        n_walkers / step_size / burn_in_steps: MCMC settings.
        n_opt_steps: Number of optimization updates.
        n_steps: Legacy cadence parameter. With ``n_mcmc_per_opt`` and
            ``n_opt_per_mcmc`` both ``None``, falls back to one MCMC step
            per ``n_steps`` optimization updates (matching molecular default).
        optimizer_type: ``"newton"`` (default) or ``"adam"``.
        learning_rate / opt_kwargs: Optimizer hyperparameters.
        params: ``[jastrow_params, linear_coeffs]``; defaults if ``None``.
        key: PRNG key.
        move_type: ``"one"`` (default) or ``"all"``.

    Returns:
        Dict with ``energies``, ``stds``, ``cost``, ``acceptance``, ``params``.
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))
    if opt_kwargs is None:
        opt_kwargs = {}

    if n_mcmc_per_opt is None and n_opt_per_mcmc is None:
        n_mcmc_per_opt = 1
        n_opt_per_mcmc = n_steps
    elif n_mcmc_per_opt is None:
        n_mcmc_per_opt = 1
    elif n_opt_per_mcmc is None:
        n_opt_per_mcmc = 1

    if params is None:
        params = [sj.jastrow.init_params(), __import__('jax').numpy.ones(len(sj.dets))]

    ansatz = _wrap_with_ewald(sj, ewald)

    walkers = initialize_walkers(sj.dets[0], cell, n_walkers=n_walkers, key=key,
                                 log_init=False)
    # Prime walker fields (cached log_psi etc.) by a batched ansatz call.
    batch_call = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
    _, walkers = batch_call(walkers, params)

    logger.info("Performing PBC burn-in (%d steps)...", burn_in_steps)
    walkers, burn_acc, key, step_size = burn_in(
        sj, cell, n_walkers, burn_in_steps, step_size, params, key,
        move_type=move_type, log=True,
    )
    logger.info("Burn-in complete (acc %.3f, adapted step_size=%.4g).",
                burn_acc, step_size)

    lattice = jax.numpy.asarray(cell.lattice_vectors())
    mcmc_step = make_mcmc_step(sj, step_size=step_size, lattice=lattice,
                               move_type=move_type,
                               max_vmap_batch_size=max_vmap_batch_size)

    loss_fn = make_variance_loss(
        ansatz=ansatz,
        optimizer_type=optimizer_type,
        use_custom_jvp=True,
        max_vmap_batch_size=max_vmap_batch_size,
    )

    if optimizer_type.lower() == "newton":
        opt_kwargs = dict(opt_kwargs)
        opt_kwargs["value_and_grad_func"] = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True)
        opt_kwargs["curvature"] = "gauss_newton"
        opt_kwargs["max_vmap_batch_size"] = max_vmap_batch_size
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        key, subkey = random.split(key)
        opt_state = optimizer.init(params, subkey, (walkers, ansatz))
        training_step = make_second_order_training_step(
            mcmc_step, optimizer,
            n_mcmc_per_opt=n_mcmc_per_opt, n_opt_per_mcmc=n_opt_per_mcmc,
        )
        training_step = jax.jit(training_step)
    else:
        optimizer = create_optimizer(optimizer_type, learning_rate, opt_kwargs)
        opt_state = optimizer.init(params)
        opt_update_step = make_opt_update_step(loss_fn, optimizer)
        training_step = make_training_step(
            mcmc_step, opt_update_step,
            n_mcmc_per_opt=n_mcmc_per_opt, n_opt_per_mcmc=n_opt_per_mcmc,
        )

    energies, stds, losses, acceptances, params_hist = [], [], [], [], []
    t0 = time.time()
    logger.info("Compiling training step + first opt step...")
    key, subkey = random.split(key)
    if optimizer_type.lower() == "newton":
        walkers, params, opt_state, loss, aux, pmove, lr = training_step(
            ansatz, walkers, params, opt_state, subkey, 0
        )
    else:
        walkers, params, opt_state, loss, aux, pmove = training_step(
            ansatz, walkers, params, opt_state, subkey
        )
        lr = None
    logger.info("First step done in %.1fs", time.time() - t0)

    def _record(opt_step, loss, aux, pmove, lr, dt):
        e_mean, e_std = jax.device_get(aux)
        energies.append(float(e_mean)); stds.append(float(e_std))
        losses.append(float(jax.device_get(loss)))
        acceptances.append(float(jax.device_get(pmove)))
        params_hist.append(tree_map(lambda x: np.asarray(jax.device_get(x))
                                     if hasattr(x, 'shape') else x, params))
        lr_s = f" | LR={lr:.4f}" if lr is not None else ""
        logger.info(
            "Step %4d | Var: %.6f | E: %+.6f±%.4f | Accept: %.3f%s | %.2fs",
            opt_step, losses[-1], energies[-1], stds[-1], acceptances[-1], lr_s, dt,
        )

    _record(0, loss, aux, pmove, lr, time.time() - t0)
    t_step = time.time()

    for opt_step in range(1, n_opt_steps):
        key, subkey = random.split(key)
        if optimizer_type.lower() == "newton":
            walkers, params, opt_state, loss, aux, pmove, lr = training_step(
                ansatz, walkers, params, opt_state, subkey, opt_step
            )
        else:
            walkers, params, opt_state, loss, aux, pmove = training_step(
                ansatz, walkers, params, opt_state, subkey
            )
            lr = None
        _record(opt_step, loss, aux, pmove, lr, time.time() - t_step)
        t_step = time.time()

    logger.info("Optimization complete.")
    return {
        "cost": np.asarray(losses),
        "energies": np.asarray(energies),
        "stds": np.asarray(stds),
        "acceptance": np.asarray(acceptances),
        "params": params_hist,
    }
