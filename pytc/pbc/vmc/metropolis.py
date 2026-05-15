"""Metropolis-Hastings driver for periodic boundary conditions.

The accept/reject body is identical to the molecular path — only the
proposal step differs, since the PBC moves wrap into the primitive cell
and require a ``lattice``. We therefore reuse the molecular
``metropolis_hastings`` shape and substitute the PBC proposal calls.

Importance-sampling (drift-diffusion) is not implemented yet; the basic
single-electron move is what the VMC examples drive in practice.
"""

import jax
import jax.numpy as jnp
from jax import random
import folx

from .moves import _all_electron_move, _one_electron_move


def metropolis_hastings(
    ansatz,
    walker,
    step_size,
    key,
    params,
    lattice,
    move_type: str = "one",
    batch_ansatz=None,
):
    """Single Metropolis-Hastings step under periodic boundary conditions.

    Args:
        ansatz: PBC-aware wavefunction (SlaterDet or SlaterJastrow with
            PBC dets / Jastrows).
        walker: Walker dataclass with cached log_psi/psi_sign/log_jastrow.
        step_size: Standard deviation of the Gaussian proposal.
        key: PRNG key.
        params: Ansatz parameters (``[jastrow_params, linear_coeffs]`` for SJ;
            ``None`` for a pure SlaterDet).
        lattice: ``(3, 3)`` lattice matrix.
        move_type: ``"one"`` (rank-1, recommended) or ``"all"``.
        batch_ansatz: Optional pre-vmapped ansatz call.

    Returns:
        ``(new_walker, acceptance_rate)``.
    """
    if move_type == "all":
        psi_values, new_psi_values, current_walker, proposals = _all_electron_move(
            ansatz, walker, step_size, key, params, lattice, batch_ansatz=batch_ansatz
        )
    elif move_type == "one":
        psi_values, new_psi_values, current_walker, proposals = _one_electron_move(
            ansatz, walker, step_size, key, params, lattice, batch_ansatz=batch_ansatz
        )
    else:
        raise ValueError("move_type must be either 'all' or 'one'")

    _, psi_logabs = psi_values
    _, new_psi_logabs = new_psi_values
    acceptance_prob = jnp.exp(2.0 * (new_psi_logabs - psi_logabs))

    key, subkey = random.split(key)
    n_walkers = walker.positions.shape[0]
    accept_mask = random.uniform(subkey, shape=(n_walkers,)) < acceptance_prob
    accept_count = jnp.sum(accept_mask)

    accept_mask_3d = accept_mask[:, None, None]
    accept_mask_4d = accept_mask[:, None, None, None]

    new_det_up = (
        jnp.where(accept_mask, proposals.det_up[0], current_walker.det_up[0]),
        jnp.where(accept_mask, proposals.det_up[1], current_walker.det_up[1]),
    )
    new_det_down = (
        jnp.where(accept_mask, proposals.det_down[0], current_walker.det_down[0]),
        jnp.where(accept_mask, proposals.det_down[1], current_walker.det_down[1]),
    )

    new_walker = current_walker.replace(
        positions=jnp.where(accept_mask_4d[:, :, :, 0], proposals.positions, current_walker.positions),
        slater_up=jnp.where(accept_mask_3d, proposals.slater_up, current_walker.slater_up),
        slater_down=jnp.where(accept_mask_3d, proposals.slater_down, current_walker.slater_down),
        inv_up=jnp.where(accept_mask_3d, proposals.inv_up, current_walker.inv_up),
        inv_down=jnp.where(accept_mask_3d, proposals.inv_down, current_walker.inv_down),
        det_up=new_det_up,
        det_down=new_det_down,
        grad_up=jnp.where(accept_mask_4d, proposals.grad_up, current_walker.grad_up),
        grad_down=jnp.where(accept_mask_4d, proposals.grad_down, current_walker.grad_down),
        lap_up=jnp.where(accept_mask_3d, proposals.lap_up, current_walker.lap_up),
        lap_down=jnp.where(accept_mask_3d, proposals.lap_down, current_walker.lap_down),
        move_mask=jnp.zeros_like(current_walker.move_mask),
        log_psi=jnp.where(accept_mask, proposals.log_psi, current_walker.log_psi),
        psi_sign=jnp.where(accept_mask, proposals.psi_sign, current_walker.psi_sign),
        log_jastrow=jnp.where(accept_mask, proposals.log_jastrow, current_walker.log_jastrow),
    )

    acceptance_rate = accept_count / n_walkers
    return new_walker, acceptance_rate


def make_mcmc_step(
    ansatz,
    step_size,
    lattice,
    move_type: str = "one",
    max_vmap_batch_size: int = 0,
):
    """Factory returning a JIT-compiled PBC MCMC step.

    Captures both ``ansatz`` and ``lattice`` at factory time so the JIT'd
    step has a stable closure. Mirrors the molecular
    :func:`pytc.vmc.metropolis.make_mcmc_step` but threads the lattice.

    Args:
        ansatz: PBC wavefunction object — captured.
        step_size: Gaussian proposal width.
        lattice: ``(3, 3)`` lattice matrix — captured.
        move_type: ``"one"`` or ``"all"``.
        max_vmap_batch_size: If > 0, use ``folx.batched_vmap`` for the
            batched ansatz call (memory control on large walker batches).

    Returns:
        ``mcmc_step(ansatz, walkers, key, params) -> (new_walkers, acc_rate)``.
        The first argument is ignored at call time — the captured ansatz
        from the factory is what drives proposals.
    """
    if move_type not in ("all", "one"):
        raise ValueError(f"move_type must be 'one' or 'all', got '{move_type}'")

    captured_ansatz = ansatz
    captured_lattice = jnp.asarray(lattice)

    if max_vmap_batch_size > 0:
        batch_ansatz = folx.batched_vmap(
            lambda w, p: captured_ansatz(w, p),
            in_axes=(0, None),
            max_batch_size=max_vmap_batch_size,
        )
    else:
        batch_ansatz = jax.vmap(lambda w, p: captured_ansatz(w, p), in_axes=(0, None))

    @jax.jit
    def _mcmc_step(walkers, key, params):
        return metropolis_hastings(
            captured_ansatz, walkers, step_size, key, params, captured_lattice,
            move_type=move_type, batch_ansatz=batch_ansatz,
        )

    def mcmc_step(ansatz, walkers, key, params):
        """Compatible with the molecular factory API. ``ansatz`` is ignored —
        the captured ansatz from factory time drives proposals, and it is
        deliberately kept out of the JIT'd function's pytree to avoid
        hashing its ``pytree_node=False`` array-valued metadata fields."""
        return _mcmc_step(walkers, key, params)

    return mcmc_step
