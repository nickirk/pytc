"""Monte Carlo move proposals with periodic boundary conditions.

PBC variants of the molecular moves in :mod:`pytc.vmc.moves`. Each proposal
is wrapped into the primitive cell after the Gaussian step, and the
drift-diffusion Green's function is evaluated using minimum-image
displacements so that detailed balance is preserved under periodicity.
"""

import jax
import jax.numpy as jnp
from jax import random

from pytc.ansatz.det import rank1_update_one_electron
from pytc.ansatz.sj import SlaterJastrow, update_jastrow_one_electron

from ..ansatz.kdet import KSlaterDet, rank1_update_one_electron_kpts
from ..utils import wrap, mic_displacement


def _all_electron_move(ansatz, walker, step_size, key, params, lattice, batch_ansatz=None):
    """Move all electrons at once for each walker, wrapping into the cell.

    Args:
        ansatz: Wavefunction object (PBC-aware).
        walker: Walker dataclass with current state (batched).
        step_size: Standard deviation of the Gaussian proposal.
        key: PRNG key.
        params: Ansatz parameters.
        lattice: ``(3, 3)`` lattice matrix with rows ``a1, a2, a3``.
        batch_ansatz: Optional pre-vmapped ansatz function.

    Returns:
        ``(psi_values, new_psi_values, current_walker, proposals)`` where
        ``proposals.positions`` lies inside the primitive cell.
    """
    if batch_ansatz is None:
        batch_ansatz = jax.vmap(lambda w, p: ansatz(w, p), in_axes=(0, None))

    psi_values, current_walker = batch_ansatz(walker, params)

    key, subkey = random.split(key)
    new_positions = walker.positions + random.normal(subkey, walker.positions.shape) * step_size
    new_positions = wrap(new_positions, lattice)

    proposals = current_walker.replace(
        positions=new_positions,
        move_mask=jnp.ones_like(walker.move_mask),
    )

    new_psi_values, proposals = batch_ansatz(proposals, params)

    return psi_values, new_psi_values, current_walker, proposals


def _one_electron_move(ansatz, walker, step_size, key, params, lattice, batch_ansatz=None):
    """Move one randomly selected electron per walker with a Sherman-Morrison
    rank-1 update, wrapping the proposed position into the primitive cell.

    Identical algebra to :func:`pytc.vmc.moves._one_electron_move` — only
    the proposed position is wrapped before the rank-1 update is invoked.
    Wrapping is mathematically a no-op when the orbital evaluator is
    periodic (GTO), but stored positions stay inside the cell so
    subsequent moves and any minimum-image distance computations are
    well-behaved.

    Args:
        ansatz: A PBC :class:`SlaterDet` or :class:`SlaterJastrow` whose
            constituent Jastrows use minimum-image distances.
        walker: Walker dataclass with current state (batched).
        step_size: Gaussian proposal standard deviation.
        key: PRNG key.
        params: Ansatz parameters.
        lattice: ``(3, 3)`` lattice matrix.
        batch_ansatz: Ignored (kept for API parity with the molecular path).

    Returns:
        ``(psi_values, new_psi_values, walker_current, proposals)``.
    """
    key, subkey = random.split(key)
    n_walkers, n_electrons = walker.positions.shape[0], walker.positions.shape[1]
    electron_indices = random.randint(subkey, (n_walkers,), 0, n_electrons)

    move_mask = jnp.arange(n_electrons)[None, :] == electron_indices[:, None]

    key, subkey = random.split(key)
    mask_3d = move_mask[:, :, None]
    new_positions = walker.positions + mask_3d * random.normal(subkey, walker.positions.shape) * step_size
    new_positions = wrap(new_positions, lattice)

    psi_values = (walker.psi_sign, walker.log_psi)

    if isinstance(ansatz, SlaterJastrow):
        det = ansatz.dets[0]
    else:
        det = ansatz

    rank1_fn = (
        rank1_update_one_electron_kpts
        if isinstance(det, KSlaterDet)
        else rank1_update_one_electron
    )

    def _single_walker_rank1(walker_i, new_pos_i, elec_idx):
        proposal_i = walker_i.replace(positions=new_pos_i)
        det_ratio, det_logabs_new, det_sign_new, proposal_updated = \
            rank1_fn(det, proposal_i, elec_idx)
        return det_ratio, det_logabs_new, det_sign_new, proposal_updated

    det_ratios, det_logabs_new, det_sign_new, proposals = jax.vmap(
        _single_walker_rank1, in_axes=(0, 0, 0)
    )(walker, new_positions, electron_indices)

    proposals = proposals.replace(move_mask=move_mask)

    if isinstance(ansatz, SlaterJastrow):
        jastrow_params = params[0]

        def _single_jastrow_update(old_pos, new_pos, elec_idx, old_log_j):
            return update_jastrow_one_electron(
                ansatz, old_pos, new_pos, elec_idx, jastrow_params, old_log_j
            )

        new_log_jastrow = jax.vmap(_single_jastrow_update)(
            walker.positions, new_positions, electron_indices, walker.log_jastrow
        )

        log_linear_coeff = jnp.log(jnp.abs(params[1][0]))
        new_psi_logabs = new_log_jastrow + log_linear_coeff + det_logabs_new
        new_psi_sign = jnp.sign(params[1][0]) * det_sign_new

        proposals = proposals.replace(
            log_psi=new_psi_logabs,
            psi_sign=new_psi_sign,
            log_jastrow=new_log_jastrow,
        )
    else:
        proposals = proposals.replace(
            log_psi=det_logabs_new,
            psi_sign=det_sign_new,
            log_jastrow=jnp.zeros_like(det_logabs_new),
        )

    new_psi_values = (proposals.psi_sign, proposals.log_psi)
    return psi_values, new_psi_values, walker, proposals


def _compute_green_function(r_target, r_source, quantum_force, time_step, lattice):
    """Transition probability density for drift-diffusion under PBC.

    Uses minimum-image displacement so that proposals which wrap across
    cell boundaries see the correct (shortest) drift contribution.
    Otherwise identical to the molecular Green's function.

    ``G(R -> R') = exp(- |MIC(R' - R) - D F(R) tau|^2 / (2 tau))``

    Args:
        r_target: Target positions, shape ``(..., n_electrons, 3)``.
        r_source: Source positions, same shape.
        quantum_force: Quantum force at source, same shape.
        time_step: Drift-diffusion time step.
        lattice: ``(3, 3)`` lattice matrix.

    Returns:
        Green's function values, broadcast over the leading axes.
    """
    drift = 0.5 * quantum_force * time_step
    diff = mic_displacement(r_target, r_source, lattice) - drift
    exponent = -jnp.sum(diff ** 2, axis=(-2, -1)) / (2.0 * time_step)
    return jnp.exp(exponent)
