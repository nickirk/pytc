"""Move generation algorithms for VMC sampling.

This module contains functions for generating different types of moves
in Monte Carlo sampling, including single-electron and multi-electron moves,
as well as importance sampling with drift-diffusion.
"""

import jax
import jax.numpy as jnp
from jax import random

from pytc.autodiff.ansatz.det import (
    SlaterDet,
    rank1_update_one_electron,
)
from pytc.autodiff.ansatz.sj import SlaterJastrow, update_jastrow_one_electron


def _all_electron_move(ansatz, walker, step_size, key, params, batch_ansatz=None):
    """Move all electrons at once for each walker.
    
    Args:
        ansatz: Wavefunction object
        walker: Walker dataclass with current state (batched)
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        params: Parameters for the ansatz
        batch_ansatz: Optional pre-vmapped ansatz function
        
    Returns:
        proposals: Walker with proposed new positions and all-True move_mask
        psi_values: Wavefunction values for current walker
        new_psi_values: Wavefunction values for proposals
        current_walker: Updated current walker (if ansatz updates it)
    """
    # Use provided batch_ansatz or create default vmap
    if batch_ansatz is None:
        batch_ansatz = jax.vmap(lambda w, p: ansatz(w, p), in_axes=(0, None))
    
    # Compute initial wavefunction values
    psi_values, current_walker = batch_ansatz(walker, params)
    
    # Generate proposals (all electrons move)
    key, subkey = random.split(key)
    new_positions = walker.positions + random.normal(subkey, walker.positions.shape) * step_size
    
    # Create proposal walker with all-True move_mask (all electrons moved)
    proposals = current_walker.replace(
        positions=new_positions,
        move_mask=jnp.ones_like(walker.move_mask)
    )
    
    # Compute new wavefunction values
    new_psi_values, proposals = batch_ansatz(proposals, params)
    
    # Return updated current_walker
    return psi_values, new_psi_values, current_walker, proposals

def _one_electron_move(ansatz, walker, step_size, key, params, batch_ansatz=None):
    """Move one randomly selected electron for each walker.
    
    Uses Sherman-Morrison rank-1 updates to avoid full O(N³) determinant
    recomputation for the proposal.  The current ψ is read from the walker
    cache (log_psi / psi_sign).  Only the *proposal* determinant is updated
    via a cheap O(N²) rank-1 update + single-electron AO evaluation.

    For SlaterJastrow ansatze, the Jastrow factor is updated by recomputing
    only the N-1 pairs involving the moved electron.
    
    Args:
        ansatz: Wavefunction object (SlaterDet or SlaterJastrow)
        walker: Walker dataclass with current state (batched).
            Must have valid log_psi, psi_sign, and log_jastrow fields.
        step_size: Standard deviation of Gaussian proposal
        key: PRNG key
        params: Parameters for the ansatz
        batch_ansatz: Ignored (kept for API compatibility; rank-1 path
            does not use a batched ansatz call).
        
    Returns:
        psi_values: Wavefunction (sign, log|ψ|) for current walker (from cache)
        new_psi_values: Wavefunction (sign, log|ψ|) for proposals
        walker_updated: Current walker (unchanged)
        proposals: Walker with proposed new positions and all Slater fields
            updated via rank-1 Sherman-Morrison.
    """
    # Select electron to move for each walker
    key, subkey = random.split(key)
    n_electrons = walker.positions.shape[1]
    n_walkers = walker.positions.shape[0]
    electron_indices = random.randint(subkey, (n_walkers,), 0, n_electrons)
    
    # Create move mask indicating which electron moved
    move_mask = (jnp.arange(n_electrons)[None, :] == electron_indices[:, None])
    
    # Generate proposals for selected electrons only
    key, subkey = random.split(key)
    mask_3d = move_mask[:, :, None]
    new_positions = walker.positions + mask_3d * random.normal(subkey, walker.positions.shape) * step_size
    
    # ---- Current ψ: use cached values from walker ----
    psi_values = (walker.psi_sign, walker.log_psi)
    
    # ---- Proposal: rank-1 update for each walker ----
    # Determine the SlaterDet to use for rank-1 update
    if isinstance(ansatz, SlaterJastrow):
        det = ansatz.dets[0]
    else:
        # ansatz is a SlaterDet
        det = ansatz

    def _single_walker_rank1(walker_i, new_pos_i, elec_idx):
        """Rank-1 update for one walker (unbatched)."""
        # Replace positions with the proposal positions
        proposal_i = walker_i.replace(positions=new_pos_i)
        # Rank-1 det update
        det_ratio, det_logabs_new, det_sign_new, proposal_updated = \
            rank1_update_one_electron(det, proposal_i, elec_idx)
        return det_ratio, det_logabs_new, det_sign_new, proposal_updated

    # vmap over the walker batch
    det_ratios, det_logabs_new, det_sign_new, proposals = jax.vmap(
        _single_walker_rank1, in_axes=(0, 0, 0)
    )(walker, new_positions, electron_indices)

    # Set move_mask on the proposals
    proposals = proposals.replace(move_mask=move_mask)

    # ---- Jastrow update (only for SlaterJastrow ansatze) ----
    if isinstance(ansatz, SlaterJastrow):
        jastrow_params = params[0]

        def _single_jastrow_update(old_pos, new_pos, elec_idx, old_log_j):
            return update_jastrow_one_electron(
                ansatz, old_pos, new_pos, elec_idx, jastrow_params, old_log_j
            )

        new_log_jastrow = jax.vmap(_single_jastrow_update)(
            walker.positions, new_positions, electron_indices, walker.log_jastrow
        )

        # Total log|ψ'| = log(J') + log|c_0| + log|det'|
        # Must match the convention in eval_sj where
        #   psi_logabs = log_jastrow + log|linear_coeffs[0]| + det_logabs
        log_linear_coeff = jnp.log(jnp.abs(params[1][0]))
        new_psi_logabs = new_log_jastrow + log_linear_coeff + det_logabs_new
        # For single-det SJ with linear_coeffs[0] = 1, sign is just det sign
        # (Jastrow is always positive: exp(u) > 0)
        new_psi_sign = jnp.sign(params[1][0]) * det_sign_new

        proposals = proposals.replace(
            log_psi=new_psi_logabs,
            psi_sign=new_psi_sign,
            log_jastrow=new_log_jastrow,
        )
    else:
        # Pure SlaterDet: no Jastrow contribution
        proposals = proposals.replace(
            log_psi=det_logabs_new,
            psi_sign=det_sign_new,
            log_jastrow=jnp.zeros_like(det_logabs_new),
        )

    new_psi_values = (proposals.psi_sign, proposals.log_psi)

    return psi_values, new_psi_values, walker, proposals


@jax.jit
def _compute_green_function(r_target, r_source, quantum_force, time_step):
    """Compute transition probability density for drift-diffusion.
    
    G(R→R') = exp(-(R'-R-D*F(R)*τ)²/(2*τ))
    
    Args:
        r_target: Target position array
        r_source: Source position array
        quantum_force: Quantum force at source
        time_step: Time step
        
    Returns:
        Green's function values for transitions
    """
    # Calculate the expected drift
    drift = 0.5 * quantum_force * time_step
    
    # Calculate the difference between actual and drift-guided movement
    diff = r_target - r_source - drift
    
    # Calculate the exponent of the Green's function
    exponent = -jnp.sum(diff**2, axis=(-2, -1)) / (2.0 * time_step)
    
    # Return the Green's function values
    return jnp.exp(exponent)
