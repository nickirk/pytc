"""Loss functions for VMC optimization.

This module contains loss functions for variational Monte Carlo optimization,
specifically designed to conform to FermiNet standards.
"""

import functools
from typing import Tuple, Optional, Callable

import chex
import jax
import jax.numpy as jnp
import kfac_jax
from ferminet import constants
from ferminet import networks
from ferminet import loss as qmc_loss_functions
import folx

# Use FermiNet's AuxiliaryLossData
AuxiliaryLossData = qmc_loss_functions.AuxiliaryLossData
LossFn = qmc_loss_functions.LossFn

def make_variance_loss(
    network: networks.LogFermiNetLike,
    local_energy: Callable,
    clip_local_energy: float = 0.0,
    clip_from_median: bool = True,
    center_at_clipped_energy: bool = True,
    max_vmap_batch_size: int = 0
) -> LossFn:
    """Creates the variance loss function for reference determinant optimization.

    This minimizes the variance of local energies with respect to a reference determinant.
    The walkers are assumed to be sampled from a reference distribution (e.g. Hartree-Fock
    or the Slater part of the ansatz) that does NOT depend on the parameters being optimized
    (typically Jastrow parameters).

    Consequently, the gradient of the loss is simply:
    ∇Var(E) = 2 * < (E_L - <E_L>) * ∇E_L >

    Args:
        network: callable which evaluates the log of the magnitude of the
            wavefunction. (Used for interface compatibility, though strictly
            not needed for the gradient if sampling is fixed, unless used for
            logging/diagnostics).
        local_energy: callable which evaluates the local energy.
        clip_local_energy: If greater than zero, clip local energies.
        clip_from_median: If true, center the clipping window at the median.
        center_at_clipped_energy: If true, center gradients.
        max_vmap_batch_size: If 0, use standard vmap. If >0, use batched_vmap.

    Returns:
        Callable with signature (params, key, data) -> (loss, aux_data).
    """
    vmap = jax.vmap if max_vmap_batch_size == 0 else functools.partial(
        folx.batched_vmap, max_batch_size=max_vmap_batch_size)
    
    batch_local_energy = vmap(
        local_energy,
        in_axes=(
            None,
            0,
            networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0),
        ),
        out_axes=(0, 0)
    )

    @jax.custom_jvp
    def variance_loss(
        params: networks.ParamTree,
        key: chex.PRNGKey,
        data: networks.FermiNetData,
    ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
        keys = jax.random.split(key, num=data.positions.shape[0])
        e_l, e_l_mat = batch_local_energy(params, keys, data)
        
        # Mean and Variance
        n = jnp.array(data.positions.shape[0], dtype=jnp.float32)
        n = constants.psum(n)
        mean_e = constants.pmean(jnp.mean(e_l))
        std_e = constants.pmean(jnp.std(e_l))
        diff = e_l - mean_e
        # Unbiased variance: multiply by N / (N - 1)
        variance = constants.pmean(jnp.mean(diff**2)) * n / (n - 1.0)
        
        return variance, AuxiliaryLossData(
            energy=mean_e,
            variance=variance,
            local_energy=e_l,
            clipped_energy=e_l, # Placeholder, updated in JVP if clipping used
            local_energy_mat=e_l_mat,
        )

    @variance_loss.defjvp
    def variance_loss_jvp(primals, tangents):
        params, key, data = primals
        params_dot, _, _ = tangents
        
        # Forward pass
        loss, aux_data = variance_loss(params, key, data)
        
        # We need the gradient of the local energy w.r.t parameters: ∇E_L
        # And we want to compute 2 * mean( (E_L - mean_E) * ∇E_L )
        
        # Define a function to compute local energies from params
        def compute_local_energies(p):
            keys = jax.random.split(key, num=data.positions.shape[0])
            e, _ = batch_local_energy(p, keys, data)
            return e

        # Compute JVP of local energies: ∇E_L . v
        # This gives us the directional derivative of E_L along the tangent vector
        e_l_primal, e_l_tangent = jax.jvp(compute_local_energies, (params,), (params_dot,))
        
        # Apply clipping if requested (to the primal values used for weighting)
        if clip_local_energy > 0.0:
             # Re-use FermiNet's clip_local_values if accessible, or reimplement
             # Since I imported qmc_loss_functions, I can't easily access its internal helper
             # unless it's exported. It is NOT exported in __init__.
             # I will implement a simple version here or assume no clipping for now if complex.
             # But let's try to be robust.
             # For now, let's just use the raw values if clipping helper is not available.
             # Actually, I can copy the helper or just implement the logic.
             # Let's implement simple clipping logic.
             tv = jnp.mean(jnp.abs(aux_data.local_energy - aux_data.energy))
             tv = constants.pmean(tv)
             center = aux_data.energy
             if clip_from_median:
                 center = jnp.median(constants.all_gather(aux_data.local_energy))
             
             delta = clip_local_energy * tv
             clipped_e_l = jnp.clip(aux_data.local_energy, center - delta, center + delta)
             
             if center_at_clipped_energy:
                 diff_center = constants.pmean(jnp.mean(clipped_e_l))
             else:
                 diff_center = aux_data.energy
                 
             diff = clipped_e_l - diff_center
             # Update aux_data with clipped values
             aux_data = aux_data.replace(clipped_energy=clipped_e_l)
        else:
             diff = aux_data.local_energy - aux_data.energy

        # Gradient of variance:
        # ∇Var = 2 * < (E_L - <E>) * ∇E_L > * N / (N - 1)
        # In JVP terms: dot(∇Var, v) = 2 * < (E_L - <E>) * (∇E_L . v) > * N / (N - 1)
        # e_l_tangent is (∇E_L . v)
        
        n = jnp.array(data.positions.shape[0], dtype=jnp.float32)
        n = constants.psum(n)
        variance_tangent = 2.0 * jnp.mean(diff * e_l_tangent)
        variance_tangent = constants.pmean(variance_tangent) * n / (n - 1.0)
        
        # Let's define batch_network
        batch_network = vmap(network, in_axes=(None, 0, 0, 0, 0), out_axes=0)
        log_psi = batch_network(params, data.positions, data.spins, data.atoms, data.charges)
        kfac_jax.register_normal_predictive_distribution(log_psi[:, None])

        return (loss, aux_data), (variance_tangent, aux_data)

    return variance_loss
