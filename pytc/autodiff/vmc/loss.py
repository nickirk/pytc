"""Loss functions for VMC optimization.

This module contains various loss functions used in variational Monte Carlo
optimization, including energy minimization and variance minimization.
"""

import functools
import jax
import jax.numpy as jnp
from jax import tree_util
from typing import Callable, Optional, Tuple
import folx
import kfac_jax


def make_energy_loss(
    ansatz,
    optimizer_type: str = "adam",
    cost_fn: Optional[Callable] = None,
    clip_multiplier: float = 5.0,
    use_custom_jvp: bool = True,
    max_vmap_batch_size: int = 0
):
    """Factory to create energy-based loss function for VMC optimization.
    
    This creates a loss function that computes the average local energy,
    with optional energy clipping and custom JVP for memory efficiency.
    
    Args:
        ansatz: Wavefunction object with local_energy method
        optimizer_type: Type of optimizer ("adam", "sgd", "kfac", etc.)
        cost_fn: Optional cost function to apply to energies (defaults to mean)
        clip_multiplier: Multiplier for energy clipping range (clips to mean ± multiplier * std)
        use_custom_jvp: Whether to use custom JVP for memory-efficient gradients
        max_vmap_batch_size: If 0, use standard vmap everywhere. If >0, use folx.batched_vmap 
                            Requires folx package. Recommended batch size: 10-50.
    
    Returns:
        Loss function with signature (params, batch_data) -> (loss, AuxData)
        where AuxData is a namedtuple with (mean_energy, energy_std, clipped_energies, diff)
    """
    # Default cost function: mean energy
    if cost_fn is None:
        cost_fn = jnp.mean
    
    # Choose vmap implementation based on max_vmap_batch_size
    if max_vmap_batch_size == 0:
        vmap_impl = jax.vmap
    else:
        vmap_impl = functools.partial(folx.batched_vmap, max_batch_size=max_vmap_batch_size)
    
    batch_local_energy = vmap_impl(
        lambda w, p: ansatz.local_energy(w, p)[0],
        in_axes=(0, None), 
        out_axes=0
    )

    batch_network = vmap_impl(
        lambda w, p: ansatz(w, p)[0][1],  # Returns log_psi
        in_axes=(0, None), 
        out_axes=0
    )
    
    # Note: We don't need batch_log_psi anymore - in the JVP we call ansatz directly on the batch
    
    if use_custom_jvp:
        # Internal structure to hold all data needed for gradient computation
        from collections import namedtuple
        AuxData = namedtuple('AuxData', ['mean_energy', 'energy_std', 'clipped_energies', 'diff'])
        
        @jax.custom_jvp
        def loss_fn(params, batch_data):
            """Energy loss function with custom JVP.
            
            Args:
                params: [jastrow_params, linear_coeffs]
                batch_data: Either walkers (Optax) or (walkers, None) (KFAC)
            
            Returns:
                loss: Scalar loss value (mean energy or custom cost)
                aux: AuxData namedtuple with (mean_energy, energy_std, clipped_energies, diff)
                     Can be indexed as aux[0], aux[1] for backward compatibility
            """
            # Extract walkers from batch
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                walkers, ansatz_arg = batch_data
                ansatz_dynamic = ansatz_arg
            else:
                walkers = batch_data
                ansatz_dynamic = ansatz
            
            # Compute all local energies using vmap (or batched_vmap)
            energies = batch_local_energy(walkers, params)
            
            # Compute statistics
            mean_energy = jnp.mean(energies)
            energy_std = jnp.mean(jnp.abs(energies - mean_energy))
            
            # Clip energies to avoid numerical instability
            clipped_energies = jnp.clip(
                energies,
                mean_energy - clip_multiplier * energy_std,
                mean_energy + clip_multiplier * energy_std
            )
            
            # Compute diff for gradient computation (store for reuse in JVP)
            mean_clipped = jnp.mean(clipped_energies)
            diff = clipped_energies - mean_clipped
            
            # Compute cost
            cost = cost_fn(clipped_energies)
            
            # Return cost and auxiliary data (namedtuple is indexable and JIT-compatible)
            return cost, AuxData(mean_energy, energy_std, clipped_energies, diff)
        
        @loss_fn.defjvp
        def loss_fn_jvp(primals, tangents):
            """Custom JVP for memory-efficient VMC gradients.
            
            This implements the VMC gradient estimator:
            ∇<E> = <(E_L - <E>) * ∇log|ψ|>
            
            Key optimization: Reuses energies and diff from forward pass!
            1. Forward pass already computed clipped_energies and diff
            2. Extract diff from auxiliary data (no need to recompute energies!)
            3. Single jvp call on batch_log_psi
            4. Gradient = dot(diff, log_psi_tangent) / N
            
            By only differentiating log|ψ| instead of the full local_energy,
            we avoid materializing gradients of non-parameter-dependent terms.
            """
            params, batch_data = primals
            params_tangent, _ = tangents
            
            # Run forward pass to get primal output and cached intermediate values
            cost_primal, aux_data = loss_fn(params, batch_data)
            mean_energy = aux_data.mean_energy
            energy_std = aux_data.energy_std
            clipped_energies = aux_data.clipped_energies
            diff = aux_data.diff
            
            # Extract walkers from batch_data
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
         # batch_network takes (walkers, params) but we only differentiate params
            # So we curry it to make a function of just params
            def log_psi_fn(p):
                return batch_network(walkers, p)
            
            # Single JVP call - now only differentiating wrt params
            log_psi_primal, log_psi_tangent = jax.jvp(
                log_psi_fn,
                (params,),
                (params_tangent,)
            )
            
            # VMC gradient: single dot product!
            # ∇⟨E⟩ = ⟨(E_L - ⟨E⟩) * ∇log|ψ|⟩ = dot(diff, ∇log|ψ|) / N
            n_walkers = diff.shape[0]
            cost_tangent = jnp.dot(diff, log_psi_tangent) / n_walkers
            
            # For KFAC compatibility - register distributions
            if optimizer_type.lower() == "kfac":
                #kfac_jax.register_normal_predictive_distribution(clipped_energies[:, None])
                kfac_jax.register_normal_predictive_distribution(log_psi_primal[:, None])
            
            # Return primal cost and tangent
            # Tangent aux_data: use zeros for cached values (they're not differentiated)
            tangent_aux = AuxData(0.0, 0.0, jnp.zeros_like(clipped_energies), jnp.zeros_like(diff))
            return (cost_primal, aux_data), (cost_tangent, tangent_aux)
        
        return loss_fn
    
    else:
        # Standard loss without custom JVP
        def loss_fn(params, batch_data):
            """Energy loss function (standard autodiff).
            
            Args:
                params: [jastrow_params, linear_coeffs]
                batch_data: Either walkers (Optax) or (walkers, None) (KFAC)
            
            Returns:
                loss: Scalar loss value (mean energy or custom cost)
                aux: Tuple of (mean_energy, energy_std)
            """
            # Extract walkers from batch
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                walkers, ansatz_arg = batch_data
                ansatz_dynamic = ansatz_arg
            else:
                walkers = batch_data
                ansatz_dynamic = ansatz
            
            # Define batch_local_energy using the current ansatz
            batch_local_energy = vmap_impl(
                lambda w, p: ansatz_dynamic.local_energy(w, p)[0],
                in_axes=(0, None), 
                out_axes=0
            )
            
            # Compute all local energies using vmap
            energies = batch_local_energy(walkers, params)
            
            # Compute statistics
            mean_energy = jnp.mean(energies)
            energy_std = jnp.mean(jnp.abs(energies - mean_energy))
            
            # Clip energies
            clipped_energies = jnp.clip(
                energies,
                mean_energy - clip_multiplier * energy_std,
                mean_energy + clip_multiplier * energy_std
            )
            
            # Compute cost
            cost = cost_fn(clipped_energies)
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac":
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return cost, (mean_energy, energy_std)
        
        return loss_fn


def make_variance_loss(
    ansatz,
    optimizer_type: str = "adam",
    use_custom_jvp: bool = True,
    max_vmap_batch_size: int = 0
):
    """Factory to create variance-based loss function for reference variance optimization.
    
    This minimizes the variance of local energies with respect to a reference determinant,
    which can improve the quality of the Jastrow factor.
    
    Uses vmap (or batched_vmap for memory efficiency) following the same pattern as make_energy_loss.
    
    Args:
        ansatz: Wavefunction object with local_energy method
        optimizer_type: Type of optimizer ("adam", "sgd", "kfac", etc.)
        use_custom_jvp: Whether to use custom JVP for memory-efficient gradients
        max_vmap_batch_size: If 0, use standard vmap. If >0, use folx.batched_vmap 
                            for memory efficiency. Recommended batch size: 10-50.
        use_hamiltonian_grad: If True and use_custom_jvp=True, use Hamiltonian-based
                            gradient method for Jastrow parameters:
                            ∇σ² = 2/(n-1) Σ(E_L - Ē)[Ĥ(∂J/∂a) - E_L·∂J/∂a]
                            This provides more memory-efficient gradients.
    
    Returns:
        Loss function with signature (params, batch_data) -> (variance, (mean_energy, energy_std))
    """
    # Choose vmap implementation based on max_vmap_batch_size
    if max_vmap_batch_size == 0:
        vmap_impl = jax.vmap
    else:
        vmap_impl = functools.partial(folx.batched_vmap, max_batch_size=max_vmap_batch_size)
    # Define batch_local_energy using the current ansatz
    batch_local_energy = vmap_impl(
        lambda w, p: ansatz.local_energy(w, p)[0],
        in_axes=(0, None), 
        out_axes=0
    )
    # Define batch_network using the current ansatz
    batch_network = vmap_impl(
        lambda w, p: ansatz(w, p)[0][1],  # Returns log_psi
        in_axes=(0, None), 
        out_axes=0
    )

    
    if use_custom_jvp:
        @jax.custom_jvp
        def loss_fn(params, batch_data):
            """Variance loss function with custom JVP.
            
            Args:
                params: [jastrow_params, linear_coeffs]
                batch_data: Either walkers (Optax) or (walkers, None) (KFAC)
            
            Returns:
                variance: Sample variance of local energies
                aux: Tuple of (mean_energy, energy_std)
            """
            # Extract walkers from batch
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Compute local energies
            energies = batch_local_energy(walkers, params)
            e_mean = jnp.mean(energies)
            e_std = jnp.std(energies)
            
            # Sample variance: sum((E - <E>)^2) / (n - 1)
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - e_mean)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            
            return variance, (e_mean, e_std)
        
        @loss_fn.defjvp
        def loss_fn_jvp(primals, tangents):
            """Custom JVP for variance minimization.
            
            """
            params, batch_data = primals
            params_tangent, _ = tangents
            jastrow_params_tangent, linear_coeffs_tangent = params_tangent
            
            # Extract walkers and ansatz
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                walkers, ansatz_arg = batch_data
                ansatz_dynamic = ansatz_arg
            else:
                walkers = batch_data
                ansatz_dynamic = ansatz
            
            if ansatz_dynamic is None:
                raise ValueError("Ansatz must be provided either in make_variance_loss or in batch_data")

            # Forward pass
            energies = batch_local_energy(walkers, params) # Pass ansatz
            e_mean = jnp.mean(energies)
            e_std = jnp.std(energies)
            
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - e_mean)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            aux_data = (e_mean, e_std)
            

            
            # ========== Standard Gradient Method ==========
            # Compute JVP of local energies
            def compute_energies(p):
                # Use the dynamic ansatz here
                return batch_local_energy(walkers, p)
            
            _, energy_tangent = jax.jvp(
                compute_energies,
                (params,),
                (params_tangent,)
            )
            # Single JVP call - now only differentiating wrt params
            #log_psi_primal = batch_network(walkers, params)
            # For KFAC compatibility
            if optimizer_type.lower() == "kfac":
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            # Variance gradient: ∇var = 2 * mean((E_L - ⟨E⟩) * ∇E_L)
            energy_diff = energies - e_mean
            if n_walkers > 1:
                variance_tangent = 2.0 * jnp.dot(energy_diff, energy_tangent) / (n_walkers - 1)
            else:
                variance_tangent = 0.0
            
            return (variance, aux_data), (variance_tangent, aux_data)
        
        return loss_fn
    
    else:
        # Standard variance loss without custom JVP
        def loss_fn(params, batch_data):
            """Variance loss function (standard autodiff).
            
            Args:
                params: [jastrow_params, linear_coeffs]
                batch_data: Either walkers (Optax) or (walkers, None) (KFAC)
            
            Returns:
                variance: Sample variance of local energies
                aux: Tuple of (mean_energy, energy_std)
            """
            # Extract walkers from batch
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                walkers, ansatz_arg = batch_data
                ansatz_dynamic = ansatz_arg
            else:
                walkers = batch_data
                ansatz_dynamic = ansatz
            
            # Define batch_local_energy using the current ansatz
            batch_local_energy = vmap_impl(
                lambda w, p: ansatz_dynamic.local_energy(w, p)[0],
                in_axes=(0, None), 
                out_axes=0
            )
            
            # Compute local energies
            energies = batch_local_energy(walkers, params)
            e_mean = jnp.mean(energies)
            e_std = jnp.std(energies)
            
            # Sample variance: sum((E - <E>)^2) / (n - 1)
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - e_mean)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac":
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return variance, (e_mean, e_std)
        
        return loss_fn
