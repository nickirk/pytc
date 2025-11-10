"""Loss functions for VMC optimization.

This module contains various loss functions used in variational Monte Carlo
optimization, including energy minimization and variance minimization.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
try:
    import kfac_jax
    KFAC_AVAILABLE = True
except ImportError:
    KFAC_AVAILABLE = False


def make_energy_loss(
    ansatz,
    optimizer_type: str = "adam",
    cost_fn: Optional[Callable] = None,
    clip_multiplier: float = 5.0,
    use_custom_jvp: bool = True
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
    
    Returns:
        Loss function with signature (params, batch_data) -> (loss, (mean_energy, energy_std))
    """
    # Default cost function: mean energy
    if cost_fn is None:
        cost_fn = jnp.mean
    
    if use_custom_jvp:
        @jax.custom_jvp
        def loss_fn(params, batch_data):
            """Energy loss function with custom JVP.
            
            Args:
                params: [jastrow_params, linear_coeffs]
                batch_data: Either walkers (Optax) or (walkers, None) (KFAC)
            
            Returns:
                loss: Scalar loss value (mean energy or custom cost)
                aux: Tuple of (mean_energy, energy_std)
            """
            # Extract walkers from batch
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Compute local energies
            energies, _ = ansatz.local_energy(walkers, params)
            
            # Clip energies to avoid numerical instability
            mean_energy = jnp.mean(energies)
            energy_std = jnp.mean(jnp.abs(energies - mean_energy))
            clipped_energies = jnp.clip(
                energies,
                mean_energy - clip_multiplier * energy_std,
                mean_energy + clip_multiplier * energy_std
            )
            
            # Compute cost
            cost = cost_fn(clipped_energies)
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return cost, (mean_energy, energy_std)
        
        @loss_fn.defjvp
        def loss_fn_jvp(primals, tangents):
            """Custom JVP for memory-efficient VMC gradients.
            
            This implements the VMC gradient estimator:
            ∇<E> = <(E_L - <E>) * ∇log|ψ|>
            
            By only differentiating log|ψ| instead of the full local_energy,
            we avoid materializing gradients of non-parameter-dependent terms
            (potential energy, etc.), dramatically reducing memory usage.
            """
            params, batch_data = primals
            params_tangent, _ = tangents
            
            # Extract walkers
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Forward pass - compute energies and cost
            energies, _ = ansatz.local_energy(walkers, params)
            
            # Clip energies (same as forward pass)
            mean_energy = jnp.mean(energies)
            energy_std = jnp.mean(jnp.abs(energies - mean_energy))
            clipped_energies = jnp.clip(
                energies,
                mean_energy - clip_multiplier * energy_std,
                mean_energy + clip_multiplier * energy_std
            )
            
            # Compute cost
            cost = cost_fn(clipped_energies)
            aux_data = (mean_energy, energy_std)
            
            # For KFAC compatibility
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            # Compute energy differences for VMC gradient estimator
            energy_diff = clipped_energies - jnp.mean(clipped_energies)
            
            # Compute JVP of log|ψ| w.r.t. parameters
            # This is the key memory optimization: only differentiate log|ψ|
            def batch_log_psi(p):
                """Evaluate log|ψ| for all walkers."""
                psi_values, _ = ansatz(walkers, p)
                _, log_psi = psi_values  # (sign, log|psi|)
                return log_psi
            
            log_psi_primal, log_psi_tangent = jax.jvp(
                batch_log_psi,
                (params,),
                (params_tangent,)
            )
            
            # VMC gradient: ∇<E> · tangent = <(E_L - <E>) * (∇log|ψ| · tangent)>
            n_walkers = energies.shape[0]
            cost_tangent = jnp.dot(energy_diff, log_psi_tangent) / n_walkers
            
            # For KFAC compatibility
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(log_psi_primal[:, None])
            
            return (cost, aux_data), (cost_tangent, aux_data)
        
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
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Compute local energies
            energies, _ = ansatz.local_energy(walkers, params)
            
            # Clip energies to avoid numerical instability
            mean_energy = jnp.mean(energies)
            energy_std = jnp.mean(jnp.abs(energies - mean_energy))
            clipped_energies = jnp.clip(
                energies,
                mean_energy - clip_multiplier * energy_std,
                mean_energy + clip_multiplier * energy_std
            )
            
            # Compute cost
            cost = cost_fn(clipped_energies)
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return cost, (mean_energy, energy_std)
        
        return loss_fn


def make_variance_loss(
    ansatz,
    optimizer_type: str = "adam",
    use_custom_jvp: bool = False
):
    """Factory to create variance-based loss function for reference variance optimization.
    
    This minimizes the variance of local energies with respect to a reference determinant,
    which can improve the quality of the Jastrow factor.
    
    Args:
        ansatz: Wavefunction object with local_energy method
        optimizer_type: Type of optimizer ("adam", "sgd", "kfac", etc.)
        use_custom_jvp: Whether to use custom JVP for memory-efficient gradients
    
    Returns:
        Loss function with signature (params, batch_data) -> (variance, (mean_energy, energy_std))
    """
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
            energies, _ = ansatz.local_energy(walkers, params)
            e_mean = jnp.mean(energies)
            e_std = jnp.std(energies)
            
            # Sample variance: sum((E - <E>)^2) / (n - 1)
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - e_mean)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return variance, (e_mean, e_std)
        
        @loss_fn.defjvp
        def loss_fn_jvp(primals, tangents):
            """Custom JVP for variance minimization.
            
            This implements: ∇variance = 2 * E[(E_L - <E_L>) * ∇E_L]
            
            Using the chain rule and the fact that ∇E_L = E_L * ∇log|ψ| for VMC,
            we get the memory-efficient gradient by only differentiating log|ψ|.
            """
            params, batch_data = primals
            params_tangent, _ = tangents
            
            # Extract walkers
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Forward pass
            energies, _ = ansatz.local_energy(walkers, params)
            e_mean = jnp.mean(energies)
            e_std = jnp.std(energies)
            
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - e_mean)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            aux_data = (e_mean, e_std)
            
            # For KFAC compatibility
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            # Compute JVP of local energies
            def compute_energies(p):
                e, _ = ansatz.local_energy(walkers, p)
                return e
            
            _, energy_tangent = jax.jvp(
                compute_energies,
                (params,),
                (params_tangent,)
            )
            
            # Variance gradient: ∇var = 2 * mean((E_L - <E>) * ∇E_L)
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
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Compute local energies
            energies, _ = ansatz.local_energy(walkers, params)
            e_mean = jnp.mean(energies)
            e_std = jnp.std(energies)
            
            # Sample variance: sum((E - <E>)^2) / (n - 1)
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - e_mean)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return variance, (e_mean, e_std)
        
        return loss_fn


def make_combined_loss(
    ansatz,
    optimizer_type: str = "adam",
    energy_weight: float = 1.0,
    variance_weight: float = 0.0,
    clip_multiplier: float = 5.0,
    use_custom_jvp: bool = True
):
    """Factory to create combined energy + variance loss function.
    
    This allows optimizing a weighted combination of energy and variance:
    L = energy_weight * <E> + variance_weight * Var(E)
    
    Args:
        ansatz: Wavefunction object with local_energy method
        optimizer_type: Type of optimizer ("adam", "sgd", "kfac", etc.)
        energy_weight: Weight for energy term (default: 1.0)
        variance_weight: Weight for variance term (default: 0.0)
        clip_multiplier: Multiplier for energy clipping range
        use_custom_jvp: Whether to use custom JVP for memory-efficient gradients
    
    Returns:
        Loss function with signature (params, batch_data) -> (loss, (mean_energy, energy_std))
    """
    if use_custom_jvp:
        @jax.custom_jvp
        def loss_fn(params, batch_data):
            """Combined energy + variance loss with custom JVP.
            
            Args:
                params: [jastrow_params, linear_coeffs]
                batch_data: Either walkers (Optax) or (walkers, None) (KFAC)
            
            Returns:
                loss: Weighted combination of energy and variance
                aux: Tuple of (mean_energy, energy_std)
            """
            # Extract walkers from batch
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Compute local energies
            energies, _ = ansatz.local_energy(walkers, params)
            
            # Compute statistics
            mean_energy = jnp.mean(energies)
            energy_std = jnp.std(energies)
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - mean_energy)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            
            # Clip energies for the energy term
            energy_dev = jnp.mean(jnp.abs(energies - mean_energy))
            clipped_energies = jnp.clip(
                energies,
                mean_energy - clip_multiplier * energy_dev,
                mean_energy + clip_multiplier * energy_dev
            )
            clipped_mean = jnp.mean(clipped_energies)
            
            # Combined loss
            loss = energy_weight * clipped_mean + variance_weight * variance
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return loss, (mean_energy, energy_std)
        
        @loss_fn.defjvp
        def loss_fn_jvp(primals, tangents):
            """Custom JVP for combined loss."""
            params, batch_data = primals
            params_tangent, _ = tangents
            
            # Extract walkers
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Forward pass
            energies, _ = ansatz.local_energy(walkers, params)
            
            mean_energy = jnp.mean(energies)
            energy_std = jnp.std(energies)
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - mean_energy)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            
            energy_dev = jnp.mean(jnp.abs(energies - mean_energy))
            clipped_energies = jnp.clip(
                energies,
                mean_energy - clip_multiplier * energy_dev,
                mean_energy + clip_multiplier * energy_dev
            )
            clipped_mean = jnp.mean(clipped_energies)
            
            loss = energy_weight * clipped_mean + variance_weight * variance
            aux_data = (mean_energy, energy_std)
            
            # For KFAC compatibility
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            # Compute JVP using log|ψ| for energy term
            def batch_log_psi(p):
                psi_values, _ = ansatz(walkers, p)
                _, log_psi = psi_values
                return log_psi
            
            log_psi_primal, log_psi_tangent = jax.jvp(
                batch_log_psi,
                (params,),
                (params_tangent,)
            )
            
            # Energy term gradient
            energy_diff = clipped_energies - jnp.mean(clipped_energies)
            energy_tangent = jnp.dot(energy_diff, log_psi_tangent) / n_walkers
            
            # Variance term gradient (using full energy JVP)
            def compute_energies(p):
                e, _ = ansatz.local_energy(walkers, p)
                return e
            
            _, energy_jvp = jax.jvp(
                compute_energies,
                (params,),
                (params_tangent,)
            )
            
            variance_diff = energies - mean_energy
            if n_walkers > 1:
                variance_tangent = 2.0 * jnp.dot(variance_diff, energy_jvp) / (n_walkers - 1)
            else:
                variance_tangent = 0.0
            
            # Combined gradient
            loss_tangent = energy_weight * energy_tangent + variance_weight * variance_tangent
            
            # For KFAC compatibility
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(log_psi_primal[:, None])
            
            return (loss, aux_data), (loss_tangent, aux_data)
        
        return loss_fn
    
    else:
        # Standard combined loss without custom JVP
        def loss_fn(params, batch_data):
            """Combined energy + variance loss (standard autodiff)."""
            # Extract walkers from batch
            if isinstance(batch_data, tuple):
                walkers = batch_data[0]
            else:
                walkers = batch_data
            
            # Compute local energies
            energies, _ = ansatz.local_energy(walkers, params)
            
            # Compute statistics
            mean_energy = jnp.mean(energies)
            energy_std = jnp.std(energies)
            n_walkers = energies.shape[0]
            variance = jnp.sum((energies - mean_energy)**2) / (n_walkers - 1) if n_walkers > 1 else 0.0
            
            # Clip energies for the energy term
            energy_dev = jnp.mean(jnp.abs(energies - mean_energy))
            clipped_energies = jnp.clip(
                energies,
                mean_energy - clip_multiplier * energy_dev,
                mean_energy + clip_multiplier * energy_dev
            )
            clipped_mean = jnp.mean(clipped_energies)
            
            # Combined loss
            loss = energy_weight * clipped_mean + variance_weight * variance
            
            # For KFAC, register predictive distribution
            if optimizer_type.lower() == "kfac" and KFAC_AVAILABLE:
                kfac_jax.register_normal_predictive_distribution(energies[:, None])
            
            return loss, (mean_energy, energy_std)
        
        return loss_fn
