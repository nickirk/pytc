"""JAX implementation of kinetic matrix elements."""
from functools import partial
import jax
import jax.numpy as jnp

def calc_K1(rho_paired, nabla_rho_paired, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K1 term with memory-efficient batching.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid)
        nabla_rho_paired: Array of shape (Nb*Nb, N_grid, 3)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,) for integration weights
        batch_size: Number of grid points to process at once
    """
    Nb2 = rho_paired.shape[0]
    N_grid = grid_points.shape[0]
    result = jnp.zeros((Nb2, Nb2))
    weights = jnp.asarray(weights)
    
    # Process grid points in batches for r2
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        batch_points = grid_points[i:i_end]
        
        @partial(jax.vmap, in_axes=(0, None))
        def get_grads(r1, r2):
            return jastrow_factor.grad_r(r1, r2, jastrow_params)  # Added params argument
        
        u_grad_batch = jax.vmap(get_grads, in_axes=(None, 0))(grid_points, batch_points)
        
        tmp = jnp.einsum('ijc,j,kjc->ik', nabla_rho_paired, weights, u_grad_batch)
        result += jnp.dot(
            rho_paired[:, i:i_end] * weights[None, i:i_end],
            tmp.T
        )
    
    return result.T

def calc_K3(rho_paired, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K3 term with memory-efficient batching.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,) for integration weights
        batch_size: Number of grid points to process at once
    """
    Nb2 = rho_paired.shape[0]
    N_grid = grid_points.shape[0]
    result = jnp.zeros((Nb2, Nb2))
    weights = jnp.asarray(weights)
    
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        batch_points = grid_points[i:i_end]
        
        @partial(jax.vmap, in_axes=(0, None))
        def get_grads(r1, r2):
            return jastrow_factor.grad_r(r1, r2, jastrow_params)  # Added params argument
        
        u_grad_batch = jax.vmap(get_grads, in_axes=(None, 0))(grid_points, batch_points)
        u_grad_squared = jnp.sum(u_grad_batch**2, axis=-1)
        
        weighted_u_squared = (u_grad_squared * 
                            weights[None, :] *
                            weights[i:i_end, None]
                            )
        
        result += jnp.dot(
            rho_paired[:, i:i_end],
            jnp.dot(weighted_u_squared, rho_paired.T)
        )
    
    return result
