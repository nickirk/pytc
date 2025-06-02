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
            return jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]  # Added params argument
        
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
            return jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]  # Added params argument
        
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


def calc_K1_isdf(C_rho, xi_rho, C_grad, xi_grad, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K1 term using ISDF intermediates with memory-efficient batching.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        C_grad: (Nb^2, n_fused, 3) Selected columns for nabla_rho
        xi_grad: (n_fused, N_grid, 3) Interpolation coeffs for grad
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    N_grid = grid_points.shape[0]
    Nb2 = C_rho.shape[0]
    n_fused = xi_rho.shape[0]
    
    result = jnp.zeros((Nb2, Nb2))
    weights = jnp.asarray(weights)
    weighted_xi_grad = xi_grad * weights[None, :, None]
    
    # Process r2 points in batches
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        batch_points = grid_points[i:i_end]
        
        @partial(jax.vmap, in_axes=(None, 0))
        def get_grads(r1, r2):
            return jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]
        
        u_grad_batch = jax.vmap(get_grads, in_axes=(0, None))(grid_points, batch_points)  # (N_grid, batch, 3)
        
        # Process each spatial component
        G1_components = []
        for c in range(3):
            xi_slice = weighted_xi_grad[:, :, c]  # (n_fused, N_grid)
            u_slice = u_grad_batch[:, :, c]      # (N_grid, batch)
            G1_c = jnp.dot(xi_slice, u_slice)     # (n_fused, batch)
            G1_components.append(G1_c)
        
        G1 = jnp.stack(G1_components, axis=-1)    # (n_fused, batch, 3)
        G1 = jnp.einsum('kmc,pkc->pm', G1, C_grad)  # (Nb^2, batch)
        
        # Contract with xi_rho and weights for this batch
        G2 = jnp.einsum('pm,lm,m->pl', G1, xi_rho[:, i:i_end], weights[i:i_end])
        
        # Accumulate result
        result += jnp.einsum('pl,ql->pq', G2, C_rho)
    
    return result


def calc_K2_isdf(C_rho, xi_rho, C_grad, xi_grad, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K2 term using ISDF intermediates.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        C_grad: (Nb^2, n_fused, 3) Selected columns for nabla_rho
        xi_grad: (n_fused, N_grid, 3) Interpolation coeffs for grad
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    # Step 1: Compute the combined nabla term (∇ϕₑϕₛ + ϕₑ∇ϕₛ)
    Nb = int(jnp.sqrt(C_grad.shape[0]))
    C_grad_reshaped = C_grad.reshape(Nb, Nb, -1, 3)
    C_grad_transposed = C_grad_reshaped.transpose(1, 0, 2, 3).reshape(-1, C_grad.shape[1], 3)
    combined_C_grad = C_grad + C_grad_transposed

    # Step 2: Reuse calc_K1_isdf with the combined gradient term
    result = -calc_K1_isdf(C_rho, xi_rho, combined_C_grad, xi_grad,
                          jastrow_factor, jastrow_params, grid_points, weights, batch_size)

    return result


def calc_K3_isdf(C_rho, xi_rho, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K3 term using ISDF intermediates with memory-efficient batching.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    N_grid = grid_points.shape[0]
    Nb2 = C_rho.shape[0]
    n_fused = xi_rho.shape[0]
    
    result = jnp.zeros((Nb2, Nb2))
    weights = jnp.asarray(weights)
    
    # Process r2 points in batches
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        batch_points = grid_points[i:i_end]
        
        @partial(jax.vmap, in_axes=(None, 0))
        def get_grads(r1, r2):
            return jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]
        
        u_grad_batch = jax.vmap(get_grads, in_axes=(0, None))(grid_points, batch_points)  # (N_grid, batch, 3)
        
        # Compute squared magnitude of gradient
        u_grad_squared = jnp.sum(u_grad_batch**2, axis=-1)  # (N_grid, batch)
        
        # Weight both coordinates
        weighted_u_squared = u_grad_squared * weights[:, None] * weights[None, i:i_end]  # (N_grid, batch)
        
        # Contract with xi_rho for this batch
        G1 = jnp.einsum('ki,ij,lj->kl', xi_rho, weighted_u_squared, xi_rho[:, i:i_end])
        
        # Accumulate result
        result += jnp.einsum('kl,pk,ql->pq', G1, C_rho, C_rho)
    
    return result
