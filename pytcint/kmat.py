import numpy as np
from . import jastrow


def calc_K1(rho, nabla_rho, grid_points, weights, jastrow_factor):
    """Evaluate the integral <pq|∇u·∇|rs> over grid points using broadcasting.
    
    Args:
        rho: Array of shape (N_grid, Nb*Nb) containing the density matrix elements
        nabla_rho: Array of shape (N_grid, Nb*Nb, 3) containing the gradients
        grid_points: Array of shape (N_grid, 3) containing the grid points
        weights: Array of shape (N_grid,) containing the weights for each grid point
        jastrow_factor: Instance of Jastrow class
        
    Returns:
        Array of shape (Nb*Nb, Nb*Nb) containing the integrated values
    """
    # Get gradients of u with respect to all grid points
    u_gradients = jastrow_factor.grad(grid_points)  # Shape: (N_grid, N_grid, 3)
    
    # Step 1: Sum over r' first at each r - O(N_grid^2 * Nb^2)
    # Multiply gradients by weights for r' integration
    weighted_grads = u_gradients * weights[np.newaxis, :, np.newaxis]  # Shape: (N_grid, N_grid, 3)
    
    # For each r, compute dot product with nabla_rho(r') and sum over r'
    # weighted_grads: (N_grid, N_grid, 3)
    # nabla_rho: (N_grid, Nb*Nb, 3)
    # Sum over j (r') and k (xyz components)
    intermediate = np.einsum('ijk,jlk->il', weighted_grads, nabla_rho)  # Shape: (N_grid, Nb*Nb)
    
    # Step 2: Final summation - O(N_grid * Nb^4)
    # For each r point, multiply rho(r) and intermediate(r) and sum
    # rho: (N_grid, Nb*Nb)
    # intermediate: (N_grid, Nb*Nb)
    # weights: (N_grid,)
    #(qs|pr) -> (pr|qs)
    result = np.einsum('i,ij,ik->jk', weights, rho, intermediate).swapaxes(0, 1)  # Shape: (Nb*Nb, Nb*Nb)
    
    return result

def calc_K2(rho, nabla_rho, grid_points, weights, jastrow_factor):
    """Evaluate the integral <pq|∇²₁u(r₁,r₂)|rs> using integration by parts.
       
    
    Args:
        rho: Array of shape (N_grid, Nb*Nb) containing ϕₚϕᵣ values
        nabla_rho: Array of shape (N_grid, Nb*Nb, 3) containing ∇ϕₑϕₛ values
        grid_points: Array of shape (N_grid, 3) containing the grid points
        weights: Array of shape (N_grid,) containing the weights for each grid point
        jastrow_factor: Instance of Jastrow class
        
    Returns:
        Array of shape (Nb*Nb, Nb*Nb) containing the integrated values
    """
    # Get the transposed term by reshaping and transposing q,s indices
    Nb = int(np.sqrt(nabla_rho.shape[1]))  # Extract Nb from the shape
    
    # Reshape to (N_grid, Nb, Nb, 3) to perform the transposition
    nabla_rho_reshaped = nabla_rho.reshape(-1, Nb, Nb, 3)
    nabla_rho_transposed = nabla_rho_reshaped.transpose(0, 2, 1, 3)
    
    # Reshape back to (N_grid, Nb*Nb, 3)
    nabla_rho_transposed = nabla_rho_transposed.reshape(-1, Nb*Nb, 3)
    
    # Sum the two terms: ∇ϕₑϕₛ + ϕₑ∇ϕₛ
    combined_nabla = nabla_rho + nabla_rho_transposed
    
    # Use nabla_u_nabla to compute the integral with the combined gradient
    # The negative sign comes from the integration by parts
    # change to r1 first and r2 second (pr|qs) chemists notation
    result = -calc_K1(rho, combined_nabla, grid_points, weights, jastrow_factor)
    
    
    return result

def calc_K3(rho_bra, grid_points, weights, jastrow_factor, rho_ket=None):
    """Evaluate the integral <pq|(∇₁u(r₁,r₂))²|rs> over grid points.
    
    Args:
        rho_bra: Array of shape (N_grid, Nb*Nb) containing ϕₚϕᵣ values
        grid_points: Array of shape (N_grid, 3) containing the grid points
        weights: Array of shape (N_grid,) containing the weights for each grid point
        jastrow_factor: Instance of Jastrow class
        rho_ket: Array of shape (N_grid, Nb*Nb) containing ϕₑϕₛ values, if None uses rho_bra
        
    Returns:
        Array of shape (Nb*Nb, Nb*Nb) containing the integrated values
    """
    # Get gradients of u with respect to all grid points
    u_gradients = jastrow_factor.grad(grid_points)  # Shape: (N_grid, N_grid, 3)
    
    # Compute the squared magnitude of the gradient
    u_grad_squared = np.sum(u_gradients**2, axis=-1)  # Shape: (N_grid, N_grid)
    
    # Multiply by weights
    weighted_u_squared = u_grad_squared * weights[:, np.newaxis] * weights[np.newaxis, :]
    
    # If rho_ket is not provided, use rho_bra
    if rho_ket is None:
        rho_ket = rho_bra
    
    # Compute final integral using einsum for efficiency
    result = np.einsum('ij,ik,jl->kl', weighted_u_squared, rho_bra, rho_ket)
    
    
    return result

