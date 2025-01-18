import numpy as np


def calc_K1(rho_paired, nabla_rho_paired, u_gradients, weights):
    """Evaluate the integral <pq|∇u·∇|rs> over grid points using broadcasting.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid)
        nabla_rho_paired: Array of shape (Nb*Nb, N_grid, 3)
        u_gradients: Array of shape (N_grid, N_grid, 3)
        weights: Array of shape (N_grid,)
    """
    # Step 1: Sum over r' first at each r - O(N_grid^2 * Nb^2)
    # Multiply gradients by weights for r' integration
    weighted_grads = u_gradients * weights[np.newaxis, :, np.newaxis]  # Shape: (N_grid, N_grid, 3)
    
    # For each r, compute dot product with nabla_rho(r') and sum over r'
    intermediate = np.einsum('ijk,ljk->li', weighted_grads, nabla_rho_paired)  # Shape: (Nb*Nb, N_grid)
    
    # Step 2: Final summation - O(N_grid * Nb^4)
    # (qs|pr) -> (pr|qs) : swap indices to follow chemists' notation for two-electron integrals
    result = np.einsum('pi,qi,i->pq', rho_paired, intermediate, weights).swapaxes(0, 1)
    
    return result

def calc_K2(rho_paired, nabla_rho_paired, u_gradients, weights):
    """Evaluate the integral <pq|∇²₁u(r₁,r₂)|rs> using integration by parts.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid)
        nabla_rho_paired: Array of shape (Nb*Nb, N_grid, 3)
        u_gradients: Array of shape (N_grid, N_grid, 3)
        weights: Array of shape (N_grid,)
    """
    # Get the transposed term by reshaping and transposing q,s indices
    Nb = int(np.sqrt(nabla_rho_paired.shape[0]))
    
    # Reshape to (Nb, Nb, N_grid, 3) to perform the transposition
    nabla_rho_reshaped = nabla_rho_paired.reshape(Nb, Nb, -1, 3)
    nabla_rho_transposed = nabla_rho_reshaped.transpose(1, 0, 2, 3)
    nabla_rho_transposed = nabla_rho_transposed.reshape(-1, nabla_rho_paired.shape[1], 3)
    
    # Sum the two terms: ∇ϕₑϕₛ + ϕₑ∇ϕₛ
    combined_nabla = nabla_rho_paired + nabla_rho_transposed
    
    # The negative sign comes from the integration by parts
    # Change to r1 first and r2 second (pr|qs) chemists notation
    result = -calc_K1(rho_paired, combined_nabla, u_gradients, weights)
    
    return result

def calc_K3(rho_paired, u_gradients, weights):
    """Evaluate the integral <pq|(∇₁u(r₁,r₂))²|rs> over grid points.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid)
        u_gradients: Array of shape (N_grid, N_grid, 3)
        weights: Array of shape (N_grid,)
    """
    # Compute the squared magnitude of the gradient
    u_grad_squared = np.sum(u_gradients**2, axis=-1)  # Shape: (N_grid, N_grid)
    
    # Multiply by weights for both r and r'
    weighted_u_squared = u_grad_squared * weights[:, np.newaxis] * weights[np.newaxis, :]
    
    # Compute final integral using einsum for efficiency
    result = np.einsum('pi,ij,qj->pq', rho_paired, weighted_u_squared, rho_paired)
    
    return result

