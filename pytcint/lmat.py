"""This module implements the three-electron matrix elements L."""

import numpy as np
from . import jastrow


def calc_v_vector(rho_qt, grid_points, weights, jastrow_factor):
    """Compute the intermediate vector V_qt(r₁) = ∫ϕₑ(r₂)∇₁u(r₁,r₂)ϕₜ(r₂)dr₂.
    
    This is the first step in computing the three-electron matrix elements,
    integrating over r₂ first to create an intermediate vector.
    
    Args:
        rho_qt: Array of shape (Nb*Nb, N_grid) containing ϕₑϕₜ values
        grid_points: Array of shape (N_grid, 3) containing the grid points
        weights: Array of shape (N_grid,) containing the weights for each grid point
        jastrow_factor: Instance of Jastrow class
        
    Returns:
        Array of shape (Nb*Nb, N_grid, 3) containing V_qt(r₁) vectors
    """
    # Get gradients of u with respect to r₁
    u_gradients = jastrow_factor.grad(grid_points)  # Shape: (N_grid, N_grid, 3)
    
    # For each r₁, multiply gradients by weights and rho_qt(r₂), then sum over r₂
    # u_gradients[:, r1, :] gives ∇₁u(r₁,r₂) for fixed r₁
    weighted_grads = u_gradients * weights[np.newaxis, :, np.newaxis]  # Shape: (N_grid, N_grid, 3)
    
    # Compute V_qt(r₁) by summing over r₂
    # rho_qt: (Nb*Nb, N_grid) -> (Nb*Nb, 1, N_grid)
    # weighted_grads: (N_grid, N_grid, 3)
    v_vector = np.sum(
        rho_qt[:, np.newaxis, :, np.newaxis] * weighted_grads[np.newaxis, :, :, :],
        axis=2
    )  # Shape: (Nb*Nb, N_grid, 3)
    
    return v_vector


def calc_L(rho_p, v_qt, v_ru, rho_s, grid_points, weights):
    """Compute the three-electron matrix element L^{pqr(123)}_{stu}.
    
    Uses the intermediate vectors V_qt and V_ru to compute:
    L^{pqr(123)}_{stu} = ∫ϕₚ(r₁)V_qt(r₁)·V_ru(r₁)ϕₛ(r₁)dr₁
    
    Args:
        rho_p: Array of shape (Nb, N_grid) containing ϕₚ values
        v_qt: Array of shape (Nb*Nb, N_grid, 3) containing V_qt vectors
        v_ru: Array of shape (Nb*Nb, N_grid, 3) containing V_ru vectors
        rho_s: Array of shape (Nb, N_grid) containing ϕₛ values
        grid_points: Array of shape (N_grid, 3) containing the grid points
        weights: Array of shape (N_grid,) containing the weights for each grid point
        
    Returns:
        Array of shape (Nb, Nb*Nb, Nb*Nb) containing the L matrix elements
    """
    # Compute dot product between V_qt and V_ru vectors at each r₁
    v_dot_v = np.sum(v_qt[:, :, :] * v_ru[:, :, :], axis=2)  # Shape: (Nb*Nb, N_grid)
    
    # Multiply by weights and basis functions ϕₚ(r₁)ϕₛ(r₁)
    weighted_integrand = (
        rho_p[:, :, np.newaxis] *  # Shape: (Nb, N_grid, 1)
        v_dot_v[np.newaxis, :, :] *  # Shape: (1, N_grid, Nb*Nb)
        rho_s[:, :, np.newaxis] *  # Shape: (Nb, N_grid, 1)
        weights[np.newaxis, :, np.newaxis]  # Shape: (1, N_grid, 1)
    )
    
    # Sum over grid points r₁
    result = np.sum(weighted_integrand, axis=1)  # Shape: (Nb, Nb*Nb)
    
    # Reshape to final form
    Nb = rho_p.shape[0]
    return result.reshape(Nb, Nb, Nb, Nb, Nb, Nb)


def calc_L_symmetric(rho_p, v_qt, v_ru, rho_s, grid_points, weights):
    """Compute the symmetrized three-electron matrix elements.
    
    Computes L^{pqr123}_{stu} + L^{pqr312}_{stu} + L^{pqr231}_{stu} with
    unique indices p ≥ s, q ≥ t, r ≥ u, ps ≥ qt ≥ ru and pqr ≥ stu.
    
    Args:
        Same as compute_l_matrix
        
    Returns:
        Array containing the symmetrized L matrix elements for unique indices
    """
    # First compute the base L matrix
    l_mat = compute_l_matrix(rho_p, v_qt, v_ru, rho_s, grid_points, weights)
    
    # Add permutations
    l_sym = (
        l_mat +  # L^{pqr123}_{stu}
        l_mat.transpose(0, 1, 2, 3, 4, 5) +  # L^{pqr312}_{stu}
        l_mat.transpose(0, 2, 1, 3, 5, 4)    # L^{pqr231}_{stu}
    )
    
    # Extract unique elements based on symmetry conditions
    # This part needs to be implemented based on specific requirements
    # for handling the unique indices p ≥ s, q ≥ t, r ≥ u, ps ≥ qt ≥ ru, pqr ≥ stu
    
    return l_sym

