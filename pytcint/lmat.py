"""This module implements the three-electron matrix elements L."""

import numpy as np
from functools import partial

# Create an optimized einsum that always uses the 'optimal' path
einsum = partial(np.einsum, optimize='optimal')


def calc_v_vector(rho_paired, u_gradients, weights):
    """Compute the intermediate vector V_qt(r₁).
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid) containing orbital products
        u_gradients: Array of shape (N_grid, N_grid, 3) containing Jastrow gradients
        weights: Array of shape (N_grid,) containing the weights
        
    Returns:
        Array of shape (Nb*Nb, N_grid, 3) containing V_qt(r₁) vectors
    """
    # Multiply gradients by weights for r₂ integration
    weighted_grads = u_gradients * weights[None, :, None]  # Shape: (N_grid, N_grid, 3)
    
    # Compute V_qt(r₁) by summing over r₂
    v_vector = einsum('ijk,li->ljk', weighted_grads, rho_paired)
    
    return v_vector


def calc_L(rho_paired, v_bra, weights, v_ket=None):
    """Compute the three-electron matrix element L^{pqr(123)}_{stu}.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid) containing orbital products
        v_bra: Array of shape (Nb*Nb, N_grid, 3) containing bra-side vectors
        weights: Array of shape (N_grid,) containing the weights
        v_ket: Optional array of shape (Nb*Nb, N_grid, 3), defaults to v_bra
        
    Returns:
        Array of shape (Nb*Nb, Nb*Nb, Nb*Nb) containing the L matrix elements
    """
    if v_ket is None:
        v_ket = v_bra
    
    # Multiply by weights and rho_paired
    result = einsum('in,jnd,knd,n->ijk', rho_paired, v_bra, v_ket, weights)
    
    return result


def calc_L_symmetric(rho_paired, v_bra, weights, v_ket=None):
    """Compute the symmetrized three-electron matrix elements.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid) containing orbital products
        v_bra: Array of shape (Nb*Nb, N_grid, 3) containing bra-side vectors
        weights: Array of shape (N_grid,) containing the weights
        v_ket: Optional array of shape (Nb*Nb, N_grid, 3), defaults to v_bra
        
    Returns:
        l_sym: Array of shape (Nb*Nb, Nb*Nb, Nb*Nb) 
        containing the symmetrized L matrix elements
        
    """
    # First compute the base L matrix
    l_mat = calc_L(rho_paired, v_bra, weights, v_ket)
    
    # Reshape for permutations (assuming Nb*Nb = N²)
    N = rho_paired.shape[0]
    l_mat = l_mat.reshape(N, N, N)
    
    # Add permutations
    l_sym = (
        l_mat +  # Original
        l_mat.transpose(0, 2, 1) +  # Permute second and third electron
        l_mat.transpose(2, 1, 0)    # Cyclic permutation
    )
    
    return l_sym
