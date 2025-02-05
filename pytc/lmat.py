"""This module implements the three-electron matrix elements L."""

import numpy as np
from functools import partial
import logging
import time
from tqdm import tqdm

# Create an optimized einsum that always uses the 'optimal' path
einsum = partial(np.einsum, optimize='optimal')


def calc_v_vector(rho_paired, jastrow_factor, grid_points, weights, batch_size=3000):
    """Compute the intermediate vector V_qt(r₁) using batched processing.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid) containing orbital products
        jastrow_factor: Jastrow instance for computing gradients
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,)
        batch_size: Integer controlling batch size
        
    Returns:
        Array of shape (Nb*Nb, N_grid, 3) containing V_qt(r₁) vectors
    """
    start_time = time.perf_counter()
    
    N_grid = len(grid_points)
    result = np.zeros((rho_paired.shape[0], N_grid, 3))
    
    # Weight the rho for r₂ integration once
    weighted_rho = rho_paired * weights[None, :]  # (Nb^2, N_grid)
    
    logging.info(f"Starting v_vector calculation with {N_grid} grid points in batches of {batch_size}")
    # Process grid points in batches with progress bar
    pbar = tqdm(range(0, N_grid, batch_size), desc="Computing v_vector")
    for i in pbar:
        i_end = min(i + batch_size, N_grid)
        progress = (i + batch_size) / N_grid * 100
        logging.info(f"v_vector progress: {progress:.1f}% (points {i} to {i_end})")
        
        batch_points = grid_points[i:i_end]
        
        # Get Jastrow gradients for this batch
        # TODO: This step is not using multiple threads.
        u_grad_batch = jastrow_factor.grad(batch_points, grid_points)  # (batch, N_grid, 3)
        
        # Process each spatial component separately using np.dot
        for c in range(3):
            # Extract the c-th component: (batch, N_grid)
            u_grad_c = u_grad_batch[..., c]
            # Compute v_vector for this component: (Nb^2, batch)
            result[:, i:i_end, c] = np.dot(weighted_rho, u_grad_c.T)
    
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"v_vector calculation completed in {elapsed_time:.2f} seconds")
    return result


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
