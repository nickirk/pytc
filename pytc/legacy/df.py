"""
This module implements the density-fitting for transcorrelated integrals.
"""
import numpy as np
from typing import Tuple 
import time  
import logging

logger = logging.getLogger(__name__)

def calculate_norm(rho: np.ndarray) -> np.ndarray:
    """Calculate the norm of the input tensor along trailing dimensions.
    Args:
        rho: Input tensor (N_b, N_grid, *trailing_dims) or (N_b, N_grid)
    Returns:
        rho_normed: Normed tensor (N_b, N_grid)
    """
    if rho.ndim > 2:
        return np.linalg.norm(rho.reshape(rho.shape[0], rho.shape[1], -1), axis=-1)
    return rho

def pivoted_cholesky(M: np.ndarray, n_rank: int, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Pivoted Cholesky decomposition with fixed rank.
    
    Args:
        M: Input matrix to decompose (positive semi-definite)
        n_rank: Number of pivots to select
        tol: Tolerance for early termination and numerical stability
    
    Returns:
        L: Lower triangular factor
        piv: Selected pivot indices
    """
    start_time = time.time()
    n = M.shape[0]
    
    perm = np.arange(n)
    L = np.zeros((n, n))
    d = np.diag(M).copy()
    
    # Initial error is the trace
    initial_err = np.sum(d)
    current_err = initial_err
    
    if initial_err < 0:
        raise ValueError("Input matrix must be positive semi-definite")
    
    for k in range(min(n_rank, n)):
        if k > 0:
            d[perm[k:]] = np.diag(M)[perm[k:]] - np.sum(L[perm[k:], :k]**2, axis=1)
        
        max_val = np.max(d[perm[k:]])
        
        if max_val < tol:
            logger.warning(f"Small pivot encountered at step {k}: {max_val:.2e}")
            end_time = time.time()
            logger.info(f"Pivoted Cholesky decomposition took {end_time - start_time:.2f} seconds")
            return L[:, :k], perm[:k]
        
        pivot = k + np.argmax(d[perm[k:]])
        
        if pivot != k:
            perm[k], perm[pivot] = perm[pivot], perm[k]
        
        L[perm[k], k] = np.sqrt(d[perm[k]])
        
        if k < n_rank - 1:
            row_k = M[perm[k], perm[k+1:]] - L[perm[k], :k] @ L[perm[k+1:], :k].T
            L[perm[k+1:], k] = row_k / L[perm[k], k]
        
        current_err = np.sum(d[perm[k+1:]])
        rel_err = current_err / initial_err
        
        if rel_err < tol:
            logger.info(f"Converged at step {k} with relative error {rel_err:.2e}")
            end_time = time.time()
            logger.info(f"Pivoted Cholesky decomposition took {end_time - start_time:.2f} seconds")
            return L[:, :k+1], perm[:k+1]
    
    end_time = time.time()
    logger.info(f"Pivoted Cholesky decomposition took {end_time - start_time:.2f} seconds")
    return L[:, :n_rank], perm[:n_rank]

def solve_least_squares(C: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Solve least squares problem for interpolation coefficients.
    Args:
        C: Selected columns (N_b*N_b, n_rank, *trailing_dims)
        rho: Original tensor (N_b*N_b, N_grid, *trailing_dims)

    Returns:
        xi: Interpolation coefficients (n_rank, N_grid, *trailing_dims)
    """
    if C.ndim > 2:
        xi_list = []
        for i in range(rho.shape[-1]):
            xi_i, *_ = np.linalg.lstsq(C[..., i], rho[..., i], rcond=None)
            xi_list.append(xi_i)
        return np.stack(xi_list, axis=-1)
    else:
        xi, *_ = np.linalg.lstsq(C, rho, rcond=None)
        return xi

def isdf_decompose_cholesky(rho: np.ndarray, n_rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """ISDF decomposition using pivoted Cholesky, generalized for tensors.
    Args:
        rho: Input tensor (N_b, N_grid, *trailing_dims) or (N_b, N_grid)
        n_rank: Number of interpolation points
    Returns:
        C: Selected columns from rho (N_b, n_rank, *trailing_dims) or (N_b, n_rank)
        xi: Interpolation coefficients (n_rank, N_grid, *trailing_dims) or (n_rank, N_grid)
    """
    start_time = time.time()
    rho_normed = calculate_norm(rho)
    S = rho_normed.T @ rho_normed
    S[np.diag_indices_from(S)] += 1e-12 * np.max(np.abs(np.diag(S)))

    _, piv = pivoted_cholesky(S, n_rank)
    C = np.take(rho, piv, axis=1)
    xi = solve_least_squares(C, rho)

    end_time = time.time()
    logger.info(f"ISDF decomposition took {end_time - start_time:.2f} seconds")
    return C, xi

def isdf_decompose_multi(rho1: np.ndarray, rho2: np.ndarray, n_rank1: int, n_rank2: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ISDF decomposition for two densities with pivot fusion.

    Args:
        rho1: First overlap density (N_b*N_b, N_grid, *trailing_dims)
        rho2: Second overlap density (N_b*N_b, N_grid, *trailing_dims)
        n_rank1: Number of interpolation points for rho1
        n_rank2: Number of interpolation points for rho2

    Returns:
        C1: Selected columns from rho1 using fused pivots (N_b*N_b, n_fused, *trailing_dims)
        xi1: Interpolation coefficients for rho1 (n_fused, N_grid, *trailing_dims)
        C2: Selected columns from rho2 using fused pivots (N_b*N_b, n_fused, *trailing_dims)
        xi2: Interpolation coefficients for rho2 (n_fused, N_grid, *trailing_dims)
        piv_fused: Combined pivot indices
    """
    rho1_normed = calculate_norm(rho1)
    rho2_normed = calculate_norm(rho2)

    S1 = rho1_normed.T @ rho1_normed
    S2 = rho2_normed.T @ rho2_normed

    # Add small diagonal shift for stability
    shift1 = 1e-12 * np.max(np.abs(np.diag(S1)))
    shift2 = 1e-12 * np.max(np.abs(np.diag(S2)))
    S1[np.diag_indices_from(S1)] += shift1
    S2[np.diag_indices_from(S2)] += shift2

    _, piv1 = pivoted_cholesky(S1, n_rank1)
    _, piv2 = pivoted_cholesky(S2, n_rank2)

    piv_fused = np.unique(np.concatenate([piv1, piv2]))

    if rho1.ndim > 2:
        C1 = np.take(rho1, piv_fused, axis=1)
    else:
        C1 = rho1[:, piv_fused]

    if rho2.ndim > 2:
        C2 = np.take(rho2, piv_fused, axis=1)
    else:
        C2 = rho2[:, piv_fused]

    xi1 = solve_least_squares(C1, rho1)
    xi2 = solve_least_squares(C2, rho2)

    return C1, xi1, C2, xi2, piv_fused

def reconstruct_rho(C: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Reconstruct tensor from decomposition.

    Args:
        C: Selected columns (N_b, n_rank, *trailing_dims)
        xi: Interpolation coefficients (n_rank, N_grid, *trailing_dims)

    Returns:
        Reconstructed tensor with same shape as original
    """
    if C.ndim > 2:
        return np.einsum('br...,rg...->bg...', C, xi)
    else:
        return C @ xi

def test_accuracy(rho_orig: np.ndarray, C: np.ndarray, P: np.ndarray) -> Tuple[float, float]:
    """Calculate reconstruction relative and absolute errors.
    
    Args:
        rho_orig: Original tensor (N_b, N_grid, *trailing_dims)
        C: Selected columns (N_b, n_rank, *trailing_dims)
        P: Interpolation coefficients (n_rank, N_grid, *trailing_dims)
        
    Returns:
        rel_error: Relative error of reconstruction
        abs_error: Absolute error of reconstruction
    """
    rho_recon = reconstruct_rho(C, P)
    abs_error = np.linalg.norm(rho_recon - rho_orig)
    rel_error = abs_error / np.linalg.norm(rho_orig)
    return rel_error, abs_error

def test_multi_accuracy(rho1: np.ndarray, rho2: np.ndarray, n_rank1: int, n_rank2: int) -> Tuple[float, float, float, float, int]:
    """Test reconstruction accuracy for two densities using fused pivots.
    
    Args:
        rho1, rho2: Input densities to approximate
        n_rank1, n_rank2: Desired ranks for each density
    
    Returns:
        rel_error1: Relative error for rho1 reconstruction
        abs_error1: Absolute error for rho1 reconstruction
        rel_error2: Relative error for rho2 reconstruction
        abs_error2: Absolute error for rho2 reconstruction
        n_fused: Number of fused pivots used
    """
    C1, xi1, C2, xi2, piv_fused = isdf_decompose_multi(rho1, rho2, n_rank1, n_rank2)
    
    rel_error1, abs_error1 = test_accuracy(rho1, C1, xi1)
    rel_error2, abs_error2 = test_accuracy(rho2, C2, xi2)
    
    return rel_error1, abs_error1, rel_error2, abs_error2, len(piv_fused)

if __name__ == "__main__":
    Nb = 10
    N_grid = 500
    rank = 5
    U = np.random.randn(Nb**2, rank)
    V = np.random.randn(N_grid, rank)
    rho = U @ V.T

    C, P = isdf_decompose_cholesky(rho, n_rank=rank)

    error = test_accuracy(rho, C, P)
    print(f"Reconstruction relative error: {error:.2e}")
    print(f"Number of auxiliary basis: {C.shape[1]}")

    rank1, rank2 = 5, 8
    U1 = np.random.randn(Nb**2, rank1)
    U2 = np.random.randn(Nb**2, rank2)
    V1 = np.random.randn(N_grid, rank1)
    V2 = np.random.randn(N_grid, rank2)
    rho1 = U1 @ V1.T
    rho2 = U2 @ V2.T

    err1, err2, n_fused = test_multi_accuracy(rho1, rho2, rank1, rank2)
    print(f"Rho1 error: {err1:.2e}")
    print(f"Rho2 error: {err2:.2e}")
    print(f"Number of fused pivots: {n_fused}")