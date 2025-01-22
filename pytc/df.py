"""
This module implements the density-fitting for transcorrelated integrals.
"""
import numpy as np
from scipy.linalg import qr

import numpy as np
from scipy.linalg import solve_triangular

def pivoted_cholesky(M, n_rank):
    """Pivoted Cholesky decomposition with fixed rank.
    
    Args:
        M: Input matrix to decompose (positive semi-definite)
        n_rank: Number of pivots to select
        
    Returns:
        L: Lower triangular factor
        piv: Selected pivot indices
    """
    n = M.shape[0]
    perm = np.arange(n)
    L = np.zeros((n, n))
    d = np.diag(M).copy()  # Diagonal elements
    
    for k in range(n_rank):
        # Find maximum diagonal element
        if k > 0:
            d[perm[k:]] = np.diag(M)[perm[k:]] - np.sum(L[perm[k:], :k]**2, axis=1)
        
        # Select pivot
        pivot = k + np.argmax(d[perm[k:]])
        if pivot != k:
            perm[k], perm[pivot] = perm[pivot], perm[k]
            
        # Update L[:, k]
        L[perm[k], k] = np.sqrt(d[perm[k]])
        if k < n_rank - 1:
            row_k = M[perm[k], perm[k+1:]] - L[perm[k], :k] @ L[perm[k+1:], :k].T
            L[perm[k+1:], k] = row_k / L[perm[k], k]
            
    return L[:, :n_rank], perm[:n_rank]

def isdf_decompose_cholesky(rho, n_rank):
    """ISDF decomposition using pivoted Cholesky.
    
    Args:
        rho: Input density matrix (N_b^2, N_grid)
        n_rank: Number of interpolation points
        
    Returns:
        C: Selected columns from rho (N_b^2, n_rank)
        xi: Interpolation coefficients (n_rank, N_grid)
    """
    N_b_sq, N_grid = rho.shape
    
    # Form overlap matrix S = rho.T @ rho
    S = rho.T @ rho
    
    # Add small diagonal shift for stability
    shift = 1e-12 * np.max(np.abs(np.diag(S)))
    S[np.diag_indices_from(S)] += shift
    
    # Get interpolation points via pivoted Cholesky
    _, piv = pivoted_cholesky(S, n_rank)
    
    # Select columns from original rho
    C = rho[:, piv]
    
    # Solve least squares problem for interpolation coefficients
    # min_xi ||rho - C @ xi||
    xi, *_ = np.linalg.lstsq(C, rho, rcond=None)
    
    return C, xi

def isdf_decompose_multi(rho1, rho2, n_rank1, n_rank2):
    """ISDF decomposition for two densities with pivot fusion.
    
    Args:
        rho1: First density matrix (N_b1^2, N_grid)
        rho2: Second density matrix (N_b2^2, N_grid)
        n_rank1: Number of interpolation points for rho1
        n_rank2: Number of interpolation points for rho2
        
    Returns:
        C1: Selected columns from rho1 using fused pivots
        xi1: Interpolation coefficients for rho1
        C2: Selected columns from rho2 using fused pivots
        xi2: Interpolation coefficients for rho2
        piv_fused: Combined pivot indices
    """
    # Get pivots for each density separately
    _, piv1 = pivoted_cholesky(rho1.T @ rho1, n_rank1)
    _, piv2 = pivoted_cholesky(rho2.T @ rho2, n_rank2)
    
    # Combine and uniquify pivots
    piv_fused = np.unique(np.concatenate([piv1, piv2]))
    
    # Select columns using fused pivots
    C1 = rho1[:, piv_fused]
    C2 = rho2[:, piv_fused]
    
    # Solve least squares problems
    xi1, *_ = np.linalg.lstsq(C1, rho1, rcond=None)
    xi2, *_ = np.linalg.lstsq(C2, rho2, rcond=None)
    
    return C1, xi1, C2, xi2, piv_fused

def reconstruct_rho(C, xi):
    """Reconstruct density using interpolation."""
    return C @ xi

def test_accuracy(rho_orig, C, P):
    """Calculate reconstruction relative error."""
    rho_recon = reconstruct_rho(C, P)
    return np.linalg.norm(rho_recon - rho_orig) 

def test_multi_accuracy(rho1, rho2, n_rank1, n_rank2):
    """Test reconstruction accuracy for two densities using fused pivots.
    
    Args:
        rho1, rho2: Input densities to approximate
        n_rank1, n_rank2: Desired ranks for each density
    
    Returns:
        error1: Relative error for rho1 reconstruction
        error2: Relative error for rho2 reconstruction
        n_fused: Number of fused pivots used
    """
    C1, xi1, C2, xi2, piv_fused = isdf_decompose_multi(rho1, rho2, n_rank1, n_rank2)
    
    # Calculate reconstruction errors
    error1 = np.linalg.norm(rho1 - C1 @ xi1) / np.linalg.norm(rho1)
    error2 = np.linalg.norm(rho2 - C2 @ xi2) / np.linalg.norm(rho2)
    
    return error1, error2, len(piv_fused)

# Example usage
if __name__ == "__main__":
    # Generate test data
    Nb = 10
    N_grid = 500
    rank = 5
    U = np.random.randn(Nb**2, rank)
    V = np.random.randn(N_grid, rank)
    rho = U @ V.T  # Construct low-rank matrix
    
    # Perform Cholesky-based ISDF decomposition with fixed rank
    C, P = isdf_decompose_cholesky(rho, n_rank=rank)
    
    # Test reconstruction accuracy
    error = test_accuracy(rho, C, P)
    print(f"Reconstruction relative error: {error:.2e}")
    print(f"Number of auxiliary basis: {C.shape[1]}")
    
    # Test with two densities of different ranks
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