import numpy as np
from functools import partial, reduce
import psutil
import logging
import time  # Add this import at the top


einsum = partial(np.einsum, optimize='optimal')

def _get_safe_batch_size(N_grid, Nb2, dtype=np.float64):
    """Calculate safe batch size based on available memory.
    
    Memory requirements for main arrays in calc_K1:
    - u_grad_batch: (N_grid, batch, 3) 
    - tmp: (Nb^2, batch)
    Plus some overhead for intermediate calculations
    """
    bytes_per_elem = np.dtype(dtype).itemsize
    mem = psutil.virtual_memory()
    available_bytes = mem.available * 0.5  # Use 50% of available memory
    
    # Calculate memory for main arrays
    def calc_mem_for_batch(batch_size):
        u_grad_mem = N_grid * batch_size * 3 * bytes_per_elem  # u_grad_batch
        tmp_mem = Nb2 * batch_size * bytes_per_elem            # tmp array
        return u_grad_mem + tmp_mem
    
    # Binary search for largest batch size that fits in memory
    left, right = 1, N_grid
    while left < right:
        mid = (left + right + 1) // 2
        if calc_mem_for_batch(mid) <= available_bytes:
            left = mid
        else:
            right = mid - 1
    
    batch_size = max(1, min(left, 5000))  
    #batch_size = max(1, left)
    logging.info(f"Selected batch_size: {batch_size} based on available memory: {mem.available / 1e9:.2f} GB")
    return batch_size

def calc_K1(rho_paired, nabla_rho_paired, jastrow_factor, grid_points, weights, batch_size=None):
    """Evaluate the integral <pq|∇u·∇|rs> over grid points using batched processing."""
    start_time = time.perf_counter()
    
    N_grid = len(grid_points)
    Nb2 = rho_paired.shape[0]
    
    result = np.zeros((rho_paired.shape[0],) * 2)
    if batch_size is None:
        batch_size = _get_safe_batch_size(N_grid, Nb2)
    
    logging.info(f"Starting K1 calculation with {N_grid} grid points in batches of {batch_size}")
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        progress = i_end / N_grid * 100
        logging.info(f"K1 progress: {progress:.1f}% (points {i} to {i_end})")
        
        # Get Jastrow gradients for this batch against all r1
        u_grad_batch = jastrow_factor.grad(grid_points, grid_points[i:i_end])  # (N_grid, batch, 3)
        
        tmp = np.zeros((rho_paired.shape[0], i_end - i))
        # Loop over c (only 3 iterations)
        for c_idx in range(3):
            # Slice A and B for current c
            A_slice = nabla_rho_paired[:, :, c_idx]  # Shape (i, n)
            B_slice = u_grad_batch[:, :, c_idx]      # Shape (n, k)
        
            # Compute (i, n) @ (n, k) = (i, k) with BLAS acceleration
            tmp += np.dot((A_slice * weights[None, :]), B_slice)  # Scale A by weights, then matmul
    
        
        result += np.dot(rho_paired[:,i:i_end]*weights[None,i:i_end], tmp.T)

    result = result.swapaxes(0, 1)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"calc_K1 completed in {elapsed_time:.2f} seconds")
    return result

def calc_K2(rho_paired, nabla_rho_paired, jastrow_factor, grid_points, weights, batch_size=None):
    """Evaluate the integral <pq|∇²₁u(r₁,r₂)|rs> using batched processing.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid)
        nabla_rho_paired: Array of shape (Nb*Nb, N_grid, 3)
        jastrow_factor: Jastrow instance for computing gradients
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,)
        batch_size: Integer for controlling batch size
    """
    start_time = time.perf_counter()
    
    # Get the transposed term
    Nb = int(np.sqrt(nabla_rho_paired.shape[0]))
    nabla_rho_reshaped = nabla_rho_paired.reshape(Nb, Nb, -1, 3)
    nabla_rho_transposed = nabla_rho_reshaped.transpose(1, 0, 2, 3)
    nabla_rho_transposed = nabla_rho_transposed.reshape(-1, nabla_rho_paired.shape[1], 3)
    
    # Sum the two terms and compute K1 with combined gradient
    combined_nabla = nabla_rho_paired + nabla_rho_transposed
    result = -calc_K1(rho_paired, combined_nabla, jastrow_factor, grid_points, weights, batch_size)
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"calc_K2 completed in {elapsed_time:.2f} seconds")
    return result

def calc_K3(rho_paired, jastrow_factor, grid_points, weights, batch_size=None):
    """Evaluate the integral <pq|(∇₁u(r₁,r₂))²|rs> using batched processing.
    
    Args:
        rho_paired: Array of shape (Nb*Nb, N_grid)
        jastrow_factor: Jastrow instance for computing gradients
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,)
        batch_size: Integer for controlling batch size
    """
    start_time = time.perf_counter()
    
    N_grid = len(grid_points)
    result = np.zeros((rho_paired.shape[0],) * 2)
    # Process grid points in batches
    # Automatically determine batch size if not provided
    if batch_size is None:
        batch_size = _get_safe_batch_size(N_grid, rho_paired.shape[0])
    
    logging.info(f"Starting K3 calculation with {N_grid} grid points in batches of {batch_size}")
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        progress = i_end / N_grid * 100
        logging.info(f"K3 progress: {progress:.1f}% (points {i} to {i_end})")
        batch_points = grid_points[i:i_end]
        
        # Get Jastrow gradients for this batch
        u_grad_batch = jastrow_factor.grad(batch_points, grid_points)
        
        # Compute squared magnitude of gradient
        u_grad_squared = np.sum(u_grad_batch**2, axis=-1)  # (batch, N_grid)
        
        # Weight both coordinates
        weighted_u_squared = u_grad_squared * weights[i:i_end, None] * weights[None, :]
        
        result += reduce(np.dot, (rho_paired[:,i:i_end], weighted_u_squared, rho_paired.T))
    
    result = result
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"calc_K3 completed in {elapsed_time:.2f} seconds")
    return result

def calc_K1_isdf(C_rho, xi_rho, C_grad, xi_grad, jastrow_factor, grid_points, weights, batch_size=None):
    """Directly evaluate <pq|∇u·∇|rs> using ISDF intermediates with batched processing.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        C_grad: (Nb^2, n_fused, 3) Selected columns for nabla_rho
        xi_grad: (n_fused, N_grid, 3) Interpolation coeffs for grad
        jastrow_factor: Jastrow instance for computing gradients
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Optional batch size for r2 coordinate
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    start_time = time.perf_counter()
    
    N_grid = len(grid_points)
    Nb2 = C_rho.shape[0]
    n_fused = xi_rho.shape[0]
    
    if batch_size is None:
        batch_size = _get_safe_batch_size(N_grid, n_fused)
        
    result = np.zeros((Nb2, Nb2))
    weighted_xi_grad = xi_grad * weights[None,:,None]
    
    # Process r2 points in batches
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        batch_size_i = i_end - i
        
        # Get Jastrow gradients for this batch
        u_grad_batch = jastrow_factor.grad(grid_points, grid_points[i:i_end])  # (N_grid, batch, 3)
        
        # Process each spatial component
        G1_components = []
        for c in range(3):
            xi_slice = weighted_xi_grad[:,:,c]           # (n_fused, N_grid)
            u_slice = u_grad_batch[:,:,c]               # (N_grid, batch)
            G1_c = np.dot(xi_slice, u_slice)           # (n_fused, batch)
            G1_components.append(G1_c)
        
        G1 = np.stack(G1_components, axis=-1)          # (n_fused, batch, 3)
        G1 = einsum('kmc,pkc->pm', G1, C_grad)         # (Nb^2, batch)
        
        # Contract with xi_rho and weights for this batch
        G2 = einsum('pm,lm,m->pl', G1, xi_rho[:,i:i_end], weights[i:i_end])
        
        # Accumulate result
        result += einsum('pl,ql->pq', G2, C_rho)
    
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"calc_K1_isdf completed in {elapsed_time:.2f} seconds")
    return result

def calc_K2_isdf(C_rho, xi_rho, C_grad, xi_grad, jastrow_factor, grid_points, weights, batch_size=None):
    """Directly evaluate <pq|∇²₁u(r₁,r₂)|rs> using ISDF intermediates.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        C_grad: (Nb^2, n_fused, 3) Selected columns for nabla_rho
        xi_grad: (n_fused, N_grid, 3) Interpolation coeffs for grad
        jastrow_factor: Jastrow instance for computing gradients
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Optional batch size for r2 coordinate
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    start_time = time.perf_counter()
    
    # Step 1: Compute the combined nabla term (∇ϕₑϕₛ + ϕₑ∇ϕₛ)
    Nb = int(np.sqrt(C_grad.shape[0]))
    C_grad_reshaped = C_grad.reshape(Nb, Nb, -1, 3)
    C_grad_transposed = C_grad_reshaped.transpose(1, 0, 2, 3).reshape(-1, C_grad.shape[1], 3)
    combined_C_grad = C_grad + C_grad_transposed

    # Step 2: Reuse calc_K1_isdf with the combined gradient term
    result = -calc_K1_isdf(C_rho, xi_rho, combined_C_grad, xi_grad,
                          jastrow_factor, grid_points, weights, batch_size)

    elapsed_time = time.perf_counter() - start_time
    logging.info(f"calc_K2_isdf completed in {elapsed_time:.2f} seconds")
    return result

def calc_K3_isdf(C_rho, xi_rho, jastrow_factor, grid_points, weights, batch_size=None):
    """Directly evaluate <pq|(∇₁u(r₁,r₂))²|rs> using ISDF intermediates with batched processing.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        jastrow_factor: Jastrow instance for computing gradients
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Optional batch size for r2 coordinate
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    start_time = time.perf_counter()
    
    N_grid = len(grid_points)
    Nb2 = C_rho.shape[0]
    n_fused = xi_rho.shape[0]
    
    if batch_size is None:
        batch_size = _get_safe_batch_size(N_grid, n_fused)
    
    result = np.zeros((Nb2, Nb2))
    
    # Process r2 points in batches
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        
        # Get Jastrow gradients for this batch
        u_grad_batch = jastrow_factor.grad(grid_points, grid_points[i:i_end])  # (N_grid, batch, 3)
        
        # Compute squared magnitude of gradient
        u_grad_squared = np.sum(u_grad_batch**2, axis=-1)  # (N_grid, batch)
        
        # Weight both coordinates
        weighted_u_squared = u_grad_squared * weights[:, None] * weights[None, i:i_end]  # (N_grid, batch)
        
        # Contract with xi_rho for this batch
        G1 = einsum('ki,ij,lj->kl', xi_rho, weighted_u_squared, xi_rho[:,i:i_end])
        
        # Accumulate result
        result += einsum('kl,pk,ql->pq', G1, C_rho, C_rho)
    
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"calc_K3_isdf completed in {elapsed_time:.2f} seconds")
    return result
