"""JAX implementation of Density Fitting / ISDF."""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import logging
import time
from typing import Callable, Tuple


def solve_normal_equations_batch(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                   B: jnp.ndarray, rcond: float = 1e-10) -> jnp.ndarray:
    """Fast solver using SVD-based pseudoinverse for structured least-squares.
    
    Solves min ||C*X - B||² where C[pq, m] = phi_piv_p[p, m] * phi_piv_q[q, m]
    
    Note: Integration weights are applied during pivot selection, not here.
    The fitting is done at discrete grid points using the selected pivots.
    
    This exploits the Kronecker-like structure:
    (C^T C)[m, m'] = (phi_piv_p^T @ phi_piv_p)[m,m'] * (phi_piv_q^T @ phi_piv_q)[m,m']
    (C^T B)[m, g] = einsum('pq,pm,qm->m', B[pq,g], phi_piv_p, phi_piv_q)
    
    Uses SVD-based pseudoinverse which is stable and accurate. Since n_fused is 
    typically < 10000, SVD is fast enough and provides better numerical accuracy
    than Cholesky with regularization.
    
    Args:
        phi_piv_p: (n_orb, n_fused) first factor (e.g., phi_piv or grad_phi_piv[:,:,c])
        phi_piv_q: (n_orb, n_fused) second factor (usually phi_piv)
        B: (n_orb^2, n_rhs) right-hand sides
        rcond: Relative condition number cutoff for SVD (default 1e-10)
        
    Returns:
        X: (n_fused, n_rhs) solutions
    """
    n_orb, n_fused = phi_piv_p.shape
    n_rhs = B.shape[1]
    
    # Compute A^T A efficiently using the Kronecker-like structure
    gram_p = phi_piv_p.T @ phi_piv_p  # (n_fused, n_fused)
    gram_q = phi_piv_q.T @ phi_piv_q  # (n_fused, n_fused)
    ATA = gram_p * gram_q  # Element-wise product
    
    # Compute A^T B efficiently
    B_reshaped = B.reshape(n_orb, n_orb, n_rhs)
    ATB = jnp.einsum('pqg,pm,qm->mg', B_reshaped, phi_piv_p, phi_piv_q)  # (n_fused, n_rhs)
    
    # Use SVD for numerically stable pseudoinverse
    U, s, Vt = jnp.linalg.svd(ATA, full_matrices=False)
    
    # Compute inverse singular values with cutoff
    s_inv = jnp.where(s > rcond * s[0], 1.0 / s, 0.0)
    
    # Solve: X = V @ diag(s_inv) @ U.T @ ATB
    X = Vt.T @ (s_inv[:, None] * (U.T @ ATB))
    
    return X


@partial(jax.jit, static_argnames=('n_rank',))
def _pivoted_cholesky_phi(phi_weighted, n_rank, shift):
    """Specialized pivoted Cholesky for phi decomposition."""
    n_grid = phi_weighted.shape[1]
    
    # Initialize diagonal
    diag_err = jnp.sum(phi_weighted**2, axis=0)**2 + shift
    
    # Storage for L factor (N_grid, n_rank)
    L = jnp.zeros((n_grid, n_rank))
    pivots = jnp.zeros(n_rank, dtype=int)
    
    def body_fn(step, state):
        diag_err, L, pivots = state
        pivot = jnp.argmax(diag_err)
        pivots = pivots.at[step].set(pivot)
        pivot_val = diag_err[pivot]
        
        # gram_col_phi logic
        dot = jnp.dot(phi_weighted.T, phi_weighted[:, pivot])
        S_col = dot**2
        S_col = S_col.at[pivot].add(shift)
        
        dot_prod = jnp.dot(L, L[pivot])
        is_small = pivot_val < 1e-12
        safe_pivot = jnp.where(is_small, 1.0, pivot_val)
        inv_sqrt_pivot = jax.lax.rsqrt(safe_pivot)
        
        l_col = (S_col - dot_prod) * inv_sqrt_pivot
        l_col = jnp.where(is_small, 0.0, l_col)
        L = L.at[:, step].set(l_col)
        diag_err = jnp.maximum(diag_err - l_col**2, 0.0)
        diag_err = jnp.where(is_small, diag_err.at[pivot].set(0.0), diag_err)
        
        return diag_err, L, pivots

    _, _, final_pivots = jax.lax.fori_loop(0, n_rank, body_fn, (diag_err, L, pivots))
    return final_pivots


@partial(jax.jit, static_argnames=('n_rank',))
def _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank, shift):
    """Specialized pivoted Cholesky for gradient decomposition."""
    n_grid = phi_weighted.shape[1]
    
    # Initialize diagonal
    A_diag = jnp.sum(phi_weighted**2, axis=0)
    B_diag = jnp.sum(jnp.sum(grad_phi_weighted**2, axis=2), axis=0)
    diag_err = A_diag * B_diag + shift
    
    # Storage for L factor (N_grid, n_rank)
    L = jnp.zeros((n_grid, n_rank))
    pivots = jnp.zeros(n_rank, dtype=int)
    
    def body_fn(step, state):
        diag_err, L, pivots = state
        pivot = jnp.argmax(diag_err)
        pivots = pivots.at[step].set(pivot)
        pivot_val = diag_err[pivot]
        
        # gram_col_grad logic
        A_col = jnp.dot(phi_weighted.T, phi_weighted[:, pivot])
        B_col = jnp.zeros(n_grid)
        for c in range(3):
            B_col += jnp.dot(grad_phi_weighted[:, :, c].T, grad_phi_weighted[:, pivot, c])
        S_col = A_col * B_col
        S_col = S_col.at[pivot].add(shift)
        
        dot_prod = jnp.dot(L, L[pivot])
        is_small = pivot_val < 1e-12
        safe_pivot = jnp.where(is_small, 1.0, pivot_val)
        inv_sqrt_pivot = jax.lax.rsqrt(safe_pivot)
        
        l_col = (S_col - dot_prod) * inv_sqrt_pivot
        l_col = jnp.where(is_small, 0.0, l_col)
        L = L.at[:, step].set(l_col)
        diag_err = jnp.maximum(diag_err - l_col**2, 0.0)
        diag_err = jnp.where(is_small, diag_err.at[pivot].set(0.0), diag_err)
        
        return diag_err, L, pivots

    _, _, final_pivots = jax.lax.fori_loop(0, n_rank, body_fn, (diag_err, L, pivots))
    return final_pivots


def compute_rhs_phi(phi_p: jnp.ndarray, phi_q: jnp.ndarray, 
                    grid_start: int, grid_end: int) -> jnp.ndarray:
    """Compute RHS for least-squares problem for a grid batch.
    
    RHS[pq, g] = phi_p[p, g] * phi_q[q, g]
    
    This handles both phi decomposition (phi_p = phi_q = phi) and 
    gradient decomposition (phi_p = grad_phi[:,:,c], phi_q = phi).
    
    Args:
        phi_p: (n_orb, n_grid) first factor (phi or grad_phi[:,:,c])
        phi_q: (n_orb, n_grid) second factor (usually phi)
        grid_start: Start index for grid batch
        grid_end: End index for grid batch
        
    Returns:
        rhs: (n_orb², batch_size) flattened RHS
    """
    phi_p_batch = phi_p[:, grid_start:grid_end]  # (n_orb, batch_size)
    phi_q_batch = phi_q[:, grid_start:grid_end]  # (n_orb, batch_size)
    n_orb, batch_size = phi_p_batch.shape
    
    # Compute outer products for all grid points in batch
    # rhs[p,q,g] = phi_p[p,g] * phi_q[q,g]
    rhs = jnp.einsum('pi,qi->pqi', phi_p_batch, phi_q_batch)  # (n_orb, n_orb, batch)
    return rhs.reshape(-1, batch_size)  # (n_orb², batch)



def isdf_decompose(phi, grad_phi, n_rank_phi, n_rank_grad, weights=None,
                   use_iterative=True, grid_batch_size=4096, rcond=1e-14):
    """Perform ISDF decomposition of orbitals and their gradients.
    
    Memory-efficient implementation using SVD-based solver to avoid
    materializing large C matrices (n_orb² × n_fused).
    
    Args:
        phi: Orbitals on grid (n_orb, n_grid)
        grad_phi: Orbital gradients on grid (n_orb, n_grid, 3)
        n_rank_phi: Rank for phi decomposition
        n_rank_grad: Rank for gradient decomposition
        weights: Optional (n_grid,) array of integration weights. 
                 If provided, pivot selection is weighted by these weights.
        use_iterative: If True, use memory-efficient solver (recommended).
                      If False, use direct solve with materialized C matrices (HIGH MEMORY).
        grid_batch_size: Number of grid points to process in each batch
        rcond: Relative condition number cutoff for SVD pseudoinverse (default 1e-10).
               Smaller values retain more singular values (more accurate but less stable).
        
    Returns:
        phi_piv: (N_orb, N_fused)
        xi_phi: (N_fused, N_grid)
        grad_phi_piv: (N_orb, N_fused, 3)
        xi_grad: (N_fused, N_grid, 3)
        pivots: (N_fused,)
    """
    n_orb, n_grid = phi.shape
    
    if weights is None:
        w_sqrt = jnp.ones(n_grid)
    else:
        w_sqrt = jnp.sqrt(jnp.abs(weights))  # Use abs to avoid NaN
        
    start_time = time.perf_counter()
    logging.info(f"Starting ISDF decomposition with n_orb={n_orb}, n_grid={n_grid}, n_rank_phi={n_rank_phi}, n_rank_grad={n_rank_grad}")
    if weights is not None:
        logging.info(f"  Using integration weights (min={jnp.min(weights):.3e}, max={jnp.max(weights):.3e})")

    # --- 1. Phi Decomposition ---
    t0 = time.perf_counter()
    
    # Apply weights to orbitals for pivot selection
    # The Gram matrix is (phi^T W phi)(phi^T W phi) where W = diag(weights)
    # Equivalently: (sqrt(W) phi)^T (sqrt(W) phi) squared
    phi_weighted = phi * w_sqrt  # (n_orb, n_grid)
        
    # Pre-compute diagonal for phi to calculate shift
    orb_sq = jnp.sum(phi_weighted**2, axis=0)  # Weighted orbital norms
    diag_phi = orb_sq**2
    shift_phi = 1e-12 * jnp.max(jnp.abs(diag_phi))

    pivots_phi = _pivoted_cholesky_phi(phi_weighted, n_rank_phi, shift_phi)
    t1 = time.perf_counter()
    logging.info(f"Phi decomposition completed in {t1 - t0:.4f} s")

    # --- 2. Gradient Decomposition ---
    t0 = time.perf_counter()
    
    # Apply weights to gradients for pivot selection
    grad_phi_weighted = grad_phi * w_sqrt[:, None]  # (n_orb, n_grid, 3)
    
    # Pre-compute diagonal for grad to calculate shift
    A_diag = jnp.sum(phi_weighted**2, axis=0)
    B_diag = jnp.sum(jnp.sum(grad_phi_weighted**2, axis=2), axis=0)
    diag_grad = A_diag * B_diag
    shift_grad = 1e-12 * jnp.max(jnp.abs(diag_grad))

    pivots_grad = _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank_grad, shift_grad)
    t1 = time.perf_counter()
    logging.info(f"Grad decomposition completed in {t1 - t0:.4f} s")
    
    # --- 3. Fuse pivots ---
    t0 = time.perf_counter()
    
    # Use numpy for unique to avoid JAX dynamic shape overhead
    pivots_all = np.concatenate([np.array(pivots_phi), np.array(pivots_grad)])
    pivots = jnp.array(np.unique(pivots_all))
    n_fused = pivots.shape[0]
    t1 = time.perf_counter()
    logging.info(f"Pivots fused: {pivots_phi.shape[0]} + {pivots_grad.shape[0]} -> {n_fused} in {t1 - t0:.4f} s")
    
    # --- 4. Extract pivot values ---
    t0 = time.perf_counter()
    
    phi_piv = phi[:, pivots]  # (n_orb, n_fused)
    grad_phi_piv = grad_phi[:, pivots, :]  # (n_orb, n_fused, 3)
    
    t1 = time.perf_counter()
    logging.info(f"Pivot values extracted in {t1 - t0:.4f} s")
    
    # --- 5. Solve for xi_phi and xi_grad using fast normal equations solver ---
    if use_iterative:
        t0 = time.perf_counter()
        logging.info("Using fast normal equations solver")
        
        # Solve for xi_phi in batches over grid points
        # Pre-allocate on host as numpy array to avoid GPU OOM
        xi_phi = np.zeros((n_fused, n_grid), dtype=phi.dtype)
        n_batches = (n_grid + grid_batch_size - 1) // grid_batch_size
        
        logging.info(f"  Processing {n_batches} batches of size {grid_batch_size}")
        t_batch_start = time.perf_counter()
        
        for batch_idx in range(n_batches):
            g_start = batch_idx * grid_batch_size
            g_end = min(g_start + grid_batch_size, n_grid)
            
            # Compute RHS for this batch: phi[p,g] * phi[q,g]
            rhs_batch = compute_rhs_phi(phi, phi, g_start, g_end)  # (n_orb², batch)
            
            # Solve using SVD-based pseudoinverse
            # For phi: C[pq,m] = phi_piv[p,m] * phi_piv[q,m]
            xi_batch = solve_normal_equations_batch(phi_piv, phi_piv, rhs_batch, rcond=rcond)
            
            # Store in host array
            xi_phi[:, g_start:g_end] = np.array(xi_batch)
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                elapsed = time.perf_counter() - t_batch_start
                rate = batch_idx / elapsed
                eta = (n_batches - batch_idx) / rate if rate > 0 else 0
                logging.info(f"    Xi_phi: batch {batch_idx}/{n_batches} ({rate:.1f} batch/s, ETA: {eta:.1f}s)")
        
        cpu_device = jax.devices("cpu")[0]
        xi_phi = jax.device_put(xi_phi, cpu_device)
        t1 = time.perf_counter()
        logging.info(f"Xi_phi solved in {t1 - t0:.4f} s ({n_batches/(t1-t0):.2f} batch/s)")
        
        # Solve for xi_grad (one component at a time)
        t0 = time.perf_counter()
        # Pre-allocate on host as numpy array
        xi_grad = np.zeros((n_fused, n_grid, 3), dtype=phi.dtype)
        
        for c in range(3):
            t_comp_start = time.perf_counter()
            
            # Extract gradient component
            grad_phi_piv_c = grad_phi_piv[:, :, c]  # (n_orb, n_fused)
            grad_phi_c = grad_phi[:, :, c]  # (n_orb, n_grid)
            
            for batch_idx in range(n_batches):
                g_start = batch_idx * grid_batch_size
                g_end = min(g_start + grid_batch_size, n_grid)
                
                # RHS: grad_phi[p,g,c] * phi[q,g]
                rhs_batch = compute_rhs_phi(grad_phi_c, phi, g_start, g_end)
                
                # For grad: C[pq,m,c] = grad_phi_piv[p,m,c] * phi_piv[q,m]
                xi_batch = solve_normal_equations_batch(grad_phi_piv_c, phi_piv,
                                                        rhs_batch, rcond=rcond)
                
                # Store in host array
                xi_grad[:, g_start:g_end, c] = np.array(xi_batch)
            
            t_comp = time.perf_counter() - t_comp_start
            logging.info(f"  Xi_grad component {c} solved in {t_comp:.4f} s")
        
        xi_grad = jax.device_put(xi_grad, cpu_device)
        t1 = time.perf_counter()
        logging.info(f"Xi_grad solved in {t1 - t0:.4f} s")
        
    else:
        # --- OLD METHOD: Direct solve with materialized C matrices ---
        t0 = time.perf_counter()
        logging.warning("Using direct solver (HIGH MEMORY!)")
        
        # Construct C matrices (MEMORY INTENSIVE!)
        C_phi = jnp.einsum('pm,qm->pqm', phi_piv, phi_piv).reshape(-1, n_fused)
        C_grad = jnp.einsum('pmc,qm->pqmc', grad_phi_piv, phi_piv).reshape(-1, n_fused, 3)
        
        # Pre-compute pseudo-inverses for least squares
        C_phi_pinv = jnp.linalg.pinv(C_phi)
        C_grad_pinv = jnp.stack([jnp.linalg.pinv(C_grad[:, :, c]) for c in range(3)], axis=0)
        
        t1 = time.perf_counter()
        logging.info(f"C matrices and pinv constructed in {t1 - t0:.4f} s")
        
        # Solve for xi_phi and xi_grad (Block-wise Least Squares)
        t0 = time.perf_counter()
        
        batch_size = grid_batch_size
        n_batches = (n_grid + batch_size - 1) // batch_size
        
        def solve_batch(batch_idx):
            start = batch_idx * batch_size
            end = jnp.minimum(start + batch_size, n_grid)
            width = end - start
            
            # 1. Solve xi_phi
            phi_batch = jax.lax.dynamic_slice(phi, (0, start), (n_orb, width))
            phi_paired_batch = jnp.einsum('pi,qi->pqi', phi_batch, phi_batch).reshape(-1, width)
            
            # Solve C_phi * xi = phi_paired_batch using pre-computed pinv
            xi_phi_batch = C_phi_pinv @ phi_paired_batch
            
            # 2. Solve xi_grad
            grad_phi_batch = jax.lax.dynamic_slice(grad_phi, (0, start, 0), (n_orb, width, 3))
            
            xi_grad_batch_list = []
            for c in range(3):
                # Construct grad_phi_paired_batch for component c
                grad_phi_paired_c = jnp.einsum('pi,qi->pqi', grad_phi_batch[:, :, c], phi_batch).reshape(-1, width)
                
                # Solve C_grad[..., c] * xi = grad_phi_paired_c using pre-computed pinv
                xi_c = C_grad_pinv[c] @ grad_phi_paired_c
                xi_grad_batch_list.append(xi_c)
                
            xi_grad_batch = jnp.stack(xi_grad_batch_list, axis=-1) # (k, width, 3)
            
            return xi_phi_batch, xi_grad_batch, width

        xi_phi_list = []
        xi_grad_list = []
        
        for i in range(n_batches):
            xi_p, xi_g, w = solve_batch(i)
            xi_phi_list.append(xi_p)
            xi_grad_list.append(xi_g)
            
        xi_phi = jnp.concatenate(xi_phi_list, axis=1)
        xi_grad = jnp.concatenate(xi_grad_list, axis=1)
        t1 = time.perf_counter()
        logging.info(f"Xi solved in {t1 - t0:.4f} s")
    
    total_time = time.perf_counter() - start_time
    logging.debug(f"Total fused ranks = {n_fused}")
    logging.info(f"ISDF decomposition total time: {total_time:.4f} s")
    
    return phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots
