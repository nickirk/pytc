import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.ops import segment_sum
import numpy as np
import scipy.linalg as sp_linalg
import h5py
import logging
import time
import os
import gc
from functools import partial

# Import original solvers from your existing file
from pytc.df import (prepare_normal_equations_solver, 
                solve_normal_equations_batch_prepared,
                isdf_decompose as isdf_incore)

from pytc.df import _pivoted_cholesky_grad, _pivoted_cholesky_phi

# JAX CPU Parallelism Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_memory_usage(named_arrays):
    """
    Identifies all live arrays on the default GPU and prints their 
    size, shape, and memory footprint in MiB.
    """
    
    print(f"\n{'Variable Name':<25} | {'Variable Shape':<25} | {'Dtype':<10} | {'Memory (MiB)':<12} | Device")
    print("-" * 55)
    
    total_mem_bytes = 0
    count = 0
    
    for name, arr in named_arrays.items():
        # Check if the array is actually on the device we are looking at
        device = getattr(arr, 'device', 'CPU')
        try:
            shape = arr.shape
            dtype = arr.dtype
            size_bytes = arr.nbytes
            
            mem_mib = size_bytes / (1024**2)
            print(f"{name:<25} | {str(shape):<25} | {str(dtype):<10} | {mem_mib:>10.2f} MiB | {device}")
            
            total_mem_bytes += size_bytes
            count += 1
        except (AttributeError, RuntimeError):
            continue

    total_mib = total_mem_bytes / (1024**2)
    print("-" * 55)
    print(f"Total Live Arrays: {count}")
    print(f"Total Memory Used: {total_mib:.2f} MiB")

def stream_diag_and_shift(h5_phi, h5_grad, w_sqrt, batch_size):
    """Computes Gram matrix diagonals for phi and grad_phi out-of-core."""
    n_orb, n_grid = h5_phi.shape
    diag_phi = jax.device_put(jnp.zeros(n_grid))
    diag_grad = jax.device_put(jnp.zeros(n_grid))

    for i in range(0, n_grid, batch_size):
        end = min(i + batch_size, n_grid)
        p_batch = jax.device_put(jnp.array(h5_phi[:, i:end])) * w_sqrt[i:end]
        g_batch = jax.device_put(jnp.array(h5_grad[:, i:end, :])) * w_sqrt[i:end, None]
        
        # Logic from Source 1: diag_phi = (sum phi^2)^2
        # diag_grad = (sum phi^2) * (sum |grad_phi|^2)
        a_diag = jnp.sum(p_batch**2, axis=0)
        # b_diag = jnp.sum(jnp.sum(g_batch**2, axis=2), axis=0)
        b_diag = jnp.einsum('igc,igc->g', g_batch, g_batch)
        
        diag_phi = diag_phi.at[i:end].set(a_diag**2)
        diag_grad = diag_grad.at[i:end].set(a_diag * b_diag)

    shift_phi = 1e-12 * jnp.max(diag_phi)
    shift_grad = 1e-12 * jnp.max(diag_grad)
    return diag_phi + shift_phi, diag_grad + shift_grad, shift_phi, shift_grad

def get_gram_col_outcore(h5_phi, h5_grad, pivot, w_sqrt, mode, shift, batch_size):
    """Computes a specific Gram column (phi or grad) by streaming from disk."""
    n_orb, n_grid = h5_phi.shape
    p_piv = jnp.array(h5_phi[:, pivot]) * w_sqrt[pivot]
    col = jnp.zeros(n_grid)
    grad_at_pivot = jax.device_put(jnp.array(h5_grad[:, pivot, :]))
    # print('nbatch: ', n_grid // batch_size)
    for i in range(0, n_grid, batch_size):
        # t0 = time.time()
        end = min(i + batch_size, n_grid)
        p_batch = jax.device_put(jnp.array(h5_phi[:, i:end])) * w_sqrt[i:end]
        a_col = jnp.dot(p_batch.T, p_piv)
        if mode == 'phi':
            col = col.at[i:end].set(a_col**2)
        else:
            g_piv = grad_at_pivot * w_sqrt[pivot]
            g_batch = jax.device_put(jnp.array(h5_grad[:, i:end, :])) * w_sqrt[i:end, None]
            b_col = jnp.zeros(end - i)
            for c in range(3):
                b_col += jnp.dot(g_batch[:, :, c].T, g_piv[:, c])
            col = col.at[i:end].set(a_col * b_col)
        # print_memory_usage({name: val for name, val in locals().items()})
        # t1 = time.time()
        # print(f'gram batch time: {t1-t0:0.2f}')
            
    col = col.at[pivot].add(shift)
    return col

# def kmeans_pivots_jax(grid_coords, weights, n_aux, max_iters=200):
#     """
#     Weighted K-Means for ISDF Pivot Selection.
#     grid_coords: (N_grid, 3)
#     density: (N_grid,) - Electron density
#     weights: (N_grid,) - Integration weights
#     """
#     n_grid = grid_coords.shape[0]
    
#     # 1. Initialize centroids using Weighted Sampling (K-Means++)
#     key = jax.random.PRNGKey(42)
#     # Probability distribution based on density
#     prob = weights / jnp.sum(weights)
#     idx = jax.random.choice(key, n_grid, shape=(n_aux,), p=prob, replace=False)
#     centroids = grid_coords[idx]

#     @jax.jit
#     def one_iteration(curr_centroids):
#         # Memory-efficient assignment: Find index of closest centroid
#         # We use the expansion: |r - c|^2 = |r|^2 + |c|^2 - 2rc
#         r2 = jnp.sum(grid_coords**2, axis=1, keepdims=True) # (N, 1)
#         c2 = jnp.sum(curr_centroids**2, axis=1)            # (K,)
#         rc = jnp.dot(grid_coords, curr_centroids.T)        # (N, K)
        
#         dist_sq = r2 + c2 - 2 * rc
#         labels = jnp.argmin(dist_sq, axis=1)               # (N,)

#         # Update Step: Compute weighted mean for each cluster
#         # def compute_new_centroid(k):
#         #     mask = (labels == k)
#         #     w_mask = weights * mask
#         #     denom = jnp.sum(w_mask)
#         #     num = jnp.sum(grid_coords * w_mask[:, None], axis=0)
#         #     return num / denom
#         # new_centroids = jax.vmap(compute_new_centroid)(jnp.arange(n_aux))
         
#         # 1. Compute denominators (sum of weights per cluster)
#         # Result shape: (n_aux,)
#         denoms = segment_sum(weights, labels, num_segments=n_aux)
        
#         # Add a small epsilon to avoid division by zero for empty clusters
#         denoms = jnp.where(denoms > 1e-15, denoms, 1.0)

#         # 2. Compute numerators (weighted sum of coordinates per cluster)
#         # We multiply coords by weights first: (N_grid, 3) * (N_grid, 1)
#         # Then segment_sum along the grid dimension
#         weighted_coords = grid_coords * weights[:, None]
#         nums = segment_sum(weighted_coords, labels, num_segments=n_aux) # Shape: (n_aux, 3)

#         # 3. Final division
#         new_centroids = nums / denoms[:, None]
            

#         return new_centroids

#     # Loop iterations
#     for i in range(max_iters):
#         centroids = one_iteration(centroids)
        

#     # Final step: Snap centroids to the nearest actual grid points
#     # (Because the weighted average might land in a 'void')
#     r2 = jnp.sum(grid_coords**2, axis=1, keepdims=True)
#     c2 = jnp.sum(centroids**2, axis=1)
#     rc = jnp.dot(grid_coords, centroids.T)
#     final_pivots = jnp.argmin(r2 + c2 - 2 * rc, axis=0)
    
#     return final_pivots


def stream_diag_and_shift_numpy(h5_phi, h5_grad, w_sqrt, batch_size):
    """NumPy version of stream_diag_and_shift. No JAX required."""
    n_orb, n_grid = h5_phi.shape
    diag_phi = np.zeros(n_grid)
    diag_grad = np.zeros(n_grid)

    for i in range(0, n_grid, batch_size):
        end = min(i + batch_size, n_grid)
        p_batch = np.array(h5_phi[:, i:end]) * w_sqrt[i:end]          # (n_orb, batch)
        g_batch = np.array(h5_grad[:, i:end, :]) * w_sqrt[i:end, None] # (n_orb, batch, 3)

        a_diag = np.sum(p_batch**2, axis=0)                            # (batch,)
        b_diag = np.einsum('igc,igc->g', g_batch, g_batch)             # (batch,)

        diag_phi[i:end] = a_diag**2
        diag_grad[i:end] = a_diag * b_diag

    shift_phi = 1e-12 * np.max(diag_phi)
    shift_grad = 1e-12 * np.max(diag_grad)
    return diag_phi + shift_phi, diag_grad + shift_grad, shift_phi, shift_grad


def get_gram_col_outcore_numpy(h5_phi, h5_grad, pivot, w_sqrt, mode, shift, batch_size):
    """NumPy version of get_gram_col_outcore. No JAX required."""
    n_orb, n_grid = h5_phi.shape
    p_piv = np.array(h5_phi[:, pivot]) * w_sqrt[pivot]                 # (n_orb,)
    col = np.zeros(n_grid)
    grad_at_pivot = np.array(h5_grad[:, pivot, :])                     # (n_orb, 3)

    for i in range(0, n_grid, batch_size):
        end = min(i + batch_size, n_grid)
        p_batch = np.array(h5_phi[:, i:end]) * w_sqrt[i:end]           # (n_orb, batch)
        a_col = p_batch.T @ p_piv                               # (batch,)
        if mode == 'phi':
            col[i:end] = a_col**2
        else:
            g_piv = grad_at_pivot * w_sqrt[pivot]
            g_batch = np.array(h5_grad[:, i:end, :]) * w_sqrt[i:end, None]  # (n_orb, batch, 3)
            b_col = np.zeros(end - i)
            for c in range(3):
                b_col += g_batch[:, :, c].T @ g_piv[:, c]
            col[i:end] = a_col * b_col

    col[pivot] += shift
    return col


def get_max_orbital_importance(input_stream_path, weights=None, batch_size=100000):
    """
    Computes the maximum orbital importance and gradient importance across a grid.
    
    Args:
        input_stream_path: Path to HDF5 file containing 'phi' and 'grad_phi'.
        weights: Optional integration weights of shape (ngrid,).
        batch_size: Number of grid points to process in one GPU batch.
    """
    
    # 1. Define JIT kernels with operator fusion
    @jax.jit
    def process_phi_batch(phi_batch, w_batch):
        # Taking absolute value ensures we find peak 'importance' regardless of sign
        val = jnp.abs(phi_batch)
        if w_batch is not None:
            val = val * w_batch  # Fused by XLA: No extra allocation
        return jnp.max(val, axis=0)

    @jax.jit
    def process_grad_batch(grad_batch, w_batch):
        # 1. Norm along Cartesian dimension (x,y,z) -> (Norb, Batch)
        norms = jnp.linalg.norm(grad_batch, axis=2)
        if w_batch is not None:
            norms = norms * w_batch # Fused by XLA
        # 2. Max along Orbital dimension -> (Batch,)
        return jnp.max(norms, axis=0)

    with h5py.File(input_stream_path, 'r') as f_in:
        h5_phi = f_in['phi']      # Shape: (Norb, Ngrid)
        h5_grad = f_in['grad_phi'] # Shape: (Norb, Ngrid, 3)
        
        n_orb, n_grid = h5_phi.shape
        
        # Initialize result containers in System RAM (NumPy) to save VRAM
        max_phi = np.zeros(n_grid)
        max_grad_norm = np.zeros(n_grid)
        
        print(f"Starting batched reduction for {n_grid} points (Batch Size: {batch_size})...")
        
        for i in range(0, n_grid, batch_size):
            end = min(i + batch_size, n_grid)
            
            # Move weight slice to GPU if it exists
            w_batch = jnp.array(weights[i:end]) if weights is not None else None
            
            # --- Handle Phi (Orbitals) ---
            batch_p = jnp.array(h5_phi[:, i:end])
            res_p = process_phi_batch(batch_p, w_batch)
            
            # Transfer result to CPU immediately
            max_phi[i:end] = np.array(res_p)
            
            # Explicit Cleanup: Clear 'batch_p' before loading the much larger gradients
            del batch_p
            res_p.block_until_ready() 

            # --- Handle Grad (Gradients) ---
            # Gradient batch is 3x the size of phi; we need the VRAM cleared above
            batch_g = jnp.array(h5_grad[:, i:end, :])
            res_g = process_grad_batch(batch_g, w_batch)
            
            max_grad_norm[i:end] = np.array(res_g)
            
            # Explicit Cleanup
            del batch_g
            res_g.block_until_ready()

            if i % (batch_size * 5) == 0:
                print(f"Progress: {end}/{n_grid} grid points processed.")

    # Return as JAX arrays for the next step (K-Means or FPS)
    return jnp.array(max_phi), jnp.array(max_grad_norm)


def get_max_orbital_importance_numpy(input_stream_path, weights=None, batch_size=100000):
    """
    NumPy version of get_max_orbital_importance. No JAX required.

    Args:
        input_stream_path: Path to HDF5 file containing 'phi' and 'grad_phi'.
        weights: Optional integration weights of shape (ngrid,).
        batch_size: Number of grid points to process in one batch.

    Returns:
        max_phi: (ngrid,) np.ndarray
        max_grad_norm: (ngrid,) np.ndarray
    """
    with h5py.File(input_stream_path, 'r') as f_in:
        h5_phi = f_in['phi']       # Shape: (Norb, Ngrid)
        h5_grad = f_in['grad_phi']  # Shape: (Norb, Ngrid, 3)

        n_orb, n_grid = h5_phi.shape
        max_phi = np.zeros(n_grid)
        max_grad_norm = np.zeros(n_grid)

        print(f"[numpy] Starting batched reduction for {n_grid} points (Batch Size: {batch_size})...")

        for i in range(0, n_grid, batch_size):
            end = min(i + batch_size, n_grid)

            # Use abs(weights): PySCF DFT grid weights can be negative, but
            # importance scores must be non-negative for K-Means sampling.
            w_batch = np.abs(weights[i:end]) if weights is not None else None

            # --- Phi ---
            batch_p = np.array(h5_phi[:, i:end])          # (n_orb, batch)
            val_p = np.abs(batch_p)
            if w_batch is not None:
                val_p = val_p * w_batch
            max_phi[i:end] = np.max(val_p, axis=0)
            del batch_p, val_p

            # --- Grad ---
            batch_g = np.array(h5_grad[:, i:end, :])      # (n_orb, batch, 3)
            norms = np.linalg.norm(batch_g, axis=2)        # (n_orb, batch)
            if w_batch is not None:
                norms = norms * w_batch
            max_grad_norm[i:end] = np.max(norms, axis=0)
            del batch_g, norms

            if i % (batch_size * 5) == 0:
                print(f"[numpy] Progress: {end}/{n_grid} grid points processed.")

    return max_phi, max_grad_norm


def _pivoted_cholesky_phi_numpy(phi_weighted, n_rank, shift, cd_sample_factor=None, cd_seed=42):
    """
    NumPy version of _pivoted_cholesky_phi (from df.py).
    Uses a plain Python for loop instead of jax.lax.fori_loop.

    Args:
        phi_weighted: (n_orb, n_grid) np.ndarray
        n_rank: int
        shift: float
        cd_sample_factor: float or None. Subset grid points to min(n_grid, int(n_orb * cd_sample_factor))
        cd_seed: int. Random seed for reproducible sampling

    Returns:
        pivots: (n_rank,) np.ndarray of int
    """
    n_orb = phi_weighted.shape[0]
    n_grid_full = phi_weighted.shape[1]
    
    if cd_sample_factor is not None:
        num_samples = min(n_grid_full, int(n_orb * cd_sample_factor))
        # sample without replacement
        rng = np.random.RandomState(cd_seed)
        grid_idx = rng.choice(n_grid_full, num_samples, replace=False)
        phi_weighted_sub = phi_weighted[:, grid_idx]
    else:
        grid_idx = np.arange(n_grid_full)
        phi_weighted_sub = phi_weighted

    n_grid = phi_weighted_sub.shape[1]
    diag_err = np.sum(phi_weighted_sub**2, axis=0)**2 + shift  # (n_grid,)
    L = np.zeros((n_grid, n_rank))
    pivots = np.zeros(n_rank, dtype=np.int32)

    for step in range(n_rank):
        pivot = int(np.argmax(diag_err))
        pivots[step] = grid_idx[pivot]
        pivot_val = diag_err[pivot]

        # Gram column: dot(phi_weighted_sub.T, phi_weighted_sub[:, pivot])^2
        dot = phi_weighted_sub.T @ phi_weighted_sub[:, pivot]  # (n_grid,)
        S_col = dot**2
        S_col[pivot] += shift

        dot_prod = L @ L[pivot]  # (n_grid,)
        is_small = pivot_val < 1e-12
        safe_pivot = 1.0 if is_small else pivot_val
        inv_sqrt_pivot = 1.0 / np.sqrt(safe_pivot)

        l_col = (S_col - dot_prod) * inv_sqrt_pivot
        if is_small:
            l_col[:] = 0.0
        L[:, step] = l_col

        diag_err = np.maximum(diag_err - l_col**2, 0.0)
        if is_small:
            diag_err[pivot] = 0.0

    return pivots


def _pivoted_cholesky_grad_numpy(phi_weighted, grad_phi_weighted, n_rank, shift, cd_sample_factor=None, cd_seed=42):
    """
    NumPy version of _pivoted_cholesky_grad (from df.py).
    Uses a plain Python for loop instead of jax.lax.fori_loop.

    Args:
        phi_weighted: (n_orb, n_grid) np.ndarray
        grad_phi_weighted: (n_orb, n_grid, 3) np.ndarray
        n_rank: int
        shift: float
        cd_sample_factor: float or None. Subset grid points to min(n_grid, int(n_orb * cd_sample_factor))
        cd_seed: int. Random seed for reproducible sampling

    Returns:
        pivots: (n_rank,) np.ndarray of int
    """
    n_orb = phi_weighted.shape[0]
    n_grid_full = phi_weighted.shape[1]
    
    if cd_sample_factor is not None:
        num_samples = min(n_grid_full, int(n_orb * cd_sample_factor))
        rng = np.random.RandomState(cd_seed)
        grid_idx = rng.choice(n_grid_full, num_samples, replace=False)
        phi_weighted_sub = phi_weighted[:, grid_idx]
        grad_phi_weighted_sub = grad_phi_weighted[:, grid_idx, :]
    else:
        grid_idx = np.arange(n_grid_full)
        phi_weighted_sub = phi_weighted
        grad_phi_weighted_sub = grad_phi_weighted

    n_grid = phi_weighted_sub.shape[1]
    A_diag = np.sum(phi_weighted_sub**2, axis=0)
    B_diag = np.sum(np.sum(grad_phi_weighted_sub**2, axis=2), axis=0)
    diag_err = A_diag * B_diag + shift
    L = np.zeros((n_grid, n_rank))
    pivots = np.zeros(n_rank, dtype=np.int32)

    for step in range(n_rank):
        pivot = int(np.argmax(diag_err))
        pivots[step] = grid_idx[pivot]
        pivot_val = diag_err[pivot]

        A_col = phi_weighted_sub.T @ phi_weighted_sub[:, pivot]  # (n_grid,)
        B_col = np.zeros(n_grid)
        for c in range(3):
            B_col += grad_phi_weighted_sub[:, :, c].T @ grad_phi_weighted_sub[:, pivot, c]
        S_col = A_col * B_col
        S_col[pivot] += shift

        dot_prod = L @ L[pivot]  # (n_grid,)
        is_small = pivot_val < 1e-12
        safe_pivot = 1.0 if is_small else pivot_val
        inv_sqrt_pivot = 1.0 / np.sqrt(safe_pivot)

        l_col = (S_col - dot_prod) * inv_sqrt_pivot
        if is_small:
            l_col[:] = 0.0
        L[:, step] = l_col

        diag_err = np.maximum(diag_err - l_col**2, 0.0)
        if is_small:
            diag_err[pivot] = 0.0

    return pivots


def _prepare_normal_equations_solver_numpy(phi_piv_p, phi_piv_q, rcond=1e-14,
                                           max_jitter_tries=8, jitter_growth=10.0):
    """
    NumPy/SciPy version of prepare_normal_equations_solver (from df.py).
    Uses scipy.linalg.cho_factor with adaptive jitter escalation.

    Args:
        phi_piv_p: (n_orb, n_fused) np.ndarray
        phi_piv_q: (n_orb, n_fused) np.ndarray
        rcond: regularisation strength
        max_jitter_tries: max jitter escalation attempts
        jitter_growth: multiplicative jitter growth

    Returns:
        chol: Cholesky factor from scipy.linalg.cho_factor
        lower: bool
    """
    gram_p = phi_piv_p.T @ phi_piv_p
    gram_q = phi_piv_q.T @ phi_piv_q
    ata = gram_p * gram_q
    ata = 0.5 * (ata + ata.T)

    diag_mean = float(np.mean(np.diag(ata)))
    eps_scale = float(np.finfo(ata.dtype).eps) * max(diag_mean, 1.0)
    base_jitter = max(diag_mean * rcond, eps_scale)
    eye = np.eye(ata.shape[0], dtype=ata.dtype)

    last_chol = None
    for attempt in range(max_jitter_tries):
        jitter = base_jitter * (jitter_growth ** attempt)
        try:
            chol = sp_linalg.cho_factor(ata + jitter * eye, lower=True)
            if np.all(np.isfinite(chol[0])):
                if attempt > 0:
                    logger.warning(
                        "[numpy] Cholesky jitter escalated: base=%.3e final=%.3e tries=%d",
                        base_jitter, jitter, attempt + 1
                    )
                return chol, True  # chol is a (factor, lower) tuple from scipy
            last_chol = chol
        except np.linalg.LinAlgError:
            pass

    raise np.linalg.LinAlgError(
        f"[numpy] Adaptive Cholesky failed after {max_jitter_tries} tries; "
        f"base_jitter={base_jitter:.3e}"
    )


def _solve_normal_equations_batch_prepared_numpy(chol, phi_piv_p, phi_piv_q,
                                                  phi_p_batch, phi_q_batch):
    """
    NumPy/SciPy version of solve_normal_equations_batch_prepared (from df.py).
    Uses scipy.linalg.cho_solve with a precomputed Cholesky factor.

    Args:
        chol: tuple returned by _prepare_normal_equations_solver_numpy (scipy cho_factor output)
        phi_piv_p: (n_orb, n_fused) np.ndarray
        phi_piv_q: (n_orb, n_fused) np.ndarray
        phi_p_batch: (n_orb, batch) np.ndarray
        phi_q_batch: (n_orb, batch) np.ndarray

    Returns:
        X: (n_fused, batch) np.ndarray
    """
    term_p = np.matmul(phi_piv_p.T, phi_p_batch)  # (n_fused, batch)
    term_q = np.matmul(phi_piv_q.T, phi_q_batch)  # (n_fused, batch)
    atb = term_p * term_q                          # (n_fused, batch)
    return sp_linalg.cho_solve(chol, atb)


def isdf_decompose_outcore(input_stream_path, output_stream_path, n_rank_phi, n_rank_grad,
                    grid_coords, weights, grid_batch_size=4096, rcond=1e-14,
                    backend='jax', cd_sample_factor=None, cd_seed=42):
    """
    Full Out-of-Core ISDF for Phi and Grad_Phi with aggressive memory cleanup.

    Args:
        input_stream_path: Path to HDF5 file containing 'phi' and 'grad_phi'.
        output_stream_path: Path to write HDF5 output ('xi_phi', 'xi_grad', 'pivots').
        n_rank_phi: Target rank for phi decomposition.
        n_rank_grad: Target rank for gradient decomposition.
        grid_coords: (N_grid, 3) array of grid coordinates.
        weights: (N_grid,) integration weights.
        grid_batch_size: Number of grid points per streaming batch.
        rcond: Regularisation strength for normal equations solver.
        backend: 'jax' (default) or 'numpy'. When 'numpy', all JAX operations
                 are replaced with pure NumPy/SciPy equivalents for CPU-only execution.
    """
    if backend not in ('jax', 'numpy'):
        raise ValueError(f"backend must be 'jax' or 'numpy', got {backend!r}")

    use_numpy = (backend == 'numpy')

    with h5py.File(input_stream_path, 'r') as f_in:
        h5_phi = f_in['phi']
        h5_grad = f_in['grad_phi']
        n_orb, n_grid = h5_phi.shape

        # --- 1. Pivoted Cholesky ---
        initial_pivs = np.arange(n_grid)

        if use_numpy:
            phi_sub = np.array(h5_phi[:, initial_pivs])          # (n_orb, n_sub)
            grad_sub = np.array(h5_grad[:, initial_pivs, :])      # (n_orb, n_sub, 3)
            w_sqrt = np.sqrt(np.abs(np.array(weights)[initial_pivs]))

            phi_weighted = phi_sub * w_sqrt
            diag_phi = np.sum(phi_weighted**2, axis=0)**2
            shift_phi = float(1e-12 * np.max(np.abs(diag_phi)))

            pivots_phi_idx = _pivoted_cholesky_phi_numpy(
                phi_weighted, n_rank_phi, shift_phi, cd_sample_factor, cd_seed
            )

            grad_phi_weighted = grad_sub * w_sqrt[:, None]
            A_diag = np.sum(phi_weighted**2, axis=0)
            B_diag = np.sum(np.sum(grad_phi_weighted**2, axis=2), axis=0)
            diag_grad = A_diag * B_diag
            shift_grad = float(1e-12 * np.max(np.abs(diag_grad)))

            pivots_grad_idx = _pivoted_cholesky_grad_numpy(
                phi_weighted, grad_phi_weighted, n_rank_grad, shift_grad, cd_sample_factor, cd_seed
            )

            del phi_sub, grad_sub, phi_weighted, grad_phi_weighted
            gc.collect()
        else:
            phi_sub = jnp.array(h5_phi[:, initial_pivs])
            grad_sub = jnp.array(h5_grad[:, initial_pivs, :])
            w_sqrt = jnp.sqrt(jnp.abs(weights[initial_pivs]))

            phi_weighted = phi_sub * w_sqrt
            diag_phi = jnp.sum(phi_weighted**2, axis=0)**2
            shift_phi = 1e-12 * jnp.max(jnp.abs(diag_phi))

            pivots_phi_idx = _pivoted_cholesky_phi(phi_weighted, n_rank_phi, shift_phi)
            pivots_phi_idx.block_until_ready()

            grad_phi_weighted = grad_sub * w_sqrt[:, None]
            A_diag = jnp.sum(phi_weighted**2, axis=0)
            B_diag = jnp.sum(jnp.sum(grad_phi_weighted**2, axis=2), axis=0)
            diag_grad = A_diag * B_diag
            shift_grad = 1e-12 * jnp.max(jnp.abs(diag_grad))

            pivots_grad_idx = _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank_grad, shift_grad)
            pivots_grad_idx.block_until_ready()

            del phi_sub, grad_sub, phi_weighted, grad_phi_weighted
            gc.collect()
            jax.clear_caches()

        pivots_final = np.unique(np.concatenate([
            initial_pivs[np.array(pivots_phi_idx)],
            initial_pivs[np.array(pivots_grad_idx)]
        ]))

        print(f'[{backend}] Passed select_pivots. Final Rank: {len(pivots_final)}')

        # --- 3. Extract Pivot Values and Prepare Solvers ---
        if use_numpy:
            phi_piv = np.array(h5_phi[:, pivots_final])        # (n_orb, n_fused)
            grad_piv = np.array(h5_grad[:, pivots_final, :])   # (n_orb, n_fused, 3)

            phi_chol, _ = _prepare_normal_equations_solver_numpy(phi_piv, phi_piv, rcond=rcond)

            g_chol = []
            for c in range(3):
                c_h, _ = _prepare_normal_equations_solver_numpy(grad_piv[:, :, c], phi_piv, rcond=rcond)
                g_chol.append(c_h)
        else:
            phi_piv = jnp.array(h5_phi[:, pivots_final])
            grad_piv = jnp.array(h5_grad[:, pivots_final, :])

            phi_chol, phi_lower = prepare_normal_equations_solver(phi_piv, phi_piv, rcond=rcond)

            g_chol, g_low = [], []
            for c in range(3):
                c_h, l_h = prepare_normal_equations_solver(grad_piv[:, :, c], phi_piv, rcond=rcond)
                g_chol.append(c_h)
                g_low.append(l_h)

        # --- 4. Solve Normal Equations in Batches ---
        with h5py.File(output_stream_path, 'w') as f_out:
            xi_p = f_out.create_dataset('xi_phi', (len(pivots_final), n_grid), dtype='f8')
            xi_g = f_out.create_dataset('xi_grad', (len(pivots_final), n_grid, 3), dtype='f8')
            f_out.create_dataset('pivots', data=pivots_final)

            for i in range(0, n_grid, grid_batch_size):
                t0 = time.time()
                end = min(i + grid_batch_size, n_grid)

                if use_numpy:
                    p_batch = np.array(h5_phi[:, i:end])  # (n_orb, batch)

                    # Solve Phi
                    res_p = _solve_normal_equations_batch_prepared_numpy(
                        phi_chol, phi_piv, phi_piv, p_batch, p_batch)
                    xi_p[:, i:end] = res_p
                    del res_p

                    # Solve Grad
                    for c in range(3):
                        g_batch = np.array(h5_grad[:, i:end, c])  # (n_orb, batch)
                        res_g = _solve_normal_equations_batch_prepared_numpy(
                            g_chol[c], grad_piv[:, :, c], phi_piv, g_batch, p_batch)
                        xi_g[:, i:end, c] = res_g
                        del g_batch, res_g

                    del p_batch
                    gc.collect()
                else:
                    p_batch = jnp.array(h5_phi[:, i:end])

                    # Solve Phi
                    res_p = solve_normal_equations_batch_prepared(
                        phi_chol, phi_lower, phi_piv, phi_piv, p_batch, p_batch)
                    res_p.block_until_ready()
                    xi_p[:, i:end] = np.array(res_p)
                    del res_p

                    # Solve Grad
                    for c in range(3):
                        g_batch = jnp.array(h5_grad[:, i:end, c])
                        res_g = solve_normal_equations_batch_prepared(
                            g_chol[c], g_low[c], grad_piv[:, :, c], phi_piv, g_batch, p_batch)
                        res_g.block_until_ready()
                        xi_g[:, i:end, c] = np.array(res_g)
                        del g_batch, res_g

                    del p_batch

                    # Periodic cache clearing for very large grids
                    if (i // grid_batch_size) % 20 == 0:
                        jax.clear_caches()
                        gc.collect()

                t1 = time.time()
                print(f"[{backend}] Normal equations progress: {end}/{n_grid} grid points processed. "
                      f"Batch took: {t1-t0:.2f}s")

    print(f'[{backend}] Passed all!')
    # Return as NumPy to ensure calling scope doesn't accidentally keep JAX pointers alive
    return pivots_final, np.array(phi_piv), np.array(grad_piv)

if __name__ == "__main__":
    # TEST CASE: Equivalence check
    N_ORB, N_GRID, N_RANK = 10, 500, 20
    key = jax.random.PRNGKey(42)
    phi = jax.random.normal(key, (N_ORB, N_GRID))
    grad_phi = jax.random.normal(key, (N_ORB, N_GRID, 3))
    weights = jax.random.normal(key, (N_GRID))
    grid_coords = jax.random.normal(key, (N_GRID, 3))

    # Save to temp HDF5
    with h5py.File('test_in.h5', 'w') as f:
        f.create_dataset('phi', data=phi)
        f.create_dataset('grad_phi', data=grad_phi)

    print("Running In-Core Decomposition...")
    res_in = isdf_incore(phi, grad_phi, N_RANK, N_RANK, weights=weights, is_incore=True)
    
    print("Running Out-of-Core Decomposition...")
    p_out, phi_p_out, g_p_out = isdf_decompose_outcore(
        'test_in.h5', 'test_out.h5', N_RANK, N_RANK, grid_coords, weights, grid_batch_size = 100)
