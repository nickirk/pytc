import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.ops import segment_sum
import numpy as np
import h5py
import logging
import time
import os
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

def kmeans_pivots_jax(grid_coords, weights, n_aux, max_iters=100, batch_size=50000):
    n_grid = grid_coords.shape[0]
    
    # 1. Initialization (K-Means++)
    key = jax.random.PRNGKey(42)
    prob = weights / jnp.sum(weights)
    idx = jax.random.choice(key, n_grid, shape=(n_aux,), p=prob, replace=False)
    centroids = grid_coords[idx]

    # Pre-calculate r2 once to save memory and time
    r2 = jnp.sum(grid_coords**2, axis=1) # (N_grid,)

    @jax.jit
    def get_batch_labels(batch_coords, batch_r2, curr_centroids, curr_c2):
        # |r-c|^2 = r^2 + c^2 - 2rc
        rc = jnp.dot(batch_coords, curr_centroids.T)
        # Broadcasting: (batch, 1) + (n_aux,) - (batch, n_aux)
        dist_sq = batch_r2[:, None] + curr_c2 - 2 * rc
        return jnp.argmin(dist_sq, axis=1)

    @jax.jit
    def sync_centroids(all_labels, grid_coords, weights):
        # Using the segment_sum trick we discussed
        denoms = jax.ops.segment_sum(weights, all_labels, num_segments=n_aux)
        denoms = jnp.where(denoms > 1e-15, denoms, 1.0)
        
        weighted_coords = grid_coords * weights[:, None]
        nums = jax.ops.segment_sum(weighted_coords, all_labels, num_segments=n_aux)
        return nums / denoms[:, None]

    print(f"Starting K-Means with {n_aux} clusters...")

    for i in range(max_iters):
        # Step 1: Assignment (Batched to save VRAM)
        c2 = jnp.sum(centroids**2, axis=1)
        labels_list = []
        
        for start in range(0, n_grid, batch_size):
            end = min(start + batch_size, n_grid)
            batch_l = get_batch_labels(
                grid_coords[start:end], 
                r2[start:end], 
                centroids, 
                c2
            )
            labels_list.append(batch_l)
        
        full_labels = jnp.concatenate(labels_list)
        
        # Step 2: Update
        new_centroids = sync_centroids(full_labels, grid_coords, weights)
        
        # Check convergence (optional but helpful)
        diff = jnp.max(jnp.abs(new_centroids - centroids))
        centroids = new_centroids
        if diff < 1e-5:
            break

    # Final Snap: Centroids to real Grid Indices
    # We must do this in batches too!
    final_pivots = []
    c2 = jnp.sum(centroids**2, axis=1)
    for start in range(0, n_grid, batch_size):
        # We don't actually need the full label list here, 
        # but we need to find which grid point is closest to each centroid
        pass 
    
    # Actually, the most robust "Snap" for ISDF is just to return 
    # the argmin of the distances across the WHOLE grid for each centroid.
    # To do this memory-efficiently:
    
    best_indices = jnp.zeros(n_aux, dtype=jnp.int32)
    min_dists = jnp.full(n_aux, jnp.inf)

    for start in range(0, n_grid, batch_size):
        end = min(start + batch_size, n_grid)
        rc = jnp.dot(grid_coords[start:end], centroids.T)
        dist_sq = r2[start:end, None] + c2 - 2 * rc # (batch, n_aux)
        
        # Find local min for this batch
        local_min_val = jnp.min(dist_sq, axis=0)
        local_min_idx = jnp.argmin(dist_sq, axis=0) + start
        
        # Update global best
        mask = local_min_val < min_dists
        min_dists = jnp.where(mask, local_min_val, min_dists)
        best_indices = jnp.where(mask, local_min_idx, best_indices)

    return best_indices

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

import gc
import jax
import numpy as np

def isdf_decompose_outcore(input_stream_path, output_stream_path, n_rank_phi, n_rank_grad, 
                    grid_coords, weights, grid_batch_size=4096, rcond=1e-14, n_rank_kmeans_factor = 5):
    """Full Out-of-Core ISDF for Phi and Grad_Phi with aggressive memory cleanup."""
    
    with h5py.File(input_stream_path, 'r') as f_in:
        h5_phi = f_in['phi']
        h5_grad = f_in['grad_phi']
        n_orb, n_grid = h5_phi.shape

        # --- 1. K-means Pivot Selection ---
        wts_phi, wts_grad = get_max_orbital_importance(input_stream_path, weights=weights, batch_size=grid_batch_size)
        
        pivots_phi_km = kmeans_pivots_jax(grid_coords, wts_phi, n_rank_kmeans_factor*n_rank_phi, batch_size=grid_batch_size)
        pivots_grad_km = kmeans_pivots_jax(grid_coords, wts_grad, n_rank_kmeans_factor*n_rank_grad, batch_size=grid_batch_size)
        
        # Ensure K-means is done before pulling data
        pivots_phi_km.block_until_ready()
        pivots_grad_km.block_until_ready()

        initial_pivs = np.unique(np.concatenate([np.array(pivots_phi_km), np.array(pivots_grad_km)]))
        pivots_jax = jnp.array(initial_pivs)
        
        # --- 2. Pivoted Cholesky Refinement ---
        phi = jnp.array(h5_phi[:, initial_pivs])
        grad_phi = jnp.array(h5_grad[:, initial_pivs, :])
        w_sqrt = jnp.sqrt(jnp.abs(weights[initial_pivs]))
        
        phi_weighted = phi * w_sqrt
        diag_phi = jnp.sum(phi_weighted**2, axis=0)**2
        shift_phi = 1e-12 * jnp.max(jnp.abs(diag_phi))
        
        pivots_phi_idx = _pivoted_cholesky_phi(phi_weighted, n_rank_phi, shift_phi)
        pivots_phi_idx.block_until_ready()

        grad_phi_weighted = grad_phi * w_sqrt[:, None]
        A_diag = jnp.sum(phi_weighted**2, axis=0)
        B_diag = jnp.sum(jnp.sum(grad_phi_weighted**2, axis=2), axis=0)
        diag_grad = A_diag * B_diag
        shift_grad = 1e-12 * jnp.max(jnp.abs(diag_grad))

        pivots_grad_idx = _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank_grad, shift_grad)
        pivots_grad_idx.block_until_ready()

        # Final global indices
        pivots_final = np.unique(np.concatenate([initial_pivs[np.array(pivots_phi_idx)], 
                                                initial_pivs[np.array(pivots_grad_idx)]]))
        pivots = jnp.array(pivots_final)
        
        # CLEANUP: Delete heavy refinement intermediates
        del phi, grad_phi, phi_weighted, grad_phi_weighted, initial_pivs, pivots_jax
        gc.collect()
        jax.clear_caches() # Free buffers from the Cholesky/Kmeans logic

        print(f'Passed select_pivots. Final Rank: {len(pivots)}')
        
        # --- 3. Extract and Solve ---
        phi_piv = jnp.array(h5_phi[:, pivots_final])
        grad_piv = jnp.array(h5_grad[:, pivots_final, :])
        
        phi_chol, phi_lower = prepare_normal_equations_solver(phi_piv, phi_piv, rcond=rcond)
        
        g_chol, g_low = [], []
        for c in range(3):
            c_h, l_h = prepare_normal_equations_solver(grad_piv[:, :, c], phi_piv, rcond=rcond)
            g_chol.append(c_h); g_low.append(l_h)

        with h5py.File(output_stream_path, 'w') as f_out:
            xi_p = f_out.create_dataset('xi_phi', (len(pivots_final), n_grid), dtype='f8')
            xi_g = f_out.create_dataset('xi_grad', (len(pivots_final), n_grid, 3), dtype='f8')
            f_out.create_dataset('pivots', data=pivots_final)
            
            for i in range(0, n_grid, grid_batch_size):
                t0 = time.time()
                end = min(i + grid_batch_size, n_grid)
                
                # Load slice
                p_batch = jnp.array(h5_phi[:, i:end])
                
                # Solve Phi
                res_p = solve_normal_equations_batch_prepared(
                    phi_chol, phi_lower, phi_piv, phi_piv, p_batch, p_batch)
                res_p.block_until_ready() # Wait for GPU before writing to disk
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
                
                # Periodic Cache Clearing: Essential for very large grids
                if (i // grid_batch_size) % 20 == 0:
                    jax.clear_caches()
                    gc.collect()

                t1 = time.time()
                print(f"Normal equations progress: {end}/{n_grid} grid points processed. Batch took: {t1-t0:0.2f}s")

    print('Passed all!')
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
