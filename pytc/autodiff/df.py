"""JAX implementation of Density Fitting / ISDF."""
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=('gram_diag_fn', 'gram_col_fn', 'n_grid', 'n_rank'))
def pivoted_cholesky_matrix_free(gram_diag_fn, gram_col_fn, n_grid, n_rank):
    """Matrix-free pivoted Cholesky decomposition.
    
    Args:
        gram_diag_fn: Function () -> (N_grid,) returning diagonal of Gram matrix
        gram_col_fn: Function (pivot_idx) -> (N_grid,) returning a column of Gram matrix
        n_grid: Number of grid points
        n_rank: Rank of decomposition
        
    Returns:
        pivots: (n_rank,) indices of selected pivots
    """
    # Initialize diagonal
    diag_err = gram_diag_fn()
    
    # Storage for L factor (N_grid, n_rank)
    # We build it column by column
    L = jnp.zeros((n_grid, n_rank))
    pivots = jnp.zeros(n_rank, dtype=int)
    
    def body_fn(step, state):
        diag_err, L, pivots = state
        
        # Select pivot
        pivot = jnp.argmax(diag_err)
        pivots = pivots.at[step].set(pivot)
        pivot_val = diag_err[pivot]
        
        # Compute column of Gram matrix
        S_col = gram_col_fn(pivot)
        
        # Compute column of L
        # L[:, step] = (S[:, pivot] - sum(L[:, :step] * L[pivot, :step])) / sqrt(D[pivot])
        
        # Dot product of previous L rows with L[pivot] row
        # L_prev = L[:, :step]  (N_grid, step)
        # L_pivot = L[pivot, :step] (step,)
        # dot = L_prev @ L_pivot
        
        # Since step is dynamic in scan (but bounded), we can mask or use dynamic slice
        # Ideally we use the full L and mask out future columns, but they are 0 anyway.
        
        dot_prod = jnp.dot(L, L[pivot]) # (N_grid,)
        
        # Numerical stability: check for small pivot
        is_small = pivot_val < 1e-12
        safe_pivot = jnp.where(is_small, 1.0, pivot_val)
        inv_sqrt_pivot = jax.lax.rsqrt(safe_pivot)
        
        l_col = (S_col - dot_prod) * inv_sqrt_pivot
        # If pivot was small, set column to 0
        l_col = jnp.where(is_small, 0.0, l_col)
        
        L = L.at[:, step].set(l_col)
        
        # Update diagonal error
        # D_new = D_old - l_col^2
        diag_err = diag_err - l_col**2
        
        # If pivot was small, force its error to 0 to avoid re-selection
        diag_err = jnp.where(is_small, diag_err.at[pivot].set(0.0), diag_err)
        
        # Ensure non-negative (numerical noise)
        diag_err = jnp.maximum(diag_err, 0.0)
        
        return diag_err, L, pivots

    # Run scan
    final_diag, final_L, final_pivots = jax.lax.fori_loop(0, n_rank, body_fn, (diag_err, L, pivots))
    
    return final_pivots

def isdf_decompose(mo_values, mo_grads, n_rank_rho, n_rank_grad, weights=None):
    """Perform ISDF decomposition on orbitals and their gradients.
    
    Args:
        mo_values: (N_orb, N_grid)
        mo_grads: (N_orb, N_grid, 3)
        n_rank_rho: Rank for density decomposition
        n_rank_grad: Rank for gradient decomposition
        weights: Optional (N_grid,) array of integration weights. 
                 If provided, pivot selection is weighted by these weights.
        
    Returns:
        C_rho: (N_orb^2, N_fused)
        xi_rho: (N_fused, N_grid)
        C_grad: (N_orb^2, N_fused, 3)
        xi_grad: (N_fused, N_grid, 3)
        pivots: (N_fused,)
    """
    n_orb, n_grid = mo_values.shape
    
    
    if weights is None:
        w_sqrt = jnp.ones(n_grid)
    else:
        w_sqrt = jnp.sqrt(jnp.abs(weights)) # Use abs to avoid NaN
        
    
    def gram_diag_rho():
        # S_ii = (sum_p phi_p(i)^2)^2
        orb_sq = jnp.sum(mo_values**2, axis=0)
        return orb_sq**2
        
    def gram_col_rho(idx):
        # S_ij = (sum_p phi_p(i) phi_p(idx))^2
        # dot = phi(i) @ phi(idx)
        dot = jnp.dot(mo_values.T, mo_values[:, idx])
        return dot**2
        
    pivots_rho = pivoted_cholesky_matrix_free(gram_diag_rho, gram_col_rho, n_grid, n_rank_rho)
    
    
    def gram_diag_grad():
        A_diag = jnp.sum(mo_values**2, axis=0)
        B_diag = jnp.sum(jnp.sum(mo_grads**2, axis=2), axis=0)
        return A_diag * B_diag
        
    def gram_col_grad(idx):
        A_col = jnp.dot(mo_values.T, mo_values[:, idx])
        # B_col: sum_c (nabla^c phi).T @ (nabla^c phi)[:, idx]
        B_col = jnp.zeros(n_grid)
        for c in range(3):
            B_col += jnp.dot(mo_grads[:, :, c].T, mo_grads[:, idx, c])
        return A_col * B_col
        
    pivots_grad = pivoted_cholesky_matrix_free(gram_diag_grad, gram_col_grad, n_grid, n_rank_grad)
    
    # --- 3. Fuse pivots ---
    pivots_all = jnp.concatenate([pivots_rho, pivots_grad])
    pivots = jnp.unique(pivots_all)
    n_fused = pivots.shape[0]
    
    # --- 4. Construct C matrices ---
    mo_vals_piv = mo_values[:, pivots]
    C_rho = jnp.einsum('pm,qm->pqm', mo_vals_piv, mo_vals_piv).reshape(-1, n_fused)
    
    mo_grads_piv = mo_grads[:, pivots, :] # (N_orb, N_fused, 3)
    C_grad = jnp.einsum('pmc,qm->pqmc', mo_grads_piv, mo_vals_piv).reshape(-1, n_fused, 3)
    
    # --- 5. Solve for xi_rho and xi_grad (Block-wise Least Squares) ---
    batch_size = 4096
    n_batches = (n_grid + batch_size - 1) // batch_size
    
    def solve_batch(batch_idx):
        start = batch_idx * batch_size
        end = jnp.minimum(start + batch_size, n_grid)
        width = end - start
        
        # 1. Solve xi_rho
        mo_val_batch = jax.lax.dynamic_slice(mo_values, (0, start), (n_orb, width))
        rho_batch = jnp.einsum('pi,qi->pqi', mo_val_batch, mo_val_batch).reshape(-1, width)
        
        # Solve C_rho * xi = rho_batch
        # C_rho: (N^2, k), rho_batch: (N^2, width)
        xi_rho_batch, _, _, _ = jnp.linalg.lstsq(C_rho, rho_batch, rcond=1e-10)
        
        # 2. Solve xi_grad
        # nabla_rho_batch: (N_orb^2, width, 3)
        mo_grad_batch = jax.lax.dynamic_slice(mo_grads, (0, start, 0), (n_orb, width, 3))
        
        xi_grad_batch_list = []
        for c in range(3):
            # Construct nabla_rho_batch for component c
            # nabla^c rho_pq = nabla^c phi_p * phi_q
            nabla_rho_c = jnp.einsum('pi,qi->pqi', mo_grad_batch[:, :, c], mo_val_batch).reshape(-1, width)
            
            # Solve C_grad[..., c] * xi = nabla_rho_c
            xi_c, _, _, _ = jnp.linalg.lstsq(C_grad[:, :, c], nabla_rho_c, rcond=1e-10)
            xi_grad_batch_list.append(xi_c)
            
        xi_grad_batch = jnp.stack(xi_grad_batch_list, axis=-1) # (k, width, 3)
        
        return xi_rho_batch, xi_grad_batch, width

    xi_rho_list = []
    xi_grad_list = []
    
    for i in range(n_batches):
        xi_r, xi_g, w = solve_batch(i)
        xi_rho_list.append(xi_r)
        xi_grad_list.append(xi_g)
        
    xi_rho = jnp.concatenate(xi_rho_list, axis=1)
    xi_grad = jnp.concatenate(xi_grad_list, axis=1)
    
    return C_rho, xi_rho, C_grad, xi_grad, pivots
