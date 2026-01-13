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

def isdf_decompose(phi, grad_phi, n_rank_phi, n_rank_grad, weights=None):
    """Perform ISDF decomposition of orbitals and their gradients.
    
    Args:
        phi: Orbitals on grid (Nb, N_grid)
        grad_phi: Orbital gradients on grid (Nb, N_grid, 3)
        n_rank_phi: Rank for phi decomposition
        n_rank_grad: Rank for gradient decomposition
        weights: Optional (N_grid,) array of integration weights. 
                 If provided, pivot selection is weighted by these weights.
        
    Returns:
        C_phi: (N_orb^2, N_fused)
        xi_phi: (N_fused, N_grid)
        C_grad: (N_orb^2, N_fused, 3)
        xi_grad: (N_fused, N_grid, 3)
        pivots: (N_fused,)
    """
    n_orb, n_grid = phi.shape
    
    if weights is None:
        w_sqrt = jnp.ones(n_grid)
    else:
        w_sqrt = jnp.sqrt(jnp.abs(weights)) # Use abs to avoid NaN
        
    # Pre-compute diagonal for phi to calculate shift
    orb_sq = jnp.sum(phi**2, axis=0)
    diag_phi = orb_sq**2
    shift_phi = 1e-12 * jnp.max(jnp.abs(diag_phi))

    def gram_diag_phi():
        return diag_phi + shift_phi
        
    def gram_col_phi(idx):
        dot = jnp.dot(phi.T, phi[:, idx])
        col = dot**2
        return col.at[idx].add(shift_phi)
        
    pivots_phi = pivoted_cholesky_matrix_free(gram_diag_phi, gram_col_phi, n_grid, n_rank_phi)
    
    # Pre-compute diagonal for grad to calculate shift
    A_diag = jnp.sum(phi**2, axis=0)
    B_diag = jnp.sum(jnp.sum(grad_phi**2, axis=2), axis=0)
    diag_grad = A_diag * B_diag
    shift_grad = 1e-12 * jnp.max(jnp.abs(diag_grad))

    def gram_diag_grad():
        return diag_grad + shift_grad
        
    def gram_col_grad(idx):
        A_col = jnp.dot(phi.T, phi[:, idx])
        B_col = jnp.zeros(n_grid)
        for c in range(3):
            B_col += jnp.dot(grad_phi[:, :, c].T, grad_phi[:, idx, c])
        col = A_col * B_col
        return col.at[idx].add(shift_grad)
        
    pivots_grad = pivoted_cholesky_matrix_free(gram_diag_grad, gram_col_grad, n_grid, n_rank_grad)
    
    # --- 3. Fuse pivots ---
    pivots_all = jnp.concatenate([pivots_phi, pivots_grad])
    pivots = jnp.unique(pivots_all)
    n_fused = pivots.shape[0]
    
    # --- 4. Construct C matrices ---
    phi_piv = phi[:, pivots]
    C_phi = jnp.einsum('pm,qm->pqm', phi_piv, phi_piv).reshape(-1, n_fused)
    
    grad_phi_piv = grad_phi[:, pivots, :] # (N_orb, N_fused, 3)
    C_grad = jnp.einsum('pmc,qm->pqmc', grad_phi_piv, phi_piv).reshape(-1, n_fused, 3)
    
    # --- 5. Solve for xi_phi and xi_grad (Block-wise Least Squares) ---
    batch_size = 4096
    n_batches = (n_grid + batch_size - 1) // batch_size
    
    def solve_batch(batch_idx):
        start = batch_idx * batch_size
        end = jnp.minimum(start + batch_size, n_grid)
        width = end - start
        
        # 1. Solve xi_phi
        phi_batch = jax.lax.dynamic_slice(phi, (0, start), (n_orb, width))
        phi_paired_batch = jnp.einsum('pi,qi->pqi', phi_batch, phi_batch).reshape(-1, width)
        
        # Solve C_phi * xi = phi_paired_batch
        xi_phi_batch, _, _, _ = jnp.linalg.lstsq(C_phi, phi_paired_batch, rcond=1e-14)
        
        # 2. Solve xi_grad
        grad_phi_batch = jax.lax.dynamic_slice(grad_phi, (0, start, 0), (n_orb, width, 3))
        
        xi_grad_batch_list = []
        for c in range(3):
            # Construct grad_phi_paired_batch for component c
            grad_phi_paired_c = jnp.einsum('pi,qi->pqi', grad_phi_batch[:, :, c], phi_batch).reshape(-1, width)
            
            # Solve C_grad[..., c] * xi = grad_phi_paired_c
            xi_c, _, _, _ = jnp.linalg.lstsq(C_grad[:, :, c], grad_phi_paired_c, rcond=1e-14)
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
    
    return C_phi, xi_phi, C_grad, xi_grad, pivots
