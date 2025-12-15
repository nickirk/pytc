"""JAX implementation of kinetic matrix elements."""
from functools import partial
import jax
import jax.numpy as jnp

def calc_K1(rho, nabla_rho, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000, ranges=None):
    """JAX implementation of K1 term with memory-efficient batching using scan.
    
    Args:
        rho: Array of shape (Nb, N_grid)
        nabla_rho: Array of shape (Nb, N_grid, 3)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,) for integration weights
        batch_size: Number of grid points to process at once
        ranges: Optional tuple of (slice_p, slice_q, slice_r, slice_s).
                If None, computes full (Nb^2, Nb^2) matrix.
    """
    n_orb = rho.shape[0]
    N_grid = grid_points.shape[0]
    weights = jnp.asarray(weights)
    
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
        Np = Nq = Nr = Ns = n_orb
    else:
        slice_p, slice_q, slice_r, slice_s = ranges
        # Helper to get size from slice
        def get_size(s, size):
            start, stop, step = s.indices(size)
            return (stop - start + (step - 1)) // step
        Np = get_size(slice_p, n_orb)
        Nq = get_size(slice_q, n_orb)
        Nr = get_size(slice_r, n_orb)
        Ns = get_size(slice_s, n_orb)

    # Slice input arrays
    rho_p = rho[slice_p]
    rho_q = rho[slice_q]
    rho_r = rho[slice_r]
    rho_s = rho[slice_s]
    
    nabla_rho_p = nabla_rho[slice_p]
    nabla_rho_r = nabla_rho[slice_r]
    
    # Pad grid to multiple of batch_size
    padded_size = ((N_grid + batch_size - 1) // batch_size) * batch_size
    padded_grid = jnp.pad(grid_points, ((0, padded_size - N_grid), (0, 0)))
    
    # Reshape for scanning
    batched_grid = padded_grid.reshape(-1, batch_size, 3)
    
    # Prepare outer scan inputs (ket side: q, s)
    # We need rho_q and rho_s for the outer loop (r2 integration)
    padded_rho_q = jnp.pad(rho_q, ((0, 0), (0, padded_size - N_grid)))
    padded_rho_s = jnp.pad(rho_s, ((0, 0), (0, padded_size - N_grid)))
    padded_weights_r2 = jnp.pad(weights, ((0, padded_size - N_grid),))
    
    batched_rho_q = padded_rho_q.reshape(Nq, -1, batch_size).transpose(1, 0, 2)
    batched_rho_s = padded_rho_s.reshape(Ns, -1, batch_size).transpose(1, 0, 2)
    batched_weights_r2 = padded_weights_r2.reshape(-1, batch_size)
    
    @jax.checkpoint
    def outer_scan(carry, args):
        r2_batch, rho_q_batch, rho_s_batch, w_batch = args
        
        # Inner scan over r1 batches (bra side: p, r)
        @jax.checkpoint
        def inner_scan(inner_carry, inner_args):
            r1_batch, nabla_rho_p_batch, rho_r_batch, nabla_rho_r_batch, rho_p_batch, weights_batch = inner_args
            
            # nabla_rho_paired_bra = nabla_rho_p * rho_r
            # Shape: (Np*Nr, batch, 3)
            nabla_rho_paired_batch = jnp.einsum('pnd,rn->prnd', nabla_rho_p_batch, rho_r_batch).reshape(Np*Nr, -1, 3)
            
            grads = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            term = jnp.einsum('ijc,j,jkc->ik', nabla_rho_paired_batch, weights_batch, grads)
            return inner_carry + term, None

        inner_bs = batch_size
        n_inner = (N_grid + inner_bs - 1) // inner_bs
        padded_inner_size = n_inner * inner_bs
        
        padded_r1 = jnp.pad(grid_points, ((0, padded_inner_size - N_grid), (0, 0)))
        padded_nabla_p = jnp.pad(nabla_rho_p, ((0, 0), (0, padded_inner_size - N_grid), (0, 0)))
        padded_rho_r = jnp.pad(rho_r, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_nabla_r = jnp.pad(nabla_rho_r, ((0, 0), (0, padded_inner_size - N_grid), (0, 0)))
        padded_rho_p = jnp.pad(rho_p, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_weights = jnp.pad(weights, ((0, padded_inner_size - N_grid),))
        
        batched_r1 = padded_r1.reshape(n_inner, inner_bs, 3)
        batched_nabla_p = padded_nabla_p.reshape(Np, n_inner, inner_bs, 3).transpose(1, 0, 2, 3)
        batched_rho_r = padded_rho_r.reshape(Nr, n_inner, inner_bs).transpose(1, 0, 2)
        batched_nabla_r = padded_nabla_r.reshape(Nr, n_inner, inner_bs, 3).transpose(1, 0, 2, 3)
        batched_rho_p = padded_rho_p.reshape(Np, n_inner, inner_bs).transpose(1, 0, 2)
        batched_weights = padded_weights.reshape(n_inner, inner_bs)
        
        tmp_init = jnp.zeros((Np*Nr, batch_size))
        tmp, _ = jax.lax.scan(inner_scan, tmp_init, (batched_r1, batched_nabla_p, batched_rho_r, batched_nabla_r, batched_rho_p, batched_weights))
        
        # rho_paired_ket = rho_q * rho_s
        rho_paired_batch = jnp.einsum('qn,sn->qsn', rho_q_batch, rho_s_batch).reshape(Nq*Ns, -1)
        
        contrib = jnp.dot(rho_paired_batch * w_batch[None, :], tmp.T)
        return carry + contrib, None

    result_init = jnp.zeros((Nq*Ns, Np*Nr))
    final_result, _ = jax.lax.scan(outer_scan, result_init, (batched_grid, batched_rho_q, batched_rho_s, batched_weights_r2))
    
    return final_result.T

def calc_K3(rho, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000, ranges=None):
    """JAX implementation of K3 term with memory-efficient batching using scan.
    
    Args:
        rho: Array of shape (Nb, N_grid)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,) for integration weights
        batch_size: Number of grid points to process at once
        ranges: Optional tuple of (slice_p, slice_q, slice_r, slice_s).
                If None, computes full (Nb^2, Nb^2) matrix.
    """
    n_orb = rho.shape[0]
    N_grid = grid_points.shape[0]
    weights = jnp.asarray(weights)
    
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
        Np = Nq = Nr = Ns = n_orb
    else:
        slice_p, slice_q, slice_r, slice_s = ranges
        # Helper to get size from slice
        def get_size(s, size):
            start, stop, step = s.indices(size)
            return (stop - start + (step - 1)) // step
        Np = get_size(slice_p, n_orb)
        Nq = get_size(slice_q, n_orb)
        Nr = get_size(slice_r, n_orb)
        Ns = get_size(slice_s, n_orb)

    # Slice input arrays
    rho_p = rho[slice_p]
    rho_q = rho[slice_q]
    rho_r = rho[slice_r]
    rho_s = rho[slice_s]
    
    # Pad grid to multiple of batch_size
    padded_size = ((N_grid + batch_size - 1) // batch_size) * batch_size
    padded_grid = jnp.pad(grid_points, ((0, padded_size - N_grid), (0, 0)))
    
    batched_grid = padded_grid.reshape(-1, batch_size, 3)
    
    # Prepare inputs for outer scan (over r2 batches, ket side: q, s)
    padded_rho_q = jnp.pad(rho_q, ((0, 0), (0, padded_size - N_grid)))
    padded_rho_s = jnp.pad(rho_s, ((0, 0), (0, padded_size - N_grid)))
    padded_weights_r2 = jnp.pad(weights, ((0, padded_size - N_grid),))
    
    batched_rho_q = padded_rho_q.reshape(Nq, -1, batch_size).transpose(1, 0, 2)
    batched_rho_s = padded_rho_s.reshape(Ns, -1, batch_size).transpose(1, 0, 2)
    batched_weights_r2 = padded_weights_r2.reshape(-1, batch_size)
    
    @jax.checkpoint
    def outer_scan(carry, args):
        r2_batch, rho_q_batch, rho_s_batch, w_batch_r2 = args
        
        # Inner scan over r1 batches (bra side: p, r)
        @jax.checkpoint
        def inner_scan(inner_carry, inner_args):
            r1_batch, rho_p_batch, rho_r_batch, weights_batch_r1 = inner_args
            
            # Compute rho_paired for this batch of r1
            rho_paired_batch_r1 = jnp.einsum('pn,rn->prn', rho_p_batch, rho_r_batch).reshape(Np*Nr, -1)
            
            # Compute gradients
            grads = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params) # (inner_bs, batch_size, 3)
            u_grad_squared = jnp.sum(grads**2, axis=-1) # (inner_bs, batch_size)
            
            # Weight: w(r1) * w(r2) * |grad u|^2
            weighted_u2 = u_grad_squared * weights_batch_r1[:, None] * w_batch_r2[None, :]
            
            # Contract with rho(r1): dot(weighted_u2, rho(r1).T)
            term = jnp.dot(rho_paired_batch_r1, weighted_u2)
            return inner_carry + term, None

        inner_bs = batch_size
        n_inner = (N_grid + inner_bs - 1) // inner_bs
        padded_inner_size = n_inner * inner_bs
        
        padded_r1 = jnp.pad(grid_points, ((0, padded_inner_size - N_grid), (0, 0)))
        padded_rho_p = jnp.pad(rho_p, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_rho_r = jnp.pad(rho_r, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_weights = jnp.pad(weights, ((0, padded_inner_size - N_grid),))
        
        batched_r1 = padded_r1.reshape(n_inner, inner_bs, 3)
        batched_rho_p = padded_rho_p.reshape(Np, n_inner, inner_bs).transpose(1, 0, 2)
        batched_rho_r = padded_rho_r.reshape(Nr, n_inner, inner_bs).transpose(1, 0, 2)
        batched_weights_r1 = padded_weights.reshape(n_inner, inner_bs)
        
        tmp_init = jnp.zeros((Np*Nr, batch_size))
        tmp, _ = jax.lax.scan(inner_scan, tmp_init, (batched_r1, batched_rho_p, batched_rho_r, batched_weights_r1))
        
        # tmp is (Np*Nr, batch_size)
        # Now accumulate result: sum_k [ rho_paired(r2_k) * tmp[p, k] ]
        
        rho_paired_batch_r2 = jnp.einsum('qn,sn->qsn', rho_q_batch, rho_s_batch).reshape(Nq*Ns, -1)
        
        contrib = jnp.dot(tmp, rho_paired_batch_r2.T) # (Np*Nr, Nq*Ns)
        return carry + contrib, None

    result_init = jnp.zeros((Np*Nr, Nq*Ns))
    final_result, _ = jax.lax.scan(outer_scan, result_init, (batched_grid, batched_rho_q, batched_rho_s, batched_weights_r2))
    
    return final_result


def calc_K1_isdf(C_rho, xi_rho, C_grad, xi_grad, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K1 term using ISDF intermediates with memory-efficient batching.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        C_grad: (Nb^2, n_fused, 3) Selected columns for nabla_rho
        xi_grad: (n_fused, N_grid, 3) Interpolation coeffs for grad
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    N_grid = grid_points.shape[0]
    Nb2 = C_rho.shape[0]
    n_fused = xi_rho.shape[0]
    
    result = jnp.zeros((Nb2, Nb2))
    weights = jnp.asarray(weights)
    weighted_xi_grad = xi_grad * weights[None, :, None]
    
    # Process r2 points in batches
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        batch_points = grid_points[i:i_end]
        
        @partial(jax.vmap, in_axes=(None, 0))
        def get_grads(r1, r2):
            return jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]
        
        u_grad_batch = jax.vmap(get_grads, in_axes=(0, None))(grid_points, batch_points)  # (N_grid, batch, 3)
        
        # Process each spatial component
        G1_components = []
        for c in range(3):
            xi_slice = weighted_xi_grad[:, :, c]  # (n_fused, N_grid)
            u_slice = u_grad_batch[:, :, c]      # (N_grid, batch)
            G1_c = jnp.dot(xi_slice, u_slice)     # (n_fused, batch)
            G1_components.append(G1_c)
        
        G1 = jnp.stack(G1_components, axis=-1)    # (n_fused, batch, 3)
        G1 = jnp.einsum('kmc,pkc->pm', G1, C_grad)  # (Nb^2, batch)
        
        # Contract with xi_rho and weights for this batch
        G2 = jnp.einsum('pm,lm,m->pl', G1, xi_rho[:, i:i_end], weights[i:i_end])
        
        # Accumulate result
        result += jnp.einsum('pl,ql->pq', G2, C_rho)
    
    return result


def calc_K2_isdf(C_rho, xi_rho, C_grad, xi_grad, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K2 term using ISDF intermediates.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        C_grad: (Nb^2, n_fused, 3) Selected columns for nabla_rho
        xi_grad: (n_fused, N_grid, 3) Interpolation coeffs for grad
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    # Step 1: Compute the combined nabla term (∇ϕₑϕₛ + ϕₑ∇ϕₛ)
    Nb = int(jnp.sqrt(C_grad.shape[0]))
    C_grad_reshaped = C_grad.reshape(Nb, Nb, -1, 3)
    C_grad_transposed = C_grad_reshaped.transpose(1, 0, 2, 3).reshape(-1, C_grad.shape[1], 3)
    combined_C_grad = C_grad + C_grad_transposed

    # Step 2: Reuse calc_K1_isdf with the combined gradient term
    result = -calc_K1_isdf(C_rho, xi_rho, combined_C_grad, xi_grad,
                          jastrow_factor, jastrow_params, grid_points, weights, batch_size)

    return result


def calc_K3_isdf(C_rho, xi_rho, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K3 term using ISDF intermediates with memory-efficient batching.
    
    Args:
        C_rho: (Nb^2, n_fused) Selected columns for rho
        xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb^2, Nb^2) array in chemists' notation (pr|qs)
    """
    N_grid = grid_points.shape[0]
    Nb2 = C_rho.shape[0]
    n_fused = xi_rho.shape[0]
    
    result = jnp.zeros((Nb2, Nb2))
    weights = jnp.asarray(weights)
    
    # Process r2 points in batches
    for i in range(0, N_grid, batch_size):
        i_end = min(i + batch_size, N_grid)
        batch_points = grid_points[i:i_end]
        
        @partial(jax.vmap, in_axes=(None, 0))
        def get_grads(r1, r2):
            return jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]
        
        u_grad_batch = jax.vmap(get_grads, in_axes=(0, None))(grid_points, batch_points)  # (N_grid, batch, 3)
        
        # Compute squared magnitude of gradient
        u_grad_squared = jnp.sum(u_grad_batch**2, axis=-1)  # (N_grid, batch)
        
        # Weight both coordinates
        weighted_u_squared = u_grad_squared * weights[:, None] * weights[None, i:i_end]  # (N_grid, batch)
        
        # Contract with xi_rho for this batch
        G1 = jnp.einsum('ki,ij,lj->kl', xi_rho, weighted_u_squared, xi_rho[:, i:i_end])
        
        # Accumulate result
        result += jnp.einsum('kl,pk,ql->pq', G1, C_rho, C_rho)
    
    return result
