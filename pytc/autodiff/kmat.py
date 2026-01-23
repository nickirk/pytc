"""JAX implementation of kinetic matrix elements."""
from functools import partial
import jax
import jax.numpy as jnp
import logging
import time

def calc_K1(phi, grad_phi, jastrow_factor, jastrow_params, grid_points, weights, ranges=None, batch_size=1000):
    r"""Calculate K1 matrix: K1_{pqrs} = \sum_{i,j} w_i w_j \phi_p(i) \phi_q(i) \nabla_i u(i, j) \phi_r(j) \phi_s(j)
    
    Args:
        phi: Orbitals on grid (Nb, N_grid)
        grad_phi: Orbital gradients on grid (Nb, N_grid, 3)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,) for integration weights
        batch_size: Number of grid points to process at once
        ranges: Optional tuple of (slice_p, slice_q, slice_r, slice_s).
                If None, computes full (Nb, Nb, Nb, Nb) matrix.
    """
    n_orb = phi.shape[0]
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
    phi_p = phi[slice_p]
    phi_q = phi[slice_q]
    phi_r = phi[slice_r]
    phi_s = phi[slice_s]
    
    grad_phi_p = grad_phi[slice_p]
    
    # Pad grid to multiple of batch_size
    padded_size = ((N_grid + batch_size - 1) // batch_size) * batch_size
    padded_grid = jnp.pad(grid_points, ((0, padded_size - N_grid), (0, 0)))
    
    # Reshape for scanning
    batched_grid = padded_grid.reshape(-1, batch_size, 3)
    
    # Prepare outer scan inputs (ket side: r, s)
    # We need phi_r and phi_s for the outer loop (r2 integration)
    padded_phi_r = jnp.pad(phi_r, ((0, 0), (0, padded_size - N_grid)))
    padded_phi_s = jnp.pad(phi_s, ((0, 0), (0, padded_size - N_grid)))
    padded_weights_r2 = jnp.pad(weights, ((0, padded_size - N_grid),))
    
    batched_phi_r = padded_phi_r.reshape(Nr, -1, batch_size).transpose(1, 0, 2)
    batched_phi_s = padded_phi_s.reshape(Ns, -1, batch_size).transpose(1, 0, 2)
    batched_weights_r2 = padded_weights_r2.reshape(-1, batch_size)
    
    @jax.checkpoint
    def outer_scan(carry, args):
        r2_batch, phi_r_batch, phi_s_batch, w_batch = args
        
        # Inner scan over r1 batches (bra side: p, q)
        @jax.checkpoint
        def inner_scan(inner_carry, inner_args):
            r1_batch, grad_phi_p_batch, phi_q_batch, weights_batch = inner_args
            
            # grad_phi_paired_bra = grad_phi_p * phi_q
            # Shape: (Np*Nq, batch, 3)
            grad_phi_paired_batch = jnp.einsum('pnd,qn->pqnd', grad_phi_p_batch, phi_q_batch).reshape(Np*Nq, -1, 3)
            
            grads = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            term = jnp.einsum('ijc,j,jkc->ik', grad_phi_paired_batch, weights_batch, grads)
            return inner_carry + term, None

        inner_bs = batch_size
        n_inner = (N_grid + inner_bs - 1) // inner_bs
        padded_inner_size = n_inner * inner_bs
        
        padded_r1 = jnp.pad(grid_points, ((0, padded_inner_size - N_grid), (0, 0)))
        padded_grad_phi_p = jnp.pad(grad_phi_p, ((0, 0), (0, padded_inner_size - N_grid), (0, 0)))
        padded_phi_q = jnp.pad(phi_q, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_weights = jnp.pad(weights, ((0, padded_inner_size - N_grid),))
        
        batched_r1 = padded_r1.reshape(n_inner, inner_bs, 3)
        batched_grad_phi_p = padded_grad_phi_p.reshape(Np, n_inner, inner_bs, 3).transpose(1, 0, 2, 3)
        batched_phi_q = padded_phi_q.reshape(Nq, n_inner, inner_bs).transpose(1, 0, 2)
        batched_weights = padded_weights.reshape(n_inner, inner_bs)
        
        tmp_init = jnp.zeros((Np*Nq, batch_size))
        tmp, _ = jax.lax.scan(inner_scan, tmp_init, (batched_r1, batched_grad_phi_p, batched_phi_q, batched_weights))
        
        # phi_paired_ket = phi_r * phi_s
        phi_paired_batch = jnp.einsum('rn,sn->rsn', phi_r_batch, phi_s_batch).reshape(Nr*Ns, -1)
        
        contrib = jnp.dot(phi_paired_batch * w_batch[None, :], tmp.T)
        return carry + contrib, None

    result_init = jnp.zeros((Nr*Ns, Np*Nq))
    final_result, _ = jax.lax.scan(outer_scan, result_init, (batched_grid, batched_phi_r, batched_phi_s, batched_weights_r2))
    
    return final_result.T

def calc_K3(phi, jastrow_factor, jastrow_params, grid_points, weights, ranges=None, batch_size=1000):
    r"""Calculate K3 matrix: K3_{pqrs} = \sum_{i,j} w_i w_j \phi_p(i) \phi_q(i) (\nabla_i u(i, j))^2 \phi_r(j) \phi_s(j)
    
    Args:
        phi: Orbitals on grid (Nb, N_grid)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,) for integration weights
        batch_size: Number of grid points to process at once
        ranges: Optional tuple of (slice_p, slice_q, slice_r, slice_s).
                If None, computes full (Nb, Nb, Nb, Nb) matrix.
    """
    n_orb = phi.shape[0]
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
    phi_p = phi[slice_p]
    phi_q = phi[slice_q]
    phi_r = phi[slice_r]
    phi_s = phi[slice_s]
    
    # Pad grid to multiple of batch_size
    padded_size = ((N_grid + batch_size - 1) // batch_size) * batch_size
    padded_grid = jnp.pad(grid_points, ((0, padded_size - N_grid), (0, 0)))
    
    batched_grid = padded_grid.reshape(-1, batch_size, 3)
    
    # Prepare inputs for outer scan (over r2 batches, ket side: r, s)
    padded_phi_r = jnp.pad(phi_r, ((0, 0), (0, padded_size - N_grid)))
    padded_phi_s = jnp.pad(phi_s, ((0, 0), (0, padded_size - N_grid)))
    padded_weights_r2 = jnp.pad(weights, ((0, padded_size - N_grid),))
    
    batched_phi_r = padded_phi_r.reshape(Nr, -1, batch_size).transpose(1, 0, 2)
    batched_phi_s = padded_phi_s.reshape(Ns, -1, batch_size).transpose(1, 0, 2)
    batched_weights_r2 = padded_weights_r2.reshape(-1, batch_size)
    
    @jax.checkpoint
    def outer_scan(carry, args):
        r2_batch, phi_r_batch, phi_s_batch, w_batch_r2 = args
        
        # Inner scan over r1 batches (bra side: p, q)
        @jax.checkpoint
        def inner_scan(inner_carry, inner_args):
            r1_batch, phi_p_batch, phi_q_batch, weights_batch_r1 = inner_args
            
            # Compute phi_paired for this batch of r1
            phi_paired_batch_r1 = jnp.einsum('pn,qn->pqn', phi_p_batch, phi_q_batch).reshape(Np*Nq, -1)
            
            # Compute gradients
            grads = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params) # (inner_bs, batch_size, 3)
            u_grad_squared = jnp.sum(grads**2, axis=-1) # (inner_bs, batch_size)
            
            # Weight: w(r1) * w(r2) * |grad u|^2
            weighted_u2 = u_grad_squared * weights_batch_r1[:, None] * w_batch_r2[None, :]
            
            # Contract with phi(r1): dot(weighted_u2, phi(r1).T)
            term = jnp.dot(phi_paired_batch_r1, weighted_u2)
            return inner_carry + term, None

        inner_bs = batch_size
        n_inner = (N_grid + inner_bs - 1) // inner_bs
        padded_inner_size = n_inner * inner_bs
        
        padded_r1 = jnp.pad(grid_points, ((0, padded_inner_size - N_grid), (0, 0)))
        padded_phi_p = jnp.pad(phi_p, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_phi_q = jnp.pad(phi_q, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_weights = jnp.pad(weights, ((0, padded_inner_size - N_grid),))
        
        batched_r1 = padded_r1.reshape(n_inner, inner_bs, 3)
        batched_phi_p = padded_phi_p.reshape(Np, n_inner, inner_bs).transpose(1, 0, 2)
        batched_phi_q = padded_phi_q.reshape(Nq, n_inner, inner_bs).transpose(1, 0, 2)
        batched_weights_r1 = padded_weights.reshape(n_inner, inner_bs)
        
        tmp_init = jnp.zeros((Np*Nq, batch_size))
        tmp, _ = jax.lax.scan(inner_scan, tmp_init, (batched_r1, batched_phi_p, batched_phi_q, batched_weights_r1))
        
        # tmp is (Np*Nq, batch_size)
        # Now accumulate result: sum_k [ phi_paired(r2_k) * tmp[p, k] ]
        
        phi_paired_batch_r2 = jnp.einsum('rn,sn->rsn', phi_r_batch, phi_s_batch).reshape(Nr*Ns, -1)
        
        contrib = jnp.dot(tmp, phi_paired_batch_r2.T) # (Np*Nq, Nr*Ns)
        return carry + contrib, None

    result_init = jnp.zeros((Np*Nq, Nr*Ns))
    final_result, _ = jax.lax.scan(outer_scan, result_init, (batched_grid, batched_phi_r, batched_phi_s, batched_weights_r2))
    
    return final_result


def calc_K1_kernel(xi_grad_r1, xi_phi_r2, weights_r1, weights_r2, jastrow_factor, jastrow_params, grid_r1, grid_r2, batch_size=1024):
    r"""Calculate K1 kernel: K1_{kl} = \sum_{g,h} w_g w_h \xi_{grad}(k,g) \nabla u(g,h) \xi_\phi(l,h)
    
    Args:
        xi_grad_r1: (N_fused, N_grid_r1, 3) - Host resident
        xi_phi_r2: (N_fused, N_grid_r2) - Host resident
        weights_r1: (N_grid_r1,)
        weights_r2: (N_grid_r2,)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_r1: (N_grid_r1, 3)
        grid_r2: (N_grid_r2, 3)
        batch_size: Batch size for grid integration
        
    Returns:
        K1_kernel: (N_fused, N_fused, 3)
    """
    N_grid_r1 = grid_r1.shape[0]
    N_grid_r2 = grid_r2.shape[0]
    n_fused = xi_phi_r2.shape[0]
    
    # Pad grids for scanning
    def get_padded_size(n):
        return ((n + batch_size - 1) // batch_size) * batch_size

    padded_size_r1 = get_padded_size(N_grid_r1)
    padded_size_r2 = get_padded_size(N_grid_r2)
    
    # Pad arrays (on host if they are large)
    if padded_size_r1 > N_grid_r1:
        grid_r1_padded = jnp.pad(grid_r1, ((0, padded_size_r1 - N_grid_r1), (0, 0)))
        weights_r1_padded = jnp.pad(weights_r1, ((0, padded_size_r1 - N_grid_r1),))
        xi_grad_r1_padded = jnp.pad(xi_grad_r1, ((0, 0), (0, padded_size_r1 - N_grid_r1), (0, 0)))
    else:
        grid_r1_padded, weights_r1_padded, xi_grad_r1_padded = grid_r1, weights_r1, xi_grad_r1
        
    if padded_size_r2 > N_grid_r2:
        grid_r2_padded = jnp.pad(grid_r2, ((0, padded_size_r2 - N_grid_r2), (0, 0)))
        weights_r2_padded = jnp.pad(weights_r2, ((0, padded_size_r2 - N_grid_r2),))
        xi_phi_r2_padded = jnp.pad(xi_phi_r2, ((0, 0), (0, padded_size_r2 - N_grid_r2)))
    else:
        grid_r2_padded, weights_r2_padded, xi_phi_r2_padded = grid_r2, weights_r2, xi_phi_r2

    n_batches_r1 = padded_size_r1 // batch_size
    n_batches_r2 = padded_size_r2 // batch_size

    def outer_scan(carry, i_batch_r2):
        # Slice r2 batch from host
        r2_batch = jax.lax.dynamic_slice(grid_r2_padded, (i_batch_r2 * batch_size, 0), (batch_size, 3))
        w2_batch = jax.lax.dynamic_slice(weights_r2_padded, (i_batch_r2 * batch_size,), (batch_size,))
        xi_phi_batch = jax.lax.dynamic_slice(xi_phi_r2_padded, (0, i_batch_r2 * batch_size), (n_fused, batch_size))
        
        def inner_scan(inner_carry, i_batch_r1):
            # Slice r1 batch from host
            r1_batch = jax.lax.dynamic_slice(grid_r1_padded, (i_batch_r1 * batch_size, 0), (batch_size, 3))
            w1_batch = jax.lax.dynamic_slice(weights_r1_padded, (i_batch_r1 * batch_size,), (batch_size,))
            xi_grad_batch = jax.lax.dynamic_slice(xi_grad_r1_padded, (0, i_batch_r1 * batch_size, 0), (n_fused, batch_size, 3))
            
            # Calculate gradients: grad_r1 u(r1, r2)
            # u_grad_batch: (batch_r1, batch_r2, 3)
            u_grad_batch = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            
            # Contract r1:
            # G1_{k,b,c} = sum_g (xi_grad_{k,g,c} * w1_g) * u_grad_batch_{g,b,c}
            # (N_fused, batch_r1, 3) * (batch_r1, batch_r2, 3) -> (N_fused, batch_r2, 3)
            G1_batch = jnp.einsum('kgc,g,gbc->kbc', xi_grad_batch, w1_batch, u_grad_batch)
            
            return inner_carry + G1_batch, None

        # Inner scan over r1 batches
        G1_init = jnp.zeros((n_fused, batch_size, 3))
        G1, _ = jax.lax.scan(inner_scan, G1_init, jnp.arange(n_batches_r1))
        
        # Contract r2:
        # K1_batch_{k,l,c} = sum_b G1_{k,b,c} * xi_phi_batch_{l,b} * w2_batch_{b}
        # Use matmul per component and stack to avoid copies
        K1_slices = []
        for c in range(3):
            G1_w = G1[:, :, c] * w2_batch[None, :]  # (N_fused, batch)
            K1_slices.append(jnp.matmul(G1_w, xi_phi_batch.T))
        K1_batch = jnp.stack(K1_slices, axis=-1)
        
        return carry + K1_batch, None

    init_val = jnp.zeros((n_fused, n_fused, 3))
    K1_kernel, _ = jax.lax.scan(outer_scan, init_val, jnp.arange(n_batches_r2))
    
    return K1_kernel


def calc_K3_kernel(xi_phi_r1, xi_phi_r2, weights_r1, weights_r2, jastrow_factor, jastrow_params, grid_r1, grid_r2, batch_size=1024):
    r"""Calculate K3 kernel: K3_{kl} = \sum_{g,h} w_g w_h \xi_\phi(k,g) |\nabla u(g,h)|^2 \xi_\phi(l,h)
    
    Args:
        xi_phi_r1: (N_fused, N_grid_r1) - Host resident
        xi_phi_r2: (N_fused, N_grid_r2) - Host resident
        weights_r1: (N_grid_r1,)
        weights_r2: (N_grid_r2,)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_r1: (N_grid_r1, 3)
        grid_r2: (N_grid_r2, 3)
        batch_size: Batch size for grid integration
        
    Returns:
        K3_kernel: (N_fused, N_fused)
    """
    N_grid_r1 = grid_r1.shape[0]
    N_grid_r2 = grid_r2.shape[0]
    n_fused = xi_phi_r2.shape[0]
    
    # Pad grids for scanning
    def get_padded_size(n):
        return ((n + batch_size - 1) // batch_size) * batch_size

    padded_size_r1 = get_padded_size(N_grid_r1)
    padded_size_r2 = get_padded_size(N_grid_r2)
    
    # Pad arrays
    if padded_size_r1 > N_grid_r1:
        grid_r1_padded = jnp.pad(grid_r1, ((0, padded_size_r1 - N_grid_r1), (0, 0)))
        weights_r1_padded = jnp.pad(weights_r1, ((0, padded_size_r1 - N_grid_r1),))
        xi_phi_r1_padded = jnp.pad(xi_phi_r1, ((0, 0), (0, padded_size_r1 - N_grid_r1)))
    else:
        grid_r1_padded, weights_r1_padded, xi_phi_r1_padded = grid_r1, weights_r1, xi_phi_r1
        
    if padded_size_r2 > N_grid_r2:
        grid_r2_padded = jnp.pad(grid_r2, ((0, padded_size_r2 - N_grid_r2), (0, 0)))
        weights_r2_padded = jnp.pad(weights_r2, ((0, padded_size_r2 - N_grid_r2),))
        xi_phi_r2_padded = jnp.pad(xi_phi_r2, ((0, 0), (0, padded_size_r2 - N_grid_r2)))
    else:
        grid_r2_padded, weights_r2_padded, xi_phi_r2_padded = grid_r2, weights_r2, xi_phi_r2

    n_batches_r1 = padded_size_r1 // batch_size
    n_batches_r2 = padded_size_r2 // batch_size

    def outer_scan(carry, i_batch_r2):
        # Slice r2 batch from host
        r2_batch = jax.lax.dynamic_slice(grid_r2_padded, (i_batch_r2 * batch_size, 0), (batch_size, 3))
        w2_batch = jax.lax.dynamic_slice(weights_r2_padded, (i_batch_r2 * batch_size,), (batch_size,))
        xi_phi_r2_batch = jax.lax.dynamic_slice(xi_phi_r2_padded, (0, i_batch_r2 * batch_size), (n_fused, batch_size))
        
        def inner_scan(inner_carry, i_batch_r1):
            # Slice r1 batch from host
            r1_batch = jax.lax.dynamic_slice(grid_r1_padded, (i_batch_r1 * batch_size, 0), (batch_size, 3))
            w1_batch = jax.lax.dynamic_slice(weights_r1_padded, (i_batch_r1 * batch_size,), (batch_size,))
            xi_phi_r1_batch = jax.lax.dynamic_slice(xi_phi_r1_padded, (0, i_batch_r1 * batch_size), (n_fused, batch_size))
            
            # Calculate gradients: grad_r1 u(r1, r2)
            # u_grad_batch: (batch_r1, batch_r2, 3)
            u_grad_batch = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            
            # Compute squared norm of gradients: (batch_r1, batch_r2)
            u_grad_norm_sq = jnp.sum(u_grad_batch**2, axis=-1)
            
            # Contract r1:
            # G3_{k,b} = sum_g (xi_phi_{k,g} * w1_g) * |grad u(g,b)|^2
            # (N_fused, batch_r1) * (batch_r1) * (batch_r1, batch_r2) -> (N_fused, batch_r2)
            G3_batch = jnp.einsum('kg,g,gb->kb', xi_phi_r1_batch, w1_batch, u_grad_norm_sq)
            
            return inner_carry + G3_batch, None

        # Inner scan over r1 batches
        G3_init = jnp.zeros((n_fused, batch_size))
        G3, _ = jax.lax.scan(inner_scan, G3_init, jnp.arange(n_batches_r1))
        
        # Contract r2:
        # K3_batch_{k,l} = sum_b G3_{k,b} * xi_phi_r2_batch_{l,b} * w2_batch_{b}
        # Use matmul: (G3 * w) @ xi_phi.T to avoid large intermediate
        G3_w = G3 * w2_batch[None, :]  # (N_fused, batch)
        K3_batch = jnp.matmul(G3_w, xi_phi_r2_batch.T)
        
        return carry + K3_batch, None

    init_val = jnp.zeros((n_fused, n_fused))
    K3_kernel, _ = jax.lax.scan(outer_scan, init_val, jnp.arange(n_batches_r2))
    
    return K3_kernel


def contract_K1_isdf(phi_piv, grad_phi_piv, U1, ranges=None):
    r"""Contract K1 using precomputed kernel and pivot values.
    
    K1_{pqrs} \approx \sum_{k,l} (\nabla\phi_p(z_k) \phi_q(z_k)) U1_{kl} (\phi_r(z_l) \phi_s(z_l))
    
    Args:
        phi_piv: (Nb, N_fused) Values of phi at pivot points
        grad_phi_piv: (Nb, N_fused, 3) Values of grad_phi at pivot points
        U1: (N_fused, N_fused, 3) Precomputed kernel
        ranges: Optional tuple of (slice_p, slice_q, slice_r, slice_s)
        
    Returns:
        K1: (Np, Nq, Nr, Ns)
    """
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
    else:
        slice_p, slice_q, slice_r, slice_s = ranges
        
    phi_p = phi_piv[slice_p]
    phi_q = phi_piv[slice_q]
    phi_r = phi_piv[slice_r]
    phi_s = phi_piv[slice_s]
    grad_phi_p = grad_phi_piv[slice_p]
    
@jax.jit
def contract_K1_isdf_jit(phi_p, phi_q, phi_r, phi_s, grad_phi_p, U1):
    """JITted version of K1 contraction.
    
    Memory-optimized: processes each spatial component (x, y, z) sequentially
    to avoid creating the full C_grad tensor of shape (Np, Nq, N_fused, 3).
    Peak memory is reduced from O(Np*Nq*N_fused*3) to O(Np*Nq*N_fused).
    """
    # C_phi_{rs, l} = phi_{r,l} phi_{s,l}
    C_phi = jnp.einsum('rl,sl->rsl', phi_r, phi_s)
    
    # Process each spatial component sequentially to avoid large C_grad
    # For each c: C_grad_c_{pq, k} = grad_phi_{p,k,c} * phi_{q,k}
    #             tmp_c_{pq, l} = sum_k C_grad_c_{pq, k} * U1_{k, l, c}
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_fused = U1.shape[0]
    
    def process_component(tmp_accum, c):
        # C_grad_c: (Np, Nq, N_fused) - only one component at a time
        C_grad_c = jnp.einsum('pk,qk->pqk', grad_phi_p[:, :, c], phi_q)
        # Contract with U1[:, :, c]: (Np, Nq, N_fused) @ (N_fused, N_fused) -> (Np, Nq, N_fused)
        tmp_c = jnp.einsum('pqk,kl->pql', C_grad_c, U1[:, :, c])
        return tmp_accum + tmp_c, None
    
    tmp_init = jnp.zeros((Np, Nq, N_fused))
    tmp, _ = jax.lax.scan(process_component, tmp_init, jnp.arange(3))
    
    return jnp.einsum('pql,rsl->pqrs', tmp, C_phi)

def contract_K1_isdf(phi_piv, grad_phi_piv, U1, ranges=None):
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
    else:
        slice_p, slice_q, slice_r, slice_s = ranges
        
    phi_p = phi_piv[slice_p]
    phi_q = phi_piv[slice_q]
    phi_r = phi_piv[slice_r]
    phi_s = phi_piv[slice_s]
    grad_phi_p = grad_phi_piv[slice_p]
    
    return contract_K1_isdf_jit(phi_p, phi_q, phi_r, phi_s, grad_phi_p, U1)


def contract_K3_isdf(phi_piv, U3, ranges=None):
    r"""Contract K3 using precomputed U3 kernel and pivot values.
    
    K3_{pqrs} \approx \sum_{k,l} (\phi_p(z_k) \phi_q(z_k)) U3_{kl} (\phi_r(z_l) \phi_s(z_l))
    
    Args:
        phi_piv: (Nb, N_fused) Values of phi at pivot points
        U3: (N_fused, N_fused) Precomputed kernel
        ranges: Optional tuple of (slice_p, slice_q, slice_r, slice_s)
        
    Returns:
        K3: (Np, Nq, Nr, Ns)
    """
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
    else:
        slice_p, slice_q, slice_r, slice_s = ranges
        
    phi_p = phi_piv[slice_p]
    phi_q = phi_piv[slice_q]
    phi_r = phi_piv[slice_r]
    phi_s = phi_piv[slice_s]
    
@jax.jit
def contract_K3_isdf_jit(phi_p, phi_q, phi_r, phi_s, U3):
    """JITted version of K3 contraction."""
    # C_phi_{pq, k} = phi_{p,k} phi_{q,k}
    C_phi_pq = jnp.einsum('pk,qk->pqk', phi_p, phi_q)
    # C_phi_{rs, l} = phi_{r,l} phi_{s,l}
    C_phi_rs = jnp.einsum('rl,sl->rsl', phi_r, phi_s)
    # K3 = sum_{k,l} C_phi_{pq,k} * U3_{k,l} * C_phi_{rs,l}
    # Break down to avoid O(N_orb^2 * N_rank^2) intermediate
    tmp = jnp.einsum('pqk,kl->pql', C_phi_pq, U3)
    return jnp.einsum('pql,rsl->pqrs', tmp, C_phi_rs)

def contract_K3_isdf(phi_piv, U3, ranges=None):
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
    else:
        slice_p, slice_q, slice_r, slice_s = ranges
        
    phi_p = phi_piv[slice_p]
    phi_q = phi_piv[slice_q]
    phi_r = phi_piv[slice_r]
    phi_s = phi_piv[slice_s]
    
    return contract_K3_isdf_jit(phi_p, phi_q, phi_r, phi_s, U3)
