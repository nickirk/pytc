"""JAX implementation of kinetic matrix elements."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

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
    n_fused_r1 = xi_grad_r1.shape[0]
    n_fused_r2 = xi_phi_r2.shape[0]

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
        xi_phi_batch = jax.lax.dynamic_slice(xi_phi_r2_padded, (0, i_batch_r2 * batch_size), (n_fused_r2, batch_size))

        def inner_scan(inner_carry, i_batch_r1):
            # Slice r1 batch from host
            r1_batch = jax.lax.dynamic_slice(grid_r1_padded, (i_batch_r1 * batch_size, 0), (batch_size, 3))
            w1_batch = jax.lax.dynamic_slice(weights_r1_padded, (i_batch_r1 * batch_size,), (batch_size,))
            xi_grad_batch = jax.lax.dynamic_slice(xi_grad_r1_padded, (0, i_batch_r1 * batch_size, 0), (n_fused_r1, batch_size, 3))

            # Calculate gradients: grad_r1 u(r1, r2)
            # u_grad_batch: (batch_r1, batch_r2, 3)
            u_grad_batch = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)

            # Contract r1:
            # G1_{k,b,c} = sum_g (xi_grad_{k,g,c} * w1_g) * u_grad_batch_{g,b,c}
            # (N_fused_r1, batch_r1, 3) * (batch_r1, batch_r2, 3) -> (N_fused_r1, batch_r2, 3)
            G1_batch = jnp.einsum('kgc,g,gbc->kbc', xi_grad_batch, w1_batch, u_grad_batch)

            return inner_carry + G1_batch, None

        # Inner scan over r1 batches
        G1_init = jnp.zeros((n_fused_r1, batch_size, 3))
        G1, _ = jax.lax.scan(inner_scan, G1_init, jnp.arange(n_batches_r1))

        # Contract r2 and accumulate each component directly into carry; avoids
        # holding all three (n_fused_r1, n_fused_r2) slices and a stacked
        # (n_fused_r1, n_fused_r2, 3) tensor concurrently. XLA can fuse the
        # scatter-add into the donated scan carry.
        new_carry = carry
        for c in range(3):
            G1_w = G1[:, :, c] * w2_batch[None, :]              # (N_fused_r1, batch)
            K1_c = jnp.matmul(G1_w, xi_phi_batch.T)             # (N_fused_r1, N_fused_r2)
            new_carry = new_carry.at[:, :, c].add(K1_c)

        return new_carry, None

    init_val = jnp.zeros((n_fused_r1, n_fused_r2, 3))
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
    n_fused_r1 = xi_phi_r1.shape[0]
    n_fused_r2 = xi_phi_r2.shape[0]

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
        xi_phi_r2_batch = jax.lax.dynamic_slice(xi_phi_r2_padded, (0, i_batch_r2 * batch_size), (n_fused_r2, batch_size))

        def inner_scan(inner_carry, i_batch_r1):
            # Slice r1 batch from host
            r1_batch = jax.lax.dynamic_slice(grid_r1_padded, (i_batch_r1 * batch_size, 0), (batch_size, 3))
            w1_batch = jax.lax.dynamic_slice(weights_r1_padded, (i_batch_r1 * batch_size,), (batch_size,))
            xi_phi_r1_batch = jax.lax.dynamic_slice(xi_phi_r1_padded, (0, i_batch_r1 * batch_size), (n_fused_r1, batch_size))

            # Calculate gradients: grad_r1 u(r1, r2)
            # u_grad_batch: (batch_r1, batch_r2, 3)
            u_grad_batch = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)

            # Compute squared norm of gradients: (batch_r1, batch_r2)
            u_grad_norm_sq = jnp.sum(u_grad_batch**2, axis=-1)

            # Contract r1:
            # G3_{k,b} = sum_g (xi_phi_{k,g} * w1_g) * |grad u(g,b)|^2
            # (N_fused_r1, batch_r1) * (batch_r1) * (batch_r1, batch_r2) -> (N_fused_r1, batch_r2)
            G3_batch = jnp.einsum('kg,g,gb->kb', xi_phi_r1_batch, w1_batch, u_grad_norm_sq)

            return inner_carry + G3_batch, None

        # Inner scan over r1 batches
        G3_init = jnp.zeros((n_fused_r1, batch_size))
        G3, _ = jax.lax.scan(inner_scan, G3_init, jnp.arange(n_batches_r1))

        # Contract r2:
        # K3_batch_{k,l} = sum_b G3_{k,b} * xi_phi_r2_batch_{l,b} * w2_batch_{b}
        # Use matmul: (G3 * w) @ xi_phi.T to avoid large intermediate
        G3_w = G3 * w2_batch[None, :]  # (N_fused_r1, batch)
        K3_batch = jnp.matmul(G3_w, xi_phi_r2_batch.T)

        return carry + K3_batch, None

    init_val = jnp.zeros((n_fused_r1, n_fused_r2))
    K3_kernel, _ = jax.lax.scan(outer_scan, init_val, jnp.arange(n_batches_r2))

    return K3_kernel


@partial(jax.jit, static_argnums=(6,))
def contract_K1_isdf_jit(phi_p, phi_q, phi_r, phi_s, grad_phi_p, U1, rank_block_size=128):
    """JITted version of K1 contraction.
    
    Memory-optimized: processes each spatial component (x, y, z) sequentially
    to avoid creating the full C_grad tensor of shape (Np, Nq, N_fused, 3).
    Peak memory is reduced from O(Np*Nq*N_fused*3) to O(Np*Nq*N_fused).
    
    Args:
        rank_block_size: Block size for scanning the ISDF rank dimension.
            Larger values = fewer scan iterations but more VRAM per step.
            This is a static argument — JAX recompiles if it changes.
    """
    # C_phi_{rs, l} = phi_{r,l} phi_{s,l}
    C_phi = jnp.einsum('rl,sl->rsl', phi_r, phi_s)
    
    # Define dimensions first
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_fused = U1.shape[0]
    
    # We iterate over blocks of l.
    n_rank = U1.shape[1]
    
    # Pad rank dimension to multiple of block size
    padded_rank = ((n_rank + rank_block_size - 1) // rank_block_size) * rank_block_size
    pad_width = padded_rank - n_rank
    
    # Pad U1 along axis 1 (l index)
    U1_padded = jnp.pad(U1, ((0, 0), (0, pad_width), (0, 0)))
    
    # Pad phi_r and phi_s along axis 1 (l index)
    phi_r_padded = jnp.pad(phi_r, ((0, 0), (0, pad_width)))
    phi_s_padded = jnp.pad(phi_s, ((0, 0), (0, pad_width)))
    
    # Reshape for scan: (n_blocks, block_size, ...)
    n_blocks = padded_rank // rank_block_size
    
    # U1: (N_fused, n_blocks, block, 3) -> (n_blocks, N_fused, block, 3)
    U1_scannable = U1_padded.reshape(N_fused, n_blocks, rank_block_size, 3).transpose(1, 0, 2, 3)
    
    # phi_r: (Nr, n_blocks, block) -> (n_blocks, Nr, block)
    phi_r_scannable = phi_r_padded.reshape(Nr, n_blocks, rank_block_size).transpose(1, 0, 2)
    # phi_s: (Ns, n_blocks, block) -> (n_blocks, Ms, block)
    phi_s_scannable = phi_s_padded.reshape(Ns, n_blocks, rank_block_size).transpose(1, 0, 2)
    
    def scan_l_block(carry, args):
        U1_block, phi_r_block, phi_s_block = args
        # 1. Compute T_block[p, q, l_local]
        # Sum over spatial component c
        def process_component(T_acc, c):
            U1_slice = U1_block[:, :, c] # (N_fused, block)
            
            # W[p, k, l'] = grad_phi_p[p, k, c] * U1_slice[k, l']
            # Broadcasting: (Np, k, 1) * (1, k, block) -> (Np, k, block)
            W = grad_phi_p[:,:,c][:,:,None] * U1_slice[None,:,:] 
            
            # Contract k: T_c[p, l', q] = sum_k W[p, k, l'] * phi_q[q, k]
            # Reshape W to treat (p, l') as batch dimensions if needed, or permute
            # W_perm: (Np, block, k)
            W_perm = jnp.transpose(W, (0, 2, 1))
            W_2d = W_perm.reshape(Np*rank_block_size, N_fused)
            
            # T_flat = W_2d @ phi_q.T
            T_flat = jnp.matmul(W_2d, phi_q.T) # (Np*block, Nq)
            
            # Reshape back to (Np, block, Nq) -> (Np, Nq, block)
            T_c = T_flat.reshape(Np, rank_block_size, Nq)
            T_c = jnp.transpose(T_c, (0, 2, 1))
            
            return T_acc + T_c, None

        T_init = jnp.zeros((Np, Nq, rank_block_size))
        T_block, _ = jax.lax.scan(process_component, T_init, jnp.arange(3))
        
        # 2. Form C_rs[r, s, l_local]
        C_rs = phi_r_block[:, None, :] * phi_s_block[None, :, :] # (Nr, Ns, block)
        
        # 3. Contract: sum_l T_block[p,q,l] * C_rs[r,s,l]
        contribution = jnp.einsum('pql,rsl->pqrs', T_block, C_rs)
        
        return carry + contribution, None

    K1_init = jnp.zeros((Np, Nq, Nr, Ns))
    K1_final, _ = jax.lax.scan(scan_l_block, K1_init, (U1_scannable, phi_r_scannable, phi_s_scannable))
    
    return K1_final

def contract_K1_isdf(phi_piv, grad_phi_piv, U1, ranges=None, rank_block_size=None,
                     gpu_max_memory_mb=None):
    """Contract K1 using ISDF decomposition.
    
    Args:
        rank_block_size: Override for the ISDF rank scan block size.
            If None, an adaptive size is computed based on the orbital
            slice dimensions and available GPU memory.
        gpu_max_memory_mb: GPU memory budget for adaptive block sizing.
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
    
    if rank_block_size is None:
        from pytc.utils.gpu_memory import adaptive_rank_block_size
        rank_block_size = adaptive_rank_block_size(
            phi_p.shape[0], phi_q.shape[0], U1.shape[0],
            gpu_max_memory_mb=gpu_max_memory_mb)
    
    return contract_K1_isdf_jit(phi_p, phi_q, phi_r, phi_s, grad_phi_p, U1,
                                rank_block_size)


@partial(jax.jit, static_argnums=(7,))
def contract_K1_minus_K2_isdf_jit(phi_p, phi_q, phi_r, phi_s,
                                   grad_phi_p, grad_phi_q, U1,
                                   rank_block_size=128):
    """Compute (K1 - K2)[p,q,r,s] in a single scan pass.

    K1 uses grad_phi on index p; K2 uses grad_phi on index q (then transposes
    p↔q).  By computing both T-blocks in the same scan body and subtracting
    before contracting with C_rs, we use **one** (Np,Nq,Nr,Ns) accumulator
    instead of two, halving the peak memory compared to separate calls.

    Peak GPU: 1×(Np,Nq,Nr,Ns) carry + 1×(Np,Nq,Nr,Ns) contribution
            + 2×(Np,Nq,block) T-blocks + 1×(Nr,Ns,block) C_rs.
    """
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_fused = U1.shape[0]
    n_rank = U1.shape[1]

    # Pad rank dimension
    padded_rank = ((n_rank + rank_block_size - 1) // rank_block_size) * rank_block_size
    pad_width = padded_rank - n_rank
    U1_padded = jnp.pad(U1, ((0, 0), (0, pad_width), (0, 0)))
    phi_r_padded = jnp.pad(phi_r, ((0, 0), (0, pad_width)))
    phi_s_padded = jnp.pad(phi_s, ((0, 0), (0, pad_width)))

    n_blocks = padded_rank // rank_block_size
    U1_scannable = U1_padded.reshape(N_fused, n_blocks, rank_block_size, 3).transpose(1, 0, 2, 3)
    phi_r_scannable = phi_r_padded.reshape(Nr, n_blocks, rank_block_size).transpose(1, 0, 2)
    phi_s_scannable = phi_s_padded.reshape(Ns, n_blocks, rank_block_size).transpose(1, 0, 2)

    def _compute_T_block(grad_phi_bra, phi_ket, U1_block, N_bra, N_ket):
        """Compute T[bra, ket, l'] = sum_{k,c} grad_phi_bra[bra,k,c] U1[k,l',c] phi_ket[ket,k]."""
        def process_component(T_acc, c):
            U1_slice = U1_block[:, :, c]  # (N_fused, block)
            W = grad_phi_bra[:, :, c][:, :, None] * U1_slice[None, :, :]  # (N_bra, k, block)
            W_perm = jnp.transpose(W, (0, 2, 1))  # (N_bra, block, k)
            W_2d = W_perm.reshape(N_bra * rank_block_size, N_fused)
            T_flat = jnp.matmul(W_2d, phi_ket.T)  # (N_bra*block, N_ket)
            T_c = T_flat.reshape(N_bra, rank_block_size, N_ket)
            T_c = jnp.transpose(T_c, (0, 2, 1))  # (N_bra, N_ket, block)
            return T_acc + T_c, None

        T_init = jnp.zeros((N_bra, N_ket, rank_block_size))
        T_block, _ = jax.lax.scan(process_component, T_init, jnp.arange(3))
        return T_block

    def scan_l_block(carry, args):
        U1_block, phi_r_block, phi_s_block = args

        # T_K1[p, q, l'] using grad_phi_p
        T_K1 = _compute_T_block(grad_phi_p, phi_q, U1_block, Np, Nq)

        # T_K2[q, p, l'] using grad_phi_q  →  transpose to [p, q, l']
        T_K2 = _compute_T_block(grad_phi_q, phi_p, U1_block, Nq, Np)
        T_K2_T = jnp.transpose(T_K2, (1, 0, 2))  # (Np, Nq, block)

        # Combined T
        T_combined = T_K1 - T_K2_T  # (Np, Nq, block)

        # C_rs[r, s, l']
        C_rs = phi_r_block[:, None, :] * phi_s_block[None, :, :]  # (Nr, Ns, block)

        contribution = jnp.einsum('pql,rsl->pqrs', T_combined, C_rs)
        return carry + contribution, None

    init = jnp.zeros((Np, Nq, Nr, Ns))
    result, _ = jax.lax.scan(scan_l_block, init, (U1_scannable, phi_r_scannable, phi_s_scannable))
    return result


def contract_K1_minus_K2_isdf_fused(phi_p, phi_q, phi_r, phi_s,
                                    grad_phi_p, grad_phi_q, U1,
                                    r_block_size=128):
    """Fused (K1 - K2) contraction that contracts the rank dimension FIRST.

    The scan-loop version (``contract_K1_minus_K2_isdf_jit``) iterates over
    rank blocks and accumulates into the full (Np, Nq, Nr, Ns) output on
    every iteration.  For large Nr/Ns (e.g. 1179 at 1200 orbitals) the
    accumulator is ~2.2 GB and is re-read/re-written ~168 times, making the
    kernel memory-bound at ~0.2 FLOP/byte.

    This version computes the small T[p,q,l] tensor once (full rank, no
    blocking), then contracts T against phi_r/phi_s in r-blocks via real
    GEMMs.  The full 4D output is written exactly once.
    """
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_fused = U1.shape[0]
    n_rank = U1.shape[1]

    # --- Step 1: compute T_K1[p,q,l] and T_K2[p,q,l] (full rank) ---
    # T[bra,ket,l] = sum_{k,c} grad_phi_bra[bra,k,c] * U1[k,l,c] * phi_ket[ket,k]
    # Decompose per Cartesian component c, then sum.
    T_K1 = jnp.zeros((Np, Nq, n_rank), dtype=phi_p.dtype)
    T_K2 = jnp.zeros((Nq, Np, n_rank), dtype=phi_p.dtype)

    for c in range(3):
        U1_c = U1[:, :, c]  # (N_fused, n_rank)
        # K1: W[p,k,l] = grad_phi_p[p,k,c] * U1[k,l,c]
        W1 = grad_phi_p[:, :, c][:, :, None] * U1_c[None, :, :]  # (Np, N_fused, n_rank)
        W1_2d = jnp.transpose(W1, (0, 2, 1)).reshape(Np * n_rank, N_fused)
        T1_c = jnp.matmul(W1_2d, phi_q.T)  # (Np*n_rank, Nq)
        T1_c = T1_c.reshape(Np, n_rank, Nq).transpose(0, 2, 1)  # (Np, Nq, n_rank)
        T_K1 = T_K1 + T1_c

        # K2: W[q,k,l] = grad_phi_q[q,k,c] * U1[k,l,c]
        W2 = grad_phi_q[:, :, c][:, :, None] * U1_c[None, :, :]  # (Nq, N_fused, n_rank)
        W2_2d = jnp.transpose(W2, (0, 2, 1)).reshape(Nq * n_rank, N_fused)
        T2_c = jnp.matmul(W2_2d, phi_p.T)  # (Nq*n_rank, Np)
        T2_c = T2_c.reshape(Nq, n_rank, Np).transpose(0, 2, 1)  # (Nq, Np, n_rank)
        T_K2 = T_K2 + T2_c

    # Combined T[p,q,l] = T_K1[p,q,l] - T_K2_T[p,q,l]
    T_K2_T = jnp.transpose(T_K2, (1, 0, 2))  # (Np, Nq, n_rank)
    T_combined = T_K1 - T_K2_T  # (Np, Nq, n_rank)

    # --- Step 2: contract T against phi_r/phi_s in r-blocks ---
    # result[p,q,r,s] = sum_l T[p,q,l] * phi_r[r,l] * phi_s[s,l]
    # For each r-block:
    #   T_r[p,q,r,l] = T[p,q,l] * phi_r[r,l]  (broadcast multiply)
    #   result[p,q,r_block,s] = T_r_flat @ phi_s  (GEMM over l)
    result = jnp.zeros((Np, Nq, Nr, Ns), dtype=phi_p.dtype)

    for r0 in range(0, Nr, r_block_size):
        r1 = min(r0 + r_block_size, Nr)
        phi_r_block = phi_r[r0:r1, :]  # (R, n_rank)
        R = r1 - r0
        # Broadcast multiply: T_r[p,q,r,l] = T[p,q,l] * phi_r_block[r,l]
        # T_combined: (Np, Nq, n_rank), phi_r_block: (R, n_rank)
        # → (Np, Nq, R, n_rank) via broadcasting
        T_r = T_combined[:, :, None, :] * phi_r_block[None, None, :, :]  # (Np, Nq, R, n_rank)
        # Flatten to (Np*Nq*R, n_rank) for GEMM
        T_r_flat = T_r.reshape(Np * Nq * R, n_rank)
        # GEMM: (Np*Nq*R, n_rank) @ (n_rank, Ns) → (Np*Nq*R, Ns)
        block_result = jnp.matmul(T_r_flat, phi_s.T)  # (Np*Nq*R, Ns)
        # Reshape and place into output
        block_result = block_result.reshape(Np, Nq, R, Ns)
        result = jax.lax.dynamic_update_slice(result, block_result, (0, 0, r0, 0))

    return result


@partial(jax.jit, static_argnums=(5,))
def contract_K1_antisym_pq_isdf_jit(phi_p, phi_r, phi_s, grad_phi_p, U1,
                                     rank_block_size=128):
    """Compute ``K1[p,q,r,s] - K1[q,p,r,s]`` for the symmetric case
    ``phi_p == phi_q`` (and ``grad_phi_p == grad_phi_q``).

    Antisymmetrises the small ``(Np, Np, block)`` T tensor inside the rank
    scan, then contracts against ``C_rs[r,s,l]``.  This avoids materialising
    the full ``(Np, Np, Nr, Ns)`` K1 intermediate and its transposed copy
    (the dominant cost of the legacy ``k12 - k12.T(1,0,2,3)`` path).
    """
    Np = phi_p.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_fused = U1.shape[0]
    n_rank = U1.shape[1]

    padded_rank = ((n_rank + rank_block_size - 1) // rank_block_size) * rank_block_size
    pad_width = padded_rank - n_rank
    U1_padded = jnp.pad(U1, ((0, 0), (0, pad_width), (0, 0)))
    phi_r_padded = jnp.pad(phi_r, ((0, 0), (0, pad_width)))
    phi_s_padded = jnp.pad(phi_s, ((0, 0), (0, pad_width)))

    n_blocks = padded_rank // rank_block_size
    U1_scannable = U1_padded.reshape(N_fused, n_blocks, rank_block_size, 3).transpose(1, 0, 2, 3)
    phi_r_scannable = phi_r_padded.reshape(Nr, n_blocks, rank_block_size).transpose(1, 0, 2)
    phi_s_scannable = phi_s_padded.reshape(Ns, n_blocks, rank_block_size).transpose(1, 0, 2)

    def scan_l_block(carry, args):
        U1_block, phi_r_block, phi_s_block = args

        # T[p,q,l'] = sum_{k,c} grad_phi_p[p,k,c] U1[k,l',c] phi_p[q,k]
        def process_component(T_acc, c):
            U1_slice = U1_block[:, :, c]                              # (N_fused, block)
            W = grad_phi_p[:, :, c][:, :, None] * U1_slice[None, :, :] # (Np, k, block)
            W_perm = jnp.transpose(W, (0, 2, 1))                       # (Np, block, k)
            W_2d = W_perm.reshape(Np * rank_block_size, N_fused)
            T_flat = jnp.matmul(W_2d, phi_p.T)                         # (Np*block, Np)
            T_c = T_flat.reshape(Np, rank_block_size, Np)
            T_c = jnp.transpose(T_c, (0, 2, 1))                        # (Np, Np, block)
            return T_acc + T_c, None

        T_init = jnp.zeros((Np, Np, rank_block_size))
        T_block, _ = jax.lax.scan(process_component, T_init, jnp.arange(3))

        # Antisymmetrise the small (Np, Np, block) T before the big contraction.
        T_anti = T_block - jnp.transpose(T_block, (1, 0, 2))

        C_rs = phi_r_block[:, None, :] * phi_s_block[None, :, :]        # (Nr, Ns, block)
        contribution = jnp.einsum('pql,rsl->pqrs', T_anti, C_rs)
        return carry + contribution, None

    init = jnp.zeros((Np, Np, Nr, Ns))
    result, _ = jax.lax.scan(scan_l_block, init, (U1_scannable, phi_r_scannable, phi_s_scannable))
    return result


def contract_K1_minus_K2_isdf(phi_piv, grad_phi_piv, U1, ranges=None,
                               rank_block_size=None, gpu_max_memory_mb=None,
                               fused=False):
    """Compute (K1 - K2)[pqrs] in one pass, halving GPU peak vs separate calls.

    K2[pqrs] = K1[qprs] transposed, so the difference can be accumulated
    in a single scan over the ISDF rank dimension.
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
    grad_phi_q = grad_phi_piv[slice_q]

    if fused:
        return contract_K1_minus_K2_isdf_fused(
            phi_p, phi_q, phi_r, phi_s, grad_phi_p, grad_phi_q, U1)

    if rank_block_size is None:
        from pytc.utils.gpu_memory import adaptive_rank_block_size
        rank_block_size = adaptive_rank_block_size(
            phi_p.shape[0], phi_q.shape[0], U1.shape[0],
            gpu_max_memory_mb=gpu_max_memory_mb)

    return contract_K1_minus_K2_isdf_jit(
        phi_p, phi_q, phi_r, phi_s,
        grad_phi_p, grad_phi_q, U1, rank_block_size)


@partial(jax.jit, static_argnums=(5,))
def contract_K3_isdf_fused(phi_p, phi_q, phi_r, phi_s, U3,
                             r_block_size=128):
    """Fused K3 contraction that contracts the rank dimension FIRST.

    Same principle as ``contract_K1_minus_K2_isdf_fused``: compute the small
    T[p,q,l] tensor once (full rank), then contract against phi_r/phi_s in
    r-blocks via real GEMMs.  Eliminates the scan-loop accumulator.
    """
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_fused = U3.shape[0]
    n_rank = U3.shape[1]

    # --- Step 1: compute T_K3[p,q,l] (full rank) ---
    # T[p,q,l] = sum_k phi_p[p,k] * U3[k,l] * phi_q[q,k]
    # = sum_k phi_p[p,k] * (U3[k,l] * phi_q[q,k])
    # W[k,l,q] = U3[k,l] * phi_q[q,k]  → (N_fused, n_rank, Nq)
    W = U3[:, :, None] * phi_q.T[:, None, :]  # (N_fused, n_rank, Nq)
    W_flat = W.reshape(N_fused, n_rank * Nq)
    T_flat = jnp.matmul(phi_p, W_flat)  # (Np, n_rank*Nq)
    T_K3 = T_flat.reshape(Np, n_rank, Nq).transpose(0, 2, 1)  # (Np, Nq, n_rank)

    # --- Step 2: contract T against phi_r/phi_s in r-blocks ---
    result = jnp.zeros((Np, Nq, Nr, Ns), dtype=phi_p.dtype)

    for r0 in range(0, Nr, r_block_size):
        r1 = min(r0 + r_block_size, Nr)
        phi_r_block = phi_r[r0:r1, :]  # (R, n_rank)
        R = r1 - r0
        T_r = T_K3[:, :, None, :] * phi_r_block[None, None, :, :]  # (Np, Nq, R, n_rank)
        T_r_flat = T_r.reshape(Np * Nq * R, n_rank)
        block_result = jnp.matmul(T_r_flat, phi_s.T)  # (Np*Nq*R, Ns)
        block_result = block_result.reshape(Np, Nq, R, Ns)
        result = jax.lax.dynamic_update_slice(result, block_result, (0, 0, r0, 0))

    return result


def contract_K3_isdf_jit(phi_p, phi_q, phi_r, phi_s, U3, rank_block_size=128):
    """JITted version of K3 contraction.
    
    Args:
        rank_block_size: Block size for scanning the ISDF rank dimension.
            This is a static argument — JAX recompiles if it changes.
    """
    
    # Process K3 in chunks of l (rank index) to avoid O(N^2 * N_rank) memory usage.
    
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_fused = U3.shape[0]
    # ``n_rank`` is the l (axis-1) size, which may differ from N_fused when
    # streaming passes a panel of axis-1 columns.
    n_rank = U3.shape[1]

    # Pad rank dimension (axis 1) to a multiple of rank_block_size.
    padded_rank = ((n_rank + rank_block_size - 1) // rank_block_size) * rank_block_size
    pad_width = padded_rank - n_rank

    # Pad U3 along axis 1 (l index)
    U3_padded = jnp.pad(U3, ((0, 0), (0, pad_width)))
    
    # Pad phi_r and phi_s
    phi_r_padded = jnp.pad(phi_r, ((0, 0), (0, pad_width)))
    phi_s_padded = jnp.pad(phi_s, ((0, 0), (0, pad_width)))
    
    n_blocks = padded_rank // rank_block_size
    
    # Reshape for scan
    # U3: (N_fused, n_blocks, block) -> (n_blocks, N_fused, block)
    U3_scannable = U3_padded.reshape(N_fused, n_blocks, rank_block_size).transpose(1, 0, 2)
    
    # phi_r/s: (N, n_blocks, block) -> (n_blocks, N, block)
    phi_r_scannable = phi_r_padded.reshape(Nr, n_blocks, rank_block_size).transpose(1, 0, 2)
    phi_s_scannable = phi_s_padded.reshape(Ns, n_blocks, rank_block_size).transpose(1, 0, 2)
    
    def scan_l_block(carry, args):
        U3_block, phi_r_block, phi_s_block = args
        # U3_block: (N_fused, block)
        
        # 1. Compute T_block[p, q, l_local]
        # W[k, l', q] = U3_block[k,l'] * phi_q[q,k]
        W = U3_block[:, :, None] * phi_q.T[:, None, :] # (k, l', 1) * (k, 1, Nq) -> (k, l', Nq)
        W_flat = W.reshape(N_fused, rank_block_size * Nq)
        
        T_flat = jnp.matmul(phi_p, W_flat) # (Np, k) @ (k, l'*Nq) -> (Np, l'*Nq)
        T_block = T_flat.reshape(Np, rank_block_size, Nq) # (Np, l', Nq)
        T_block = jnp.transpose(T_block, (0, 2, 1)) # (Np, Nq, l')
        
        # 2. Form C_rs[r, s, l_local]
        C_rs = phi_r_block[:, None, :] * phi_s_block[None, :, :] # (Nr, Ns, block)
        
        # 3. Contract
        contribution = jnp.einsum('pql,rsl->pqrs', T_block, C_rs)
        
        return carry + contribution, None

    K3_init = jnp.zeros((Np, Nq, Nr, Ns))
    K3_final, _ = jax.lax.scan(scan_l_block, K3_init, (U3_scannable, phi_r_scannable, phi_s_scannable))
    
    return K3_final

def _pad_axis(arr, axis, pad):
    """Pad ``arr`` with zeros by ``pad`` along ``axis``. Works for np or jnp."""
    if pad <= 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, pad)
    if isinstance(arr, np.ndarray):
        return np.pad(arr, pad_width)
    return jnp.pad(arr, pad_width)


def _stream_l_panels(U, phi_r, phi_s, panel_size):
    """Iterate (U_panel, phi_r_panel, phi_s_panel) along axis-1 of U.

    Each yielded panel has axis-1 size exactly ``panel_size`` (last panel is
    zero-padded to keep a single JIT shape). U panels are moved to device with
    ``jax.device_put`` when U lives on host.
    """
    n_fused = U.shape[1]
    for l0 in range(0, n_fused, panel_size):
        l1 = min(l0 + panel_size, n_fused)
        pad = panel_size - (l1 - l0)

        U_slice = U[:, l0:l1, ...] if U.ndim == 3 else U[:, l0:l1]
        U_slice = _pad_axis(U_slice, 1, pad)
        if isinstance(U_slice, np.ndarray):
            U_slice = jax.device_put(U_slice)

        phi_r_slice = _pad_axis(phi_r[:, l0:l1], 1, pad)
        phi_s_slice = _pad_axis(phi_s[:, l0:l1], 1, pad)
        yield U_slice, phi_r_slice, phi_s_slice


def contract_K1_isdf_streaming(phi_p, phi_q, phi_r, phi_s,
                                grad_phi_p, U1,
                                rank_block_size=128,
                                panel_size=None):
    """Streaming-capable wrapper around :func:`contract_K1_isdf_jit` (symmetric,
    p == q case). Same panel-on-axis-1 strategy as
    :func:`contract_K1_minus_K2_isdf`."""
    n_fused = U1.shape[1]
    if panel_size is None or panel_size >= n_fused:
        if isinstance(U1, np.ndarray):
            U1 = jax.device_put(U1)
        return contract_K1_isdf_jit(phi_p, phi_q, phi_r, phi_s, grad_phi_p, U1, rank_block_size)

    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    result = jnp.zeros((Np, Nq, Nr, Ns))
    for U1_panel, phi_r_panel, phi_s_panel in _stream_l_panels(U1, phi_r, phi_s, panel_size):
        partial = contract_K1_isdf_jit(
            phi_p, phi_q, phi_r_panel, phi_s_panel, grad_phi_p, U1_panel,
            rank_block_size,
        )
        result = result + partial
    return result


def contract_K1_minus_K2_isdf_streaming(phi_p, phi_q, phi_r, phi_s,
                                         grad_phi_p, grad_phi_q, U1,
                                         rank_block_size=128,
                                         panel_size=None):
    """Streaming-capable wrapper around :func:`contract_K1_minus_K2_isdf_jit`.

    ``U1`` can be a device ``jax.Array`` or a host numpy ndarray.

    * ``panel_size=None`` or ``panel_size >= U1.shape[1]`` → delegate to the
      JIT once with the full K1; behaviour is bit-identical to the resident
      fast-path.
    * Otherwise, iterate over axis-1 (rank-column) panels of ``panel_size``;
      each panel is ``jax.device_put`` just before its call, phi_r/phi_s
      sliced to the matching slab, and partial contributions summed on
      device. Axis 0 of U1 (the k axis) is untouched.

    The last panel is zero-padded to ``panel_size`` so the JIT compiles once
    for the whole loop. Zero-padded rows/columns contribute 0 to the sum.

    Note: a function named ``contract_K1_minus_K2_isdf`` (without the
    ``_streaming`` suffix) already exists as a range-based wrapper that
    slices from a full ``phi_piv`` / ``grad_phi_piv`` — keep the names
    distinct.
    """
    n_fused = U1.shape[1]
    if panel_size is None or panel_size >= n_fused:
        if isinstance(U1, np.ndarray):
            U1 = jax.device_put(U1)
        return contract_K1_minus_K2_isdf_jit(
            phi_p, phi_q, phi_r, phi_s, grad_phi_p, grad_phi_q, U1,
            rank_block_size,
        )

    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    result = jnp.zeros((Np, Nq, Nr, Ns))
    for U1_panel, phi_r_panel, phi_s_panel in _stream_l_panels(U1, phi_r, phi_s, panel_size):
        partial = contract_K1_minus_K2_isdf_jit(
            phi_p, phi_q, phi_r_panel, phi_s_panel,
            grad_phi_p, grad_phi_q, U1_panel, rank_block_size,
        )
        result = result + partial
    return result


def contract_K1_antisym_pq_isdf_streaming(phi_p, phi_r, phi_s, grad_phi_p, U1,
                                           rank_block_size=128,
                                           panel_size=None):
    """Streaming wrapper around :func:`contract_K1_antisym_pq_isdf_jit`.

    Mirrors :func:`contract_K1_minus_K2_isdf_streaming` (axis-1 panels of
    ``U1``, host→device on demand, partial sums accumulated on device);
    use this in the ``slice_p == slice_q`` tile path where the legacy code
    materialised ``k12`` and computed ``k12 - k12.T(1, 0, 2, 3)``.
    """
    n_fused = U1.shape[1]
    if panel_size is None or panel_size >= n_fused:
        if isinstance(U1, np.ndarray):
            U1 = jax.device_put(U1)
        return contract_K1_antisym_pq_isdf_jit(
            phi_p, phi_r, phi_s, grad_phi_p, U1, rank_block_size,
        )

    Np = phi_p.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    result = jnp.zeros((Np, Np, Nr, Ns))
    for U1_panel, phi_r_panel, phi_s_panel in _stream_l_panels(U1, phi_r, phi_s, panel_size):
        partial = contract_K1_antisym_pq_isdf_jit(
            phi_p, phi_r_panel, phi_s_panel, grad_phi_p, U1_panel,
            rank_block_size,
        )
        result = result + partial
    return result


def contract_K3_isdf_streaming(phi_p, phi_q, phi_r, phi_s, U3,
                                rank_block_size=128,
                                panel_size=None):
    """Streaming-capable wrapper around :func:`contract_K3_isdf_jit`.

    Same panel-on-axis-1 strategy as :func:`contract_K1_minus_K2_isdf`, but
    U3 is 2-D ``(n_fused, n_fused)`` with no ``c`` component axis.
    """
    n_fused = U3.shape[1]
    if panel_size is None or panel_size >= n_fused:
        if isinstance(U3, np.ndarray):
            U3 = jax.device_put(U3)
        return contract_K3_isdf_jit(phi_p, phi_q, phi_r, phi_s, U3, rank_block_size)

    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    result = jnp.zeros((Np, Nq, Nr, Ns))
    for U3_panel, phi_r_panel, phi_s_panel in _stream_l_panels(U3, phi_r, phi_s, panel_size):
        partial = contract_K3_isdf_jit(
            phi_p, phi_q, phi_r_panel, phi_s_panel, U3_panel, rank_block_size,
        )
        result = result + partial
    return result


def contract_K3_isdf(phi_piv, U3, ranges=None, rank_block_size=None,
                     gpu_max_memory_mb=None, fused=False):
    """Contract K3 using ISDF decomposition.
    
    Args:
        rank_block_size: Override for the ISDF rank scan block size.
            If None, an adaptive size is computed.
        gpu_max_memory_mb: GPU memory budget for adaptive block sizing.
        fused: If True, use the fused kernel (contracts rank dimension first).
    """
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
    else:
        slice_p, slice_q, slice_r, slice_s = ranges
        
    phi_p = phi_piv[slice_p]
    phi_q = phi_piv[slice_q]
    phi_r = phi_piv[slice_r]
    phi_s = phi_piv[slice_s]
    
    if fused:
        return contract_K3_isdf_fused(phi_p, phi_q, phi_r, phi_s, U3)

    if rank_block_size is None:
        from pytc.utils.gpu_memory import adaptive_rank_block_size
        rank_block_size = adaptive_rank_block_size(
            phi_p.shape[0], phi_q.shape[0], U3.shape[0],
            gpu_max_memory_mb=gpu_max_memory_mb)
    
    return contract_K3_isdf_jit(phi_p, phi_q, phi_r, phi_s, U3, rank_block_size)
