"""JAX implementation of kinetic matrix elements."""
from functools import partial
import jax
import jax.numpy as jnp
import logging
import time

def calc_K1(phi, grad_phi, jastrow_factor, jastrow_params, grid_points, weights, ranges=None, batch_size=1000):
    r"""Calculate K1 matrix: K1_{pqrs} = \sum_i w_i \phi_p(i) \phi_q(i) \nabla_i u(i, j) \phi_r(j) \phi_s(j)
    
    Args:
        phi: Orbitals on grid (Nb, N_grid)
        grad_phi: Orbital gradients on grid (Nb, N_grid, 3)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,) for integration weights
        batch_size: Number of grid points to process at once
        ranges: Optional tuple of (slice_p, slice_q, slice_r, slice_s).
                If None, computes full (Nb^2, Nb^2) matrix.
    """
    n_orb = phi.shape[0]
    N_grid = grid_points.shape[0]
    weights = jnp.asarray(weights)
    
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
        Np = Nq = Nr = Ns = n_orb
    else:
        slice_p, slice_r, slice_q, slice_s = ranges
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
    grad_phi_r = grad_phi[slice_r]
    
    # Pad grid to multiple of batch_size
    padded_size = ((N_grid + batch_size - 1) // batch_size) * batch_size
    padded_grid = jnp.pad(grid_points, ((0, padded_size - N_grid), (0, 0)))
    
    # Reshape for scanning
    batched_grid = padded_grid.reshape(-1, batch_size, 3)
    
    # Prepare outer scan inputs (ket side: q, s)
    # We need phi_q and phi_s for the outer loop (r2 integration)
    padded_phi_q = jnp.pad(phi_q, ((0, 0), (0, padded_size - N_grid)))
    padded_phi_s = jnp.pad(phi_s, ((0, 0), (0, padded_size - N_grid)))
    padded_weights_r2 = jnp.pad(weights, ((0, padded_size - N_grid),))
    
    batched_phi_q = padded_phi_q.reshape(Nq, -1, batch_size).transpose(1, 0, 2)
    batched_phi_s = padded_phi_s.reshape(Ns, -1, batch_size).transpose(1, 0, 2)
    batched_weights_r2 = padded_weights_r2.reshape(-1, batch_size)
    
    @jax.checkpoint
    def outer_scan(carry, args):
        r2_batch, phi_q_batch, phi_s_batch, w_batch = args
        
        # Inner scan over r1 batches (bra side: p, r)
        @jax.checkpoint
        def inner_scan(inner_carry, inner_args):
            r1_batch, grad_phi_p_batch, phi_r_batch, grad_phi_r_batch, phi_p_batch, weights_batch = inner_args
            
            # grad_phi_paired_bra = grad_phi_p * phi_r
            # Shape: (Np*Nr, batch, 3)
            grad_phi_paired_batch = jnp.einsum('pnd,rn->prnd', grad_phi_p_batch, phi_r_batch).reshape(Np*Nr, -1, 3)
            
            grads = jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            term = jnp.einsum('ijc,j,jkc->ik', grad_phi_paired_batch, weights_batch, grads)
            return inner_carry + term, None

        inner_bs = batch_size
        n_inner = (N_grid + inner_bs - 1) // inner_bs
        padded_inner_size = n_inner * inner_bs
        
        padded_r1 = jnp.pad(grid_points, ((0, padded_inner_size - N_grid), (0, 0)))
        padded_grad_phi_p = jnp.pad(grad_phi_p, ((0, 0), (0, padded_inner_size - N_grid), (0, 0)))
        padded_phi_r = jnp.pad(phi_r, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_grad_phi_r = jnp.pad(grad_phi_r, ((0, 0), (0, padded_inner_size - N_grid), (0, 0)))
        padded_phi_p = jnp.pad(phi_p, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_weights = jnp.pad(weights, ((0, padded_inner_size - N_grid),))
        
        batched_r1 = padded_r1.reshape(n_inner, inner_bs, 3)
        batched_grad_phi_p = padded_grad_phi_p.reshape(Np, n_inner, inner_bs, 3).transpose(1, 0, 2, 3)
        batched_phi_r = padded_phi_r.reshape(Nr, n_inner, inner_bs).transpose(1, 0, 2)
        batched_grad_phi_r = padded_grad_phi_r.reshape(Nr, n_inner, inner_bs, 3).transpose(1, 0, 2, 3)
        batched_phi_p = padded_phi_p.reshape(Np, n_inner, inner_bs).transpose(1, 0, 2)
        batched_weights = padded_weights.reshape(n_inner, inner_bs)
        
        tmp_init = jnp.zeros((Np*Nr, batch_size))
        tmp, _ = jax.lax.scan(inner_scan, tmp_init, (batched_r1, batched_grad_phi_p, batched_phi_r, batched_grad_phi_r, batched_phi_p, batched_weights))
        
        # phi_paired_ket = phi_q * phi_s
        phi_paired_batch = jnp.einsum('qn,sn->qsn', phi_q_batch, phi_s_batch).reshape(Nq*Ns, -1)
        
        contrib = jnp.dot(phi_paired_batch * w_batch[None, :], tmp.T)
        return carry + contrib, None

    result_init = jnp.zeros((Nq*Ns, Np*Nr))
    final_result, _ = jax.lax.scan(outer_scan, result_init, (batched_grid, batched_phi_q, batched_phi_s, batched_weights_r2))
    
    return final_result.T

def calc_K3(phi, jastrow_factor, jastrow_params, grid_points, weights, ranges=None, batch_size=1000):
    r"""Calculate K3 matrix: K3_{pqrs} = \sum_i w_i \phi_p(i) \phi_q(i) (\nabla_i u(i, j))^2 \phi_r(j) \phi_s(j)
    
    Args:
        phi: Orbitals on grid (Nb, N_grid)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: Array of shape (N_grid, 3)
        weights: Array of shape (N_grid,) for integration weights
        batch_size: Number of grid points to process at once
        ranges: Optional tuple of (slice_p, slice_q, slice_r, slice_s).
                If None, computes full (Nb^2, Nb^2) matrix.
    """
    n_orb = phi.shape[0]
    N_grid = grid_points.shape[0]
    weights = jnp.asarray(weights)
    
    if ranges is None:
        slice_p = slice_q = slice_r = slice_s = slice(None)
        Np = Nq = Nr = Ns = n_orb
    else:
        slice_p, slice_r, slice_q, slice_s = ranges
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
    
    # Prepare inputs for outer scan (over r2 batches, ket side: q, s)
    padded_phi_q = jnp.pad(phi_q, ((0, 0), (0, padded_size - N_grid)))
    padded_phi_s = jnp.pad(phi_s, ((0, 0), (0, padded_size - N_grid)))
    padded_weights_r2 = jnp.pad(weights, ((0, padded_size - N_grid),))
    
    batched_phi_q = padded_phi_q.reshape(Nq, -1, batch_size).transpose(1, 0, 2)
    batched_phi_s = padded_phi_s.reshape(Ns, -1, batch_size).transpose(1, 0, 2)
    batched_weights_r2 = padded_weights_r2.reshape(-1, batch_size)
    
    @jax.checkpoint
    def outer_scan(carry, args):
        r2_batch, phi_q_batch, phi_s_batch, w_batch_r2 = args
        
        # Inner scan over r1 batches (bra side: p, r)
        @jax.checkpoint
        def inner_scan(inner_carry, inner_args):
            r1_batch, phi_p_batch, phi_r_batch, weights_batch_r1 = inner_args
            
            # Compute phi_paired for this batch of r1
            phi_paired_batch_r1 = jnp.einsum('pn,rn->prn', phi_p_batch, phi_r_batch).reshape(Np*Nr, -1)
            
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
        padded_phi_r = jnp.pad(phi_r, ((0, 0), (0, padded_inner_size - N_grid)))
        padded_weights = jnp.pad(weights, ((0, padded_inner_size - N_grid),))
        
        batched_r1 = padded_r1.reshape(n_inner, inner_bs, 3)
        batched_phi_p = padded_phi_p.reshape(Np, n_inner, inner_bs).transpose(1, 0, 2)
        batched_phi_r = padded_phi_r.reshape(Nr, n_inner, inner_bs).transpose(1, 0, 2)
        batched_weights_r1 = padded_weights.reshape(n_inner, inner_bs)
        
        tmp_init = jnp.zeros((Np*Nr, batch_size))
        tmp, _ = jax.lax.scan(inner_scan, tmp_init, (batched_r1, batched_phi_p, batched_phi_r, batched_weights_r1))
        
        # tmp is (Np*Nr, batch_size)
        # Now accumulate result: sum_k [ phi_paired(r2_k) * tmp[p, k] ]
        
        phi_paired_batch_r2 = jnp.einsum('qn,sn->qsn', phi_q_batch, phi_s_batch).reshape(Nq*Ns, -1)
        
        contrib = jnp.dot(tmp, phi_paired_batch_r2.T) # (Np*Nr, Nq*Ns)
        return carry + contrib, None

    result_init = jnp.zeros((Np*Nr, Nq*Ns))
    final_result, _ = jax.lax.scan(outer_scan, result_init, (batched_grid, batched_phi_q, batched_phi_s, batched_weights_r2))
    
    return final_result


def calc_K1_isdf(phi, xi_phi_r2, grad_phi, xi_grad_r1, jastrow_factor, jastrow_params, 
                 grid_r1, weights_r1, grid_r2, weights_r2, batch_size=1000):
    """JAX implementation of K1 term using ISDF intermediates with memory-efficient batching.
    
    Args:
        phi: (Nb, n_fused) Selected columns for phi
        xi_phi_r2: (n_fused, N_grid_r2) Interpolation coeffs for phi (r2)
        grad_phi: (Nb, n_fused, 3) Selected columns for grad_phi
        xi_grad_r1: (n_fused, N_grid_r1, 3) Interpolation coeffs for grad (r1)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_r1: (N_grid_r1, 3) Grid points for r1
        weights_r1: (N_grid_r1,) Grid weights for r1
        grid_r2: (N_grid_r2, 3) Grid points for r2
        weights_r2: (N_grid_r2,) Grid weights for r2
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb, Nb, Nb, Nb) array in chemists' notation (pr|qs)
    """
    N_grid_r2 = grid_r2.shape[0]
    Nb = phi.shape[0]
    n_fused = xi_phi_r2.shape[0]
    
    weights_r1 = jnp.asarray(weights_r1)
    weights_r2 = jnp.asarray(weights_r2)
    weighted_xi_grad_r1 = xi_grad_r1 * weights_r1[None, :, None]
    
    # Pad r2 inputs to be divisible by batch_size
    padded_size = ((N_grid_r2 + batch_size - 1) // batch_size) * batch_size
    padding = padded_size - N_grid_r2
    
    if padding > 0:
        grid_r2_padded = jnp.pad(grid_r2, ((0, padding), (0, 0)))
        weights_r2_padded = jnp.pad(weights_r2, ((0, padding),))
        xi_phi_r2_padded = jnp.pad(xi_phi_r2, ((0, 0), (0, padding)))
    else:
        grid_r2_padded = grid_r2
        weights_r2_padded = weights_r2
        xi_phi_r2_padded = xi_phi_r2
        
    # Reshape for scanning
    n_batches = padded_size // batch_size
    grid_r2_batches = grid_r2_padded.reshape(n_batches, batch_size, 3)
    weights_r2_batches = weights_r2_padded.reshape(n_batches, batch_size)
    xi_phi_r2_batches = xi_phi_r2_padded.reshape(n_fused, n_batches, batch_size).transpose(1, 0, 2)
    
    def scan_body(carry, args):
        r2_batch, w2_batch, xi_phi_batch = args
        
        # Calculate gradients: grad_r1 u(r1, r2)
        # r1 is full grid, r2 is batch
        # u_grad_batch: (N_grid_r1, batch, 3)
        
        @partial(jax.vmap, in_axes=(None, 0))
        def get_grads(r1, r2):
            return jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]
        
        u_grad_batch = jax.vmap(get_grads, in_axes=(0, None))(grid_r1, r2_batch)
        
        # Process each spatial component
        # G1_{k,b,c} = sum_g weighted_xi_grad_r1_{k,g,c} * u_grad_batch_{g,b,c}
        # (n_fused, N_grid_r1, 3) * (N_grid_r1, batch, 3) -> (n_fused, batch, 3)
        G1 = jnp.einsum('kgc,gbc->kbc', weighted_xi_grad_r1, u_grad_batch)
        
        # Factorized form: C_grad_{pq, k, c} = grad_phi_{p,k,c} phi_{q,k}
        # (n_fused, batch, 3) * (Nb, n_fused, 3) * (Nb, n_fused) -> (Nb, Nb, batch)
        G1_factorized = jnp.einsum('kmc,pkc,qk->pqm', G1, grad_phi, phi)
        
        # Contract with xi_phi_r2 and weights_r2 for this batch
        # (Nb, Nb, batch) * (n_fused, batch) * (batch,) -> (Nb, Nb, n_fused)
        G2 = jnp.einsum('pqm,lm,m->pql', G1_factorized, xi_phi_batch, w2_batch)
        
        # Accumulate result
        # Factorized form: C_phi_{rs, l} = phi_{r,l} phi_{s,l}
        # (Nb, Nb, n_fused) * (Nb, n_fused) * (Nb, n_fused) -> (Nb, Nb, Nb, Nb)
        term = jnp.einsum('pql,rl,sl->pqrs', G2, phi, phi)
        
        return carry + term, None

    result_init = jnp.zeros((Nb, Nb, Nb, Nb))
    final_result, _ = jax.lax.scan(scan_body, result_init, (grid_r2_batches, weights_r2_batches, xi_phi_r2_batches))
    
    return final_result


def calc_K2_isdf(phi, xi_phi, grad_phi, xi_grad, jastrow_factor, jastrow_params, grid_points, weights, batch_size=1000):
    """JAX implementation of K2 term using ISDF intermediates.
    
    Args:
        phi: (Nb, n_fused) Selected columns for phi
        xi_phi: (n_fused, N_grid) Interpolation coeffs for phi
        grad_phi: (Nb, n_fused, 3) Selected columns for grad_phi
        xi_grad: (n_fused, N_grid, 3) Interpolation coeffs for grad
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_points: (N_grid, 3) Grid points
        weights: (N_grid,) Grid weights
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb, Nb, Nb, Nb) array in chemists' notation (pq|rs)
    """
    # Pass grid_points and weights for both r1 and r2
    res1 = calc_K1_isdf(phi, xi_phi, grad_phi, xi_grad, jastrow_factor, jastrow_params, 
                        grid_points, weights, grid_points, weights, batch_size)
    # Transpose p and q for the second part of combined_C_grad
    res2 = jnp.einsum('pqrs->qprs', res1)
    
    return -(res1 + res2)


def calc_K3_isdf(phi, xi_phi_r1, xi_phi_r2, jastrow_factor, jastrow_params, 
                 grid_r1, weights_r1, grid_r2, weights_r2, batch_size=1000):
    """JAX implementation of K3 term using ISDF intermediates with memory-efficient batching.
    
    Args:
        phi: (Nb, n_fused) Selected columns for phi
        xi_phi_r1: (n_fused, N_grid_r1) Interpolation coeffs for phi (r1)
        xi_phi_r2: (n_fused, N_grid_r2) Interpolation coeffs for phi (r2)
        jastrow_factor: JAX Jastrow factor instance
        jastrow_params: Parameters for the Jastrow factor
        grid_r1: (N_grid_r1, 3) Grid points for r1
        weights_r1: (N_grid_r1,) Grid weights for r1
        grid_r2: (N_grid_r2, 3) Grid points for r2
        weights_r2: (N_grid_r2,) Grid weights for r2
        batch_size: Number of grid points to process at once
    
    Returns:
        (Nb, Nb, Nb, Nb) array in chemists' notation (pr|qs)
    """
    N_grid_r2 = grid_r2.shape[0]
    Nb = phi.shape[0]
    n_fused = xi_phi_r1.shape[0]
    
    weights_r1 = jnp.asarray(weights_r1)
    weights_r2 = jnp.asarray(weights_r2)
    
    # Pad r2 inputs
    padded_size = ((N_grid_r2 + batch_size - 1) // batch_size) * batch_size
    padding = padded_size - N_grid_r2
    
    if padding > 0:
        grid_r2_padded = jnp.pad(grid_r2, ((0, padding), (0, 0)))
        weights_r2_padded = jnp.pad(weights_r2, ((0, padding),))
        xi_phi_r2_padded = jnp.pad(xi_phi_r2, ((0, 0), (0, padding)))
    else:
        grid_r2_padded = grid_r2
        weights_r2_padded = weights_r2
        xi_phi_r2_padded = xi_phi_r2
        
    # Reshape for scanning
    n_batches = padded_size // batch_size
    grid_r2_batches = grid_r2_padded.reshape(n_batches, batch_size, 3)
    weights_r2_batches = weights_r2_padded.reshape(n_batches, batch_size)
    xi_phi_r2_batches = xi_phi_r2_padded.reshape(n_fused, n_batches, batch_size).transpose(1, 0, 2)
    
    def scan_body(carry, args):
        r2_batch, w2_batch, xi_phi_batch = args
        
        @partial(jax.vmap, in_axes=(None, 0))
        def get_grads(r1, r2):
            return jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]
        
        u_grad_batch = jax.vmap(get_grads, in_axes=(0, None))(grid_r1, r2_batch)  # (N_grid_r1, batch, 3)
        
        # Compute squared magnitude of gradient
        u_grad_squared = jnp.sum(u_grad_batch**2, axis=-1)  # (N_grid_r1, batch)
        
        # Weight both coordinates
        # (N_grid_r1, batch) * (N_grid_r1, 1) * (1, batch)
        weighted_u_squared = u_grad_squared * weights_r1[:, None] * w2_batch[None, :]
        
        # Contract with xi_phi for this batch
        # G1 = xi_phi_r1 @ weighted_u_squared @ xi_phi_batch.T
        # (n_fused, N_grid_r1) @ (N_grid_r1, batch) -> (n_fused, batch)
        # (n_fused, batch) @ (batch, n_fused) -> (n_fused, n_fused)
        
        # First contract r1:
        # temp = xi_phi_r1 @ weighted_u_squared  -> (n_fused, batch)
        temp = jnp.dot(xi_phi_r1, weighted_u_squared)
        
        # Then contract r2 (batch):
        # G1 = temp @ xi_phi_batch.T -> (n_fused, n_fused)
        G1 = jnp.dot(temp, xi_phi_batch.T)
        
        # Accumulate result
        # Factorized form: C_phi_{pq, k} = phi_{p,k} phi_{q,k}
        term = jnp.einsum('kl,pk,qk,rl,sl->pqrs', G1, phi, phi, phi, phi)
        
        return carry + term, None

    result_init = jnp.zeros((Nb, Nb, Nb, Nb))
    final_result, _ = jax.lax.scan(scan_body, result_init, (grid_r2_batches, weights_r2_batches, xi_phi_r2_batches))
    
    return final_result
