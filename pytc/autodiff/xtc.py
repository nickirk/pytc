"""JAX implementation of X transcorrelated methods."""

from functools import partial, reduce
import numpy as np
import logging
import time
import gc
import jax
import jax.numpy as jnp
import h5py
from flax import struct
from .tc import TC, ISDFTC
from . import tc_helper
from . import kmat as kmat_jax

@struct.dataclass
class XTC(TC):
    """JAX implementation of extended transcorrelated methods using flax dataclass.
    
    Attributes:
        mo_occ: Molecular orbital occupation numbers (N_orb,)
        energy_nuc: Nuclear repulsion energy (static)
    """
    mo_occ: jnp.ndarray = struct.field(default=None)
    energy_nuc: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def from_pyscf(cls, mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
        """Initialize XTC object from PySCF mean-field object."""
        # Create base TC object
        tc_obj = super().from_pyscf(mf, jastrow_factor, mo_coeff, grid_lvl)
        
        # Extract additional fields
        mo_occ = jnp.asarray(mf.mo_occ)
        energy_nuc = mf.energy_nuc()
        
        # Return XTC object with all fields
        return cls(
            grid_points=tc_obj.grid_points,
            weights=tc_obj.weights,
            phi=tc_obj.phi,
            grad_phi=tc_obj.grad_phi,
            n_orb=tc_obj.n_orb,
            grid_lvl=tc_obj.grid_lvl,
            jastrow_factor=tc_obj.jastrow_factor,
            mo_coeff=tc_obj.mo_coeff,
            nocc=tc_obj.nocc,
            mo_occ=mo_occ,
            energy_nuc=energy_nuc
        )
    
    @property
    def n_grid(self):
        """Number of grid points."""
        return len(self.grid_points)
    
    def _calc_v_block(self, r1_batch, phi, weights, jastrow_params, slice_rows, slice_cols, batch_size=1000):
        """Calculate V_qt(r₁) for a batch of r1 points and specific row/col slices.
        
        Args:
            r1_batch: (batch_size, 3)
            phi: (Nb, N_grid)
            weights: (N_grid,)
            jastrow_params: Jastrow parameters
            slice_rows: slice object for row indices (q)
            slice_cols: slice object for col indices (t)
            batch_size: Inner batch size for r2 scan
            
        Returns:
            V_batch: (N_rows, N_cols, batch_size, 3)
        """
        n_orb, n_grid = phi.shape
        
        # Extract relevant phi blocks
        phi_rows = phi[slice_rows]  # (N_rows, N_grid)
        phi_cols = phi[slice_cols]  # (N_cols, N_grid)
        
        n_rows = phi_rows.shape[0]
        n_cols = phi_cols.shape[0]
        
        # Pad grid for r2 scan
        padded_size = ((n_grid + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(self.grid_points, ((0, padded_size - n_grid), (0, 0)))
        padded_weights = jnp.pad(weights, (0, padded_size - n_grid))
        padded_phi_rows = jnp.pad(phi_rows, ((0, 0), (0, padded_size - n_grid)))
        padded_phi_cols = jnp.pad(phi_cols, ((0, 0), (0, padded_size - n_grid)))
        
        # Reshape for scanning
        r2_batches = padded_grid.reshape(-1, batch_size, 3)
        weights_batches = padded_weights.reshape(-1, batch_size)
        phi_rows_batches = padded_phi_rows.reshape(n_rows, -1, batch_size)
        phi_cols_batches = padded_phi_cols.reshape(n_cols, -1, batch_size)
        
        def scan_body(carry, args):
            r2_batch, w_batch, phi_row_batch, phi_col_batch = args
            
            # Compute phi_paired for this r2 batch: phi_q(r2) * phi_t(r2)
            # (N_rows, batch) * (N_cols, batch) -> (N_rows, N_cols, batch)
            phi_paired_r2 = jnp.einsum('ib,jb->ijb', phi_row_batch, phi_col_batch)
            weighted_phi_r2 = phi_paired_r2 * w_batch[None, None, :]
            
            # Compute gradients: grad_J(r1, r2) -> (batch_r1, batch_r2, 3)
            grads = self.jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            
            # Contract: sum_{r2} phi(r2) * grad(r1, r2)
            # weighted_phi_r2: (N_rows, N_cols, batch_r2)
            # grads: (batch_r1, batch_r2, 3)
            # Result: (N_rows, N_cols, batch_r1, 3)
            term = jnp.einsum('ijb,obd->ijod', weighted_phi_r2, grads)
            
            return carry + term, None

        init_val = jnp.zeros((n_rows, n_cols, len(r1_batch), 3))
        final_val, _ = jax.lax.scan(scan_body, init_val, 
                                   (r2_batches, weights_batches, 
                                    phi_rows_batches.transpose(1, 0, 2), 
                                    phi_cols_batches.transpose(1, 0, 2)))
        
        return final_val

    @partial(jax.jit, static_argnames=('batch_size',))
    def _calc_v_vector(self, phi_paired, jastrow_params, batch_size=1000):
        """Calculate V_qt(r₁) for all r₁ points.
        
        Args:
            phi_paired: (Nb^2, N_grid)
            jastrow_params: Jastrow parameters
            batch_size: Batch size for r1
            
        Returns:
            V: (Nb^2, N_grid, 3)
        """
        n_grid = self.n_grid
        Nb2 = phi_paired.shape[0]
        
        # Pad grid for r1 scan
        padded_size = ((n_grid + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(self.grid_points, ((0, padded_size - n_grid), (0, 0)))
        padded_phi_paired = jnp.pad(phi_paired, ((0, 0), (0, padded_size - n_grid)))
        padded_weights = jnp.pad(self.weights, (0, padded_size - n_grid))
        
        # Reshape for scanning
        r1_batches = padded_grid.reshape(-1, batch_size, 3)
        phi_paired_batches = padded_phi_paired.reshape(Nb2, -1, batch_size)
        weights_batches = padded_weights.reshape(-1, batch_size)
        
        def scan_body(carry, args):
            r1_batch, phi_paired_batch, w_batch = args
            
            # Use _calc_v_block with full grid
            # We need to reshape phi_paired_batch back to (Nb, Nb, batch)
            Nb = self.n_orb
            
            # V_batch: (Nb, Nb, batch_r1, 3)
            # We can use a simplified version of _calc_v_block here
            def inner_scan(carry_inner, args_inner):
                r2_batch, w2_batch, phi_paired_r2 = args_inner
                
                # grads: (batch_r1, batch_r2, 3)
                grads = self.jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
                
                # sum_j w_j phi_q(r2_j) phi_t(r2_j) grad_i u(r1_i, r2_j)
                # (Nb, Nb, batch_r2) * (batch_r1, batch_r2, 3) -> (Nb, Nb, batch_r1, 3)
                V_up = jnp.einsum('ijb,abk->ijak', phi_paired_r2 * w2_batch[None, None, :], grads)
                return carry_inner + V_up, None

            # Prepare r2 batches (full grid)
            n_grid_full = self.n_grid
            padded_size_r2 = ((n_grid_full + batch_size - 1) // batch_size) * batch_size
            padded_grid_r2 = jnp.pad(self.grid_points, ((0, padded_size_r2 - n_grid_full), (0, 0)))
            padded_weights_r2 = jnp.pad(self.weights, (0, padded_size_r2 - n_grid_full))
            padded_phi_paired_r2 = jnp.pad(phi_paired, ((0, 0), (0, padded_size_r2 - n_grid_full)))
            
            r2_batches_full = padded_grid_r2.reshape(-1, batch_size, 3)
            weights_batches_full = padded_weights_r2.reshape(-1, batch_size)
            phi_paired_batches_full = padded_phi_paired_r2.reshape(Nb, Nb, -1, batch_size).transpose(2, 0, 1, 3)
            
            V_acc, _ = jax.lax.scan(inner_scan, jnp.zeros((Nb, Nb, r1_batch.shape[0], 3)), 
                                    (r2_batches_full, weights_batches_full, phi_paired_batches_full))
            
            return carry, V_acc.reshape(Nb2, -1, 3)
        
        _, V = jax.lax.scan(scan_body, None, (r1_batches, phi_paired_batches, weights_batches))
        return V.reshape(-1, Nb2, 3).transpose(1, 0, 2)[:, :n_grid, :]

    def _calc_delta_U(self, v_vector=None, phi_paired=None, dm1=None):
        """Calculate ΔU matrix.
        
        Args:
            v_vector: (Nb^2, N_grid, 3)
            phi_paired: (Nb^2, N_grid)
            dm1: Density matrix (Nb, Nb)
            
        Returns:
            delta_U: (Nb, Nb, Nb, Nb)
        """
        Nb = self.n_orb
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Reshape v_vector to (Nb, Nb, N_grid, 3)
        V_reshaped = v_vector.reshape(Nb, Nb, -1, 3)
        
        # Reshape phi_paired to (Nb, Nb, N_grid)
        phi_paired_reshaped = phi_paired.reshape(Nb, Nb, -1)
        
        # W_k(r) = sum_{rs} V_{rs,k}(r) dm1_{rs}
        # (Nb, Nb, N_grid, 3) * (Nb, Nb) -> (N_grid, 3)
        W = jnp.einsum('rskc,rs->kc', V_reshaped, dm1)
        
        # Wbar(r) = |W(r)|^2
        Wbar = jnp.sum(W**2, axis=1) # (N_grid,)
        
        # Vbar_{qt,k}(r) = sum_{rs} V_{rs,k}(r) dm1_{rq} dm1_{st}
        # (Nb, Nb, N_grid, 3) * (Nb, Nb) * (Nb, Nb) -> (Nb, Nb, N_grid, 3)
        Vbar = jnp.einsum('rskc,rq,st->qtkc', V_reshaped, dm1, dm1)
        
        # Zbar_{qt}(r) = sum_k W_k(r) V_{qt,k}(r)
        # (N_grid, 3) * (Nb, Nb, N_grid, 3) -> (Nb, Nb, N_grid)
        Zbar = jnp.einsum('kc,qtkc->qtc', W, V_reshaped)
        
        # G_{qt,k}(r) = sum_l W_l(r) Vbar_{qt,l}(r)
        # (N_grid, 3) * (Nb, Nb, N_grid, 3) -> (Nb, Nb, N_grid)
        G = jnp.einsum('lc,qtlc->qtc', W, Vbar)
        
        # Term 1: sum_c w_c phi_p(c) phi_q(c) (Vbar_{rs}(c) - Zbar_{rs}(c))
        # A_{rs,c} = Vbar_{rs,c} - Zbar_{rs,c}
        A = Vbar - Zbar[:, :, None, :] # (Nb, Nb, N_grid, 3) - (Nb, Nb, 1, N_grid) -> (Nb, Nb, N_grid, 3)
        
        # term1 = sum_c w_c phi_p(c) phi_q(c) A_{rs,c}
        # (Nb, N_grid) * (Nb, N_grid) * (Nb, Nb, N_grid, 3) * (N_grid,) -> (Nb, Nb, Nb, Nb, 3)
        term1 = jnp.einsum('pc,qc,rscd,c->pqrsd', self.phi, self.phi, A, self.weights)
        
        # Term 2: sum_c w_c V_{pq}(c) (0.5 Wbar(c) V_{rs}(c) - G_{rs}(c))
        # B_{rs,c} = 0.5 Wbar(c) V_{rs}(c) - G_{rs}(c)
        B = 0.5 * Wbar[None, None, :, None] * V_reshaped - G[:, :, None, :] # (Nb, Nb, N_grid, 3)
        
        # term2 = sum_c w_c V_{pq}(c) B_{rs}(c)
        # (Nb, Nb, N_grid, 3) * (Nb, Nb, N_grid, 3) * (N_grid,) -> (Nb, Nb, Nb, Nb, 3)
        term2 = jnp.einsum('pqcd,rscd,c->pqrsd', V_reshaped, B, self.weights)
        
        # Sum over the gradient components (d)
        result = -(jnp.sum(term1, axis=-1) + jnp.sum(term2, axis=-1))
        
        # Symmetrize
        final = result + result.transpose(2, 3, 0, 1)
        return final

    def get_delta_U(self, jastrow_params, dm1=None, ranges=None, batch_size=1000):
        """Get delta_U matrix with memory-efficient batching and multi-GPU support.
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (must be diagonal if provided)
            ranges: Tuple of slices (p, q, r, s) for block calculation
            batch_size: Batch size for grid integration
            
        Returns:
            delta_U: The correction term.
                     If ranges provided: (Np, Nr, Nq, Ns)
                     Otherwise: (N, N, N, N)
        """
        start_time = time.perf_counter()
        logging.debug("Starting XTC.get_delta_U")
        n_devices = jax.local_device_count()
        n_grid = self.n_grid
        
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Check if dm1 is diagonal (for debugging/verification outside JIT)
        # is_diagonal = jnp.allclose(dm1, jnp.diag(jnp.diagonal(dm1)))
        # if not is_diagonal:
        #     raise ValueError("Non-diagonal density matrix for XTC calculation is not supported.")
        
        n_occ_vec = jnp.diagonal(dm1)
        
        # Pad grid to be divisible by n_devices
        remainder = n_grid % n_devices
        if remainder != 0:
            padding = n_devices - remainder
            padded_grid_points = jnp.pad(self.grid_points, ((0, padding), (0, 0)))
            padded_weights = jnp.pad(self.weights, ((0, padding),))
            padded_phi = jnp.pad(self.phi, ((0, 0), (0, padding)))
        else:
            padded_grid_points = self.grid_points
            padded_weights = self.weights
            padded_phi = self.phi
            
        n_grid_padded = padded_grid_points.shape[0]
        n_per_device = n_grid_padded // n_devices
        
        # Shard arrays for r1: (n_devices, n_per_device, ...)
        sharded_grid_r1 = padded_grid_points.reshape(n_devices, n_per_device, 3)
        sharded_weights_r1 = padded_weights.reshape(n_devices, n_per_device)
        # phi: (Nb, N) -> (Nb, n_dev, N_per) -> (n_dev, Nb, N_per)
        sharded_phi_r1 = padded_phi.reshape(self.n_orb, n_devices, n_per_device).transpose(1, 0, 2)
        
        # Define ranges
        if ranges is None:
            full_slice = slice(None)
            ranges = (full_slice, full_slice, full_slice, full_slice)
            
        slice_p, slice_q, slice_r, slice_s = ranges
        slice_occ = slice(0, self.nocc) if self.nocc is not None else slice(None)
        
        # Slice n_occ_vec to match slice_occ
        n_occ_vec_active = n_occ_vec[slice_occ]
        
        # Helper to get size
        def get_size(s, size):
            start, stop, step = s.indices(size)
            return (stop - start + (step - 1)) // step
            
        Np = get_size(slice_p, self.n_orb)
        Nq = get_size(slice_q, self.n_orb)
        Nr = get_size(slice_r, self.n_orb)
        Ns = get_size(slice_s, self.n_orb)
        Nocc = get_size(slice_occ, self.n_orb)

        def compute_on_device(grid_r1, weights_r1, phi_r1, jastrow_params):
            n_local = grid_r1.shape[0]
            local_remainder = n_local % batch_size
            if local_remainder != 0:
                local_padding = batch_size - local_remainder
                grid_r1_batched = jnp.pad(grid_r1, ((0, local_padding), (0, 0)))
                weights_r1_batched = jnp.pad(weights_r1, ((0, local_padding),))
                phi_r1_batched = jnp.pad(phi_r1, ((0, 0), (0, local_padding)))
            else:
                grid_r1_batched = grid_r1
                weights_r1_batched = weights_r1
                phi_r1_batched = phi_r1
                
            # Reshape for scanning
            r1_batches = grid_r1_batched.reshape(-1, batch_size, 3)
            weights_batches = weights_r1_batched.reshape(-1, batch_size)
            phi_batches = phi_r1_batched.reshape(self.n_orb, -1, batch_size)
            
            @jax.checkpoint
            def scan_body(carry, args):
                r1_batch, w_batch, phi_batch = args
                
                # Current batch size (might be padded)
                curr_batch_size = r1_batch.shape[0]
                
                # --- Compute V blocks ---
                # We need V blocks for:
                # (p, r), (q, s)
                # (occ, p), (occ, r), (occ, q), (occ, s)
                # (occ, occ) for W
                
                # Optimization: Compute unique blocks only
                # V is symmetric in orbital indices, so V_ij = V_ji
                # But _calc_v_block returns (rows, cols, batch, 3)
                # So V_ji = V_ij.swapaxes(0, 1)
                
                # 1. V_occ_occ (for W)
                V_occ_occ = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                              slice_occ, slice_occ, batch_size)
                
                # 2. V_pq
                V_pq = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                         slice_p, slice_q, batch_size)
                
                # 3. V_rs
                V_rs = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                         slice_r, slice_s, batch_size)
                
                # 4. V_occ_blocks
                # We need V_{k,p}, V_{k,q}, V_{k,r}, V_{k,s} where k in occ
                # We can compute V_{occ, p} etc.
                V_occ_p = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                            slice_occ, slice_p, batch_size)
                V_occ_q = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                            slice_occ, slice_q, batch_size)
                V_occ_r = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                            slice_occ, slice_r, batch_size)
                V_occ_s = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                            slice_occ, slice_s, batch_size)
                
                # --- Compute Intermediates (Diagonal dm1) ---
                
                # W = 2 * sum_k V_{kk} n_k
                # V_occ_occ: (Nocc, Nocc, batch, 3)
                # Diagonal V_{kk}: (Nocc, batch, 3)
                V_kk = jnp.einsum('ii...->i...', V_occ_occ)
                # n_occ_vec: (Nocc,)
                # W: (batch, 3)
                W = 2 * jnp.einsum('i,ibd->bd', n_occ_vec_active, V_kk)
                
                # Wbar = 2 * sum_k phi_{kk} n_k
                # phi_batch: (N_orb, batch)
                phi_occ = phi_batch[slice_occ] # (Nocc, batch)
                # phi_{kk} is just phi_occ * phi_occ? No, phi_{kk}(r) = |phi_k(r)|^2
                phi_kk = phi_occ * phi_occ
                # Wbar: (batch,)
                Wbar = 2 * jnp.einsum('i,ib->b', n_occ_vec_active, phi_kk)
                
                # --- Block (p, q) Terms ---
                # V_{pk} = V_{kp} = V_occ_p (Nocc, Np, batch, 3)
                # V_{qk} = V_{kq} = V_occ_q (Nocc, Nq, batch, 3)
                # Zbar_{pq}: (Np, Nq, batch)
                Zbar_pq = jnp.einsum('i,ipbd,iqbd->pqb', n_occ_vec_active, V_occ_p, V_occ_q)
                
                # G_{pq} = sum_k (phi_{kp} V_{qk} + phi_{kq} V_{pk}) n_k
                # phi_{kp} = phi_k * phi_p
                phi_p = phi_batch[slice_p] # (Np, batch)
                phi_q = phi_batch[slice_q] # (Nq, batch)
                # phi_{kp}: (Nocc, Np, batch)
                phi_kp = jnp.einsum('ib,pb->ipb', phi_occ, phi_p)
                phi_kq = jnp.einsum('ib,qb->iqb', phi_occ, phi_q)
                
                # G_{pq}: (Np, Nq, batch, 3)
                G_pq = jnp.einsum('i,ipb,iqbd->pqbd', n_occ_vec_active, phi_kp, V_occ_q) + \
                       jnp.einsum('i,iqb,ipbd->pqbd', n_occ_vec_active, phi_kq, V_occ_p)
                       
                # Vbar_{pq} = sum_d W_d * V_{pq,d}
                Vbar_pq = jnp.einsum('bd,pqbd->pqb', W, V_pq)
                A_pq = Vbar_pq - Zbar_pq # (Np, Nq, batch)
                
                # B_{pq} = 0.5 * Wbar * V_{pq} - G_{pq}
                # B_{pq}: (Np, Nq, batch, 3)
                B_pq = 0.5 * Wbar[None, None, :, None] * V_pq - G_pq
                
                # --- Block (r, s) Terms ---
                # Symmetric to (p, q)
                
                # Zbar_{rs}
                Zbar_rs = jnp.einsum('i,irbd,isbd->rsb', n_occ_vec_active, V_occ_r, V_occ_s)
                
                # G_{rs}
                phi_r = phi_batch[slice_r]
                phi_s = phi_batch[slice_s]
                phi_kr = jnp.einsum('ib,rb->irb', phi_occ, phi_r)
                phi_ks = jnp.einsum('ib,sb->isb', phi_occ, phi_s)
                
                G_rs = jnp.einsum('i,irb,isbd->rsbd', n_occ_vec_active, phi_kr, V_occ_s) + \
                       jnp.einsum('i,isb,irbd->rsbd', n_occ_vec_active, phi_ks, V_occ_r)
                       
                # A_{rs}
                Vbar_rs = jnp.einsum('bd,rsbd->rsb', W, V_rs)
                A_rs = Vbar_rs - Zbar_rs
                
                # B_{rs}
                B_rs = 0.5 * Wbar[None, None, :, None] * V_rs - G_rs
                
                # --- Combine Terms ---
                
                # term1 = phi_{pq} * A_{rs}
                # phi_{pq} = phi_p * phi_q
                phi_pq = jnp.einsum('pb,qb->pqb', phi_p, phi_q)
                # Weighted phi_pq for integration
                phi_pq_w = phi_pq * w_batch[None, None, :]
                
                # term1: (Np, Nq, Nr, Ns)
                # einsum: pqb, rsb -> pqrs (sum over b)
                term1 = jnp.einsum('pqb,rsb->pqrs', phi_pq_w, A_rs)
                
                # term2 = V_{pq} * B_{rs}
                # V_{pq}: (Np, Nq, batch, 3)
                # B_{rs}: (Nr, Ns, batch, 3)
                # term2: (Np, Nq, Nr, Ns)
                # We can weight V_pq
                V_pq_w = V_pq * w_batch[None, None, :, None]
                term2 = jnp.einsum('pqbd,rsbd->pqrs', V_pq_w, B_rs)
                
                # term1_sym = phi_{rs} * A_{pq}
                phi_rs = jnp.einsum('rb,sb->rsb', phi_r, phi_s)
                phi_rs_w = phi_rs * w_batch[None, None, :]
                term1_sym = jnp.einsum('rsb,pqb->pqrs', phi_rs_w, A_pq)
                
                # term2_sym = V_{rs} * B_{pq}
                V_rs_w = V_rs * w_batch[None, None, :, None]
                term2_sym = jnp.einsum('rsbd,pqbd->pqrs', V_rs_w, B_pq)
                
                # Total for this batch
                contrib = term1 + term2 + term1_sym + term2_sym
                
                return carry + contrib, None

            init_val = jnp.zeros((Np, Nq, Nr, Ns))
            
            local_delta_U, _ = jax.lax.scan(scan_body, init_val, (r1_batches, weights_batches, phi_batches.transpose(1, 0, 2)))
            
            # Sum results across devices
            total_delta_U = jax.lax.psum(local_delta_U, axis_name='devices')
            return total_delta_U

        # Execute pmap
        pmapped_compute = jax.pmap(compute_on_device, axis_name='devices', in_axes=(0, 0, 0, None))
        
        delta_U_replicated = pmapped_compute(sharded_grid_r1, sharded_weights_r1, sharded_phi_r1, jastrow_params)
        
        total_delta_U = delta_U_replicated[0]
        
        total_time = time.perf_counter() - start_time
        logging.debug(f"XTC.get_delta_U completed in {total_time:.4f} s")
        return -total_delta_U

    def get_delta_h(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        """Get or compute delta_h with memory optimization."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
            
        if ranges is None:
            # Default to full matrix if no ranges specified
            ranges = (slice(None), slice(None), slice(None), slice(None))
            
        slice_p, slice_q, slice_r, slice_s = ranges
        
        slice_occ = slice(0, self.nocc)
        
        # (pq|rs) term
        ranges_pqrs = (slice_p, slice_q, slice_occ, slice_occ)
        delta_U_pqrs = self.get_delta_U(jastrow_params, dm1, ranges=ranges_pqrs, batch_size=batch_size)
        # delta_U_pqrs shape: (Np, Nq, Nocc, Nocc)
        
        dm1_diag = jnp.diagonal(dm1)[slice_occ] # (Nocc,)
        
        # (pq|rs) * dm_rs -> (pq|oo) * dm_oo -> (pq)
        term1 = 2 * jnp.einsum('pqoo,o->pq', delta_U_pqrs, dm1_diag)
        
        # (ps|rq) term
        ranges_psrq = (slice_p, slice_occ, slice_occ, slice_q)
        delta_U_psrq = self.get_delta_U(jastrow_params, dm1, ranges=ranges_psrq, batch_size=batch_size)
        # delta_U_psrq shape: (Np, Nocc, Nocc, Nq)
        
        # (ps|rq) * dm_sr -> (po|oq) * dm_oo -> (pq)
        term2 = jnp.einsum('pooq,o->pq', delta_U_psrq, dm1_diag)
        
        delta_h = -0.5 * (term1 - term2)
        return delta_h

    @partial(jax.jit, static_argnames=('block_str', 'ranges', 'batch_size'))
    def get_1b(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        """Get one-body operator correction."""
        return self.get_delta_h(jastrow_params, dm1, block_str, ranges, batch_size)

    @partial(jax.jit, static_argnames=('block_str', 'ranges', 'batch_size'))
    def get_2b(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        """Compute two-body integrals correction."""
        start_time = time.perf_counter()
        logging.debug("Starting XTC.get_2b")
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
        
        tc_correction = super().get_2b(jastrow_params, ranges=ranges)
        
        delta_U = self.get_delta_U(jastrow_params, dm1, ranges=ranges, batch_size=batch_size)
        
        total_time = time.perf_counter() - start_time
        logging.debug(f"XTC.get_2b completed in {total_time:.4f} s")
        return tc_correction + delta_U

    @jax.jit
    def get_const(self, jastrow_params, dm1=None):
        """Compute constant contribution."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        
        delta_h = self.get_delta_h(jastrow_params, dm1)
        const = -2/3 * jnp.einsum('qp,pq->', delta_h, dm1)
        const += self.energy_nuc
        return const
    
    @jax.jit
    def _calc_delta_h(self, delta_U, dm1=None):
        """Calculate δh using δU and density matrix."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        term1 = 2*jnp.einsum('qpsr,rs->qp', delta_U, dm1)
        term2 = jnp.einsum('spqr,rs->qp', delta_U, dm1)
        delta_h = -0.5 * (term1 - term2)
        return delta_h

    def _get_mf_dm(self):
        """Get mean-field 1-body density matrix for closed shell system."""
        dm1 = jnp.diag(self.mo_occ)/2
        return dm1

    def get_3b(self):
        """Get three-body extended correlation."""
        raise NotImplementedError("JAX implementation pending")
        
    def make_eris(self, mf, jastrow_params):
        """Create ChemistsERIs object for CCSD calculation.
        
        Args:
            mf: PySCF mean-field object (required for initializing RCCSD)
            jastrow_params: Parameters for the Jastrow factor
        """
        from pyscf.cc import rccsd
        mycc = rccsd.RCCSD(mf)
        nocc = np.sum(mf.mo_occ > 0)
        
        eris = rccsd._ChemistsERIs(mycc)
        
        # Get standard integrals from helper
        eri_std = tc_helper.get_eri(mf, self.mo_coeff)
        h1e_std = tc_helper.get_hcore(mf, self.mo_coeff)
        
        # Get corrections
        # Force concrete value computation
        const = np.asarray(self.get_const(jastrow_params))
        h1e_corr = np.asarray(self.get_1b(jastrow_params))
        h2e_corr = np.asarray(self.get_2b(jastrow_params))
        
        # Combine
        h1e = h1e_std + h1e_corr
        h2e = eri_std + h2e_corr
        
        # Now use the concrete NumPy arrays
        eris.e_core = np.float64(const)
        eris.fock = h1e.copy()
        
        # Modify fock matrix
        fock_modification = (2 * np.einsum('pqii->pq', h2e[:,:,:nocc,:nocc]) - 
                           np.einsum('piiq->pq', h2e[:,:nocc,:nocc,:]))
        eris.fock += fock_modification
        eris.mo_energy = np.diag(eris.fock).copy()
        
        # Store ERI blocks
        eris.oooo = h2e[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = h2e[:nocc,nocc:,:nocc,:nocc].copy()
        eris.ooov = h2e[:nocc,:nocc,:nocc,nocc:].copy()
        eris.vooo = h2e[nocc:,:nocc,:nocc,:nocc].copy()
        eris.ovov = h2e[:nocc,nocc:,:nocc,nocc:].copy()
        eris.vovo = h2e[nocc:,:nocc,nocc:,:nocc].copy()
        eris.ovvo = h2e[:nocc,nocc:,nocc:,:nocc].copy()
        eris.voov = h2e[nocc:,:nocc,:nocc,nocc:].copy()
        eris.oovv = h2e[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvv = h2e[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vovv = h2e[nocc:,:nocc,nocc:,nocc:].copy()
        eris.vvov = h2e[nocc:,nocc:,:nocc,nocc:].copy()
        eris.vvvv = h2e[nocc:,nocc:,nocc:,nocc:].copy()

        return eris


@struct.dataclass
class ISDFXTC(XTC, ISDFTC):
    """JAX implementation of extended transcorrelated methods using ISDF.
    
    Attributes:
        xi_phi: ISDF coefficients for density (N_fused, N_grid)
        xi_grad: ISDF coefficients for gradients (N_fused, N_grid, 3)
        pivots: ISDF pivot indices (N_fused,)
        phi_isdf: ISDF basis for density (Nb, N_fused)
        grad_phi_isdf: ISDF basis for gradients (Nb, N_fused, 3)
    """
    # Fields are inherited from ISDFTC

    @classmethod
    def from_xtc(cls, xtc_obj, n_rank=None, is_incore=False, save_path=None):
        """Initialize ISDFXTC object from XTC object."""
        from . import df
        
        if n_rank is None:
            n_rank = xtc_obj.grid_points.shape[0] // 4
            
        # Perform ISDF decomposition
        phi_isdf, xi_phi, grad_phi_isdf, xi_grad, pivots, actual_save_path = df.isdf_decompose(
            xtc_obj.phi, xtc_obj.grad_phi, n_rank, n_rank, weights=xtc_obj.weights,
            is_incore=is_incore, save_path=save_path
        )
        
        return cls(
            grid_points=xtc_obj.grid_points,
            weights=xtc_obj.weights,
            phi=xtc_obj.phi,
            grad_phi=xtc_obj.grad_phi,
            n_orb=xtc_obj.n_orb,
            grid_lvl=xtc_obj.grid_lvl,
            jastrow_factor=xtc_obj.jastrow_factor,
            mo_coeff=xtc_obj.mo_coeff,
            mo_occ=xtc_obj.mo_occ,
            energy_nuc=xtc_obj.energy_nuc,
            xi_phi=xi_phi,
            xi_grad=xi_grad,
            pivots=pivots,
            phi_isdf=phi_isdf,
            grad_phi_isdf=grad_phi_isdf,
            isdf_kernels=None,
            is_incore=is_incore,
            save_path=actual_save_path
        )

    def isdf(self, jastrow_params, save_path=None, batch_size=1000):
        """Compute ISDF intermediates and store them.
        
        Computes K1_kernel, K3_kernel, L_aux, D1, D4, X2, X3 kernels and stores them in self.isdf_kernels.
        Also stores pivot values of phi and grad_phi.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor.
            save_path: Optional path to save intermediates to HDF5.
            batch_size: Batch size for computation.
        """
        # 1. Compute K1_kernel, K3_kernel, L_aux (via ISDFTC)
        # This returns a new ISDFTC object with kernels
        isdf_tc = super().isdf(jastrow_params, save_path=self.save_path, batch_size=batch_size)
        kernels = dict(isdf_tc.isdf_kernels)
        
        logging.info("Computing ISDF intermediates (Delta U)...")
        start_time = time.perf_counter()
        
        # 2. Compute Delta U kernels (D, X)
        # Pass L_aux to avoid redundant calculation
        delta_u_kernels = self.compute_delta_u_kernels(jastrow_params, batch_size, L_aux=kernels.get('L_aux'))
        kernels.update(delta_u_kernels)
        
        # 3. Discard L_aux from ISDFXTC kernels to save RAM (it's still in ISDFTC if needed)
        if 'L_aux' in kernels:
            del kernels['L_aux']
        
        if save_path:
            import h5py
            with h5py.File(save_path, 'w') as f:
                for k, v in kernels.items():
                    f.create_dataset(k, data=np.array(v))
                f.create_dataset('phi_piv', data=np.array(self.phi_isdf))
                f.create_dataset('grad_phi_piv', data=np.array(self.grad_phi_isdf))
                f.create_dataset('pivots', data=np.array(self.pivots))
                
        logging.info(f"ISDF intermediates (Delta U) computed in {time.perf_counter() - start_time:.4f} s")
        
        return self.replace(isdf_kernels=kernels)

    def compute_delta_u_kernels(self, jastrow_params, batch_size=1000, L_aux=None):
        """Compute D, X kernels for Delta U."""
        full_slice = slice(None)
        ranges = (full_slice, full_slice, full_slice, full_slice)
        return self._compute_delta_u_kernels_raw(jastrow_params, ranges, batch_size, L_aux=L_aux)



    def _compute_delta_u_kernels_raw(self, jastrow_params, ranges, batch_size=1024, L_aux=None):
        """Compute raw kernels for Delta U."""
        # 1. Compute L_aux (G) if not provided
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)
        
        # Ensure L_aux is on CPU
        cpu_device = jax.devices("cpu")[0]
        L_aux = jax.device_put(L_aux, cpu_device)
        
        n_devices = jax.local_device_count()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        dm1 = self._get_mf_dm() # Use MF density
        
        # Pad grid
        remainder = n_grid % n_devices
        padding = (n_devices - remainder) if remainder != 0 else 0
        n_grid_padded = n_grid + padding
        n_per_device = n_grid_padded // n_devices
        devices = jax.local_devices()
        
        sharded_grid = jnp.pad(self.grid_points, ((0, padding), (0, 0))).reshape(n_devices, n_per_device, 3)
        sharded_weights = jnp.pad(self.weights, ((0, padding),)).reshape(n_devices, n_per_device)
        
        # Shard G (-L_aux) for r1 directly to avoid large copies
        sharded_G_list = []
        for d in range(n_devices):
            start = d * n_per_device
            end = min((d + 1) * n_per_device, n_grid)
            actual_len = end - start
            # G = -L_aux
            G_d = -L_aux[:, start:end, :]
            if actual_len < n_per_device:
                G_d = jnp.pad(G_d, ((0, 0), (0, n_per_device - actual_len), (0, 0)))
            sharded_G_list.append(jax.device_put(G_d, devices[d]))
            del G_d
        sharded_G = jax.device_put_sharded(sharded_G_list, devices)
        del sharded_G_list
        gc.collect()
        
        if self.xi_phi is not None:
            padded_xi_phi = jnp.pad(self.xi_phi, ((0, 0), (0, padding)))
            sharded_xi_phi_list = [padded_xi_phi[:, d*n_per_device:(d+1)*n_per_device] for d in range(n_devices)]
            sharded_xi_phi = jax.device_put_sharded(sharded_xi_phi_list, devices)
            full_xi_phi = jax.device_put(self.xi_phi, devices[0])
            del padded_xi_phi, sharded_xi_phi_list
            gc.collect()
        else:
            logging.info(f"  _compute_delta_u_kernels_raw: Streaming xi_phi from {self.save_path}")
            sharded_xi_phi_list = []
            with h5py.File(self.save_path, 'r') as f:
                xi_phi_ds = f['xi_phi']
                for d in range(n_devices):
                    start = d * n_per_device
                    end = min((d + 1) * n_per_device, n_grid)
                    actual_len = end - start
                    xi_phi_d = xi_phi_ds[:, start:end]
                    if actual_len < n_per_device:
                        xi_phi_d = np.pad(xi_phi_d, ((0, 0), (0, n_per_device - actual_len)))
                    sharded_xi_phi_list.append(jax.device_put(xi_phi_d, devices[d]))
                    del xi_phi_d
                full_xi_phi = jax.device_put(xi_phi_ds[:], devices[0])
            sharded_xi_phi = jax.device_put_sharded(sharded_xi_phi_list, devices)
            del sharded_xi_phi_list
            gc.collect()
            
        Gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        
        full_grid = jax.device_put(np.asarray(self.grid_points), devices[0])
        full_weights = jax.device_put(np.asarray(self.weights), devices[0])
        # full_xi_phi is already device-resident
        gc.collect()


        phi_isdf = self.phi_isdf
        n_orb = self.n_orb
        
        # Debug output for array sizes
        def get_gb(shape):
            return np.prod(shape) * 8 / 1024**3
            
        logging.info(f"Delta U Kernel Sizes (GB):")
        logging.info(f"  phi_isdf: {get_gb(phi_isdf.shape):.3f} GB {phi_isdf.shape}")
        if self.xi_phi is not None:
            logging.info(f"  xi_phi:   {get_gb(self.xi_phi.shape):.3f} GB {self.xi_phi.shape}")
        else:
            logging.info(f"  xi_phi:   None (Out-of-core)")
        logging.info(f"  dm1:      {get_gb(dm1.shape):.3f} GB {dm1.shape}")
        
        # Estimated sizes in _calc_delta_U_kernels_shard
        n_rank = phi_isdf.shape[1]
        logging.info(f"Estimated GPU intermediates (per device):")
        logging.info(f"  c_phi_ru: {get_gb((n_orb, n_orb, n_rank)):.3f} GB")
        logging.info(f"  L:        {get_gb((n_orb, n_orb, n_rank)):.3f} GB")
        logging.info(f"  Y/Z/M/N:  {get_gb((n_orb, n_orb, batch_size, 3)):.3f} GB each")
        logging.info(f"  YZ/L_tilde/C_tilde: {get_gb((n_orb, n_orb, n_rank)):.3f} GB each")

        dm1_diag = jnp.diagonal(dm1)
        phi_dm = phi_isdf * dm1_diag[:, None]
        Q = jnp.dot(phi_dm.T, phi_isdf) # (N_rank, N_rank)
        
        calc_D_fn = self._calc_D_shard
        calc_X2_fn = self._calc_X2_shard
        calc_X3_1_fn = self._calc_X3_1_shard
        calc_X3_2_fn = self._calc_X3_2_shard
        
        # Convert remaining inputs to NumPy for pmap
        sharded_grid = np.asarray(sharded_grid)
        sharded_weights = np.asarray(sharded_weights)
        
        # Log estimated peak memory for the new loop-based approach
        # Peak intermediate per batch: 2 * (batch * Nr * N_rank) + (Nr * Ns * N_rank)
        # For 512, 800, 5000: 2 * 7.6 GB + 12 GB = 27.2 GB.
        # This is much better than the 93 GB before.
        logging.info(f"  Estimated peak intermediate per batch: {2 * get_gb((batch_size, n_orb, n_rank)) + get_gb((n_orb, n_orb, n_rank)):.3f} GB")
        
        gc.collect()

        # Stage 1: D Term
        def compute_D_on_device(grid_shard, weights_shard, xi_shard, G_shard, jastrow_params, Gb, dm1, phi_isdf, n_orb):
            return calc_D_fn(
                jastrow_params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, phi_isdf, ranges, n_orb, batch_size
            )
        pmapped_D = jax.pmap(compute_D_on_device, axis_name='devices', in_axes=(0, 0, 0, 0, None, None, None, None, None))
        D_rep = pmapped_D(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, jastrow_params, Gb, dm1, phi_isdf, n_orb)
        D = jnp.sum(D_rep, axis=0)
        del D_rep
        gc.collect()

        # Stage 2: X2 Term
        def compute_X2_on_device(grid_shard, weights_shard, xi_shard, G_shard, jastrow_params, Gb, dm1, phi_isdf, n_orb, Q):
            return calc_X2_fn(
                jastrow_params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, phi_isdf, ranges, n_orb, batch_size, Q
            )
        pmapped_X2 = jax.pmap(compute_X2_on_device, axis_name='devices', in_axes=(0, 0, 0, 0, None, None, None, None, None, None))
        X2_rep = pmapped_X2(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, jastrow_params, Gb, dm1, phi_isdf, n_orb, Q)
        X2 = jnp.sum(X2_rep, axis=0)
        del X2_rep
        gc.collect()

        # Stage 3: X3_1 Term
        def compute_X3_1_on_device(grid_shard, weights_shard, xi_shard, G_shard, jastrow_params, Gb, dm1, phi_isdf, n_orb, Q):
            return calc_X3_1_fn(
                jastrow_params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, phi_isdf, ranges, n_orb, batch_size, Q
            )
        pmapped_X3_1 = jax.pmap(compute_X3_1_on_device, axis_name='devices', in_axes=(0, 0, 0, 0, None, None, None, None, None, None))
        X3_1_rep = pmapped_X3_1(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, jastrow_params, Gb, dm1, phi_isdf, n_orb, Q)
        X3_1 = jnp.sum(X3_1_rep, axis=0)
        del X3_1_rep
        gc.collect()

        # Stage 4: X3_2 Term
        def compute_X3_2_on_device(grid_shard, weights_shard, xi_shard, G_shard, jastrow_params, Gb, dm1, phi_isdf, n_orb, Q):
            return calc_X3_2_fn(
                jastrow_params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, phi_isdf, ranges, n_orb, batch_size, Q
            )
        pmapped_X3_2 = jax.pmap(compute_X3_2_on_device, axis_name='devices', in_axes=(0, 0, 0, 0, None, None, None, None, None, None))
        X3_2_rep = pmapped_X3_2(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, jastrow_params, Gb, dm1, phi_isdf, n_orb, Q)
        X3_2 = jnp.sum(X3_2_rep, axis=0)
        del X3_2_rep
        gc.collect()

        X = X2 + X3_1 + X3_2
        
        return {'D': D, 'X': X}

    def _calc_D_shard(self, jastrow_params, dm1, grid_points, weights, xi_phi, G_shard,
                      Gb, phi, ranges, n_orb, batch_size=1024):
        """Calculate D kernel for a shard."""
        N_rank = phi.shape[1]
        N_shard = grid_points.shape[0]
        
        # Pad grid for scanning
        padded_size = ((N_shard + batch_size - 1) // batch_size) * batch_size
        weights_padded = jnp.pad(weights, (0, padded_size - N_shard))
        xi_padded = jnp.pad(xi_phi, ((0, 0), (0, padded_size - N_shard)))
        G_padded = jnp.pad(G_shard, ((0, 0), (0, padded_size - N_shard), (0, 0)))
        
        n_batches = padded_size // batch_size

        def scan_D(D_acc, i_batch):
            w_batch = jax.lax.dynamic_slice(weights_padded, (i_batch * batch_size,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(xi_padded, (0, i_batch * batch_size), (N_rank, batch_size))
            G_batch = jax.lax.dynamic_slice(G_padded, (0, i_batch * batch_size, 0), (N_rank, batch_size, 3))
            
            G_flat = G_batch.reshape(N_rank, -1)
            
            # D1 part
            H = jnp.einsum('b,bik->ik', Gb, G_batch)
            V = jnp.einsum('ik,dik->di', H, G_batch)
            D1_update = jnp.einsum('i,ai,di->ad', w_batch, xi_batch, V)
            
            # D4 part
            w_tilde = w_batch * jnp.einsum('b,bi->i', Gb, xi_batch)
            G_weighted = G_batch * w_tilde[None, :, None]
            G_weighted_flat = G_weighted.reshape(N_rank, -1)
            D4_update = jnp.dot(G_weighted_flat, G_flat.T)
            
            return D_acc + 2 * D1_update + D4_update, None
        
        D_final, _ = jax.lax.scan(scan_D, jnp.zeros((N_rank, N_rank)), jnp.arange(n_batches))
        return D_final

    def _calc_X2_shard(self, jastrow_params, dm1, grid_points, weights, xi_phi, G_shard,
                       Gb, phi, ranges, n_orb, batch_size=1024, Q=None):
        """Calculate X2 kernel for a shard using Q-matrix optimization."""
        N_rank = phi.shape[1]
        N_shard = grid_points.shape[0]
        
        slice_p, slice_q, slice_r, slice_s = ranges
        phi_r = phi[slice_r]
        phi_s = phi[slice_s]
        Nr = phi_r.shape[0]
        Ns = phi_s.shape[0]
        
        # Pad grid for scanning
        padded_size = ((N_shard + batch_size - 1) // batch_size) * batch_size
        weights_padded = jnp.pad(weights, (0, padded_size - N_shard))
        xi_padded = jnp.pad(xi_phi, ((0, 0), (0, padded_size - N_shard)))
        G_padded = jnp.pad(G_shard, ((0, 0), (0, padded_size - N_shard), (0, 0)))
        
        n_batches = padded_size // batch_size

        def scan_X2(X_acc, i_batch):
            w_batch = jax.lax.dynamic_slice(weights_padded, (i_batch * batch_size,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(xi_padded, (0, i_batch * batch_size), (N_rank, batch_size))
            G_batch = jax.lax.dynamic_slice(G_padded, (0, i_batch * batch_size, 0), (N_rank, batch_size, 3))
            
            YZ = 0
            for k in range(3):
                G_k = G_batch[:, :, k] # (N_rank, batch)
                
                @jax.checkpoint
                def compute_YZ_ki(G_ki):
                    # G_ki: (N_rank,)
                    phi_r_G = phi_r * G_ki[None, :]
                    phi_s_G = phi_s * G_ki[None, :]
                    return phi_r_G @ Q @ phi_s_G.T # (Nr, Ns)
                
                YZ += jax.vmap(compute_YZ_ki, in_axes=0)(G_k.T) # (batch, Nr, Ns)
            # YZ is (batch, Nr, Ns). We need (Nr, Ns, batch) for X2_update.
            YZ = YZ.transpose(1, 2, 0) # (Nr, Ns, batch)
            
            # Use matmul for better memory efficiency: (Nr, Ns, batch) @ (batch, N_rank)
            X2_update = jnp.matmul(YZ * w_batch[None, None, :], xi_batch.T)
            return X_acc + X2_update, None
        
        X2_final, _ = jax.lax.scan(scan_X2, jnp.zeros((Nr, Ns, N_rank)), jnp.arange(n_batches))
        return X2_final

    def _calc_X3_1_shard(self, jastrow_params, dm1, grid_points, weights, xi_phi, G_shard,
                         Gb, phi, ranges, n_orb, batch_size=1024, Q=None):
        """Calculate X3_1 kernel for a shard using Q-matrix optimization."""
        N_rank = phi.shape[1]
        N_shard = grid_points.shape[0]
        
        slice_p, slice_q, slice_r, slice_s = ranges
        phi_r = phi[slice_r]
        phi_s = phi[slice_s]
        Nr = phi_r.shape[0]
        Ns = phi_s.shape[0]
        
        # Pad grid for scanning
        padded_size = ((N_shard + batch_size - 1) // batch_size) * batch_size
        weights_padded = jnp.pad(weights, (0, padded_size - N_shard))
        xi_padded = jnp.pad(xi_phi, ((0, 0), (0, padded_size - N_shard)))
        G_padded = jnp.pad(G_shard, ((0, 0), (0, padded_size - N_shard), (0, 0)))
        
        n_batches = padded_size // batch_size

        def scan_X3_1(X_acc, i_batch):
            w_batch = jax.lax.dynamic_slice(weights_padded, (i_batch * batch_size,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(xi_padded, (0, i_batch * batch_size), (N_rank, batch_size))
            G_batch = jax.lax.dynamic_slice(G_padded, (0, i_batch * batch_size, 0), (N_rank, batch_size, 3))
            
            X3_1_update = 0
            for k in range(3):
                G_k = G_batch[:, :, k] # (N_rank, batch)
                
                @jax.checkpoint
                def compute_M_ki(G_ki, xi_i):
                    # G_ki: (N_rank,), xi_i: (N_rank,)
                    phi_r_G = phi_r * G_ki[None, :]
                    phi_s_xi = phi_s * xi_i[None, :]
                    return phi_r_G @ Q @ phi_s_xi.T # (Nr, Ns)
                
                M_k = jax.vmap(compute_M_ki, in_axes=(0, 0))(G_k.T, xi_batch.T) # (batch, Nr, Ns)
                X3_1_update += jnp.matmul(M_k.transpose(1, 2, 0) * w_batch[None, None, :], G_k.T)
            
            return X_acc + X3_1_update, None
        
        X3_1_final, _ = jax.lax.scan(scan_X3_1, jnp.zeros((Nr, Ns, N_rank)), jnp.arange(n_batches))
        return X3_1_final

    def _calc_X3_2_shard(self, jastrow_params, dm1, grid_points, weights, xi_phi, G_shard,
                         Gb, phi, ranges, n_orb, batch_size=1024, Q=None):
        """Calculate X3_2 kernel for a shard using Q-matrix optimization."""
        N_rank = phi.shape[1]
        N_shard = grid_points.shape[0]
        
        slice_p, slice_q, slice_r, slice_s = ranges
        phi_r = phi[slice_r]
        phi_s = phi[slice_s]
        Nr = phi_r.shape[0]
        Ns = phi_s.shape[0]
        
        # Pad grid for scanning
        padded_size = ((N_shard + batch_size - 1) // batch_size) * batch_size
        weights_padded = jnp.pad(weights, (0, padded_size - N_shard))
        xi_padded = jnp.pad(xi_phi, ((0, 0), (0, padded_size - N_shard)))
        G_padded = jnp.pad(G_shard, ((0, 0), (0, padded_size - N_shard), (0, 0)))
        
        n_batches = padded_size // batch_size

        def scan_X3_2(X_acc, i_batch):
            w_batch = jax.lax.dynamic_slice(weights_padded, (i_batch * batch_size,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(xi_padded, (0, i_batch * batch_size), (N_rank, batch_size))
            G_batch = jax.lax.dynamic_slice(G_padded, (0, i_batch * batch_size, 0), (N_rank, batch_size, 3))
            
            X3_2_update = 0
            for k in range(3):
                G_k = G_batch[:, :, k] # (N_rank, batch)
                
                @jax.checkpoint
                def compute_N_ki(G_ki, xi_i):
                    # G_ki: (N_rank,), xi_i: (N_rank,)
                    phi_r_xi = phi_r * xi_i[None, :]
                    phi_s_G = phi_s * G_ki[None, :]
                    return phi_r_xi @ Q @ phi_s_G.T # (Nr, Ns)
                
                N_k = jax.vmap(compute_N_ki, in_axes=(0, 0))(G_k.T, xi_batch.T) # (batch, Nr, Ns)
                X3_2_update += jnp.matmul(N_k.transpose(1, 2, 0) * w_batch[None, None, :], G_k.T)
            
            return X_acc + X3_2_update, None
        
        X3_2_final, _ = jax.lax.scan(scan_X3_2, jnp.zeros((Nr, Ns, N_rank)), jnp.arange(n_batches))
        return X3_2_final



    def get_delta_U(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        """Get delta_U matrix using ISDF with pmap support."""
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
            
        if ranges is None:
            full_slice = slice(None)
            ranges = (full_slice, full_slice, full_slice, full_slice)
            
        start_time = time.perf_counter()
        logging.info("Starting ISDFXTC.get_delta_U")
        
        # Check if kernels are available
        if self.isdf_kernels is None:
             # Compute kernels on the fly if not available
             # This aligns with kmat functions logic
             kernels = self.compute_delta_u_kernels(jastrow_params, batch_size)
        else:
            kernels = self.isdf_kernels
            
        result = self._contract_delta_U_kernels(kernels, ranges)
        
        # Symmetrize the result
        slice_p, slice_q, slice_r, slice_s = ranges
        
        if slice_p == slice_r and slice_q == slice_s:
            final_result = -(result + result.transpose(2, 3, 0, 1))
        else:
            # Non-symmetric block
            # We need the transpose block (rs|pq)
            ranges_T = (slice_r, slice_s, slice_p, slice_q)
            
            # Reuse kernels for transpose block
            result_T = self._contract_delta_U_kernels(kernels, ranges_T)
                
            final_result = -(result + result_T.transpose(2, 3, 0, 1))

        total_time = time.perf_counter() - start_time
        logging.info(f"ISDFXTC.get_delta_U completed in {total_time:.4f} s")
        return final_result

    @staticmethod
    @jax.jit
    def _contract_delta_U_kernels_jit(D, X_sliced, phi_p, phi_q, phi_r, phi_s):
        """JITted version of Delta U contraction."""
        # C_phi_{pq, a} = phi_{p,a} phi_{q,a}
        c_phi_pq = jnp.einsum('pa,qa->pqa', phi_p, phi_q)
        c_phi_rs = jnp.einsum('ra,sa->rsa', phi_r, phi_s)
        
        # Term 1 & 4: sum_{a,d} c_phi_pq[a] * D[a,d] * c_phi_rs[d]
        # Break down to avoid O(N_orb^2 * N_rank^2) intermediate
        tmp = jnp.einsum('pqa,ad->pqd', c_phi_pq, D)
        term_d = jnp.einsum('pqd,rsd->pqrs', tmp, c_phi_rs)
        
        # Term 2 & 3: - sum_a c_phi_pq[a] * X[r,s,a]
        term_x = -jnp.einsum('pqa,rsa->pqrs', c_phi_pq, X_sliced)
        
        return term_d + term_x

    def _contract_delta_U_kernels(self, kernels, ranges):
        """Contract precomputed kernels to get Delta U block."""
        D = kernels['D']
        X = kernels['X']
        
        slice_p, slice_q, slice_r, slice_s = ranges
        
        phi_p = self.phi_isdf[slice_p]
        phi_q = self.phi_isdf[slice_q]
        phi_r = self.phi_isdf[slice_r]
        phi_s = self.phi_isdf[slice_s]
        
        # X is (N_orb, N_orb, N_rank)
        # We need to slice it for r, s
        X_sliced = X[slice_r, slice_s]
        
        return self._contract_delta_U_kernels_jit(D, X_sliced, phi_p, phi_q, phi_r, phi_s)
    
