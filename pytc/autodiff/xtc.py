"""JAX implementation of X transcorrelated methods."""

from functools import partial, reduce
from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from .tc import TC
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
            rho=tc_obj.rho,
            nabla_rho=tc_obj.nabla_rho,
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
    
    def _calc_v_block(self, r1_batch, rho, weights, jastrow_params, slice_rows, slice_cols, batch_size=1000):
        """Calculate V_qt(r₁) for a batch of r1 points and specific row/col slices.
        
        Args:
            r1_batch: (batch_size, 3)
            rho: (Nb, N_grid)
            weights: (N_grid,)
            jastrow_params: Jastrow parameters
            slice_rows: slice object for row indices (q)
            slice_cols: slice object for col indices (t)
            batch_size: Inner batch size for r2 scan
            
        Returns:
            V_batch: (N_rows, N_cols, batch_size, 3)
        """
        n_orb, n_grid = rho.shape
        
        # Extract relevant rho blocks
        rho_rows = rho[slice_rows]  # (N_rows, N_grid)
        rho_cols = rho[slice_cols]  # (N_cols, N_grid)
        
        n_rows = rho_rows.shape[0]
        n_cols = rho_cols.shape[0]
        
        # Pad grid for r2 scan
        padded_size = ((n_grid + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(self.grid_points, ((0, padded_size - n_grid), (0, 0)))
        padded_weights = jnp.pad(weights, (0, padded_size - n_grid))
        padded_rho_rows = jnp.pad(rho_rows, ((0, 0), (0, padded_size - n_grid)))
        padded_rho_cols = jnp.pad(rho_cols, ((0, 0), (0, padded_size - n_grid)))
        
        # Reshape for scanning
        r2_batches = padded_grid.reshape(-1, batch_size, 3)
        weights_batches = padded_weights.reshape(-1, batch_size)
        rho_rows_batches = padded_rho_rows.reshape(n_rows, -1, batch_size)
        rho_cols_batches = padded_rho_cols.reshape(n_cols, -1, batch_size)
        
        def scan_body(carry, args):
            r2_batch, w_batch, rho_row_batch, rho_col_batch = args
            
            # Compute rho_paired for this r2 batch: rho_q(r2) * rho_t(r2)
            # (N_rows, batch) * (N_cols, batch) -> (N_rows, N_cols, batch)
            rho_paired_r2 = jnp.einsum('ib,jb->ijb', rho_row_batch, rho_col_batch)
            weighted_rho_r2 = rho_paired_r2 * w_batch[None, None, :]
            
            # Compute gradients: grad_J(r1, r2) -> (batch_r1, batch_r2, 3)
            grads = self.jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            
            # Contract: sum_{r2} rho(r2) * grad(r1, r2)
            # weighted_rho_r2: (N_rows, N_cols, batch_r2)
            # grads: (batch_r1, batch_r2, 3)
            # Result: (N_rows, N_cols, batch_r1, 3)
            term = jnp.einsum('ijb,obd->ijod', weighted_rho_r2, grads)
            
            return carry + term, None

        init_val = jnp.zeros((n_rows, n_cols, len(r1_batch), 3))
        final_val, _ = jax.lax.scan(scan_body, init_val, 
                                   (r2_batches, weights_batches, 
                                    rho_rows_batches.transpose(1, 0, 2), 
                                    rho_cols_batches.transpose(1, 0, 2)))
        
        return final_val

    @partial(jax.jit, static_argnames=('batch_size',))
    def _calc_v_batch(self, r1_batch, rho, weights, jastrow_params, batch_size=1000):
        """Calculate V_qt(r₁) for a batch of r1 points by scanning over r2.
        
        Args:
            r1_batch: (batch_size, 3)
            rho: (Nb, N_grid)
            weights: (N_grid,)
            jastrow_params: Jastrow parameters
            batch_size: Inner batch size for r2 scan
            
        Returns:
            V_batch: (Nb^2, batch_size, 3)
        """
        n_orb = rho.shape[0]
        # Use the block implementation with full slices
        full_slice = slice(None)
        v_block = self._calc_v_block(r1_batch, rho, weights, jastrow_params, full_slice, full_slice, batch_size)
        return v_block.reshape(n_orb * n_orb, -1, 3)

    @jax.jit
    def calc_delta_U(self, v_vector, rho_paired, dm1, weights):
        """Calculate delta_U contribution for a batch.
        
        Args:
            v_vector: (Nb^2, batch_size, 3)
            rho_paired: (Nb^2, batch_size)
            dm1: (Nb, Nb)
            weights: (batch_size,)
            
        Returns:
            contribution: (Nb, Nb, Nb, Nb)
        """
        nb = dm1.shape[1]
        
        V = jnp.reshape(v_vector, (nb, nb, -1, 3))
        rho = jnp.reshape(rho_paired, (nb, nb, -1))
        rho_weighted = rho * weights[None, None, :]
        
        compute_W = jax.vmap(lambda V, dm: 2 * jnp.einsum('utx,tu->x', V, dm), 
                            in_axes=(2, None), out_axes=0)
        W = compute_W(V, dm1)
        
        compute_Vbar = jax.vmap(lambda W, V: jnp.einsum('d,srd->sr', W, V), 
                               in_axes=(0, 2))
        Vbar = compute_Vbar(W, V)
        
        compute_X = jax.vmap(lambda V, dm: jnp.einsum('stx,tu->sux', V, dm), 
                            in_axes=(2, None), out_axes=0)
        X = compute_X(V, dm1)
        
        compute_Zbar = jax.vmap(lambda V, X: jnp.einsum('urx,sux->sr', V, X), 
                               in_axes=(2, 0))
        Zbar = compute_Zbar(V, X)
        
        Wbar = 2 * jnp.einsum('uti,tu->i', rho_weighted, dm1)
        
        compute_Y = jax.vmap(lambda V, dm: jnp.einsum('urx,tu->trx', V, dm), 
                            in_axes=(2, None), out_axes=0)
        Y = compute_Y(V, dm1)
        
        compute_G = jax.vmap(lambda rho, X, Y: 
                            jnp.einsum('ur,sux->srx', rho, X) + 
                            jnp.einsum('trx,st->srx', Y, rho),
                            in_axes=(2, 0, 0))
        G = compute_G(rho_weighted, X, Y)

        A = (Vbar - Zbar).transpose(1,2,0)
        B = 0.5 * Wbar[None, None, :, None] * V - jnp.transpose(G, (1, 2, 0, 3))
        
        term1 = jnp.einsum('qpi,sri->qpsr', rho_weighted, A)
        term2 = jnp.einsum('qpix,srix->qpsr', V, B)
        
        result = -(term1 + term2)
        return result

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
        n_devices = jax.local_device_count()
        n_grid = self.n_grid
        
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Check if dm1 is diagonal
        is_diagonal = jnp.allclose(dm1, jnp.diag(jnp.diagonal(dm1)))
        if not is_diagonal:
            raise ValueError("Non-diagonal density matrix for XTC calculation is not supported.")
        
        n_occ_vec = jnp.diagonal(dm1)
        
        # Pad grid to be divisible by n_devices
        remainder = n_grid % n_devices
        if remainder != 0:
            padding = n_devices - remainder
            padded_grid_points = jnp.pad(self.grid_points, ((0, padding), (0, 0)))
            padded_weights = jnp.pad(self.weights, ((0, padding),))
            padded_rho = jnp.pad(self.rho, ((0, 0), (0, padding)))
        else:
            padded_grid_points = self.grid_points
            padded_weights = self.weights
            padded_rho = self.rho
            
        n_grid_padded = padded_grid_points.shape[0]
        n_per_device = n_grid_padded // n_devices
        
        # Shard arrays for r1: (n_devices, n_per_device, ...)
        sharded_grid_r1 = padded_grid_points.reshape(n_devices, n_per_device, 3)
        sharded_weights_r1 = padded_weights.reshape(n_devices, n_per_device)
        # rho: (Nb, N) -> (Nb, n_dev, N_per) -> (n_dev, Nb, N_per)
        sharded_rho_r1 = padded_rho.reshape(self.n_orb, n_devices, n_per_device).transpose(1, 0, 2)
        
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

        def compute_on_device(grid_r1, weights_r1, rho_r1):
            n_local = grid_r1.shape[0]
            local_remainder = n_local % batch_size
            if local_remainder != 0:
                local_padding = batch_size - local_remainder
                grid_r1_batched = jnp.pad(grid_r1, ((0, local_padding), (0, 0)))
                weights_r1_batched = jnp.pad(weights_r1, ((0, local_padding),))
                rho_r1_batched = jnp.pad(rho_r1, ((0, 0), (0, local_padding)))
            else:
                grid_r1_batched = grid_r1
                weights_r1_batched = weights_r1
                rho_r1_batched = rho_r1
                
            # Reshape for scanning
            r1_batches = grid_r1_batched.reshape(-1, batch_size, 3)
            weights_batches = weights_r1_batched.reshape(-1, batch_size)
            rho_batches = rho_r1_batched.reshape(self.n_orb, -1, batch_size)
            
            def scan_body(carry, args):
                r1_batch, w_batch, rho_batch = args
                
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
                V_occ_occ = self._calc_v_block(r1_batch, self.rho, self.weights, jastrow_params, 
                                              slice_occ, slice_occ, batch_size)
                
                # 2. V_pr
                V_pr = self._calc_v_block(r1_batch, self.rho, self.weights, jastrow_params, 
                                         slice_p, slice_r, batch_size)
                
                # 3. V_qs
                V_qs = self._calc_v_block(r1_batch, self.rho, self.weights, jastrow_params, 
                                         slice_q, slice_s, batch_size)
                
                # 4. V_occ_blocks
                # We need V_{k,p}, V_{k,r}, V_{k,q}, V_{k,s} where k in occ
                # We can compute V_{occ, p} etc.
                V_occ_p = self._calc_v_block(r1_batch, self.rho, self.weights, jastrow_params, 
                                            slice_occ, slice_p, batch_size)
                V_occ_r = self._calc_v_block(r1_batch, self.rho, self.weights, jastrow_params, 
                                            slice_occ, slice_r, batch_size)
                V_occ_q = self._calc_v_block(r1_batch, self.rho, self.weights, jastrow_params, 
                                            slice_occ, slice_q, batch_size)
                V_occ_s = self._calc_v_block(r1_batch, self.rho, self.weights, jastrow_params, 
                                            slice_occ, slice_s, batch_size)
                
                # --- Compute Intermediates (Diagonal dm1) ---
                
                # W = 2 * sum_k V_{kk} n_k
                # V_occ_occ: (Nocc, Nocc, batch, 3)
                # Diagonal V_{kk}: (Nocc, batch, 3)
                V_kk = jnp.einsum('ii...->i...', V_occ_occ)
                # n_occ_vec: (Nocc,)
                # W: (batch, 3)
                W = 2 * jnp.einsum('i,ibd->bd', n_occ_vec_active, V_kk)
                
                # Wbar = 2 * sum_k rho_{kk} n_k
                # rho_batch: (N_orb, batch)
                rho_occ = rho_batch[slice_occ] # (Nocc, batch)
                # rho_{kk} is just rho_occ * rho_occ? No, rho_{kk}(r) = |phi_k(r)|^2
                rho_kk = rho_occ * rho_occ
                # Wbar: (batch,)
                Wbar = 2 * jnp.einsum('i,ib->b', n_occ_vec_active, rho_kk)
                
                # --- Block (q, s) Terms ---
                
                # Zbar_{qs} = sum_k V_{kq} V_{sk} n_k
                # V_{kq} = V_{qk} = V_occ_q (Nocc, Nq, batch, 3)
                # V_{sk} = V_{ks} = V_occ_s (Nocc, Ns, batch, 3)
                # Zbar_{qs}: (Nq, Ns, batch)
                Zbar_qs = jnp.einsum('i,iqbd,isbd->qsb', n_occ_vec_active, V_occ_q, V_occ_s)
                
                # G_{qs} = sum_k (rho_{kq} V_{sk} + rho_{ks} V_{qk}) n_k
                # rho_{kq} = rho_k * rho_q
                rho_q = rho_batch[slice_q] # (Nq, batch)
                rho_s = rho_batch[slice_s] # (Ns, batch)
                # rho_{kq}: (Nocc, Nq, batch)
                rho_kq = jnp.einsum('ib,qb->iqb', rho_occ, rho_q)
                rho_ks = jnp.einsum('ib,sb->isb', rho_occ, rho_s)
                
                # G_{qs}: (Nq, Ns, batch, 3)
                G_qs = jnp.einsum('i,iqb,isbd->qsbd', n_occ_vec_active, rho_kq, V_occ_s) + \
                       jnp.einsum('i,isb,iqbd->qsbd', n_occ_vec_active, rho_ks, V_occ_q)
                       
                # Vbar_{qs} = sum_d W_d * V_{qs,d}
                Vbar_qs = jnp.einsum('bd,qsbd->qsb', W, V_qs)
                A_qs = Vbar_qs - Zbar_qs # (Nq, Ns, batch)
                
                # B_{qs} = 0.5 * Wbar * V_{qs} - G_{qs}
                # B_{qs}: (Nq, Ns, batch, 3)
                B_qs = 0.5 * Wbar[None, None, :, None] * V_qs - G_qs
                
                # --- Block (p, r) Terms ---
                # Symmetric to (q, s)
                
                # Zbar_{pr}
                Zbar_pr = jnp.einsum('i,ipbd,irbd->prb', n_occ_vec_active, V_occ_p, V_occ_r)
                
                # G_{pr}
                rho_p = rho_batch[slice_p]
                rho_r = rho_batch[slice_r]
                rho_kp = jnp.einsum('ib,pb->ipb', rho_occ, rho_p)
                rho_kr = jnp.einsum('ib,rb->irb', rho_occ, rho_r)
                
                G_pr = jnp.einsum('i,ipb,irbd->prbd', n_occ_vec_active, rho_kp, V_occ_r) + \
                       jnp.einsum('i,irb,ipbd->prbd', n_occ_vec_active, rho_kr, V_occ_p)
                       
                # A_{pr}
                Vbar_pr = jnp.einsum('bd,prbd->prb', W, V_pr)
                A_pr = Vbar_pr - Zbar_pr
                
                # B_{pr}
                B_pr = 0.5 * Wbar[None, None, :, None] * V_pr - G_pr
                
                # --- Combine Terms ---
                
                # term1 = rho_{pr} * A_{qs}
                # rho_{pr} = rho_p * rho_r
                rho_pr = jnp.einsum('pb,rb->prb', rho_p, rho_r)
                # Weighted rho_pr for integration
                rho_pr_w = rho_pr * w_batch[None, None, :]
                
                # term1: (Np, Nr, Nq, Ns)
                # einsum: prb, qsb -> prqs (sum over b)
                term1 = jnp.einsum('prb,qsb->prqs', rho_pr_w, A_qs)
                
                # term2 = V_{pr} * B_{qs}
                # V_{pr}: (Np, Nr, batch, 3)
                # B_{qs}: (Nq, Ns, batch, 3)
                # term2: (Np, Nr, Nq, Ns)
                # We can weight V_pr
                V_pr_w = V_pr * w_batch[None, None, :, None]
                term2 = jnp.einsum('prbd,qsbd->prqs', V_pr_w, B_qs)
                
                # term1_sym = rho_{qs} * A_{pr}
                rho_qs = jnp.einsum('qb,sb->qsb', rho_q, rho_s)
                rho_qs_w = rho_qs * w_batch[None, None, :]
                term1_sym = jnp.einsum('qsb,prb->prqs', rho_qs_w, A_pr)
                
                # term2_sym = V_{qs} * B_{pr}
                V_qs_w = V_qs * w_batch[None, None, :, None]
                term2_sym = jnp.einsum('qsbd,prbd->prqs', V_qs_w, B_pr)
                
                # Total for this batch
                contrib = term1 + term2 + term1_sym + term2_sym
                
                return carry + contrib, None

            init_val = jnp.zeros((Np, Nr, Nq, Ns))
            
            local_delta_U, _ = jax.lax.scan(scan_body, init_val, (r1_batches, weights_batches, rho_batches.transpose(1, 0, 2)))
            
            # Sum results across devices
            total_delta_U = jax.lax.psum(local_delta_U, axis_name='devices')
            return total_delta_U

        # Execute pmap
        pmapped_compute = jax.pmap(compute_on_device, axis_name='devices')
        
        delta_U_replicated = pmapped_compute(sharded_grid_r1, sharded_weights_r1, sharded_rho_r1)
        
        total_delta_U = delta_U_replicated[0]
        
        # Result is -(term1 + term2 + term1_sym + term2_sym)
        # Note: term1_sym + term2_sym corresponds to the transpose part in the full code
        # In full code: result + result.T
        # Here we computed both explicitly.
        # But wait, term1_sym indices are (p, r, q, s) because we einsum'd that way.
        # Is that correct?
        # Full code: result + result.transpose(2, 3, 0, 1)
        # result indices: q, p, s, r (internal) -> p, r, q, s (external)
        # result.T indices: s, r, q, p (internal) -> q, s, p, r (external)
        # Here:
        # term1 indices: p, r, q, s
        # term1_sym indices: p, r, q, s (constructed as rho_qs * A_pr)
        # Does term1_sym correspond to result.T?
        # result.T corresponds to swapping (p,r) with (q,s).
        # term1: (p,r) from rho/V, (q,s) from A/B.
        # term1_sym: (q,s) from rho/V, (p,r) from A/B.
        # Yes, this covers both symmetric parts.
        
        return -total_delta_U

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

    def get_delta_h(self, jastrow_params, dm1=None):
        """Get or compute delta_h."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        delta_U = self.get_delta_U(jastrow_params, dm1)
        return self._calc_delta_h(delta_U, dm1)

    @jax.jit
    def get_1b(self, jastrow_params, dm1=None):
        """Get one-body operator correction."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        return self.get_delta_h(jastrow_params, dm1)

    @partial(jax.jit, static_argnames=('block_str', 'ranges'))
    def get_2b(self, jastrow_params, dm1=None, block_str=None, ranges=None):
        """Compute two-body integrals correction."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
        
        # Get TC's two-body correction (negative of K terms)
        tc_correction = super().get_2b(jastrow_params, ranges=ranges)
        
        # Add delta_U
        delta_U = self.get_delta_U(jastrow_params, dm1, ranges=ranges)
        
        return tc_correction + delta_U

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
class ISDFXTC(XTC):
    """JAX implementation of Extended Transcorrelated method using ISDF.
    
    Attributes:
        C_rho: ISDF basis for density (Nb^2, N_fused)
        xi_rho: ISDF coefficients for density (N_fused, N_grid)
        C_grad: ISDF basis for gradients (Nb^2, N_fused, 3)
        xi_grad: ISDF coefficients for gradients (N_fused, N_grid, 3)
        pivots: ISDF pivot indices (N_fused,)
    """
    C_rho: jnp.ndarray = struct.field(default=None)
    xi_rho: jnp.ndarray = struct.field(default=None)
    C_grad: jnp.ndarray = struct.field(default=None)
    xi_grad: jnp.ndarray = struct.field(default=None)
    pivots: jnp.ndarray = struct.field(default=None)

    @classmethod
    def from_xtc(cls, xtc_obj, n_rank=None):
        """Initialize ISDFXTC object from XTC object."""
        from . import df
        
        if n_rank is None:
            n_rank = xtc_obj.grid_points.shape[0] // 4
            
        # Perform ISDF decomposition
        C_rho, xi_rho, C_grad, xi_grad, pivots = df.isdf_decompose(
            xtc_obj.rho, xtc_obj.nabla_rho, n_rank, n_rank, weights=xtc_obj.weights
        )
        
        return cls(
            grid_points=xtc_obj.grid_points,
            weights=xtc_obj.weights,
            rho=xtc_obj.rho,
            nabla_rho=xtc_obj.nabla_rho,
            n_orb=xtc_obj.n_orb,
            grid_lvl=xtc_obj.grid_lvl,
            jastrow_factor=xtc_obj.jastrow_factor,
            mo_coeff=xtc_obj.mo_coeff,
            mo_occ=xtc_obj.mo_occ,
            energy_nuc=xtc_obj.energy_nuc,
            C_rho=C_rho,
            xi_rho=xi_rho,
            C_grad=C_grad,
            xi_grad=xi_grad,
            pivots=pivots
        )

    def get_delta_U(self, jastrow_params, dm1=None, batch_size=1000):
        """Get delta_U matrix using ISDF with pmap support."""
        n_devices = jax.local_device_count()
        n_grid = self.n_grid
        
        if dm1 is None:
            dm1 = self._get_mf_dm()
        
        # Pad grid to be divisible by n_devices
        remainder = n_grid % n_devices
        if remainder != 0:
            padding = n_devices - remainder
            padded_grid_points = jnp.pad(self.grid_points, ((0, padding), (0, 0)))
            padded_weights = jnp.pad(self.weights, ((0, padding),))
            padded_xi_rho = jnp.pad(self.xi_rho, ((0, 0), (0, padding)))
        else:
            padded_grid_points = self.grid_points
            padded_weights = self.weights
            padded_xi_rho = self.xi_rho
            
        n_grid_padded = padded_grid_points.shape[0]
        n_per_device = n_grid_padded // n_devices
        
        # Shard arrays
        sharded_grid = padded_grid_points.reshape(n_devices, n_per_device, 3)
        sharded_weights = padded_weights.reshape(n_devices, n_per_device)
        # xi_rho: (N_rank, N) -> (N_rank, n_dev, N_per) -> (n_dev, N_rank, N_per)
        sharded_xi_rho = padded_xi_rho.reshape(self.C_rho.shape[1], n_devices, n_per_device).transpose(1, 0, 2)
        
        # Pre-compute device-independent quantities
        Gb = jnp.einsum('tub,tu->b', self.C_rho.reshape(self.n_orb, self.n_orb, -1), dm1)
        
        # Full arrays for integration (replicated on each device)
        full_grid = self.grid_points
        full_weights = self.weights
        full_xi_rho = self.xi_rho

        def compute_on_device(grid_shard, weights_shard, xi_shard):
            return self._calc_delta_U_isdf_shard(
                jastrow_params, dm1, grid_shard, weights_shard, xi_shard, 
                full_grid, full_weights, full_xi_rho, Gb, batch_size
            )

        # Execute pmap
        pmapped_compute = jax.pmap(compute_on_device, axis_name='devices')
        
        # Returns tuple of accumulators: (X, X1, Q, Q3)
        # Each has shape (n_devices, ...)
        X_rep, X1_rep, Q_rep, Q3_rep = pmapped_compute(sharded_grid, sharded_weights, sharded_xi_rho)
        
        # Sum over devices
        X = jnp.sum(X_rep, axis=0)
        X1 = jnp.sum(X1_rep, axis=0)
        Q = jnp.sum(Q_rep, axis=0)
        Q3 = jnp.sum(Q3_rep, axis=0)
        
        # Reconstruct Terms
        Nb = self.n_orb
        N_rank = self.C_rho.shape[1]
        C_rho_reshaped = self.C_rho.reshape(Nb, Nb, N_rank)
        
        # Term 1: 2 * C^a C^d X1_{ad}
        term1 = 2 * jnp.einsum('pqa,ad,rsd->pqrs', C_rho_reshaped, X1, C_rho_reshaped)
        
        # Term 2: - C^a Q_{rsa}
        term2 = -jnp.einsum('pqa,rsa->pqrs', C_rho_reshaped, Q)
        
        # Term 3: - C^a Q3_{rsa}
        term3 = -jnp.einsum('pqa,rsa->pqrs', C_rho_reshaped, Q3)
        
        # Term 4: C^a C^c X_{ac}
        term4 = jnp.einsum('pqa,rsc,ac->pqrs', C_rho_reshaped, C_rho_reshaped, X)
        
        result = term1 + term2 + term3 + term4
        
        # Symmetrize
        return -(result + result.transpose(2, 3, 0, 1))

    def _calc_delta_U_isdf_shard(self, jastrow_params, dm1, grid_points, weights, xi_rho, 
                                 full_grid, full_weights, full_xi_rho, Gb, batch_size=1000):
        """Calculate ΔU contribution for a shard of grid points."""
        Nb = self.n_orb
        N_rank = self.C_rho.shape[1]
        N_shard = grid_points.shape[0]
        
        C_rho_reshaped = self.C_rho.reshape(Nb, Nb, N_rank)
        
        # Pre-compute L for Terms 2 & 3
        L = jnp.einsum('tu,tsc->usc', dm1, C_rho_reshaped)
        
        # Initialize accumulators
        X_acc = jnp.zeros((N_rank, N_rank))
        X1_acc = jnp.zeros((N_rank, N_rank))
        Q_acc = jnp.zeros((Nb, Nb, N_rank))
        Q3_acc = jnp.zeros((Nb, Nb, N_rank))
        
        # Prepare batches for the SHARD
        # We loop over the shard points (index i)
        padded_size = ((N_shard + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(grid_points, ((0, padded_size - N_shard), (0, 0)))
        padded_weights = jnp.pad(weights, (0, padded_size - N_shard))
        padded_xi = jnp.pad(xi_rho, ((0, 0), (0, padded_size - N_shard)))
        
        r_batches = padded_grid.reshape(-1, batch_size, 3)
        w_batches = padded_weights.reshape(-1, batch_size)
        xi_batches = padded_xi.reshape(N_rank, -1, batch_size).transpose(1, 0, 2)
        
        def scan_body(carry, args):
            r_batch, w_batch, xi_batch = args
            X_curr, X1_curr, Q_curr, Q3_curr = carry
            
            # Compute Gradients: grad(full_grid, r_batch)
            # This gives gradients for all j (full_grid) wrt r_batch (i)
            # Shape: (N_full, batch, 3)
            grads = self.jastrow_factor.grad_r_batch(full_grid, r_batch, jastrow_params)
            
            # Compute G: (N_rank, batch, 3)
            # G_{bi\alpha} = \sum_j w_j \xi_b(j) \nabla_\alpha J(r_j, r_i)
            G = jnp.einsum('j,bj,jic->bic', full_weights, full_xi_rho, grads)
            
            # --- Term 4 (Easy Term) ---
            # \tilde{w}_i = w_i \sum_b G_b \xi_b(i)
            w_tilde = w_batch * jnp.einsum('b,bi->i', Gb, xi_batch)
            
            # X_{ac} += \sum_i \tilde{w}_i K_{ac}(i)
            # K_{ac}(i) = \sum_k G_{ak}(i) G_{ck}(i)
            X_update = jnp.einsum('i,aik,cik->ac', w_tilde, G, G)
            
            # --- Term 1 (Easy Term) ---
            # H_k(i) = \sum_b G_b G_{bk}(i)
            H = jnp.einsum('b,bik->ik', Gb, G)
            # V_d(i) = \sum_k H_k(i) G_{dk}(i)
            V = jnp.einsum('ik,dik->di', H, G)
            # X1_{ad} += \sum_i w_i \xi_a(i) V_d(i)
            X1_update = jnp.einsum('i,ai,di->ad', w_batch, xi_batch, V)
            
            # --- Term 2 (Hard Term) ---
            # Y_{usk}(i) = \sum_c L_{usc} G_{ck}(i)
            Y = jnp.einsum('usc,cik->usik', L, G)
            
            # Z_{ruk}(i) = \sum_b C_{rub} G_{bk}(i)
            Z = jnp.einsum('rub,bik->ruik', C_rho_reshaped, G)
            
            # Q_{rsa} += \sum_i w_i \xi_a(i) \sum_{uk} Y_{usk}(i) Z_{ruk}(i)
            YZ = jnp.einsum('usik,ruik->rsi', Y, Z)
            Q_update = jnp.einsum('i,ai,rsi->rsa', w_batch, xi_batch, YZ)
            
            # --- Term 3 (Hard Term) ---
            # Part 1: Q3_1
            L_tilde = jnp.einsum('ci,usc->usi', xi_batch, L)
            GZ = jnp.einsum('aik,ruik->ruia', G, Z)
            Q3_1_update = jnp.einsum('i,usi,ruia->rsa', w_batch, L_tilde, GZ)
            
            # Part 2: Q3_2
            C_tilde = jnp.einsum('bi,rub->rui', xi_batch, C_rho_reshaped)
            YC = jnp.einsum('usik,rui->rski', Y, C_tilde)
            Q3_2_update = jnp.einsum('i,aik,rski->rsa', w_batch, G, YC)
            
            return (X_curr + X_update, X1_curr + X1_update, Q_curr + Q_update, Q3_curr + Q3_1_update + Q3_2_update), None

        final_accumulators, _ = jax.lax.scan(scan_body, (X_acc, X1_acc, Q_acc, Q3_acc), (r_batches, w_batches, xi_batches))
        
        return final_accumulators

    def get_2b(self, jastrow_params, dm1=None):
        """Compute two-body integrals correction using ISDF."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        k_nabla = kmat_jax.calc_K1_isdf(
            self.C_rho, self.xi_rho, self.C_grad, self.xi_grad,
            self.jastrow_factor, jastrow_params, self.grid_points, self.weights
        )
        # k_laplacian = -(k_nabla + k_nabla^T)
        k_nabla = k_nabla.reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        k_laplacian = -(k_nabla + k_nabla.swapaxes(0, 1))
        
        k_square = kmat_jax.calc_K3_isdf(
            self.C_rho, self.xi_rho,
            self.jastrow_factor, jastrow_params, self.grid_points, self.weights
        )
        
        k_square = k_square.reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        tc_correction = 0.5 * (k_laplacian + k_square) + k_nabla
        tc_correction += tc_correction.transpose(2, 3, 0, 1)
        tc_correction = -tc_correction
        
        # Add delta_U
        delta_U = self.get_delta_U(jastrow_params, dm1)
        
        return tc_correction + delta_U
