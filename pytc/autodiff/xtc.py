"""JAX implementation of X transcorrelated methods."""

from functools import partial, reduce
import numpy as np
import os
import gc
import logging
import time
import jax
import jax.numpy as jnp
import h5py
from flax import struct
from .tc import TC, ISDFTC
from . import tc_helper
from . import kmat as kmat_jax

logger = logging.getLogger(__name__)

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
        logger.debug("Starting XTC.get_delta_U")
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
        logger.debug(f"XTC.get_delta_U completed in {total_time:.4f} s")
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

    def get_1b(self, jastrow_params, dm1=None, block_str=None, ranges=None, orb_block_size=256, batch_size=1000):
        """Get one-body operator correction."""
        return self.get_delta_h(jastrow_params, dm1, block_str, ranges, orb_block_size, batch_size)

    def get_2b(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        """Compute two-body integrals correction."""
        start_time = time.perf_counter()
        logger.debug("Starting XTC.get_2b")
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
        
        tc_correction = super().get_2b(jastrow_params, ranges=ranges)
        
        delta_U = self.get_delta_U(jastrow_params, dm1, ranges=ranges, batch_size=batch_size)
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"XTC.get_2b completed in {time.perf_counter() - start_time:.4f} s")
        return tc_correction + delta_U

    def get_const(self, jastrow_params, dm1=None):
        """Compute constant contribution."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        
        delta_h = self.get_delta_h(jastrow_params, dm1)
        const = -2/3 * jnp.einsum('qp,pq->', delta_h, dm1)
        const += self.energy_nuc
        return const
    
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
    def from_xtc(cls, xtc_obj, n_rank=None, is_incore=False, save_path=None, ls_grid_batch_size=16384):
        """Initialize ISDFXTC object from XTC object.
        
        Args:
            xtc_obj: XTC object
            n_rank: Number of ISDF ranks
            is_incore: Whether to perform in-core decomposition
            save_path: Path to save ISDF kernels
            ls_grid_batch_size: Batch size for grid evaluation in linear solver in ISDF decomposition (default: 16384)
        """
        from . import df
        
        if n_rank is None:
            n_rank = xtc_obj.grid_points.shape[0] // 4
            
        # Perform ISDF decomposition
        phi_isdf, xi_phi, grad_phi_isdf, xi_grad, pivots, actual_save_path = df.isdf_decompose(
            xtc_obj.phi, xtc_obj.grad_phi, n_rank, n_rank, weights=xtc_obj.weights,
            is_incore=is_incore, save_path=save_path, grid_batch_size=ls_grid_batch_size
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
            nocc=xtc_obj.nocc,
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

    def isdf(self, jastrow_params, save_path=None, batch_size=1000, orb_block_size=128, host_grid_block_size=None):
        """Compute ISDF intermediates and store them.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor.
            save_path: Optional path to save intermediates to HDF5.
            batch_size: Batch size for computation.
            orb_block_size: Block size for orbital batching of X kernel.
            host_grid_block_size: Block size for grid batching on host.
        """
        logger.info("Computing ISDF intermediates (XTC)...")
        start_time = time.perf_counter()
        
        # Use save_path if provided, otherwise use self.save_path
        out_path = save_path if save_path else self.save_path
        
        # 1. Compute TC kernels (K1, K3, L_aux) using base class
        isdf_tc = super().isdf(jastrow_params, save_path=out_path, batch_size=batch_size, host_grid_block_size=host_grid_block_size)
        kernels = isdf_tc.isdf_kernels
        
        # 2. Compute Delta U kernels (D, X) with orbital batching
        # Check if D and X already exist in HDF5
        if out_path and os.path.exists(out_path):
            try:
                f = h5py.File(out_path, 'r')
                if 'D' in f and 'X' in f:
                    logger.info(f"  Found existing D and X in {out_path}. Reading from file...")
                    kernels['D'] = f['D'][:]
                    if self.is_incore:
                        logger.debug("  incore mode: Loading X into RAM")
                        kernels['X'] = f['X'][:]
                        f.close()
                    else:
                        # Stream X from file. 
                        # Return the dataset object directly. 
                        # Do NOT close 'f' here; the dataset object keeps the file open.
                        logger.debug("  out-of-core mode: Streaming X from file")
                        kernels['X'] = f['X']
                    logger.debug(f"ISDF intermediates (Delta U) loaded from file in {time.perf_counter() - start_time:.4f} s")
                    return self.replace(isdf_kernels=kernels, save_path=out_path)
            except (IOError, KeyError) as e:
                logger.warning(f"  Error reading Delta U kernels from {out_path}: {e}. Recomputing...")

        # Pass L_aux to avoid redundant calculation
        delta_u_kernels = self.compute_delta_u_kernels(
            jastrow_params, batch_size, L_aux=kernels.get('L_aux'),
            orb_block_size=orb_block_size,
            save_path=out_path,
            host_grid_block_size=host_grid_block_size
        )
        kernels.update(delta_u_kernels)
        
        # 3. Discard L_aux from ISDFXTC kernels to save RAM and avoid JAX types error
        # L_aux is used to compute D and X, but not needed for get_2b or get_delta_U
        if 'L_aux' in kernels:
            del kernels['L_aux']
        
        # Persistence for other kernels (phi_isdf, etc.) if out_path provided
        if out_path:
            with h5py.File(out_path, 'a') as f:
                if 'phi_isdf' not in f: f.create_dataset('phi_isdf', data=np.array(self.phi_isdf))
                if 'grad_phi_isdf' not in f: f.create_dataset('grad_phi_isdf', data=np.array(self.grad_phi_isdf))
                if 'pivots' not in f: f.create_dataset('pivots', data=np.array(self.pivots))
                
        logger.info(f"ISDF intermediates (Delta U) computed in {time.perf_counter() - start_time:.4f} s")
        
        return self.replace(isdf_kernels=kernels, save_path=out_path)

    def compute_delta_u_kernels(self, jastrow_params, batch_size=1000, L_aux=None, orb_block_size=128, save_path=None, host_grid_block_size=None):
        """Compute D, X kernels for Delta U with orbital and grid batching."""
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)
            
        n_orb = self.n_orb
        n_rank = self.phi_isdf.shape[1]
        dm1 = self._get_mf_dm()
        
        # Precompute Gb and L_Q (low-rank factor of Q) to avoid redundant work
        Gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        
        # Q = phi_dm.T @ phi_isdf has rank <= n_orb
        # Factor as Q = L_Q @ L_Q.T where L_Q has shape (N_rank, n_orb)
        dm1_diag = jnp.diagonal(dm1)
        sqrt_dm1 = jnp.sqrt(jnp.maximum(dm1_diag, 0.0))  # Ensure non-negative
        L_Q = self.phi_isdf.T * sqrt_dm1[None, :]  # (N_rank, n_orb)
        
        # 1. Compute D kernel
        logger.info("Computing D kernel...")
        D = self._compute_D_kernel(jastrow_params, batch_size, L_aux, Gb=Gb, host_grid_block_size=host_grid_block_size)
        
        # 2. Compute X kernel with orbital batching
        logger.info("Computing X kernel...")
        
        if save_path:
            # If L_aux is a dataset from the same file, we must load it or close it.
            if isinstance(L_aux, h5py.Dataset):
                if L_aux.file.filename == os.path.abspath(save_path):
                    logger.info("  L_aux is a dataset from the target file. Loading into RAM to allow reopening in 'a' mode.")
                    L_aux = L_aux[:]
            
            f = h5py.File(save_path, 'a')
            if 'D' in f: del f['D']
            f.create_dataset('D', data=np.array(D))
            if 'X' in f: del f['X']
            X = f.create_dataset('X', (n_orb, n_orb, n_rank), dtype='f8')
        else:
            X = np.zeros((n_orb, n_orb, n_rank), dtype='f8')
            
        # Exploit symmetry: X[r,s,a] = X[s,r,a], only compute upper triangle blocks
        for r0 in range(0, n_orb, orb_block_size):
            r1 = min(r0 + orb_block_size, n_orb)
            logger.info(f"  compute_delta_u_kernels: Computing X blocks for r-range [{r0}:{r1}]...")
            for s0 in range(r0, n_orb, orb_block_size):  # Start from r0 for upper triangle
                s1 = min(s0 + orb_block_size, n_orb)
            
                ranges = (slice(None), slice(None), slice(r0, r1), slice(s0, s1))
                X_block = self._compute_X_kernel(jastrow_params, ranges, batch_size, L_aux, Gb=Gb, L_Q=L_Q, host_grid_block_size=host_grid_block_size)
                X_block_np = np.array(X_block)
                X[r0:r1, s0:s1, :] = X_block_np
                # Fill symmetric block (only if not diagonal)
                if r0 != s0:
                    X[s0:s1, r0:r1, :] = X_block_np.transpose(1, 0, 2)
                del X_block, X_block_np
                gc.collect()
                
        if save_path:
            # Return dataset object for X to allow streaming
            return {'D': f['D'][:], 'X': X}
        else:
            return {'D': D, 'X': X}

    def _compute_D_kernel(self, jastrow_params, batch_size=1024, L_aux=None, Gb=None, host_grid_block_size=None):
        """Compute D kernel for Delta U with grid-blocking to save host RAM."""
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)
            
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        dm1 = self._get_mf_dm()
        
        if host_grid_block_size is None:
            host_grid_block_size = n_grid
            
        if Gb is None:
            Gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        phi_isdf = self.phi_isdf
        n_orb = self.n_orb
        
        # Initialize D on host
        D = np.zeros((n_rank, n_rank))
        
        # Open xi_phi dataset if needed
        xi_phi_ds = None
        f_xi = None
        if self.xi_phi is None and self.save_path:
            f_xi = h5py.File(self.save_path, 'r')
            xi_phi_ds = f_xi['xi_phi']
            
        # 4. Run pmap
        def compute_D_on_device(grid_shard, weights_shard, xi_shard, G_shard, jastrow_params, Gb, dm1, phi_isdf, n_orb):
            return self._calc_D_shard(
                jastrow_params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, phi_isdf, None, n_orb, batch_size
            )
            
        pmapped_D = jax.pmap(compute_D_on_device, axis_name='devices', in_axes=(0, 0, 0, 0, None, None, None, None, None))

        try:
            for g0 in range(0, n_grid, host_grid_block_size):
                g1 = min(g0 + host_grid_block_size, n_grid)
                n_block = g1 - g0
                logger.debug(f"    _compute_D_kernel: Processing grid block [{g0}:{g1}]...")
                
                remainder = n_block % n_devices
                padding = (n_devices - remainder) if remainder != 0 else 0
                n_block_padded = n_block + padding
                n_per_device = n_block_padded // n_devices
                
                # 1. Shard Grid and Weights
                grid_block = self.grid_points[g0:g1]
                weights_block = self.weights[g0:g1]
                if padding > 0:
                    grid_block = jnp.pad(grid_block, ((0, padding), (0, 0)))
                    weights_block = jnp.pad(weights_block, ((0, padding),))
                
                sharded_grid = grid_block.reshape(n_devices, n_per_device, 3)
                sharded_weights = weights_block.reshape(n_devices, n_per_device)
                
                # 2. Shard G (L_aux)
                sharded_G_list = []
                for d in range(n_devices):
                    start = g0 + d * n_per_device
                    end = min(g0 + (d + 1) * n_per_device, g1)
                    actual_len = end - start
                    
                    # Slice L_aux (could be HDF5 dataset or JAX array)
                    G_d = -L_aux[:, start:end, :]
                    if actual_len < n_per_device:
                        G_d = jnp.pad(G_d, ((0, 0), (0, n_per_device - actual_len), (0, 0)))
                    sharded_G_list.append(jax.device_put(G_d, devices[d]))
                sharded_G = jax.device_put_sharded(sharded_G_list, devices)
                
                # 3. Shard xi_phi
                sharded_xi_phi_list = []
                for d in range(n_devices):
                    start = g0 + d * n_per_device
                    end = min(g0 + (d + 1) * n_per_device, g1)
                    actual_len = end - start
                    
                    if self.xi_phi is not None:
                        xi_phi_d = self.xi_phi[:, start:end]
                    else:
                        xi_phi_d = xi_phi_ds[:, start:end]
                        
                    if actual_len < n_per_device:
                        xi_phi_d = np.pad(xi_phi_d, ((0, 0), (0, n_per_device - actual_len)))
                    sharded_xi_phi_list.append(jax.device_put(xi_phi_d, devices[d]))
                sharded_xi_phi = jax.device_put_sharded(sharded_xi_phi_list, devices)
                
                D_rep = pmapped_D(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, jastrow_params, Gb, dm1, phi_isdf, n_orb)
                D += np.array(jnp.sum(D_rep, axis=0))
                
                # Explicitly clear memory
                del sharded_G, sharded_xi_phi, sharded_grid, sharded_weights, D_rep
                gc.collect()
                
        finally:
            if f_xi: f_xi.close()
            
        return D

    def _compute_X_kernel(self, jastrow_params, ranges, batch_size=1024, L_aux=None, Gb=None, L_Q=None, host_grid_block_size=None):
        """Compute X kernel for Delta U for a specific orbital range with grid-blocking.
        
        Uses low-rank factorization: Q = L_Q @ L_Q.T where L_Q has shape (N_rank, n_orb).
        This reduces the expensive O(batch × Nr × N_rank²) matmuls to O(batch × Nr × N_rank × n_orb).
        """
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)
            
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        dm1 = self._get_mf_dm()
        
        if host_grid_block_size is None:
            host_grid_block_size = n_grid
            
        if Gb is None:
            Gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        if L_Q is None:
            # Compute L_Q if not provided
            dm1 = self._get_mf_dm()
            dm1_diag = jnp.diagonal(dm1)
            sqrt_dm1 = jnp.sqrt(jnp.maximum(dm1_diag, 0.0))
            L_Q = self.phi_isdf.T * sqrt_dm1[None, :]
            
        phi_isdf = self.phi_isdf
        n_orb = self.n_orb
        
        # Initialize X block on host
        slice_p, slice_q, slice_r, slice_s = ranges
        Nr = self.phi_isdf[slice_r].shape[0]
        Ns = self.phi_isdf[slice_s].shape[0]
        X = np.zeros((Nr, Ns, n_rank))
        
        # Open xi_phi dataset if needed
        xi_phi_ds = None
        f_xi = None
        if self.xi_phi is None and self.save_path:
            f_xi = h5py.File(self.save_path, 'r')
            xi_phi_ds = f_xi['xi_phi']
            
        # 4. Run pmap
        def compute_X_on_device(grid_shard, weights_shard, xi_shard, G_shard, jastrow_params, Gb, dm1, phi_isdf, n_orb, L_Q):
            return self._calc_X_shard(
                jastrow_params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, phi_isdf, ranges, n_orb, batch_size, L_Q
            )
            
        pmapped_X = jax.pmap(compute_X_on_device, axis_name='devices', in_axes=(0, 0, 0, 0, None, None, None, None, None, None))

        try:
            for g0 in range(0, n_grid, host_grid_block_size):
                g1 = min(g0 + host_grid_block_size, n_grid)
                logger.debug(f"    _compute_X_kernel: Processing grid block [{g0}:{g1}]...")
                n_block = g1 - g0
                
                remainder = n_block % n_devices
                padding = (n_devices - remainder) if remainder != 0 else 0
                n_block_padded = n_block + padding
                n_per_device = n_block_padded // n_devices
                
                # 1. Shard Grid and Weights
                grid_block = self.grid_points[g0:g1]
                weights_block = self.weights[g0:g1]
                if padding > 0:
                    grid_block = jnp.pad(grid_block, ((0, padding), (0, 0)))
                    weights_block = jnp.pad(weights_block, ((0, padding),))
                
                sharded_grid = grid_block.reshape(n_devices, n_per_device, 3)
                sharded_weights = weights_block.reshape(n_devices, n_per_device)
                
                # 2. Shard G (L_aux)
                sharded_G_list = []
                for d in range(n_devices):
                    start = g0 + d * n_per_device
                    end = min(g0 + (d + 1) * n_per_device, g1)
                    actual_len = end - start
                    G_d = -L_aux[:, start:end, :]
                    if actual_len < n_per_device:
                        G_d = jnp.pad(G_d, ((0, 0), (0, n_per_device - actual_len), (0, 0)))
                    sharded_G_list.append(jax.device_put(G_d, devices[d]))
                sharded_G = jax.device_put_sharded(sharded_G_list, devices)
                
                # 3. Shard xi_phi
                sharded_xi_phi_list = []
                for d in range(n_devices):
                    start = g0 + d * n_per_device
                    end = min(g0 + (d + 1) * n_per_device, g1)
                    actual_len = end - start
                    
                    if self.xi_phi is not None:
                        xi_phi_d = self.xi_phi[:, start:end]
                    else:
                        xi_phi_d = xi_phi_ds[:, start:end]
                        
                    if actual_len < n_per_device:
                        xi_phi_d = np.pad(xi_phi_d, ((0, 0), (0, n_per_device - actual_len)))
                    sharded_xi_phi_list.append(jax.device_put(xi_phi_d, devices[d]))
                sharded_xi_phi = jax.device_put_sharded(sharded_xi_phi_list, devices)
                
                X_rep = pmapped_X(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, jastrow_params, Gb, dm1, phi_isdf, n_orb, L_Q)
                X += np.array(jnp.sum(X_rep, axis=0))
                
                # Explicitly clear memory
                del sharded_G, sharded_xi_phi, sharded_grid, sharded_weights, X_rep
                gc.collect()
                
        finally:
            if f_xi: f_xi.close()
            
        return X


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
            # D1_update = einsum('i,ai,di->ad') but use matmul to avoid large intermediate
            # (xi * w).T @ V.T = (N_rank, batch) @ (batch, N_rank) -> (N_rank, N_rank)
            xi_w = xi_batch * w_batch[None, :]  # (N_rank, batch)
            D1_update = jnp.matmul(xi_w, V.T)
            
            # D4 part
            w_tilde = w_batch * jnp.einsum('b,bi->i', Gb, xi_batch)
            G_weighted = G_batch * w_tilde[None, :, None]
            G_weighted_flat = G_weighted.reshape(N_rank, -1)
            D4_update = jnp.dot(G_weighted_flat, G_flat.T)
            
            return D_acc + 2 * D1_update + D4_update, None
        
        D_final, _ = jax.lax.scan(scan_D, jnp.zeros((N_rank, N_rank)), jnp.arange(n_batches))
        return D_final

    def _calc_X_shard(self, jastrow_params, dm1, grid_points, weights, xi_phi, G_shard,
                      Gb, phi, ranges, n_orb, batch_size=1024, L_Q=None):
        """Calculate X kernel for a shard (merged X2, X3_1, X3_2).
        
        Uses low-rank factorization: Q = L_Q @ L_Q.T where L_Q has shape (N_rank, n_orb).
        Instead of: einsum('bra,ac->brc', X, Q) which is O(batch × Nr × N_rank²)
        We compute: (X @ L_Q) @ L_Q.T which is O(batch × Nr × N_rank × n_orb) - ~9x faster!
        """
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
        
        # Helper: compute X @ L_Q @ L_Q.T using two-step matmul (O(N_rank × n_orb) instead of O(N_rank²))
        def apply_Q(X):
            """Apply Q = L_Q @ L_Q.T to X via two matmuls: (X @ L_Q) @ L_Q.T"""
            # X: (batch, Nr, N_rank), L_Q: (N_rank, n_orb)
            # Step 1: (batch, Nr, N_rank) @ (N_rank, n_orb) -> (batch, Nr, n_orb)
            tmp = jnp.einsum('bra,ao->bro', X, L_Q)
            # Step 2: (batch, Nr, n_orb) @ (n_orb, N_rank) -> (batch, Nr, N_rank)
            return jnp.einsum('bro,ao->bra', tmp, L_Q)

        def scan_X(X_acc, i_batch):
            w_batch = jax.lax.dynamic_slice(weights_padded, (i_batch * batch_size,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(xi_padded, (0, i_batch * batch_size), (N_rank, batch_size))
            G_batch = jax.lax.dynamic_slice(G_padded, (0, i_batch * batch_size, 0), (N_rank, batch_size, 3))
            
            xi_T = xi_batch.T  # (batch, N_rank)
            
            # Precompute k-independent terms ONCE per batch (these don't depend on G_k)
            phi_s_xi = phi_s[None, :, :] * xi_T[:, None, :]  # (batch, Ns, N_rank)
            phi_r_xi = phi_r[None, :, :] * xi_T[:, None, :]  # (batch, Nr, N_rank)
            tmp_r_xi_Q = apply_Q(phi_r_xi)  # O(batch × Nr × N_rank × n_orb)
            
            # Loop over k to avoid 4D tensor materialization (saves 3x VRAM)
            for k in range(3):
                G_k_T = G_batch[:, :, k].T  # (batch, N_rank)
                
                # phi_r_G: (batch, Nr, N_rank)
                phi_r_G = phi_r[None, :, :] * G_k_T[:, None, :]
                tmp_r_G_Q = apply_Q(phi_r_G)  # O(batch × Nr × N_rank × n_orb)
                
                # phi_s_G: (batch, Ns, N_rank)
                phi_s_G = phi_s[None, :, :] * G_k_T[:, None, :]
                
                # X2: (brs * w).T @ xi_T -> (Nr, Ns, N_rank)
                YZ_k = jnp.einsum('bra,bsa->brs', tmp_r_G_Q, phi_s_G)
                X_acc = X_acc + jnp.matmul((YZ_k * w_batch[:, None, None]).transpose(1, 2, 0), xi_T)
                
                # X3_1
                M_k = jnp.einsum('bra,bsa->brs', tmp_r_G_Q, phi_s_xi)
                X_acc = X_acc + jnp.matmul((M_k * w_batch[:, None, None]).transpose(1, 2, 0), G_k_T)
                
                # X3_2
                N_k = jnp.einsum('bra,bsa->brs', tmp_r_xi_Q, phi_s_G)
                X_acc = X_acc + jnp.matmul((N_k * w_batch[:, None, None]).transpose(1, 2, 0), G_k_T)
            
            return X_acc, None
        
        X_final, _ = jax.lax.scan(scan_X, jnp.zeros((Nr, Ns, N_rank)), jnp.arange(n_batches))
        return X_final



    def get_delta_U(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        """Get delta_U matrix using ISDF with pmap support."""
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
            
        if ranges is None:
            full_slice = slice(None)
            ranges = (full_slice, full_slice, full_slice, full_slice)
            
        start_time = time.perf_counter()
        logger.debug("Starting ISDFXTC.get_delta_U")
        
        # Check if kernels are available
        if self.isdf_kernels is None:
             # Compute kernels on the fly if not available
             logger.warning("ISDF kernels missing in get_delta_U. Computing on-the-fly with orbital batching. "
                             "This might be slow. Consider calling .isdf() first.")
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
        logger.debug(f"ISDFXTC.get_delta_U completed in {total_time:.4f} s")
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

    def get_delta_h(self, jastrow_params, dm1=None, 
                    block_str=None, ranges=None, 
                    orb_block_size=256,
                    batch_size=1000):
        """Get or compute delta_h using ISDF kernels efficiently.
        
        Evaluates $\delta h_{pq} = \sum_{rs} (2 \Delta U_{pqrs} - \Delta U_{psrq}) \gamma_{rs}$
        directly from ISDF kernels D and X.
        
        Note on Symmetry:
        $\Delta U_{pqrs} = - (R_{pqrs} + R_{rspq})$ where $R_{pqrs} = (\phi_p \phi_q | \text{kernel} | \phi_r \phi_s)$.
        Term 1 (J-like): $2 \sum \Delta U_{pqrs} \gamma_{rs} = -2 (J + J_{sym})$.
        Term 2 (K-like): $\sum \Delta U_{psrq} \gamma_{rs} = - (K + K_{sym})$.
        $\delta h = -0.5 * (Term 1 - Term 2) = (J + J_{sym}) - 0.5 (K + K_{sym})$.
        
        $J$ involves $R_{pqrs}$, $J_{sym}$ involves $R_{rspq}$.
        $K$ involves $R_{psrq}$, $K_{sym}$ involves $R_{rqps}$.
        """
        logger.debug("Starting ISDFXTC.get_delta_h")
        start_time = time.perf_counter()
        
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Ensure kernels are available
        if self.isdf_kernels is None:
             logger.warning("ISDF kernels missing in get_delta_h. Computing on-the-fly.")
             kernels = self.compute_delta_u_kernels(jastrow_params, batch_size)
        else:
             kernels = self.isdf_kernels
             
        D = kernels['D']
        X = kernels['X']
        phi = self.phi_isdf
        
        slice_p = slice(None)
        slice_q = slice(None)
        if ranges is not None:
             slice_p, slice_q = ranges[0], ranges[1]
             
        # Precompute common intermediates
        # Gb = sum_{rs} phi_rb phi_sb dm_rs
        Gb = jnp.einsum('rb,sb,rs->b', phi, phi, dm1)
        
        # P_phi = sum_{rs} phi_sa dm_rs phi_rb = (phi.T @ dm1 @ phi)
        P_phi = jnp.linalg.multi_dot([phi.T, dm1, phi])
        
        # phi_tilde = dm1 @ phi
        phi_tilde = jnp.dot(dm1, phi)
        
        # wc = sum_{rs} X_rsc dm_rs
        # Y_all = sum_r X_rqc phi_tilde_rc
        
        # Check if X is HDF5 dataset
        is_hdf5 = isinstance(X, (h5py.Dataset, h5py.File)) # or checking attribute?
        # A file object wouldn't be passed, but let's be safe.
        # Actually kernels['X'] is dataset or array.
        
        wc = jnp.zeros((phi.shape[1],)) # (N_rank,)
        Y_all = jnp.zeros((self.n_orb, phi.shape[1])) # (N_orb, N_rank)
        
        if is_hdf5:
            # Process strictly in chunks to respect memory
            # For J_X_sym, we need X[p,q] which is a slice. If slice is small, h5py handles it.
            # But for Y_all and wc, we need full contraction.
            logger.debug("  Streaming X in chunks from HDF5")
            chunk_size = orb_block_size # Adjust based on memory
            for i in range(0, self.n_orb, chunk_size):
                logger.debug(f"  Processing chunk {i}, {i*chunk_size}-{i*chunk_size+chunk_size} out of {self.n_orb}")
                sl = slice(i, min(i+chunk_size, self.n_orb))
                X_chunk = X[sl] # (chunk, N, N_rank) -> Numpy array
                
                # Update wc
                # wc += sum_{r_chunk, s} X[r,s,c] * dm1[r,s]
                # dm1 slice: dm1[sl, :]
                wc += jnp.einsum('rsc,rs->c', X_chunk, dm1[sl])
                
                # Update Y_all
                # Y_all[q, c] += sum_{r_chunk} X[r,q,c] * phi_tilde[r,c]
                # phi_tilde slice: phi_tilde[sl]
                Y_all += jnp.einsum('rqc,rc->qc', X_chunk, phi_tilde[sl])
                
        else:
            # In-memory array
            wc = jnp.einsum('rsc,rs->c', X, dm1)
            Y_all = jnp.einsum('rqc,rc->qc', X, phi_tilde)
        
        # Sliced inputs
        phi_p = phi[slice_p]
        phi_q = phi[slice_q]
        Y_p = Y_all[slice_p]
        Y_q = Y_all[slice_q]
        
        # J terms
        # J_D from R_{pqrs}: sum_{ad} phi_pa phi_qa D_{ad} G_d
        # J_D_sym from R_{rspq}: sum_{ad} phi_pa phi_qa D_{da} G_d = sum_{ad} phi_pa phi_qa D.T_{ad} G_d
        # Total J_D contrib: phi_p * ((D + D.T) @ G) * phi_q
        D_sym = D + D.T
        tmp_a = jnp.dot(D_sym, Gb)
        J_D_total = jnp.dot(phi_p * tmp_a[None, :], phi_q.T)
        
        # J_X: - sum phi_p phi_q w_c
        J_X = - jnp.dot(phi_p * wc[None, :], phi_q.T)
        
        # J_X_sym: - sum X_pq G_c
        # Need X[p, q]
        if is_hdf5:
            # If slices are full, this might be big. But usually p,q are ranges (blocks).
            # H5py slicing returns numpy array.
            X_pq = X[slice_p, slice_q] 
        else:
            X_pq = X[slice_p, slice_q]
            
        J_X_sym = - jnp.einsum('pqc,c->pq', X_pq, Gb)
        
        # J_total = J_D_total + J_X + J_X_sym
        J_total = J_D_total + J_X + J_X_sym
        
        # K terms
        # K_D from R_{psrq}: sum (D * P)_{ad} phi_pa phi_qd
        # K_D_sym from R_{rqps}: sum (D * P)_{da} phi_pa phi_qd = sum (D * P).T_{ad} phi_pa phi_qd
        # Total K_D contrib: phi_p @ (DP + DP.T) @ phi_q.T
        DP = D * P_phi
        DP_sym = DP + DP.T
        K_D_total = jnp.linalg.multi_dot([phi_p, DP_sym, phi_q.T])
        
        # K_X_1: - sum phi_p Y_q
        K_X_1 = - jnp.dot(phi_p, Y_q.T)
        
        # K_X_2: - sum Y_p phi_q
        K_X_2 = - jnp.dot(Y_p, phi_q.T)
        
        # K_total = K_D_total + K_X_1 + K_X_2
        K_total = K_D_total + K_X_1 + K_X_2
        
        # delta_h = J_total - 0.5 * K_total
        delta_h = J_total - 0.5 * K_total
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"ISDFXTC.get_delta_h completed in {total_time:.4f} s")
        return delta_h

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
    

