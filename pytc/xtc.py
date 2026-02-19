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

# ---------------------------------------------------------------------------
# Host-side cache for X orbital slices read from HDF5.
#
# Within each CCSD phase (ovvv / vovv / vvvv), every block iteration calls
# _contract_delta_U_kernels with the *same* (slice_r, slice_s) — only the
# (p, q) orbital indices vary.  Caching the last-read X slice avoids
# re-reading tens of GB from HDF5 per block.
#
# The cache holds at most one slice.  When the phase changes (and the
# (slice_r, slice_s) key changes), the old slice is replaced.
# ---------------------------------------------------------------------------
_X_HDF5_CACHE: dict = {"key": None, "data": None}


def _read_X_slice(X, slice_r, slice_s):
    """Read ``X[slice_r, slice_s]``, reusing a host-side cache when possible.

    If *X* is an HDF5 dataset and the requested slice matches the
    previous call, the cached numpy array is returned directly — no I/O.
    """
    if isinstance(X, h5py.Dataset):
        key = (id(X), _slice_key(slice_r), _slice_key(slice_s))
        if _X_HDF5_CACHE["key"] == key:
            logger.debug("X slice cache HIT  (%s, %s)", slice_r, slice_s)
            return _X_HDF5_CACHE["data"]
        logger.debug("X slice cache MISS (%s, %s) — reading from HDF5", slice_r, slice_s)
        data = X[slice_r, slice_s]
        _X_HDF5_CACHE["key"] = key
        _X_HDF5_CACHE["data"] = data
        return data
    # In-memory array: just slice directly.
    return X[slice_r, slice_s]


def invalidate_X_cache():
    """Explicitly free the cached X slice (e.g., at end of CCSD iteration)."""
    _X_HDF5_CACHE["key"] = None
    _X_HDF5_CACHE["data"] = None


def _slice_key(sl):
    """Hashable representation of a slice or index array."""
    if isinstance(sl, slice):
        return ("slice", sl.start, sl.stop, sl.step)
    return ("idx", tuple(np.asarray(sl).ravel()))


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

    def get_delta_h(self, jastrow_params, dm1=None, block_str=None, ranges=None, orb_block_size=None, batch_size=1000):
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
        
        # Accumulate on host to avoid holding two output-sized GPU tensors.
        # ISDFTC.get_2b already returns via host internally.
        tc_result = super().get_2b(jastrow_params, ranges=ranges)
        result_np = np.array(tc_result)  # writable host copy
        del tc_result
        
        delta_U = self.get_delta_U(jastrow_params, dm1, ranges=ranges, batch_size=batch_size)
        result_np += np.asarray(delta_U)
        del delta_U
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"XTC.get_2b completed in {time.perf_counter() - start_time:.4f} s")
        return jnp.asarray(result_np)

    def get_const(self, jastrow_params, dm1=None, delta_h=None):
        """Compute constant contribution."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        if delta_h is None:
            delta_h = self.get_delta_h(jastrow_params, dm1)
        logger.debug("Starting XTC.get_const")
        start_time = time.perf_counter()
        const = -2/3 * jnp.einsum('qp,pq->', delta_h, dm1)
        const += self.energy_nuc
        logger.debug(f"XTC.get_const completed in {time.perf_counter() - start_time:.4f} s")
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


@partial(jax.jit, static_argnums=(6,))
def _contract_delta_U_kernels_jit(D, X_sliced, phi_p, phi_q, phi_r, phi_s,
                                   rank_block_size=128):
    """JITted version of Delta U contraction.
    
    Args:
        rank_block_size: Block size for scanning the ISDF rank dimension.
            This is a static argument — JAX recompiles if it changes.
    """
    
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_rank_D = D.shape[0] 
    N_rank_X = X_sliced.shape[2]
    
    # Term 1 & 4: T_D = sum_{a,d} (phi_p*phi_q)_a * D[a,d] * (phi_r*phi_s)_d
    # Scan over blocks of d (second index of D)
    
    # Pad D
    padded_rank_D = ((N_rank_D + rank_block_size - 1) // rank_block_size) * rank_block_size
    pad_width_D = padded_rank_D - N_rank_D
    D_padded = jnp.pad(D, ((0, 0), (0, pad_width_D)))
    
    # Pad phi_r and phi_s (associated with index d)
    phi_r_padded = jnp.pad(phi_r, ((0, 0), (0, pad_width_D)))
    phi_s_padded = jnp.pad(phi_s, ((0, 0), (0, pad_width_D)))
    
    n_blocks_D = padded_rank_D // rank_block_size
    
    # Reshape for scan
    # D: (N_rank, n_blocks, block) -> (n_blocks, N_rank, block)
    D_scannable = D_padded.reshape(N_rank_D, n_blocks_D, rank_block_size).transpose(1, 0, 2)
    # phi_r/s: (N, n_blocks, block) -> (n_blocks, N, block)
    phi_r_scannable = phi_r_padded.reshape(Nr, n_blocks_D, rank_block_size).transpose(1, 0, 2)
    phi_s_scannable = phi_s_padded.reshape(Ns, n_blocks_D, rank_block_size).transpose(1, 0, 2)

    def scan_d_block(carry, args):
        D_block, phi_r_block, phi_s_block = args
        # D_block: (N_rank_D, block)
        
        # intermediate V[p,q,d_local] = sum_a (phi_p[p,a] * phi_q[q,a]) * D_block[a, d_local]
        # V[p,q,d'] = sum_a phi_p[p,a] * W[a, d', q]
        # W[a, d', q] = phi_q[q,a] * D_block[a, d']
        W = D_block[:, :, None] * phi_q.T[:, None, :] # (a, d', 1) * (a, 1, q) -> (a, d', q)
        W_flat = W.reshape(N_rank_D, rank_block_size * Nq)
        
        V_flat = jnp.matmul(phi_p, W_flat) # (p, a) @ (a, d'q) -> (p, d'q)
        V_block = V_flat.reshape(Np, rank_block_size, Nq)
        V_block = jnp.transpose(V_block, (0, 2, 1)) # (p, q, d')
        
        # C_rs[r, s, d_local]
        C_rs = phi_r_block[:, None, :] * phi_s_block[None, :, :]
        
        # Contract
        contribution = jnp.einsum('pqd,rsd->pqrs', V_block, C_rs)
        return carry + contribution, None

    term_d_init = jnp.zeros((Np, Nq, Nr, Ns))
    term_d, _ = jax.lax.scan(scan_d_block, term_d_init, (D_scannable, phi_r_scannable, phi_s_scannable))
    
    # Term 2 & 3: T_X = - sum_c (phi_p*phi_q)_c * X[r,s,c]
    # Scan over blocks of c (rank index of X)
    
    # Pad X
    padded_rank_X = ((N_rank_X + rank_block_size - 1) // rank_block_size) * rank_block_size
    pad_width_X = padded_rank_X - N_rank_X
    # X is (Nr, Ns, c)
    X_padded = jnp.pad(X_sliced, ((0,0), (0,0), (0, pad_width_X)))
    
    # Pad phi_p and phi_q (associated with index c/a)
    phi_p_padded = jnp.pad(phi_p, ((0, 0), (0, pad_width_X)))
    phi_q_padded = jnp.pad(phi_q, ((0, 0), (0, pad_width_X)))
    
    n_blocks_X = padded_rank_X // rank_block_size
    
    # Reshape
    # X: (Nr, Ns, n_blocks, block) -> (n_blocks, Nr, Ns, block)
    X_scannable = X_padded.reshape(Nr, Ns, n_blocks_X, rank_block_size).transpose(2, 0, 1, 3)
    phi_p_scannable = phi_p_padded.reshape(Np, n_blocks_X, rank_block_size).transpose(1, 0, 2)
    phi_q_scannable = phi_q_padded.reshape(Nq, n_blocks_X, rank_block_size).transpose(1, 0, 2)
    
    def scan_c_block(carry, args):
        X_block, phi_p_block, phi_q_block = args
        # X_block: (Nr, Ns, block)
        
        # C_pq[p, q, c_local]
        C_pq = phi_p_block[:, None, :] * phi_q_block[None, :, :] # (Np, Nq, block)
        
        # Contract: - sum_c C_pq * X_block
        contribution = -jnp.einsum('pqc,rsc->pqrs', C_pq, X_block)
        return carry + contribution, None

    term_x_init = jnp.zeros((Np, Nq, Nr, Ns))
    term_x, _ = jax.lax.scan(scan_c_block, term_x_init, (X_scannable, phi_p_scannable, phi_q_scannable))

    return term_d + term_x


@jax.jit
def _delta_h_jk_terms(D, Gb, P_phi, phi_p, phi_q, Y_p, Y_q, wc,
                       J_D_total, J_X, J_X_sym):
    """JIT-compiled J/K algebra for get_delta_h (avoids re-tracing each call)."""
    J_total = J_D_total + J_X + J_X_sym

    DP = D * P_phi
    DP_sym = DP + DP.T
    K_D_total = jnp.linalg.multi_dot([phi_p, DP_sym, phi_q.T])
    K_X_1 = -jnp.dot(phi_p, Y_q.T)
    K_X_2 = -jnp.dot(Y_p, phi_q.T)
    K_total = K_D_total + K_X_1 + K_X_2

    return J_total - 0.5 * K_total


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
                    logger.info(f"  Loading D with shape: {f['D'].shape} on host RAM")
                    kernels['D'] = f['D'][:]
                    if self.is_incore:
                        logger.debug("incore mode: Loading X with shape: {f['X'].shape} on host RAM")
                        kernels['X'] = f['X'][:]
                        f.close()
                    else:
                        # Stream X from file. 
                        # Return the dataset object directly. 
                        # Do NOT close 'f' here; the dataset object keeps the file open.
                        logger.debug(f"out-of-core mode: Streaming X from file. X shape: {f['X'].shape}")
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
            # If L_aux is a dataset from the same file, we must close the read-only handle 
            # and reopen in 'a' mode to write D and X, while keeping L_aux streaming.
            f = None
            if isinstance(L_aux, h5py.Dataset):
                # Check if it's the same file. Use realpath to be safe.
                try:
                    l_aux_path = os.path.abspath(L_aux.file.filename)
                    target_path = os.path.abspath(save_path)
                    if l_aux_path == target_path:
                        logger.info("  L_aux is from target file. Switching handle to read-write for streaming...")
                        ds_name = L_aux.name
                        if L_aux.file: L_aux.file.close()
                        f = h5py.File(save_path, 'a')
                        L_aux = f[ds_name] # Re-bind L_aux
                except Exception as e:
                    logger.warning(f"  Could not check L_aux file path: {e}")
            
            if f is None:
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

        from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read

        def _prepare_D_block(g0_loc):
            """Prepare sharded data for one D-kernel grid block (background-safe)."""
            g1_loc = min(g0_loc + host_grid_block_size, n_grid)
            n_blk = g1_loc - g0_loc
            rem = n_blk % n_devices
            pad = (n_devices - rem) if rem != 0 else 0
            n_blk_p = n_blk + pad
            n_per_dev = n_blk_p // n_devices

            gb = np.asarray(self.grid_points[g0_loc:g1_loc])
            wb = np.asarray(self.weights[g0_loc:g1_loc])
            if pad > 0:
                gb = np.pad(gb, ((0, pad), (0, 0)))
                wb = np.pad(wb, ((0, pad),))
            s_grid = gb.reshape(n_devices, n_per_dev, 3)
            s_weights = wb.reshape(n_devices, n_per_dev)

            G_list = []
            xi_list = []
            for d in range(n_devices):
                start = g0_loc + d * n_per_dev
                end = min(g0_loc + (d + 1) * n_per_dev, g1_loc)
                alen = end - start
                G_d = -safe_hdf5_read(L_aux, (slice(None), slice(start, end), slice(None)))
                if alen < n_per_dev:
                    G_d = np.pad(G_d, ((0, 0), (0, n_per_dev - alen), (0, 0)))
                G_list.append(jax.device_put(G_d, devices[d]))
                if self.xi_phi is not None:
                    xp = safe_hdf5_read(self.xi_phi, (slice(None), slice(start, end)))
                else:
                    xp = safe_hdf5_read(xi_phi_ds, (slice(None), slice(start, end)))
                if alen < n_per_dev:
                    xp = np.pad(xp, ((0, 0), (0, n_per_dev - alen)))
                xi_list.append(jax.device_put(xp, devices[d]))
            s_G = jax.device_put_sharded(G_list, devices)
            s_xi = jax.device_put_sharded(xi_list, devices)
            return s_grid, s_weights, s_G, s_xi

        try:
            pending_D = None
            for g0 in range(0, n_grid, host_grid_block_size):
                g1 = min(g0 + host_grid_block_size, n_grid)
                logger.debug(f"_compute_D_kernel: Processing grid block [{g0}:{g1}]...")

                if pending_D is not None:
                    sharded_grid, sharded_weights, sharded_G, sharded_xi_phi = await_read(pending_D)
                    pending_D = None
                else:
                    sharded_grid, sharded_weights, sharded_G, sharded_xi_phi = _prepare_D_block(g0)
                
                D_rep = pmapped_D(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, jastrow_params, Gb, dm1, phi_isdf, n_orb)

                # While GPU runs pmap, prefetch next block
                next_g0 = g0 + host_grid_block_size
                if next_g0 < n_grid:
                    pending_D = async_read(lambda _g=next_g0: _prepare_D_block(_g))

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

        from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read

        def _prepare_X_block(g0_loc):
            """Prepare sharded data for one X-kernel grid block (background-safe)."""
            g1_loc = min(g0_loc + host_grid_block_size, n_grid)
            n_blk = g1_loc - g0_loc
            rem = n_blk % n_devices
            pad = (n_devices - rem) if rem != 0 else 0
            n_blk_p = n_blk + pad
            n_per_dev = n_blk_p // n_devices

            gb = np.asarray(self.grid_points[g0_loc:g1_loc])
            wb = np.asarray(self.weights[g0_loc:g1_loc])
            if pad > 0:
                gb = np.pad(gb, ((0, pad), (0, 0)))
                wb = np.pad(wb, ((0, pad),))
            s_grid = gb.reshape(n_devices, n_per_dev, 3)
            s_weights = wb.reshape(n_devices, n_per_dev)

            G_list = []
            xi_list = []
            for d in range(n_devices):
                start = g0_loc + d * n_per_dev
                end = min(g0_loc + (d + 1) * n_per_dev, g1_loc)
                alen = end - start
                G_d = -safe_hdf5_read(L_aux, (slice(None), slice(start, end), slice(None)))
                if alen < n_per_dev:
                    G_d = np.pad(G_d, ((0, 0), (0, n_per_dev - alen), (0, 0)))
                G_list.append(jax.device_put(G_d, devices[d]))
                if self.xi_phi is not None:
                    xp = safe_hdf5_read(self.xi_phi, (slice(None), slice(start, end)))
                else:
                    xp = safe_hdf5_read(xi_phi_ds, (slice(None), slice(start, end)))
                if alen < n_per_dev:
                    xp = np.pad(xp, ((0, 0), (0, n_per_dev - alen)))
                xi_list.append(jax.device_put(xp, devices[d]))
            s_G = jax.device_put_sharded(G_list, devices)
            s_xi = jax.device_put_sharded(xi_list, devices)
            return s_grid, s_weights, s_G, s_xi

        try:
            pending_X = None
            for g0 in range(0, n_grid, host_grid_block_size):
                g1 = min(g0 + host_grid_block_size, n_grid)
                logger.debug(f"_compute_X_kernel: Processing grid block [{g0}:{g1}]...")

                if pending_X is not None:
                    sharded_grid, sharded_weights, sharded_G, sharded_xi_phi = await_read(pending_X)
                    pending_X = None
                else:
                    sharded_grid, sharded_weights, sharded_G, sharded_xi_phi = _prepare_X_block(g0)
                
                X_rep = pmapped_X(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, jastrow_params, Gb, dm1, phi_isdf, n_orb, L_Q)

                # While GPU runs pmap, prefetch next block
                next_g0 = g0 + host_grid_block_size
                if next_g0 < n_grid:
                    pending_X = async_read(lambda _g=next_g0: _prepare_X_block(_g))

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
        
        # Helper: compute P = (phi * vec_T) @ L_Q using chunking over Rank to avoid OOM
        def compute_proj(phi, vec_T, L_Q, chunk_size=2048):
            """Compute P[b,r,o] = sum_a phi[r,a] * vec_T[b,a] * L_Q[a,o]"""
            Ns = phi.shape[0]
            Nb = vec_T.shape[0]
            No = L_Q.shape[1]
            Na = L_Q.shape[0]
            
            num_chunks = (Na + chunk_size - 1) // chunk_size
            
            # Pad dimensions to be multiples of chunk_size
            pad_len = num_chunks * chunk_size - Na
            if pad_len > 0:
                phi_p = jnp.pad(phi, ((0,0), (0, pad_len)))
                vec_p = jnp.pad(vec_T, ((0,0), (0, pad_len)))
                lq_p = jnp.pad(L_Q, ((0, pad_len), (0,0)))
            else:
                phi_p, vec_p, lq_p = phi, vec_T, L_Q
                
            def body_fn(carry, i):
                start = i * chunk_size
                p_c = jax.lax.dynamic_slice(phi_p, (0, start), (Ns, chunk_size))
                v_c = jax.lax.dynamic_slice(vec_p, (0, start), (Nb, chunk_size))
                l_c = jax.lax.dynamic_slice(lq_p, (start, 0), (chunk_size, No))
                
                # (batch, Nr, chunk) * (chunk, n_orb) -> (batch, Nr, n_orb)
                # v_c[:, None, :] broadcasts to (batch, 1, chunk)
                # p_c[None, :, :] broadcasts to (1, Nr, chunk)
                # product is (batch, Nr, chunk)
                term = jnp.matmul(p_c[None, :, :] * v_c[:, None, :], l_c)
                return carry + term, None
            
            res, _ = jax.lax.scan(body_fn, jnp.zeros((Nb, Ns, No)), jnp.arange(num_chunks))
            return res

        def scan_X(X_acc, i_batch):
            w_batch = jax.lax.dynamic_slice(weights_padded, (i_batch * batch_size,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(xi_padded, (0, i_batch * batch_size), (N_rank, batch_size))
            G_batch = jax.lax.dynamic_slice(G_padded, (0, i_batch * batch_size, 0), (N_rank, batch_size, 3))
            
            xi_T = xi_batch.T  # (batch, N_rank)
            
            # Precompute projections for xi (phi_s_xi and phi_r_xi terms)
            # P_s_xi = (phi_s * xi) @ L_Q -> (batch, Ns, N_orb)
            P_s_xi = compute_proj(phi_s, xi_T, L_Q)
            P_r_xi = compute_proj(phi_r, xi_T, L_Q)
            
            # Loop over k (x,y,z components)
            for k in range(3):
                G_k_T = G_batch[:, :, k].T  # (batch, N_rank)
                
                # Projections for G_k
                P_r_G = compute_proj(phi_r, G_k_T, L_Q)
                P_s_G = compute_proj(phi_s, G_k_T, L_Q)
                
                # Reconstruct X contributions using low-rank outer products
                # Original: YZ_k = (phi_r_G_Q) @ (phi_s_G).T = (P_r_G @ L_Q.T) @ (P_s_G @ L_Q.T).T
                # Wait, original was: YZ_k = einsum('bra,bsa->brs', tmp_r_G_Q, phi_s_G)
                # where tmp_r_G_Q = (phi_r * G) @ L_Q @ L_Q.T = P_r_G @ L_Q.T
                # and phi_s_G = phi_s * G
                # So YZ_k = (P_r_G @ L_Q.T) @ (phi_s * G).T
                #         = P_r_G @ ( (phi_s * G) @ L_Q ).T
                #         = P_r_G @ P_s_G.T
                
                # X2: YZ_k = P_r_G @ P_s_G.T
                # einsum('bro,bso->brs', P_r_G, P_s_G)
                YZ_k = jnp.matmul(P_r_G, P_s_G.transpose(0, 2, 1))
                X_acc = X_acc + jnp.matmul((YZ_k * w_batch[:, None, None]).transpose(1, 2, 0), xi_T)
                
                # X3_1: M_k = (P_r_G @ L_Q.T) @ (phi_s * xi).T
                #           = P_r_G @ P_s_xi.T
                M_k = jnp.matmul(P_r_G, P_s_xi.transpose(0, 2, 1))
                X_acc = X_acc + jnp.matmul((M_k * w_batch[:, None, None]).transpose(1, 2, 0), G_k_T)
                
                # X3_2: N_k = (P_r_xi @ L_Q.T) @ (phi_s * G).T
                #           = P_r_xi @ P_s_G.T
                N_k = jnp.matmul(P_r_xi, P_s_G.transpose(0, 2, 1))
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
        
        # Symmetrize the result: final = -(result + T_full.transpose(2,3,0,1))
        slice_p, slice_q, slice_r, slice_s = ranges
        
        if slice_p == slice_r and slice_q == slice_s:
            result = -(result + result.transpose(2, 3, 0, 1))
        else:
            # Transfer direct block to host, then sub-chunk the transpose
            # block on GPU → accumulate on host.  Avoids holding two full
            # output-sized tensors on GPU simultaneously.
            result_np = -np.asarray(result)
            del result
            nmo = self.phi_isdf.shape[0]
            r_start = slice_r.start if slice_r.start is not None else 0
            r_stop = slice_r.stop if slice_r.stop is not None else nmo
            r_len = r_stop - r_start
            n_sub = 2
            chunk_size = max(1, (r_len + n_sub - 1) // n_sub)
            for i0 in range(0, r_len, chunk_size):
                i1 = min(i0 + chunk_size, r_len)
                sub_ranges = (slice(r_start + i0, r_start + i1),
                              slice_s, slice_p, slice_q)
                tmp = self._contract_delta_U_kernels(kernels, sub_ranges)
                chunk_np = np.asarray(tmp.transpose(2, 3, 0, 1))
                del tmp
                result_np[:, :, i0:i1, :] -= chunk_np
                del chunk_np
            result = jnp.asarray(result_np)

        total_time = time.perf_counter() - start_time
        logger.debug(f"ISDFXTC.get_delta_U completed in {total_time:.4f} s")
        return result


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
             
        Gb = jnp.einsum('rb,sb,rs->b', phi, phi, dm1)
        P_phi = jnp.linalg.multi_dot([phi.T, dm1, phi])
        phi_tilde = jnp.dot(dm1, phi)
        
        # Check if X is HDF5 dataset
        is_hdf5 = isinstance(X, (h5py.Dataset, h5py.File))
        
        wc = jnp.zeros((phi.shape[1],)) # (N_rank,)
        Y_all = jnp.zeros((self.n_orb, phi.shape[1])) # (N_orb, N_rank)
        
        if is_hdf5:
            # Process strictly in chunks to respect memory
            logger.debug("Streaming X in chunks from HDF5")
            chunk_size = orb_block_size # Adjust based on memory
            from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read

            pending_h = None
            for i in range(0, self.n_orb, chunk_size):
                start = i
                stop = min(i + chunk_size, self.n_orb)
                sl = slice(start, stop)
                logger.debug(f"Processing slice {start}-{stop}")

                if pending_h is not None:
                    X_chunk = await_read(pending_h)
                    pending_h = None
                else:
                    X_chunk = safe_hdf5_read(X, sl)

                # Prefetch next chunk while einsum runs
                next_start = stop
                if next_start < self.n_orb:
                    next_stop = min(next_start + chunk_size, self.n_orb)
                    pending_h = async_read(lambda _s=slice(next_start, next_stop): safe_hdf5_read(X, _s))
                
                wc += jnp.einsum('rsc,rs->c', X_chunk, dm1[sl])
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
        D_sym = D + D.T
        tmp_a = jnp.dot(D_sym, Gb)
        J_D_total = jnp.dot(phi_p * tmp_a[None, :], phi_q.T)
        
        # J_X: - sum phi_p phi_q w_c
        J_X = - jnp.dot(phi_p * wc[None, :], phi_q.T)
        
        # J_X_sym: - sum X_pq G_c
        if is_hdf5:
            start_p, stop_p, step_p = slice_p.indices(self.n_orb)
            start_q, stop_q, step_q = slice_q.indices(self.n_orb)
            
            Np = (stop_p - start_p + step_p - 1) // step_p
            Nq = (stop_q - start_q + step_q - 1) // step_q
            
            J_X_sym_blocks = []
            
            # Iterate p in chunks relative to result
            from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read
            pending_jx = None
            for i in range(0, Np, orb_block_size):
                i_end = min(i + orb_block_size, Np)
                p_abs_start = start_p + i * step_p
                p_abs_stop = start_p + i_end * step_p
                p_abs_slice = slice(p_abs_start, p_abs_stop, step_p)
                
                # Load X block (prefetched or inline)
                if pending_jx is not None:
                    X_chunk = await_read(pending_jx)
                    pending_jx = None
                else:
                    X_chunk = safe_hdf5_read(X, (p_abs_slice, slice_q))

                block_res = - jnp.einsum('pqc,c->pq', X_chunk, Gb)

                # Prefetch next X block while einsum runs
                next_i = i + orb_block_size
                if next_i < Np:
                    ni_end = min(next_i + orb_block_size, Np)
                    np_start = start_p + next_i * step_p
                    np_stop = start_p + ni_end * step_p
                    n_slice = slice(np_start, np_stop, step_p)
                    pending_jx = async_read(lambda _sl=n_slice: safe_hdf5_read(X, (_sl, slice_q)))

                J_X_sym_blocks.append(block_res)
                
            J_X_sym = jnp.concatenate(J_X_sym_blocks, axis=0)

        else:
            X_pq = X[slice_p, slice_q]
            J_X_sym = - jnp.einsum('pqc,c->pq', X_pq, Gb)
        
        delta_h = _delta_h_jk_terms(D, Gb, P_phi, phi_p, phi_q,
                                    Y_p, Y_q, wc, J_D_total, J_X, J_X_sym)
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"ISDFXTC.get_delta_h completed in {total_time:.4f} s")
        return delta_h

    def _contract_delta_U_kernels(self, kernels, ranges):
        """Contract precomputed kernels to get Delta U block."""
        D = kernels['D']
        X = kernels['X']
        
        slice_p, slice_q, slice_r, slice_s = ranges
        
        # Helper to get length and indices
        def get_info(sl, total):
            if isinstance(sl, slice):
                idx = np.arange(*sl.indices(total))
            else:
                idx = np.array(sl)
            return len(idx), idx

        Np, _ = get_info(slice_p, self.n_orb)
        Nq, _ = get_info(slice_q, self.n_orb)
        Nr, r_idx = get_info(slice_r, self.n_orb)
        Ns, s_idx = get_info(slice_s, self.n_orb)
        N_rank = X.shape[2]
        
        # Check size of X_sliced vs available GPU memory
        from pytc.utils.gpu_memory import adaptive_rank_block_size, _get_gpu_free_bytes
        gpu_free_bytes = _get_gpu_free_bytes()
        available_gb = gpu_free_bytes / (1024.0**3)
        
        # Estimate total GPU memory needed for delta_U calculation.
        # _contract_delta_U_kernels_jit runs TWO sequential lax.scans:
        #   1. D-scan: accumulates term_d (Np, Nq, Nr, Ns)
        #   2. X-scan: accumulates term_x, with term_d still alive
        # Peak during X-scan:
        #   X_sliced (JIT input, stays resident) + X_padded (~14% larger copy)
        #   + D (JIT input) + term_d + carry + contribution
        #   ≈ X_sliced * 2 + D + 3 × carry
        # Peak during D-scan:
        #   X_sliced (alive for later) + D + 2 × carry + W + C_rs
        x_sliced_size_gb = (float(Nr) * float(Ns) * float(N_rank) * 8.0) / (1024.0**3)
        d_size_gb = (float(N_rank) * float(N_rank) * 8.0) / (1024.0**3)
        scan_carry_gb = (float(Np) * float(Nq) * float(Nr) * float(Ns) * 8.0) / (1024.0**3)
        # X_sliced input + padded copy ≈ 2× X_sliced
        # + D matrix + 3× carry (term_d + carry + contribution)
        total_needed_gb = x_sliced_size_gb * 2.0 + d_size_gb + 3.0 * scan_carry_gb
        
        # Threshold: use 80% of actually free GPU memory (not budget).
        # This is more accurate than the budget-based estimate since it
        # accounts for pre-allocated tensors (phi_isdf, etc.).
        threshold = available_gb * 0.5
        
        logger.debug(f"delta_U memory estimate: X_sliced={x_sliced_size_gb:.2f} GB, "
                     f"scan_carry={scan_carry_gb:.2f} GB, total={total_needed_gb:.2f} GB "
                     f"(Threshold: {threshold:.2f} GB, dims: Np={Np}, Nq={Nq}, Nr={Nr}, Ns={Ns})")
        
        phi_p = self.phi_isdf[slice_p]
        phi_q = self.phi_isdf[slice_q]
        
        # Use fixed rank_block_size (worst-case over all phases) to avoid
        # JIT recompilation when (Np, Nq) changes across CCSD blocks.
        _rbs = self._get_fixed_rank_block_size()
        if _rbs is None:
            _rbs = adaptive_rank_block_size(
                Np, Nq, N_rank,
                gpu_max_memory_mb=getattr(self, 'gpu_max_memory', None))

        # Read the full X orbital slice into host RAM (cached across
        # repeated calls with the same (slice_r, slice_s) — typical
        # within a CCSD ERI phase).  The chunking path sub-slices
        # from this host array rather than re-reading from HDF5.
        X_full = _read_X_slice(X, slice_r, slice_s)
        
        if total_needed_gb < threshold:
            phi_r = self.phi_isdf[slice_r]
            phi_s = self.phi_isdf[slice_s]
            return _contract_delta_U_kernels_jit(D, X_full, phi_p, phi_q, phi_r, phi_s,
                                                  _rbs)
        
        # Chunking strategy to avoid VRAM exhaustion
        logger.warning(f"  delta_U memory estimate ({total_needed_gb:.2f} GB) exceeds {threshold:.2f} GB limit. Chunking orbital indices.")
        
        # Pre-allocate result on host memory
        result = np.zeros((Np, Nq, Nr, Ns), dtype=np.float64)
        
        # Choose chunk size to keep TOTAL memory (X_chunk + 3× carry) under budget.
        # For r-chunking (reducing Nr to Nr_chunk):
        #   X_chunk = (Nr_chunk, Ns, N_rank)
        #   carry_chunk = (Np, Nq, Nr_chunk, Ns)
        #   Total per Nr_chunk_unit = Ns * N_rank * 8 * 2 (input+padded)
        #                            + 3 * Np * Nq * Ns * 8
        # For s-chunking (reducing Ns to Ns_chunk):
        #   X_chunk = (Nr, Ns_chunk, N_rank)
        #   carry_chunk = (Np, Nq, Nr, Ns_chunk)
        #   Total per Ns_chunk_unit = Nr * N_rank * 8 * 2 (input+padded)
        #                            + 3 * Np * Nq * Nr * 8
        # Target: total + D fits in threshold.
        target_gb = threshold - d_size_gb
        target_gb = max(target_gb, 1.0)  # safety
        
        if Nr >= Ns:
            # Chunk over r
            per_r_unit_gb = (float(Ns) * float(N_rank) * 8.0 * 2.0
                            + 3.0 * float(Np) * float(Nq) * float(Ns) * 8.0) / (1024.0**3)
            max_Nr_chunk = max(1, int(target_gb / per_r_unit_gb)) if per_r_unit_gb > 0 else Nr
            orb_chunk_size = min(max_Nr_chunk, Nr)
            
            chunk_total_gb = orb_chunk_size * per_r_unit_gb + d_size_gb
            logger.debug(f"Chunking over 'r' index. Chunk size: {orb_chunk_size} "
                         f"(est. per chunk: {chunk_total_gb:.2f} GB)")
            
            phi_s = self.phi_isdf[slice_s]

            # Async prefetch: overlap host→device transfer of next chunk
            # with the current GPU JIT computation.
            from pytc.utils.prefetch import async_read, await_read

            def _prepare_r_chunk(i_start):
                """Prepare phi_r_chunk and X_chunk for a given r-index range."""
                ie = min(i_start + orb_chunk_size, Nr)
                alen = ie - i_start
                pr = self.phi_isdf[r_idx[i_start:ie]]
                xc = jnp.asarray(X_full[i_start:ie])
                if alen < orb_chunk_size:
                    pad = orb_chunk_size - alen
                    pr = jnp.pad(pr, ((0, pad), (0, 0)))
                    xc = jnp.pad(xc, ((0, pad), (0, 0), (0, 0)))
                return pr, xc, alen

            # Pre-compute first chunk synchronously
            phi_r_chunk, X_chunk, actual_len = _prepare_r_chunk(0)
            next_future = None

            for i in range(0, Nr, orb_chunk_size):
                # Use the already-prepared arrays
                cur_phi_r = phi_r_chunk
                cur_X = X_chunk
                cur_actual = actual_len

                # Start JIT computation on GPU
                res_chunk = _contract_delta_U_kernels_jit(
                    D, cur_X, phi_p, phi_q, cur_phi_r, phi_s, _rbs)

                # While GPU is busy, prepare the next chunk in background
                next_i = i + orb_chunk_size
                if next_i < Nr:
                    next_future = async_read(_prepare_r_chunk, next_i)

                result[:, :, i:i+cur_actual, :] = np.asarray(res_chunk)[:, :, :cur_actual, :]
                del res_chunk

                # Await next chunk if submitted
                if next_future is not None and next_i < Nr:
                    phi_r_chunk, X_chunk, actual_len = await_read(next_future)
                    next_future = None

                gc.collect()
        else:
            # Chunk over s
            per_s_unit_gb = (float(Nr) * float(N_rank) * 8.0 * 2.0
                            + 3.0 * float(Np) * float(Nq) * float(Nr) * 8.0) / (1024.0**3)
            max_Ns_chunk = max(1, int(target_gb / per_s_unit_gb)) if per_s_unit_gb > 0 else Ns
            orb_chunk_size = min(max_Ns_chunk, Ns)
            
            chunk_total_gb = orb_chunk_size * per_s_unit_gb + d_size_gb
            logger.debug(f"Chunking over 's' index. Chunk size: {orb_chunk_size} "
                         f"(est. per chunk: {chunk_total_gb:.2f} GB)")

            phi_r = self.phi_isdf[slice_r]

            # Async prefetch: overlap host→device transfer of next chunk.
            from pytc.utils.prefetch import async_read, await_read

            def _prepare_s_chunk(i_start):
                """Prepare phi_s_chunk and X_chunk for a given s-index range."""
                ie = min(i_start + orb_chunk_size, Ns)
                alen = ie - i_start
                ps = self.phi_isdf[s_idx[i_start:ie]]
                xc = jnp.asarray(X_full[:, i_start:ie])
                if alen < orb_chunk_size:
                    pad = orb_chunk_size - alen
                    ps = jnp.pad(ps, ((0, pad), (0, 0)))
                    xc = jnp.pad(xc, ((0, 0), (0, pad), (0, 0)))
                return ps, xc, alen

            # Pre-compute first chunk synchronously
            phi_s_chunk, X_chunk, actual_len = _prepare_s_chunk(0)
            next_future = None

            for i in range(0, Ns, orb_chunk_size):
                cur_phi_s = phi_s_chunk
                cur_X = X_chunk
                cur_actual = actual_len

                # Start JIT computation on GPU
                res_chunk = _contract_delta_U_kernels_jit(
                    D, cur_X, phi_p, phi_q, phi_r, cur_phi_s, _rbs)

                # While GPU is busy, prepare the next chunk in background
                next_i = i + orb_chunk_size
                if next_i < Ns:
                    next_future = async_read(_prepare_s_chunk, next_i)

                result[:, :, :, i:i+cur_actual] = np.asarray(res_chunk)[:, :, :, :cur_actual]
                del res_chunk

                # Await next chunk if submitted
                if next_future is not None and next_i < Ns:
                    phi_s_chunk, X_chunk, actual_len = await_read(next_future)
                    next_future = None

                gc.collect()
                
        return jnp.asarray(result)
    

