"""JAX implementation of Transcorrelated method."""

from functools import partial
from typing import Any, Optional
import numpy as np
import logging
import time
import jax
import jax.numpy as jnp
from flax import struct
from pyscf import dft
from . import kmat as kmat_jax

def _compute_2b_shard(phi, grad_phi, grid, weights, jastrow_params, jastrow_factor, ranges, batch_size):
    """Compute K terms for a grid shard (pmapped)."""
    # Unpack ranges (p, q, r, s)
    slice_p, slice_q, slice_r, slice_s = ranges
    
    # Helper to get size
    def get_size(s, size):
        start, stop, step = s.indices(size)
        return (stop - start + (step - 1)) // step
    
    n_orb = phi.shape[0]
    Np = get_size(slice_p, n_orb)
    Nq = get_size(slice_q, n_orb)
    Nr = get_size(slice_r, n_orb)
    Ns = get_size(slice_s, n_orb)
    
    # Compute K1 (nabla on p)
    k1_raw = kmat_jax.calc_K1(
        phi, grad_phi,
        jastrow_factor, jastrow_params,
        grid, weights,
        ranges=ranges,
        batch_size=batch_size
    )
    k1 = k1_raw.reshape(Np, Nq, Nr, Ns)
    
    # Compute K2 (nabla on q)
    if slice_p == slice_q:
        # If p and q ranges are identical, K2 is just K1 with p,q swapped
        k2 = k1.transpose(1, 0, 2, 3)
    else:
        # Must compute explicitly: swap p and q in ranges
        ranges_k2 = (slice_q, slice_p, slice_r, slice_s)
        k2_raw = kmat_jax.calc_K1(
            phi, grad_phi,
            jastrow_factor, jastrow_params,
            grid, weights,
            ranges=ranges_k2,
            batch_size=batch_size
        )
        # Result is (Nq, Np, Nr, Ns), transpose to (Np, Nq, Nr, Ns)
        k2 = k2_raw.reshape(Nq, Np, Nr, Ns).transpose(1, 0, 2, 3)
        
    # Compute K3
    k3_raw = kmat_jax.calc_K3(
        phi, jastrow_factor, jastrow_params,
        grid, weights,
        ranges=ranges,
        batch_size=batch_size
    )
    k3 = k3_raw.reshape(Np, Nq, Nr, Ns)
    
    # Combine: 0.5 * (K1 - K2 + K3)
    result_local = 0.5 * (k1 - k2 + k3)
    
    # Sum results across devices
    result_sum = jax.lax.psum(result_local, axis_name='devices')
    
    return result_sum

@struct.dataclass
class TC:
    """JAX implementation of Transcorrelated method using flax dataclass.
    
    Attributes:
        grid_points: Grid points for numerical integration (N_grid, 3)
        weights: Grid weights (N_grid,)
        phi: Basis functions evaluated on grid (N_orb, N_grid)
        grad_phi: Basis function gradients on grid (N_orb, N_grid, 3)
        n_orb: Number of orbitals (static)
        grid_lvl: Grid level (static)
        jastrow_factor: Jastrow factor instance (PyTree)
        mo_coeff: Molecular orbital coefficients (N_ao, N_orb)
    """
    grid_points: jnp.ndarray
    weights: jnp.ndarray
    phi: jnp.ndarray
    grad_phi: jnp.ndarray
    n_orb: int = struct.field(pytree_node=False)
    grid_lvl: int = struct.field(pytree_node=False)
    jastrow_factor: Any = struct.field(pytree_node=True)
    mo_coeff: jnp.ndarray = struct.field(default=None)
    nocc: int = struct.field(pytree_node=False, default=None)

    @classmethod
    def from_pyscf(cls, mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
        """Initialize TC object from PySCF mean-field object.
        
        Args:
            mf: PySCF mean-field object
            jastrow_factor: JAX Jastrow factor instance
            mo_coeff: Optional molecular orbital coefficients
            grid_lvl: Grid level for numerical integration
            
        Returns:
            TC: Initialized TC object
        """
        mol = mf.mol
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        n_orb = mo_coeff.shape[1]
        nocc = int(np.sum(mf.mo_occ > 0))
        
        # Initialize grid
        grids = dft.gen_grid.Grids(mol)
        grids.level = grid_lvl
        grids.build()
        
        grid_points = jnp.asarray(grids.coords)
        weights = jnp.asarray(grids.weights)
        
        # Evaluate basis on grid
        # Use PySCF to evaluate AOs with numpy arrays
        ao = dft.numint.eval_ao(mol, grids.coords, deriv=1)
        ao_values = ao[0].T  # (N_ao, N_grid)
        ao_gradients = ao[1:4].transpose(2, 1, 0)  # (N_ao, N_grid, 3)
        
        # Transform to MO basis
        mo_values = np.dot(mo_coeff.T, ao_values)
        mo_gradients = np.einsum('ji,jnc->inc', mo_coeff, ao_gradients)
        
        phi = jnp.asarray(mo_values)
        grad_phi = jnp.asarray(mo_gradients)
        
        return cls(
            grid_points=grid_points,
            weights=weights,
            phi=phi,
            grad_phi=grad_phi,
            n_orb=n_orb,
            grid_lvl=grid_lvl,
            jastrow_factor=jastrow_factor,
            mo_coeff=jnp.asarray(mo_coeff),
            nocc=nocc
        )
    
    def _get_block_ranges(self, block_str):
        """Parse block string into slice ranges.
        
        block_str is expected to be in chemists' notation (p, q, r, s),
        where p, q share coordinate 1 and r, s share coordinate 2.
        
        Returns ranges in the order (p, q, r, s) expected by calc_K1.
        """
        if self.nocc is None:
            raise ValueError("nocc must be set to use block_str")
        
        ranges_list = []
        for char in block_str:
            if char == 'o':
                ranges_list.append(slice(0, self.nocc))
            elif char == 'v':
                ranges_list.append(slice(self.nocc, self.n_orb))
            elif char == 'g':
                ranges_list.append(slice(0, self.n_orb))
            else:
                raise ValueError(f"Invalid block character: {char}")
        
        # block_str indices: 0->p, 1->q, 2->r, 3->s
        # calc_K1 expects: (p, q, r, s)
        if len(ranges_list) == 4:
            p = ranges_list[0]
            q = ranges_list[1]
            r = ranges_list[2]
            s = ranges_list[3]
        elif len(ranges_list) == 2:
            p = ranges_list[0]
            q = ranges_list[1]
            r = ranges_list[0]
            s = ranges_list[1]
        else:
            raise ValueError("block_str must have 2 or 4 characters")
        
        return (p, q, r, s)

    def get_2b(self, jastrow_params, block_str=None, ranges=None, batch_size=1000):
        """Calculate TC correction terms (K1 + K2 + K3) with multi-GPU support.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor
            block_str: Optional string specifying the block (e.g. 'oovv')
            ranges: Optional tuple of slices (slice_p, slice_q, slice_r, slice_s)
            
        Returns:
            jnp.ndarray: The TC correction term.
                         If block_str/ranges is provided, returns the raw block (Np, Nr, Nq, Ns).
                         Otherwise, returns the full symmetrized correction (N, N, N, N).
        """
        start_time = time.perf_counter()
        logging.debug("Starting TC.get_2b")
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
        n_devices = jax.local_device_count()
        n_grid = self.grid_points.shape[0]
        
        # Pad grid to be divisible by n_devices
        remainder = n_grid % n_devices
        if remainder != 0:
            padding = n_devices - remainder
            padded_grid_points = jnp.pad(self.grid_points, ((0, padding), (0, 0)))
            padded_weights = jnp.pad(self.weights, ((0, padding),))
            padded_phi = jnp.pad(self.phi, ((0, 0), (0, padding)))
            padded_grad_phi = jnp.pad(self.grad_phi, ((0, 0), (0, padding), (0, 0)))
        else:
            padded_grid_points = self.grid_points
            padded_weights = self.weights
            padded_phi = self.phi
            padded_grad_phi = self.grad_phi
            
        n_grid_padded = padded_grid_points.shape[0]
        n_per_device = n_grid_padded // n_devices
        
        # Shard arrays: (n_devices, n_per_device, ...)
        # grid: (N, 3) -> (n_dev, N_per, 3)
        sharded_grid = padded_grid_points.reshape(n_devices, n_per_device, 3)
        # weights: (N,) -> (n_dev, N_per)
        sharded_weights = padded_weights.reshape(n_devices, n_per_device)
        # phi: (Nb, N) -> (Nb, n_dev, N_per) -> (n_dev, Nb, N_per)
        sharded_phi = padded_phi.reshape(self.n_orb, n_devices, n_per_device).transpose(1, 0, 2)
        # grad_phi: (Nb, N, 3) -> (Nb, n_dev, N_per, 3) -> (n_dev, Nb, N_per, 3)
        sharded_grad_phi = padded_grad_phi.reshape(self.n_orb, n_devices, n_per_device, 3).transpose(1, 0, 2, 3)
        
        # Execute pmap
        pmapped_compute = jax.pmap(
            _compute_2b_shard, 
            axis_name='devices',
            in_axes=(0, 0, 0, 0, None, None, None, None),
            static_broadcasted_argnums=(6, 7)
        )
        
        if ranges is None:
            full_slice = slice(None)
            ranges = (full_slice, full_slice, full_slice, full_slice)
            
        # Compute main block: 0.5 * (K1 - K2 + K3)
        result_sum = pmapped_compute(
            sharded_phi, sharded_grad_phi, sharded_grid, sharded_weights,
            jastrow_params, self.jastrow_factor, ranges, batch_size
        )
        result = result_sum[0]
        
        # Add transpose block: (r, s, p, q)
        # Check if ranges imply symmetry
        slice_p, slice_q, slice_r, slice_s = ranges
        
        if slice_p == slice_r and slice_q == slice_s:
            # Symmetric block (e.g. 'oooo'), just add transpose of result
            result += result.transpose(2, 3, 0, 1)
        else:
            ranges_T = (slice_r, slice_s, slice_p, slice_q)
            
            result_sum_T = pmapped_compute(
                sharded_phi, sharded_grad_phi, sharded_grid, sharded_weights,
                jastrow_params, self.jastrow_factor, ranges_T, batch_size
            )
            result_T = result_sum_T[0]
            
            result += result_T.transpose(2, 3, 0, 1)
        
        total_time = time.perf_counter() - start_time
        logging.debug(f"TC.get_2b completed in {total_time:.4f} s")
        return -result

    def get_1b_fock(self, jastrow_params, dm1=None):
        """Get one-body Fock matrix correction (AO basis)."""
        return jnp.zeros((self.n_orb, self.n_orb))

    def get_2b_fock(self, jastrow_params, dm1, T=None):
        """Get 2-body Fock matrix correction.
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix
            T: Optional cached 2-body tensor (N, N, N, N).
        """
        if T is None:
            k_2b = self.get_2b(jastrow_params)
            T = k_2b
            
        # Coulomb-like contribution
        # \sum_{r,s} T_{pqrs} P_{rs}
        J_mat = jnp.einsum('pqrs,rs->pq', T, dm1)
        # Exchange-like contribution
        # \sum_{r,s} T_{psrq} P_{rs}
        K_mat = jnp.einsum('psrq,rs->pq', T, dm1)
        
        return J_mat - 0.5 * K_mat


    def get_3b_fock(self, jastrow_params, dm1):
        """Get 3-body Fock matrix correction (on-the-fly).
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis)
            
        Returns:
            Fock matrix contribution (N, N)
        """
        density_g = jnp.einsum('mg,ng,mn->g', self.phi, self.phi, dm1)
        
        N_grid = self.grid_points.shape[0]
        batch_size = 1000
        
        def compute_W_batch(r_batch):
            def inner_scan(carry, chunk_idx):
                start = chunk_idx * batch_size
                end = jnp.minimum(start + batch_size, N_grid)
                
                # Using dynamic_slice
                slice_len = batch_size # Fixed size slice
                r2_chunk = jax.lax.dynamic_slice(self.grid_points, (start, 0), (slice_len, 3))
                w_chunk = jax.lax.dynamic_slice(self.weights, (start,), (slice_len,))
                density_chunk = jax.lax.dynamic_slice(density_g, (start,), (slice_len,))
                
                # grad(r_batch, r2_chunk) -> (B, B_inner, 3)
                grads = self.jastrow_factor.grad_r_batch(r_batch, r2_chunk, jastrow_params)
                
                # sum_j w_j density_j grad_ij
                weighted_grads = grads * (w_chunk * density_chunk)[None, :, None]
                chunk_sum = jnp.sum(weighted_grads, axis=1)
                
                return carry + chunk_sum, None
                
            n_chunks = (N_grid + batch_size - 1) // batch_size
            W_batch, _ = jax.lax.scan(inner_scan, jnp.zeros((r_batch.shape[0], 3)), jnp.arange(n_chunks))
            return W_batch

        # Compute W for all grid points
        # Scan over r_batch
        n_batches = (N_grid + batch_size - 1) // batch_size
        
        # Pad grid arrays to be divisible by batch_size to avoid slicing issues
        padded_size = n_batches * batch_size
        padding = padded_size - N_grid
        if padding > 0:
            grid_padded = jnp.pad(self.grid_points, ((0, padding), (0, 0)))
            weights_padded = jnp.pad(self.weights, ((0, padding),))
            density_g_padded = jnp.pad(density_g, ((0, padding),))
        else:
            grid_padded = self.grid_points
            weights_padded = self.weights
            density_g_padded = density_g
            
        def compute_W_batch_padded(r_batch, grid_p, weights_p, density_p):
            def inner_scan_p(carry, chunk_idx):
                start = chunk_idx * batch_size
                # Fixed size slice on padded arrays
                r2_chunk = jax.lax.dynamic_slice(grid_p, (start, 0), (batch_size, 3))
                w_chunk = jax.lax.dynamic_slice(weights_p, (start,), (batch_size,))
                density_chunk = jax.lax.dynamic_slice(density_p, (start,), (batch_size,))
                
                grads = self.jastrow_factor.grad_r_batch(r_batch, r2_chunk, jastrow_params)
                weighted_grads = grads * (w_chunk * density_chunk)[None, :, None]
                chunk_sum = jnp.sum(weighted_grads, axis=1)
                return carry + chunk_sum, None
            
            n_chunks = (grid_p.shape[0]) // batch_size
            W_batch, _ = jax.lax.scan(inner_scan_p, jnp.zeros((r_batch.shape[0], 3), dtype=jnp.complex128), jnp.arange(n_chunks))
            return W_batch

        def outer_scan(carry, batch_idx):
            start = batch_idx * batch_size
            # Slice from padded grid
            r_batch = jax.lax.dynamic_slice(grid_padded, (start, 0), (batch_size, 3))
            W_batch = compute_W_batch_padded(r_batch, grid_padded, weights_padded, density_g_padded)
            return carry, W_batch 
            
        _, W_all = jax.lax.scan(outer_scan, None, jnp.arange(n_batches))
        W_all = W_all.reshape(-1, 3)[:N_grid] # Flatten and trim padding if any
        
        # 3. Compute V_3b_direct(r) = |W(r)|^2
        V_3b_g = jnp.sum(W_all**2, axis=1) # (N_grid,)
        
        # 4. Integrate to get Fock matrix elements
        # F_mn = \int \phi_m(r) \phi_n(r) V_3b(r) dr
        #      = \sum_g w_g \phi_m(g) \phi_n(g) V_3b(g)
        
        # (N_orb, N_grid) * (N_grid,) -> (N_orb, N_grid)
        weighted_phi = self.phi * (self.weights * V_3b_g)[None, :]
        F_3b = jnp.dot(weighted_phi, self.phi.T)
        
        return F_3b

    def get_3b_fock_full(self, jastrow_params, dm1):
        """Get 3-body Fock matrix correction (full tensor calculation).
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis)
            
        Returns:
            Fock matrix contribution (N, N)
        """
        # 1. Compute density on grid
        density_g = jnp.einsum('mg,ng,mn->g', self.phi, self.phi, dm1)
        
        # 2. Compute W(r) on grid using full broadcasting
        # W(r_i) = \sum_j w_j rho(r_j) \nabla_i u(r_i, r_j)
        
        # grads: (N_grid, N_grid, 3)
        # This might be memory intensive for large grids!
        grads = self.jastrow_factor.grad_r_batch(self.grid_points, self.grid_points, jastrow_params)
        
        # Weighted density: (N_grid,)
        w_density = self.weights * density_g
        
        # Contract: (N_i, N_j, 3) * (N_j,) -> (N_i, 3)
        W_all = jnp.einsum('ijc,j->ic', grads, w_density)
        
        # 3. Compute V_3b_direct(r) = |W(r)|^2
        V_3b_g = jnp.sum(W_all**2, axis=1) # (N_grid,)
        
        # 4. Integrate to get Fock matrix elements
        weighted_phi = self.phi * (self.weights * V_3b_g)[None, :]
        F_3b = jnp.dot(weighted_phi, self.phi.T)
        
        return F_3b




@struct.dataclass
class ISDFTC(TC):
    """JAX implementation of Transcorrelated method using ISDF.
    
    Attributes:
        xi_rho: ISDF coefficients for density (N_fused, N_grid)
        xi_grad: ISDF coefficients for gradients (N_fused, N_grid, 3)
        pivots: ISDF pivot indices (N_fused,)
        phi: ISDF basis for density (Nb, N_fused)
        grad_phi: ISDF basis for gradients (Nb, N_fused, 3)
        isdf_kernels: Dictionary storing precomputed kernels (U1, U3)
    """
    xi_rho: jnp.ndarray = struct.field(default=None)
    xi_grad: jnp.ndarray = struct.field(default=None)
    pivots: jnp.ndarray = struct.field(default=None)
    phi_isdf: jnp.ndarray = struct.field(default=None)
    grad_phi_isdf: jnp.ndarray = struct.field(default=None)
    isdf_kernels: dict = struct.field(default=None, pytree_node=True)

    @classmethod
    def from_tc(cls, tc_obj, n_rank=None):
        """Initialize ISDFTC object from TC object.
        
        Args:
            tc_obj: TC object
            n_rank: Rank for ISDF decomposition (default: N_grid // 4)
            
        Returns:
            ISDFTC: Initialized ISDFTC object
        """
        from . import df
        
        if n_rank is None:
            n_rank = tc_obj.grid_points.shape[0] // 4
            
        # Perform ISDF decomposition
        # We use the same rank for both phi and grad for simplicity, 
        # matching the numpy implementation default behavior
        phi_isdf, xi_rho, grad_phi_isdf, xi_grad, pivots = df.isdf_decompose(
            tc_obj.phi, tc_obj.grad_phi, n_rank, n_rank, weights=tc_obj.weights
        )
        
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
            xi_rho=xi_rho,
            xi_grad=xi_grad,
            pivots=pivots,
            phi_isdf=phi_isdf,
            grad_phi_isdf=grad_phi_isdf,
            isdf_kernels=None
        )

    def compute_kmat_kernels(self, jastrow_params, batch_size=1000):
        """Compute K1 and K3 kernels with multi-GPU support.
        
        Returns:
            dict: {'K1_kernel': K1_kernel, 'K3_kernel': K3_kernel}
        """
        n_devices = jax.local_device_count()
        n_grid = self.grid_points.shape[0]
        
        # Pad grid to be divisible by n_devices
        remainder = n_grid % n_devices
        if remainder != 0:
            padding = n_devices - remainder
            padded_grid_points = jnp.pad(self.grid_points, ((0, padding), (0, 0)))
            padded_weights = jnp.pad(self.weights, ((0, padding),))
            padded_xi_rho = jnp.pad(self.xi_rho, ((0, 0), (0, padding)))
            padded_xi_grad = jnp.pad(self.xi_grad, ((0, 0), (0, padding), (0, 0)))
        else:
            padded_grid_points = self.grid_points
            padded_weights = self.weights
            padded_xi_rho = self.xi_rho
            padded_xi_grad = self.xi_grad
            
        n_grid_padded = padded_grid_points.shape[0]
        n_per_device = n_grid_padded // n_devices
        
        # Shard arrays for r1 (bra side)
        sharded_grid = padded_grid_points.reshape(n_devices, n_per_device, 3)
        sharded_weights = padded_weights.reshape(n_devices, n_per_device)
        # xi_rho: (N_rank, N) -> (N_rank, n_dev, N_per) -> (n_dev, N_rank, N_per)
        sharded_xi_rho = padded_xi_rho.reshape(self.phi_isdf.shape[1], n_devices, n_per_device).transpose(1, 0, 2)
        # xi_grad: (N_rank, N, 3) -> (N_rank, n_dev, N_per, 3) -> (n_dev, N_rank, N_per, 3)
        sharded_xi_grad = padded_xi_grad.reshape(self.phi_isdf.shape[1], n_devices, n_per_device, 3).transpose(1, 0, 2, 3)
        
        # Full arrays for r2 (ket side) - broadcasted to all devices
        full_grid = self.grid_points
        full_weights = self.weights
        full_xi_rho = self.xi_rho
        
        def compute_on_device(grid_shard, weights_shard, xi_rho_shard, xi_grad_shard, jastrow_params):
            K1_shard = kmat_jax.calc_K1_kernel(
                xi_grad_shard, full_xi_rho, weights_shard, full_weights,
                self.jastrow_factor, jastrow_params,
                grid_shard, full_grid, batch_size
            )
            
            K3_shard = kmat_jax.calc_K3_kernel(
                xi_rho_shard, full_xi_rho, weights_shard, full_weights,
                self.jastrow_factor, jastrow_params,
                grid_shard, full_grid, batch_size
            )
            return K1_shard, K3_shard
            
        pmapped_compute = jax.pmap(compute_on_device, axis_name='devices', in_axes=(0, 0, 0, 0, None))
        
        K1_shards, K3_shards = pmapped_compute(sharded_grid, sharded_weights, sharded_xi_rho, sharded_xi_grad, jastrow_params)
        
        # Sum over devices
        K1_kernel = jnp.sum(K1_shards, axis=0)
        K3_kernel = jnp.sum(K3_shards, axis=0)
        
        return {'K1_kernel': K1_kernel, 'K3_kernel': K3_kernel}

    def get_2b(self, jastrow_params, block_str=None, ranges=None, batch_size=1000):
        """Calculate TC correction terms using ISDF with multi-GPU support."""
        start_time = time.perf_counter()
        logging.debug("Starting ISDFTC.get_2b")
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
            
        # Check if kernels are available, if not compute them
        if self.isdf_kernels is None:
            # We can't update self in a jitted/frozen dataclass easily if it's not designed for it.
            # But here we are just computing them for this call if they don't exist.
            # Ideally, the user should call isdf() first to populate them.
            # For now, let's compute them on the fly if missing.
            kernels = self.compute_kmat_kernels(jastrow_params, batch_size)
        else:
            kernels = self.isdf_kernels
            
        U1 = kernels['K1_kernel']
        U3 = kernels['K3_kernel']
        
        # Contract using pivot values
        # We need phi and grad_phi at pivot points.
        # self.phi_isdf is (Nb, N_fused), which ARE the values at pivot points (columns of phi).
        # self.grad_phi_isdf is (Nb, N_fused, 3).
        
        # K1 term (nabla on p)
        K1 = kmat_jax.contract_K1_isdf(self.phi_isdf, self.grad_phi_isdf, U1, ranges)
        
        # K3 term
        K3 = kmat_jax.contract_K3_isdf(self.phi_isdf, U3, ranges)
        
        # K2 term (nabla on q) - transpose of K1 if symmetric
        slice_p, slice_q, slice_r, slice_s = ranges if ranges else (slice(None), slice(None), slice(None), slice(None))
        
        if slice_p == slice_q:
            K2 = K1.transpose(1, 0, 2, 3)
        else:
            # Compute K2 explicitly
            ranges_k2 = (slice_q, slice_p, slice_r, slice_s)
            K2_transposed = kmat_jax.contract_K1_isdf(self.phi_isdf, self.grad_phi_isdf, U1, ranges_k2)
            K2 = K2_transposed.transpose(1, 0, 2, 3)
            
        result = 0.5 * (K1 - K2 + K3)
        
        # Symmetrize result (add transpose block) to match TC.get_2b
        if slice_p == slice_r and slice_q == slice_s:
            result += result.transpose(2, 3, 0, 1)
        else:
            # For non-symmetric blocks, we would need to compute the transpose block explicitly
            # But ISDFTC.get_2b currently assumes we want the full result or a specific block.
            # If ranges are provided, we compute that block.
            # TC.get_2b computes the transpose block if ranges are not symmetric.
            # Here we should probably do the same if we want to match TC.get_2b behavior exactly.
            
            # However, ISDF allows computing arbitrary blocks efficiently.
            # If the user asks for a block, they might expect just that block.
            # But TC.get_2b returns the symmetrized contribution.
            
            # Let's match TC.get_2b logic:
            ranges_T = (slice_r, slice_s, slice_p, slice_q)
            
            # We need to compute result for ranges_T
            # This requires re-computing K1, K2, K3 for ranges_T
            
            # K1_T
            K1_T = kmat_jax.contract_K1_isdf(self.phi_isdf, self.grad_phi_isdf, U1, ranges_T)
            
            # K3_T
            K3_T = kmat_jax.contract_K3_isdf(self.phi_isdf, U3, ranges_T)
            
            # K2_T
            slice_p_T, slice_q_T, slice_r_T, slice_s_T = ranges_T
            if slice_p_T == slice_q_T:
                K2_T = K1_T.transpose(1, 0, 2, 3)
            else:
                ranges_k2_T = (slice_q_T, slice_p_T, slice_r_T, slice_s_T)
                K2_transposed_T = kmat_jax.contract_K1_isdf(self.phi_isdf, self.grad_phi_isdf, U1, ranges_k2_T)
                K2_T = K2_transposed_T.transpose(1, 0, 2, 3)
                
            result_T = 0.5 * (K1_T - K2_T + K3_T)
            result += result_T.transpose(2, 3, 0, 1)
        
        total_time = time.perf_counter() - start_time
        logging.debug(f"ISDFTC.get_2b completed in {total_time:.4f} s")
        return -result

    def get_3b_fock(self, jastrow_params, dm1, L_aux=None):
        """Get 3-body Fock matrix correction using ISDF.
        
        Scaling: O(N_aux * N_grid) per SCF step (after precomputation).
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix
            L_aux: Optional precomputed auxiliary potential (N_aux, N_grid, 3).
                   If None, it will be computed on the fly.
        """
        # 1. Check cache / Compute L_aux
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params)
        
        # 2. Compute density at pivots
        # rho(mu) = sum_pq D_pq phi_p(mu) phi_q(mu)
        # phi_pivots = self.phi_isdf (N_orb, N_aux)
        rho_pivots = jnp.einsum('ma,na,mn->a', self.phi_isdf, self.phi_isdf, dm1)
        
        # 3. Compute W on full grid using L_aux
        # W(r) = sum_mu rho(mu) L_mu(r)
        # L_aux: (N_aux, N_grid, 3)
        # rho_pivots: (N_aux,)
        W_g = jnp.einsum('a,agc->gc', rho_pivots, L_aux)
        
        # 4. Compute V_3b
        V_3b_g = jnp.sum(W_g**2, axis=1)
        
        # 5. Integrate Fock matrix using ISDF
        # F_mn = sum_mu (sum_g w_g xi_mu(g) V_3b(g)) phi_m(mu) phi_n(mu)
        weighted_V = self.weights * V_3b_g
        V_proj = jnp.dot(self.xi_rho, weighted_V) # (N_aux,)
        
        phi_pivots = self.phi_isdf # (N_orb, N_aux)
        weighted_phi = phi_pivots * V_proj[None, :]
        F_3b = jnp.dot(weighted_phi, phi_pivots.T)
        
        return F_3b
