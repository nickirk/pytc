"""JAX implementation of Transcorrelated method."""

from typing import Any
import numpy as np
import os
import logging
import time
import gc
import jax
import jax.numpy as jnp
import h5py
from flax import struct
from pyscf import dft
from . import kmat as kmat_jax

logger = logging.getLogger(__name__)

# Module-level cache for the fixed rank_block_size computed once per
# (n_orb, N_fused) pair.  Using the worst-case orbital dimensions
# (n_orb, n_orb) produces the most conservative (smallest) power-of-2
# block size, which is safe for every (Np, Nq) slice encountered
# during CCSD iterations.  This eliminates JIT recompilation from
# changing static_argnums values across ovvv / vovv / vvvv phases.
_FIXED_RBS_CACHE: dict = {}

def _compute_2b_shard(phi, grad_phi, grid, weights, jastrow_params, jastrow_factor, ranges, batch_size):
    """Compute K terms for a grid shard (pmapped)."""
    # Helper to calculate size from tuple (start, stop, step)
    def get_size(r, size):
        start, stop, step = slice(*r).indices(size)
        return (stop - start + (step - 1)) // step
    
    n_orb = phi.shape[0]
    # Unpack ranges (p, q, r, s) tuples
    t_p, t_q, t_r, t_s = ranges
    
    Np = get_size(t_p, n_orb)
    Nq = get_size(t_q, n_orb)
    Nr = get_size(t_r, n_orb)
    Ns = get_size(t_s, n_orb)
    
    # Compute K1 (nabla on p)
    # Convert tuples back to slices for kmat functions
    slices = tuple(slice(*r) for r in ranges)
    
    k1_raw = kmat_jax.calc_K1(
        phi, grad_phi,
        jastrow_factor, jastrow_params,
        grid, weights,
        ranges=slices,
        batch_size=batch_size
    )
    k1 = k1_raw.reshape(Np, Nq, Nr, Ns)
    
    # Compute K2 (nabla on q)
    if t_p == t_q:
        # If p and q ranges are identical, K2 is just K1 with p,q swapped
        k2 = k1.transpose(1, 0, 2, 3)
    else:
        # Must compute explicitly: swap p and q
        ranges_k2 = (slices[1], slices[0], slices[2], slices[3])
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
        ranges=slices,
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
        logger.info(f"TC: Initializing grid with level {grid_lvl}")
        start_time = time.perf_counter()
        grids = dft.gen_grid.Grids(mol)
        grids.level = grid_lvl
        grids.build()
        logger.debug(f"TC: Grid initialized in {time.perf_counter() - start_time:.3f} seconds")
        
        grid_points = jnp.asarray(grids.coords)
        weights = jnp.asarray(grids.weights)
        
        # Evaluate basis on grid
        # Use PySCF to evaluate AOs with numpy arrays
        logger.info(f"TC: Evaluating basis on grid")
        start_time = time.perf_counter()
        ao = dft.numint.eval_ao(mol, grids.coords, deriv=1)
        ao_values = ao[0].T  # (N_ao, N_grid)
        ao_gradients = ao[1:4].transpose(2, 1, 0)  # (N_ao, N_grid, 3)
        logger.debug(f"TC: AO basis evaluated in {time.perf_counter() - start_time:.3f} seconds")
        
        # Transform to MO basis using JAX/GPU for speed
        logger.info(f"TC: Transforming to MO basis (GPU)")
        start_time = time.perf_counter()
        
        # Move to GPU
        mo_coeff_jax = jnp.asarray(mo_coeff)
        ao_values_jax = jnp.asarray(ao_values)
        ao_gradients_jax = jnp.asarray(ao_gradients)
        
        # phi = mo_coeff.T @ ao_values
        phi = jnp.matmul(mo_coeff_jax.T, ao_values_jax)
        
        # grad_phi = mo_coeff.T @ ao_gradients (reshaped)
        n_mo = mo_coeff.shape[1]
        n_ao = mo_coeff.shape[0]
        n_grid = grid_points.shape[0]
        
        # Reshape ao_gradients to (n_ao, n_grid * 3) for matmul
        ao_grad_reshaped = ao_gradients_jax.reshape(n_ao, -1)
        grad_phi_reshaped = jnp.matmul(mo_coeff_jax.T, ao_grad_reshaped)
        grad_phi = grad_phi_reshaped.reshape(n_mo, n_grid, 3)
        
        logger.debug(f"TC: MO basis transformed in {time.perf_counter() - start_time:.3f} seconds")
        
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
        logger.debug("Starting TC.get_2b")
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
            
        # Convert slices to hashable tuples for JAX static args
        ranges_tuple = tuple((s.start, s.stop, s.step) for s in ranges)
            
        # Compute main block: 0.5 * (K1 - K2 + K3)
        result_sum = pmapped_compute(
            sharded_phi, sharded_grad_phi, sharded_grid, sharded_weights,
            jastrow_params, self.jastrow_factor, ranges_tuple, batch_size
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
            ranges_T_tuple = tuple((s.start, s.stop, s.step) for s in ranges_T)
            
            result_sum_T = pmapped_compute(
                sharded_phi, sharded_grad_phi, sharded_grid, sharded_weights,
                jastrow_params, self.jastrow_factor, ranges_T_tuple, batch_size
            )
            result_T = result_sum_T[0]
            
            result += jax.lax.transpose(result_T, (2, 3, 0, 1))
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"TC.get_2b completed in {total_time:.4f} s")
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
    xi_phi: jnp.ndarray = struct.field(default=None)
    xi_grad: jnp.ndarray = struct.field(default=None)
    pivots: jnp.ndarray = struct.field(default=None)
    phi_isdf: jnp.ndarray = struct.field(default=None)
    grad_phi_isdf: jnp.ndarray = struct.field(default=None)
    isdf_kernels: dict = struct.field(default=None, pytree_node=True)
    is_incore: bool = struct.field(default=False, pytree_node=False)
    save_path: str = struct.field(default=None, pytree_node=False)

    def _get_fixed_rank_block_size(self):
        """Return a fixed rank_block_size that is safe for all orbital slices.

        Uses worst-case dimensions ``(n_orb, n_orb)`` so that the resulting
        power-of-2 block size is the smallest (most conservative) one.  Any
        smaller ``(Np, Nq)`` combination would yield a larger or equal block
        size, so this value is safe everywhere and avoids JIT recompilation
        from changing ``static_argnums`` across CCSD phases.

        The result is cached at module level keyed by ``(n_orb, N_fused)``
        so the GPU budget query happens only once per run.

        Returns ``None`` when ``phi_isdf`` is not yet available (pre-ISDF).
        """
        if self.phi_isdf is None:
            return None
        N_fused = self.phi_isdf.shape[1]
        key = (int(self.n_orb), int(N_fused))
        if key not in _FIXED_RBS_CACHE:
            from pytc.utils.gpu_memory import adaptive_rank_block_size
            rbs = adaptive_rank_block_size(self.n_orb, self.n_orb, N_fused)
            logger.info(f"  Fixed rank_block_size = {rbs} "
                        f"(worst-case n_orb={self.n_orb}, N_fused={N_fused})")
            _FIXED_RBS_CACHE[key] = rbs
        return _FIXED_RBS_CACHE[key]

    @classmethod
    def from_tc(cls, tc_obj, n_rank=None, is_incore=False, save_path=None, ls_grid_batch_size=16384):
        """Initialize ISDFTC object from TC object.
        
        Args:
            tc_obj: TC object
            n_rank: Rank for ISDF decomposition (default: N_grid // 4)
            is_incore: Whether to perform in-core decomposition
            save_path: Path to save ISDF kernels
            ls_grid_batch_size: Batch size for grid decomposition in isdf_decompose
            
        Returns:
            ISDFTC: Initialized ISDFTC object
        """
        from . import df
        
        if n_rank is None:
            n_rank = tc_obj.grid_points.shape[0] // 4
            
        # Perform ISDF decomposition
        phi_isdf, xi_phi, grad_phi_isdf, xi_grad, pivots, actual_save_path = df.isdf_decompose(
            tc_obj.phi, tc_obj.grad_phi, n_rank, n_rank, weights=tc_obj.weights,
            is_incore=is_incore, save_path=save_path, grid_batch_size=ls_grid_batch_size
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
            xi_phi=xi_phi,
            xi_grad=xi_grad,
            pivots=pivots,
            phi_isdf=phi_isdf,
            grad_phi_isdf=grad_phi_isdf,
            isdf_kernels=None,
            is_incore=is_incore,
            save_path=actual_save_path
        )

    def compute_kmat_kernels(self, jastrow_params, batch_size=1024, host_grid_block_size=None):
        """Compute K1 and K3 kernels with multi-GPU support.
        
        Returns:
            dict: {'K1_kernel': K1_kernel, 'K3_kernel': K3_kernel}
        """
        n_devices = jax.local_device_count()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        
        # Pad grid to be divisible by n_devices
        logger.debug(f"     compute_kmat_kernels: Padding grid to be divisible by {n_devices} devices...")
        devices = jax.local_devices()
        
        if host_grid_block_size is None:
            host_grid_block_size = n_grid
            
        # Initialize kernels on host
        K1_kernel = np.zeros((n_rank, n_rank, 3))
        K3_kernel = np.zeros((n_rank, n_rank))
        
        # Open datasets if needed
        xi_phi_ds = None
        xi_grad_ds = None
        f_xi = None
        if self.xi_phi is None and self.save_path:
            f_xi = h5py.File(self.save_path, 'r')
            xi_phi_ds = f_xi['xi_phi']
            xi_grad_ds = f_xi['xi_grad']
            
        # Full arrays for r2 (ket side) - on CPU
        full_grid = jax.device_put(np.asarray(self.grid_points), devices[0])
        full_weights = jax.device_put(np.asarray(self.weights), devices[0])
        
        if self.xi_phi is not None:
            full_xi_phi = jax.device_put(self.xi_phi, devices[0])
        else:
            full_xi_phi = jax.device_put(xi_phi_ds[:], devices[0])
            
        jastrow_factor = self.jastrow_factor
        
        def compute_on_device(grid_shard, weights_shard, xi_phi_shard, xi_grad_shard, jastrow_params, 
                              full_grid, full_weights, full_xi_phi):
            K1_shard = kmat_jax.calc_K1_kernel(
                xi_grad_shard, full_xi_phi, weights_shard, full_weights,
                jastrow_factor, jastrow_params,
                grid_shard, full_grid, batch_size
            )
            
            K3_shard = kmat_jax.calc_K3_kernel(
                xi_phi_shard, full_xi_phi, weights_shard, full_weights,
                jastrow_factor, jastrow_params,
                grid_shard, full_grid, batch_size
            )
            return K1_shard, K3_shard
            
        # Convert ALL inputs to NumPy to avoid "incompatible devices" error in pmap.
        # JAX will then handle broadcasting and streaming from Host RAM.
        # Note: sharded_xi_phi and sharded_xi_grad are already device-resident or NumPy.

        logger.info(f"  compute_kmat_kernels: Starting pmap for K-kernels (n_fused={n_rank}, n_grid={n_grid})...")
        pmapped_compute = jax.pmap(compute_on_device, axis_name='devices', in_axes=(0, 0, 0, 0, None, None, None, None))

        from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read

        def _prepare_kmat_block(g0_loc):
            """Prepare sharded data for one grid block (runs in background thread)."""
            g1_loc = min(g0_loc + host_grid_block_size, n_grid)
            n_block_loc = g1_loc - g0_loc
            remainder_loc = n_block_loc % n_devices
            padding_loc = (n_devices - remainder_loc) if remainder_loc != 0 else 0
            n_block_padded_loc = n_block_loc + padding_loc
            n_per_dev = n_block_padded_loc // n_devices

            grid_block = self.grid_points[g0_loc:g1_loc]
            weights_block = self.weights[g0_loc:g1_loc]
            if padding_loc > 0:
                grid_block = np.pad(grid_block, ((0, padding_loc), (0, 0)))
                weights_block = np.pad(weights_block, ((0, padding_loc),))
            s_grid = grid_block.reshape(n_devices, n_per_dev, 3)
            s_weights = weights_block.reshape(n_devices, n_per_dev)

            phi_list = []
            grad_list = []
            for d in range(n_devices):
                start = g0_loc + d * n_per_dev
                end = min(g0_loc + (d + 1) * n_per_dev, g1_loc)
                actual_len = end - start
                if self.xi_phi is not None:
                    xi_phi_d = safe_hdf5_read(self.xi_phi, (slice(None), slice(start, end)))
                    xi_grad_d = safe_hdf5_read(self.xi_grad, (slice(None), slice(start, end), slice(None)))
                else:
                    xi_phi_d = safe_hdf5_read(xi_phi_ds, (slice(None), slice(start, end)))
                    xi_grad_d = safe_hdf5_read(xi_grad_ds, (slice(None), slice(start, end), slice(None)))
                if actual_len < n_per_dev:
                    xi_phi_d = np.pad(xi_phi_d, ((0, 0), (0, n_per_dev - actual_len)))
                    xi_grad_d = np.pad(xi_grad_d, ((0, 0), (0, n_per_dev - actual_len), (0, 0)))
                phi_list.append(jax.device_put(xi_phi_d, devices[d]))
                grad_list.append(jax.device_put(xi_grad_d, devices[d]))
            s_xi_phi = jax.device_put_sharded(phi_list, devices)
            s_xi_grad = jax.device_put_sharded(grad_list, devices)
            return s_grid, s_weights, s_xi_phi, s_xi_grad

        try:
            pending_kmat = None
            for g0 in range(0, n_grid, host_grid_block_size):
                g1 = min(g0 + host_grid_block_size, n_grid)
                logger.info(f"    compute_kmat_kernels: Processing grid block [{g0}:{g1}]...")

                # Fetch prepared data (prefetched or inline)
                if pending_kmat is not None:
                    sharded_grid, sharded_weights, sharded_xi_phi, sharded_xi_grad = await_read(pending_kmat)
                    pending_kmat = None
                else:
                    sharded_grid, sharded_weights, sharded_xi_phi, sharded_xi_grad = _prepare_kmat_block(g0)

                K1_shards, K3_shards = pmapped_compute(sharded_grid, sharded_weights, sharded_xi_phi, sharded_xi_grad, jastrow_params,
                                                       full_grid, full_weights, full_xi_phi)

                # While GPU runs pmap, prefetch next block's data in background
                next_g0 = g0 + host_grid_block_size
                if next_g0 < n_grid:
                    pending_kmat = async_read(lambda _g=next_g0: _prepare_kmat_block(_g))

                K1_kernel += np.array(jnp.sum(K1_shards, axis=0))
                K3_kernel += np.array(jnp.sum(K3_shards, axis=0))
                
                # Explicitly clear memory
                del sharded_grid, sharded_weights, sharded_xi_phi, sharded_xi_grad, K1_shards, K3_shards
                gc.collect()
                
        finally:
            if f_xi: f_xi.close()
            
        return {'K1_kernel': jnp.asarray(K1_kernel), 'K3_kernel': jnp.asarray(K3_kernel)}

    def _compute_L_aux(self, jastrow_params, batch_size=1024, save_path=None, host_grid_block_size=None):
        """Compute L_aux (G) for the full grid with grid-blocking to save host RAM."""
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        
        # If host_grid_block_size is None, process the whole grid in one block
        if host_grid_block_size is None:
            host_grid_block_size = n_grid
            
        # Initialize L_aux on host or HDF5
        L_aux_out = None
        f_out = None
        if save_path:
            f_out = h5py.File(save_path, 'a')
            if 'L_aux' in f_out: del f_out['L_aux']
            L_aux_out = f_out.create_dataset('L_aux', (n_rank, n_grid, 3), dtype='f8')
        else:
            L_aux_out = np.zeros((n_rank, n_grid, 3))
            
        # Open xi_phi dataset if needed
        xi_phi_ds = None
        f_xi = None
        if self.xi_phi is None and self.save_path:
            f_xi = h5py.File(self.save_path, 'r')
            xi_phi_ds = f_xi['xi_phi']
            
        def compute_block_on_device(grid_eval_shard, jastrow_params, grid_int, weights_int, xi_phi_int):
            def scan_body(carry, i):
                r_eval = grid_eval_shard
                g_batch = jax.lax.dynamic_slice(grid_int, (i * batch_size, 0), (batch_size, 3))
                w_batch = jax.lax.dynamic_slice(weights_int, (i * batch_size,), (batch_size,))
                xi_batch = jax.lax.dynamic_slice(xi_phi_int, (0, i * batch_size), (n_rank, batch_size))
                
                u_grad = self.jastrow_factor.grad_r_batch(r_eval, g_batch, jastrow_params)
                xi_weighted = xi_batch * w_batch[None, :]
                update = jnp.einsum('ab,ibk->aik', xi_weighted, u_grad)
                return carry + update, None

            n_int = grid_int.shape[0]
            n_batches = (n_int + batch_size - 1) // batch_size
            
            # Pad integration grid for scan
            pad_int = n_batches * batch_size - n_int
            if pad_int > 0:
                grid_int = jnp.pad(grid_int, ((0, pad_int), (0, 0)))
                weights_int = jnp.pad(weights_int, (0, pad_int))
                xi_phi_int = jnp.pad(xi_phi_int, ((0, 0), (0, pad_int)))
                
            init_val = jnp.zeros((n_rank, grid_eval_shard.shape[0], 3))
            res, _ = jax.lax.scan(scan_body, init_val, jnp.arange(n_batches))
            return res

        pmapped_compute = jax.pmap(compute_block_on_device, in_axes=(0, None, None, None, None))

        try:
            # Outer loop: Evaluation blocks (r)
            for r0 in range(0, n_grid, host_grid_block_size):
                r1 = min(r0 + host_grid_block_size, n_grid)
                n_eval = r1 - r0
                logger.debug(f"    _compute_L_aux: Processing evaluation block [{r0}:{r1}]...")
                
                remainder = n_eval % n_devices
                padding = (n_devices - remainder) if remainder != 0 else 0
                n_eval_padded = n_eval + padding
                n_per_device_local = n_eval_padded // n_devices
                
                grid_eval_block = self.grid_points[r0:r1]
                if padding > 0:
                    grid_eval_block = jnp.pad(grid_eval_block, ((0, padding), (0, 0)))
                sharded_grid_eval = grid_eval_block.reshape(n_devices, n_per_device_local, 3)
                
                # Initialize accumulator on device
                # We can use the first result to initialize, or create zeros
                res_rep_accum = None
                
                # Inner loop: Integration blocks (g)
                # Also controlled by host_grid_block_size to limit peak memory of inputs
                from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read

                def _prepare_Laux_int_block(g0_loc):
                    """Load integration chunk to device (background-thread safe)."""
                    g1_loc = min(g0_loc + host_grid_block_size, n_grid)
                    g_chunk = jax.device_put(np.asarray(self.grid_points[g0_loc:g1_loc]))
                    w_chunk = jax.device_put(np.asarray(self.weights[g0_loc:g1_loc]))
                    if self.xi_phi is not None:
                        xi_chunk = jax.device_put(safe_hdf5_read(self.xi_phi, (slice(None), slice(g0_loc, g1_loc))))
                    else:
                        xi_chunk = jax.device_put(safe_hdf5_read(xi_phi_ds, (slice(None), slice(g0_loc, g1_loc))))
                    return g_chunk, w_chunk, xi_chunk

                pending_int = None
                for g0 in range(0, n_grid, host_grid_block_size):
                    g1 = min(g0 + host_grid_block_size, n_grid)
                    
                    # Fetch prepared data (prefetched or inline)
                    if pending_int is not None:
                        grid_int_chunk, weights_int_chunk, xi_phi_chunk = await_read(pending_int)
                        pending_int = None
                    else:
                        grid_int_chunk, weights_int_chunk, xi_phi_chunk = _prepare_Laux_int_block(g0)
                            
                    # Compute partial update
                    res_partial = pmapped_compute(sharded_grid_eval, jastrow_params, grid_int_chunk, weights_int_chunk, xi_phi_chunk)

                    # While GPU runs pmap, prefetch next integration block
                    next_g0 = g0 + host_grid_block_size
                    if next_g0 < n_grid:
                        pending_int = async_read(lambda _g=next_g0: _prepare_Laux_int_block(_g))
                    
                    if res_rep_accum is None:
                        res_rep_accum = res_partial
                    else:
                        res_rep_accum += res_partial
                    
                    # Explicitly free memory
                    del grid_int_chunk, weights_int_chunk, xi_phi_chunk, res_partial
                
                # Store result for this evaluation block
                res_block = res_rep_accum.transpose(1, 0, 2, 3).reshape(n_rank, -1, 3)
                L_aux_out[:, r0:r1, :] = np.array(res_block[:, :n_eval, :])
                
                del sharded_grid_eval, res_rep_accum, res_block
                gc.collect()
                
        finally:
            if f_xi: f_xi.close()
            # If we are returning a dataset, we MUST NOT close f_out here.
            # The caller or the dataset object itself will manage the lifecycle.
            # However, h5py datasets require the file to remain open.
            # To be safe and consistent with other parts, we'll close it if we're not returning it.
            if f_out and not isinstance(L_aux_out, h5py.Dataset):
                f_out.close()
            
        return L_aux_out
	

    def isdf(self, jastrow_params, save_path=None, batch_size=1000, host_grid_block_size=None):
        """Compute ISDF intermediates and store them.
        
        Computes K1_kernel, K3_kernel, and L_aux.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor.
            save_path: Optional path to save intermediates to HDF5.
            batch_size: Batch size for computation.
            host_grid_block_size: Block size for grid batching on host.
        """
        logger.info("Computing ISDF intermediates (TC)...")
        start_time = time.perf_counter()
        
        # Use save_path if provided, otherwise use self.save_path
        out_path = save_path if save_path else self.save_path
        
        # Check if kernels already exist in HDF5
        kernels = {}
        if out_path and os.path.exists(out_path):
            try:
                f = h5py.File(out_path, 'r')
                if 'K1_kernel' in f and 'K3_kernel' in f and 'L_aux' in f:
                    logger.info(f"  Found existing K1, K3, and L_aux in {out_path}. Reading from file...")
                    logger.info(f"  Loading K1 with shape: {f['K1_kernel'].shape} on host RAM.")
                    kernels['K1_kernel'] = f['K1_kernel'][:]
                    logger.info(f"  Loading K3 with shape: {f['K3_kernel'].shape} on host RAM")
                    kernels['K3_kernel'] = f['K3_kernel'][:]
                    if self.is_incore:
                        logger.debug(f"  incore mode: Loading L_aux with shape: {f['L_aux'].shape} on host RAM")
                        kernels['L_aux'] = f['L_aux'][:]
                        f.close()
                    else:
                        logger.debug(f"  out-of-core mode: Streaming L_aux with shape: {f['L_aux'].shape} from {out_path}")
                        kernels['L_aux'] = f['L_aux'] 

                    logger.info(f"ISDF intermediates loaded from file in {time.perf_counter() - start_time:.4f} s")
                    return self.replace(isdf_kernels=kernels)
                
                # If we are here, keys are missing. Close the file!
                f.close()
            except (IOError, KeyError) as e:
                logger.warning(f"  Error reading kernels from {out_path}: {e}. Recomputing...")

        # 1. Compute K1_kernel and K3_kernel
        logger.info("  Computing K1 and K3 kernels...")
        
        kernels = self.compute_kmat_kernels(jastrow_params, batch_size, host_grid_block_size=host_grid_block_size)
        logger.info(f"   K1 kernel on device size: {kernels['K1_kernel'].size * 8 / 1024**3:.2f} GB")
        logger.info(f"   K3 kernel on device size: {kernels['K3_kernel'].size * 8 / 1024**3:.2f} GB")
        
        # 2. Compute L_aux
        logger.info("  Computing L_aux...")
        L_aux = self._compute_L_aux(jastrow_params, batch_size, save_path=out_path if not self.is_incore else None, host_grid_block_size=host_grid_block_size)
        
        # Move L_aux to CPU RAM to avoid GPU OOM (it can be very large)
        # If it's an HDF5 dataset, we keep it as is.
        if not isinstance(L_aux, (np.ndarray, jnp.ndarray)):
             logger.info("  L_aux is streaming from HDF5")
             kernels['L_aux'] = L_aux
        else:
            logger.info("  Moving L_aux to CPU")
            logger.info(f"    L_aux shape: {L_aux.shape}")
            logger.info(f"    L_aux size: {L_aux.size * 8 / 1024**3:.2f} GB")
            cpu_device = jax.devices("cpu")[0]
            L_aux = jax.device_put(L_aux, cpu_device)
            kernels['L_aux'] = L_aux
        
        if out_path and not self.is_incore:
            with h5py.File(out_path, 'a') as f:
                for k, v in kernels.items():
                    if k == 'L_aux': continue # Already saved
                    if k in f: del f[k]
                    f.create_dataset(k, data=np.array(v))
                # Basics are already saved by isdf_decompose, but let's ensure they are there
                if 'phi_isdf' not in f: f.create_dataset('phi_isdf', data=np.array(self.phi_isdf))
                if 'grad_phi_isdf' not in f: f.create_dataset('grad_phi_isdf', data=np.array(self.grad_phi_isdf))
                if 'pivots' not in f: f.create_dataset('pivots', data=np.array(self.pivots))
                
        logger.info(f"ISDF intermediates computed in {time.perf_counter() - start_time:.4f} s")
        
        return self.replace(isdf_kernels=kernels, save_path=out_path)

    def _accumulate_transpose_block(self, result_np, U1, U3, ranges_T,
                                    scale, n_sub=2):
        """Compute transpose block in sub-chunks on GPU, accumulate on host.

        Each sub-chunk is computed on GPU, transferred to host via np.asarray(),
        and accumulated into the NumPy array ``result_np``.  GPU only ever holds
        one sub-chunk at a time → peak GPU ≈ S/n_sub (not S).

        Parameters
        ----------
        result_np : numpy array (Np, Nq, Nr, Ns) — host-side accumulator.
        U1, U3 : ISDF kernels (on GPU).
        ranges_T : (slice_r, slice_s, slice_p, slice_q) for the transpose block.
        scale : float multiplier.
        n_sub : int — number of sub-chunks (default 2).
        """
        slice_r_T, slice_s_T, slice_p_T, slice_q_T = ranges_T

        # Determine start/stop for the first axis of the transpose block
        nmo = self.phi_isdf.shape[0]
        r_start = slice_r_T.start if slice_r_T.start is not None else 0
        r_stop = slice_r_T.stop if slice_r_T.stop is not None else nmo
        r_len = r_stop - r_start

        # Use fixed rank_block_size to avoid JIT recompilation.
        rbs = self._get_fixed_rank_block_size()

        # Determine chunk boundaries
        chunk_size = max(1, (r_len + n_sub - 1) // n_sub)
        for i0 in range(0, r_len, chunk_size):
            i1 = min(i0 + chunk_size, r_len)
            sub_slice_r = slice(r_start + i0, r_start + i1)
            sub_ranges = (sub_slice_r, slice_s_T, slice_p_T, slice_q_T)

            # K1-K2 sub-chunk on GPU
            if sub_slice_r == slice_s_T:
                tmp = kmat_jax.contract_K1_isdf(
                    self.phi_isdf, self.grad_phi_isdf, U1, sub_ranges,
                    rank_block_size=rbs)
                tmp = tmp - tmp.transpose(1, 0, 2, 3)
            else:
                tmp = kmat_jax.contract_K1_minus_K2_isdf(
                    self.phi_isdf, self.grad_phi_isdf, U1, sub_ranges,
                    rank_block_size=rbs)

            # K3 sub-chunk on GPU
            tmp = tmp + kmat_jax.contract_K3_isdf(
                self.phi_isdf, U3, sub_ranges, rank_block_size=rbs)

            # Transfer to host and accumulate — frees GPU memory for next chunk
            chunk_np = np.asarray(tmp.transpose(2, 3, 0, 1))
            del tmp
            result_np[:, :, i0:i1, :] += chunk_np * scale
            del chunk_np

    def get_2b(self, jastrow_params, block_str=None, ranges=None, batch_size=1000):
        """Calculate TC correction terms using ISDF with multi-GPU support.

        Memory-optimized: computes each piece on GPU, immediately transfers
        to host, and accumulates on host (NumPy).  GPU only ever holds one
        contraction output at a time → peak GPU ≈ 1S (one output block).
        Returns a JAX array.
        """
        start_time = time.perf_counter()
        logger.debug("Starting ISDFTC.get_2b")
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
            
        # Check if kernels are available, if not compute them
        if self.isdf_kernels is None:
            kernels = self.compute_kmat_kernels(jastrow_params, batch_size)
        else:
            kernels = self.isdf_kernels
            
        U1 = kernels['K1_kernel']
        U3 = kernels['K3_kernel']
        
        slice_p, slice_q, slice_r, slice_s = ranges if ranges else (slice(None), slice(None), slice(None), slice(None))

        # Use a fixed rank_block_size for all kmat calls to avoid
        # JIT recompilation when (Np, Nq) changes across CCSD phases.
        rbs = self._get_fixed_rank_block_size()
        
        # --- Direct block: (K1 - K2 + K3)[pqrs] * 0.5 ---
        # Compute K1-K2, transfer to host, then K3, transfer to host.
        # Never hold two output-sized tensors on GPU simultaneously.
        if slice_p == slice_q:
            k12 = kmat_jax.contract_K1_isdf(
                self.phi_isdf, self.grad_phi_isdf, U1, ranges,
                rank_block_size=rbs)
            k12 = k12 - k12.transpose(1, 0, 2, 3)
        else:
            k12 = kmat_jax.contract_K1_minus_K2_isdf(
                self.phi_isdf, self.grad_phi_isdf, U1, ranges,
                rank_block_size=rbs)
        
        result_np = np.array(k12)  # writable host copy
        del k12

        k3 = kmat_jax.contract_K3_isdf(
            self.phi_isdf, U3, ranges, rank_block_size=rbs)
        result_np += np.asarray(k3)
        del k3
        result_np *= 0.5
        
        # --- Transpose block: (K1 - K2 + K3)[rspq] * 0.5, transposed to (pqrs) ---
        if slice_p == slice_r and slice_q == slice_s:
            result_np += result_np.transpose(2, 3, 0, 1)
        else:
            ranges_T = (slice_r, slice_s, slice_p, slice_q)
            # Sub-chunk the transpose block on GPU, accumulate on host.
            # GPU only holds one sub-chunk at a time.
            self._accumulate_transpose_block(
                result_np, U1, U3, ranges_T, scale=0.5, n_sub=2)
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"ISDFTC.get_2b completed in {total_time:.4f} s")
        return jnp.asarray(-result_np)

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
            if self.isdf_kernels is not None and 'L_aux' in self.isdf_kernels:
                L_aux = self.isdf_kernels['L_aux']
            else:
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
        
        if self.xi_phi is not None:
            V_proj = jnp.dot(self.xi_phi, weighted_V) # (N_aux,)
        else:
            with h5py.File(self.save_path, 'r') as f:
                xi_phi = f['xi_phi'][:]
                V_proj = jnp.dot(xi_phi, weighted_V)
        
        phi_pivots = self.phi_isdf # (N_orb, N_aux)
        weighted_phi = phi_pivots * V_proj[None, :]
        F_3b = jnp.dot(weighted_phi, phi_pivots.T)
        
        return F_3b

    def get_2b_fock(self, jastrow_params, dm1, T=None):
        """Get 2-body Fock matrix correction using ISDF kernels.
        
        This implementation avoids forming the full O(N^4) tensor by contracting
        the density matrix directly with the ISDF kernels. It computes:
        v_2b = Fock(result) = J(result) - 0.5 * K(result)
        where result = 0.5 * (K1 - K2 + K3) + transpose(2,3,0,1)
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis)
            T: Optional precomputed 2-body tensor. If provided, uses base class implementation.
            
        Returns:
            Fock matrix contribution (N, N)
        """
        if T is not None:
            return super().get_2b_fock(jastrow_params, dm1, T)
            
        # Check if kernels are available
        if self.isdf_kernels is None:
             kernels = self.compute_kmat_kernels(jastrow_params)
        else:
            kernels = self.isdf_kernels
            
        U1 = kernels['K1_kernel'] # (k, l, c)
        U3 = kernels['K3_kernel'] # (k, l)
        
        phi = self.phi_isdf # (n_orb, n_fused)
        grad = self.grad_phi_isdf # (n_orb, n_fused, 3)
        dm1 = jnp.array(dm1)
        
        # --- Direct Part Intermediates ---
        # rho[l] = sum_rs D_rs phi_r(l) phi_s(l)
        rho = jnp.einsum('rl,sl,rs->l', phi, phi, dm1)
        # m[k,l] = sum_rs phi_s(k) D_rs phi_r(l)
        m = jnp.einsum('sk,rl,rs->kl', phi, phi, dm1)
        # m_grad[k,l,c] = sum_rs grad_s(k,c) D_rs phi_r(l)
        m_grad = jnp.einsum('skc,rl,rs->klc', grad, phi, dm1)
        
        # --- Direct Part Terms ---
        # K3: phi phi U3 phi phi
        J3 = jnp.einsum('pk,qk,kl,l->pq', phi, phi, U3, rho)
        K3 = jnp.einsum('pk,kl,ql,lk->pq', phi, U3, phi, m)
        
        # K1: grad phi U1 phi phi
        U1_rho = jnp.einsum('klc,l->kc', U1, rho)
        J1 = jnp.einsum('pkc,qk,kc->pq', grad, phi, U1_rho)
        K1 = jnp.einsum('pkc,klc,ql,lk->pq', grad, U1, phi, m)
        
        # K2: phi grad U1 phi phi
        J2 = jnp.einsum('pk,qkc,kc->pq', phi, grad, U1_rho)
        K2 = jnp.einsum('pk,klc,ql,klc->pq', phi, U1, phi, m_grad)
        
        J_dir = 0.5 * (J1 - J2 + J3)
        K_dir = 0.5 * (K1 - K2 + K3)
        
        # --- Transpose Part Intermediates ---
        # rho_grad[k,c] = sum_rs D_rs grad_r(k,c) phi_s(k)
        rho_grad = jnp.einsum('rkc,sk,rs->kc', grad, phi, dm1)
        # rho_grad_rev[k,c] = sum_rs D_rs phi_r(k) grad_s(k,c)
        rho_grad_rev = jnp.einsum('rk,skc,rs->kc', phi, grad, dm1)
        # m_grad_rev[k,l,c] = sum_rs phi_s(k) D_rs grad_r(l,c)
        m_grad_rev = jnp.einsum('sk,rlc,rs->klc', phi, grad, dm1)
        
        U1T = U1.transpose(1, 0, 2)
        U3T = U3.T
        
        # --- Transpose Part Terms ---
        # K3T: phi phi U3T phi phi
        J3T = jnp.einsum('pk,qk,kl,l->pq', phi, phi, U3T, rho)
        K3T = jnp.einsum('pk,kl,ql,lk->pq', phi, U3T, phi, m)
        
        # K1T: phi phi U1T grad phi
        U1T_rho1 = jnp.einsum('klc,lc->k', U1T, rho_grad)
        J1T = jnp.einsum('pk,qk,k->pq', phi, phi, U1T_rho1)
        K1T = jnp.einsum('pk,klc,ql,klc->pq', phi, U1T, phi, m_grad_rev)
        
        # K2T: phi phi U1T phi grad
        U1T_rho2 = jnp.einsum('klc,lc->k', U1T, rho_grad_rev)
        J2T = jnp.einsum('pk,qk,k->pq', phi, phi, U1T_rho2)
        K2T = jnp.einsum('pk,klc,qlc,lk->pq', phi, U1T, grad, m)
        
        J_trans = 0.5 * (J1T - J2T + J3T)
        K_trans = 0.5 * (K1T - K2T + K3T)
        
        J_tot = J_dir + J_trans
        K_tot = K_dir + K_trans
        
        return -(J_tot - 0.5 * K_tot)


