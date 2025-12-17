"""JAX implementation of Transcorrelated method."""

from functools import partial
from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from pyscf import dft
from . import kmat as kmat_jax

def _compute_2b_shard(rho, nabla_rho, grid, weights, jastrow_params, jastrow_factor, ranges, batch_size):
    """Compute K terms for a grid shard (pmapped)."""
    # Unpack ranges (p, q, r, s)
    slice_p, slice_r, slice_q, slice_s = ranges
    
    # Helper to get size
    def get_size(s, size):
        start, stop, step = s.indices(size)
        return (stop - start + (step - 1)) // step
    
    n_orb = rho.shape[0]
    Np = get_size(slice_p, n_orb)
    Nq = get_size(slice_q, n_orb)
    Nr = get_size(slice_r, n_orb)
    Ns = get_size(slice_s, n_orb)
    
    # Compute K1 (nabla on p)
    k1_raw = kmat_jax.calc_K1(
        rho, nabla_rho,
        jastrow_factor, jastrow_params,
        grid, weights,
        ranges=ranges,
        batch_size=batch_size
    )
    k1 = k1_raw.reshape(Np, Nr, Nq, Ns)
    
    # Compute K2 (nabla on r)
    if slice_p == slice_r:
        # If p and r ranges are identical, K2 is just K1 with p,r swapped
        k2 = k1.swapaxes(0, 1)
    else:
        # Must compute explicitly: swap p and r in ranges
        ranges_k2 = (slice_r, slice_p, slice_q, slice_s)
        k2_raw = kmat_jax.calc_K1(
            rho, nabla_rho,
            jastrow_factor, jastrow_params,
            grid, weights,
            ranges=ranges_k2,
            batch_size=batch_size
        )
        # Result is (Nr, Np, Nq, Ns), transpose to (Np, Nr, Nq, Ns)
        k2 = k2_raw.reshape(Nr, Np, Nq, Ns).transpose(1, 0, 2, 3)
        
    # Compute K3
    k3_raw = kmat_jax.calc_K3(
        rho, jastrow_factor, jastrow_params,
        grid, weights,
        ranges=ranges,
        batch_size=batch_size
    )
    k3 = k3_raw.reshape(Np, Nr, Nq, Ns)
    
    # Combine: 0.5 * (K1 - K2 + K3)
    # Note: get_2b (full) logic:
    # k_nabla = K1
    # k_laplacian = -(K1 + K2)
    # result = 0.5 * (k_laplacian + k_square) + k_nabla
    #        = 0.5 * (-K1 - K2 + K3) + K1
    #        = 0.5 * (K1 - K2 + K3)
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
        rho: Basis functions evaluated on grid (N_orb, N_grid)
        nabla_rho: Basis function gradients on grid (N_orb, N_grid, 3)
        n_orb: Number of orbitals (static)
        grid_lvl: Grid level (static)
        jastrow_factor: Jastrow factor instance (PyTree)
        mo_coeff: Molecular orbital coefficients (N_ao, N_orb)
    """
    grid_points: jnp.ndarray
    weights: jnp.ndarray
    rho: jnp.ndarray
    nabla_rho: jnp.ndarray
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
        
        rho = jnp.asarray(mo_values)
        nabla_rho = jnp.asarray(mo_gradients)
        
        return cls(
            grid_points=grid_points,
            weights=weights,
            rho=rho,
            nabla_rho=nabla_rho,
            n_orb=n_orb,
            grid_lvl=grid_lvl,
            jastrow_factor=jastrow_factor,
            mo_coeff=jnp.asarray(mo_coeff),
            nocc=nocc
        )
    
    def _get_block_ranges(self, block_str):
        """Parse block string into slice ranges.
        
        block_str is expected to be in chemists' notation (p, r, q, s),
        where p, r share coordinate 1 and q, s share coordinate 2.
        
        Returns ranges in the order (p, r, q, s) expected by calc_K1.
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
        
        # block_str indices: 0->p, 1->r, 2->q, 3->s
        # calc_K1 expects: (p, q, r, s)
        if len(ranges_list) == 4:
            p = ranges_list[0]
            r = ranges_list[1]
            q = ranges_list[2]
            s = ranges_list[3]
        elif len(ranges_list) == 2:
            p = ranges_list[0]
            r = ranges_list[1]
            q = ranges_list[0]
            s = ranges_list[1]
        else:
            raise ValueError("block_str must have 2 or 4 characters")
        
        return (p, r, q, s)

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
            padded_rho = jnp.pad(self.rho, ((0, 0), (0, padding)))
            padded_nabla_rho = jnp.pad(self.nabla_rho, ((0, 0), (0, padding), (0, 0)))
        else:
            padded_grid_points = self.grid_points
            padded_weights = self.weights
            padded_rho = self.rho
            padded_nabla_rho = self.nabla_rho
            
        n_grid_padded = padded_grid_points.shape[0]
        n_per_device = n_grid_padded // n_devices
        
        # Shard arrays: (n_devices, n_per_device, ...)
        # grid: (N, 3) -> (n_dev, N_per, 3)
        sharded_grid = padded_grid_points.reshape(n_devices, n_per_device, 3)
        # weights: (N,) -> (n_dev, N_per)
        sharded_weights = padded_weights.reshape(n_devices, n_per_device)
        # rho: (Nb, N) -> (Nb, n_dev, N_per) -> (n_dev, Nb, N_per)
        sharded_rho = padded_rho.reshape(self.n_orb, n_devices, n_per_device).transpose(1, 0, 2)
        # nabla_rho: (Nb, N, 3) -> (Nb, n_dev, N_per, 3) -> (n_dev, Nb, N_per, 3)
        sharded_nabla_rho = padded_nabla_rho.reshape(self.n_orb, n_devices, n_per_device, 3).transpose(1, 0, 2, 3)
        
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
            sharded_rho, sharded_nabla_rho, sharded_grid, sharded_weights,
            jastrow_params, self.jastrow_factor, ranges, batch_size
        )
        result = result_sum[0]
        
        # Add transpose block: (q, s, p, r)
        # Check if ranges imply symmetry
        slice_p, slice_r, slice_q, slice_s = ranges
        
        if slice_p == slice_q and slice_r == slice_s:
            # Symmetric block (e.g. 'oooo'), just add transpose of result
            result += result.transpose(2, 3, 0, 1)
        else:
            ranges_T = (slice_q, slice_s, slice_p, slice_r)
            
            result_sum_T = pmapped_compute(
                sharded_rho, sharded_nabla_rho, sharded_grid, sharded_weights,
                jastrow_params, self.jastrow_factor, ranges_T, batch_size
            )
            result_T = result_sum_T[0]
            
            result += result_T.transpose(2, 3, 0, 1)
        
        return -result

    def get_1b_fock(self, jastrow_params, dm1=None):
        """Get one-body Fock matrix correction (AO basis)."""
        return jnp.zeros((self.n_orb, self.n_orb))

    def get_2b_fock(self, jastrow_params, dm1):
        """Get 2-body Fock matrix correction.
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis), shape (N, N)
            
        Returns:
            Fock matrix contribution (N, N)
        """
        k_2b = self.get_2b(jastrow_params) # (p, r, q, s) in chemists notation?
        T = k_2b # (p, r, q, s)
        
        # Coulomb-like contribution
        # \sum_{q,s} T_{prqs} P_{qs}
        J_mat = jnp.einsum('prqs,qs->pr', T, dm1)
        K_mat = jnp.einsum('pqrs,qs->pr', T, dm1)
        
        return J_mat - 0.5 * K_mat

    def get_3b_fock(self, jastrow_params, dm1):
        """Get 3-body Fock matrix correction (on-the-fly).
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis)
            
        Returns:
            Fock matrix contribution (N, N)
        """
        rho_g = jnp.einsum('mg,ng,mn->g', self.rho, self.rho, dm1)
        
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
                rho_chunk = jax.lax.dynamic_slice(rho_g, (start,), (slice_len,))
                
                # grad(r_batch, r2_chunk) -> (B, B_inner, 3)
                grads = self.jastrow_factor.grad_r_batch(r_batch, r2_chunk, jastrow_params)
                
                # sum_j w_j rho_j grad_ij
                weighted_grads = grads * (w_chunk * rho_chunk)[None, :, None]
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
            rho_g_padded = jnp.pad(rho_g, ((0, padding),))
        else:
            grid_padded = self.grid_points
            weights_padded = self.weights
            rho_g_padded = rho_g
            
        def compute_W_batch_padded(r_batch, grid_p, weights_p, rho_p):
            def inner_scan_p(carry, chunk_idx):
                start = chunk_idx * batch_size
                # Fixed size slice on padded arrays
                r2_chunk = jax.lax.dynamic_slice(grid_p, (start, 0), (batch_size, 3))
                w_chunk = jax.lax.dynamic_slice(weights_p, (start,), (batch_size,))
                rho_chunk = jax.lax.dynamic_slice(rho_p, (start,), (batch_size,))
                
                grads = self.jastrow_factor.grad_r_batch(r_batch, r2_chunk, jastrow_params)
                weighted_grads = grads * (w_chunk * rho_chunk)[None, :, None]
                chunk_sum = jnp.sum(weighted_grads, axis=1)
                return carry + chunk_sum, None
            
            n_chunks = (grid_p.shape[0]) // batch_size
            W_batch, _ = jax.lax.scan(inner_scan_p, jnp.zeros((r_batch.shape[0], 3), dtype=jnp.complex128), jnp.arange(n_chunks))
            return W_batch

        def outer_scan(carry, batch_idx):
            start = batch_idx * batch_size
            # Slice from padded grid
            r_batch = jax.lax.dynamic_slice(grid_padded, (start, 0), (batch_size, 3))
            W_batch = compute_W_batch_padded(r_batch, grid_padded, weights_padded, rho_g_padded)
            return carry, W_batch 
            
        _, W_all = jax.lax.scan(outer_scan, None, jnp.arange(n_batches))
        W_all = W_all.reshape(-1, 3)[:N_grid] # Flatten and trim padding if any
        
        # 3. Compute V_3b_direct(r) = |W(r)|^2
        V_3b_g = jnp.sum(W_all**2, axis=1) # (N_grid,)
        
        # 4. Integrate to get Fock matrix elements
        # F_mn = \int \phi_m(r) \phi_n(r) V_3b(r) dr
        #      = \sum_g w_g \phi_m(g) \phi_n(g) V_3b(g)
        
        # (N_orb, N_grid) * (N_grid,) -> (N_orb, N_grid)
        weighted_phi = self.rho * (self.weights * V_3b_g)[None, :]
        F_3b = jnp.dot(weighted_phi, self.rho.T)
        
        return F_3b

    def get_3b_fock_full(self, jastrow_params, dm1):
        """Get 3-body Fock matrix correction (full tensor calculation).
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis)
            
        Returns:
            Fock matrix contribution (N, N)
        """
        # 1. Compute rho on grid
        rho_g = jnp.einsum('mg,ng,mn->g', self.rho, self.rho, dm1)
        
        # 2. Compute W(r) on grid using full broadcasting
        # W(r_i) = \sum_j w_j rho(r_j) \nabla_i u(r_i, r_j)
        
        # grads: (N_grid, N_grid, 3)
        # This might be memory intensive for large grids!
        grads = self.jastrow_factor.grad_r_batch(self.grid_points, self.grid_points, jastrow_params)
        
        # Weighted rho: (N_grid,)
        w_rho = self.weights * rho_g
        
        # Contract: (N_i, N_j, 3) * (N_j,) -> (N_i, 3)
        W_all = jnp.einsum('ijc,j->ic', grads, w_rho)
        
        # 3. Compute V_3b_direct(r) = |W(r)|^2
        V_3b_g = jnp.sum(W_all**2, axis=1) # (N_grid,)
        
        # 4. Integrate to get Fock matrix elements
        weighted_phi = self.rho * (self.weights * V_3b_g)[None, :]
        F_3b = jnp.dot(weighted_phi, self.rho.T)
        
        return F_3b




@struct.dataclass
class ISDFTC(TC):
    """JAX implementation of Transcorrelated method using ISDF.
    
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
        # We use the same rank for both rho and grad for simplicity, 
        # matching the numpy implementation default behavior
        C_rho, xi_rho, C_grad, xi_grad, pivots = df.isdf_decompose(
            tc_obj.rho, tc_obj.nabla_rho, n_rank, n_rank, weights=tc_obj.weights
        )
        
        return cls(
            grid_points=tc_obj.grid_points,
            weights=tc_obj.weights,
            rho=tc_obj.rho,
            nabla_rho=tc_obj.nabla_rho,
            n_orb=tc_obj.n_orb,
            grid_lvl=tc_obj.grid_lvl,
            jastrow_factor=tc_obj.jastrow_factor,
            mo_coeff=tc_obj.mo_coeff,
            C_rho=C_rho,
            xi_rho=xi_rho,
            C_grad=C_grad,
            xi_grad=xi_grad,
            pivots=pivots
        )

    def get_2b(self, jastrow_params, block_str=None, ranges=None, batch_size=1000):
        """Calculate TC correction terms using ISDF."""
        if block_str is not None or ranges is not None:
            raise NotImplementedError("Block calculation not implemented for ISDFTC yet.")
            
        # Use ISDF method
        k_nabla = kmat_jax.calc_K1_isdf(
            self.C_rho,
            self.xi_rho,
            self.C_grad,
            self.xi_grad,
            self.jastrow_factor,
            jastrow_params,
            self.grid_points,
            self.weights,
            batch_size=batch_size
        )
        # k_laplacian = -(k_nabla + k_nabla^T)
        # We reshape first to swap axes correctly
        k_nabla = k_nabla.reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        k_laplacian = -(k_nabla + k_nabla.swapaxes(0, 1))
        
        k_square = kmat_jax.calc_K3_isdf(
            self.C_rho,
            self.xi_rho,
            self.jastrow_factor,
            jastrow_params,
            self.grid_points,
            self.weights,
            batch_size=batch_size
        )
        
        # Reshape results
        # k_nabla is already reshaped
        # k_laplacian is already reshaped
        k_square = k_square.reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        # Combine results
        result = 0.5 * (k_laplacian + k_square)
        result += k_nabla
        result += result.transpose(2, 3, 0, 1)
        
        return -result
