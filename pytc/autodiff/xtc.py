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
            mo_occ=mo_occ,
            energy_nuc=energy_nuc
        )
    
    @property
    def n_grid(self):
        """Number of grid points."""
        return len(self.grid_points)
    
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
        n_orb, n_grid = rho.shape
        nb2 = n_orb * n_orb
        
        # Pad grid for r2 scan
        padded_size = ((n_grid + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(self.grid_points, ((0, padded_size - n_grid), (0, 0)))
        padded_weights = jnp.pad(weights, (0, padded_size - n_grid))
        padded_rho = jnp.pad(rho, ((0, 0), (0, padded_size - n_grid)))
        
        # Reshape for scanning
        r2_batches = padded_grid.reshape(-1, batch_size, 3)
        weights_batches = padded_weights.reshape(-1, batch_size)
        rho_batches = padded_rho.reshape(n_orb, -1, batch_size)
        
        def scan_body(carry, args):
            r2_batch, w_batch, rho_batch_r2 = args
            
            # Compute rho_paired for this r2 batch
            rho_paired_r2 = jnp.einsum('in,jn->ijn', rho_batch_r2, rho_batch_r2).reshape(nb2, -1)
            weighted_rho_r2 = rho_paired_r2 * w_batch[None, :]
            
            # Compute gradients
            grads = self.jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            
            # Contract: sum_{r2} rho(r2) * grad(r1, r2)
            term = jnp.einsum('ki,oij->okj', weighted_rho_r2, grads)
            
            return carry + term, None

        init_val = jnp.zeros((len(r1_batch), nb2, 3))
        final_val, _ = jax.lax.scan(scan_body, init_val, (r2_batches, weights_batches, rho_batches.transpose(1, 0, 2)))
        
        return final_val.transpose(1, 0, 2)

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

    def get_delta_U(self, jastrow_params, dm1=None, batch_size=1000):
        """Get delta_U matrix with memory-efficient batching and multi-GPU support."""
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
        
        # Define n_orb and nb2 for closure
        n_orb = self.n_orb
        nb2 = n_orb * n_orb

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
            rho_batches = rho_r1_batched.reshape(n_orb, -1, batch_size)
            
            def scan_body(carry, args):
                r1_batch, w_batch, rho_batch = args
                
                # Compute rho_paired for this batch
                rho_paired_batch = jnp.einsum('in,jn->ijn', rho_batch, rho_batch).reshape(nb2, -1)
                
                # Compute V vector for this batch (integrating over all r2)
                v_batch = self._calc_v_batch(r1_batch, self.rho, self.weights, jastrow_params, batch_size)
                
                # Compute contribution to delta_U
                contrib = self.calc_delta_U(v_batch, rho_paired_batch, dm1, w_batch)
                
                return carry + contrib, None

            init_val = jnp.zeros((n_orb, n_orb, n_orb, n_orb))
            
            local_delta_U, _ = jax.lax.scan(scan_body, init_val, (r1_batches, weights_batches, rho_batches.transpose(1, 0, 2)))
            
            # Sum results across devices
            total_delta_U = jax.lax.psum(local_delta_U, axis_name='devices')
            return total_delta_U

        # Execute pmap
        pmapped_compute = jax.pmap(compute_on_device, axis_name='devices')
        
        delta_U_replicated = pmapped_compute(sharded_grid_r1, sharded_weights_r1, sharded_rho_r1)
        
        total_delta_U = delta_U_replicated[0]
        
        # Symmetrize
        return total_delta_U + total_delta_U.transpose(2, 3, 0, 1)

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

    @jax.jit
    def get_2b(self, jastrow_params, dm1=None):
        """Compute two-body integrals correction."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        
        # Get TC's two-body correction (negative of K terms)
        tc_correction = super().get_2b(jastrow_params)
        
        # Add delta_U
        delta_U = self.get_delta_U(jastrow_params, dm1)
        
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
        """Get delta_U matrix using ISDF."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        return self._calc_delta_U_isdf(jastrow_params, dm1, batch_size)

    def _calc_delta_U_isdf(self, jastrow_params, dm1, batch_size=1000):
        """Calculate ΔU^{QS}_{PR} using ISDF intermediates with batched processing."""
        from . import kmat as kmat_jax
        
        N_grid = self.n_grid
        Nb = self.n_orb
        N_rank = self.C_rho.shape[1]
        
        # Reshape C_rho for easier contraction
        C_rho_reshaped = self.C_rho.reshape(Nb, Nb, N_rank)
        
        # Initialize accumulator for M tensor
        M = jnp.zeros((N_rank, N_rank, N_rank))
        
        # Process r2 points in batches
        def scan_body(carry, args):
            r2_batch, w_batch, xi_rho_batch = args
            M_acc = carry
            
            grads = self.jastrow_factor.grad_r_batch(self.grid_points, r2_batch, jastrow_params) # (N_grid, batch, 3)
            
            G = jnp.einsum('j,bj,jic->bic', self.weights, self.xi_rho, grads) # (N_rank, batch, 3)
            
            K = jnp.einsum('bic,dic->bdi', G, G) # (N_rank, N_rank, batch)
            
            weighted_xi = xi_rho_batch * w_batch[None, :] # (N_rank, batch)
            
            M_update = jnp.einsum('ai,bdi->abd', weighted_xi, K)
            
            return M_acc + M_update, None

        # Prepare batched inputs
        padded_size = ((N_grid + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(self.grid_points, ((0, padded_size - N_grid), (0, 0)))
        padded_weights = jnp.pad(self.weights, (0, padded_size - N_grid))
        padded_xi_rho = jnp.pad(self.xi_rho, ((0, 0), (0, padded_size - N_grid)))
        
        r2_batches = padded_grid.reshape(-1, batch_size, 3)
        w_batches = padded_weights.reshape(-1, batch_size)
        xi_rho_batches = padded_xi_rho.reshape(N_rank, -1, batch_size).transpose(1, 0, 2)
        
        M, _ = jax.lax.scan(scan_body, M, (r2_batches, w_batches, xi_rho_batches))
        
        Gb = jnp.einsum('tub,tu->b', C_rho_reshaped, dm1)
        
        GM = jnp.einsum('b,abd->ad', Gb, M)
        
        T = jnp.einsum('pqa,ad->pqd', C_rho_reshaped, GM)
        
        term1 = 2 * jnp.einsum('pqd,rsd->pqrs', T, C_rho_reshaped)
        
        L = jnp.einsum('tu,tsc->usc', dm1, C_rho_reshaped)
        
        T2 = jnp.einsum('abc,usc->usab', M, L)
        
        Q = jnp.einsum('usab,rub->rsa', T2, C_rho_reshaped)
        
        term2 = -jnp.einsum('pqa,rsa->pqrs', C_rho_reshaped, Q)
        
        Mt = M + M.transpose(2, 1, 0)
        T3 = jnp.einsum('cab,usc->usab', Mt, L)
        Q3 = jnp.einsum('usab,rub->rsa', T3, C_rho_reshaped)
        term3 = -jnp.einsum('pqa,rsa->pqrs', C_rho_reshaped, Q3)
        
        term4_inter = jnp.einsum('b,bac->ac', Gb, M)
        term4_inter = jnp.einsum('ac,pqa->pqc', term4_inter, C_rho_reshaped)
        term4 = jnp.einsum('pqc,rsc->pqrs', term4_inter, C_rho_reshaped)
        term4_inter = jnp.einsum('b,bac->ac', Gb, M)
        term4_inter = jnp.einsum('ac,pqa->pqc', term4_inter, C_rho_reshaped)
        term4 = jnp.einsum('pqc,rsc->pqrs', term4_inter, C_rho_reshaped)
        
        result = term1 + term2 + term3 + term4
        
        # Symmetrize
        return -(result + result.transpose(2, 3, 0, 1))

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
