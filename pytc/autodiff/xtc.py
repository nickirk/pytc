"""JAX implementation of X transcorrelated methods."""

from functools import partial, reduce
from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from .tc import TC
from . import tc_helper

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
    
    @jax.jit
    def calc_v_vector(self, rho_paired, jastrow_params, batch_size=1000, inner_batch_size=1000):
        """Calculate V_qt(r₁) vector with memory-efficient nested batching.
        
        Args:
            rho_paired: Array of shape (Nb*Nb, N_grid)
            jastrow_params: Parameters for the Jastrow factor
            batch_size: Number of r1 points to process at once (outer loop)
            inner_batch_size: Number of r2 points to process at once (inner loop)
        """
        Nb2, N_grid = rho_paired.shape
        weighted_rho = rho_paired * self.weights[None, :]

        # Pad grid to multiple of batch_size for outer loop
        padded_size = ((N_grid + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(self.grid_points, ((0, padded_size - N_grid), (0, 0)))
        
        padded_weighted_rho = jnp.pad(weighted_rho, ((0, 0), (0, padded_size - N_grid)))
        
        batched_grid = padded_grid.reshape(-1, batch_size, 3)
        
        # Inner loop batches
        inner_batched_grid = padded_grid.reshape(-1, inner_batch_size, 3)
        inner_batched_rho = padded_weighted_rho.reshape(Nb2, -1, inner_batch_size) # (Nb2, n_batches, inner_bs)

        # Create a mask for valid grid points
        grid_mask = jnp.arange(padded_size) < N_grid
        inner_batched_mask = grid_mask.reshape(-1, inner_batch_size)

        def outer_scan_body(carry, r1_batch):
            # r1_batch: (batch_size, 3)
            
            def inner_scan_body(inner_carry, args):
                r2_batch, rho_batch, mask_batch = args
                # r2_batch: (inner_batch_size, 3)
                # rho_batch: (Nb2, inner_batch_size)
                # mask_batch: (inner_batch_size,)
                
                # Compute gradients for this block of r1 and r2
                # Shape: (batch_size, inner_batch_size, 3)
                
                @partial(jax.vmap, in_axes=(None, 0))
                def grad_fn(r1, r2):
                    return self.jastrow_factor.grad_r(r1[None], r2[None], jastrow_params)[0]
            
                grads = jax.vmap(grad_fn, in_axes=(0, None))(r1_batch, r2_batch)
                
                # Apply mask to grads
                grads = grads * mask_batch[None, :, None]
                
                # Contract: sum_j rho(j) * grad(i, j)
                # rho_batch: (Nb2, inner_batch_size)
                # grads: (batch_size, inner_batch_size, 3)
                grads_reshaped = grads.transpose(1, 0, 2).reshape(inner_batch_size, -1)
                
                # rho_batch @ grads_reshaped -> (Nb2, batch_size * 3)
                block_contribution_flat = jnp.dot(rho_batch, grads_reshaped)
                
                # Reshape back to (Nb2, batch_size, 3)
                block_contribution = block_contribution_flat.reshape(Nb2, batch_size, 3)
                
                return inner_carry + block_contribution, None

            # Initialize accumulation for this r1_batch
            init_val = jnp.zeros((Nb2, batch_size, 3))
            
            # Scan over inner batches
            final_val, _ = jax.lax.scan(inner_scan_body, init_val, (inner_batched_grid, inner_batched_rho.transpose(1, 0, 2), inner_batched_mask))
            
            return carry, final_val

        _, results = jax.lax.scan(outer_scan_body, 0, batched_grid)
        
        # results: (n_outer_batches, Nb2, batch_size, 3)
        # Transpose to (Nb2, n_outer_batches, batch_size, 3) -> reshape -> (Nb2, padded_size, 3)
        return results.transpose(1, 0, 2, 3).reshape(Nb2, padded_size, 3)[:, :N_grid, :]

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
    
    @jax.jit
    def calc_delta_U(self, v_vector, rho_paired, dm1):
        """Calculate delta_U matrix."""
        nb = dm1.shape[1]
        
        V = jnp.reshape(v_vector, (nb, nb, -1, 3))
        rho = jnp.reshape(rho_paired, (nb, nb, -1))
        rho_weighted = rho * self.weights[None, None, :]
        
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
        return result + result.transpose(2, 3, 0, 1)

    def _get_mf_dm(self):
        """Get mean-field 1-body density matrix for closed shell system."""
        dm1 = jnp.diag(self.mo_occ)/2
        return dm1

    def get_delta_h(self, jastrow_params, dm1=None):
        """Get or compute delta_h."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        delta_U = self.get_delta_U(jastrow_params, dm1)
        return self._calc_delta_h(delta_U, dm1)
    
    def get_delta_U(self, jastrow_params, dm1):
        """Get delta_U matrix."""
        n_orb = self.n_orb
        rho_paired = jnp.einsum('in,jn->ijn', self.rho, self.rho).reshape((n_orb * n_orb, -1))
        v_vector = self.calc_v_vector(rho_paired, jastrow_params)
        return self.calc_delta_U(v_vector, rho_paired, dm1)

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
