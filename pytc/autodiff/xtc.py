"""JAX implementation of X transcorrelated methods."""

from functools import partial, reduce
import numpy as np
import jax
import jax.numpy as jnp
from .tc import TC

class XTC(TC):
    """JAX implementation of extended transcorrelated methods."""
    
    def __init__(self, mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
        """Initialize the XTC calculator.
        
        Args:
            jastrow_factor: JAX Jastrow instance
            grid_points: Array of shape (N_grid, 3)
            weights: Array of shape (N_grid,)
        """
        self.jastrow_factor = jastrow_factor
        super().__init__(mf, jastrow_factor, mo_coeff, grid_lvl)
        self._delta_U = None  # Cache for delta_U
        self._delta_h = None  # Cache for delta_h
    
    @property
    def n_grid(self):
        """Number of grid points."""
        return len(self.grid_points)
    
    @partial(jax.jit, static_argnums=(0,2))  # Only self and batch_size should be static
    def calc_v_vector(self, rho_paired, batch_size=1000):
        """Calculate V_qt(r₁) vector.
        
        Args:
            rho_paired: Array of shape (Nb*Nb, N_grid)
            batch_size: Number of grid points to process at once
        
        Returns:
            Array of shape (Nb*Nb, N_grid, 3)
        """
        Nb2, N_grid = rho_paired.shape
        weighted_rho = rho_paired * self.weights[None, :]

        # Pad grid to multiple of batch_size
        padded_size = ((N_grid + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(self.grid_points, ((0, padded_size - N_grid), (0, 0)))
        padded_weighted_rho = jnp.pad(weighted_rho, ((0, 0), (0, padded_size - N_grid)))
        batched_grid = padded_grid.reshape(-1, batch_size, 3)

        def scan_body(carry, batch_grid):
            @partial(jax.vmap, in_axes=(None, 0))
            def grad_fn(r1, r2):
                return self.jastrow_factor.grad_r(r1, r2)
        
            grads = jax.vmap(grad_fn, in_axes=(0, None))(batch_grid, padded_grid)
            contribution = jnp.einsum('nj,ijk->nik', padded_weighted_rho, grads)
            return carry, contribution
        
        _, results = jax.lax.scan(scan_body, None, batched_grid)
        return results.reshape(rho_paired.shape[0], -1, 3)[:,:N_grid,:]

    @partial(jax.jit, static_argnums=(0,))  # Only self should be static
    def _calc_delta_h(self, delta_U=None, dm1=None):
        """Calculate δh using δU and density matrix.
        
        δh^q_p = -1/2 * (δU^{qs}_{pr} - δU^{sq}_{pr}) * γ^r_s
        """
        if delta_U is None:
            delta_U = self._calc_delta_U()  # To be implemented
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Calculate δh using einstein summation
        term1 = 2*jnp.einsum('qpsr,rs->qp', delta_U, dm1)
        term2 = jnp.einsum('spqr,rs->qp', delta_U, dm1)
        delta_h = -0.5 * (term1 - term2)
        return delta_h
    
    @partial(jax.jit, static_argnums=(0,))  # Only self should be static
    def calc_delta_U(self, v_vector, rho_paired, dm1):
        """Calculate delta_U matrix.
        
        Args:
            v_vector: Array from calc_v_vector
            rho_paired: Array of shape (Nb*Nb, N_grid)
            dm1: One-particle density matrix
        
        Returns:
            Array of shape (Nb, Nb, Nb, Nb)
        """
        nb = int(np.sqrt(v_vector.shape[0]))
        V = v_vector.reshape(nb, nb, -1, 3)

        rho = rho_paired.reshape(nb, nb, -1)
        rho_weighted = rho * self.weights[None, None, :]
        
        # Use existing vectorized computations
        compute_W = jax.vmap(lambda V, dm: 2 * jnp.einsum('utx,tu->x', V, dm), 
                            in_axes=(2, None), out_axes=1)
        W = compute_W(V, dm1)
        
        # ... rest of the delta_U calculation remains the same ...
        compute_Vbar = jax.vmap(lambda W, V: jnp.einsum('x,srx->sr', W, V), 
                               in_axes=(0, 2))
        Vbar = compute_Vbar(W, V)
        
        compute_X = jax.vmap(lambda V, dm: jnp.einsum('stx,tu->sux', V, dm), 
                            in_axes=2, out_axes=2)
        X = compute_X(V, dm1)
        
        compute_Zbar = jax.vmap(lambda V, X: jnp.einsum('urx,sux->sr', V, X), 
                               in_axes=2)
        Zbar = compute_Zbar(V, X)
        
        Wbar = 2 * jnp.einsum('uti,tu->i', rho_weighted, dm1)
        
        compute_Y = jax.vmap(lambda V, dm: jnp.einsum('urx,tu->trx', V, dm), 
                            in_axes=2, out_axes=2)
        Y = compute_Y(V, dm1)
        
        compute_G = jax.vmap(lambda rho, X, Y: 
                            jnp.einsum('ur,sux->srx', rho, X) + 
                            jnp.einsum('trx,st->srx', Y, rho),
                            in_axes=(2, 2, 2))
        G = compute_G(rho_weighted, X, Y)
        
        A = Vbar - Zbar
        B = 0.5 * Wbar[None, None, :, None] * V - G
        
        term1 = jnp.einsum('qpi,sri->qpsr', rho_weighted, A)
        term2 = jnp.einsum('qpix,srix->qpsr', V, B)
        
        result = -(term1 + term2)
        return result + result.transpose(2, 3, 0, 1)

    def _get_mf_dm(self):
        """Get mean-field 1-body density matrix for closed shell system.
        
        Returns:
            numpy.ndarray: Diagonal density matrix with 2.0 for occupied orbitals
        """
        #nelec = self.mol.nelectron
        #nocc = nelec // 2
        #dm1 = np.zeros((self.n_orb, self.n_orb))
        #np.fill_diagonal(dm1[:nocc, :nocc], 2.0)
        dm1 = jnp.diag(self.mf.mo_occ)
        return dm1

    def get_delta_h(self, dm1=None):
        """Get or compute delta_h with caching."""
        if self._delta_h is None:
            if dm1 is None:
                dm1 = self._get_mf_dm()
            delta_U = self.get_delta_U(dm1)
            self._delta_h = self._calc_delta_h(delta_U, dm1)
        return self._delta_h
    
    def get_delta_U(self, dm1):
        """Get delta_U matrix."""
        if self._delta_U is None:
            # Convert numpy arrays to jax arrays explicitly
            rho = jnp.asarray(self._rho)
            rho_paired = jnp.einsum('in,jn->ijn', rho, rho).reshape(-1, rho.shape[1])
            v_vector = self.calc_v_vector(rho_paired)
            self._delta_U = self.calc_delta_U(v_vector, rho_paired, dm1)
        return self._delta_U

    def get_1b(self, dm1=None, dm2=None):
        """Get one-body operator h1e = T + V + δh."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Get core Hamiltonian and transform to MO basis
        h1e = self.mf.get_hcore()
        h1e = jnp.asarray(reduce(np.dot, (self.mo_coeff.T, h1e, self.mo_coeff)))
        
        # Add delta_h using cached value
        h1e += self.get_delta_h(dm1)
        
        return h1e

    def get_2b(self, dm1=None):
        """Compute two-body integrals."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        
        # Get TC's two-body contribution
        result = super().get_2b()
        
        # Add cached delta_U
        result += self.get_delta_U(dm1)
        return result

    def get_3b(self):
        """Get three-body extended correlation."""
        raise NotImplementedError("JAX implementation pending")
