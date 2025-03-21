"""JAX implementation of Transcorrelated method."""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import dft, ao2mo
from . import kmat as kmat_jax

class TC:
    """JAX implementation of Transcorrelated method."""
    
    def __init__(self, mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
        """Initialize the TC object.
        
        Args:
            mf: PySCF mean-field object
            jastrow_factor: JAX Jastrow factor instance without parameters
            mo_coeff: Optional molecular orbital coefficients
            grid_lvl: Grid level for numerical integration
        """
        self.mf = mf
        self.mol = mf.mol
        self.mo_coeff = mo_coeff if mo_coeff is not None else mf.mo_coeff
        self.n_orb = self.mo_coeff.shape[1]
        self.verbose = mf.verbose if hasattr(mf, 'verbose') else 0
        self.jastrow_factor = jastrow_factor
        
        # Cache for evaluated quantities
        self._rho = None
        self._nabla_rho = None
        self._eri1 = None
        
        # Initialize grid
        self._init_grid(grid_lvl)
        self._eval_basis_on_grid()
    
    def _init_grid(self, grid_lvl=2):
        """Initialize numerical integration grid using PySCF."""
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = grid_lvl
        grids.build()
        
        # Convert to JAX arrays immediately
        self.grid_points = jnp.asarray(grids.coords)
        self.weights = jnp.asarray(grids.weights)
    
    def _eval_basis_on_grid(self):
        """Evaluate basis functions using PySCF's numint."""
        if self._rho is not None and self._nabla_rho is not None:
            return self._rho, self._nabla_rho
        
        # Use PySCF to evaluate AOs with numpy arrays
        ao = dft.numint.eval_ao(self.mol, jax.device_get(self.grid_points), deriv=1)
        ao_values = ao[0].T  # (N_ao, N_grid)
        ao_gradients = ao[1:4].transpose(2, 1, 0)  # (N_ao, N_grid, 3)
        
        # Transform to MO basis
        if self.mo_coeff is not None:
            mo_values = np.dot(self.mo_coeff.T, ao_values)
            mo_gradients = np.einsum('ji,jnc->inc', self.mo_coeff, ao_gradients)
            ao_values, ao_gradients = mo_values, mo_gradients
        
        # Cache as JAX arrays
        self._rho = jnp.asarray(ao_values)
        self._nabla_rho = jnp.asarray(ao_gradients)
        
        return self._rho, self._nabla_rho
    
    @partial(jax.jit, static_argnums=(0,))
    def get_2b(self, jastrow_params, dm1=None, dm2=None):
        """Calculate two-body terms K1 + K2 + K3.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor
            dm1: Optional one-body density matrix
            dm2: Optional two-body density matrix
        """
        # Get orbital values on grid
        rho, nabla_rho = self._eval_basis_on_grid()
        
        # Prepare paired quantities
        rho_paired = jnp.einsum('in,jn->ijn', rho, rho).reshape(-1, len(self.weights))
        rho_nabla_rho_paired = jnp.einsum('pnd,rn->prnd', nabla_rho, rho).reshape(-1, len(self.weights), 3)
        
        # Compute K terms with explicit parameter passing
        k_nabla = kmat_jax.calc_K1(
            rho_paired, rho_nabla_rho_paired,
            self.jastrow_factor, jastrow_params,
            self.grid_points, self.weights
        )
        
        k_square = kmat_jax.calc_K3(
            rho_paired, self.jastrow_factor, jastrow_params,
            self.grid_points, self.weights
        )
        
        # Reshape results
        k_nabla = k_nabla.reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        k_laplacian = -(k_nabla + k_nabla.swapaxes(0,1))
        k_square = k_square.reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        # Combine results
        result = 0.5 * (k_laplacian + k_square)
        result += k_nabla
        result += result.transpose(2, 3, 0, 1)
        
        # Get ERI and convert at the end
        eri1 = jnp.asarray(self.get_eri1())
        return eri1 - result
    
    def get_eri1(self):
        """Get two-body integrals in MO basis."""
        if self._eri1 is None:
            # Compute ERI using PySCF with numpy arrays
            eri1 = ao2mo.incore.full(self.mf._eri, self.mo_coeff, compact=False)
            eri1 = ao2mo.restore(1, eri1, self.mo_coeff.shape[1])
            self._eri1 = eri1
        return self._eri1

    def get_3b(self):
        """Compute all three-body integrals."""
        raise NotImplementedError("JAX implementation pending")
