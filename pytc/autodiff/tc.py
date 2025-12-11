"""JAX implementation of Transcorrelated method."""

from functools import partial
from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from pyscf import dft
from . import kmat as kmat_jax

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
            mo_coeff=jnp.asarray(mo_coeff)
        )
    
    @jax.jit
    def get_2b(self, jastrow_params):
        """Calculate TC correction terms (K1 + K2 + K3).
        
        Args:
            jastrow_params: Parameters for the Jastrow factor
            
        Returns:
            jnp.ndarray: The TC correction term (negative of the K terms sum)
                         such that H_TC = H_MF + correction
        """
        # Prepare paired quantities
        # rho shape: (n_orb, n_grid)
        rho_paired = jnp.einsum('in,jn->ijn', self.rho, self.rho).reshape(-1, len(self.weights))
        rho_nabla_rho_paired = jnp.einsum('pnd,rn->prnd', self.nabla_rho, self.rho).reshape(-1, len(self.weights), 3)
        
        # Compute K terms
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
        
        return -result

    def get_3b(self):
        """Compute all three-body integrals."""
        raise NotImplementedError("JAX implementation pending")
