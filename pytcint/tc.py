"""This module implements the Transcorrelated method."""

import numpy as np
from pyscf import dft
from pyscf.dft import gen_grid
from . import kmat
from . import lmat

class TC:
    """Transcorrelated method implementation."""
    
    def __init__(self, mf, mo_coeff=None, grid_lvl=2):
        """Initialize the TC object.
        
        Args:
            mf: PySCF mean-field object
            mo_coeff: Optional molecular orbital coefficients. If None, uses mf.mo_coeff
            grid_lvl: Grid level for numerical integration (default: 2)
        """
        self.mf = mf
        self.mol = mf.mol
        self.mo_coeff = mo_coeff if mo_coeff is not None else mf.mo_coeff
        self.n_orb = self.mo_coeff.shape[1]
        # Initialize grid
        self.grid_lvl = grid_lvl
        self.grid_points = None
        self.weights = None
        self._init_grid(grid_lvl)
        
        # Cache for evaluated quantities
        self._cache = {}
    
    def _init_grid(self, grid_lvl=2):
        """Initialize numerical integration grid.
        
        Uses PySCF's grid generation for DFT to create atom-centered grids
        with Treutler-Ahlrichs radial grids and Lebedev angular grids.
        
        Args:
            grid_lvl: Grid level for accuracy (0-9, higher is more accurate)
        """
        # Create grid object
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = grid_lvl
        grids.build()
        
        # Store grid points and weights
        self.grid_points = grids.coords
        self.weights = grids.weights
    
    def _eval_basis_on_grid(self):
        """Evaluate basis functions and their gradients on the grid points."""
        if 'ao_values' in self._cache:
            return self._cache['ao_values'], self._cache['ao_gradients']
        
        # Evaluate AO values and gradients on grid
        ao = dft.numint.eval_ao(self.mol, self.grid_points, deriv=1)
        ao_values = ao[0]  # Shape: (N_grid, N_ao)
        ao_gradients = ao[1:4].transpose(1, 0, 2)  # Shape: (N_grid, 3, N_ao)
        
        # Transform to MO basis if mo_coeff is available
        if self.mo_coeff is not None:
            mo_values = np.dot(ao_values, self.mo_coeff)
            mo_gradients = np.dot(ao_gradients, self.mo_coeff)
            ao_values, ao_gradients = mo_values, mo_gradients
        
        # Cache results
        self._cache['ao_values'] = ao_values
        self._cache['ao_gradients'] = ao_gradients
        
        return ao_values, ao_gradients
    
    def get_2b(self, jastrow_factor):
        """Compute all two-body integrals involving the Jastrow factor.
        
        Assembles the K^{pq}_{rs} integrals including:
        - ∇u·∇ terms
        - ∇²u terms
        - (∇u)² terms
        
        Args:
            jastrow_factor: Instance of Jastrow class
            
        Returns:
            Dictionary containing different types of two-body integrals
        """
        # Get basis functions and gradients on grid
        rho, nabla_rho = self._eval_basis_on_grid()  # rho: (N_grid, N_ao), nabla_rho: (N_grid, 3, N_ao)
        n_orb = rho.shape[1]
        
        # Prepare paired indices with N_grid as first dimension
        # For rho: (N_grid, N_ao) -> (N_grid, N_ao*N_ao)
        rho_paired = np.einsum('ni,nj->nij', rho, rho).reshape(rho.shape[0], -1)
        
        # For nabla_rho: (N_grid, 3, N_ao) -> (N_grid, N_ao*N_ao, 3)
        # Compute outer product for all components at once
        # nabla_rho: (N_grid, 3, N_ao), rho: (N_grid, N_ao) -> (N_grid, N_ao*N_ao, 3)
        nabla_rho_paired = np.einsum('ndi,nj->nijd', nabla_rho, rho).reshape(rho.shape[0], -1, 3)
        
        # Compute different types of integrals
        k_nabla = self._get_K1(rho_paired, nabla_rho_paired, jastrow_factor)
        k_laplacian = self._get_K2(rho_paired, nabla_rho_paired, jastrow_factor)
        k_square = self._get_K3(rho_paired, jastrow_factor)
        
        result = 0.5 * (k_laplacian + k_square) + k_nabla

        # symmetrize wrt r1 and r2
        result += result.transpose(2, 3, 0, 1)
        return result
    
    def _get_K1(self, rho_paired, nabla_rho_paired, jastrow_factor):
        """Compute the K1 integral."""
        return kmat.calc_K1(rho_paired, nabla_rho_paired, self.grid_points, self.weights, jastrow_factor).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
    
    def _get_K2(self, rho_paired, nabla_rho_paired, jastrow_factor):
        """Compute the K2 integral."""
        return kmat.calc_K2(rho_paired, nabla_rho_paired, self.grid_points, self.weights, jastrow_factor).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
    
    def _get_K3(self, rho_paired, jastrow_factor):
        """Compute the K3 integral."""
        return kmat.calc_K3(rho_paired, self.grid_points, self.weights, jastrow_factor).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)

    def get_3b(self, jastrow_factor):
        """Compute all three-body integrals involving the Jastrow factor. Use the lmat module.
        """
        return lmat.calc_l_matrix_symmetric(self.mol, self.mo_coeff, self.grid_points, self.weights, jastrow_factor)
