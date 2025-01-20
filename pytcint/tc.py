"""This module implements the Transcorrelated method."""

import numpy as np
from functools import partial
from pyscf import dft
from . import kmat
from . import lmat

# Create an optimized einsum that always uses the 'optimal' path
einsum = partial(np.einsum, optimize='optimal')


class TC:
    """Transcorrelated method implementation."""
    
    def __init__(self, mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
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
        self.jastrow_factor = jastrow_factor
        self._init_grid(grid_lvl)
        
        # Cache for evaluated quantities
        self._cache = {}
        # Add cache for intermediates
        self._rho = None
        self._nabla_rho = None
        self._u_gradients = None
        self._rho_paired = None
    
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
        if 'mo_values' in self._cache and 'mo_gradients' in self._cache:
            return self._cache['mo_values'], self._cache['mo_gradients']
        
        # Evaluate AO values and gradients on grid
        ao = dft.numint.eval_ao(self.mol, self.grid_points, deriv=1)
        ao_values = ao[0].T  # Shape: (N_ao, N_grid)
        # Shape: (N_ao, N_grid, 3)
        ao_gradients = ao[1:4].transpose(2, 1, 0)  
        
        # Transform to MO basis if mo_coeff is available
        if self.mo_coeff is not None:
            # Shape: (N_mo, N_grid)
            mo_values = np.dot(self.mo_coeff.T, ao_values)
            # Shape: (N_mo, N_grid, 3)
            mo_gradients = einsum('ji,jnc->inc', self.mo_coeff.T, ao_gradients)
            ao_values, ao_gradients = mo_values, mo_gradients
        
        # Cache results
        self._cache['mo_values'] = ao_values
        self._cache['mo_gradients'] = ao_gradients
        
        return ao_values, ao_gradients

    def _get_intermediates(self):
        """Get or compute intermediate quantities with caching."""
        if self._rho is None:
            self._rho, self._nabla_rho = self._eval_basis_on_grid()
            self._u_gradients = self.jastrow_factor.grad(self.grid_points)
            self._rho_paired = einsum('in,jn->ijn', self._rho, self._rho).reshape(-1, self._rho.shape[1])
        return self._rho, self._nabla_rho, self._u_gradients, self._rho_paired
    
    def get_2b(self):
        """Compute all two-body integrals involving the Jastrow factor.
        
        Assembles the K^{pq}_{rs} integrals including:
        - ∇u·∇ terms
        - ∇²u terms
        - (∇u)² terms
        
        Args:
            jastrow_factor: Instance of Jastrow class
            
        Returns:
            Array of shape (n_orb, n_orb, n_orb, n_orb) containing the two-body integrals
        """
        # Get cached intermediates
        rho, nabla_rho, u_gradients, rho_paired = self._get_intermediates()
        
        # Update nabla_rho_paired for new array shapes
        nabla_rho_paired = einsum('inc,jn->ijnc', nabla_rho, rho).reshape(-1, rho.shape[1], 3)
        
        # Compute integrals
        k_nabla = self._get_K1(rho_paired, nabla_rho_paired, u_gradients)
        k_laplacian = self._get_K2(rho_paired, nabla_rho_paired, u_gradients)
        k_square = self._get_K3(rho_paired, u_gradients)
        
        result = 0.5 * (k_laplacian + k_square) + k_nabla
        result += result.transpose(2, 3, 0, 1)
        return result
    
    def _get_K1(self, rho_paired, nabla_rho_paired, u_gradients):
        """Compute the K1 integral <pq|∇u·∇|rs>.
        
        Args:
            rho_paired: Array of shape (N_grid, Nb*Nb) containing orbital products
            nabla_rho_paired: Array of shape (3, N_grid, Nb*Nb) containing gradients
            u_gradients: Array of shape (3, N_grid, N_grid) containing Jastrow gradients
            
        Returns:
            Array of shape (n_orb, n_orb, n_orb, n_orb) containing reshaped K1 integrals
        """
        return kmat.calc_K1(rho_paired, nabla_rho_paired, u_gradients, self.weights).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
    
    def _get_K2(self, rho_paired, nabla_rho_paired, u_gradients):
        """Compute the K2 integral <pq|∇²₁u|rs>.
        
        Args:
            rho_paired: Array of shape (N_grid, Nb*Nb) containing orbital products
            nabla_rho_paired: Array of shape (3, N_grid, Nb*Nb) containing gradients
            u_gradients: Array of shape (3, N_grid, N_grid) containing Jastrow gradients
            
        Returns:
            Array of shape (n_orb, n_orb, n_orb, n_orb) containing reshaped K2 integrals
        """
        return kmat.calc_K2(rho_paired, nabla_rho_paired, u_gradients, self.weights).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
    
    def _get_K3(self, rho_paired, u_gradients):
        """Compute the K3 integral <pq|(∇₁u)²|rs>.
        
        Args:
            rho_paired: Array of shape (N_grid, Nb*Nb) containing orbital products
            u_gradients: Array of shape (3, N_grid, N_grid) containing Jastrow gradients
            
        Returns:
            Array of shape (n_orb, n_orb, n_orb, n_orb) containing reshaped K3 integrals
        """
        return kmat.calc_K3(rho_paired, u_gradients, self.weights).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)

    def get_3b(self, rho_paired, u_gradients):
        """Compute all three-body integrals involving the Jastrow factor. Use the lmat module.
        """
        return lmat.calc_L_symmetric(self.mol, self.mo_coeff, self.grid_points, self.weights, jastrow_factor)