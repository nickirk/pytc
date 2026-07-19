"""This module implements the Transcorrelated method."""

import numpy as np
from functools import partial
from pyscf import dft
from . import lmat
from .df import isdf_decompose_multi, test_accuracy
from ..tc_helper import get_eri
import logging

logger = logging.getLogger(__name__)

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
        self.verbose = mf.verbose if hasattr(mf, 'verbose') else 0  
        self.grid_lvl = grid_lvl
        self.grid_points = None
        self.weights = None
        self.jastrow_factor = jastrow_factor
        self._init_grid(grid_lvl)
        
        self._cache = {}
        self._rho = None
        self._nabla_rho = None
        self._u_gradients = None
        self._isdf_results = None
    
    def _init_grid(self, grid_lvl=2):
        """Initialize numerical integration grid.
        
        Uses PySCF's grid generation for DFT to create atom-centered grids
        with Treutler-Ahlrichs radial grids and Lebedev angular grids.
        
        Args:
            grid_lvl: Grid level for accuracy (0-9, higher is more accurate)
        """
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = grid_lvl
        grids.build()
        
        self.grid_points = grids.coords
        self.weights = grids.weights
    
    def _eval_basis_on_grid(self):
        """Evaluate basis functions and their gradients on the grid points."""
        if 'mo_values' in self._cache and 'mo_gradients' in self._cache:
            return self._cache['mo_values'], self._cache['mo_gradients']
        
        ao = dft.numint.eval_ao(self.mol, self.grid_points, deriv=1)
        ao_values = ao[0].T  # Shape: (N_ao, N_grid)
        # Shape: (N_ao, N_grid, 3)
        ao_gradients = ao[1:4].transpose(2, 1, 0)  
        
        if self.mo_coeff is not None:
            # Shape: (N_mo, N_grid)
            mo_values = np.dot(self.mo_coeff.T, ao_values)
            # Shape: (N_mo, N_grid, 3)
            mo_gradients = einsum('ji,jnc->inc', self.mo_coeff, ao_gradients)
            ao_values, ao_gradients = mo_values, mo_gradients
        
        self._cache['mo_values'] = ao_values
        self._cache['mo_gradients'] = ao_gradients
        
        return ao_values, ao_gradients

    def _get_intermediates(self):
        """Get or compute intermediate quantities."""
        if self._rho is None:
            self._rho, self._nabla_rho = self._eval_basis_on_grid()
        return self._rho, self._nabla_rho

    def isdf(self, n_rank=None):
        """Perform ISDF decomposition on paired densities and gradients.
    
        Args:
            n_rank: Number of interpolation points to use. If None, use 1/4 of grid points.
            
        Returns:
            dict: Dictionary containing:
                'C_rho': Selected columns for rho_paired
                'xi_rho': Interpolation coefficients for rho_paired
                'C_grad': Selected columns for grad_normed  
                'xi_grad': Interpolation coefficients for grad_normed
                'pivots': Fused pivot indices
        """
    
        rho, nabla_rho = self._get_intermediates()
        rho_paired = einsum('in,jn->ijn', rho, rho).reshape(-1, rho.shape[1])
        nabla_rho_paired = einsum('pnc,rn->prnc', nabla_rho, rho).reshape(-1, rho.shape[1], 3)
    
        if n_rank is None:
            n_rank = len(self.weights) // 4
    
        C_rho, xi_rho, C_grad, xi_grad, pivots = isdf_decompose_multi(
            rho_paired, nabla_rho_paired, n_rank, n_rank
        )

        if self.verbose > 4:
            rel_error_rho, abs_error_rho = test_accuracy(rho_paired, C_rho, xi_rho)
            
            rel_error_grad, abs_error_grad = test_accuracy(nabla_rho_paired, C_grad, xi_grad)
            
            log_message = (
                f"ISDF Reconstruction Errors:\n"
                f"  Rho Paired: Relative Error = {rel_error_rho:.2e}, Absolute Error = {abs_error_rho:.2e}\n"
                f"  Grad Paired: Relative Error = {rel_error_grad:.2e}, Absolute Error = {abs_error_grad:.2e}"
            )
            logger.info(log_message)

        result = {
            'C_rho': C_rho,
            'xi_rho': xi_rho,
            'C_grad': C_grad,
            'xi_grad': xi_grad,
            'pivots': pivots,
        }
        self._isdf_results = result
        return result

    def get_2b(self, dm1=None, dm2=None):
        """Calculate two-body terms K1 + K2 + K3."""
        from .kmat import (calc_K1, calc_K2, calc_K3, 
                             calc_K1_isdf, calc_K2_isdf, calc_K3_isdf)
        
        
        if self._isdf_results is not None:
            k_nabla = calc_K1_isdf(
                self._isdf_results['C_rho'],
                self._isdf_results['xi_rho'],
                self._isdf_results['C_grad'],
                self._isdf_results['xi_grad'],
                self.jastrow_factor,
                self.grid_points,
                self.weights
            )
            k_laplacian = calc_K2_isdf(
                self._isdf_results['C_rho'],
                self._isdf_results['xi_rho'],
                self._isdf_results['C_grad'],
                self._isdf_results['xi_grad'],
                self.jastrow_factor,
                self.grid_points,
                self.weights
            )
            k_square = calc_K3_isdf(
                self._isdf_results['C_rho'],
                self._isdf_results['xi_rho'],
                self.jastrow_factor,
                self.grid_points,
                self.weights
            )
        else:
            rho, nabla_rho = self._get_intermediates()
            rho_paired = np.einsum('in,jn->ijn', rho, rho).reshape(-1, len(self.weights))
            # r1 is the first index, r2 is the second index, grad on r1
            rho_nabla_rho_paired = np.einsum('pnd, rn ->prnd',nabla_rho,rho).reshape(-1, len(self.weights), 3)
            k_nabla = calc_K1(rho_paired, rho_nabla_rho_paired, 
                         self.jastrow_factor, self.grid_points, self.weights)
            k_square = calc_K3(rho_paired, self.jastrow_factor,
                         self.grid_points, self.weights)
        
        k_nabla = k_nabla.reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        k_laplacian = - (k_nabla + k_nabla.swapaxes(0, 1))
        k_square = k_square.reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        result = 0.5 * (k_laplacian + k_square)
        result += k_nabla
        result += result.transpose(2, 3, 0, 1)
        
        eri1 = get_eri(self.mf, self.mo_coeff)
        
        return eri1-result

    def get_3b(self, rho_paired, u_gradients):
        """Compute all three-body integrals involving the Jastrow factor. Use the lmat module.
        """
        return lmat.calc_L_symmetric(self.mol, self.mo_coeff, self.grid_points, self.weights )
