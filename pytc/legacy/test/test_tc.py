"""Test for transcorrelated methods."""

import unittest
import numpy as np
from pyscf import gto, scf
from pytc.legacy.tc import TC
from pytc.legacy.jastrow import Jastrow
from pytc.tc_helper import get_eri


def get_h2_sto3g():
    """Return a simple H2 molecule with STO-3G basis for testing."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


class REXP(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    def __call__(self, r1, r2, r_nuc=None):
        r1 = np.atleast_2d(r1)
        r2 = np.atleast_2d(r2)
        delta_r = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        return -1./self.params[0]*np.exp(-self.params[0] * np.linalg.norm(delta_r, axis=-1))
    
    def _process_grad_batch(self, r1_batch, r2):
        r1_batch = np.atleast_2d(r1_batch)
        r2 = np.atleast_2d(r2)
        delta_r = r1_batch[:, np.newaxis, :] - r2[np.newaxis, :, :]
        norm = np.linalg.norm(delta_r, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)  # Avoid division by zero
        return delta_r / norm * self.__call__(r1_batch, r2)[..., np.newaxis]


class TestTC(unittest.TestCase):
    """Test TC class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple H2 molecule for all tests in this class."""
        cls.mol, cls.mf = get_h2_sto3g()
        cls.jastrow = REXP([1])  # alpha = 0.5
        cls.tc = TC(cls.mf, cls.jastrow, grid_lvl=1)  # Use coarse grid for testing
    
    def test_grid_initialization(self):
        """Test if grid is properly initialized."""
        self.assertIsNotNone(self.tc.grid_points)
        self.assertIsNotNone(self.tc.weights)
        self.assertEqual(self.tc.grid_points.shape[-1], 3)
        self.assertEqual(self.tc.grid_points.shape[0], len(self.tc.weights))
    
    def test_basis_evaluation(self):
        """Test if basis functions are properly evaluated on grid."""
        rho, nabla_rho = self.tc._eval_basis_on_grid()
        n_grid = len(self.tc.weights)
        n_ao = self.mol.nao
        
        self.assertEqual(rho.shape, (n_ao, n_grid))
        self.assertEqual(nabla_rho.shape, (n_ao, n_grid, 3))
        
        rho2, nabla_rho2 = self.tc._eval_basis_on_grid()
        np.testing.assert_array_equal(rho, rho2)
        np.testing.assert_array_equal(nabla_rho, nabla_rho2)
    
    def test_two_body_terms(self):
        """Test calculation of two-body terms."""
        from pytc.legacy.kmat import calc_K1, calc_K2, calc_K3
        
        rho, nabla_rho = self.tc._get_intermediates()
        rho_paired = np.einsum('in,jn->ijn', rho, rho).reshape(-1, len(self.tc.weights))
        nabla_rho_paired = np.einsum('ind,jn->ijnd', nabla_rho, rho).reshape(-1, len(self.tc.weights), 3)
        
        k1 = calc_K1(rho_paired, nabla_rho_paired, 
                     self.tc.jastrow_factor, self.tc.grid_points, self.tc.weights)
        k2 = calc_K2(rho_paired, nabla_rho_paired,
                     self.tc.jastrow_factor, self.tc.grid_points, self.tc.weights)
        k3 = calc_K3(rho_paired, self.tc.jastrow_factor,
                     self.tc.grid_points, self.tc.weights)
        
        combined = self.tc.get_2b()
        
        n_orb = self.tc.n_orb
        k1 = k1.reshape(n_orb, n_orb, n_orb, n_orb)
        k3 = k3.reshape(n_orb, n_orb, n_orb, n_orb)
        
        # TC uses - (K1 + K1^T) for Laplacian part (K2)
        k_laplacian = - (k1 + k1.swapaxes(0, 1))
        
        result = 0.5 * (k_laplacian + k3)
        result += k1
        result += result.transpose(2, 3, 0, 1)
        
        eri1 = get_eri(self.tc.mf, self.tc.mo_coeff)
        
        expected = eri1 - result
        
        np.testing.assert_array_almost_equal(combined, expected)
    

if __name__ == '__main__':
    unittest.main()
