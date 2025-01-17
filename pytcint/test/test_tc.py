"""Test for transcorrelated methods."""

import unittest
import numpy as np
from pyscf import gto, scf
from pytcint.tc import TC
from pytcint.jastrow import Jastrow


def get_h2_sto3g():
    """Return a simple H2 molecule with STO-3G basis for testing."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


class SimpleJastrow(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    def jastrow_function(self, delta_r):
        return np.exp(-self.parameters[0] * np.linalg.norm(delta_r, axis=-1))
    
    def jastrow_gradient(self, delta_r):
        norm = np.linalg.norm(delta_r, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)  # Avoid division by zero
        return -self.parameters[0] * delta_r / norm * self.jastrow_function(delta_r)[..., np.newaxis]


class TestTC(unittest.TestCase):
    """Test TC class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple H2 molecule for all tests in this class."""
        cls.mol, cls.mf = get_h2_sto3g()
        cls.tc = TC(cls.mf, grid_lvl=1)  # Use coarse grid for testing
        cls.jastrow = SimpleJastrow([0.5])  # alpha = 0.5
    
    def test_grid_initialization(self):
        """Test if grid is properly initialized."""
        self.assertIsNotNone(self.tc.grid_points)
        self.assertIsNotNone(self.tc.weights)
        self.assertEqual(self.tc.grid_points.shape[1], 3)
        self.assertEqual(self.tc.grid_points.shape[0], len(self.tc.weights))
    
    def test_basis_evaluation(self):
        """Test if basis functions are properly evaluated on grid."""
        rho, nabla_rho = self.tc._eval_basis_on_grid()
        n_grid = len(self.tc.weights)
        n_ao = self.mol.nao
        
        self.assertEqual(rho.shape, (n_grid, n_ao))
        self.assertEqual(nabla_rho.shape, (n_grid, 3, n_ao))
        
        # Test if cached values are returned
        rho2, nabla_rho2 = self.tc._eval_basis_on_grid()
        np.testing.assert_array_equal(rho, rho2)
        np.testing.assert_array_equal(nabla_rho, nabla_rho2)
    
    def test_2b_shape(self):
        """Test if get_2b returns correct shape."""
        result = self.tc.get_2b(self.jastrow)
        self.assertEqual(result.shape, (self.mol.nao,)*4)
    
    def test_3b_shape(self):
        """Test if get_3b returns correct shape."""
        result = self.tc.get_3b(self.jastrow)
        self.assertEqual(result.shape, (self.mol.nao,)*6)


if __name__ == '__main__':
    unittest.main() 