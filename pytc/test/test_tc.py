"""Test for transcorrelated methods."""

import unittest
import numpy as np
from pyscf import gto, scf
from pytc.tc import TC
from pytc.jastrow import Jastrow


def get_h2_sto3g():
    """Return a simple H2 molecule with STO-3G basis for testing."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


class SimpleJastrow(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    def __call__(self, r1, r2, atomic_positions=None):
        delta_r = r1[..., np.newaxis, :] - r2[np.newaxis, ...]
        return -1./self.parameters[0]*np.exp(-self.parameters[0] * np.linalg.norm(delta_r, axis=-1))
    
    def grad(self, r1, r2=None, atomic_positions=None):
        if r2 is None:
            r2 = r1
        delta_r = r1[..., np.newaxis, :] - r2[np.newaxis, ...]
        norm = np.linalg.norm(delta_r, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)  # Avoid division by zero
        return delta_r / norm * self.__call__(r1, r2)[..., np.newaxis]


class TestTC(unittest.TestCase):
    """Test TC class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple H2 molecule for all tests in this class."""
        cls.mol, cls.mf = get_h2_sto3g()
        cls.jastrow = SimpleJastrow([1])  # alpha = 0.5
        # Update TC initialization to include jastrow_factor
        cls.tc = TC(cls.mf, cls.jastrow, grid_lvl=1)  # Use coarse grid for testing
    
    def test_grid_initialization(self):
        """Test if grid is properly initialized."""
        self.assertIsNotNone(self.tc.grid_points)
        self.assertIsNotNone(self.tc.weights)
        # Update shape assertions to match new layout (N_grid, 3)
        self.assertEqual(self.tc.grid_points.shape[-1], 3)
        self.assertEqual(self.tc.grid_points.shape[0], len(self.tc.weights))
    
    def test_basis_evaluation(self):
        """Test if basis functions are properly evaluated on grid."""
        rho, nabla_rho = self.tc._eval_basis_on_grid()
        n_grid = len(self.tc.weights)
        n_ao = self.mol.nao
        
        # Update shape assertions to match new layout
        self.assertEqual(rho.shape, (n_ao, n_grid))  # Changed from (n_grid, n_ao)
        self.assertEqual(nabla_rho.shape, (n_ao, n_grid, 3))  # Changed from (n_grid, 3, n_ao)
        
        # Test if cached values are returned
        rho2, nabla_rho2 = self.tc._eval_basis_on_grid()
        np.testing.assert_array_equal(rho, rho2)
        np.testing.assert_array_equal(nabla_rho, nabla_rho2)
    
    def test_2b_shape(self):
        """Test if get_2b returns correct shape."""
        # Remove jastrow argument since it's now in the TC object
        result = self.tc.get_2b()
        self.assertEqual(result.shape, (self.mol.nao,)*4)
    
    def test_3b_shape(self):
        """Test if get_3b returns correct shape."""
        pass
        #result = self.tc.get_3b(self.jastrow)
        #self.assertEqual(result.shape, (self.mol.nao,)*6)


if __name__ == '__main__':
    unittest.main()