"""Tests for Slater determinant implementation."""

import unittest
import numpy as np
from pyscf import gto, scf
from pytc.vmc.det import SlaterDeterminant

class TestSlaterDeterminant(unittest.TestCase):
    """Test Slater determinant implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test cases for all tests in this class."""
        # Create H2 molecule
        cls.mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g')
        cls.mol.build()
        
        # Get MO coefficients
        mf = scf.RHF(cls.mol)
        mf.kernel()
        cls.mo_coeff = mf.mo_coeff
        
        # Common test coordinates
        cls.test_coords = np.array([
            [0.0, 0.0, 0.1],  # near first H
            [0.0, 0.0, 1.3],  # near second H
        ])
    
    def setUp(self):
        """Set up each test."""
        self.det = SlaterDeterminant(self.mol, self.mo_coeff, n_up=1, n_down=1)

    def test_init_restricted(self):
        """Test initialization with restricted orbitals."""
        self.assertEqual(self.det.n_up, 1)
        self.assertEqual(self.det.n_down, 1)
        self.assertFalse(self.det.unrestricted)
        self.assertIs(self.det.mo_coeff_alpha, self.det.mo_coeff_beta)

    def test_init_unrestricted(self):
        """Test initialization with unrestricted orbitals."""
        mo_coeffs = [self.mo_coeff, self.mo_coeff]  # Simulate UHF
        det = SlaterDeterminant(self.mol, mo_coeffs, n_up=1, n_down=1)
        self.assertTrue(det.unrestricted)
        self.assertIsNot(det.mo_coeff_alpha, det.mo_coeff_beta)

    def test_determinant_value(self):
        """Test basic determinant evaluation."""
        value = self.det.value(self.test_coords)
        self.assertIsInstance(value, float)
        self.assertNotEqual(value, 0.0)

    def test_update_mechanism(self):
        """Test the update mechanism for moving electrons."""
        # Initialize
        self.det.init_inverse(self.test_coords)
        init_value = self.det.total_value()
        
        # Move up electron
        new_pos = np.array([0.1, 0.1, 0.1])
        ratio = self.det.update(0, new_pos)
        
        # Check ratio against direct calculation
        new_coords = self.test_coords.copy()
        new_coords[0] = new_pos
        direct_value = self.det.value(new_coords)
        
        self.assertAlmostEqual(ratio * init_value, direct_value, places=10)

    def test_value_sign_change(self):
        """Test if determinant changes sign when electrons are exchanged."""
        det = SlaterDeterminant(self.mol, self.mo_coeff, n_up=2, n_down=0)
        coords1 = self.test_coords
        coords2 = np.array([coords1[1], coords1[0]])  # Exchange positions
        
        val1 = det.value(coords1)
        val2 = det.value(coords2)
        
        self.assertAlmostEqual(val1, -val2, places=10)

    def test_boundary_conditions(self):
        """Test behavior at large distances."""
        far_coords = np.array([[0.0, 0.0, 10.0],
                             [0.0, 0.0, -10.0]])
        value = self.det.value(far_coords)
        self.assertLess(abs(value), 1e-5)

    def test_numerical_gradient(self):
        """Test against numerical differentiation."""
        eps = 1e-6
        coords = self.test_coords
        
        # Compute numerical gradient for first electron
        numerical_grad = np.zeros(3)
        for d in range(3):
            h = np.zeros(3)
            h[d] = eps
            coords_plus = coords.copy()
            coords_minus = coords.copy()
            coords_plus[0] += h
            coords_minus[0] -= h
            
            grad = (self.det.value(coords_plus) - self.det.value(coords_minus)) / (2*eps)
            numerical_grad[d] = grad
            
        # Compare with analytical gradient (if implemented)
        if hasattr(self.det, 'grad'):
            analytical_grad = self.det.grad(coords)[0]
            np.testing.assert_array_almost_equal(numerical_grad, analytical_grad, decimal=5)

if __name__ == '__main__':
    unittest.main()
