"""Tests for Slater determinant implementation."""

import unittest
import numpy as np
from pyscf import gto, scf
from pytc.autodiff.ansatz.det import SlaterDet  

class TestSlaterDet(unittest.TestCase):
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
        
        # Create a slightly more complex molecule for advanced tests
        cls.mol_water = gto.M(atom='O 0 0 0; H 0.75 0.5 0; H -0.75 0.5 0', basis='6-31g')
        cls.mol_water.build()
        mf_water = scf.RHF(cls.mol_water)
        mf_water.kernel()
        cls.mo_coeff_water = mf_water.mo_coeff
        
        # Common test coordinates
        cls.test_coords = np.array([
            [0.0, 0.0, 0.1],  # near first H
            [0.0, 0.0, 1.3],  # near second H
        ])
        
        # More electrons for water
        cls.water_coords = np.array([
            [0.1, 0.1, 0.1],  # near O
            [0.2, 0.1, 0.1],  # near O
            [0.3, 0.1, 0.1],  # near O
            [0.4, 0.1, 0.1],  # near O
            [0.5, 0.1, 0.1],  # near O
            [0.7, 0.5, 0.0],  # near H1
            [0.8, 0.5, 0.0],  # near H1
            [-0.7, 0.5, 0.0],  # near H2
            [-0.8, 0.5, 0.0],  # near H2
            [-0.9, 0.5, 0.0]   # near H2
        ])
    
    def test_init_restricted(self):
        """Test initialization with restricted orbitals."""
        det = SlaterDet(self.mol, self.mo_coeff)
        self.assertEqual(det.n_alpha, 1)
        self.assertEqual(det.n_beta, 1)
        self.assertFalse(det.unrestricted)
        self.assertIs(det.mo_coeff_alpha, det.mo_coeff_beta)
        
        # Check occupied orbital indices
        self.assertEqual(det.alpha_occ, [0])
        self.assertEqual(det.beta_occ, [0])
        
        # Check occupied MO coefficients shape
        self.assertEqual(det.mo_coeff_alpha_occ.shape, (self.mol.nao, 1))
        self.assertEqual(det.mo_coeff_beta_occ.shape, (self.mol.nao, 1))

    def test_init_unrestricted(self):
        """Test initialization with unrestricted orbitals."""
        # Simulate UHF with different coefficients
        mo_coeffs = [self.mo_coeff * 0.9, self.mo_coeff * 1.1]
        det = SlaterDet(self.mol, mo_coeffs)
        self.assertTrue(det.unrestricted)
        self.assertIsNot(det.mo_coeff_alpha, det.mo_coeff_beta)
        
        # Check coefficient values differ
        self.assertFalse(np.allclose(det.mo_coeff_alpha, det.mo_coeff_beta))

    def test_init_with_nelec(self):
        """Test initialization with custom electron counts."""
        det = SlaterDet(self.mol, self.mo_coeff, nelec=(2, 0))
        self.assertEqual(det.n_alpha, 2)
        self.assertEqual(det.n_beta, 0)
        self.assertEqual(len(det.alpha_occ), 2)
        self.assertEqual(len(det.beta_occ), 0)

    def test_init_with_excitation(self):
        """Test initialization with excitations."""
        # Water molecule has more orbitals to play with
        # Single excitation: move one alpha electron from orbital 0 to 5
        excitation = (([0], [5]), ([], []))
        det = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5), excitations=excitation)
        
        # Check the occupied orbitals
        self.assertNotIn(0, det.alpha_occ)
        self.assertIn(5, det.alpha_occ)
        self.assertEqual(len(det.alpha_occ), 5)  # Still 5 orbitals
        
        # No change in beta
        self.assertEqual(det.beta_occ, list(range(5)))

    def test_invalid_excitation(self):
        """Test that invalid excitations raise appropriate errors."""
        # Try to excite from unoccupied orbital
        with self.assertRaises(ValueError):
            SlaterDet(self.mol, self.mo_coeff, nelec=(1,1), 
                     excitations=(([2], [3]), ([], [])))
        
        # Try to excite to occupied orbital
        with self.assertRaises(ValueError):
            SlaterDet(self.mol, self.mo_coeff, nelec=(1,1), 
                     excitations=(([0], [0]), ([], [])))
        
        # Mismatched from/to indices
        with self.assertRaises(ValueError):
            SlaterDet(self.mol, self.mo_coeff, nelec=(1,1), 
                     excitations=(([0], [1, 2]), ([], [])))

    def test_determinant_value(self):
        """Test basic determinant evaluation."""
        det = SlaterDet(self.mol, self.mo_coeff)
        value = det.value(self.test_coords)[0]
        self.assertIsInstance(value, float)
        self.assertNotEqual(value, 0.0)
        
        # Test __call__ convenience method
        call_value = det(self.test_coords)
        self.assertEqual(value, call_value)

    def test_matrix_shape(self):
        """Test shape of Slater matrices."""
        det = SlaterDet(self.mol, self.mo_coeff, nelec=(1, 1))
        slater_up, slater_down = det.matrix(self.test_coords)
        
        self.assertEqual(slater_up.shape, (1, 1))
        self.assertEqual(slater_down.shape, (1, 1))
        
        # Test with more electrons
        det_water = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5))
        water_up, water_down = det_water.matrix(self.water_coords)
        
        self.assertEqual(water_up.shape, (5, 5))
        self.assertEqual(water_down.shape, (5, 5))

    def test_update_mechanism(self):
        """Test the update mechanism for moving electrons."""
        det = SlaterDet(self.mol, self.mo_coeff)
        
        # Initialize
        det.init_inverse(self.test_coords)
        init_value = det.total_value()
        
        # Move up electron
        new_pos = np.array([0.1, 0.1, 0.1])
        ratio = det.update(0, new_pos)
        
        # Check ratio against direct calculation
        new_coords = self.test_coords.copy()
        new_coords[0] = new_pos
        direct_value = det.value(new_coords)[0]
        
        self.assertAlmostEqual(ratio * init_value, direct_value, places=10)
        
        # Sequential moves - move down electron
        new_pos2 = np.array([0.2, 0.2, 1.2])
        ratio2 = det.update(1, new_pos2)
        
        # Check against direct calculation
        new_coords[1] = new_pos2
        direct_value2 = det.value(new_coords)[0]
        
        self.assertAlmostEqual(ratio2 * ratio * init_value, direct_value2, places=10)

    def test_value_sign_change(self):
        """Test if determinant changes sign when electrons are exchanged."""
        # Need 2 electrons of same spin to test exchange
        det = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(2, 0))
        coords1 = self.water_coords[:2]  # Just take first two electrons
        coords2 = np.array([coords1[1], coords1[0]])  # Exchange positions
        
        val1 = det.value(coords1)
        val2 = det.value(coords2)
        
        # Determinant should change sign when two rows are swapped
        self.assertAlmostEqual(val1, -val2, places=10)

    def test_boundary_conditions(self):
        """Test behavior at large distances."""
        det = SlaterDet(self.mol, self.mo_coeff)
        far_coords = np.array([
            [0.0, 0.0, 10.0],  # far from molecule
            [0.0, 0.0, -10.0]   # far from molecule
        ])
        value = det.value(far_coords)
        # Determinant should decay to zero far from molecule
        self.assertLess(abs(value), 1e-3)

    def test_numerical_gradient(self):
        """Test gradient against numerical differentiation."""
        det = SlaterDet(self.mol, self.mo_coeff)
        eps = 1e-5
        coords = self.test_coords
        
        # Compute numerical gradient for first electron, x direction
        d = 0  # x-direction
        e_idx = 0  # first electron
        
        # Get the Slater matrix at the original position
        slater_up_orig, _ = det.matrix(coords)
        
        # Compute numerical derivative using central difference
        h = np.zeros(3)
        h[d] = eps
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[e_idx] += h
        coords_minus[e_idx] -= h
        
        slater_up_plus, _ = det.matrix(coords_plus)
        slater_up_minus, _ = det.matrix(coords_minus)
        
        numeric_grad = (slater_up_plus - slater_up_minus) / (2 * eps)
        
        # Get analytical gradient
        grad_up, _ = det.grad(coords)
        
        # Compare numerical vs analytical for this specific element
        self.assertAlmostEqual(
            grad_up[e_idx, 0, d],  # [electron, orbital, direction]
            numeric_grad[e_idx, 0],  # [electron, orbital]
            places=3
        )

    def test_laplacian(self):
        """Test Laplacian calculation."""
        det = SlaterDet(self.mol, self.mo_coeff)
        
        # We can at least verify it returns the expected shape
        lapl_up, lapl_down = det.laplacian(self.test_coords)
        self.assertEqual(lapl_up.shape, (1, 1))
        self.assertEqual(lapl_down.shape, (1, 1))
        
        # Testing against numerical Laplacian would require more complex code

    def test_excitation_det_value(self):
        """Test determinant value with excitation."""
        # Regular determinant
        det_normal = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5))
        
        # Excited determinant (HOMO → LUMO)
        det_excited = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5),
                              excitations=(([4], [5]), ([], [])))  
        
        # Values should be different
        val_normal = det_normal.value(self.water_coords)
        val_excited = det_excited.value(self.water_coords)
        
        self.assertNotEqual(val_normal, val_excited)
        


if __name__ == '__main__':
    unittest.main()
