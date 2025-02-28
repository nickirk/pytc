"""Tests for the Ansatz class."""

import unittest
import numpy as np
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.autodiff.ansatz import SlaterJastrow, SlaterDet
from pytc.autodiff.jastrow import Poly



class TestAnsatzH2(unittest.TestCase):
    """Test cases for Ansatz class using real H2 molecule."""
    
    def setUp(self):  
        """Set up H2 molecule and compute RHF."""
        self.mol = gto.M(  
            atom='H 0 0 0; H 0 0 0.742',  
            basis='sto-3g',
            unit='angstrom',
            spin=2
        )
        
        # Run RHF for triplet state
        self.mf = scf.RHF(self.mol)  
        self.mf.kernel()
        
        # Create determinant with RHF orbitals
        self.det = SlaterDet(self.mol, self.mf.mo_coeff)
        
        # Simple Jastrow with one parameter
        self.jastrow = Poly(jnp.array([0.5]))
        
        # Create ansatz with single determinant
        self.coeffs = jnp.array([1.0])
        self.ansatz = SlaterJastrow(self.mol, self.jastrow, [self.det], self.coeffs)
        
        # Test positions: two electrons slightly offset from nuclei
        self.test_pos = jnp.array([
            [0.0, 0.1, 0.0],    # electron 1 near first H
            [0.0, 0.1, 0.742],  # electron 2 near second H
        ])

    def test_wavefunction_evaluation(self):
        """Test full wavefunction evaluation for H2."""
        value = self.ansatz(self.test_pos)
        
        # Value should be real for ground state
        self.assertTrue(np.isreal(value))
        
        # Value should be non-zero
        self.assertNotEqual(float(value), 0.0)
        
        # Test that moving electrons far apart gives smaller value
        far_pos = jnp.array([
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
        ])
        far_value = self.ansatz(far_pos)
        self.assertLess(abs(far_value), abs(value))

    def test_jastrow_parameter_sensitivity(self):
        """Test sensitivity to Jastrow parameter changes."""
        value_original = self.ansatz(self.test_pos)
        
        # Change Jastrow parameter more significantly
        new_ansatz = self.ansatz.update_jastrow(jnp.array([2.0]))  # Bigger change
        value_new = new_ansatz(self.test_pos)
        
        # Values should be different
        self.assertNotAlmostEqual(float(value_original), float(value_new))

    def test_antisymmetry(self):
        """Test that wavefunction is antisymmetric under electron exchange."""
        value1 = self.ansatz(self.test_pos)
        
        # Swap electrons and check sign change
        # Note: For H2 in RHF, we need to swap within same spin block to see antisymmetry
        # First electron is spin-up, second is spin-down, so swapping won't show antisymmetry
        # Let's modify the test to use two spin-up electrons
        spin_up_pos = jnp.array([
            [0.0, 0.1, 0.0],    # first spin-up electron
            [0.0, 0.1, 1.0],    # second spin-up electron
        ])
        
        value1 = self.ansatz(spin_up_pos)
        swapped_pos = spin_up_pos[::-1]
        value2 = self.ansatz(swapped_pos)
        
        # Values should be equal and opposite
        np.testing.assert_allclose(value1, -value2)

    def test_jastrow_terms(self):
        """Test computation of Jastrow gradient and laplacian terms."""
        grad_J, lap_J = self.ansatz._compute_jastrow_terms(self.test_pos)
        
        # Check shapes
        self.assertEqual(grad_J.shape, (2, 3))  # (n_electrons, xyz)
        self.assertEqual(lap_J.shape, (2,))     # (n_electrons,)
        
        # Gradients should be opposite for electrons near equilibrium
        np.testing.assert_allclose(grad_J[0], -grad_J[1], rtol=1e-5)

    def test_kinetic_matrix(self):
        """Test computation of kinetic energy matrix."""
        grad_J, lap_J = self.ansatz._compute_jastrow_terms(self.test_pos)
        inv_up, inv_down, B_kin_up, B_kin_down = self.ansatz._compute_kinetic_matrix(
            self.test_pos, grad_J, lap_J)
        
        # Check shapes
        n_up = self.det.n_alpha
        n_down = self.det.n_beta
        self.assertEqual(B_kin_up.shape, (n_up, n_up))
        self.assertEqual(B_kin_down.shape, (n_down, n_down))
        
        # Kinetic energy should be real
        self.assertTrue(np.allclose(B_kin_up.imag, 0))
        self.assertTrue(np.allclose(B_kin_down.imag, 0))
        
        # Inverse matrices should be correct
        slater_up, slater_down = self.det.matrix(self.test_pos)
        np.testing.assert_allclose(inv_up @ slater_up, np.eye(n_up), atol=1e-7)

    def test_potential_matrix(self):
        """Test computation of potential energy matrix."""
        slater_up, slater_down = self.det.matrix(self.test_pos)
        B_pot_up, B_pot_down = self.ansatz._compute_potential_matrix(
            self.test_pos, slater_up, slater_down)
        
        # Check shapes
        self.assertEqual(B_pot_up.shape, slater_up.shape)
        self.assertEqual(B_pot_down.shape, slater_down.shape)
        
        # Potential energy should be real
        self.assertTrue(np.allclose(B_pot_up.imag, 0))
        self.assertTrue(np.allclose(B_pot_down.imag, 0))
        
        # Test that potential increases as electrons move apart
        far_pos = jnp.array([
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
        ])
        far_slater_up, far_slater_down = self.det.matrix(far_pos)
        far_B_pot_up, far_B_pot_down = self.ansatz._compute_potential_matrix(
            far_pos, far_slater_up, far_slater_down)
        
        # Energy should be higher for separated electrons
        self.assertGreater(
            float(jnp.abs(far_B_pot_up).mean()), 
            float(jnp.abs(B_pot_up).mean())
        )

    def test_local_energy(self):
        """Test local energy computation."""
        energy = self.ansatz.local_energy(self.test_pos)
        
        # Energy should be real
        self.assertTrue(np.isreal(energy))
        
        # Energy should be finite
        self.assertTrue(np.isfinite(energy))
        
        # Test virial theorem: <T> ≈ -<V> for ground state
        # This requires computing T and V separately
        grad_J, lap_J = self.ansatz._compute_jastrow_terms(self.test_pos)
        inv_up, inv_down, B_kin_up, B_kin_down = self.ansatz._compute_kinetic_matrix(
            self.test_pos, grad_J, lap_J)
        
        slater_up, slater_down = self.det.matrix(self.test_pos)
        B_pot_up, B_pot_down = self.ansatz._compute_potential_matrix(
            self.test_pos, slater_up, slater_down)
        
        T = float(jnp.trace(inv_up @ B_kin_up) + jnp.trace(inv_down @ B_kin_down))
        V = float(jnp.trace(inv_up @ B_pot_up) + jnp.trace(inv_down @ B_pot_down))
        
        # Check if T ≈ -V (allow for some deviation due to non-optimal wavefunction)
        self.assertLess(abs(T + V), abs(T))  # |T + V| should be smaller than |T|

if __name__ == '__main__':
    unittest.main()
