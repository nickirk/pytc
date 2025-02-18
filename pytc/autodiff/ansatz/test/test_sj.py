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
        self.ansatz = SlaterJastrow(self.jastrow, [self.det], self.coeffs)
        
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

if __name__ == '__main__':
    unittest.main()
