import unittest
import numpy as np
from pyscf import gto, scf
from pytc.autodiff.jastrow.ncusp import NuclearCuspJastrow

class TestNuclearCuspJastrow(unittest.TestCase):
    """Test cases for NuclearCuspJastrow class."""
    
    def setUp(self):
        """Set up H2 molecule and compute RHF."""
        self.mol = gto.M(atom='H 0 0 1.4; O 0 0 0; H 0 0 -1.4', basis='cc-pvdz')
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        # Initialize and setup NuclearCuspJastrow
        self.ncusp = NuclearCuspJastrow(n_radial=1000)
        self.ncusp.setup_for_molecule(self.mol, self.mf.mo_coeff)
        
    def test_mo_values_symmetry(self):
        """Test that MO values follow expected symmetry for H2O.
        The two H atoms should have similar magnitude but potentially opposite signs
        due to the molecular orbital symmetry."""
        # Test points
        distances = np.linspace(0.1, 2.0, 1000)
        
        # Get values for both H atoms
        val1 = self.ncusp.eval_mo_at_r(0, distances)  # First H
        val2 = self.ncusp.eval_mo_at_r(2, distances)  # Second H
        
        # Test that magnitudes are similar
        np.testing.assert_allclose(abs(val1), abs(val2), rtol=1e-5,
                                 err_msg=f"MO value magnitudes not symmetric for H atoms")
        
        # Test that they have opposite signs (due to molecular orbital symmetry)
        np.testing.assert_allclose(val1, -val2, rtol=1e-5,
                                 err_msg=f"MO values don't show expected antisymmetry for H atoms")

    def test_mo_sums_debug(self):
        """Debug MO sums calculation."""
        # Print debug information
        nocc = self.mol.nelec[0]
        
        print("\nAO values at first point for each nucleus:")
        for i in range(self.ncusp.n_nuclei):
            print(f"\nNucleus {i}:")
            print("AO values:", self.ncusp.ao_values[i][0])
            print("s-type indices:", self.ncusp.s_indices_per_atom[i])
            
            # Test MO values
            s_ao_vals = self.ncusp.ao_values[i][0]
            print("mo_coeff:", self.mf.mo_coeff[self.ncusp.s_indices_per_atom[i], :nocc])
            mo_vals = np.dot(s_ao_vals, self.mf.mo_coeff[self.ncusp.s_indices_per_atom[i], :nocc])
            print(f"MO values at nucleus {i}:", mo_vals)
            
            # Basic sanity checks
            self.assertTrue(np.all(np.isfinite(mo_vals)), 
                          f"Non-finite MO values found for nucleus {i}")
    
    def test_distance_scaling(self):
        """Test that MO values decay with distance."""
        for i in range(self.ncusp.n_nuclei):
            near_val = abs(self.ncusp.eval_mo_at_r(i, 0.1))
            far_val = abs(self.ncusp.eval_mo_at_r(i, 2.0))
            self.assertGreater(near_val, far_val, 
                             "MO values should decrease with distance")

if __name__ == '__main__':
    unittest.main()
