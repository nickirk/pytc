import unittest
import numpy as np
from pyscf import gto, scf
from pytc.autodiff.jastrow.ncusp import NuclearCuspJastrow

class TestNuclearCuspJastrow(unittest.TestCase):
    """Test cases for NuclearCuspJastrow class."""
    
    def setUp(self):
        """Set up H2 molecule and compute RHF."""
        self.mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='cc-pvdz')
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        # Initialize and setup NuclearCuspJastrow
        self.ncusp = NuclearCuspJastrow(n_radial=100)
        self.ncusp.setup_for_molecule(self.mol, self.mf.mo_coeff)
        
    def test_mo_values_symmetry(self):
        """Test that MO values are symmetric for H2."""
        # Test points
        distances = [0.1, 0.5, 1.0, 2.0]
        
        for d in distances:
            val1 = self.ncusp.eval_mo_at_r(0, d)
            val2 = self.ncusp.eval_mo_at_r(1, d)
            # For H2, absolute values should be similar due to symmetry
            np.testing.assert_allclose(abs(val1), abs(val2), rtol=1e-5,
                                     err_msg=f"MO values not symmetric at r={d}")
    
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
