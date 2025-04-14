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
        #np.testing.assert_allclose(val1, -val2, rtol=1e-5,
        #                         err_msg=f"MO values don't show expected antisymmetry for H atoms")

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
    
    def test_phi_values_and_derivatives(self):
        """Test φ_s values and derivatives at key points."""
        # Test at r=0, r=rc, and r=∞ (far point)
        rc = 1.0/self.mol.atom_charges()[0]  # rc = 1/Z for first nucleus
        test_points = [1e-8, rc, 4.0]
        
        for nucleus_idx in range(self.ncusp.n_nuclei):
            for r in test_points:
                phi_vals = self.ncusp._get_phi_s_derivatives(nucleus_idx, r)
                phi, phi_d1, phi_d2 = phi_vals
                
                # Basic sanity checks
                self.assertTrue(np.isfinite(phi))
                self.assertTrue(np.isfinite(phi_d1))
                self.assertTrue(np.isfinite(phi_d2))
                
                # Value should decrease with distance
                if r > rc:
                    near_val = self.ncusp.eval_mo_at_r(nucleus_idx, rc)
                    self.assertLess(abs(phi), abs(near_val))
    
    def test_cusp_correction(self):
        """Test the cusp correction values against φ_s."""
        params = self.ncusp.init_params()
        
        for nucleus_idx in range(self.ncusp.n_nuclei):
            Z = self.mol.atom_charges()[nucleus_idx]
            Z_idx = self.ncusp.Z_to_idx[int(Z)]
            rc = params['rc'][Z_idx]
            poly_coeffs = params['poly_coeff'][Z_idx]
            C = params['C'][Z_idx]
            
            # Test matching conditions at r = rc
            phi_rc_vals = self.ncusp._get_phi_s_derivatives(nucleus_idx, rc)
            phi_s, phi_s_d1, phi_s_d2 = phi_rc_vals
            
            # Compute φ_cusp and its derivatives at rc
            poly_val = self.ncusp._eval_poly(rc, poly_coeffs)
            phi_cusp = np.exp(poly_val) + C
            
            # X1: Value matching at rc
            np.testing.assert_allclose(
                np.log(abs(phi_cusp - C)), 
                np.log(abs(phi_s)), 
                rtol=1e-5,
                err_msg=f"X1 condition failed at rc for nucleus {nucleus_idx}"
            )
            
            # X2: First derivative matching at rc
            R_rc = np.exp(poly_val)  # R(rc) = exp(p(rc))
            deriv1_cusp = R_rc * self.ncusp._eval_poly(rc, np.arange(5) * poly_coeffs)
            np.testing.assert_allclose(
                deriv1_cusp/R_rc,
                phi_s_d1/phi_s,
                rtol=1e-5,
                err_msg=f"X2 condition failed at rc for nucleus {nucleus_idx}"
            )
            
            # X3: Second derivative matching at rc
            deriv2_cusp = R_rc * (
                self.ncusp._eval_poly(rc, np.arange(5) * np.arange(5) * poly_coeffs) +
                self.ncusp._eval_poly(rc, np.arange(5) * poly_coeffs)**2
            )
            np.testing.assert_allclose(
                deriv2_cusp/R_rc,
                phi_s_d2/phi_s,
                rtol=1e-5,
                err_msg=f"X3 condition failed at rc for nucleus {nucleus_idx}"
            )
            
            # X4: Cusp condition at r = 0
            self.assertAlmostEqual(
                poly_coeffs[1],
                -Z,
                places=4,
                msg=f"X4 cusp condition failed for nucleus {nucleus_idx}"
            )
            
            # X5: Value matching at r = 0
            phi_s_0 = self.ncusp.eval_mo_at_r(nucleus_idx, 1e-8)
            phi_cusp_0 = np.exp(poly_coeffs[0]) + C
            np.testing.assert_allclose(
                np.log(abs(phi_cusp_0 - C)),
                np.log(abs(phi_s_0)),
                rtol=1e-5,
                err_msg=f"X5 condition failed at r=0 for nucleus {nucleus_idx}"
            )
            
            # ... existing distance scaling tests ...
    
    def test_nuclear_cusp_condition(self):
        """Test that cusp condition is satisfied at r=0."""
        params = self.ncusp.init_params()
        r_test = 1e-6  # Close to nucleus
        
        for nucleus_idx in range(self.ncusp.n_nuclei):
            Z = self.mol.atom_charges()[nucleus_idx]
            Z_idx = self.ncusp.Z_to_idx[int(Z)]
            
            # Get derivatives of corrected wavefunction near r=0
            poly_coeffs = params['poly_coeff'][Z_idx]
            # First derivative of polynomial at r=0 should be -Z
            self.assertAlmostEqual(poly_coeffs[1], -Z, places=4,
                msg=f"Cusp condition not satisfied for nucleus {nucleus_idx}")
    
    def test_cutoff_behavior(self):
        """Test the smooth cutoff behavior."""
        params = self.ncusp.init_params()
        
        for nucleus_idx in range(self.ncusp.n_nuclei):
            Z_idx = self.ncusp.Z_to_idx[int(self.mol.atom_charges()[nucleus_idx])]
            rc = params['rc'][Z_idx]
            
            # Test points before, at, and after rc
            r_vals = [0.5*rc, rc, 2.0*rc]
            cutoffs = [self.ncusp._cutoff_function(r, rc) for r in r_vals]
            
            # Cutoff should decrease monotonically
            self.assertGreater(cutoffs[0], cutoffs[1])
            self.assertGreater(cutoffs[1], cutoffs[2])
            
            # Value at rc should be intermediate
            self.assertAlmostEqual(cutoffs[1], 0.5, places=1)

    def test_param_initialization(self):
        """Test parameter initialization and constraints."""
        params = self.ncusp.init_params()
        
        for Z_type, Z_idx in self.ncusp.Z_to_idx.items():
            Z = float(Z_type)
            rc = params['rc'][Z_idx]
            
            # Check rc initialization
            self.assertAlmostEqual(rc, 1.0/Z)
            
            # Check polynomial coefficients
            coeffs = params['poly_coeff'][Z_idx]
            self.assertEqual(len(coeffs), 5)
            
            # First derivative coefficient should match cusp condition
            self.assertAlmostEqual(coeffs[1], -Z, places=4)

if __name__ == '__main__':
    unittest.main()
