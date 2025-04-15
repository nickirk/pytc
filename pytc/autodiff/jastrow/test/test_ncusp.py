import unittest
import numpy as np
from pyscf import gto, scf
from pytc.autodiff.jastrow.ncusp import NuclearCuspJastrow
import jax 
jax.config.update("jax_enable_x64", True)
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
                atol=1e-5,
                err_msg=f"X1 condition failed at rc for nucleus {nucleus_idx}"
            )
            
            # X2: First derivative matching at rc
            R_rc = np.exp(poly_val)  # R(rc) = exp(p(rc))
            # For first derivative, powers reduce by 1 and skip 0th power
            deriv1_cusp = R_rc * np.sum(np.arange(5)[1:] * poly_coeffs[1:] * rc**(np.arange(5)[1:]-1))
            np.testing.assert_allclose(
                deriv1_cusp/R_rc,
                phi_s_d1/phi_s,
                atol=1e-5,
                err_msg=f"X2 condition failed at rc for nucleus {nucleus_idx}"
            )
            
            # X3: Second derivative matching at rc
            # For second derivative, powers reduce by 2 and skip 0th and 1st power
            p_d2 = np.sum(np.arange(5)[2:] * (np.arange(5)[2:]-1) * poly_coeffs[2:] * rc**(np.arange(5)[2:]-2))
            # First derivative squared term uses reduced powers as well
            p_d1 = np.sum(np.arange(5)[1:] * poly_coeffs[1:] * rc**(np.arange(5)[1:]-1))
            deriv2_cusp = R_rc * (p_d2 + p_d1**2)
            np.testing.assert_allclose(
                deriv2_cusp/R_rc,
                phi_s_d2/phi_s,
                atol=1e-6,
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
                atol=1e-5,
                err_msg=f"X5 condition failed at r=0 for nucleus {nucleus_idx}"
            )
            
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
