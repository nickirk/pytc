import unittest
import numpy as np
from pyscf import gto, scf
from pytc.autodiff.jastrow.ncusp import NuclearCusp
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
        self.ncusp = NuclearCusp(self.mol, n_radial=1000)
        
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
        
        # Create Z_dict dynamically from the array-based representation
        Z_dict = {}
        for Z in self.ncusp.unique_Z:
            Z_int = int(Z)
            Z_idx = int(self.ncusp.Z_to_idx[Z_int])
            Z_dict[Z_int] = Z_idx
            
        for Z_type, Z_idx in Z_dict.items():
            Z = float(Z_type)
            rc = params['rc'][Z_idx]
            
            # Check rc initialization
            self.assertAlmostEqual(rc, 1.0/Z)
            
            # Check polynomial coefficients
            coeffs = params['poly_coeff'][Z_idx]
            self.assertEqual(len(coeffs), 5)
            
            # First derivative coefficient should match cusp condition
            self.assertAlmostEqual(coeffs[1], -Z, places=4)
    
    def test_eval_mo_against_numpy_spline(self):
        """Compare eval_mo_at_r results with NumPy's cubic spline interpolation."""
        from scipy.interpolate import CubicSpline
        
        # Initialize parameters
        params = self.ncusp.init_params()
        
        # Test for each nucleus
        for nucleus_idx in range(self.ncusp.n_nuclei):
            # Use ncusp's internal radial grid and AO values
            r_grid = self.ncusp.r_grids[nucleus_idx]
            ao_values = self.ncusp.ao_values[nucleus_idx]
            
            # Calculate MO values on the grid points
            nocc = self.mol.nelec[0]
            s_indices = self.ncusp.s_indices_per_atom[nucleus_idx]
            mo_values = ao_values[:, 0]
            
            # Construct a NumPy cubic spline using the same data points
            # The result is the sum of splines for each MO contribution
            numpy_spline = CubicSpline(r_grid, mo_values)
            
            # Test points at different distances
            Z_idx = self.ncusp.Z_to_idx[int(self.mol.atom_charges()[nucleus_idx])]
            test_points = np.linspace(r_grid[0], params['rc'][Z_idx], 50)
            
            # Get values from both methods
            jax_values = self.ncusp.eval_mo_at_r(nucleus_idx, test_points)
            numpy_values = numpy_spline(test_points)
            
            # Compare results
            np.testing.assert_allclose(
                jax_values, numpy_values, rtol=1e-4, atol=1e-4,
                err_msg=f"JAX spline doesn't match NumPy spline for nucleus {nucleus_idx}"
            )
            
            # Test a scalar input
            single_point = np.min(r_grid) + 0.05
            jax_value = self.ncusp.eval_mo_at_r(nucleus_idx, single_point)
            numpy_value = numpy_spline(single_point)
            
            np.testing.assert_allclose(
                jax_value, numpy_value, rtol=1e-4, atol=1e-4,
                err_msg=f"JAX spline doesn't match NumPy spline for scalar input (nucleus {nucleus_idx})"
            )
            
            # Print values for debugging
            if nucleus_idx == 0:
                print(f"\nSpline comparison for nucleus {nucleus_idx}:")
                print(f"Grid shape: {r_grid.shape}, AO values shape: {ao_values.shape}, MO values shape: {mo_values.shape}")
                for i in range(0, len(test_points), 10):
                    print(f"r = {test_points[i]:.6f}, JAX = {jax_values[i]:.6f}, "
                         f"NumPy = {numpy_values[i]:.6f}, "
                         f"diff = {jax_values[i] - numpy_values[i]:.6f}")
    
    
    def test_compute_function(self):
        """Test the _compute function of NuclearCuspJastrow."""
        params = self.ncusp.init_params()
        
        # Test points: one near O atom, one near H atom, one far from all atoms
        test_points = [
            (np.array([0.0, 0.0, 0.001]),  # near O atom
             np.array([0.0, 0.0, 1.0])), # somewhere else
            (np.array([0.0, 0.0, 1.3]),  # near H atom
             np.array([0.0, 0.0, -1.0])),
            (np.array([2.0, 2.0, 2.0]),  # far from all atoms
             np.array([3.0, 3.0, 3.0]))
        ]
        
        for r1, r2 in test_points:
            # Compute the Jastrow value
            u = self.ncusp._compute(r1, r2, params)
            
            # Basic checks
            #self.assertTrue(np.isfinite(u), 
            #              f"Non-finite Jastrow value for r1={r1}, r2={r2}")
            
            # Print detailed information for debugging
            print(f"\nTest point:")
            print(f"r1: {r1}")
            print(f"r2: {r2}")
            print(f"Jastrow value u: {u}")
            
            # Check each nucleus contribution
            for nucleus_idx in range(self.ncusp.n_nuclei):
                nucleus_pos = self.mol.atom_coords()[nucleus_idx]
                r1_dist = np.linalg.norm(r1 - nucleus_pos)
                r2_dist = np.linalg.norm(r2 - nucleus_pos)
                
                Z = self.mol.atom_charges()[nucleus_idx]
                Z_idx = self.ncusp.Z_to_idx[int(Z)]
                rc = params['rc'][Z_idx]
                
                print(f"\nNucleus {nucleus_idx} (Z={Z}):")
                print(f"Distance from r1: {r1_dist:.6f}")
                print(f"Distance from r2: {r2_dist:.6f}")
                print(f"rc value: {rc:.6f}")
                
                # Test if distances are within rc
                phi_s = self.ncusp.eval_mo_at_r(nucleus_idx, r1_dist)
                print(f"φ_s at r1: {phi_s:.6f}")
                phi_s = self.ncusp.eval_mo_at_r(nucleus_idx, r2_dist)
                print(f"φ_s at r2: {phi_s:.6f}")

    def test_log_gradients(self):
        """Test gradient and laplacian calculations."""
        params = self.ncusp.init_params()
        
        # Test points near each nucleus
        for nucleus_idx in range(self.ncusp.n_nuclei):
            nucleus_pos = self.mol.atom_coords()[nucleus_idx]
            Z = self.mol.atom_charges()[nucleus_idx]
            Z_idx = self.ncusp.Z_to_idx[int(Z)]
            rc = params['rc'][Z_idx]
            
            # Test point slightly offset from nucleus
            r1 = nucleus_pos + np.array([0.0, 0.0, 0.01])
            r2 = nucleus_pos + np.array([0.0, 0.0, -.01])
            
            value = self.ncusp._compute(r1, r2, params)
            # Get gradients and laplacian
            grad_u, lap_u = self.ncusp.get_log_grads_r1(r1, r2, params)
            
            print(f"\nGradient test for nucleus {nucleus_idx} (Z={Z}):")
            print(f"Test point r1: {r1}")
            print(f"Reference point r2: {r2}")
            print(f"Distance from nucleus: {np.linalg.norm(r1 - nucleus_pos):.6f}")
            print(f"rc value: {rc:.6f}")
            print(f"Jastrow value: {value:.6f}")
            print(f"Gradient: {grad_u}")
            print(f"Laplacian: {lap_u}")
            
            # Check basic physics:
            # 1. Gradient should point away from nucleus when inside rc
            r_vec = r1 - nucleus_pos
            r_dist = np.linalg.norm(r_vec)
            if r_dist < rc:
                grad_radial = np.dot(grad_u, r_vec/r_dist)
                print(f"Radial gradient component: {grad_radial:.6f}")
                self.assertGreater(grad_radial, -100.0, 
                    "Gradient should not be strongly attractive inside rc")
            
            # 2. Values should be finite
            self.assertTrue(np.all(np.isfinite(grad_u)), 
                          f"Non-finite gradient values found")
            self.assertTrue(np.isfinite(lap_u),
                          f"Non-finite laplacian value found")
            
            # 3. Print intermediates from _compute for debugging
            u = self.ncusp._compute(r1, r2, params)
            print(f"Jastrow value u: {u}")
            phi_s = self.ncusp.eval_mo_at_r(nucleus_idx, r_dist)
            print(f"φ_s value: {phi_s}")

    def test_local_energy_along_axis(self):
        """Test local energy evaluation along x-axis through O atom."""
        from pytc.autodiff.ansatz.sj import SlaterJastrow
        from pytc.autodiff.ansatz.det import SlaterDet
        import jax.numpy as jnp
        
        # Create RHF determinant
        det = SlaterDet(self.mol, self.mf.mo_coeff)
        
        # Create SlaterJastrow with nuclear cusp
        sj = SlaterJastrow(self.mol, self.ncusp, [det])
        
        # Initialize parameters
        jastrow_params = self.ncusp.init_params()
        linear_coeffs = jnp.array([1.0])  # Single determinant
        
        # Set up fixed positions for other electrons (random but fixed)
        np.random.seed(42)
        n_electrons = 10  # Water molecule has 10 electrons
        fixed_positions = np.random.randn(n_electrons-1, 3)+1.  # 9 electrons at random positions
        
        # Create grid points along x-axis through O atom
        x_points = np.linspace(-0.2, 0.2, 100)
        energies = []
        jastrow_vals = []
        
        # Create a vmap function for Jastrow evaluation
        vmap_jastrow = jax.vmap(lambda pos: self.ncusp._compute(
            pos, fixed_positions[0], jastrow_params))
        
        # Evaluate Jastrow and local energy at each point
        for x in x_points:
            # Place electron 0 at point along x-axis
            pos = np.array([x, 0.0, 0.0])
            
            # Compute Jastrow value
            u = self.ncusp._compute(pos, fixed_positions[0], jastrow_params)
            jastrow_vals.append(float(u))
            
            # Create full electron configuration
            elec_coords = np.vstack([[pos], fixed_positions])
            elec_coords = elec_coords.reshape((1, n_electrons, 3))
            
            # Compute local energy
            E_L = sj.local_energy(elec_coords, jastrow_params, linear_coeffs)[0]
            energies.append(float(E_L))
        
        # Print results
        print("\nEnergies and Jastrow values along x-axis through O atom:")
        print("x-coordinate  Energy (hartree)    Jastrow value")
        print("-" * 50)
        for x, E, u in zip(x_points, energies, jastrow_vals):
            print(f"{x:10.4f}  {E:15.6f}  {u:15.6f}")

if __name__ == '__main__':
    unittest.main()
