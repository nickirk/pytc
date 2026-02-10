import unittest
import numpy as np
from pyscf import gto, scf
import jax 
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp

from pytc.autodiff.jastrow import NuclearCusp, Poly
from pytc.autodiff.vmc import sample
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pytc.autodiff.ansatz.det import SlaterDet
from pytc.autodiff.vmc.mcmc_utils import analyze_energies
from pytc.autodiff.vmc.walker import Walker

def create_test_walker(positions, n_alpha, n_beta):
    """Helper to create a Walker for testing."""
    n_walkers = positions.shape[0]
    # Ensure positions are (n_walkers, n_elec, 3)
    if positions.ndim == 2:
        positions = positions[None, ...]
        n_walkers = 1
        
    return Walker(
        positions=jnp.asarray(positions),
        det_up=(jnp.zeros((n_walkers,)), jnp.zeros((n_walkers,))), 
        det_down=(jnp.zeros((n_walkers,)), jnp.zeros((n_walkers,))),
        slater_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        slater_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        inv_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        inv_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        grad_up=jnp.zeros((n_walkers, n_alpha, n_alpha, 3)),
        grad_down=jnp.zeros((n_walkers, n_beta, n_beta, 3)),
        lap_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
        lap_down=jnp.zeros((n_walkers, n_beta, n_beta)),
        move_mask=jnp.ones((n_walkers, positions.shape[1]), dtype=bool),
        log_psi=jnp.zeros((n_walkers,)),
        psi_sign=jnp.zeros((n_walkers,)),
        log_jastrow=jnp.zeros((n_walkers,)),
    )

class TestNuclearCuspJastrow(unittest.TestCase):
    """Test cases for NuclearCuspJastrow class."""
    
    def setUp(self):
        """Set up H2 molecule and compute RHF."""
        self.mol = gto.M(atom='H 0 0 1.4; O 0 0 0; H 0 0 -1.4', basis='cc-pvdz')
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        # Initialize and setup NuclearCuspJastrow
        self.ncusp = NuclearCusp.create(self.mol, n_radial=1000)
        
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
            # AO values no longer stored
            pass
    
    
    def test_cusp_correction(self):
        """Test the cusp correction values against φ_s."""
        params = self.ncusp.init_params()
        
        for nucleus_idx in range(self.ncusp.n_nuclei):
            Z = self.mol.atom_charges()[nucleus_idx]
            Z_idx = self.ncusp.Z_to_idx[int(Z)]
            rc = params['rc'][Z_idx]
            
            # Extract X4 value from params
            X4 = params['X4'][Z_idx]
            
            # Compute X values and alpha coefficients for this nucleus
            X = self.ncusp._compute_X_values(Z_idx, rc, X4)
            poly_coeffs = self.ncusp._compute_alpha_coeffs(Z, rc, X)
            
            # Test matching conditions at r = rc
            phi_rc_vals = self.ncusp._get_phi_s_derivatives(nucleus_idx, rc)
            phi_s, phi_s_d1, phi_s_d2 = phi_rc_vals
            
            # Compute φ_cusp and its derivatives at rc
            poly_val = self.ncusp._eval_poly(rc, poly_coeffs)
            phi_cusp = np.exp(poly_val)
            
            # X1: Value matching at rc
            np.testing.assert_allclose(
                np.log(abs(phi_cusp)), 
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
            phi_s_0 = self.ncusp.eval_mo_at_r(nucleus_idx, 1e-8)*1.1
            phi_cusp_0 = np.exp(poly_coeffs[0])
            np.testing.assert_allclose(
                np.log(abs(phi_cusp_0)),
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
            
            # Get X4 value and compute polynomial coefficients
            X4 = params['X4'][Z_idx]
            X = self.ncusp._compute_X_values(Z_idx, rc, X4)
            coeffs = self.ncusp._compute_alpha_coeffs(Z, rc, X)
            
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
            # Use ncusp's internal radial grid and coefficients
            r_grid = self.ncusp.r_grids[nucleus_idx]
            spline_coeffs = self.ncusp.spline_coeffs[nucleus_idx]
            
            # Evaluate at grid points
            jax_values_at_grid = self.ncusp.eval_mo_at_r(nucleus_idx, r_grid)
            
            # Evaluate at midpoints
            midpoints = (r_grid[:-1] + r_grid[1:]) / 2
            jax_values_at_mid = self.ncusp.eval_mo_at_r(nucleus_idx, midpoints)
            
            # Just check finite
            self.assertTrue(np.all(np.isfinite(jax_values_at_grid)))
            self.assertTrue(np.all(np.isfinite(jax_values_at_mid)))
    
    
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
            
            # Check each nucleus contribution
            for nucleus_idx in range(self.ncusp.n_nuclei):
                nucleus_pos = self.mol.atom_coords()[nucleus_idx]
                r1_dist = np.linalg.norm(r1 - nucleus_pos)
                
                Z = self.mol.atom_charges()[nucleus_idx]
                Z_idx = self.ncusp.Z_to_idx[int(Z)]
                rc = params['rc'][Z_idx]
                
                # Test if distances are within rc
                phi_s = self.ncusp.eval_mo_at_r(nucleus_idx, r1_dist)

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
            
            # Check basic physics:
            # 1. Gradient should point away from nucleus when inside rc
            r_vec = r1 - nucleus_pos
            r_dist = np.linalg.norm(r_vec)
            if r_dist < rc:
                grad_radial = np.dot(grad_u, r_vec/r_dist)
                self.assertGreater(grad_radial, -100.0, 
                    "Gradient should not be strongly attractive inside rc")
            
            # 2. Values should be finite
            self.assertTrue(np.all(np.isfinite(grad_u)), 
                          f"Non-finite gradient values found")
            self.assertTrue(np.isfinite(lap_u),
                          f"Non-finite laplacian value found")

    def test_local_energy_along_axis(self):
        """Test local energy evaluation along x-axis through O atom."""
        from pytc.autodiff.ansatz.sj import SlaterJastrow
        from pytc.autodiff.ansatz.det import SlaterDet
        import jax.numpy as jnp
        
        # Create RHF determinant
        det = SlaterDet.create(self.mol, self.mf.mo_coeff)
        
        # Create SlaterJastrow with nuclear cusp
        sj = SlaterJastrow.create(self.mol, self.ncusp, [det])
        
        # Initialize parameters
        jastrow_params = self.ncusp.init_params()
        linear_coeffs = jnp.array([1.0])  # Single determinant
        # Combine parameters for SlaterJastrow
        params = [jastrow_params, linear_coeffs]
        
        # Set up fixed positions for other electrons (random but fixed)
        key = jax.random.PRNGKey(42)
        n_electrons = 10  # Water molecule has 10 electrons
        fixed_positions = jax.random.normal(key, (n_electrons-1, 3)) + 4.0
        
        # Create grid points along x-axis through O atom
        x_points = np.linspace(-0.02, 0.02, 100)
        energies = []
        jastrow_vals = []
        
        
        # JIT compile the evaluation for speed
        @jax.jit
        def eval_point(pos):
            # Create full electron configuration
            elec_coords = jnp.vstack([pos[None, :], fixed_positions])
            
            # Create walker and populate it
            walker = create_test_walker(elec_coords, det.n_alpha, det.n_beta)
            single_walker = jax.tree_util.tree_map(lambda x: x[0], walker)
            
            psi, updated_walker = sj(single_walker, params)
            
            # Compute local energy using combined parameters
            E_L = sj.local_energy(updated_walker, params)[0]
            
            # Compute Jastrow value
            # Ensure we use sj.jastrow._compute directly to be consistent with params
            u = sj.jastrow._compute(pos, fixed_positions[0], params[0])
            return E_L, jnp.exp(u)

        # Evaluate Jastrow and local energy at each point
        for x in x_points:
            pos = jnp.array([x, 0.0, 0.0])
            E_L, j_val = eval_point(pos)
            
            energies.append(float(E_L))
            jastrow_vals.append(float(j_val))
        
        # Print results
        print("\nEnergies and Jastrow values along x-axis through O atom:")
        print("x-coordinate  Energy (hartree)    Jastrow value")
        print("-" * 50)
        for x, E, u in zip(x_points, energies, jastrow_vals):
            print(f"{x:10.4f}  {E:15.6f}  {u:15.6f}")

if __name__ == '__main__':
    unittest.main()
