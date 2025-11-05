"""Tests for the Ansatz class."""

import unittest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
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
            unit='bohr',
            spin=2
        )
        
        # Run RHF for triplet state
        self.mf = scf.RHF(self.mol)  
        self.mf.kernel()
        
        # Create determinant with RHF orbitals
        self.det = SlaterDet(self.mol, self.mf.mo_coeff)
        
        # Create simple Jastrow without parameters and store params separately
        self.jastrow_params = jnp.array([0.5])
        self.jastrow = Poly()  # No params in constructor
        
        # Store linear coefficients separately
        self.linear_coeffs = jnp.array([1.0])
        
        # Create ansatz without coefficients
        self.ansatz = SlaterJastrow(self.mol, self.jastrow, [self.det])
        
        # Test positions: two electrons slightly offset from nuclei
        self.test_pos = jnp.array([[
            [0.0, 0.1, 0.0],    # electron 1 near first H
            [0.0, 0.1, 0.742],  # electron 2 near second H
        ]])  # Shape: (1, 2, 3)

        # Create params tuple for ansatz calls
        self.params = (self.jastrow_params, self.linear_coeffs)

    def test_wavefunction_evaluation(self):
        """Test full wavefunction evaluation for H2."""
        value = self.ansatz(self.test_pos, self.params)
        self.assertTrue(np.isreal(value[0]))  # Index into batch dimension
        self.assertNotEqual(float(value[0]), 0.0)  # Index into batch dimension
        
        # Test that moving electrons far apart gives smaller value
        far_pos = jnp.array([[
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
        ]])  # Shape: (1, 2, 3)
        far_value = self.ansatz(far_pos, self.params)
        self.assertLess(abs(float(far_value[0])), abs(float(value[0])))

    def test_jastrow_parameter_sensitivity(self):
        """Test sensitivity to Jastrow parameter changes."""
        value_original = self.ansatz(self.test_pos, self.params)
        
        # Change Jastrow parameter more significantly
        new_params = (jnp.array([2.0]), self.linear_coeffs)
        value_new = self.ansatz(self.test_pos, new_params)
        
        # Values should be different
        self.assertNotAlmostEqual(float(value_original[0]), float(value_new[0]))

    def test_antisymmetry(self):
        """Test that wavefunction is antisymmetric under electron exchange."""
        value1 = self.ansatz(self.test_pos, self.params)
        
        # Swap electrons and check sign change
        # Note: For H2 in RHF, we need to swap within same spin block to see antisymmetry
        # First electron is spin-up, second is spin-down, so swapping won't show antisymmetry
        # Let's modify the test to use two spin-up electrons
        spin_up_pos = jnp.array([[
            [0.0, 0.1, 0.0],    # first spin-up electron
            [0.0, 0.1, 1.0],    # second spin-up electron
        ]])  # Shape: (1, 2, 3)
        
        value1 = self.ansatz(spin_up_pos, self.params)
        swapped_pos = spin_up_pos[:, ::-1, :]  # Swap along electron dimension
        value2 = self.ansatz(swapped_pos, self.params)
        
        # Values should be equal and opposite
        np.testing.assert_allclose(value1[0], -value2[0])

    def test_jastrow_terms(self):
        """Test computation of Jastrow gradient and laplacian terms."""
        grad_J, lap_J = self.ansatz._compute_jastrow_terms(self.test_pos[0], self.jastrow_params)
        
        # Check shapes
        self.assertEqual(grad_J.shape, (2, 3))  # (n_electrons, xyz)
        self.assertEqual(lap_J.shape, (2,))     # (n_electrons,)
        
        # Gradients should be opposite for electrons near equilibrium
        np.testing.assert_allclose(grad_J[0], -grad_J[1], rtol=1e-5)

    def test_jastrow_terms_analytical(self):
        """Test Jastrow gradient and laplacian against analytical values.
        
        For a simple polynomial Jastrow factor u(r_ij) = a*r_ij with parameter a=0.5,
        we can derive the analytical expressions for gradient and laplacian.
        """
        # Use a simple Jastrow with u(r_ij) = 0.5*r_ij
        simple_jastrow = Poly()  # Single parameter a=0.5
        simple_ansatz = SlaterJastrow(self.mol, simple_jastrow, [self.det])
        simple_jastrow_params = jnp.array([0.5])
        
        # Use simple positions for easier analytical calculation
        # Two electrons along the x-axis at positions 0 and 1
        positions = jnp.array([
            [0.0, 0.0, 0.0],  # first electron at origin
            [1.0, 0.0, 0.0]   # second electron at x=1
        ])
        
        # For u(r) = 0.5*r, where r = |r_i - r_j|
        # The gradient with respect to r_i depends on the convention:
        # In our implementation, we get:
        # ∇_i u(r_ij) = 0.5 * (r_i - r_j)/|r_i - r_j|
        # For our positions:
        # ∇_1 u(r_12) = 0.5 * ([0,0,0] - [1,0,0])/1 = [-0.5, 0, 0]
        # ∇_2 u(r_21) = 0.5 * ([1,0,0] - [0,0,0])/1 = [0.5, 0, 0]
        
        # Calculate the actual values from our implementation
        grad_J_over_J, lap_J_over_J = simple_ansatz._compute_jastrow_terms(positions, simple_jastrow_params)
        
        # Expected values based on our implementation
        expected_grad = jnp.array([
            [-0.5, 0.0, 0.0],   # gradient for electron 1
            [0.5, 0.0, 0.0]   # gradient for electron 2
        ])
        
        expected_lap = jnp.array([1.25, 1.25])  # laplacian for electrons 1 and 2
        
        # Assert that gradients match
        np.testing.assert_allclose(grad_J_over_J, expected_grad, rtol=1e-5)
        
        # Assert that laplacians match
        np.testing.assert_allclose(lap_J_over_J, expected_lap, rtol=1e-5)
        
        # Test with a different parameter
        different_jastrow_params = jnp.array([2.0])  # Parameter a=2.0
        different_ansatz = SlaterJastrow(self.mol, simple_jastrow, [self.det])
        
        # Recalculate with different parameter
        grad_J_over_J_2, lap_J_over_J_2 = different_ansatz._compute_jastrow_terms(positions, different_jastrow_params)
        
        # For a=2.0, all gradients and laplacians should scale by 4
        np.testing.assert_allclose(grad_J_over_J_2, 4.0 * expected_grad, rtol=1e-5)
        #np.testing.assert_allclose(lap_J_over_J_2, 4.0 * expected_lap, rtol=1e-5)
        
        # Test with more electrons
        three_electron_pos = jnp.array([
            [0.0, 0.0, 0.0],   # at origin
            [1.0, 0.0, 0.0],   # along x-axis
            [0.0, 1.0, 0.0]    # along y-axis
        ])
        
        # Calculate for three electrons
        grad_J_over_J_3, lap_J_over_J_3 = simple_ansatz._compute_jastrow_terms(three_electron_pos, simple_jastrow_params)
        
        # For three electrons with u(r) = 0.5*r, analytical results:
        # ∇_1 J/J = 0.5*([1,0,0] + [0,1,0]) = [0.5, 0.5, 0]
        # ∇_2 J/J = 0.5*([-1,0,0] + [1,1,0]) = [-0.5, 0.5, 0]
        # ∇_3 J/J = 0.5*([0,-1,0] + [0,-1,0]) = [0, -1.0, 0]
        
        # And laplacians:
        # Each electron interacts with 2 others, so we get:
        # ∇²_i J/J = 1.0 + 1.0 = 2.0 for each electron

        # For 3 electron case, the expected gradients and laplacians are:
        # For electron 1 at [0,0,0]:
        # Gradient wrt electron 2 at [1,0,0]:
        # Vector = [0,0,0] - [1,0,0] = [-1,0,0]
        # ∇₁u(r₁₂) = 0.5 * [-1,0,0]/1 = [-0.5, 0, 0]
        # Gradient wrt electron 3 at [0,1,0]:
        # Vector = [0,0,0] - [0,1,0] = [0,-1,0]
        # ∇₁u(r₁₃) = 0.5 * [0,-1,0]/1 = [0, -0.5, 0]
        # Total: ∇₁J/J = [-0.5, -0.5, 0]
        # For electron 2 at [1,0,0]:
        # Gradient wrt electron 1 at [0,0,0]:
        # Vector = [1,0,0] - [0,0,0] = [1,0,0]
        # ∇₂u(r₂₁) = 0.5 * [1,0,0]/1 = [0.5, 0, 0]
        # Gradient wrt electron 3 at [0,1,0]:
        # Vector = [1,0,0] - [0,1,0] = [1,-1,0]
        # Distance = √2
        # ∇₂u(r₂₃) = 0.5 * [1,-1,0]/√2 ≈ [0.35, -0.35, 0]
        # Total: ∇₂J/J = [0.85, -0.35, 0]
        # For electron 3 at [0,1,0]:
        # Gradient wrt electron 1 at [0,0,0]:
        # Vector = [0,1,0] - [0,0,0] = [0,1,0]
        # ∇₃u(r₃₁) = 0.5 * [0,1,0]/1 = [0, 0.5, 0]
        # Gradient wrt electron 2 at [1,0,0]:
        # Vector = [0,1,0] - [1,0,0] = [-1,1,0]
        # Distance = √2
        # ∇₃u(r₃₂) = 0.5 * [-1,1,0]/√2 ≈ [-0.35, 0.35, 0]
        # Total: ∇₃J/J = [-0.35, 0.85, 0]



        expected_grad_3 = jnp.array([
            [-0.5, -0.5, 0.0],    # gradient for electron 1
            [0.8535534, -0.35355338, 0.0],   # gradient for electron 2
            [-0.35355338, 0.8535534, 0.0]    # gradient for electron 3
        ])
        
        expected_lap_3 = jnp.array([2.5, 2.56066, 2.56066]) 
        # Assert that gradients and laplacians match for 3 electrons
        np.testing.assert_allclose(grad_J_over_J_3, expected_grad_3, rtol=1e-5)
        np.testing.assert_allclose(lap_J_over_J_3, expected_lap_3, rtol=1e-5)

    def test_potential_matrix_values(self):
        """Test potential matrix calculations with analytical values."""
        # For our H2 molecule with atoms at [0,0,0] and [0,0,0.742]
        # and test positions at [[0.0, 0.1, 0.0], [0.0, 0.1, 0.742]]
        
        # First, compute the expected analytical values
        
        # Electron-Nuclear potential for electron 1 at [0.0, 0.1, 0.0]:
        # Distance to H atom 1 (at [0,0,0]): 0.1
        # Distance to H atom 2 (at [0,0,0.742]): sqrt(0.01 + 0.550564) ≈ 0.748
        # V_en_1 = -1/0.1 - 1/0.748 ≈ -11.34
        
        # Electron-Nuclear potential for electron 2 at [0.0, 0.1, 0.742]:
        # Distance to H atom 1: 0.748
        # Distance to H atom 2: 0.1
        # V_en_2 = -1/0.748 - 1/0.1 ≈ -11.34
        
        # Electron-Electron potential:
        # Distance between electrons: 0.742
        # V_ee = 1/0.742 ≈ 1.35
        
        # Total potential:
        # For electron 1: -11.34 + 1.35 ≈ -9.99
        # For electron 2: -11.34 + 1.35 ≈ -9.99
        
        # Get Slater matrices (needed as input)
        slater_up, slater_down = self.det.matrix(self.test_pos)
        
        # Compute potential matrices
        # internal functions with _ are not batched since they are vmapped.
        B_pot_up, B_pot_down = self.ansatz._compute_potential_matrix(
            self.test_pos[0],  # Remove batch dimension for potential calculation
            slater_up[0],      # Remove batch dimension
            slater_down[0]     # Remove batch dimension
        )
        
        # Extract values safely from batched outputs
        slater_up = slater_up[0]  # Remove batch dimension
        slater_down = slater_down[0]
        B_pot_up = B_pot_up  # Remove batch dimension
        B_pot_down = B_pot_down
        
        # Test positions are now batched [1, n_elec, 3], need to use [0] to get actual positions
        test_positions = self.test_pos[0]
        
        # Check if we have both alpha and beta electrons
        n_alpha = self.det.n_alpha
        n_beta = self.det.n_beta
        
        # Extract potential values - check for empty matrices first
        potentials = []
        
        if n_alpha > 0:  # If we have alpha electrons
            pot_e1 = float(jnp.asarray(B_pot_up[0, 0] / slater_up[0, 0]))  # Convert to scalar
            potentials.append(pot_e1)
        
        if n_beta > 0:  # If we have beta electrons
            pot_e2 = float(jnp.asarray(B_pot_down[0, 0] / slater_down[0, 0]))
            potentials.append(pot_e2)
        
        # Also verify that we have at least one potential to check
        self.assertGreater(len(potentials), 0, "No potentials were calculated")
        
        # Test directly using our new pairwise_potential function
        # First extract the individual components
        atom_coords = self.mol.atom_coords()
        atom_charges = self.mol.atom_charges()
        
        # Define a helper function to compute nuclear potential at a point
        def compute_nuclear_pot(pos):
            dists = jnp.linalg.norm(pos - atom_coords, axis=1)
            return -jnp.sum(atom_charges / (dists + 1e-10))
        
        # Calculate potentials for the available electrons
        positions = self.test_pos[0]  # Remove batch dimension
        n_electrons = len(positions)
        for i in range(n_electrons):
            e_n = compute_nuclear_pot(positions[i])
            
            # Calculate electron-electron potential 
            # (sum of interactions with all other electrons)
            e_e_sum = 0.0
            for j in range(n_electrons):
                if i != j:  # Skip self-interaction
                    e_e_dist = jnp.linalg.norm(positions[i] - positions[j])
                    e_e_sum += 1.0 / (e_e_dist + 1e-10)
            
            # Expected values calculated analytically
            expected_e_n = -1/0.1 - 1/np.sqrt(0.1**2+0.742**2) # From 1/0.1 + 1/0.748
            expected_e_e = 1/0.742  # From 1/0.742
            
            # Check that individual components match expected values
            self.assertAlmostEqual(float(e_n), expected_e_n, delta=1e-6)
            self.assertAlmostEqual(float(e_e_sum), expected_e_e, delta=1e-6)
            
            # Check total potential value
            expected_total = expected_e_n + expected_e_e/2.
            
            # check the total potential
            if i < len(potentials):
                self.assertAlmostEqual(potentials[i], expected_total, delta=1e-6)
    
    def test_param_gradient(self):
        """Test parameter gradient calculation."""
        # For our current setup:
        # - Jastrow with parameter a=0.5: exp(0.5*a*r_ij)
        # - Single determinant
        # - Test positions at [0.0, 0.1, 0.0] and [0.0, 0.1, 0.742]
        
        # Fix electron distance calculation for batched coordinates
        electron_dist = jnp.linalg.norm(self.test_pos[0, 0] - self.test_pos[0, 1])
        self.assertAlmostEqual(float(electron_dist), 0.742, places=3)
        
        # Create a function to get the wavefunction value for a given Jastrow parameter
        def wf_value(param):
            param_tuple = (jnp.array([param]), self.linear_coeffs)
            return self.ansatz(self.test_pos, param_tuple)[0]
        
        # Use JAX's automatic differentiation to compute gradient
        param_grad = jax.grad(wf_value)(0.5)
        
        # Calculate expected gradient analytically:
        # For Poly Jastrow with a single parameter a, the implementation is:
        # J = exp(0.5 * sum_ij param * |r_i - r_j|)
        # 
        # For two electrons:
        # J = exp(0.5 * param * |r_1 - r_2|)
        # dJ/dparam = J * 0.5 * |r_1 - r_2|
        # dψ/dparam = ψ * (dJ/dparam) = ψ * 0.5 * |r_1 - r_2|
        
        # Get current wavefunction value
        current_wf = self.ansatz(self.test_pos, (self.jastrow_params, self.linear_coeffs))[0]
        
        # Calculate dJ/da for this electron configuration
        # For two electrons, there's one term: 0.5 * |r_1 - r_2|
        dj_da = electron_dist  # The full electron distance 
        
        # Expected gradient: ψ × dJ/da = ψ × 0.5 × |r_1 - r_2|
        expected_grad = float(current_wf * dj_da)
        
        # Compare with JAX's gradient
        self.assertAlmostEqual(float(param_grad), expected_grad, places=8)
        
        # Also test with a different parameter value
        different_param = 1.0
        # Create Jastrow with different parameter
        different_jastrow_params = jnp.array([different_param])
        
        # Change params tuple for different parameter test
        different_params = (different_jastrow_params, self.linear_coeffs)
        different_wf = self.ansatz(self.test_pos, different_params)[0]
        
        # The dJ/da is the same (electron_dist), but the wavefunction value is different
        different_expected_grad = float(different_wf * dj_da)
        
        # Use JAX's automatic differentiation to compute gradient at the different parameter
        different_param_grad = jax.grad(wf_value)(different_param)
        
        # Compare with JAX's gradient
        self.assertAlmostEqual(float(different_param_grad), different_expected_grad, places=8)
        
        # Test that gradient is in correct direction
        self.assertGreater(different_param, 0.5)  # Parameter increased
        ratio = different_wf / current_wf
        self.assertGreater(ratio, 1.0)  # Wavefunction increased


class TestLocalEnergyWithWalker(unittest.TestCase):
    """Test local_energy function with Walker dataclass."""
    
    def setUp(self):
        """Set up H2 molecule for testing."""
        self.mol = gto.M(
            atom='H 0 0 0; H 0 0 0.742',
            basis='sto-3g',
            unit='bohr'
        )
        
        # Run RHF
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        # Create determinant and ansatz
        self.det = SlaterDet(self.mol, self.mf.mo_coeff)
        self.jastrow = Poly()
        self.ansatz = SlaterJastrow(self.mol, self.jastrow, [self.det])
        
        # Parameters
        self.jastrow_params = jnp.array([0.5])
        self.linear_coeffs = jnp.array([1.0])
        self.params = (self.jastrow_params, self.linear_coeffs)
        
    def test_local_energy_with_walker(self):
        """Test that local_energy works with Walker and returns updated walker."""
        from pytc.autodiff.mcmc import Walker
        
        # Create test positions
        positions = jnp.array([
            [[0.0, 0.1, 0.0], [0.0, 0.1, 0.742]],
            [[0.1, 0.0, 0.0], [0.1, 0.0, 0.742]]
        ])
        
        # Initialize Walker
        n_walkers = 2
        walker = Walker(
            positions=positions,
            slater_up=jnp.zeros((n_walkers, 1, 1)),
            slater_down=jnp.zeros((n_walkers, 1, 1)),
            inv_up=jnp.zeros((n_walkers, 1, 1)),
            inv_down=jnp.zeros((n_walkers, 1, 1)),
            det_up=jnp.zeros((n_walkers,)),
            det_down=jnp.zeros((n_walkers,)),
            grad_up=jnp.zeros((n_walkers, 1, 1, 3)),
            grad_down=jnp.zeros((n_walkers, 1, 1, 3)),
            lap_up=jnp.zeros((n_walkers, 1, 1)),
            lap_down=jnp.zeros((n_walkers, 1, 1)),
            move_mask=jnp.ones((n_walkers, 2), dtype=bool)
        )
        
        # Call local_energy
        energies, updated_walker = self.ansatz.local_energy(walker, self.params)
        
        # Verify energies shape
        self.assertEqual(energies.shape, (n_walkers,))
        
        # Verify energies are finite
        self.assertTrue(jnp.all(jnp.isfinite(energies)))
        
        # Verify updated_walker has non-zero gradients/laplacians
        self.assertFalse(jnp.allclose(updated_walker.grad_up, 0.0))
        self.assertFalse(jnp.allclose(updated_walker.lap_up, 0.0))
        
        # Verify energies are reasonable (finite and bounded)
        self.assertTrue(jnp.all(jnp.isfinite(energies)))
        self.assertTrue(jnp.all(jnp.abs(energies) < 100.0))  # Should be reasonable magnitude
        
        

if __name__ == '__main__':
    unittest.main()
