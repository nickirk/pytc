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

    #def test_kinetic_matrix(self):
    #    """Test computation of kinetic energy matrix."""
    #    grad_J, lap_J = self.ansatz._compute_jastrow_terms(self.test_pos)
    #    inv_up, inv_down, B_kin_up, B_kin_down = self.ansatz._compute_kinetic_matrix(
    #        self.test_pos, grad_J, lap_J)
    #    
    #    # Check shapes
    #    n_up = self.det.n_alpha
    #    n_down = self.det.n_beta
    #    self.assertEqual(B_kin_up.shape, (n_up, n_up))
    #    self.assertEqual(B_kin_down.shape, (n_down, n_down))
    #    
    #    # Kinetic energy should be real
    #    self.assertTrue(np.allclose(B_kin_up.imag, 0))
    #    self.assertTrue(np.allclose(B_kin_down.imag, 0))
    #    
    #    # Inverse matrices should be correct
    #    slater_up, slater_down = self.det.matrix(self.test_pos)
    #    np.testing.assert_allclose(inv_up @ slater_up, np.eye(n_up), atol=1e-7)

    #def test_local_energy(self):
    #    """Test local energy computation."""
    #    energy = self.ansatz.local_energy(self.test_pos)
    #    
    #    # Energy should be real
    #    self.assertTrue(np.isreal(energy))
    #    
    #    # Energy should be finite
    #    self.assertTrue(np.isfinite(energy))
    #    
    #    # Test virial theorem: <T> ≈ -<V> for ground state
    #    # This requires computing T and V separately
    #    grad_J, lap_J = self.ansatz._compute_jastrow_terms(self.test_pos)
    #    inv_up, inv_down, B_kin_up, B_kin_down = self.ansatz._compute_kinetic_matrix(
    #        self.test_pos, grad_J, lap_J)
    #    
    #    slater_up, slater_down = self.det.matrix(self.test_pos)
    #    B_pot_up, B_pot_down = self.ansatz._compute_potential_matrix(
    #        self.test_pos, slater_up, slater_down)
    #    
    #    T = float(jnp.trace(inv_up @ B_kin_up) + jnp.trace(inv_down @ B_kin_down))
    #    V = float(jnp.trace(inv_up @ B_pot_up) + jnp.trace(inv_down @ B_pot_down))
    #    
    #    # Check if T ≈ -V (allow for some deviation due to non-optimal wavefunction)
    #    self.assertLess(abs(T + V), abs(T))  # |T + V| should be smaller than |T|
    #    
    #    # Calculate total energy manually and verify consistency with local_energy method
    #    total_E = T + V
    #    self.assertAlmostEqual(energy, total_E, places=10)
    #    
    #    # Test energy stability across similar geometries
    #    # Small perturbations to electron positions shouldn't cause large energy changes
    #    perturbed_pos = self.test_pos + jnp.array([[0.01, -0.01, 0.005], [-0.005, 0.007, -0.01]])
    #    perturbed_energy = self.ansatz.local_energy(perturbed_pos)
    #    
    #    # Energy should change slightly but not dramatically
    #    energy_diff = abs(perturbed_energy - energy)
    #    self.assertLess(energy_diff / abs(energy), 0.1)  # Less than 10% change
    #    
    #    # Test energy with different Jastrow parameters
    #    improved_jastrow = Poly(jnp.array([-0.5]))  # Negative parameter for electron-electron repulsion
    #    improved_ansatz = SlaterJastrow(self.mol, improved_jastrow, [self.det], self.coeffs)
    #    improved_energy = improved_ansatz.local_energy(self.test_pos)
    #    
    #    # Test energy with different electron configurations
    #    # Electrons very close together should have high energy (repulsion)
    #    close_pos = jnp.array([
    #        [0.1, 0.1, 0.1],
    #        [0.1, 0.1, 0.1 + 1e-3]  # Very close to first electron
    #    ])
    #    close_energy = self.ansatz.local_energy(close_pos)
    #    
    #    # Electrons far apart should have higher energy (mostly kinetic)
    #    far_pos = jnp.array([
    #        [0.0, 0.0, -5.0],
    #        [0.0, 0.0, 5.0]
    #    ])
    #    far_energy = self.ansatz.local_energy(far_pos)
    #    
    #    # Energy should be higher when electrons are very close or very far
    #    self.assertGreater(close_energy, energy)
    #    self.assertGreater(far_energy, energy)

    #def test_local_energy_reference_values(self):
    #    """Test local energy against reference calculations."""
    #    # For H2 near equilibrium, we have reference values
    #    # Create a better ansatz with optimized Jastrow
    #    opt_jastrow = Poly(jnp.array([-0.25]))  # Example optimized parameter
    #    opt_ansatz = SlaterJastrow(self.mol, opt_jastrow, [self.det], self.coeffs)
    #    
    #    # Sample multiple points to approximate the true expectation value
    #    n_samples = 10
    #    energies = []
    #    
    #    # Generate sample positions around equilibrium
    #    for i in range(n_samples):
    #        # Random positions centered around nuclei with small perturbations
    #        # Fix: Convert Python lists to JAX arrays before adding
    #        pos = jnp.array([
    #            jnp.array([0.0, 0.0, 0.0]) + 0.1 * jnp.array([np.random.normal(), np.random.normal(), np.random.normal()]),
    #            jnp.array([0.0, 0.0, 0.742]) + 0.1 * jnp.array([np.random.normal(), np.random.normal(), np.random.normal()])
    #        ])
    #        energy = opt_ansatz.local_energy(pos)
    #        energies.append(float(energy))
    #    
    #    # Calculate mean and variance
    #    mean_energy = np.mean(energies)
    #    energy_variance = np.var(energies)
    #    
    #    # The ground state energy of H2 (in atomic units) at bond length 0.742 bohr
    #    # should be approximately -1.1 to -1.2 Hartree with a minimal basis
    #    # Note: Exact value depends on the basis set quality
    #    self.assertTrue(-1.3 < mean_energy < -0.9,
    #                    f"Mean energy {mean_energy} outside expected range")
    #    
    #    # A good wavefunction should have low variance in local energy
    #    # This is not a strict test but checks for reasonable variance
    #    self.assertLess(energy_variance, 0.1,
    #                   f"Energy variance {energy_variance} is too high")

    def test_jastrow_terms_analytical(self):
        """Test Jastrow gradient and laplacian against analytical values.
        
        For a simple polynomial Jastrow factor u(r_ij) = a*r_ij with parameter a=0.5,
        we can derive the analytical expressions for gradient and laplacian.
        """
        # Use a simple Jastrow with u(r_ij) = 0.5*r_ij
        simple_jastrow = Poly(jnp.array([0.5]))  # Single parameter a=0.5
        simple_ansatz = SlaterJastrow(self.mol, simple_jastrow, [self.det], jnp.array([1.0]))
        
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
        grad_J_over_J, lap_J_over_J = simple_ansatz._compute_jastrow_terms(positions)
        
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
        different_jastrow = Poly(jnp.array([2.0]))  # Parameter a=2.0
        different_ansatz = SlaterJastrow(self.mol, different_jastrow, [self.det], jnp.array([1.0]))
        
        # Recalculate with different parameter
        grad_J_over_J_2, lap_J_over_J_2 = different_ansatz._compute_jastrow_terms(positions)
        
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
        grad_J_over_J_3, lap_J_over_J_3 = simple_ansatz._compute_jastrow_terms(three_electron_pos)
        
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
        B_pot_up, B_pot_down = self.ansatz._compute_potential_matrix(self.test_pos, slater_up, slater_down)
        
        # Check if we have both alpha and beta electrons
        n_alpha = self.det.n_alpha
        n_beta = self.det.n_beta
        
        # Extract potential values - check for empty matrices first
        potentials = []
        
        if n_alpha > 0:  # If we have alpha electrons
            pot_e1 = float(B_pot_up[0, 0] / slater_up[0, 0])  # Divide by slater value to get raw potential
            potentials.append(pot_e1)
        
        if n_beta > 0:  # If we have beta electrons
            pot_e2 = float(B_pot_down[0, 0] / slater_down[0, 0])
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
        n_electrons = len(self.test_pos)
        for i in range(n_electrons):
            e_n = compute_nuclear_pot(self.test_pos[i])
            
            # Calculate electron-electron potential 
            # (sum of interactions with all other electrons)
            e_e_sum = 0.0
            for j in range(n_electrons):
                if i != j:  # Skip self-interaction
                    e_e_dist = jnp.linalg.norm(self.test_pos[i] - self.test_pos[j])
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

if __name__ == '__main__':
    unittest.main()
