"""Tests for the Metropolis-Hastings sampling implementation."""

import unittest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import time

# Import PySCF-related functionality
from pyscf import gto, scf

# Import our modules
from pytc.vmc import (
    optimize, optimize_ref_var, sample, Walker, initialize_walker_state, 
    initialize_walkers, metropolis_hastings, _one_electron_move, _all_electron_move
)
from pytc.vmc.mcmc_utils import init_electron_configs

from pytc.vmc.mcmc_utils import analyze_energies
from pytc.ansatz.sj import SlaterJastrow
from pytc.jastrow import REXP, Poly, CompositeJastrow, NuclearCusp, BoysHandy
from pytc.ansatz.det import SlaterDet 



class TestJastrowFunctions(unittest.TestCase):
    """Test Jastrow factor behavior."""
    
    def test_zero_jastrow_is_identity(self):
        """Test that a Jastrow factor with zero parameters evaluates to 1."""
        jastrow = Poly()
        jastrow_params = jnp.zeros(1)
        
        key = random.PRNGKey(0)
        for _ in range(10):
            key, subkey = random.split(key)
            r1 = random.normal(subkey, (3,))
            key, subkey = random.split(key)
            r2 = random.normal(subkey, (3,))
            
            # Should evaluate to 0, making Jastrow factor exp(0) = 1
            val = jastrow._compute(r1, r2, jastrow_params)
            np.testing.assert_allclose(val, 0.0, atol=1e-10)
            
            # Derivatives should be zero
            grads, laps = jastrow.get_log_grads_r1(r1, r2, jastrow_params)
            np.testing.assert_allclose(grads, jnp.zeros(3), atol=1e-10)
            np.testing.assert_allclose(laps, 0.0, atol=1e-10)


class TestWalkerDataclass(unittest.TestCase):
    """Test Walker dataclass and related functions."""
    
    def setUp(self):
        """Set up a simple molecule for testing."""
        self.mol = gto.Mole()
        self.mol.atom = 'H 0 0 0; H 0 0 1.0'
        self.mol.basis = 'sto-3g'
        self.mol.build()
        
        # Create simple ansatz
        mf = scf.RHF(self.mol)
        mf.kernel()
        det = SlaterDet.create(self.mol, mf.mo_coeff)
        jastrow = Poly()
        self.ansatz = SlaterJastrow.create(self.mol, jastrow, [det])
        
        self.n_walkers = 10
        self.n_electrons = self.mol.nelectron
        self.n_alpha = self.ansatz.n_alpha
        self.n_beta = self.n_electrons - self.n_alpha
    
    def test_walker_initialization(self):
        """Test that Walker is initialized correctly."""
        key = random.PRNGKey(42)
        positions = init_electron_configs(
            self.mol.atom_coords(), self.mol.atom_charges(),
            self.n_electrons, self.n_walkers, key
        )
        
        walker = initialize_walker_state(self.ansatz, positions)
        
        # Check it's a Walker instance
        self.assertIsInstance(walker, Walker)
        
        # Check shapes
        self.assertEqual(walker.positions.shape, (self.n_walkers, self.n_electrons, 3))
        self.assertEqual(walker.slater_up.shape, (self.n_walkers, self.n_alpha, self.n_alpha))
        self.assertEqual(walker.slater_down.shape, (self.n_walkers, self.n_beta, self.n_beta))
        self.assertEqual(walker.inv_up.shape, (self.n_walkers, self.n_alpha, self.n_alpha))
        self.assertEqual(walker.inv_down.shape, (self.n_walkers, self.n_beta, self.n_beta))
        self.assertEqual(walker.det_up[0].shape, (self.n_walkers,))
        self.assertEqual(walker.det_up[1].shape, (self.n_walkers,))
        self.assertEqual(walker.det_down[0].shape, (self.n_walkers,))
        self.assertEqual(walker.det_down[1].shape, (self.n_walkers,))
        self.assertEqual(walker.move_mask.shape, (self.n_walkers, self.n_electrons))
        
        # Check move_mask is all True initially
        self.assertTrue(jnp.all(walker.move_mask))
        
        # Check other fields are zeros
        self.assertTrue(jnp.allclose(walker.slater_up, 0.0))
        self.assertTrue(jnp.allclose(walker.det_up[0], 0.0))
        self.assertTrue(jnp.allclose(walker.det_up[1], 0.0))
    
    def test_initialize_walkers(self):
        """Test initialize_walkers function."""
        key = random.PRNGKey(42)
        walker = initialize_walkers(self.ansatz, self.n_walkers, key=key)
        
        # Check it returns a Walker
        self.assertIsInstance(walker, Walker)
        self.assertEqual(walker.positions.shape, (self.n_walkers, self.n_electrons, 3))
        self.assertTrue(jnp.all(walker.move_mask))
    
    def test_walker_immutability(self):
        """Test that Walker.replace creates new instance."""
        key = random.PRNGKey(42)
        walker = initialize_walkers(self.ansatz, self.n_walkers, key=key)
        
        # Create new walker with modified positions
        new_positions = walker.positions + 0.1
        new_walker = walker.replace(positions=new_positions)
        
        # Original walker should be unchanged
        self.assertFalse(jnp.allclose(walker.positions, new_walker.positions))
        self.assertTrue(jnp.allclose(new_walker.positions, walker.positions + 0.1))
    
    def test_one_electron_move_mask(self):
        """Test that _one_electron_move sets move_mask correctly."""
        key = random.PRNGKey(42)
        walker = initialize_walkers(self.ansatz, self.n_walkers, key=key)
        
        # Reset move_mask to False for current walker
        walker = walker.replace(move_mask=jnp.zeros_like(walker.move_mask))
        
        # Create parameters
        jastrow_params = jnp.zeros(1)
        linear_coeffs = jnp.ones(1)
        params = [jastrow_params, linear_coeffs]
        
        # Perform one electron move
        key, subkey = random.split(key)
        psi_old, psi_new, walker_updated, proposals = _one_electron_move(
            self.ansatz, walker, step_size=0.1, key=subkey, params=params
        )
        
        # Check that proposals have exactly one True per walker
        n_true_per_walker = jnp.sum(proposals.move_mask, axis=1)
        self.assertTrue(jnp.all(n_true_per_walker == 1))
        
        # Check that positions changed only for masked electrons
        for i in range(self.n_walkers):
            electron_idx = jnp.where(proposals.move_mask[i])[0][0]
            # Moved electron should have different position
            self.assertFalse(jnp.allclose(
                walker.positions[i, electron_idx], 
                proposals.positions[i, electron_idx]
            ))
            # Other electrons should have same position
            for j in range(self.n_electrons):
                if j != electron_idx:
                    self.assertTrue(jnp.allclose(
                        walker.positions[i, j], 
                        proposals.positions[i, j]
                    ))
    
    def test_all_electron_move_mask(self):
        """Test that _all_electron_move sets move_mask to all True."""
        key = random.PRNGKey(42)
        walker = initialize_walkers(self.ansatz, self.n_walkers, key=key)
        
        # Reset move_mask to False
        walker = walker.replace(move_mask=jnp.zeros_like(walker.move_mask))
        
        # Create parameters
        jastrow_params = jnp.zeros(1)
        linear_coeffs = jnp.ones(1)
        params = [jastrow_params, linear_coeffs]
        
        # Perform all electron move
        key, subkey = random.split(key)
        psi_old, psi_new, walker_updated, proposals = _all_electron_move(
            self.ansatz, walker, step_size=0.1, key=subkey, params=params
        )
        
        # Check that proposals have all True
        self.assertTrue(jnp.all(proposals.move_mask))
        
        # Check that all positions changed
        self.assertFalse(jnp.allclose(walker.positions, proposals.positions))
    
    def test_metropolis_hastings_resets_mask(self):
        """Test that metropolis_hastings resets move_mask after acceptance."""
        key = random.PRNGKey(42)
        walker = initialize_walkers(self.ansatz, self.n_walkers, key=key)
        
        # Reset move_mask to False
        walker = walker.replace(move_mask=jnp.zeros_like(walker.move_mask))
        
        # Create parameters
        jastrow_params = jnp.zeros(1)
        linear_coeffs = jnp.ones(1)
        params = [jastrow_params, linear_coeffs]
        
        # Perform one MH step
        key, subkey = random.split(key)
        new_walker, acceptance_rate = metropolis_hastings(
            self.ansatz, walker, step_size=0.1, key=subkey, 
            params=params, move_type="one"
        )
        
        # Check that move_mask is reset to all False
        self.assertTrue(jnp.all(~new_walker.move_mask))
        
        # Check acceptance_rate is reasonable
        self.assertGreaterEqual(acceptance_rate, 0.0)
        self.assertLessEqual(acceptance_rate, 1.0)


class TestElectronInitialization(unittest.TestCase):
    """Test electron configuration initialization."""
    
    def test_init_electron_configs(self):
        """Test that electron configuration initialization is reasonable."""
        mol = gto.Mole()
        mol.atom = 'H 0 0 0; H 0 0 1.0'
        mol.basis = 'sto-3g'
        mol.build()
        
        atom_coords = mol.atom_coords()
        atom_charges = mol.atom_charges()
        n_electrons = mol.nelectron
        n_walkers = 5
        
        key = random.PRNGKey(0)
        configs = init_electron_configs(atom_coords, atom_charges, n_electrons, n_walkers, key)
        
        # Check shape and distribution
        self.assertEqual(configs.shape, (n_walkers, n_electrons, 3))
        
        # Check electrons are reasonably close to nuclei
        for i in range(n_walkers):
            for j in range(n_electrons):
                pos = configs[i, j]
                min_dist = min(jnp.linalg.norm(pos - atom_pos) for atom_pos in atom_coords)
                self.assertLess(min_dist, 5.0, f"Electron too far from nuclei: {min_dist} bohr")


class TestHartreeFockEnergy(unittest.TestCase):
    """Test that HF energy is correctly reproduced via sampling."""
    
    def run_hf_energy_test(self, molecule_spec):
        """Run HF energy test on the specified molecule."""
        # Create molecule
        mol = gto.Mole()
        mol.atom = molecule_spec
        mol.basis = 'sto6g'
        mol.unit = 'A'
        mol.build()
        
        # Run PySCF calculation for reference energy
        mf = scf.RHF(mol)
        mf.kernel()
        hf_energy_reference = mf.e_tot
        
        # Extract orbitals and occupation
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        
        # Create determinant from HF solution
        det = SlaterDet.create(mol, mo_coeff)
        
        # Create PolyJastrow with zero parameters (equals identity)
        jastrow = Poly()
        jastrow_params = jnp.zeros(1)
        
        # Create SlaterJastrow ansatz (equivalent to HF with Jastrow=1)
        sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
        jastrow_params = jnp.zeros(1)  # Initialize to zero for HF test
        linear_coeffs = jnp.ones(1)  # Single determinant
        
        # Use small settings for test speed
        # For production, use larger values
        n_walkers = 5000
        n_steps = 5000
        step_size = 0.1
        burn_in_steps = 1000
        thinning = 10
        key = random.PRNGKey(42)  # Fixed seed for reproducibility
        
        # Run sampling
        print(f"Starting sampling for {mol.atom}...")
        start_time = time.time()
        sampling_results = sample(
            sj_ansatz,
            params=[jastrow_params, linear_coeffs],
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            use_importance_sampling=True,
            burn_in_steps=burn_in_steps,  # Updated parameter name
            thinning=thinning,
            key=key
        )
        end_time = time.time()
        print(f"Sampling completed in {end_time - start_time:.2f} seconds")
        
        # Pass plot=False to avoid opening matplotlib windows during tests
        energy_stats = analyze_energies(sampling_results)
        
        # Extract mean and error
        energy_mean = float(energy_stats["mean"])
        energy_error = float(energy_stats["error"])
        
        # Print results
        print(f"Reference HF energy: {hf_energy_reference:.6f}")
        print(f"Sampled energy: {energy_mean:.6f} ± {energy_error:.6f}")
        
        # Check if energies agree within a reasonable tolerance
        rel_error = abs(energy_mean - hf_energy_reference) / abs(hf_energy_reference)
        
        self.assertLessEqual(abs(energy_mean - hf_energy_reference), 3 * energy_error,
                            "Reference energy outside 3-sigma error bars of sampled energy")
        
        # Return values to be used in other tests if needed
        return {
            "reference_energy": hf_energy_reference,
            "sampled_energy": energy_mean,
            "energy_error": energy_error,
            "sampling_results": sampling_results
        }
    
    def test_be_atom(self):
        """Test HF energy sampling for Be atom."""
        results = self.run_hf_energy_test("Be 0 0 0")
    def test_lih(self):
        """Test HF energy sampling for LiH molecule."""
        results = self.run_hf_energy_test("Li 0 0 0; H 0 0 1.6")



class TestJastrowOptimization(unittest.TestCase):
    """Test optimization of the Jastrow factor."""
    
    def run_optimization_test(self, molecule_spec, jastrow_params=None, basis='sto-3g'):
        """Run optimization test on the specified molecule."""
        # Create molecule
        mol = gto.Mole()
        mol.atom = molecule_spec
        mol.basis = basis
        mol.unit = 'A'
        mol.cart = False
        mol.build()
        
        # Run PySCF calculation for reference energy
        mf = scf.RHF(mol)
        mf.kernel()
        hf_energy_reference = mf.e_tot


        
        # Create determinant from HF solution
        det = SlaterDet.create(mol, mf.mo_coeff)
        
        # Create REXP jastrow with given or default parameters
        rexp = REXP()
        bh = BoysHandy.create(mol)
        jnuclear_cusp = NuclearCusp.create(mol)    
        #jastrow = NuclearCusp(mol)    
        jastrow = CompositeJastrow.create([jnuclear_cusp, bh])
        jastrow_params = jastrow.init_params() if jastrow_params is None else jastrow_params 
        # Create SlaterJastrow ansatz
        sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
        linear_coeffs = jnp.ones(1)  # Single determinant
        
        # Use small settings for test speed
        n_walkers = 1000
        n_steps = 20
        step_size = 0.01
        burn_in_steps = 1000
        n_opt_steps = 100
        key = random.PRNGKey(42)
        
        # Run optimization
        print(f"Starting Jastrow optimization for {mol.atom}...")
        start_time = time.time()
        opt_results = optimize_ref_var(
            sj_ansatz,
            params=[jastrow_params, linear_coeffs],
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            n_opt_steps=n_opt_steps,
            optimizer_type='newton',
            learning_rate=0.1,
            max_vmap_batch_size=0,
            opt_kwargs={'damping': 1e-6, 'solver': 'exact'},
            key=key
        )
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        
        # Check energy improvement
        initial_energy = jnp.asarray(opt_results["energies"][:500]).mean()
        final_energy = jnp.asarray(opt_results["energies"][-500:]).mean()
        print(f"Initial energy: {initial_energy:.6f}")
        print(f"Final energy: {final_energy:.6f}")
        print(f"Reference HF energy: {hf_energy_reference:.6f}")
        
        return opt_results
    
    def test_be(self):
        """Test optimization of Jastrow parameters for Be atom."""
        self.run_optimization_test('Be 0 0 0;', basis='ccpvtz')
    


if __name__ == "__main__":
    unittest.main()
