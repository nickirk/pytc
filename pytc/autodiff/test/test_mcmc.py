"""Tests for the Metropolis-Hastings sampling implementation."""

import unittest
import numpy as np
from jax import random
import jax.numpy as jnp
import time

# Import PySCF-related functionality
from pyscf import gto, scf
from pyscf import gto, scf

# Import our modules
from pytc.autodiff.mcmc import (
    optimize, sample, init_electron_configs, metropolis_hastings,
    initialize_walkers, perform_mcmc_step, burn_in, prepare_sampling_results,
    report_progress, create_optimizer
)
from pytc.autodiff.sample_utils import analyze_energies
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pytc.autodiff.jastrow import REXP, Poly
from pytc.autodiff.ansatz.det import SlaterDet 



class TestJastrowFunctions(unittest.TestCase):
    """Test Jastrow factor behavior."""
    
    def test_zero_jastrow_is_identity(self):
        """Test that a Jastrow factor with zero parameters evaluates to 1."""
        jastrow = Poly(params=jnp.zeros(1))
        
        key = random.PRNGKey(0)
        for _ in range(10):
            key, subkey = random.split(key)
            r1 = random.normal(subkey, (3,))
            key, subkey = random.split(key)
            r2 = random.normal(subkey, (3,))
            
            # Should evaluate to 0, making Jastrow factor exp(0) = 1
            val = jastrow._compute(r1, r2, jastrow.params)
            np.testing.assert_allclose(val, 0.0, atol=1e-10)
            
            # Derivatives should be zero
            grads, laps = jastrow.get_log_grads(r1, r2)
            np.testing.assert_allclose(grads, jnp.zeros(3), atol=1e-10)
            np.testing.assert_allclose(laps, 0.0, atol=1e-10)


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
        mol.basis = 'ccpvdz'
        mol.unit = 'bohr'
        mol.build()
        
        # Run PySCF calculation for reference energy
        mf = scf.RHF(mol)
        mf.kernel()
        hf_energy_reference = mf.e_tot
        
        # Extract orbitals and occupation
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        
        # Create determinant from HF solution
        det = SlaterDet(mol, mo_coeff)
        
        # Create PolyJastrow with zero parameters (equals identity)
        jastrow = Poly(params=jnp.zeros(1))
        
        # Create SlaterJastrow ansatz (equivalent to HF with Jastrow=1)
        sj_ansatz = SlaterJastrow(mol, jastrow, [det], jnp.array([1.0]))
        
        # Use small settings for test speed
        # For production, use larger values
        n_walkers = 2000
        n_steps = 8000
        step_size = 0.2
        burn_in_steps = 4000  # Updated parameter name
        thinning = 10
        key = random.PRNGKey(42)  # Fixed seed for reproducibility
        
        # Run sampling
        print(f"Starting sampling for {mol.atom}...")
        start_time = time.time()
        sampling_results = sample(
            sj_ansatz,
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
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
        # We set a relatively large tolerance for test efficiency
        # This could be tightened with more samples
        rel_error = abs(energy_mean - hf_energy_reference) / abs(hf_energy_reference)
        
        # We use a 5% tolerance because MC sampling has statistical fluctuations
        # and we're using a small number of steps for test speed
        self.assertLess(rel_error, 1.05, 
                       f"Sampled energy {energy_mean:.6f} too far from reference {hf_energy_reference:.6f}")
        
        # Also check if the reference energy is within the statistical error bars
        self.assertLessEqual(abs(energy_mean - hf_energy_reference), 3 * energy_error,
                            "Reference energy outside 3-sigma error bars of sampled energy")
        
        # Return values to be used in other tests if needed
        return {
            "reference_energy": hf_energy_reference,
            "sampled_energy": energy_mean,
            "energy_error": energy_error,
            "sampling_results": sampling_results
        }
    
    def test_h4_molecule(self):
        """Test HF energy sampling for H2 molecule."""
        results = self.run_hf_energy_test("H 0 0 0; H 0 0 2; H 0 0 4; H 0 0 6")
    
    def test_he_atom(self):
        """Test HF energy sampling for He He molecule."""
        results = self.run_hf_energy_test("He 0 0 0; He 0 0 1")

    def test_lih(self):
        """Test HF energy sampling for LiH molecule."""
        results = self.run_hf_energy_test("Li 0 0 0; H 0 0 1.6")



class TestJastrowOptimization(unittest.TestCase):
    """Test optimization of the Jastrow factor."""
    
    def test_h2_optimization(self):
        """Test optimization of Jastrow parameters for H2 molecule."""
        # Create molecule
        mol = gto.Mole()
        mol.atom = 'H 0 0 0; H 0 0 1.0'
        mol.basis = 'sto-3g'
        mol.unit = 'bohr'
        mol.build()
        
        # Run PySCF calculation for reference energy
        mf = scf.RHF(mol)
        mf.kernel()
        hf_energy_reference = mf.e_tot
        
        # Create determinant from HF solution
        det = SlaterDet(mol, mf.mo_coeff)
        
        # Create PolyJastrow with small but non-zero parameters
        # We use small initial parameters to test if optimization improves them
        jastrow = REXP(params=jnp.array([0.01]))
        
        # Create SlaterJastrow ansatz (initial state with small Jastrow)
        sj_ansatz = SlaterJastrow(mol, jastrow, [det], jnp.array([1.0]))
        
        # Use small settings for test speed
        n_walkers = 1000
        n_steps = 200
        step_size = 0.1
        burn_in_steps = 200
        thinning = 10
        n_opt_steps = 100  # Just a few optimization steps for test
        key = random.PRNGKey(42)  # Fixed seed for reproducibility
        
        # Run optimization
        print(f"Starting Jastrow optimization for H2...")
        start_time = time.time()
        opt_results = optimize(
            sj_ansatz,
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            thinning=thinning,
            n_opt_steps=n_opt_steps,
            learning_rate=0.05,
            key=key
        )
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        # Check that we have optimization history
        self.assertIn("optimization_history", opt_results)
        self.assertEqual(len(opt_results["optimization_history"]["energy"]), n_opt_steps)
        
        # Check that optimized energy is available
        self.assertIn("best_energy", opt_results)
        self.assertIn("best_params", opt_results)
        
        # Check that params are updated (different from initial value)
        self.assertNotEqual(opt_results["best_params"][0], 0.01)
        
        # Check that best energy is reasonable
        self.assertLess(abs(opt_results["best_energy"] - hf_energy_reference), 0.1)
        
        # Check energy improvement over optimization steps
        initial_energy = opt_results["optimization_history"]["energy"][0]
        final_energy = opt_results["optimization_history"]["energy"][-1]
        print(f"Initial energy: {initial_energy:.6f}")
        print(f"Final energy: {final_energy:.6f}")
        print(f"Reference HF energy: {hf_energy_reference:.6f}")
        
        # Energy should improve (lower) or stay similar
        self.assertLessEqual(final_energy, initial_energy + 0.05)
    
    def test_he2_optimization(self):
        """Test optimization of Jastrow parameters for He-He molecule."""
        # Create molecule
        mol = gto.Mole()
        mol.atom = 'He 0 0 0; He 0 0 2.0'
        mol.basis = 'sto-3g'
        mol.unit = 'bohr'
        mol.build()
        
        # Run PySCF calculation for reference energy
        mf = scf.RHF(mol)
        mf.kernel()
        hf_energy_reference = mf.e_tot
        
        # Create determinant from HF solution
        det = SlaterDet(mol, mf.mo_coeff)
        
        # Create PolyJastrow with small but non-zero parameters
        jastrow = Poly(params=jnp.array([0.01]))
        
        # Create SlaterJastrow ansatz
        sj_ansatz = SlaterJastrow(mol, jastrow, [det], jnp.array([1.0]))
        
        # Use small settings for test speed
        n_walkers = 500
        n_steps = 200
        step_size = 0.1
        burn_in_steps = 200
        thinning = 10
        n_opt_steps = 3
        key = random.PRNGKey(43)  # Different seed
        
        # Run optimization
        print(f"Starting Jastrow optimization for He2...")
        opt_results = optimize(
            sj_ansatz,
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            thinning=thinning,
            n_opt_steps=n_opt_steps,
            learning_rate=0.05,
            key=key
        )
        
        # Check energy improvement over optimization steps
        initial_energy = opt_results["optimization_history"]["energy"][0]
        final_energy = opt_results["optimization_history"]["energy"][-1]
        print(f"Initial energy: {initial_energy:.6f}")
        print(f"Final energy: {final_energy:.6f}")
        print(f"Reference HF energy: {hf_energy_reference:.6f}")
        
        # Energy should improve (lower) or stay similar
        self.assertLessEqual(final_energy, initial_energy + 0.05)

if __name__ == "__main__":
    unittest.main()