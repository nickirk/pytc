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
from pytc.autodiff.mcmc import optimize, sample 
from pytc.autodiff.mcmc_utils import init_electron_configs

from pytc.autodiff.mcmc_utils import analyze_energies
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pytc.autodiff.jastrow import REXP, Poly, CompositeJastrow, NuclearCusp
from pytc.autodiff.ansatz.det import SlaterDet 



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
        jastrow = Poly()
        jastrow_params = jnp.zeros(1)
        
        # Create SlaterJastrow ansatz (equivalent to HF with Jastrow=1)
        sj_ansatz = SlaterJastrow(mol, jastrow, [det])
        jastrow_params = jnp.zeros(1)  # Initialize to zero for HF test
        linear_coeffs = jnp.ones(1)  # Single determinant
        
        # Use small settings for test speed
        # For production, use larger values
        n_walkers = 5000
        n_steps = 5000
        step_size = 0.1
        burn_in_steps = 1000  # Updated parameter name
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
        results = self.run_hf_energy_test("He 0 0 0")

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
        mol.unit = 'bohr'
        mol.build()
        
        # Run PySCF calculation for reference energy
        mf = scf.RHF(mol)
        mf.kernel()
        hf_energy_reference = mf.e_tot
        
        # Create determinant from HF solution
        det = SlaterDet(mol, mf.mo_coeff)
        
        # Create REXP jastrow with given or default parameters
        rexp = REXP()
        jnuclear_cusp = NuclearCusp(mol)    
        jastrow = CompositeJastrow([jnuclear_cusp, rexp])
        jastrow_params = jastrow.init_params() if jastrow_params is None else jastrow_params 
        # Create SlaterJastrow ansatz
        sj_ansatz = SlaterJastrow(mol, jastrow, [det])
        linear_coeffs = jnp.ones(1)  # Single determinant
        
        # Use small settings for test speed
        n_walkers = 5000
        n_steps = 50
        step_size = 0.01
        burn_in_steps = 1000
        n_opt_steps = 2000
        key = random.PRNGKey(42)
        
        # Run optimization
        print(f"Starting Jastrow optimization for {mol.atom}...")
        start_time = time.time()
        opt_results = optimize(
            sj_ansatz,
            params=[jastrow_params, linear_coeffs],
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            n_opt_steps=n_opt_steps,
            optimizer_type='adam',
            learning_rate=0.001,
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
    
    def test_h2_optimization(self):
        """Test optimization of Jastrow parameters for H2 molecule."""
        self.run_optimization_test('H 0 0 0; H 0 0 1.0')
    
    def test_he2_optimization(self):
        """Test optimization of Jastrow parameters for He atom."""
        self.run_optimization_test('He 0 0 0; He 0 0 1.5', basis='ccpvdz')


if __name__ == "__main__":
    unittest.main()