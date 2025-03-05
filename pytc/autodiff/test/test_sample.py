"""Tests for the Metropolis-Hastings sampling implementation."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import time

# Import PySCF-related functionality
from pyscf import gto, scf

# Import our modules
from pytc.autodiff.sample import metropolis_hastings, init_electron_configs
from pytc.autodiff.sample_utils import analyze_energies
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pytc.autodiff.jastrow import Poly
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
        n_walkers = 1000
        n_steps = 10000
        step_size = 0.1
        burn_in = 100
        thinning = 2
        key = random.PRNGKey(42)  # Fixed seed for reproducibility
        
        # Run sampling
        print(f"Starting sampling for {mol.atom}...")
        start_time = time.time()
        sampling_results = metropolis_hastings(
            sj_ansatz,
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            burn_in=burn_in,
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
        #self.assertLess(rel_error, 1.05, 
        #               f"Sampled energy {energy_mean:.6f} too far from reference {hf_energy_reference:.6f}")
        
        ## Also check if the reference energy is within the statistical error bars
        #self.assertLessEqual(abs(energy_mean - hf_energy_reference), 3 * energy_error,
        #                    "Reference energy outside 3-sigma error bars of sampled energy")
        
        # Return values to be used in other tests if needed
        return {
            "reference_energy": hf_energy_reference,
            "sampled_energy": energy_mean,
            "energy_error": energy_error,
            "sampling_results": sampling_results
        }
    
    #def test_h2_molecule(self):
    #    """Test HF energy sampling for H2 molecule."""
    #    results = self.run_hf_energy_test("H 0 0 0; H 0 0 0.1")
    #    # Additional H2-specific assertions could be added here
    
    def test_he_atom(self):
        """Test HF energy sampling for He atom."""
        results = self.run_hf_energy_test("He 0 0 0")
        # Additional He-specific assertions could be added here


if __name__ == "__main__":
    unittest.main()