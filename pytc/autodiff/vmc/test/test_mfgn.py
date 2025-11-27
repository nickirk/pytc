
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
from pytc.autodiff.vmc import optimize_ref_var
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pytc.autodiff.jastrow import Poly, CompositeJastrow, NuclearCusp, BoysHandy
from pytc.autodiff.ansatz.det import SlaterDet 

class TestMFGNOptimization(unittest.TestCase):
    """Test optimization of the Jastrow factor using Matrix-Free Gauss-Newton."""
    
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
        
        # Create Jastrow
        # Using simple Poly Jastrow for faster testing
        jnuc = NuclearCusp.create(mol)
        jastrow = CompositeJastrow.create([jnuc, BoysHandy.create(mol)])
        jastrow_params = jastrow.init_params() if jastrow_params is None else jastrow_params 
        
        # Create SlaterJastrow ansatz
        sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
        linear_coeffs = jnp.ones(1)  # Single determinant
        
        # Use small settings for test speed
        n_walkers = 2000
        n_steps = 10
        step_size = 0.01
        burn_in_steps = 1000
        n_opt_steps = 1000
        key = random.PRNGKey(42)
        
        # Run optimization with MFGN
        print(f"Starting MFGN optimization for {mol.atom}...")
        start_time = time.time()
        opt_results = optimize_ref_var(
            sj_ansatz,
            params=[jastrow_params, linear_coeffs],
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            n_opt_steps=n_opt_steps,
            optimizer_type='mfgn',
            opt_kwargs={'damping': 1e-5, 'maxiter': 10},
            key=key
        )
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        # Check variance improvement
        initial_variance = opt_results["cost"][0]
        final_variance = opt_results["cost"][-1]
        
        print(f"Initial variance: {initial_variance:.6f}")
        print(f"Final variance: {final_variance:.6f}")
        
        # Variance should decrease
        self.assertLess(final_variance, initial_variance)
        
        return opt_results
    
    def test_be_mfgn(self):
        """Test MFGN optimization for Be atom."""
        self.run_optimization_test('Be 0 0 0', basis='ccpvdz')

if __name__ == "__main__":
    unittest.main()
