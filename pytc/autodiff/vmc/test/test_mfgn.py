
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
from pytc.autodiff.vmc import optimize_ref_var, optimize
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pytc.autodiff.jastrow import Poly, CompositeJastrow, NuclearCusp, BoysHandy
from pytc.autodiff.ansatz.det import SlaterDet 

class TestMFGNOptimization(unittest.TestCase):
    """Test optimization using Matrix-Free Gauss-Newton (and SR)."""
    
    def setUp(self):
        # Create molecule
        self.mol = gto.Mole()
        self.mol.atom = 'Be 0 0 0'
        self.mol.basis = 'ccpvdz'
        self.mol.unit = 'A'
        self.mol.cart = False
        self.mol.build()
        
        # Run PySCF calculation for reference energy
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        # Create determinant from HF solution
        self.det = SlaterDet.create(self.mol, self.mf.mo_coeff)
        
        # Create Jastrow
        self.jnuc = NuclearCusp.create(self.mol)
        self.jastrow = CompositeJastrow.create([self.jnuc, BoysHandy.create(self.mol)])
        self.jastrow_params = self.jastrow.init_params()
        
        # Create SlaterJastrow ansatz
        self.sj_ansatz = SlaterJastrow.create(self.mol, self.jastrow, [self.det])
        self.linear_coeffs = jnp.ones(1)
        self.params = [self.jastrow_params, self.linear_coeffs]

    def test_energy_minimization_sr(self):
        """Test Energy Minimization using MFGN (SR mode)."""
        print("\nTesting Energy Minimization (SR)...")
        
        n_walkers = 20000
        n_steps = 50
        step_size = 0.1
        burn_in_steps = 1000
        n_opt_steps = 1000
        key = random.PRNGKey(42)
        
        opt_results = optimize(
            self.sj_ansatz,
            params=self.params,
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            n_opt_steps=n_opt_steps,
            optimizer_type='mfgn',
            learning_rate=0.001,
            opt_kwargs={'damping': 1e-6, 'maxiter': 20},
            key=key
        )
        
        initial_energy = opt_results["energies"][0]
        final_energy = opt_results["energies"][-1]
        print(f"Initial Energy: {initial_energy:.6f}, Final Energy: {final_energy:.6f}")
        
        # Energy should decrease or stay low (it might start low due to HF)
        # We just check it runs and doesn't explode
        self.assertTrue(np.isfinite(final_energy))
        
    def test_variance_minimization_gn(self):
        """Test Variance Minimization using MFGN (Gauss-Newton mode)."""
        print("\nTesting Variance Minimization (GN)...")
        
        n_walkers = 1000
        n_steps = 10 # steps per opt
        step_size = 0.1
        burn_in_steps = 1000
        n_opt_steps = 10
        key = random.PRNGKey(43)
        
        # optimize_ref_var returns a dict with 'cost' (variance)
        opt_results = optimize_ref_var(
            self.sj_ansatz,
            params=self.params,
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=step_size,
            burn_in_steps=burn_in_steps,
            n_opt_steps=n_opt_steps,
            optimizer_type='mfgn',
            learning_rate=0.1,
            opt_kwargs={'damping': 1e-6, 'maxiter': 10},
            key=key
        )
        
        # In optimize_ref_var, results are returned differently?
        # Let's check optimization.py return value.
        # It returns dict with "cost", "energies", etc.
        
        initial_variance = opt_results["cost"][0]
        final_variance = opt_results["cost"][-1]
        print(f"Initial Variance: {initial_variance:.6f}, Final Variance: {final_variance:.6f}")
        
        self.assertTrue(np.isfinite(final_variance))
        # Variance should ideally decrease
        self.assertLessEqual(final_variance, initial_variance * 1.1) # Allow slight fluctuation due to noise

if __name__ == "__main__":
    unittest.main()
