import os
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import time
from pyscf import gto, scf
from pytc.integrals.tc import TC, ISDFTC
from pytc.integrals.xtc import XTC, ISDFXTC
from pytc.tc_helper import get_eri

jax.config.update("jax_enable_x64", True)
from pytc.jastrow.rexp import REXP

# Set by GitHub Actions; used to skip tests that exceed the 16 GB hosted-runner RAM.
_ON_CI = os.environ.get("CI", "").lower() == "true"

class TestISDF(unittest.TestCase):
    def setUp(self):
        # System: H2O molecule
        self.mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='321g', verbose=0)
        self.mf = scf.RHF(self.mol).run()
        
        # Jastrow factor
        self.jastrow_jax = REXP()
        self.jastrow_params_jax = {'alpha': jnp.array([1.0])}
        
        # NumPy Jastrow
        from pytc.legacy.jastrow.rexp import REXP as REXP_numpy
        self.jastrow_numpy = REXP_numpy(params=np.array([1.0]), mol=self.mol)
        self.jastrow_params_numpy = np.array([1.0])
        
        # Initialize TC and XTC objects (Grid Level 2)
        self.tc_jax = TC.from_pyscf(self.mf, self.jastrow_jax, grid_lvl=2)
        self.xtc_jax = XTC.from_pyscf(self.mf, self.jastrow_jax, grid_lvl=2)
        
        from pytc.legacy.tc import TC as TC_numpy
        from pytc.legacy.xtc import XTC as XTC_numpy
        self.tc_numpy = TC_numpy(self.mf, self.jastrow_numpy, grid_lvl=2)
        self.xtc_numpy = XTC_numpy(self.mf, self.jastrow_numpy, grid_lvl=2)
        
        print(f"Grid size: {len(self.tc_jax.grid_points)}")

    def test_isdf_delta_U_accuracy(self):
        """Compare JAX ISDF delta_U directly with JAX Exact delta_U using convergence test."""
        ranks = [100, 200, 300]
        
        # --- JAX Exact Delta U with timing ---
        print("\nRunning JAX Exact Delta U...")
        start_exact = time.time()
        delta_U_exact_jax = self.xtc_jax.get_delta_U(self.jastrow_params_jax).block_until_ready()
        time_exact = time.time() - start_exact
        norm_exact_jax = np.linalg.norm(np.array(delta_U_exact_jax))
        print(f"Exact JAX time: {time_exact:.4f} s")
        
        # --- NumPy Exact Delta U ---
        print("Running NumPy Exact Delta U...")
        # Ensure we use the same parameters
        # NumPy get_delta_U uses self.jastrow_factor.params which is set in setUp
        # We need to make sure they are identical.
        # In setUp: self.jastrow_params_numpy = np.array([1.0])
        # self.jastrow_numpy = REXP_numpy(params=np.array([1.0]), mol=self.mol)
        delta_U_exact_numpy = self.xtc_numpy.get_delta_U()
        norm_exact_numpy = np.linalg.norm(delta_U_exact_numpy)
        
        # --- Compare Exact Versions ---
        diff_exact = np.linalg.norm(np.array(delta_U_exact_jax) - delta_U_exact_numpy)
        abs_err_exact = np.max(np.abs(np.array(delta_U_exact_jax) - delta_U_exact_numpy))
        rel_err_exact = diff_exact / norm_exact_numpy
        print(f"Exact Delta U Relative Error (JAX vs NumPy): {rel_err_exact:.2e}")
        print(f"Exact Delta U Absolute Error (JAX vs NumPy): {abs_err_exact:.2e}")
        self.assertTrue(rel_err_exact < 1e-10, f"Exact Delta U mismatch: {rel_err_exact}")
        
        print(f"\n{'Rank':<10} {'Rel Error':<15} {'Max Abs Error':<15} {'Time (s)':<12} {'Speedup':<10}")
        print("-" * 67)
        
        prev_error = float('inf')
        
        for n_rank in ranks:
            start_time = time.time()
            isdf_xtc_jax = ISDFXTC.from_xtc(self.xtc_jax, n_rank=n_rank)
            delta_U_isdf = isdf_xtc_jax.get_delta_U(self.jastrow_params_jax).block_until_ready()
            isdf_time = time.time() - start_time
            
            speedup = time_exact / isdf_time
            
            diff_dU = np.linalg.norm(np.array(delta_U_exact_jax) - np.array(delta_U_isdf))
            rel_err_dU = diff_dU / norm_exact_jax
            max_abs_dU = np.max(np.abs(np.array(delta_U_exact_jax) - np.array(delta_U_isdf)))
            
            print(f"{n_rank:<10} {rel_err_dU:<15.2e} {max_abs_dU:<15.2e} {isdf_time:<12.4f} {speedup:<10.2f}x")
            
            # Check for convergence or low error
            if n_rank > 100:
                 # Error should decrease or be already very small
                 if rel_err_dU > 1e-4:
                     self.assertLess(rel_err_dU, prev_error, f"Error increased at rank {n_rank}")
            
            prev_error = rel_err_dU
            
        # Final assertion for high rank
        self.assertLess(rel_err_dU, 1e-4, f"Final relative error {rel_err_dU} is too high")

    @unittest.skipIf(_ON_CI, "OOMs on 16 GB GitHub-hosted runner; legacy K3 path needs >16 GB on H2O/grid_lvl=2")
    def test_isdf_kmat_accuracy(self):
        """Compare JAX ISDF K matrices directly with JAX Exact K matrices using convergence test."""
        # --- JAX Exact 2-Body Correction with timing ---
        print("\nRunning JAX Exact 2-Body Correction...")
        start_exact = time.time()
        k2b_exact_jax = self.tc_jax.get_2b(self.jastrow_params_jax).block_until_ready()
        time_exact = time.time() - start_exact
        norm_exact_jax = np.linalg.norm(np.array(k2b_exact_jax))
        print(f"Exact JAX time: {time_exact:.4f} s")
        
        # --- NumPy Exact 2-Body Correction ---
        print("Running NumPy Exact 2-Body Correction...")
        k2b_numpy_full = self.tc_numpy.get_2b()
        # Compute ERI to isolate TC correction
        eri = get_eri(self.mf)
        k2b_exact_numpy = k2b_numpy_full - eri
        
        # --- Compare Exact Versions ---
        diff_exact = np.linalg.norm(np.array(k2b_exact_jax) - k2b_exact_numpy)
        rel_err_exact = diff_exact / np.linalg.norm(k2b_exact_numpy)
        print(f"Exact 2-Body Correction Relative Error (JAX vs NumPy): {rel_err_exact:.2e}")
        self.assertTrue(rel_err_exact < 1e-10, f"Exact 2-Body Correction mismatch: {rel_err_exact}")
        
        # --- JAX ISDF Convergence ---
        ranks = [100, 200, 300]
        print(f"\n{'Rank':<10} {'Rel Error':<15} {'Max Abs Error':<15} {'Time (s)':<12} {'Speedup':<10}")
        print("-" * 67)
        
        prev_error = float('inf')
        for n_rank in ranks:
            start_time = time.time()
            isdf_tc_jax = ISDFTC.from_tc(self.tc_jax, n_rank=n_rank)
            k2b_isdf = isdf_tc_jax.get_2b(self.jastrow_params_jax).block_until_ready()
            isdf_time = time.time() - start_time
            
            speedup = time_exact / isdf_time
            
            diff_2b = np.linalg.norm(np.array(k2b_exact_jax) - np.array(k2b_isdf))
            rel_err_2b = diff_2b / norm_exact_jax
            max_abs_2b = np.max(np.abs(np.array(k2b_exact_jax) - np.array(k2b_isdf)))
            
            print(f"{n_rank:<10} {rel_err_2b:<15.2e} {max_abs_2b:<15.2e} {isdf_time:<12.4f} {speedup:<10.2f}x")
            
            if n_rank > 100:
                if rel_err_2b > 1e-4:
                    self.assertLess(rel_err_2b, prev_error, f"Error increased at rank {n_rank}")
            
            prev_error = rel_err_2b
            
        # Final assertion for high rank
        self.assertLess(rel_err_2b, 1e-4, f"Final relative error {rel_err_2b} is too high")
    
    def test_get_2b_convergence(self):
        """Verify that get_2b (overall ISDFXTC) converges with rank."""
        # --- JAX Exact 2-Body Correction (Overall) with timing ---
        print("\nRunning JAX Exact Overall 2-Body Correction...")
        start_exact = time.time()
        k2b_exact_jax = self.xtc_jax.get_2b(self.jastrow_params_jax).block_until_ready()
        time_exact = time.time() - start_exact
        norm_exact_jax = np.linalg.norm(np.array(k2b_exact_jax))
        print(f"Exact JAX time: {time_exact:.4f} s")
        
        ranks = [100, 200, 300]
        
        print(f"\n{'Rank':<10} {'Rel Error':<15} {'Max Abs Error':<15} {'Time (s)':<12} {'Speedup':<10}")
        print("-" * 67)
        
        prev_error = float('inf')
        for n_rank in ranks:
            start_time = time.time()
            isdf_xtc_jax = ISDFXTC.from_xtc(self.xtc_jax, n_rank=n_rank)
            k2b_isdf = isdf_xtc_jax.get_2b(self.jastrow_params_jax).block_until_ready()
            isdf_time = time.time() - start_time
            
            speedup = time_exact / isdf_time
            
            diff_2b = np.linalg.norm(np.array(k2b_exact_jax) - np.array(k2b_isdf))
            rel_err_2b = diff_2b / norm_exact_jax
            max_abs_2b = np.max(np.abs(np.array(k2b_exact_jax) - np.array(k2b_isdf)))
            
            print(f"{n_rank:<10} {rel_err_2b:<15.2e} {max_abs_2b:<15.2e} {isdf_time:<12.4f} {speedup:<10.2f}x")
            
            if n_rank > 100:
                if rel_err_2b > 1e-4:
                    self.assertLess(rel_err_2b, prev_error, f"Error increased at rank {n_rank}")
            
            prev_error = rel_err_2b
            
        # Final assertion for high rank
        self.assertLess(rel_err_2b, 1e-4, f"Final relative error {rel_err_2b} is too high")

if __name__ == '__main__':
    unittest.main()
