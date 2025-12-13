import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.autodiff.tc import TC, ISDFTC
from pytc.autodiff.xtc import XTC, ISDFXTC

jax.config.update("jax_enable_x64", True)
from pytc.autodiff.jastrow.rexp import REXP

class TestISDF(unittest.TestCase):
    def setUp(self):
        # System: H2O molecule
        self.mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='ccpvdz', verbose=0)
        self.mf = scf.RHF(self.mol).run()
        
        # Jastrow factor
        self.jastrow_jax = REXP()
        self.jastrow_params_jax = {'alpha': jnp.array([1.0])}
        
        # NumPy Jastrow
        from pytc.jastrow.rexp import REXP as REXP_numpy
        self.jastrow_numpy = REXP_numpy(params=np.array([1.0]), mol=self.mol)
        self.jastrow_params_numpy = np.array([1.0])
        
        # Initialize TC and XTC objects (Grid Level 2)
        self.tc_jax = TC.from_pyscf(self.mf, self.jastrow_jax, grid_lvl=2)
        self.xtc_jax = XTC.from_pyscf(self.mf, self.jastrow_jax, grid_lvl=2)
        
        from pytc.tc import TC as TC_numpy
        from pytc.xtc import XTC as XTC_numpy
        self.tc_numpy = TC_numpy(self.mf, self.jastrow_numpy, grid_lvl=2)
        self.xtc_numpy = XTC_numpy(self.mf, self.jastrow_numpy, grid_lvl=2)
        
        print(f"Grid size: {len(self.tc_jax.grid_points)}")

    def test_isdf_vs_numpy_integrals(self):
        """Compare JAX ISDF integrals directly with NumPy ISDF integrals."""
        n_rank = 400 # Sufficient rank based on convergence benchmark
        
        # --- NumPy ISDF ---
        print("\nRunning NumPy ISDF...")
        self.tc_numpy.isdf(n_rank=n_rank)
        # Note: XTC numpy uses TC's isdf results if available, or we need to run it on xtc?
        # Checking xtc.py: XTC inherits from TC. It uses self._isdf_results.
        # So running isdf on xtc_numpy is enough.
        self.xtc_numpy.isdf(n_rank=n_rank)
        
        # Compute Integrals (NumPy)
        # get_2b returns (K_nabla + K_lap + K_sq) + ERI
        # NumPy get_2b does NOT take params argument, it uses self.jastrow_factor.params
        k2b_numpy_full = self.tc_numpy.get_2b()
        delta_U_numpy = self.xtc_numpy.get_delta_U()
        
        # Compute ERI to isolate TC correction
        from pyscf import ao2mo
        eri = ao2mo.incore.full(self.mf._eri, self.mf.mo_coeff, compact=False)
        eri = ao2mo.restore(1, eri, self.mf.mo_coeff.shape[1])
        k2b_numpy_correction = k2b_numpy_full - eri
        
        # --- JAX ISDF ---
        print("Running JAX ISDF...")
        isdf_tc_jax = ISDFTC.from_tc(self.tc_jax, n_rank=n_rank)
        isdf_xtc_jax = ISDFXTC.from_xtc(self.xtc_jax, n_rank=n_rank)
        
        # Compute Integrals (JAX)
        # JAX get_2b returns ONLY the correction term (-K)
        k2b_jax_correction = isdf_tc_jax.get_2b(self.jastrow_params_jax)
        delta_U_jax = isdf_xtc_jax.get_delta_U(self.jastrow_params_jax)
        
        # --- Comparison ---
        # 1. Delta U
        diff_dU = np.linalg.norm(delta_U_numpy - np.array(delta_U_jax))
        norm_dU = np.linalg.norm(delta_U_numpy)
        rel_err_dU = diff_dU / norm_dU
        print(f"Delta U Relative Error (vs NumPy): {rel_err_dU:.2e}")
        
        # 2. 2-Body Integrals (Correction only)
        diff_2b = np.linalg.norm(k2b_numpy_correction - np.array(k2b_jax_correction))
        norm_2b = np.linalg.norm(k2b_numpy_correction)
        rel_err_2b = diff_2b / norm_2b
        print(f"2-Body Correction Relative Error (vs NumPy): {rel_err_2b:.2e}")
        
        # Component-wise comparison
        print("\nComponent-wise Comparison:")
        # NumPy components
        from pytc.kmat import calc_K1_isdf, calc_K2_isdf, calc_K3_isdf
        res_np = self.tc_numpy._isdf_results
        k1_np = calc_K1_isdf(res_np['C_rho'], res_np['xi_rho'], res_np['C_grad'], res_np['xi_grad'],
                             self.jastrow_numpy, self.tc_numpy.grid_points, self.tc_numpy.weights)
        k2_np = calc_K2_isdf(res_np['C_rho'], res_np['xi_rho'], res_np['C_grad'], res_np['xi_grad'],
                             self.jastrow_numpy, self.tc_numpy.grid_points, self.tc_numpy.weights)
        k3_np = calc_K3_isdf(res_np['C_rho'], res_np['xi_rho'],
                             self.jastrow_numpy, self.tc_numpy.grid_points, self.tc_numpy.weights)
        
        # JAX components
        from pytc.autodiff import kmat as kmat_jax
        k1_jax = kmat_jax.calc_K1_isdf(isdf_tc_jax.C_rho, isdf_tc_jax.xi_rho, isdf_tc_jax.C_grad, isdf_tc_jax.xi_grad,
                                       self.jastrow_jax, self.jastrow_params_jax, self.tc_jax.grid_points, self.tc_jax.weights)
        # For K2, JAX ISDFTC uses -(K1 + K1.T) now, but let's check what calc_K2_isdf returns vs that
        k2_jax_func = kmat_jax.calc_K2_isdf(isdf_tc_jax.C_rho, isdf_tc_jax.xi_rho, isdf_tc_jax.C_grad, isdf_tc_jax.xi_grad,
                                            self.jastrow_jax, self.jastrow_params_jax, self.tc_jax.grid_points, self.tc_jax.weights)
        
        k1_jax = np.array(k1_jax)
        k2_jax_func = np.array(k2_jax_func)
        
        # Compare K1
        err_k1 = np.linalg.norm(k1_np - k1_jax) / np.linalg.norm(k1_np)
        print(f"K1 Relative Error: {err_k1:.2e}")

        # --- Cross-Check: JAX K1 with NumPy Intermediates ---
        print("\nK1 Cross-Check (NumPy Intermediates -> JAX K1):")
        C_rho_np = jnp.array(res_np['C_rho'])
        xi_rho_np = jnp.array(res_np['xi_rho'])
        C_grad_np = jnp.array(res_np['C_grad'])
        xi_grad_np = jnp.array(res_np['xi_grad'])
        
        k1_jax_cross = kmat_jax.calc_K1_isdf(C_rho_np, xi_rho_np, C_grad_np, xi_grad_np,
                                             self.jastrow_jax, self.jastrow_params_jax, 
                                             self.tc_jax.grid_points, self.tc_jax.weights)
        k1_jax_cross = np.array(k1_jax_cross)
        err_k1_cross = np.linalg.norm(k1_np - k1_jax_cross) / np.linalg.norm(k1_np)
        print(f"K1 Cross-Check Error: {err_k1_cross:.2e}")
        
        # Also check K2 cross-check
        k2_jax_cross = kmat_jax.calc_K2_isdf(C_rho_np, xi_rho_np, C_grad_np, xi_grad_np,
                                             self.jastrow_jax, self.jastrow_params_jax,
                                             self.tc_jax.grid_points, self.tc_jax.weights)
        k2_jax_cross = np.array(k2_jax_cross)
        err_k2_cross = np.linalg.norm(k2_np - k2_jax_cross) / np.linalg.norm(k2_np)
        print(f"K2 Cross-Check Error: {err_k2_cross:.2e}")
        
        # Compare K2 (NumPy uses calc_K2_isdf, JAX uses calc_K2_isdf for this check)
        err_k2 = np.linalg.norm(k2_np - k2_jax_func) / np.linalg.norm(k2_np)
        print(f"K2 (func) Relative Error: {err_k2:.2e}")
        
        # Compare K3
        k3_jax = kmat_jax.calc_K3_isdf(isdf_tc_jax.C_rho, isdf_tc_jax.xi_rho,
                                       self.jastrow_jax, self.jastrow_params_jax, self.tc_jax.grid_points, self.tc_jax.weights)
        k3_jax = np.array(k3_jax)
        err_k3 = np.linalg.norm(k3_np - k3_jax) / np.linalg.norm(k3_np)
        print(f"K3 Relative Error: {err_k3:.2e}")
        
        self.assertTrue(rel_err_dU < 1e-4, f"Delta U mismatch: {rel_err_dU}")
        self.assertTrue(rel_err_2b < 1e-4, f"2-Body correction mismatch: {rel_err_2b}")
        self.assertTrue(err_k1 < 1e-4, f"K1 mismatch: {err_k1}")
        self.assertTrue(err_k2 < 1e-4, f"K2 mismatch: {err_k2}")
        self.assertTrue(err_k3 < 1e-4, f"K3 mismatch: {err_k3}")

    def test_isdf_convergence(self):
        """Verify convergence of ISDF intermediates (Rho/Grad) with rank."""
        from pytc.autodiff.df import isdf_decompose
        
        ranks = [50, 100, 200]
        errors = []
        
        rho = self.tc_jax.rho
        nabla_rho = self.tc_jax.nabla_rho
        
        # Construct ground truth for full grid
        rho_paired = jnp.einsum('in,jn->ijn', rho, rho).reshape(rho.shape[0]**2, -1)
        # Match C_grad definition: nabla on first orbital
        nabla_rho_paired = jnp.einsum('inc,jn->ijnc', nabla_rho, rho).reshape(rho.shape[0]**2, -1, 3)
        
        print("\nConvergence Check:")
        for rank in ranks:
            C_rho, xi_rho, C_grad, xi_grad, _ = isdf_decompose(rho, nabla_rho, rank, rank)
            
            # Calculate errors
            rho_recon = C_rho @ xi_rho
            rho_err = jnp.linalg.norm(rho_recon - rho_paired) / jnp.linalg.norm(rho_paired)
            
            # Grad error
            grad_recon = jnp.einsum('pmc,mnc->pnc', C_grad, xi_grad) # (Nb^2, N_sub, 3)
            grad_err = jnp.linalg.norm(grad_recon - nabla_rho_paired) / jnp.linalg.norm(nabla_rho_paired)
            
            errors.append(rho_err) # Store rho error for convergence check
            print(f"Rank {rank}: Rho Error = {rho_err:.2e}, Grad Error = {grad_err:.2e}")
            
        # Check if error decreases
        self.assertTrue(errors[-1] < errors[0], "Error should decrease with rank")
        self.assertTrue(errors[-1] < 1e-10, f"Error at rank {ranks[-1]} should be small")

    def test_isdf_tc_accuracy(self):
        """Verify accuracy of JAX ISDF TC against JAX Exact TC."""
        n_rank = 400
        
        # Create ISDFTC
        isdf_tc = ISDFTC.from_tc(self.tc_jax, n_rank=n_rank)
        
        # Compute 2-body term with exact TC (JAX)
        # Note: TC.get_2b in JAX returns correction only (-K)
        exact_tc_2b = self.tc_jax.get_2b(self.jastrow_params_jax)
        
        # Compute 2-body term with ISDFTC
        isdf_tc_2b = isdf_tc.get_2b(self.jastrow_params_jax)
        
        # Compare
        diff = jnp.linalg.norm(exact_tc_2b - isdf_tc_2b)
        norm = jnp.linalg.norm(exact_tc_2b)
        rel_err = diff / norm
        
        print(f"\nJAX ISDF vs Exact Accuracy: {rel_err:.2e}")
        self.assertTrue(rel_err < 1e-4, f"ISDF TC error too high: {rel_err}")

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
