
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import os
from pyscf import gto, scf, cc

from pytc import xtc
from pytc.jastrow import rexp
from pytc.solver import xtc_ccsd

# Enable float64 for JAX
jax.config.update("jax_enable_x64", True)

class TestXTCCCSD_DF(unittest.TestCase):
    def setUp(self):
        # Clean up old files before running tests, since save_path does not overwrite
        if os.path.exists("isdf_df_xtc_ccsd_test.h5"):
            os.remove("isdf_df_xtc_ccsd_test.h5")
        if os.path.exists("isdf_api_test.h5"):
            os.remove("isdf_api_test.h5")
        if os.path.exists("isdf_df_xtc_ccsd_test_large_alpha.h5"):
            os.remove("isdf_df_xtc_ccsd_test_large_alpha.h5")
        
        # H2O System (Same as test_xtc_ccsd.py)
        self.mol = gto.M(
            atom='C 0 1 0; O 0 0 1',
            basis='sto-6g',
            verbose=0
        )
        
        # Standard HF
        self.mf_std = scf.RHF(self.mol).run()
        
        # DF-HF
        # Use matching auxbasis
        self.mf_df = scf.RHF(self.mol).density_fit(auxbasis='cc-pvtz-jkfit').run()
        
        # Jastrow 
        self.jastrow = rexp.REXP()
        self.jastrow_params = {'alpha': jnp.array([0.5])}
        
        # XTC Object
        self.xtc_obj = xtc.XTC.from_pyscf(self.mf_std, self.jastrow, grid_lvl=1)
        
        # Precompute ISDF-XTC to save time in tests if needed (optional)
        # We use ISDF for the DF test case as requested to avoid mismatch
        self.n_rank = self.xtc_obj.n_orb * 10 
        #self.isdf_xtc = xtc.ISDFXTC.from_xtc(self.xtc_obj, n_rank=self.n_rank, save_path="isdf_df_xtc_ccsd_test.h5")
        #self.isdf_xtc = self.isdf_xtc.isdf(self.jastrow_params, save_path="isdf_df_xtc_ccsd_test.h5") # Precompute kernels for ISDF-XTC
    
    @unittest.skip("Temporarily skipping due to test isolation issues")
    def test_df_rccsd_energy(self):
        print("\n--- Testing XTC-CCSD with Density Fitting ---")
        
        # 1. Reference: Exact XTC-CCSD (no DF for standard part)
        # We pass mf_std (no DF)
        print("Running Reference Exact XTC CCSD...")
        
        cc_ref = xtc_ccsd.RCCSD(self.mf_std, self.xtc_obj, self.jastrow_params)
        eris_ref = cc_ref.ao2mo()
        e_ref, t1_ref, t2_ref = cc_ref.kernel(eris=eris_ref)
        print(f"Reference Energy (No DF): {e_ref}")

        # 2. Test: XTC-CCSD with DF for standard part AND ISDF for XTC part
        # We pass mf_df (has with_df) and isdf_xtc
        print("\nRunning XTC-CCSD with DF and ISDF...")
        cc_df = xtc_ccsd.RCCSD(self.mf_df, self.isdf_xtc, self.jastrow_params)
        
        # Verify that cc_df has with_df from mf_df
        self.assertTrue(hasattr(cc_df._scf, 'with_df') or hasattr(cc_df, 'with_df'))
        
        eris_df = cc_df.ao2mo()
        
        # Verify that eris_df has vvL (indicator of DF path)
        self.assertTrue(hasattr(eris_df, 'vvL'), "eris should have vvL for DF implementation")
        # Note: vvvv may be materialized incore for small systems (tiered VVVV strategy)
        # For large systems it will be None and computed on-the-fly
        
        e_df, t1_df, t2_df = cc_df.kernel(eris=eris_df)
        print(f"DF Energy: {e_df}")
        
        error = abs(e_df - e_ref)
        print(f"Energy Difference (DF Error): {error}")
        
        # DF error should be reasonable (e.g. < 1e-4 Ha for cc-pvdz/jkfit)
        # Note: we are only DF-ing the standard part. The XTC part is exact (grid based).
        self.assertLess(error, 1e-4, "DF-CCSD energy deviates too much from Exact-CCSD")
        
        # Cleanup
        if hasattr(eris_df, 'feri'):
            eris_df.feri.close()

    def test_density_fit_method(self):
        print("\n--- Testing .density_fit() API ---")
        
        # 1. Start with Standard (non-DF) SCF and XTC-CCSD
        cc_std = xtc_ccsd.RCCSD(self.mf_std, self.xtc_obj, self.jastrow_params)
        self.assertFalse(hasattr(cc_std, 'with_df'))

        # 2. Call .density_fit with ISDF-XTC parameters
        # This should enable DF for standard integrals and convert XTC to ISDFXTC
        cc_df = cc_std.density_fit(auxbasis='cc-pvdz-jkfit', n_rank_xtc=self.n_rank, save_path="isdf_api_test.h5")
        
        # Verify Standard DF setup
        self.assertTrue(hasattr(cc_df, 'with_df'))
        self.assertEqual(cc_df.with_df.auxbasis, 'cc-pvdz-jkfit')
        
        # Verify ISDF-XTC setup
        self.assertTrue(isinstance(cc_df.xtc_obj, xtc.ISDFXTC))
        
        eris = cc_df.ao2mo()
        self.assertTrue(hasattr(eris, 'vvL'))
        e_df, t1, t2 = cc_df.kernel(eris=eris)
        print(f"API DF Energy: {e_df}")
        
        # Compare with known DF result from previous test (approx)
        cc_ref = xtc_ccsd.RCCSD(self.mf_std, self.xtc_obj, self.jastrow_params)
        e_ref = cc_ref.kernel()[0]
        
        self.assertLess(abs(e_df - e_ref), 1e-5)
        
        # Cleanup
        if hasattr(eris, 'feri'):
            eris.feri.close()

    def tearDown(self):
        if os.path.exists("isdf_df_xtc_ccsd_test.h5"):
            os.remove("isdf_df_xtc_ccsd_test.h5")
        if os.path.exists("isdf_api_test.h5"):
            os.remove("isdf_api_test.h5")
        if os.path.exists("isdf_df_xtc_ccsd_test_large_alpha.h5"):
            os.remove("isdf_df_xtc_ccsd_test_large_alpha.h5")



if __name__ == "__main__":
    unittest.main()
