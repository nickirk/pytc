
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf, lib, cc

from pytc import xtc
from pytc.jastrow import rexp
from pytc.solver import xtc_ccsd


jax.config.update("jax_enable_x64", True)

class TestXTCCCSD(unittest.TestCase):
    def setUp(self):

        self.mol = gto.M(
            atom='C 0 0 0; O 0 0 1.128',
            basis='sto-6g',
            verbose=4
        )
        self.mf = scf.RHF(self.mol).run()
        
        self.jastrow = rexp.REXP()
        self.jastrow_params = {'alpha': jnp.array([0.5])}
        
        self.xtc_obj = xtc.XTC.from_pyscf(self.mf, self.jastrow, grid_lvl=1)
        
        self.n_rank = self.xtc_obj.n_orb * 12 # Sufficiently high rank
        self.isdf_xtc = xtc.ISDFXTC.from_xtc(self.xtc_obj, n_rank=self.n_rank, save_path="isdf_xtc_ccsd_test.h5")

    def test_rccsd_energy(self):
        print("\nRunning Reference Exact XTC CCSD...")
        eris_exact = self.xtc_obj.make_eris(self.mf, self.jastrow_params)
        cc_exact = xtc_ccsd.RCCSD(self.mf, self.xtc_obj, self.jastrow_params)
        
        print("\nBuilding New ISDF-XTC-RCCSD ERIs...")
        cc_new = xtc_ccsd.RCCSD(self.mf, self.isdf_xtc, self.jastrow_params)
        eris_new = cc_new.ao2mo()

        
        # Compare blocks BEFORE zeroing vvvv
        fock_diff = np.linalg.norm(eris_new.fock - eris_exact.fock)
        print(f"Fock Matrix Difference Norm: {fock_diff}")
    
        try:
            if eris_new.oooo.shape == eris_exact.oooo.shape:
                oooo_diff = np.linalg.norm(eris_new.oooo - eris_exact.oooo)
                print(f"OOOO Difference Norm: {oooo_diff}")
            else:
                print(f"OOOO Shape mismatch")
        except Exception as e:
            print(f"OOOO check failed: {e}")
        
        try:
            if eris_new.oovv.shape == eris_exact.oovv.shape:
                oovv_diff = np.linalg.norm(eris_new.oovv - eris_exact.oovv)
                print(f"OOVV Difference Norm: {oovv_diff}")
            else:
                print(f"OOVV Shape mismatch")
        except Exception as e:
            print(f"OOVV check failed: {e}")

        try:
            if eris_new.ovov.shape == eris_exact.ovov.shape:
                ovov_diff = np.linalg.norm(eris_new.ovov - eris_exact.ovov)
                print(f"OVOV Difference Norm: {ovov_diff}")
            else:
                print(f"OVOV Shape mismatch")
        except Exception as e:
            print(f"OVOV check failed: {e}")
        
        try:
            if eris_new.ovoo.shape == eris_exact.ovoo.shape:
                ovoo_diff = np.linalg.norm(eris_new.ovoo - eris_exact.ovoo)
                print(f"OVOO Difference Norm: {ovoo_diff}")
            else:
                print(f"OVOO Shape mismatch")
        except Exception as e:
            print(f"OVOO check failed: {e}")

        if eris_exact.ovvv is not None:
            def unpack_if_needed(ov):
                if ov.ndim == 3: # packed (nocc, nvir, pair)
                    nocc, nvir, pair = ov.shape
                    unpacked = np.zeros((nocc, nvir, nvir, nvir))
                    idx = np.tril_indices(nvir)
                    unpacked[:, :, idx[0], idx[1]] = ov
                    unpacked[:, :, idx[1], idx[0]] = ov
                    return unpacked
                return ov # already unpacked (nocc, nvir, nvir, nvir)
            
            ovvv_ref = unpack_if_needed(eris_exact.ovvv)
            ovvv_new = unpack_if_needed(eris_new.ovvv)
            
            if ovvv_ref.shape == ovvv_new.shape:
                ovvv_diff = np.linalg.norm(ovvv_ref - ovvv_new)
                print(f"OVVV Difference Norm: {ovvv_diff}")
                print(f"OVVV Ref max: {np.max(np.abs(ovvv_ref))}")
                print(f"OVVV New max: {np.max(np.abs(ovvv_new))}")
            else:
                print(f"OVVV shapes still differ: {ovvv_new.shape} vs {ovvv_ref.shape}")
        
        if eris_exact.vvvv is not None and eris_new.vvvv is not None:
            # Save copies before zeroing for comparison
            # Use np.array() instead of .copy() to handle both numpy arrays and HDF5 Datasets
            vvvv_ref_saved = np.array(eris_exact.vvvv) if hasattr(eris_exact, 'vvvv') else None
            vvvv_new_saved = np.array(eris_new.vvvv) if hasattr(eris_new, 'vvvv') else None
            if vvvv_ref_saved is not None and vvvv_new_saved is not None:
                vvvv_diff = np.linalg.norm(vvvv_ref_saved - vvvv_new_saved)
                print(f"VVVV Difference Norm: {vvvv_diff}")
                print(f"VVVV Ref max: {np.max(np.abs(vvvv_ref_saved))}")
                print(f"VVVV New max: {np.max(np.abs(vvvv_new_saved))}")

        print("\nRunning Reference CCSD (Exact XTC)...")
        e_ref, t1_ref, t2_ref = cc_exact.kernel(eris=eris_exact)
        print(f"Reference Correlation Energy: {e_ref}")
        
        print("\nRunning New ISDF-XTC-CCSD...")
        e_new, t1_new, t2_new = cc_new.kernel(eris=eris_new)

        
        print(f"New Correlation Energy: {e_new}")
        
        error = abs(e_new - e_ref)
        print(f"Energy Difference: {error}")
        
        self.assertLess(error, 1e-3, "Correlation energy mismatch > 1e-3 Ha")
        
        t1_diff = np.linalg.norm(t1_new - t1_ref)
        self.assertLess(t1_diff, 1e-2, "T1 amplitude mismatch")

    def test_hermitian_limit(self):
        print("\nTesting Hermitian Limit (Large Alpha)...")
        # Use large alpha to make Jastrow negligible
        large_alpha_params = {'alpha': jnp.array([1000.0])}
        
        # Exact XTC with large alpha should match PySCF standard
        cc_xtc = xtc_ccsd.RCCSD(self.mf, self.xtc_obj, large_alpha_params)
        eris_xtc = self.xtc_obj.make_eris(self.mf, large_alpha_params)
        e_xtc, _, _ = cc_xtc.kernel(eris=eris_xtc)
        
        cc_std = cc.rccsd.RCCSD(self.mf)
        e_std, _, _ = cc_std.kernel()
        
        print(f"XTC (Large Alpha) Energy: {e_xtc}")
        print(f"PySCF Standard Energy: {e_std}")
        
        print("\nComparing mo_energy:")
        print(f"XTC mo_energy: {eris_xtc.mo_energy}")
        print(f"STD mo_energy: {self.mf.mo_energy}")

        
        error = abs(e_xtc - e_std)

        print(f"Energy Difference (Hermitian Limit): {error}")
        
        self.assertLess(error, 1e-6, "Hermitian limit agreement failed")


    def tearDown(self):
        import os
        if os.path.exists("isdf_xtc_ccsd_test.h5"):
            os.remove("isdf_xtc_ccsd_test.h5")

if __name__ == "__main__":
    unittest.main()
