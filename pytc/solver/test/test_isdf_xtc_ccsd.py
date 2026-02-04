
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf, lib, cc

from pytc.autodiff import xtc
from pytc.autodiff.jastrow import rexp
from pytc.solver import isdf_xtc_ccsd

# Enable float64 for JAX
jax.config.update("jax_enable_x64", True)

class TestISDFXTCCCSD(unittest.TestCase):
    def setUp(self):
        # H2O System
        self.mol = gto.M(
            atom='O 0 0 0; H 0 1 0; H 0 0 1',
            basis='sto-6g',
            verbose=4
        )
        self.mf = scf.RHF(self.mol).run()
        
        # Jastrow (Standard parameters)
        self.jastrow = rexp.REXP()
        self.jastrow_params = {'alpha': jnp.array([1.0])}
        
        # XTC Object (Low grid level for speed)
        self.xtc_obj = xtc.XTC.from_pyscf(self.mf, self.jastrow, grid_lvl=2)
        
        # Reference calculation (Exact XTC)
        self.n_rank = self.xtc_obj.n_orb * 10 # Sufficiently high rank
        self.isdf_xtc = xtc.ISDFXTC.from_xtc(self.xtc_obj, n_rank=self.n_rank, save_path="isdf_xtc_ccsd_test.h5")
        self.isdf_xtc = self.isdf_xtc.isdf(self.jastrow_params) # Precompute kernels

    def test_rccsd_energy(self):
        print("\nRunning Reference Exact XTC CCSD...")
        eris_exact = self.xtc_obj.make_eris(self.mf, self.jastrow_params)
        
        print("\nBuilding New ISDF-XTC-RCCSD ERIs...")
        cc_new = isdf_xtc_ccsd.RCCSD(self.mf, self.isdf_xtc, self.jastrow_params)
        eris_new = cc_new.ao2mo()
        
        # Compare blocks BEFORE zeroing vvvv
        with open('test_output.txt', 'w') as f:
            # Compare Fock
            fock_diff = np.linalg.norm(eris_new.fock - eris_exact.fock)
            f.write(f"Fock Matrix Difference Norm: {fock_diff}\n")
            print(f"Fock Matrix Difference Norm: {fock_diff}")
        
            # Compare OOOO
            try:
                if eris_new.oooo.shape == eris_exact.oooo.shape:
                    oooo_diff = np.linalg.norm(eris_new.oooo - eris_exact.oooo)
                    f.write(f"OOOO Difference Norm: {oooo_diff}\n")
                else:
                    f.write(f"OOOO Shape mismatch\n")
            except Exception as e:
                f.write(f"OOOO check failed: {e}\n")
            
            # Compare OOVV (uses oo and vv pairs)
            try:
                if eris_new.oovv.shape == eris_exact.oovv.shape:
                    oovv_diff = np.linalg.norm(eris_new.oovv - eris_exact.oovv)
                    f.write(f"OOVV Difference Norm: {oovv_diff}\n")
                else:
                    f.write(f"OOVV Shape mismatch\n")
            except Exception as e:
                f.write(f"OOVV check failed: {e}\n")

            # Compare OVOV (uses ov pairs)
            try:
                if eris_new.ovov.shape == eris_exact.ovov.shape:
                    ovov_diff = np.linalg.norm(eris_new.ovov - eris_exact.ovov)
                    f.write(f"OVOV Difference Norm: {ovov_diff}\n")
                else:
                    f.write(f"OVOV Shape mismatch\n")
            except Exception as e:
                f.write(f"OVOV check failed: {e}\n")
            
            # Compare OVOO (uses ov and oo pairs)
            try:
                if eris_new.ovoo.shape == eris_exact.ovoo.shape:
                    ovoo_diff = np.linalg.norm(eris_new.ovoo - eris_exact.ovoo)
                    f.write(f"OVOO Difference Norm: {ovoo_diff}\n")
                else:
                    f.write(f"OVOO Shape mismatch\n")
            except Exception as e:
                f.write(f"OVOO check failed: {e}\n")

            # Compare OVVV with explicit unpacking
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
                    f.write(f"OVVV Difference Norm: {ovvv_diff}\n")
                    f.write(f"OVVV Ref max: {np.max(np.abs(ovvv_ref))}\n")
                    f.write(f"OVVV New max: {np.max(np.abs(ovvv_new))}\n")
                else:
                    f.write(f"OVVV shapes still differ: {ovvv_new.shape} vs {ovvv_ref.shape}\n")
            
            # Compare VVVV (before zeroing)
            if eris_exact.vvvv is not None and eris_new.vvvv is not None:
                # Save copies before zeroing for comparison
                vvvv_ref_saved = eris_exact.vvvv.copy() if hasattr(eris_exact, 'vvvv') else None
                vvvv_new_saved = eris_new.vvvv.copy() if hasattr(eris_new, 'vvvv') else None
                if vvvv_ref_saved is not None and vvvv_new_saved is not None:
                    vvvv_diff = np.linalg.norm(vvvv_ref_saved - vvvv_new_saved)
                    f.write(f"VVVV Difference Norm: {vvvv_diff}\n")
                    f.write(f"VVVV Ref max: {np.max(np.abs(vvvv_ref_saved))}\n")
                    f.write(f"VVVV New max: {np.max(np.abs(vvvv_new_saved))}\n")

        # Now run both CCSD calculations (with full vvvv for now - remove debug zeroing)
        # Remove vvvv zeroing from isdf_xtc_ccsd.py before this test
        print("\nRunning Reference CCSD...")
        cc_ref = cc.rccsd.RCCSD(self.mf)
        e_ref, t1_ref, t2_ref = cc_ref.kernel(eris=eris_exact)
        print(f"Reference Correlation Energy: {e_ref}")
        
        # Update output file
        with open('test_output.txt', 'a') as f:
            f.write(f"Reference Correlation Energy: {e_ref}\n")
        
        print("\nRunning New ISDF-XTC-CCSD...")
        e_new, t1_new, t2_new = cc_new.kernel(eris=eris_new)
        
        print(f"New Correlation Energy: {e_new}")
        
        error = abs(e_new - e_ref)
        print(f"Energy Difference: {error}")
        
        self.assertLess(error, 1e-3, "Correlation energy mismatch > 1e-3 Ha")
        
        t1_diff = np.linalg.norm(t1_new - t1_ref)
        self.assertLess(t1_diff, 1e-2, "T1 amplitude mismatch")

    def tearDown(self):
        import os
        if os.path.exists("isdf_xtc_ccsd_test.h5"):
            os.remove("isdf_xtc_ccsd_test.h5")
        if os.path.exists("test_output.txt"):
            pass

if __name__ == "__main__":
    unittest.main()
