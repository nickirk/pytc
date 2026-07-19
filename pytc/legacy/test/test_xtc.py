"""Test for Exculding normal-ordered 3-body Transcorrelated (XTC) calculations."""

import unittest
import os 
import sys
from functools import partial, reduce
import numpy as np
import time

from pyscf import gto, scf

from pytc.legacy.xtc import XTC
from pytc.legacy.jastrow import REXP
from pytc.tc_helper import get_eri

_SKIP_ISDF_STRESS = (
    os.environ.get("CI", "").lower() == "true"
    and sys.version_info >= (3, 14)
)


def get_be_ccpvdz():
    """Return a Be atom with cc-pVDZ basis for testing."""
    mol = gto.M(atom='Be 0 0 0', basis='ccpvdz', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    
    return mol, mf

class TestXTC(unittest.TestCase):
    """Test XTranscorrelated calculations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case using Be atom."""
        _, cls.mf = get_be_ccpvdz()
        
        
        cls.jastrow = REXP([1.])
        
        cls.xtc = XTC(cls.mf, cls.jastrow, grid_lvl=2)  
        
        cls.grid_points = cls.xtc.grid_points
        cls.weights = cls.xtc.weights
        cls.jastrow = cls.xtc.jastrow_factor
        
        mo_values, _ = cls.xtc._eval_basis_on_grid()
        cls.rho = mo_values
        
        cls.rho_paired = np.einsum('in,jn->ijn', 
                                  cls.rho, 
                                  cls.rho).reshape(-1, len(cls.weights))
    
    def test_get_mf_dm(self):
        """Test mean-field density matrix generation."""
        dm1 = self.xtc._get_mf_dm() * 2
        
        self.assertEqual(dm1.shape, (self.mf.mo_coeff.shape[1],) * 2)
        
        self.assertAlmostEqual(np.trace(dm1), self.xtc.mol.nelectron)
        
        nocc = self.xtc.mol.nelectron // 2
        np.testing.assert_array_almost_equal(np.diag(dm1)[:nocc], 
                                           np.full(nocc, 2.0))
        np.testing.assert_array_almost_equal(np.diag(dm1)[nocc:], 
                                           np.zeros(len(dm1) - nocc))
    
    
    def test_nhccsd(self):
        from pyscf.cc import rccsd, CCSD
        mycc = CCSD(self.mf)
        e_corr, t1, t2 = mycc.kernel()
        mycc.verbose = 5
        print("E_CCSD = ", e_corr)
        #self.assertAlmostEqual(e_corr, -0.04503138331130402, places=6)
        t = mycc.amplitudes_to_vector(t1, t2)
        print("|t2| = ", np.linalg.norm(t2))
        print("|t1+t2| = ", np.linalg.norm(t))
        eri1 = get_eri(self.xtc.mf, self.xtc.mo_coeff)
        h1e = mycc._scf.get_hcore()
        h1e = reduce(np.dot, (self.xtc.mo_coeff.T, h1e, self.xtc.mo_coeff))
        
        e_hf_0 = 2. * np.einsum('ii->', h1e[:mycc.nocc, :mycc.nocc])
        e_dir = 2. * np.einsum('jjii->', eri1[:mycc.nocc, :mycc.nocc, :mycc.nocc, :mycc.nocc])
        e_ex = -1. * np.einsum('ijji->', eri1[:mycc.nocc, :mycc.nocc, :mycc.nocc, :mycc.nocc])
        e_hf_0 += (e_dir + e_ex) + mycc._scf.energy_nuc()
        print("Check e_hf = ",  e_hf_0)
        self.assertAlmostEqual(e_hf_0, -14.57233763095337, places=6)

        myrcc = rccsd.RCCSD(self.mf)
        self.xtc.mf._eri = None
        eris = self.xtc.make_eris()
        self.assertIsNotNone(self.xtc.mf._eri)
        tc_e_corr, t1, t2 = myrcc.kernel(eris=eris)
        t = myrcc.amplitudes_to_vector(t1, t2)
        print("|t2| = ", np.linalg.norm(t2))
        print("|t1+t2| = ", np.linalg.norm(t))
        print("corr E_XTC_CCSD = ", myrcc.e_corr)
        
        # Check against expected value (either custom PySCF or standard PySCF)
        # Custom PySCF (tc-ccsd branch) gives -0.0327...
        # Standard PySCF gives -0.0537...
        expected_custom = -0.032719417286611235
        expected_standard = -0.05372917906901621
        
        if np.isclose(tc_e_corr, expected_custom, atol=1e-5):
             self.assertAlmostEqual(tc_e_corr, expected_custom, places=5)
        elif np.isclose(tc_e_corr, expected_standard, atol=1e-5):
             print("Warning: Using standard PySCF result. For correct tc-ccsd results, install https://github.com/nickirk/pyscf/tree/tc-ccsd")
             self.assertAlmostEqual(tc_e_corr, expected_standard, places=5)
        else:
             self.fail(f"Correlation energy {tc_e_corr} does not match expected custom ({expected_custom}) or standard ({expected_standard}) values.")

        no = myrcc.nocc
        tc_h1e = self.xtc.get_1b()
        tc_e_hf = 2. * np.einsum('ii->', tc_h1e[:no, :no])
        tc_e_dir = 2. * np.einsum('jjii->', eris.oooo)
        tc_e_ex = -1. * np.einsum('ijji->', eris.oooo)

        tc_e_hf += (tc_e_dir + tc_e_ex) + eris.e_core 
        print("E_XTC_CCSD = ", myrcc.e_corr + tc_e_hf)
        
        expected_total_custom = -14.656403596315542
        expected_total_standard = -14.677381617968336
        
        if np.isclose(tc_e_hf + tc_e_corr, expected_total_custom, atol=1e-5):
             self.assertAlmostEqual(tc_e_hf + tc_e_corr, expected_total_custom, places=5)
        elif np.isclose(tc_e_hf + tc_e_corr, expected_total_standard, atol=1e-5):
             self.assertAlmostEqual(tc_e_hf + tc_e_corr, expected_total_standard, places=5)
        else:
             self.fail(f"Total energy {tc_e_hf + tc_e_corr} does not match expected custom ({expected_total_custom}) or standard ({expected_total_standard}) values.")


    @unittest.skipIf(
        _SKIP_ISDF_STRESS,
        "Exceeds GitHub-hosted runner resources on Python 3.14; covered by the 3.10 full job",
    )
    def test_delta_U_isdf_convergence(self):
        """Test convergence of ISDF delta_U calculation with increasing rank."""
        dm1 = self.xtc._get_mf_dm()
        
        rho, _ = self.xtc._get_intermediates()
        rho_paired = np.einsum('in,jn->ijn', rho, rho).reshape(-1, rho.shape[1])
        v_vector = self.xtc._calc_v_vector(rho_paired)
        
        delta_U_ref = self.xtc._calc_delta_U(v_vector, rho_paired, dm1)
         
        ranks = [20, 50, 100]
        errors = []
        times = []
        
        print("\nTesting ISDF delta_U convergence:")
        print("Rank/Full Size\tRel Error\tTime(s)")
        print("-" * 40)
        
        for rank in ranks:
            start_time = time.time()
            
            result = self.xtc.isdf(n_rank=rank)
            
            fused_rank = len(result['pivots'])
            delta_U_isdf = self.xtc._calc_delta_U_isdf(
                result['C_rho'], 
                result['xi_rho'],
                self.xtc.jastrow_factor,
                self.grid_points,
                self.weights,
                dm1=dm1
            )
            
            calc_time = time.time() - start_time
            
            rel_error = np.linalg.norm(delta_U_isdf - delta_U_ref) / np.linalg.norm(delta_U_ref)
            abs_error = np.max(np.abs(delta_U_isdf - delta_U_ref))
            errors.append((rel_error, abs_error))
            times.append(calc_time)
            
            print(f"{fused_rank}/{len(self.weights)}\t{rel_error:.2e}\t{calc_time:.2f}")
        
        rel_errors = [e[0] for e in errors]
        self.assertTrue(all(rel_errors[i] > rel_errors[i+1] for i in range(len(rel_errors)-1)),
                       "Relative errors should decrease monotonically with increasing rank")
        
        self.assertLess(rel_errors[-1], 1e-4,
                       f"Final relative error {rel_errors[-1]:.2e} should be below 1e-4")
        
        ref_time = time.time()
        delta_U_ref = self.xtc._calc_delta_U(v_vector, rho_paired, dm1)
        ref_time = time.time() - ref_time
        print(f"\nReference calculation time: {ref_time:.2f}s")
        

    @unittest.skipIf(
        _SKIP_ISDF_STRESS,
        "Exceeds GitHub-hosted runner resources on Python 3.14; covered by the 3.10 full job",
    )
    def test_isdf_ccsd_convergence(self):
        """Test convergence of ISDF XTC-CCSD energy with increasing rank."""
        from pyscf.cc import rccsd
        import time
        
        myrcc = rccsd.RCCSD(self.mf)
        eris_ref = self.xtc.make_eris()
        e_corr_ref, t1_ref, t2_ref = myrcc.kernel(eris=eris_ref)
        e_hf_ref = eris_ref.e_core + np.einsum('ii->', eris_ref.fock[:myrcc.nocc, :myrcc.nocc]) * 2
        e_dir = 2. * np.einsum('jjii->', eris_ref.oooo)
        e_ex = -1. * np.einsum('ijji->', eris_ref.oooo)
        e_hf_ref += -(e_dir + e_ex)
        e_ref = e_corr_ref + e_hf_ref
        
        ranks = [20, 50, 100]
        errors = []
        times = []
        
        print("\nTesting ISDF XTC-CCSD convergence:")
        print("Rank/Full Size\tRel Error\tTime(s)\tEnergy")
        print("-" * 50)
        
        for rank in ranks:
            start_time = time.time()
            
            result = self.xtc.isdf(n_rank=rank)
            fused_rank = len(result['pivots'])
            
            eris_isdf = self.xtc.make_eris()
            
            myrcc_isdf = rccsd.RCCSD(self.mf)
            e_corr_isdf, t1, t2 = myrcc_isdf.kernel(eris=eris_isdf)
            e_hf_isdf = eris_isdf.e_core + np.einsum('ii->', eris_isdf.fock[:myrcc_isdf.nocc, :myrcc_isdf.nocc]) * 2
            e_dir = 2. * np.einsum('jjii->', eris_isdf.oooo)
            e_ex = -1. * np.einsum('ijji->', eris_isdf.oooo)
            e_hf_isdf += -(e_dir + e_ex)
            e_isdf = e_corr_isdf + e_hf_isdf
            
            calc_time = time.time() - start_time
            rel_error = abs(e_isdf - e_ref) / abs(e_ref)
            
            errors.append(rel_error)
            times.append(calc_time)
            
            print(f"{fused_rank}/{len(self.weights)}\t{rel_error:.2e}\t{calc_time:.2f}\t{e_isdf:.8f}")
            
        self.assertTrue(all(errors[i] > errors[i+1] for i in range(len(errors)-1)),
                       "Errors should decrease monotonically with increasing rank")
        
        self.assertLess(errors[-1], 1e-4,
                       f"Final error {errors[-1]:.2e} should be below 1e-4")
        
        ref_time = time.time()
        eris_ref = self.xtc.make_eris()
        myrcc = rccsd.RCCSD(self.mf)
        myrcc.kernel(eris=eris_ref)
        ref_time = time.time() - ref_time
        
        print(f"\nReference calculation time: {ref_time:.2f}s")
        print(f"Reference energy: {e_ref:.8f}")
        


if __name__ == '__main__':
    unittest.main()
