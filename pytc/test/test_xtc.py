"""Test for Exculding normal-ordered 3-body Transcorrelated (XTC) calculations."""

import unittest
import os 
from functools import partial, reduce
import numpy as np
import time

from pyscf import gto, scf, ao2mo

from pytc.xtc import XTC
from pytc.jastrow import REXP

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
        # Get mean-field data
        _, cls.mf = get_be_ccpvdz()
        
        
        cls.jastrow = REXP([1.])
        
        # Initialize XTC with REXP Jastrow
        cls.xtc = XTC(cls.mf, cls.jastrow, grid_lvl=2)  
        
        # Get grid points and weights from TC parent class
        cls.grid_points = cls.xtc.grid_points
        cls.weights = cls.xtc.weights
        cls.jastrow = cls.xtc.jastrow_factor
        
        # Get orbital values on grid
        mo_values, _ = cls.xtc._eval_basis_on_grid()
        cls.rho = mo_values
        
        # Prepare paired indices
        cls.rho_paired = np.einsum('in,jn->ijn', 
                                  cls.rho, 
                                  cls.rho).reshape(-1, len(cls.weights))
    
    def test_get_mf_dm(self):
        """Test mean-field density matrix generation."""
        dm1 = self.xtc._get_mf_dm() * 2
        
        # Check shape
        self.assertEqual(dm1.shape, (self.mf.mo_coeff.shape[1],) * 2)
        
        # Check trace equals number of electrons
        self.assertAlmostEqual(np.trace(dm1), self.xtc.mol.nelectron)
        
        # Check diagonal elements are 2.0 for occupied and 0.0 for virtual
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
        eri1 = ao2mo.incore.full(self.xtc.mf._eri, self.xtc.mo_coeff, compact=False)
        eri1 = ao2mo.restore(1, eri1, self.xtc.mo_coeff.shape[1])
        h1e = mycc._scf.get_hcore()
        h1e = reduce(np.dot, (self.xtc.mo_coeff.T, h1e, self.xtc.mo_coeff))
        
        e_hf_0 = 2. * np.einsum('ii->', h1e[:mycc.nocc, :mycc.nocc])
        e_dir = 2. * np.einsum('jjii->', eri1[:mycc.nocc, :mycc.nocc, :mycc.nocc, :mycc.nocc])
        e_ex = -1. * np.einsum('ijji->', eri1[:mycc.nocc, :mycc.nocc, :mycc.nocc, :mycc.nocc])
        e_hf_0 += (e_dir + e_ex) + mycc._scf.energy_nuc()
        print("Check e_hf = ",  e_hf_0)
        self.assertAlmostEqual(e_hf_0, -14.57233763095337, places=6)

        myrcc = rccsd.RCCSD(self.mf)
        #myrcc.verbose = 5
        eris = self.xtc.make_eris()
        tc_e_corr, t1, t2 = myrcc.kernel(eris=eris)
        t = myrcc.amplitudes_to_vector(t1, t2)
        print("|t2| = ", np.linalg.norm(t2))
        print("|t1+t2| = ", np.linalg.norm(t))
        print("corr E_XTC_CCSD = ", myrcc.e_corr)
        self.assertAlmostEqual(tc_e_corr, -0.03272155333587409, places=6)
        # get the hf energy using fock and eris
        no = myrcc.nocc
        tc_h1e = self.xtc.get_1b()
        tc_e_hf = 2. * np.einsum('ii->', tc_h1e[:no, :no])
        tc_e_dir = 2. * np.einsum('jjii->', eris.oooo)
        tc_e_ex = -1. * np.einsum('ijji->', eris.oooo)

        tc_e_hf += (tc_e_dir + tc_e_ex) + eris.e_core 
        print("E_XTC_CCSD = ", myrcc.e_corr + tc_e_hf)
        self.assertAlmostEqual(tc_e_hf + tc_e_corr, -14.656373992235194, places=6)


    def test_delta_U_isdf_convergence(self):
        """Test convergence of ISDF delta_U calculation with increasing rank."""
        # Get reference delta_U using original method
        dm1 = self.xtc._get_mf_dm()
        
        # Compute V vector with batched processing
        rho, _ = self.xtc._get_intermediates()
        rho_paired = np.einsum('in,jn->ijn', rho, rho).reshape(-1, rho.shape[1])
        v_vector = self.xtc._calc_v_vector(rho_paired)
        
        delta_U_ref = self.xtc._calc_delta_U(v_vector, rho_paired, dm1)
         
        # Test range of ranks
        ranks = [20, 50, 100]
        #ranks = [300, 400]
        errors = []
        times = []
        
        print("\nTesting ISDF delta_U convergence:")
        print("Rank/Full Size\tRel Error\tTime(s)")
        print("-" * 40)
        
        for rank in ranks:
            start_time = time.time()
            
            # Get ISDF decomposition
            result = self.xtc.isdf(n_rank=rank)
            
            fused_rank = len(result['pivots'])
            # Calculate delta_U using ISDF
            delta_U_isdf = self.xtc._calc_delta_U_isdf(
                result['C_rho'], 
                result['xi_rho'],
                self.xtc.jastrow_factor,
                self.grid_points,
                self.weights,
                dm1=dm1
            )
            
            calc_time = time.time() - start_time
            
            # Calculate errors
            rel_error = np.linalg.norm(delta_U_isdf - delta_U_ref) / np.linalg.norm(delta_U_ref)
            abs_error = np.max(np.abs(delta_U_isdf - delta_U_ref))
            errors.append((rel_error, abs_error))
            times.append(calc_time)
            
            print(f"{fused_rank}/{len(self.weights)}\t{rel_error:.2e}\t{calc_time:.2f}")
        
        # Check that errors decrease with increasing rank
        rel_errors = [e[0] for e in errors]
        self.assertTrue(all(rel_errors[i] > rel_errors[i+1] for i in range(len(rel_errors)-1)),
                       "Relative errors should decrease monotonically with increasing rank")
        
        # Check final accuracy is reasonable
        self.assertLess(rel_errors[-1], 1e-4,
                       f"Final relative error {rel_errors[-1]:.2e} should be below 1e-4")
        
        # Verify timing improvement
        ref_time = time.time()
        delta_U_ref = self.xtc._calc_delta_U(v_vector, rho_paired, dm1)
        ref_time = time.time() - ref_time
        print(f"\nReference calculation time: {ref_time:.2f}s")
        

    def test_isdf_ccsd_convergence(self):
        """Test convergence of ISDF XTC-CCSD energy with increasing rank."""
        from pyscf.cc import rccsd
        import time
        
        # First get reference XTC-CCSD energy
        myrcc = rccsd.RCCSD(self.mf)
        eris_ref = self.xtc.make_eris()
        e_corr_ref, t1_ref, t2_ref = myrcc.kernel(eris=eris_ref)
        e_hf_ref = eris_ref.e_core + np.einsum('ii->', eris_ref.fock[:myrcc.nocc, :myrcc.nocc]) * 2
        e_dir = 2. * np.einsum('jjii->', eris_ref.oooo)
        e_ex = -1. * np.einsum('ijji->', eris_ref.oooo)
        e_hf_ref += -(e_dir + e_ex)
        e_ref = e_corr_ref + e_hf_ref
        
        # Test range of ranks
        ranks = [20, 50, 100]
        errors = []
        times = []
        
        print("\nTesting ISDF XTC-CCSD convergence:")
        print("Rank/Full Size\tRel Error\tTime(s)\tEnergy")
        print("-" * 50)
        
        for rank in ranks:
            start_time = time.time()
            
            # Get ISDF decomposition
            result = self.xtc.isdf(n_rank=rank)
            fused_rank = len(result['pivots'])
            
            # Make ERIS with ISDF
            eris_isdf = self.xtc.make_eris()
            
            # Run CCSD
            myrcc_isdf = rccsd.RCCSD(self.mf)
            e_corr_isdf, t1, t2 = myrcc_isdf.kernel(eris=eris_isdf)
            e_hf_isdf = eris_isdf.e_core + np.einsum('ii->', eris_isdf.fock[:myrcc_isdf.nocc, :myrcc_isdf.nocc]) * 2
            e_dir = 2. * np.einsum('jjii->', eris_isdf.oooo)
            e_ex = -1. * np.einsum('ijji->', eris_isdf.oooo)
            e_hf_isdf += -(e_dir + e_ex)
            e_isdf = e_corr_isdf + e_hf_isdf
            
            # Calculate timing and errors
            calc_time = time.time() - start_time
            rel_error = abs(e_isdf - e_ref) / abs(e_ref)
            
            errors.append(rel_error)
            times.append(calc_time)
            
            print(f"{fused_rank}/{len(self.weights)}\t{rel_error:.2e}\t{calc_time:.2f}\t{e_isdf:.8f}")
            
        # Check that errors decrease with increasing rank
        self.assertTrue(all(errors[i] > errors[i+1] for i in range(len(errors)-1)),
                       "Errors should decrease monotonically with increasing rank")
        
        # Check final accuracy is reasonable
        self.assertLess(errors[-1], 1e-4,
                       f"Final error {errors[-1]:.2e} should be below 1e-4")
        
        # Verify timing improvement
        ref_time = time.time()
        eris_ref = self.xtc.make_eris()
        myrcc = rccsd.RCCSD(self.mf)
        myrcc.kernel(eris=eris_ref)
        ref_time = time.time() - ref_time
        
        print(f"\nReference calculation time: {ref_time:.2f}s")
        print(f"Reference energy: {e_ref:.8f}")
        


if __name__ == '__main__':
    unittest.main()
