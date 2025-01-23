"""Test for Exculding normal-ordered 3-body Transcorrelated (XTC) calculations."""

import unittest
import os 
from functools import partial, reduce
import numpy as np

from pyscf import gto, scf, ao2mo

from pytc.xtc import XTC
from pytc.lmat import calc_v_vector
from pytc.jastrow import SM7, SimpleJastrow
from pytc.df import isdf_decompose_cholesky 

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
        
        
        # Create SM7 instance with Be coefficients
        #cls.jastrow = SM7(atom='He')
        cls.jastrow = SimpleJastrow([1.4])
        
        # Initialize XTC with SM7 Jastrow
        cls.xtc = XTC(cls.mf, cls.jastrow, grid_lvl=1)  # Use grid_lvl=1 for testing
        
        # Get grid points and weights from TC parent class
        cls.grid_points = cls.xtc.grid_points
        cls.weights = cls.xtc.weights
        
        # Get orbital values on grid
        mo_values, _ = cls.xtc._eval_basis_on_grid()
        cls.rho = mo_values
        
        # Prepare paired indices
        cls.rho_paired = np.einsum('in,jn->ijn', 
                                  cls.rho, 
                                  cls.rho).reshape(-1, len(cls.weights))
        
        # Update to use jastrow from xtc object
        cls.u_gradients = cls.xtc.jastrow_factor.grad(cls.grid_points)
        
        # Calculate v_vector
        cls.v_vector = calc_v_vector(cls.rho_paired, cls.u_gradients, cls.weights)
    
    def test_get_mf_dm(self):
        """Test mean-field density matrix generation."""
        dm1 = self.xtc._get_mf_dm()
        
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
    
    #def test_calc_delta_U(self):
    #    """Test calculation of delta_U tensor."""
    #    delta_U = self.xtc._calc_delta_U(self.v_vector, self.rho_paired)
    #    
    #    # Check shape
    #    n_orb = self.mf.mo_coeff.shape[1]
    #    self.assertEqual(delta_U.shape, (n_orb,) * 4)
    #    
    #    # Test symmetry property
    #    np.testing.assert_array_almost_equal(
    #        delta_U, 
    #        delta_U.transpose(2,3,0,1)
    #    )
    
    #def test_calc_delta_h(self):
    #    """Test calculation of delta_h matrix."""
    #    delta_U = self.xtc._calc_delta_U(self.v_vector, self.rho_paired)
    #    dm1 = self.xtc._get_mf_dm()
    #    delta_h = self.xtc._calc_delta_h(delta_U, dm1)
    #    
    #    # Check shape
    #    n_orb = self.mf.mo_coeff.shape[1]
    #    self.assertEqual(delta_h.shape, (n_orb,) * 2)
    #    
    #    # Test hermiticity
    #    np.testing.assert_array_almost_equal(
    #        delta_h, 
    #        delta_h.T.conj()
    #    )
    
    #def test_get_const(self):
    #    """Test calculation of constant term."""
    #    dm1 = self.xtc._get_mf_dm()
    #    const = self.xtc.get_const(dm1=dm1)
    #    
    #    # Check that const is real
    #    self.assertTrue(np.isreal(const))
    
    def test_nhccsd(self):
        from pyscf.cc import rccsd, CCSD
        mycc = CCSD(self.mf)
        e_corr, t1, t2 = mycc.kernel()
        mycc.verbose = 5
        print("E_CCSD = ", e_corr)
        self.assertAlmostEqual(e_corr, -0.04503138331130402, places=6)
        print("|t2| = ", np.linalg.norm(t2))
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
        print("|t2| = ", np.linalg.norm(t2))
        print("corr E_XTC_CCSD = ", myrcc.e_corr)
        #self.assertAlmostEqual(tc_e_corr, -0.0443956329631562, places=6)
        # get the hf energy using fock and eris
        no = myrcc.nocc
        tc_h1e = self.xtc.get_1b()
        tc_e_hf = 2. * np.einsum('ii->', tc_h1e[:no, :no])
        tc_e_dir = 2. * np.einsum('jjii->', eris.oooo)
        tc_e_ex = -1. * np.einsum('ijji->', eris.oooo)

        tc_e_hf += (tc_e_dir + tc_e_ex) + eris.e_core 
        print("E_XTC_CCSD = ", myrcc.e_corr + tc_e_hf)
        #self.assertAlmostEqual(tc_e_hf, -14.592606059260131, places=6)

    def test_isdf(self):
        """Test ISDF decomposition and delta_U calculation."""
        # Get decomposition with small rank for testing
        n_rank = 50
        result = self.xtc.isdf(n_rank=n_rank)
        
        # Check that all expected keys are present
        expected_keys = ['C_rho', 'xi_rho', 'C_grad', 'xi_grad', 'pivots']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check reconstruction accuracy
        rho_recon = result['C_rho'] @ result['xi_rho']
        rel_error_rho = np.linalg.norm(rho_recon - self.rho_paired) / np.linalg.norm(self.rho_paired)
        
        # Check reasonable errors for ISDF decomposition
        self.assertLess(rel_error_rho, 1e-4)
        
        # Check number of pivots is reasonable
        self.assertLessEqual(len(result['pivots']), 2 * n_rank)
        self.assertGreaterEqual(len(result['pivots']), n_rank)
        
        # Test delta_U calculation using both methods
        dm1 = self.xtc._get_mf_dm()
        delta_U_orig = self.xtc._calc_delta_U(self.v_vector, self.rho_paired, dm1)
        delta_U_isdf = self.xtc._calc_delta_U_isdf(result['C_rho'], result['xi_rho'], 
                                                  self.u_gradients, dm1)
        
        # Compare results
        rel_error = np.linalg.norm(delta_U_isdf - delta_U_orig) / np.linalg.norm(delta_U_orig)
        abs_error = np.max(np.abs(delta_U_isdf - delta_U_orig))
        
        print(f"Relative error in delta_U: {rel_error:.2e}")
        print(f"Maximum absolute error: {abs_error:.2e}")
        
        # Check errors are within tolerance
        self.assertLess(rel_error, 1e-6)
        self.assertLess(abs_error, 1e-6)
        
        # Test symmetry property of ISDF delta_U
        np.testing.assert_array_almost_equal(
            delta_U_isdf, 
            delta_U_isdf.transpose(2,3,0,1)
        )

if __name__ == '__main__':
    unittest.main()
