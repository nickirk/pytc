"""Test for Exculding normal-ordered 3-body Transcorrelated (XTC) calculations."""

import unittest
import os 
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np

from pyscf import gto, scf

from pytcint.xtc import XTC
from pytcint.lmat import calc_v_vector
from pytcint.jastrow import SM7

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
        cls.jastrow = SM7(atom='Be')
        
        # Initialize XTC with SM7 Jastrow
        cls.xtc = XTC(cls.mf, cls.jastrow, grid_lvl=1)  # Use grid_lvl=1 for testing
        
        # Get grid points and weights from TC parent class
        cls.grid_points = cls.xtc.grid_points
        cls.weights = cls.xtc.weights
        
        # Get orbital values on grid
        ao_values, _ = cls.xtc._eval_basis_on_grid()
        cls.rho = ao_values
        
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
    
    def test_calc_delta_U(self):
        """Test calculation of delta_U tensor."""
        delta_U = self.xtc._calc_delta_U(self.v_vector, self.rho_paired)
        
        # Check shape
        n_orb = self.mf.mo_coeff.shape[1]
        self.assertEqual(delta_U.shape, (n_orb,) * 4)
        
        # Test symmetry property
        np.testing.assert_array_almost_equal(
            delta_U, 
            delta_U.transpose(2,3,0,1)
        )
    
    def test_calc_delta_h(self):
        """Test calculation of delta_h matrix."""
        delta_U = self.xtc._calc_delta_U(self.v_vector, self.rho_paired)
        dm1 = self.xtc._get_mf_dm()
        delta_h = self.xtc._calc_delta_h(delta_U, dm1)
        
        # Check shape
        n_orb = self.mf.mo_coeff.shape[1]
        self.assertEqual(delta_h.shape, (n_orb,) * 2)
        
        # Test hermiticity
        np.testing.assert_array_almost_equal(
            delta_h, 
            delta_h.T.conj()
        )
    
    def test_get_const(self):
        """Test calculation of constant term."""
        dm1 = self.xtc._get_mf_dm()
        const = self.xtc.get_const(dm1=dm1)
        
        # Check that const is real
        self.assertTrue(np.isreal(const))
    
    def test_nhccsd(self):
        from pyscf.cc import rccsd, CCSD
        mycc = CCSD(self.mf).run()
        print("E_CCSD = ", mycc.e_corr)

        myrcc = rccsd.RCCSD(self.mf)
        eris = self.xtc.make_eris()
        myrcc.kernel(eris=eris)
        print("E_XTC_CCSD = ", myrcc.e_corr)
    
    #def test_scan_param(self):
    #    """Test scanning over parameter space for finding optimal Jastrow factor 
    #    in terms of smallest xtc-mp2 t2 norm."""
    #    from pyscf import mp, cc
    #    
    #    # Scan over alpha values
    #    for alpha in np.linspace(1.0, 1.5, 10):
    #        jastrow = SimpleJastrow([alpha])
    #        xtc = XTC(self.mf, jastrow, grid_lvl=1)
    #        eris = xtc.make_eris()
    #        #mymp2 = mp.MP2(xtc.mf)
    #        mycc = cc.rccsd.RCCSD(xtc.mf)
    #        e_corr, t1, t2 = mycc.kernel(eris=eris)
    #        print(f"alpha = {alpha:.2f}, E_XTC_MP2 = {e_corr:.6f}, t2_norm = {np.linalg.norm(t2):.6f}")



if __name__ == '__main__':
    unittest.main()
