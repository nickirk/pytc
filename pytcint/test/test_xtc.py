"""Test for Exculding normal-ordered 3-body Transcorrelated (XTC) calculations."""

import unittest
import numpy as np
from pyscf import gto, scf

from pytcint.xtc import XTC
from pytcint.lmat import calc_v_vector
from pytcint.jastrow import Jastrow

def get_h2_sto3g():
    """Return a simple H2 molecule with STO-3G basis for testing."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

class SimpleJastrow(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    def __call__(self, r1, r2, atomic_positions=None):
        delta_r = r1[..., np.newaxis, :] - r2[np.newaxis, ...]
        return np.exp(-self.parameters[0] * np.linalg.norm(delta_r, axis=-1))
    
    def grad(self, r1, r2=None, atomic_positions=None):
        if r2 is None:
            r2 = r1
        delta_r = r1[..., np.newaxis, :] - r2[np.newaxis, ...]
        norm = np.linalg.norm(delta_r, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)  # Avoid division by zero
        return -self.parameters[0] * delta_r / norm * self.__call__(r1, r2)[..., np.newaxis]

class TestXTC(unittest.TestCase):
    """Test XTranscorrelated calculations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case using H2 molecule from test_lmat."""
        # Get mean-field data from test_lmat
        _, cls.mf = get_h2_sto3g()
        cls.jastrow = SimpleJastrow([0.5])  # alpha = 0.5
        
        # Initialize XTC object
        cls.xtc = XTC(cls.mf, grid_lvl=1)  # Use grid_lvl=1 for testing
        
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
        
        # Get test Jastrow gradients from TestLmat
        #test_lmat = TestLmat()
        cls.u_gradients = cls.jastrow.grad(cls.grid_points)
        
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
        delta_U = self.xtc._calc_delta_U(self.v_vector, self.rho_paired)
        dm1 = self.xtc._get_mf_dm()
        delta_h = self.xtc._calc_delta_h(delta_U, dm1)
        const = self.xtc.get_const(dm1=dm1, delta_h=delta_h)
        
        # Check that const is real
        self.assertTrue(np.isreal(const))


if __name__ == '__main__':
    unittest.main()
