"""Test for density-fitting and tensor decomposition implementations."""

import unittest
import numpy as np
from pyscf import gto, scf

from pytc.xtc import XTC
from pytc.jastrow import SimpleJastrow
from pytc.df import isdf_decompose_cholesky, reconstruct_rho, test_accuracy, test_multi_accuracy

def get_be_ccpvdz():
    """Return a Be atom with cc-pVDZ basis for testing."""
    mol = gto.M(atom='Be 0 0 0', basis='ccpvdz', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

class TestDF(unittest.TestCase):
    """Test density-fitting and tensor decomposition methods."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case using Be atom."""
        # Get mean-field data
        _, cls.mf = get_be_ccpvdz()
        
        # Create SimpleJastrow instance
        cls.jastrow = SimpleJastrow([1.4])
        
        # Initialize XTC
        cls.xtc = XTC(cls.mf, cls.jastrow, grid_lvl=1)
        
        # Get orbital values on grid
        mo_values, _ = cls.xtc._eval_basis_on_grid()
        rho = mo_values
        
        # Get paired density
        cls.rho_paired = np.einsum('in,jn->ijn', rho, rho).reshape(-1, len(cls.xtc.weights))
        
    def test_rank_convergence(self):
        """Test ISDF decomposition with different ranks."""
        ranks = [10, 20, 40]
        errors = []
        
        for rank in ranks:
            # Perform decomposition
            C, xi = isdf_decompose_cholesky(self.rho_paired, n_rank=rank)
            
            # Check shapes
            self.assertEqual(C.shape[1], rank)
            self.assertEqual(xi.shape[0], rank)
            
            # Reconstruct and check error
            rho_reconstructed = reconstruct_rho(C, xi)
            rel_error = np.linalg.norm(self.rho_paired - rho_reconstructed) / np.linalg.norm(self.rho_paired)
            errors.append(rel_error)
            
        # Check that error decreases with increasing rank
        self.assertTrue(all(errors[i] > errors[i+1] for i in range(len(errors)-1)))
    
    def test_reconstruction(self):
        """Test reconstruction accuracy."""
        for rank in range(10, self.rho_paired.shape[1]//10, 10):
            # Perform decomposition
            C, xi = isdf_decompose_cholesky(self.rho_paired, n_rank=rank)
        
            # Test reconstruction
            rho_reconstructed = reconstruct_rho(C, xi)
            rel_error = test_accuracy(self.rho_paired, C, xi)
            print(f"Rank {rank} error: {rel_error}")
        self.assertLess(rel_error, 1e-4)
        
        # Check shape preservation
        self.assertEqual(rho_reconstructed.shape, self.rho_paired.shape)

    def test_multi_decomposition(self):
        """Test decomposition of multiple densities."""
        # Create two test densities with different ranks
        rank1, rank2 = 10, 15
        N = self.rho_paired.shape[1]
        U1 = np.random.randn(self.rho_paired.shape[0], rank1)
        U2 = np.random.randn(self.rho_paired.shape[0], rank2)
        V1 = np.random.randn(N, rank1)
        V2 = np.random.randn(N, rank2)
        rho1 = U1 @ V1.T
        rho2 = U2 @ V2.T
        
        # Test decomposition
        err1, err2, n_fused = test_multi_accuracy(rho1, rho2, rank1, rank2)
        
        # Check errors are reasonable
        self.assertLess(err1, 1e-4)
        self.assertLess(err2, 1e-4)
        
        # Check number of fused pivots
        self.assertLessEqual(n_fused, rank1 + rank2)
        self.assertGreaterEqual(n_fused, max(rank1, rank2))

if __name__ == '__main__':
    unittest.main()
