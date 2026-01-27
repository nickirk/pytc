"""Test for density-fitting and tensor decomposition implementations."""

import unittest
import numpy as np
from pyscf import gto, scf

from pytc.xtc import XTC
from pytc.jastrow import REXP
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

        # Create REXP instance
        cls.jastrow = REXP([1.4])

        # Initialize XTC
        cls.xtc = XTC(cls.mf, cls.jastrow, grid_lvl=1)

        # Get orbital values on grid
        mo_values, _ = cls.xtc._eval_basis_on_grid()
        rho = mo_values

        # Get paired density
        cls.rho_paired = np.einsum('in,jn->ijn', rho, rho).reshape(-1, len(cls.xtc.weights))

        # Get mo_grad for tensor tests
        _, cls.mo_grad = cls.xtc._eval_basis_on_grid()
        cls.rho_grad = cls.mo_grad  # Shape (Nb, N_grid, 3)

    def test_tensor_decomposition(self):
        """Test ISDF decomposition with tensor input."""
        n_rank = 20
        # Input has shape (Nb, N_grid, 3)
        C, xi = isdf_decompose_cholesky(self.rho_grad, n_rank=n_rank)

        # Check shapes including trailing dimensions
        self.assertEqual(C.shape, (self.rho_grad.shape[0], n_rank, 3))
        self.assertEqual(xi.shape, (n_rank, self.rho_grad.shape[1], 3))

        # Test reconstruction
        rho_reconstructed = reconstruct_rho(C, xi)
        self.assertEqual(rho_reconstructed.shape, self.rho_grad.shape)

        # Check error using norm along spatial dimensions
        error = np.linalg.norm(rho_reconstructed - self.rho_grad) / np.linalg.norm(self.rho_grad)
        self.assertLess(error, 1e-4)

    def test_multi_decomposition(self):
        """Test decomposition of scalar and tensor densities."""
        # Use both scalar and tensor valued densities
        rank = 50

        # Create tensor test density
        rho_tensor = self.rho_grad  # Shape (Nb, N_grid, 3)

        # Test decomposition with mixed types
        rel_error1, abs_error1, rel_error2, abs_error2, n_fused = test_multi_accuracy(
            self.rho_paired,  # Matrix input
            rho_tensor,       # Tensor input
            rank, rank
        )

        # Check errors are reasonable
        self.assertLess(rel_error1, 1e-4)
        self.assertLess(rel_error2, 1e-4)

        # Check number of fused pivots
        self.assertLessEqual(n_fused, 2 * rank)
        self.assertGreaterEqual(n_fused, rank)

    def test_rank_convergence(self):
        """Test ISDF decomposition with different ranks."""
        # Test both matrix and tensor inputs
        for input_data in [self.rho_paired, self.rho_grad]:
            ranks = [10, 20, 40]
            errors = []

            for rank in ranks:
                C, xi = isdf_decompose_cholesky(input_data, n_rank=rank)

                # Reconstruct and check error
                reconstructed = reconstruct_rho(C, xi)
                error = np.linalg.norm(reconstructed - input_data) / np.linalg.norm(input_data)
                errors.append(error)

            # Check that error decreases with increasing rank
            # Check that error generally decreases (highest rank should have lowest error)
            self.assertLess(errors[-1], errors[0])

    def test_reconstruction_shapes(self):
        """Test shape preservation in reconstruction."""
        rank = 150

        # Test matrix input
        C_mat, xi_mat = isdf_decompose_cholesky(self.rho_paired, n_rank=rank)
        recon_mat = reconstruct_rho(C_mat, xi_mat)
        self.assertEqual(recon_mat.shape, self.rho_paired.shape)

        # Check reconstruction accuracy
        rel_error, _ = test_accuracy(self.rho_paired, C_mat, xi_mat)
        self.assertLess(rel_error, 1e-4)

        # Test tensor input
        C_tens, xi_tens = isdf_decompose_cholesky(self.rho_grad, n_rank=rank)
        recon_tens = reconstruct_rho(C_tens, xi_tens)
        self.assertEqual(recon_tens.shape, self.rho_grad.shape)

        # Check reconstruction accuracy
        rel_error, _ = test_accuracy(self.rho_grad, C_tens, xi_tens)
        self.assertLess(rel_error, 1e-4)

if __name__ == '__main__':
    unittest.main()