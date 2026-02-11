"""Test for two-body matrix elements K."""

import unittest
import numpy as np
from pyscf import gto, scf, dft
from pytc.legacy.jastrow import REXP
import time

def get_be_ccpvdz():
    """Return a Be atom with cc-pVDZ basis for testing."""
    mol = gto.M(atom='Be 0 0 0;  ', basis='ccpvdz', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

class TestKmat(unittest.TestCase):
    """Test K matrix elements."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a Be atom for all tests in this class."""
        cls.mol, cls.mf = get_be_ccpvdz()
        cls.n_orb = cls.mf.mo_coeff.shape[1]
        cls.jastrow = REXP([0.5])  # alpha = 0.5
        
        # Set up grid points for testing
        grids = dft.gen_grid.Grids(cls.mol)
        grids.level = 1  # Use coarse grid for testing
        grids.build()
        cls.grid_points = grids.coords  # Keep original shape (N_grid, 3)
        cls.weights = grids.weights
        
        # Prepare basis functions on grid with correct shapes
        ao = dft.numint.eval_ao(cls.mol, cls.grid_points, deriv=1)
        cls.rho = np.dot(ao[0], cls.mf.mo_coeff).T  # Shape: (N_orb, N_grid)
        cls.nabla_rho = np.dot(ao[1:4].transpose(1,0,2), 
                              cls.mf.mo_coeff).transpose(2,0,1)  # Shape: (N_orb, N_grid, 3)
        
        # Prepare paired indices for testing
        cls.rho_paired = np.einsum('in,jn->ijn', 
                                  cls.rho, 
                                  cls.rho).reshape(-1, len(cls.weights))
        
        cls.nabla_rho_paired = np.einsum('ind,jn->ijnd', 
                                        cls.nabla_rho, 
                                        cls.rho).reshape(-1, len(cls.weights), 3)
        
    
    def test_k_shapes(self):
        """Verify K matrix shapes."""
        from pytc.legacy.kmat import calc_K1, calc_K2, calc_K3
        import time 
        time_start = time.time() 
        k1 = calc_K1(self.rho_paired, self.nabla_rho_paired,
                   self.jastrow, self.grid_points, self.weights).reshape((self.n_orb,)*4)
        k1_time = time.time() - time_start
        time_start = time.time()
        k2 = calc_K2(self.rho_paired, self.nabla_rho_paired,
                   self.jastrow, self.grid_points, self.weights).reshape((self.n_orb,)*4)
        k2_time = time.time() - time_start
        time_start = time.time()
        k3 = calc_K3(self.rho_paired, self.jastrow,
                   self.grid_points, self.weights).reshape((self.n_orb,)*4)
        k3_time = time.time() - time_start
        print(f"K1 time: {k1_time}")
        print(f"K2 time: {k2_time}")
        print(f"K3 time: {k3_time}")
        
        
        for k in (k1, k2, k3):
            self.assertEqual(k.shape, (self.n_orb,)*4)
    
    def test_k2_k3_symmetry(self):
        """Test symmetry properties of K2 (laplacian) and K3 (square) integrals."""
        from pytc.legacy.kmat import calc_K2, calc_K3
        
        k2 = calc_K2(
            self.rho_paired,
            self.nabla_rho_paired,
            self.jastrow,
            self.grid_points,
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        k3 = calc_K3(
            self.rho_paired,
            self.jastrow,
            self.grid_points,
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        # Test symmetries using array operations
        for K, name in [(k2, 'K2'), (k3, 'K3')]:
            self.assertTrue(np.allclose(K.transpose(1,0,3,2), K),
                          msg=f"{name} transpose symmetry failed")
    
    def test_k1_k2_symmetry(self):
        """Test if K1 + K2 is equal to K1 with p and r indices swapped."""
        from pytc.legacy.kmat import calc_K1, calc_K2
        
        k1 = calc_K1(
            self.rho_paired,
            self.nabla_rho_paired,
            self.jastrow,
            self.grid_points,
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        k2 = calc_K2(
            self.rho_paired,
            self.nabla_rho_paired,
            self.jastrow,
            self.grid_points,
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        tmp = k1 + k2
        # swap p and r indices
        tmp = tmp.swapaxes(0, 1)
        # compare each element
        self.assertTrue(np.allclose(k1, -tmp))


class TestISDF(TestKmat):
    """Test ISDF implementation of K matrices."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case with Be atom."""
        super().setUpClass()
        
        # Get reference K matrices
        from pytc.legacy.kmat import calc_K1, calc_K2, calc_K3
        
        cls.k1_ref = calc_K1(cls.rho_paired, cls.nabla_rho_paired,
                           cls.jastrow, cls.grid_points, cls.weights).reshape(cls.n_orb, cls.n_orb, cls.n_orb, cls.n_orb)
        cls.k2_ref = calc_K2(cls.rho_paired, cls.nabla_rho_paired,
                           cls.jastrow, cls.grid_points, cls.weights).reshape(cls.n_orb, cls.n_orb, cls.n_orb, cls.n_orb)
        cls.k3_ref = calc_K3(cls.rho_paired, cls.jastrow,
                           cls.grid_points, cls.weights).reshape(cls.n_orb, cls.n_orb, cls.n_orb, cls.n_orb)
    
    def test_isdf_convergence(self):
        """Test if K1_isdf, K2_isdf, and K3_isdf converge to original values with increasing rank."""
        from pytc.legacy.df import isdf_decompose_multi
        from pytc.legacy.kmat import calc_K1_isdf, calc_K2_isdf, calc_K3_isdf

        # Test different ranks as fractions of grid points
        ranks = [len(self.weights) // n for n in [1000, 100, 50]]
        errors_k1, errors_k2, errors_k3 = [], [], []
        times_k1, times_k2, times_k3 = [], [], []

        for rank in ranks:
            # Perform ISDF decomposition
            C_rho, xi_rho, C_grad, xi_grad, fused_pivots = isdf_decompose_multi(
                self.rho_paired, 
                self.nabla_rho_paired,
                rank, rank
            )

            # Compute K1, K2, and K3 using ISDF
            start_time = time.time()
            k1_isdf = calc_K1_isdf(
                C_rho, xi_rho, C_grad, xi_grad,
                self.jastrow, self.grid_points, self.weights
            ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
            times_k1.append(time.time() - start_time)

            start_time = time.time()
            k2_isdf = calc_K2_isdf(
                C_rho, xi_rho, C_grad, xi_grad,
                self.jastrow, self.grid_points, self.weights
            ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
            times_k2.append(time.time() - start_time)
            # assert the relation betweek K1 and K2
            tmp = k1_isdf + k2_isdf
            tmp = tmp.swapaxes(0, 1)
            self.assertTrue(np.allclose(k1_isdf, -tmp))

            start_time = time.time()
            k3_isdf = calc_K3_isdf(
                C_rho, xi_rho,
                self.jastrow, self.grid_points, self.weights
            ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
            times_k3.append(time.time() - start_time)

            # Calculate relative errors
            error_k1 = np.linalg.norm(k1_isdf - self.k1_ref) / np.linalg.norm(self.k1_ref)
            error_k2 = np.linalg.norm(k2_isdf - self.k2_ref) / np.linalg.norm(self.k2_ref)
            error_k3 = np.linalg.norm(k3_isdf - self.k3_ref) / np.linalg.norm(self.k3_ref)

            errors_k1.append(error_k1)
            errors_k2.append(error_k2)
            errors_k3.append(error_k3)

            print(f"ISDF rank {len(fused_pivots)}: K_err=({error_k1:.1e},{error_k2:.1e},{error_k3:.1e})")

        # Check if errors decrease with increasing rank
        self.assertTrue(all(errors_k1[i] > errors_k1[i+1] for i in range(len(errors_k1)-1)))
        self.assertTrue(all(errors_k2[i] > errors_k2[i+1] for i in range(len(errors_k2)-1)))
        self.assertTrue(all(errors_k3[i] > errors_k3[i+1] for i in range(len(errors_k3)-1)))

        # Check if final errors are below thresholds
        self.assertLess(errors_k1[-1], 1e-5)
        self.assertLess(errors_k2[-1], 1e-5)
        self.assertLess(errors_k3[-1], 1e-5)


if __name__ == '__main__':
    unittest.main()
