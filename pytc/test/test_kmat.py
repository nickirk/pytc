"""Test for two-body matrix elements K."""

import unittest
import numpy as np
from pyscf import gto, scf, dft
from pytc.jastrow import Jastrow, SimpleJastrow
import time

def get_be_ccpvdz():
    """Return a Be atom with cc-pVDZ basis for testing."""
    mol = gto.M(atom='Be 0 0 0; Be 0 0 1; Be 0 0 2', basis='ccpvdz', unit='Bohr')
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
        cls.jastrow = SimpleJastrow([0.5])  # alpha = 0.5
        
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
        
        # Pre-compute u_gradients for all tests
        cls.u_gradients = cls.jastrow.grad(cls.grid_points)
    
    def test_k1_shape(self):
        """Test if K1 (nabla) integral has correct shape."""
        from pytc.kmat import calc_K1
        k1 = calc_K1(
            self.rho_paired,
            self.nabla_rho_paired,
            self.u_gradients,  # Pre-computed gradients
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        self.assertEqual(k1.shape, (self.n_orb,)*4)
    
    def test_k2_shape(self):
        """Test if K2 (laplacian) integral has correct shape."""
        from pytc.kmat import calc_K2
        k2 = calc_K2(
            self.rho_paired,
            self.nabla_rho_paired,
            self.u_gradients,  # Pre-computed gradients
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        self.assertEqual(k2.shape, (self.n_orb,)*4)
    
    def test_k3_shape(self):
        """Test if K3 (square) integral has correct shape."""
        from pytc.kmat import calc_K3
        k3 = calc_K3(
            self.rho_paired,
            self.u_gradients,  # Pre-computed gradients
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        self.assertEqual(k3.shape, (self.n_orb,)*4)
    
    def test_k2_k3_symmetry(self):
        """Test symmetry properties of K2 (laplacian) and K3 (square) integrals."""
        from pytc.kmat import calc_K2, calc_K3
        
        k2 = calc_K2(
            self.rho_paired,
            self.nabla_rho_paired,
            self.u_gradients,
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        k3 = calc_K3(
            self.rho_paired,
            self.u_gradients,
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        # Test symmetries for laplacian and square terms
        for K, name in [(k2, 'K2'), (k3, 'K3')]:
            for p in range(2):
                for q in range(2):
                    for r in range(2):
                        for s in range(2):
                            self.assertAlmostEqual(
                                K[p,r,q,s], 
                                K[r,p,s,q], 
                                places=10,
                                msg=f"Symmetry failed for {name}[{p},{q},{r},{s}]"
                            )
                            self.assertAlmostEqual(
                                K[q,s,p,r],
                                K[s,q,r,p],
                                places=10,
                                msg=f"Symmetry failed for {name}[{p},{q},{r},{s}]"
                            )
    
    def test_k1_k2_symmetry(self):
        """Test if K1 + K2 is equal to K1 with p and r indices swapped."""
        from pytc.kmat import calc_K1, calc_K2
        
        k1 = calc_K1(
            self.rho_paired,
            self.nabla_rho_paired,
            self.u_gradients,
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        k2 = calc_K2(
            self.rho_paired,
            self.nabla_rho_paired,
            self.u_gradients,
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        
        tmp = k1 + k2
        # swap p and r indices
        tmp = tmp.swapaxes(0, 1)
        # compare each element
        self.assertTrue(np.allclose(k1, -tmp))


class TestISDF(unittest.TestCase):
    """Test ISDF implementation of K matrices."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case with Be atom."""
        # Reuse setup from TestKmat
        cls.mol, cls.mf = get_be_ccpvdz()
        cls.n_orb = cls.mf.mo_coeff.shape[1]
        cls.jastrow = SimpleJastrow([0.5])
        
        # Set up grid
        grids = dft.gen_grid.Grids(cls.mol)
        grids.level = 1  # Use finer grid for ISDF tests
        grids.build()
        cls.grid_points = grids.coords
        cls.weights = grids.weights
        
        # Get basis functions and gradients
        ao = dft.numint.eval_ao(cls.mol, cls.grid_points, deriv=1)
        cls.rho = np.dot(ao[0], cls.mf.mo_coeff).T
        cls.nabla_rho = np.dot(ao[1:4].transpose(1,0,2), 
                              cls.mf.mo_coeff).transpose(2,0,1)
        
        # Prepare paired quantities
        cls.rho_paired = np.einsum('in,jn->ijn', 
                                  cls.rho, 
                                  cls.rho).reshape(-1, len(cls.weights))
        cls.nabla_rho_paired = np.einsum('rnc,pn->prnc', 
                                        cls.nabla_rho, 
                                        cls.rho).reshape(-1, len(cls.weights), 3)
        
        # Get reference K1, K2, and K3
        cls.u_gradients = cls.jastrow.grad(cls.grid_points)
        from pytc.kmat import calc_K1, calc_K2, calc_K3
        
        start_time = time.time()
        cls.k1_ref = calc_K1(
            cls.rho_paired,
            cls.nabla_rho_paired,
            cls.u_gradients,
            cls.weights
        ).reshape(cls.n_orb, cls.n_orb, cls.n_orb, cls.n_orb)
        cls.k1_time = time.time() - start_time
        
        start_time = time.time()
        cls.k2_ref = calc_K2(
            cls.rho_paired,
            cls.nabla_rho_paired,
            cls.u_gradients,
            cls.weights
        ).reshape(cls.n_orb, cls.n_orb, cls.n_orb, cls.n_orb)
        cls.k2_time = time.time() - start_time
        
        start_time = time.time()
        cls.k3_ref = calc_K3(
            cls.rho_paired,
            cls.u_gradients,
            cls.weights
        ).reshape(cls.n_orb, cls.n_orb, cls.n_orb, cls.n_orb)
        cls.k3_time = time.time() - start_time
    
    def test_isdf_convergence(self):
        """Test if K1_isdf, K2_isdf, and K3_isdf converge to original values with increasing rank."""
        from pytc.df import isdf_decompose_multi
        from pytc.kmat import calc_K1_isdf, calc_K2_isdf, calc_K3_isdf

        # Test different ranks as fractions of grid points
        ranks = [len(self.weights) // n for n in [1000, 100, 50, 10, 5]]
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
                self.u_gradients, self.weights
            ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
            times_k1.append(time.time() - start_time)

            start_time = time.time()
            k2_isdf = calc_K2_isdf(
                C_rho, xi_rho, C_grad, xi_grad,
                self.u_gradients, self.weights
            ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
            times_k2.append(time.time() - start_time)

            start_time = time.time()
            k3_isdf = calc_K3_isdf(
                C_rho, xi_rho,
                self.u_gradients, self.weights
            ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
            times_k3.append(time.time() - start_time)

            # Calculate relative errors
            error_k1 = np.linalg.norm(k1_isdf - self.k1_ref) / np.linalg.norm(self.k1_ref)
            error_k2 = np.linalg.norm(k2_isdf - self.k2_ref) / np.linalg.norm(self.k2_ref)
            error_k3 = np.linalg.norm(k3_isdf - self.k3_ref) / np.linalg.norm(self.k3_ref)

            errors_k1.append(error_k1)
            errors_k2.append(error_k2)
            errors_k3.append(error_k3)

            print(f"Rank {len(fused_pivots)}/{len(self.weights)}:")
            print(f"  K1 relative error = {error_k1:.2e}, ISDF time = {times_k1[-1]:.2f}s, ref time = {self.k1_time:.2f}s")
            print(f"  K2 relative error = {error_k2:.2e}, ISDF time = {times_k2[-1]:.2f}s, ref time = {self.k2_time:.2f}s")
            print(f"  K3 relative error = {error_k3:.2e}, ISDF time = {times_k3[-1]:.2f}s, ref time = {self.k3_time:.2f}s")

        # Check if errors decrease with increasing rank
        self.assertTrue(all(errors_k1[i] > errors_k1[i+1] for i in range(len(errors_k1)-1)))
        self.assertTrue(all(errors_k2[i] > errors_k2[i+1] for i in range(len(errors_k2)-1)))
        self.assertTrue(all(errors_k3[i] > errors_k3[i+1] for i in range(len(errors_k3)-1)))

        # Check if final errors are below thresholds
        self.assertLess(errors_k1[-1], 1e-5)
        self.assertLess(errors_k2[-1], 1e-5)
        self.assertLess(errors_k3[-1], 1e-5)

        # Print timing comparison
        print(f"Reference K1 time: {self.k1_time:.2f}s")
        print(f"Reference K2 time: {self.k2_time:.2f}s")
        print(f"Reference K3 time: {self.k3_time:.2f}s")
        print(f"ISDF K1 times: {times_k1}")
        print(f"ISDF K2 times: {times_k2}")
        print(f"ISDF K3 times: {times_k3}")

if __name__ == '__main__':
    unittest.main()