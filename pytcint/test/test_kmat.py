"""Test for two-body matrix elements K."""

import unittest
import numpy as np
from pyscf import gto, scf, dft
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


class TestKmat(unittest.TestCase):
    """Test K matrix elements."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple H2 molecule for all tests in this class."""
        cls.mol, cls.mf = get_h2_sto3g()
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
        from pytcint.kmat import calc_K1
        k1 = calc_K1(
            self.rho_paired,
            self.nabla_rho_paired,
            self.u_gradients,  # Pre-computed gradients
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        self.assertEqual(k1.shape, (self.n_orb,)*4)
    
    def test_k2_shape(self):
        """Test if K2 (laplacian) integral has correct shape."""
        from pytcint.kmat import calc_K2
        k2 = calc_K2(
            self.rho_paired,
            self.nabla_rho_paired,
            self.u_gradients,  # Pre-computed gradients
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        self.assertEqual(k2.shape, (self.n_orb,)*4)
    
    def test_k3_shape(self):
        """Test if K3 (square) integral has correct shape."""
        from pytcint.kmat import calc_K3
        k3 = calc_K3(
            self.rho_paired,
            self.u_gradients,  # Pre-computed gradients
            self.weights
        ).reshape(self.n_orb, self.n_orb, self.n_orb, self.n_orb)
        self.assertEqual(k3.shape, (self.n_orb,)*4)
    
    def test_k2_k3_symmetry(self):
        """Test symmetry properties of K2 (laplacian) and K3 (square) integrals."""
        from pytcint.kmat import calc_K2, calc_K3
        
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
        from pytcint.kmat import calc_K1, calc_K2
        
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


if __name__ == '__main__':
    unittest.main()