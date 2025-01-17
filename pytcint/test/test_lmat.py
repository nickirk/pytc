"""Test for three-body matrix elements L."""

import unittest
import numpy as np
from pyscf import gto, scf
from pytcint.jastrow import Jastrow
from pytcint import lmat


def get_h2_sto3g():
    """Return a simple H2 molecule with STO-3G basis for testing."""
    mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


class SimpleJastrow(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    def jastrow_function(self, delta_r):
        return np.exp(-self.parameters[0] * np.linalg.norm(delta_r, axis=-1))
    
    def jastrow_gradient(self, delta_r):
        norm = np.linalg.norm(delta_r, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)  # Avoid division by zero
        return -self.parameters[0] * delta_r / norm * self.jastrow_function(delta_r)[..., np.newaxis]


class TestLmat(unittest.TestCase):
    """Test L matrix elements."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple H2 molecule for all tests in this class."""
        cls.mol, cls.mf = get_h2_sto3g()
        cls.n_orb = cls.mf.mo_coeff.shape[1]
        cls.jastrow = SimpleJastrow([0.5])  # alpha = 0.5
        
        # Set up grid points for testing
        from pyscf.dft import gen_grid
        grids = gen_grid.Grids(cls.mol)
        grids.level = 1  # Use coarse grid for testing
        grids.build()
        cls.grid_points = grids.coords
        cls.weights = grids.weights
        
        # Prepare basis functions on grid
        from pyscf.dft import numint
        ao = numint.eval_ao(cls.mol, cls.grid_points, deriv=1)
        cls.rho = np.dot(ao[0], cls.mf.mo_coeff)  # Shape: (N_grid, N_orb)
        cls.nabla_rho = np.dot(ao[1:4].transpose(1,0,2), cls.mf.mo_coeff)  # Shape: (N_grid, 3, N_orb)
    
    def test_v_vector_shape(self):
        """Test if V vector computation returns correct shape."""
        # Prepare paired indices
        rho_qt = np.einsum('ni,nj->ij', self.rho, self.rho).reshape(-1, len(self.weights))
        
        v_vector = lmat.compute_v_vector(
            rho_qt, 
            self.grid_points, 
            self.weights, 
            self.jastrow
        )
        
        self.assertEqual(v_vector.shape, (self.n_orb*self.n_orb, len(self.weights), 3))
    
    def test_l_matrix_shape(self):
        """Test if L matrix computation returns correct shape."""
        # Prepare paired indices
        rho_qt = np.einsum('ni,nj->ij', self.rho, self.rho).reshape(-1, len(self.weights))
        rho_ru = np.einsum('ni,nj->ij', self.rho, self.rho).reshape(-1, len(self.weights))
        
        # Compute V vectors
        v_qt = lmat.compute_v_vector(rho_qt, self.grid_points, self.weights, self.jastrow)
        v_ru = lmat.compute_v_vector(rho_ru, self.grid_points, self.weights, self.jastrow)
        
        # Compute L matrix
        l_mat = lmat.compute_l_matrix(
            self.rho, 
            v_qt, 
            v_ru, 
            self.rho, 
            self.grid_points, 
            self.weights
        )
        
        expected_shape = (self.n_orb,)*6  # (Nb, Nb, Nb, Nb, Nb, Nb)
        self.assertEqual(l_mat.shape, expected_shape)
    
    def test_l_matrix_symmetry(self):
        """Test symmetry properties of L matrix elements."""
        # Get full L matrix
        l_mat = lmat.get_3b(
            self.mol, 
            self.mf.mo_coeff, 
            self.grid_points, 
            self.weights, 
            self.jastrow
        )
        
        # Reshape to 6-index tensor
        l_mat = l_mat.reshape((self.n_orb,)*6)
        
        # Test L^{pqr123}_{stu} = L^{pqr312}_{stu} symmetry
        for p in range(2):
            for q in range(2):
                for r in range(2):
                    for s in range(2):
                        for t in range(2):
                            for u in range(2):
                                # Original
                                val1 = l_mat[p,q,r,s,t,u]
                                # Permuted (312)
                                val2 = l_mat[p,r,q,s,u,t]
                                self.assertAlmostEqual(
                                    val1, 
                                    val2, 
                                    places=10,
                                    msg=f"Symmetry failed for [{p},{q},{r},{s},{t},{u}]"
                                )
    
    def test_l_matrix_hermiticity(self):
        """Test Hermiticity of L matrix elements."""
        l_mat = lmat.get_3b(
            self.mol, 
            self.mf.mo_coeff, 
            self.grid_points, 
            self.weights, 
            self.jastrow
        )
        
        # Reshape to 6-index tensor
        l_mat = l_mat.reshape((self.n_orb,)*6)
        
        # Test L^{pqr}_{stu} = L^{stu}_{pqr}*
        for p in range(2):
            for q in range(2):
                for r in range(2):
                    for s in range(2):
                        for t in range(2):
                            for u in range(2):
                                self.assertAlmostEqual(
                                    l_mat[p,q,r,s,t,u], 
                                    l_mat[s,t,u,p,q,r].conj(), 
                                    places=10,
                                    msg=f"Hermiticity failed for [{p},{q},{r},{s},{t},{u}]"
                                )


if __name__ == '__main__':
    unittest.main()
