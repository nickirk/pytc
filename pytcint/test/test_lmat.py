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


class TestLmat(unittest.TestCase):
    """Test L matrix elements."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple H2 molecule for all tests in this class."""
        cls.mol, cls.mf = get_h2_sto3g()
        cls.n_orb = cls.mf.mo_coeff.shape[1]
        cls.jastrow = SimpleJastrow([0.5])  # alpha = 0.5
        
        # Set up grid points for testing
        from pyscf import dft
        grids = dft.gen_grid.Grids(cls.mol)
        grids.level = 1  # Use coarse grid for testing
        grids.build()
        cls.grid_points = grids.coords  # Shape: (N_grid, 3)
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
    
    def test_v_vector_shape(self):
        """Test if V vector computation returns correct shape."""
        v_vector = lmat.calc_v_vector(
            self.rho_paired, 
            self.u_gradients,
            self.weights
        )
        
        expected_shape = (self.n_orb**2, len(self.weights), 3)
        self.assertEqual(v_vector.shape, expected_shape)
    
    def test_l_matrix_shape(self):
        """Test if L matrix computation returns correct shape."""
        # Get Jastrow gradients and compute V vectors
        v_bra = lmat.calc_v_vector(self.rho_paired, self.u_gradients, self.weights)
        
        # Compute L matrix
        l_mat = lmat.calc_L(
            self.rho_paired,
            v_bra,
            self.weights
        )
        
        expected_shape = (self.n_orb**2, self.n_orb**2, self.n_orb**2)
        self.assertEqual(l_mat.shape, expected_shape)
    
    def test_l_matrix_symmetry(self):
        """Test symmetry properties of L matrix elements."""
        # Get Jastrow gradients and compute V vectors
        v_bra = lmat.calc_v_vector(self.rho_paired, self.u_gradients, self.weights)
        
        # Get symmetric L matrix
        l_mat = lmat.calc_L_symmetric(
            self.rho_paired,
            v_bra,
            self.weights
        )
        
        # Test permutation symmetry
        l_tensor = l_mat.reshape((self.n_orb**2,)*3)
        for p in range(self.n_orb):
            for q in range(self.n_orb):
                for r in range(self.n_orb):
                    val1 = l_tensor[p,q,r]
                    val2 = l_tensor[p,r,q]  # Permute second and third electron
                    val3 = l_tensor[r,q,p]  # Cyclic permutation
                    np.testing.assert_almost_equal(val1, val2)
                    np.testing.assert_almost_equal(val1, val3)
    
    def test_calc_v_vector(self):
        """Test the calc_v_vector function for correct output."""
        # Prepare test data with correct shapes
        n_grid = 2
        rho_paired = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Shape: (3, 2)
        u_gradients = np.array([  # Shape: (2, 2, 3)
            [[1, 0, 1], [0, 1, 1]],
            [[2, 1, 2], [1, 2, 2]]
        ])
        weights = np.array([1.0, 1.0])
        
        v_vector = lmat.calc_v_vector(rho_paired, u_gradients, weights)
        
        # Test shape and contents
        self.assertEqual(v_vector.shape, (3, 2, 3))
        # ...rest of assertions remain the same...


if __name__ == '__main__':
    unittest.main()
