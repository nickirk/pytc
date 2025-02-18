"""Tests for JAX implementation of Extended Transcorrelated methods."""

import unittest
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from pyscf import gto, scf

from pytc.xtc import XTC as XTC_numpy
from pytc.autodiff.xtc import XTC as XTC_jax
from pytc.autodiff.jastrow import Poly

# Enable float64 support
jax.config.update("jax_enable_x64", True)

def get_be_ccpvdz():
    """Return a Be atom with cc-pVDZ basis for testing."""
    mol = gto.M(atom='Be 0 0 0', basis='ccpvdz', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

class TestXTC(unittest.TestCase):
    """Test JAX implementation of Extended Transcorrelated methods."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case using Be atom."""
        # Get mean-field data
        _, cls.mf = get_be_ccpvdz()
        
        # Create simple Jastrow factors for both implementations
        cls.params = jnp.array([1.4])
        cls.jastrow_jax = Poly(cls.params)
        
        # Create numpy version of same jastrow
        class PolyNumpy:
            def grad(self, r1, r2):
                diff = r1[:, None, :] - r2[None, :, :]
                r12 = np.sqrt(np.sum(diff * diff, axis=-1))
                mask = r12 > 1e-10
                grad = np.where(mask[..., None], 
                              cls.params[0]*diff / np.maximum(r12[..., None], 1e-10),
                              np.zeros_like(diff))
                return grad
        cls.jastrow_numpy = PolyNumpy()
        
        # Initialize XTC calculators with low grid level for testing
        cls.xtc_jax = XTC_jax(cls.mf, cls.jastrow_jax, grid_lvl=1)
        cls.xtc_numpy = XTC_numpy(cls.mf, cls.jastrow_numpy, grid_lvl=1)
        
    def test_grid_initialization(self):
        """Test grid initialization and conversion to JAX arrays."""
        np.testing.assert_allclose(
            np.asarray(self.xtc_jax.grid_points),
            self.xtc_numpy.grid_points,
            rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(self.xtc_jax.weights),
            self.xtc_numpy.weights,
            rtol=1e-7, atol=1e-7
        )

    def test_basis_evaluation(self):
        """Test basis function evaluation on grid."""
        rho_jax, _ = self.xtc_jax._eval_basis_on_grid()
        rho_numpy, _ = self.xtc_numpy._eval_basis_on_grid()
        
        np.testing.assert_allclose(
            np.asarray(rho_jax),
            rho_numpy,
            rtol=1e-5, atol=1e-5
        )

    def test_v_vector(self):
        """Test v_vector calculation."""
        # Get orbital values on grid
        mo_values_jax, _ = self.xtc_jax._eval_basis_on_grid()
        mo_values_numpy, _ = self.xtc_numpy._eval_basis_on_grid()
        
        # Prepare paired indices
        rho_paired_jax = jnp.einsum('in,jn->ijn', 
                                   mo_values_jax, 
                                   mo_values_jax).reshape(-1, len(self.xtc_jax.weights))
        rho_paired_numpy = np.einsum('in,jn->ijn', 
                                    mo_values_numpy, 
                                    mo_values_numpy).reshape(-1, len(self.xtc_numpy.weights))
        
        # Calculate v_vector
        v_vector_jax = self.xtc_jax.calc_v_vector(rho_paired_jax)
        v_vector_numpy = self.xtc_numpy._calc_v_vector(rho_paired_numpy)
        
        np.testing.assert_allclose(
            np.asarray(v_vector_jax),
            v_vector_numpy,
            rtol=1e-5, atol=1e-5
        )
        
        # Update gradient usage to match new interface
        @partial(jax.vmap, in_axes=(0, None))
        def get_grads(r1, r2):
            return self.xtc.jastrow_factor.grad_r(r1[None], r2[None])[0]

    def test_delta_U(self):
        """Test delta_U calculation."""
        # First test delta_U matrices
        dm1_jax = self.xtc_jax._get_mf_dm()
        dm1_numpy = self.xtc_numpy._get_mf_dm()
        
        delta_U_jax = self.xtc_jax.get_delta_U(dm1_jax)
        delta_U_numpy = self.xtc_numpy._calc_delta_U()
        
        np.testing.assert_allclose(
            np.asarray(delta_U_jax),
            delta_U_numpy,
            rtol=1e-5, atol=1e-5
        )
        
        # Test symmetry property
        np.testing.assert_allclose(
            np.asarray(delta_U_jax),
            np.asarray(delta_U_jax.transpose(2,3,0,1)),
            rtol=1e-7, atol=1e-7
        )
        
        # Test gradient calculations with defined points
        test_r1 = np.array([[0.0, 0.0, 0.0]])
        test_r2 = np.array([[0.0, 0.0, 1.0]])
        
        grad_jax = self.xtc_jax.jastrow_factor.grad_r(test_r1, test_r2)
        grad_numpy = self.jastrow_numpy.grad(test_r1, test_r2)
        
        np.testing.assert_allclose(
            np.asarray(grad_jax[:, None, :]),  # Add middle dimension to match numpy shape
            grad_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Gradients don't match"
        )

    def test_delta_h(self):
        """Test delta_h calculation."""
        delta_h_jax = self.xtc_jax.get_delta_h()
        delta_h_numpy = self.xtc_numpy._calc_delta_h()
        
        np.testing.assert_allclose(
            np.asarray(delta_h_jax),
            delta_h_numpy,
            rtol=1e-5, atol=1e-5
        )
        
        # Test hermiticity
        np.testing.assert_allclose(
            np.asarray(delta_h_jax),
            np.asarray(delta_h_jax.T.conj()),
            rtol=1e-7, atol=1e-7
        )

    def test_one_body(self):
        """Test one-body operator calculation."""
        h1e_jax = self.xtc_jax.get_1b()
        h1e_numpy = self.xtc_numpy.get_1b()
        
        np.testing.assert_allclose(
            np.asarray(h1e_jax),
            h1e_numpy,
            rtol=1e-5, atol=1e-5
        )

    def test_two_body(self):
        """Test two-body term calculation."""
        v2e_jax = self.xtc_jax.get_2b()
        v2e_numpy = self.xtc_numpy.get_2b()
        
        np.testing.assert_allclose(
            np.asarray(v2e_jax),
            v2e_numpy,
            rtol=1e-5, atol=1e-5
        )

if __name__ == '__main__':
    unittest.main()
