"""Tests for JAX implementation of Transcorrelated method."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.tc import TC as TC_numpy
from pytc.autodiff.tc import TC as TC_jax
from pytc.autodiff.jastrow import SimpleJastrow

# Enable float64 support
jax.config.update("jax_enable_x64", True)

class TestTC(unittest.TestCase):
    """Test JAX implementation of TC method."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple molecule
        self.mol = gto.M(atom='He 0 0 0', basis='sto-3g')
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        # Create simple Jastrow factors for both implementations
        self.params = jnp.array([1.0])
        self.jastrow_jax = SimpleJastrow(self.params)
        
        # Create numpy version of same jastrow for comparison
        class SimpleJastrowNumpy:
            def grad(self, r1, r2):
                diff = r1[:, None, :] - r2[None, :, :]
                r12 = np.sqrt(np.sum(diff * diff, axis=-1))
                mask = r12 > 1e-10
                grad = np.where(mask[..., None], 
                              diff / np.maximum(r12[..., None], 1e-10),
                              np.zeros_like(diff))
                return grad
        self.jastrow_numpy = SimpleJastrowNumpy()
        
        # Create TC objects
        self.tc_jax = TC_jax(self.mf, self.jastrow_jax)
        self.tc_numpy = TC_numpy(self.mf, self.jastrow_numpy)
        
    def test_grid_initialization(self):
        """Test grid initialization and conversion to JAX arrays."""
        self.assertIsNotNone(self.tc_jax.grid_points)
        self.assertIsNotNone(self.tc_jax.weights)
        self.assertTrue(isinstance(self.tc_jax.grid_points, jnp.ndarray))
        self.assertTrue(isinstance(self.tc_jax.weights, jnp.ndarray))
        
    def test_basis_evaluation(self):
        """Test basis function evaluation on grid."""
        rho_jax, nabla_rho_jax = self.tc_jax._eval_basis_on_grid()
        rho_numpy, nabla_rho_numpy = self.tc_numpy._eval_basis_on_grid()
        
        np.testing.assert_allclose(
            np.asarray(rho_jax), rho_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="JAX and numpy basis evaluations don't match"
        )
        np.testing.assert_allclose(
            np.asarray(nabla_rho_jax), nabla_rho_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="JAX and numpy basis gradients don't match"
        )
        
    def test_get_2b_against_numpy(self):
        """Test two-body term calculation against numpy version."""
        # Compute two-body terms
        result_jax = self.tc_jax.get_2b()
        result_numpy = self.tc_numpy.get_2b()
        
        # Convert JAX array to numpy for comparison
        result_jax = np.asarray(result_jax)
        
        np.testing.assert_allclose(
            result_jax, result_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="JAX and numpy two-body terms don't match"
        )
    
    def test_mo_coeff_handling(self):
        """Test handling of molecular orbital coefficients."""
        # Test with explicit mo_coeff
        new_mo = self.mf.mo_coeff + 0.1
        tc_jax_new = TC_jax(self.mf, self.jastrow_jax, mo_coeff=new_mo)
        tc_numpy_new = TC_numpy(self.mf, self.jastrow_numpy, mo_coeff=new_mo)
        
        result_jax = tc_jax_new.get_2b()
        result_numpy = tc_numpy_new.get_2b()
        
        np.testing.assert_allclose(
            np.asarray(result_jax), result_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Results don't match with explicit mo_coeff"
        )

if __name__ == '__main__':
    unittest.main()
