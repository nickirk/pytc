"""Tests for JAX implementation of Transcorrelated method."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.tc import TC as TC_numpy
from pytc.autodiff.tc import TC as TC_jax
from pytc.autodiff.jastrow import Poly

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
        self.jastrow_jax = Poly()  # No params in constructor
        
        # Create numpy version of same jastrow for comparison
        class PolyNumpy:
            """Numpy version of Poly for comparison."""
            def __init__(self, params):
                self.params = params

            def __call__(self, r1, r2):
                r12 = np.sqrt(np.sum((r1 - r2)**2, axis=-1))
                return self.params[0] * r12

            def grad(self, r1, r2):
                """Combined gradient calculation for comparison."""
                diff = r1[:, None, :] - r2[None, :, :]
                r12_sq = np.sum(diff * diff, axis=-1)
                r12 = np.sqrt(r12_sq + 1e-10)  # Add epsilon for stability
                cutoff = 1.0 / (1.0 + np.exp(-(r12 - 1e-5) * 1e6))  # Sigmoid cutoff
                grad = np.where(r12_sq[..., None] > 1e-10,
                               diff * self.params[0] * cutoff[..., None] / r12[..., None],
                               np.zeros_like(diff))
                return grad
        self.jastrow_numpy = PolyNumpy(self.params)
        
        # Create TC objects
        self.tc_jax = TC_jax(self.mf, self.jastrow_jax)  # No params in constructor
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
        # Compute two-body terms with explicit parameter passing
        result_jax = self.tc_jax.get_2b(self.params)
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
        
        result_jax = tc_jax_new.get_2b(self.params)
        result_numpy = tc_numpy_new.get_2b()
        
        np.testing.assert_allclose(
            np.asarray(result_jax), result_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Results don't match with explicit mo_coeff"
        )

    def test_two_body_terms(self):
        """Test two-body term calculation."""
        r1 = np.array([[0.0, 0.0, 0.0]])
        r2 = np.array([[0.0, 0.0, 1.0]])
        grad_jax = self.jastrow_jax.grad_r(r1, r2, self.params)[:, None, :]
        grad_numpy = self.jastrow_numpy.grad(r1, r2)
        
        np.testing.assert_allclose(
            grad_jax, grad_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Gradients don't match"
        )

if __name__ == '__main__':
    unittest.main()
