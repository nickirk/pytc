"""Tests for JAX SimpleJastrow implementation."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pytc.autodiff.jastrow import SimpleJastrow

# Enable float64 support
jax.config.update("jax_enable_x64", True)

def numerical_gradient_params(jastrow, r1, r2, eps=1e-4):
    """Compute numerical gradient with respect to parameters for single points."""
    params = jastrow.params
    grad = jnp.zeros_like(params)
    
    for i in range(len(params)):
        # Forward step
        params_plus = params.at[i].add(eps)
        params_minus = params.at[i].add(-eps)
        
        j_plus = SimpleJastrow(params_plus)._compute(r1, r2, params_plus)
        j_minus = SimpleJastrow(params_minus)._compute(r1, r2, params_minus)
        
        # Central difference - no normalization needed
        grad = grad.at[i].set((j_plus - j_minus) / (2 * eps))
    
    return grad

def numerical_gradient_r1(jastrow, r1, r2, eps=1e-7):
    """Compute numerical gradient with respect to r1 for single points."""
    grad = jnp.zeros_like(r1)
    
    for j in range(3):  # x, y, z components
        # Forward step
        r1_plus = r1.at[j].add(eps)
        r1_minus = r1.at[j].add(-eps)
        
        j_plus = jastrow._compute(r1_plus, r2, jastrow.params)
        j_minus = jastrow._compute(r1_minus, r2, jastrow.params)
        
        # Central difference
        grad = grad.at[j].set((j_plus - j_minus) / (2 * eps))
    
    return grad

class TestSimpleJastrowJAX(unittest.TestCase):
    """Test cases for SimpleJastrowJAX class."""
    
    def setUp(self):
        self.params = jnp.array([1.0])
        self.jastrow = SimpleJastrow(self.params)
    
    def test_single_point_evaluation(self):
        """Test single point Jastrow evaluation."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        value = self.jastrow._compute(r1, r2, self.params)
        self.assertTrue(jnp.isfinite(value))
        self.assertEqual(value, 1.0)  # Should be param * |r1-r2| = 1.0 * 1.0
    
    def test_batch_evaluation(self):
        """Test batched Jastrow evaluation."""
        r1 = jnp.array([[0., 0., 0.], [1., 1., 1.]])  # (2, 3)
        r2 = jnp.array([[1., 0., 0.]])  # (1, 3) - single point for r2
        
        values = self.jastrow(r1, r2)
        # Should broadcast to (2, 1)
        self.assertEqual(values.shape, (2, 1))
        self.assertTrue(jnp.all(jnp.isfinite(values)))
    
    def test_param_gradient(self):
        """Test parameter gradient computation for single points."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        # Use raw gradients for comparison
        grad_analytical = self.jastrow.grad_params(r1, r2)
        grad_numerical = numerical_gradient_params(self.jastrow, r1, r2)
        
        np.testing.assert_allclose(
            grad_analytical, grad_numerical,
            rtol=1e-5, atol=1e-5,
            err_msg="Single point parameter gradients don't match"
        )
    
    def test_position_gradient(self):
        """Test position gradient computation for single points."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        # Use new method name _compute_grad_r instead of _grad_r1_single
        grad_analytical = self.jastrow.grad_r(r1, r2)
        grad_numerical = numerical_gradient_r1(self.jastrow, r1, r2)
        
        np.testing.assert_allclose(
            grad_analytical, grad_numerical,
            rtol=1e-5, atol=1e-5,
            err_msg="Single point position gradients don't match"
        )
    
    def test_batch_consistency(self):
        """Test that batched results match single point computations."""
        r1_single = jnp.array([0., 0., 0.])
        r2_single = jnp.array([1., 0., 0.])
        r1_batch = jnp.array([[0., 0., 0.]])
        r2_batch = jnp.array([[1., 0., 0.]])
        
        single_value = self.jastrow._compute(r1_single, r2_single, self.params)
        batch_value = self.jastrow(r1_batch, r2_batch)
        
        np.testing.assert_allclose(
            single_value, batch_value[0, 0],
            rtol=1e-10, atol=1e-10,
            err_msg="Batch and single point results don't match"
        )

if __name__ == '__main__':
    unittest.main()
