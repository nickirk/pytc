"""Tests for JAX implementation of kinetic matrix elements."""

import unittest
import numpy as np
import jax
# Enable float64 support
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pytc.kmat import calc_K1 as calc_K1_numpy, calc_K3 as calc_K3_numpy
from pytc.kmat import calc_K1, calc_K3
from pytc.jastrow import Poly


class TestKmat(unittest.TestCase):
    """Test JAX implementation of K matrix elements."""
    
    def setUp(self):
        """Set up test fixtures."""
        rng = np.random.RandomState(42)
        
        # Create multiple test systems with different sizes
        self.test_configs = [
            # Small system
            {'Nb': 2, 'N_grid': 3, 'name': 'small'},
            # Medium system
            {'Nb': 4, 'N_grid': 10, 'name': 'medium'},
            # Larger system
            {'Nb': 6, 'N_grid': 20, 'name': 'large'}
        ]
        
        for config in self.test_configs:
            Nb, N_grid = config['Nb'], config['N_grid']
            # Create test data for each configuration
            config['grid_points'] = rng.randn(N_grid, 3)
            config['weights'] = rng.rand(N_grid)  # Random weights
            
            # Generate orbitals (phi) and gradients (grad_phi)
            config['phi'] = rng.randn(Nb, N_grid)
            config['grad_phi'] = rng.randn(Nb, N_grid, 3)
            
            # Compute paired densities for NumPy reference (which expects pairs)
            # phi_paired_ij = phi_i * phi_j
            config['phi_paired'] = np.einsum('in,jn->ijn', config['phi'], config['phi']).reshape(Nb * Nb, N_grid)
            
            # grad_phi_paired_ij = grad_phi_i * phi_j
            # Note: This matches JAX calc_K1 logic (grad on first index)
            config['grad_phi_paired'] = np.einsum('ind,jn->ijnd', config['grad_phi'], config['phi']).reshape(Nb * Nb, N_grid, 3)
        
        # Create Jastrow factors
        self.params = jnp.array([1.0])
        self.jastrow_jax = Poly()
        
        class PolyNumpy:
            """NumPy implementation to match original implementation."""
            def __init__(self, params):
                self.params = params
                
            def grad(self, r1, r2):
                """Numpy gradient computation handling both single and batched inputs."""
                # Handle single point inputs
                if r1.ndim == 1:
                    r1 = r1[None, :]
                if r2.ndim == 1:
                    r2 = r2[None, :]
                    
                diff = r1[:, None, :] - r2[None, :, :]
                r12 = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)  # Match epsilon
                grad = diff / r12[..., None]
                grad = grad * self.params[0]
                
                # Return single point result without batch dimensions
                if grad.shape[0] == 1 and grad.shape[1] == 1:
                    return grad[0, 0]
                return grad
                
        self.jastrow_numpy = PolyNumpy(self.params)
    
    def test_K1_shapes(self):
        """Test K1 output shapes for different input sizes."""
        for config in self.test_configs:
            with self.subTest(size=config['name']):
                Nb = config['Nb']
                result = calc_K1(
                    jnp.asarray(config['phi']),
                    jnp.asarray(config['grad_phi']),
                    self.jastrow_jax,
                    self.params,  # Add params argument
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights'])
                )
                self.assertEqual(result.shape, (Nb * Nb, Nb * Nb))
    
    def test_K1_against_numpy_all_sizes(self):
        """Compare JAX K1 implementation against numpy for different sizes."""
        for config in self.test_configs:
            with self.subTest(size=config['name']):
                k1_jax_raw = calc_K1(
                    jnp.asarray(config['phi']),
                    jnp.asarray(config['grad_phi']),
                    self.jastrow_jax,
                    self.params,  # Add params argument
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights'])
                )
                # JAX returns (Nb, Nb, Nb, Nb) flattened to (Nb^2, Nb^2)
                # This matches NumPy (Nb^2, Nb^2)
                k1_jax = k1_jax_raw
                
                k1_numpy = calc_K1_numpy(
                    config['phi_paired'],
                    config['grad_phi_paired'],
                    self.jastrow_numpy,
                    config['grid_points'],
                    config['weights']
                )
                
                np.testing.assert_allclose(
                    np.asarray(k1_jax), k1_numpy,
                    rtol=1e-5, atol=1e-5,
                    err_msg=f"JAX and numpy K1 don't match for {config['name']} system"
                )
    
    def test_K3_shapes(self):
        """Test K3 output shapes for different input sizes."""
        for config in self.test_configs:
            with self.subTest(size=config['name']):
                Nb = config['Nb']
                result = calc_K3(
                    jnp.asarray(config['phi']),
                    self.jastrow_jax,
                    self.params,  # Add params argument
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights'])
                )
                self.assertEqual(result.shape, (Nb * Nb, Nb * Nb))
    
    def test_K3_against_numpy_all_sizes(self):
        """Compare JAX K3 implementation against numpy for different sizes."""
        for config in self.test_configs:
            with self.subTest(size=config['name']):
                k3_jax = calc_K3(
                    jnp.asarray(config['phi']),
                    self.jastrow_jax,
                    self.params,  # Add params argument
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights'])
                )
                
                k3_numpy = calc_K3_numpy(
                    config['phi_paired'],
                    self.jastrow_numpy,
                    config['grid_points'],
                    config['weights']
                )
                
                np.testing.assert_allclose(
                    np.asarray(k3_jax), k3_numpy,
                    rtol=1e-5, atol=1e-5,
                    err_msg=f"JAX and numpy K3 don't match for {config['name']} system"
                )
    
    def test_batch_size_handling(self):
        """Test different batch sizes produce same results."""
        config = self.test_configs[-1]  # Use largest system
        batch_sizes = [1, 5, 10, 20]
        
        # Get reference result with default batch size
        ref_k1 = calc_K1(
            jnp.asarray(config['phi']),
            jnp.asarray(config['grad_phi']),
            self.jastrow_jax,
            self.params,  # Add params argument
            jnp.asarray(config['grid_points']),
            jnp.asarray(config['weights'])
        )
        
        ref_k3 = calc_K3(
            jnp.asarray(config['phi']),
            self.jastrow_jax,
            self.params,  # Add params argument
            jnp.asarray(config['grid_points']),
            jnp.asarray(config['weights'])
        )
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Test K1
                k1 = calc_K1(
                    jnp.asarray(config['phi']),
                    jnp.asarray(config['grad_phi']),
                    self.jastrow_jax,
                    self.params,  # Add params argument
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights']),
                    batch_size=batch_size
                )
                np.testing.assert_allclose(k1, ref_k1, rtol=1e-5, atol=1e-5)
                
                # Test K3
                k3 = calc_K3(
                    jnp.asarray(config['phi']),
                    self.jastrow_jax,
                    self.params,  # Add params argument
                    jnp.asarray(config['grid_points']),
                    jnp.asarray(config['weights']),
                    batch_size=batch_size
                )
                np.testing.assert_allclose(k3, ref_k3, rtol=1e-5, atol=1e-5)
    
    def test_single_point_gradient(self):
        """Test single point gradient computation matches between JAX and NumPy."""
        r1 = np.array([0., 0., 0.])
        r2 = np.array([1., 0., 0.])
        
        grad_jax = self.jastrow_jax.grad_r(r1, r2, self.params)
        grad_numpy = self.jastrow_numpy.grad(r1, r2)
        
        np.testing.assert_allclose(
            np.asarray(grad_jax), grad_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Single point gradients don't match"
        )



if __name__ == '__main__':
    unittest.main()

