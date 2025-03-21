import unittest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pyscf import gto, scf
import psutil
import gc

from pytc.autodiff.ansatz.det import SlaterDet, value, grad, laplacian, matrix

class TestDetJax(unittest.TestCase):
    """Tests for JAX wrappers of SlaterDet methods"""
    
    def setUp(self):
        """Create a simple H2 molecule with a SlaterDet for testing"""
        # Create a simple H2 molecule
        self.mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz', unit='angstrom')
        
        # Get RHF orbitals
        mf = scf.RHF(self.mol)
        mf.kernel()
        
        # Create SlaterDet with RHF orbitals
        self.det = SlaterDet(self.mol, mo_coeff=mf.mo_coeff)
    
    def test_value_single(self):
        """Test the JAX wrapper for value with a single walker"""
        # Create random electron positions and add batch dimension
        coords_np = np.random.rand(self.det.n_electrons, 3)
        coords_np_batched = coords_np.reshape(1, -1, 3)
        
        # Compare numpy and JAX results
        np_value = self.det.value(coords_np)
        jax_value = value(self.det, jnp.array(coords_np_batched))[0]
        
        # Check results are close
        self.assertTrue(jnp.allclose(np_value, jax_value))
        
        # Test jit compilation - mark det as static
        jax_value_jit = jax.jit(value, static_argnums=0)
        jit_value = jax_value_jit(self.det, jnp.array(coords_np_batched))[0]
        
        self.assertTrue(jnp.allclose(np_value, jit_value))
    
    def test_value_batch(self):
        """Test the JAX wrapper for value with batched walkers"""
        # Create batched random electron positions
        n_batch = 5
        coords_np = np.random.rand(n_batch, self.det.n_electrons, 3)
        
        # Compare numpy and JAX results
        np_value = self.det.value(coords_np)
        jax_value = value(self.det, jnp.array(coords_np))
        
        # Check shapes and values
        self.assertEqual(jax_value.shape, (n_batch,))
        self.assertTrue(jnp.allclose(np_value, jax_value))
    
    def test_grad_single(self):
        """Test the JAX wrapper for grad with a single walker"""
        # Create random electron positions and add batch dimension
        coords_np = np.random.rand(self.det.n_electrons, 3)
        coords_np_batched = coords_np.reshape(1, -1, 3)
        
        # Compare numpy and JAX results
        np_grad_up, np_grad_down = self.det.grad(coords_np)
        jax_result = grad(self.det, jnp.array(coords_np_batched))
        jax_grad_up, jax_grad_down = jax_result
        
        # Extract first batch element
        jax_grad_up, jax_grad_down = jax_grad_up[0], jax_grad_down[0]
        
        # Check shapes and values
        self.assertEqual(jax_grad_up.shape, (self.det.n_alpha, self.det.n_alpha, 3))
        self.assertEqual(jax_grad_down.shape, (self.det.n_beta, self.det.n_beta, 3))
        self.assertTrue(jnp.allclose(np_grad_up, jax_grad_up))
        self.assertTrue(jnp.allclose(np_grad_down, jax_grad_down))
        
        # Test with jit
        jax_grad_jit = jax.jit(grad, static_argnums=0)
        jit_grad_up, jit_grad_down = jax_grad_jit(self.det, jnp.array(coords_np_batched))
        self.assertTrue(jnp.allclose(np_grad_up, jit_grad_up[0]))
        self.assertTrue(jnp.allclose(np_grad_down, jit_grad_down[0]))
    
    def test_grad_batch(self):
        """Test the JAX wrapper for grad with batched walkers"""
        # Create batched random electron positions
        n_batch = 5
        coords_np = np.random.rand(n_batch, self.det.n_electrons, 3)
        
        # Compare numpy and JAX results
        np_grad_up, np_grad_down = self.det.grad(coords_np)
        jax_result = grad(self.det, jnp.array(coords_np))
        jax_grad_up, jax_grad_down = jax_result
        
        # Check shapes and values
        self.assertEqual(jax_grad_up.shape, (n_batch, self.det.n_alpha, self.det.n_alpha, 3))
        self.assertEqual(jax_grad_down.shape, (n_batch, self.det.n_beta, self.det.n_beta, 3))
        self.assertTrue(jnp.allclose(np_grad_up, jax_grad_up))
        self.assertTrue(jnp.allclose(np_grad_down, jax_grad_down))
    
    def test_laplacian_single(self):
        """Test the JAX wrapper for laplacian with a single walker"""
        # Create random electron positions and add batch dimension
        coords_np = np.random.rand(self.det.n_electrons, 3)
        coords_np_batched = coords_np.reshape(1, -1, 3)
        
        # Compare numpy and JAX results
        np_lap_up, np_lap_down = self.det.laplacian(coords_np)
        jax_result = laplacian(self.det, jnp.array(coords_np_batched))
        jax_lap_up, jax_lap_down = jax_result[0], jax_result[1]
        
        # Extract first batch element
        jax_lap_up, jax_lap_down = jax_lap_up[0], jax_lap_down[0]
        
        # Check shapes and values
        self.assertEqual(jax_lap_up.shape, (self.det.n_alpha, self.det.n_alpha))
        self.assertEqual(jax_lap_down.shape, (self.det.n_beta, self.det.n_beta))
        self.assertTrue(jnp.allclose(np_lap_up, jax_lap_up))
        self.assertTrue(jnp.allclose(np_lap_down, jax_lap_down))
        
        # Test with jit
        jax_lap_jit = jax.jit(laplacian, static_argnums=0)
        jit_lap = jax_lap_jit(self.det, jnp.array(coords_np_batched))
        jit_lap_up, jit_lap_down = jit_lap[0][0], jit_lap[1][0]
        self.assertTrue(jnp.allclose(np_lap_up, jit_lap_up))
        self.assertTrue(jnp.allclose(np_lap_down, jit_lap_down))
    
    def test_matrix_batch(self):
        """Test the JAX wrapper for matrix with batched walkers"""
        # Create batched random electron positions
        n_batch = 5
        coords_np = np.random.rand(n_batch, self.det.n_electrons, 3)
        
        # Compare numpy and JAX results
        np_mat_up, np_mat_down = self.det.matrix(coords_np)
        jax_result = matrix(self.det, jnp.array(coords_np))
        jax_mat_up, jax_mat_down = jax_result
        
        # Check shapes and values
        self.assertEqual(jax_mat_up.shape, (n_batch, self.det.n_alpha, self.det.n_alpha))
        self.assertEqual(jax_mat_down.shape, (n_batch, self.det.n_beta, self.det.n_beta))
        self.assertTrue(jnp.allclose(np_mat_up, jax_mat_up))
        self.assertTrue(jnp.allclose(np_mat_down, jax_mat_down))
    
    def test_jax_transformations(self):
        """Test that JAX transformations work with jit"""
        # Create random electron positions with batch dimension
        coords_np = np.random.rand(1, self.det.n_electrons, 3)
        coords_jax = jnp.array(coords_np)
        
        # Test jit compilation with static argument
        value_jit = jax.jit(value, static_argnums=0)
        value_result = value_jit(self.det, coords_jax)
        
        # Just check that we got some output
        self.assertIsNotNone(value_result)
        self.assertEqual(value_result.shape, (1,))
    
    def test_memory_leak(self):
        """Test that repeated calls to JAX wrappers don't cause memory leaks"""
        process = psutil.Process()
        
        # Force garbage collection
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Number of iterations for repeated function calls
        n_iterations = 100
        n_walkers = 10000
        
        # Test all wrapper functions
        for _ in range(n_iterations):
            # Generate new coordinates each time
            coords_np = np.random.rand(n_walkers, self.det.n_electrons, 3)
            coords_jax = jnp.array(coords_np)
            
            # Call all wrapper functions
            value(self.det, coords_jax)
            grad(self.det, coords_jax)
            laplacian(self.det, coords_jax)
            matrix(self.det, coords_jax)
            final_memory = process.memory_info().rss / 1024 / 1024  # in MB
            print(f"Iter: {_}, Memory usage during iteration: {final_memory:.2f}MB")
        
        # Force garbage collection again
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Check memory growth
        memory_growth = final_memory - initial_memory
        print(f"Memory usage: initial={initial_memory:.2f}MB, final={final_memory:.2f}MB, growth={memory_growth:.2f}MB")
        
        # Allow some reasonable growth, but not excessive
        self.assertLess(memory_growth, 50.0, "Excessive memory growth detected, possible memory leak")
    
    def test_memory_leak_jit(self):
        """Test that repeated calls to JIT-compiled JAX wrappers don't cause memory leaks"""
        # Create JIT versions of all functions
        
        process = psutil.Process()
        
        # Force garbage collection
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Number of iterations for repeated function calls
        n_iterations = 100
        n_walkers = 500000
        
        # Test all wrapper functions
        for _ in range(n_iterations):
            # Generate new coordinates each time
            coords_np = np.random.rand(n_walkers, self.det.n_electrons, 3)
            coords_jax = jnp.array(coords_np)
            value_jit = jax.jit(value, static_argnums=0)
            grad_jit = jax.jit(grad, static_argnums=0)
            laplacian_jit = jax.jit(laplacian, static_argnums=0)
            matrix_jit = jax.jit(matrix, static_argnums=0)
            
            # Call all JIT-compiled wrapper functions
            value_jit(self.det, coords_jax)
            grad_jit(self.det, coords_jax)
            laplacian_jit(self.det, coords_jax)
            matrix_jit(self.det, coords_jax)
            final_memory = process.memory_info().rss / 1024 / 1024  # in MB
            print(f"value = {value_jit(self.det, coords_jax)}")
            print(f"Iter: {_}, Memory usage during iteration: {final_memory:.2f}MB")
        
        # Force garbage collection again
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Check memory growth
        memory_growth = final_memory - initial_memory
        print(f"JIT Memory usage: initial={initial_memory:.2f}MB, final={final_memory:.2f}MB, growth={memory_growth:.2f}MB")
        
        # Allow some reasonable growth, but not excessive
        self.assertLess(memory_growth, 50.0, "Excessive memory growth detected in JIT functions, possible memory leak")

if __name__ == '__main__':
    unittest.main()
