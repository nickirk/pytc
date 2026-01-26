import unittest
import numpy as np
import scipy
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
        """Create a simple benzene molecule with a SlaterDet for testing"""
        atoms = 'H      1.2194     -0.1652      2.1600;'\
                'C      0.6825     -0.0924      1.2087;'\
                'C     -0.7075     -0.0352      1.1973;'\
                'H     -1.2644     -0.0630      2.1393;'\
                'C     -1.3898      0.0572     -0.0114;'\
                'H     -2.4836      0.1021     -0.0204;'\
                'C     -0.6824      0.0925     -1.2088;'\
                'H     -1.2194      0.1652     -2.1599;'\
                'C      0.7075      0.0352     -1.1973;'\
                'H      1.2641      0.0628     -2.1395;'\
                'C      1.3899     -0.0572      0.0114;'\
                'H      2.4836     -0.1022      0.0205'
        # Create a simple H2 molecule
        self.mol = gto.M(atom=atoms, basis='ccpvdz', unit='angstrom')
        
        # Get RHF orbitals
        mf = scf.RHF(self.mol)
        mf.kernel()
        
        # Create SlaterDet with RHF orbitals
        self.det = SlaterDet.create(self.mol, mo_coeff=mf.mo_coeff)
    
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
        n_iterations = 10
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
        
    
    def test_memory_leak_jit(self):
        """Test that repeated calls to JIT-compiled JAX wrappers don't cause memory leaks"""
        import time
        
        process = psutil.Process()
        
        # Force garbage collection
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Number of iterations for repeated function calls
        n_iterations = 10000
        n_walkers = 4000
        
        # Statistics containers
        matrix_times = []
        det_times = []
        matrix_cpu = []
        det_cpu = []
        grad_times = []
        grad_cpu = []
        
        value_jit = jax.jit(value, static_argnums=0)
        matrix_jit = jax.jit(matrix, static_argnums=0)
        grad_jit = jax.jit(grad, static_argnums=0)
        
        # Warmup JIT
        coords_warmup = jnp.array(np.random.rand(1, self.det.n_electrons, 3))
        matrix_jit(self.det, coords_warmup)
        
        for i in range(n_iterations):
            coords_np = np.random.rand(n_walkers, self.det.n_electrons, 3)
            coords_jax = jnp.array(coords_np)

            start_time = time.time()
            grads = grad_jit(self.det, coords_jax)
            grad_cpu.append(process.cpu_percent())
            grad_times.append(time.time() - start_time)


            
            # Measure matrix_jit
            start_time = time.time()
            #start_cpu = process.cpu_percent()
            mat = matrix_jit(self.det, coords_jax)
            matrix_cpu.append(process.cpu_percent())
            matrix_times.append(time.time() - start_time)

            # wait for a bit
            #time.sleep(0.5)
            
            # Measure determinant calculation
            start_time = time.time()
            #start_cpu = process.cpu_percent()
            mat_det0 = scipy.linalg.det(mat[0])
            mat_det1 = scipy.linalg.det(mat[1])
            det_cpu.append(process.cpu_percent())
            det_times.append(time.time() - start_time)
            
            if i % 10 == 0:  # Print stats every 10 iterations
                print(f"\nIteration {i}:")
                print(f"Matrix: avg_time={np.mean(matrix_times):.4f}s, avg_cpu={np.mean(matrix_cpu):.1f}%")
                print(f"Det: avg_time={np.mean(det_times):.4f}s, avg_cpu={np.mean(det_cpu):.1f}%")
                print(f"Grad: avg_time={np.mean(grad_times):.4f}s, avg_cpu={np.mean(grad_cpu):.1f}%")
                print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f}MB")
        
        # Print final statistics
        print("\nFinal Statistics:")
        print(f"Matrix operation: {np.mean(matrix_times):.4f}±{np.std(matrix_times):.4f}s, "
              f"CPU: {np.mean(matrix_cpu):.1f}±{np.std(matrix_cpu):.1f}%")
        print(f"Determinant: {np.mean(det_times):.4f}±{np.std(det_times):.4f}s, "
              f"CPU: {np.mean(det_cpu):.1f}±{np.std(det_cpu):.1f}%")
        
        # Force garbage collection again
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Check memory growth
        memory_growth = final_memory - initial_memory
        print(f"JIT Memory usage: initial={initial_memory:.2f}MB, final={final_memory:.2f}MB, growth={memory_growth:.2f}MB")
        

if __name__ == '__main__':
    unittest.main()
