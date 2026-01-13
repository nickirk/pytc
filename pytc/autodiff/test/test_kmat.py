"""Tests for JAX implementation of kinetic matrix elements."""

import unittest
import numpy as np
import jax
# Enable float64 support
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pytc.kmat import calc_K1 as calc_K1_numpy, calc_K3 as calc_K3_numpy
from pytc.autodiff.kmat import calc_K1, calc_K3
from pytc.autodiff.jastrow import Poly


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

def get_be_ccpvdz():
    """Return a Be atom with cc-pVDZ basis for testing."""
    from pyscf import gto, scf, dft
    mol = gto.M(atom='Be 0 0 0;  ', basis='ccpvdz', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


class TestISDF(unittest.TestCase):
    """Test ISDF implementation of K matrices against numpy reference."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a Be atom for all ISDF tests."""
        from pyscf import dft
        from pytc.jastrow import REXP as REXP_NUMPY
        from pytc.autodiff.jastrow import REXP
        
        cls.mol, cls.mf = get_be_ccpvdz()
        cls.n_orb = cls.mf.mo_coeff.shape[1]
        
        # Set up Jastrow factors
        cls.params = {'alpha': jnp.array([0.5])}  # alpha = 0.5
        cls.jastrow_jax = REXP()
        cls.jastrow_numpy = REXP_NUMPY([0.5])
        
        # Set up grid points for testing
        grids = dft.gen_grid.Grids(cls.mol)
        grids.level = 1  # Use coarse grid for testing
        grids.build()
        cls.grid_points = grids.coords
        cls.weights = grids.weights
        
        # Prepare basis functions on grid
        ao = dft.numint.eval_ao(cls.mol, cls.grid_points, deriv=1)
        cls.phi = np.dot(ao[0], cls.mf.mo_coeff).T  # Shape: (N_orb, N_grid)
        cls.grad_phi = np.dot(ao[1:4].transpose(1,0,2), 
                              cls.mf.mo_coeff).transpose(2,0,1)  # Shape: (N_orb, N_grid, 3)
        
        # Prepare paired indices for testing
        cls.phi_paired = np.einsum('in,jn->ijn', 
                                  cls.phi, 
                                  cls.phi).reshape(-1, len(cls.weights))
        
        cls.grad_phi_paired = np.einsum('ind,jn->ijnd', 
                                        cls.grad_phi, 
                                        cls.phi).reshape(-1, len(cls.weights), 3)
    
    def test_isdf_shapes(self):
        """Test ISDF output shapes for different input sizes."""
        from pytc.df import isdf_decompose_multi
        from pytc.autodiff.kmat import calc_K1_isdf, calc_K2_isdf, calc_K3_isdf
        
        # Test with a moderate rank
        rank = len(self.weights) // 100
        C_phi, xi_phi, C_grad, xi_grad, fused_pivots = isdf_decompose_multi(
            self.phi_paired, 
            self.grad_phi_paired,
            rank, rank
        )
        
        # Test K1_isdf
        result_k1 = calc_K1_isdf(
            jnp.asarray(C_phi), jnp.asarray(xi_phi), 
            jnp.asarray(C_grad), jnp.asarray(xi_grad),
            self.jastrow_jax, self.params,
            jnp.asarray(self.grid_points), jnp.asarray(self.weights)
        )
        self.assertEqual(result_k1.shape, (self.n_orb, self.n_orb, self.n_orb, self.n_orb))
        
        # Test K2_isdf
        result_k2 = calc_K2_isdf(
            jnp.asarray(C_phi), jnp.asarray(xi_phi), 
            jnp.asarray(C_grad), jnp.asarray(xi_grad),
            self.jastrow_jax, self.params,
            jnp.asarray(self.grid_points), jnp.asarray(self.weights)
        )
        self.assertEqual(result_k2.shape, (self.n_orb, self.n_orb, self.n_orb, self.n_orb))
        
        # Test K3_isdf
        result_k3 = calc_K3_isdf(
            jnp.asarray(C_phi), jnp.asarray(xi_phi),
            self.jastrow_jax, self.params,
            jnp.asarray(self.grid_points), jnp.asarray(self.weights)
        )
        self.assertEqual(result_k3.shape, (self.n_orb, self.n_orb, self.n_orb, self.n_orb))
    
    def test_isdf_against_numpy(self):
        """Compare JAX ISDF implementation against numpy ISDF."""
        from pytc.df import isdf_decompose_multi
        from pytc.kmat import calc_K1_isdf as calc_K1_isdf_numpy
        from pytc.kmat import calc_K2_isdf as calc_K2_isdf_numpy
        from pytc.kmat import calc_K3_isdf as calc_K3_isdf_numpy
        from pytc.autodiff.kmat import calc_K1_isdf, calc_K2_isdf, calc_K3_isdf
        
        # Test with a moderate rank
        rank = len(self.weights) // 100
        C_phi, xi_phi, C_grad, xi_grad, fused_pivots = isdf_decompose_multi(
            self.phi_paired, 
            self.grad_phi_paired,
            rank, rank
        )
        
        # Test K1_isdf
        k1_jax = calc_K1_isdf(
            jnp.asarray(C_phi), jnp.asarray(xi_phi), 
            jnp.asarray(C_grad), jnp.asarray(xi_grad),
            self.jastrow_jax, self.params,
            jnp.asarray(self.grid_points), jnp.asarray(self.weights)
        )
        
        k1_numpy = calc_K1_isdf_numpy(
            C_phi, xi_phi, C_grad, xi_grad,
            self.jastrow_numpy, self.grid_points, self.weights
        )
        
        np.testing.assert_allclose(
            np.asarray(k1_jax).reshape(-1, self.n_orb**2), k1_numpy,
            atol=1e-6,
            err_msg="JAX and numpy K1_isdf don't match"
        )
        
        # Test K2_isdf
        k2_jax = calc_K2_isdf(
            jnp.asarray(C_phi), jnp.asarray(xi_phi), 
            jnp.asarray(C_grad), jnp.asarray(xi_grad),
            self.jastrow_jax, self.params,
            jnp.asarray(self.grid_points), jnp.asarray(self.weights)
        )
        
        k2_numpy = calc_K2_isdf_numpy(
            C_phi, xi_phi, C_grad, xi_grad,
            self.jastrow_numpy, self.grid_points, self.weights
        )
        
        np.testing.assert_allclose(
            np.asarray(k2_jax).reshape(-1, self.n_orb**2), k2_numpy,
            atol=1e-6,
            err_msg="JAX and numpy K2_isdf don't match"
        )
        
        # Test K3_isdf
        k3_jax = calc_K3_isdf(
            jnp.asarray(C_phi), jnp.asarray(xi_phi),
            self.jastrow_jax, self.params,
            jnp.asarray(self.grid_points), jnp.asarray(self.weights)
        )
        
        k3_numpy = calc_K3_isdf_numpy(
            C_phi, xi_phi,
            self.jastrow_numpy, self.grid_points, self.weights
        )
        
        np.testing.assert_allclose(
            np.asarray(k3_jax).reshape(-1, self.n_orb**2), k3_numpy,
            atol=1e-6,
            err_msg="JAX and numpy K3_isdf don't match"
        )
    
    def test_isdf_convergence(self):
        """Test if ISDF functions converge with increasing rank."""
        from pytc.df import isdf_decompose_multi
        from pytc.kmat import calc_K1_isdf as calc_K1_isdf_numpy
        from pytc.kmat import calc_K2_isdf as calc_K2_isdf_numpy
        from pytc.kmat import calc_K3_isdf as calc_K3_isdf_numpy
        from pytc.autodiff.kmat import calc_K1_isdf, calc_K2_isdf, calc_K3_isdf
        
        # Test different ranks as fractions of grid points
        ranks = [len(self.weights) // n for n in [100, 50, 10]]
        errors_k1, errors_k2, errors_k3 = [], [], []
        
        for rank in ranks:
            # Perform ISDF decomposition
            C_phi, xi_phi, C_grad, xi_grad, fused_pivots = isdf_decompose_multi(
                self.phi_paired, 
                self.grad_phi_paired,
                rank, rank
            )
            
            # Compute JAX ISDF functions
            k1_jax = calc_K1_isdf(
                jnp.asarray(C_phi), jnp.asarray(xi_phi), 
                jnp.asarray(C_grad), jnp.asarray(xi_grad),
                self.jastrow_jax, self.params,
                jnp.asarray(self.grid_points), jnp.asarray(self.weights)
            )
            
            k2_jax = calc_K2_isdf(
                jnp.asarray(C_phi), jnp.asarray(xi_phi), 
                jnp.asarray(C_grad), jnp.asarray(xi_grad),
                self.jastrow_jax, self.params,
                jnp.asarray(self.grid_points), jnp.asarray(self.weights)
            )
            
            k3_jax = calc_K3_isdf(
                jnp.asarray(C_phi), jnp.asarray(xi_phi),
                self.jastrow_jax, self.params,
                jnp.asarray(self.grid_points), jnp.asarray(self.weights)
            )
            
            # Compute numpy ISDF functions as reference
            k1_numpy = calc_K1_isdf_numpy(
                C_phi, xi_phi, C_grad, xi_grad,
                self.jastrow_numpy, self.grid_points, self.weights
            )
            
            k2_numpy = calc_K2_isdf_numpy(
                C_phi, xi_phi, C_grad, xi_grad,
                self.jastrow_numpy, self.grid_points, self.weights
            )
            
            k3_numpy = calc_K3_isdf_numpy(
                C_phi, xi_phi,
                self.jastrow_numpy, self.grid_points, self.weights
            )
            
            # Calculate abs errors
            error_k1 = np.linalg.norm(np.asarray(k1_jax).reshape(-1, self.n_orb**2) - k1_numpy)
            error_k2 = np.linalg.norm(np.asarray(k2_jax).reshape(-1, self.n_orb**2) - k2_numpy)
            error_k3 = np.linalg.norm(np.asarray(k3_jax).reshape(-1, self.n_orb**2) - k3_numpy)
            
            errors_k1.append(error_k1)
            errors_k2.append(error_k2)
            errors_k3.append(error_k3)
            
            print(f"ISDF rank {len(fused_pivots)}: K_err=({error_k1:.1e},{error_k2:.1e},{error_k3:.1e})")
            
            # Check errors are within tolerance
            self.assertLess(error_k1, 1e-6, f"K1 error {error_k1} exceeds tolerance for rank {rank}")
            self.assertLess(error_k2, 1e-6, f"K2 error {error_k2} exceeds tolerance for rank {rank}")
            self.assertLess(error_k3, 1e-6, f"K3 error {error_k3} exceeds tolerance for rank {rank}")
    
    def test_isdf_batch_sizes(self):
        """Test different batch sizes produce same results."""
        from pytc.df import isdf_decompose_multi
        from pytc.autodiff.kmat import calc_K1_isdf, calc_K2_isdf, calc_K3_isdf
        
        # Use moderate rank
        rank = len(self.weights) // 10
        C_phi, xi_phi, C_grad, xi_grad, fused_pivots = isdf_decompose_multi(
            self.phi_paired, 
            self.grad_phi_paired,
            rank, rank
        )
        
        batch_sizes = [100, 500, 1000]
        
        # Get reference result with default batch size
        ref_k1 = calc_K1_isdf(
            jnp.asarray(C_phi), jnp.asarray(xi_phi), 
            jnp.asarray(C_grad), jnp.asarray(xi_grad),
            self.jastrow_jax, self.params,
            jnp.asarray(self.grid_points), jnp.asarray(self.weights)
        )
        
        ref_k3 = calc_K3_isdf(
            jnp.asarray(C_phi), jnp.asarray(xi_phi),
            self.jastrow_jax, self.params,
            jnp.asarray(self.grid_points), jnp.asarray(self.weights)
        )
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Test K1_isdf
                k1 = calc_K1_isdf(
                    jnp.asarray(C_phi), jnp.asarray(xi_phi), 
                    jnp.asarray(C_grad), jnp.asarray(xi_grad),
                    self.jastrow_jax, self.params,
                    jnp.asarray(self.grid_points), jnp.asarray(self.weights),
                    batch_size=batch_size
                )
                np.testing.assert_allclose(k1, ref_k1, atol=1e-6)
                
                # Test K3_isdf
                k3 = calc_K3_isdf(
                    jnp.asarray(C_phi), jnp.asarray(xi_phi),
                    self.jastrow_jax, self.params,
                    jnp.asarray(self.grid_points), jnp.asarray(self.weights),
                    batch_size=batch_size
                )
                np.testing.assert_allclose(k3, ref_k3, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
