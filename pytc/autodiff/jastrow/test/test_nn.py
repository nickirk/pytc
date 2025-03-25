"""Tests for Neural Network based Jastrow implementation."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf import gto
from pytc.autodiff.jastrow import NeuralJastrow

# Enable float64 support
jax.config.update("jax_enable_x64", True)

def get_h2_molecule(bond_length=1.4):
    """Create H2 molecule."""
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {bond_length}',
        basis='sto-3g',
        unit='bohr'
    )
    return mol

def get_h2o_molecule():
    """Create H2O molecule."""
    mol = gto.M(
        atom='''
        O  0.0000000   0.0000000   0.0000000
        H  0.7569685   0.5858752   0.0000000
        H -0.7569685   0.5858752   0.0000000
        ''',
        basis='sto-3g',
        unit='angstrom'
    )
    return mol

class TestNeuralJastrow(unittest.TestCase):
    """Test cases for NeuralJastrow class."""
    
    def setUp(self):
        # Test systems
        h2_pos = jnp.array([[0., 0., -0.7], [0., 0., 0.7]])
        h2_charges = jnp.array([1., 1.])
        h2o_pos = jnp.array([[0., 0., 0.], [0., 1.43233673, -0.96104039],
                            [0., -1.43233673, -0.96104039]])
        h2o_charges = jnp.array([8., 1., 1.])
        
        key1 = random.PRNGKey(0)
        # Initialize networks with smaller width for testing
        self.jastrow_h2 = NeuralJastrow(h2_pos, h2_charges, 
                                       layer_widths=[4, 4])
        self.jastrow_h2o = NeuralJastrow(h2o_pos, h2o_charges, 
                                        layer_widths=[4, 4])
        
        # Split the key for three networks
        key1, key2, key3 = random.split(key1, 3)
        self.params_h2 = self.jastrow_h2.init_params(key1, key2, key3)
        key1, key2, key3 = random.split(key1, 3)
        self.params_h2o = self.jastrow_h2o.init_params(key1, key2, key3)
    
    def test_params_shape(self):
        """Test parameter count and shapes."""
        n_nuclei_h2 = 2
        expected_params_h2 = (
            self.jastrow_h2.get_param_count_single(2 * n_nuclei_h2) +  # en network
            self.jastrow_h2.get_param_count_single(1) +                # ee network
            self.jastrow_h2.get_param_count_single(1 + 2 * n_nuclei_h2)  # een network
        )
        self.assertEqual(self.params_h2.size, expected_params_h2)
        
        n_nuclei_h2o = 3
        expected_params_h2o = (
            self.jastrow_h2o.get_param_count_single(2 * n_nuclei_h2o) +  # en network
            self.jastrow_h2o.get_param_count_single(1) +                 # ee network
            self.jastrow_h2o.get_param_count_single(1 + 2 * n_nuclei_h2o)  # een network
        )
        self.assertEqual(self.params_h2o.size, expected_params_h2o)
    
    def test_feature_construction(self):
        """Test feature vector construction."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        en_features, ee_features, een_features = self.jastrow_h2._construct_features(r1, r2)
        
        # Check shapes
        self.assertEqual(en_features.shape, (1, 4))  # 2 nuclei * 2 electrons
        self.assertEqual(ee_features.shape, (1, 1))  # 1 e-e distance
        self.assertEqual(een_features.shape, (1, 5))  # 1 e-e + 2*2 e-n distances
    
    def test_compute_basics(self):
        """Test basic compute functionality."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        # Check output is scalar
        value_h2 = self.jastrow_h2._compute(r1, r2, self.params_h2)
        value_h2o = self.jastrow_h2o._compute(r1, r2, self.params_h2o)
        
        self.assertEqual(value_h2.shape, ())
        self.assertEqual(value_h2o.shape, ())
        
        # Check values are finite
        self.assertTrue(jnp.isfinite(value_h2))
        self.assertTrue(jnp.isfinite(value_h2o))
    
    def test_electron_coalescence(self):
        """Test behavior when electrons approach each other."""
        r1 = jnp.array([0., 0., 0.])
        r2_close = jnp.array([0., 0., 1e-3])
        r2_far = jnp.array([0., 0., 1.0])
        
        value_close = self.jastrow_h2._compute(r1, r2_close, self.params_h2)
        value_far = self.jastrow_h2._compute(r1, r2_far, self.params_h2)
        
        # Value should be larger in magnitude when electrons are close
        self.assertTrue(jnp.abs(value_close) > jnp.abs(value_far))
    
    def test_h2o_symmetry(self):
        """Test approximate symmetry of Jastrow for H2O."""
        # Test points symmetric about the O atom at z=0 plane
        z_offset = 0.3
        r1 = jnp.array([0., 0.5, z_offset])
        r2 = jnp.array([0., -0.5, z_offset])  # Symmetric position
        
        value1 = self.jastrow_h2o._compute(r1, r2, self.params_h2o)
        value2 = self.jastrow_h2o._compute(r2, r1, self.params_h2o)
        
        # Check values are close but not necessarily identical
        np.testing.assert_allclose(value1, value2, rtol=1e-5)
    
    def test_gradients(self):
        """Test gradient computation."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        # Test parameter gradients
        grad_params = jax.grad(lambda p: self.jastrow_h2._compute(r1, r2, p))(self.params_h2)
        self.assertEqual(grad_params.shape, self.params_h2.shape)
        self.assertTrue(jnp.all(jnp.isfinite(grad_params)))
        
        # Test spatial gradients
        grad_r1 = jax.grad(lambda x: self.jastrow_h2._compute(x, r2, self.params_h2))(r1)
        self.assertEqual(grad_r1.shape, (3,))
        self.assertTrue(jnp.all(jnp.isfinite(grad_r1)))

    def test_gradient_symmetry(self):
        """Test that grad_r1 = -grad_r2 for the Jastrow factor."""
        r1 = jnp.array([0.2, 0.3, 0.1])
        r2 = jnp.array([0.5, -0.1, 0.4])
        
        # Compute gradients with respect to both electron positions
        grad_r1 = jax.grad(lambda x: self.jastrow_h2._compute(x, r2, self.params_h2))(r1)
        grad_r2 = jax.grad(lambda x: self.jastrow_h2._compute(r1, x, self.params_h2))(r2)
        
        # Check that grad_r1 = -grad_r2
        np.testing.assert_allclose(grad_r1, -grad_r2, rtol=1e-7, 
                                 err_msg="Gradient symmetry violated: grad_r1 ≠ -grad_r2")

if __name__ == '__main__':
    unittest.main()
