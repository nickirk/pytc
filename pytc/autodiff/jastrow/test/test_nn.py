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
        # Set up H2 molecule
        mol_h2 = get_h2_molecule()
        self.nuclear_pos_h2 = mol_h2.atom_coords()
        self.nuclear_charges_h2 = mol_h2.atom_charges()
        
        # Set up H2O molecule
        mol_h2o = get_h2o_molecule()
        self.nuclear_pos_h2o = mol_h2o.atom_coords()
        self.nuclear_charges_h2o = mol_h2o.atom_charges()
        
        # Initialize jastrows with fixed random seed for reproducibility
        key = random.PRNGKey(42)
        key1, key2 = random.split(key)
        
        # Use smaller networks for testing
        self.jastrow_h2 = NeuralJastrow(
            self.nuclear_pos_h2,
            self.nuclear_charges_h2,
            layer_widths=[4, 4],
            key=key1
        )
        self.jastrow_h2o = NeuralJastrow(
            self.nuclear_pos_h2o,
            self.nuclear_charges_h2o,
            layer_widths=[8, 8],
            key=key2
        )
        
        # Initialize parameters
        self.params_h2 = self.jastrow_h2.init_params(key1)
        self.params_h2o = self.jastrow_h2o.init_params(key2)
    
    def test_params_shape(self):
        """Test parameter count and shapes."""
        # Test H2
        expected_params_h2 = (
            (1 + 2*len(self.nuclear_charges_h2)) * 4  # First layer weights
            + 4  # First layer bias
            + 4 * 4  # Second layer weights
            + 4  # Second layer bias
            + 4 * 1  # Output layer weights
            + 1  # Output layer bias
        )
        self.assertEqual(len(self.params_h2), expected_params_h2)
        
        # Test H2O
        expected_params_h2o = (
            (1 + 2*len(self.nuclear_charges_h2o)) * 8  # First layer weights
            + 8  # First layer bias
            + 8 * 8  # Second layer weights
            + 8  # Second layer bias
            + 8 * 1  # Output layer weights
            + 1  # Output layer bias
        )
        self.assertEqual(len(self.params_h2o), expected_params_h2o)
    
    def test_feature_construction(self):
        """Test feature vector construction."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        net_features_h2, cusp_features_h2 = self.jastrow_h2._construct_features(r1, r2)
        net_features_h2o, cusp_features_h2o = self.jastrow_h2o._construct_features(r1, r2)
        
        # Check network feature dimensions (1 e-e + 2N nuclear distances)
        expected_dim_h2 = 1 + 2*len(self.nuclear_charges_h2)
        expected_dim_h2o = 1 + 2*len(self.nuclear_charges_h2o)
        
        self.assertEqual(net_features_h2.shape, (1, expected_dim_h2))
        self.assertEqual(net_features_h2o.shape, (1, expected_dim_h2o))
        
        # Check cusp features structure
        cusp_r12_h2, cusp_r1n_h2, cusp_r2n_h2 = cusp_features_h2
        self.assertEqual(cusp_r12_h2.shape, ())  # scalar
        self.assertEqual(cusp_r1n_h2.shape, (len(self.nuclear_charges_h2),))
        self.assertEqual(cusp_r2n_h2.shape, (len(self.nuclear_charges_h2),))
    
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

if __name__ == '__main__':
    unittest.main()
