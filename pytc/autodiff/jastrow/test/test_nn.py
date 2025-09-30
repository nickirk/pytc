"""Tests for Neural Network based Jastrow implementations."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf import gto
from pytc.autodiff.jastrow import NeuralEN, NeuralEE, NeuralEEN
from pytc.autodiff.jastrow import CompositeJastrow

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

class TestNeuralBase(unittest.TestCase):
    """Common setup and utilities for neural network tests."""
    
    def setUp(self):
        self.h2_pos = jnp.array([[0., 0., -0.7], [0., 0., 0.7]])
        self.h2_charges = jnp.array([1., 1.])
        self.h2o_pos = jnp.array([[0., 0., 0.], 
                                 [0., 1.43233673, -0.96104039],
                                 [0., -1.43233673, -0.96104039]])
        self.h2o_charges = jnp.array([8., 1., 1.])
        self.key = random.PRNGKey(0)

        # Add molecule instances
        self.h2_mol = get_h2_molecule()
        self.h2o_mol = get_h2o_molecule()

    def assert_gradient_symmetry(self, jastrow, params):
        """Test that grad_r1 = -grad_r2 for the Jastrow factor."""
        nelec = 2
        walker = random.normal(self.key, shape=(nelec, 3))
        
        grad_r1_fn = jax.vmap(jax.vmap(
            jax.grad(lambda x, y: jastrow._compute(x, y, params)), 
            in_axes=(None, 0)), in_axes=(0, None))
        
        grad_r2_fn = jax.vmap(jax.vmap(
            jax.grad(lambda x, y: jastrow._compute(x, y, params), 1), 
            in_axes=(None, 0)), in_axes=(0, None))
        
        grad_r1 = grad_r1_fn(walker, walker)
        grad_r2 = grad_r2_fn(walker, walker)
        
        np.testing.assert_allclose(grad_r1, -grad_r2, rtol=1e-7)

class TestNeuralEN(TestNeuralBase):
    """Test electron-nuclear component."""
    
    def setUp(self):
        super().setUp()
        self.jastrow_h2 = NeuralEN(self.h2_mol, layer_widths=[4, 4])
        self.jastrow_h2o = NeuralEN(self.h2o_mol, layer_widths=[4, 4])
        self.params_h2 = self.jastrow_h2.init_params(key=self.key)
        self.params_h2o = self.jastrow_h2o.init_params(key=self.key)

    def test_compute(self):
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        value_h2 = self.jastrow_h2._compute(r1, r2, self.params_h2)
        value_h2o = self.jastrow_h2o._compute(r1, r2, self.params_h2o)
        
        self.assertEqual(value_h2.shape, ())
        self.assertEqual(value_h2o.shape, ())
        self.assertTrue(jnp.isfinite(value_h2))
        self.assertTrue(jnp.isfinite(value_h2o))

class TestNeuralEE(TestNeuralBase):
    """Test electron-electron component."""
    
    def setUp(self):
        super().setUp()
        # Pass any molecule since EE only depends on electron coordinates
        self.jastrow = NeuralEE(self.h2_mol, layer_widths=[4, 4])
        self.params = self.jastrow.init_params(key=self.key)

    def test_symmetry(self):
        self.assert_gradient_symmetry(self.jastrow, self.params)

class TestNeuralEEN(TestNeuralBase):
    """Test electron-electron-nuclear component."""
    
    def setUp(self):
        super().setUp()
        self.jastrow_h2o = NeuralEEN(mol=self.h2o_mol, layer_widths=[4, 4])
        self.params_h2o = self.jastrow_h2o.init_params(key=self.key)

    def test_h2o_symmetry(self):
        z_offset = 0.3
        r1 = jnp.array([0., 0.5, z_offset])
        r2 = jnp.array([0., -0.3, z_offset])
        
        value1 = self.jastrow_h2o._compute(r1, r2, self.params_h2o)
        value2 = self.jastrow_h2o._compute(r2, r1, self.params_h2o)
        
        np.testing.assert_allclose(value1, value2, rtol=1e-5)
    
    def test_h2o_gradients(self):
        # Test gradients with respect to r1 and r2 should not be equal
        r1 = jnp.array([0., 0.5, 0.3])
        r2 = jnp.array([0., -0.3, 0.3])
        grad_r1 = jax.grad(lambda x: self.jastrow_h2o._compute(x, r2, self.params_h2o))(r1)
        grad_r2 = jax.grad(lambda x: self.jastrow_h2o._compute(r1, x, self.params_h2o))(r2)
        grad_r1_1 = jax.grad(lambda x: self.jastrow_h2o._compute(r2, x, self.params_h2o))(r1)
        grad_r2_1 = jax.grad(lambda x: self.jastrow_h2o._compute(x, r1, self.params_h2o))(r2)
        np.testing.assert_allclose(grad_r1, grad_r1_1, rtol=1e-5)
        np.testing.assert_allclose(grad_r2, grad_r2_1, rtol=1e-5)

class TestCompositeNeural(TestNeuralBase):
    """Test combined neural Jastrow components."""
    
    def setUp(self):
        super().setUp()
        # Create individual components
        self.en = NeuralEN(self.h2_mol, layer_widths=[4, 4])
        self.ee = NeuralEE(self.h2_mol, layer_widths=[4, 4])
        self.een = NeuralEEN(mol=self.h2_mol, layer_widths=[4, 4])
        
        # Create composite with initialized params
        self.jastrow = CompositeJastrow([self.en, self.ee, self.een])
        self.params = [
            self.en.init_params(key=self.key),
            self.ee.init_params(key=self.key),
            self.een.init_params(key=self.key)
        ]

    def test_composite_compute(self):
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([1., 0., 0.])
        
        value = self.jastrow._compute(r1, r2, self.params)
        self.assertTrue(jnp.isfinite(value))
        
        # Test that composite gradient matches sum of individual gradients
        grad_composite = jax.grad(lambda x: self.jastrow._compute(x, r2, self.params))(r1)
        grad_parts = [
            jax.grad(lambda x: self.en._compute(x, r2, self.params[0]))(r1),
            jax.grad(lambda x: self.ee._compute(x, r2, self.params[1]))(r1),
            jax.grad(lambda x: self.een._compute(x, r2, self.params[2]))(r1)
        ]
        grad_sum = sum(grad_parts)
        
        np.testing.assert_allclose(grad_composite, grad_sum, rtol=1e-7)

if __name__ == '__main__':
    unittest.main()
