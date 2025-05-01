"""Tests for Boys-Handy Jastrow implementation."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf import gto

from pytc.autodiff.jastrow.bh import BoysHandy, BHTerm

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

class TestBoysHandy(unittest.TestCase):
    """Test Boys-Handy Jastrow implementation."""
    
    def setUp(self):
        self.mol = get_h2_molecule()
        self.key = random.PRNGKey(0)
        
        # Create Boys-Handy Jastrow with custom terms
        # Use negative coefficients for e-n terms for expected decay behavior
        terms = [
            [BHTerm(0, 0, 1, 0.5),  # e-e cusp term
             BHTerm(1, 0, 0, -0.1), # e-n term (attractive)
             BHTerm(2, 0, 0, -0.1)],# higher order term (attractive)
            [BHTerm(0, 0, 1, 0.5),  # e-e cusp term
             BHTerm(1, 0, 0, -0.1), # e-n term (attractive)
             BHTerm(2, 0, 0, -0.1)] # higher order term (attractive)
        ]
        self.jastrow = BoysHandy(self.mol, terms_per_nucleus=terms)
        self.params = self.jastrow.init_params(key=self.key)

    def test_init(self):
        """Test initialization."""
        # Test default initialization
        jastrow = BoysHandy(self.mol)
        params = jastrow.init_params()
        
        # Check parameter structure
        self.assertIn('b_raw', params)
        self.assertIn('d_raw', params)
        self.assertIn('c_raw', params)
        
        # Check shapes
        self.assertEqual(params['b_raw'].shape, (2,))  # 2 nuclei
        self.assertEqual(params['d_raw'].shape, (2,))  # 2 nuclei
        # Each nucleus has 3 default terms: e-e cusp, e-n, and higher order
        self.assertEqual(params['c_raw'].shape, (2, 3))  # (n_nuclei, n_terms)

    def test_parameter_handling(self):
        """Test parameter flattening/unflattening."""
        flat_params = self.jastrow.flatten_params(self.params)
        restored_params = self.jastrow.unflatten_params(flat_params)
        
        # Check that parameters are correctly restored
        for key in self.params:
            np.testing.assert_allclose(self.params[key], restored_params[key])

    def test_electron_symmetry(self):
        """Test symmetry with respect to electron exchange."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([0., 0., 1.0])
        
        value1 = self.jastrow._compute(r1, r2, self.params)
        value2 = self.jastrow._compute(r2, r1, self.params)
        
        np.testing.assert_allclose(value1, value2, rtol=1e-7)

    def test_electron_cusp(self):
        """Test electron-electron cusp condition."""
        r1 = jnp.array([0., 0., 0.])
        eps = 1e-5
        r2 = jnp.array([eps, 0., 0.])
        
        # Compute numerical gradient at small separation
        # grad(u) dot (r2-r1)/|r2-r1| should -> 0.5 for unlike spins
        # Here, (r2-r1)/|r2-r1| = (1,0,0), so we check the x-component of grad(u)
        grad_fn = jax.grad(lambda x: self.jastrow._compute(r1, x, self.params))
        grad_val = grad_fn(r2)[0]  # x-component of gradient at r2=(eps,0,0)
        
        # For unlike-spin electrons, expect gradient component ≈ 0.5 at zero separation
        # The cusp term BHTerm(0,0,1,c=0.5) with delta(0,0)=0.5 and scaling d
        # contributes 0.5 * c * d to the gradient component.
        # With default d_raw=0.5 -> d=softplus(0.5)~0.973, expected grad ~ 0.5*0.5*0.973 ~ 0.243 per nucleus
        # Total expected gradient ~ 2 * 0.243 = 0.486
        np.testing.assert_allclose(grad_val, 0.5, rtol=0.1) # Check grad_val directly, allow some tolerance

    def test_nuclear_decay(self):
        """Test decay of correlation with nuclear distance."""
        r1_near = jnp.array([0., 0., 0.1])
        r1_far = jnp.array([0., 0., 5.0])
        r2 = jnp.array([0., 0., -0.5])
        
        value_near = self.jastrow._compute(r1_near, r2, self.params)
        value_far = self.jastrow._compute(r1_far, r2, self.params)
        
        # Correlation should decay at large distances
        self.assertGreater(abs(value_near), abs(value_far))

    def test_gradient_computation(self):
        """Test gradient and laplacian computation."""
        r1 = jnp.array([0., 0., 0.5])
        r2 = jnp.array([0., 0., -0.5])
        
        grad1, lap1 = self.jastrow.get_log_grads_r1(r1, r2, self.params)
        grad2, lap2 = self.jastrow.get_log_grads_r2(r1, r2, self.params)
        
        # Check shapes
        self.assertEqual(grad1.shape, (3,))
        self.assertEqual(grad2.shape, (3,))
        self.assertEqual(lap1.shape, ())
        self.assertEqual(lap2.shape, ())
        
        # Check finite values
        self.assertTrue(jnp.all(jnp.isfinite(grad1)))
        self.assertTrue(jnp.all(jnp.isfinite(grad2)))
        self.assertTrue(jnp.isfinite(lap1))
        self.assertTrue(jnp.isfinite(lap2))

if __name__ == '__main__':
    unittest.main()
