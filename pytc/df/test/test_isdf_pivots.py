"""Regression controls for production ISDF pivot selection."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pytc.df.isdf import _pivoted_cholesky_grad, _pivoted_cholesky_phi


class TestISDFPivotSelection(unittest.TestCase):
    def setUp(self):
        self.phi = jnp.array([[1.0, 1e-3, 1e-3, 1e-3]])
        self.grad = jnp.repeat(self.phi.T, 3, axis=1)[None, :, :]

    def test_phi_tie_break_does_not_reselect_a_live_pivot(self):
        pivots = np.asarray(_pivoted_cholesky_phi(self.phi, 4, jnp.array(1e-12)))
        self.assertEqual(len(np.unique(pivots)), 4)

    def test_gradient_tie_break_does_not_reselect_a_live_pivot(self):
        pivots = np.asarray(
            _pivoted_cholesky_grad(self.phi, self.grad, 4, jnp.array(3e-12))
        )
        self.assertEqual(len(np.unique(pivots)), 4)


if __name__ == "__main__":
    unittest.main()
