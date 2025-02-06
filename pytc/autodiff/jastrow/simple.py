"""JAX implementation of simple Jastrow factor."""

import jax
import jax.numpy as jnp
from pytc.autodiff.jastrow import Jastrow

class SimpleJastrow(Jastrow):
    """Simple Jastrow factor implemented in JAX."""
    
    def __init__(self, params, atomic_coords=None):
        """Initialize with parameters."""
        super().__init__(params, atomic_coords)
        # Create JIT-compiled versions of core functions
        self._call_single = jax.jit(self._compute_single)
        self._grad_r_single = jax.jit(self._compute_grad_r)
        # Add vectorized version of call_single
        self._call_batch = jax.jit(jax.vmap(
            jax.vmap(self._compute_single, in_axes=(None, None, 0)),
            in_axes=(None, 0, None)
        ))
    
    def _compute_single(self, params, r1, r2):
        """Evaluate Jastrow for single positions.
        
        Args:
            params: Jastrow parameters
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
        """
        r12_sq = jnp.sum((r1 - r2)**2)
        r12 = jnp.sqrt(r12_sq)
        return jnp.where(r12_sq > 1e-10, 
                        jnp.sum(params * r12),
                        0.0)
    
    def _compute_grad_r(self, params, r1, r2):
        """Compute gradient with respect to r1 for single positions."""
        diff = r1 - r2
        r12_sq = jnp.sum(diff**2)
        r12 = jnp.sqrt(r12_sq)
        return jnp.where(r12_sq > 1e-10,
                        params[0] * diff / r12,
                        jnp.zeros_like(diff))
    
    def __call__(self, r1, r2):
        """Evaluate Jastrow factor for batched positions.
        
        Args:
            r1: Array of shape (N, 3) for first electron positions
            r2: Array of shape (M, 3) for second electron positions
            
        Returns:
            Array of shape (N, M) containing Jastrow values
        """
        # Handle both single point and batched inputs
        r1 = jnp.atleast_2d(r1)
        r2 = jnp.atleast_2d(r2)
        return self._call_batch(self.params, r1, r2)
    
    def grad_r(self, r1, r2):
        """Compute gradient with respect to r1 position.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            
        Returns:
            Array of shape (3,) containing gradient with respect to r1
        """
        return self._grad_r_single(self.params, r1, r2)
    
    def grad_params(self, r1, r2):
        """Compute gradient with respect to parameters."""
        return jax.grad(self._compute_single)(self.params, r1, r2)
