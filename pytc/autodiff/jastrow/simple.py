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
        self._compute_jit = jax.jit(self._compute)
        # Fix the vectorization to handle proper batch dimensions
        self._call_batch = jax.jit(jax.vmap(
            jax.vmap(lambda x, y, p: self._compute(x, y, p),
                    in_axes=(None, 0, None)),
            in_axes=(0, None, None)
        ))

    def _compute(self, r1, r2, params):
        """Core computation of Jastrow factor.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
        """
        diff = r1 - r2
        r12_sq = jnp.sum(diff**2, axis=-1)
        # Add small epsilon to prevent division by zero
        r12 = jnp.sqrt(r12_sq + 1e-10)
        # Smooth cutoff using sigmoid
        cutoff = jax.nn.sigmoid((r12 - 1e-5) * 1e6)
        return jnp.sum(params * r12 * cutoff)
    
    def __call__(self, r1, r2):
        """Evaluate Jastrow factor for batched positions.
        
        Args:
            r1: Array of shape (N, 3) for first electron positions
            r2: Array of shape (M, 3) for second electron positions
            
        Returns:
            Array of shape (N, M) containing Jastrow values
        """
        r1 = jnp.atleast_2d(r1)  # Shape (N, 3)
        r2 = jnp.atleast_2d(r2)  # Shape (M, 3)
        return self._call_batch(r1, r2, self.params)  # Should return shape (N, M)
