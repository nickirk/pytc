"""JAX implementation of simple Jastrow factor."""

import jax
import jax.numpy as jnp
from pytc.autodiff.jastrow import Jastrow

class Poly(Jastrow):
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
            params: Jastrow parameters (contains a_i in first half, c_i in second half)
        """
        diff = r1 - r2
        r12 = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-10)
        
        # Split params into a_i and c_i coefficients
        n_terms = len(params)
        #a_params = params[:n_terms]
        c_params = params[:]
        
        
        # Compute powers of r_rescaled and multiply by c_i
        powers = jnp.arange(1, n_terms + 1)
        terms = c_params * r12**powers
        
        return jnp.sum(terms) 
    
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
