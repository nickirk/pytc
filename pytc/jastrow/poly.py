"""Polynomial Jastrow factor implementation."""

import jax.numpy as jnp
from .jastrow import Jastrow
from flax import struct

@struct.dataclass
class Poly(Jastrow):
    """Polynomial Jastrow factor."""
    n: int = struct.field(pytree_node=False, default=1)
    name: str = struct.field(pytree_node=False, default=None)
    
    def _compute(self, r1, r2, params):
        """Compute Jastrow exponent u(r).
        
        Args:
            r1: Array of shape (..., 3) for first electron position
            r2: Array of shape (..., 3) for second electron position
            params: Array containing polynomial coefficients
            
        Returns:
            Jastrow exponent value u with shape (...)
        """
        diff = r1 - r2
        r = jnp.sqrt(jnp.sum(diff*diff, axis=-1) + 1e-10)  # Add epsilon for stability
        return params[0] * r  # Simple linear form for testing
    
    def init_params(self, **kwargs):
        return jnp.array([0.5]+[0.0]*(self.n-1))
