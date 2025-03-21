"""Polynomial Jastrow factor implementation."""

import jax.numpy as jnp
from .jastrow import Jastrow

class Poly(Jastrow):
    """Polynomial Jastrow factor."""
    
    def __init__(self):
        """Initialize polynomial Jastrow factor structure."""
        super().__init__()
    
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
        r = jnp.sqrt(jnp.sum(diff * diff) + 1e-10)  # Add epsilon for stability
        return params[0] * r  # Simple linear form for testing
