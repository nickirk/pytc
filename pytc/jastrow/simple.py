"""Simple Jastrow factor implementation."""

import numpy as np
from . import Jastrow

class SimpleJastrow(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    
    def __call__(self, r1, r2):
        """Evaluate Jastrow factor at given positions."""
        r1 = np.atleast_2d(r1)  # Ensure 2D array with shape (N, 3)
        r2 = np.atleast_2d(r2)  # Ensure 2D array with shape (M, 3)
        
        delta_r = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        result = 0.5*1./self.params[0]*np.exp(self.params[0] * np.linalg.norm(delta_r, axis=-1))
        
        # Handle single point inputs
        if r1.shape[0] == 1 and r2.shape[0] == 1:
            result = result.reshape(1, 1)
            
        return result
    
    def _process_grad_batch(self, r1_batch, r2):
        """Process a batch of r1 points efficiently using vectorized operations."""
        # Reshape for broadcasting
        r1_batch = np.asarray(r1_batch)[:, None, :]  # (batch, 1, 3)
        r2 = np.asarray(r2)[None, :, :]             # (1, N2, 3)
        
        # Vectorized operations for the batch
        diff = r1_batch - r2                         # (batch, N2, 3)
        norm = np.linalg.norm(diff, axis=-1)[..., None]  # (batch, N2, 1)
        norm = np.where(norm == 0, 1.0, norm)
        
        # Compute jastrow values efficiently for the batch
        r = norm.squeeze(-1)  # Remove last dimension for jastrow calculation
        jastrow_values = 0.5/self.params[0] * np.exp(-self.params[0] * r)[..., None]
        
        return -0.5*self.params[0] * diff / norm * jastrow_values