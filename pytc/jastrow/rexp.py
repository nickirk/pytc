
import numpy as np
from pytc.jastrow import Jastrow


class REXP(Jastrow):

    def __call__(self, r1, r2):
        """Evaluate Jastrow factor at given positions."""
        r1 = np.atleast_2d(r1)
        r2 = np.atleast_2d(r2)
        r12 = np.linalg.norm(r1[:, None, :] - r2[None, :], axis=-1)
        r12 += 1e-8  # Regularization to avoid division by zero
        return 0.5 * r12 * np.exp(-self.params[0] * r12)
    

    def _process_grad_batch(self, r1, r2):
        """Compute gradient of Jastrow factor 1/2*r_{12}*exp(-param*r_{12}) with respect to r1.
        
        Args:
            r1: Position vectors with shape (n_r1, 3)
            r2: Position vectors with shape (n_r2, 3)
        
        Returns:
            Gradient vectors with shape (n_r1, 3)
        """
        # Compute pairwise differences between all points
        diff = r1[:, None, :] - r2[None, :, :]  # Shape: (n_r1, n_r2, 3)
        
        # Compute distances between all pairs
        r12_squared = np.sum(diff**2, axis=-1)  # Shape: (n_r1, n_r2)
        r12 = np.sqrt(r12_squared)  # Shape: (n_r1, n_r2)
        
        # Regularization to avoid division by zero
        #safe_r12 = np.maximum(r12, 1e-8)
        safe_r12 = r12 + 1e-8
        
        # Calculate derivative of 1/2*r*exp(-param*r) with respect to r
        # df/dr = 1/2 * exp(-param*r) * (1 - param*r)
        df_dr = 0.5 * np.exp(-self.params[0] * safe_r12) * (1 - self.params[0] * safe_r12)
        
        # Compute gradient direction (r1-r2)/r12
        direction = diff / safe_r12[..., None]
        
        # Apply the derivative magnitude to the direction
        gradients = df_dr[..., None] * direction
        
        # Zero out gradients where r12 is near zero
        #mask = (r12 > 1e-10)[..., None]
        #gradients = np.where(mask, gradients, np.zeros_like(gradients))
        
        # Sum over r2 dimension to get net gradient at each r1
        return gradients