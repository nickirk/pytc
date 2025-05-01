from pytc.autodiff.jastrow import Jastrow
from pytc.autodiff.jastrow import NuclearCusp
import jax.numpy as jnp

class CompositeJastrow(Jastrow):
    """Combines multiple Jastrow factors by adding their exponents."""
    
    def __init__(self, jastrows):
        """Initialize with list of Jastrow factors.
        
        Args:
            jastrows: List of Jastrow instances to combine
        """
        super().__init__()
        self.jastrows = jastrows
        # Track jastrow identifiers for filtering
        self.jastrow_types = [j.__class__.__name__ for j in jastrows]
        self.jastrow_names = [j.name for j in jastrows]
        
    def _compute(self, r1, r2, params):
        """Compute sum of Jastrow exponents.
        
        Args:
            r1, r2: Electron positions
            params: List of parameter sets, one per Jastrow factor
        """
        total = 0.0
        idx = 0
        for jastrow in self.jastrows:
            total += jastrow._compute(r1, r2, params[idx])
            idx += 1
        return total
    
    def get_log_grads_r1(self, r1, r2, params):
        """Sum gradients and laplacians, handling NCusp normalization."""
        grad_total = jnp.zeros(3)
        lap_total = 0.0
        idx = 0
        
        for jastrow in self.jastrows:
            grad_u, lap_u = jastrow.get_log_grads_r1(r1, r2, params[idx])
            grad_total += grad_u
            lap_total += lap_u
            idx += 1
            
        return grad_total, lap_total
    
    def get_log_grads_r2(self, r1, r2, params):
        """Sum gradients and laplacians, handling NCusp normalization."""
        grad_total = jnp.zeros(3)
        lap_total = 0.0
        idx = 0
        
        for jastrow in self.jastrows:
            grad_u, lap_u = jastrow.get_log_grads_r2(r1, r2, params[idx])
            grad_total += grad_u
            lap_total += lap_u
            idx += 1
            
        return grad_total, lap_total

    def grad_params(self, r1, r2, params):
        """Compute gradient of u w.r.t parameters.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: List of parameter sets, one per Jastrow factor
            
        Returns:
            List of gradients, matching the structure of params
        """
        grads = []
        for i, jastrow in enumerate(self.jastrows):
            grad = jastrow.grad_params(r1, r2, params[i])
            grads.append(grad)
        return grads

    def init_params(self):
        """Initialize parameters for all Jastrow factors."""
        return [j.init_params() for j in self.jastrows]
