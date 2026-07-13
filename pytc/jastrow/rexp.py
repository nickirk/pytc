from pytc import jastrow
import jax
import jax.numpy as jnp
from flax import struct

def _safe_norm_np(x, epsilon):
    # Match NumPy behavior: r + eps (not sqrt(r^2 + eps^2))
    # This ensures JAX and NumPy implementations give identical results
    return jnp.sqrt(jnp.sum(x*x, axis=-1)) + epsilon

@struct.dataclass
class REXP(jastrow.Jastrow):
    """Exponential Jastrow factor: u(r) = 0.5 * r * exp(-alpha * r)."""
    
    epsilon: float = struct.field(pytree_node=False, default=1e-8)
    name: str = struct.field(pytree_node=False, default=None)
        
    def _compute(self, r1, r2, params):
        r12 = r1-r2
        # Use custom norm to match NumPy behavior
        r12_norm = _safe_norm_np(r12, jnp.array(self.epsilon))
        # Squeeze to a scalar: alpha is stored shape-(1,), and the pair-value
        # contract (e.g. the log-Jastrow scan carry) requires a scalar.
        return jnp.squeeze(0.5*jnp.exp(-params['alpha'] * r12_norm) * r12_norm)

    def __call__(self, r1, r2, params):
        return super().__call__(r1, r2, params)
    
    def init_params(self, **kwargs):
        alpha = kwargs.get('alpha', 0.5)
        return {'alpha': jnp.array([alpha])}

    def grad_r_batch(self, r1_batch, r2_batch, params):
        """Compute gradients for a batch of r1 and r2 points analytically.
        
        Args:
            r1_batch: (batch_size_out, 3)
            r2_batch: (batch_size_in, 3)
            params: Jastrow parameters
            
        Returns:
            Gradients of shape (batch_size_out, batch_size_in, 3)
        """
        # r1_batch: (N_out, 3)
        # r2_batch: (N_in, 3)
        
        # diff: (N_out, N_in, 3)
        diff = r1_batch[:, None, :] - r2_batch[None, :, :]
        
        # dist: (N_out, N_in)
        dist = _safe_norm_np(diff, jnp.array(self.epsilon))
        
        alpha = params['alpha'][0]
        
        # u = 0.5 * r * exp(-alpha * r)
        # grad = 0.5 * exp(-alpha * r) * (1 - alpha * r) * (diff / r)
        
        prefactor = 0.5 * jnp.exp(-alpha * dist) * (1 - alpha * dist)
        
        # Avoid division by zero (handled by safe norm, but explicit safety for direction)
        # safe_dist = dist + epsilon (already done in _safe_norm_np if we used it directly, 
        # but _safe_norm_np returns r+eps)
        
        # diff / dist: (N_out, N_in, 3)
        direction = diff / dist[..., None]
        
        return prefactor[..., None] * direction
