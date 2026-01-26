from pytc.autodiff import jastrow
import jax
import jax.numpy as jnp
from flax import struct

@jax.custom_jvp
def _safe_norm_np(x, epsilon):
    r = jnp.sqrt(jnp.sum(x*x, axis=-1))
    return r + epsilon

@_safe_norm_np.defjvp
def _safe_norm_np_jvp(primals, tangents):
    x, epsilon = primals
    x_dot, _ = tangents # epsilon is constant
    r = jnp.sqrt(jnp.sum(x*x, axis=-1))
    safe_r = r + epsilon
    primal_out = safe_r
    
    # Gradient of (r + eps) w.r.t x is x / (r + eps) in NumPy's logic
    # (Note: true gradient of r+eps is x/r, but NumPy uses x/(r+eps) for direction)
    # tangent = dot(grad, x_dot)
    
    # We need to handle r=0 case safely for the division
    # If r=0, x=0, so numerator is 0. safe_r = eps. Result is 0.
    # But x/safe_r is well defined everywhere since safe_r >= eps > 0.
    
    tangent_out = jnp.sum(x * x_dot, axis=-1) / safe_r
    return primal_out, tangent_out

@struct.dataclass
class REXP(jastrow.Jastrow):
    """Exponential Jastrow factor: u(r) = 0.5 * r * exp(-alpha * r)."""
    
    epsilon: float = struct.field(pytree_node=False, default=1e-8)
    name: str = struct.field(pytree_node=False, default=None)
        
    def _compute(self, r1, r2, params):
        r12 = r1-r2
        # Use custom norm to match NumPy behavior
        r12_norm = _safe_norm_np(r12, self.epsilon)
        return 0.5*jnp.exp(-params['alpha'] * r12_norm) * r12_norm

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
        dist = _safe_norm_np(diff, self.epsilon)
        
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
