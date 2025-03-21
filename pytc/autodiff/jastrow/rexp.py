from pytc.autodiff import jastrow
import jax.numpy as jnp

class REXP(jastrow.Jastrow):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon
        
    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
    def _compute(self, r1, r2, params):
        r12 = r1-r2
        r12_norm = self._safe_norm(r12)
        return 0.5*jnp.exp(-params[0] * r12_norm) * r12_norm

    def __call__(self, r1, r2, params):
        return super().__call__(r1, r2, params)