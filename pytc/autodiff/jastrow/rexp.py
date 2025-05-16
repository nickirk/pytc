from pytc.autodiff import jastrow
import jax.numpy as jnp

class REXP(jastrow.Jastrow):
    """
    A class representing a radial exponential (REXP) Jastrow factor.
    
    Parameters
    ----------
    name : str, optional
        An optional name for the Jastrow factor. This can be used to identify
        or label the instance. Defaults to None.
    epsilon : float, optional
        A small positive value added to norms to prevent division by zero
        or other numerical instabilities. Defaults to 1e-8.
    """
    def __init__(self, name=None, epsilon=1e-8):
        super().__init__(name=name)
        self.epsilon = epsilon
        
    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
    def _compute(self, r1, r2, params):
        r12 = r1-r2
        r12_norm = self._safe_norm(r12)
        return 0.5*jnp.exp(-params['alpha'] * r12_norm) * r12_norm

    def __call__(self, r1, r2, params):
        return super().__call__(r1, r2, params)
    
    def init_params(self, **kwargs):
        return {'alpha': jnp.array([0.5])}