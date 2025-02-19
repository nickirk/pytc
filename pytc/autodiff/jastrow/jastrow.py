"""JAX-based autodiff implementations for pytc."""

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp

class Jastrow(ABC):
    """Abstract base class for JAX-based Jastrow factors."""
    
    def __init__(self, params):
        """Initialize Jastrow factor.
        
        Args:
            params: Parameters for the Jastrow factor
        """
        self.params = params
    
    @abstractmethod
    def _compute(self, r1, r2, params):
        """Core computation of Jastrow exponent u.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
            
        Returns:
            Jastrow exponent value u
        """
        pass

    def __call__(self, r1, r2):
        """Evaluate Jastrow factor J = exp(u) for a single pair.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            
        Returns:
            Jastrow factor value J = exp(u)
        """
        return jnp.exp(self._compute(r1, r2, self.params))
    
    def grad_r(self, r1, r2):
        """Compute gradient of u w.r.t r1 coordinates."""
        def scalar_fn(x):
            # Ensure scalar output by selecting the single value
            return self._compute(x, r2, self.params).reshape(-1)[0]
        return jax.grad(scalar_fn)(r1)
    
    def laplacian_r(self, r1, r2):
        """Compute Laplacian of u w.r.t r1 coordinates."""
        def scalar_fn(x):
            # Ensure scalar output by selecting the single value
            return self._compute(x, r2, self.params).reshape(-1)[0]
        return jnp.trace(jax.hessian(scalar_fn)(r1))
    
    def grad_params(self, r1, r2):
        """Compute gradient of u w.r.t parameters.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            
        Returns:
            Gradient array with same shape as params
        """
        return jax.grad(lambda p: self._compute(r1, r2, p))(self.params)
    
    def get_log_grads(self, r1, r2):
        """Compute both ∇u and ∇²u for the Jastrow exponent.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            
        Returns:
            tuple (grad_u, lapl_u) containing:
                grad_u: gradient of u, shape (3,)
                lapl_u: laplacian of u (scalar)
        """
        grad_u = self.grad_r(r1, r2)
        lapl_u = self.laplacian_r(r1, r2)
        return grad_u, lapl_u
    
    def update(self, new_params):
        """Return new instance with updated parameters.
        
        Args:
            new_params: New parameters for the Jastrow factor
            
        Returns:
            New Jastrow instance
        """
        return self.__class__(new_params)
