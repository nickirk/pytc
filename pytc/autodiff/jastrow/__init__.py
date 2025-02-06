"""JAX-based autodiff implementations for pytc."""

from abc import ABC, abstractmethod
import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 support
import jax.numpy as jnp

class Jastrow(ABC):
    """Abstract base class for JAX-based Jastrow factors."""
    
    def __init__(self, params, atomic_coords=None):
        """Initialize Jastrow factor with float64 precision.
        
        Args:
            params: Parameters for the Jastrow factor
            atomic_coords: Optional array of shape (N_atoms, 3) for nuclear coordinates
        """
        self.params = jnp.asarray(params, dtype=jnp.float64)
        self.atomic_coords = None if atomic_coords is None else jnp.asarray(atomic_coords, dtype=jnp.float64)
    
    @abstractmethod
    def __call__(self, r1, r2):
        """Evaluate Jastrow factor.
        
        Args:
            r1: Array of shape (..., 3) for first electron positions
            r2: Array of shape (..., 3) for second electron positions
            
        Returns:
            Jastrow factor value
        """
        pass
    
    @abstractmethod
    def grad_params(self, r1, r2):
        """Compute gradient with respect to parameters.
        
        Args:
            r1: Array of shape (..., 3) for first electron positions
            r2: Array of shape (..., 3) for second electron positions
            
        Returns:
            Gradient with respect to parameters
        """
        pass
    
    @abstractmethod
    def grad_r(self, r1, r2):
        """Compute gradient with respect to r1 positions.
        
        Args:
            r1: Array of shape (..., 3) for first electron positions
            r2: Array of shape (..., 3) for second electron positions
            
        Returns:
            Gradient with respect to r1 positions
        """
        pass
    
    def update(self, new_params):
        """Update parameters and return a new instance.
        
        This method creates a new instance of the same class with updated parameters
        while preserving other attributes. This is important for maintaining JAX's
        automatic differentiation chain.
        
        Args:
            new_params: New parameters for the Jastrow factor
            
        Returns:
            A new instance with updated parameters
        """
        new_instance = self.__class__(new_params)
        # Copy any additional attributes that should persist
        new_instance.atomic_coords = self.atomic_coords
        return new_instance

from .simple import SimpleJastrow
__all__ = ['Jastrow', 'SimpleJastrow']