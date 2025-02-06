"""JAX-based autodiff implementations for pytc."""

from abc import ABC, abstractmethod
import jax.numpy as jnp

class Jastrow(ABC):
    """Abstract base class for JAX-based Jastrow factors."""
    
    def __init__(self, params, atomic_coords=None):
        """Initialize Jastrow factor.
        
        Args:
            params: Parameters for the Jastrow factor
            atomic_coords: Optional array of shape (N_atoms, 3) for nuclear coordinates
        """
        self.params = jnp.asarray(params)
        self.atomic_coords = None if atomic_coords is None else jnp.asarray(atomic_coords)
    
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

from .simple import SimpleJastrow
__all__ = ['Jastrow', 'SimpleJastrow']