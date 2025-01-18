"""This module implements different types of Jastrow factors."""

import numpy as np
from abc import ABC, abstractmethod

class Jastrow(ABC):
    """Base class for Jastrow factors."""

    def __init__(self, parameters=None):
        """Initialize the Jastrow factor with parameters."""
        self.parameters = parameters

    @abstractmethod
    def __call__(self, r1, r2, atomic_positions=None):
        """Evaluate Jastrow factor at given positions.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Array of shape (..., 3) representing electron positions
            atomic_positions: Optional array of shape (N_atoms, 3) for nuclear coordinates
        
        Returns:
            Array of shape (N1, N2) where N1, N2 are the batch dimensions of r1, r2
        """
        pass

    @abstractmethod
    def grad(self, r1, r2=None, atomic_positions=None):
        """Compute gradient with coordinates as last dimension.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Optional array of shape (..., 3). If None, use r1
            atomic_positions: Optional array of shape (N_atoms, 3) for nuclear coordinates
            
        Returns:
            Array of shape (N1, N2, 3) containing gradients
        """
        pass


class SimpleJastrow(Jastrow):
    """Simple Jastrow factor of form: u(r1,r2) = a|r1-r2|/(1 + b|r1-r2|)"""
    
    def __init__(self, parameters):
        """Initialize with parameters [a, b]."""
        super().__init__(parameters)
        self.a = parameters[0]
        self.b = parameters[1]

    def __call__(self, r1, r2, atomic_positions=None):
        """Evaluate electron-electron Jastrow factor.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Array of shape (..., 3) representing electron positions
            atomic_positions: Not used in this implementation
        """
        # Ensure inputs are arrays and expand dims for broadcasting
        r1 = np.asarray(r1)[..., np.newaxis, :]
        r2 = np.asarray(r2)[np.newaxis, ...]
        
        diff = r1 - r2
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        dist = np.where(dist < 1e-10, 1e-10, dist)
        
        denom = 1.0 + self.b * dist
        return self.a * dist / denom

    def grad(self, r1, r2=None, atomic_positions=None):
        """Compute gradient with coordinates as last dimension.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Optional array of shape (..., 3). If None, use r1
            atomic_positions: Not used in this implementation
        """
        r1 = np.asarray(r1)
        r2 = np.asarray(r2) if r2 is not None else r1
        
        diff = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)
        dist = np.where(dist < 1e-10, 1e-10, dist)
        
        denom = 1.0 + self.b * dist
        du_dr = self.a / (denom * denom)
        
        return du_dr * diff
