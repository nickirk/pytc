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

class SM7(Jastrow):
    def __init__(self, coefficients):
        """Initialize with coefficient table for m,n,o terms.
        
        Args:
            coefficients: Dictionary with (m,n,o) tuple keys and coefficient values
        """
        super().__init__(coefficients)
        self.coefficients = coefficients
    
    def _scaled_r(self, r):
        """Convert distance to scaled distance r/(1+r)."""
        return r / (1.0 + r)
    
    def _scaled_r_grad(self, r):
        """Gradient of scaled distance with respect to r."""
        return 1.0 / (1.0 + r)**2
    
    def __call__(self, r1, r2, atomic_positions=None):
        """Evaluate SM7 Jastrow factor.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Array of shape (..., 3) representing electron positions
            atomic_positions: Ignored (assumes nucleus at origin)
        """
        r1 = np.asarray(r1)[..., np.newaxis, :]
        r2 = np.asarray(r2)[np.newaxis, ...]
        
        # Get electron-nucleus distances
        r1_dist = np.sqrt(np.sum(r1 * r1, axis=-1))
        r2_dist = np.sqrt(np.sum(r2 * r2, axis=-1))
        
        # Get electron-electron distances
        diff = r1 - r2
        r12_dist = np.sqrt(np.sum(diff * diff, axis=-1))
        r12_dist = np.where(r12_dist < 1e-10, 1e-10, r12_dist)
        
        # Convert to scaled distances
        r1_scaled = self._scaled_r(r1_dist)
        r2_scaled = self._scaled_r(r2_dist)
        r12_scaled = self._scaled_r(r12_dist)
        
        # Compute sum over m,n,o terms
        result = np.zeros_like(r12_dist)
        for (m,n,o), coeff in self.coefficients.items():
            term = coeff * (r1_scaled**m * r2_scaled**n + r2_scaled**m * r1_scaled**n) * r12_scaled**o
            result += term
            
        return result

    def grad(self, r1, r2=None, atomic_positions=None):
        """Compute gradient of SM7 Jastrow factor with respect to r1.
        
        Args:
            r1: Array of shape (..., 3) representing electron positions
            r2: Array of shape (..., 3) representing electron positions
            atomic_positions: Ignored (assumes nucleus at origin)
            
        Returns:
            Array of shape (N1, N2, 3) containing gradients
        """
        r1 = np.asarray(r1)
        r2 = np.asarray(r2) if r2 is not None else r1
        
        # Reshape for broadcasting
        r1_expanded = r1[:, np.newaxis, :]
        r2_expanded = r2[np.newaxis, :, :]
        
        # Calculate electron-nucleus distances and their directional gradients
        r1_dist = np.sqrt(np.sum(r1_expanded * r1_expanded, axis=-1, keepdims=True))
        r1_grad_dir = r1_expanded / np.where(r1_dist < 1e-10, 1e-10, r1_dist)
        
        # Calculate electron-electron distances and their directional gradients
        diff = r1_expanded - r2_expanded
        r12_dist = np.sqrt(np.sum(diff * diff, axis=-1, keepdims=True))
        r12_grad_dir = diff / np.where(r12_dist < 1e-10, 1e-10, r12_dist)
        
        # Calculate scaled distances
        r1_scaled = self._scaled_r(r1_dist)
        r2_scaled = self._scaled_r(np.sqrt(np.sum(r2_expanded * r2_expanded, axis=-1, keepdims=True)))
        r12_scaled = self._scaled_r(r12_dist)
        
        # Calculate gradients of scaled distances
        r1_scaled_grad = self._scaled_r_grad(r1_dist) * r1_grad_dir
        r12_scaled_grad = self._scaled_r_grad(r12_dist) * r12_grad_dir
        
        # Initialize total gradient
        total_grad = np.zeros_like(diff)
        
        # Sum up all terms
        for (m,n,o), coeff in self.coefficients.items():
            # First term: c_mno * r1^m * r2^n * r12^o
            if m > 0:  # Gradient of r1^m term
                grad = coeff * m * r1_scaled**(m-1) * r2_scaled**n * r12_scaled**o * r1_scaled_grad
                total_grad += grad
            
            if o > 0:  # Gradient of r12^o term
                grad = coeff * r1_scaled**m * r2_scaled**n * o * r12_scaled**(o-1) * r12_scaled_grad
                total_grad += grad
            
            # Second term (symmetric): c_mno * r2^m * r1^n * r12^o
            if n > 0:  # Gradient of r1^n term
                grad = coeff * n * r1_scaled**(n-1) * r2_scaled**m * r12_scaled**o * r1_scaled_grad
                total_grad += grad
            
            if o > 0:  # Gradient of r12^o term
                grad = coeff * r2_scaled**m * r1_scaled**n * o * r12_scaled**(o-1) * r12_scaled_grad
                total_grad += grad
        
        return total_grad
