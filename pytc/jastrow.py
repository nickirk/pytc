"""This module implements different types of Jastrow factors."""

import numpy as np
from abc import ABC, abstractmethod

class Jastrow(ABC):
    """Base class for Jastrow factors."""

    def __init__(self, params=None):
        """Initialize the Jastrow factor with parameters."""
        self.params = params

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
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    
    def __call__(self, r1, r2, atomic_positions=None):
        """Evaluate Jastrow factor at given positions."""
        r1 = np.atleast_2d(r1)  # Ensure 2D array with shape (N, 3)
        r2 = np.atleast_2d(r2)  # Ensure 2D array with shape (M, 3)
        
        delta_r = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        result = 0.5*1./self.params[0]*np.exp(-self.params[0] * np.linalg.norm(delta_r, axis=-1))
        
        # Handle single point inputs
        if r1.shape[0] == 1 and r2.shape[0] == 1:
            result = result.reshape(1, 1)
            
        return result
    
    def grad(self, r1, r2=None, atomic_positions=None):
        """Compute gradient with respect to coordinates."""
        r1 = np.atleast_2d(r1)
        r2 = np.atleast_2d(r2) if r2 is not None else r1
        
        diff = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        
        # Note: Use the reshaped call result for proper broadcasting
        jastrow_values = self.__call__(r1, r2)[..., np.newaxis]
        return -0.5*self.params[0] * diff / norm * jastrow_values

class SM7(Jastrow):
    """Schmidt-Moskowitz 7-parameter Jastrow factor."""
    
    # Class-level coefficient table
    _coeff_table = {
        'He': {(0,0,1): 0.50000, (0,0,2): 0.50516, (0,0,3): -0.19313, (0,0,4): 0.30276,
               (2,0,0): -0.16995, (3,0,0): -0.34505, (4,0,0): -0.54777},
        'Li': {(0,0,1): 0.50000, (0,0,2): 0.03104, (0,0,3): 0.48928, (0,0,4): -0.62908,
               (2,0,0): -0.07185, (3,0,0): -0.48761, (4,0,0): 0.40450},
        'Be': {(0,0,1): 0.50000, (0,0,2): -0.05254, (0,0,3): 0.15355, (0,0,4): -0.30549,
               (2,0,0): -0.11928, (3,0,0): -0.17144, (4,0,0): 0.16652},
        'B':  {(0,0,1): 0.50000, (0,0,2): -0.13852, (0,0,3): -0.06687, (0,0,4): -0.02026,
               (2,0,0): -0.12573, (3,0,0): -0.05320, (4,0,0): 0.06421},
        'C':  {(0,0,1): 0.50000, (0,0,2): -0.14368, (0,0,3): -0.34102, (0,0,4): 0.30267,
               (2,0,0): -0.12272, (3,0,0): -0.05622, (4,0,0): 0.08462},
        'N':  {(0,0,1): 0.50000, (0,0,2): -0.41390, (0,0,3): 0.10406, (0,0,4): 0.06374,
               (2,0,0): -0.12400, (3,0,0): 0.01909, (4,0,0): -0.00383},
        'O':  {(0,0,1): 0.50000, (0,0,2): -0.57077, (0,0,3): 0.44725, (0,0,4): -0.16075,
               (2,0,0): -0.11696, (3,0,0): -0.01442, (4,0,0): 0.03312},
        'F':  {(0,0,1): 0.50000, (0,0,2): -0.73946, (0,0,3): 0.81463, (0,0,4): -0.41861,
               (2,0,0): -0.11872, (3,0,0): -0.01973, (4,0,0): 0.02779},
        'Ne': {(0,0,1): 0.50000, (0,0,2): -0.79266, (0,0,3): 1.05232, (0,0,4): -0.65615,
               (2,0,0): -0.13312, (3,0,0): -0.00131, (4,0,0): 0.09083}
    }

    def __init__(self, params=None, atom=None):
        """Initialize with either explicit parameters or atom name."""
        if atom is not None:
            if atom not in self._coeff_table:
                raise ValueError(f"Parameters not available for atom: {atom}")
            params = self._coeff_table[atom]
        elif params is None:
            raise ValueError("Either params or atom must be provided")
        super().__init__(params)
    
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
        for (m,n,o), coeff in self.params.items():
            if m == n: 
                coeff_ = 0.5 * coeff
            term = coeff_ * (r1_scaled**m * r2_scaled**n + r2_scaled**m * r1_scaled**n) * r12_scaled**o
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
        for (m,n,o), coeff in self.params.items():
            if m == n: 
                coeff_ = 0.5 * coeff
            else:
                coeff_ = coeff
            # First term: c_mno * r1^m * r2^n * r12^o
            if m > 0:  # Gradient of r1^m term
                grad = coeff_ * m * r1_scaled**(m-1) * r2_scaled**n * r12_scaled**o * r1_scaled_grad
                total_grad += grad
            
            if o > 0:  # Gradient of r12^o term
                grad = coeff_ * r1_scaled**m * r2_scaled**n * o * r12_scaled**(o-1) * r12_scaled_grad
                total_grad += grad
            
            # Second term (symmetric): c_mno * r2^m * r1^n * r12^o
            if n > 0:  # Gradient of r1^n term
                grad = coeff_ * n * r1_scaled**(n-1) * r2_scaled**m * r12_scaled**o * r1_scaled_grad
                total_grad += grad
            
            if o > 0:  # Gradient of r12^o term
                grad = coeff_ * r2_scaled**m * r1_scaled**n * o * r12_scaled**(o-1) * r12_scaled_grad
                total_grad += grad
        
        return total_grad
