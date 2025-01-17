"""This module implements different types of Jastrow factors."""

import numpy as np


class Jastrow:
    """Base class for Jastrow factors."""

    def __init__(self, parameters):
        """Initialize the Jastrow factor with a list of parameters."""
        self.parameters = parameters
        # Cache for storing computed values
        self._cache = {}

    def eval(self, grid_points, weights=None):
        """Evaluate the Jastrow function on the grid points using broadcasting.
        
        Args:
            grid_points: Array of shape (N_grid, 3) containing the grid points
            weights: Array of shape (N_grid,) containing the weights for each grid point
        """
        grid_points = np.array(grid_points)
        delta_r = grid_points[:, np.newaxis, :] - grid_points[np.newaxis, :, :]
        values = self.jastrow_function(delta_r)
        np.fill_diagonal(values, 0)  # Set diagonal to zero since i == j
        return values

    def grad(self, grid_points, weights=None):
        """Evaluate the gradient of the Jastrow function with respect to each grid point using broadcasting.
        
        Args:
            grid_points: Array of shape (N_grid, 3) containing the grid points
            weights: Array of shape (N_grid,) containing the weights for each grid point
        """
        # Convert to array and get a hashable key for caching
        grid_points = np.array(grid_points)
        cache_key = grid_points.tobytes()
        
        # Check if result is in cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # If not in cache, compute gradients
        delta_r = grid_points[:, np.newaxis, :] - grid_points[np.newaxis, :, :]
        gradients = self.jastrow_gradient(delta_r)
        np.fill_diagonal(gradients[:, :, 0], 0)  # Set diagonal to zero for x component
        np.fill_diagonal(gradients[:, :, 1], 0)  # Set diagonal to zero for y component
        np.fill_diagonal(gradients[:, :, 2], 0)  # Set diagonal to zero for z component
        
        # Store in cache and return
        self._cache[cache_key] = gradients
        return gradients

    def jastrow_function(self, delta_r):
        """Evaluate the Jastrow function using broadcasting."""
        return np.exp(-np.linalg.norm(delta_r, axis=-1))

    def jastrow_gradient(self, delta_r):
        """Evaluate the gradient of the Jastrow function using broadcasting."""
        norm = np.linalg.norm(delta_r, axis=-1, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero
        return -delta_r / norm * np.exp(-norm)
