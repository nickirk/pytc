"""Test for Jastrow factors."""

import unittest
import numpy as np
from pytcint.jastrow import Jastrow


class SimpleJastrow(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    def jastrow_function(self, delta_r):
        return np.exp(-self.parameters[0] * np.linalg.norm(delta_r, axis=-1))
    
    def jastrow_gradient(self, delta_r):
        norm = np.linalg.norm(delta_r, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)  # Avoid division by zero
        return -self.parameters[0] * delta_r / norm * self.jastrow_function(delta_r)[..., np.newaxis]


class TestJastrow(unittest.TestCase):
    """Test Jastrow class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test cases for all tests in this class."""
        cls.grid_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        cls.jastrow = SimpleJastrow([0.5])  # alpha = 0.5
    
    def test_eval_shape(self):
        """Test if eval returns correct shape."""
        values = self.jastrow.eval(self.grid_points)
        n_points = len(self.grid_points)
        self.assertEqual(values.shape, (n_points, n_points))
    
    def test_grad_shape(self):
        """Test if grad returns correct shape."""
        gradients = self.jastrow.grad(self.grid_points)
        n_points = len(self.grid_points)
        self.assertEqual(gradients.shape, (n_points, n_points, 3))
    
    def test_eval_symmetry(self):
        """Test if Jastrow factor is symmetric: f(r₁-r₂) = f(r₂-r₁)."""
        values = self.jastrow.eval(self.grid_points)
        for i in range(len(self.grid_points)):
            for j in range(len(self.grid_points)):
                self.assertAlmostEqual(values[i,j], values[j,i], places=10)
    
    def test_grad_antisymmetry(self):
        """Test if gradient is antisymmetric: ∇₁f(r₁-r₂) = -∇₂f(r₁-r₂)."""
        gradients = self.jastrow.grad(self.grid_points)
        for i in range(len(self.grid_points)):
            for j in range(len(self.grid_points)):
                if i != j:
                    np.testing.assert_array_almost_equal(
                        gradients[i,j], 
                        -gradients[j,i],
                        decimal=10
                    )
    
    def test_numerical_gradient(self):
        """Test gradient against numerical differentiation."""
        eps = 1e-6
        
        def numerical_gradient(point1, point2):
            """Compute gradient numerically using central difference."""
            grad = np.zeros(3)
            for d in range(3):
                h = np.zeros(3)
                h[d] = eps
                forward = self.jastrow.jastrow_function(point1 + h - point2)
                backward = self.jastrow.jastrow_function(point1 - h - point2)
                grad[d] = (forward - backward) / (2*eps)
            return grad
        
        for i in range(len(self.grid_points)):
            for j in range(len(self.grid_points)):
                if i != j:
                    numerical = numerical_gradient(
                        self.grid_points[i], 
                        self.grid_points[j]
                    )
                    analytical = self.jastrow.grad(self.grid_points)[i,j]
                    np.testing.assert_array_almost_equal(
                        numerical, 
                        analytical,
                        decimal=5
                    )
    
    def test_cusp_condition(self):
        """Test if Jastrow factor satisfies cusp condition as r→0 (but r≠0).
        
        For simple exponential Jastrow f(r)=exp(-alpha*r), 
        the gradient magnitude should approach alpha as r→0.
        We test this by evaluating at small but nonzero distances
        in various directions to ensure isotropic behavior.
        """
        # Test with displacements in various directions
        eps_values = [1e-5, 1e-6, 1e-7, 1e-8]
        
        # Test along single axes
        directions = [
            [1.0, 0.0, 0.0],  # x-axis
            [0.0, 1.0, 0.0],  # y-axis
            [0.0, 0.0, 1.0],  # z-axis
            [1.0, 1.0, 1.0],  # diagonal
            [1.0, -1.0, 0.5], # arbitrary direction
        ]
        
        for eps in eps_values:
            for direction in directions:
                # Normalize direction vector
                direction = np.array(direction) / np.linalg.norm(direction)
                # Create points with small displacement in given direction
                points = np.array([
                    [0.0, 0.0, 0.0],
                    eps * direction  # Small displacement in given direction
                ])
                gradients = self.jastrow.grad(points)
                # Check gradient at first point with respect to second point
                grad_norm = np.linalg.norm(gradients[0,1])
                self.assertAlmostEqual(
                    grad_norm, 
                    self.jastrow.parameters[0], 
                    places=5,
                    msg=f"Cusp condition failed for eps={eps}, direction={direction}"
                )


if __name__ == '__main__':
    unittest.main()



