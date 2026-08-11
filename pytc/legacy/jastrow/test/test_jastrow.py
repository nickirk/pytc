"""Test for Jastrow factors."""

import unittest
import numpy as np
from pytc.legacy.jastrow import Jastrow, SM7


class SimpleTestJastrow(Jastrow):
    """Simple Jastrow factor for testing: f(r) = exp(-alpha*r)."""
    
    def __call__(self, r1, r2, atomic_positions=None):
        """Evaluate Jastrow factor at given positions."""
        r1 = np.atleast_2d(r1)  # Ensure 2D array with shape (N, 3)
        r2 = np.atleast_2d(r2)  # Ensure 2D array with shape (M, 3)
        
        delta_r = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]
        result = np.exp(-self.params[0] * np.linalg.norm(delta_r, axis=-1))
        
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
        
        jastrow_values = self.__call__(r1, r2)[..., np.newaxis]
        return -self.params[0] * diff / norm * jastrow_values

    def _process_grad_batch(self, r1_batch, r2):
        """Implement the abstract batch hook used by the legacy base class."""
        diff = r1_batch[:, np.newaxis, :] - r2[np.newaxis, :, :]
        norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        values = self.__call__(r1_batch, r2)[..., np.newaxis]
        return -self.params[0] * diff / norm * values


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
        cls.jastrow = SimpleTestJastrow([0.5])  # alpha = 0.5
    
    def test_eval_shape(self):
        """Test if eval returns correct shape."""
        values = self.jastrow(self.grid_points, self.grid_points)
        n_points = len(self.grid_points)
        self.assertEqual(values.shape, (n_points, n_points))
    
    def test_grad_shape(self):
        """Test if grad returns correct shape."""
        gradients = self.jastrow.grad(self.grid_points)
        n_points = len(self.grid_points)
        self.assertEqual(gradients.shape, (n_points, n_points, 3))

        r1 = np.random.rand(5, 3)
        r2 = np.random.rand(7, 3)
        gradients = self.jastrow.grad(r1, r2)
        self.assertEqual(gradients.shape, (5, 7, 3))
    
    def test_eval_symmetry(self):
        """Test if Jastrow factor is symmetric: f(r₁-r₂) = f(r₂-r₁)."""
        values = self.jastrow(self.grid_points, self.grid_points)
        n_points = len(self.grid_points)
        for i in range(n_points):
            for j in range(n_points):
                self.assertAlmostEqual(values[i,j], values[j,i], places=10)
    
    def test_grad_antisymmetry(self):
        """Test if gradient is antisymmetric: ∇₁f(r₁-r₂) = -∇₂f(r₁-r₂)."""
        gradients = self.jastrow.grad(self.grid_points)
        n_points = len(self.grid_points)
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    np.testing.assert_array_almost_equal(
                        gradients[i, j], 
                        -gradients[j, i],
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
                f_forward = self.jastrow(point1[np.newaxis, :] + h, point2[np.newaxis, :])[0, 0]
                f_backward = self.jastrow(point1[np.newaxis, :] - h, point2[np.newaxis, :])[0, 0]
                grad[d] = (f_forward - f_backward) / (2 * eps)
            return grad
        
        test_points = self.grid_points[:3]
        
        n_points = len(test_points)
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    numerical = numerical_gradient(
                        test_points[i], 
                        test_points[j]
                    )
                    analytical = self.jastrow.grad(test_points[i:i+1], test_points[j:j+1])[0, 0]
                    np.testing.assert_array_almost_equal(
                        numerical, 
                        analytical,
                        decimal=5,
                        err_msg=f"Failed at points {i},{j}"
                    )
    
    def test_cusp_condition(self):
        """Test if Jastrow factor satisfies cusp condition as r→0 (but r≠0)."""
        eps_values = [1e-5, 1e-6, 1e-7, 1e-8]
        directions = [
            [1.0, 0.0, 0.0],  # x-axis
            [0.0, 1.0, 0.0],  # y-axis
            [0.0, 0.0, 1.0],  # z-axis
            [1.0, 1.0, 1.0],  # diagonal
            [1.0, -1.0, 0.5], # arbitrary direction
        ]
        
        for eps in eps_values:
            for direction in directions:
                direction = np.array(direction) / np.linalg.norm(direction)
                points = np.array([
                    [0.0, 0.0, 0.0],
                    eps * direction
                ])
                gradients = self.jastrow.grad(points)  # Shape: (N_grid, N_grid, 3)
                grad_norm = np.linalg.norm(gradients[0, 1])
                self.assertAlmostEqual(
                    grad_norm, 
                    self.jastrow.params[0], places=5, 
                    msg=f"Cusp condition failed for eps={eps}, direction={direction}")

    def test_flexible_input(self):
        """Test if Jastrow accepts different input shapes."""
        point1 = np.array([1.0, 0.0, 0.0])
        point2 = np.array([0.0, 1.0, 0.0])
        value = self.jastrow(point1, point2)
        self.assertEqual(value.shape, (1, 1))

        batch1 = np.random.rand(5, 3)
        batch2 = np.random.rand(7, 3)
        value = self.jastrow(batch1, batch2)
        self.assertEqual(value.shape, (5, 7))

        atoms = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        value = self.jastrow(batch1, batch2, atoms)
        self.assertEqual(value.shape, (5, 7))
                
class TestSM7(unittest.TestCase):
    """Test SM7 Jastrow implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test cases."""
        from pytc.legacy.jastrow import SM7
        
        cls.jastrow = SM7(atom='He')
        
        cls.grid_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

    def test_eval_shape(self):
        """Test if eval returns correct shape."""
        values = self.jastrow(self.grid_points, self.grid_points)
        n_points = len(self.grid_points)
        self.assertEqual(values.shape, (n_points, n_points))

    def test_grad_shape(self):
        """Test if grad returns correct shape."""
        gradients = self.jastrow.grad(self.grid_points)
        n_points = len(self.grid_points)
        self.assertEqual(gradients.shape, (n_points, n_points, 3))

    def test_eval_symmetry(self):
        """Test if Jastrow factor is symmetric: f(r₁,r₂) = f(r₂,r₁)."""
        values = self.jastrow(self.grid_points, self.grid_points)
        n_points = len(self.grid_points)
        for i in range(n_points):
            for j in range(n_points):
                self.assertAlmostEqual(values[i,j], values[j,i], places=10)

    def test_numerical_gradient(self):
        """Test gradient against numerical differentiation."""
        eps = 1e-6
        
        def numerical_gradient(point1, point2):
            grad = np.zeros(3)
            for d in range(3):
                h = np.zeros(3)
                h[d] = eps
                f_forward = self.jastrow(point1[np.newaxis, :] + h, point2[np.newaxis, :])[0, 0]
                f_backward = self.jastrow(point1[np.newaxis, :] - h, point2[np.newaxis, :])[0, 0]
                grad[d] = (f_forward - f_backward) / (2 * eps)
            return grad
        
        test_points = self.grid_points[:2]
        
        numerical = numerical_gradient(test_points[0], test_points[1])
        analytical = self.jastrow.grad(test_points[0:1], test_points[1:2])[0, 0]
        np.testing.assert_array_almost_equal(numerical, analytical, decimal=5)

class TestSM7Specific(unittest.TestCase):
    """Additional tests specific to SM7 implementation."""
    
    def setUp(self):
        """Set up test case."""
        self.jastrow = SM7(atom='He')
        self.test_points = np.array([
            [0.0, 0.0, 0.0],  # nucleus
            [0.01, 0.0, 0.0],  # near nucleus
            [1.0, 0.0, 0.0],  # unit distance
            [5.0, 0.0, 0.0],  # far from nucleus
        ])
    
    def test_electron_electron_scaling(self):
        """Test electron-electron correlation scaling."""
        r1 = np.array([[0.0, 0.0, 0.0]])
        r2_points = np.array([
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0]
        ])
        
        values = []
        for r2 in r2_points:
            val = self.jastrow(r1, r2[None,:])[0,0]
            values.append(val)
        
        values = np.array(values)
        self.assertTrue(np.all(np.diff(values) < 0))
    
    def test_electron_electron_cusp(self):
        """Test electron-electron cusp condition.
        
        For same-spin electrons: ∂f/∂r12|_{r12=0} = 1/4
        For opposite-spin electrons: ∂f/∂r12|_{r12=0} = 1/2
        """
        eps_values = [1e-5, 1e-6, 1e-7]
        directions = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]/np.sqrt(3)
        ]
        
        expected_cusp = 0.5
        
        for eps in eps_values:
            for direction in directions:
                direction = np.array(direction)
                r1 = np.array([[0.0, 0.0, 0.0]])
                r2 = eps * direction[None,:]
                
                grad = self.jastrow.grad(r1, r2)[0,0]
                grad_norm = np.linalg.norm(grad)
                
                self.assertAlmostEqual(
                    grad_norm, 
                    expected_cusp, 
                    places=2,
                    msg=f"Electron-electron cusp condition failed for eps={eps}, direction={direction}"
                )
                
                expected_direction = -direction
                calculated_direction = grad / grad_norm
                np.testing.assert_array_almost_equal(
                    calculated_direction,
                    expected_direction,
                    decimal=2
                )

if __name__ == '__main__':
    unittest.main()
