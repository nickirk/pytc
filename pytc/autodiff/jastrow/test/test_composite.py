import unittest
import numpy as np
from pyscf import gto, scf
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pytc.autodiff.jastrow import NuclearCusp, REXP
from pytc.autodiff.jastrow.composite import CompositeJastrow

class TestCompositeJastrow(unittest.TestCase):
    """Test cases for CompositeJastrow class."""
    
    def setUp(self):
        """Set up H2O molecule and initialize jastrows."""
        # Create H2O molecule
        self.mol = gto.M(atom='H 0 0 1.4; O 0 0 0; H 0 0 -1.4', basis='cc-pvdz')
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        # Initialize individual jastrows
        self.ncusp = NuclearCusp.create(self.mol)
        self.rexp = REXP()
        
        # Create composite jastrow
        self.composite = CompositeJastrow.create([self.ncusp, self.rexp])
        
        # Initialize parameters
        self.ncusp_params = self.ncusp.init_params()
        self.rexp_params = self.rexp.init_params()
        self.composite_params = [self.ncusp_params, self.rexp_params]

    def test_compute_values(self):
        """Test that composite _compute matches sum of individual computes."""
        # Create points along x-axis through O atom
        x_points = jnp.linspace(-2.0, 2.0, 10)
        r2 = jnp.array([0.0, 0.0, 1.0])  # Fixed reference point
        
        for x in x_points:
            r1 = jnp.array([x, 0.0, 0.0])
            
            # Compute individual values
            ncusp_val = self.ncusp._compute(r1, r2, self.ncusp_params)
            rexp_val = self.rexp._compute(r1, r2, self.rexp_params)
            expected_sum = ncusp_val + rexp_val
            
            # Compute composite value
            composite_val = self.composite._compute(r1, r2, self.composite_params)
            
            # Compare results
            np.testing.assert_allclose(composite_val, expected_sum, rtol=1e-7,
                                     err_msg=f"Mismatch at x={x}")
            
            # Print values for inspection
            print(f"\nAt x = {x:.2f}:")
            # Handle 0-d or 1-d arrays
            ncusp_scalar = float(ncusp_val) if ncusp_val.ndim == 0 else float(ncusp_val[0])
            rexp_scalar = float(rexp_val) if rexp_val.ndim == 0 else float(rexp_val[0])
            sum_scalar = float(expected_sum) if expected_sum.ndim == 0 else float(expected_sum[0])
            composite_scalar = float(composite_val) if composite_val.ndim == 0 else float(composite_val[0])
            
            print(f"NCusp value: {ncusp_scalar:.6f}")
            print(f"REXP value: {rexp_scalar:.6f}")
            print(f"Sum: {sum_scalar:.6f}")
            print(f"Composite: {composite_scalar:.6f}")

    def test_gradients_and_laplacians(self):
        """Test that composite gradients/laplacians match sum of individuals."""
        x_points = jnp.linspace(-0.2, 0.2, 20)
        r2 = jnp.array([0.0, 0.0, 1.0])
        
        for x in x_points:
            r1 = jnp.array([x, 0.0, 0.0])
            
            # Get individual gradients and laplacians
            ncusp_grad, ncusp_lap = self.ncusp.get_log_grads_r1(r1, r2, self.ncusp_params)
            rexp_grad, rexp_lap = self.rexp.get_log_grads_r1(r1, r2, self.rexp_params)
            
            # Sum individual results
            expected_grad = ncusp_grad + rexp_grad
            expected_lap = ncusp_lap + rexp_lap
            
            # Get composite results
            composite_grad, composite_lap = self.composite.get_log_grads_r1(
                r1, r2, self.composite_params)
            
            # Compare results
            np.testing.assert_allclose(composite_grad, expected_grad, rtol=1e-7,
                                     err_msg=f"Gradient mismatch at x={x}")
            np.testing.assert_allclose(composite_lap, expected_lap, rtol=1e-7,
                                     err_msg=f"Laplacian mismatch at x={x}")
            
            # Print values
            print(f"\nAt x = {x:.2f}:")
            print(f"NCusp gradient: {ncusp_grad}")
            print(f"REXP gradient: {rexp_grad}")
            print(f"Composite gradient: {composite_grad}")
            print(f"NCusp laplacian: {ncusp_lap:.6f}")
            print(f"REXP laplacian: {rexp_lap:.6f}")
            print(f"Composite laplacian: {composite_lap:.6f}")

    def test_param_gradients(self):
        """Test that parameter gradients have correct structure and values."""
        r1 = jnp.array([0.1, 0.0, 0.0])
        r2 = jnp.array([0.0, 0.0, 1.0])
        
        # Get individual parameter gradients
        ncusp_grad = self.ncusp.grad_params(r1, r2, self.ncusp_params)
        rexp_grad = self.rexp.grad_params(r1, r2, self.rexp_params)
        
        # Get composite parameter gradients
        composite_grads = self.composite.grad_params(r1, r2, self.composite_params)
        
        # Check structure
        self.assertEqual(len(composite_grads), 2, 
                        "Composite grads should have same length as params list")
        
        # Compare values
        np.testing.assert_allclose(composite_grads[0]['rc'], ncusp_grad['rc'], 
                                 rtol=1e-7)
        np.testing.assert_allclose(composite_grads[0]['X4'], ncusp_grad['X4'], 
                                 rtol=1e-7)
        np.testing.assert_allclose(composite_grads[1]['alpha'], rexp_grad['alpha'], rtol=1e-7)

if __name__ == '__main__':
    unittest.main()
