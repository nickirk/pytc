"""Tests for Slater determinant implementation."""

import unittest
import numpy as np
from pyscf import gto, scf
from pytc.autodiff.ansatz.det import SlaterDet  

class TestSlaterDet(unittest.TestCase):
    """Test Slater determinant implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test cases for all tests in this class."""
        # Create H2 molecule
        cls.mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g')
        cls.mol.build()
        
        # Get MO coefficients
        mf = scf.RHF(cls.mol)
        mf.kernel()
        cls.mo_coeff = mf.mo_coeff
        
        # Create a slightly more complex molecule for advanced tests
        cls.mol_water = gto.M(atom='O 0 0 0; H 0.75 0.5 0; H -0.75 0.5 0', basis='6-31g')
        cls.mol_water.build()
        mf_water = scf.RHF(cls.mol_water)
        mf_water.kernel()
        cls.mo_coeff_water = mf_water.mo_coeff
        
        # Common test coordinates
        cls.test_coords = np.array([
            [0.0, 0.0, 0.1],  # near first H
            [0.0, 0.0, 1.3],  # near second H
        ])
        
        # More electrons for water
        cls.water_coords = np.array([
            [0.1, 0.1, 0.1],  # near O
            [0.2, 0.1, 0.1],  # near O
            [0.3, 0.1, 0.1],  # near O
            [0.4, 0.1, 0.1],  # near O
            [0.5, 0.1, 0.1],  # near O
            [0.7, 0.5, 0.0],  # near H1
            [0.8, 0.5, 0.0],  # near H1
            [-0.7, 0.5, 0.0],  # near H2
            [-0.8, 0.5, 0.0],  # near H2
            [-0.9, 0.5, 0.0]   # near H2
        ])
    
    def test_init_restricted(self):
        """Test initialization with restricted orbitals."""
        det = SlaterDet(self.mol, self.mo_coeff)
        self.assertEqual(det.n_alpha, 1)
        self.assertEqual(det.n_beta, 1)
        self.assertFalse(det.unrestricted)
        self.assertIs(det.mo_coeff_alpha, det.mo_coeff_beta)
        
        # Check occupied orbital indices
        self.assertEqual(det.alpha_occ, [0])
        self.assertEqual(det.beta_occ, [0])
        
        # Check occupied MO coefficients shape
        self.assertEqual(det.mo_coeff_alpha_occ.shape, (self.mol.nao, 1))
        self.assertEqual(det.mo_coeff_beta_occ.shape, (self.mol.nao, 1))

    def test_init_unrestricted(self):
        """Test initialization with unrestricted orbitals."""
        # Simulate UHF with different coefficients
        mo_coeffs = [self.mo_coeff * 0.9, self.mo_coeff * 1.1]
        det = SlaterDet(self.mol, mo_coeffs)
        self.assertTrue(det.unrestricted)
        self.assertIsNot(det.mo_coeff_alpha, det.mo_coeff_beta)
        
        # Check coefficient values differ
        self.assertFalse(np.allclose(det.mo_coeff_alpha, det.mo_coeff_beta))

    def test_init_with_nelec(self):
        """Test initialization with custom electron counts."""
        det = SlaterDet(self.mol, self.mo_coeff, nelec=(2, 0))
        self.assertEqual(det.n_alpha, 2)
        self.assertEqual(det.n_beta, 0)
        self.assertEqual(len(det.alpha_occ), 2)
        self.assertEqual(len(det.beta_occ), 0)

    def test_init_with_excitation(self):
        """Test initialization with excitations."""
        # Water molecule has more orbitals to play with
        # Single excitation: move one alpha electron from orbital 0 to 5
        excitation = (([0], [5]), ([], []))
        det = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5), excitations=excitation)
        
        # Check the occupied orbitals
        self.assertNotIn(0, det.alpha_occ)
        self.assertIn(5, det.alpha_occ)
        self.assertEqual(len(det.alpha_occ), 5)  # Still 5 orbitals
        
        # No change in beta
        self.assertEqual(det.beta_occ, list(range(5)))

    def test_invalid_excitation(self):
        """Test that invalid excitations raise appropriate errors."""
        # Try to excite from unoccupied orbital
        with self.assertRaises(ValueError):
            SlaterDet(self.mol, self.mo_coeff, nelec=(1,1), 
                     excitations=(([2], [3]), ([], [])))
        
        # Try to excite to occupied orbital
        with self.assertRaises(ValueError):
            SlaterDet(self.mol, self.mo_coeff, nelec=(1,1), 
                     excitations=(([0], [0]), ([], [])))
        
        # Mismatched from/to indices
        with self.assertRaises(ValueError):
            SlaterDet(self.mol, self.mo_coeff, nelec=(1,1), 
                     excitations=(([0], [1, 2]), ([], [])))

    def test_determinant_value(self):
        """Test basic determinant evaluation."""
        det = SlaterDet(self.mol, self.mo_coeff)
        value = det.value(self.test_coords)
        self.assertIsInstance(value, (float, np.ndarray))
        self.assertNotEqual(value, 0.0)
        
        # Test __call__ convenience method
        call_value = det(self.test_coords)
        np.testing.assert_allclose(value, call_value)

    def test_matrix_shape(self):
        """Test shape of Slater matrices."""
        det = SlaterDet(self.mol, self.mo_coeff, nelec=(1, 1))
        slater_up, slater_down = det.matrix(self.test_coords)
        
        self.assertEqual(slater_up.shape, (1, 1, 1))
        self.assertEqual(slater_down.shape, (1, 1, 1))
        
        # Test with more electrons
        det_water = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5))
        water_up, water_down = det_water.matrix(self.water_coords)
        
        self.assertEqual(water_up.shape, (1, 5, 5))
        self.assertEqual(water_down.shape, (1, 5, 5))

    def test_update_mechanism(self):
        """Test the update mechanism for moving electrons."""
        det = SlaterDet(self.mol, self.mo_coeff)
        
        # Initialize with batch dimension
        batch_coords = self.test_coords[np.newaxis, :, :]
        
        # Create move mask for first electron
        move_mask = np.zeros((1, 2), dtype=bool)
        move_mask[0, 0] = True
        
        # Get initial value
        init_value = det.value(batch_coords)
        
        # Move up electron with mask
        new_pos = np.array([0.1, 0.1, 0.1])
        new_coords = batch_coords.copy()
        new_coords[0, 0] = new_pos
        
        # Test with move mask
        value_with_mask = det.value(new_coords, move_mask)
        
        # Test without mask (full recalculation)
        value_without_mask = det.value(new_coords)
        
        # Values should match regardless of method
        np.testing.assert_allclose(value_with_mask, value_without_mask)

    def test_batched_one_electron_moves(self):
        """Test batched one-electron moves with move masks."""
        det = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5))
        
        # Create batch of 3 walkers
        batch_coords = np.stack([self.water_coords] * 3)
        
        # Move different electrons in each walker
        move_mask = np.zeros((3, 10), dtype=bool)
        move_mask[0, 0] = True   # Move first electron in first walker
        move_mask[1, 4] = True   # Move fifth electron in second walker
        move_mask[2, 8] = True   # Move ninth electron in third walker
        
        # Make moves
        shifts = np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [-0.1, -0.1, -0.1]
        ])
        
        new_coords = batch_coords.copy()
        for i in range(3):
            electron_idx = np.where(move_mask[i])[0][0]
            new_coords[i, electron_idx] += shifts[i]
        
        # Compare values with and without mask
        values_with_mask = det.value(new_coords, move_mask)
        values_without_mask = det.value(new_coords)
        
        np.testing.assert_allclose(values_with_mask, values_without_mask)
        
    def test_sequential_one_electron_moves(self):
        """Test sequence of one-electron moves."""
        det = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5))
        coords = self.water_coords[np.newaxis, :, :]
        
        # Make series of moves
        moves = [(0, [0.1, 0.1, 0.1]), 
                (4, [-0.1, 0.2, 0.0]),
                (8, [0.3, -0.1, 0.2])]
        
        current_coords = coords.copy()
        current_value = det.value(current_coords)
        
        for electron_idx, shift in moves:
            # Create move mask for this electron
            move_mask = np.zeros((1, 10), dtype=bool)
            move_mask[0, electron_idx] = True
            
            # Apply move
            new_coords = current_coords.copy()
            new_coords[0, electron_idx] += shift
            
            # Compare values with and without mask
            value_with_mask = det.value(new_coords, move_mask)
            value_without_mask = det.value(new_coords)
            
            np.testing.assert_allclose(value_with_mask, value_without_mask)
            
            # Update for next move
            current_coords = new_coords
            current_value = value_with_mask

    def test_value_sign_change(self):
        """Test if determinant changes sign when electrons are exchanged."""
        # Need 2 electrons of same spin to test exchange
        det = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(2, 0))
        coords1 = self.water_coords[:2]  # Just take first two electrons
        coords2 = np.array([coords1[1], coords1[0]])  # Exchange positions
        
        val1 = det.value(coords1)
        val2 = det.value(coords2)
        
        # Determinant should change sign when two rows are swapped
        # Handle both scalar and array outputs
        if isinstance(val1, np.ndarray):
            np.testing.assert_allclose(val1, -val2)
        else:
            self.assertAlmostEqual(val1, -val2, places=10)

    def test_boundary_conditions(self):
        """Test behavior at large distances."""
        det = SlaterDet(self.mol, self.mo_coeff)
        far_coords = np.array([
            [0.0, 0.0, 10.0],  # far from molecule
            [0.0, 0.0, -10.0]   # far from molecule
        ])
        value = det.value(far_coords)
        # Determinant should decay to zero far from molecule
        value = value[0] if isinstance(value, np.ndarray) else value
        self.assertLess(abs(value), 1e-3)

    def test_numerical_gradient(self):
        """Test gradient against numerical differentiation."""
        det = SlaterDet(self.mol, self.mo_coeff)
        eps = 1e-5
        coords = self.test_coords
        
        # Get analytical gradient and matrix
        (matrix_up, matrix_down), (grad_up, grad_down) = det.grad(coords)
        
        # Compute numerical gradient for first electron, x direction
        d = 0  # x-direction
        e_idx = 0  # first electron
        
        # Get the Slater matrix at the original position - already computed above
        slater_up_orig = matrix_up
        
        # Compute numerical derivative using central difference
        h = np.zeros(3)
        h[d] = eps
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[e_idx] += h
        coords_minus[e_idx] -= h
        
        slater_up_plus, _ = det.matrix(coords_plus)
        slater_up_minus, _ = det.matrix(coords_minus)
        
        numeric_grad = (slater_up_plus - slater_up_minus) / (2 * eps)
        
        # Compare numerical vs analytical for this specific element
        self.assertAlmostEqual(
            grad_up[0, e_idx, 0, d],  # [electron, orbital, direction]
            numeric_grad[e_idx, 0],  # [electron, orbital]
            places=3
        )

    def test_laplacian(self):
        """Test Laplacian calculation."""
        det = SlaterDet(self.mol, self.mo_coeff)
        
        # Get all derivatives at once
        (matrix_up, matrix_down), (grad_up, grad_down), (lapl_up, lapl_down) = det.laplacian(self.test_coords)
        
        # Check shapes
        self.assertEqual(matrix_up.shape, (1, 1, 1))
        self.assertEqual(matrix_down.shape, (1, 1, 1))
        self.assertEqual(grad_up.shape, (1, 1, 1, 3))
        self.assertEqual(grad_down.shape, (1, 1, 1, 3))
        self.assertEqual(lapl_up.shape, (1, 1, 1))
        self.assertEqual(lapl_down.shape, (1, 1, 1))
        
        # Verify consistency with individual calls
        matrix_only_up, matrix_only_down = det.matrix(self.test_coords)
        np.testing.assert_allclose(matrix_up, matrix_only_up)
        np.testing.assert_allclose(matrix_down, matrix_only_down)
        
        (matrix_grad_up, matrix_grad_down), (grad_only_up, grad_only_down) = det.grad(self.test_coords)
        np.testing.assert_allclose(matrix_up, matrix_grad_up)
        np.testing.assert_allclose(matrix_down, matrix_grad_down)
        np.testing.assert_allclose(grad_up, grad_only_up)
        np.testing.assert_allclose(grad_down, grad_only_down)

    def test_excitation_det_value(self):
        """Test determinant value with excitation."""
        # Regular determinant
        det_normal = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5))
        
        # Excited determinant (HOMO → LUMO)
        det_excited = SlaterDet(self.mol_water, self.mo_coeff_water, nelec=(5, 5),
                              excitations=(([4], [5]), ([], [])))  
        
        # Values should be different
        val_normal = det_normal.value(self.water_coords)
        val_excited = det_excited.value(self.water_coords)
        self.assertNotEqual(val_normal, val_excited)

    def test_parallel_det_speedup(self):
        """Test that parallel determinant evaluation with threading."""
        import time
        import os
        from pytc.lib import np_helper
        
        # Skip if CPU count is too low
        cpu_count = os.cpu_count()
        if cpu_count is None or cpu_count < 4:
            self.skipTest("Need at least 4 CPU cores for meaningful parallel test")
        
        # Create a large batch of random matrices
        batch_size = 100_000
        matrix_size = 50
        rng = np.random.default_rng(42)
        matrices = rng.normal(size=(batch_size, matrix_size, matrix_size))
        
        # Warm up
        _ = np_helper.batched_det(matrices[:100], parallel=True)
        
        # Time with 1 thread (serial)
        start = time.perf_counter()
        det_serial = np_helper.batched_det(matrices, parallel=False)
        time_1_thread = time.perf_counter() - start
        
        # Time with 8 threads (parallel)
        start = time.perf_counter()
        det_parallel = np_helper.batched_det(matrices, parallel=True)
        time_8_threads = time.perf_counter() - start
        
        # Verify results match
        np.testing.assert_allclose(det_serial, det_parallel, rtol=1e-10, atol=1e-12)
        
        # Check speedup
        speedup = time_1_thread / time_8_threads
        print(f"\n1 thread:  {time_1_thread:.3f}s")
        print(f"8 threads: {time_8_threads:.3f}s")
        print(f"Speedup:   {speedup:.2f}x")
        
        # Parallel should be at least as fast (may not be much faster due to BLAS threading)
        self.assertLessEqual(time_8_threads, time_1_thread * 1.1, 
                            f"8 threads slower than 1 thread: {speedup:.2f}x")

    def test_laplacian_with_walker(self):
        """Test laplacian function with Walker dataclass for selective updates."""
        import jax
        import jax.numpy as jnp
        from pytc.autodiff.mcmc import Walker
        from pytc.autodiff.ansatz.det import laplacian
        
        # Enable 64-bit precision for JAX
        jax.config.update("jax_enable_x64", True)
        
        # Create SlaterDet (nelec is a tuple)
        det = SlaterDet(self.mol, self.mo_coeff, nelec=(1, 1))
        
        # Initialize walker with 3 walkers manually (without needing full ansatz)
        n_walkers = 3
        n_electrons = 2
        positions = jnp.array([
            [[0.0, 0.0, 0.1], [0.0, 0.0, 1.3]],  # Walker 0: different positions for each electron
            [[0.1, 0.1, 0.2], [0.1, 0.1, 1.4]],  # Walker 1
            [[0.2, 0.2, 0.3], [0.2, 0.2, 1.5]]   # Walker 2
        ])
        
        # Verify positions are different
        assert not np.allclose(positions[0, 0], positions[0, 1]), "Electrons should have different positions"
        
        # Manually create Walker with zeros (simulating uninitialized state)
        walker = Walker(
            positions=positions,
            slater_up=jnp.zeros((n_walkers, det.n_alpha, det.n_alpha)),
            slater_down=jnp.zeros((n_walkers, det.n_beta, det.n_beta)),
            inv_up=jnp.zeros((n_walkers, det.n_alpha, det.n_alpha)),
            inv_down=jnp.zeros((n_walkers, det.n_beta, det.n_beta)),
            det_up=jnp.zeros((n_walkers,)),
            det_down=jnp.zeros((n_walkers,)),
            grad_up=jnp.zeros((n_walkers, det.n_alpha, det.n_alpha, 3)),
            grad_down=jnp.zeros((n_walkers, det.n_beta, det.n_beta, 3)),
            lap_up=jnp.zeros((n_walkers, det.n_alpha, det.n_alpha)),
            lap_down=jnp.zeros((n_walkers, det.n_beta, det.n_beta)),
            move_mask=jnp.ones((n_walkers, n_electrons), dtype=bool)
        )
        
        # First call should trigger full recomputation (grad/lap uninitialized)
        (matrix_up_1, matrix_down_1), (grad_up_1, grad_down_1), (lap_up_1, lap_down_1), updated_walker_1 = laplacian(det, walker)
        
        # Verify shapes
        self.assertEqual(matrix_up_1.shape, (n_walkers, det.n_alpha, det.n_alpha))
        self.assertEqual(matrix_down_1.shape, (n_walkers, det.n_beta, det.n_beta))
        self.assertEqual(grad_up_1.shape, (n_walkers, det.n_alpha, det.n_alpha, 3))
        self.assertEqual(grad_down_1.shape, (n_walkers, det.n_beta, det.n_beta, 3))
        self.assertEqual(lap_up_1.shape, (n_walkers, det.n_alpha, det.n_alpha))
        self.assertEqual(lap_down_1.shape, (n_walkers, det.n_beta, det.n_beta))
        
        # Verify grad/lap are not all zeros after initialization
        self.assertFalse(np.allclose(grad_up_1, 0.0))
        self.assertFalse(np.allclose(lap_up_1, 0.0))
        
        # Verify updated_walker has non-zero grad/lap
        self.assertFalse(np.allclose(updated_walker_1.grad_up, 0.0))
        self.assertFalse(np.allclose(updated_walker_1.lap_up, 0.0))
        
        # Now simulate a move: update positions and set move_mask
        new_positions = positions.at[0, 0].set(jnp.array([0.05, 0.05, 0.15]))  # Move first electron of first walker
        move_mask = jnp.zeros((n_walkers, n_electrons), dtype=bool)
        move_mask = move_mask.at[0, 0].set(True)
        
        # Update walker with new positions and move_mask, keeping grad/lap from previous call
        walker_with_move = updated_walker_1.replace(
            positions=new_positions,
            move_mask=move_mask
        )
        
        # Need to call value() first to update Slater matrices
        from pytc.autodiff.ansatz.det import value
        _, walker_with_updated_matrices = value(det, walker_with_move)
        
        # Now call laplacian with updated matrices
        (matrix_up_2, matrix_down_2), (grad_up_2, grad_down_2), (lap_up_2, lap_down_2), updated_walker_2 = laplacian(det, walker_with_updated_matrices)
        
        # For H2 with (1,1) electrons:
        # - grad_up has shape (n_walkers, 1, 1, 3) - gradient for 1 alpha electron at 1 alpha MO
        # - grad_down has shape (n_walkers, 1, 1, 3) - gradient for 1 beta electron at 1 beta MO  
        # When we move electron 0 (alpha), only grad_up should change
        # When we move electron 1 (beta), only grad_down should change
        
        # Verify shapes
        self.assertEqual(grad_up_1.shape, (n_walkers, 1, 1, 3))
        self.assertEqual(grad_down_1.shape, (n_walkers, 1, 1, 3))
        
        # Walkers 1 and 2 should be completely unchanged (no moves)
        np.testing.assert_allclose(grad_up_2[1], grad_up_1[1], rtol=1e-10)
        np.testing.assert_allclose(grad_up_2[2], grad_up_1[2], rtol=1e-10)
        np.testing.assert_allclose(lap_up_2[1], lap_up_1[1], rtol=1e-10)
        np.testing.assert_allclose(lap_up_2[2], lap_up_1[2], rtol=1e-10)
        
        # Walker 0: alpha electron (electron 0) moved, so grad_up should change
        self.assertFalse(np.allclose(grad_up_2[0, 0, 0], grad_up_1[0, 0, 0], rtol=1e-10))
        self.assertFalse(np.allclose(lap_up_2[0, 0, 0], lap_up_1[0, 0, 0], rtol=1e-10))
        
        # Walker 0: beta electron (electron 1) did NOT move, so grad_down should be unchanged
        np.testing.assert_allclose(grad_down_2[0], grad_down_1[0], rtol=1e-10)
        np.testing.assert_allclose(lap_down_2[0], lap_down_1[0], rtol=1e-10)

if __name__ == '__main__':
    unittest.main()
