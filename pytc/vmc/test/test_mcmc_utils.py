import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from pytc.vmc.mcmc_utils import (
    save_optimization_history, load_optimization_history,
    save_walkers, load_walkers,
)
from pytc.vmc.walker import Walker

class TestMCMCUtils(unittest.TestCase):
    def test_save_load_optimization_history(self):
        """Test round-trip save and load of VMC optimization history data to HDF5.
        
        Ensures that:
        1. Tuples, lists, and dict formats in the PyTree are correctly maintained after round-tripping.
        2. Leaves of the PyTree (across different steps) are correctly stacked sequentially (axis=0).
        3. Real and complex float arrays are seamlessly restored.
        """
        params_history = [
            # step 0
            (
                {"weight": np.array([1.0, 2.0]), "bias": np.array([0.5])},
                [np.array([1+1j, 2-1j])] # testing complex nested elements
            ),
            # step 1
            (
                {"weight": np.array([3.0, 4.0]), "bias": np.array([1.5])},
                [np.array([3+2j, 4-2j])]
            ),
            # step 2
            (
                {"weight": np.array([5.0, 6.0]), "bias": np.array([2.5])},
                [np.array([5+3j, 6-3j])]
            )
        ]
        
        data = {
            'cost': np.array([10.5, 8.2, 7.1]),
            'energies': np.array([-1.1, -1.5, -1.8]),
            'stds': np.array([0.1, 0.05, 0.02]),
            'acceptance': np.array([0.6, 0.7, 0.65]),
            'params': params_history
        }

        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            filepath = tmp.name
            
            # Save history
            saved_path = save_optimization_history(data, filepath)
            self.assertEqual(saved_path, filepath)

            # Load history
            loaded_data = load_optimization_history(filepath)

            # 1. Check top level primitives
            self.assertTrue('cost' in loaded_data)
            np.testing.assert_array_almost_equal(loaded_data['cost'], data['cost'])
            np.testing.assert_array_almost_equal(loaded_data['energies'], data['energies'])
            
            # 2. Check params structural reconstruction
            stacked_params = loaded_data['params']
            
            # Should have restored top level as a tuple
            self.assertTrue(isinstance(stacked_params, tuple), "Top level param should be a tuple")
            self.assertEqual(len(stacked_params), 2)
            
            # Element 0 of tuple is a dict
            self.assertTrue(isinstance(stacked_params[0], dict))
            self.assertIn("weight", stacked_params[0])
            self.assertIn("bias", stacked_params[0])
            
            # 3. Check axis stacking
            # weight shape should be (3, 2) since there are 3 steps and the array is len 2
            self.assertEqual(stacked_params[0]["weight"].shape, (3, 2))
            np.testing.assert_array_almost_equal(stacked_params[0]["weight"][0], np.array([1.0, 2.0]))
            np.testing.assert_array_almost_equal(stacked_params[0]["weight"][2], np.array([5.0, 6.0]))
            
            # Element 1 of tuple is a list
            self.assertTrue(isinstance(stacked_params[1], list))
            self.assertEqual(len(stacked_params[1]), 1)
            
            # 4. Check complex arrays 
            # 3 steps, 2 elements per array -> shape (3, 2)
            complex_arr = stacked_params[1][0]
            self.assertEqual(complex_arr.shape, (3, 2))
            self.assertTrue(np.iscomplexobj(complex_arr))
            np.testing.assert_array_almost_equal(complex_arr[0], np.array([1+1j, 2-1j]))
            np.testing.assert_array_almost_equal(complex_arr[2], np.array([5+3j, 6-3j]))
            
            # 5. Validate tree_map manipulation works identically across original and new
            last_params = jax.tree_util.tree_map(lambda x: x[-1], stacked_params)
            self.assertTrue(isinstance(last_params, tuple))
            np.testing.assert_array_almost_equal(last_params[0]["weight"], np.array([5.0, 6.0]))
            np.testing.assert_array_almost_equal(last_params[1][0], np.array([5+3j, 6-3j]))

    def test_save_load_walkers(self):
        """Round-trip a Walker's full state (positions plus cached
        psi/det/grad/lap fields, and the det_up/det_down (sign, log|det|)
        tuples) through save_walkers/load_walkers -- task #12(b)'s
        continued-walkers checkpoint mechanism.
        """
        n_walkers, n_alpha, n_beta = 4, 2, 1
        n_electrons = n_alpha + n_beta
        walkers = Walker(
            positions=jnp.arange(n_walkers * n_electrons * 3, dtype=jnp.float64).reshape(
                n_walkers, n_electrons, 3),
            det_up=(jnp.ones(n_walkers), jnp.full(n_walkers, -0.5)),
            det_down=(jnp.ones(n_walkers), jnp.full(n_walkers, -0.25)),
            slater_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
            slater_down=jnp.zeros((n_walkers, n_beta, n_beta)),
            inv_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
            inv_down=jnp.zeros((n_walkers, n_beta, n_beta)),
            grad_up=jnp.zeros((n_walkers, n_alpha, n_alpha, 3)),
            grad_down=jnp.zeros((n_walkers, n_beta, n_beta, 3)),
            lap_up=jnp.zeros((n_walkers, n_alpha, n_alpha)),
            lap_down=jnp.zeros((n_walkers, n_beta, n_beta)),
            move_mask=jnp.ones((n_walkers, n_electrons), dtype=bool),
            log_psi=jnp.linspace(-3.0, -1.0, n_walkers),
            psi_sign=jnp.ones(n_walkers),
            log_jastrow=jnp.linspace(0.1, 0.4, n_walkers),
        )

        with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
            filepath = tmp.name
            saved_path = save_walkers(walkers, filepath)
            self.assertEqual(saved_path, filepath)

            loaded = load_walkers(filepath)

        self.assertIsInstance(loaded, Walker)
        np.testing.assert_array_almost_equal(loaded.positions, walkers.positions)
        np.testing.assert_array_almost_equal(loaded.log_psi, walkers.log_psi)
        np.testing.assert_array_almost_equal(loaded.log_jastrow, walkers.log_jastrow)
        np.testing.assert_array_equal(loaded.move_mask, walkers.move_mask)
        # det_up/det_down are (sign, log|det|) tuples -- confirm the tuple
        # structure survives, not just the leaf values.
        self.assertIsInstance(loaded.det_up, tuple)
        self.assertEqual(len(loaded.det_up), 2)
        np.testing.assert_array_almost_equal(loaded.det_up[1], walkers.det_up[1])
        np.testing.assert_array_almost_equal(loaded.det_down[1], walkers.det_down[1])


if __name__ == '__main__':
    unittest.main()
