import unittest
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
try:
    from jax.experimental.shard_map import shard_map
except ImportError:
    from jax.shard_map import shard_map
import folx
from pytc.utils.folx_fix import apply_fix

# Apply the fix
apply_fix()

jax.config.update("jax_enable_x64", True)

class TestFolxSharding(unittest.TestCase):
    def test_forward_laplacian_sharding(self):
        """Test folx.forward_laplacian inside shard_map with the fix."""
        mesh = Mesh(jax.devices(), axis_names=('walkers',))
        n_walkers = 8
        
        def func(r1):
            return jnp.sum(r1**3)

        x = jnp.ones((n_walkers, 3))
        
        def local_fn(x_chunk):
            def single_walker_lap(r):
                return folx.forward_laplacian(func)(r)
            return jax.vmap(single_walker_lap)(x_chunk)

        sharded_fn = shard_map(
            local_fn,
            mesh=mesh,
            in_specs=P('walkers'),
            out_specs=P('walkers')
        )
        
        # Should run without error
        result = sharded_fn(x)
        
        # Verify result correctness (lap of sum(r^3) is 6*sum(r))
        # At r=1, lap = 6*(1+1+1) = 18
        expected = 18.0
        # result is a ForwardLaplacian object with .laplacian attribute?
        # No, result of shard_map(...) returns what local_fn returns.
        # local_fn returns vmap(...) which returns ForwardLaplacian(x=..., jacobian=..., laplacian=...)
        # Wait, vmap of PyTree returns PyTree.
        # So result should be a ForwardLaplacian object where fields are arrays.
        
        self.assertTrue(hasattr(result, 'laplacian'))
        self.assertEqual(result.laplacian.shape, (n_walkers,))
        # Check values
        np_result = jax.device_get(result.laplacian)
        for val in np_result:
            self.assertAlmostEqual(val, expected, places=5)

if __name__ == '__main__':
    unittest.main()
