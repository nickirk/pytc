import os
# Must be set before any JAX import
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
try:
    from jax.shard_map import shard_map
except ImportError:
    try:
        from jax.experimental.shard_map import shard_map
    except ImportError:
        shard_map = None
import functools

class TestShardMap(unittest.TestCase):
    """Test fundamental shard_map with local batching logic."""

    def test_shard_map_with_local_batching(self):
        """Verify that shard_map can distribute walkers and batch locally."""
        import numpy as np
        
        if shard_map is None:
            self.skipTest("shard_map not available")

        devices = jax.devices()
        if len(devices) < 2:
            self.skipTest("Need at least 2 devices")

        mesh = Mesh(devices, axis_names=('x',))
        
        n_walkers = 16
        n_devices = len(devices)
        max_batch_size = 2 
        
        # Ensure n_walkers is divisible by n_devices
        self.assertEqual(n_walkers % n_devices, 0)
        
        x = jnp.arange(n_walkers).reshape(n_walkers, 1)
        
        # Mock batching logic similar to folx.batched_vmap
        def mock_local_batching(x_shard):
            batch_size = x_shard.shape[0]
            num_batches = batch_size // max_batch_size
            
            # Reshape into (num_batches, max_batch_size, ...)
            loop_args = x_shard.reshape(num_batches, max_batch_size, *x_shard.shape[1:])
            
            def scan_fn(carry, batch):
                return carry, batch * 2
            
            _, result = jax.lax.scan(scan_fn, None, loop_args)
            return result.reshape(-1, *x_shard.shape[1:])

        # Use shard_map to distribute walkers across mesh
        @functools.partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
        def parallel_batched_fn(x_shard):
            return mock_local_batching(x_shard)

        result = parallel_batched_fn(x)
        
        # Verify shape and values
        self.assertEqual(result.shape, (n_walkers, 1))
        np_result = jax.device_get(result)
        expected = np.arange(n_walkers).reshape(n_walkers, 1) * 2
        np.testing.assert_allclose(np_result, expected)
        
        # Verify it's still sharded
        self.assertEqual(result.sharding.spec, P('x'))

if __name__ == "__main__":
    unittest.main()
