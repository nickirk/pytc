import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from pytc.vmc.sharding import create_mesh, shard_walker, get_vmap_fn
from pytc.vmc.walker import initialize_walkers
from pyscf import gto, scf
from pytc.ansatz.det import SlaterDet

# If a prior test module imported jax first, the XLA_FLAGS setdefault above is
# too late and we end up with one CPU device. Skip — these tests need their own
# process to honour the env var.
@unittest.skipUnless(
    jax.device_count() >= 4,
    f"Multi-device tests require 4 simulated devices; got {jax.device_count()}. "
    "Run this module in its own process so XLA_FLAGS is honoured.",
)
class TestMultiGPUBatched(unittest.TestCase):
    def test_sharded_batched_vmap_preserves_sharding(self):
        """Verify that get_vmap_fn with max_vmap_batch_size > 0 preserves sharding."""
        mol = gto.Mole()
        mol.atom = 'H 0 0 0; H 0 0 1.4'
        mol.basis = 'sto-3g'
        mol.build()
        mf = scf.RHF(mol)
        mf.kernel()
        det = SlaterDet.create(mol, mf.mo_coeff)
        
        n_walkers = 16
        key = random.PRNGKey(0)
        walkers = initialize_walkers(det, n_walkers, key=key)
        
        mesh = create_mesh()
        ws = shard_walker(walkers, mesh)
        
        vmap_fn = get_vmap_fn(max_vmap_batch_size=4, mesh=mesh)
        
        result = vmap_fn(lambda w: det(w, None)[0][1])(ws)
        
        print(f"Result sharding: {result.sharding}")
        
        from jax.sharding import PartitionSpec as P
        self.assertEqual(result.sharding.spec, P('walkers'))
        print("✓ Sharding preserved with batching!")

if __name__ == "__main__":
    unittest.main()
