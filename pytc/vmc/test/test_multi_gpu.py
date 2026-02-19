"""Tests for multi-GPU sharding utilities and data-parallel VMC.

These tests simulate multiple devices using
``XLA_FLAGS=--xla_force_host_platform_device_count=N`` which must be set
**before** JAX is imported.  The flag is set at the top of this module.

Run with::

    conda run -n pytc --no-capture-output python -m unittest \
        pytc.vmc.test.test_multi_gpu -v
"""

import os
# Must be set before any JAX import
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)

import unittest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P

from pyscf import gto, scf

from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import CompositeJastrow, NuclearCusp, BoysHandy
from pytc.vmc.walker import Walker, initialize_walker_state, initialize_walkers
from pytc.vmc.optimizer import NewtonOptimizer
from pytc.vmc.sharding import (
    create_mesh, shard_walker, replicate,
    pad_n_walkers, pad_walker,
    get_vmap_fn, is_multi_gpu, n_devices, initialize_walkers_sharded,
)


def _make_h2():
    """Build H2/STO-3G SlaterJastrow and return (sj, det, params, mol, mf)."""
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1.4'
    mol.basis = 'sto-3g'
    mol.unit = 'Bohr'
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    det = SlaterDet.create(mol, mf.mo_coeff)
    bh = BoysHandy.create(mol)
    jnuc = NuclearCusp.create(mol)
    jastrow = CompositeJastrow.create([jnuc, bh])
    jastrow_params = jastrow.init_params()
    sj = SlaterJastrow.create(mol, jastrow, [det])
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]
    return sj, det, params, mol, mf


# ======================================================================
# Test utilities
# ======================================================================

class TestShardingUtilities(unittest.TestCase):
    """Test the sharding helper functions."""

    def test_device_count(self):
        """Should see 4 simulated CPU devices."""
        self.assertEqual(jax.device_count(), 4)
        self.assertTrue(is_multi_gpu())
        self.assertEqual(n_devices(), 4)

    def test_create_mesh(self):
        """create_mesh should return a Mesh with the expected axis."""
        mesh = create_mesh()
        self.assertEqual(len(mesh.devices.flat), 4)
        self.assertIn("walkers", mesh.axis_names)

    def test_pad_n_walkers(self):
        """pad_n_walkers rounds up to nearest multiple of n_devices."""
        self.assertEqual(pad_n_walkers(10, 4), 12)
        self.assertEqual(pad_n_walkers(8, 4), 8)
        self.assertEqual(pad_n_walkers(1, 4), 4)
        self.assertEqual(pad_n_walkers(100, 3), 102)

    def test_shard_walker_positions(self):
        """Shard a plain array along axis 0."""
        mesh = create_mesh()
        x = jnp.ones((8, 3))
        xs = shard_walker(x, mesh)
        self.assertEqual(xs.shape, (8, 3))
        # Should be sharded on the 'walkers' axis
        spec = xs.sharding.spec
        self.assertEqual(spec[0], "walkers")

    def test_shard_walker_dataclass(self):
        """Shard a Walker dataclass — all fields along axis 0."""
        sj, det, params, mol, mf = _make_h2()
        walkers = initialize_walkers(det, 8, key=random.PRNGKey(0))

        mesh = create_mesh()
        ws = shard_walker(walkers, mesh)

        # positions should be sharded
        self.assertEqual(ws.positions.sharding.spec[0], "walkers")
        # det_up is a tuple of arrays — both should be sharded
        self.assertEqual(ws.det_up[0].sharding.spec[0], "walkers")
        self.assertEqual(ws.det_up[1].sharding.spec[0], "walkers")

    def test_replicate_params(self):
        """Replicate params — every shard should be the same."""
        mesh = create_mesh()
        p = jnp.array([1.0, 2.0, 3.0])
        pr = replicate(p, mesh)
        self.assertEqual(pr.sharding.spec, P())
        np.testing.assert_allclose(pr, p)

    def test_get_vmap_fn(self):
        """get_vmap_fn returns shard_vmap in multi-device environment."""
        from pytc.vmc.sharding import shard_vmap
        
        # Auto-detects 4 devices
        fn = get_vmap_fn()
        # It returns a partial(shard_vmap, ...)
        self.assertEqual(fn.func, shard_vmap)

        # Still returns shard_vmap if max_vmap_batch_size=0
        fn2 = get_vmap_fn(max_vmap_batch_size=0)
        self.assertEqual(fn2.func, shard_vmap)

    def test_pad_walker(self):
        """pad_walker should extend walker to target size."""
        sj, det, params, mol, mf = _make_h2()
        walkers = initialize_walkers(det, 6, key=random.PRNGKey(0))

        padded, original_n = pad_walker(walkers, 8)
        self.assertEqual(original_n, 6)
        self.assertEqual(padded.positions.shape[0], 8)
        # First 6 rows should be unchanged
        np.testing.assert_allclose(padded.positions[:6], walkers.positions)

    def test_initialize_walkers_sharded(self):
        """initialize_walkers_sharded creates directly sharded walker state."""
        sj, det, params, mol, mf = _make_h2()
        mesh = create_mesh()
        walkers = initialize_walkers_sharded(det, 8, mesh, key=random.PRNGKey(7))
        self.assertEqual(walkers.positions.shape[0], 8)
        self.assertEqual(walkers.positions.sharding.spec[0], "walkers")

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()


# ======================================================================
# Test sharded computation correctness
# ======================================================================

class TestShardedComputation(unittest.TestCase):
    """Verify that sharded vmap+reduction gives the same result as single-device."""

    def setUp(self):
        jax.clear_caches()

    def test_vmap_value_and_grad_over_sharded_walkers(self):
        """vmap(value_and_grad(E_L)) over sharded walkers matches single-device."""
        sj, det, params, mol, mf = _make_h2()
        key = random.PRNGKey(42)
        n_walkers = 8

        walkers = initialize_walkers(det, n_walkers, key=key)

        # Warm up cache
        batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
        _, walkers = batch_ansatz(walkers, params)

        # --- Single-device reference ---
        def single_e_and_grad(w, p):
            return jax.value_and_grad(lambda pp: sj.local_energy(w, pp)[0])(p)

        energies_ref, jac_ref = jax.vmap(single_e_and_grad, in_axes=(0, None))(walkers, params)
        e_mean_ref = jnp.mean(energies_ref)

        # --- Sharded ---
        mesh = create_mesh()
        ws = shard_walker(walkers, mesh)
        ps = replicate(params, mesh)

        energies_s, jac_s = jax.vmap(single_e_and_grad, in_axes=(0, None))(ws, ps)
        e_mean_s = jnp.mean(energies_s)

        # Values should match
        np.testing.assert_allclose(float(e_mean_s), float(e_mean_ref), rtol=1e-10)
        np.testing.assert_allclose(np.array(energies_s), np.array(energies_ref), rtol=1e-10)

        # Sharding should be preserved for per-walker outputs
        self.assertEqual(energies_s.sharding.spec[0], "walkers")
        # Mean should be replicated
        self.assertEqual(e_mean_s.sharding.spec, P())

        print(f"✓ Sharded vmap(value_and_grad) matches: E_mean={float(e_mean_ref):.6f}")

    def test_curvature_matrix_matches(self):
        """J^T @ J from sharded walkers matches single-device."""
        sj, det, params, mol, mf = _make_h2()
        key = random.PRNGKey(42)
        n_walkers = 8

        walkers = initialize_walkers(det, n_walkers, key=key)
        batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
        _, walkers = batch_ansatz(walkers, params)

        def single_e_and_grad(w, p):
            return jax.value_and_grad(lambda pp: sj.local_energy(w, pp)[0])(p)

        # --- Reference ---
        energies_ref, jac_ref = jax.vmap(single_e_and_grad, in_axes=(0, None))(walkers, params)
        jac_flat_ref = jnp.concatenate(
            [jnp.reshape(leaf, (n_walkers, -1))
             for leaf in jax.tree_util.tree_leaves(jac_ref)], axis=1
        )
        e_mean_ref = jnp.mean(energies_ref)
        diff_ref = energies_ref - e_mean_ref
        grad_ref = (2.0 / (n_walkers - 1)) * (jac_flat_ref.T @ diff_ref)
        jac_c_ref = jac_flat_ref - jnp.mean(jac_flat_ref, axis=0, keepdims=True)
        curv_ref = (2.0 / n_walkers) * (jac_c_ref.T @ jac_c_ref)

        # --- Sharded ---
        mesh = create_mesh()
        ws = shard_walker(walkers, mesh)
        ps = replicate(params, mesh)

        energies_s, jac_s = jax.vmap(single_e_and_grad, in_axes=(0, None))(ws, ps)
        jac_flat_s = jnp.concatenate(
            [jnp.reshape(leaf, (n_walkers, -1))
             for leaf in jax.tree_util.tree_leaves(jac_s)], axis=1
        )
        e_mean_s = jnp.mean(energies_s)
        diff_s = energies_s - e_mean_s
        grad_s = (2.0 / (n_walkers - 1)) * (jac_flat_s.T @ diff_s)
        jac_c_s = jac_flat_s - jnp.mean(jac_flat_s, axis=0, keepdims=True)
        curv_s = (2.0 / n_walkers) * (jac_c_s.T @ jac_c_s)

        # Curvature and gradient must match
        np.testing.assert_allclose(np.array(curv_s), np.array(curv_ref), rtol=1e-10)
        np.testing.assert_allclose(np.array(grad_s), np.array(grad_ref), rtol=1e-10)

        # Curvature should be replicated (small P×P matrix)
        self.assertEqual(curv_s.sharding.spec, P())
        self.assertEqual(grad_s.sharding.spec, P())

        print(f"✓ Curvature matrix matches: P={curv_ref.shape[0]}, "
              f"max_diff={float(jnp.max(jnp.abs(curv_s - curv_ref))):.2e}")

    def test_mcmc_step_with_sharded_walkers(self):
        """MCMC step should work with sharded walkers."""
        from pytc.vmc.metropolis import metropolis_hastings

        sj, det, params, mol, mf = _make_h2()
        key = random.PRNGKey(42)
        n_walkers = 8

        walkers = initialize_walkers(det, n_walkers, key=key)
        batch_ansatz = jax.vmap(lambda w, p: det(w, p), in_axes=(0, None))
        _, walkers = batch_ansatz(walkers, params)

        mesh = create_mesh()
        ws = shard_walker(walkers, mesh)

        key, subkey = random.split(key)
        new_walkers, accept = metropolis_hastings(
            det, ws, 0.5, subkey, params, move_type="one"
        )

        accept_val = float(accept)
        self.assertGreater(accept_val, 0.0)
        self.assertLess(accept_val, 1.0)
        # New walkers should still be sharded
        self.assertEqual(new_walkers.positions.sharding.spec[0], "walkers")

        print(f"✓ MCMC step with sharded walkers: acceptance={accept_val:.3f}")

    def test_sharded_mcmc_uses_independent_device_rng(self):
        """Replicated key should still yield different proposals across devices."""
        from pytc.vmc.metropolis import make_mcmc_step

        sj, det, params, mol, mf = _make_h2()
        mesh = create_mesh()
        n_walkers = 8

        walkers = initialize_walkers(det, n_walkers, key=random.PRNGKey(0))
        # Make all walkers identical so divergence must come from RNG streams.
        walkers = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[:1], n_walkers, axis=0) if isinstance(x, jnp.ndarray) and x.ndim > 0 else x,
            walkers,
        )

        ws = shard_walker(walkers, mesh)
        ps = replicate(params, mesh)
        key = replicate(random.PRNGKey(123), mesh)

        mcmc_step = make_mcmc_step(
            det, 0.5, move_type="all", max_vmap_batch_size=0, mesh=mesh
        )
        new_walkers, _ = mcmc_step(det, ws, key, ps)

        shard_positions = [np.array(s.data) for s in new_walkers.positions.addressable_shards]
        # At least one device shard should differ from shard 0 if RNG is independent.
        any_diff = any(not np.allclose(shard_positions[0], arr) for arr in shard_positions[1:])
        self.assertTrue(any_diff, "Expected per-device independent RNG streams in sharded MCMC.")

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()


# ======================================================================
# Test Newton optimizer with multi_gpu flag
# ======================================================================

class TestNewtonMultiGPU(unittest.TestCase):
    """Test that NewtonOptimizer with multi_gpu=True gives correct results."""

    def setUp(self):
        jax.clear_caches()

    def test_newton_step_multi_gpu(self):
        """Newton GN exact step with multi_gpu=True matches single-device."""
        sj, det, params, mol, mf = _make_h2()
        key = random.PRNGKey(42)
        n_walkers = 8

        walkers = initialize_walkers(det, n_walkers, key=key)
        batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
        _, walkers = batch_ansatz(walkers, params)

        # Loss function for value_and_grad (needed by Newton init, but
        # gauss_newton exact path doesn't use it)
        from pytc.vmc.loss import make_variance_loss
        loss_fn = make_variance_loss(ansatz=sj, optimizer_type='newton',
                                     max_vmap_batch_size=0)
        loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

        # --- Single-device reference ---
        opt_ref = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1,
            damping=1e-5,
            curvature_type="gauss_newton",
            solver="exact",
        )
        key_ref = random.PRNGKey(99)
        state_ref = opt_ref.init(params, key_ref, (walkers, sj))
        new_params_ref, _, stats_ref = opt_ref.step(
            params, state_ref, key_ref, (walkers, sj)
        )
        loss_ref = float(stats_ref['loss'])

        # --- Multi-GPU ---
        mesh = create_mesh()
        ws = shard_walker(walkers, mesh)
        ps = replicate(params, mesh)

        opt_mg = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1,
            damping=1e-5,
            curvature_type="gauss_newton",
            solver="exact",
        )
        key_mg = random.PRNGKey(99)
        state_mg = opt_mg.init(ps, key_mg, (ws, sj))
        new_params_mg, _, stats_mg = opt_mg.step(
            ps, state_mg, key_mg, (ws, sj)
        )
        loss_mg = float(stats_mg['loss'])

        # Should match — relax to 1e-9 because multi-device reductions
        # change floating-point summation order
        np.testing.assert_allclose(loss_mg, loss_ref, rtol=1e-9,
                                   err_msg="Loss mismatch between multi_gpu and single")

        # Compare updated params
        for i, (p_ref, p_mg) in enumerate(zip(
            jax.tree_util.tree_leaves(new_params_ref),
            jax.tree_util.tree_leaves(new_params_mg)
        )):
            np.testing.assert_allclose(
                np.array(p_mg), np.array(p_ref), rtol=1e-9,
                err_msg=f"Param leaf {i} mismatch"
            )

        print(f"✓ Newton multi_gpu step matches single-device: loss={loss_ref:.6f}")

    def test_newton_nondivisible_jacobian_sample_size(self):
        """Non-divisible jacobian_sample_size should be auto-adjusted in multi-device mode."""
        sj, det, params, mol, mf = _make_h2()
        key = random.PRNGKey(123)
        n_walkers = 8

        walkers = initialize_walkers(det, n_walkers, key=key)
        batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
        _, walkers = batch_ansatz(walkers, params)

        from pytc.vmc.loss import make_variance_loss
        loss_fn = make_variance_loss(ansatz=sj, optimizer_type='newton', max_vmap_batch_size=0)
        loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

        mesh = create_mesh()
        ws = shard_walker(walkers, mesh)
        ps = replicate(params, mesh)

        opt = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1,
            damping=1e-5,
            curvature_type="gauss_newton",
            solver="exact",
            jacobian_sample_size=5,  # not divisible by 4 devices
        )
        state = opt.init(ps, key, (ws, sj))
        new_params, _, stats = opt.step(ps, state, key, (ws, sj))

        self.assertTrue(np.isfinite(float(stats['loss'])))
        self.assertEqual(new_params[1].shape, ps[1].shape)

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()


# ======================================================================
# Integration test: full optimize_ref_var with multi_gpu
# ======================================================================

class TestOptimizeRefVarMultiGPU(unittest.TestCase):
    """Integration test: optimize_ref_var with multi_gpu=True."""

    def setUp(self):
        jax.clear_caches()

    def test_optimize_ref_var_multi_gpu_runs(self):
        """optimize_ref_var(multi_gpu=True) should complete without error."""
        from pytc.vmc import optimize_ref_var

        sj, det, params, mol, mf = _make_h2()
        key = random.PRNGKey(42)

        results = optimize_ref_var(
            sj,
            params=params,
            n_walkers=8,  # 8 walkers / 4 devices = 2 per device
            n_steps=2,
            step_size=1.0,
            burn_in_steps=10,
            n_opt_steps=3,
            optimizer_type='newton',
            learning_rate=0.1,
            opt_kwargs={'damping': 1e-5, 'solver': 'exact'},
            max_vmap_batch_size=0,
            key=key,
        )

        self.assertEqual(len(results['energies']), 3)
        self.assertTrue(all(np.isfinite(results['energies'])),
                        f"Non-finite energies: {results['energies']}")

        print(f"✓ optimize_ref_var(multi_gpu=True): "
              f"E={results['energies'][-1]:.6f}, Var={results['cost'][-1]:.4f}")

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()


if __name__ == '__main__':
    unittest.main()
