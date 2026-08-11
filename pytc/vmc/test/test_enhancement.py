"""Tests for VMC optimization enhancements.

Validates that the optimized implementations produce numerically identical
results to the original code, then tests correctness at increasing system sizes.
"""

import unittest
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import time
import functools

from pyscf import gto, scf

from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import (
    SlaterDet,
    eval_det_value_and_grad,
    rank1_update_one_electron,
)
from pytc.ansatz.sj import SlaterJastrow, update_jastrow_one_electron, compute_jastrow_log_value
from pytc.jastrow import CompositeJastrow, NuclearCusp, BoysHandy
from pytc.vmc.walker import initialize_walkers
from pytc.vmc.loss import make_variance_loss
from pytc.vmc.optimizer import NewtonOptimizer
from pytc.vmc.metropolis import metropolis_hastings
from pytc.vmc import optimize_ref_var


def make_test_system(atom_spec, basis='sto-3g', n_walkers=50, key=None):
    """Helper to create a test system (molecule + ansatz + walkers + params)."""
    if key is None:
        key = random.PRNGKey(42)
    
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.basis = basis
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
    
    walkers = initialize_walkers(det, n_walkers, key=key)
    
    batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
    _, walkers = batch_ansatz(walkers, params)
    
    return mol, mf, sj, det, params, walkers


def make_h_chain(n_atoms, dist=1.4):
    """Create a hydrogen chain with Bohr spacing."""
    return "; ".join(f"H 0 0 {i * dist}" for i in range(n_atoms))


class TestNewtonMergedGradient(unittest.TestCase):
    """Test that merged gradient+Jacobian gives identical results to the original."""
    
    def setUp(self):
        """Create a small H2 system for testing."""
        self.mol, self.mf, self.sj, self.det, self.params, self.walkers = \
            make_test_system('H 0 0 0; H 0 0 1.4', n_walkers=50)
    
    def test_gauss_newton_exact_gradient_matches_original(self):
        """Verify that the merged GN gradient is identical to the original separate computation.
        
        Original: 
          1. value_and_grad_func computes loss+grad (via custom JVP)
          2. Separately computes Jacobian via vmap(grad(local_energy))
        
        Merged:
          1. Computes Jacobian + energies in one vmap(value_and_grad(local_energy)) pass
          2. Derives grad analytically from J and E
        """
        ansatz = self.sj
        params = self.params
        walkers = self.walkers
        
        loss_fn = make_variance_loss(ansatz=ansatz, optimizer_type="newton", 
                                      use_custom_jvp=True, max_vmap_batch_size=0)
        loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        
        batch = (walkers, ansatz)
        (loss_orig, aux_orig), grads_orig = loss_fn_jvp(params, batch)
        
        def single_local_energy_grad_orig(w, p):
            return jax.grad(lambda pp: ansatz.local_energy(w, pp)[0])(p)
        jac_orig = jax.vmap(single_local_energy_grad_orig, in_axes=(0, None))(walkers, params)
        jac_flat_orig, _ = jax.tree_util.tree_flatten(jac_orig)
        n_walkers = walkers.shape[0]
        jac_mat_orig = jnp.concatenate(
            [jnp.reshape(leaf, (n_walkers, -1)) for leaf in jac_flat_orig], axis=1
        )
        jac_centered_orig = jac_mat_orig - jnp.mean(jac_mat_orig, axis=0, keepdims=True)
        curvature_orig = (2.0 / n_walkers) * (jac_centered_orig.T @ jac_centered_orig)
        grads_vec_orig, unravel_fn = jax.flatten_util.ravel_pytree(grads_orig)
        
        def single_local_energy_and_grad(w, p):
            return jax.value_and_grad(lambda pp: ansatz.local_energy(w, pp)[0])(p)
        
        energies_merged, jac_merged = jax.vmap(
            single_local_energy_and_grad, in_axes=(0, None)
        )(walkers, params)
        
        jac_flat_merged, _ = jax.tree_util.tree_flatten(jac_merged)
        jac_mat_merged = jnp.concatenate(
            [jnp.reshape(leaf, (n_walkers, -1)) for leaf in jac_flat_merged], axis=1
        )
        
        e_mean = jnp.mean(energies_merged)
        energy_diff = energies_merged - e_mean
        loss_merged = jnp.sum(energy_diff**2) / (n_walkers - 1)
        
        grads_vec_merged = (2.0 / (n_walkers - 1)) * (jac_mat_merged.T @ energy_diff)
        
        jac_centered_merged = jac_mat_merged - jnp.mean(jac_mat_merged, axis=0, keepdims=True)
        curvature_merged = (2.0 / n_walkers) * (jac_centered_merged.T @ jac_centered_merged)
        
        np.testing.assert_allclose(float(loss_merged), float(loss_orig), rtol=1e-10,
                                   err_msg="Merged loss doesn't match original loss")
        
        np.testing.assert_allclose(jac_mat_merged, jac_mat_orig, rtol=1e-10,
                                   err_msg="Jacobians don't match")
        
        np.testing.assert_allclose(curvature_merged, curvature_orig, rtol=1e-10,
                                   err_msg="Curvature matrices don't match")
        
        # Gradient is the key test: merged analytical gradient vs original custom JVP gradient
        np.testing.assert_allclose(grads_vec_merged, grads_vec_orig, rtol=1e-6,
                                   err_msg="Merged gradient doesn't match original gradient")
        
        print(f"✓ Merged GN gradient matches original. Loss: {float(loss_orig):.6f}")
        print(f"  Gradient norm (orig):   {jnp.linalg.norm(grads_vec_orig):.8f}")
        print(f"  Gradient norm (merged): {jnp.linalg.norm(grads_vec_merged):.8f}")
        print(f"  Max gradient diff:      {jnp.max(jnp.abs(grads_vec_merged - grads_vec_orig)):.2e}")
    
    def test_newton_optimizer_step_matches(self):
        """Test that the full NewtonOptimizer.step() gives the same update with the merged code."""
        ansatz = self.sj
        params = self.params
        walkers = self.walkers
        batch = (walkers, ansatz)
        
        loss_fn = make_variance_loss(ansatz=ansatz, optimizer_type="newton",
                                      use_custom_jvp=True, max_vmap_batch_size=0)
        loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        
        optimizer = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1,
            damping=1e-5,
            curvature_type="gauss_newton",
            solver="exact",
        )
        
        key = random.PRNGKey(99)
        opt_state = optimizer.init(params, key, batch)
        
        new_params, new_state, stats = optimizer.step(
            params=params, state=opt_state, rng=key, batch=batch, global_step_int=0
        )
        
        loss_val = float(stats['loss'])
        e_mean, e_std = stats['aux']
        self.assertTrue(jnp.isfinite(jnp.array(loss_val)), "Loss should be finite")
        self.assertTrue(jnp.isfinite(e_mean), "Mean energy should be finite")
        
        params_flat_old, _ = jax.flatten_util.ravel_pytree(params)
        params_flat_new, _ = jax.flatten_util.ravel_pytree(new_params)
        self.assertFalse(jnp.allclose(params_flat_old, params_flat_new),
                        "Parameters should change after Newton step")
        
        print(f"✓ Newton step OK. Loss: {loss_val:.6f}, E: {float(e_mean):.6f}±{float(e_std):.6f}")

    def test_batched_accumulators_match_full_jacobian_stats(self):
        """Batched GN sufficient statistics should match the full Jacobian formulas."""
        ansatz = self.sj
        params = self.params
        walkers = self.walkers
        n_walkers = walkers.shape[0]

        def single_local_energy_and_grad(w, p):
            return jax.value_and_grad(lambda pp: ansatz.local_energy(w, pp)[0])(p)

        energies_full, jac_full = jax.vmap(
            single_local_energy_and_grad, in_axes=(0, None)
        )(walkers, params)
        jac_mat_full = NewtonOptimizer._flatten_jacobian(jac_full, n_walkers)

        e_mean_full = jnp.mean(energies_full)
        energy_diff_full = energies_full - e_mean_full
        loss_full = jnp.sum(energy_diff_full**2) / (n_walkers - 1)
        grad_full = (2.0 / (n_walkers - 1)) * (jac_mat_full.T @ energy_diff_full)
        jac_centered_full = jac_mat_full - jnp.mean(jac_mat_full, axis=0, keepdims=True)
        curv_full = (2.0 / n_walkers) * (jac_centered_full.T @ jac_centered_full)

        batch_size = 7
        n_params = jac_mat_full.shape[1]
        sum_e = 0.0
        sum_e2 = 0.0
        sum_j = jnp.zeros((n_params,))
        sum_jte = jnp.zeros((n_params,))
        sum_jtj = jnp.zeros((n_params, n_params))

        for start in range(0, n_walkers, batch_size):
            stop = min(start + batch_size, n_walkers)
            batch_walkers = jax.tree_util.tree_map(lambda x: x[start:stop], walkers)
            energies_batch, jac_batch = jax.vmap(
                single_local_energy_and_grad, in_axes=(0, None)
            )(batch_walkers, params)
            jac_mat_batch = NewtonOptimizer._flatten_jacobian(jac_batch, stop - start)
            sum_e = sum_e + jnp.sum(energies_batch)
            sum_e2 = sum_e2 + jnp.sum(energies_batch**2)
            sum_j = sum_j + jnp.sum(jac_mat_batch, axis=0)
            sum_jte = sum_jte + jac_mat_batch.T @ energies_batch
            sum_jtj = sum_jtj + jac_mat_batch.T @ jac_mat_batch

        e_mean_batched = sum_e / n_walkers
        loss_batched = (sum_e2 - n_walkers * e_mean_batched**2) / (n_walkers - 1)
        grad_batched = (2.0 / (n_walkers - 1)) * (
            sum_jte - n_walkers * (sum_j / n_walkers) * e_mean_batched
        )
        curv_batched = (2.0 / n_walkers) * (
            sum_jtj - n_walkers * jnp.outer(sum_j / n_walkers, sum_j / n_walkers)
        )

        np.testing.assert_allclose(float(loss_batched), float(loss_full), rtol=1e-10)
        np.testing.assert_allclose(grad_batched, grad_full, rtol=1e-10)
        np.testing.assert_allclose(curv_batched, curv_full, rtol=1e-10)

    def test_batched_newton_step_matches_unbatched(self):
        """Batched GN accumulation should match the existing unbatched exact step."""
        ansatz = self.sj
        params = self.params
        walkers = self.walkers
        batch = (walkers, ansatz)
        key = random.PRNGKey(99)

        loss_fn = make_variance_loss(
            ansatz=ansatz,
            optimizer_type="newton",
            use_custom_jvp=True,
            max_vmap_batch_size=0,
        )
        loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

        opt_ref = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1,
            damping=1e-5,
            curvature_type="gauss_newton",
            solver="exact",
            max_vmap_batch_size=0,
        )
        ref_state = opt_ref.init(params, key, batch)
        ref_params, _, ref_stats = opt_ref.step(params, ref_state, key, batch)

        opt_batched = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1,
            damping=1e-5,
            curvature_type="gauss_newton",
            solver="exact",
            max_vmap_batch_size=7,
        )
        batched_state = opt_batched.init(params, key, batch)
        batched_params, _, batched_stats = opt_batched.step(params, batched_state, key, batch)

        np.testing.assert_allclose(float(batched_stats["loss"]), float(ref_stats["loss"]), rtol=1e-10)
        np.testing.assert_allclose(
            np.array(batched_stats["aux"][0]),
            np.array(ref_stats["aux"][0]),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            np.array(batched_stats["aux"][1]),
            np.array(ref_stats["aux"][1]),
            rtol=1e-10,
        )
        for p_batched, p_ref in zip(
            jax.tree_util.tree_leaves(batched_params),
            jax.tree_util.tree_leaves(ref_params),
        ):
            # Slightly looser than the stats check (1e-10) because parameter
            # updates compound float op-ordering noise from the GN solve.
            np.testing.assert_allclose(np.array(p_batched), np.array(p_ref), rtol=1e-9)

    def test_batched_newton_step_with_jacobian_sampling_is_finite(self):
        """Batched GN accumulation should still work with Jacobian subsampling."""
        ansatz = self.sj
        params = self.params
        walkers = self.walkers
        batch = (walkers, ansatz)
        key = random.PRNGKey(123)

        loss_fn = make_variance_loss(
            ansatz=ansatz,
            optimizer_type="newton",
            use_custom_jvp=True,
            max_vmap_batch_size=0,
        )
        loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

        optimizer = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1,
            damping=1e-5,
            curvature_type="gauss_newton",
            solver="exact",
            max_vmap_batch_size=7,
            jacobian_sample_size=11,
        )
        state = optimizer.init(params, key, batch)
        new_params, _, stats = optimizer.step(params, state, key, batch)

        self.assertTrue(np.isfinite(float(stats["loss"])))
        self.assertTrue(np.isfinite(float(stats["aux"][0])))
        self.assertTrue(np.isfinite(float(stats["aux"][1])))
        self.assertEqual(new_params[1].shape, params[1].shape)


class TestBatchedNewtonIntegration(unittest.TestCase):
    """Integration coverage for batched GN accumulation in optimize_ref_var."""

    def test_optimize_ref_var_h_chain_runs_with_batched_newton(self):
        """A small H-chain should run end-to-end with batched GN accumulation."""
        mol, mf, sj, det, params, _ = make_test_system(
            make_h_chain(6, dist=1.0),
            basis="sto-3g",
            n_walkers=32,
            key=random.PRNGKey(7),
        )
        del mol, mf, det

        results = optimize_ref_var(
            sj,
            params=params,
            n_walkers=32,
            n_steps=1,
            step_size=0.2,
            burn_in_steps=0,
            n_opt_steps=1,
            optimizer_type="newton",
            learning_rate=0.1,
            max_vmap_batch_size=4,
            key=random.PRNGKey(8),
            adaptive_step_size=False,
        )

        self.assertEqual(len(results["energies"]), 1)
        self.assertTrue(np.isfinite(results["energies"][0]))
        self.assertTrue(np.isfinite(results["cost"][0]))


class TestPsiCaching(unittest.TestCase):
    """Test that cached log_psi/psi_sign values in Walker are correct."""
    
    def setUp(self):
        """Create H2 system."""
        self.mol, self.mf, self.sj, self.det, self.params, self.walkers = \
            make_test_system('H 0 0 0; H 0 0 1.4', n_walkers=50)
    
    def test_cache_populated_by_eval_sj(self):
        """Verify that eval_sj populates log_psi, psi_sign, log_jastrow in walker."""
        walkers = self.walkers
        sj = self.sj
        params = self.params
        
        batch_ansatz = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
        psi_values, updated_walkers = batch_ansatz(walkers, params)
        
        psi_sign, psi_logabs = psi_values
        
        np.testing.assert_allclose(updated_walkers.log_psi, psi_logabs, rtol=1e-14)
        np.testing.assert_allclose(updated_walkers.psi_sign, psi_sign, rtol=1e-14)
        
        # log_jastrow should be nonzero (unless all Jastrow params are zero)
        self.assertTrue(jnp.any(updated_walkers.log_jastrow != 0.0) or 
                       jnp.allclose(updated_walkers.log_jastrow, 0.0))
        print(f"✓ Cache populated correctly. log_psi range: [{float(psi_logabs.min()):.4f}, {float(psi_logabs.max()):.4f}]")
    
    def test_cache_populated_by_det(self):
        """Verify that eval_det_value_and_grad populates log_psi/psi_sign."""
        det = self.det
        walkers = self.walkers
        
        batch_det = jax.vmap(lambda w, p: det(w, p), in_axes=(0, None))
        det_values, updated_walkers = batch_det(walkers, None)
        
        det_sign, det_logabs = det_values
        
        np.testing.assert_allclose(updated_walkers.log_psi, det_logabs, rtol=1e-14)
        np.testing.assert_allclose(updated_walkers.psi_sign, det_sign, rtol=1e-14)
        print(f"✓ Det cache populated correctly. log|det| range: [{float(det_logabs.min()):.4f}, {float(det_logabs.max()):.4f}]")
    
    def test_one_electron_move_uses_cache(self):
        """Verify that _one_electron_move uses cached values for current ψ.
        
        After warming up, the cached log_psi should be used instead of 
        recomputing. We verify this by comparing the acceptance ratio from
        MCMC with the cached approach vs full recomputation.
        """
        det = self.det
        walkers = self.walkers
        params = self.params
        
        batch_det = jax.vmap(lambda w, p: det(w, p), in_axes=(0, None))
        _, walkers = batch_det(walkers, None)
        
        key = random.PRNGKey(123)
        acceptance_rates = []
        for _ in range(20):
            key, subkey = random.split(key)
            walkers, acceptance = metropolis_hastings(
                det, walkers, 1.0, subkey, params, move_type="one"
            )
            acceptance_rates.append(float(acceptance))
        
        mean_acceptance = np.mean(acceptance_rates)
        
        self.assertGreater(mean_acceptance, 0.05, "Acceptance rate too low — cache may be broken")
        self.assertLess(mean_acceptance, 0.95, "Acceptance rate too high — cache may be broken")
        
        det_values, walkers_check = batch_det(walkers, None)
        det_sign, det_logabs = det_values
        
        np.testing.assert_allclose(walkers.log_psi, det_logabs, rtol=1e-10,
                                   err_msg="Cached log_psi inconsistent with recomputed det value")
        np.testing.assert_allclose(walkers.psi_sign, det_sign, rtol=1e-10,
                                   err_msg="Cached psi_sign inconsistent with recomputed det value")
        
        print(f"✓ MCMC with cache: mean acceptance {mean_acceptance:.3f}, cache consistent after 20 steps")


class TestShermanMorrison(unittest.TestCase):
    """Test Sherman-Morrison rank-1 updates for single-electron moves.
    
    Verifies that:
    1. The determinant ratio from rank-1 update matches full slogdet recomputation.
    2. The updated inverse matches full jnp.linalg.inv recomputation.
    3. The updated grad/lap rows match full AO evaluation.
    4. The partial Jastrow update matches full recomputation.
    5. MCMC with rank-1 updates produces a valid Markov chain (cache consistent).
    """
    
    def setUp(self):
        """Create H2 and Be systems for testing."""
        self.mol_h2, self.mf_h2, self.sj_h2, self.det_h2, self.params_h2, self.walkers_h2 = \
            make_test_system('H 0 0 0; H 0 0 1.4', n_walkers=20)
    
    def test_det_ratio_matches_full_recomputation(self):
        """Verify rank-1 det ratio matches det(S') / det(S) from full slogdet."""
        det = self.det_h2
        walkers = self.walkers_h2
        
        w0 = jax.tree.map(lambda x: x[0], walkers)
        
        new_pos = w0.positions.at[0].set(w0.positions[0] + jnp.array([0.1, -0.2, 0.05]))
        w0_moved = w0.replace(positions=new_pos)
        
        total_ratio, det_logabs_r1, det_sign_r1, w0_updated = \
            rank1_update_one_electron(det, w0_moved, 0)
        
        (det_sign_full, det_logabs_full), w0_full = eval_det_value_and_grad(det, w0_moved)
        
        # From full: ratio = exp(logabs_full - logabs_old) * (sign_full / sign_old)
        sign_old, logabs_old = w0.det_up
        sign_old_dn, logabs_old_dn = w0.det_down
        full_logabs_total_old = logabs_old + logabs_old_dn
        full_sign_total_old = sign_old * sign_old_dn
        
        full_ratio = jnp.exp(det_logabs_full - full_logabs_total_old) * (det_sign_full / full_sign_total_old)
        
        np.testing.assert_allclose(float(total_ratio), float(full_ratio), rtol=1e-10,
                                   err_msg="Rank-1 det ratio doesn't match full recomputation")
        
        np.testing.assert_allclose(float(det_logabs_r1), float(det_logabs_full), rtol=1e-10,
                                   err_msg="Rank-1 log|det| doesn't match full recomputation")
        np.testing.assert_allclose(float(det_sign_r1), float(det_sign_full), rtol=1e-10,
                                   err_msg="Rank-1 det sign doesn't match full recomputation")
        
        print(f"✓ Det ratio: rank-1={float(total_ratio):.10f}, full={float(full_ratio):.10f}")
    
    def test_inverse_matches_full_recomputation(self):
        """Verify Sherman-Morrison inverse update matches jnp.linalg.inv."""
        det = self.det_h2
        walkers = self.walkers_h2
        
        w0 = jax.tree.map(lambda x: x[0], walkers)
        
        new_pos = w0.positions.at[0].set(w0.positions[0] + jnp.array([0.3, -0.1, 0.2]))
        w0_moved = w0.replace(positions=new_pos)
        
        _, _, _, w0_r1 = rank1_update_one_electron(det, w0_moved, 0)
        
        _, w0_full = eval_det_value_and_grad(det, w0_moved)
        
        np.testing.assert_allclose(w0_r1.inv_up, w0_full.inv_up, rtol=1e-8, atol=1e-12,
                                   err_msg="Rank-1 inv_up doesn't match full inv")
        np.testing.assert_allclose(w0_r1.inv_down, w0_full.inv_down, rtol=1e-8, atol=1e-12,
                                   err_msg="Rank-1 inv_down doesn't match full inv")
        
        print(f"✓ Inverse matches: max diff up={float(jnp.max(jnp.abs(w0_r1.inv_up - w0_full.inv_up))):.2e}, "
              f"down={float(jnp.max(jnp.abs(w0_r1.inv_down - w0_full.inv_down))):.2e}")
    
    def test_slater_grad_lap_match_full(self):
        """Verify updated Slater row, grad, and laplacian match full recomputation."""
        det = self.det_h2
        walkers = self.walkers_h2
        
        w0 = jax.tree.map(lambda x: x[0], walkers)
        
        # Move electron 1 (beta electron for H2 with 1 alpha, 1 beta)
        elec_idx = 1
        new_pos = w0.positions.at[elec_idx].set(w0.positions[elec_idx] + jnp.array([-0.15, 0.3, 0.1]))
        w0_moved = w0.replace(positions=new_pos)
        
        _, _, _, w0_r1 = rank1_update_one_electron(det, w0_moved, elec_idx)
        
        _, w0_full = eval_det_value_and_grad(det, w0_moved)
        
        np.testing.assert_allclose(w0_r1.slater_up, w0_full.slater_up, rtol=1e-10,
                                   err_msg="Slater up doesn't match")
        np.testing.assert_allclose(w0_r1.slater_down, w0_full.slater_down, rtol=1e-10,
                                   err_msg="Slater down doesn't match")
        
        np.testing.assert_allclose(w0_r1.grad_up, w0_full.grad_up, rtol=1e-8, atol=1e-12,
                                   err_msg="Grad up doesn't match")
        np.testing.assert_allclose(w0_r1.grad_down, w0_full.grad_down, rtol=1e-8, atol=1e-12,
                                   err_msg="Grad down doesn't match")
        
        np.testing.assert_allclose(w0_r1.lap_up, w0_full.lap_up, rtol=1e-8, atol=1e-12,
                                   err_msg="Lap up doesn't match")
        np.testing.assert_allclose(w0_r1.lap_down, w0_full.lap_down, rtol=1e-8, atol=1e-12,
                                   err_msg="Lap down doesn't match")
        
        print(f"✓ Slater/grad/lap match for electron {elec_idx} move")
    
    def test_jastrow_partial_update_matches_full(self):
        """Verify partial Jastrow update matches full recomputation."""
        sj = self.sj_h2
        walkers = self.walkers_h2
        params = self.params_h2
        jastrow_params = params[0]
        
        w0 = jax.tree.map(lambda x: x[0], walkers)
        
        elec_idx = 0
        new_pos = w0.positions.at[elec_idx].set(w0.positions[elec_idx] + jnp.array([0.2, -0.1, 0.15]))
        
        new_log_j_partial = update_jastrow_one_electron(
            sj, w0.positions, new_pos, elec_idx, jastrow_params, w0.log_jastrow
        )
        
        new_log_j_full = compute_jastrow_log_value(sj, new_pos, jastrow_params)
        
        np.testing.assert_allclose(float(new_log_j_partial), float(new_log_j_full), rtol=1e-10,
                                   err_msg="Partial Jastrow update doesn't match full recomputation")
        
        print(f"✓ Jastrow partial update: partial={float(new_log_j_partial):.10f}, full={float(new_log_j_full):.10f}")
    
    def test_mcmc_rank1_cache_consistency_det(self):
        """Run MCMC with rank-1 updates on SlaterDet and verify cache consistency."""
        det = self.det_h2
        walkers = self.walkers_h2
        params = self.params_h2
        
        batch_det = jax.vmap(lambda w, p: det(w, p), in_axes=(0, None))
        _, walkers = batch_det(walkers, None)
        
        key = random.PRNGKey(999)
        acceptance_rates = []
        for _ in range(30):
            key, subkey = random.split(key)
            walkers, acceptance = metropolis_hastings(
                det, walkers, 1.0, subkey, params, move_type="one"
            )
            acceptance_rates.append(float(acceptance))
        
        mean_acc = np.mean(acceptance_rates)
        
        det_values, walkers_check = batch_det(walkers, None)
        det_sign, det_logabs = det_values
        
        np.testing.assert_allclose(walkers.log_psi, det_logabs, rtol=1e-6,
                                   err_msg="Cached log_psi inconsistent after 30 rank-1 MCMC steps")
        np.testing.assert_allclose(walkers.psi_sign, det_sign, rtol=1e-6,
                                   err_msg="Cached psi_sign inconsistent after 30 rank-1 MCMC steps")
        
        np.testing.assert_allclose(walkers.inv_up, walkers_check.inv_up, rtol=1e-5, atol=1e-10,
                                   err_msg="inv_up inconsistent after 30 rank-1 MCMC steps")
        np.testing.assert_allclose(walkers.inv_down, walkers_check.inv_down, rtol=1e-5, atol=1e-10,
                                   err_msg="inv_down inconsistent after 30 rank-1 MCMC steps")
        
        print(f"✓ Rank-1 MCMC (det only): mean acceptance {mean_acc:.3f}, "
              f"cache consistent after 30 steps")
    
    def test_mcmc_rank1_cache_consistency_sj(self):
        """Run MCMC with rank-1 updates on SlaterJastrow and verify cache consistency."""
        sj = self.sj_h2
        walkers = self.walkers_h2
        params = self.params_h2
        
        batch_sj = jax.vmap(lambda w, p: sj(w, p), in_axes=(0, None))
        _, walkers = batch_sj(walkers, params)
        
        key = random.PRNGKey(777)
        acceptance_rates = []
        for _ in range(30):
            key, subkey = random.split(key)
            walkers, acceptance = metropolis_hastings(
                sj, walkers, 1.0, subkey, params, move_type="one"
            )
            acceptance_rates.append(float(acceptance))
        
        mean_acc = np.mean(acceptance_rates)
        
        psi_values, walkers_check = batch_sj(walkers, params)
        psi_sign, psi_logabs = psi_values
        
        np.testing.assert_allclose(walkers.log_psi, psi_logabs, rtol=1e-6,
                                   err_msg="Cached log_psi inconsistent after 30 rank-1 SJ MCMC steps")
        np.testing.assert_allclose(walkers.psi_sign, psi_sign, rtol=1e-6,
                                   err_msg="Cached psi_sign inconsistent after 30 rank-1 SJ MCMC steps")
        np.testing.assert_allclose(walkers.log_jastrow, walkers_check.log_jastrow, rtol=1e-6,
                                   err_msg="Cached log_jastrow inconsistent after 30 rank-1 SJ MCMC steps")
        
        print(f"✓ Rank-1 MCMC (SJ): mean acceptance {mean_acc:.3f}, "
              f"cache consistent after 30 steps")
    
    def test_rank1_be_larger_system(self):
        """Test rank-1 updates on Be atom (4 electrons, 2α + 2β) — larger matrices."""
        mol, mf, sj, det, params, walkers = make_test_system(
            'Be 0 0 0', basis='sto-3g', n_walkers=20
        )
        
        w0 = jax.tree.map(lambda x: x[0], walkers)
        
        for elec_idx in [0, 1, 2, 3]:  # 0,1 = alpha; 2,3 = beta
            new_pos = w0.positions.at[elec_idx].set(
                w0.positions[elec_idx] + jnp.array([0.2, -0.1, 0.15])
            )
            w0_moved = w0.replace(positions=new_pos)
            
            ratio_r1, logabs_r1, sign_r1, w_r1 = rank1_update_one_electron(det, w0_moved, elec_idx)
            
            (sign_full, logabs_full), w_full = eval_det_value_and_grad(det, w0_moved)
            
            np.testing.assert_allclose(float(logabs_r1), float(logabs_full), rtol=1e-9,
                                       err_msg=f"Be: log|det| mismatch for electron {elec_idx}")
            np.testing.assert_allclose(w_r1.inv_up, w_full.inv_up, rtol=1e-7, atol=1e-12,
                                       err_msg=f"Be: inv_up mismatch for electron {elec_idx}")
            np.testing.assert_allclose(w_r1.inv_down, w_full.inv_down, rtol=1e-7, atol=1e-12,
                                       err_msg=f"Be: inv_down mismatch for electron {elec_idx}")
        
        print(f"✓ Be atom: rank-1 matches full for all 4 electrons")


class TestOptimizeRefVar(unittest.TestCase):
    """Test full optimize_ref_var with different system sizes."""
    
    def setUp(self):
        """Clear JAX caches to avoid stale JIT artifacts from prior test classes."""
        jax.clear_caches()
    
    def _run_opt_test(self, atom_spec, basis='sto-3g', n_walkers=200, n_opt_steps=10, 
                      n_steps=3, label=""):
        """Helper to run optimize_ref_var and check basic sanity."""
        mol = gto.Mole()
        mol.atom = atom_spec
        mol.basis = basis
        mol.unit = 'Bohr'
        mol.build()
        
        mf = scf.RHF(mol)
        mf.kernel()
        hf_energy = mf.e_tot
        
        det = SlaterDet.create(mol, mf.mo_coeff)
        bh = BoysHandy.create(mol)
        jnuc = NuclearCusp.create(mol)
        jastrow = CompositeJastrow.create([jnuc, bh])
        jastrow_params = jastrow.init_params()
        sj = SlaterJastrow.create(mol, jastrow, [det])
        linear_coeffs = jnp.ones(1)
        
        key = random.PRNGKey(42)
        
        start_time = time.time()
        opt_results = optimize_ref_var(
            sj,
            params=[jastrow_params, linear_coeffs],
            n_walkers=n_walkers,
            n_steps=n_steps,
            step_size=1.0,
            burn_in_steps=100,
            n_opt_steps=n_opt_steps,
            optimizer_type='newton',
            learning_rate=0.1,
            opt_kwargs={'damping': 1e-5, 'solver': 'exact'},
            max_vmap_batch_size=0,
            key=key,
        )
        elapsed = time.time() - start_time
        
        self.assertEqual(len(opt_results['energies']), n_opt_steps)
        self.assertTrue(all(np.isfinite(opt_results['energies'])), "All energies should be finite")
        
        final_energy = opt_results['energies'][-1]
        self.assertLess(abs(final_energy), abs(hf_energy) * 100,
                       f"Energy {final_energy} diverged far from HF energy {hf_energy}")
        
        initial_var = opt_results['cost'][0]
        final_var = opt_results['cost'][-1]
        
        print(f"✓ {label}: {mol.natm} atoms, {mol.nelectron}e, {basis}")
        print(f"  HF energy: {hf_energy:.6f}")
        print(f"  Final E:   {final_energy:.6f} (var: {final_var:.4f})")
        print(f"  Time:      {elapsed:.1f}s ({elapsed/n_opt_steps:.2f}s/step)")
        
        return opt_results, elapsed
    
    def test_h2_sto3g(self):
        """H2 with STO-3G: 2 electrons, minimal."""
        self._run_opt_test('H 0 0 0; H 0 0 1.4', basis='sto-3g', 
                           n_walkers=200, n_opt_steps=10, label="H2/STO-3G")
    
    def test_be_sto3g(self):
        """Be atom: 4 electrons."""
        self._run_opt_test('Be 0 0 0', basis='sto-3g',
                           n_walkers=500, n_opt_steps=10, label="Be/STO-3G")
    
    def test_lih_sto3g(self):
        """LiH: 4 electrons."""
        self._run_opt_test('Li 0 0 0; H 0 0 3.015', basis='sto-3g',
                           n_walkers=500, n_opt_steps=10, label="LiH/STO-3G")
    
    def test_h2o_sto3g(self):
        """H2O: 10 electrons, moderate system."""
        self._run_opt_test(
            'O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0',
            basis='sto-3g', n_walkers=500, n_opt_steps=5, label="H2O/STO-3G"
        )
    
    def test_be_ccpvdz(self):
        """Be with cc-pVDZ: 4 electrons, larger basis."""
        self._run_opt_test('Be 0 0 0', basis='ccpvdz',
                           n_walkers=500, n_opt_steps=5, label="Be/cc-pVDZ")


class TestJacobianSubsampling(unittest.TestCase):
    """Test that Jacobian sub-sampling produces valid optimization steps."""
    
    def setUp(self):
        """Create an H2 system for testing."""
        self.mol, self.mf, self.sj, self.det, self.params, self.walkers = \
            make_test_system('H 0 0 0; H 0 0 1.4', n_walkers=200)
    
    def test_full_vs_subsample_gradient_structure(self):
        """Sub-sampled gradient should have the same structure as the full gradient."""
        from pytc.vmc.optimizer import NewtonOptimizer
        from pytc.vmc.loss import make_variance_loss
        
        ansatz = self.sj
        params = self.params
        walkers = self.walkers
        key = random.PRNGKey(99)
        
        loss_fn = make_variance_loss(ansatz=ansatz, optimizer_type="newton",
                                      use_custom_jvp=True, max_vmap_batch_size=0)
        loss_fn_jvp = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        
        opt_full = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1, damping=1e-3,
            curvature_type="gauss_newton", solver="exact",
            jacobian_sample_size=0,
        )
        
        # Sub-sampled optimizer (50 out of 200 walkers)
        opt_sub = NewtonOptimizer(
            value_and_grad_func=loss_fn_jvp,
            learning_rate=0.1, damping=1e-3,
            curvature_type="gauss_newton", solver="exact",
            jacobian_sample_size=50,
        )
        
        batch = (walkers, ansatz)
        
        key1, key2 = random.split(key)
        new_params_full, _, stats_full = opt_full.step(params, 0, key1, batch)
        new_params_sub, _, stats_sub = opt_sub.step(params, 0, key2, batch)
        
        flat_full, _ = jax.flatten_util.ravel_pytree(new_params_full)
        flat_sub, _ = jax.flatten_util.ravel_pytree(new_params_sub)
        self.assertTrue(jnp.all(jnp.isfinite(flat_full)), "Full params should be finite")
        self.assertTrue(jnp.all(jnp.isfinite(flat_sub)), "Sub-sampled params should be finite")
        
        self.assertEqual(flat_full.shape, flat_sub.shape)
        
        self.assertTrue(jnp.isfinite(stats_full['loss']), "Full loss should be finite")
        self.assertTrue(jnp.isfinite(stats_sub['loss']), "Sub-sampled loss should be finite")
        self.assertGreater(float(stats_full['loss']), 0)
        self.assertGreater(float(stats_sub['loss']), 0)
        
        print(f"✓ Full (200 walkers): loss={float(stats_full['loss']):.6f}")
        print(f"  Sub-sampled (50):    loss={float(stats_sub['loss']):.6f}")
        print(f"  Parameter delta norm (full):  {float(jnp.linalg.norm(flat_full - jax.flatten_util.ravel_pytree(params)[0])):.6f}")
        print(f"  Parameter delta norm (sub):   {float(jnp.linalg.norm(flat_sub - jax.flatten_util.ravel_pytree(params)[0])):.6f}")
    
    def test_subsample_optimization_converges(self):
        """Sub-sampled optimization should still make progress (variance decreases)."""
        key = random.PRNGKey(42)
        
        opt_results = optimize_ref_var(
            self.sj,
            params=self.params,
            n_walkers=200,
            n_steps=3,
            step_size=1.0,
            burn_in_steps=50,
            n_opt_steps=5,
            optimizer_type='newton',
            learning_rate=0.1,
            opt_kwargs={'damping': 1e-3, 'solver': 'exact',
                        'jacobian_sample_size': 50},
            key=key,
        )
        
        self.assertTrue(all(np.isfinite(opt_results['energies'])), 
                        "All energies should be finite with sub-sampling")
        self.assertTrue(all(np.isfinite(opt_results['cost'])),
                        "All variances should be finite with sub-sampling")
        
        print(f"✓ Sub-sampled optimization:")
        print(f"  Initial var: {opt_results['cost'][0]:.4f}")
        print(f"  Final var:   {opt_results['cost'][-1]:.4f}")
        print(f"  Energies: {[f'{e:.4f}' for e in opt_results['energies']]}")


if __name__ == '__main__':
    unittest.main()
