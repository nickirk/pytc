"""Complex-correctness fix for pytc/df/solvers.py's structured
normal-equations solver (task #15 design review, isdf-coulomb-cuda,
2026-07-13).

`_build_normal_matrix`, `solve_normal_equations_batch`, and
`solve_normal_equations_batch_prepared` used plain `.T` throughout --
correct for real inputs, but WRONG for complex ones (least squares needs
the conjugate transpose `C^dagger`, not `C^T`). Independently reproduced
before fixing: an explicit dense `C[(p,q),mu]`/`B[(p,q),g]` +
`np.linalg.lstsq` oracle disagreed with the (buggy) structured solver by
relative error ~2.0 on a well-conditioned complex synthetic case; the
`.conj().T` fix reproduces the dense reference to 6.4e-10. Real-valued
inputs are bit-identically unaffected (`.conj()` is a no-op on real
arrays) -- this file also pins that exact parity.

This bug was latent, not exercised by any existing production caller
(TC/xTC's own ISDF fitting via `pytc/df/isdf.py:isdf_decompose` is
real-valued only today) -- task #15 (a free-space-Poisson interpolation-
vector builder that must support complex factors for a future periodic/
k-point extension) is the first caller that needs this path correct.

Enables jax_enable_x64 explicitly at module level (this module's own
float64 tolerances require it; must not depend on another test module
enabling it first).
"""
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pytc.df.solvers import (
    solve_normal_equations_batch,
    prepare_normal_equations_solver,
    solve_normal_equations_batch_prepared,
    prepare_spd_cholesky,
)


def _dense_lstsq_reference(phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch):
    """Independent oracle: explicitly form the pair-index matrices
    C[(p,q),mu] and B[(p,q),g] (materializing the pair index, unlike the
    production separable-structure solver) and solve via
    np.linalg.lstsq -- a genuinely different code path from the
    function under test."""
    phi_piv_p = np.asarray(phi_piv_p)
    phi_piv_q = np.asarray(phi_piv_q)
    phi_p_batch = np.asarray(phi_p_batch)
    phi_q_batch = np.asarray(phi_q_batch)
    n_p, n_fused = phi_piv_p.shape
    n_q = phi_piv_q.shape[0]
    C = np.einsum('pm,qm->pqm', phi_piv_p, phi_piv_q).reshape(n_p * n_q, n_fused)
    B = np.einsum('pg,qg->pqg', phi_p_batch, phi_q_batch).reshape(n_p * n_q, -1)
    X, *_ = np.linalg.lstsq(C, B, rcond=None)
    return X


class TestNormalEquationsComplexCorrectness(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _complex_case(self, n_orb=3, n_fused=3, n_grid=5):
        rng = self.rng
        phi_piv_p = jnp.asarray(rng.standard_normal((n_orb, n_fused))
                                 + 1j * rng.standard_normal((n_orb, n_fused)))
        phi_piv_q = jnp.asarray(rng.standard_normal((n_orb, n_fused))
                                 + 1j * rng.standard_normal((n_orb, n_fused)))
        phi_p_batch = jnp.asarray(rng.standard_normal((n_orb, n_grid))
                                   + 1j * rng.standard_normal((n_orb, n_grid)))
        phi_q_batch = jnp.asarray(rng.standard_normal((n_orb, n_grid))
                                   + 1j * rng.standard_normal((n_orb, n_grid)))
        return phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch

    def test_one_shot_solver_matches_dense_lstsq_complex(self):
        phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch = self._complex_case()
        X = solve_normal_equations_batch(phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch, rcond=1e-12)
        X_ref = _dense_lstsq_reference(phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch)
        np.testing.assert_allclose(np.asarray(X), X_ref, atol=1e-8, rtol=1e-8)

    def test_prepared_solver_matches_dense_lstsq_complex(self):
        phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch = self._complex_case()
        chol, lower = prepare_normal_equations_solver(phi_piv_p, phi_piv_q, rcond=1e-12)
        X = solve_normal_equations_batch_prepared(
            chol, lower, phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch)
        X_ref = _dense_lstsq_reference(phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch)
        np.testing.assert_allclose(np.asarray(X), X_ref, atol=1e-8, rtol=1e-8)

    def test_one_shot_and_prepared_agree_complex(self):
        phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch = self._complex_case()
        X_one_shot = solve_normal_equations_batch(
            phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch, rcond=1e-12)
        chol, lower = prepare_normal_equations_solver(phi_piv_p, phi_piv_q, rcond=1e-12)
        X_prepared = solve_normal_equations_batch_prepared(
            chol, lower, phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch)
        np.testing.assert_allclose(np.asarray(X_one_shot), np.asarray(X_prepared),
                                    atol=1e-10, rtol=1e-10)

    def test_real_inputs_are_bit_identical_before_after_fix(self):
        # The fix (.conj().T) must be a pure no-op for real arrays --
        # this pins that exact parity as a permanent regression guard.
        rng = np.random.default_rng(1)
        n_orb, n_fused, n_grid = 4, 3, 6
        phi_piv_p = jnp.asarray(rng.standard_normal((n_orb, n_fused)))
        phi_piv_q = jnp.asarray(rng.standard_normal((n_orb, n_fused)))
        phi_p_batch = jnp.asarray(rng.standard_normal((n_orb, n_grid)))
        phi_q_batch = jnp.asarray(rng.standard_normal((n_orb, n_grid)))

        X = solve_normal_equations_batch(phi_piv_p, phi_piv_q, phi_p_batch, phi_q_batch, rcond=1e-10)

        # Manually reconstruct the PRE-FIX (.T, not .conj().T) computation
        # inline -- since .conj() is a no-op on real input, this must
        # match the (already-fixed) production function exactly.
        gram_p = phi_piv_p.T @ phi_piv_p
        gram_q = phi_piv_q.T @ phi_piv_q
        ATA = gram_p * gram_q
        term_p = jnp.matmul(phi_piv_p.T, phi_p_batch)
        term_q = jnp.matmul(phi_piv_q.T, phi_q_batch)
        ATB = term_p * term_q
        diag_mean = jnp.mean(jnp.diag(ATA))
        jitter = diag_mean * 1e-10
        ATA_reg = ATA + jitter * jnp.eye(ATA.shape[0])
        X_pre_fix_equivalent = jnp.linalg.solve(ATA_reg, ATB)

        np.testing.assert_array_equal(np.asarray(X), np.asarray(X_pre_fix_equivalent))

    def test_return_info_false_is_backward_compatible_two_tuple(self):
        phi_piv_p, phi_piv_q, _, _ = self._complex_case()
        result = prepare_normal_equations_solver(phi_piv_p, phi_piv_q, rcond=1e-10)
        self.assertEqual(len(result), 2)
        chol, lower = result
        self.assertIsInstance(lower, bool)

    def test_return_info_true_returns_four_tuple_with_real_jitter_facts(self):
        phi_piv_p, phi_piv_q, _, _ = self._complex_case()
        result = prepare_normal_equations_solver(
            phi_piv_p, phi_piv_q, rcond=1e-10, return_info=True)
        self.assertEqual(len(result), 4)
        chol, lower, jitter_used, n_tries = result
        self.assertIsInstance(lower, bool)
        self.assertIsInstance(n_tries, int)
        self.assertGreaterEqual(n_tries, 1)
        self.assertTrue(np.isfinite(jitter_used))

        # Cross-check against prepare_spd_cholesky's own direct 4-tuple
        # for the SAME normal matrix -- must agree exactly (this is a
        # thin wrapper, not an independent computation).
        from pytc.df.solvers import _build_normal_matrix
        ata = _build_normal_matrix(phi_piv_p, phi_piv_q)
        chol_direct, lower_direct, jitter_direct, n_tries_direct = prepare_spd_cholesky(
            ata, rcond=1e-10)
        self.assertEqual(lower, lower_direct)
        self.assertEqual(n_tries, n_tries_direct)
        self.assertEqual(jitter_used, jitter_direct)
        np.testing.assert_array_equal(np.asarray(chol), np.asarray(chol_direct))


if __name__ == "__main__":
    unittest.main()
