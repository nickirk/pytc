"""Numerical regression test for ``kernel_process_ovvv_block``.

The production kernel was refactored to avoid paired ``'kdac'`` vs
``'kcad'`` einsum subscripts against the same ``ovvv_blk`` tensor. That
pattern forces XLA to fuse an axis-1↔axis-3 transpose of a large f64
tensor into several downstream contractions (``input_transpose_fusion``)
and, at pCV5Z-class shapes, can blow the autotuner:

    INTERNAL: Autotuning failed for HLO: %input_transpose_fusion ...
              NOT_FOUND: No valid config found!

The refactor materializes ``ovvv_swap = transpose(ovvv_blk, (0,3,2,1))``
once and rewrites every contraction in the native ``'kdac'`` form. That
is supposed to be *numerically* identical (up to floating-point
reassociation inside the contractions) to the textbook kernel. This
test pins that equivalence so future edits to the kernel cannot
silently drift.
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pytc.solver.jax_xtc_ccsd import kernel_process_ovvv_block


def _reference_kernel_process_ovvv_block(ovvv_blk, t1, t2, tau):
    """Textbook implementation with paired 'kdac'/'kcad' einsums.

    This is the implementation the production kernel used before the
    Fix-B transpose refactor. Kept here verbatim as the ground truth
    for the regression test.
    """
    t1_upd  = 2 * jnp.einsum('kdac,ikcd->ia', ovvv_blk, t2)
    t1_upd -=     jnp.einsum('kcad,ikcd->ia', ovvv_blk, t2)
    t1_upd += 2 * jnp.einsum('kdac,kd,ic->ia', ovvv_blk, t1, t1)
    t1_upd -=     jnp.einsum('kcad,kd,ic->ia', ovvv_blk, t1, t1)

    Lvv_blk  = 2 * jnp.einsum('kdac,kd->ac', ovvv_blk, t1)
    Lvv_blk -=     jnp.einsum('kcad,kd->ac', ovvv_blk, t1)

    Wvoov_blk = jnp.einsum('kcad,id->akic', ovvv_blk, t1)
    Wvovo_blk = jnp.einsum('kdac,id->akci', ovvv_blk, t1)

    tmp_a_blk = jnp.einsum('kdac,ijcd->kaij', ovvv_blk, tau)
    tmp_b_blk = jnp.einsum('kcbd,ijcd->kbij', ovvv_blk, tau)

    return t1_upd, Lvv_blk, Wvoov_blk, Wvovo_blk, tmp_a_blk, tmp_b_blk


_OUTPUT_NAMES = ("t1_upd", "Lvv_blk", "Wvoov_blk", "Wvovo_blk",
                 "tmp_a_blk", "tmp_b_blk")


class TestKernelProcessOvvvBlock(unittest.TestCase):
    def _run_both(self, nocc, nvir, blk, seed=0):
        rng = np.random.default_rng(seed)
        # ovvv_blk: (nocc, nvir, blk, nvir) — sliced along axis 2 (`a`).
        ovvv_blk = rng.standard_normal((nocc, nvir, blk, nvir))
        t1 = rng.standard_normal((nocc, nvir))
        t2 = rng.standard_normal((nocc, nocc, nvir, nvir))
        tau = rng.standard_normal((nocc, nocc, nvir, nvir))

        ref = _reference_kernel_process_ovvv_block(
            jnp.asarray(ovvv_blk), jnp.asarray(t1),
            jnp.asarray(t2), jnp.asarray(tau),
        )
        new = kernel_process_ovvv_block(
            jnp.asarray(ovvv_blk), jnp.asarray(t1),
            jnp.asarray(t2), jnp.asarray(tau),
        )
        return ref, new

    def _assert_equal(self, ref, new, rtol=1e-12, atol=1e-12):
        for name, r, n in zip(_OUTPUT_NAMES, ref, new):
            r_np = np.asarray(r)
            n_np = np.asarray(n)
            self.assertEqual(
                r_np.shape, n_np.shape,
                f"{name}: shape mismatch ref={r_np.shape} new={n_np.shape}",
            )
            max_abs = float(np.max(np.abs(r_np - n_np)))
            denom = max(float(np.max(np.abs(r_np))), 1.0)
            max_rel = max_abs / denom
            self.assertTrue(
                np.allclose(r_np, n_np, rtol=rtol, atol=atol),
                f"{name}: max|Δ|={max_abs:.3e}  max rel={max_rel:.3e}",
            )

    def test_square_block(self):
        """blk == nvir exercises the perfectly-square case."""
        ref, new = self._run_both(nocc=4, nvir=6, blk=6, seed=1)
        self._assert_equal(ref, new)

    def test_partial_block(self):
        """blk < nvir is the common case mid-pipeline."""
        ref, new = self._run_both(nocc=3, nvir=7, blk=4, seed=2)
        self._assert_equal(ref, new)

    def test_block_of_one(self):
        """blk == 1 is the tail tile at the end of a slab."""
        ref, new = self._run_both(nocc=3, nvir=5, blk=1, seed=3)
        self._assert_equal(ref, new)

    def test_nocc_equals_nvir(self):
        """A degenerate shape where nocc == nvir catches label-vs-axis
        bugs that go undetected when every dimension has a unique size."""
        ref, new = self._run_both(nocc=4, nvir=4, blk=4, seed=4)
        self._assert_equal(ref, new)

    def test_large_randomized(self):
        """Slightly bigger shape to make sure the refactor still agrees
        when the contractions produce non-trivial amounts of cancellation."""
        ref, new = self._run_both(nocc=5, nvir=9, blk=5, seed=5)
        # Looser tolerance here because the reference kernel evaluates
        # 2*A - B as two separate einsums while the new kernel folds
        # the linear combination into a single contraction, so
        # floating-point reassociation differs at the last bit or two.
        self._assert_equal(ref, new, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
