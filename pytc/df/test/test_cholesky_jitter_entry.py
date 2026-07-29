"""hermitian_sandwich_solve's cholesky_jitter mode.

Every test here exists because a specific earlier claim about this path was
wrong, not to cover the happy path twice:

1) The mode was benchmarked while an unconditional np.linalg.eigh ran BEFORE
   the branch was reached, so every "Cholesky" timing had silently paid for a
   full eigendecomposition. test_cholesky_never_calls_eigh instruments both
   eigh entry points rather than reading the source.
2) The device path accepted the mode, ran eig, and labelled the provenance
   "Cholesky" -- a false claim inside a data structure. Asserted still refused.
3) An earlier revision passed the spectral rtol (1e-4) in as the jitter scale,
   the same conflation the helper's own docstring records for tsvd_rcond.
4) The mode has no spectrum, so retained-mode keys must be ABSENT rather than
   filled with placeholders a caller could gate on.
"""

import subprocess
import sys
import unittest

import jax
import numpy as np

from pytc.df import solvers
from pytc.df.solvers import _DEFAULT_JITTER_RCOND, hermitian_sandwich_solve

jax.config.update("jax_enable_x64", True)

_RETAINED_MODE_KEYS = frozenset({
    "n_retained", "n_discarded", "s_max", "s_min_retained",
    "truncation_residual", "retained_solve_residual", "cond_pi_retained",
    "retention_marginal", "adaptive_retention_used", "n_retained_pin",
    "target_truncation_residual",
    # Was missing from this set while the docstring claimed no retained fields
    # are emitted; the helper returns it as None and it leaked through review.
    "retained_singular_value_range",
})


def _hermitian_pd_pair(n=64, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return A @ A.conj().T + n * np.eye(n), B @ B.conj().T


class TestCholeskyJitterEntry(unittest.TestCase):
    def setUp(self):
        self.Pi, self.V = _hermitian_pd_pair()

    def test_solves_the_stated_equation(self):
        W, _ = hermitian_sandwich_solve(self.Pi, self.V,
                                        retention_mode="cholesky_jitter")
        residual = (np.linalg.norm(self.Pi @ W @ self.Pi - self.V)
                    / np.linalg.norm(self.V))
        self.assertLess(residual, 1e-12)

    def test_cholesky_never_calls_eigh(self):
        # The regression that motivated the whole file: dispatch used to sit
        # AFTER the unconditional eigh. Counting real calls is the only check
        # that would have caught it -- the branch itself looked correct.
        calls = []
        real_np, real_jnp = np.linalg.eigh, jax.numpy.linalg.eigh

        def spy_np(*a, **kw):
            calls.append("numpy")
            return real_np(*a, **kw)

        def spy_jnp(*a, **kw):
            calls.append("jax")
            return real_jnp(*a, **kw)

        np.linalg.eigh = spy_np
        jax.numpy.linalg.eigh = spy_jnp
        try:
            hermitian_sandwich_solve(self.Pi, self.V,
                                     retention_mode="cholesky_jitter")
            self.assertEqual(calls, [], f"cholesky_jitter called eigh: {calls}")
            # Control: the spies do fire on the truncating path, so an empty
            # list above means "not called", not "spy never installed".
            hermitian_sandwich_solve(self.Pi, self.V, rtol=1e-10)
            self.assertTrue(calls, "spy never fired -- the assertion above was vacuous")
        finally:
            np.linalg.eigh = real_np
            jax.numpy.linalg.eigh = real_jnp

    def test_returns_helper_metadata(self):
        _, info = hermitian_sandwich_solve(self.Pi, self.V,
                                           retention_mode="cholesky_jitter")
        for key in ("solver", "jitter_used", "n_tries", "dtype", "fit_residual"):
            self.assertIn(key, info)
        self.assertEqual(info["solver"], "unscaled_cholesky_jitter")
        self.assertEqual(info["dtype"], "complex128")
        self.assertEqual(len(info["jitter_used"]), 2)
        # same_sector: Pi is on both sides, so one factorization is reused and
        # both entries must be the identical value, not merely close.
        self.assertEqual(info["jitter_used"][0], info["jitter_used"][1])
        self.assertEqual(info["n_tries"][0], info["n_tries"][1])

    def test_invents_no_retained_mode_fields(self):
        _, info = hermitian_sandwich_solve(self.Pi, self.V,
                                           retention_mode="cholesky_jitter")
        self.assertFalse(info["fallback_triggered"],
                         "precondition: this case must not have fallen back to TSVD")
        leaked = _RETAINED_MODE_KEYS & set(info)
        self.assertEqual(leaked, set(),
                         f"regularizing solver reported truncation fields: {sorted(leaked)}")
        self.assertIsNone(info["rtol"])

    def test_tsvd_fallback_is_reported_not_hidden(self):
        # The mode does NOT guarantee a Cholesky solve. The shared helper gates on
        # an unregularized-bias threshold and switches to TSVD when it trips, which
        # matters twice over: TSVD is a full SVD, so it is more expensive than the
        # eigh this mode exists to replace, and being a truncating solver it DOES
        # report n_retained. A caller that reads retention_mode and assumes it knows
        # which solver ran would be wrong; solver/fallback_triggered are the truth.
        _, info = hermitian_sandwich_solve(self.Pi, self.V, jitter_rcond=1e-6,
                                           retention_mode="cholesky_jitter")
        self.assertTrue(info["fallback_triggered"])
        self.assertEqual(info["solver"], "tsvd")
        self.assertIn("n_retained", info)
        self.assertIn("preceding_cholesky_jitter_used", info)

    def test_backend_label_names_the_solver_that_actually_ran(self):
        # Reporting backend='jax_cho_solve' beside solver='tsvd' asserted two
        # contradictory things about one solve.
        _, normal = hermitian_sandwich_solve(self.Pi, self.V,
                                             retention_mode="cholesky_jitter")
        self.assertFalse(normal["fallback_triggered"])
        self.assertEqual(normal["backend"], "jax_cho_solve")
        _, fell = hermitian_sandwich_solve(self.Pi, self.V, jitter_rcond=1e-6,
                                           retention_mode="cholesky_jitter")
        self.assertTrue(fell["fallback_triggered"])
        self.assertEqual(fell["solver"], "tsvd")
        self.assertEqual(fell["backend"], "jax_tsvd")

    def test_residual_convention_names_the_periodic_operands(self):
        # The shared helper spells the molecular fit problem. The arithmetic is
        # the same; the label named the wrong operands, which is a false claim
        # about which quantity was measured.
        _, info = hermitian_sandwich_solve(self.Pi, self.V,
                                           retention_mode="cholesky_jitter")
        self.assertEqual(info["residual_norm_convention"], "||Pi W Pi - V|| / ||V||")
        self.assertNotIn("CC", info["residual_norm_convention"])

    def test_rejects_rtol(self):
        with self.assertRaises(ValueError) as ctx:
            hermitian_sandwich_solve(self.Pi, self.V, rtol=1e-6,
                                     retention_mode="cholesky_jitter")
        self.assertIn("jitter_rcond", str(ctx.exception))

    def test_rejects_retention_arguments_that_presume_a_spectrum(self):
        for kwargs in ({"n_retained_pin": 10},
                       {"target_truncation_residual": 1e-8}):
            with self.assertRaises(ValueError):
                hermitian_sandwich_solve(self.Pi, self.V,
                                         retention_mode="cholesky_jitter", **kwargs)

    def test_jitter_rcond_defaults_to_1e_14_and_is_not_rtol(self):
        _, info = hermitian_sandwich_solve(self.Pi, self.V,
                                           retention_mode="cholesky_jitter")
        self.assertEqual(info["jitter_rcond"], 1e-14)
        self.assertEqual(_DEFAULT_JITTER_RCOND, 1e-14)
        # Independence, behaviourally: the jitter actually applied must track
        # jitter_rcond. Were rtol still feeding this scale, forcing a larger
        # jitter_rcond would leave jitter_used unchanged. Stops at 1e-11 because
        # 1e-10 trips the bias gate on this matrix and returns TSVD provenance,
        # which carries no jitter_used to compare.
        _, loose = hermitian_sandwich_solve(self.Pi, self.V, jitter_rcond=1e-11,
                                            retention_mode="cholesky_jitter")
        self.assertFalse(loose["fallback_triggered"])
        self.assertAlmostEqual(
            loose["jitter_used"][0] / info["jitter_used"][0], 1e3, delta=1.0)

    def test_jitter_rcond_rejected_on_truncating_modes(self):
        with self.assertRaises(ValueError) as ctx:
            hermitian_sandwich_solve(self.Pi, self.V, jitter_rcond=1e-14,
                                     retention_mode="single")
        self.assertIn("cholesky_jitter", str(ctx.exception))

    def test_device_path_still_refuses_the_mode(self):
        # Stage 3 work. Until then the device path must refuse rather than
        # quietly running eig under a Cholesky label, as it once did.
        Pi_d, V_d = self.Pi, self.V
        with self.assertRaises(ValueError):
            solvers.hermitian_sandwich_solve_device(
                Pi_d, V_d, retention_mode="cholesky_jitter")

    def test_agrees_with_the_truncating_solver_on_a_well_conditioned_system(self):
        W_chol, _ = hermitian_sandwich_solve(self.Pi, self.V,
                                             retention_mode="cholesky_jitter")
        W_eigh, _ = hermitian_sandwich_solve(self.Pi, self.V, rtol=1e-12)
        rel = np.linalg.norm(W_chol - W_eigh) / np.linalg.norm(W_eigh)
        self.assertLess(rel, 1e-10)


class TestX64Refusal(unittest.TestCase):
    """x64 is process-global and already on elsewhere in this suite, so the
    real production failure -- c128 in, silently downcast to c64 -- can only be
    reproduced in a fresh interpreter. A same-process c64-input test would
    exercise a different branch and quietly not cover the case that matters.
    """

    def test_refuses_when_x64_is_disabled(self):
        script = (
            "import numpy as np\n"
            "from pytc.df.solvers import hermitian_sandwich_solve\n"
            "import jax\n"
            "assert not jax.config.jax_enable_x64\n"
            "n = 16\n"
            "rng = np.random.default_rng(0)\n"
            "A = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))\n"
            "Pi = (A @ A.conj().T + n*np.eye(n)).astype(np.complex128)\n"
            "assert Pi.dtype == np.complex128\n"
            "try:\n"
            "    hermitian_sandwich_solve(Pi, Pi, retention_mode='cholesky_jitter')\n"
            "except ValueError as e:\n"
            "    assert 'x64' in str(e), e\n"
            "    print('REFUSED')\n"
            "else:\n"
            "    print('ACCEPTED')\n"
        )
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, timeout=300)
        self.assertIn("REFUSED", out.stdout,
                      f"stdout={out.stdout!r} stderr={out.stderr[-2000:]!r}")


if __name__ == "__main__":
    unittest.main()
