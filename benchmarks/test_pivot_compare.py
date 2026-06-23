#!/usr/bin/env python
"""Synthetic unit tests for pivot_experiment.compare-results verdict logic.

Two critical cases per Rick R-v4:
  1. Identical residuals + shifted exact arrays → *inconclusive* (no false selection)
  2. Real residual collapse → *SELECTION*

These tests fabricate result NPZs to exercise the fail-closed validation and
verdict logic without requiring a real JAX GPU run. Run with:

    PYTHONPATH=. python benchmarks/test_pivot_compare.py
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Helpers to fabricate result NPZs that pass the fail-closed schema gates
# ---------------------------------------------------------------------------
_MANIFEST_SHA = "deadbeef" * 4
_SYSTEM = "H2O_ccpVDZ"


def _make_meta(device, mode, pivots, e_corr_isdf=None, with_ecorr=False):
    meta = {
        "system": _SYSTEM, "device": device, "mode": mode,
        "n_rank": len(pivots), "manifest_sha": _MANIFEST_SHA,
        "pivots": pivots, "with_ecorr": with_ecorr,
    }
    if e_corr_isdf is not None:
        meta["e_corr_isdf"] = e_corr_isdf
    return meta


def _save(d, path):
    np.savez(path, du_exact=d["ex"], du_isdf=d["is"], meta=json.dumps(d["meta"]))


# ---------------------------------------------------------------------------
# The compare_results function (imported from the main script so we test the
# real code path).
# ---------------------------------------------------------------------------
def _run_compare(cpu_native, gpu_native, gpu_fixed, exact_tol=1e-9):
    """Invoke cmd_compare with the given NPZs and return the verdict dict."""
    import argparse
    from benchmarks.pivot_experiment import cmd_compare

    ns = argparse.Namespace(
        cpu_native=cpu_native,
        gpu_native=gpu_native,
        gpu_fixed=gpu_fixed,
        exact_tol=exact_tol,
    )
    # Capture stdout
    import io
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        cmd_compare(ns)
    finally:
        sys.stdout = old_stdout
    return json.loads(buf.getvalue())


class TestPivotCompare(unittest.TestCase):
    """Synthetic verdict tests."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _path(self, name):
        return os.path.join(self.tmp, name)

    # ------------------------------------------------------------------
    # Case 1: identical residuals + shifted exact → INCONCLUSIVE
    #   The exact path differs between CPU and GPU, which means the ISDF
    #   residuals are being computed over a different base. The verdict
    #   must NOT falsely attribute this to pivot selection.
    # ------------------------------------------------------------------
    def test_identical_residuals_shifted_exact(self):
        pivots = [0, 1, 2, 3, 4]

        # CPU native: exact ≠ GPU exact, but ISDF residuals are identical
        arr = np.random.RandomState(42).randn(10, 10).astype(np.float64)
        ex_cpu = arr.copy()
        is_cpu = arr + 0.001  # isdf_dU_err ~1e-3 on GPU-like scale
        shift = 1e-6 * np.ones_like(arr)
        ex_gpu = arr + shift       # exact shifted by 1e-6 = 1000× tol
        is_gpu = ex_gpu + (is_cpu - ex_cpu)  # same residual as CPU
        ex_fixed = ex_gpu.copy()
        is_fixed = ex_gpu + (is_cpu - ex_cpu)  # same residual (fixed doesn't help)

        _save({"ex": ex_cpu, "is": is_cpu, "meta": _make_meta("cpu", "native", pivots)},
              self._path("c_nat.npz"))
        _save({"ex": ex_gpu, "is": is_gpu, "meta": _make_meta("gpu", "native", pivots)},
              self._path("g_nat.npz"))
        _save({"ex": ex_fixed, "is": is_fixed, "meta": _make_meta("gpu", "fixed", pivots)},
              self._path("g_fix.npz"))

        res = _run_compare(self._path("c_nat.npz"),
                           self._path("g_nat.npz"),
                           self._path("g_fix.npz"))
        self.assertIn("inconclusive", res["verdict"].lower(),
                      f"shifted exact should be inconclusive, got: {res['verdict']}")
        self.assertIn("exact-path control failed", res["verdict"],
                      "verdict should mention exact-path failure")

    # ------------------------------------------------------------------
    # Case 2: real residual collapse → SELECTION
    #   CPU and GPU exact paths are identical (frozen-state control holds),
    #   GPU native shows a large residual gap, but fixed-GPU(cpu-pivots)
    #   collapses the residual back to the CPU level.
    # ------------------------------------------------------------------
    def test_residual_collapse_selection(self):
        piv_cpu = [0, 1, 2, 3, 4]
        piv_gpu = [0, 1, 7, 3, 4]  # different GPU selection (pivot 2→7)

        arr = np.random.RandomState(99).randn(10, 10).astype(np.float64)
        r_cpu = arr * 1e-12     # CPU residual ~0
        r_gpu_nat = arr * 1e-3  # GPU native residual is LARGE (different pivots)
        r_gpu_fix = arr * 1e-12 # fixed-GPU residual collapses back (CPU pivots forced)

        # Exact paths are identical (frozen-state control holds):
        ex_cpu = arr.copy()
        ex_gpu = arr.copy()
        ex_fixed = arr.copy()

        is_cpu = ex_cpu + r_cpu
        is_gpu_nat = ex_gpu + r_gpu_nat
        is_gpu_fix = ex_fixed + r_gpu_fix

        _save({"ex": ex_cpu, "is": is_cpu,
               "meta": _make_meta("cpu", "native", piv_cpu)},
              self._path("c_nat.npz"))
        _save({"ex": ex_gpu, "is": is_gpu_nat,
               "meta": _make_meta("gpu", "native", piv_gpu)},
              self._path("g_nat.npz"))
        _save({"ex": ex_fixed, "is": is_gpu_fix,
               "meta": _make_meta("gpu", "fixed", piv_cpu)},  # fixed uses CPU pivots
              self._path("g_fix.npz"))

        res = _run_compare(self._path("c_nat.npz"),
                           self._path("g_nat.npz"),
                           self._path("g_fix.npz"))
        self.assertIn("SELECTION", res["verdict"],
                      f"residual collapse should be SELECTION, got: {res['verdict']}")

    # ------------------------------------------------------------------
    # Case 3: native residual gap too small → INCONCLUSIVE
    # ------------------------------------------------------------------
    def test_no_native_gap(self):
        pivots = [0, 1, 2]
        arr = np.random.RandomState(7).randn(10, 10).astype(np.float64)
        ex = arr.copy()
        r_small = arr * 1e-15
        is_c = ex + r_small
        is_g = ex + r_small
        is_f = ex + r_small

        _save({"ex": ex, "is": is_c, "meta": _make_meta("cpu", "native", pivots)},
              self._path("c_nat.npz"))
        _save({"ex": ex, "is": is_g, "meta": _make_meta("gpu", "native", pivots)},
              self._path("g_nat.npz"))
        _save({"ex": ex, "is": is_f, "meta": _make_meta("gpu", "fixed", pivots)},
              self._path("g_fix.npz"))

        res = _run_compare(self._path("c_nat.npz"),
                           self._path("g_nat.npz"),
                           self._path("g_fix.npz"))
        self.assertIn("inconclusive", res["verdict"].lower())
        self.assertIn("no meaningful native", res["verdict"])

    # ------------------------------------------------------------------
    # Case 4: e_corr fail-closed — with_ecorr without values → FATAL
    # ------------------------------------------------------------------
    def test_ecorr_fail_closed_missing_values(self):
        pivots = [0, 1, 2]
        arr = np.random.RandomState(1).randn(10, 10).astype(np.float64)

        _save({"ex": arr, "is": arr,
               "meta": _make_meta("cpu", "native", pivots, with_ecorr=True, e_corr_isdf=-0.4)},
              self._path("c_nat.npz"))
        _save({"ex": arr, "is": arr,
               "meta": _make_meta("gpu", "native", pivots, with_ecorr=True, e_corr_isdf=-0.4)},
              self._path("g_nat.npz"))
        # Fixed result has with_ecorr=True but NO e_corr_isdf key
        _save({"ex": arr, "is": arr,
               "meta": _make_meta("gpu", "fixed", pivots, with_ecorr=True)},
              self._path("g_fix.npz"))

        with self.assertRaises(SystemExit) as ctx:
            _run_compare(self._path("c_nat.npz"),
                         self._path("g_nat.npz"),
                         self._path("g_fix.npz"))
        self.assertIn("FATAL", str(ctx.exception))
        self.assertIn("e_corr_isdf", str(ctx.exception))

    # ------------------------------------------------------------------
    # Case 5: e_corr fail-closed — success with all finite values
    # ------------------------------------------------------------------
    def test_ecorr_success_with_all_finite(self):
        pivots = [0, 1, 2]
        arr = np.random.RandomState(2).randn(10, 10).astype(np.float64)

        _save({"ex": arr, "is": arr,
               "meta": _make_meta("cpu", "native", pivots, with_ecorr=True, e_corr_isdf=-0.3968)},
              self._path("c_nat.npz"))
        _save({"ex": arr, "is": arr,
               "meta": _make_meta("gpu", "native", pivots, with_ecorr=True, e_corr_isdf=-0.3969)},
              self._path("g_nat.npz"))
        _save({"ex": arr, "is": arr,
               "meta": _make_meta("gpu", "fixed", pivots, with_ecorr=True, e_corr_isdf=-0.3968)},
              self._path("g_fix.npz"))

        res = _run_compare(self._path("c_nat.npz"),
                           self._path("g_nat.npz"),
                           self._path("g_fix.npz"))
        self.assertIn("e_corr_isdf_cpu", res)
        self.assertIn("e_corr_isdf_gpu_native", res)
        self.assertIn("e_corr_isdf_gpu_fixed", res)
        self.assertIn("e_corr_isdf_native_cpu_vs_gpu_delta", res)
        self.assertIn("e_corr_isdf_fixed_cpu_vs_gpu_delta", res)

    # ------------------------------------------------------------------
    # Case 5b: e_corr fail-closed — NaN value → FATAL
    # ------------------------------------------------------------------
    def test_ecorr_nan_is_fatal(self):
        pivots = [0, 1, 2]
        arr = np.random.RandomState(3).randn(10, 10).astype(np.float64)

        _save({"ex": arr, "is": arr,
               "meta": _make_meta("cpu", "native", pivots, with_ecorr=True, e_corr_isdf=-0.4)},
              self._path("c_nat.npz"))
        _save({"ex": arr, "is": arr,
               "meta": _make_meta("gpu", "native", pivots, with_ecorr=True, e_corr_isdf=-0.4)},
              self._path("g_nat.npz"))
        _save({"ex": arr, "is": arr,
               "meta": _make_meta("gpu", "fixed", pivots, with_ecorr=True, e_corr_isdf=float("nan"))},
              self._path("g_fix.npz"))

        with self.assertRaises(SystemExit) as ctx:
            _run_compare(self._path("c_nat.npz"),
                         self._path("g_nat.npz"),
                         self._path("g_fix.npz"))
        self.assertIn("FATAL", str(ctx.exception))

    # ------------------------------------------------------------------
    # Case 5c: e_corr fail-closed — boolean value → FATAL (bool is int subclass)
    # ------------------------------------------------------------------
    def test_ecorr_boolean_is_fatal(self):
        pivots = [0, 1, 2]
        arr = np.random.RandomState(4).randn(10, 10).astype(np.float64)

        _save({"ex": arr, "is": arr,
               "meta": _make_meta("cpu", "native", pivots, with_ecorr=True, e_corr_isdf=-0.4)},
              self._path("c_nat.npz"))
        _save({"ex": arr, "is": arr,
               "meta": _make_meta("gpu", "native", pivots, with_ecorr=True, e_corr_isdf=-0.4)},
              self._path("g_nat.npz"))
        _save({"ex": arr, "is": arr,
               "meta": _make_meta("gpu", "fixed", pivots, with_ecorr=True, e_corr_isdf=True)},
              self._path("g_fix.npz"))

        with self.assertRaises(SystemExit) as ctx:
            _run_compare(self._path("c_nat.npz"),
                         self._path("g_nat.npz"),
                         self._path("g_fix.npz"))
        self.assertIn("FATAL", str(ctx.exception))

    # ------------------------------------------------------------------
    # Case 6: fixed-leg exact path fails → INCONCLUSIVE
    # ------------------------------------------------------------------
    def test_fixed_exact_path_divergence(self):
        pivots = [0, 1, 2, 3]
        arr = np.random.RandomState(55).randn(10, 10).astype(np.float64)
        r_c = arr * 1e-12
        r_g = arr * 1e-3   # large native residual

        ex_c = arr.copy()
        ex_g = arr.copy()   # native exact path matches
        ex_f = arr + 1e-5   # fixed exact path DIVERGES (frozen control failed on fixed)

        is_c = ex_c + r_c
        is_g = ex_g + r_g
        is_f = ex_f + r_c   # fixed residual = CPU residual (collapse) BUT exact diverged

        _save({"ex": ex_c, "is": is_c, "meta": _make_meta("cpu", "native", pivots)},
              self._path("c_nat.npz"))
        _save({"ex": ex_g, "is": is_g, "meta": _make_meta("gpu", "native", pivots)},
              self._path("g_nat.npz"))
        _save({"ex": ex_f, "is": is_f, "meta": _make_meta("gpu", "fixed", pivots)},
              self._path("g_fix.npz"))

        res = _run_compare(self._path("c_nat.npz"),
                           self._path("g_nat.npz"),
                           self._path("g_fix.npz"))
        self.assertIn("inconclusive", res["verdict"].lower(),
                      f"fixed exact divergence should be inconclusive, got: {res['verdict']}")
        self.assertIn("fixed exact delta", res["verdict"])


if __name__ == "__main__":
    unittest.main()
