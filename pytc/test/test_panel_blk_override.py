"""Focused tests for the PYTC_PANEL_BLK / PYTC_GPU_MAX_MEMORY_MB env-var
panel/block-size overrides and the _FIXED_RBS_CACHE cache-key invariants.

Covers:
- _panel_blk_overrides() parsing (default None, set values, bad values).
- Cache-key correctness: a same-process env change invalidates a stale cached
  rank_block_size, and repeated same-env calls hit the cache without recompute.
"""
import os
import unittest

import pytc.tc as tc
from pytc.utils.gpu_memory import adaptive_rank_block_size


class TestPanelBlkOverrides(unittest.TestCase):
    def _overrides(self):
        return tc._panel_blk_overrides()

    def tearDown(self):
        for k in ("PYTC_PANEL_BLK", "PYTC_GPU_MAX_MEMORY_MB"):
            os.environ.pop(k, None)

    def test_default_unset_is_none(self):
        self.assertEqual(self._overrides(), (None, None))

    def test_panel_blk_set(self):
        os.environ["PYTC_PANEL_BLK"] = "64"
        self.assertEqual(self._overrides(), (64, None))

    def test_gpu_max_memory_mb_set(self):
        os.environ["PYTC_GPU_MAX_MEMORY_MB"] = "40000"
        self.assertEqual(self._overrides(), (None, 40000.0))

    def test_both_set(self):
        os.environ["PYTC_PANEL_BLK"] = "96"
        os.environ["PYTC_GPU_MAX_MEMORY_MB"] = "60000"
        self.assertEqual(self._overrides(), (96, 60000.0))

    def test_bad_panel_blk_ignored(self):
        for v in ("garbage", "", "0", "-5", "3.5"):
            os.environ["PYTC_PANEL_BLK"] = v
            self.assertIsNone(self._overrides()[0], f"bad value {v!r} should be ignored")

    def test_bad_gpu_max_memory_mb_ignored(self):
        for v in ("garbage", "", "nan", "inf", "-100"):
            os.environ["PYTC_GPU_MAX_MEMORY_MB"] = v
            self.assertIsNone(self._overrides()[1], f"bad value {v!r} should be ignored")


class TestFixedRbsCacheKey(unittest.TestCase):
    """Cache-key correctness for _FIXED_RBS_CACHE.

    Env overrides must be part of the cache key: a same-process env change
    must invalidate the stale cached rank_block_size and return a fresh value.
    Repeated same-env calls must hit the cache without re-entering
    adaptive_rank_block_size.

    These tests call the real ``ISDFTC._get_fixed_rank_block_size`` method
    via a stub instance.
    """

    def tearDown(self):
        tc._FIXED_RBS_CACHE.clear()
        for k in ("PYTC_PANEL_BLK", "PYTC_GPU_MAX_MEMORY_MB"):
            os.environ.pop(k, None)

    def _make_stub(self, n_orb=137, n_fused=300):
        # Minimal object exposing the attrs the real method reads.
        import jax.numpy as jnp
        class _Stub:
            pass
        s = _Stub()
        s.n_orb = n_orb
        s.phi_isdf = jnp.ones((n_orb, n_fused))  # sets N_fused = n_fused
        s.isdf_kernels = None                   # -> streaming=False
        return s

    def _install_counter(self):
        # Count re-entries into adaptive_rank_block_size via both the
        # gpu_memory module attr and tc's bound name.
        from pytc.utils import gpu_memory
        calls = {"n": 0}
        _orig = gpu_memory.adaptive_rank_block_size
        def _counting(*a, **k):
            calls["n"] += 1
            return _orig(*a, **k)
        gpu_memory.adaptive_rank_block_size = _counting
        tc.adaptive_rank_block_size = _counting
        self.addCleanup(setattr, gpu_memory, "adaptive_rank_block_size", _orig)
        return calls

    def _method(self):
        return tc.ISDFTC._get_fixed_rank_block_size

    def test_repeated_same_env_calls_hit_cache(self):
        calls = self._install_counter()
        m = self._method()
        s = self._make_stub()
        r1 = m(s); r2 = m(s); r3 = m(s)
        # Same env -> cache hits after the first call: adaptive called once.
        self.assertEqual((r1, r2, r3), (300, 300, 300))
        self.assertEqual(calls["n"], 1,
                         f"expected 1 adaptive call, got {calls['n']} (cache miss)")

    def test_env_change_same_process_invalidates_stale_rbs(self):
        calls = self._install_counter()
        m = self._method()
        s = self._make_stub()
        r_unset = m(s)
        self.assertEqual(r_unset, 300)
        os.environ["PYTC_PANEL_BLK"] = "64"
        r_capped = m(s)
        self.assertEqual(r_capped, 64)
        # New key -> adaptive re-entered.
        self.assertEqual(calls["n"], 2)
        os.environ.pop("PYTC_PANEL_BLK")
        r_unset2 = m(s)
        # Default key still cached from the first call -> cache hit (no new call).
        self.assertEqual(r_unset2, 300)
        self.assertEqual(calls["n"], 2)

    def test_env_change_repeat_capped_calls_hit_cache(self):
        calls = self._install_counter()
        m = self._method()
        s = self._make_stub()
        os.environ["PYTC_PANEL_BLK"] = "64"
        r1 = m(s); r2 = m(s)
        self.assertEqual((r1, r2), (64, 64))
        self.assertEqual(calls["n"], 1,
                         f"capped repeat should hit cache; adaptive={calls['n']}")

    def test_gpu_max_memory_mb_change_produces_distinct_cache_entry(self):
        calls = self._install_counter()
        m = self._method()
        s = self._make_stub()
        r_default = m(s)
        os.environ["PYTC_GPU_MAX_MEMORY_MB"] = "40000"
        r_budget = m(s)
        # Different key -> adaptive re-entered.
        self.assertEqual(calls["n"], 2)
        # Both keys are cached distinctly (2 entries for this (n_orb, N_fused)).
        self.assertEqual(
            len([k for k in tc._FIXED_RBS_CACHE
                 if k[0] == 137 and k[1] == 300]), 2)


if __name__ == "__main__":
    unittest.main()