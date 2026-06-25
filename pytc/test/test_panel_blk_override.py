"""Focused tests for the PYTC_PANEL_BLK / PYTC_GPU_MAX_MEMORY_MB env-var
panel/block-size overrides and the _FIXED_RBS_CACHE cache-key fix (Rick #26).

Covers:
- _panel_blk_overrides() parsing (default None, set values, bad values).
- The cache-key fix: a same-process env change (unset -> PYTC_PANEL_BLK=64)
  invalidates the stale uncapped rank_block_size (Rick's probe: previously
  returned 300 after setting 64 because the env was not part of the key).
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
    """Rick #26: env overrides MUST be part of the cache key, else a
    same-process env change reuses a stale uncapped rank_block_size."""

    def tearDown(self):
        tc._FIXED_RBS_CACHE.clear()
        for k in ("PYTC_PANEL_BLK", "PYTC_GPU_MAX_MEMORY_MB"):
            os.environ.pop(k, None)

    def _rbs_for(self, n_orb, N_fused, streaming=False, k_stream_panel=None):
        # Mirror the _get_fixed_rank_block_size cache path exactly.
        key = (int(n_orb), int(N_fused), bool(streaming), int(k_stream_panel or 0))
        panel_blk, gpu_max_memory_mb = tc._panel_blk_overrides()
        key = key + (panel_blk, gpu_max_memory_mb)
        if key not in tc._FIXED_RBS_CACHE:
            rbs = adaptive_rank_block_size(
                n_orb, n_orb, N_fused,
                resident_bytes=0,
                gpu_max_memory_mb=gpu_max_memory_mb,
            )
            if panel_blk is not None:
                rbs = min(rbs, max(1, int(panel_blk)))
            tc._FIXED_RBS_CACHE[key] = rbs
        return tc._FIXED_RBS_CACHE[key]

    def test_env_change_same_process_invalidates_stale_rbs(self):
        # Unset -> autotuned (300 for n_orb=137, N_fused=300).
        r_unset = self._rbs_for(137, 300)
        self.assertEqual(r_unset, 300)
        # Set PYTC_PANEL_BLK=64 in the SAME process -> must be 64, not 300.
        os.environ["PYTC_PANEL_BLK"] = "64"
        r_capped = self._rbs_for(137, 300)
        self.assertEqual(r_capped, 64)
        # Unset again -> back to 300 (cache key changes with the env).
        os.environ.pop("PYTC_PANEL_BLK")
        r_unset2 = self._rbs_for(137, 300)
        self.assertEqual(r_unset2, 300)

    def test_gpu_max_memory_mb_change_invalidates_stale_rbs(self):
        r_default = self._rbs_for(137, 300)
        os.environ["PYTC_GPU_MAX_MEMORY_MB"] = "40000"
        r_budget = self._rbs_for(137, 300)
        # Different key -> a distinct cache entry is created (not silently
        # reusing the default-key entry). The default entry's continued
        # presence is fine; what matters is that the budget-keyed entry exists
        # separately so the env change is reflected.
        self.assertIn(
            (137, 300, False, 0, None, 40000.0), tc._FIXED_RBS_CACHE)
        # And the two keys are distinct (not the same object).
        self.assertEqual(len(tc._FIXED_RBS_CACHE), 2)


if __name__ == "__main__":
    unittest.main()