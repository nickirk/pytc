import json
import unittest

import jax.numpy as jnp

from pytc.utils import tile_timers


class TestTileTimers(unittest.TestCase):
    """The timers are strictly opt-in: disabled = zero accumulation and
    zero behaviour change; enabled = terms accumulate with count/mean and
    the JSON dump has the documented shape."""

    def setUp(self):
        self._saved = tile_timers._ENABLED
        tile_timers._reset_for_tests()

    def tearDown(self):
        tile_timers._ENABLED = self._saved
        tile_timers._reset_for_tests()

    def test_disabled_accumulates_nothing(self):
        tile_timers._ENABLED = False
        with tile_timers.term("off_term") as t:
            value = t.sync(object())  # sync must be a pass-through
        self.assertIsNotNone(value)
        self.assertEqual(tile_timers.report(), {})

    def test_enabled_accumulates_count_and_mean(self):
        tile_timers._ENABLED = True
        for _ in range(3):
            with tile_timers.term("on_term") as t:
                t.sync(jnp.ones(4))
        rep = tile_timers.report()
        self.assertIn("on_term", rep)
        self.assertEqual(rep["on_term"]["count"], 3)
        self.assertGreater(rep["on_term"]["total_s"], 0.0)
        self.assertAlmostEqual(
            rep["on_term"]["mean_s"],
            rep["on_term"]["total_s"] / 3, places=12)

    def test_json_dump_shape(self):
        tile_timers._ENABLED = True
        with tile_timers.term("dump_term") as t:
            t.sync(jnp.ones(4))
        import tempfile, os
        saved_path = tile_timers._JSON_PATH
        try:
            with tempfile.TemporaryDirectory() as d:
                tile_timers._JSON_PATH = os.path.join(d, "timers.json")
                tile_timers._dump_at_exit()
                with open(tile_timers._JSON_PATH) as f:
                    payload = json.load(f)
            self.assertIn("terms", payload)
            self.assertIn("dump_term", payload["terms"])
            self.assertEqual(payload["terms"]["dump_term"]["count"], 1)
            self.assertGreater(payload["total_tracked_s"], 0.0)
        finally:
            tile_timers._JSON_PATH = saved_path

    def test_incr_noop_when_disabled_and_accumulates_when_enabled(self):
        tile_timers._ENABLED = False
        tile_timers.incr("off_counter")
        self.assertEqual(tile_timers._STATE["counters"], {})
        tile_timers._ENABLED = True
        tile_timers.incr("on_counter", 2)
        self.assertEqual(tile_timers._STATE["counters"].get("on_counter"), 2)


if __name__ == "__main__":
    unittest.main()
