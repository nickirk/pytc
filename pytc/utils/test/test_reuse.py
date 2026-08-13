import unittest

from pytc.utils.reuse import ReuseScope


class TestReuseScope(unittest.TestCase):
    def test_hit_miss_eviction_and_entry_bound(self):
        calls = []
        scope = ReuseScope(max_entries=2)

        self.assertEqual(scope.get_or_compute("a", lambda: calls.append("a") or 1), 1)
        self.assertEqual(scope.get_or_compute("a", lambda: 99), 1)
        self.assertEqual(scope.get_or_compute("b", lambda: calls.append("b") or 2), 2)
        self.assertEqual(scope.get_or_compute("c", lambda: calls.append("c") or 3), 3)

        self.assertEqual(calls, ["a", "b", "c"])
        self.assertEqual(scope.stats().hits, 1)
        self.assertEqual(scope.stats().misses, 3)
        self.assertEqual(scope.stats().evictions, 1)
        self.assertEqual(scope.stats().entries, 2)

    def test_failed_computation_is_not_cached(self):
        scope = ReuseScope(max_entries=1)

        def fail():
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            scope.get_or_compute("x", fail)
        self.assertEqual(scope.stats().entries, 0)
        self.assertEqual(scope.get_or_compute("x", lambda: 4), 4)
        self.assertEqual(scope.stats().misses, 2)

    def test_context_exit_releases_values(self):
        with ReuseScope(max_entries=1) as scope:
            scope.get_or_compute("x", lambda: object())
            self.assertEqual(scope.stats().entries, 1)
        self.assertEqual(scope.stats().entries, 0)

    def test_rejects_invalid_bounds(self):
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReuseScope(value)
        for value in (True, 1.5):
            with self.subTest(value=value), self.assertRaises(TypeError):
                ReuseScope(value)


if __name__ == "__main__":
    unittest.main()
