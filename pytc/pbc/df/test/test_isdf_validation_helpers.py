"""Direct tests for the shared validation helpers in pytc.pbc.df.isdf.

Both are called from more than one build path, so a change here reaches several
callers at once; the pin normaliser in particular replaced two copies that had
already drifted against each other.
"""

import unittest

import numpy as np

from pytc.pbc.df.isdf import _normalize_n_retained_pin, _require_mesh3


class TestRequireMesh3(unittest.TestCase):
    def test_returns_coerced_tuple(self):
        got = _require_mesh3([38, 38, 38])
        self.assertEqual(got, (38, 38, 38))
        self.assertTrue(all(isinstance(m, int) for m in got))

    def test_coerces_numpy_and_float_inputs(self):
        self.assertEqual(_require_mesh3(np.array([4, 5, 6])), (4, 5, 6))
        self.assertEqual(_require_mesh3([4.0, 5.0, 6.0]), (4, 5, 6))

    def test_anisotropic_mesh_is_fine(self):
        self.assertEqual(_require_mesh3((19, 38, 57)), (19, 38, 57))

    def test_rejects_wrong_length(self):
        for bad in ([38, 38], [38, 38, 38, 38], []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _require_mesh3(bad)

    def test_rejects_non_positive(self):
        for bad in ([0, 38, 38], [38, -1, 38], [38, 38, 0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _require_mesh3(bad)

    def test_message_names_the_argument(self):
        # Callers pass different meshes; a message that does not say which one
        # failed sends the reader to the wrong call site.
        with self.assertRaises(ValueError) as ctx:
            _require_mesh3([1, 2], name="staging_mesh")
        self.assertIn("staging_mesh", str(ctx.exception))


class TestNormalizeNRetainedPin(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(_normalize_n_retained_pin(None, 8))

    def test_scalar_broadcasts_to_every_q(self):
        # The case the device and host copies had already disagreed about:
        # a scalar pin is valid and must not be subscripted directly.
        self.assertEqual(_normalize_n_retained_pin(3, 4), [3, 3, 3, 3])

    def test_numpy_integer_scalar_broadcasts(self):
        self.assertEqual(_normalize_n_retained_pin(np.int64(2), 3), [2, 2, 2])

    def test_sequence_passes_through_as_list(self):
        got = _normalize_n_retained_pin((1, 2, 3), 3)
        self.assertEqual(got, [1, 2, 3])
        self.assertIsInstance(got, list)

    def test_ndarray_sequence_becomes_a_list(self):
        got = _normalize_n_retained_pin(np.array([4, 5]), 2)
        self.assertEqual(list(got), [4, 5])
        self.assertIsInstance(got, list)

    def test_rejects_wrong_length_sequence(self):
        with self.assertRaises(ValueError) as ctx:
            _normalize_n_retained_pin([1, 2], 3)
        self.assertIn("length 3", str(ctx.exception))

    def test_result_is_independent_of_the_input_object(self):
        # Callers index the result per q; aliasing the caller's array would let
        # a later mutation change a build already in progress.
        src = [1, 2, 3]
        got = _normalize_n_retained_pin(src, 3)
        src[0] = 99
        self.assertEqual(got, [1, 2, 3])

    def test_one_entry_per_q_for_every_input_form(self):
        for pin in (5, [5, 5, 5, 5], (5, 5, 5, 5), np.array([5, 5, 5, 5])):
            with self.subTest(pin=type(pin).__name__):
                self.assertEqual(len(_normalize_n_retained_pin(pin, 4)), 4)


class TestBothCallersAgree(unittest.TestCase):
    """The device and host builders must normalise pins identically.

    They previously held separate copies of this logic and drifted; the helper
    exists so they cannot. This asserts they still share it.
    """

    def test_device_and_host_builders_use_the_shared_helper(self):
        import inspect

        from pytc.pbc.df import isdf

        for fn in (isdf.build_coul_kpt_device, isdf.build_coul_kpt_host):
            with self.subTest(fn=fn.__name__):
                src = inspect.getsource(fn)
                self.assertIn("_normalize_n_retained_pin(", src)
                self.assertNotIn("pin_per_q = [n_retained_pin] * n_kpts", src)


if __name__ == "__main__":
    unittest.main()
