"""Tests for pytc.utils.x_store (rank-innermost -> rank-major conversion)."""

from __future__ import annotations

import os
import tempfile
import unittest

import h5py
import numpy as np

from pytc.utils.x_store import convert_x_to_rank_major


class TestConvertXToRankMajor(unittest.TestCase):

    def test_roundtrip_content(self):
        rng = np.random.default_rng(20260730)
        x = rng.normal(size=(7, 7, 8))  # (nmo, nmo, rank), nmo != rank
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.h5")
            dst_path = os.path.join(tmp, "dst.h5")
            with h5py.File(src_path, "w") as fh:
                fh.create_dataset("X", data=x, dtype="f8")
            convert_x_to_rank_major(src_path, dst_path, row_block=3)
            with h5py.File(dst_path, "r") as fh:
                out = np.asarray(fh["X"])
        self.assertEqual(out.shape, (8, 7, 7))
        np.testing.assert_array_equal(out, x.transpose(2, 0, 1))

    def test_rejects_non_innermost_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.h5")
            dst_path = os.path.join(tmp, "dst.h5")
            with h5py.File(src_path, "w") as fh:
                fh.create_dataset("X", data=np.zeros((8, 7, 7)), dtype="f8")
            with self.assertRaises(ValueError):
                convert_x_to_rank_major(src_path, dst_path)


if __name__ == "__main__":
    unittest.main()
