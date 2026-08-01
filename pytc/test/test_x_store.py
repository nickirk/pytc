"""Tests for the X-store layout helpers in pytc.df.thc."""

from __future__ import annotations

import os
import tempfile
import unittest

import h5py
import numpy as np

from pytc.df.thc import (add_rank_major, convert_store_to_rank_major,
    convert_x_to_rank_major)


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

    def test_whole_store_conversion(self):
        rng = np.random.default_rng(99)
        x = rng.normal(size=(7, 7, 8))
        k1 = rng.normal(size=(8, 8))
        d = rng.normal(size=(8, 8))
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.h5")
            dst_path = os.path.join(tmp, "dst.h5")
            with h5py.File(src_path, "w") as fh:
                fh.attrs["provenance"] = "toy-store-v1"
                fh.create_dataset("X", data=x, dtype="f8")
                fh.create_dataset("K1_kernel", data=k1, dtype="f8")
                fh.create_dataset("D", data=d, dtype="f8")
            convert_store_to_rank_major(src_path, dst_path, row_block=3)
            with h5py.File(dst_path, "r") as fh:
                self.assertEqual(fh.attrs["provenance"], "toy-store-v1")
                np.testing.assert_array_equal(np.asarray(fh["X"]),
                                              x.transpose(2, 0, 1))
                self.assertEqual(fh["X"].attrs["x_layout"], "rank_major")
                np.testing.assert_array_equal(np.asarray(fh["K1_kernel"]), k1)
                np.testing.assert_array_equal(np.asarray(fh["D"]), d)

    def test_add_rank_major_in_place(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(7, 7, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.h5")
            with h5py.File(path, "w") as fh:
                fh.create_dataset("X", data=x, dtype="f8")
                fh.create_dataset("D", data=np.ones((8, 8)), dtype="f8")
            add_rank_major(path, row_block=3)
            with h5py.File(path, "r") as fh:
                # X untouched, X_rm appended and correct.
                np.testing.assert_array_equal(np.asarray(fh["X"]), x)
                np.testing.assert_array_equal(np.asarray(fh["X_rm"]),
                                              x.transpose(2, 0, 1))
                self.assertEqual(fh["X_rm"].attrs["x_layout"], "rank_major")
            # Idempotent: a second run verifies and keeps.
            add_rank_major(path, row_block=3)
            # A corrupt X_rm is rejected, not silently kept.
            with h5py.File(path, "r+") as fh:
                del fh["X_rm"]
                fh.create_dataset("X_rm", data=np.zeros((3, 3, 3)), dtype="f8")
            with self.assertRaises(ValueError):
                add_rank_major(path)

    def test_integrity_guards(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(7, 7, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.h5")
            with h5py.File(path, "w") as fh:
                ds = fh.create_dataset("X", data=x, dtype="f8")
                ds.attrs["build_meta"] = "toy-build"
            # Invalid row_block is rejected, no silent zero-write.
            with self.assertRaises(ValueError):
                add_rank_major(path, row_block=0)
            with self.assertRaises(ValueError):
                convert_x_to_rank_major(path, os.path.join(tmp, "o.h5"),
                                        row_block=-2)
            # Source X attrs are preserved; no stray temp dataset remains.
            add_rank_major(path, row_block=3)
            with h5py.File(path, "r") as fh:
                self.assertEqual(fh["X_rm"].attrs["build_meta"], "toy-build")
                self.assertEqual(fh["X_rm"].attrs["x_layout"], "rank_major")
                self.assertNotIn("X_rm.tmp", fh)
            # A leftover temp from an interrupted run is cleaned on retry.
            with h5py.File(path, "r+") as fh:
                del fh["X_rm"]
                fh.create_dataset("X_rm.tmp", data=np.zeros((2, 2, 2)),
                                  dtype="f8")
            add_rank_major(path, row_block=3)
            with h5py.File(path, "r") as fh:
                np.testing.assert_array_equal(np.asarray(fh["X_rm"]),
                                              x.transpose(2, 0, 1))
                self.assertNotIn("X_rm.tmp", fh)


if __name__ == "__main__":
    unittest.main()
