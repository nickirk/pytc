"""Direct tests for safe_hdf5_read and hdf5_slice_loader read path."""
import unittest
import numpy as np
import os
import tempfile


class TestSafeHDF5Read(unittest.TestCase):
    """Test both HDF5 and numpy branches of safe_hdf5_read."""

    def test_numpy_array_slice(self):
        """Plain numpy array: returns correct slice, no HDF5 lock needed."""
        from pytc.utils.prefetch import safe_hdf5_read

        arr = np.arange(100).reshape(10, 10)
        result = safe_hdf5_read(arr, (slice(2, 5), slice(None)))
        self.assertEqual(result.shape, (3, 10))
        np.testing.assert_array_equal(result, arr[2:5])

    def test_hdf5_dataset_slice(self):
        """HDF5 dataset: returns correct slice under lock."""
        from pytc.utils.prefetch import safe_hdf5_read
        import h5py

        arr = np.arange(100).reshape(10, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            with h5py.File(path, "w") as f:
                ds = f.create_dataset("data", data=arr)
                result = safe_hdf5_read(ds, (slice(3, 7), slice(None)))
            self.assertEqual(result.shape, (4, 10))
            np.testing.assert_array_equal(result, arr[3:7])

    def test_returns_numpy_array(self):
        """Result is always a numpy array (not h5py.Dataset)."""
        from pytc.utils.prefetch import safe_hdf5_read
        import h5py

        arr = np.arange(20).reshape(4, 5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            with h5py.File(path, "w") as f:
                ds = f.create_dataset("data", data=arr)
                result = safe_hdf5_read(ds, (slice(0, 2),))
            self.assertIsInstance(result, np.ndarray)


class TestHDF5SliceLoader(unittest.TestCase):
    """Test hdf5_slice_loader with both numpy arrays and HDF5 datasets."""

    def test_numpy_axis0(self):
        from pytc.utils.prefetch import hdf5_slice_loader

        arr = np.arange(100).reshape(10, 10)
        loader = hdf5_slice_loader(arr, axis=0)
        result = loader((2, 5))
        self.assertEqual(result.shape, (3, 10))
        np.testing.assert_array_equal(result, arr[2:5])

    def test_numpy_axis1(self):
        from pytc.utils.prefetch import hdf5_slice_loader

        arr = np.arange(100).reshape(10, 10)
        loader = hdf5_slice_loader(arr, axis=1)
        result = loader((3, 7))
        self.assertEqual(result.shape, (10, 4))
        np.testing.assert_array_equal(result, arr[:, 3:7])

    def test_hdf5_axis0(self):
        from pytc.utils.prefetch import hdf5_slice_loader
        import h5py

        arr = np.arange(100).reshape(10, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            with h5py.File(path, "w") as f:
                ds = f.create_dataset("data", data=arr)
                loader = hdf5_slice_loader(ds, axis=0)
                result = loader((4, 8))
            self.assertEqual(result.shape, (4, 10))
            np.testing.assert_array_equal(result, arr[4:8])

    def test_hdf5_and_numpy_identical(self):
        """Both branches must produce byte-identical results."""
        from pytc.utils.prefetch import hdf5_slice_loader
        import h5py

        arr = np.arange(200).reshape(10, 20)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            with h5py.File(path, "w") as f:
                ds = f.create_dataset("data", data=arr)
                hdf5_loader = hdf5_slice_loader(ds, axis=0)
                hdf5_result = hdf5_loader((3, 7))

        numpy_loader = hdf5_slice_loader(arr, axis=0)
        numpy_result = numpy_loader((3, 7))

        np.testing.assert_array_equal(hdf5_result, numpy_result)

    def test_no_shape_returns_dataset(self):
        """When dataset has no .shape, _load returns it directly."""
        from pytc.utils.prefetch import hdf5_slice_loader

        class NoShape:
            pass

        obj = NoShape()
        loader = hdf5_slice_loader(obj, axis=0)
        self.assertIs(loader((0, 1)), obj)


if __name__ == "__main__":
    unittest.main()
