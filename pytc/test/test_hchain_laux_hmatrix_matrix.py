"""Controls for the four-leg H-chain L_aux/X measurement driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


def _load_driver_module():
    driver_path = Path(__file__).resolve().parents[2] / "tools" / "hchain_laux_hmatrix_matrix.py"
    specification = importlib.util.spec_from_file_location("hchain_laux_hmatrix_matrix", driver_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class TestHChainLauxHMatrixMatrixDriver(unittest.TestCase):
    """The hierarchical leg must share only static gauge/ISDF state."""

    def test_static_clone_excludes_derived_laux_and_exchange_kernels(self):
        driver = _load_driver_module()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "direct.h5"
            target_path = Path(directory) / "hierarchical.h5"
            with h5py.File(source_path, "w") as source:
                source.create_dataset("xi_phi", data=np.arange(12.0).reshape(3, 4))
                source.create_dataset("xi_grad", data=np.arange(36.0).reshape(3, 4, 3))
                source.create_dataset("pivots", data=np.array([0, 1, 3]))
                source.create_dataset("phi_isdf", data=np.arange(9.0).reshape(3, 3))
                source.create_dataset("grad_phi_isdf", data=np.arange(27.0).reshape(3, 3, 3))
                source.create_dataset("mo_coeff", data=np.eye(3))
                source.create_dataset("K1_kernel", data=np.ones((3, 3, 3)))
                source.create_dataset("K3_kernel", data=np.ones((3, 3)))
                source.create_dataset("L_aux", data=np.ones((3, 4, 3)))
                source.create_dataset("H_aux", data=np.ones((3, 4)))
                source.create_dataset("D", data=np.ones((3, 3)))
                source.create_dataset("X", data=np.ones((3, 3, 3)))
                source.attrs["persistent"] = "keep"
                source.attrs["pytc_kmat_kernel_mode"] = "aux-recovery"
                source.attrs["pytc_xtc_x_mode"] = "full"
                driver.clone_static_isdf_cache(source, target_path)

            with h5py.File(target_path, "r") as target:
                self.assertEqual(target.attrs["persistent"], "keep")
                self.assertNotIn("pytc_kmat_kernel_mode", target.attrs)
                self.assertNotIn("pytc_xtc_x_mode", target.attrs)
                self.assertTrue(
                    {"xi_phi", "xi_grad", "pivots", "phi_isdf", "grad_phi_isdf", "mo_coeff"}
                    <= set(target.keys())
                )
                self.assertFalse(driver.DERIVED_DATASETS.intersection(target.keys()))

    def test_tucker_laux_streams_from_full_x_store_after_kernel_eviction(self):
        """ISDFXTC evicts L_aux in memory after building full X."""
        driver = _load_driver_module()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "direct.h5"
            with h5py.File(source_path, "w") as source:
                source.create_dataset("L_aux", data=np.arange(24.0).reshape(2, 4, 3))
                source.create_dataset("X", data=np.ones((2, 2, 2)))

                class FullXView:
                    isdf_kernels = {"X": source["X"]}

                l_aux = driver.tucker_l_aux_dataset(
                    FullXView(), {"path": str(source_path)}
                )
                self.assertIsInstance(l_aux, h5py.Dataset)
                np.testing.assert_array_equal(l_aux[:], source["L_aux"][:])


if __name__ == "__main__":
    unittest.main()
