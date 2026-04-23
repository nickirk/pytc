import os
import tempfile
import unittest
from unittest import mock

import h5py
import jax
import numpy as np

from pytc import xtc as xtc_mod


jax.config.update("jax_enable_x64", True)


class _FakeISDF:
    def __init__(self, phi_isdf):
        self.phi_isdf = np.asarray(phi_isdf)
        self.n_orb = self.phi_isdf.shape[0]
        self.gpu_max_memory = None

    def _get_fixed_rank_block_size(self):
        return 2

    _get_delta_u_direct_tile = xtc_mod.ISDFXTC._get_delta_u_direct_tile


class TestDeltaUChunking(unittest.TestCase):
    def test_chunked_contract_matches_direct_without_full_x_slice(self):
        rng = np.random.default_rng(9)
        n_orb = 8
        n_rank = 3
        phi_isdf = rng.normal(size=(n_orb, n_rank))
        D = rng.normal(size=(n_rank, n_rank))
        X_full = rng.normal(size=(n_orb, n_orb, n_rank))
        fake = _FakeISDF(phi_isdf)
        ranges = (slice(0, 2), slice(2, 4), slice(1, 6), slice(0, 4))

        kernels_direct = {"D": D, "X": X_full}
        with mock.patch("pytc.utils.gpu_memory._get_gpu_free_bytes", return_value=10**9):
            ref = xtc_mod.ISDFXTC._contract_delta_U_kernels(fake, kernels_direct, ranges)

        full_call = (ranges[2], ranges[3])
        read_calls = []
        original_read = xtc_mod._read_X_slice

        def tracking_read(X, slice_r, slice_s):
            read_calls.append((slice_r, slice_s))
            return original_read(X, slice_r, slice_s)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "x.h5")
            with h5py.File(path, "w") as fh:
                ds = fh.create_dataset("X", data=X_full)
                kernels_chunk = {"D": D, "X": ds}
                with mock.patch("pytc.utils.gpu_memory._get_gpu_free_bytes", return_value=2200):
                    with mock.patch("pytc.xtc._read_X_slice", side_effect=tracking_read):
                        got = xtc_mod.ISDFXTC._contract_delta_U_kernels(fake, kernels_chunk, ranges)

        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-10, rtol=1e-10)
        self.assertTrue(read_calls, "Expected chunked path to read X slices")
        self.assertNotIn(full_call, read_calls)
        self.assertGreater(len(read_calls), 1)


if __name__ == "__main__":
    unittest.main()
