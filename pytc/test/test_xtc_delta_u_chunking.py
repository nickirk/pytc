import os
import tempfile
import types
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


class TestDeltaUAutoshrinkGuard(unittest.TestCase):
    """Regression tests for the layout-aware genuine-OOM guard in _assemble_delta_u_tile."""

    def _make_fake_isdfxtc(self, nmo, n_rank):
        """Return a minimal ISDFXTC-duck with phi_isdf and bound methods."""
        rng = np.random.default_rng(42)
        phi = rng.normal(size=(nmo, n_rank)).astype(np.float64)
        fake = _FakeISDF(phi)
        # Wire up _assemble_delta_u_tile from the real class so the guard runs.
        import types
        fake._assemble_delta_u_tile = types.MethodType(
            xtc_mod.ISDFXTC._assemble_delta_u_tile, fake
        )
        fake._get_delta_u_direct_tile = types.MethodType(
            xtc_mod.ISDFXTC._get_delta_u_direct_tile, fake
        )
        fake._get_isdf_device_cache = None
        return fake

    def _make_kernels(self, nmo, n_rank):
        rng = np.random.default_rng(7)
        D = rng.normal(size=(n_rank, n_rank)).astype(np.float64)
        X_arr = rng.normal(size=(nmo, nmo, n_rank)).astype(np.float64)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = os.path.join(tmpdir.name, "x.h5")
        fh = h5py.File(path, "w")
        X = fh.create_dataset("X", data=X_arr)
        return {"D": D, "X": X}, fh

    def test_qr_layout_guard_does_not_raise_when_padded_axes_fit(self):
        # "qr" layout: p is NOT padded (full nvir), only q and r are padded.
        # safe_ps can be < p_len; the guard must NOT raise in that case.
        nmo, n_rank = 12, 4
        nocc, nvir = 2, 10  # p_len = nvir = 10; panel_size = max(nocc, blk) = 3
        panel_size = 3       # q_len = r_len = 3 ≤ panel_size → safe_ps = 3 ≥ 3 = min_padded
        fake = self._make_fake_isdfxtc(nmo, n_rank)
        kernels, fh = self._make_kernels(nmo, n_rank)

        # ranges for "qr" layout: slice_p covers all virtual (full nvir), slice_q and
        # slice_r are sub-panels of size panel_size.
        ranges = (
            slice(nocc, nmo),        # p — full virtual, NOT padded in "qr"
            slice(nocc, nocc + 3),   # q — padded axis, length 3 = panel_size
            slice(nocc, nocc + 3),   # r — padded axis, length 3 = panel_size
            slice(nocc, nmo),        # s — NOT padded in "qr" (only q and r are)
        )

        # Stub a large-enough free budget so isdf_tile_peak_bytes(3, ...) fits
        large_free = 10 * 1024 ** 3  # 10 GiB — easily fits a tiny tile
        with mock.patch("pytc.xtc._get_device_free_bytes", return_value=large_free):
            with mock.patch("pytc.utils.gpu_memory._get_gpu_free_bytes",
                            return_value=large_free):
                # Should not raise: safe_ps >= max(q_len=3, r_len=3) = 3
                result = fake._assemble_delta_u_tile(
                    kernels, ranges, device=None,
                    panel_size=panel_size, panel_layout="qr"
                )
        self.assertIsNotNone(result)
        fh.close()

    def test_pr_layout_genuine_oom_raises_with_layout_in_message(self):
        # When safe_ps < min_padded (here both p and r are padded and too large
        # to fit), RuntimeError must fire and mention the layout.
        nmo, n_rank = 8, 4
        panel_size = 6
        fake = self._make_fake_isdfxtc(nmo, n_rank)
        kernels, fh = self._make_kernels(nmo, n_rank)
        ranges = (
            slice(0, 6), slice(0, 8),
            slice(0, 6), slice(0, 8),
        )
        # Stub tiny free memory so no tile fits.
        tiny_free = 1  # 1 byte — nothing will fit
        with mock.patch("pytc.xtc._get_device_free_bytes", return_value=tiny_free):
            with self.assertRaises(RuntimeError) as ctx:
                fake._assemble_delta_u_tile(
                    kernels, ranges, device=None,
                    panel_size=panel_size, panel_layout="pr"
                )
        self.assertIn("layout=", str(ctx.exception))
        self.assertIn("min_padded=", str(ctx.exception))
        fh.close()

    def test_resident_d_not_double_counted_in_shrink_guard(self):
        # When D is already resident in the device cache, _get_device_free_bytes
        # has already excluded D's bytes from the free total.  The pre-dispatch
        # shrink guard must use include_d=False — otherwise it double-counts D
        # and can over-shrink panel_size or falsely trip the genuine-OOM guard.
        #
        # We wire a fake _get_isdf_device_cache that reports D as resident, then
        # record the include_d argument that _isdf_tile_peak_bytes receives from
        # the guard's _tile_bytes closure.  All calls must use include_d=False.
        nmo, n_rank = 8, 3
        panel_size = 4
        fake = self._make_fake_isdfxtc(nmo, n_rank)
        kernels, fh = self._make_kernels(nmo, n_rank)
        ranges = (slice(0, 4), slice(0, 8), slice(0, 4), slice(0, 8))

        D_sentinel = object()  # truthy sentinel — D is "resident"

        def fake_cache_getter(kernels, device=None,
                              include_grad=False, include_delta_u=False):
            # Must include phi_isdf so _get_delta_u_direct_tile can slice it.
            return {"D": D_sentinel, "phi_isdf": fake.phi_isdf}

        fake._get_isdf_device_cache = fake_cache_getter

        recorded_include_d = []
        original_tile_peak = xtc_mod._isdf_tile_peak_bytes

        def recording_tile_peak(Np, Nq, Nr, Ns, N_rank, include_d=True):
            recorded_include_d.append(include_d)
            return original_tile_peak(Np, Nq, Nr, Ns, N_rank, include_d=include_d)

        large_free = 10 * 1024 ** 3
        with mock.patch("pytc.xtc._isdf_tile_peak_bytes",
                        side_effect=recording_tile_peak):
            with mock.patch("pytc.xtc._get_device_free_bytes",
                            return_value=large_free):
                with mock.patch("pytc.utils.gpu_memory._get_gpu_free_bytes",
                                return_value=large_free):
                    fake._assemble_delta_u_tile(
                        kernels, ranges, device=None,
                        panel_size=panel_size, panel_layout="pr"
                    )

        self.assertTrue(recorded_include_d, "Expected _isdf_tile_peak_bytes to be called")
        self.assertTrue(
            all(not incl_d for incl_d in recorded_include_d),
            f"include_d=True seen — D was double-counted: {recorded_include_d}",
        )
        fh.close()


if __name__ == "__main__":
    unittest.main()
