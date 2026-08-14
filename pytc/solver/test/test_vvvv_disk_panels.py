"""Regression controls for bounded disk-backed VVVV panels."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from pytc.solver import xtc_ccsd


class _RecordingDataset:
    """Small stand-in that records HDF5 first-axis slab writes."""

    def __init__(self):
        self.writes = []

    def __setitem__(self, key, value):
        first_axis = key[0]
        self.writes.append((first_axis.start, first_axis.stop, value.shape, value.nbytes))


class _SynchronousWriter:
    """Keep the writer test deterministic while exercising its slab loop."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def submit(self, function, *args, **kwargs):
        function(*args)

    def drain(self):
        pass

    def log_summary(self, log):
        pass


class _FakeXTC:
    phi_isdf = None

    def __init__(self, nvir):
        self.nvir = nvir
        self.isdf_kernels = {"X": np.zeros((1, 1, 1))}

    def get_2b(self, params, *, ranges):
        width = ranges[0].stop - ranges[0].start
        return np.zeros((width, self.nvir, self.nvir, self.nvir))


class TestVVVVDiskPanels(unittest.TestCase):
    def test_mid_size_writer_uses_multiple_bounded_slabs(self):
        """A 60-virtual source must not regress to one 100-MiB disk write."""
        nocc = 1
        nvir = 60
        panel = 16
        cc = SimpleNamespace(
            gpu_max_memory=70_000,
            max_memory=120_800,
            vvvv_p_block_size=panel,
            vvvv_r_block_size=panel,
        )
        eris = SimpleNamespace(vvvv=_RecordingDataset())
        xtc_obj = _FakeXTC(nvir)
        mo_coeff = np.zeros((1, nocc + nvir))

        def fake_general(mol, orbitals, compact=False):
            width = orbitals[0].shape[1]
            return np.zeros(width * nvir ** 3)

        # Force the unbounded estimator to request one panel.  The explicit
        # cap must still drive the actual writer loop into four disk writes.
        with mock.patch.object(xtc_ccsd, "estimate_blksize", return_value=(nvir, 0)), \
             mock.patch.object(xtc_ccsd.ao2mo, "general", side_effect=fake_general), \
             mock.patch.object(xtc_ccsd, "_AsyncHDF5Writer", _SynchronousWriter):
            xtc_ccsd._compute_vvvv_block_ao2mo(
                eris, xtc_obj, None, None, mo_coeff, nocc, nvir, nocc + nvir, cc
            )

        self.assertEqual(
            [(start, stop) for start, stop, _, _ in eris.vvvv.writes],
            [(0, 16), (16, 32), (32, 48), (48, 60)],
        )
        self.assertEqual(
            eris.vvvv_write_receipt,
            {
                "nvir": nvir,
                "p_blksize": panel,
                "r_blksize": panel,
                "n_p_blocks": 4,
                "n_r_blocks": 4,
            },
        )
        self.assertTrue(all(shape[0] <= panel for _, _, shape, _ in eris.vvvv.writes))
        self.assertLess(max(size for _, _, _, size in eris.vvvv.writes), 32 * 1024**2)

    def test_df_writer_publishes_the_panel_receipt_it_uses(self):
        """The driver-visible receipt must come from the real DF writer loop."""
        nocc = 1
        nvir = 60
        panel = 16
        cc = SimpleNamespace(
            gpu_max_memory=24_000,
            vvvv_p_block_size=panel,
            vvvv_r_block_size=panel,
        )
        eris = SimpleNamespace(vvvv=_RecordingDataset())
        xtc_obj = _FakeXTC(nvir)
        device = SimpleNamespace(id=0)

        def synchronous_pipeline(specs, issue_tile, consume_tile, *, devices):
            for spec in specs:
                handle = issue_tile(spec, device)
                consume_tile(spec, device, handle, lambda: None)

        def fake_tile(xtc_obj, params, ranges, *, device, panel_size):
            p_len = ranges[0].stop - ranges[0].start
            r_len = ranges[2].stop - ranges[2].start
            return np.zeros((p_len, nvir, r_len, nvir))

        with mock.patch.object(
                xtc_ccsd, "resolve_vvvv_panel_block_sizes", return_value=(panel, panel)), \
             mock.patch.object(xtc_ccsd, "_solver_local_devices", return_value=[device]), \
             mock.patch.object(xtc_ccsd, "_round_robin_pipeline", side_effect=synchronous_pipeline), \
             mock.patch.object(xtc_ccsd.xtc_mod, "compute_2b_tile", side_effect=fake_tile), \
             mock.patch.object(xtc_ccsd, "_AsyncHDF5Writer", _SynchronousWriter):
            xtc_ccsd._compute_vvvv_block_df(
                eris, xtc_obj, None, np.zeros((nvir, nvir, 1)),
                nocc, nvir, nocc + nvir, cc,
            )

        self.assertEqual(
            eris.vvvv_write_receipt,
            {
                "nvir": nvir,
                "p_blksize": panel,
                "r_blksize": panel,
                "n_p_blocks": 4,
                "n_r_blocks": 4,
            },
        )
        self.assertEqual(len(eris.vvvv.writes), 4)

    def test_reader_and_writer_share_the_explicit_disk_cap(self):
        cc = SimpleNamespace(
            gpu_max_memory=70_000,
            max_memory=120_800,
            vvvv_p_block_size=16,
            vvvv_r_block_size=16,
        )
        with mock.patch.object(xtc_ccsd, "estimate_blksize", return_value=(60, 0)):
            self.assertEqual(
                xtc_ccsd.resolve_vvvv_disk_block_size(1, 60, cc, kind="vvvv"), 16
            )
            self.assertEqual(
                xtc_ccsd.resolve_vvvv_disk_block_size(1, 60, cc, kind="vvvv_gpu"), 16
            )

    def test_mismatched_panel_overrides_fail_closed(self):
        cc = SimpleNamespace(
            gpu_max_memory=2_000,
            max_memory=2_000,
            vvvv_p_block_size=16,
            vvvv_r_block_size=8,
        )
        with self.assertRaisesRegex(ValueError, "equal"):
            xtc_ccsd.resolve_vvvv_disk_block_size(1, 60, cc, kind="vvvv")


if __name__ == "__main__":
    unittest.main()
