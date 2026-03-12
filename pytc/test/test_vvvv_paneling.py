import os
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import lib

from pytc import xtc as xtc_mod
from pytc.solver import jax_xtc_ccsd, xtc_ccsd
from pytc.utils import gpu_memory


jax.config.update("jax_enable_x64", True)


class _FakeXTC:
    def __init__(self, tc_full, nocc, n_fused=12, return_jax=False):
        self._tc_full = np.asarray(tc_full)
        self._nocc = nocc
        self._return_jax = return_jax
        self.phi_isdf = np.zeros((nocc + tc_full.shape[0], n_fused))
        self.isdf_kernels = {"K1_kernel": 0, "K3_kernel": 0, "D": 0, "X": 0}

    def get_2b(self, jastrow_params, ranges):
        slice_p, slice_q, slice_r, slice_s = ranges
        block = self._tc_full[
            slice(slice_p.start - self._nocc, slice_p.stop - self._nocc),
            slice(slice_q.start - self._nocc, slice_q.stop - self._nocc),
            slice(slice_r.start - self._nocc, slice_r.stop - self._nocc),
            slice(slice_s.start - self._nocc, slice_s.stop - self._nocc),
        ]
        if self._return_jax:
            return jnp.asarray(block)
        return np.asarray(block)

    def _assemble_2b_tile(self, jastrow_params, kernels, ranges, device=None, panel_size=None):
        del jastrow_params, kernels
        block = np.asarray(self.get_2b(None, ranges))
        if panel_size is not None:
            padded = np.zeros((panel_size, block.shape[1], panel_size, block.shape[3]))
            padded[:block.shape[0], :, :block.shape[2], :] = block
            block = padded
        if self._return_jax:
            if device is not None:
                return jax.device_put(block, device)
            return jnp.asarray(block)
        return np.asarray(block)


def _pack_vvL(l_vv_full):
    return lib.pack_tril(l_vv_full.transpose(2, 0, 1)).T


class TestVVVVPanelSizing(unittest.TestCase):
    def test_auto_defaults_to_estimated_square_tiles(self):
        cc = SimpleNamespace(
            gpu_max_memory=4096,
            vvvv_p_block_size=None,
            vvvv_r_block_size=None,
        )
        with mock.patch.object(gpu_memory, "estimate_vvvv_panel_blksize", return_value=(7, 1024)):
            p_blk, r_blk = gpu_memory.resolve_vvvv_panel_block_sizes(
                2, 11, gpu_max_memory_mb=cc.gpu_max_memory)
        self.assertEqual((p_blk, r_blk), (7, 7))

    def test_partial_override_keeps_other_axis_auto(self):
        cc = SimpleNamespace(
            gpu_max_memory=4096,
            vvvv_p_block_size=3,
            vvvv_r_block_size=None,
        )
        with mock.patch.object(gpu_memory, "estimate_vvvv_panel_blksize", return_value=(7, 1024)):
            p_blk, r_blk = gpu_memory.resolve_vvvv_panel_block_sizes(
                2, 11,
                p_block_size=cc.vvvv_p_block_size,
                r_block_size=cc.vvvv_r_block_size,
                gpu_max_memory_mb=cc.gpu_max_memory)
        self.assertEqual((p_blk, r_blk), (3, 3))

    def test_equal_overrides_win(self):
        cc = SimpleNamespace(
            gpu_max_memory=4096,
            vvvv_p_block_size=3,
            vvvv_r_block_size=3,
        )
        with mock.patch.object(gpu_memory, "estimate_vvvv_panel_blksize", return_value=(7, 1024)):
            p_blk, r_blk = gpu_memory.resolve_vvvv_panel_block_sizes(
                2, 11,
                p_block_size=cc.vvvv_p_block_size,
                r_block_size=cc.vvvv_r_block_size,
                gpu_max_memory_mb=cc.gpu_max_memory)
        self.assertEqual((p_blk, r_blk), (3, 3))

    def test_mismatched_overrides_raise(self):
        cc = SimpleNamespace(
            gpu_max_memory=4096,
            vvvv_p_block_size=3,
            vvvv_r_block_size=5,
        )
        with self.assertRaises(ValueError):
            gpu_memory.resolve_vvvv_panel_block_sizes(
                2, 11,
                p_block_size=cc.vvvv_p_block_size,
                r_block_size=cc.vvvv_r_block_size,
                gpu_max_memory_mb=cc.gpu_max_memory)


class TestSolverRoundRobin(unittest.TestCase):
    def test_round_robin_pipeline_cycles_devices(self):
        seen = []

        def issue(spec, device):
            seen.append(("issue", spec, device))
            return spec * 10

        def consume(spec, device, handle):
            seen.append(("consume", spec, device, handle))

        xtc_ccsd._round_robin_pipeline(
            [0, 1, 2, 3, 4],
            issue,
            consume,
            devices=("d0", "d1"),
        )

        issue_devices = [entry[2] for entry in seen if entry[0] == "issue"]
        consume_devices = [entry[2] for entry in seen if entry[0] == "consume"]
        self.assertEqual(issue_devices, ["d0", "d1", "d0", "d1", "d0"])
        self.assertEqual(consume_devices, ["d0", "d1", "d0", "d1", "d0"])

    def test_forced_two_local_devices_drive_jax_vvvv_scheduler(self):
        script = textwrap.dedent(
            """
            import json
            import os
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

            import jax
            import jax.numpy as jnp
            import numpy as np
            from pyscf import lib
            from types import SimpleNamespace

            from pytc.solver import jax_xtc_ccsd, xtc_ccsd
            from pytc import xtc as xtc_mod

            calls = []

            def fake_compute(xtc_obj, jastrow_params, ranges, device=None, panel_size=None):
                calls.append(getattr(device, "id", None))
                p = ranges[0].stop - ranges[0].start
                q = ranges[1].stop - ranges[1].start
                r = ranges[2].stop - ranges[2].start
                s = ranges[3].stop - ranges[3].start
                block = np.zeros((p, q, r, s))
                if panel_size is not None:
                    padded = np.zeros((panel_size, q, panel_size, s))
                    padded[:p, :, :r, :] = block
                    block = padded
                if device is not None:
                    return jax.device_put(block, device)
                return jnp.asarray(block)

            xtc_mod.compute_2b_tile = fake_compute

            nocc = 1
            nvir = 4
            naux = 1
            nmo = nocc + nvir
            l_vv_full = np.zeros((nvir, nvir, naux))
            vvL = lib.pack_tril(l_vv_full.transpose(2, 0, 1)).T

            cc = SimpleNamespace(
                nocc=nocc,
                nmo=nmo,
                xtc_obj=SimpleNamespace(phi_isdf=np.zeros((nmo, 3))),
                jastrow_params=None,
                with_df=object(),
                _scf=SimpleNamespace(with_df=None),
                gpu_max_memory=4096,
                max_memory=4096,
                vvvv_p_block_size=2,
                vvvv_r_block_size=2,
                mol=None,
            )
            eris = SimpleNamespace(vvvv=None, vvL=vvL)
            out = np.zeros((nocc, nocc, nvir, nvir))

            jax_xtc_ccsd._contract_vvvv_t2(cc, jnp.zeros_like(out), eris, out)
            print(json.dumps({"n_devices": jax.local_device_count(), "calls": calls}))
            """
        )
        env = os.environ.copy()
        proc = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            cwd="/Users/keliao/Work/project/pytc",
            env=env,
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["n_devices"], 2)
        self.assertEqual(payload["calls"], [0, 1, 0, 1])


class TestVVVVPaneling(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.nocc = 1
        self.nvir = 5
        self.nmo = self.nocc + self.nvir
        self.naux = 3
        self.tc_full = rng.normal(size=(self.nvir, self.nvir, self.nvir, self.nvir))
        l_raw = rng.normal(size=(self.nvir, self.nvir, self.naux))
        self.l_vv_full = 0.5 * (l_raw + l_raw.transpose(1, 0, 2))
        self.vvvv_full = self.tc_full + np.tensordot(
            self.l_vv_full, self.l_vv_full, axes=((2,), (2,))
        )

    def test_compute_vvvv_block_df_matches_dense_reference(self):
        xtc_obj = _FakeXTC(self.tc_full, self.nocc, return_jax=False)
        cc = SimpleNamespace(
            gpu_max_memory=4096,
            max_memory=4096,
            vvvv_p_block_size=2,
            vvvv_r_block_size=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vvvv.h5")
            with h5py.File(path, "w") as fh:
                eris = SimpleNamespace(vvvv=fh.create_dataset("vvvv", self.vvvv_full.shape, dtype="f8"))
                xtc_ccsd._compute_vvvv_block_df(
                    eris,
                    xtc_obj,
                    None,
                    self.l_vv_full,
                    self.nocc,
                    self.nvir,
                    self.nmo,
                    cc,
                )
                np.testing.assert_allclose(
                    eris.vvvv[:],
                    self.vvvv_full,
                    atol=1e-10,
                    rtol=1e-10,
                )

    def test_jax_contract_vvvv_t2_matches_dense_reference(self):
        xtc_obj = _FakeXTC(self.tc_full, self.nocc, return_jax=True)
        vvL = _pack_vvL(self.l_vv_full)
        eris = SimpleNamespace(vvvv=None, vvL=vvL)
        cc = SimpleNamespace(
            nocc=self.nocc,
            nmo=self.nmo,
            xtc_obj=xtc_obj,
            jastrow_params=None,
            with_df=object(),
            _scf=SimpleNamespace(with_df=None),
            gpu_max_memory=4096,
            max_memory=4096,
            vvvv_p_block_size=2,
            vvvv_r_block_size=2,
        )
        rng = np.random.default_rng(11)
        t2 = rng.normal(size=(self.nocc, self.nocc, self.nvir, self.nvir))
        out = np.zeros_like(t2)

        jax_xtc_ccsd._contract_vvvv_t2(cc, jnp.asarray(t2), eris, out)

        ref = np.einsum(
            "abcd,ijcd->ijab",
            self.vvvv_full.transpose(0, 2, 1, 3),
            t2,
        )
        np.testing.assert_allclose(out, ref, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
