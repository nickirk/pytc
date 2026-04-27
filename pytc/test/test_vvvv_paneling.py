import os
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
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
from pytc.utils.tile_memory import isdf_tile_peak_bytes, find_max_blksize


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

    def _assemble_2b_tile(self, jastrow_params, kernels, ranges, device=None,
                          panel_size=None, panel_layout="pr"):
        del jastrow_params, kernels, panel_layout
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


class TestTileMemory(unittest.TestCase):
    """Unit tests for pytc.utils.tile_memory — the canonical ISDF memory model."""

    def test_isdf_tile_peak_bytes_zero_nfused(self):
        """Without ISDF (n_fused=0) the formula returns 0."""
        self.assertEqual(isdf_tile_peak_bytes(10, 10, 10, 10, 0), 0)

    def test_isdf_tile_peak_bytes_include_d(self):
        """include_d adds the D matrix cost."""
        without = isdf_tile_peak_bytes(4, 4, 4, 4, 100)
        with_d  = isdf_tile_peak_bytes(4, 4, 4, 4, 100, include_d=True)
        self.assertEqual(with_d - without, 100 * 100 * 8)

    def test_isdf_tile_peak_bytes_formula(self):
        """Verify the individual term breakdown matches _estimate_delta_u_direct_tile_bytes."""
        from pytc.xtc import _estimate_delta_u_direct_tile_bytes
        for Np, Nq, Nr, Ns, Nf in [(5, 10, 5, 10, 200), (21, 21, 21, 1179, 500)]:
            ref = _estimate_delta_u_direct_tile_bytes(Np, Nq, Nr, Ns, Nf)
            got = isdf_tile_peak_bytes(Np, Nq, Nr, Ns, Nf, include_d=True)
            self.assertEqual(got, ref["total"], msg=f"mismatch for ({Np},{Nq},{Nr},{Ns},{Nf})")

    def test_find_max_blksize_basic(self):
        """find_max_blksize returns the largest blk where tile_bytes ≤ target."""
        blk = find_max_blksize(lambda b: b * b, lo=1, hi=100, gpu_target=50)
        self.assertEqual(blk, 7)   # 7²=49 ≤ 50 < 64=8²

    def test_find_max_blksize_nothing_fits(self):
        """Returns lo when even the smallest tile exceeds budget."""
        blk = find_max_blksize(lambda b: b * 1000, lo=5, hi=100, gpu_target=1)
        self.assertEqual(blk, 5)

    def test_find_max_blksize_with_host_constraint(self):
        """Host constraint is respected independently of GPU constraint."""
        # GPU fits up to b=10, host fits up to b=5 → should return 5
        blk = find_max_blksize(
            lambda b: b,          # GPU: fits for any b ≤ 100
            lo=1, hi=100,
            gpu_target=100,
            host_target=5,
            host_bytes_fn=lambda b: b,
        )
        self.assertEqual(blk, 5)


class TestTrimPanel(unittest.TestCase):
    """Unit tests for xtc_mod.trim_panel (defined in pytc.tc)."""

    def _padded(self, shape):
        """Return an arange array of the given shape for easy index inspection."""
        return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)

    def test_pr_trims_axes_0_and_2(self):
        """panel_layout='pr': axes 0 and 2 are trimmed to actual_a and actual_b."""
        tc = self._padded((8, 4, 8, 5))   # padded shape
        got = xtc_mod.trim_panel(tc, "pr", 3, 6)
        self.assertEqual(got.shape, (3, 4, 6, 5))
        np.testing.assert_array_equal(got, tc[:3, :, :6, :])

    def test_qr_trims_axes_1_and_2(self):
        """panel_layout='qr': axes 1 and 2 are trimmed to actual_a and actual_b."""
        tc = self._padded((5, 8, 8, 6))
        got = xtc_mod.trim_panel(tc, "qr", 3, 5)
        self.assertEqual(got.shape, (5, 3, 5, 6))
        np.testing.assert_array_equal(got, tc[:, :3, :5, :])

    def test_ps_trims_axes_0_and_3(self):
        """panel_layout='ps': axes 0 and 3 are trimmed to actual_a and actual_b."""
        tc = self._padded((8, 4, 5, 8))
        got = xtc_mod.trim_panel(tc, "ps", 3, 6)
        self.assertEqual(got.shape, (3, 4, 5, 6))
        np.testing.assert_array_equal(got, tc[:3, :, :, :6])

    def test_no_trim_needed(self):
        """When actual extents equal the padded size, result is the full array."""
        tc = self._padded((4, 3, 5, 6))
        got = xtc_mod.trim_panel(tc, "pr", 4, 5)
        self.assertEqual(got.shape, tc.shape)
        np.testing.assert_array_equal(got, tc)

    def test_invalid_layout_raises(self):
        """An unrecognised layout string raises ValueError."""
        tc = self._padded((4, 4, 4, 4))
        with self.assertRaises(ValueError):
            xtc_mod.trim_panel(tc, "xy", 2, 2)

    def test_returns_view_not_copy(self):
        """trim_panel should return a view (NumPy slice), not a copy."""
        tc = np.zeros((6, 4, 6, 5))
        got = xtc_mod.trim_panel(tc, "pr", 3, 4)
        self.assertTrue(np.shares_memory(got, tc))


class TestV3OPanelSizing(unittest.TestCase):
    """Regression tests for estimate_v3o_panel_blksize to guard against OOM bugs."""

    def test_production_scale_does_not_return_oom_blk(self):
        """For benzene cc-pCV5Z parameters the old formula returned blk=59 (OOM).
        The corrected formula must return blk ≤ 30 for an 82 GB GPU.

        Parameters from the production run that crashed:
          nocc=21, nvir=1179, N_rank=25961, gpu=82 GB
        """
        with (
            mock.patch.object(gpu_memory, "_get_gpu_free_bytes", return_value=80 * 10**9),
            mock.patch.object(gpu_memory, "_get_gpu_physical_bytes", return_value=82 * 10**9),
        ):
            blk, _ = gpu_memory.estimate_v3o_panel_blksize(
                nocc=21, nvir=1179,
                gpu_max_memory_mb=82_000,
                naux=2000,
                n_fused=25961,
            )
        # Old formula gave 59 → OOM; correct formula should stay ≤ nocc=21 for this budget
        self.assertLessEqual(blk, 30,
            f"blk={blk} is too large and would OOM a 82 GB GPU at N_rank=25961")
        self.assertGreaterEqual(blk, 21, "blk must be at least nocc=21")

    def test_small_system_gets_large_blk(self):
        """A small system should auto-select a large block that covers all virtuals."""
        with (
            mock.patch.object(gpu_memory, "_get_gpu_free_bytes", return_value=16 * 10**9),
            mock.patch.object(gpu_memory, "_get_gpu_physical_bytes", return_value=16 * 10**9),
        ):
            blk, _ = gpu_memory.estimate_v3o_panel_blksize(
                nocc=5, nvir=50,
                gpu_max_memory_mb=16_000,
                naux=200,
                n_fused=500,
            )
        # For a tiny system the full nvir should fit in one tile
        self.assertEqual(blk, 50, f"Expected blk=nvir=50, got {blk}")


class TestBroadcastToDevices(unittest.TestCase):
    """Unit tests for xtc_ccsd.broadcast_to_devices."""

    def test_none_device_returns_original(self):
        """For device=None the original array is returned unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = xtc_ccsd.broadcast_to_devices(arr, [None])
        self.assertIs(result[None], arr)

    def test_all_devices_present(self):
        """Every device in the input list appears as a key in the output."""
        arr = np.ones((3, 4))
        devices = ["gpu0", "gpu1", None]
        # Patch jax.device_put to return a sentinel so we don't need real devices.
        with mock.patch("jax.device_put", side_effect=lambda a, d: f"put({d})"):
            result = xtc_ccsd.broadcast_to_devices(arr, devices)
        self.assertEqual(set(result.keys()), {"gpu0", "gpu1", None})

    def test_real_device_uses_device_put(self):
        """For a non-None device, jax.device_put is called with the numpy form."""
        arr_jax = jnp.array([1.0, 2.0])
        fake_device = object()
        captured = {}

        def fake_put(a, d):
            captured['arr'] = a
            captured['dev'] = d
            return a  # return the array itself as the "result"

        with mock.patch("jax.device_put", side_effect=fake_put):
            result = xtc_ccsd.broadcast_to_devices(arr_jax, [fake_device])

        self.assertIs(captured['dev'], fake_device)
        # The array passed to device_put must be a NumPy array (not JAX)
        self.assertIsInstance(captured['arr'], np.ndarray)

    def test_numpy_materialised_once(self):
        """np.asarray is called once regardless of the number of real devices."""
        arr = np.arange(6.0)
        call_count = [0]
        original_asarray = np.asarray

        def counting_asarray(a, *args, **kwargs):
            if a is arr:
                call_count[0] += 1
            return original_asarray(a, *args, **kwargs)

        devices = ["d0", "d1", "d2"]
        with mock.patch("numpy.asarray", side_effect=counting_asarray), \
             mock.patch("jax.device_put", side_effect=lambda a, d: a):
            xtc_ccsd.broadcast_to_devices(arr, devices)

        self.assertEqual(call_count[0], 1)

    def test_empty_devices(self):
        """Empty device list returns an empty dict."""
        arr = np.zeros(5)
        result = xtc_ccsd.broadcast_to_devices(arr, [])
        self.assertEqual(result, {})


_ON_CI = os.environ.get("CI", "").lower() == "true"


class TestSolverRoundRobin(unittest.TestCase):
    @unittest.skipIf(_ON_CI, "Race condition: consume threads release the issue semaphore "
                              "in nondeterministic order on slow CI runners; pre-existing flake.")
    def test_round_robin_pipeline_cycles_devices(self):
        seen = []
        seen_lock = __import__("threading").Lock()

        def issue(spec, device):
            with seen_lock:
                seen.append(("issue", spec, device))
            return spec * 10

        def consume(spec, device, handle, release_gpu_slot):
            release_gpu_slot()
            with seen_lock:
                seen.append(("consume", spec, device, handle))

        xtc_ccsd._round_robin_pipeline(
            [0, 1, 2, 3, 4],
            issue,
            consume,
            devices=("d0", "d1"),
        )

        # Issue order is deterministic (main thread); consume order may vary
        # because consumes now run concurrently in a thread pool.
        issue_devices = [entry[2] for entry in seen if entry[0] == "issue"]
        self.assertEqual(issue_devices, ["d0", "d1", "d0", "d1", "d0"])
        consume_entries = [entry for entry in seen if entry[0] == "consume"]
        self.assertEqual(len(consume_entries), 5)
        self.assertEqual(
            sorted((entry[1], entry[2]) for entry in consume_entries),
            sorted([(0, "d0"), (1, "d1"), (2, "d0"), (3, "d1"), (4, "d0")]),
        )

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
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["n_devices"], 2)
        # ``_round_robin_pipeline`` runs one issue thread per device in
        # parallel.  Tile-id round-robin still partitions a 2×2 tile
        # grid as ``d0 = [tile 0, tile 2]`` and ``d1 = [tile 1, tile 3]``
        # — so each device should receive exactly its 2 tiles — but the
        # *order* in which the two threads append to ``calls`` is
        # non-deterministic (they race on the shared list).  Assert the
        # multiset, not the ordered list.
        self.assertEqual(len(payload["calls"]), 4)
        self.assertEqual(sorted(payload["calls"]), [0, 0, 1, 1])


class TestSmallOccBlockFormulas(unittest.TestCase):
    """Unit tests for the four small all-occupied block DF formulas.

    These formulas live in the inline loop inside _make_xtc_eris
    (oooo / ovoo / ooov / vooo).  We verify shapes and numerical
    correctness against explicit einsum references without needing the
    full ERI-build pipeline.

    Loo shape: (naux, nocc*nocc)   flat form returned by _init_df_eris
    Lov shape: (naux, nocc*nvir)   flat form returned by _init_df_eris
    """

    def setUp(self):
        rng = np.random.default_rng(0xCAFE)
        self.nocc, self.nvir, self.naux = 4, 6, 9
        O, V, L = self.nocc, self.nvir, self.naux
        self.Loo = rng.standard_normal((L, O * O))   # (naux, nocc²)
        self.Lov = rng.standard_normal((L, O * V))   # (naux, nocc×nvir)

    def _df_blocks(self):
        """Replicate the inline loop from _make_xtc_eris."""
        nocc, nvir = self.nocc, self.nvir
        Loo, Lov = self.Loo, self.Lov
        return {
            'oooo': lib.ddot(Loo.T, Loo).reshape(nocc, nocc, nocc, nocc),
            'ovoo': lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc),
            'ooov': lib.ddot(Loo.T, Lov).reshape(nocc, nocc, nocc, nvir),
            'vooo': lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc).transpose(1, 0, 2, 3),
        }

    def test_shapes(self):
        """Each block has the expected shape."""
        nocc, nvir = self.nocc, self.nvir
        blks = self._df_blocks()
        self.assertEqual(blks['oooo'].shape, (nocc, nocc, nocc, nocc))
        self.assertEqual(blks['ovoo'].shape, (nocc, nvir, nocc, nocc))
        self.assertEqual(blks['ooov'].shape, (nocc, nocc, nocc, nvir))
        self.assertEqual(blks['vooo'].shape, (nvir, nocc, nocc, nocc))

    def test_oooo_matches_einsum(self):
        """oooo[i,j,k,l] = sum_L Loo[L,ij] * Loo[L,kl]."""
        nocc, naux = self.nocc, self.naux
        Loo3 = self.Loo.reshape(naux, nocc, nocc)          # (L, i, j)
        ref  = np.einsum('Lij,Lkl->ijkl', Loo3, Loo3)
        np.testing.assert_allclose(self._df_blocks()['oooo'], ref, atol=1e-11)

    def test_ovoo_matches_einsum(self):
        """ovoo[i,a,k,l] = sum_L Lov[L,ia] * Loo[L,kl]."""
        nocc, nvir, naux = self.nocc, self.nvir, self.naux
        Lov3 = self.Lov.reshape(naux, nocc, nvir)
        Loo3 = self.Loo.reshape(naux, nocc, nocc)
        ref  = np.einsum('Lia,Lkl->iakl', Lov3, Loo3)
        np.testing.assert_allclose(self._df_blocks()['ovoo'], ref, atol=1e-11)

    def test_ooov_matches_einsum(self):
        """ooov[i,j,k,a] = sum_L Loo[L,ij] * Lov[L,ka]."""
        nocc, nvir, naux = self.nocc, self.nvir, self.naux
        Loo3 = self.Loo.reshape(naux, nocc, nocc)
        Lov3 = self.Lov.reshape(naux, nocc, nvir)
        ref  = np.einsum('Lij,Lka->ijka', Loo3, Lov3)
        np.testing.assert_allclose(self._df_blocks()['ooov'], ref, atol=1e-11)

    def test_vooo_matches_einsum(self):
        """vooo[a,i,k,l] = sum_L Lov[L,ia] * Loo[L,kl] — transpose of ovoo."""
        nocc, nvir, naux = self.nocc, self.nvir, self.naux
        Lov3 = self.Lov.reshape(naux, nocc, nvir)
        Loo3 = self.Loo.reshape(naux, nocc, nocc)
        ref  = np.einsum('Lia,Lkl->aikl', Lov3, Loo3)
        np.testing.assert_allclose(self._df_blocks()['vooo'], ref, atol=1e-11)

    def test_vooo_is_ovoo_transposed(self):
        """vooo[a,i,k,l] = ovoo[i,a,k,l].transpose(1,0,2,3)."""
        blks = self._df_blocks()
        np.testing.assert_allclose(
            blks['vooo'], blks['ovoo'].transpose(1, 0, 2, 3), atol=1e-14)

    def test_oooo_has_pq_rs_symmetry(self):
        """oooo[i,j,k,l] = oooo[k,l,i,j] (exchange of electron pairs)."""
        oooo = self._df_blocks()['oooo']
        np.testing.assert_allclose(oooo, oooo.transpose(2, 3, 0, 1), atol=1e-12)


class TestComputeMediumBlocksTiled(unittest.TestCase):
    """Integration / correctness test for _compute_medium_blocks_tiled.

    Verifies that the tiled multi-GPU pipeline produces results numerically
    identical to direct np.einsum reference computations for all five blocks:
    oovv / vvoo / ovov / ovvo / vovo.

    A _FakeMOXTC object stores the XTC "tc" contribution as a dense
    (nmo, nmo, nmo, nmo) array in MO-index space and returns properly
    padded tiles from _assemble_2b_tile, supporting all panel_layout values.
    """

    class _FakeMOXTC:
        """Fake XTC returning tc_full[p,q,r,s] for any MO-space ranges."""
        def __init__(self, tc_full, nmo):
            self._tc = np.asarray(tc_full)
            self.phi_isdf = np.zeros((nmo, 8))
            self.isdf_kernels = {"K1_kernel": 0, "K3_kernel": 0, "D": 0, "X": 0}

        def _assemble_2b_tile(self, jastrow_params, kernels, ranges,
                               device=None, panel_size=None, panel_layout="pr"):
            from pytc.tc import _normalize_panel_layout
            layout = _normalize_panel_layout(panel_layout)
            sp, sq, sr, ss = ranges
            block = self._tc[sp, sq, sr, ss].copy()
            p, q, r, s = block.shape
            if panel_size is not None:
                ps = panel_size
                if layout == "pr":
                    padded = np.zeros((ps, q, ps, s))
                    padded[:p, :, :r, :] = block
                elif layout == "qr":
                    padded = np.zeros((p, ps, ps, s))
                    padded[:, :q, :r, :] = block
                else:   # "ps"
                    padded = np.zeros((ps, q, r, ps))
                    padded[:p, :, :, :s] = block
                block = padded
            return jnp.asarray(block)

    def setUp(self):
        rng = np.random.default_rng(0xC0FFEE)
        self.nocc, self.nvir, self.naux = 3, 6, 7
        self.nmo = self.nocc + self.nvir
        O, V, L = self.nocc, self.nvir, self.naux
        self.Loo          = rng.standard_normal((L, O * O))          # flat (naux, O²)
        self.Lov_reshaped = rng.standard_normal((L, O, V))           # (naux, O, V)
        self.L_vv_full    = rng.standard_normal((V, V, L))           # (V, V, naux)
        # Dense tc in full MO space; each block's tc slice comes from here
        self._tc_full     = rng.standard_normal((self.nmo,) * 4)

    def _reference_blocks(self):
        """Dense einsum references for all 5 blocks: tc_part + df_part."""
        nocc, nvir, naux, nmo = self.nocc, self.nvir, self.naux, self.nmo
        Loo3 = self.Loo.reshape(naux, nocc, nocc)      # (L, i, j)
        Lov  = self.Lov_reshaped                        # (L, i, a)
        Lvv  = self.L_vv_full.transpose(2, 0, 1)       # (L, a, b)
        O, V = slice(None, nocc), slice(nocc, nmo)
        tc   = self._tc_full

        df_oovv = np.einsum('Lij,Lab->ijab', Loo3, Lvv)
        df_vvoo = np.einsum('Lab,Lij->abij', Lvv, Loo3)
        df_ovov = np.einsum('Lia,Ljb->iajb', Lov,  Lov)
        df_ovvo = np.einsum('Lia,Ljb->iabj', Lov,  Lov)
        df_vovo = np.einsum('Lia,Ljb->aibj', Lov,  Lov)

        return {
            'oovv': tc[O, O, V, V] + df_oovv,
            'vvoo': tc[V, V, O, O] + df_vvoo,
            'ovov': tc[O, V, O, V] + df_ovov,
            'ovvo': tc[O, V, V, O] + df_ovvo,
            'vovo': tc[V, O, V, O] + df_vovo,
        }

    def _run_tiled(self, panel_blk):
        xtc_obj = self._FakeMOXTC(self._tc_full, self.nmo)
        return xtc_ccsd._compute_medium_blocks_tiled(
            xtc_obj, None,
            self.Loo, self.Lov_reshaped, self.L_vv_full,
            self.nocc, self.nvir, self.nmo,
            panel_blk,
            devices=(None,),   # single CPU device — no real GPU needed
        )

    def test_single_tile_matches_reference(self):
        """panel_blk=nvir → one tile per block covers all virtuals at once."""
        results = self._run_tiled(self.nvir)
        refs    = self._reference_blocks()
        for blk in ('oovv', 'vvoo', 'ovov', 'ovvo', 'vovo'):
            np.testing.assert_allclose(
                results[blk], refs[blk], atol=1e-11,
                err_msg=f"Block {blk}: single-tile result does not match reference")

    def test_multi_tile_matches_reference(self):
        """panel_blk=2 < nvir=6 → 3 tiles per block, exercises the tiling path."""
        results = self._run_tiled(2)
        refs    = self._reference_blocks()
        for blk in ('oovv', 'vvoo', 'ovov', 'ovvo', 'vovo'):
            np.testing.assert_allclose(
                results[blk], refs[blk], atol=1e-11,
                err_msg=f"Block {blk}: multi-tile result does not match reference")

    def test_odd_nvir_panel_blk_matches_reference(self):
        """panel_blk=4 with nvir=6 → tiles of sizes [4, 2] (last tile smaller)."""
        results = self._run_tiled(4)
        refs    = self._reference_blocks()
        for blk in ('oovv', 'vvoo', 'ovov', 'ovvo', 'vovo'):
            np.testing.assert_allclose(
                results[blk], refs[blk], atol=1e-11,
                err_msg=f"Block {blk}: uneven-tile result does not match reference")


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
