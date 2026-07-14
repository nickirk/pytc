"""Tests for the staging-policy layer in pytc.pbc.df.isdf (design v2.1
section 6): predicted_byte_model, choose_staging_policy (including
forced-demotion cases), and the memmap/recompute mechanics."""

import tempfile
import unittest
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pytc.pbc.df.isdf import (
    build_pi_eta,
    choose_staging_policy,
    jit_memory_analysis_smoke,
    predicted_byte_model,
    stage_eta_memmap,
    stage_eta_recompute_tile,
)


def _tr_symmetric_fixture(rng, n_kpts, neg, shape):
    X = np.zeros((n_kpts,) + shape, dtype=np.complex128)
    done = set()
    for k in range(n_kpts):
        if k in done:
            continue
        nk = int(neg[k])
        if nk == k:
            X[k] = rng.normal(size=shape)
        else:
            re, im = rng.normal(size=shape), rng.normal(size=shape)
            X[k] = re + 1j * im
            X[nk] = re - 1j * im
            done.add(nk)
        done.add(k)
    return X


class TestPredictedByteModel(unittest.TestCase):
    def test_eta_store_bytes_matches_closed_form(self):
        model = predicted_byte_model(n_kpts=3, n_ip=5, n_grid=1000, n_ao=7, block_size=200)
        self.assertEqual(model["eta_store_bytes"], 3 * 5 * 1000 * 16)

    def test_selection_traffic_scales_with_rank_and_grid(self):
        small = predicted_byte_model(n_kpts=2, n_ip=4, n_grid=500, n_ao=6, block_size=100)
        larger_rank = predicted_byte_model(n_kpts=2, n_ip=8, n_grid=500, n_ao=6, block_size=100)
        self.assertEqual(larger_rank["selection_traffic_bytes"], 2 * small["selection_traffic_bytes"])

    def test_total_predicted_is_eta_plus_selection(self):
        model = predicted_byte_model(n_kpts=2, n_ip=4, n_grid=500, n_ao=6, block_size=100)
        self.assertEqual(
            model["total_predicted_bytes"],
            model["eta_store_bytes"] + model["selection_traffic_bytes"],
        )

    def test_rejects_nonpositive_shape_params(self):
        with self.assertRaises(ValueError):
            predicted_byte_model(n_kpts=0, n_ip=4, n_grid=500, n_ao=6, block_size=100)
        with self.assertRaises(ValueError):
            predicted_byte_model(n_kpts=2, n_ip=4, n_grid=500, n_ao=6, block_size=-1)


class TestChooseStagingPolicy(unittest.TestCase):
    def test_chooses_ram_when_eta_fits_headroom(self):
        model = predicted_byte_model(n_kpts=2, n_ip=4, n_grid=500, n_ao=6, block_size=100)
        decision = choose_staging_policy(
            model, available_host_bytes=10 * model["eta_store_bytes"],
            available_disk_bytes=10 * model["eta_store_bytes"],
        )
        self.assertEqual(decision["policy"], "ram")
        self.assertIsNone(decision["observed_peak_host_bytes"])
        self.assertEqual(decision["observed_status"], "unmeasured")

    def test_forced_demotion_ram_to_memmap(self):
        # eta exceeds RAM headroom but fits disk headroom -> memmap.
        model = predicted_byte_model(n_kpts=4, n_ip=50, n_grid=200_000, n_ao=20, block_size=1000)
        decision = choose_staging_policy(
            model,
            available_host_bytes=int(model["eta_store_bytes"] * 0.5),  # below 50% headroom
            available_disk_bytes=10 * model["eta_store_bytes"],
        )
        self.assertEqual(decision["policy"], "memmap")

    def test_forced_demotion_memmap_to_recompute(self):
        # eta exceeds BOTH ram and disk headroom -> recompute.
        model = predicted_byte_model(n_kpts=4, n_ip=50, n_grid=200_000, n_ao=20, block_size=1000)
        decision = choose_staging_policy(
            model,
            available_host_bytes=int(model["eta_store_bytes"] * 0.1),
            available_disk_bytes=int(model["eta_store_bytes"] * 0.1),
        )
        self.assertEqual(decision["policy"], "recompute")

    def test_headroom_fraction_is_overridable(self):
        model = predicted_byte_model(n_kpts=2, n_ip=4, n_grid=500, n_ao=6, block_size=100)
        eta_bytes = model["eta_store_bytes"]
        # Exactly at the default 50% boundary: fits with a looser
        # fraction, does not fit with the default.
        host_bytes = int(eta_bytes * 1.5)
        default_decision = choose_staging_policy(
            model, available_host_bytes=host_bytes, available_disk_bytes=host_bytes,
        )
        loose_decision = choose_staging_policy(
            model, available_host_bytes=host_bytes, available_disk_bytes=host_bytes,
            ram_headroom_fraction=0.9,
        )
        self.assertEqual(default_decision["policy"], "memmap")
        self.assertEqual(loose_decision["policy"], "ram")

    def test_rejects_bad_fractions_and_negative_resources(self):
        model = predicted_byte_model(n_kpts=2, n_ip=4, n_grid=500, n_ao=6, block_size=100)
        with self.assertRaises(ValueError):
            choose_staging_policy(model, available_host_bytes=-1, available_disk_bytes=100)
        with self.assertRaises(ValueError):
            choose_staging_policy(
                model, available_host_bytes=100, available_disk_bytes=100, ram_headroom_fraction=0.0
            )
        with self.assertRaises(ValueError):
            choose_staging_policy(
                model, available_host_bytes=100, available_disk_bytes=100, ram_headroom_fraction=1.5
            )


class TestStageEtaMemmap(unittest.TestCase):
    def test_streamed_chunks_reconstruct_full_array(self):
        rng = np.random.default_rng(100)
        n_kpts, n_ip, n_grid = 2, 3, 17
        full = (rng.normal(size=(n_kpts, n_ip, n_grid)) + 1j * rng.normal(size=(n_kpts, n_ip, n_grid))).astype(
            np.complex128
        )
        chunks = [(0, 6, full[:, :, 0:6]), (6, 13, full[:, :, 6:13]), (13, 17, full[:, :, 13:17])]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "eta.memmap")
            mm = stage_eta_memmap(iter(chunks), (n_kpts, n_ip, n_grid), path)
            np.testing.assert_allclose(np.asarray(mm), full, atol=0.0)

            reopened = np.memmap(path, dtype=np.complex128, mode="r", shape=(n_kpts, n_ip, n_grid))
            np.testing.assert_allclose(np.asarray(reopened), full, atol=0.0)

    def test_rejects_malformed_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bad.memmap")
            with self.assertRaises(ValueError):
                stage_eta_memmap(iter([]), (2, 3), path)


class TestStageEtaRecomputeTile(unittest.TestCase):
    def _cell_and_mesh(self):
        from pyscf.pbc.gto import Cell

        from pytc.pbc.df.kpts import canonicalize_kpts

        cell = Cell()
        cell.atom = "He 1.0 1.0 1.0"
        cell.a = np.diag([2.0, 2.0, 2.0])
        cell.unit = "A"
        cell.verbose = 0
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.build()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        return cell, mesh_obj

    def test_matches_direct_build_pi_eta(self):
        from pytc.pbc.df.isdf import stream_ao_blocks

        cell, mesh_obj = self._cell_and_mesh()
        grid_coords = cell.get_uniform_grids(cell.mesh)
        rng = np.random.default_rng(101)
        n_ip = 3
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))

        def ao_block_source():
            return (blk for _, _, blk in stream_ao_blocks(cell, mesh_obj.canonical_kpts, grid_coords, 13))

        Pi_recompute, eta_recompute = stage_eta_recompute_tile(X, ao_block_source, mesh_obj.kmesh)

        ao_full = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords, kpts=list(mesh_obj.canonical_kpts)),
            dtype=np.complex128,
        )
        Pi_direct, eta_direct = build_pi_eta(X, ao_full, mesh_obj.kmesh)

        np.testing.assert_allclose(Pi_recompute, Pi_direct, atol=1e-12)
        np.testing.assert_allclose(eta_recompute, eta_direct, atol=1e-12)

    def test_q_slice_applied_after_build(self):
        from pytc.pbc.df.isdf import stream_ao_blocks

        cell, mesh_obj = self._cell_and_mesh()
        grid_coords = cell.get_uniform_grids(cell.mesh)
        rng = np.random.default_rng(102)
        n_ip = 3
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))

        def ao_block_source():
            return (blk for _, _, blk in stream_ao_blocks(cell, mesh_obj.canonical_kpts, grid_coords, 13))

        Pi_full, eta_full = stage_eta_recompute_tile(X, ao_block_source, mesh_obj.kmesh)
        Pi_sliced, eta_sliced = stage_eta_recompute_tile(X, ao_block_source, mesh_obj.kmesh, q_slice=0)
        np.testing.assert_allclose(Pi_sliced, Pi_full[0], atol=0.0)
        np.testing.assert_allclose(eta_sliced, eta_full[0], atol=0.0)


class TestJitMemoryAnalysisSmoke(unittest.TestCase):
    def test_returns_labeled_stats_matching_the_actual_running_backend(self):
        # Deliberately backend-agnostic: this suite runs on whatever
        # jax.default_backend() reports on the machine executing it
        # (CPU dev box, or a real GPU node) -- hardcoding "cpu" here
        # would fail on a genuine GPU run for the RIGHT reason (the
        # label fix works), which is exactly what happened on a real
        # V100 validation run. Assert internal consistency against the
        # actual backend instead of assuming which one is present.
        @jax.jit
        def f(x):
            return x @ x

        result = jit_memory_analysis_smoke(f, jnp.ones((4, 4), dtype=jnp.complex128))
        backend = jax.default_backend()
        expected_label = (
            "cpu_backend_structural_smoke_not_hbm"
            if backend == "cpu"
            else "gpu_backend_compiled_memory_stats"
        )
        self.assertEqual(result["backend"], backend)
        self.assertEqual(result["label"], expected_label)
        self.assertIn("temp_size_in_bytes", result)
        self.assertIsInstance(result["temp_size_in_bytes"], int)

    def test_label_reflects_actual_backend_not_hardcoded(self):
        # A non-cpu backend must NOT be mislabeled "not_hbm" -- simulate
        # by monkeypatching jax.default_backend rather than requiring a
        # real GPU in this test environment.
        import unittest.mock as mock

        from pytc.pbc.df import isdf as isdf_module

        @jax.jit
        def f(x):
            return x @ x

        with mock.patch.object(isdf_module.jax, "default_backend", return_value="gpu"):
            result = jit_memory_analysis_smoke(f, jnp.ones((4, 4), dtype=jnp.complex128))
        self.assertEqual(result["backend"], "gpu")
        self.assertEqual(result["label"], "gpu_backend_compiled_memory_stats")


if __name__ == "__main__":
    unittest.main()
