"""Pipeline-level n_retained_pin tests: the fused per-q device path and
build_coul_kpt_device honor a fixed effective rank identically to the
NumPy oracle, and a pin set to each q's rtol-selected count reproduces
the rtol pipeline run exactly (the mechanism the pinned-comparison
regression relies on)."""

import os
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import (
    RawKernelProvider,
    apply_kernel_and_solve_device,
    apply_raw_kernel_and_solve,
    build_coul_kpt_deterministic_cpu,
    build_coul_kpt_device,
    build_pi_eta,
)
from pytc.pbc.df.kpts import canonicalize_kpts


def _make_cell():
    cell = Cell()
    cell.atom = "He 1.0 1.0 1.0"
    cell.a = np.diag([2.0, 2.0, 2.0])
    cell.unit = "A"
    cell.verbose = 0
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 40.0
    cell.build()
    return cell


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


def _setup(kmesh=(1, 1, 3), seed=96, n_ip=3):
    cell = _make_cell()
    rng = np.random.default_rng(seed)
    kpts = cell.make_kpts(kmesh, wrap_around=False)
    mesh_obj = canonicalize_kpts(cell, kpts)
    grids = cell.get_uniform_grids(cell.mesh)
    X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))
    ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (grids.shape[0], cell.nao))
    Pi, eta = build_pi_eta(X, ao, mesh_obj.phase, mesh_obj.neg)
    provider = RawKernelProvider(
        cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
    )
    return cell, mesh_obj, grids, Pi, eta, provider


class TestPinnedPipeline(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"),
        "deterministic CPU solve requires Linux CPU-affinity APIs",
    )
    def test_deterministic_cpu_worker_is_bitwise_repeatable_and_receipted(self):
        _, mesh_obj, grids, Pi, eta, provider = _setup()
        coul_device, kern, _, calls = build_coul_kpt_device(
            provider, Pi, eta, grids, mesh_obj, rtol=1e-8,
        )
        first = build_coul_kpt_deterministic_cpu(
            Pi, kern, grids, mesh_obj, rtol=1e-8,
        )
        second = build_coul_kpt_deterministic_cpu(
            Pi, kern, grids, mesh_obj, rtol=1e-8,
        )
        coul_first, kern_first, infos_first, calls_first = first
        coul_second, kern_second, infos_second, calls_second = second

        self.assertEqual(calls_first, calls)
        self.assertEqual(calls_second, calls)
        np.testing.assert_array_equal(np.asarray(coul_first), np.asarray(coul_second))
        np.testing.assert_array_equal(np.asarray(kern_first), np.asarray(kern_second))
        np.testing.assert_allclose(
            np.asarray(coul_first), np.asarray(coul_device), atol=1e-10, rtol=1e-10,
        )
        for info_first, info_second in zip(infos_first, infos_second):
            self.assertEqual(info_first["worker_affinity_count"], 1)
            self.assertEqual(info_second["worker_affinity_count"], 1)
            self.assertEqual(
                info_first["execution_backend"],
                "deterministic_cpu_subprocess",
            )
            self.assertEqual(info_first["n_retained"], info_second["n_retained"])

    def test_pinned_fused_path_matches_numpy_oracle_per_q(self):
        cell, mesh_obj, grids, Pi, eta, provider = _setup()
        for q in range(mesh_obj.n_kpts):
            K = 2
            W_np, kern_np, info_np = apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids, grid_mesh=cell.mesh, n_retained_pin=K,
            )
            W_dev, kern_dev, info_dev = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], grid_coords=grids, n_retained_pin=K,
            )
            np.testing.assert_allclose(
                np.asarray(kern_dev), kern_np, atol=1e-12, err_msg=f"kern_q q={q}"
            )
            np.testing.assert_allclose(
                np.asarray(W_dev), W_np, atol=1e-10, err_msg=f"W_q q={q}"
            )
            self.assertEqual(info_dev["n_retained"], info_np["n_retained"])
            self.assertEqual(info_dev["n_retained_pin"], K)
            self.assertIsNone(info_dev["rtol"])

    def test_pin_at_rtol_count_reproduces_rtol_run_exactly(self):
        # The mechanism the pinned-comparison regression relies on: pinning
        # each q to the count its rtol run selected must reproduce that run.
        _, mesh_obj, grids, Pi, eta, provider = _setup()
        for q in range(mesh_obj.n_kpts):
            W_rtol, _, info_rtol = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], grid_coords=grids, rtol=1e-8,
            )
            m = info_rtol["n_retained"]
            W_pin, _, info_pin = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], grid_coords=grids, n_retained_pin=m,
            )
            np.testing.assert_array_equal(np.asarray(W_pin), np.asarray(W_rtol))
            self.assertEqual(info_pin["n_retained"], m)

    def test_build_coul_kpt_device_per_q_pin_sequence(self):
        _, mesh_obj, grids, Pi, eta, provider = _setup()
        n_kpts = mesh_obj.n_kpts
        pins = [2, 3, 2][:n_kpts]
        _, _, infos, _ = build_coul_kpt_device(
            provider, Pi, eta, grids, mesh_obj, n_retained_pin=pins,
        )
        for q in range(n_kpts):
            nq = int(mesh_obj.neg[q])
            want = pins[q] if q <= nq else pins[nq]
            self.assertEqual(infos[q]["n_retained_pin"], want, f"q={q}")
            self.assertEqual(infos[q]["n_retained"], want, f"q={q}")

    def test_isdfdf_build_passthrough_records_pin_per_q(self):
        from pytc.pbc import coulomb
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        built = coulomb.build(cell, kpts, rank=4, block_size=11, n_retained_pin=2,
                              retention_mode="single")
        self.assertGreaterEqual(built["n_selected"], 2)
        for q, info in enumerate(built["solve_infos"]):
            self.assertEqual(info["n_retained_pin"], 2, f"q={q}")
            self.assertEqual(info["n_retained"], 2, f"q={q}")
            self.assertIsNone(info["rtol"], f"q={q}")

    def test_isdfdf_build_pin_plus_rtol_raises(self):
        from pytc.pbc import coulomb
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        with self.assertRaises(ValueError):
            coulomb.build(cell, kpts, rank=4, block_size=11, rtol=1e-8, n_retained_pin=2)

    def test_pin_misuse_raises_through_pipeline(self):
        _, mesh_obj, grids, Pi, eta, provider = _setup()
        n_ip = Pi.shape[1]
        with self.assertRaises(ValueError):
            apply_kernel_and_solve_device(
                provider, 0, Pi[0], eta[0], grid_coords=grids, rtol=1e-8, n_retained_pin=2,
            )
        with self.assertRaises(ValueError):
            apply_kernel_and_solve_device(
                provider, 0, Pi[0], eta[0], grid_coords=grids, n_retained_pin=n_ip + 1,
            )
        with self.assertRaises(ValueError):
            apply_kernel_and_solve_device(
                provider, 0, Pi[0], eta[0], grid_coords=grids, n_retained_pin=0,
            )
        with self.assertRaises(ValueError):
            build_coul_kpt_device(
                provider, Pi, eta, grids, mesh_obj, n_retained_pin=[2],
            )


if __name__ == "__main__":
    unittest.main()
