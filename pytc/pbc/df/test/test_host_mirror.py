"""build_coul_kpt_host must track build_coul_kpt_device.

The host mirror exists so an accuracy question can be settled without first
porting a solver to the device path. That is only sound while the two agree:
the moment the loops drift -- conjugate shortcut, self_paired at nq == q, per-q
ordering -- the mirror stops being a reference and becomes a second, unvalidated
implementation. These tests are what make the mirror usable as evidence.
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import (
    RawKernelProvider,
    apply_raw_kernel_and_solve,
    build_coul_kpt_device,
    build_coul_kpt_host,
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


def _tr_symmetric(rng, n_kpts, neg, shape):
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


def _setup(kmesh, seed=96, n_ip=3):
    cell = _make_cell()
    mesh = canonicalize_kpts(cell, cell.make_kpts(kmesh, wrap_around=False))
    grids = cell.get_uniform_grids(cell.mesh)
    rng = np.random.default_rng(seed)
    X = _tr_symmetric(rng, mesh.n_kpts, mesh.neg, (n_ip, cell.nao))
    ao = _tr_symmetric(rng, mesh.n_kpts, mesh.neg, (grids.shape[0], cell.nao))
    Pi, eta = build_pi_eta(X, ao, mesh.phase, mesh.neg)
    return cell, mesh, grids, Pi, eta


class TestHostMirrorMatchesDevice(unittest.TestCase):
    def test_matches_device_at_gamma_and_with_a_conjugate_pair(self):
        # (1,1,3) is the case that exercises the nq != q conjugate shortcut; Gamma
        # alone would leave that branch untested.
        for kmesh in ((1, 1, 1), (1, 1, 3)):
            with self.subTest(kmesh=kmesh):
                cell, mesh, grids, Pi, eta = _setup(kmesh)
                provider = RawKernelProvider(
                    cell=cell, canonical_kpts=mesh.canonical_kpts, grid_mesh=cell.mesh)
                coul_d, kern_d, _, n_calls_d = build_coul_kpt_device(
                    provider, Pi, eta, grids, mesh, rtol=1e-6)
                coul_h, kern_h, infos, n_calls_h = build_coul_kpt_host(
                    cell, Pi, eta, grids, mesh, rtol=1e-6)
                self.assertEqual(n_calls_h, n_calls_d)
                np.testing.assert_allclose(np.asarray(coul_d), coul_h, atol=1e-12)
                np.testing.assert_allclose(np.asarray(kern_d), kern_h, atol=1e-11)
                self.assertEqual(len(infos), mesh.n_kpts)
                self.assertTrue(all(i is not None for i in infos))

    def test_cholesky_jitter_runs_where_the_device_path_refuses(self):
        # The reason the mirror exists: this mode has no device implementation.
        cell, mesh, grids, Pi, eta = _setup((1, 1, 1))
        coul, _, infos, _ = build_coul_kpt_host(
            cell, Pi, eta, grids, mesh, retention_mode="cholesky_jitter")
        self.assertEqual(coul.shape[0], mesh.n_kpts)
        self.assertTrue(np.all(np.isfinite(coul)))
        for info in infos:
            self.assertIn(info["solver"], ("unscaled_cholesky_jitter", "tsvd"))

    def test_scalar_and_sequence_n_retained_pin_both_work(self):
        # A scalar pin is documented as valid and raised TypeError on the first
        # subscript, because the host loop indexed it without normalizing.
        cell, mesh, grids, Pi, eta = _setup((1, 1, 3))
        coul_scalar, _, _, _ = build_coul_kpt_host(
            cell, Pi, eta, grids, mesh, n_retained_pin=2)
        coul_seq, _, _, _ = build_coul_kpt_host(
            cell, Pi, eta, grids, mesh, n_retained_pin=[2] * mesh.n_kpts)
        np.testing.assert_allclose(coul_scalar, coul_seq, atol=1e-14)
        with self.assertRaises(ValueError):
            build_coul_kpt_host(cell, Pi, eta, grids, mesh, n_retained_pin=[2])

    def test_pipeline_calls_counts_the_conjugate_shortcut(self):
        # 1x1x3 has one q/-q pair, so the pipeline runs twice, not three times.
        # This was hard-coded to Nk, overstating the work at every paired mesh.
        cell, mesh, grids, Pi, eta = _setup((1, 1, 3))
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh.canonical_kpts, grid_mesh=cell.mesh)
        _, _, _, n_dev = build_coul_kpt_device(provider, Pi, eta, grids, mesh, rtol=1e-6)
        _, _, _, n_host = build_coul_kpt_host(cell, Pi, eta, grids, mesh, rtol=1e-6)
        self.assertEqual(n_host, n_dev)
        self.assertLess(n_host, mesh.n_kpts)

    def test_rtol_is_rejected_rather_than_ignored_in_cholesky_mode(self):
        # A caller sweeping rtol over a mode that ignores it would get identical
        # runs and read them as insensitivity to rtol.
        cell, mesh, grids, Pi, eta = _setup((1, 1, 1))
        with self.assertRaises(ValueError):
            build_coul_kpt_host(cell, Pi, eta, grids, mesh, rtol=1e-6,
                                retention_mode="cholesky_jitter")

    def test_jitter_rcond_rejected_on_truncating_modes(self):
        cell, mesh, grids, Pi, eta = _setup((1, 1, 1))
        with self.assertRaises(ValueError):
            apply_raw_kernel_and_solve(
                Pi[0], eta[0], cell=cell, q_kpt=mesh.canonical_kpts[0],
                grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-6,
                jitter_rcond=1e-14)


if __name__ == "__main__":
    unittest.main()
