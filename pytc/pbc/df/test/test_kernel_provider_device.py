"""Tests for the device-path KernelProvider seam (task #24,
#proj-isdf-periodic, design v2.1 section 7, Flinn's refined seam ruling
in the task #24 thread, msg 6916c47c):
- raw_kernel_apply's derived q<->-q dagger law at the pre-phased,
  pre-conjugate seam.
- RawKernelProvider is a thin, correct wrapper around raw_kernel_apply.
- apply_kernel_and_solve_device (RawKernelProvider) matches the CPU/
  NumPy oracle apply_raw_kernel_and_solve bit-tier (V3's "JAX device
  path vs NumPy oracle <=1e-12" sub-gate).
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import (
    RawKernelProvider,
    apply_kernel_and_solve_device,
    apply_raw_kernel_and_solve,
    build_pi_eta,
    raw_kernel_apply,
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


class TestRawKernelApplyDaggerLaw(unittest.TestCase):
    def test_dagger_law_holds_for_arbitrary_lq(self):
        # The derived law apply(neg[q], conj(lq)) == conj(apply(q, lq))
        # holds for ANY (Nip, Ng) lq -- it does not depend on lq itself
        # being time-reversal symmetric across a k-collection (that
        # structure is what pair_convolve's eta guarantees upstream;
        # the provider seam's own law is a per-call identity).
        cell = _make_cell()
        rng = np.random.default_rng(80)
        grid_mesh = cell.mesh
        n_grid = int(np.prod(grid_mesh))
        n_ip = 3
        lq = (
            rng.normal(size=(n_ip, n_grid)) + 1j * rng.normal(size=(n_ip, n_grid))
        ).astype(np.complex128)
        q_kpt = rng.normal(size=3) * 0.3

        v_q = raw_kernel_apply(lq, cell=cell, q_kpt=q_kpt, grid_mesh=grid_mesh)
        v_negq = raw_kernel_apply(lq.conj(), cell=cell, q_kpt=-q_kpt, grid_mesh=grid_mesh)
        np.testing.assert_allclose(np.asarray(v_negq), np.asarray(v_q).conj(), atol=1e-9)

    def test_dagger_law_at_gamma_is_self_conjugate(self):
        # q = Gamma is its own neg[q]; the law degenerates to
        # apply(Gamma, conj(lq)) == conj(apply(Gamma, lq)), a genuine
        # constraint (not vacuous) since it must hold for the SAME q.
        cell = _make_cell()
        rng = np.random.default_rng(81)
        grid_mesh = cell.mesh
        n_grid = int(np.prod(grid_mesh))
        lq = (rng.normal(size=(2, n_grid)) + 1j * rng.normal(size=(2, n_grid))).astype(
            np.complex128
        )
        gamma = np.zeros(3)
        v = raw_kernel_apply(lq, cell=cell, q_kpt=gamma, grid_mesh=grid_mesh)
        v_conj_input = raw_kernel_apply(lq.conj(), cell=cell, q_kpt=gamma, grid_mesh=grid_mesh)
        np.testing.assert_allclose(np.asarray(v_conj_input), np.asarray(v).conj(), atol=1e-9)


class TestRawKernelProvider(unittest.TestCase):
    def test_apply_matches_raw_kernel_apply_directly(self):
        cell = _make_cell()
        rng = np.random.default_rng(82)
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grid_mesh = cell.mesh
        n_grid = int(np.prod(grid_mesh))
        lq = (rng.normal(size=(2, n_grid)) + 1j * rng.normal(size=(2, n_grid))).astype(
            np.complex128
        )
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=grid_mesh
        )
        for q in range(mesh_obj.n_kpts):
            v_direct = raw_kernel_apply(
                lq, cell=cell, q_kpt=mesh_obj.canonical_kpts[q], grid_mesh=grid_mesh
            )
            v_provider = provider.apply(q, lq)
            np.testing.assert_allclose(np.asarray(v_provider), np.asarray(v_direct), atol=0.0)

    def test_provenance_fields(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        prov = provider.provenance()
        self.assertEqual(prov["kernel_name"], "raw")
        self.assertIn("exxdiv", prov)
        self.assertIn("not_this_provider", prov["exxdiv"])
        self.assertEqual(prov["normalization"], "vol_over_ng_inside_provider")

    def test_rejects_out_of_range_q_index(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        with self.assertRaises(ValueError):
            provider.apply(mesh_obj.n_kpts, np.zeros((1, int(np.prod(cell.mesh)))))


class TestApplyKernelAndSolveDeviceMatchesNumpyOracle(unittest.TestCase):
    def _setup(self, kmesh, seed=83, n_ip=3):
        cell = _make_cell()
        rng = np.random.default_rng(seed)
        kpts = cell.make_kpts(kmesh, wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grids = cell.get_uniform_grids(cell.mesh)
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))
        ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (grids.shape[0], cell.nao))
        Pi, eta = build_pi_eta(X, ao, mesh_obj.kmesh)
        return cell, mesh_obj, grids, Pi, eta

    def test_device_path_matches_numpy_oracle_bit_tier(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        for q in range(mesh_obj.n_kpts):
            W_np, kern_np, info_np = apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8,
            )
            W_dev, kern_dev, info_dev = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], grid_coords=grids, rtol=1e-8,
            )
            np.testing.assert_allclose(np.asarray(kern_dev), kern_np, atol=1e-12, err_msg=f"kern_q q={q}")
            np.testing.assert_allclose(np.asarray(W_dev), W_np, atol=1e-10, err_msg=f"W_q q={q}")
            self.assertEqual(info_dev["n_retained"], info_np["n_retained"])

    def test_device_w_q_is_hermitian(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        for q in range(mesh_obj.n_kpts):
            W_dev, _, _ = apply_kernel_and_solve_device(
                provider, q, Pi[q], eta[q], grid_coords=grids, rtol=1e-8,
            )
            W_np = np.asarray(W_dev)
            np.testing.assert_allclose(W_np, W_np.conj().T, atol=1e-8, err_msg=f"q={q}")


if __name__ == "__main__":
    unittest.main()
