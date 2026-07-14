"""Structural/synthetic tests for pytc.pbc.df.isdf.apply_raw_kernel_and_solve
(task #21, design v2.1 sections 5/7) -- Hermiticity, bare-kernel-only
structure, and malformed-input rejection, independent of the real
fftisdf comparison in test_v2_reference_replay.py.
"""

import unittest

import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import apply_raw_kernel_and_solve, build_pi_eta
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


class TestApplyRawKernelAndSolve(unittest.TestCase):
    def _setup(self, kmesh, seed=70, n_ip=3):
        cell = _make_cell()
        rng = np.random.default_rng(seed)
        kpts = cell.make_kpts(kmesh, wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grids = cell.get_uniform_grids(cell.mesh)
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))
        ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (grids.shape[0], cell.nao))
        Pi, eta = build_pi_eta(X, ao, mesh_obj.phase)
        return cell, mesh_obj, grids, Pi, eta

    def test_kern_q_and_w_q_are_hermitian(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        for q in range(mesh_obj.n_kpts):
            W_q, kern_q, info = apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8,
            )
            np.testing.assert_allclose(
                kern_q, kern_q.conj().T, atol=1e-8, err_msg=f"kern_q q={q}"
            )
            np.testing.assert_allclose(W_q, W_q.conj().T, atol=1e-8, err_msg=f"W_q q={q}")

    def test_retained_solve_residual_is_tiny(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        for q in range(mesh_obj.n_kpts):
            _, _, info = apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8,
            )
            self.assertLess(info["retained_solve_residual"], 1e-8, msg=f"q={q}")

    def test_no_exxdiv_anywhere_bare_kernel_only(self):
        # The function signature has no exxdiv parameter at all -- this
        # test documents/enforces that invariant structurally.
        import inspect
        sig = inspect.signature(apply_raw_kernel_and_solve)
        self.assertNotIn("exxdiv", sig.parameters)

    def test_self_paired_forces_kern_q_exactly_real(self):
        # design v2.1 section 5 (retention-policy fix follow-up): for a
        # self-paired q (neg[q]==q), physics requires kern_q real; its
        # own FFT-chain construction leaves measured floating-point
        # noise on the imaginary part that self_paired=True removes
        # before the solve.
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        q = 0  # Gamma is always self-paired
        self.assertEqual(int(mesh_obj.neg[q]), q)
        _, kern_q, _ = apply_raw_kernel_and_solve(
            Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
            grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8, self_paired=True,
        )
        np.testing.assert_array_equal(kern_q.imag, 0.0)

    def test_self_paired_false_is_the_default_and_leaves_kern_q_unmodified(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 3])
        q = 0
        _, kern_q_default, _ = apply_raw_kernel_and_solve(
            Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
            grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8,
        )
        _, kern_q_explicit_false, _ = apply_raw_kernel_and_solve(
            Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
            grid_coords=grids, grid_mesh=cell.mesh, rtol=1e-8, self_paired=False,
        )
        np.testing.assert_array_equal(kern_q_default, kern_q_explicit_false)

    def test_rejects_malformed_shapes(self):
        cell, mesh_obj, grids, Pi, eta = self._setup([1, 1, 2])
        q = 0
        with self.assertRaises(ValueError):
            apply_raw_kernel_and_solve(
                Pi[q][:, :2], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids, grid_mesh=cell.mesh,
            )
        with self.assertRaises(ValueError):
            apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids[:-1], grid_mesh=cell.mesh,
            )
        with self.assertRaises(ValueError):
            apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=np.array([0.0, 0.0]),
                grid_coords=grids, grid_mesh=cell.mesh,
            )
        with self.assertRaises(ValueError):
            apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grids, grid_mesh=(2, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()
