"""Structural/synthetic tests for pytc.pbc.df.isdf.apply_raw_kernel_and_solve
(task #21, design v2.1 sections 5/7) -- Hermiticity, bare-kernel-only
structure, and malformed-input rejection, independent of the real
fftisdf comparison in test_v2_reference_replay.py.
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from pyscf.pbc.gto import Cell

import gc
import weakref

from pytc.pbc.df.isdf import (
    RawKernelProvider,
    p_blocked_peak_bytes,
    apply_raw_kernel_and_solve,
    build_pi_eta,
    build_pi_kern_p_blocked,
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


class TestApplyRawKernelAndSolve(unittest.TestCase):
    def _setup(self, kmesh, seed=70, n_ip=3):
        cell = _make_cell()
        rng = np.random.default_rng(seed)
        kpts = cell.make_kpts(kmesh, wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grids = cell.get_uniform_grids(cell.mesh)
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))
        ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (grids.shape[0], cell.nao))
        Pi, eta = build_pi_eta(X, ao, mesh_obj.phase, mesh_obj.neg)
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


class TestPBlockedKern(unittest.TestCase):
    """kern built panel-by-panel over the interpolation-point axis, never
    materialising eta. The panel schedule and the Hermitian mirror are certified
    against the dense path rather than against the algebra."""

    def _fixture(self, n_ip=6, kmesh=(2, 1, 1), seed=7):
        cell = _make_cell()
        rng = np.random.default_rng(seed)
        mesh_obj = canonicalize_kpts(cell, cell.make_kpts(list(kmesh), wrap_around=False))
        grids = cell.get_uniform_grids(cell.mesh)
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))
        ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg,
                                   (grids.shape[0], cell.nao))
        provider = RawKernelProvider(cell=cell, canonical_kpts=mesh_obj.canonical_kpts,
                                     grid_mesh=cell.mesh)
        return cell, mesh_obj, grids, X, ao, provider

    @staticmethod
    def _dense_kern(mesh_obj, grids, eta, provider, n_ip):
        kern = np.zeros((mesh_obj.n_kpts, n_ip, n_ip), dtype=np.complex128)
        n_grid = grids.shape[0]
        for q in range(mesh_obj.n_kpts):
            gphase = np.exp(-1j * (grids @ np.asarray(mesh_obj.canonical_kpts)[q]))
            lq = np.asarray(eta[q]) * gphase[None, :]
            rq = np.conj(np.asarray(provider.apply(q, lq)))
            kern[q] = (lq @ rq.T) / np.sqrt(n_grid)
        return kern

    def test_every_panel_schedule_reproduces_the_dense_kern(self):
        n_ip = 6
        cell, mesh_obj, grids, X, ao, provider = self._fixture(n_ip=n_ip)
        _, eta = build_pi_eta(X, ao, mesh_obj.phase, mesh_obj.neg)
        want = self._dense_kern(mesh_obj, grids, eta, provider, n_ip)
        # One panel (no blocking) through to one row per panel (maximum blocking).
        for panel_rows in (6, 3, 2, 1):
            with self.subTest(panel_rows=panel_rows):
                _, got = build_pi_kern_p_blocked(
                    X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
                    panel_rows=panel_rows)
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_pi_is_unaffected_by_the_panel_schedule(self):
        cell, mesh_obj, grids, X, ao, provider = self._fixture()
        want, _ = build_pi_eta(X, ao, mesh_obj.phase, mesh_obj.neg)
        got, _ = build_pi_kern_p_blocked(
            X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
            panel_rows=2)
        np.testing.assert_array_equal(got, want)

    def test_rejects_a_malformed_knob(self):
        cell, mesh_obj, grids, X, ao, provider = self._fixture()
        # True is an int in Python and 1.9 truncates silently; both were accepted
        # before review.
        for bad in (0, -1, True, 1.9):
            with self.subTest(panel_rows=bad):
                with self.assertRaises(ValueError):
                    build_pi_kern_p_blocked(X, lambda: [ao], mesh_obj.phase,
                                            mesh_obj.neg, provider, grids,
                                            panel_rows=bad)


class TestPBlockedStorageContract(unittest.TestCase):
    """The storage claims, tested directly. Equivalence tests certify arithmetic
    and say nothing about memory -- which is how an earlier revision that made
    peak memory WORSE passed the whole suite."""

    def _fixture(self, n_ip=6, n_blocks=4):
        cell = _make_cell()
        rng = np.random.default_rng(11)
        mesh_obj = canonicalize_kpts(cell, cell.make_kpts([2, 1, 1], wrap_around=False))
        grids = cell.get_uniform_grids(cell.mesh)
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (n_ip, cell.nao))
        ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg,
                                   (grids.shape[0], cell.nao))
        # Split the AO array into several genuine blocks along the grid axis.
        edges = np.linspace(0, grids.shape[0], n_blocks + 1).astype(int)
        chunks = [np.ascontiguousarray(ao[:, a:b, :]) for a, b in zip(edges, edges[1:])]
        provider = RawKernelProvider(cell=cell, canonical_kpts=mesh_obj.canonical_kpts,
                                     grid_mesh=cell.mesh)
        return cell, mesh_obj, grids, X, chunks, provider

    def test_only_one_ao_block_is_live_at_a_time(self):
        cell, mesh_obj, grids, X, chunks, provider = self._fixture()
        seen = []

        def factory():
            def gen():
                for chunk in chunks:
                    block = np.array(chunk)      # a fresh object per yield
                    ref = weakref.ref(block)
                    # Everything handed over earlier must already be collectable.
                    gc.collect()
                    alive = [r for r in seen if r() is not None]
                    if alive:
                        raise AssertionError(
                            f"{len(alive)} earlier AO block(s) still live")
                    seen.append(ref)
                    yield block
                    del block
            return gen()

        n_ip, panel_rows = 6, 2
        build_pi_kern_p_blocked(X, factory, mesh_obj.phase, mesh_obj.neg,
                                provider, grids, panel_rows=panel_rows)
        # Every block of every panel build passed through, and none outlived its
        # successor's creation -- which the loop above asserts as it goes.
        n_panels = -(-n_ip // panel_rows)
        n_builds = n_panels * (n_panels + 1) // 2
        self.assertEqual(len(seen), n_builds * len(chunks))

    def test_factory_call_count_is_the_triangular_schedule(self):
        cell, mesh_obj, grids, X, chunks, provider = self._fixture()
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return iter(chunks)

        for panel_rows, n_panels in ((6, 1), (3, 2), (2, 3), (1, 6)):
            calls["n"] = 0
            build_pi_kern_p_blocked(X, factory, mesh_obj.phase, mesh_obj.neg,
                                    provider, grids, panel_rows=panel_rows)
            # P(P+1)/2 panel builds -- a superpanel schedule, NOT a resident cache.
            self.assertEqual(calls["n"], n_panels * (n_panels + 1) // 2,
                             f"panel_rows={panel_rows}")

    def test_byte_model_names_the_floor_no_knob_can_reach(self):
        # Synthetic 444/k222 shape: asserted, never allocated.
        big = dict(n_kpts=8, n_ip=37120, n_grid=438948, ao_block_cols=4096, n_ao=3712)
        wide = p_blocked_peak_bytes(panel_rows=37120, **big)
        narrow = p_blocked_peak_bytes(panel_rows=2000, **big)
        # The knob moves the eta term and only the eta term.
        self.assertLess(narrow["eta_panels"], wide["eta_panels"] / 15)
        for term in ("Pi", "kern", "grid_phases", "one_ao_block"):
            self.assertEqual(narrow[term], wide[term], term)
        # Pi + kern are irreducible here and already exceed a 300 GB target.
        floor = narrow["Pi"] + narrow["kern"]
        self.assertGreater(floor, 300e9)
        self.assertGreater(narrow["total"], floor)

    def test_streamed_eta_rejects_a_grid_width_mismatch(self):
        cell, mesh_obj, grids, X, chunks, provider = self._fixture()
        with self.assertRaises(ValueError):
            build_pi_kern_p_blocked(X, lambda: iter(chunks[:-1]), mesh_obj.phase,
                                    mesh_obj.neg, provider, grids, panel_rows=3)
