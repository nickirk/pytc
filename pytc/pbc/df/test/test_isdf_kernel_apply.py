"""Structural/synthetic tests for pytc.pbc.df.isdf.apply_raw_kernel_and_solve
(task #21, design v2.1 sections 5/7) -- Hermiticity, bare-kernel-only
structure, and malformed-input rejection, independent of the real
fftisdf comparison in test_v2_reference_replay.py.
"""

import unittest
import unittest.mock

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from pyscf.pbc.gto import Cell

import gc
import weakref

from pytc.pbc.df.isdf import (
    RawKernelProvider,
    build_coul_kpt_device,
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

    def test_pair_progress_count_matches_the_documented_formula(self):
        """The COST note promises P(P+1)/2 units. Derive P with ceil() from the
        schedule rather than hand-entering it -- a hand-entered count cannot catch a
        panel-enumeration error, it only restates one. Includes non-divisor
        schedules so the short final panel is exercised."""
        n_ip = 6
        cell, mesh_obj, grids, X, ao, provider = self._fixture(n_ip=n_ip)
        for panel_rows in (6, 5, 4, 3, 2, 1):
            with self.subTest(panel_rows=panel_rows):
                n_panels = -(-n_ip // panel_rows)          # ceil, from the schedule
                want = n_panels * (n_panels + 1) // 2
                seen = []
                with self.assertLogs("pytc.pbc.df.isdf", level="INFO") as cm:
                    build_pi_kern_p_blocked(
                        X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider,
                        grids, panel_rows=panel_rows,
                        on_panel=lambda d, t, e: seen.append((d, t, e)))
                lines = [m for m in cm.output if "p_blocked: pair" in m]
                self.assertEqual(len(lines), want)
                self.assertEqual([d for d, _, _ in seen], list(range(1, want + 1)))
                self.assertTrue(all(t == want for _, t, _ in seen))
                el = [e for _, _, e in seen]
                self.assertEqual(el, sorted(el))

    def test_progress_is_counted_after_the_pairs_kernel_work(self):
        """The defect this replaces: the count advanced at the eta build, BEFORE the
        pair's kernel work, so every reported elapsed omitted a kernel term -- the
        whole of it at the first line. Drive a deterministic clock from the
        provider's own apply calls and require the final elapsed to cover ALL
        modelled work. Under the old placement the last line was short by the final
        apply, so this test fails against it."""
        cell, mesh_obj, grids, X, ao, provider = self._fixture(n_ip=6)

        applies = {"n": 0}
        real_apply = provider.apply

        class CountingProvider:
            canonical_kpts = provider.canonical_kpts
            is_self_adjoint_per_q = getattr(
                provider, "is_self_adjoint_per_q", False)

            def apply(self, q, lq):
                applies["n"] += 1
                return real_apply(q, lq)

        # Clock advances only with modelled work, so elapsed is exact, not timed.
        def fake_clock():
            return float(applies["n"])

        seen = []
        with unittest.mock.patch("pytc.pbc.df.isdf.time.perf_counter", fake_clock):
            build_pi_kern_p_blocked(
                X, lambda: [ao], mesh_obj.phase, mesh_obj.neg,
                CountingProvider(), grids, panel_rows=2,
                on_panel=lambda d, t, e: seen.append((d, t, e)))

        total_applies = float(applies["n"])
        self.assertGreater(total_applies, 0)
        # The last event must account for every apply. Counting at the eta build
        # leaves the final pair's applies outside the window.
        self.assertEqual(seen[-1][0], seen[-1][1])
        self.assertEqual(seen[-1][2], total_applies)
        # And no event may report more work than had happened by then.
        self.assertTrue(all(e <= total_applies for _, _, e in seen))

    def test_self_adjoint_path_reuses_each_right_panel_across_left_panels(self):
        cell, mesh_obj, grids, X, ao, provider = self._fixture(n_ip=6)
        applies = {"n": 0}
        real_apply = provider.apply

        class CountingProvider:
            canonical_kpts = provider.canonical_kpts
            is_self_adjoint_per_q = True

            def apply(self, q, lq):
                applies["n"] += 1
                return real_apply(q, lq)

        panel_rows = 2
        n_panels = -(-X.shape[1] // panel_rows)
        with self.assertLogs("pytc.pbc.df.isdf", level="INFO") as cm:
            build_pi_kern_p_blocked(
                X,
                lambda: [ao],
                mesh_obj.phase,
                mesh_obj.neg,
                CountingProvider(),
                grids,
                panel_rows=panel_rows,
            )

        self.assertEqual(applies["n"], mesh_obj.n_kpts * n_panels)
        reuse_lines = [m for m in cm.output if "right-factor reuse" in m]
        self.assertEqual(len(reuse_lines), 1)
        self.assertIn(f"misses={n_panels}", reuse_lines[0])
        self.assertIn(
            f"hits={n_panels * (n_panels - 1) // 2}", reuse_lines[0]
        )

    def test_progress_needs_no_callback(self):
        """Positive control: with on_panel omitted the log must STILL fire. If only
        the hook-based assertions were checked, default-on was never covered."""
        cell, mesh_obj, grids, X, ao, provider = self._fixture(n_ip=4)
        with self.assertLogs("pytc.pbc.df.isdf", level="INFO") as cm:
            build_pi_kern_p_blocked(
                X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
                panel_rows=2)
        self.assertEqual(
            len([m for m in cm.output if "p_blocked: pair" in m]), 3)

    def test_progress_reports_no_projected_total(self):
        """A projection was withdrawn as unsound without schedule-aware weights.
        Assert it has not crept back in: an unweighted extrapolation is wrong in the
        optimistic direction, which is the one that licenses a bad kill decision."""
        cell, mesh_obj, grids, X, ao, provider = self._fixture(n_ip=4)
        with self.assertLogs("pytc.pbc.df.isdf", level="INFO") as cm:
            build_pi_kern_p_blocked(
                X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
                panel_rows=2)
        joined = " ".join(cm.output)
        for banned in ("projected", "mean", "eta_s/", "/build"):
            self.assertNotIn(banned, joined)

    def test_progress_does_not_change_kern(self):
        """Instrumentation must be numerically inert: with and without a hook the
        result must agree BIT-for-bit, not to a tolerance."""
        cell, mesh_obj, grids, X, ao, provider = self._fixture(n_ip=6)
        _, bare = build_pi_kern_p_blocked(
            X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
            panel_rows=2)
        _, hooked = build_pi_kern_p_blocked(
            X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
            panel_rows=2, on_panel=lambda *a: None)
        self.assertTrue(np.array_equal(bare, hooked))

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


class TestPBlockedProviderAndSelfPaired(unittest.TestCase):
    """The two contract questions the first review left open: whether the
    Hermitian mirror is safe for a given provider, and what gets projected at
    self-paired q."""

    def _setup(self, kmesh, n_ip=4, seed=23):
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
    def _dense(mesh_obj, grids, eta, provider, n_ip, self_paired=None, Pi=None):
        kern = np.zeros((mesh_obj.n_kpts, n_ip, n_ip), dtype=np.complex128)
        for q in range(mesh_obj.n_kpts):
            gphase = np.exp(-1j * (grids @ np.asarray(mesh_obj.canonical_kpts)[q]))
            lq = np.asarray(eta[q]) * gphase[None, :]
            kern[q] = (lq @ np.conj(np.asarray(provider.apply(q, lq))).T) / np.sqrt(grids.shape[0])
            if self_paired is not None and self_paired(q):
                kern[q] = kern[q].real.astype(np.complex128)
        return kern

    def test_gamma_and_odd_mesh_reproduce_the_dense_kern(self):
        # Gamma has one self-paired q; (3,1,1) carries a genuine q/-q pair.
        for kmesh in ((1, 1, 1), (3, 1, 1)):
            with self.subTest(kmesh=kmesh):
                cell, mesh_obj, grids, X, ao, provider = self._setup(kmesh)
                _, eta = build_pi_eta(X, ao, mesh_obj.phase, mesh_obj.neg)
                want = self._dense(mesh_obj, grids, eta, provider, X.shape[1])
                _, got = build_pi_kern_p_blocked(
                    X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
                    panel_rows=2)
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_self_paired_projects_pi_as_well_as_kern(self):
        cell, mesh_obj, grids, X, ao, provider = self._setup((1, 1, 1))
        neg = np.asarray(mesh_obj.neg)
        sp = lambda q: int(neg[q]) == q
        Pi, kern = build_pi_kern_p_blocked(
            X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
            panel_rows=2, self_paired=sp)
        for q in range(mesh_obj.n_kpts):
            if sp(q):
                # The dense solve makes BOTH real; projecting only kern would
                # hand the solve a complex Pi it would have projected.
                self.assertEqual(np.max(np.abs(kern[q].imag)), 0.0, f"kern q={q}")
                self.assertEqual(np.max(np.abs(Pi[q].imag)), 0.0, f"Pi q={q}")

    def test_mirror_is_skipped_when_the_provider_makes_no_such_claim(self):
        cell, mesh_obj, grids, X, ao, provider = self._setup((2, 1, 1), n_ip=6)

        class Opaque:
            """Same maths, no self-adjointness claim."""
            def __init__(self, inner):
                self._inner = inner
                self.canonical_kpts = inner.canonical_kpts
            def apply(self, q, lq):
                return self._inner.apply(q, lq)

        opaque = Opaque(provider)
        self.assertFalse(getattr(opaque, "is_self_adjoint_per_q", False))
        _, mirrored = build_pi_kern_p_blocked(
            X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids, panel_rows=2)
        _, computed = build_pi_kern_p_blocked(
            X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, opaque, grids, panel_rows=2)
        # For this provider both paths must agree -- the point is that the
        # opaque one earned its result rather than assuming symmetry.
        np.testing.assert_allclose(computed, mirrored, rtol=0, atol=1e-12)


class TestPBlockedThroughTheSolve(unittest.TestCase):
    """The panel-blocked kern routed through the real solve, not a reimplemented
    one. A second solve path is where the residual gate, retention mode and pin
    would quietly diverge, so the precomputed kern goes through the same call."""

    def test_coul_kpt_matches_the_eta_path_end_to_end(self):
        cell = _make_cell()
        rng = np.random.default_rng(31)
        mesh_obj = canonicalize_kpts(cell, cell.make_kpts([2, 1, 1], wrap_around=False))
        grids = cell.get_uniform_grids(cell.mesh)
        X = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg, (4, cell.nao))
        ao = _tr_symmetric_fixture(rng, mesh_obj.n_kpts, mesh_obj.neg,
                                   (grids.shape[0], cell.nao))
        provider = RawKernelProvider(cell=cell, canonical_kpts=mesh_obj.canonical_kpts,
                                     grid_mesh=cell.mesh)
        Pi, eta = build_pi_eta(X, ao, mesh_obj.phase, mesh_obj.neg)
        W_eta, kern_eta, _, _ = build_coul_kpt_device(
            provider, Pi, eta, grids, mesh_obj, rtol=1e-6)

        neg = np.asarray(mesh_obj.neg)
        Pi_b, kern_b = build_pi_kern_p_blocked(
            X, lambda: [ao], mesh_obj.phase, mesh_obj.neg, provider, grids,
            panel_rows=2, self_paired=lambda q: int(neg[q]) == q)
        # eta is deliberately absent on this path.
        W_pb, kern_pb, _, _ = build_coul_kpt_device(
            provider, Pi_b, None, grids, mesh_obj, rtol=1e-6, kern=kern_b)

        np.testing.assert_allclose(np.asarray(W_pb), np.asarray(W_eta),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(kern_pb), np.asarray(kern_eta),
                                   rtol=0, atol=1e-12)

    def test_kern_shape_is_validated_when_eta_is_absent(self):
        cell = _make_cell()
        mesh_obj = canonicalize_kpts(cell, cell.make_kpts([2, 1, 1], wrap_around=False))
        grids = cell.get_uniform_grids(cell.mesh)
        provider = RawKernelProvider(cell=cell, canonical_kpts=mesh_obj.canonical_kpts,
                                     grid_mesh=cell.mesh)
        Pi = np.zeros((mesh_obj.n_kpts, 3, 3), dtype=np.complex128)
        bad = np.zeros((mesh_obj.n_kpts + 1, 3, 3), dtype=np.complex128)
        with self.assertRaises(ValueError):
            build_coul_kpt_device(provider, Pi, None, grids, mesh_obj, kern=bad)


class TestFusedRightFactor(unittest.TestCase):
    """Gate for the fused right factor (task #81 step 2b).

    Reference is the GENERIC composition through ``provider.apply`` -- the same
    path ``_right_factor`` takes when ``apply_right_factor`` is absent. The bound
    was frozen before any fused value was produced. Bitwise identity is NOT
    required: XLA may reassociate inside the fused region, which is exactly why
    this is a numerical bound rather than an equality.
    """

    BOUND = 1e-13

    @staticmethod
    def _generic_right_factor(provider, q, eta, gphase):
        lq = eta * gphase[None, :]
        rq = np.conj(np.asarray(provider.apply(q, lq)))
        rq *= gphase[None, :]
        return rq

    def _worst_rel(self, kpts, n_ip, seed):
        cell = _make_cell()
        mesh = tuple(int(m) for m in cell.mesh)
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=kpts, grid_mesh=mesh
        )
        n_grid = int(np.prod(mesh))
        rng = np.random.default_rng(seed)
        eta = (rng.standard_normal((n_ip, n_grid))
               + 1j * rng.standard_normal((n_ip, n_grid))).astype(np.complex128)
        worst = 0.0
        for q in range(len(kpts)):
            gphase = np.exp(-1j * rng.standard_normal(n_grid)).astype(np.complex128)
            ref = self._generic_right_factor(provider, q, eta, gphase)
            fused = np.asarray(provider.apply_right_factor(q, eta, gphase))
            self.assertEqual(fused.shape, ref.shape)
            self.assertEqual(fused.dtype, np.complex128)
            denom = np.abs(ref).max()
            worst = max(worst, np.abs(fused - ref).max() / (denom if denom > 0 else 1.0))
        return worst

    def test_fused_matches_generic_at_gamma(self):
        self.assertLessEqual(self._worst_rel(np.zeros((1, 3)), 4, 11), self.BOUND)

    def test_fused_matches_generic_at_finite_q_including_q_and_negq(self):
        kpts = np.array([[0.0, 0.0, 0.0],
                         [0.11, -0.07, 0.23],
                         [-0.11, 0.07, -0.23]])
        self.assertLessEqual(self._worst_rel(kpts, 4, 13), self.BOUND)

    def test_right_factor_dispatches_to_the_fused_path_and_agrees(self):
        """The hook must actually fire AND agree with the fallback it replaces.

        A provider exposing only ``apply`` drives the generic branch; the real
        provider drives the fused one. Comparing them here is what makes the
        dispatch observable rather than inferred from a log line.
        """
        from pytc.pbc.df import isdf as _isdf

        cell = _make_cell()
        mesh = tuple(int(m) for m in cell.mesh)
        kpts = np.array([[0.0, 0.0, 0.0], [0.11, -0.07, 0.23]])
        provider = RawKernelProvider(cell=cell, canonical_kpts=kpts, grid_mesh=mesh)
        self.assertTrue(hasattr(provider, "apply_right_factor"))

        class _ApplyOnly:
            def __init__(self, inner):
                self._inner = inner

            def apply(self, q_index, lq):
                return self._inner.apply(q_index, lq)

        self.assertIsNone(getattr(_ApplyOnly(provider), "apply_right_factor", None))

        n_grid = int(np.prod(mesh))
        rng = np.random.default_rng(17)
        eta = (rng.standard_normal((4, n_grid))
               + 1j * rng.standard_normal((4, n_grid))).astype(np.complex128)
        gphase = np.exp(-1j * rng.standard_normal(n_grid)).astype(np.complex128)

        _isdf._RIGHT_FACTOR_PATH_LOGGED = False
        fused = np.asarray(_isdf._right_factor(provider, 1, eta, gphase))
        _isdf._RIGHT_FACTOR_PATH_LOGGED = False
        fallback = np.asarray(_isdf._right_factor(_ApplyOnly(provider), 1, eta, gphase))

        denom = np.abs(fallback).max()
        rel = np.abs(fused - fallback).max() / (denom if denom > 0 else 1.0)
        self.assertLessEqual(rel, self.BOUND)

    def test_malformed_inputs_are_rejected(self):
        cell = _make_cell()
        mesh = tuple(int(m) for m in cell.mesh)
        provider = RawKernelProvider(
            cell=cell, canonical_kpts=np.zeros((1, 3)), grid_mesh=mesh
        )
        n_grid = int(np.prod(mesh))
        eta = np.zeros((3, n_grid), dtype=np.complex128)
        gphase = np.ones(n_grid, dtype=np.complex128)
        with self.assertRaises(ValueError):
            provider.apply_right_factor(1, eta, gphase)          # q out of range
        with self.assertRaises(ValueError):
            provider.apply_right_factor(0, eta[0], gphase)        # eta not 2-D
        with self.assertRaises(ValueError):
            provider.apply_right_factor(0, eta, gphase[:-1])      # gphase wrong length


class TestFusedRightFactorIndependentOracles(unittest.TestCase):
    """Checks that do NOT reduce to "two implementations of the same recipe agree".

    The bound test compares the fused path against the ``apply``-composed one.
    Both share ``coulG`` and the same transform recipe, so a common-mode error in
    that recipe is invisible to it. These two checks attack that blind spot from
    different sides:

      1. the ALGEBRAIC IDENTITY the construction exists to provide, which fixes
         where the phase and the conjugate belong;
      2. an INDEPENDENT COMPOSITION built on numpy's transforms rather than
         jax's, catching axis/reshape/scaling mistakes in the traced core.

    Neither can catch an error inside ``coulG`` itself -- that comes from pyscf,
    is shared with production, and is not something this change introduces. The
    blind spot is named rather than papered over.
    """

    BOUND = 1e-13

    def test_algebraic_identity_phase_may_move_to_the_right_factor(self):
        """(eta_i*g) @ conj(apply(eta_j*g)).T  ==  eta_i @ right_factor(eta_j).T

        This is the identity quoted in _right_factor's docstring and the reason
        the left operand can stay unphased. It pins the phase placement and the
        conjugate independently of whether the two implementations agree.
        """
        cell = _make_cell()
        mesh = tuple(int(m) for m in cell.mesh)
        kpts = np.array([[0.0, 0.0, 0.0], [0.11, -0.07, 0.23]])
        provider = RawKernelProvider(cell=cell, canonical_kpts=kpts, grid_mesh=mesh)
        n_grid = int(np.prod(mesh))
        rng = np.random.default_rng(23)
        eta_i = (rng.standard_normal((3, n_grid))
                 + 1j * rng.standard_normal((3, n_grid))).astype(np.complex128)
        eta_j = (rng.standard_normal((4, n_grid))
                 + 1j * rng.standard_normal((4, n_grid))).astype(np.complex128)
        for q in range(len(kpts)):
            gphase = np.exp(-1j * rng.standard_normal(n_grid)).astype(np.complex128)
            lhs = (eta_i * gphase[None, :]) @ np.conj(
                np.asarray(provider.apply(q, eta_j * gphase[None, :]))).T
            rhs = eta_i @ np.asarray(
                provider.apply_right_factor(q, eta_j, gphase)).T
            denom = np.abs(lhs).max()
            rel = np.abs(rhs - lhs).max() / (denom if denom > 0 else 1.0)
            self.assertLessEqual(rel, self.BOUND, f"identity failed at q={q}")

    def test_matches_an_independent_numpy_composition(self):
        """Oracle built with numpy transforms, not the jax core under test."""
        from pyscf.pbc import tools as pbctools

        cell = _make_cell()
        mesh = tuple(int(m) for m in cell.mesh)
        kpts = np.array([[0.0, 0.0, 0.0], [0.17, 0.05, -0.09]])
        provider = RawKernelProvider(cell=cell, canonical_kpts=kpts, grid_mesh=mesh)
        n_grid = int(np.prod(mesh))
        Gv = cell.get_Gv(list(mesh))
        rng = np.random.default_rng(29)
        eta = (rng.standard_normal((3, n_grid))
               + 1j * rng.standard_normal((3, n_grid))).astype(np.complex128)

        for q, kpt in enumerate(kpts):
            gphase = np.exp(-1j * rng.standard_normal(n_grid)).astype(np.complex128)
            coulG = pbctools.get_coulG(
                cell, k=np.asarray(kpt, dtype=np.float64), exx=False,
                Gv=Gv, mesh=list(mesh))
            coulG = np.asarray(coulG, dtype=np.float64) * (cell.vol / n_grid)

            lq = eta * gphase[None, :]
            w = np.fft.fftn(lq.reshape((-1,) + mesh), axes=(1, 2, 3))
            w = w * coulG.reshape(mesh)[None, :, :, :]
            v = np.fft.ifftn(w, axes=(1, 2, 3)).reshape(eta.shape[0], -1)
            oracle = np.conj(v) * gphase[None, :]

            fused = np.asarray(provider.apply_right_factor(q, eta, gphase))
            denom = np.abs(oracle).max()
            rel = np.abs(fused - oracle).max() / (denom if denom > 0 else 1.0)
            self.assertLessEqual(rel, self.BOUND, f"numpy oracle mismatch at q={q}")
