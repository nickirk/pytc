"""build_coul_kpt_host must track build_coul_kpt_device.

The host mirror exists so an accuracy question can be settled without first
porting a solver to the device path. That is only sound while the two agree:
the moment the loops drift -- conjugate shortcut, self_paired at nq == q, per-q
ordering -- the mirror stops being a reference and becomes a second, unvalidated
implementation. These tests are what make the mirror usable as evidence.
"""

import unittest
from unittest import mock

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
from pytc.pbc.coulomb import (FROZEN_BPC_POLICY, ISDFDF, build,
                              predicted_cached_ao_bytes, validate_option_compatibility)
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
            # "tsvd" was accepted here while the fallback existed. It cannot occur
            # now, and leaving it in the tuple would let a resurrected fallback pass.
            self.assertEqual(info["solver"], "unscaled_cholesky_jitter")
            self.assertFalse(info["fallback_triggered"])

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


class TestOptionsRefusedAtConstruction(unittest.TestCase):
    """The refusals existed already; what is tested here is WHERE they fire.

    They previously lived only inside build(), which reaches them after pivot
    selection, so a combination knowable in microseconds killed a production run
    ~3 h in. These assert construction-time failure -- a test of placement, which
    a test of the error message alone would not catch.
    """

    def setUp(self):
        self.cell = _make_cell()
        self.kpts = self.cell.make_kpts((1, 1, 1), wrap_around=False)

    def _isdfdf(self, **kw):
        return ISDFDF(self.cell, self.kpts, rank=12, block_size=200, **kw)

    def test_incompatible_levers_raise_before_any_work(self):
        cases = (
            dict(p_block_rows=4, kern_blocking={"staging_root": "/tmp"}),
            dict(p_block_rows=4, stage_eta_root="/tmp"),
            dict(solve_backend="host", p_block_rows=4),
            dict(solve_backend="device", jitter_rcond=1e-14),
            dict(solve_backend="nonsense"),
        )
        for kw in cases:
            with self.subTest(**kw):
                # ISDFDF.__init__ itself must raise. If this ever regresses to
                # raising only in build(), the failure moves hours downstream.
                with self.assertRaises(ValueError):
                    self._isdfdf(**kw)

    def test_host_rejects_custom_provider_before_mesh_or_ao_work(self):
        class _CustomProvider(RawKernelProvider):
            pass

        with mock.patch(
            "pytc.pbc.coulomb.canonicalize_kpts",
            side_effect=AssertionError("mesh work must not start"),
        ):
            with self.assertRaisesRegex(ValueError, "custom provider would be ignored"):
                build(
                    self.cell, self.kpts, rank=12, block_size=200,
                    solve_backend="host", provider_cls=_CustomProvider,
                )

    def test_valid_production_shape_constructs(self):
        # The 444 production config's shape: panel blocking alone, no staging levers.
        df = self._isdfdf(p_block_rows=4)
        self.assertEqual(df.p_block_rows, 4)
        self.assertIsNone(df.kern_blocking)

    def test_build_still_validates_for_direct_callers(self):
        # build() must keep its own check: not every caller goes through ISDFDF,
        # and construction-time validation must not become the only gate.
        with self.assertRaises(ValueError):
            validate_option_compatibility(p_block_rows=4, stage_eta_root="/tmp")


class TestStaticallyInvalidConfigsRefusedAtConstruction(unittest.TestCase):
    """Review found three statically invalid configurations that constructed fine and
    failed only after pivot selection: (device, cholesky_jitter, jitter=None),
    (host, single, jitter_rcond set), and (host, cholesky_jitter, rtol set). The
    first validator took neither retention_mode nor rtol -- fixing placement for
    SOME options and not others is not a fix. These pin the exact placements."""

    def setUp(self):
        self.cell = _make_cell()
        self.kpts = self.cell.make_kpts((1, 1, 1), wrap_around=False)

    def test_the_three_placements_fail_at_construction(self):
        cases = (
            # device + cholesky_jitter is legal as of 2026-08-10; covered by
            # TestCholeskyBiasPolicyPrecondition instead.
            dict(solve_backend="host", retention_mode="single", jitter_rcond=1e-14),
            dict(solve_backend="host", retention_mode="cholesky_jitter", rtol=1e-6),
            dict(solve_backend="host", retention_mode="cholesky_jitter", n_retained_pin=3),
        )
        for kw in cases:
            with self.subTest(**kw):
                with self.assertRaises(ValueError):
                    ISDFDF(self.cell, self.kpts, rank=12, block_size=200, **kw)

    def test_bpc_n_topup_rejects_bool_and_float_before_coercion(self):
        # int() accepts True as 1 and 3.7 as 3, then the clamp records the coerced
        # value as though the caller had asked for it.
        for bad in (True, 3.7):
            with self.subTest(bpc_n_topup=bad):
                with self.assertRaises(ValueError):
                    build(self.cell, self.kpts, rank=12, block_size=200, rtol=1e-6,
                          selection_mode="bpc_streamed", bpc_n_topup=bad)


class TestCholeskyBiasPolicyPrecondition(unittest.TestCase):
    """The confinement these tests protected is GONE as of 2026-08-10: the jitted
    path now runs cholesky_jitter, so the mode can reach a P-blocked run and
    warning-only is no longer justified by unreachability.

    The owner settled the policy rather than the confinement being removed by
    accident: judge this mode against an energy reference, not a hard residual
    gate. What must therefore still hold is the weaker but load-bearing property
    that the bias is REPORTED. A silent large-bias solve is the failure these
    tests exist to prevent, and it is now the only one they can prevent.
    """

    def setUp(self):
        self.cell = _make_cell()
        self.kpts = self.cell.make_kpts((1, 1, 1), wrap_around=False)

    def _isdfdf(self, **kw):
        return ISDFDF(self.cell, self.kpts, rank=12, block_size=200, **kw)

    def test_the_reference_path_still_refuses_the_production_lever(self):
        # The host path remains reference-grade; only its pairing with the
        # jitted path changed.
        for kw in (dict(solve_backend="host", p_block_rows=4),
                   dict(solve_backend="host", retention_mode="cholesky_jitter",
                        p_block_rows=4)):
            with self.subTest(**kw):
                with self.assertRaises(ValueError):
                    self._isdfdf(**kw)

    def test_the_jitted_path_now_accepts_the_mode(self):
        # Was refused; refusing it again would mean the wiring regressed.
        df = self._isdfdf(solve_backend="device", retention_mode="cholesky_jitter",
                          jitter_rcond=1e-6)
        self.assertEqual(df.retention_mode, "cholesky_jitter")

    def test_a_large_bias_is_reported_not_swallowed(self):
        # The property the confinement used to guarantee. Without this the mode
        # can run P-blocked at production scale and say nothing about its bias.
        import numpy as np
        from pytc.df.solvers import hermitian_sandwich_solve_device
        n = 40
        rng = np.random.default_rng(0)
        q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        pi = (q * np.logspace(0, -9, n)) @ q.conj().T
        pi = (pi + pi.conj().T) / 2
        v = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        v = (v + v.conj().T) / 2
        with self.assertLogs("pytc.df.solvers", level="WARNING") as captured:
            _, info = hermitian_sandwich_solve_device(
                pi, v, retention_mode="cholesky_jitter", jitter_rcond=1e-6)
        self.assertTrue(
            any("two-sided fit residual" in line for line in captured.output),
            f"bias not reported: {captured.output}",
        )
        self.assertGreater(info["retained_solve_residual"], 1e-10)

    def test_the_reference_path_itself_is_reachable(self):
        # Control: the refusals above must not be vacuous. Without the production
        # lever the mode is available, which is what makes it a reference path
        # rather than dead code.
        df = self._isdfdf(solve_backend="host", retention_mode="cholesky_jitter")
        self.assertEqual(df.retention_mode, "cholesky_jitter")
        self.assertIsNone(df.p_block_rows)


class TestBpcPolicyAndCacheGate(unittest.TestCase):
    """Task #57. The task's premise was that batch_size exceeds a small cell's
    candidate pool; isolation showed otherwise -- batch_size=999 and
    candidate_oversampling=99 both pass at rank=12, because batch_size self-limits.
    The only size-dependent constraint is n_topup <= rank."""

    def setUp(self):
        self.cell = _make_cell()
        self.kpts = self.cell.make_kpts((1, 1, 1), wrap_around=False)

    def _build(self, **kw):
        return build(self.cell, self.kpts, rank=12, block_size=200, rtol=1e-6, **kw)

    def test_frozen_policy_no_longer_fails_on_a_small_rank(self):
        # The exact combination that broke 19 tests: n_topup=16 against rank=12.
        out = self._build(selection_mode="bpc_cached_gemm", **FROZEN_BPC_POLICY)
        prov = out["selection_provenance"]
        self.assertEqual(prov["bpc_n_topup_requested"], 16)
        self.assertEqual(prov["bpc_n_topup"], 12)
        self.assertTrue(prov["bpc_n_topup_clamped_to_rank"])

    def test_clamp_is_recorded_not_silent(self):
        # A silent clamp would make two different runs look identical in provenance.
        out = self._build(selection_mode="bpc_cached_gemm",
                          **{**FROZEN_BPC_POLICY, "bpc_n_topup": 4})
        prov = out["selection_provenance"]
        self.assertEqual(prov["bpc_n_topup"], 4)
        self.assertFalse(prov["bpc_n_topup_clamped_to_rank"])

    def test_auto_picks_cached_when_it_fits_and_streamed_when_it_does_not(self):
        big = self._build(selection_mode="bpc_auto", **FROZEN_BPC_POLICY)
        self.assertEqual(big["selection_provenance"]["auto_resolved_to"], "cached")
        tiny = self._build(selection_mode="bpc_auto", cached_ao_max_bytes=1024,
                           **FROZEN_BPC_POLICY)
        self.assertEqual(tiny["selection_provenance"]["auto_resolved_to"], "streamed")

    def test_explicit_cached_over_ceiling_fails_closed(self):
        # Refuse before allocating, rather than OOM mid-selection.
        with self.assertRaises(ValueError):
            self._build(selection_mode="bpc_cached_gemm", cached_ao_max_bytes=1024,
                        **FROZEN_BPC_POLICY)

    def test_predicted_bytes_matches_the_real_allocation(self):
        # The gate is only worth having if its prediction is the actual size.
        out = self._build(selection_mode="bpc_cached_gemm", **FROZEN_BPC_POLICY)
        prov = out["selection_provenance"]
        self.assertEqual(prov["predicted_cached_ao_bytes"], prov["cache_bytes"])
        self.assertEqual(
            prov["predicted_cached_ao_bytes"],
            predicted_cached_ao_bytes(prov["candidate_count"], 1, self.cell.nao))


if __name__ == "__main__":
    unittest.main()
