"""End-to-end test for pytc.pbc.coulomb.build (design doc §2): S1-S4 on a
tiny real cell vs a from-scratch stage-by-stage reconstruction.
get_k/get_j structural tests live in test_coulomb_get_k_get_j.py."""

import dataclasses
import itertools
import json
import os
import unittest
from unittest import mock

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc import coulomb
from pytc.pbc.df.isdf import (
    RawKernelProvider,
    apply_raw_kernel_and_solve,
    build_cached_periodic_bpc_gemm_oracle,
    build_periodic_pivot_oracle,
    build_pi_eta,
    pivoted_cholesky_hermitian,
    stream_ao_blocks,
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


class _AlternateRawProvider(RawKernelProvider):
    def provenance(self):
        return {
            **super().provenance(),
            "kernel_name": "alternate_raw_test",
        }


class _MissingNormalizationProvider(RawKernelProvider):
    def provenance(self):
        provenance = super().provenance()
        provenance.pop("normalization")
        return provenance


class _UnserializableProvider(RawKernelProvider):
    def provenance(self):
        return {**super().provenance(), "opaque": object()}


class TestBuildPlan(unittest.TestCase):
    @staticmethod
    def _resolve(**overrides):
        options = dict(
            n_grid=10,
            n_kpts=2,
            n_ao=3,
            rank=3,
            block_size=9,
            provider_cls=RawKernelProvider,
            provider_details={
                "normalization": "unit_test",
                "grid_mesh": np.asarray([1, 2, 3], dtype=np.int64),
            },
            selection_mode="streamed",
        )
        options.update(overrides)
        return coulomb.resolve_build_plan(**options)

    def test_route_matrix_accepts_exactly_the_static_legal_combinations(self):
        """Exhaust the independent route switches without doing AO work."""
        custom_provider = type("_CustomProvider", (RawKernelProvider,), {})
        values = itertools.product(
            ("device", "host", "deterministic_cpu"),
            (False, True),  # panel blocking
            (False, True),  # eta staging
            (False, True),  # kernel blocking
            (*coulomb.RETENTION_MODES, "not_a_mode"),
            (False, True),  # jitter_rcond
            (False, True),  # custom provider
            (False, True),  # rtol
            (False, True),  # retained-rank pin
            (False, True),  # adaptive residual target
        )
        checked = 0
        for (backend, panel, staged, kern_blocked, retention, jitter,
             custom, with_rtol, pin, target) in values:
            with self.subTest(
                backend=backend, panel=panel, staged=staged,
                kern_blocked=kern_blocked, retention=retention,
                jitter=jitter, custom=custom, rtol=with_rtol, pin=pin,
                target=target,
            ):
                legal = retention in coulomb.RETENTION_MODES
                legal &= not (panel and kern_blocked)
                legal &= not (panel and staged)
                legal &= not (backend == "host" and (panel or kern_blocked))
                legal &= not (backend == "host" and custom)
                legal &= not (
                    backend == "deterministic_cpu"
                    and (not panel or retention != "single")
                )
                if retention == "cholesky_jitter":
                    # Runs on both backends now: the jitted path takes a fixed
                    # jitter instead of the host loop's escalation.
                    legal &= not with_rtol and not pin and not target
                else:
                    legal &= not jitter
                if retention != "single":
                    legal &= not pin and not target
                if pin:
                    legal &= not with_rtol and not target

                kwargs = dict(
                    solve_backend=backend,
                    p_block_rows=2 if panel else None,
                    stage_eta_root="stage" if staged else None,
                    kern_blocking={"row_block": 2} if kern_blocked else None,
                    retention_mode=retention,
                    jitter_rcond=1e-12 if jitter else None,
                    provider_cls=custom_provider if custom else RawKernelProvider,
                    rtol=1e-6 if with_rtol else None,
                    n_retained_pin=2 if pin else None,
                    target_truncation_residual=1e-3 if target else None,
                )
                if not legal:
                    with self.assertRaises(ValueError):
                        self._resolve(**kwargs)
                    checked += 1
                    continue

                plan = self._resolve(**kwargs)
                resolved = plan.to_dict()["resolved"]
                self.assertEqual(resolved["solve_backend"], backend)
                self.assertEqual(resolved["retention_mode"], retention)
                self.assertEqual(
                    resolved["eta_strategy"],
                    "panel_blocked" if panel else (
                        "staged" if staged else "resident_streamed"
                    ),
                )
                self.assertEqual(
                    resolved["kernel_strategy"],
                    "panel_precomputed" if panel else (
                        "grid_blocked" if kern_blocked else "dense"
                    ),
                )
                # The numpy array in provider provenance must be normalized at
                # the plan boundary, before an expensive build can complete.
                json.dumps(plan.to_dict())
                checked += 1
        self.assertEqual(checked, 3840)

    def test_retained_rank_pin_is_normalized_and_validated_pre_ao(self):
        for pin, expected in (
            (2, 2),
            ([1, 2], [1, 2]),
            (np.asarray([2, 3], dtype=np.int32), [2, 3]),
        ):
            with self.subTest(pin=repr(pin)):
                resolved = self._resolve(n_retained_pin=pin).to_dict()["resolved"]
                self.assertEqual(resolved["n_retained_pin"], expected)

        for pin in (True, 0, 4, [1], [1, 4], [1.0, 2.0]):
            with self.subTest(invalid_pin=repr(pin)):
                with self.assertRaises(ValueError):
                    self._resolve(n_retained_pin=pin)

    def test_full_build_routes_panel_kernel_to_deterministic_cpu_backend(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 1], wrap_around=False)

        def fake_deterministic(Pi, kern, grid_coords, mesh_obj, **kwargs):
            self.assertEqual(Pi.shape, kern.shape)
            self.assertEqual(Pi.shape[0], mesh_obj.n_kpts)
            self.assertGreater(grid_coords.shape[0], 0)
            self.assertEqual(kwargs["retention_mode"], "single")
            zeros = np.zeros_like(Pi)
            infos = [{"execution_backend": "test-double"}] * mesh_obj.n_kpts
            return zeros, kern, infos, 1

        with mock.patch.object(
            coulomb,
            "build_coul_kpt_deterministic_cpu",
            side_effect=fake_deterministic,
        ) as isolated_solve:
            built = coulomb.build(
                cell,
                kpts,
                rank=3,
                block_size=11,
                p_block_rows=2,
                solve_backend="deterministic_cpu",
                retention_mode="single",
                selection_mode="streamed",
            )
        isolated_solve.assert_called_once()
        self.assertEqual(
            built["build_plan"]["resolved"]["solve_backend"],
            "deterministic_cpu",
        )
        self.assertEqual(
            built["solve_infos"][0]["execution_backend"], "test-double"
        )

    def test_invalid_retention_routes_refuse_before_mesh_or_ao(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        invalid = (
            {"retention_mode": "not_a_mode"},
            {"retention_mode": "pairwise", "n_retained_pin": 2},
            {"retention_mode": "svd_lstsq", "n_retained_pin": 2},
        )
        with mock.patch.object(
            coulomb, "canonicalize_kpts",
            side_effect=AssertionError("mesh construction was reached"),
        ):
            for kwargs in invalid:
                with self.subTest(**kwargs):
                    with self.assertRaises(ValueError):
                        coulomb.build(
                            cell, kpts, rank=3, block_size=9,
                            selection_mode="streamed", **kwargs,
                        )

    def test_selection_matrix_resolves_or_rejects_every_public_mode(self):
        predicted = coulomb.predicted_cached_ao_bytes(10, 2, 3)
        modes = (
            None,
            "bpc_auto",
            "bpc_cached_gemm",
            "bpc_streamed",
            "streamed",
            "fixed_pivots",
            next(iter(coulomb.RETIRED_SELECTION_MODES)),
            "not_a_mode",
        )
        for mode, fixed, over_ceiling in itertools.product(
            modes, (False, True), (False, True)
        ):
            with self.subTest(mode=mode, fixed=fixed, over=over_ceiling):
                ceiling = predicted - 1 if over_ceiling else predicted
                legal = mode not in coulomb.RETIRED_SELECTION_MODES
                legal &= mode in coulomb.SELECTOR_STORAGE or mode is None
                legal &= not (mode == "fixed_pivots" and not fixed)
                legal &= not (
                    fixed and mode not in (None, "streamed", "fixed_pivots")
                )
                effective = "fixed_pivots" if fixed else (
                    coulomb.DEFAULT_SELECTION_MODE if mode is None else mode
                )
                legal &= not (
                    effective == "bpc_cached_gemm" and over_ceiling
                )
                kwargs = dict(
                    selection_mode=mode,
                    fixed_pivots=np.asarray([0, 1, 2]) if fixed else None,
                    cached_ao_max_bytes=ceiling,
                )
                if not legal:
                    with self.assertRaises(ValueError):
                        self._resolve(**kwargs)
                    continue

                plan = self._resolve(**kwargs)
                record = plan.to_dict()
                self.assertEqual(record["requested"]["selection_mode"], mode)
                if fixed:
                    expected_mode, expected_storage = "fixed_pivots", None
                elif effective == "bpc_auto":
                    expected_mode = (
                        "bpc_streamed" if over_ceiling else "bpc_cached_gemm"
                    )
                    expected_storage = "streamed" if over_ceiling else "cached"
                else:
                    expected_mode = effective
                    expected_storage = coulomb.SELECTOR_STORAGE[effective][1]
                self.assertEqual(
                    record["resolved"]["selection_mode"], expected_mode
                )
                self.assertEqual(record["resolved"]["storage"], expected_storage)

    def test_gamma_metric_sizes_only_one_kpoint_and_disables_eta_cache_reuse(self):
        full = self._resolve(
            n_grid=10, n_kpts=8, n_ao=3,
            selection_mode="bpc_cached_gemm",
            selection_metric="full_k",
        ).to_dict()
        gamma = self._resolve(
            n_grid=10, n_kpts=8, n_ao=3,
            selection_mode="bpc_cached_gemm",
            selection_metric="gamma",
        ).to_dict()
        self.assertEqual(
            full["resolved"]["predicted_cached_ao_bytes"],
            8 * gamma["resolved"]["predicted_cached_ao_bytes"],
        )
        self.assertEqual(gamma["resolved"]["selection_metric_n_kpts"], 1)
        self.assertTrue(gamma["requested"]["reuse_ao_cache_for_eta"])
        self.assertFalse(gamma["resolved"]["reuse_ao_cache_for_eta"])
        self.assertEqual(
            gamma["resolved"]["reuse_ao_cache_for_eta_disabled_reason"],
            "gamma_selection_cache_has_one_kpoint",
        )

    def test_invalid_selection_metric_refuses_before_ao(self):
        with self.assertRaisesRegex(ValueError, "selection_metric"):
            self._resolve(selection_metric="q_dependent")

    def test_plan_is_deeply_immutable_and_returns_fresh_json_values(self):
        details = {"normalization": "unit_test", "nested": {"mesh": [1, 2, 3]}}
        plan = self._resolve(provider_details=details)
        details["nested"]["mesh"][0] = 999
        self.assertEqual(
            plan.to_dict()["resolved"]["provider"]["details"]["nested"]["mesh"],
            [1, 2, 3],
        )
        first = plan.to_dict()
        first["resolved"]["provider"]["details"]["nested"]["mesh"][0] = 888
        self.assertEqual(
            plan.to_dict()["resolved"]["provider"]["details"]["nested"]["mesh"],
            [1, 2, 3],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan._resolved_json = "{}"


class TestBuild(unittest.TestCase):
    def test_gamma_selection_metric_uses_one_kpoint_then_returns_to_full_mesh(self):
        """The production call path, not only provenance, must drop the Nk sum."""
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        original = cell.pbc_eval_gto
        selection_finished = False
        seen_during_selection = []
        seen_after_selection = []

        def record_eval(*args, **kwargs):
            target = (seen_after_selection if selection_finished
                      else seen_during_selection)
            target.append(len(kwargs["kpts"]))
            return original(*args, **kwargs)

        def mark_selection_done(_pivots, provenance):
            nonlocal selection_finished
            self.assertEqual(provenance["metric_kpoint_policy"], "gamma")
            self.assertEqual(provenance["metric_n_kpts"], 1)
            self.assertEqual(len(provenance["metric_kpoint_indices"]), 1)
            selection_finished = True

        with mock.patch.object(cell, "pbc_eval_gto", side_effect=record_eval):
            result = coulomb.build(
                cell, kpts, rank=3, block_size=9,
                selection_mode="bpc_cached_gemm",
                selection_metric="gamma",
                on_selection=mark_selection_done,
            )

        self.assertTrue(seen_during_selection, "selection made no AO calls")
        self.assertEqual(set(seen_during_selection), {1})
        self.assertIn(len(kpts), seen_after_selection)
        self.assertEqual(
            result["build_plan"]["resolved"]["selection_metric"], "gamma"
        )
        self.assertFalse(
            result["build_plan"]["resolved"]["reuse_ao_cache_for_eta"]
        )

    def test_build_records_exact_provider_and_normalization_provenance(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 1], wrap_around=False)
        device = coulomb.build(
            cell, kpts, rank=3, block_size=9,
            provider_cls=_AlternateRawProvider,
        )
        host = coulomb.build(
            cell, kpts, rank=3, block_size=9, solve_backend="host",
        )

        device_provider = device["kernel_provider"]
        self.assertEqual(
            device_provider["provider_class"],
            f"{_AlternateRawProvider.__module__}.{_AlternateRawProvider.__qualname__}",
        )
        self.assertEqual(
            device_provider["details"]["kernel_name"], "alternate_raw_test"
        )
        self.assertEqual(
            device_provider["details"]["normalization"],
            "vol_over_ng_inside_provider",
        )

        host_provider = host["kernel_provider"]
        self.assertEqual(
            host_provider["provider_class"],
            f"{RawKernelProvider.__module__}.{RawKernelProvider.__qualname__}",
        )
        self.assertEqual(
            host_provider["details"]["normalization"],
            "vol_over_ng_inside_provider",
        )
        # The complete requested/resolved plan must survive direct artifact
        # persistence, including the provider's normalized grid_mesh value.
        self.assertEqual(
            json.loads(json.dumps(device["build_plan"])), device["build_plan"]
        )
        self.assertEqual(
            device["build_plan"]["resolved"]["provider"], device_provider
        )
        self.assertEqual(
            device["build_plan"]["requested"]["selection_mode"], None
        )
        self.assertEqual(
            device["build_plan"]["requested"]["selection_metric"], "full_k"
        )
        self.assertEqual(
            device["build_plan"]["resolved"]["selection_metric"], "full_k"
        )
        self.assertEqual(device["build_plan"]["requested"]["rank"], 3)
        self.assertEqual(device["build_plan"]["requested"]["block_size"], 9)
        self.assertEqual(
            device["build_plan"]["resolved"]["bpc_policy"]["n_topup"], 3
        )
        self.assertIn(
            device["build_plan"]["resolved"]["selection_mode"],
            ("bpc_cached_gemm", "bpc_streamed"),
        )

    def test_missing_provider_normalization_is_rejected_before_ao_work(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 1], wrap_around=False)
        with mock.patch.object(
            cell, "pbc_eval_gto", side_effect=AssertionError("AO work must not start")
        ):
            with self.assertRaisesRegex(ValueError, "non-empty 'normalization'"):
                coulomb.build(
                    cell, kpts, rank=3, block_size=9,
                    provider_cls=_MissingNormalizationProvider,
                )

    def test_unserializable_plan_value_is_rejected_before_ao_work(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 1], wrap_around=False)
        with mock.patch.object(
            cell, "pbc_eval_gto", side_effect=AssertionError("AO work must not start")
        ):
            with self.assertRaisesRegex(TypeError, "non-JSON value"):
                coulomb.build(
                    cell, kpts, rank=3, block_size=9,
                    provider_cls=_UnserializableProvider,
                )

    def test_selection_provenance_carries_per_round_stage_stats(self):
        """Per-stage timings must reach a REAL build, not only a unit test.

        `stage_stats` was a selector parameter no production caller ever
        supplied -- the only `stage_stats=` argument in the product was in
        test_isdf_selector. So the per-stage timings, including the
        `materialisation_seconds` bucket added to attribute the async device
        read that a wall-clock gap was hiding in, observed nothing during an
        actual build. A profiling run would have produced no breakdown at all.
        """
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 1], wrap_around=False)
        seen = []
        coulomb.build(cell, kpts, rank=3, block_size=9,
                      selection_mode="bpc_streamed",
                      on_selection=lambda p, prov: seen.append(prov))
        self.assertEqual(len(seen), 1)
        stats = seen[0].get("bpc_stage_stats")
        self.assertIsNotNone(
            stats, "per-stage timings never reach the callback, so a build "
                   "cannot be profiled from its own receipt")
        self.assertTrue(stats, "stage stats present but EMPTY -- a vacuous pass")
        for item in stats:
            for field in ("candidate_eval_seconds", "projection_seconds",
                          "materialisation_seconds",
                          "within_batch_pivot_seconds", "factor_update_seconds"):
                self.assertIn(field, item)
                self.assertGreaterEqual(item[field], 0.0)

    def test_on_selection_fires_once_before_the_build_and_enables_resume(self):
        """The hook exists so hours of selection survive an interruption. Assert
        the property that matters -- what it hands back is sufficient to resume --
        rather than merely that it was called."""
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        seen = []
        result = coulomb.build(cell, kpts, rank=3, block_size=9,
                               on_selection=lambda p, prov: seen.append((p, prov)))
        self.assertEqual(len(seen), 1)
        pivots, prov = seen[0]
        # Sufficient to resume: identical to what the completed build reports.
        self.assertEqual(pivots.tolist(),
                         result["selection_provenance"]["pivot_indices"])
        self.assertIn("mode", prov)
        # Handed a copy, so a caller mutating it cannot corrupt the build.
        pivots[0] = -12345
        self.assertNotEqual(result["selection_provenance"]["pivot_indices"][0], -12345)

        # The round trip is the point: feeding it back reproduces the build.
        resumed = coulomb.build(cell, kpts, rank=3, block_size=9,
                                fixed_pivots=np.asarray(
                                    result["selection_provenance"]["pivot_indices"]))
        np.testing.assert_allclose(resumed["coul_kpt"], result["coul_kpt"],
                                   rtol=0, atol=1e-12)

    def test_on_selection_failure_does_not_take_the_build_with_it(self):
        """A checkpoint that raises must cost the checkpoint, not the run. The
        build is hours of work; the callback is a file write."""
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)

        def boom(pivots, prov):
            raise RuntimeError("scratch filesystem full")

        with self.assertLogs("pytc.pbc.coulomb", level="WARNING") as cm:
            result = coulomb.build(cell, kpts, rank=3, block_size=9,
                                   on_selection=boom)
        self.assertTrue(any("scratch filesystem full" in m for m in cm.output))
        # Recorded, not silent: a later reader can tell the pivots were not saved.
        self.assertIn("on_selection_error",
                      result["selection_provenance"])

    def test_retired_selection_mode_names_its_replacement(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        for retired, replacement in coulomb.RETIRED_SELECTION_MODES.items():
            with self.assertRaises(ValueError) as ctx:
                coulomb.build(cell, kpts, rank=3, block_size=9, selection_mode=retired)
            self.assertIn(replacement, str(ctx.exception))

    def test_retired_keyword_is_a_type_error_not_a_directive(self):
        # Kwargs removed with the retired selectors bind-fail before any
        # mode validation runs, so a legacy call raises TypeError rather than
        # the directive ValueError above. Asserted so the distinction is
        # documented rather than discovered.
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        with self.assertRaises(TypeError):
            coulomb.build(cell, kpts, rank=3, block_size=9,
                          selection_peak_max_bytes=1 << 30)

    def test_default_is_bpc_with_the_frozen_policy_and_gated_storage(self):
        # Promoted 2026-07-29 on owner instruction. This test previously asserted
        # the opposite and is UPDATED rather than deleted: it guards the reasons the
        # default may move, not the value. Both blockers had to be answered first --
        # n_topup=16 is now clamped to rank (it was refused at rank<16), and storage
        # is chosen on predicted bytes rather than allocating uncapped.
        self.assertEqual(coulomb.DEFAULT_SELECTION_MODE, "bpc_auto")
        self.assertEqual(coulomb.FROZEN_BPC_POLICY, {
            "bpc_batch_size": 64, "bpc_min_separation": 2.0,
            "bpc_candidate_oversampling": 4, "bpc_n_topup": 16,
            "bpc_blocked_projection": True,
        })
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        # rank=3 is far below n_topup=16: exactly the combination that used to raise.
        result = coulomb.build(cell, kpts, rank=3, block_size=9)
        prov = result["selection_provenance"]
        self.assertEqual(prov["selector"], "bpc")
        self.assertIn(prov["storage"], ("cached", "streamed"))
        self.assertEqual(prov["auto_resolved_to"], prov["storage"])
        # The tuning must travel with the mode; defaulting one without the other
        # ships a configuration nothing validated.
        self.assertEqual(prov["bpc_batch_size"], 64)
        self.assertEqual(prov["bpc_n_topup_requested"], 16)
        self.assertEqual(prov["bpc_n_topup"], 3)
        self.assertTrue(prov["bpc_n_topup_clamped_to_rank"])

    def test_build_produces_self_consistent_artifact(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        result = coulomb.build(cell, kpts, rank=4, block_size=11)

        mesh_obj = result["mesh_obj"]
        self.assertEqual(mesh_obj.n_kpts, 3)
        n_selected = result["n_selected"]
        self.assertLessEqual(n_selected, 4)
        self.assertGreater(n_selected, 0)

        self.assertEqual(result["inpv_kpt"].shape, (3, n_selected, cell.nao))
        self.assertEqual(result["coul_kpt"].shape, (3, n_selected, n_selected))
        self.assertEqual(result["kern_kpt"].shape, (3, n_selected, n_selected))
        self.assertEqual(len(result["solve_infos"]), 3)
        self.assertLessEqual(result["n_pipeline_calls"], 3)

        for q in range(3):
            W_q = np.asarray(result["coul_kpt"][q])
            np.testing.assert_allclose(W_q, W_q.conj().T, atol=1e-8, err_msg=f"q={q}")

    def test_fixed_pivots_reuse_exact_ao_downstream(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grid_coords = cell.get_uniform_grids(cell.mesh)
        diagonal, column = build_periodic_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size=13,
        )
        pivots, _, _ = pivoted_cholesky_hermitian(diagonal, column, rank=3)
        selected = coulomb.build(
            cell, kpts, rank=3, block_size=13,
            selection_mode="streamed", fixed_pivots=pivots,
        )
        baseline = coulomb.build(
            cell, kpts, rank=3, block_size=13,
            selection_mode="streamed",
        )
        self.assertEqual(selected["selection_provenance"]["mode"], "fixed_pivots_experimental")
        self.assertEqual(selected["selection_provenance"]["pivot_indices"], pivots.tolist())
        np.testing.assert_allclose(selected["inpv_kpt"], baseline["inpv_kpt"], atol=0.0)
        np.testing.assert_allclose(selected["coul_kpt"], baseline["coul_kpt"], atol=1e-10, rtol=1e-10)

    def test_fixed_pivot_mode_requires_explicit_indices(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        with self.assertRaisesRegex(ValueError, "requires an explicit fixed_pivots"):
            coulomb.build(
                cell, kpts, rank=3, block_size=13,
                selection_mode="fixed_pivots",
            )

    def test_build_matches_manual_stage_by_stage_reconstruction(self):
        # Strongest check: reconstruct the SAME artifact by manually
        # driving the individual stage functions (as opposed to
        # build()'s own orchestration) and compare.
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        rank, block_size = 3, 9

        # The manual reconstruction below drives the exact streamed oracle, so
        # request it explicitly rather than inheriting whatever the default is.
        result = coulomb.build(cell, kpts, rank=rank, block_size=block_size,
                               rtol=1e-8, retention_mode="single",
                               selection_mode="streamed")

        mesh_obj = canonicalize_kpts(cell, kpts)
        grid_coords = cell.get_uniform_grids(cell.mesh)
        diag, col_eval = build_periodic_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size
        )
        pivots, _, n_selected = pivoted_cholesky_hermitian(diag, col_eval, rank=rank)
        inpv_kpt = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[pivots], kpts=list(mesh_obj.canonical_kpts)),
            dtype=np.complex128,
        )
        ao_full = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords, kpts=list(mesh_obj.canonical_kpts)),
            dtype=np.complex128,
        )
        Pi, eta = build_pi_eta(inpv_kpt, ao_full, mesh_obj.phase, mesh_obj.neg)

        self.assertEqual(result["n_selected"], n_selected)
        np.testing.assert_allclose(result["inpv_kpt"], inpv_kpt, atol=0.0)

        for q in range(mesh_obj.n_kpts):
            W_np, kern_np, _ = apply_raw_kernel_and_solve(
                Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
                grid_coords=grid_coords, grid_mesh=cell.mesh, rtol=1e-8,
            )
            np.testing.assert_allclose(
                np.asarray(result["coul_kpt"][q]), W_np, atol=1e-9, err_msg=f"q={q}"
            )

class TestDiamond111DevicePrecisionGuard(unittest.TestCase):
    """The device W-solve requires complex128: without jax_enable_x64 JAX
    downcasts silently and the solve loses ~9 digits, tripping the machine-tier
    retained-solve gate. Guards both branches -- x64 on passes the gate at every
    self-paired q; x64 off fails closed at the boundary with a dtype error."""

    @staticmethod
    def _diamond_111():
        cell = Cell()
        cell.atom = "C 0 0 0; C .8917 .8917 .8917"
        cell.a = "0 1.7834 1.7834\n1.7834 0 1.7834\n1.7834 1.7834 0"
        cell.unit = "A"
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.ke_cutoff = 20.0
        cell.verbose = 0
        cell.build()
        return cell

    def test_x64_on_full_build_passes_machine_tier_gate_all_self_paired_q(self):
        cell = self._diamond_111()
        kpts = cell.make_kpts([2, 2, 2])
        mesh_obj = canonicalize_kpts(cell, kpts)
        # Precondition the regression asserts it actually covers: diamond-111/
        # k222 is fully self-paired, so every q exercises the Pi_q.real path.
        self.assertTrue(
            all(int(mesh_obj.neg[q]) == q for q in range(mesh_obj.n_kpts)),
            msg="diamond-111/k222 is expected to be a fully self-paired mesh",
        )
        result = coulomb.build(
            cell, kpts, rank=6 * cell.nao_nr(), block_size=64, rtol=1e-4,
            retention_mode="single", selection_mode="streamed",
        )
        for q, info in enumerate(result["solve_infos"]):
            self.assertLessEqual(
                info["retained_solve_residual"], 1e-10,
                msg=f"q={q} retained_solve_residual exceeds the 1e-10 gate in float64",
            )

    def test_x64_off_full_build_fails_closed_with_actionable_dtype_error(self):
        cell = self._diamond_111()
        kpts = cell.make_kpts([2, 2, 2])
        with jax.enable_x64(False):
            with self.assertRaises(ValueError) as ctx:
                coulomb.build(
                    cell, kpts, rank=6 * cell.nao_nr(), block_size=64,
                    selection_mode="streamed",
                )
        message = str(ctx.exception)
        self.assertIn("jax_enable_x64", message)
        self.assertIn("complex64", message)


class TestBpcCachedGemmEtaReuse(unittest.TestCase):
    """The bpc_cached_gemm oracle caches AO features in a 2-D (Ng, Nk*Nao)
    layout, but build_pi_eta consumes 3-D (Nk, Ng, Nao) blocks; the reuse path
    must invert that pack exactly. Pins both layers: the reshape reproduces the
    stream_ao_blocks output, and a full build round-trips."""

    def test_reshape_recovers_stream_ao_blocks_exactly(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        grid_coords = cell.get_uniform_grids(cell.mesh)
        block_size = 9
        _, _, features = build_cached_periodic_bpc_gemm_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size
        )
        n_grid = len(grid_coords)
        n_kpts = len(mesh_obj.canonical_kpts)
        n_ao = cell.nao_nr()
        reshaped = features.reshape(n_grid, n_kpts, n_ao).transpose(1, 0, 2)
        streamed = np.concatenate(
            [blk for _, _, blk in stream_ao_blocks(
                cell, mesh_obj.canonical_kpts, grid_coords, block_size)],
            axis=1,
        )
        self.assertEqual(reshaped.shape, (n_kpts, n_grid, n_ao))
        # Exact layout equivalence -- the reuse must be the same numbers
        # stream_ao_blocks would have produced, not merely the right shape.
        np.testing.assert_array_equal(reshaped, streamed)

    def test_bpc_cached_gemm_full_build_roundtrips_on_diamond_111(self):
        cell = Cell()
        cell.atom = "C 0 0 0; C .8917 .8917 .8917"
        cell.a = "0 1.7834 1.7834\n1.7834 0 1.7834\n1.7834 1.7834 0"
        cell.unit = "A"
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.ke_cutoff = 20.0
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts([2, 2, 2])
        # Exercises the eta-reuse path end to end (build_pi_eta consumes the
        # reshaped bpc cache); with the pre-fix code this raised in pair_convolve.
        result = coulomb.build(
            cell, kpts, rank=6 * cell.nao_nr(), block_size=64, rtol=1e-4,
            retention_mode="single", selection_mode="bpc_cached_gemm",
        )
        for q, info in enumerate(result["solve_infos"]):
            self.assertLessEqual(
                info["retained_solve_residual"], 1e-10,
                msg=f"q={q} retained_solve_residual exceeds the 1e-10 gate",
            )

    def test_reuse_ao_cache_for_eta_false_gives_equivalent_build(self):
        # Freeing the AO cache before eta re-streams the AOs, which must not
        # change the result: same pivots and inpv_kpt, coul_kpt to the solve's
        # own fp-tie. Correctness is independent of the memory strategy.
        cell = Cell()
        cell.atom = "C 0 0 0; C .8917 .8917 .8917"
        cell.a = "0 1.7834 1.7834\n1.7834 0 1.7834\n1.7834 1.7834 0"
        cell.unit = "A"
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.ke_cutoff = 20.0
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts([2, 2, 2])
        kw = dict(rank=6 * cell.nao_nr(), block_size=64,
                  selection_mode="bpc_cached_gemm", bpc_batch_size=64,
                  bpc_min_separation=2.0, bpc_candidate_oversampling=4, bpc_n_topup=16)
        reuse = coulomb.build(cell, kpts, reuse_ao_cache_for_eta=True, **kw)
        freed = coulomb.build(cell, kpts, reuse_ao_cache_for_eta=False, **kw)
        np.testing.assert_array_equal(
            reuse["selection_provenance"]["pivot_indices"],
            freed["selection_provenance"]["pivot_indices"],
        )
        np.testing.assert_array_equal(
            np.asarray(reuse["inpv_kpt"]), np.asarray(freed["inpv_kpt"]),
        )
        # 1e-10 was the eigh path's fp-tie. On the cholesky_jitter default the
        # two AO streaming orders agree to ~1.9e-09 (0.9% of elements exceed
        # 1e-10), so the tie is looser -- the jittered Cholesky is more
        # order-sensitive than the eigendecomposition was. Kept on the default
        # deliberately: this asserts memory-strategy equivalence for the path we
        # ship, and the bound is set from the shipped path's actual behaviour
        # rather than inherited from a solve we no longer use.
        np.testing.assert_allclose(
            np.asarray(reuse["coul_kpt"]), np.asarray(freed["coul_kpt"]),
            rtol=0.0, atol=1e-8,
        )

    def test_stage_eta_root_matches_in_ram_build_and_cleans_up(self):
        # Staging eta must not change the result, and the staging file must
        # always be removed.
        import glob
        import tempfile

        cell = Cell()
        cell.atom = "C 0 0 0; C .8917 .8917 .8917"
        cell.a = "0 1.7834 1.7834\n1.7834 0 1.7834\n1.7834 1.7834 0"
        cell.unit = "A"
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.ke_cutoff = 20.0
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts([2, 2, 2])
        kw = dict(rank=6 * cell.nao_nr(), block_size=64,
                  selection_mode="bpc_cached_gemm", bpc_batch_size=64,
                  bpc_min_separation=2.0, bpc_candidate_oversampling=4,
                  bpc_n_topup=16)
        in_ram = coulomb.build(cell, kpts, **kw)
        with tempfile.TemporaryDirectory() as staging_root:
            # The campaign combination: free the AO cache AND stage eta, so the
            # AO blocks arrive as a stream rather than one resident array.
            staged = coulomb.build(cell, kpts, reuse_ao_cache_for_eta=False,
                                   stage_eta_root=staging_root,
                                   stage_eta_block=4096, **kw)
            leftover = glob.glob(os.path.join(staging_root, "*"))
        self.assertEqual(leftover, [], "staging file was not cleaned up")

        np.testing.assert_array_equal(
            in_ram["selection_provenance"]["pivot_indices"],
            staged["selection_provenance"]["pivot_indices"],
        )
        np.testing.assert_array_equal(
            np.asarray(in_ram["inpv_kpt"]), np.asarray(staged["inpv_kpt"]),
        )
        np.testing.assert_allclose(
            np.asarray(in_ram["coul_kpt"]), np.asarray(staged["coul_kpt"]),
            rtol=0.0, atol=1e-10,
        )
        # Grid-major: a flush is appended whole, so the contiguous run is the
        # entire record -- Nk * Nip * staging_block elements, not one row of
        # staging_block. Guards the property the layout exists for: the old
        # q-major slice ran at 64 KiB per row and measured ~30-40 MiB/s on NFS.
        stats = staged["eta_staging"]
        n_kpts = len(staged["inpv_kpt"])
        n_ip = np.asarray(staged["inpv_kpt"]).shape[1]
        self.assertEqual(stats["layout"], "grid_major_records")
        self.assertEqual(
            stats["write_run_bytes"], n_kpts * n_ip * 4096 * 16,
        )
        self.assertGreater(stats["write_run_bytes"], 1 << 20)
        self.assertEqual(sum(stats["record_cols"]), staged["eta_staging"]["staged_bytes"]
                         // (n_kpts * n_ip * 16))
        self.assertGreater(stats["staged_bytes"], 0)
        self.assertIsNone(in_ram["eta_staging"])

    def test_blocked_solve_matches_resident_build(self):
        # Full staged+blocked configuration must match the resident path.
        # Tolerance is 1e-9, not 1e-10: the grid-chunked Gram reorders a
        # summation and the pseudo-inverse amplifies that tie (kern_q agrees to
        # ~5e-17; coul_kpt to ~5e-10, rel ~6e-11).
        import glob
        import tempfile

        cell = Cell()
        cell.atom = "C 0 0 0; C .8917 .8917 .8917"
        cell.a = "0 1.7834 1.7834\n1.7834 0 1.7834\n1.7834 1.7834 0"
        cell.unit = "A"
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.ke_cutoff = 20.0
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts([2, 2, 2])
        kw = dict(rank=6 * cell.nao_nr(), block_size=64, rtol=1e-4,
                  retention_mode="single",
                  selection_mode="bpc_cached_gemm", bpc_batch_size=64,
                  bpc_min_separation=2.0, bpc_candidate_oversampling=4,
                  bpc_n_topup=16)
        resident = coulomb.build(cell, kpts, **kw)
        with tempfile.TemporaryDirectory() as root:
            blocked = coulomb.build(
                cell, kpts, reuse_ao_cache_for_eta=False, stage_eta_root=root,
                stage_eta_block=4096,
                kern_blocking=dict(staging_root=root, row_block=64,
                                   grid_chunk=256),
                **kw)
            leftover = glob.glob(os.path.join(root, "*"))
        self.assertEqual(leftover, [], "staging files were not cleaned up")

        np.testing.assert_array_equal(
            resident["selection_provenance"]["pivot_indices"],
            blocked["selection_provenance"]["pivot_indices"],
        )
        np.testing.assert_allclose(
            np.asarray(resident["kern_kpt"]), np.asarray(blocked["kern_kpt"]),
            rtol=0.0, atol=1e-9,
        )
        np.testing.assert_allclose(
            np.asarray(resident["coul_kpt"]), np.asarray(blocked["coul_kpt"]),
            rtol=0.0, atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()


class TestPanelBlockedBuildPath(unittest.TestCase):
    """build() selecting the panel-blocked path, which forms kern without ever
    materialising eta. The default path passing says nothing about this one, so
    it is exercised end to end and compared against it."""

    def test_panel_blocked_matches_the_default_path(self):
        # Without x64 both paths run in complex64 and would agree with each
        # other while both being wrong -- the comparison would prove nothing.
        self.assertTrue(jax.config.jax_enable_x64)
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        common = dict(rank=4, block_size=9)
        want = coulomb.build(cell, kpts, **common)
        # Several panels, so the off-diagonal pair work and the AO re-sweep both
        # actually run -- one panel would exercise neither.
        got = coulomb.build(cell, kpts, p_block_rows=2, **common)
        for key in ("coul_kpt", "kern_kpt"):
            with self.subTest(key=key):
                np.testing.assert_allclose(
                    np.asarray(got[key]), np.asarray(want[key]), rtol=0, atol=1e-10)

    def test_panel_blocked_preserves_requested_retention_mode(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        for requested in ("single", "pairwise", "svd_lstsq"):
            with self.subTest(retention_mode=requested):
                built = coulomb.build(
                    cell, kpts, rank=4, block_size=9, p_block_rows=2,
                    retention_mode=requested,
                )
                self.assertTrue(built["solve_infos"])
                self.assertTrue(all(
                    info["retention_mode"] == requested
                    for info in built["solve_infos"]
                ))

    def test_incompatible_memory_levers_are_refused(self):
        # Both are memory levers but they are alternatives: the panel path forms
        # kern directly, so the other two would silently do nothing. Accepting
        # the combination would let a caller believe two levers were active.
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        common = dict(rank=4, block_size=9,  p_block_rows=2)
        with self.assertRaises(ValueError):
            coulomb.build(cell, kpts, kern_blocking=dict(
                staging_root="/tmp", row_block=2, grid_chunk=8), **common)
        with self.assertRaises(ValueError):
            coulomb.build(cell, kpts, stage_eta_root="/tmp", **common)

    def test_panel_blocked_non_mirror_branch_agrees_with_the_mirror(self):
        # The `not mirror` branch had NO coverage: is_self_adjoint_per_q is True
        # on the only provider, so nothing exercised the path that computes the
        # transposed panel explicitly instead of mirroring it. Task #81 changed
        # that branch (lq_i is now built lazily, only where apply() needs it
        # materialised), so it needs a test rather than an argument.
        #
        # The provider below IS self-adjoint but does not declare it, so both
        # paths must produce the same kern: computing the transpose explicitly
        # and mirroring it are then two routes to one answer.
        self.assertTrue(jax.config.jax_enable_x64)

        class _UndeclaredSelfAdjoint(coulomb.RawKernelProvider):
            is_self_adjoint_per_q = False

        cell = _make_cell()
        for kmesh in ([1, 1, 1], [1, 1, 2]):
            kpts = cell.make_kpts(kmesh, wrap_around=False)
            for panel in (1, 2, 3):        # 3 leaves a ragged final panel
                with self.subTest(kmesh=tuple(kmesh), p_block_rows=panel):
                    common = dict(rank=4, block_size=9,
                                  p_block_rows=panel)
                    mirrored = coulomb.build(cell, kpts, **common)
                    explicit = coulomb.build(
                        cell, kpts, provider_cls=_UndeclaredSelfAdjoint, **common)
                    np.testing.assert_allclose(
                        np.asarray(explicit["kern_kpt"]),
                        np.asarray(mirrored["kern_kpt"]), rtol=0, atol=1e-13)

    def test_panel_blocked_uses_a_providers_fused_right_factor(self):
        # Proves the fused hook is actually TAKEN, not merely present: legacy
        # apply() raises, so any result at all means the panel loop went through
        # apply_right_factor. Without this, a provider could ship a fused method
        # that is silently never called and nothing would notice.
        #
        # The fused method here computes the reference formula, so the kern must
        # match the fallback path exactly -- this isolates "is the hook wired"
        # from "is the fused arithmetic right", which is a separate gate.
        self.assertTrue(jax.config.jax_enable_x64)

        class _FusedOnly(coulomb.RawKernelProvider):
            def apply(self, q, lq):                      # noqa: D102
                raise AssertionError(
                    "legacy apply() must not be called when the provider "
                    "implements apply_right_factor")

            def apply_right_factor(self, q, eta_q, gphase):
                lq = eta_q * gphase[None, :]
                rq = np.conj(np.asarray(
                    coulomb.RawKernelProvider.apply(self, q, lq)))
                rq *= gphase[None, :]
                return rq

        cell = _make_cell()
        for kmesh in ([1, 1, 1], [1, 1, 2]):
            kpts = cell.make_kpts(kmesh, wrap_around=False)
            for panel in (1, 2, 3):
                with self.subTest(kmesh=tuple(kmesh), p_block_rows=panel):
                    common = dict(rank=4, block_size=9,
                                  p_block_rows=panel)
                    fallback = coulomb.build(cell, kpts, **common)
                    fused = coulomb.build(cell, kpts, provider_cls=_FusedOnly,
                                          **common)
                    np.testing.assert_allclose(
                        np.asarray(fused["kern_kpt"]),
                        np.asarray(fallback["kern_kpt"]), rtol=0, atol=1e-13)

    def test_panel_blocked_agrees_across_panel_sizes(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        common = dict(rank=4, block_size=9)
        one = coulomb.build(cell, kpts, p_block_rows=4, **common)
        many = coulomb.build(cell, kpts, p_block_rows=1, **common)
        np.testing.assert_allclose(np.asarray(many["coul_kpt"]),
                                   np.asarray(one["coul_kpt"]), rtol=0, atol=1e-10)
