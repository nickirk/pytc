"""Numerical gates for the standalone Phase-A factor-direct prototype."""

from __future__ import annotations

import json
import os
import unittest

import jax

# This must happen before the first test array is created.  The Phase-A
# acceptance threshold is a FP64 reassociation threshold, not a FP32 test.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from pytc.solver import factor_direct_vvvv as factor_direct
from pytc.utils import tile_timers as _tile_timers


def _random_inputs(*, nocc: int = 2, nvir: int = 5, rank: int = 7):
    keys = jax.random.split(jax.random.key(314159), 7)
    normal = lambda key, shape: jax.random.normal(key, shape, dtype=jnp.float64)
    t2_raw = normal(keys[0], (nocc, nocc, nvir, nvir))
    # The RCCSD t2 permutation is (i,j,a,b) -> (j,i,b,a).
    t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))
    return {
        "t2": t2,
        "p": normal(keys[1], (nvir, rank)),
        "grad_p": normal(keys[2], (nvir, rank, 3)),
        "u1": normal(keys[3], (rank, rank, 3)),
        "u3": normal(keys[4], (rank, rank)),
        "d": normal(keys[5], (rank, rank)),
        "x": normal(keys[6], (nvir, nvir, rank)),
    }


def _dense_full(a, b, z, c, e):
    """Test-only V^4 reference in PyTC's (a,c,b,d) tile order."""

    return jnp.einsum("am,cm,mn,bn,dn->acbd", a, c, z, b, e)


def _dense_x(p, x):
    """Test-only partial-THC X reference in (a,c,b,d) order."""

    return jnp.einsum("am,cm,bdm->acbd", p, p, x)


def _tile_t2(tile, t2):
    return jnp.einsum("acbd,ijcd->ijab", tile, t2)


def _dense_reference_terms(data):
    p = data["p"]
    grad_p = data["grad_p"]
    u1 = data["u1"]
    t2 = data["t2"]
    k1_tile = sum(
        (_dense_full(grad_p[:, :, gamma], p, u1[:, :, gamma], p, p)
         for gamma in range(3)),
        start=jnp.zeros((p.shape[0],) * 4, dtype=t2.dtype),
    )
    k2_tile = sum(
        (_dense_full(p, p, u1[:, :, gamma], grad_p[:, :, gamma], p)
         for gamma in range(3)),
        start=jnp.zeros((p.shape[0],) * 4, dtype=t2.dtype),
    )
    k3_tile = _dense_full(p, p, data["u3"], p, p)
    d_tile = _dense_full(p, p, data["d"], p, p)
    x_tile = _dense_x(p, data["x"])

    def direct_and_pair(tile):
        return _tile_t2(tile, t2), _tile_t2(tile.transpose(2, 3, 0, 1), t2)

    k1_direct, k1_pair = direct_and_pair(k1_tile)
    k2_direct, k2_pair = direct_and_pair(k2_tile)
    k3_direct, k3_pair = direct_and_pair(k3_tile)
    d_direct, d_pair = direct_and_pair(d_tile)
    x_direct, x_pair = direct_and_pair(x_tile)
    tc_direct = 0.5 * (k1_direct - k2_direct + k3_direct)
    tc_pair = 0.5 * (k1_pair - k2_pair + k3_pair)
    delta_direct = d_direct - x_direct
    delta_pair = d_pair - x_pair
    tc = -(tc_direct + tc_pair)
    delta_u = -(delta_direct + delta_pair)
    return {
        "k1_direct": k1_direct,
        "k1_pair": k1_pair,
        "k2_direct": k2_direct,
        "k2_pair": k2_pair,
        "k3_direct": k3_direct,
        "k3_pair": k3_pair,
        "d_direct": d_direct,
        "d_pair": d_pair,
        "x_direct": x_direct,
        "x_pair": x_pair,
        "tc_direct": tc_direct,
        "tc_pair": tc_pair,
        "delta_direct": delta_direct,
        "delta_pair": delta_pair,
        "tc": tc,
        "delta_u": delta_u,
        "final": tc + delta_u,
    }


def _relative_l2(actual, reference):
    return float(jnp.linalg.norm(actual - reference) / jnp.linalg.norm(reference))


def _nested_jaxprs(value):
    """Yield nested JAXPRs, including the body of a staged jit call."""

    if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        yield value.jaxpr
    elif hasattr(value, "eqns"):
        yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _nested_jaxprs(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nested_jaxprs(item)


def _jaxpr_output_shapes(jaxpr):
    shapes = []
    for equation in jaxpr.eqns:
        for var in equation.outvars:
            aval = getattr(var, "aval", None)
            shape = getattr(aval, "shape", None)
            if shape is not None:
                shapes.append(tuple(shape))
        for nested in _nested_jaxprs(equation.params):
            shapes.extend(_jaxpr_output_shapes(nested))
    return shapes


class TestFactorDirectRandomFP64(unittest.TestCase):
    """Every unsymmetrized and pair-swapped Phase-A branch is exact in FP64."""

    def setUp(self):
        self.data = _random_inputs()
        self.reference = _dense_reference_terms(self.data)
        self.actual = factor_direct.contract_isdf_factor_direct_terms_t2(
            **self.data,
            occupied_pair_batch_size=2,
            rank_panel_size=3,
        )
        jax.block_until_ready(tuple(self.actual.values()))

    def test_every_term_and_pair_swapped_term_matches_dense_tile_path(self):
        self.assertEqual(set(self.reference), set(self.actual))
        for name, expected in self.reference.items():
            with self.subTest(term=name):
                rel_l2 = _relative_l2(self.actual[name], expected)
                self.assertLessEqual(rel_l2, 1e-11, f"{name}: relative L2={rel_l2:.3e}")
                np.testing.assert_allclose(
                    np.asarray(self.actual[name]), np.asarray(expected),
                    rtol=1e-11, atol=1e-11,
                )

    def test_final_residual_has_rccsd_pair_symmetry(self):
        final = self.actual["final"]
        np.testing.assert_allclose(
            np.asarray(final),
            np.asarray(final.transpose(1, 0, 3, 2)),
            rtol=1e-11,
            atol=1e-11,
        )

    def test_profile_records_per_term_wall_and_compiled_memory_accounting(self):
        terms, profiles = factor_direct.profile_isdf_factor_direct_terms_t2(
            **self.data,
            occupied_pair_batch_size=2,
            rank_panel_size=3,
        )
        self.assertEqual(
            set(profiles),
            {
                "k1_direct", "k1_pair", "k2_direct", "k2_pair",
                "k3_direct", "k3_pair", "d_direct", "d_pair",
                "x_direct", "x_pair",
            },
        )
        self.assertIn("final", terms)
        for name, profile in profiles.items():
            with self.subTest(term=name):
                self.assertGreaterEqual(profile.wall_seconds, 0.0)
                self.assertGreater(profile.schedule_intermediate_estimate_bytes, 0)
                self.assertGreater(profile.compiled_xla_temporary_bytes, 0)
                self.assertGreater(profile.compiled_xla_argument_bytes, 0)
                self.assertGreater(profile.compiled_xla_output_bytes, 0)
                self.assertEqual(
                    profile.compiled_xla_total_bytes,
                    profile.compiled_xla_argument_bytes
                    + profile.compiled_xla_output_bytes
                    + profile.compiled_xla_temporary_bytes
                    - profile.compiled_xla_alias_bytes,
                )
                self.assertFalse(profile.materializes_v4)
                self.assertEqual(profile.occupied_pair_batch_size, 2)
                self.assertEqual(profile.rank_panel_size, 3)

    def test_profiles_use_the_actual_compiled_full_and_x_executables(self):
        _, profiles = factor_direct.profile_isdf_factor_direct_terms_t2(
            **self.data,
            occupied_pair_batch_size=2,
            rank_panel_size=3,
        )
        full = factor_direct.compiled_full_thc_memory(
            self.data["t2"], self.data["p"], self.data["p"], self.data["u3"],
            self.data["p"], self.data["p"],
            occupied_pair_batch_size=2,
            rank_panel_size=3,
        )
        x_left = factor_direct.compiled_partial_x_left_memory(
            self.data["t2"], self.data["p"], self.data["p"], self.data["x"],
            occupied_pair_batch_size=2,
            rank_panel_size=3,
        )
        k1_direct = factor_direct._compiled_k12_memory(
            factor_direct._contract_k1_direct_t2_jit,
            self.data["t2"], self.data["p"], self.data["grad_p"], self.data["u1"],
            occupied_pair_batch_size=2,
            rank_panel_size=3,
        )
        self.assertEqual(profiles["k3_direct"].compiled_xla_temporary_bytes, full.temporary_bytes)
        self.assertEqual(profiles["k3_direct"].compiled_xla_argument_bytes, full.argument_bytes)
        self.assertEqual(profiles["k3_direct"].compiled_xla_output_bytes, full.output_bytes)
        self.assertEqual(profiles["x_direct"].compiled_xla_temporary_bytes, x_left.temporary_bytes)
        self.assertEqual(profiles["x_direct"].compiled_xla_argument_bytes, x_left.argument_bytes)
        self.assertEqual(profiles["k1_direct"].compiled_xla_temporary_bytes, k1_direct.temporary_bytes)
        self.assertEqual(profiles["k1_direct"].compiled_xla_total_bytes, k1_direct.total_bytes)
        self.assertGreater(
            profiles["k1_direct"].compiled_xla_total_bytes,
            profiles["k3_direct"].compiled_xla_total_bytes,
        )

    def test_staged_jaxprs_never_create_a_virtual_four_index_tile(self):
        data = self.data
        nvir = data["p"].shape[0]
        v4_shape = (nvir, nvir, nvir, nvir)
        calls = (
            lambda t2: factor_direct.contract_full_thc_t2(
                t2, data["p"], data["p"], data["u3"], data["p"], data["p"],
                occupied_pair_batch_size=2, rank_panel_size=3,
            ),
            lambda t2: factor_direct.contract_partial_x_left_t2(
                t2, data["p"], data["p"], data["x"],
                occupied_pair_batch_size=2, rank_panel_size=3,
            ),
            lambda t2: factor_direct.contract_partial_x_right_t2(
                t2, data["p"], data["p"], data["x"],
                occupied_pair_batch_size=2, rank_panel_size=3,
            ),
        )
        for call in calls:
            with self.subTest(call=call):
                closed = jax.make_jaxpr(call)(data["t2"])
                self.assertNotIn(v4_shape, _jaxpr_output_shapes(closed.jaxpr))


@unittest.skipUnless(
    os.environ.get("PYTC_RUN_FACTOR_DIRECT_H10") == "1",
    "set PYTC_RUN_FACTOR_DIRECT_H10=1 to run the physical H10 Phase-A gate",
)
class TestFactorDirectPhysicalH10(unittest.TestCase):
    """Opt-in physical deck used by the H10 GPU validation run card.

    This is intentionally not ordinary CI: it builds the actual current ISDF
    factors, compares every rebracketed branch to a dense virtual tile on the
    small H10/STO-3G deck, and independently checks the final factor-direct
    residual against ``ISDFXTC._assemble_2b_tile``.
    """

    @classmethod
    def setUpClass(cls):
        from pyscf import gto, scf

        from pytc.jastrow.rexp import REXP
        from pytc.xtc import ISDFXTC, XTC

        atoms = "; ".join(f"H 0 0 {1.4 * atom}" for atom in range(10))
        mol = gto.M(atom=atoms, basis="sto-3g", unit="Bohr", verbose=0)
        mf = scf.RHF(mol).run()
        jparams = {"alpha": jnp.array([0.5], dtype=jnp.float64)}
        xtc = XTC.from_pyscf(mf, REXP(), grid_lvl=0)
        base = ISDFXTC.from_xtc(
            xtc, n_rank=max(16, 2 * xtc.n_orb), is_incore=True,
        )
        cls.isdf_xtc = base.isdf(
            jparams, batch_size=256, orb_block_size=4,
            host_grid_block_size=1024,
        )
        cls.jparams = jparams
        cls.nocc = mol.nelec[0]

    def test_h10_current_isdf_factors_and_current_tile_path(self):
        nmo = self.isdf_xtc.n_orb
        virtual = slice(self.nocc, nmo)
        nvir = nmo - self.nocc
        rank = self.isdf_xtc.phi_isdf.shape[1]
        data = _random_inputs(nocc=self.nocc, nvir=nvir, rank=rank)
        kernels = self.isdf_xtc.isdf_kernels
        data.update(
            p=jnp.asarray(self.isdf_xtc.phi_isdf[virtual]),
            grad_p=jnp.asarray(self.isdf_xtc.grad_phi_isdf[virtual]),
            u1=jnp.asarray(kernels["K1_kernel"]),
            u3=jnp.asarray(kernels["K3_kernel"]),
            d=jnp.asarray(kernels["D"]),
            x=jnp.asarray(kernels["X"])[virtual, virtual],
        )
        actual = factor_direct.contract_isdf_factor_direct_terms_t2(
            **data, occupied_pair_batch_size=2, rank_panel_size=8,
        )
        reference = _dense_reference_terms(data)
        for name, expected in reference.items():
            with self.subTest(term=name):
                self.assertLessEqual(_relative_l2(actual[name], expected), 1e-10)

        current_tile = self.isdf_xtc._assemble_2b_tile(
            self.jparams,
            kernels,
            (virtual, virtual, virtual, virtual),
        )
        current_residual = _tile_t2(jnp.asarray(current_tile), data["t2"])
        np.testing.assert_allclose(
            np.asarray(actual["final"]), np.asarray(current_residual),
            rtol=1e-10, atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(actual["final"]),
            np.asarray(actual["final"].transpose(1, 0, 3, 2)),
            rtol=1e-10, atol=1e-10,
        )

        _, profiles = factor_direct.profile_isdf_factor_direct_terms_t2(
            **data, occupied_pair_batch_size=2, rank_panel_size=8,
        )
        for name, profile in profiles.items():
            self.assertFalse(profile.materializes_v4)
            self.assertGreater(profile.schedule_intermediate_estimate_bytes, 0)
            self.assertGreater(profile.compiled_xla_temporary_bytes, 0)
            print(
                "FACTOR_DIRECT_TERM " + json.dumps(
                    {
                        "term": name,
                        "wall_seconds": profile.wall_seconds,
                        "schedule_intermediate_estimate_bytes": profile.schedule_intermediate_estimate_bytes,
                        "compiled_xla_temporary_bytes": profile.compiled_xla_temporary_bytes,
                        "compiled_xla_argument_bytes": profile.compiled_xla_argument_bytes,
                        "compiled_xla_output_bytes": profile.compiled_xla_output_bytes,
                        "compiled_xla_alias_bytes": profile.compiled_xla_alias_bytes,
                        "compiled_xla_total_bytes": profile.compiled_xla_total_bytes,
                        "rank_panel_size": profile.rank_panel_size,
                        "occupied_pair_batch_size": profile.occupied_pair_batch_size,
                        "materializes_v4": profile.materializes_v4,
                    },
                    sort_keys=True,
                )
            )


class TestStreamedXParity(unittest.TestCase):
    """Streamed-X contraction matches the full-block lift term for term.

    The streamed path is the only route whose device X residency is one rank
    panel, so it is the 1200-orbital path; these gates pin it to the
    reviewed full-block math at FP64 reassociation level.  Panel boundaries
    are the adversarial cases: rank 7 is exercised with panel sizes 2 and 3
    (ragged tails) and 7 (single panel, the full-lift degenerate).
    """

    def setUp(self):
        self.data = _random_inputs()          # nocc=2, nvir=5, rank=7
        self.nocc, self.nvir, self.rank = 2, 5, 7
        nmo = self.nocc + self.nvir
        backing = np.zeros((nmo, nmo, self.rank), dtype=np.float64)
        backing[self.nocc:, self.nocc:, :] = np.asarray(self.data["x"])
        self.backing = backing

    def test_left_and_right_streamed_match_full_block(self):
        for panel in (2, 3, 7):
            with self.subTest(rank_panel_size=panel):
                left_full = factor_direct.contract_partial_x_left_t2(
                    self.data["t2"], self.data["p"], self.data["p"], self.data["x"],
                    occupied_pair_batch_size=2, rank_panel_size=panel)
                left_stream = factor_direct.contract_partial_x_left_t2_streamed(
                    self.data["t2"], self.data["p"], self.data["p"],
                    self.backing, self.nocc,
                    occupied_pair_batch_size=2, rank_panel_size=panel)
                self.assertLessEqual(_relative_l2(left_stream, left_full), 1e-12)

                right_full = factor_direct.contract_partial_x_right_t2(
                    self.data["t2"], self.data["p"], self.data["p"], self.data["x"],
                    occupied_pair_batch_size=2, rank_panel_size=panel)
                right_stream = factor_direct.contract_partial_x_right_t2_streamed(
                    self.data["t2"], self.data["p"], self.data["p"],
                    self.backing, self.nocc,
                    occupied_pair_batch_size=2, rank_panel_size=panel)
                self.assertLessEqual(_relative_l2(right_stream, right_full), 1e-12)

    def test_streamed_reads_hdf5_backing(self):
        import h5py
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x_store.h5")
            with h5py.File(path, "w") as fh:
                fh.create_dataset("X", data=self.backing, dtype="f8")
            with h5py.File(path, "r") as fh:
                streamed = factor_direct.contract_partial_x_left_t2_streamed(
                    self.data["t2"], self.data["p"], self.data["p"],
                    fh["X"], self.nocc,
                    occupied_pair_batch_size=2, rank_panel_size=3)
        in_memory = factor_direct.contract_partial_x_left_t2_streamed(
            self.data["t2"], self.data["p"], self.data["p"],
            self.backing, self.nocc,
            occupied_pair_batch_size=2, rank_panel_size=3)
        self.assertLessEqual(_relative_l2(streamed, in_memory), 1e-12)

    def test_dispatcher_xstream_matches_full_dispatcher(self):
        full = factor_direct.contract_isdf_factor_direct_terms_t2(
            **self.data, occupied_pair_batch_size=2, rank_panel_size=3)
        streamed = factor_direct.contract_isdf_factor_direct_terms_t2_xstream(
            self.data["t2"], self.data["p"], self.data["grad_p"],
            self.data["u1"], self.data["u3"], self.data["d"],
            self.backing, self.nocc,
            occupied_pair_batch_size=2, rank_panel_size=3)
        for name in ("x_direct", "x_pair", "delta_direct", "delta_pair",
                     "tc", "delta_u", "final"):
            with self.subTest(term=name):
                self.assertLessEqual(
                    _relative_l2(streamed[name], full[name]), 1e-12)

    def test_panel_reader_pads_and_bounds(self):
        # The panel reader is the only place host data enters the streamed
        # path: every read must be exactly panel-shaped, and a ragged tail
        # must be zero-padded, never shorter.
        rank = self.rank
        for m0, expected_width in ((0, 3), (3, 3), (6, 1)):
            panel = factor_direct._read_x_rank_panel(
                self.backing, self.nocc, m0, min(m0 + 3, rank), 3)
            self.assertEqual(panel.shape, (self.nvir, self.nvir, 3))
            np.testing.assert_array_equal(
                panel[:, :, :expected_width],
                self.backing[self.nocc:, self.nocc:, m0:m0 + expected_width])
            if expected_width < 3:
                np.testing.assert_array_equal(
                    panel[:, :, expected_width:],
                    np.zeros((self.nvir, self.nvir, 3 - expected_width)))


class TestTieredXAuto(unittest.TestCase):
    """Three-tier auto dispatcher: forced tiers match the full-lift math.

    Tiers 2/3 regroup the rank reduction (working-set panels instead of the
    full-lift compiled rank scan), so agreement with the full-lift reference
    is at FP64 reassociation level (relative L2 <= 1e-12), not bitwise.
    The env pins are read at call time inside the functions, so each test
    sets and restores them.
    """

    _ENV_PINS = ("PYTC_X_FORCE_TIER", "PYTC_X_PANEL_BUDGET_GB")

    def setUp(self):
        self.data = _random_inputs()          # nocc=2, nvir=5, rank=7
        self.nocc, self.nvir, self.rank = 2, 5, 7
        nmo = self.nocc + self.nvir
        backing = np.zeros((nmo, nmo, self.rank), dtype=np.float64)
        backing[self.nocc:, self.nocc:, :] = np.asarray(self.data["x"])
        self.backing = backing
        self.reference = factor_direct.contract_isdf_factor_direct_terms_t2(
            **self.data, occupied_pair_batch_size=2, rank_panel_size=3)
        self._saved_env = {name: os.environ.get(name) for name in self._ENV_PINS}

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _run_auto(self, tier):
        # The panel budget pins a width of 6 rank columns (6 * 5 * 5 * 8 =
        # 1200 bytes), so the pipelined loop runs two panels over rank 7:
        # one full and one ragged, at a width != rank_panel_size=3.
        os.environ["PYTC_X_FORCE_TIER"] = str(tier)
        os.environ["PYTC_X_PANEL_BUDGET_GB"] = str(1200 / 1024 ** 3)
        counters_before = dict(_tile_timers._STATE["counters"])
        terms = factor_direct.contract_isdf_factor_direct_terms_t2_auto(
            self.data["t2"], self.data["p"], self.data["grad_p"],
            self.data["u1"], self.data["u3"], self.data["d"],
            self.backing, self.nocc,
            occupied_pair_batch_size=2, rank_panel_size=3)
        counters_after = _tile_timers._STATE["counters"]
        fired = {
            name: counters_after.get(name, 0) - counters_before.get(name, 0)
            for name in counters_after
        }
        return terms, fired

    def _assert_terms_match(self, actual, expected, tol=1e-12):
        self.assertEqual(set(expected), set(actual))
        for name, ref in expected.items():
            with self.subTest(term=name):
                self.assertLessEqual(_relative_l2(actual[name], ref), tol)

    def test_forced_tier2_host_resident_matches_full_lift(self):
        terms, fired = self._run_auto(2)
        self.assertEqual(fired.get("fd_x_tier2_host_resident", 0), 1)
        self._assert_terms_match(terms, self.reference)

    def test_forced_tier3_stream_matches_full_lift(self):
        terms, fired = self._run_auto(3)
        self.assertEqual(fired.get("fd_x_tier3_stream", 0), 1)
        self._assert_terms_match(terms, self.reference)

    def test_forced_tier2_matches_tier3(self):
        tier2, _ = self._run_auto(2)
        tier3, _ = self._run_auto(3)
        self._assert_terms_match(tier2, tier3)

    def test_pipelined_wrappers_bitwise_match_streamed_at_same_panel_width(self):
        # A tiny budget forces panel_size == rank_panel_size, so the
        # pipelined loop groups the rank reduction exactly like the streamed
        # path and must reproduce it bitwise.
        os.environ["PYTC_X_PANEL_BUDGET_GB"] = str(8 / 1024 ** 3)
        left_stream = factor_direct.contract_partial_x_left_t2_streamed(
            self.data["t2"], self.data["p"], self.data["p"],
            self.backing, self.nocc,
            occupied_pair_batch_size=2, rank_panel_size=3)
        left_pipe = factor_direct.contract_partial_x_left_t2_pipelined(
            self.data["t2"], self.data["p"], self.data["p"],
            self.backing, self.nocc,
            occupied_pair_batch_size=2, rank_panel_size=3)
        self.assertEqual(_relative_l2(left_pipe, left_stream), 0.0)
        right_stream = factor_direct.contract_partial_x_right_t2_streamed(
            self.data["t2"], self.data["p"], self.data["p"],
            self.backing, self.nocc,
            occupied_pair_batch_size=2, rank_panel_size=3)
        right_pipe = factor_direct.contract_partial_x_right_t2_pipelined(
            self.data["t2"], self.data["p"], self.data["p"],
            self.backing, self.nocc,
            occupied_pair_batch_size=2, rank_panel_size=3)
        self.assertEqual(_relative_l2(right_pipe, right_stream), 0.0)

    def test_pipelined_retries_at_half_panel_on_device_oom(self):
        # Regression cover for the v4 1200 run (JID 20609700): the measured
        # panel budget cannot see BFC fragmentation, so a prefetch
        # ``device_put`` can still OOM; the loop must restart at half the
        # panel width and produce the same result.
        from unittest import mock
        os.environ["PYTC_X_PANEL_BUDGET_GB"] = str(1200 / 1024 ** 3)  # width 6
        real_device_put = jax.device_put
        calls = {"n": 0}

        def fail_first(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise jax.errors.JaxRuntimeError(
                    "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1B.")
            return real_device_put(*args, **kwargs)

        with mock.patch.object(jax, "device_put", side_effect=fail_first):
            result = factor_direct.contract_partial_x_left_t2_pipelined(
                self.data["t2"], self.data["p"], self.data["p"],
                self.backing, self.nocc,
                occupied_pair_batch_size=2, rank_panel_size=3)
        reference = factor_direct.contract_partial_x_left_t2(
            self.data["t2"], self.data["p"], self.data["p"], self.data["x"],
            occupied_pair_batch_size=2, rank_panel_size=3)
        self.assertLessEqual(_relative_l2(result, reference), 1e-12)
        self.assertGreaterEqual(calls["n"], 2)  # the retry really re-ran

    def test_pipelined_reraises_non_oom_jax_error(self):
        # Only device-OOM errors trigger the halving retry; anything else
        # must propagate unchanged.
        from unittest import mock
        os.environ["PYTC_X_PANEL_BUDGET_GB"] = str(1200 / 1024 ** 3)

        def fail_always(*args, **kwargs):
            raise jax.errors.JaxRuntimeError("INTERNAL: something else broke")

        with mock.patch.object(jax, "device_put", side_effect=fail_always):
            with self.assertRaises(jax.errors.JaxRuntimeError):
                factor_direct.contract_partial_x_left_t2_pipelined(
                    self.data["t2"], self.data["p"], self.data["p"],
                    self.backing, self.nocc,
                    occupied_pair_batch_size=2, rank_panel_size=3)
    """The tier gate's memory measurements, including their fallback paths.

    Regression cover for the 1200 validation OOM: the gate measured node
    RAM (psutil) while the job's real cap was the SLURM cgroup ``--mem``
    limit, and a device stats dict without ``bytes_available`` forced the
    minimum panel width on a GPU job.
    """

    def test_cgroup_v2_parsing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "memory.max"), "w") as fh:
                fh.write("1000\n")
            with open(os.path.join(root, "memory.current"), "w") as fh:
                fh.write("250\n")
            self.assertEqual(
                factor_direct._cgroup_memory_available_bytes(root), 750)

    def test_cgroup_v2_unlimited_falls_through_to_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "memory.max"), "w") as fh:
                fh.write("max\n")
            with open(os.path.join(root, "memory.current"), "w") as fh:
                fh.write("250\n")
            self.assertIsNone(factor_direct._cgroup_memory_available_bytes(root))

    def test_cgroup_v1_parsing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "memory"))
            with open(os.path.join(root, "memory", "memory.limit_in_bytes"), "w") as fh:
                fh.write("2048\n")
            with open(os.path.join(root, "memory", "memory.usage_in_bytes"), "w") as fh:
                fh.write("48\n")
            self.assertEqual(
                factor_direct._cgroup_memory_available_bytes(root), 2000)

    def test_cgroup_missing_returns_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(factor_direct._cgroup_memory_available_bytes(root))

    def _make_v2_tree(self, root, levels):
        # levels: dict of relative cgroup dir -> (memory.max, memory.current)
        for rel, (max_v, cur_v) in levels.items():
            d = os.path.join(root, rel) if rel else root
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "memory.max"), "w") as fh:
                fh.write(f"{max_v}\n")
            with open(os.path.join(d, "memory.current"), "w") as fh:
                fh.write(f"{cur_v}\n")

    def _make_proc_cgroup(self, path, rel, controllers=""):
        with open(path, "w") as fh:
            fh.write(f"0::{controllers}/{rel}\n" if controllers else f"0::/{rel}\n")

    def test_job_cgroup_discovered_via_proc_self_cgroup(self):
        # The SLURM layout from the bouchet diagnostic: the limit lives at
        # the "user" level ONE UP from the leaf /proc/self/cgroup path.
        import tempfile
        with tempfile.TemporaryDirectory() as root, \
                tempfile.NamedTemporaryFile("w", delete=False) as proc:
            rel = "system.slice/slurmstepd.scope/job_1/step_0/user/task_0"
            parent = os.path.dirname(rel)
            self._make_v2_tree(root, {
                rel: ("max", 1000),
                parent: (274877906944, 1867776),   # the 256 GiB SLURM cap
                os.path.dirname(parent): ("max", 1000),
            })
            self._make_proc_cgroup(proc.name, rel)
            self.assertEqual(
                factor_direct._cgroup_memory_available_bytes(
                    root, proc_cgroup=proc.name),
                274877906944 - 1867776)

    def test_most_restrictive_ancestor_wins(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root, \
                tempfile.NamedTemporaryFile("w", delete=False) as proc:
            rel = "a/b/c"
            self._make_v2_tree(root, {
                rel: (10_000, 0),
                "a/b": (5_000, 1_000),   # tighter: 4000 remaining
                "a": (8_000, 0),
            })
            self._make_proc_cgroup(proc.name, rel)
            self.assertEqual(
                factor_direct._cgroup_memory_available_bytes(
                    root, proc_cgroup=proc.name), 4_000)

    def test_v1_sentinel_treated_as_unlimited(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root, \
                tempfile.NamedTemporaryFile("w", delete=False) as proc:
            d = os.path.join(root, "memory", "job")
            os.makedirs(d)
            sentinel = 9223372036854771712  # v1 "no limit" value
            with open(os.path.join(d, "memory.limit_in_bytes"), "w") as fh:
                fh.write(f"{sentinel}\n")
            with open(os.path.join(d, "memory.usage_in_bytes"), "w") as fh:
                fh.write("1000\n")
            with open(proc.name, "w") as fh:
                fh.write("2:memory:/job\n")
            self.assertIsNone(
                factor_direct._cgroup_memory_available_bytes(
                    root, proc_cgroup=proc.name))

    def test_free_host_capped_by_cgroup(self):
        # With a cgroup tighter than node RAM, the host measurement must
        # report the cgroup remainder (the tier-2 lift is killed by the
        # cgroup OOM killer, not by node exhaustion).  psutil and the
        # cgroup probe are stubbed with fixed values: the live
        # node-available figure drifts between calls and made this test
        # flaky.
        import sys
        import types
        import unittest.mock as mock
        fake_psutil = types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(available=10_000))
        real_cgroup = factor_direct._cgroup_memory_available_bytes
        try:
            with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
                factor_direct._cgroup_memory_available_bytes = lambda **_: 4_000
                self.assertEqual(factor_direct._measure_free_host_bytes(), 4_000)
                factor_direct._cgroup_memory_available_bytes = lambda **_: 50_000
                self.assertEqual(factor_direct._measure_free_host_bytes(), 10_000)
                factor_direct._cgroup_memory_available_bytes = lambda **_: None
                self.assertEqual(factor_direct._measure_free_host_bytes(), 10_000)
        finally:
            factor_direct._cgroup_memory_available_bytes = real_cgroup

    def test_device_stats_fallback_keys(self):
        # bytes_available missing but limit/in_use present -> difference.
        class _FakeDevice:
            def __init__(self, stats):
                self._stats = stats
            def memory_stats(self):
                return self._stats
        real_local_devices = jax.local_devices
        try:
            jax.local_devices = lambda: [_FakeDevice(
                {"bytes_limit": 1000, "bytes_in_use": 300})]
            self.assertEqual(factor_direct._measure_free_device_bytes(), 700)
            jax.local_devices = lambda: [_FakeDevice({"some_other_key": 1})]
            self.assertIsNone(factor_direct._measure_free_device_bytes())
            jax.local_devices = lambda: [_FakeDevice(None)]
            self.assertIsNone(factor_direct._measure_free_device_bytes())
        finally:
            jax.local_devices = real_local_devices
