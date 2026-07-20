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
