"""bpc_blocked_projection wiring from coulomb.build through to the selector.

The numerics are covered in pytc/pbc/df/test/test_isdf_selector.py; the path
from the public argument down to the selector call is what is tested here,
because the option existed and was tested for a week while no production
caller could reach it.
"""

import unittest
from unittest import mock

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc import coulomb


def _make_cell():
    cell = Cell()
    cell.atom = "He 1.0 1.0 1.0"
    cell.a = np.diag([2.0, 2.0, 2.0])
    cell.unit = "A"
    cell.verbose = 0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 40.0
    cell.build()
    return cell


class TestPlanResolution(unittest.TestCase):
    def _plan(self, **kw):
        return coulomb.resolve_build_plan(
            n_grid=512, n_kpts=1, n_ao=4, rank=8, block_size=64,
            provider_cls=coulomb.RawKernelProvider,
            provider_details={"normalization": "unit_test"},
            selection_mode="bpc_cached_gemm", **kw
        ).to_dict()

    def test_defaults_to_on(self):
        # Flipped 2026-08-10 by owner decision. Pivots stay byte-identical
        # (test_isdf_selector); factor values move in the last bits, so a
        # comparison against a result banked before the flip is not exact.
        self.assertTrue(coulomb.FROZEN_BPC_POLICY["bpc_blocked_projection"])
        self.assertTrue(self._plan()["resolved"]["bpc_blocked_projection"])

    def test_can_still_be_turned_off(self):
        # The slow path stays reachable: it is the reference the blocked path
        # was gated against.
        plan = self._plan(bpc_blocked_projection=False)
        self.assertFalse(plan["resolved"]["bpc_blocked_projection"])

    def test_requested_value_survives_into_the_resolved_plan(self):
        plan = self._plan(bpc_blocked_projection=True)
        self.assertTrue(plan["requested"]["bpc_blocked_projection"])
        self.assertTrue(plan["resolved"]["bpc_blocked_projection"])

    def test_rejects_non_bool(self):
        for bad in (1, 0, "yes", None, 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as ctx:
                self._plan(bpc_blocked_projection=bad)
            self.assertIn("bpc_blocked_projection", str(ctx.exception))

    def test_numpy_bool_is_accepted(self):
        self.assertTrue(
            self._plan(bpc_blocked_projection=np.True_)["resolved"][
                "bpc_blocked_projection"]
        )

    def test_absent_for_a_non_bpc_selector(self):
        plan = coulomb.resolve_build_plan(
            n_grid=512, n_kpts=1, n_ao=4, rank=8, block_size=64,
            provider_cls=coulomb.RawKernelProvider,
            provider_details={"normalization": "unit_test"},
            selection_mode="streamed",
        ).to_dict()
        self.assertIsNone(plan["resolved"]["bpc_blocked_projection"])


class TestReachesTheSelector(unittest.TestCase):
    """A resolved plan value that never arrives at the selector is the bug
    this option already had."""

    def _run_and_capture(self, **kw):
        cell = _make_cell()
        real = coulomb.pivoted_cholesky_batched_hermitian
        seen = {}

        def spy(*args, **kwargs):
            seen.update(kwargs)
            result = real(*args, **kwargs)
            seen["returned_pivots"] = result[0]
            return result

        with mock.patch.object(
            coulomb, "pivoted_cholesky_batched_hermitian", spy
        ):
            built = coulomb.build(
                cell, cell.make_kpts([1, 1, 1]),
                rank=2 * cell.nao_nr(), block_size=64,
                selection_mode="bpc_cached_gemm", bpc_n_topup=2,
                bpc_batch_size=4, **kw
            )
        return seen, built

    def test_true_is_forwarded(self):
        seen, _ = self._run_and_capture(bpc_blocked_projection=True)
        self.assertIs(seen["blocked_projection"], True)

    def test_default_forwards_true(self):
        seen, _ = self._run_and_capture()
        self.assertIs(seen["blocked_projection"], True)

    def test_recorded_in_selection_provenance(self):
        _, built = self._run_and_capture(bpc_blocked_projection=True)
        self.assertTrue(
            built["selection_provenance"]["bpc_blocked_projection"]
        )

    def test_same_pivots_either_way_on_a_real_cell(self):
        # The selector-level test asserts this on synthetic input; this is the
        # same claim reached through the public argument.
        off, _ = self._run_and_capture(bpc_blocked_projection=False)
        on, _ = self._run_and_capture(bpc_blocked_projection=True)
        np.testing.assert_array_equal(
            off["returned_pivots"], on["returned_pivots"]
        )


class TestDfAdapterForwards(unittest.TestCase):
    def test_isdfdf_stores_and_forwards(self):
        cell = _make_cell()
        df = coulomb.ISDFDF(
            cell, cell.make_kpts([1, 1, 1]), rank=2 * cell.nao_nr(),
            block_size=64, bpc_blocked_projection=True,
        )
        self.assertTrue(df.bpc_blocked_projection)

        captured = {}

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return {"n_selected": 0}

        with mock.patch.object(coulomb, "build", spy):
            df.build()
        self.assertIs(captured["bpc_blocked_projection"], True)


if __name__ == "__main__":
    unittest.main()
