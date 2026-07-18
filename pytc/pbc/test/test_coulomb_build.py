"""End-to-end test for pytc.pbc.coulomb.build (design doc §2): S1-S4 on a
tiny real cell vs a from-scratch stage-by-stage reconstruction.
get_k/get_j structural tests live in test_coulomb_get_k_get_j.py."""

import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc import coulomb
from pytc.pbc.df.isdf import (
    RawKernelProvider,
    apply_raw_kernel_and_solve,
    build_periodic_pivot_oracle,
    build_pi_eta,
    pivoted_cholesky_hermitian,
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


class TestBuild(unittest.TestCase):
    def test_translation_mode_matches_complex_cache_end_to_end(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        kwargs = dict(
            rank=3, block_size=13, rtol=1e-8,
            selection_peak_max_bytes=10**9,
        )
        cached = coulomb.build(
            cell, kpts, selection_mode="jax_cached_matrix_free", **kwargs,
        )
        translated = coulomb.build(
            cell, kpts, selection_mode="jax_translation_matrix_free", **kwargs,
        )
        np.testing.assert_array_equal(
            cached["selection_provenance"]["pivot_indices"],
            translated["selection_provenance"]["pivot_indices"],
        )
        self.assertEqual(
            2 * translated["selection_provenance"]["cache_bytes"],
            cached["selection_provenance"]["cache_bytes"],
        )
        self.assertEqual(
            translated["selection_provenance"]["eta_ao_source"],
            "blocked_reconstruction_from_translation_classes",
        )
        provenance = translated["selection_provenance"]
        self.assertEqual(
            provenance["translation_reconstruction_calls"],
            provenance["translation_reconstruction_calls_selection"]
            + int(np.ceil(cell.get_uniform_grids(cell.mesh).shape[0] / kwargs["block_size"])),
        )
        self.assertEqual(
            provenance["translation_reconstruction_grid_points"],
            provenance["translation_reconstruction_grid_points_selection"]
            + cell.get_uniform_grids(cell.mesh).shape[0],
        )
        for key in ("inpv_kpt", "coul_kpt", "kern_kpt"):
            np.testing.assert_allclose(
                np.asarray(translated[key]), np.asarray(cached[key]), atol=2e-10, rtol=2e-10,
            )

    def test_build_produces_self_consistent_artifact(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        result = coulomb.build(cell, kpts, rank=4, block_size=11, rtol=1e-8)

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
            cell, kpts, rank=3, block_size=13, rtol=1e-8,
            selection_mode="streamed", fixed_pivots=pivots,
        )
        baseline = coulomb.build(
            cell, kpts, rank=3, block_size=13, rtol=1e-8,
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
                cell, kpts, rank=3, block_size=13, rtol=1e-8,
                selection_mode="fixed_pivots",
            )

    def test_build_matches_manual_stage_by_stage_reconstruction(self):
        # Strongest check: reconstruct the SAME artifact by manually
        # driving the individual stage functions (as opposed to
        # build()'s own orchestration) and compare.
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        rank, block_size = 3, 9

        result = coulomb.build(cell, kpts, rank=rank, block_size=block_size, rtol=1e-8)

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
    """Regression for task #47. The ISDF device W-solve silently ran in single
    precision whenever jax_enable_x64 was off: hermitian_sandwich_solve_device
    casts to complex128, but JAX downcasts to complex64 without x64, so the
    eigh/solve loses ~9 digits and trips the 1e-10 machine-tier retained-solve
    gate three stages downstream. It surfaced at the diamond-111 primitive cell
    because that was the first NON-TEST build ever routed through the device
    solve (every test module and accepted production runner enables x64), and
    diamond-111/k222 is a fully self-paired (all-TRIM) mesh -- every q is its
    own negative, so all q's take the Pi_q.real self-paired path. Guards BOTH
    branches: x64 on passes the gate at all self-paired q's; x64 off now fails
    closed at the boundary with the actionable dtype error instead of the opaque
    downstream gate trip."""

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
            selection_mode="streamed",
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
                    cell, kpts, rank=6 * cell.nao_nr(), block_size=64, rtol=1e-4,
                    selection_mode="streamed",
                )
        message = str(ctx.exception)
        self.assertIn("jax_enable_x64", message)
        self.assertIn("complex64", message)


if __name__ == "__main__":
    unittest.main()
