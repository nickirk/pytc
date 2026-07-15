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

if __name__ == "__main__":
    unittest.main()
