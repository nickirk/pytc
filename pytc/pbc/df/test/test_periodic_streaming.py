"""Tests for pytc.pbc.df.isdf.stream_ao_blocks (S1) and
build_periodic_pivot_oracle (S2), design v2.1 sections 3/6."""

import unittest

import numpy as np
from pyscf.pbc.gto import Cell

from pytc.pbc.df.isdf import (
    build_periodic_pivot_oracle,
    pivoted_cholesky_hermitian,
    stream_ao_blocks,
)


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


def _dense_reference_metric(cell, kpts, grid_coords):
    kpts_list = list(np.asarray(kpts, dtype=np.float64))
    ao = np.asarray(cell.pbc_eval_gto("GTOval", grid_coords, kpts=kpts_list), dtype=np.complex128)
    n_kpts = ao.shape[0]
    pooled = ao.transpose(1, 0, 2).reshape(ao.shape[1], -1)  # (Ng, Nk*Nao)
    gram = pooled.conj() @ pooled.T  # (Ng, Ng)
    return (np.abs(gram) ** 2) / n_kpts


class TestStreamAoBlocks(unittest.TestCase):
    def test_blocks_concatenate_to_single_shot_evaluation(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        grid_coords = cell.get_uniform_grids(cell.mesh)[:37]

        blocks = list(stream_ao_blocks(cell, kpts, grid_coords, block_size=10))
        reassembled = np.concatenate([blk for _, _, blk in blocks], axis=1)

        expected = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords, kpts=list(kpts)), dtype=np.complex128
        )
        np.testing.assert_allclose(reassembled, expected, atol=1e-12)

    def test_block_bounds_cover_grid_exactly_once(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 1], wrap_around=False)
        grid_coords = cell.get_uniform_grids(cell.mesh)[:23]

        bounds = [(g0, g1) for g0, g1, _ in stream_ao_blocks(cell, kpts, grid_coords, block_size=7)]
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 23)
        for (_, g1_a), (g0_b, _) in zip(bounds, bounds[1:]):
            self.assertEqual(g1_a, g0_b)

    def test_rejects_malformed_inputs(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 1], wrap_around=False)
        grid_coords = cell.get_uniform_grids(cell.mesh)[:10]
        with self.assertRaises(ValueError):
            list(stream_ao_blocks(cell, kpts, grid_coords[:, :2], block_size=5))
        with self.assertRaises(ValueError):
            list(stream_ao_blocks(cell, kpts, grid_coords, block_size=0))
        with self.assertRaises(ValueError):
            list(stream_ao_blocks(cell, kpts, grid_coords, block_size=-3))


class TestBuildPeriodicPivotOracle(unittest.TestCase):
    def test_diag_matches_dense_reference(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        grid_coords = cell.get_uniform_grids(cell.mesh)[:29]
        diag, _ = build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size=8)
        M = _dense_reference_metric(cell, kpts, grid_coords)
        np.testing.assert_allclose(diag, np.diag(M), atol=1e-10)

    def test_col_eval_matches_dense_reference(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        grid_coords = cell.get_uniform_grids(cell.mesh)[:29]
        _, col_eval = build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size=8)
        M = _dense_reference_metric(cell, kpts, grid_coords)
        for j in (0, 5, 17, 28):
            np.testing.assert_allclose(np.asarray(col_eval(j)).real, M[:, j], atol=1e-10, err_msg=f"j={j}")

    def test_diag_and_col_eval_agree_at_the_diagonal(self):
        # Two independent code paths (one streamed sweep vs a per-column
        # sweep) must agree on M[j,j] exactly.
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 3], wrap_around=False)
        grid_coords = cell.get_uniform_grids(cell.mesh)[:19]
        diag, col_eval = build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size=6)
        for j in (0, 4, 11, 18):
            self.assertAlmostEqual(diag[j], np.asarray(col_eval(j))[j].real, places=8, msg=f"j={j}")

    def test_gamma_only_matches_molecular_pair_density_gram(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 1], wrap_around=False)
        grid_coords = cell.get_uniform_grids(cell.mesh)[:15]
        diag, col_eval = build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size=5)

        ao = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords, kpts=list(kpts)), dtype=np.complex128
        )[0]  # (Ng, Nao), single k-point
        gram_molecular = ao.conj() @ ao.T
        expected = np.abs(gram_molecular) ** 2  # Nk=1, so /Nk is a no-op

        np.testing.assert_allclose(diag, np.diag(expected), atol=1e-10)
        np.testing.assert_allclose(np.asarray(col_eval(3)).real, expected[:, 3], atol=1e-10)

    def test_pivoted_cholesky_hermitian_reconstructs_dense_reference(self):
        cell = _make_cell()
        kpts = cell.make_kpts([1, 1, 2], wrap_around=False)
        grid_coords = cell.get_uniform_grids(cell.mesh)[:25]
        diag, col_eval = build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size=9)
        M = _dense_reference_metric(cell, kpts, grid_coords)

        rank = np.linalg.matrix_rank(M, tol=1e-8)
        pivots, L, n_selected = pivoted_cholesky_hermitian(diag, col_eval, rank=min(rank + 2, M.shape[0]))
        recon = L @ L.conj().T
        np.testing.assert_allclose(recon, M, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
