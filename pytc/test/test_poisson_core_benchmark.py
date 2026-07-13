"""Unit tests for the pure protocol helpers in the P2c benchmark harness."""

import unittest

import numpy as np
from pyscf import gto

from pytc.utils.poisson_core_benchmark import (
    BenchmarkCase,
    centered_uniform_mesh,
    matrix_case,
    mp2_energy_from_ovov,
    recommended_matrix,
)


class TestPoissonCoreBenchmarkProtocol(unittest.TestCase):
    def test_centered_mesh_is_odd_contains_margin_and_is_deterministic(self):
        mol = gto.M(atom="H 0 0 -0.7; H 0 0 0.7", basis="sto-3g", verbose=0)
        shape, origin, coords = centered_uniform_mesh(mol, spacing=0.3, margin=2.0)
        self.assertTrue(all(n % 2 == 1 for n in shape))
        self.assertEqual(coords.shape, (np.prod(shape), 3))
        atom_coords = mol.atom_coords(unit="Bohr")
        grid_lo = np.asarray(origin)
        grid_hi = grid_lo + (np.asarray(shape) - 1) * 0.3
        self.assertTrue(np.all(grid_lo <= atom_coords.min(axis=0) - 2.0 + 1e-12))
        self.assertTrue(np.all(grid_hi >= atom_coords.max(axis=0) + 2.0 - 1e-12))
        shape2, origin2, coords2 = centered_uniform_mesh(mol, spacing=0.3, margin=2.0)
        self.assertEqual(shape, shape2)
        np.testing.assert_array_equal(origin, origin2)
        np.testing.assert_array_equal(coords, coords2)

    def test_mp2_formula_zero_eri_is_zero(self):
        eri = np.zeros((1, 2, 1, 2))
        mo_energy = np.array([-1.0, 0.2, 0.4])
        self.assertEqual(mp2_energy_from_ovov(eri, mo_energy, n_occ=1), 0.0)

    def test_case_rejects_invalid_geometry_and_backend(self):
        with self.assertRaises(ValueError):
            BenchmarkCase(spacing=0.0)
        with self.assertRaises(ValueError):
            BenchmarkCase(pad_factor=1)
        with self.assertRaises(ValueError):
            BenchmarkCase(backend="cuda")

    def test_recommended_matrix_is_unique_and_has_all_sweep_axes(self):
        matrix = recommended_matrix("jax")
        encoded = {tuple(sorted(case.items())) for case in matrix}
        self.assertEqual(len(encoded), len(matrix))
        self.assertEqual({case["system"] for case in matrix}, {"H2O_ccpVDZ", "benzene_ccpVDZ"})
        self.assertTrue(all(case["backend"] == "jax" for case in matrix))
        self.assertIn(3, {case["pad_factor"] for case in matrix})
        self.assertIn(0.25, {case["spacing"] for case in matrix if case["system"] == "H2O_ccpVDZ"})
        self.assertIn(6.0, {case["rank_factor"] for case in matrix if case["system"] == "benzene_ccpVDZ"})

    def test_matrix_case_resolves_and_bounds_checks_job_array_index(self):
        matrix = recommended_matrix("jax")
        self.assertEqual(matrix_case(0, "jax"), BenchmarkCase(**matrix[0]))
        self.assertEqual(matrix_case(len(matrix) - 1, "jax"), BenchmarkCase(**matrix[-1]))
        with self.assertRaises(ValueError):
            matrix_case(-1, "jax")
        with self.assertRaises(ValueError):
            matrix_case(len(matrix), "jax")


if __name__ == "__main__":
    unittest.main()
