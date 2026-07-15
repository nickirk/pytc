"""CPU feasibility test for reciprocal primitive-AO translation."""

import time
import unittest

import numpy as np
from pyscf.pbc import gto, tools

from pytc.pbc.df.kpts import canonicalize_kpts
from pytc.pbc.df.reciprocal_ao_pilot import (
    metric_column_from_ao_groups,
    pivot_prefix_from_ao_groups,
    reciprocal_translate_bloch_ao,
    relative_frobenius_error,
)


def _diamond_211():
    primitive = gto.Cell()
    primitive.atom = "C 0.0 0.0 0.0; C 0.8917 0.8917 0.8917"
    primitive.a = """0.0 1.7834 1.7834
1.7834 0.0 1.7834
1.7834 1.7834 0.0"""
    primitive.unit = "A"
    primitive.basis = "gth-dzvp"
    primitive.pseudo = "gth-pbe"
    primitive.ke_cutoff = 30.0
    primitive.verbose = 0
    primitive.build()
    return tools.super_cell(primitive, [2, 1, 1])


class TestReciprocalPrimitiveAOPilot(unittest.TestCase):
    def test_211_direct_shift_reciprocal_convergence_and_selector_prefix(self):
        cell = _diamond_211()
        mesh = np.asarray(cell.mesh)
        coords = cell.get_uniform_grids(mesh)
        kpts = cell.make_kpts([3, 2, 1], wrap_around=False)
        mesh_obj = canonicalize_kpts(cell, kpts)
        selected = np.array([0, 1, 2])
        self.assertEqual(mesh_obj.neg[0], 0)
        self.assertEqual(mesh_obj.neg[1], 1)
        self.assertNotEqual(mesh_obj.neg[2], 2)

        n_primitive_ao = cell.nao_nr() // 2
        translation = cell.atom_coords()[2] - cell.atom_coords()[0]
        selected_kpts = mesh_obj.canonical_kpts[selected]
        direct = np.asarray(cell.pbc_eval_gto(
            "GTOval", coords, kpts=list(selected_kpts),
        ), dtype=np.complex128)
        shifted = np.asarray(cell.pbc_eval_gto(
            "GTOval", coords - translation, kpts=list(selected_kpts),
        ), dtype=np.complex128)
        targets = direct[:, :, n_primitive_ao:]
        seed = direct[:, :, :n_primitive_ao]
        np.testing.assert_allclose(targets, shifted[:, :, :n_primitive_ao], atol=3e-13)

        phase_errors = []
        generated = []
        g_vectors = cell.get_Gv(mesh)
        for k_index, kpt in enumerate(selected_kpts):
            continuous = (
                np.exp(-1j * (kpt @ translation))
                * np.exp(1j * (coords @ kpt))[:, None]
                * np.exp(-1j * ((coords - translation) @ kpt))[:, None]
                * shifted[k_index, :, :n_primitive_ao]
            )
            phase_errors.append(relative_frobenius_error(targets[k_index], continuous))
            generated.append(reciprocal_translate_bloch_ao(
                seed[k_index], coords, mesh, kpt, translation, g_vectors,
            ))
        generated = np.asarray(generated)
        for error in phase_errors:
            self.assertLess(error["max_abs"], 4e-13)
            self.assertLess(error["relative_frobenius"], 4e-13)

        same_mesh_error = relative_frobenius_error(targets, generated)
        self.assertLess(same_mesh_error["relative_frobenius"], 2e-3)

        generic_kpt = selected_kpts[2]
        padded_mesh = 2 * mesh
        padded_coords = cell.get_uniform_grids(padded_mesh)
        padded_direct = np.asarray(cell.pbc_eval_gto(
            "GTOval", padded_coords, kpts=[generic_kpt],
        ), dtype=np.complex128)[0]
        padded_generated = reciprocal_translate_bloch_ao(
            padded_direct[:, :n_primitive_ao], padded_coords, padded_mesh,
            generic_kpt, translation, cell.get_Gv(padded_mesh),
        )
        padded_error = relative_frobenius_error(
            padded_direct[:, n_primitive_ao:], padded_generated,
        )
        self.assertLess(padded_error["relative_frobenius"], 1e-10)
        self.assertLess(
            padded_error["relative_frobenius"],
            same_mesh_error["relative_frobenius"] / 1e6,
        )

        direct_groups = lambda: iter((direct[:, :, :n_primitive_ao], targets))
        generated_groups = lambda: iter((direct[:, :, :n_primitive_ao], generated))
        pivots = (0, len(coords) // 3, len(coords) - 1)
        for pivot in pivots:
            metric_error = relative_frobenius_error(
                metric_column_from_ao_groups(direct_groups(), pivot, len(selected)),
                metric_column_from_ao_groups(generated_groups(), pivot, len(selected)),
            )
            self.assertLess(metric_error["max_abs"], 4e-3)
            self.assertLess(metric_error["relative_frobenius"], 3e-3)

        started = time.perf_counter()
        direct_prefix, _, direct_count = pivot_prefix_from_ao_groups(
            direct_groups, len(coords), len(selected), rank=4,
        )
        repeated_pivot_seconds = time.perf_counter() - started
        generated_prefix, _, generated_count = pivot_prefix_from_ao_groups(
            generated_groups, len(coords), len(selected), rank=4,
        )
        self.assertEqual(generated_count, direct_count)
        np.testing.assert_array_equal(generated_prefix, direct_prefix)
        self.assertGreaterEqual(repeated_pivot_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
