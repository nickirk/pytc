"""CPU feasibility test for reciprocal primitive-AO translation."""

import dataclasses
import time
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc import gto, tools

from pytc.pbc import coulomb
from pytc.pbc.df.kpts import canonicalize_kpts
from pytc.pbc.df.reciprocal_ao_pilot import (
    ReciprocalOrbitPartition,
    ReciprocalOrbitPartitionError,
    ReciprocalSameGridCapacityError,
    reciprocal_same_grid_byte_model,
    select_reciprocal_same_grid,
    metric_column_from_ao_groups,
    downsample_uniform_grid_values,
    pivot_prefix_from_ao_groups,
    reciprocal_translate_bloch_ao,
    relative_frobenius_error,
    uniform_grid_downsample_indices,
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
    cell = tools.super_cell(primitive, [2, 1, 1])
    cell.mesh = np.array([27, 13, 13], dtype=np.int32)
    return cell


def _diamond_211_partition(cell):
    return ReciprocalOrbitPartition(
        seed_shell_slice=(0, 6),
        replica_shell_slices=((6, 12),),
        replica_translations=np.asarray([cell.atom_coords()[2] - cell.atom_coords()[0]]),
        supercell_matrix=np.diag([2, 1, 1]),
    )


class _HybridOrbitCell:
    """Small shell-sliced AO fixture that rejects accidental full AO requests."""

    nbas = 3
    mesh = np.array([2, 2, 2], dtype=np.int64)
    _bas = np.array([[0, 1], [1, 1], [2, 2]], dtype=np.int32)

    def ao_loc_nr(self):
        return np.array([0, 1, 2, 3], dtype=np.int64)

    def atom_coords(self):
        return np.array([[0., 0., 0.], [0., 0., 0.], [.3, .2, .1]])

    def get_Gv(self, mesh):
        return np.zeros((int(np.prod(mesh)), 3))

    def pbc_eval_gto(self, name, coords, *, kpts, shls_slice=None):
        if shls_slice is None:
            raise AssertionError("selector must never request a full-supercell AO tensor")
        n_grid = len(coords)
        columns = []
        for shell in range(*shls_slice):
            if shell in (0, 1):
                columns.append(np.linspace(1.0, 2.0, n_grid))
            else:
                columns.append(np.linspace(2.0, 1.0, n_grid))
        return np.asarray(columns, dtype=np.complex128).T[None]


class _ThreeReplicaOrbitCell(_HybridOrbitCell):
    """Three identical one-AO replica groups for residency regression."""

    nbas = 3
    _bas = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32)

    def atom_coords(self):
        return np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])

    def pbc_eval_gto(self, name, coords, *, kpts, shls_slice=None):
        if shls_slice is None:
            raise AssertionError("selector must never request a full-supercell AO tensor")
        values = np.linspace(1.0, 2.0, len(coords))
        return np.repeat(values[None, :, None], shls_slice[1] - shls_slice[0], axis=2)


class TestReciprocalPrimitiveAOPilot(unittest.TestCase):
    def test_hybrid_orbit_partition_accumulates_reused_and_unique_groups(self):
        cell = _HybridOrbitCell()
        coords = np.zeros((8, 3))
        partition = ReciprocalOrbitPartition(
            seed_shell_slice=(0, 1),
            replica_shell_slices=((1, 2),),
            replica_translations=np.zeros((1, 3)),
            unique_shell_slices=((2, 3),),
            supercell_matrix=np.diag([2, 1, 1]),
        )
        pivots, factor, count, provenance = select_reciprocal_same_grid(
            cell, np.zeros((1, 3)), coords, rank=2, partition=partition,
            selection_peak_max_bytes=10**9, process_pipeline_allowance_bytes=2**20,
            return_factor=True,
        )
        seed = cell.pbc_eval_gto("GTOval", coords, kpts=[np.zeros(3)], shls_slice=(0, 1))
        replica = cell.pbc_eval_gto("GTOval", coords, kpts=[np.zeros(3)], shls_slice=(1, 2))
        unique = cell.pbc_eval_gto("GTOval", coords, kpts=[np.zeros(3)], shls_slice=(2, 3))
        expected, _, expected_count = pivot_prefix_from_ao_groups(
            lambda: iter((seed, replica, unique)), len(coords), 1, rank=2,
        )
        self.assertEqual(count, expected_count)
        np.testing.assert_array_equal(pivots, expected)
        self.assertEqual(factor.dtype, np.float64)
        self.assertFalse(provenance["hidden_full_supercell_ao_allocation"])
        self.assertEqual(provenance["NAO_reused"], 1)
        self.assertEqual(provenance["NAO_unique"], 1)
        self.assertEqual(provenance["realized_cache_factor"], 1.5)
        self.assertEqual(provenance["generated_group_live_limit"], 1)
        self.assertEqual(provenance["reconstruction_count"], count + 1)
        self.assertEqual(provenance["unique_direct_evaluations"], count + 1)

    def test_invalid_hybrid_orbit_and_capacity_fail_closed(self):
        cell = _HybridOrbitCell()
        invalid = ReciprocalOrbitPartition(
            seed_shell_slice=(0, 1),
            replica_shell_slices=((1, 2),),
            replica_translations=np.array([[1.0, 0.0, 0.0]]),
            unique_shell_slices=((2, 3),),
            supercell_matrix=np.diag([2, 1, 1]),
        )
        with self.assertRaisesRegex(ReciprocalOrbitPartitionError, "INVALID_ORBIT_PARTITION"):
            select_reciprocal_same_grid(
                cell, np.zeros((1, 3)), np.zeros((8, 3)), rank=2, partition=invalid,
                selection_peak_max_bytes=10**9, process_pipeline_allowance_bytes=2**20,
            )
        model = reciprocal_same_grid_byte_model(
            1, 8, 1, 1, 2, 2, selection_peak_max_bytes=1,
            process_pipeline_allowance_bytes=2**20,
        )
        self.assertEqual(model["capacity_condition"], "RECIPROCAL_SAME_GRID_SELECTION_PEAK_EXCEEDS_POLICY")
        valid = dataclasses.replace(invalid, replica_translations=np.zeros((1, 3)))
        with self.assertRaisesRegex(
            ReciprocalSameGridCapacityError,
            "PROCESS_PIPELINE_ALLOWANCE_REQUIRED",
        ):
            select_reciprocal_same_grid(
                cell, np.zeros((1, 3)), np.zeros((8, 3)), rank=2, partition=valid,
                selection_peak_max_bytes=10**9,
            )
        with self.assertRaisesRegex(ReciprocalSameGridCapacityError, "RECIPROCAL_SAME_GRID_SELECTION_PEAK_EXCEEDS_POLICY"):
            select_reciprocal_same_grid(
                cell, np.zeros((1, 3)), np.zeros((8, 3)), rank=2, partition=valid,
                selection_peak_max_bytes=1, process_pipeline_allowance_bytes=2**20,
            )

    def test_three_replica_groups_release_before_next_generation(self):
        cell = _ThreeReplicaOrbitCell()
        coords = np.zeros((8, 3))
        partition = ReciprocalOrbitPartition(
            seed_shell_slice=(0, 1),
            replica_shell_slices=((1, 2), (2, 3)),
            replica_translations=np.array([[1., 0., 0.], [2., 0., 0.]]),
            supercell_matrix=np.diag([3, 1, 1]),
        )
        pivots, _, count, provenance = select_reciprocal_same_grid(
            cell, np.zeros((1, 3)), coords, rank=2, partition=partition,
            selection_peak_max_bytes=10**9, process_pipeline_allowance_bytes=2**20,
        )
        reference = cell.pbc_eval_gto("GTOval", coords, kpts=[np.zeros(3)], shls_slice=(0, 1))
        expected, _, expected_count = pivot_prefix_from_ao_groups(
            lambda: iter((reference, reference, reference)), len(coords), 1, rank=2,
        )
        self.assertEqual(count, expected_count)
        np.testing.assert_array_equal(pivots, expected)
        self.assertEqual(provenance["generated_group_live_limit"], 1)
        self.assertEqual(provenance["n_replicas"], 3)
        self.assertEqual(provenance["reconstruction_count"], 2 * (count + 1))

    def test_diamond_same_grid_selector_uses_seed_and_one_generated_group(self):
        cell = _diamond_211()
        mesh_obj = canonicalize_kpts(cell, cell.make_kpts([3, 2, 1], wrap_around=False))
        coords = cell.get_uniform_grids(cell.mesh)
        pivots, factor, count, provenance = select_reciprocal_same_grid(
            cell, mesh_obj.canonical_kpts, coords, rank=4,
            partition=_diamond_211_partition(cell), selection_peak_max_bytes=10**9,
            process_pipeline_allowance_bytes=2**20, return_factor=True,
        )
        direct = np.asarray(cell.pbc_eval_gto(
            "GTOval", coords, kpts=list(mesh_obj.canonical_kpts),
        ), dtype=np.complex128)
        generated = np.asarray([
            reciprocal_translate_bloch_ao(
                direct[k, :, :26], coords, cell.mesh, kpt,
                cell.atom_coords()[2] - cell.atom_coords()[0], cell.get_Gv(cell.mesh),
            )
            for k, kpt in enumerate(mesh_obj.canonical_kpts)
        ])
        expected, _, expected_count = pivot_prefix_from_ao_groups(
            lambda: iter((direct[:, :, :26], generated)), len(coords), len(mesh_obj.canonical_kpts),
            rank=4,
        )
        self.assertEqual(count, expected_count)
        np.testing.assert_array_equal(pivots, expected)
        self.assertEqual(factor.dtype, np.float64)
        self.assertFalse(provenance["hidden_full_supercell_ao_allocation"])
        self.assertEqual(provenance["NAO_reused"], 26)
        self.assertEqual(provenance["NAO_unique"], 0)
        self.assertEqual(provenance["n_replicas"], 2)
        self.assertEqual(provenance["reconstruction_count"], count + 1)
        self.assertEqual(provenance["fft_count"], 2 * len(mesh_obj.canonical_kpts) * (count + 1))

    def test_nondefault_build_uses_exact_downstream_aos(self):
        cell = _diamond_211()
        kpts = cell.make_kpts([3, 2, 1], wrap_around=False)
        result = coulomb.build(
            cell, kpts, rank=2, block_size=256, rtol=1e-5,
            selection_mode="reciprocal_same_grid",
            reciprocal_orbit_partition=_diamond_211_partition(cell),
            selection_peak_max_bytes=10**9,
            process_pipeline_allowance_bytes=2**20,
        )
        mesh_obj = result["mesh_obj"]
        pivots = np.asarray(result["selection_provenance"]["pivot_indices"])
        exact_inpv = np.asarray(cell.pbc_eval_gto(
            "GTOval", cell.get_uniform_grids(cell.mesh)[pivots],
            kpts=list(mesh_obj.canonical_kpts),
        ), dtype=np.complex128)
        np.testing.assert_allclose(result["inpv_kpt"], exact_inpv, atol=0.0, rtol=0.0)
        self.assertEqual(result["selection_provenance"]["eta_ao_source"], "exact_streamed_pyscf_ao")
        self.assertFalse(result["selection_provenance"]["hidden_full_supercell_ao_allocation"])

    def test_oversampled_target_grid_metrics_and_on_demand_pivots(self):
        cell = _diamond_211()
        mesh = np.asarray(cell.mesh)
        fine_mesh = 2 * mesh
        coords = cell.get_uniform_grids(mesh)
        fine_coords = cell.get_uniform_grids(fine_mesh)
        index_map = uniform_grid_downsample_indices(mesh, fine_mesh)
        np.testing.assert_allclose(coords, fine_coords[index_map], atol=0.0)
        mesh_obj = canonicalize_kpts(cell, cell.make_kpts([3, 2, 1], wrap_around=False))
        selected = np.array([0, 1, 2])
        kpts = mesh_obj.canonical_kpts[selected]
        n_primitive_ao = cell.nao_nr() // 2
        translation = cell.atom_coords()[2] - cell.atom_coords()[0]
        fine_gvectors = cell.get_Gv(fine_mesh)
        direct = np.asarray(cell.pbc_eval_gto(
            "GTOval", coords, kpts=list(kpts),
        ), dtype=np.complex128)
        coarse_target = direct[:, :, n_primitive_ao:]

        started = time.perf_counter()
        fine_seeds = np.empty(
            (len(kpts), len(fine_coords), n_primitive_ao), dtype=np.complex128,
        )
        for k_index, kpt in enumerate(kpts):
            fine_direct = np.asarray(cell.pbc_eval_gto(
                "GTOval", fine_coords, kpts=[kpt],
            ), dtype=np.complex128)
            fine_seeds[k_index] = fine_direct[0, :, :n_primitive_ao]
        del fine_direct
        seed_build_seconds = time.perf_counter() - started
        self.assertGreaterEqual(seed_build_seconds, 0.0)
        self.assertEqual(
            fine_seeds.nbytes,
            len(kpts) * len(fine_coords) * n_primitive_ao * np.dtype(np.complex128).itemsize,
        )

        def generate_target_grid():
            return np.asarray([
                downsample_uniform_grid_values(
                    reciprocal_translate_bloch_ao(
                        fine_seeds[k], fine_coords, fine_mesh, kpt,
                        translation, fine_gvectors,
                    ),
                    mesh, fine_mesh,
                )
                for k, kpt in enumerate(kpts)
            ])

        started = time.perf_counter()
        accurate_target = generate_target_grid()
        one_sweep_seconds = time.perf_counter() - started
        self.assertGreaterEqual(one_sweep_seconds, 0.0)
        target_error = relative_frobenius_error(coarse_target, accurate_target)
        self.assertLess(target_error["max_abs"], 5e-11)
        self.assertLess(target_error["relative_frobenius"], 5e-11)

        direct_groups = lambda: iter((direct[:, :, :n_primitive_ao], coarse_target))
        accurate_groups = lambda: iter((direct[:, :, :n_primitive_ao], accurate_target))
        for pivot in (0, len(coords) // 3, len(coords) - 1):
            np.testing.assert_allclose(
                metric_column_from_ao_groups(direct_groups(), pivot, len(kpts)),
                metric_column_from_ao_groups(accurate_groups(), pivot, len(kpts)),
                atol=3e-10, rtol=3e-10,
            )
        direct_prefix, _, direct_count = pivot_prefix_from_ao_groups(
            direct_groups, len(coords), len(kpts), rank=16,
        )
        accurate_prefix, _, accurate_count = pivot_prefix_from_ao_groups(
            accurate_groups, len(coords), len(kpts), rank=16,
        )
        self.assertEqual(accurate_count, direct_count)
        np.testing.assert_array_equal(accurate_prefix, direct_prefix)

        regenerations = {"count": 0}

        def on_demand_groups():
            regenerations["count"] += 1
            return iter((direct[:, :, :n_primitive_ao], generate_target_grid()))

        rank = 4
        started = time.perf_counter()
        on_demand_prefix, _, on_demand_count = pivot_prefix_from_ao_groups(
            on_demand_groups, len(coords), len(kpts), rank=rank,
        )
        on_demand_seconds = time.perf_counter() - started
        self.assertGreaterEqual(on_demand_seconds, 0.0)
        self.assertEqual(regenerations["count"], rank + 1)
        self.assertEqual(on_demand_count, rank)
        np.testing.assert_array_equal(on_demand_prefix, direct_prefix[:rank])

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
