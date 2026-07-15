"""CPU parity test for the standalone JAX periodic spherical AO pilot."""

import time
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from pytc.pbc.df.jax_periodic_ao_pilot import (
    PeriodicSphericalAOPilot,
    compile_spherical_block_evaluator,
    stream_periodic_spherical_ao_blocks,
)
from pytc.pbc.df.kpts import canonicalize_kpts
from pytc.pbc.df.reciprocal_ao_pilot import (
    metric_column_from_ao_groups,
    pivot_prefix_from_ao_groups,
    relative_frobenius_error,
)
from pytc.pbc.df.test.test_reciprocal_ao_pilot import _diamond_211


class TestJAXPeriodicAOPilot(unittest.TestCase):
    def test_spherical_images_match_pyscf_and_metric_prefix(self):
        cell = _diamond_211()
        coords = cell.get_uniform_grids(cell.mesh)[:256]
        mesh_obj = canonicalize_kpts(cell, cell.make_kpts([3, 2, 1], wrap_around=False))
        selected = np.array([0, 1, 2])
        kpts = mesh_obj.canonical_kpts[selected]
        pilot = PeriodicSphericalAOPilot.create(cell)
        self.assertEqual(pilot.nao, cell.nao_nr())
        self.assertEqual(pilot.lattice_vectors.shape[0], cell.get_lattice_Ls().shape[0])

        direct_started = time.perf_counter()
        direct = np.asarray(cell.pbc_eval_gto("GTOval", coords, kpts=list(kpts)))
        direct_seconds = time.perf_counter() - direct_started
        compile_started = time.perf_counter()
        compiled = compile_spherical_block_evaluator(pilot, grid_block_size=64)
        compile_seconds = time.perf_counter() - compile_started

        generated = []
        stats = {}
        warm_started = time.perf_counter()
        for kpt in kpts:
            generated.append(np.concatenate([
                block for _, _, block in stream_periodic_spherical_ao_blocks(
                    pilot, compiled, coords, kpt, grid_block_size=64,
                    image_block_size=8, stats=stats,
                )
            ], axis=0))
        warm_seconds = time.perf_counter() - warm_started
        generated = np.asarray(generated)
        error = relative_frobenius_error(direct, generated)
        self.assertLess(error["max_abs"], 4e-11)
        self.assertLess(error["relative_frobenius"], 4e-11)
        self.assertGreater(direct_seconds, 0.0)
        self.assertGreater(compile_seconds, 0.0)
        self.assertGreater(warm_seconds, 0.0)
        self.assertEqual(stats["grid_blocks"], 12)
        self.assertEqual(stats["image_blocks"], 12 * 89)
        self.assertEqual(stats["jax_image_evaluations"], 12 * 711)

        direct_groups = lambda: iter((direct[:, :, :26], direct[:, :, 26:]))
        generated_groups = lambda: iter((generated[:, :, :26], generated[:, :, 26:]))
        for pivot in (0, 85, 255):
            np.testing.assert_allclose(
                metric_column_from_ao_groups(direct_groups(), pivot, len(kpts)),
                metric_column_from_ao_groups(generated_groups(), pivot, len(kpts)),
                atol=2e-10, rtol=2e-10,
            )
        direct_prefix, _, direct_count = pivot_prefix_from_ao_groups(
            direct_groups, len(coords), len(kpts), rank=4,
        )
        generated_prefix, _, generated_count = pivot_prefix_from_ao_groups(
            generated_groups, len(coords), len(kpts), rank=4,
        )
        self.assertEqual(generated_count, direct_count)
        np.testing.assert_array_equal(generated_prefix, direct_prefix)


if __name__ == "__main__":
    unittest.main()
