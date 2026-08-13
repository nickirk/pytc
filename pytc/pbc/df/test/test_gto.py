import unittest
from unittest import mock

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pyscf.pbc import gto

from pytc.pbc import coulomb
from pytc.pbc.df.gto import (
    PeriodicGTOEvaluator,
    _eval_periodic_ao,
    eval_periodic_ao,
)


def make_carbon_cell(basis="cc-pvtz", *, cart=False):
    cell = gto.Cell()
    cell.atom = "C 0.3 0.4 0.5"
    cell.a = np.eye(3) * 7.0
    cell.unit = "B"
    cell.basis = basis
    cell.cart = cart
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()
    return cell


class TestPeriodicGTOEvaluator(unittest.TestCase):
    def test_contracted_spdf_gamma_and_kpoint_match_pyscf(self):
        cell = make_carbon_cell()
        self.assertEqual(max(cell.bas_angular(i) for i in range(cell.nbas)), 3)
        kpts = cell.make_kpts([2, 1, 1])
        coords = np.asarray(
            [
                [0.1, 0.2, 0.3],
                [1.3, 2.1, 3.7],
                [6.8, 0.1, 5.5],
            ],
            dtype=np.float64,
        )
        evaluator = PeriodicGTOEvaluator.from_cell(cell, kpts)
        actual = np.asarray(eval_periodic_ao(evaluator, jnp.asarray(coords)))
        expected = np.asarray(
            cell.pbc_eval_gto("GTOval_sph", coords, kpts=list(kpts)),
            dtype=np.complex128,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
        self.assertEqual(evaluator.lmax, 3)
        self.assertGreater(evaluator.n_lattice, 1)
        self.assertEqual(evaluator.image_block_size, 8)
        self.assertGreaterEqual(
            evaluator.lattice_vectors.shape[0] * evaluator.image_block_size,
            evaluator.n_lattice,
        )
        self.assertLess(
            evaluator.lattice_vectors.shape[0] * evaluator.image_block_size,
            evaluator.n_lattice + evaluator.image_block_size,
        )

    def test_eager_and_jit_are_identical(self):
        cell = make_carbon_cell("cc-pvdz")
        kpts = cell.make_kpts([2, 1, 1])
        coords = jnp.asarray([[0.2, 0.4, 0.8], [6.9, 0.1, 3.3]])
        evaluator = PeriodicGTOEvaluator.from_cell(cell, kpts)
        eager = np.asarray(_eval_periodic_ao(evaluator, coords))
        compiled = np.asarray(eval_periodic_ao(evaluator, coords))
        np.testing.assert_array_equal(compiled, eager)

    def test_cartesian_basis_fails_closed(self):
        cell = make_carbon_cell("cc-pvdz", cart=True)
        with self.assertRaisesRegex(NotImplementedError, "spherical AOs only"):
            PeriodicGTOEvaluator.from_cell(cell, np.zeros((1, 3)))

    def test_angular_momentum_above_f_fails_closed(self):
        cell = make_carbon_cell("cc-pvqz")
        self.assertGreater(max(cell.bas_angular(i) for i in range(cell.nbas)), 3)
        with self.assertRaisesRegex(NotImplementedError, "through f"):
            PeriodicGTOEvaluator.from_cell(cell, np.zeros((1, 3)))

    def test_full_build_does_not_call_pyscf_pbc_eval_gto(self):
        cell = gto.Cell()
        cell.atom = "He 0 0 0"
        cell.a = np.eye(3) * 5.0
        cell.unit = "B"
        cell.basis = "gth-dzvp"
        cell.pseudo = "gth-pbe"
        cell.mesh = [4, 4, 4]
        cell.precision = 1e-9
        cell.verbose = 0
        cell.build()
        with mock.patch.object(
            cell,
            "pbc_eval_gto",
            side_effect=AssertionError("PySCF AO path must not be called"),
        ):
            result = coulomb.build(
                cell,
                np.zeros((1, 3)),
                rank=2,
                block_size=16,
                selection_mode="fixed_pivots",
                fixed_pivots=np.asarray([0, 1]),
            )
        self.assertEqual(result["n_selected"], 2)
        self.assertEqual(
            result["selection_provenance"]["ao_backend"]["backend"],
            "jax_periodic_spherical_gto",
        )


if __name__ == "__main__":
    unittest.main()
