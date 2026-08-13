"""Exact-oracle gates for periodic Boys--Handy channel factorization."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf.pbc import gto

from pytc.jastrow.bh import BHTerm
from pytc.pbc.bh_channels import (
    apply_boys_handy_channel_plan,
    apply_boys_handy_channels,
    boys_handy_coefficient_families,
    channel_plan_nbytes,
    prepare_boys_handy_channels,
)
from pytc.pbc.fft_tc import fft_pair_potential
from pytc.pbc.jastrow import BoysHandy
from pytc.utils.reuse import ReuseScope


jax.config.update("jax_enable_x64", True)


def _cell(natom=2):
    cell = gto.Cell()
    cell.atom = "; ".join(f"H {1.4 * atom} 0 0" for atom in range(natom))
    cell.basis = "sto-3g"
    cell.a = np.eye(3) * 8.0
    cell.unit = "B"
    cell.cart = True
    cell.mesh = [2, 2, 2]
    cell.verbose = 0
    cell.build()
    return cell


class TestBoysHandyChannels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _cell()
        cls.mesh = (2, 2, 2)
        cls.grid = jnp.asarray(cls.cell.gen_uniform_grids(cls.mesh))
        cls.weights = jnp.full(cls.grid.shape[0], cls.cell.vol / cls.grid.shape[0])
        cls.jastrow = BoysHandy.create(cls.cell)
        cls.params = cls.jastrow.init_params()
        cls.right = jnp.stack(
            (
                jnp.linspace(0.2, 0.9, cls.grid.shape[0]),
                jnp.cos(jnp.arange(cls.grid.shape[0])),
            )
        )

    def test_gradient_apply_matches_left_row_oracle(self):
        actual, diagnostics = apply_boys_handy_channels(
            self.grid,
            self.weights,
            self.mesh,
            self.jastrow,
            self.params,
            self.right,
        )
        expected = fft_pair_potential(
            self.grid,
            self.weights,
            self.mesh,
            self.jastrow,
            self.params,
            self.right,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
        self.assertEqual(diagnostics["n_gradient_channels"], 26)

    def test_squared_gradient_apply_matches_left_row_oracle(self):
        actual, diagnostics = apply_boys_handy_channels(
            self.grid,
            self.weights,
            self.mesh,
            self.jastrow,
            self.params,
            self.right[:1],
            squared_gradient=True,
        )
        expected = fft_pair_potential(
            self.grid,
            self.weights,
            self.mesh,
            self.jastrow,
            self.params,
            self.right[:1],
            squared_gradient=True,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
        self.assertEqual(diagnostics["raw_k3_cross_count"], 351)
        self.assertLess(
            diagnostics["n_radial_kernel_transforms"],
            diagnostics["raw_k3_cross_count"],
        )

    def test_same_centre_matrix_has_no_cross_centre_edges(self):
        terms = [[BHTerm(2, 0, 2, 0.25)]]
        jastrow = BoysHandy.create(self.cell, terms_per_nucleus=terms)
        labels, families = boys_handy_coefficient_families(
            jastrow, jastrow.init_params()
        )
        matrix = families[0].coefficient
        index = {label: i for i, label in enumerate(labels)}
        self.assertEqual(matrix[index[(0, 2)], index[(1, 2)]], 0.0)
        self.assertNotEqual(matrix[index[(0, 2)], index[(None, 0)]], 0.0)
        self.assertNotEqual(matrix[index[(1, 2)], index[(None, 0)]], 0.0)

    def test_default_inventory_includes_radial_power_six(self):
        _, families = boys_handy_coefficient_families(
            self.jastrow, self.params
        )
        self.assertEqual(
            [family.radial_power for family in families], [0, 1, 2, 3, 4, 6]
        )

    def test_prepared_plan_is_eager_jit_and_reuse_stable(self):
        plan = prepare_boys_handy_channels(
            self.grid,
            self.weights,
            self.mesh,
            self.jastrow,
            self.params,
        )
        for squared_gradient in (False, True):
            eager = apply_boys_handy_channel_plan(
                plan,
                self.weights,
                self.right,
                squared_gradient=squared_gradient,
            )[0]
            compiled = jax.jit(
                lambda right: apply_boys_handy_channel_plan(
                    plan,
                    self.weights,
                    right,
                    squared_gradient=squared_gradient,
                )[0]
            )(self.right)
            repeated = apply_boys_handy_channel_plan(
                plan,
                self.weights,
                self.right,
                squared_gradient=squared_gradient,
            )[0]
            np.testing.assert_allclose(
                compiled, eager, atol=2e-15, rtol=2e-15
            )
            np.testing.assert_array_equal(repeated, eager)

        reuse = ReuseScope(max_entries=1)
        first = fft_pair_potential(
            self.grid,
            self.weights,
            self.mesh,
            self.jastrow,
            self.params,
            self.right,
            channel_factorized=True,
            _reuse=reuse,
        )
        second = fft_pair_potential(
            self.grid,
            self.weights,
            self.mesh,
            self.jastrow,
            self.params,
            self.right,
            channel_factorized=True,
            _reuse=reuse,
        )
        np.testing.assert_array_equal(second, first)
        self.assertEqual((reuse.stats().misses, reuse.stats().hits), (1, 1))
        self.assertGreater(channel_plan_nbytes(plan), 0)
        for channel in plan.channels:
            self.assertLessEqual(channel.left.size, 3 * plan.n_grid)
            self.assertEqual(channel.right.size, plan.n_grid)
            self.assertLessEqual(channel.radial.size, 3 * plan.n_grid)

    def test_default_channel_rank_scales_linearly_with_identical_centres(self):
        for natom in (2, 4, 8):
            cell = _cell(natom)
            jastrow = BoysHandy.create(cell)
            _, families = boys_handy_coefficient_families(
                jastrow, jastrow.init_params()
            )
            n_channels = sum(
                family.envelope_numerical_rank
                + (family.numerical_rank if family.radial_power else 0)
                for family in families
            )
            self.assertEqual(n_channels, 7 * natom + 12)


if __name__ == "__main__":
    unittest.main()
