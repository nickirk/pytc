import unittest

import jax
import jax.numpy as jnp
import numpy as np
from pyscf.pbc import gto, scf

from pytc import kmat
from pytc.integrals.xtc import XTC
from pytc.pbc.fft_tc import (
    calc_isdf_kernels_fft,
    calc_isdf_l_aux_fft,
    fft_pair_potential,
)
from pytc.pbc.jastrow import BoysHandy
from pytc.pbc.tc import create_tc_fft
from pytc.pbc.xtc import create_xtc_fft
from pytc.utils.reuse import ReuseScope


jax.config.update("jax_enable_x64", True)


def _cell():
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.4"
    cell.basis = "sto-3g"
    cell.a = np.eye(3) * 6.0
    cell.unit = "B"
    cell.cart = True
    cell.verbose = 0
    cell.build()
    return cell


def _fcc_carbon_cell():
    cell = gto.Cell()
    cell.atom = "C 0 0 0; C 0.8917 0.8917 0.8917"
    cell.basis = "sto-3g"
    cell.a = np.array(
        [
            [0.0, 1.7834, 1.7834],
            [1.7834, 0.0, 1.7834],
            [1.7834, 1.7834, 0.0],
        ]
    )
    cell.unit = "A"
    cell.verbose = 0
    cell.build()
    return cell


class TestFFTTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _cell()
        cls.mf = scf.RHF(cls.cell)
        cls.mf.exxdiv = None
        cls.mf.kernel()
        cls.jastrow = BoysHandy.create(cls.cell)
        cls.params = cls.jastrow.init_params()
        cls.tc = create_tc_fft(cls.mf, cls.jastrow, mesh=(2, 2, 2))
        cls.xtc = create_xtc_fft(cls.mf, cls.jastrow, mesh=(2, 2, 2))

    def test_fft_pair_potential_matches_direct_pair_sum(self):
        right = jnp.arange(16, dtype=jnp.float64).reshape(2, 8) / 10
        gradients = self.jastrow.grad_r_batch(
            self.tc.grid_points, self.tc.grid_points, self.params
        )

        actual = fft_pair_potential(
            self.tc.grid_points,
            self.tc.weights,
            self.tc.fft_mesh,
            self.jastrow,
            self.params,
            right,
        )
        expected = jnp.einsum(
            "xyc,ay,y->axc", gradients, right, self.tc.weights
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

        actual_squared = fft_pair_potential(
            self.tc.grid_points,
            self.tc.weights,
            self.tc.fft_mesh,
            self.jastrow,
            self.params,
            right,
            squared_gradient=True,
        )
        expected_squared = jnp.einsum(
            "xy,ay,y->ax",
            jnp.sum(gradients * gradients, axis=-1),
            right,
            self.tc.weights,
        )
        np.testing.assert_allclose(
            actual_squared, expected_squared, atol=1e-12, rtol=1e-12
        )

    def test_fft_pair_potential_reuses_one_compiled_gradient_evaluator(self):
        right = jnp.arange(16, dtype=jnp.float64).reshape(2, 8) / 10
        reuse = ReuseScope(max_entries=1)
        vector = fft_pair_potential(
            self.tc.grid_points,
            self.tc.weights,
            self.tc.fft_mesh,
            self.jastrow,
            self.params,
            right,
            _reuse=reuse,
        )
        squared = fft_pair_potential(
            self.tc.grid_points,
            self.tc.weights,
            self.tc.fft_mesh,
            self.jastrow,
            self.params,
            right,
            squared_gradient=True,
            _reuse=reuse,
        )
        stats = reuse.stats()
        self.assertEqual((stats.misses, stats.hits, stats.entries), (1, 1, 1))

        gradients = self.jastrow.grad_r_batch(
            self.tc.grid_points, self.tc.grid_points, self.params
        )
        np.testing.assert_allclose(
            vector,
            jnp.einsum("xyc,ay,y->axc", gradients, right, self.tc.weights),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            squared,
            jnp.einsum(
                "xy,ay,y->ax",
                jnp.sum(gradients * gradients, axis=-1),
                right,
                self.tc.weights,
            ),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_reused_gradient_row_matches_fcc_boundary_oracle(self):
        cell = _fcc_carbon_cell()
        mesh = (4, 4, 4)
        grid = jnp.asarray(cell.gen_uniform_grids(mesh))
        jastrow = BoysHandy.create(cell)
        params = jastrow.init_params()
        compiled = jax.jit(
            lambda left: jastrow.grad_r_batch(
                left[None, :], grid, params
            )[0]
        )

        eager = jastrow.grad_r_batch(grid, grid, params)
        reused = jnp.stack([compiled(left) for left in grid])
        np.testing.assert_allclose(reused, eager, atol=1e-12, rtol=1e-12)

    def test_fft_tc_two_body_matches_direct_uniform_grid_oracle(self):
        n_orb = self.tc.n_orb
        k1 = kmat.calc_K1(
            self.tc.phi,
            self.tc.grad_phi,
            self.jastrow,
            self.params,
            self.tc.grid_points,
            self.tc.weights,
            batch_size=8,
        ).reshape((n_orb,) * 4)
        k3 = kmat.calc_K3(
            self.tc.phi,
            self.jastrow,
            self.params,
            self.tc.grid_points,
            self.tc.weights,
            batch_size=8,
        ).reshape((n_orb,) * 4)
        direct = 0.5 * (k1 - k1.transpose(1, 0, 2, 3) + k3)
        direct = -(direct + direct.transpose(2, 3, 0, 1))
        np.testing.assert_allclose(
            self.tc.get_2b(self.params), direct, atol=1e-12, rtol=1e-12
        )

    def test_fft_isdf_kernels_match_direct_uniform_grid_oracle(self):
        xi_phi = jnp.array(
            [
                [0.2, 0.8, -0.1, 0.5, 0.3, -0.2, 0.4, 0.7],
                [0.6, -0.3, 0.9, 0.1, -0.4, 0.2, 0.8, -0.5],
            ]
        )
        xi_grad = jnp.stack(
            (xi_phi, 0.5 * xi_phi, -0.25 * xi_phi), axis=-1
        )
        actual_u1, actual_u3 = calc_isdf_kernels_fft(
            xi_phi,
            xi_grad,
            self.tc.weights,
            self.tc.grid_points,
            self.tc.fft_mesh,
            self.jastrow,
            self.params,
        )
        expected_u1 = kmat.calc_K1_kernel(
            xi_grad,
            xi_phi,
            self.tc.weights,
            self.tc.weights,
            self.jastrow,
            self.params,
            self.tc.grid_points,
            self.tc.grid_points,
            batch_size=8,
        )
        expected_u3 = kmat.calc_K3_kernel(
            xi_phi,
            xi_phi,
            self.tc.weights,
            self.tc.weights,
            self.jastrow,
            self.params,
            self.tc.grid_points,
            self.tc.grid_points,
            batch_size=8,
        )
        np.testing.assert_allclose(actual_u1, expected_u1, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual_u3, expected_u3, atol=1e-12, rtol=1e-12)

    def test_fft_isdf_l_aux_matches_direct_uniform_grid_oracle(self):
        xi_phi = jnp.array(
            [
                [0.2, 0.8, -0.1, 0.5, 0.3, -0.2, 0.4, 0.7],
                [0.6, -0.3, 0.9, 0.1, -0.4, 0.2, 0.8, -0.5],
            ]
        )
        actual = calc_isdf_l_aux_fft(
            xi_phi,
            self.tc.weights,
            self.tc.grid_points,
            self.tc.fft_mesh,
            self.jastrow,
            self.params,
        )
        gradients = self.jastrow.grad_r_batch(
            self.tc.grid_points, self.tc.grid_points, self.params
        )
        expected = jnp.einsum(
            "ay,y,xyc->axc", xi_phi, self.tc.weights, gradients
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_fft_xtc_delta_u_matches_direct_uniform_grid_oracle(self):
        direct = self._direct_xtc()
        expected = direct.get_delta_U(self.params, batch_size=8)
        actual = self.xtc.get_delta_U(self.params, batch_size=8)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_fft_xtc_integrals_match_direct_uniform_grid_oracle(self):
        direct = self._direct_xtc()
        dm1 = self.xtc._get_mf_dm()
        np.testing.assert_allclose(
            self.xtc.get_2b(self.params, dm1=dm1, batch_size=8),
            direct.get_2b(self.params, dm1=dm1, batch_size=8),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            self.xtc.get_1b(self.params, dm1=dm1, batch_size=8),
            direct.get_1b(self.params, dm1=dm1, batch_size=8),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            self.xtc.get_3b_fock(self.params, dm1),
            direct.get_3b_fock_full(self.params, dm1),
            atol=1e-12,
            rtol=1e-12,
        )

    def _direct_xtc(self):
        return XTC(
            grid_points=self.xtc.grid_points,
            weights=self.xtc.weights,
            phi=self.xtc.phi,
            grad_phi=self.xtc.grad_phi,
            n_orb=self.xtc.n_orb,
            grid_lvl=self.xtc.grid_lvl,
            jastrow_factor=self.xtc.jastrow_factor,
            mo_coeff=self.xtc.mo_coeff,
            nocc=self.xtc.nocc,
            mo_occ=self.xtc.mo_occ,
            energy_nuc=self.xtc.energy_nuc,
        )

    def test_fft_backend_rejects_nonuniform_weights(self):
        weights = np.asarray(self.tc.weights).copy()
        weights[0] *= 2
        with self.assertRaisesRegex(ValueError, "uniform periodic quadrature"):
            fft_pair_potential(
                self.tc.grid_points,
                weights,
                self.tc.fft_mesh,
                self.jastrow,
                self.params,
                np.ones(8),
            )


if __name__ == "__main__":
    unittest.main()
