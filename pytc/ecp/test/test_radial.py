"""Tests for pytc.ecp.radial."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pytc.ecp.radial import (
    eval_radial_channel,
    eval_v_loc,
    eval_v_nl,
    find_nonlocal_cutoff,
)


def _ref_radial(r, n_powers, zetas, coeffs):
    """Reference radial sum c_k * r^(n_k - 2) * exp(-zeta_k r^2)."""
    r = np.asarray(r)
    out = np.zeros_like(r, dtype=np.float64)
    for n, z, c in zip(n_powers, zetas, coeffs):
        out += c * r ** (int(n) - 2) * np.exp(-z * r ** 2)
    return out


class TestEvalRadialChannel(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_single_term(self):
        # 4.0 * r^(-1) * exp(-14.435 r^2)
        n = jnp.array([1])
        z = jnp.array([14.43502])
        c = jnp.array([4.0])
        r = jnp.linspace(0.1, 4.0, 32)
        got = eval_radial_channel(r, n, z, c)
        ref = _ref_radial(r, [1], [14.43502], [4.0])
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-10, atol=1e-12)

    def test_carbon_ccecp_local(self):
        # C ccECP local channel (from inspection of mol._ecp earlier):
        #   4.0   * r^(-1) * exp(-14.43502 r^2)
        #  -25.81955  * r^0  * exp(-7.38188  r^2)
        #   57.74008  * r^1  * exp(-8.39889  r^2)
        n = jnp.array([1, 2, 3])
        z = jnp.array([14.43502, 7.38188, 8.39889])
        c = jnp.array([4.0, -25.81955, 57.74008])
        r = jnp.linspace(0.05, 5.0, 50)
        got = eval_radial_channel(r, n, z, c)
        ref = _ref_radial(r, n, z, c)
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-10, atol=1e-12)

    def test_zero_coeff_padding(self):
        # Padding terms (c = 0) should contribute nothing even if zeta = n = 0.
        n = jnp.array([0, 1])
        z = jnp.array([1.0, 0.0])  # zero zeta on padding entry
        c = jnp.array([0.0, 0.0])
        r = jnp.array([0.5, 1.0, 2.0])
        got = eval_radial_channel(r, n, z, c)
        np.testing.assert_allclose(np.asarray(got), 0.0, atol=1e-14)

    def test_zero_distance_does_not_nan(self):
        # r = 0 should be regularized (no nan/inf), since QMC sampling never
        # places electrons exactly on top of nuclei but a JIT trace still
        # needs to be finite.
        n = jnp.array([1])  # gives r^(-1) at the bare formula
        z = jnp.array([1.0])
        c = jnp.array([1.0])
        got = eval_radial_channel(jnp.array(0.0), n, z, c)
        self.assertTrue(np.isfinite(float(got)))


class TestEvalVLocAndNL(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_shapes(self):
        n_e, n_atoms, k_loc, l_plus_1, k_nl = 3, 2, 4, 2, 3
        r_iA = jnp.linspace(0.5, 3.0, n_e * n_atoms).reshape(n_e, n_atoms)
        loc_n = jnp.zeros((n_atoms, k_loc), dtype=jnp.int32)
        loc_z = jnp.ones((n_atoms, k_loc))
        loc_c = jnp.zeros((n_atoms, k_loc))
        nl_n = jnp.zeros((n_atoms, l_plus_1, k_nl), dtype=jnp.int32)
        nl_z = jnp.ones((n_atoms, l_plus_1, k_nl))
        nl_c = jnp.zeros((n_atoms, l_plus_1, k_nl))
        v_loc = eval_v_loc(r_iA, loc_n, loc_z, loc_c)
        v_nl = eval_v_nl(r_iA, nl_n, nl_z, nl_c)
        self.assertEqual(v_loc.shape, (n_e, n_atoms))
        self.assertEqual(v_nl.shape, (n_e, n_atoms, l_plus_1))

    def test_zero_when_no_ecp(self):
        # All-zero coefficients (non-ECP atom) yield zero potential.
        n_e, n_atoms = 4, 1
        r = jnp.array([[0.5], [1.0], [2.0], [3.0]])
        loc_n = jnp.zeros((n_atoms, 2), dtype=jnp.int32)
        loc_z = jnp.ones((n_atoms, 2))
        loc_c = jnp.zeros((n_atoms, 2))
        v = eval_v_loc(r, loc_n, loc_z, loc_c)
        np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-14)

    def test_carbon_ccecp_loc_via_v_loc(self):
        # Same C ccECP local channel, evaluated through eval_v_loc with
        # n_atoms = 1, broadcast over a batch of electrons.
        n_atoms = 1
        n = jnp.array([[1, 2, 3]])
        z = jnp.array([[14.43502, 7.38188, 8.39889]])
        c = jnp.array([[4.0, -25.81955, 57.74008]])
        r = jnp.linspace(0.1, 4.0, 16).reshape(-1, 1)
        v = eval_v_loc(r, n, z, c)
        ref = _ref_radial(
            np.linspace(0.1, 4.0, 16),
            [1, 2, 3], [14.43502, 7.38188, 8.39889],
            [4.0, -25.81955, 57.74008],
        )
        np.testing.assert_allclose(np.asarray(v[:, 0]), ref, rtol=1e-10)


class TestFindNonlocalCutoff(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_zero_when_no_nonlocal(self):
        # All-zero coefficients -> r_cut = 0.
        n_atoms, l_plus_1, k = 2, 2, 3
        nl_n = jnp.zeros((n_atoms, l_plus_1, k), dtype=jnp.int32)
        nl_z = jnp.ones((n_atoms, l_plus_1, k))
        nl_c = jnp.zeros((n_atoms, l_plus_1, k))
        r_c = find_nonlocal_cutoff(nl_n, nl_z, nl_c, tol=1e-5)
        np.testing.assert_allclose(np.asarray(r_c), 0.0, atol=1e-14)

    def test_carbon_ccecp_l0(self):
        # C ccECP only has a single l=0 non-local term:
        #   52.13345 * r^0 * exp(-7.76079 r^2)
        # Solve for r where |V| = 1e-5:
        #   |V(r)| = 52.13345 * exp(-7.76079 r^2) < 1e-5
        # => 7.76079 r^2 > ln(52.13345 / 1e-5) = ln(5.21e6) ~ 15.466
        # => r > sqrt(15.466 / 7.76079) ~ 1.412
        n_atoms = 1
        l_plus_1 = 1
        k = 1
        nl_n = jnp.array([[[2]]], dtype=jnp.int32)
        nl_z = jnp.array([[[7.76079]]])
        nl_c = jnp.array([[[52.13345]]])
        r_c = find_nonlocal_cutoff(nl_n, nl_z, nl_c, tol=1e-5, r_max=5.0, n_grid=4096)
        # Expect ~1.41 Bohr; allow a coarse tolerance because of the grid step.
        self.assertGreater(float(r_c[0]), 1.30)
        self.assertLess(float(r_c[0]), 1.55)

    def test_cutoff_monotone_in_tol(self):
        # Tighter tolerance -> larger cutoff.
        nl_n = jnp.array([[[2]]], dtype=jnp.int32)
        nl_z = jnp.array([[[7.76079]]])
        nl_c = jnp.array([[[52.13345]]])
        r_c_loose = find_nonlocal_cutoff(nl_n, nl_z, nl_c, tol=1e-3, r_max=5.0)
        r_c_tight = find_nonlocal_cutoff(nl_n, nl_z, nl_c, tol=1e-8, r_max=5.0)
        self.assertLess(float(r_c_loose[0]), float(r_c_tight[0]))


if __name__ == "__main__":
    unittest.main()
