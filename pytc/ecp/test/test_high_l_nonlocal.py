"""Tests for the ECP non-local kernel at high angular momentum (l = 2).

Production tests on this branch (C ccECP, CO ccECP, H2O BFD, Cu ccECP) all
involve ECPs with L_max_NL ≤ 1, so the kernel's l ≥ 2 path is never
exercised end-to-end on a PySCF molecule.  These synthetic tests verify
the angular machinery for l = 2 directly.

Two complementary checks:

1. ``_legendre_p_stack`` returns the correct Legendre polynomial values for
   l = 0..5 at arbitrary cos θ, against analytic formulas.

2. **Projector isolation**: for a wavefunction proportional to a real
   spherical harmonic ``Y_{L,M}`` of degree L, the kernel's angular sum

      Σ_q w_q P_l(cos θ_q) Y_{L,M}(Ω̂_q)/Y_{L,M}(Ω̂_i)  =  δ_{l,L}/(2L+1)

   when the grid integrates spherical harmonics of degree ≤ ``l + L``
   exactly.  This is the orthogonality of Legendre projectors with respect
   to spherical harmonics.  For the 12-point icosahedral grid (exact through
   degree 5) this holds for ``l + L ≤ 5``; for the 26-point Lebedev grid
   (exact through 7), for ``l + L ≤ 7``.

   We test the l = 2 self-projection (L=2, l_test=2 → l+L=4 ≤ 5) and the
   l = 2 cross-channel orthogonality against Y_{0,0} and Y_{1,0} (degree
   2 + 0 = 2 and 2 + 1 = 3, both ≤ 5).  This covers the angular machinery
   for the ℓ = 2 channel that would be activated by a 4d/5d ccECP.

Higher-degree cases (L = 3, 4, 5 self-projection) require grids beyond the
12-point default; those are out of scope for these tests because no
realistic QMC ECP has L_max_NL > 2.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pytc.ecp.energy import _legendre_p_stack
from pytc.ecp.quadrature import icosahedral_12, lebedev_26


def _Y_lm_real_unit_sphere(L, M, omega):
    """Real spherical harmonic Y_{L,M}(Ω̂) on the unit sphere (PySCF order)."""
    x, y, z = omega
    if (L, M) == (0, 0):
        return np.sqrt(1.0 / (4 * np.pi))
    if (L, M) == (1, 0):  # p_z
        return np.sqrt(3.0 / (4 * np.pi)) * z
    if (L, M) == (2, 0):  # d_{z²}
        return np.sqrt(5.0 / (16 * np.pi)) * (3 * z * z - 1)
    raise NotImplementedError((L, M))


class TestLegendrePStack(unittest.TestCase):
    """``_legendre_p_stack`` must return correct P_l(x) for l = 0..5."""

    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.x = jnp.array([-0.9, -0.4, 0.0, 0.3, 0.7, 1.0])

    def test_legendre_values(self):
        x = self.x
        got = np.asarray(_legendre_p_stack(6, x))   # (6, 6)
        x_np = np.asarray(x)
        ref = np.stack([
            np.ones_like(x_np),                                         # P_0
            x_np,                                                       # P_1
            0.5 * (3 * x_np**2 - 1),                                    # P_2
            0.5 * (5 * x_np**3 - 3 * x_np),                             # P_3
            (1/8) * (35 * x_np**4 - 30 * x_np**2 + 3),                  # P_4
            (1/8) * (63 * x_np**5 - 70 * x_np**3 + 15 * x_np),          # P_5
        ], axis=0)
        np.testing.assert_allclose(got, ref, atol=1e-13, rtol=1e-12)


class TestProjectorL2Isolation(unittest.TestCase):
    """The l = 2 projector must pick up Y_2 with coefficient 1/(2L+1) = 1/5
    and reject Y_0, Y_1 to machine precision, on the default 12-point grid.
    """

    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.grid = icosahedral_12()
        omega_i = np.array([0.2, -0.4, np.sqrt(1 - 0.2**2 - 0.4**2)])
        self.omega_i = omega_i / np.linalg.norm(omega_i)
        self.cos_theta = np.asarray(self.grid.directions) @ self.omega_i

    def _projection_value(self, L_psi, l_test):
        """Σ_q w_q P_{l_test}(cos θ_q) · Y_{L,0}(Ω̂_q) / Y_{L,0}(Ω̂_i)."""
        psi_i = _Y_lm_real_unit_sphere(L_psi, 0, self.omega_i)
        psi_q = np.array([
            _Y_lm_real_unit_sphere(L_psi, 0, np.asarray(self.grid.directions[q]))
            for q in range(self.grid.n_points)
        ])
        ratio = psi_q / psi_i
        P_l = np.asarray(
            _legendre_p_stack(l_test + 1, jnp.asarray(self.cos_theta))
        )[l_test]
        return float(np.sum(np.asarray(self.grid.weights) * P_l * ratio))

    def test_l2_self_projection(self):
        # Σ_q w_q P_2(cos θ_q) Y_{2,0}(Ω̂_q)/Y_{2,0}(Ω̂_i) = 1/(2·2+1) = 1/5.
        # Integrand degree: l+L = 2+2 = 4 ≤ 5 → exact on 12-pt icosahedral.
        got = self._projection_value(L_psi=2, l_test=2)
        np.testing.assert_allclose(got, 1.0 / 5.0, atol=1e-12)

    def test_l2_rejects_Y0(self):
        # Σ_q w_q P_2(cos θ_q) Y_{0,0}(Ω̂_q)/Y_{0,0}(Ω̂_i) = 0.
        # Integrand degree: 2 + 0 = 2 ≤ 5 → exact.
        got = self._projection_value(L_psi=0, l_test=2)
        np.testing.assert_allclose(got, 0.0, atol=1e-12)

    def test_l2_rejects_Y1(self):
        # Integrand degree: 2 + 1 = 3 ≤ 5 → exact.
        got = self._projection_value(L_psi=1, l_test=2)
        np.testing.assert_allclose(got, 0.0, atol=1e-12)

    def test_l0_and_l1_cross_rejected_for_d_state(self):
        # ψ ∝ Y_2 → only the l=2 projector should fire; l=0 and l=1 must be 0.
        # Integrand degrees: l + L ≤ 3 ≤ 5 → exact.
        for l in (0, 1):
            got = self._projection_value(L_psi=2, l_test=l)
            np.testing.assert_allclose(
                got, 0.0, atol=1e-12,
                err_msg=f"l={l} projection of Y_2: got {got}, want 0",
            )


class TestProjectorL2OnLebedev26(unittest.TestCase):
    """Same checks on the 26-point Lebedev grid — the optional higher-order
    quadrature available via the ``ecp_quad_grid`` kwarg.  Useful for 4d/5d
    ECPs with L_max_NL = 2 where the 12-point grid may be too coarse for
    trial wavefunctions with sizable g/h content (l + L can exceed 5).
    """

    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.grid = lebedev_26()
        omega_i = np.array([0.1, 0.6, np.sqrt(1 - 0.1**2 - 0.6**2)])
        self.omega_i = omega_i / np.linalg.norm(omega_i)
        self.cos_theta = np.asarray(self.grid.directions) @ self.omega_i

    def test_l2_self_projection(self):
        psi_i = _Y_lm_real_unit_sphere(2, 0, self.omega_i)
        psi_q = np.array([
            _Y_lm_real_unit_sphere(2, 0, np.asarray(self.grid.directions[q]))
            for q in range(self.grid.n_points)
        ])
        ratio = psi_q / psi_i
        P_2 = np.asarray(
            _legendre_p_stack(3, jnp.asarray(self.cos_theta))
        )[2]
        got = float(np.sum(np.asarray(self.grid.weights) * P_2 * ratio))
        np.testing.assert_allclose(got, 1.0 / 5.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
