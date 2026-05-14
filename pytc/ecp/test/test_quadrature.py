"""Tests for pytc.ecp.quadrature."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pytc.ecp.quadrature import get_grid, icosahedral_12, lebedev_26


def _real_spherical_harmonic(l, m, dirs):
    """Real Y_lm evaluated at unit directions. Unnormalized order is fine
    for the integration test: we only need a closed-form l ≥ 1 function
    whose integral over the sphere is zero.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    if (l, m) == (0, 0):
        return np.ones_like(x)
    if (l, m) == (1, -1):
        return y
    if (l, m) == (1, 0):
        return z
    if (l, m) == (1, 1):
        return x
    if (l, m) == (2, -2):
        return x * y
    if (l, m) == (2, -1):
        return y * z
    if (l, m) == (2, 0):
        return 3.0 * z * z - 1.0
    if (l, m) == (2, 1):
        return x * z
    if (l, m) == (2, 2):
        return x * x - y * y
    if (l, m) == (3, 0):
        # 5 z^3 - 3 z * (x^2 + y^2 + z^2) on the unit sphere reduces to
        # 5 z^3 - 3 z.
        return 5.0 * z ** 3 - 3.0 * z
    if (l, m) == (4, 0):
        return 35.0 * z ** 4 - 30.0 * z ** 2 + 3.0
    if (l, m) == (5, 0):
        return 63.0 * z ** 5 - 70.0 * z ** 3 + 15.0 * z
    if (l, m) == (6, 0):
        return 231.0 * z ** 6 - 315.0 * z ** 4 + 105.0 * z ** 2 - 5.0
    if (l, m) == (7, 0):
        return 429.0 * z ** 7 - 693.0 * z ** 5 + 315.0 * z ** 3 - 35.0 * z
    raise NotImplementedError((l, m))


class TestIcosahedral12(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.grid = icosahedral_12()

    def test_12_points(self):
        self.assertEqual(self.grid.n_points, 12)

    def test_weights_sum_to_one(self):
        w = np.asarray(self.grid.weights)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)

    def test_unit_vectors(self):
        dirs = np.asarray(self.grid.directions)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_exact_for_l_le_5(self):
        dirs = np.asarray(self.grid.directions)
        w = np.asarray(self.grid.weights)
        for l, m in [
            (1, 0), (1, 1), (1, -1),
            (2, 0), (2, 1), (2, -1), (2, 2), (2, -2),
            (3, 0),
            (4, 0),
            (5, 0),
        ]:
            f = _real_spherical_harmonic(l, m, dirs)
            integral = float(np.sum(w * f))
            self.assertAlmostEqual(
                integral, 0.0, places=10,
                msg=f"non-zero integral for Y_{l}{m}: {integral}",
            )

    def test_constant_function(self):
        dirs = np.asarray(self.grid.directions)
        w = np.asarray(self.grid.weights)
        integral = float(np.sum(w * np.ones(len(dirs))))
        self.assertAlmostEqual(integral, 1.0, places=12)


class TestLebedev26(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        self.grid = lebedev_26()

    def test_26_points(self):
        self.assertEqual(self.grid.n_points, 26)

    def test_weights_sum_to_one(self):
        w = np.asarray(self.grid.weights)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)

    def test_unit_vectors(self):
        dirs = np.asarray(self.grid.directions)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_exact_for_l_le_7(self):
        dirs = np.asarray(self.grid.directions)
        w = np.asarray(self.grid.weights)
        for l, m in [
            (1, 0), (2, 0), (3, 0), (4, 0),
            (5, 0), (6, 0), (7, 0),
        ]:
            f = _real_spherical_harmonic(l, m, dirs)
            integral = float(np.sum(w * f))
            self.assertAlmostEqual(
                integral, 0.0, places=10,
                msg=f"non-zero integral for Y_{l}{m}: {integral}",
            )


class TestGridLookup(unittest.TestCase):
    def test_default(self):
        g = get_grid()
        self.assertEqual(g.n_points, 12)

    def test_by_name(self):
        g = get_grid("lebedev_26")
        self.assertEqual(g.n_points, 26)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_grid("not-a-grid")


if __name__ == "__main__":
    unittest.main()
