import unittest
import numpy as np
import jax
import jax.numpy as jnp

from pytc.pbc.utils import (
    wrap,
    mic_displacement,
    mic_distance,
    generate_images,
)


class TestWrap(unittest.TestCase):
    def test_wrap_orthorhombic_idempotent(self):
        lattice = jnp.diag(jnp.array([5.0, 6.0, 7.0]))
        positions = jnp.array([
            [12.3, -1.4, 14.9],
            [0.0, 0.0, 0.0],
            [4.999, 5.999, 6.999],
        ])
        wrapped = wrap(positions, lattice)
        # All wrapped coords lie in [0, L_i) for each axis
        L = jnp.diag(lattice)
        self.assertTrue(bool(jnp.all(wrapped >= 0.0)))
        self.assertTrue(bool(jnp.all(wrapped < L)))
        # Wrapping twice is a no-op
        np.testing.assert_allclose(wrap(wrapped, lattice), wrapped, atol=1e-12)

    def test_wrap_preserves_periodic_equivalence(self):
        """A wrapped position differs from the original by an integer combination
        of lattice vectors."""
        rng = np.random.default_rng(0)
        lattice = jnp.array([
            [3.0, 0.5, 0.0],
            [0.4, 4.0, 0.2],
            [0.1, 0.0, 5.0],
        ])
        positions = jnp.asarray(rng.uniform(-20, 20, size=(10, 3)))
        wrapped = wrap(positions, lattice)
        delta = positions - wrapped
        # delta in fractional coords should be integer
        frac_delta = delta @ jnp.linalg.inv(lattice)
        np.testing.assert_allclose(
            frac_delta, jnp.round(frac_delta), atol=1e-10
        )

    def test_wrap_batched(self):
        lattice = jnp.diag(jnp.array([4.0, 4.0, 4.0]))
        positions = jnp.arange(2 * 5 * 3, dtype=jnp.float64).reshape(2, 5, 3) * 1.7
        wrapped = wrap(positions, lattice)
        self.assertEqual(wrapped.shape, positions.shape)
        self.assertTrue(bool(jnp.all((wrapped >= 0) & (wrapped < 4.0))))


class TestMIC(unittest.TestCase):
    def test_mic_orthorhombic_against_brute_force(self):
        lattice = jnp.diag(jnp.array([5.0, 6.0, 7.0]))
        r1 = jnp.array([0.1, 0.2, 0.3])
        r2 = jnp.array([4.9, 5.9, 6.9])
        # Brute-force minimum image: search over 3x3x3 image cells
        images = generate_images(lattice, rcut=15.0)
        candidates = (r1 - r2) - images  # r1 - (r2 + T)
        brute_dist = float(jnp.min(jnp.linalg.norm(candidates, axis=-1)))
        d = float(mic_distance(r1, r2, lattice))
        self.assertAlmostEqual(d, brute_dist, places=10)

    def test_mic_displacement_inside_half_cell(self):
        lattice = jnp.diag(jnp.array([5.0, 6.0, 7.0]))
        rng = np.random.default_rng(1)
        r1 = jnp.asarray(rng.uniform(-10, 10, size=(20, 3)))
        r2 = jnp.asarray(rng.uniform(-10, 10, size=(20, 3)))
        disp = mic_displacement(r1, r2, lattice)
        L = jnp.diag(lattice)
        # Fractional displacement should lie in [-0.5, 0.5)
        frac = disp @ jnp.linalg.inv(lattice)
        self.assertTrue(bool(jnp.all(frac >= -0.5 - 1e-10)))
        self.assertTrue(bool(jnp.all(frac < 0.5 + 1e-10)))
        # Cartesian magnitude bounded by half-diagonal
        self.assertTrue(bool(jnp.all(jnp.linalg.norm(disp, axis=-1) <= jnp.linalg.norm(L) / 2 + 1e-10)))

    def test_mic_distance_jit_and_grad(self):
        lattice = jnp.diag(jnp.array([5.0, 6.0, 7.0]))
        r1 = jnp.array([0.1, 0.2, 0.3])
        r2 = jnp.array([4.9, 5.9, 6.9])
        d_jit = jax.jit(mic_distance)(r1, r2, lattice)
        d = mic_distance(r1, r2, lattice)
        np.testing.assert_allclose(d_jit, d, atol=1e-12)
        # gradient w.r.t. r1 should be a unit vector along the MIC displacement
        g = jax.grad(lambda r: mic_distance(r, r2, lattice))(r1)
        disp = mic_displacement(r1, r2, lattice)
        np.testing.assert_allclose(g, disp / jnp.linalg.norm(disp), atol=1e-10)


class TestGenerateImages(unittest.TestCase):
    def test_origin_included_and_first(self):
        lattice = jnp.eye(3) * 2.0
        images = generate_images(lattice, rcut=3.0)
        np.testing.assert_array_equal(images[0], np.zeros(3))

    def test_origin_excluded(self):
        lattice = jnp.eye(3) * 2.0
        images = generate_images(lattice, rcut=3.0, include_origin=False)
        norms = np.linalg.norm(images, axis=-1)
        self.assertTrue(bool(np.all(norms > 0)))

    def test_count_for_simple_cubic(self):
        # In a unit cube with rcut=1.0, integer points satisfying |n| <= 1
        # are: origin (1), face-centered ±e_i (6) = 7 total.
        lattice = jnp.eye(3)
        images = generate_images(lattice, rcut=1.0)
        self.assertEqual(images.shape[0], 7)

    def test_all_within_cutoff(self):
        lattice = jnp.array([
            [3.0, 0.5, 0.0],
            [0.4, 4.0, 0.2],
            [0.1, 0.0, 5.0],
        ])
        rcut = 7.0
        images = generate_images(lattice, rcut=rcut)
        norms = np.linalg.norm(images, axis=-1)
        self.assertTrue(bool(np.all(norms <= rcut + 1e-12)))

    def test_completeness_against_oversized_box(self):
        """A larger n_max search should find no additional images within rcut
        beyond what generate_images returns."""
        lattice = jnp.array([
            [3.0, 0.0, 0.0],
            [0.6, 2.5, 0.0],
            [0.0, 0.1, 4.0],
        ])
        rcut = 5.0
        images = generate_images(lattice, rcut=rcut)
        # Reference: enumerate a much larger box and filter
        N = 10
        lat = np.asarray(lattice)
        ns = np.arange(-N, N + 1)
        grid = np.stack(np.meshgrid(ns, ns, ns, indexing='ij'), axis=-1).reshape(-1, 3)
        ref = grid @ lat
        ref = ref[np.linalg.norm(ref, axis=-1) <= rcut]
        self.assertEqual(images.shape[0], ref.shape[0])


if __name__ == '__main__':
    unittest.main()
