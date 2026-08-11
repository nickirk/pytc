"""Rigorous reference gates for the experimental single-IBP Coulomb form.

The tests intentionally keep this path separate from the production Coulomb
default.  They cover the vector FFT algebra, dense oracle agreement, analytic
derivative interpolation with fixed pivots, smooth-density convergence,
finite-box and grid-phase sensitivity, complex sesquilinearity, and the
unsymmetrized core contraction.

Task #18, #proj-isdf-coulomb-cuda, 2026-07-13.
"""

import math
import unittest

import numpy as np

from pytc.integrals.coulomb import (
    build_free_space_poisson_kernel,
    solve_free_space_poisson,
)
from pytc.utils.single_ibp_coulomb_benchmark import (
    build_gradient_interpolation_vectors,
    build_single_ibp_kernel,
    single_ibp_core,
    single_ibp_direct_sum_oracle,
    solve_single_ibp_vector,
)


def _mesh(shape, spacing, origin=(0.0, 0.0, 0.0), *, fft_kind="rfft", dtype=np.float64):
    return build_free_space_poisson_kernel(
        shape, spacing, origin=origin, backend="numpy", fft_kind=fft_kind, dtype=dtype
    )


def _gaussian(alpha, center, coords):
    delta = coords - np.asarray(center)
    r2 = np.sum(delta * delta, axis=-1)
    rho = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)
    grad = -2.0 * alpha * delta * rho[:, None]
    return rho, grad


def _coords(mesh):
    axes = [mesh.origin[i] + np.arange(mesh.shape[i]) * mesh.spacing[i] for i in range(3)]
    xyz = np.meshgrid(*axes, indexing="ij")
    return np.stack([x.reshape(-1) for x in xyz], axis=1)


class TestSingleIBPVectorConvolution(unittest.TestCase):
    def _oracle_case(self, shape, spacing, dtype, fft_kind):
        rng = np.random.default_rng(7)
        poisson = _mesh(shape, spacing, fft_kind=fft_kind, dtype=dtype)
        kernel = build_single_ibp_kernel(poisson.mesh, fft_kind=fft_kind, dtype=dtype)
        rho = rng.standard_normal((2,) + shape)
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            rho = rho + 1j * rng.standard_normal((2,) + shape)
        rho = rho.astype(dtype)
        actual = solve_single_ibp_vector(rho, kernel)
        reference = single_ibp_direct_sum_oracle(rho, poisson.mesh)
        tol = 2e-5 if np.dtype(dtype).itemsize <= 4 else 2e-12
        np.testing.assert_allclose(actual, reference, atol=tol, rtol=tol)

    def test_real_odd_even_anisotropic_dense_oracle(self):
        self._oracle_case((3, 4, 2), (0.2, 0.35, 0.5), np.float64, "rfft")

    def test_complex_odd_even_anisotropic_dense_oracle(self):
        self._oracle_case((2, 3, 3), (0.4, 0.25, 0.3), np.complex128, "fft")

    def test_float32_dense_oracle(self):
        self._oracle_case((3, 3, 4), (0.3, 0.2, 0.4), np.float32, "rfft")

    def test_one_cell_self_vector_is_exactly_zero(self):
        poisson = _mesh((1, 1, 1), (0.2, 0.3, 0.4))
        kernel = build_single_ibp_kernel(poisson.mesh)
        rho = np.array([[[2.0]]])
        np.testing.assert_array_equal(solve_single_ibp_vector(rho, kernel), 0.0)

    def test_complex_linearity(self):
        shape = (3, 3, 3)
        poisson = _mesh(shape, (0.3, 0.3, 0.3), fft_kind="fft", dtype=np.complex128)
        kernel = build_single_ibp_kernel(poisson.mesh, fft_kind="fft", dtype=np.complex128)
        rng = np.random.default_rng(9)
        a = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex128)
        b = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex128)
        alpha = 0.7 - 0.2j
        np.testing.assert_allclose(
            solve_single_ibp_vector(a + alpha * b, kernel),
            solve_single_ibp_vector(a, kernel) + alpha * solve_single_ibp_vector(b, kernel),
            atol=2e-12, rtol=2e-12,
        )

    def test_padding_factor_invariance_and_linear_storage(self):
        shape = (4, 3, 5)
        spacing = (0.2, 0.3, 0.25)
        rho = np.random.default_rng(10).normal(size=shape)
        poisson2 = build_free_space_poisson_kernel(
            shape, spacing, pad_factor=2, backend="numpy", dtype=np.float64
        )
        poisson3 = build_free_space_poisson_kernel(
            shape, spacing, pad_factor=3, backend="numpy", dtype=np.float64
        )
        kernel2 = build_single_ibp_kernel(poisson2.mesh)
        kernel3 = build_single_ibp_kernel(poisson3.mesh)
        np.testing.assert_allclose(
            solve_single_ibp_vector(rho, kernel2),
            solve_single_ibp_vector(rho, kernel3),
            atol=2e-12, rtol=2e-12,
        )
        # Three vector spectra, each proportional to the padded grid; no
        # N_g x N_g production allocation is retained.
        self.assertLessEqual(kernel2.spectrum.size, 3 * math.prod(poisson2.mesh.padded_shape))

    def test_rejects_shape_dtype_and_kind_mismatch(self):
        poisson = _mesh((3, 3, 3), (0.3, 0.3, 0.3))
        kernel = build_single_ibp_kernel(poisson.mesh)
        with self.assertRaises(ValueError):
            solve_single_ibp_vector(np.zeros((3, 3, 2), dtype=np.float64), kernel)
        with self.assertRaises(ValueError):
            solve_single_ibp_vector(np.zeros((3, 3, 3), dtype=np.float32), kernel)
        with self.assertRaises(ValueError):
            build_single_ibp_kernel(poisson.mesh, fft_kind="rfft", dtype=np.complex128)


class TestFixedPivotAnalyticGradientTheta(unittest.TestCase):
    def test_matches_independent_dense_lstsq_and_grid_batching(self):
        rng = np.random.default_rng(11)
        factor_p = rng.normal(size=(2, 13))
        factor_q = rng.normal(size=(3, 13))
        grad_p = rng.normal(size=(3, 2, 13))
        grad_q = rng.normal(size=(3, 3, 13))
        pivots = np.array([0, 2, 5, 8, 11])
        actual = build_gradient_interpolation_vectors(
            factor_p, factor_q, grad_p, grad_q, pivots, rcond=1e-14,
            grid_batch_size=4,
        )
        c = np.einsum(
            "pm,qm->pqm", factor_p[:, pivots], factor_q[:, pivots]
        ).reshape(6, len(pivots))
        expected = []
        for axis in range(3):
            db = (
                np.einsum("pg,qg->pqg", grad_p[axis], factor_q)
                + np.einsum("pg,qg->pqg", factor_p, grad_q[axis])
            ).reshape(6, 13)
            expected.append(np.linalg.lstsq(c, db, rcond=None)[0])
        expected = np.stack(expected, axis=1)
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
        unbatched = build_gradient_interpolation_vectors(
            factor_p, factor_q, grad_p, grad_q, pivots, rcond=1e-14
        )
        np.testing.assert_allclose(actual, unbatched, atol=2e-12, rtol=2e-12)


class TestSingleIBPCoreDenseOracle(unittest.TestCase):
    def test_real_and_complex_unsymmetrized_core(self):
        for dtype, fft_kind in [(np.float64, "rfft"), (np.complex128, "fft")]:
            rng = np.random.default_rng(13)
            shape = (3, 2, 3)
            poisson = _mesh(shape, (0.2, 0.35, 0.4), fft_kind=fft_kind, dtype=dtype)
            kernel = build_single_ibp_kernel(poisson.mesh, fft_kind=fft_kind, dtype=dtype)
            n_grid = math.prod(shape)
            theta = rng.normal(size=(4, n_grid))
            grad = rng.normal(size=(5, 3, n_grid))
            if np.issubdtype(np.dtype(dtype), np.complexfloating):
                theta = theta + 1j * rng.normal(size=theta.shape)
                grad = grad + 1j * rng.normal(size=grad.shape)
            theta, grad = theta.astype(dtype), grad.astype(dtype)
            vector = single_ibp_direct_sum_oracle(
                theta.reshape((4,) + shape), poisson.mesh
            ).reshape(4, 3, n_grid)
            expected = -0.5 * math.prod(poisson.mesh.spacing) * np.einsum(
                "mcg,ncg->mn", grad.conj(), vector
            )
            actual = single_ibp_core(grad, theta, kernel, nu_block_size=3)
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)


class TestSingleIBPAnalyticGaussian(unittest.TestCase):
    @staticmethod
    def _energy(h, half_width, shift_fraction=0.0):
        n = int(round(2.0 * half_width / h))
        origin = -half_width + 0.5 * h + shift_fraction * h
        poisson = _mesh((n, n, n), (h, h, h), (origin, origin, origin))
        coords = _coords(poisson.mesh)
        rho, grad = _gaussian(1.0, (0.0, 0.0, 0.0), coords)
        rho_grid = rho.reshape(poisson.mesh.shape)
        grad_theta = grad.T.reshape(1, 3, -1)
        kernel = build_single_ibp_kernel(poisson.mesh)
        single = single_ibp_core(grad_theta, rho.reshape(1, -1), kernel)[0, 0]
        potential = solve_free_space_poisson(rho_grid, poisson)
        direct = math.prod(poisson.mesh.spacing) * np.sum(rho_grid * potential)
        return float(direct), float(single)

    def test_convergence_is_monotone_and_beats_direct(self):
        exact = math.sqrt(2.0 / math.pi)
        hs = [0.6, 0.4, 0.3]
        values = [self._energy(h, 4.2) for h in hs]
        direct_errors = np.array([abs(v[0] - exact) for v in values])
        single_errors = np.array([abs(v[1] - exact) for v in values])
        self.assertTrue(np.all(np.diff(direct_errors) < 0), direct_errors)
        self.assertTrue(np.all(np.diff(single_errors) < 0), single_errors)
        self.assertLess(single_errors[-1], direct_errors[-1] / 20.0)
        observed_order = math.log(single_errors[1] / single_errors[2]) / math.log(0.4 / 0.3)
        self.assertGreater(observed_order, 3.5)

    def test_finite_box_error_is_exposed_not_hidden(self):
        exact = math.sqrt(2.0 / math.pi)
        small = abs(self._energy(0.3, 2.1)[1] - exact)
        large = abs(self._energy(0.3, 4.2)[1] - exact)
        self.assertGreater(small, 100.0 * large)
        self.assertLess(large, 1e-4)

    def test_half_cell_phase_spread_is_small_for_smooth_gaussian(self):
        values = [self._energy(0.3, 4.2, shift)[1] for shift in (0.0, 0.25, 0.5)]
        self.assertLess(max(values) - min(values), 2e-7)

    def test_complex_cross_energy_dagger_relation_converged_box(self):
        h = 0.35
        n = 24
        origin = -4.2 + 0.5 * h
        poisson = _mesh(
            (n, n, n), (h, h, h), (origin, origin, origin),
            fft_kind="fft", dtype=np.complex128,
        )
        coords = _coords(poisson.mesh)
        g1, dg1 = _gaussian(1.0, (-0.4, 0.1, 0.0), coords)
        g2, dg2 = _gaussian(0.7, (0.6, -0.2, 0.3), coords)
        g3, dg3 = _gaussian(1.3, (0.0, 0.4, -0.5), coords)
        rho_a, grad_a = g1 + 0.25j * g2, dg1 + 0.25j * dg2
        rho_b, grad_b = 0.6 * g2 - 0.4j * g3, 0.6 * dg2 - 0.4j * dg3
        kernel = build_single_ibp_kernel(
            poisson.mesh, fft_kind="fft", dtype=np.complex128
        )
        e_ab = single_ibp_core(
            grad_a.T.reshape(1, 3, -1), rho_b.reshape(1, -1), kernel
        )[0, 0]
        e_ba = single_ibp_core(
            grad_b.T.reshape(1, 3, -1), rho_a.reshape(1, -1), kernel
        )[0, 0]
        self.assertLess(abs(e_ab - e_ba.conjugate()) / max(abs(e_ab), 1e-15), 2e-4)


if __name__ == "__main__":
    unittest.main()
