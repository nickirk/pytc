"""Tests for pytc.integrals.coulomb's free-space Poisson solver section
(task #13, isdf-coulomb-cuda, P2a, 2026-07-13).

Acceptance criteria this file covers (Alice's 3 design-review rounds):
  1. Normalized Gaussian density vs analytic erf(sqrt(a)r)/r, including
     the finite r->0 limit, with h/box/padding convergence.
  2. free_space_poisson_direct_sum_oracle vs the FFT path on small
     real+complex arrays.
  3. Translation/no-wrap near a boundary; odd/even and anisotropic
     meshes; batch-vs-single equality; Hermiticity/realness; positive
     self-energy (restricted to nonnegative real densities only).
  4. NumPy vs JAX CPU agreement in float64, plus a pinned float32/
     complex64 regression (NumPy FFT dtype-preservation behavior is
     version-dependent, per Alice's review -- do not assume it).
  5. Benchmark timing/memory, confirm no O(N_g^2) production allocation.
  Plus: integer wrapped-offset correctness (even/odd P), the
  rectangular_cell-vs-equivalent_sphere quantified approximation error
  and exact scale-homogeneity identities, mismatch-rejection, and a
  one-cell/two-charge normalization check.

Enables jax_enable_x64 explicitly at module level (this module's own
float64 acceptance thresholds require it; must not depend on another
test module enabling it first) -- see test_molecular_df_reference.py's
docstring for the same standing lesson.
"""
import math
import time
import tracemalloc
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy import integrate, special

from pytc.integrals.coulomb import (
    FreeSpacePoissonMesh,
    FreeSpacePoissonKernel,
    build_free_space_poisson_kernel,
    solve_free_space_poisson,
    free_space_poisson_direct_sum_oracle,
    _wrapped_integer_offsets,
    _rectangular_cell_self_potential,
    _equivalent_sphere_self_potential,
)


class TestFreeSpacePoissonMeshValidation(unittest.TestCase):
    def test_valid_mesh_constructs(self):
        mesh = FreeSpacePoissonMesh(
            shape=(4, 5, 6), spacing=(0.1, 0.2, 0.3), origin=(1.0, -2.0, 0.0),
            padded_shape=(8, 10, 12), self_cell_scheme="rectangular_cell",
            fft_normalization="backward",
        )
        self.assertEqual(mesh.shape, (4, 5, 6))

    def test_rejects_nonpositive_shape(self):
        with self.assertRaises(ValueError):
            FreeSpacePoissonMesh(shape=(0, 5, 6), spacing=(0.1, 0.1, 0.1),
                                  origin=(0, 0, 0), padded_shape=(8, 10, 12),
                                  self_cell_scheme="rectangular_cell", fft_normalization="backward")

    def test_rejects_nonpositive_spacing(self):
        with self.assertRaises(ValueError):
            FreeSpacePoissonMesh(shape=(4, 5, 6), spacing=(0.0, 0.1, 0.1),
                                  origin=(0, 0, 0), padded_shape=(8, 10, 12),
                                  self_cell_scheme="rectangular_cell", fft_normalization="backward")

    def test_rejects_nonfinite_origin(self):
        with self.assertRaises(ValueError):
            FreeSpacePoissonMesh(shape=(4, 5, 6), spacing=(0.1, 0.1, 0.1),
                                  origin=(float("inf"), 0, 0), padded_shape=(8, 10, 12),
                                  self_cell_scheme="rectangular_cell", fft_normalization="backward")

    def test_rejects_insufficient_padding(self):
        with self.assertRaises(ValueError):
            FreeSpacePoissonMesh(shape=(4, 5, 6), spacing=(0.1, 0.1, 0.1),
                                  origin=(0, 0, 0), padded_shape=(4, 5, 6),
                                  self_cell_scheme="rectangular_cell", fft_normalization="backward")

    def test_rejects_unsupported_scheme(self):
        with self.assertRaises(ValueError):
            FreeSpacePoissonMesh(shape=(4, 5, 6), spacing=(0.1, 0.1, 0.1),
                                  origin=(0, 0, 0), padded_shape=(8, 10, 12),
                                  self_cell_scheme="cubic_average", fft_normalization="backward")

    def test_rejects_unsupported_normalization(self):
        with self.assertRaises(ValueError):
            FreeSpacePoissonMesh(shape=(4, 5, 6), spacing=(0.1, 0.1, 0.1),
                                  origin=(0, 0, 0), padded_shape=(8, 10, 12),
                                  self_cell_scheme="rectangular_cell", fft_normalization="forward")


class TestWrappedIntegerOffsets(unittest.TestCase):
    def test_matches_fftfreq_even_and_odd(self):
        for P in (1, 2, 7, 8, 9, 16, 17, 31, 32):
            got = _wrapped_integer_offsets(P)
            ref = np.round(np.fft.fftfreq(P) * P).astype(int)
            np.testing.assert_array_equal(got, ref, err_msg=f"mismatch at P={P}")

    def test_no_floating_point_involved(self):
        # Pure integer dtype throughout -- no roundoff-truncation risk.
        idx = _wrapped_integer_offsets(64)
        self.assertTrue(np.issubdtype(idx.dtype, np.integer))


class TestSelfCellSchemes(unittest.TestCase):
    """rectangular_cell (exact closed form) vs equivalent_sphere
    (documented approximation) -- quantified against an INDEPENDENT
    scipy quadrature reference, not the production formula re-run
    against itself."""

    @staticmethod
    def _quadrature_reference_integral(dx, dy, dz, eps_frac=1e-4):
        eps = eps_frac * min(dx, dy, dz)
        ball_contrib = 2 * np.pi * eps**2

        def f(z, y, x):
            r2 = x * x + y * y + z * z
            if r2 <= eps * eps:
                return 0.0
            return 1.0 / np.sqrt(r2)

        val, _ = integrate.tplquad(f, -dx / 2, dx / 2, -dy / 2, dy / 2, -dz / 2, dz / 2,
                                    epsabs=1e-8, epsrel=1e-8)
        return val + ball_contrib

    def test_rectangular_cell_matches_independent_quadrature(self):
        for dx, dy, dz, label in [
            (1.0, 1.0, 1.0, "cube"),
            (1.0, 1.0, 4.0, "aniso 1:1:4"),
            (1.0, 1.0, 10.0, "aniso 1:1:10"),
        ]:
            I_ref = self._quadrature_reference_integral(dx, dy, dz)
            I, K_self = _rectangular_cell_self_potential(dx, dy, dz)
            dV = dx * dy * dz
            self.assertAlmostEqual(I, I_ref, delta=1e-3 * abs(I_ref),
                                    msg=f"{label}: closed-form I disagrees with independent quadrature")
            self.assertAlmostEqual(K_self * dV, I, places=10)

    def test_rectangular_cell_scale_homogeneity(self):
        # I(s*h) = s^2 * I(h); K_self(s*h) = K_self(h)/s -- asserted
        # SEPARATELY, not one inferred from the other (Alice's review).
        dx, dy, dz = 0.7, 1.3, 2.1
        s = 0.37
        I1, K1 = _rectangular_cell_self_potential(dx, dy, dz)
        I2, K2 = _rectangular_cell_self_potential(s * dx, s * dy, s * dz)
        self.assertAlmostEqual(I2, s**2 * I1, places=10)
        self.assertAlmostEqual(K2, K1 / s, places=10)

    def test_equivalent_sphere_overestimates_and_error_grows_with_anisotropy(self):
        # Documented approximation error (not a regression that just
        # freezes it -- checks the SIGN and rough MAGNITUDE bounds, and
        # that error strictly grows with anisotropy ratio).
        cases = [(1.0, 1.0, 1.0), (1.0, 1.0, 4.0), (1.0, 1.0, 10.0)]
        rel_errors = []
        for dx, dy, dz in cases:
            I_rect, K_rect = _rectangular_cell_self_potential(dx, dy, dz)
            I_sph, K_sph, R = _equivalent_sphere_self_potential(dx, dy, dz)
            self.assertIsNotNone(R)
            rel_err = (I_sph - I_rect) / I_rect
            rel_errors.append(rel_err)
        # equivalent_sphere overestimates in every case (positive error).
        for e in rel_errors:
            self.assertGreater(e, 0.0)
        # error strictly increases with anisotropy.
        self.assertLess(rel_errors[0], rel_errors[1])
        self.assertLess(rel_errors[1], rel_errors[2])
        # rough documented bounds: isotropic small, 1:1:10 large.
        self.assertLess(rel_errors[0], 0.05)
        self.assertGreater(rel_errors[2], 0.5)

    def test_equivalent_sphere_radius_none_for_rectangular_cell(self):
        kernel = build_free_space_poisson_kernel((4, 4, 4), (0.2, 0.2, 0.2), backend="numpy")
        self.assertEqual(kernel.mesh.self_cell_scheme, "rectangular_cell")
        self.assertIsNone(kernel.equivalent_sphere_radius)

    def test_equivalent_sphere_radius_set_for_equivalent_sphere_scheme(self):
        kernel = build_free_space_poisson_kernel(
            (4, 4, 4), (0.2, 0.2, 0.2), self_cell_scheme="equivalent_sphere", backend="numpy")
        self.assertIsNotNone(kernel.equivalent_sphere_radius)
        self.assertGreater(kernel.equivalent_sphere_radius, 0.0)


class TestFreeSpacePoissonGaussianConvergence(unittest.TestCase):
    """Acceptance test 1: normalized Gaussian density vs the analytic
    free-space potential erf(sqrt(a)r)/r, including the finite r->0
    limit (2*sqrt(a/pi)), with h-convergence."""

    @staticmethod
    def _gaussian_case(h, N, alpha=2.0):
        shape = (N, N, N)
        spacing = (h, h, h)
        c = N // 2
        coords_1d = (np.arange(N) - c) * h
        X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        rho = (alpha / np.pi) ** 1.5 * np.exp(-alpha * R**2)
        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy", dtype=np.float64)
        v = solve_free_space_poisson(rho, kernel)
        v_analytic = np.where(
            R > 1e-12, special.erf(np.sqrt(alpha) * R) / np.where(R > 1e-12, R, 1.0),
            2.0 * np.sqrt(alpha / np.pi))
        return v, v_analytic, N

    def test_finite_r_to_zero_limit(self):
        alpha = 2.0
        expected = 2.0 * np.sqrt(alpha / np.pi)
        v, v_analytic, N = self._gaussian_case(h=0.15, N=24, alpha=alpha)
        c = N // 2
        self.assertAlmostEqual(v_analytic[c, c, c], expected, places=10)
        # Numeric FFT value at the center should be close to the finite
        # limit (small discretization error, not exact).
        self.assertLess(abs(v[c, c, c] - expected) / expected, 0.02)

    def test_h_convergence(self):
        errs = []
        for h, N in [(0.30, 12), (0.15, 24), (0.075, 48)]:
            v, v_analytic, N_ = self._gaussian_case(h=h, N=N)
            margin = N_ // 6
            interior = slice(margin, N_ - margin)
            err = np.abs(v[interior, interior, interior] - v_analytic[interior, interior, interior])
            errs.append(err.max())
        # Strictly decreasing as h shrinks.
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[2], errs[1])

    def test_padding_convergence_invariant_beyond_minimum(self):
        # Increasing padding beyond the minimum linear-convolution bound
        # must not change the result for the SAME finite-support density
        # -- a structural invariance, not a physical convergence sweep.
        h, N, alpha = 0.15, 16, 2.0
        c = N // 2
        coords_1d = (np.arange(N) - c) * h
        X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        R = np.sqrt(X**2 + Y**2 + Z**2)
        rho = (alpha / np.pi) ** 1.5 * np.exp(-alpha * R**2)
        v2 = solve_free_space_poisson(
            rho, build_free_space_poisson_kernel((N, N, N), (h, h, h), pad_factor=2, backend="numpy"))
        v3 = solve_free_space_poisson(
            rho, build_free_space_poisson_kernel((N, N, N), (h, h, h), pad_factor=3, backend="numpy"))
        np.testing.assert_allclose(v2, v3, atol=1e-10, rtol=1e-10)


class TestFreeSpacePoissonDirectSumOracle(unittest.TestCase):
    """Acceptance test 2: small real+complex arrays, FFT vs oracle."""

    def test_real_small_system(self):
        shape = (4, 5, 3)
        kernel = build_free_space_poisson_kernel(shape, (0.3, 0.25, 0.4), backend="numpy")
        rng = np.random.default_rng(0)
        rho = rng.standard_normal(shape)
        v_fft = solve_free_space_poisson(rho, kernel)
        v_oracle = free_space_poisson_direct_sum_oracle(rho, kernel.mesh)
        np.testing.assert_allclose(v_fft, v_oracle, atol=1e-10, rtol=1e-10)

    def test_complex_small_system(self):
        shape = (4, 3, 5)
        kernel = build_free_space_poisson_kernel(shape, (0.3, 0.3, 0.3),
                                                   fft_kind="fft", dtype=np.complex128, backend="numpy")
        rng = np.random.default_rng(1)
        rho = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        v_fft = solve_free_space_poisson(rho, kernel)
        v_oracle = free_space_poisson_direct_sum_oracle(rho, kernel.mesh)
        np.testing.assert_allclose(v_fft, v_oracle, atol=1e-10, rtol=1e-10)

    def test_equivalent_sphere_scheme_oracle_agreement(self):
        # The oracle must agree with the FFT path under EITHER self-cell
        # scheme -- both share the same _coulomb_kernel_values helper.
        shape = (3, 4, 3)
        kernel = build_free_space_poisson_kernel(
            shape, (0.2, 0.2, 0.2), self_cell_scheme="equivalent_sphere", backend="numpy")
        rng = np.random.default_rng(2)
        rho = rng.standard_normal(shape)
        v_fft = solve_free_space_poisson(rho, kernel)
        v_oracle = free_space_poisson_direct_sum_oracle(rho, kernel.mesh)
        np.testing.assert_allclose(v_fft, v_oracle, atol=1e-10, rtol=1e-10)


class TestFreeSpacePoissonBoundaryAndMeshVariants(unittest.TestCase):
    """Acceptance test 3 (part 1): no-wrap/translation, odd/even meshes,
    anisotropic spacing."""

    def test_no_wrap_charge_near_boundary(self):
        # A charge placed at the mesh's corner cell must NOT show a
        # spurious periodic-image contribution from the opposite corner
        # -- verified against the oracle (which has no periodicity at
        # all), not just "looks physical."
        shape = (10, 10, 10)
        kernel = build_free_space_poisson_kernel(shape, (0.25, 0.25, 0.25), backend="numpy")
        rho = np.zeros(shape)
        rho[0, 0, 0] = 5.0
        v_fft = solve_free_space_poisson(rho, kernel)
        v_oracle = free_space_poisson_direct_sum_oracle(rho, kernel.mesh)
        np.testing.assert_allclose(v_fft, v_oracle, atol=1e-9, rtol=1e-9)
        # potential must fall off with distance from the corner charge,
        # not wrap around and spike near the opposite corner.
        self.assertGreater(v_fft[0, 0, 0], v_fft[-1, -1, -1])

    def test_odd_and_even_mesh_sizes(self):
        for shape in [(5, 5, 5), (6, 6, 6), (5, 6, 7), (4, 5, 6)]:
            kernel = build_free_space_poisson_kernel(shape, (0.3, 0.3, 0.3), backend="numpy")
            rng = np.random.default_rng(hash(shape) % (2**31))
            rho = rng.standard_normal(shape)
            v_fft = solve_free_space_poisson(rho, kernel)
            v_oracle = free_space_poisson_direct_sum_oracle(rho, kernel.mesh)
            np.testing.assert_allclose(v_fft, v_oracle, atol=1e-9, rtol=1e-9,
                                        err_msg=f"shape={shape}")

    def test_anisotropic_spacing(self):
        shape = (5, 4, 6)
        kernel = build_free_space_poisson_kernel(shape, (0.1, 0.5, 0.9), backend="numpy")
        rng = np.random.default_rng(3)
        rho = rng.standard_normal(shape)
        v_fft = solve_free_space_poisson(rho, kernel)
        v_oracle = free_space_poisson_direct_sum_oracle(rho, kernel.mesh)
        np.testing.assert_allclose(v_fft, v_oracle, atol=1e-9, rtol=1e-9)


class TestFreeSpacePoissonBatching(unittest.TestCase):
    def test_batch_matches_single_solves(self):
        shape = (5, 4, 5)
        kernel = build_free_space_poisson_kernel(shape, (0.3, 0.3, 0.3), backend="numpy")
        rng = np.random.default_rng(4)
        rho_batch = rng.standard_normal((3,) + shape)
        v_batch = solve_free_space_poisson(rho_batch, kernel)
        v_single = np.stack([solve_free_space_poisson(rho_batch[i], kernel) for i in range(3)])
        np.testing.assert_array_equal(v_batch, v_single)

    def test_multi_axis_batch(self):
        shape = (4, 4, 4)
        kernel = build_free_space_poisson_kernel(shape, (0.25, 0.25, 0.25), backend="numpy")
        rng = np.random.default_rng(5)
        rho_batch = rng.standard_normal((2, 3) + shape)
        v_batch = solve_free_space_poisson(rho_batch, kernel)
        self.assertEqual(v_batch.shape, (2, 3) + shape)


class TestFreeSpacePoissonHermiticityAndEnergy(unittest.TestCase):
    """Acceptance test 3 (part 2): Hermiticity for the general signed/
    complex case; positive self-energy RESTRICTED to nonnegative real
    densities only (Alice's review -- positive diagonal does not imply
    the full discrete operator is PSD in general)."""

    def _operator(self, kernel, x):
        return solve_free_space_poisson(x, kernel)

    def test_hermiticity_complex(self):
        shape = (4, 3, 4)
        kernel = build_free_space_poisson_kernel(
            shape, (0.2, 0.2, 0.2), fft_kind="fft", dtype=np.complex128, backend="numpy")
        rng = np.random.default_rng(6)
        a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        b = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        Ka = self._operator(kernel, a)
        Kb = self._operator(kernel, b)
        lhs = np.vdot(a, Kb)  # <a, Kb>
        rhs = np.vdot(Ka, b)  # <Ka, b>
        np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)

    def test_quadratic_form_is_real(self):
        shape = (4, 4, 3)
        kernel = build_free_space_poisson_kernel(
            shape, (0.2, 0.2, 0.2), fft_kind="fft", dtype=np.complex128, backend="numpy")
        rng = np.random.default_rng(7)
        a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        Ka = self._operator(kernel, a)
        quad = np.vdot(a, Ka)
        self.assertLess(abs(quad.imag), 1e-9 * abs(quad.real if quad.real != 0 else 1.0) + 1e-9)

    def test_positive_self_energy_nonnegative_density_only(self):
        shape = (5, 5, 5)
        kernel = build_free_space_poisson_kernel(shape, (0.3, 0.3, 0.3), backend="numpy")
        rng = np.random.default_rng(8)
        rho = np.abs(rng.standard_normal(shape))  # elementwise nonnegative
        v = solve_free_space_poisson(rho, kernel)
        dV = 0.3**3
        self_energy = 0.5 * np.sum(rho * v) * dV
        self.assertGreater(self_energy, 0.0)

    def test_no_positivity_claim_for_signed_density(self):
        # A signed density's self-energy is NOT guaranteed positive --
        # confirms this test suite does not assert positivity there
        # (documents the restriction, doesn't just skip it silently).
        shape = (5, 5, 5)
        kernel = build_free_space_poisson_kernel(shape, (0.3, 0.3, 0.3), backend="numpy")
        rng = np.random.default_rng(9)
        rho = rng.standard_normal(shape)  # signed
        v = solve_free_space_poisson(rho, kernel)
        # No assertion on sign here -- Hermiticity (already tested above)
        # is the correctness property that holds unconditionally; the
        # self-energy quadratic form itself is real for a real+symmetric
        # kernel with real rho, which we DO check, without claiming sign.
        dV = 0.3**3
        self_energy = 0.5 * np.sum(rho * v) * dV
        self.assertTrue(np.isfinite(self_energy))


class TestFreeSpacePoissonBackendAgreement(unittest.TestCase):
    """Acceptance test 4: NumPy vs JAX CPU agreement in float64, plus a
    pinned float32/complex64 regression (dtype-preservation behavior of
    FFT libraries is version-dependent -- do not assume it)."""

    def test_numpy_vs_jax_float64(self):
        shape = (5, 4, 6)
        spacing = (0.2, 0.25, 0.3)
        rng = np.random.default_rng(10)
        rho = rng.standard_normal(shape)

        k_np = build_free_space_poisson_kernel(shape, spacing, backend="numpy", dtype=np.float64)
        k_jax = build_free_space_poisson_kernel(shape, spacing, backend="jax", dtype=np.float64)
        v_np = solve_free_space_poisson(rho, k_np)
        v_jax = solve_free_space_poisson(jnp.asarray(rho), k_jax)
        np.testing.assert_allclose(v_np, np.asarray(v_jax), atol=1e-10, rtol=1e-10)

    def test_pinned_float32_complex64_behavior(self):
        # Pin the CURRENT NumPy version's FFT dtype-preservation behavior
        # (verified during design: this numpy preserves float32 ->
        # complex64) as an explicit regression, not an assumed fact.
        shape = (4, 4, 4)
        kernel = build_free_space_poisson_kernel(shape, (0.2, 0.2, 0.2),
                                                   backend="numpy", dtype=np.float32)
        self.assertEqual(kernel.input_dtype, str(np.dtype(np.float32)))
        self.assertIn("complex64", kernel.spectrum_dtype)

        rng = np.random.default_rng(11)
        rho = rng.standard_normal(shape).astype(np.float32)
        v = solve_free_space_poisson(rho, kernel)
        self.assertEqual(v.dtype, np.float32)

        kernel_c = build_free_space_poisson_kernel(
            shape, (0.2, 0.2, 0.2), fft_kind="fft", backend="numpy", dtype=np.complex64)
        rho_c = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        v_c = solve_free_space_poisson(rho_c, kernel_c)
        self.assertEqual(v_c.dtype, np.complex64)

    def test_jax_x64_truncation_raises(self):
        with jax.enable_x64(False):
            with self.assertRaises(ValueError):
                build_free_space_poisson_kernel(
                    (4, 4, 4), (0.2, 0.2, 0.2), backend="jax", dtype=np.float64)

    def test_jax_backend_preserves_device_no_host_copy_marker(self):
        # Confirms the kernel's device field is read from the REALIZED
        # spectrum array, not assumed from jax.devices()[0].
        kernel = build_free_space_poisson_kernel((4, 4, 4), (0.2, 0.2, 0.2), backend="jax")
        self.assertEqual(kernel.device, str(kernel.spectrum.device))
        self.assertIsInstance(kernel.spectrum, jax.Array)


class TestFreeSpacePoissonMismatchRejection(unittest.TestCase):
    def setUp(self):
        self.shape = (4, 4, 4)
        self.kernel_np_real = build_free_space_poisson_kernel(
            self.shape, (0.2, 0.2, 0.2), backend="numpy", dtype=np.float64)

    def test_shape_mismatch_rejected(self):
        rho = np.zeros((5, 5, 5))
        with self.assertRaises(ValueError):
            solve_free_space_poisson(rho, self.kernel_np_real)

    def test_backend_mismatch_rejected(self):
        rho = jnp.zeros(self.shape)
        with self.assertRaises(ValueError):
            solve_free_space_poisson(rho, self.kernel_np_real)

    def test_dtype_mismatch_rejected(self):
        rho = np.zeros(self.shape, dtype=np.float32)
        with self.assertRaises(ValueError):
            solve_free_space_poisson(rho, self.kernel_np_real)

    def test_fft_kind_mismatch_rejected(self):
        rho_complex = np.zeros(self.shape, dtype=np.complex128)
        with self.assertRaises(ValueError):
            solve_free_space_poisson(rho_complex, self.kernel_np_real)

    def test_non_kernel_object_rejected(self):
        with self.assertRaises(TypeError):
            solve_free_space_poisson(np.zeros(self.shape), object())

    def test_non_mesh_object_rejected_by_oracle(self):
        with self.assertRaises(TypeError):
            free_space_poisson_direct_sum_oracle(np.zeros(self.shape), object())


class TestFreeSpacePoissonNormalizationAndInvariance(unittest.TestCase):
    def test_single_occupied_cell_reproduces_self_term(self):
        shape = (6, 6, 6)
        spacing = (0.3, 0.3, 0.3)
        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy")
        rho = np.zeros(shape)
        rho[2, 3, 4] = 1.7
        v = solve_free_space_poisson(rho, kernel)
        dV = spacing[0] * spacing[1] * spacing[2]
        expected_self = kernel.self_cell_integral * rho[2, 3, 4]
        self.assertAlmostEqual(v[2, 3, 4], expected_self, places=8)
        # Sanity: I = K_self * dV.
        self.assertAlmostEqual(kernel.self_cell_integral, kernel.K_self * dV, places=8)

    def test_two_point_charges_off_diagonal_coupling(self):
        shape = (8, 8, 8)
        spacing = (0.25, 0.25, 0.25)
        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy")
        rho = np.zeros(shape)
        rho[1, 1, 1] = 2.0
        rho[5, 1, 1] = 3.0
        v = solve_free_space_poisson(rho, kernel)
        dV = spacing[0] * spacing[1] * spacing[2]
        d = 4 * spacing[0]  # separation along x
        # potential at cell (5,1,1) from the OTHER charge at (1,1,1),
        # excluding its own self term: v_i = dV * rho_j * K(i-j), and
        # K(offset) = 1/|offset| off the self cell -- a single dV factor,
        # not dV^2 (dV appears once, from the Riemann-sum discretization
        # of v_i = dV*sum_j rho_j/|r_i-r_j|, not from the kernel itself).
        expected_cross_only = dV * rho[1, 1, 1] / d
        v_cross_only = v[5, 1, 1] - kernel.self_cell_integral * rho[5, 1, 1]
        self.assertAlmostEqual(v_cross_only, expected_cross_only, places=6)

    def test_pad_factor_2_vs_3_invariance(self):
        shape = (6, 5, 7)
        spacing = (0.3, 0.3, 0.3)
        rng = np.random.default_rng(12)
        rho = rng.standard_normal(shape)
        v2 = solve_free_space_poisson(
            rho, build_free_space_poisson_kernel(shape, spacing, pad_factor=2, backend="numpy"))
        v3 = solve_free_space_poisson(
            rho, build_free_space_poisson_kernel(shape, spacing, pad_factor=3, backend="numpy"))
        v4 = solve_free_space_poisson(
            rho, build_free_space_poisson_kernel(shape, spacing, pad_factor=4, backend="numpy"))
        np.testing.assert_allclose(v2, v3, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(v2, v4, atol=1e-10, rtol=1e-10)


class TestFreeSpacePoissonBenchmark(unittest.TestCase):
    """Acceptance test 5: timing/memory sanity, confirm no O(N_g^2)
    production allocation. Kept small/CI-safe (task #6 review's
    CI-safety lesson: avoid heavy tests in the default suite) -- this
    is a structural/allocation check, not a performance regression gate."""

    def test_no_ng_squared_allocation_and_kernel_reuse_is_cheap(self):
        shape = (24, 24, 24)
        spacing = (0.2, 0.2, 0.2)
        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy")
        rng = np.random.default_rng(13)
        rho_batch = rng.standard_normal((4,) + shape)

        tracemalloc.start()
        t0 = time.perf_counter()
        v = solve_free_space_poisson(rho_batch, kernel)
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        Ng = shape[0] * shape[1] * shape[2]
        # An O(N_g^2) dense kernel for this system would be
        # Ng^2 * 8 bytes ~ 1.5e9 bytes (~1.5 GB) -- peak traced Python
        # allocation here must stay far below that. Measured actual
        # usage for this problem size is ~18 MB (padded-array scale,
        # (2*24)^3 elements plus a handful of FFT working buffers);
        # the threshold below (~153 MB, 10% of the O(N_g^2) bound) is
        # a full order of magnitude looser than the observed value,
        # so this is an order-of-magnitude discriminator against an
        # accidental O(N_g^2) allocation, not a tight perf regression
        # gate that could flake on FFT-library buffer-count changes.
        self.assertLess(peak, Ng**2 * 8 / 10)
        self.assertEqual(v.shape, (4,) + shape)
        self.assertLess(dt, 30.0)  # generous CI-safety ceiling, not a perf gate


if __name__ == "__main__":
    unittest.main()
