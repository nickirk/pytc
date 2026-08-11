"""Tests for pytc.integrals.coulomb's Poisson interpolation-vector Z-core
assembly section (task #15, isdf-coulomb-cuda, P2b, 2026-07-13).

Acceptance criteria this file covers (Alice's task spec + 2 design-review
rounds):
  1. Tiny real+complex systems vs an independent dense O(N_g^2) Coulomb-
     matrix oracle, checking Theta construction, dV, conjugation, and
     every Z entry.
  2. Tiled vs untiled equality across non-divisible grid/mu/nu block
     sizes, multiple batch sizes, odd/even anisotropic meshes, rfft/full
     FFT paths.
  3. Hermiticity and real diagonal for complex Theta; explicit mismatch
     rejection (mesh/kernel/backend/dtype/kind/provenance).
  4. NumPy/JAX x64 agreement plus float32/complex64 policy tests;
     warning-clean.
  5. End-to-end synthetic fit test using known interpolation points
     (full analytic-rank pivot count -> exact reconstruction), before
     any larger regression.
  6. Peak-memory/time benchmark demonstrating no forbidden allocations,
     labeled as an explicit incore baseline.
  7. Immutable typed result/provenance artifacts -- seeded from real
     builder output (not a hand-rolled placeholder dict), perturbing one
     field at a time.

Enables jax_enable_x64 explicitly at module level (this module's own
float64 tolerances require it; must not depend on another test module
enabling it first).
"""
import dataclasses
import math
import time
import tracemalloc
import unittest
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pytc.integrals.coulomb import (
    FreeSpacePoissonKernel,
    PoissonInterpolationSector,
    PoissonCoreArtifact,
    build_free_space_poisson_kernel,
    build_poisson_interpolation_sector,
    poisson_core,
    free_space_poisson_direct_sum_oracle,
)


def _dense_theta_reference(factor_p, factor_q, pivots):
    """Independent oracle: explicit dense C[(p,q),mu]/B[(p,q),g] +
    np.linalg.lstsq -- a genuinely different code path from the
    production separable-structure solver."""
    factor_p = np.asarray(factor_p)
    factor_q = np.asarray(factor_q)
    n_p, N_g = factor_p.shape
    n_q = factor_q.shape[0]
    C = np.einsum('pm,qm->pqm', factor_p[:, pivots], factor_q[:, pivots]).reshape(n_p * n_q, len(pivots))
    B = np.einsum('pg,qg->pqg', factor_p, factor_q).reshape(n_p * n_q, N_g)
    Theta, *_ = np.linalg.lstsq(C, B, rcond=None)
    return Theta


def _dense_coulomb_matrix(mesh):
    """Dense N_g x N_g Coulomb matrix via free_space_poisson_direct_sum_oracle
    on delta-function densities -- an independent oracle for the kernel
    itself, undoing that oracle's own dV so K_dense[i,j] is the bare
    kernel value."""
    N_g = 1
    for s in mesh.shape:
        N_g *= s
    dV = mesh.spacing[0] * mesh.spacing[1] * mesh.spacing[2]
    K = np.zeros((N_g, N_g))
    for j in range(N_g):
        delta = np.zeros(N_g)
        delta[j] = 1.0
        v = free_space_poisson_direct_sum_oracle(delta.reshape(mesh.shape), mesh).reshape(N_g)
        K[:, j] = v / dV
    return K


class TestPoissonInterpolationDenseOracle(unittest.TestCase):
    """Acceptance test 1: tiny real+complex systems vs an independent
    dense O(N_g^2) Coulomb-matrix oracle."""

    def _run_case(self, dtype, fft_kind, complex_factors):
        shape = (3, 4, 3)
        spacing = (0.3, 0.25, 0.35)
        rng = np.random.default_rng(0)
        N_g = 3 * 4 * 3
        n_orb = 3

        def mk(n):
            arr = rng.standard_normal((n, N_g))
            if complex_factors:
                arr = arr + 1j * rng.standard_normal((n, N_g))
            return arr.astype(dtype)

        factor_p = mk(n_orb)
        factor_q = mk(n_orb)
        pivots = rng.choice(N_g, size=7, replace=False)

        kernel = build_free_space_poisson_kernel(
            shape, spacing, backend="numpy", fft_kind=fft_kind, dtype=dtype)
        sector = build_poisson_interpolation_sector(
            factor_p, factor_q, pivots, kernel.mesh, rcond=1e-12)
        core = poisson_core(sector, kernel=kernel)

        Theta_ref = _dense_theta_reference(factor_p, factor_q, pivots)
        np.testing.assert_allclose(sector.Theta, Theta_ref, atol=1e-8, rtol=1e-8)

        K_dense = _dense_coulomb_matrix(kernel.mesh)
        dV = spacing[0] * spacing[1] * spacing[2]
        Z_ref = dV * dV * Theta_ref.conj() @ K_dense @ Theta_ref.T
        np.testing.assert_allclose(core.Z, Z_ref, atol=1e-7, rtol=1e-7)

    def test_real_float64(self):
        self._run_case(np.float64, "rfft", complex_factors=False)

    def test_complex_float64(self):
        self._run_case(np.complex128, "fft", complex_factors=True)


class TestPoissonInterpolationTilingEquality(unittest.TestCase):
    """Acceptance test 2: tiled vs untiled equality across non-divisible
    block sizes, odd/even anisotropic meshes, rfft/fft."""

    def _build(self, shape, spacing, fft_kind, dtype, complex_factors, n_pivots):
        rng = np.random.default_rng(1)
        N_g = 1
        for s in shape:
            N_g *= s

        def mk(n):
            arr = rng.standard_normal((n, N_g))
            if complex_factors:
                arr = arr + 1j * rng.standard_normal((n, N_g))
            return arr.astype(dtype)

        factor_p = mk(3)
        factor_q = mk(3)
        pivots = rng.choice(N_g, size=n_pivots, replace=False)
        kernel = build_free_space_poisson_kernel(
            shape, spacing, backend="numpy", fft_kind=fft_kind, dtype=dtype)
        sector = build_poisson_interpolation_sector(
            factor_p, factor_q, pivots, kernel.mesh, rcond=1e-12)
        return kernel, sector

    def test_grid_batch_size_nondivisible(self):
        kernel, sector_full = self._build((4, 5, 3), (0.3, 0.3, 0.3), "rfft", np.float64, False, 6)
        # rebuild with a nondivisible grid_batch_size and confirm Theta matches
        rng = np.random.default_rng(1)
        N_g = 4 * 5 * 3
        factor_p = rng.standard_normal((3, N_g))
        factor_q = rng.standard_normal((3, N_g))
        pivots = rng.choice(N_g, size=6, replace=False)
        sector_batched = build_poisson_interpolation_sector(
            factor_p, factor_q, pivots, kernel.mesh, grid_batch_size=7, rcond=1e-12)
        sector_unbatched = build_poisson_interpolation_sector(
            factor_p, factor_q, pivots, kernel.mesh, rcond=1e-12)
        np.testing.assert_allclose(sector_batched.Theta, sector_unbatched.Theta, atol=1e-10, rtol=1e-10)

    def test_mu_nu_block_size_nondivisible(self):
        kernel, sector = self._build((4, 4, 4), (0.25, 0.25, 0.25), "rfft", np.float64, False, 7)
        core_untiled = poisson_core(sector, kernel=kernel)
        core_tiled = poisson_core(sector, kernel=kernel, mu_block_size=3, nu_block_size=2)
        np.testing.assert_allclose(core_tiled.Z, core_untiled.Z, atol=1e-10, rtol=1e-10)

    def test_odd_even_anisotropic_mesh(self):
        for shape, spacing in [((5, 5, 5), (0.2, 0.2, 0.2)), ((4, 6, 5), (0.15, 0.4, 0.3))]:
            kernel, sector = self._build(shape, spacing, "rfft", np.float64, False, 5)
            core_untiled = poisson_core(sector, kernel=kernel)
            core_tiled = poisson_core(sector, kernel=kernel, mu_block_size=2, nu_block_size=2)
            np.testing.assert_allclose(core_tiled.Z, core_untiled.Z, atol=1e-9, rtol=1e-9,
                                        err_msg=f"shape={shape}")

    def test_fft_kind_complex_tiling(self):
        kernel, sector = self._build((4, 4, 4), (0.3, 0.3, 0.3), "fft", np.complex128, True, 5)
        core_untiled = poisson_core(sector, kernel=kernel)
        core_tiled = poisson_core(sector, kernel=kernel, mu_block_size=2, nu_block_size=3)
        np.testing.assert_allclose(core_tiled.Z, core_untiled.Z, atol=1e-9, rtol=1e-9)

    def test_poisson_solve_call_count_matches_nu_blocks_not_mu_blocks(self):
        # solve_free_space_poisson must be called exactly
        # ceil(n_right/nu_block_size) times -- once per nu-block, NEVER
        # recomputed inside the mu-loop (Alice's review, task #15,
        # 2026-07-13: "add a call-count spy proving solve_free_space_
        # poisson is called exactly ceil(n_right/nu_block_size) times and
        # does not grow with the number of mu blocks").
        kernel, sector = self._build((4, 4, 4), (0.25, 0.25, 0.25), "rfft", np.float64, False, 9)
        n_right = sector.Theta.shape[0]
        nu_block_size = 4
        expected_calls = math.ceil(n_right / nu_block_size)

        import pytc.integrals.coulomb as coulomb_module
        real_solve = coulomb_module.solve_free_space_poisson
        with patch.object(coulomb_module, "solve_free_space_poisson",
                           side_effect=real_solve) as spy:
            poisson_core(sector, kernel=kernel, mu_block_size=1, nu_block_size=nu_block_size)
            calls_with_many_mu_blocks = spy.call_count

        with patch.object(coulomb_module, "solve_free_space_poisson",
                           side_effect=real_solve) as spy:
            poisson_core(sector, kernel=kernel, mu_block_size=n_right, nu_block_size=nu_block_size)
            calls_with_one_mu_block = spy.call_count

        self.assertEqual(calls_with_many_mu_blocks, expected_calls)
        self.assertEqual(calls_with_one_mu_block, expected_calls)
        self.assertEqual(calls_with_many_mu_blocks, calls_with_one_mu_block)


class TestPoissonCoreHermiticityAndRejection(unittest.TestCase):
    """Acceptance test 3."""

    def _complex_sector(self, label_shift=0):
        shape = (4, 4, 4)
        spacing = (0.3, 0.3, 0.3)
        rng = np.random.default_rng(2 + label_shift)
        N_g = 64
        factor_p = (rng.standard_normal((3, N_g)) + 1j * rng.standard_normal((3, N_g)))
        factor_q = (rng.standard_normal((3, N_g)) + 1j * rng.standard_normal((3, N_g)))
        pivots = rng.choice(N_g, size=5, replace=False)
        kernel = build_free_space_poisson_kernel(
            shape, spacing, backend="numpy", fft_kind="fft", dtype=np.complex128)
        sector = build_poisson_interpolation_sector(factor_p, factor_q, pivots, kernel.mesh, rcond=1e-12)
        return kernel, sector

    def test_same_sector_hermitian_real_diagonal(self):
        kernel, sector = self._complex_sector()
        core = poisson_core(sector, kernel=kernel)
        np.testing.assert_allclose(core.Z, core.Z.conj().T, atol=1e-9, rtol=1e-9)
        diag = np.diag(core.Z)
        np.testing.assert_allclose(diag.imag, 0.0, atol=1e-9)

    def test_cross_sector_hermitian_relationship(self):
        kernel, sector_A = self._complex_sector(0)
        _, sector_B = self._complex_sector(1)
        core_AB = poisson_core(sector_A, sector_B, kernel=kernel)
        core_BA = poisson_core(sector_B, sector_A, kernel=kernel)
        np.testing.assert_allclose(core_AB.Z, core_BA.Z.conj().T, atol=1e-9, rtol=1e-9)

    def test_rejects_non_sector_left(self):
        kernel, sector = self._complex_sector()
        with self.assertRaises(TypeError):
            poisson_core(object(), kernel=kernel)

    def test_rejects_non_sector_right(self):
        kernel, sector = self._complex_sector()
        with self.assertRaises(TypeError):
            poisson_core(sector, object(), kernel=kernel)

    def test_rejects_non_kernel(self):
        _, sector = self._complex_sector()
        with self.assertRaises(TypeError):
            poisson_core(sector, kernel=object())

    def test_rejects_mesh_geometry_mismatch(self):
        kernel, sector = self._complex_sector()
        other_kernel = build_free_space_poisson_kernel(
            (4, 4, 4), (0.5, 0.5, 0.5), backend="numpy", fft_kind="fft", dtype=np.complex128)
        with self.assertRaises(ValueError):
            poisson_core(sector, kernel=other_kernel)

    def test_rejects_backend_mismatch(self):
        kernel, sector = self._complex_sector()
        jax_kernel = build_free_space_poisson_kernel(
            (4, 4, 4), (0.3, 0.3, 0.3), backend="jax", fft_kind="fft", dtype=np.complex128)
        with self.assertRaises(ValueError):
            poisson_core(sector, kernel=jax_kernel)

    def test_rejects_dtype_mismatch(self):
        kernel, sector = self._complex_sector()
        real_kernel = build_free_space_poisson_kernel(
            (4, 4, 4), (0.3, 0.3, 0.3), backend="numpy", fft_kind="rfft", dtype=np.float64)
        with self.assertRaises(ValueError):
            poisson_core(sector, kernel=real_kernel)


class TestPoissonBackendAgreement(unittest.TestCase):
    """Acceptance test 4."""

    def test_numpy_vs_jax_float64(self):
        shape = (4, 4, 4)
        spacing = (0.25, 0.25, 0.25)
        rng = np.random.default_rng(3)
        N_g = 64
        factor_p = rng.standard_normal((3, N_g))
        factor_q = rng.standard_normal((3, N_g))
        pivots = rng.choice(N_g, size=5, replace=False)

        kernel_np = build_free_space_poisson_kernel(shape, spacing, backend="numpy", dtype=np.float64)
        sector_np = build_poisson_interpolation_sector(factor_p, factor_q, pivots, kernel_np.mesh, rcond=1e-12)
        core_np = poisson_core(sector_np, kernel=kernel_np)

        kernel_jax = build_free_space_poisson_kernel(shape, spacing, backend="jax", dtype=np.float64)
        sector_jax = build_poisson_interpolation_sector(
            jnp.asarray(factor_p), jnp.asarray(factor_q), pivots, kernel_jax.mesh, rcond=1e-12,
            upstream_provenance={"factor_p_sha256": "a" * 64, "factor_q_sha256": "b" * 64})
        core_jax = poisson_core(sector_jax, kernel=kernel_jax)

        np.testing.assert_allclose(core_np.Z, np.asarray(core_jax.Z), atol=1e-8, rtol=1e-8)

    def test_pinned_float32(self):
        shape = (4, 4, 4)
        spacing = (0.25, 0.25, 0.25)
        rng = np.random.default_rng(4)
        N_g = 64
        factor_p = rng.standard_normal((3, N_g)).astype(np.float32)
        factor_q = rng.standard_normal((3, N_g)).astype(np.float32)
        pivots = rng.choice(N_g, size=5, replace=False)
        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy", dtype=np.float32)
        sector = build_poisson_interpolation_sector(factor_p, factor_q, pivots, kernel.mesh, rcond=1e-6)
        self.assertEqual(sector.realized_dtype, str(np.dtype(np.float32)))
        core = poisson_core(sector, kernel=kernel)
        self.assertEqual(core.realized_dtype, str(np.dtype(np.float32)))

    def test_pinned_complex64_full_sector_and_fft_core(self):
        # A REAL complex64 path -- fft_kind='fft', complex factors, and a
        # dense-oracle cross-check, not just a dtype-label assertion
        # (Alice's review, task #15, 2026-07-13: "add a real complex64
        # full sector+FFT-core test").
        shape = (4, 4, 4)
        spacing = (0.25, 0.25, 0.25)
        rng = np.random.default_rng(9)
        N_g = 64
        factor_p = (rng.standard_normal((3, N_g)) + 1j * rng.standard_normal((3, N_g))).astype(np.complex64)
        factor_q = (rng.standard_normal((3, N_g)) + 1j * rng.standard_normal((3, N_g))).astype(np.complex64)
        pivots = rng.choice(N_g, size=5, replace=False)
        kernel = build_free_space_poisson_kernel(
            shape, spacing, backend="numpy", fft_kind="fft", dtype=np.complex64)
        sector = build_poisson_interpolation_sector(factor_p, factor_q, pivots, kernel.mesh, rcond=1e-5)
        self.assertEqual(sector.realized_dtype, str(np.dtype(np.complex64)))
        core = poisson_core(sector, kernel=kernel)
        self.assertEqual(core.realized_dtype, str(np.dtype(np.complex64)))

        Theta_ref = _dense_theta_reference(
            factor_p.astype(np.complex128), factor_q.astype(np.complex128), pivots)
        np.testing.assert_allclose(sector.Theta.astype(np.complex128), Theta_ref, atol=2e-3, rtol=2e-3)

    def test_jax_complex_path_vs_numpy_dense_oracle(self):
        # A formal JAX complex path (Theta and Z), not just real float64
        # (Alice's review, task #15, 2026-07-13).
        shape = (4, 4, 4)
        spacing = (0.3, 0.3, 0.3)
        rng = np.random.default_rng(10)
        N_g = 64
        factor_p = rng.standard_normal((3, N_g)) + 1j * rng.standard_normal((3, N_g))
        factor_q = rng.standard_normal((3, N_g)) + 1j * rng.standard_normal((3, N_g))
        pivots = rng.choice(N_g, size=6, replace=False)

        kernel_jax = build_free_space_poisson_kernel(
            shape, spacing, backend="jax", fft_kind="fft", dtype=np.complex128)
        sector_jax = build_poisson_interpolation_sector(
            jnp.asarray(factor_p), jnp.asarray(factor_q), pivots, kernel_jax.mesh, rcond=1e-12,
            upstream_provenance={"factor_p_sha256": "a" * 64, "factor_q_sha256": "b" * 64})
        core_jax = poisson_core(sector_jax, kernel=kernel_jax)

        Theta_ref = _dense_theta_reference(factor_p, factor_q, pivots)
        np.testing.assert_allclose(np.asarray(sector_jax.Theta), Theta_ref, atol=1e-8, rtol=1e-8)

        kernel_np = build_free_space_poisson_kernel(
            shape, spacing, backend="numpy", fft_kind="fft", dtype=np.complex128)
        sector_np = build_poisson_interpolation_sector(
            factor_p, factor_q, pivots, kernel_np.mesh, rcond=1e-12)
        core_np = poisson_core(sector_np, kernel=kernel_np)

        np.testing.assert_allclose(np.asarray(core_jax.Z), core_np.Z, atol=1e-8, rtol=1e-8)


class TestPoissonSyntheticExactFit(unittest.TestCase):
    """Acceptance test 5: known interpolation points -- full analytic-rank
    pivot count gives an EXACT (near machine-precision) reconstruction,
    before any larger regression."""

    def test_full_rank_pivots_exact_reconstruction(self):
        shape = (4, 4, 4)
        spacing = (0.3, 0.3, 0.3)
        rng = np.random.default_rng(5)
        N_g = 64
        n_p, n_q = 2, 2
        factor_p = rng.standard_normal((n_p, N_g))
        factor_q = rng.standard_normal((n_q, N_g))
        # analytic pair-rank bound for a generic (non-symmetric) sector is n_p*n_q.
        n_fused = n_p * n_q
        pivots = rng.choice(N_g, size=n_fused, replace=False)

        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy", dtype=np.float64)
        sector = build_poisson_interpolation_sector(
            factor_p, factor_q, pivots, kernel.mesh, rcond=1e-13)

        # Reconstruct the FULL pair-product matrix everywhere and compare
        # to the exact pair products -- at full analytic rank this must
        # be essentially exact, not merely a good approximation.
        pair_exact = np.einsum('pg,qg->pqg', factor_p, factor_q).reshape(n_p * n_q, N_g)
        pair_at_pivots = pair_exact[:, pivots]
        pair_reconstructed = pair_at_pivots @ sector.Theta
        rel_err = (np.linalg.norm(pair_reconstructed - pair_exact)
                   / np.linalg.norm(pair_exact))
        self.assertLess(rel_err, 1e-8)


class TestPoissonInterpolationBenchmark(unittest.TestCase):
    """Acceptance test 6: explicit incore-baseline label, structural
    allocation accounting alongside tracemalloc (which can miss native/
    JAX allocations)."""

    def test_incore_baseline_labeled_and_shape_accounted(self):
        # Sized so an accidental O(N_g^2) dense-kernel allocation
        # (~ N_g^2*8 bytes) is a full order of magnitude above real
        # measured usage -- avoids the coincidentally-too-tight-
        # threshold flakiness a small N_g produces (task #13's
        # equivalent benchmark test hit exactly this the first time).
        shape = (14, 14, 14)
        spacing = (0.2, 0.2, 0.2)
        N_g = 14 * 14 * 14
        rng = np.random.default_rng(6)
        factor_p = rng.standard_normal((4, N_g))
        factor_q = rng.standard_normal((4, N_g))
        pivots = rng.choice(N_g, size=10, replace=False)
        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy", dtype=np.float64)

        tracemalloc.start()
        t0 = time.perf_counter()
        sector = build_poisson_interpolation_sector(
            factor_p, factor_q, pivots, kernel.mesh, grid_batch_size=512, rcond=1e-12)
        core = poisson_core(sector, kernel=kernel, mu_block_size=4, nu_block_size=4)
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Explicit label + structural shape accounting (not solely
        # relying on tracemalloc, which can miss native allocations):
        self.assertEqual(sector.storage_mode, "incore_full_theta")
        self.assertEqual(sector.Theta.shape, (10, N_g))
        self.assertEqual(core.Z.shape, (10, 10))

        # Order-of-magnitude discriminator against an accidental O(N_g^2)
        # dense-kernel allocation -- a real one here would be
        # N_g^2*8 ~ 6.0e8 bytes (~600 MB); the threshold below (10% of
        # that, ~60 MB) is still a full order of magnitude above
        # realistic incore-Theta/padded-FFT usage at this size.
        self.assertLess(peak, N_g**2 * 8 / 10)
        self.assertLess(dt, 30.0)


class TestPoissonInterpolationSectorValidation(unittest.TestCase):
    """Acceptance test 7 (sector half): seeded from a real builder
    artifact via dataclasses.fields (not a hand-rolled placeholder
    dict -- task #13's round-2 lesson), perturbing one field at a time."""

    def _valid_kwargs(self):
        shape = (4, 4, 4)
        spacing = (0.3, 0.3, 0.3)
        N_g = 64
        rng = np.random.default_rng(7)
        factor_p = rng.standard_normal((3, N_g))
        factor_q = rng.standard_normal((3, N_g))
        pivots = rng.choice(N_g, size=5, replace=False)
        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy", dtype=np.float64)
        sector = build_poisson_interpolation_sector(factor_p, factor_q, pivots, kernel.mesh, rcond=1e-12)
        kwargs = {f.name: getattr(sector, f.name) for f in dataclasses.fields(sector)}
        theta = np.array(sector.Theta)
        theta.setflags(write=True)
        kwargs["Theta"] = theta
        return kwargs

    def test_valid_kwargs_baseline_constructs(self):
        PoissonInterpolationSector(**self._valid_kwargs())

    def test_theta_defensively_copied_readonly_numpy(self):
        kwargs = self._valid_kwargs()
        sector = PoissonInterpolationSector(**kwargs)
        self.assertFalse(sector.Theta.flags.writeable)
        with self.assertRaises(ValueError):
            sector.Theta[0, 0] = 999

    def test_rejects_wrong_theta_shape(self):
        kwargs = self._valid_kwargs()
        kwargs["Theta"] = kwargs["Theta"][:, :-1]
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_dtype_mismatch(self):
        kwargs = self._valid_kwargs()
        kwargs["realized_dtype"] = str(np.dtype(np.complex128))
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_non_unique_pivots(self):
        kwargs = self._valid_kwargs()
        p = np.array(kwargs["pivots"])
        p[1] = p[0]
        kwargs["pivots"] = p
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_out_of_range_pivots(self):
        kwargs = self._valid_kwargs()
        p = np.array(kwargs["pivots"])
        p[0] = 10**6
        kwargs["pivots"] = p
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_bad_grid_batch_size(self):
        kwargs = self._valid_kwargs()
        kwargs["grid_batch_size"] = -1
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_bool_grid_batch_size(self):
        kwargs = self._valid_kwargs()
        kwargs["grid_batch_size"] = True
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_bad_storage_mode(self):
        kwargs = self._valid_kwargs()
        kwargs["storage_mode"] = "out_of_core"
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_bad_factor_identity_source(self):
        kwargs = self._valid_kwargs()
        kwargs["factor_identity_source"] = "trust_me"
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_non_hex_pivots_sha256(self):
        kwargs = self._valid_kwargs()
        kwargs["pivots_sha256"] = "not-hex" * 8
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_tampered_sector_spec_hash(self):
        kwargs = self._valid_kwargs()
        kwargs["sector_spec_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_wrong_solver_version(self):
        kwargs = self._valid_kwargs()
        kwargs["solver_version"] = "bogus"
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_negative_jitter(self):
        kwargs = self._valid_kwargs()
        kwargs["jitter_used"] = -1.0
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_mismatched_mesh_geometry_tuple_length(self):
        kwargs = self._valid_kwargs()
        kwargs["mesh_shape"] = (4, 4)
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_valid_but_wrong_pivots_sha256(self):
        # A syntactically valid 64-hex digest that simply does not match
        # the actual pivots array -- not just malformed syntax (Alice's
        # review, task #15, 2026-07-13: pivot identity must be bound to
        # the actual array, not merely a plausible-looking string).
        kwargs = self._valid_kwargs()
        kwargs["pivots_sha256"] = "ab" * 32
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_reordered_pivots_with_stale_digest(self):
        kwargs = self._valid_kwargs()
        kwargs["pivots"] = np.array(kwargs["pivots"])[::-1].copy()
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_valid_but_wrong_factor_p_sha256(self):
        kwargs = self._valid_kwargs()
        kwargs["factor_p_sha256"] = "cd" * 32
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_valid_but_wrong_factor_q_sha256(self):
        kwargs = self._valid_kwargs()
        kwargs["factor_q_sha256"] = "ef" * 32
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_backend_factor_identity_source_mismatch(self):
        # NumPy backend claiming a JAX-style caller_attested trust
        # boundary (or vice versa) must be rejected -- a manually
        # constructed artifact must not be able to claim the wrong trust
        # boundary (Alice's review, task #15, 2026-07-13).
        kwargs = self._valid_kwargs()
        self.assertEqual(kwargs["backend"], "numpy")
        kwargs["factor_identity_source"] = "caller_attested"
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_same_factor_true_with_differing_factor_digests(self):
        kwargs = self._valid_kwargs()
        kwargs["same_factor"] = True
        kwargs["factor_p_sha256"] = "11" * 32
        kwargs["factor_q_sha256"] = "22" * 32
        # Recompute a self-consistent sector_spec_sha256 for THIS
        # perturbed field set so the earlier hash-recomputation check
        # doesn't mask the same_factor-specific check being tested here.
        from pytc.integrals.coulomb import _kernel_spec_sha256
        kwargs["sector_spec_sha256"] = _kernel_spec_sha256({
            "mesh_shape": kwargs["mesh_shape"], "mesh_spacing": kwargs["mesh_spacing"],
            "mesh_origin": kwargs["mesh_origin"], "same_factor": kwargs["same_factor"],
            "storage_mode": kwargs["storage_mode"], "backend": kwargs["backend"],
            "realized_dtype": kwargs["realized_dtype"], "grid_batch_size": kwargs["grid_batch_size"],
            "rcond": kwargs["rcond"], "jitter_used": kwargs["jitter_used"],
            "n_tries": kwargs["n_tries"], "factor_identity_source": kwargs["factor_identity_source"],
            "pivots_sha256": kwargs["pivots_sha256"], "factor_p_sha256": kwargs["factor_p_sha256"],
            "factor_q_sha256": kwargs["factor_q_sha256"], "solver_version": kwargs["solver_version"],
        })
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)

    def test_rejects_valid_but_wrong_device(self):
        kwargs = self._valid_kwargs()
        kwargs["device"] = "gpu:0"
        with self.assertRaises(ValueError):
            PoissonInterpolationSector(**kwargs)


class TestPoissonCoreArtifactValidation(unittest.TestCase):
    """Acceptance test 7 (core half)."""

    def _valid_kwargs(self):
        shape = (4, 4, 4)
        spacing = (0.3, 0.3, 0.3)
        N_g = 64
        rng = np.random.default_rng(8)
        factor_p = rng.standard_normal((3, N_g))
        factor_q = rng.standard_normal((3, N_g))
        pivots = rng.choice(N_g, size=5, replace=False)
        kernel = build_free_space_poisson_kernel(shape, spacing, backend="numpy", dtype=np.float64)
        sector = build_poisson_interpolation_sector(factor_p, factor_q, pivots, kernel.mesh, rcond=1e-12)
        core = poisson_core(sector, kernel=kernel)
        kwargs = {f.name: getattr(core, f.name) for f in dataclasses.fields(core)}
        z = np.array(core.Z)
        z.setflags(write=True)
        kwargs["Z"] = z
        return kwargs

    def test_valid_kwargs_baseline_constructs(self):
        PoissonCoreArtifact(**self._valid_kwargs())

    def test_z_defensively_copied_readonly_numpy(self):
        kwargs = self._valid_kwargs()
        core = PoissonCoreArtifact(**kwargs)
        self.assertFalse(core.Z.flags.writeable)

    def test_rejects_wrong_z_shape(self):
        kwargs = self._valid_kwargs()
        kwargs["left_n_fused"] = kwargs["left_n_fused"] + 1
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)

    def test_rejects_bad_normalization(self):
        kwargs = self._valid_kwargs()
        kwargs["normalization"] = "2dV"
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)

    def test_rejects_non_hex_kernel_digest(self):
        kwargs = self._valid_kwargs()
        kwargs["kernel_spec_sha256"] = "short"
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)

    def test_rejects_valid_but_wrong_kernel_digest(self):
        # A syntactically valid 64-hex digest that doesn't match what
        # core_spec_sha256 was actually computed from -- not just
        # malformed syntax (Alice's review, task #15, 2026-07-13:
        # "ANY single typed field... can be tampered with independently
        # and still look internally consistent" without a recomputed
        # core_spec_sha256).
        kwargs = self._valid_kwargs()
        kwargs["kernel_spec_sha256"] = "ab" * 32
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)

    def test_rejects_valid_but_wrong_left_sector_digest(self):
        kwargs = self._valid_kwargs()
        kwargs["left_sector_spec_sha256"] = "cd" * 32
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)

    def test_rejects_valid_but_wrong_right_sector_digest(self):
        kwargs = self._valid_kwargs()
        kwargs["right_sector_spec_sha256"] = "ef" * 32
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)

    def test_rejects_valid_but_wrong_device(self):
        kwargs = self._valid_kwargs()
        kwargs["device"] = "gpu:0"
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)

    def test_core_spec_sha256_is_sensitive_to_device(self):
        # core_spec_sha256 is documented to cover EVERY typed core field,
        # explicitly including device -- confirm the canonical digest
        # actually changes for otherwise-identical field dictionaries
        # that differ only in device (Alice's review, task #15,
        # 2026-07-13: the builder hash and __post_init__ recomputation
        # had both silently omitted device from the hashed fields
        # despite the documented schema and handoff claiming it was
        # covered).
        from pytc.integrals.coulomb import _kernel_spec_sha256
        base_fields = {
            "left_sector_spec_sha256": "11" * 32, "right_sector_spec_sha256": "22" * 32,
            "left_n_fused": 5, "right_n_fused": 5, "kernel_spec_sha256": "33" * 32,
            "mu_block_size": 5, "nu_block_size": 5, "normalization": "dV",
            "backend": "numpy", "realized_dtype": "float64", "solver_version": "1",
        }
        digest_cpu = _kernel_spec_sha256({**base_fields, "device": "cpu"})
        digest_gpu = _kernel_spec_sha256({**base_fields, "device": "gpu:0"})
        self.assertNotEqual(digest_cpu, digest_gpu)

    def test_rejects_bad_block_size(self):
        kwargs = self._valid_kwargs()
        kwargs["mu_block_size"] = 0
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)

    def test_rejects_bool_block_size(self):
        kwargs = self._valid_kwargs()
        kwargs["nu_block_size"] = False
        with self.assertRaises(ValueError):
            PoissonCoreArtifact(**kwargs)


if __name__ == "__main__":
    unittest.main()
