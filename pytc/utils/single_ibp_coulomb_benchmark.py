"""Experimental single-integration-by-parts Coulomb reference and benchmark.

This module is deliberately *not* a production Coulomb policy.  It provides a
small, NumPy-only implementation used to decide whether

    (rho_A | rho_B) = -1/2 int grad(rho_A)^* . (rhat * rho_B) dr

is a useful replacement for point-sampled ``1/r`` quadrature on a uniform
Cartesian mesh.  The vector kernel is evaluated by the same zero-padded linear
convolution convention as :mod:`pytc.integrals.coulomb`; its centered-cell
self contribution is exactly zero by inversion symmetry.

The molecular benchmark holds orbitals, mesh, pivots, and pair-collocation
matrix fixed across analytic DF, direct ``1/r``, and single-IBP cores.  The
derivative interpolation vectors are obtained by differentiating the pair
products analytically and applying the *same fixed-pivot normal-equation
factorization* as the ordinary interpolation vectors.  No finite differences
are used.

Task #18, #proj-isdf-coulomb-cuda, 2026-07-13.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Any

import numpy as np
from pyscf import dft, gto, mp, scf

from pytc.df.solvers import (
    prepare_normal_equations_solver,
    solve_normal_equations_batch_prepared,
)
from pytc.integrals.coulomb import (
    FreeSpacePoissonMesh,
    _wrapped_integer_offsets,
    build_free_space_poisson_kernel,
    build_poisson_interpolation_sector,
    compute_C_streamed,
    compute_Z,
    pair_collocation_at_pivots,
    poisson_core,
    reconstruct_eri_block,
    select_sector_pivots,
    solve_free_space_poisson,
    weight_mo_values,
)
from pytc.utils.poisson_core_benchmark import (
    SYSTEMS,
    centered_uniform_mesh,
    mp2_energy_from_ovov,
)


@dataclasses.dataclass(frozen=True)
class SingleIBPKernel:
    """Cached NumPy vector-kernel spectrum for the experimental solver."""

    mesh: FreeSpacePoissonMesh
    spectrum: np.ndarray
    fft_kind: str
    input_dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, FreeSpacePoissonMesh):
            raise TypeError("mesh must be a FreeSpacePoissonMesh")
        if self.fft_kind not in ("rfft", "fft"):
            raise ValueError("fft_kind must be 'rfft' or 'fft'")
        dtype = np.dtype(self.input_dtype)
        if self.fft_kind == "rfft" and dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("rfft single-IBP kernels require float32 or float64 input")
        if self.fft_kind == "fft" and dtype not in (np.dtype("complex64"), np.dtype("complex128")):
            raise ValueError("fft single-IBP kernels require complex64 or complex128 input")
        spectrum = np.asarray(self.spectrum)
        expected_spatial = (
            self.mesh.padded_shape[:-1] + (self.mesh.padded_shape[-1] // 2 + 1,)
            if self.fft_kind == "rfft"
            else self.mesh.padded_shape
        )
        if spectrum.shape != (3,) + expected_spatial:
            raise ValueError(
                f"spectrum.shape={spectrum.shape} != expected {(3,) + expected_spatial}"
            )
        spectrum = np.array(spectrum, copy=True)
        spectrum.setflags(write=False)
        object.__setattr__(self, "spectrum", spectrum)
        object.__setattr__(self, "input_dtype", str(dtype))


def build_single_ibp_kernel(mesh: FreeSpacePoissonMesh, *, fft_kind="rfft", dtype=None):
    """Build the zero-padded spectrum of ``r/|r|`` on ``mesh``.

    The zero-offset vector is exactly zero: the average of this odd vector
    field over any centered rectangular cell vanishes by inversion symmetry.
    """
    if not isinstance(mesh, FreeSpacePoissonMesh):
        raise TypeError("mesh must be a FreeSpacePoissonMesh")
    if fft_kind not in ("rfft", "fft"):
        raise ValueError("fft_kind must be 'rfft' or 'fft'")
    dtype = np.dtype(
        dtype if dtype is not None else (np.float64 if fft_kind == "rfft" else np.complex128)
    )
    if fft_kind == "rfft" and dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("rfft single-IBP kernels require float32 or float64 input")
    if fft_kind == "fft" and dtype not in (np.dtype("complex64"), np.dtype("complex128")):
        raise ValueError("fft single-IBP kernels require complex64 or complex128 input")
    real_dtype = np.float32 if dtype in (np.dtype("float32"), np.dtype("complex64")) else np.float64

    offsets = [
        _wrapped_integer_offsets(p).astype(real_dtype) * spacing
        for p, spacing in zip(mesh.padded_shape, mesh.spacing)
    ]
    ox = offsets[0][:, None, None]
    oy = offsets[1][None, :, None]
    oz = offsets[2][None, None, :]
    radius = np.sqrt(ox * ox + oy * oy + oz * oz)
    vectors = np.zeros((3,) + mesh.padded_shape, dtype=real_dtype)
    np.divide(ox, radius, out=vectors[0], where=radius != 0)
    np.divide(oy, radius, out=vectors[1], where=radius != 0)
    np.divide(oz, radius, out=vectors[2], where=radius != 0)

    axes = (-3, -2, -1)
    if fft_kind == "rfft":
        spectrum = np.fft.rfftn(vectors, s=mesh.padded_shape, axes=axes, norm="backward")
    else:
        spectrum = np.fft.fftn(
            vectors.astype(dtype), s=mesh.padded_shape, axes=axes, norm="backward"
        )
    return SingleIBPKernel(mesh=mesh, spectrum=spectrum, fft_kind=fft_kind,
                           input_dtype=str(dtype))


def solve_single_ibp_vector(rho: np.ndarray, kernel: SingleIBPKernel) -> np.ndarray:
    """Return ``dV sum_j rhat_(i-j) rho_j`` with a component axis before xyz."""
    if not isinstance(kernel, SingleIBPKernel):
        raise TypeError("kernel must be a SingleIBPKernel")
    if not isinstance(rho, np.ndarray):
        raise TypeError("the experimental single-IBP solver accepts NumPy arrays only")
    if rho.dtype != np.dtype(kernel.input_dtype):
        raise ValueError(f"rho.dtype={rho.dtype} != kernel.input_dtype={kernel.input_dtype}")
    mesh = kernel.mesh
    if tuple(rho.shape[-3:]) != mesh.shape:
        raise ValueError(f"rho trailing shape {rho.shape[-3:]} != mesh.shape {mesh.shape}")
    rho_is_complex = np.issubdtype(rho.dtype, np.complexfloating)
    if rho_is_complex != (kernel.fft_kind == "fft"):
        raise ValueError("rho real/complex family does not match kernel.fft_kind")

    batch_shape = rho.shape[:-3]
    padded = np.zeros(batch_shape + mesh.padded_shape, dtype=rho.dtype)
    nx, ny, nz = mesh.shape
    padded[..., :nx, :ny, :nz] = rho
    axes = (-3, -2, -1)
    if kernel.fft_kind == "rfft":
        rho_hat = np.fft.rfftn(padded, s=mesh.padded_shape, axes=axes, norm="backward")
        vector_hat = rho_hat[..., None, :, :, :] * kernel.spectrum
        result = np.fft.irfftn(
            vector_hat, s=mesh.padded_shape, axes=axes, norm="backward"
        )
    else:
        rho_hat = np.fft.fftn(padded, s=mesh.padded_shape, axes=axes, norm="backward")
        vector_hat = rho_hat[..., None, :, :, :] * kernel.spectrum
        result = np.fft.ifftn(
            vector_hat, s=mesh.padded_shape, axes=axes, norm="backward"
        )
    dV = math.prod(mesh.spacing)
    return (result[..., :nx, :ny, :nz] * dV).astype(rho.dtype)


def single_ibp_direct_sum_oracle(rho: np.ndarray, mesh: FreeSpacePoissonMesh) -> np.ndarray:
    """Dense ``O(N_g^2)`` vector-convolution oracle for tiny tests only."""
    if not isinstance(mesh, FreeSpacePoissonMesh):
        raise TypeError("mesh must be a FreeSpacePoissonMesh")
    rho = np.asarray(rho)
    if tuple(rho.shape[-3:]) != mesh.shape:
        raise ValueError(f"rho trailing shape {rho.shape[-3:]} != mesh.shape {mesh.shape}")
    axes = [np.arange(n) * h for n, h in zip(mesh.shape, mesh.spacing)]
    xyz = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([x.reshape(-1) for x in xyz], axis=1)
    diff = coords[:, None, :] - coords[None, :, :]
    radius = np.linalg.norm(diff, axis=-1)
    rhat = np.zeros_like(diff)
    np.divide(diff, radius[..., None], out=rhat, where=radius[..., None] != 0)
    n_grid = coords.shape[0]
    rho_flat = rho.reshape(rho.shape[:-3] + (n_grid,))
    result = math.prod(mesh.spacing) * np.einsum("...j,ijc->...ci", rho_flat, rhat)
    return result.reshape(rho.shape[:-3] + (3,) + mesh.shape)


def single_ibp_core(gradient_theta_left: np.ndarray, theta_right: np.ndarray,
                    kernel: SingleIBPKernel, *, nu_block_size=None) -> np.ndarray:
    """Contract an unsymmetrized single-IBP core from analytic derivatives.

    ``gradient_theta_left`` has shape ``(n_mu,3,N_g)`` and ``theta_right``
    has shape ``(n_nu,N_g)``.  The returned expression is the literal
    one-sided discretization; it is intentionally not symmetrized so that
    dagger-symmetry error remains an observable rather than being hidden.
    """
    grad = np.asarray(gradient_theta_left)
    theta = np.asarray(theta_right)
    n_grid = math.prod(kernel.mesh.shape)
    if grad.ndim != 3 or grad.shape[1:] != (3, n_grid):
        raise ValueError(f"gradient_theta_left must have shape (n_mu,3,{n_grid})")
    if theta.ndim != 2 or theta.shape[1] != n_grid:
        raise ValueError(f"theta_right must have shape (n_nu,{n_grid})")
    if grad.dtype != theta.dtype or theta.dtype != np.dtype(kernel.input_dtype):
        raise ValueError("gradient_theta_left, theta_right, and kernel dtype must match")
    n_nu = theta.shape[0]
    block = n_nu if nu_block_size is None else int(nu_block_size)
    if block <= 0:
        raise ValueError("nu_block_size must be positive")
    z = np.empty((grad.shape[0], n_nu), dtype=theta.dtype)
    dV = math.prod(kernel.mesh.spacing)
    for start in range(0, n_nu, block):
        stop = min(start + block, n_nu)
        vector = solve_single_ibp_vector(
            theta[start:stop].reshape((stop - start,) + kernel.mesh.shape), kernel
        ).reshape(stop - start, 3, n_grid)
        z[:, start:stop] = -0.5 * dV * np.einsum(
            "mcg,ncg->mn", grad.conj(), vector, optimize=True
        )
    return z


def build_gradient_interpolation_vectors(
    factor_p: np.ndarray,
    factor_q: np.ndarray,
    gradient_p: np.ndarray,
    gradient_q: np.ndarray,
    pivots: np.ndarray,
    *,
    rcond=1e-14,
    grid_batch_size=None,
) -> np.ndarray:
    """Differentiate pair products and apply the fixed interpolation solve.

    ``gradient_p/q`` use shape ``(3,n_orb,N_g)``.  Pivots and the
    normal-equation factorization are constructed only from the undifferentiated
    factors; differentiation never changes the interpolation points or fit.
    """
    factor_p = np.asarray(factor_p)
    factor_q = np.asarray(factor_q)
    gradient_p = np.asarray(gradient_p)
    gradient_q = np.asarray(gradient_q)
    if factor_p.ndim != 2 or factor_q.ndim != 2:
        raise ValueError("factor_p and factor_q must be 2-D")
    if gradient_p.shape != (3,) + factor_p.shape:
        raise ValueError("gradient_p must have shape (3,) + factor_p.shape")
    if gradient_q.shape != (3,) + factor_q.shape:
        raise ValueError("gradient_q must have shape (3,) + factor_q.shape")
    if not (factor_p.dtype == factor_q.dtype == gradient_p.dtype == gradient_q.dtype):
        raise ValueError("all factor and gradient arrays must share one dtype")
    if factor_p.shape[1] != factor_q.shape[1]:
        raise ValueError("factor_p and factor_q must share the grid axis")
    pivots = np.asarray(pivots)
    if pivots.ndim != 1 or not np.issubdtype(pivots.dtype, np.integer):
        raise ValueError("pivots must be a 1-D integer array")
    n_grid = factor_p.shape[1]
    if pivots.size == 0 or np.unique(pivots).size != pivots.size:
        raise ValueError("pivots must be nonempty and unique")
    if pivots.min() < 0 or pivots.max() >= n_grid:
        raise ValueError("pivots are out of range")
    batch = n_grid if grid_batch_size is None else int(grid_batch_size)
    if batch <= 0:
        raise ValueError("grid_batch_size must be positive")

    p_piv = factor_p[:, pivots]
    q_piv = factor_q[:, pivots]
    chol, lower = prepare_normal_equations_solver(p_piv, q_piv, rcond=rcond)
    components = []
    for axis in range(3):
        chunks = []
        for start in range(0, n_grid, batch):
            stop = min(start + batch, n_grid)
            left = solve_normal_equations_batch_prepared(
                chol, lower, p_piv, q_piv,
                gradient_p[axis, :, start:stop], factor_q[:, start:stop],
            )
            right = solve_normal_equations_batch_prepared(
                chol, lower, p_piv, q_piv,
                factor_p[:, start:stop], gradient_q[axis, :, start:stop],
            )
            chunks.append(left + right)
        components.append(np.concatenate(chunks, axis=1))
    return np.stack(components, axis=1)


def _evaluate_mos_and_gradients(mol, mo_coeff, coords, batch_size):
    n_mo = mo_coeff.shape[1]
    n_grid = coords.shape[0]
    values = np.empty((n_mo, n_grid), dtype=np.result_type(mo_coeff, float))
    gradients = np.empty((3, n_mo, n_grid), dtype=values.dtype)
    for start in range(0, n_grid, batch_size):
        stop = min(start + batch_size, n_grid)
        ao = dft.numint.eval_ao(mol, coords[start:stop], deriv=1)
        values[:, start:stop] = (ao[0] @ mo_coeff).T
        for axis in range(3):
            gradients[axis, :, start:stop] = (ao[axis + 1] @ mo_coeff).T
    return values, gradients


@dataclasses.dataclass(frozen=True)
class SingleIBPBenchmarkCase:
    system: str = "H2O_ccpVDZ"
    spacing: float = 0.30
    margin: float = 5.0
    rank_factor: float = 4.0
    auxbasis: str = "weigend"
    grid_shift_fraction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ao_batch_size: int = 8192
    grid_batch_size: int = 8192
    nu_block_size: int = 8
    rcond: float = 1e-12

    def __post_init__(self):
        if self.system not in SYSTEMS:
            raise ValueError(f"unknown system {self.system!r}")
        for name in ("spacing", "margin", "rank_factor", "rcond"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("ao_batch_size", "grid_batch_size", "nu_block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        shift = tuple(float(x) for x in self.grid_shift_fraction)
        if len(shift) != 3 or any(not math.isfinite(x) or abs(x) > 0.5 for x in shift):
            raise ValueError("grid_shift_fraction must contain three values in [-0.5, 0.5]")
        object.__setattr__(self, "grid_shift_fraction", shift)


def _relative_error(value, reference):
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def run_single_ibp_case(case: SingleIBPBenchmarkCase) -> dict[str, Any]:
    """Run one fixed-factor H2O/benzene comparison in a fresh process."""
    spec = SYSTEMS[case.system]
    mol = gto.M(atom=spec["atom"], basis=spec["basis"], verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=case.auxbasis).run()
    mo_coeff = np.asarray(mf.mo_coeff)
    n_occ = mol.nelectron // 2
    n_vir = mo_coeff.shape[1] - n_occ
    shape, origin, coords = centered_uniform_mesh(
        mol, case.spacing, case.margin, case.grid_shift_fraction
    )
    values, gradients = _evaluate_mos_and_gradients(
        mol, mo_coeff, coords, case.ao_batch_size
    )
    occ, vir = np.ascontiguousarray(values[:n_occ]), np.ascontiguousarray(values[n_occ:])
    grad_occ = np.ascontiguousarray(gradients[:, :n_occ])
    grad_vir = np.ascontiguousarray(gradients[:, n_occ:])
    dV = case.spacing ** 3
    weights = np.full(coords.shape[0], dV)
    rank = min(int(math.ceil(case.rank_factor * mo_coeff.shape[1])), n_occ * n_vir)
    pivots = np.asarray(select_sector_pivots(
        weight_mo_values(occ, weights), weight_mo_values(vir, weights), rank
    ))
    p_matrix = pair_collocation_at_pivots(occ[:, pivots], vir[:, pivots])

    c = compute_C_streamed(
        mf, p_matrix, mo_coeff[:, :n_occ], mo_coeff[:, n_occ:], auxbasis=case.auxbasis
    )
    z_df, _ = compute_Z(p_matrix, c)
    eri_df = reconstruct_eri_block(p_matrix, z_df, p_matrix).reshape(
        n_occ, n_vir, n_occ, n_vir
    )
    eri_exact = mf.with_df.ao2mo(
        (mo_coeff[:, :n_occ], mo_coeff[:, n_occ:],
         mo_coeff[:, :n_occ], mo_coeff[:, n_occ:]), compact=False
    ).reshape(n_occ, n_vir, n_occ, n_vir)

    t0 = time.perf_counter()
    poisson_kernel = build_free_space_poisson_kernel(
        shape, (case.spacing,) * 3, origin=origin, fft_kind="rfft",
        backend="numpy", dtype=np.float64,
    )
    direct_kernel_build_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    sector = build_poisson_interpolation_sector(
        occ, vir, pivots, poisson_kernel.mesh,
        grid_batch_size=case.grid_batch_size, rcond=case.rcond,
    )
    theta_build_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    z_direct = poisson_core(
        sector, kernel=poisson_kernel, nu_block_size=case.nu_block_size
    ).Z
    direct_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    grad_theta = build_gradient_interpolation_vectors(
        occ, vir, grad_occ, grad_vir, pivots,
        rcond=case.rcond, grid_batch_size=case.grid_batch_size,
    )
    gradient_theta_build_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    ibp_kernel = build_single_ibp_kernel(poisson_kernel.mesh, dtype=np.float64)
    ibp_kernel_build_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    z_ibp = single_ibp_core(
        grad_theta, sector.Theta, ibp_kernel, nu_block_size=case.nu_block_size
    )
    ibp_seconds = time.perf_counter() - t0
    # Diagnostic only: retain the literal one-sided result above as the
    # primary observable, then show exactly how much a posteriori dagger
    # projection changes.  Never hide a quadrature defect by reporting only
    # the projected matrix.
    z_ibp_sym = 0.5 * (z_ibp + z_ibp.conj().T)

    eri_direct = reconstruct_eri_block(p_matrix, z_direct, p_matrix).reshape(
        n_occ, n_vir, n_occ, n_vir
    )
    eri_ibp = reconstruct_eri_block(p_matrix, z_ibp, p_matrix).reshape(
        n_occ, n_vir, n_occ, n_vir
    )
    eri_ibp_sym = reconstruct_eri_block(p_matrix, z_ibp_sym, p_matrix).reshape(
        n_occ, n_vir, n_occ, n_vir
    )
    e_exact = mp2_energy_from_ovov(eri_exact, mf.mo_energy, n_occ)
    e_df = mp2_energy_from_ovov(eri_df, mf.mo_energy, n_occ)
    e_direct = mp2_energy_from_ovov(eri_direct, mf.mo_energy, n_occ)
    e_ibp = mp2_energy_from_ovov(eri_ibp, mf.mo_energy, n_occ)
    e_ibp_sym = mp2_energy_from_ovov(eri_ibp_sym, mf.mo_energy, n_occ)
    boundary = np.zeros(shape, dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    theta_grid = np.asarray(sector.Theta).reshape((len(pivots),) + shape)
    grad_grid = grad_theta.reshape((len(pivots), 3) + shape)

    return {
        "case": dataclasses.asdict(case),
        "shape": shape,
        "origin": origin,
        "n_grid": int(coords.shape[0]),
        "n_pivots": int(len(pivots)),
        "pivots": pivots.tolist(),
        "poisson_kernel_spec_sha256": poisson_kernel.kernel_spec_sha256,
        "poisson_sector_spec_sha256": sector.sector_spec_sha256,
        "same_p_relative_eri_error_direct": _relative_error(eri_direct, eri_df),
        "same_p_relative_eri_error_single_ibp": _relative_error(eri_ibp, eri_df),
        "same_p_relative_eri_error_single_ibp_dagger_projected": _relative_error(
            eri_ibp_sym, eri_df
        ),
        "total_relative_eri_error_df": _relative_error(eri_df, eri_exact),
        "total_relative_eri_error_direct": _relative_error(eri_direct, eri_exact),
        "total_relative_eri_error_single_ibp": _relative_error(eri_ibp, eri_exact),
        "total_relative_eri_error_single_ibp_dagger_projected": _relative_error(
            eri_ibp_sym, eri_exact
        ),
        "mp2_hartree": {
            "exact": e_exact,
            "df_isdf": e_df,
            "direct": e_direct,
            "single_ibp": e_ibp,
            "single_ibp_dagger_projected": e_ibp_sym,
            "same_p_direct_delta_mha": 1000.0 * (e_direct - e_df),
            "same_p_single_ibp_delta_mha": 1000.0 * (e_ibp - e_df),
            "same_p_single_ibp_dagger_projected_delta_mha": 1000.0 * (e_ibp_sym - e_df),
        },
        "z_dagger_residual_direct": _relative_error(z_direct, z_direct.conj().T),
        "z_dagger_residual_single_ibp": _relative_error(z_ibp, z_ibp.conj().T),
        "max_abs_theta_on_boundary": float(np.max(np.abs(theta_grid[:, boundary]))),
        "max_abs_grad_theta_on_boundary": float(np.max(np.abs(grad_grid[:, :, boundary]))),
        "timing_seconds": {
            "direct_kernel_build": direct_kernel_build_seconds,
            "theta_build_shared": theta_build_seconds,
            "direct_core": direct_seconds,
            "single_ibp_gradient_theta_build": gradient_theta_build_seconds,
            "single_ibp_kernel_build": ibp_kernel_build_seconds,
            "single_ibp_core": ibp_seconds,
        },
        "reference_dfmp2_hartree": float(mp.dfmp2.DFMP2(mf).run().e_corr),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="H2O_ccpVDZ")
    parser.add_argument("--spacing", type=float, default=0.30)
    parser.add_argument("--margin", type=float, default=5.0)
    parser.add_argument("--rank-factor", type=float, default=4.0)
    parser.add_argument("--grid-shift-fraction", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run_single_ibp_case(SingleIBPBenchmarkCase(
        system=args.system, spacing=args.spacing, margin=args.margin,
        rank_factor=args.rank_factor,
        grid_shift_fraction=tuple(args.grid_shift_fraction),
    ))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
