"""P2c production validation/benchmark for molecular Poisson ISDF cores.

This harness compares two Coulomb-core builders while holding the molecule,
orbitals, uniform Cartesian mesh, interpolation points, and pair-collocation
matrix fixed:

* ``MolecularDFReference``: streamed analytic DF factors, ``C = P B^dagger``,
  followed by the shared two-sided core fit.
* ``Poisson``: raw physical interpolation vectors on the same uniform mesh,
  followed by the isolated/free-space FFT Coulomb core.

The comparison reports three distinct errors:

1. analytic-DF ISDF ERIs versus exact analytic DF ERIs (rank/truncation error),
2. Poisson ERIs versus analytic-DF ISDF ERIs (mesh/kernel error at fixed P),
3. Poisson ERIs versus exact analytic DF ERIs (total error).

Each invocation runs one case so host/device peak-memory measurements are not
contaminated by earlier sweep points.  ``--emit-matrix`` prints the recommended
one-factor-at-a-time H2O/benzene sweep as JSON; a cluster job array should run
one emitted case per fresh process.

Examples::

    python -m pytc.utils.poisson_core_benchmark --system H2O_ccpVDZ \
        --spacing 0.35 --margin 6 --rank-factor 2 --backend numpy

    python -m pytc.utils.poisson_core_benchmark --emit-matrix --backend jax \
        --output p2c_matrix.json

    python -m pytc.utils.poisson_core_benchmark --matrix-index 0 --backend jax \
        --output p2c_case_0.json

    python -m pytc.utils.poisson_core_benchmark --system H2O_ccpVDZ \
        --spacing 0.12 --rank-factor 4 --grid-shift-fraction 0.5 0.5 0.5 \
        --backend jax --output p2c_h2o_shifted.json

Task #16, #proj-isdf-coulomb-cuda, 2026-07-13.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pyscf import dft, gto, lib, mp, scf
import pyscf

from pytc.integrals.coulomb import (
    build_free_space_poisson_kernel,
    build_poisson_interpolation_sector,
    compute_C_streamed,
    compute_Z,
    pair_collocation_at_pivots,
    poisson_core,
    reconstruct_eri_block,
    select_sector_pivots,
    weight_mo_values,
)


def _benzene_geom(cc: float = 1.397, ch: float = 1.084) -> str:
    atoms: list[str] = []
    rh = cc + ch
    for i in range(6):
        theta = np.deg2rad(60.0 * i)
        atoms.append(f"C {cc*np.cos(theta):.8f} {cc*np.sin(theta):.8f} 0")
    for i in range(6):
        theta = np.deg2rad(60.0 * i)
        atoms.append(f"H {rh*np.cos(theta):.8f} {rh*np.sin(theta):.8f} 0")
    return "; ".join(atoms)


SYSTEMS: dict[str, dict[str, Any]] = {
    "H2O_ccpVDZ": {
        "atom": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        "basis": "ccpvdz",
    },
    "benzene_ccpVDZ": {
        "atom": _benzene_geom(),
        "basis": "ccpvdz",
    },
}


@dataclass(frozen=True)
class BenchmarkCase:
    system: str = "H2O_ccpVDZ"
    spacing: float = 0.35
    margin: float = 6.0
    pad_factor: int = 2
    rank_factor: float = 2.0
    backend: str = "numpy"
    auxbasis: str = "weigend"
    ao_batch_size: int = 8192
    grid_batch_size: int = 8192
    mu_block_size: int = 64
    nu_block_size: int = 16
    rcond: float = 1e-12
    repeats: int = 2
    num_threads: int = 1
    grid_shift_fraction: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.system not in SYSTEMS:
            raise ValueError(f"unknown system {self.system!r}; choose from {sorted(SYSTEMS)}")
        if self.backend not in ("numpy", "jax"):
            raise ValueError("backend must be 'numpy' or 'jax'")
        for name in ("spacing", "margin", "rank_factor", "rcond"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if self.pad_factor < 2:
            raise ValueError("pad_factor must be >= 2")
        for name in (
            "ao_batch_size", "grid_batch_size", "mu_block_size",
            "nu_block_size", "repeats", "num_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        try:
            shift = tuple(float(x) for x in self.grid_shift_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("grid_shift_fraction must contain three finite floats") from exc
        if len(shift) != 3 or any(not math.isfinite(x) or abs(x) > 0.5 for x in shift):
            raise ValueError(
                "grid_shift_fraction must contain three finite values in [-0.5, 0.5], "
                f"got {self.grid_shift_fraction!r}"
            )
        object.__setattr__(self, "grid_shift_fraction", shift)


def centered_uniform_mesh(mol: gto.Mole, spacing: float, margin: float,
                          shift_fraction=(0.0, 0.0, 0.0)):
    """Return an odd, molecule-centered Cartesian mesh in Bohr.

    ``margin`` is added beyond the outermost nucleus independently along
    each axis.  Odd shapes put the molecular box center on a grid point and
    make spacing/domain sweeps deterministic.
    """
    shift_fraction = tuple(float(x) for x in shift_fraction)
    if len(shift_fraction) != 3 or any(
        not math.isfinite(x) or abs(x) > 0.5 for x in shift_fraction
    ):
        raise ValueError("shift_fraction must contain three finite values in [-0.5, 0.5]")
    atom_coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    lo = atom_coords.min(axis=0)
    hi = atom_coords.max(axis=0)
    center = 0.5 * (lo + hi)
    half_extent = 0.5 * (hi - lo) + float(margin)
    shape = []
    for extent in half_extent:
        n = int(math.ceil(2.0 * extent / spacing)) + 1
        if n % 2 == 0:
            n += 1
        shape.append(n)
    shape_t = tuple(shape)
    origin = tuple(
        center - 0.5 * (np.asarray(shape_t) - 1) * spacing
        + np.asarray(shift_fraction) * spacing
    )

    axes = [origin[i] + np.arange(shape_t[i]) * spacing for i in range(3)]
    xyz = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([a.reshape(-1) for a in xyz], axis=1)
    return shape_t, origin, coords


def evaluate_mos_batched(mol: gto.Mole, mo_coeff: np.ndarray,
                         coords: np.ndarray, batch_size: int) -> np.ndarray:
    """Evaluate MOs as ``(n_mo, n_grid)`` without retaining a full AO grid."""
    n_grid = coords.shape[0]
    values = np.empty((mo_coeff.shape[1], n_grid), dtype=np.result_type(mo_coeff, float))
    for start in range(0, n_grid, batch_size):
        stop = min(start + batch_size, n_grid)
        ao = dft.numint.eval_ao(mol, coords[start:stop], deriv=0)
        values[:, start:stop] = (ao @ mo_coeff).T
    return values


def mp2_energy_from_ovov(eri_ovov: np.ndarray, mo_energy: np.ndarray,
                         n_occ: int) -> float:
    """Restricted closed-shell MP2 energy from chemist-ordered ``(i,a,j,b)`` ERIs."""
    e_occ = np.asarray(mo_energy[:n_occ])
    e_vir = np.asarray(mo_energy[n_occ:])
    denom = (
        e_occ[:, None, None, None] + e_occ[None, None, :, None]
        - e_vir[None, :, None, None] - e_vir[None, None, None, :]
    )
    t2 = eri_ovov / denom
    return float(np.einsum(
        "iajb,iajb->", t2,
        2.0 * eri_ovov - eri_ovov.transpose(0, 3, 2, 1),
        optimize=True,
    ).real)


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    return float(np.linalg.norm(value - reference) / denom) if denom else float("nan")


def _array_sha256(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _sync(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def _timed_call(fn: Callable[[], Any], sync_getter: Callable[[Any], Any]):
    t0 = time.perf_counter()
    result = fn()
    _sync(sync_getter(result))
    return result, time.perf_counter() - t0


def _warm_median(fn: Callable[[], Any], sync_getter: Callable[[Any], Any], repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        result, elapsed = _timed_call(fn, sync_getter)
        samples.append(elapsed)
    return result, float(statistics.median(samples)), [float(x) for x in samples]


def _host_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux and most BSD cluster nodes report KiB.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _device_memory_stats() -> dict[str, Any]:
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    keep = {}
    for key, value in stats.items():
        if isinstance(value, (int, float)) and (
            "byte" in key.lower() or "peak" in key.lower() or "limit" in key.lower()
        ):
            keep[key] = int(value)
    return {"device": str(device), "platform": device.platform, "stats": keep}


@contextlib.contextmanager
def _pyscf_threads(n_threads: int):
    """Temporarily set PySCF's global thread count and restore it."""
    previous = lib.num_threads()
    lib.num_threads(n_threads)
    try:
        yield
    finally:
        lib.num_threads(previous)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _structural_bytes(*arrays: np.ndarray) -> int:
    return int(sum(np.asarray(a).nbytes for a in arrays))


def run_case(case: BenchmarkCase) -> dict[str, Any]:
    spec = SYSTEMS[case.system]
    with _pyscf_threads(case.num_threads):
        t0 = time.perf_counter()
        mol = gto.M(atom=spec["atom"], basis=spec["basis"], verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis=case.auxbasis).run()
        scf_seconds = time.perf_counter() - t0

        mo_coeff = np.asarray(mf.mo_coeff)
        n_mo = mo_coeff.shape[1]
        n_occ = mol.nelectron // 2
        n_vir = n_mo - n_occ
        mo_occ = mo_coeff[:, :n_occ]
        mo_vir = mo_coeff[:, n_occ:]

        shape, origin, coords = centered_uniform_mesh(
            mol, case.spacing, case.margin, case.grid_shift_fraction,
        )
        n_grid = int(np.prod(shape))
        dV = case.spacing ** 3

        t0 = time.perf_counter()
        mo_values = evaluate_mos_batched(mol, mo_coeff, coords, case.ao_batch_size)
        grid_eval_seconds = time.perf_counter() - t0
        occ_raw = np.ascontiguousarray(mo_values[:n_occ])
        vir_raw = np.ascontiguousarray(mo_values[n_occ:])
        weights = np.full(n_grid, dV, dtype=np.float64)

        requested_rank = min(
            int(math.ceil(case.rank_factor * n_mo)),
            n_occ * n_vir,
        )
        t0 = time.perf_counter()
        occ_weighted = weight_mo_values(occ_raw, weights)
        vir_weighted = weight_mo_values(vir_raw, weights)
        pivots, pivot_provenance = select_sector_pivots(
            occ_weighted, vir_weighted, requested_rank,
            effective_rank_rtol=1e-6, return_provenance=True,
        )
        pivots = np.asarray(pivots)
        P = pair_collocation_at_pivots(occ_raw[:, pivots], vir_raw[:, pivots])
        _sync(occ_weighted)
        pivot_seconds = time.perf_counter() - t0

        # Analytic MolecularDFReference route, excluding common pivot selection.
        t0 = time.perf_counter()
        C, c_provenance = compute_C_streamed(
            mf, P, mo_occ, mo_vir, auxbasis=case.auxbasis,
            return_provenance=True,
        )
        df_c_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        Z_df, z_df_provenance = compute_Z(P, C)
        df_z_seconds = time.perf_counter() - t0

        # Exact analytic DF ERI/MP2 references are system invariants, not part
        # of either core-builder timing.
        t0 = time.perf_counter()
        eri_exact = mf.with_df.ao2mo(
            (mo_occ, mo_vir, mo_occ, mo_vir), compact=False,
        ).reshape(n_occ, n_vir, n_occ, n_vir)
        exact_eri_seconds = time.perf_counter() - t0
        eri_df = reconstruct_eri_block(P, Z_df, P).reshape(
            n_occ, n_vir, n_occ, n_vir,
        )
        e_mp2_exact_from_eri = mp2_energy_from_ovov(eri_exact, mf.mo_energy, n_occ)
        pt = mp.dfmp2.DFMP2(mf).run()
        e_mp2_pyscf = float(pt.e_corr)
        e_mp2_df = mp2_energy_from_ovov(eri_df, mf.mo_energy, n_occ)

        backend = case.backend
        if backend == "jax":
            factor_p = jnp.asarray(occ_raw)
            factor_q = jnp.asarray(vir_raw)
            upstream = {
                "factor_p_sha256": _array_sha256(occ_raw),
                "factor_q_sha256": _array_sha256(vir_raw),
                "benchmark": "P2c",
            }
        else:
            factor_p, factor_q = occ_raw, vir_raw
            upstream = {"benchmark": "P2c"}

        kernel_fn = lambda: build_free_space_poisson_kernel(
            shape, (case.spacing,) * 3, origin=origin,
            pad_factor=case.pad_factor, backend=backend,
            fft_kind="rfft", dtype=np.float64,
        )
        kernel, kernel_cold_seconds = _timed_call(kernel_fn, lambda x: x.spectrum)
        kernel, kernel_warm_seconds, kernel_samples = _warm_median(
            kernel_fn, lambda x: x.spectrum, case.repeats,
        )

        theta_fn = lambda: build_poisson_interpolation_sector(
            factor_p, factor_q, pivots, kernel.mesh,
            grid_batch_size=case.grid_batch_size, rcond=case.rcond,
            upstream_provenance=dict(upstream),
        )
        poisson_sector, theta_cold_seconds = _timed_call(theta_fn, lambda x: x.Theta)
        poisson_sector, theta_warm_seconds, theta_samples = _warm_median(
            theta_fn, lambda x: x.Theta, case.repeats,
        )

        core_fn = lambda: poisson_core(
            poisson_sector, kernel=kernel,
            mu_block_size=min(case.mu_block_size, len(pivots)),
            nu_block_size=min(case.nu_block_size, len(pivots)),
        )
        poisson_artifact, core_cold_seconds = _timed_call(core_fn, lambda x: x.Z)
        poisson_artifact, core_warm_seconds, core_samples = _warm_median(
            core_fn, lambda x: x.Z, case.repeats,
        )

        Z_poisson = np.asarray(poisson_artifact.Z)
        eri_poisson = reconstruct_eri_block(P, Z_poisson, P).reshape(
            n_occ, n_vir, n_occ, n_vir,
        )
        e_mp2_poisson = mp2_energy_from_ovov(eri_poisson, mf.mo_energy, n_occ)

    df_total = df_c_seconds + df_z_seconds
    poisson_total_cold = kernel_cold_seconds + theta_cold_seconds + core_cold_seconds
    poisson_total_warm = kernel_warm_seconds + theta_warm_seconds + core_warm_seconds

    result = {
        "schema_version": 2,
        "case": asdict(case),
        "status": "ok",
        "environment": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "jax": jax.__version__,
            "pyscf": pyscf.__version__,
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "device": _device_memory_stats(),
        },
        "sizes": {
            "n_atom": int(mol.natm), "n_ao": int(mol.nao), "n_mo": int(n_mo),
            "n_occ": int(n_occ), "n_vir": int(n_vir), "n_pair_ov": int(n_occ*n_vir),
            "mesh_shape": shape, "mesh_origin_bohr": origin, "n_grid": n_grid,
            "dV_bohr3": dV, "requested_rank": requested_rank,
            "selected_rank": int(len(pivots)),
        },
        "timing_seconds": {
            "scf": scf_seconds, "uniform_grid_mo_eval": grid_eval_seconds,
            "common_pivot_selection": pivot_seconds,
            "exact_df_eri_reference": exact_eri_seconds,
            "df_C_stream": df_c_seconds, "df_Z_fit": df_z_seconds,
            "df_total_excluding_pivots": df_total,
            "poisson_kernel_cold": kernel_cold_seconds,
            "poisson_kernel_warm_median": kernel_warm_seconds,
            "poisson_theta_cold": theta_cold_seconds,
            "poisson_theta_warm_median": theta_warm_seconds,
            "poisson_core_cold": core_cold_seconds,
            "poisson_core_warm_median": core_warm_seconds,
            "poisson_total_cold_excluding_pivots": poisson_total_cold,
            "poisson_total_warm_excluding_pivots": poisson_total_warm,
            "df_over_poisson_cold_speed_ratio": df_total / poisson_total_cold,
            "df_over_poisson_warm_speed_ratio": df_total / poisson_total_warm,
            "warm_samples": {
                "kernel": kernel_samples, "theta": theta_samples, "core": core_samples,
            },
        },
        "accuracy": {
            "df_isdf_eri_relerr_vs_exact_df": _relative_error(eri_df, eri_exact),
            "poisson_eri_relerr_vs_df_isdf_same_P": _relative_error(eri_poisson, eri_df),
            "poisson_eri_relerr_vs_exact_df": _relative_error(eri_poisson, eri_exact),
            "poisson_core_relerr_vs_df_core_same_P": _relative_error(Z_poisson, Z_df),
            "mp2_pyscf_df_Ha": e_mp2_pyscf,
            "mp2_exact_eri_formula_Ha": e_mp2_exact_from_eri,
            "mp2_df_isdf_Ha": e_mp2_df,
            "mp2_poisson_Ha": e_mp2_poisson,
            "mp2_df_isdf_error_mHa": 1000.0 * (e_mp2_df - e_mp2_pyscf),
            "mp2_poisson_error_mHa": 1000.0 * (e_mp2_poisson - e_mp2_pyscf),
            "mp2_poisson_minus_df_isdf_mHa": 1000.0 * (e_mp2_poisson - e_mp2_df),
            "mp2_formula_vs_pyscf_mHa": 1000.0 * (e_mp2_exact_from_eri - e_mp2_pyscf),
        },
        "memory": {
            "host_peak_rss_bytes": _host_peak_rss_bytes(),
            "factor_arrays_bytes": _structural_bytes(occ_raw, vir_raw),
            "P_bytes": int(np.asarray(P).nbytes), "C_bytes": int(np.asarray(C).nbytes),
            "Theta_bytes": int(np.prod(poisson_sector.Theta.shape) * poisson_sector.Theta.dtype.itemsize),
            "Z_df_bytes": int(np.asarray(Z_df).nbytes),
            "Z_poisson_bytes": int(np.asarray(Z_poisson).nbytes),
            "kernel_spectrum_bytes": int(np.prod(kernel.spectrum.shape) * kernel.spectrum.dtype.itemsize),
        },
        "provenance": {
            "pivot": pivot_provenance,
            "mo_coeff_sha256": _array_sha256(mo_coeff),
            "grid_coords_sha256": _array_sha256(coords),
            "pivots_sha256": _array_sha256(pivots),
            "pair_collocation_P_sha256": _array_sha256(P),
            "df_factor_sha256": c_provenance["df_factor_sha256"],
            "df_solver": z_df_provenance,
            "poisson_sector_spec_sha256": poisson_sector.sector_spec_sha256,
            "poisson_kernel_spec_sha256": kernel.kernel_spec_sha256,
            "poisson_core_spec_sha256": poisson_artifact.core_spec_sha256,
        },
    }
    return _jsonable(result)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def recommended_matrix(backend: str = "jax") -> list[dict[str, Any]]:
    """One-factor-at-a-time production matrix with duplicate baselines removed."""
    cases: list[BenchmarkCase] = []

    def add(base: BenchmarkCase, **updates):
        data = asdict(base)
        data.update(updates)
        case = BenchmarkCase(**data)
        if case not in cases:
            cases.append(case)

    h2o = BenchmarkCase(system="H2O_ccpVDZ", spacing=0.35, margin=6.0,
                        rank_factor=2.0, pad_factor=2, backend=backend)
    for h in (0.50, 0.35, 0.25):
        add(h2o, spacing=h)
    for margin in (4.0, 6.0, 8.0):
        add(h2o, margin=margin)
    for pad in (2, 3):
        add(h2o, pad_factor=pad)
    for rank in (1.0, 2.0, 4.0):
        add(h2o, rank_factor=rank)

    benzene = BenchmarkCase(system="benzene_ccpVDZ", spacing=0.45, margin=6.0,
                            rank_factor=4.0, pad_factor=2, backend=backend,
                            grid_batch_size=4096, mu_block_size=64, nu_block_size=8)
    for h in (0.60, 0.45, 0.35):
        add(benzene, spacing=h)
    for margin in (4.0, 6.0):
        add(benzene, margin=margin)
    for pad in (2, 3):
        add(benzene, pad_factor=pad)
    for rank in (2.0, 4.0, 6.0):
        add(benzene, rank_factor=rank)

    return [asdict(c) for c in cases]


def matrix_case(index: int, backend: str = "jax") -> BenchmarkCase:
    """Resolve one recommended sweep case for a cluster job-array index."""
    matrix = recommended_matrix(backend)
    if isinstance(index, bool) or index < 0 or index >= len(matrix):
        raise ValueError(f"matrix index must be in [0, {len(matrix)}), got {index!r}")
    return BenchmarkCase(**matrix[index])


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--emit-matrix", action="store_true")
    selection.add_argument(
        "--matrix-index", type=int,
        help="run this zero-based case from the recommended matrix; other case flags are ignored",
    )
    p.add_argument("--system", choices=sorted(SYSTEMS), default="H2O_ccpVDZ")
    p.add_argument("--spacing", type=float, default=0.35)
    p.add_argument("--margin", type=float, default=6.0)
    p.add_argument("--pad-factor", type=int, default=2)
    p.add_argument("--rank-factor", type=float, default=2.0)
    p.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    p.add_argument("--auxbasis", default="weigend")
    p.add_argument("--ao-batch-size", type=int, default=8192)
    p.add_argument("--grid-batch-size", type=int, default=8192)
    p.add_argument("--mu-block-size", type=int, default=64)
    p.add_argument("--nu-block-size", type=int, default=16)
    p.add_argument("--rcond", type=float, default=1e-12)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument(
        "--grid-shift-fraction", type=float, nargs=3, metavar=("SX", "SY", "SZ"),
        default=(0.0, 0.0, 0.0),
        help="translate the grid by these fractions of one cell per axis (diagnostic)",
    )
    p.add_argument("--output", help="optional JSON output path; stdout is always emitted")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.emit_matrix:
        payload = json.dumps(recommended_matrix(args.backend), indent=2, sort_keys=True)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        print(payload)
        return 0
    if args.matrix_index is not None:
        case = matrix_case(args.matrix_index, args.backend)
    else:
        case = BenchmarkCase(
            system=args.system, spacing=args.spacing, margin=args.margin,
            pad_factor=args.pad_factor, rank_factor=args.rank_factor,
            backend=args.backend, auxbasis=args.auxbasis,
            ao_batch_size=args.ao_batch_size, grid_batch_size=args.grid_batch_size,
            mu_block_size=args.mu_block_size, nu_block_size=args.nu_block_size,
            rcond=args.rcond, repeats=args.repeats, num_threads=args.num_threads,
            grid_shift_fraction=tuple(args.grid_shift_fraction),
        )
    result = run_case(case)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
