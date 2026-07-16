"""CPU-only paired same-grid reciprocal-AO pivot study for diamond 211.

The approximate reciprocal AOs are used solely to choose pivot indices.
Every factorization, SCF, ERI, and MP2 calculation receives exact PySCF AO
values at the chosen indices and on the full uniform grid.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pyscf
from pyscf.pbc import gto, mp, scf, tools
from pyscf.pbc.df import FFTDF

from pytc.pbc import coulomb
from pytc.pbc.df.isdf import (
    build_cached_periodic_pivot_oracle,
    build_periodic_pivot_oracle,
    pivoted_cholesky_hermitian,
)
from pytc.pbc.df.kpts import build_kconserv, canonicalize_kpts
from pytc.pbc.df.reciprocal_ao_pilot import reciprocal_translate_bloch_ao


RANK_MULTIPLIERS = (2, 4, 6, 8, 10, 12, 14)
FROZEN_GRID_MESH = (27, 13, 13)
RTOL = 1e-5
RETENTION_MODE = "single"
SCF_CONTROLS = {
    "conv_tol": 1e-9,
    "max_cycle": 50,
    "init_guess": "minao",
    "exxdiv": "ewald",
}


def _git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()


def _configure_scf(mf):
    mf.conv_tol = SCF_CONTROLS["conv_tol"]
    mf.max_cycle = SCF_CONTROLS["max_cycle"]
    mf.init_guess = SCF_CONTROLS["init_guess"]
    mf.exxdiv = SCF_CONTROLS["exxdiv"]
    return mf


def diamond_211():
    """Return the frozen R1 diamond supercell input."""
    primitive = gto.Cell()
    primitive.atom = "C 0.0 0.0 0.0; C 0.8917 0.8917 0.8917"
    primitive.a = """0.0 1.7834 1.7834
1.7834 0.0 1.7834
1.7834 1.7834 0.0"""
    primitive.unit = "A"
    primitive.basis = "gth-dzvp"
    primitive.pseudo = "gth-pbe"
    primitive.ke_cutoff = 30.0
    primitive.verbose = 0
    primitive.build()
    cell = tools.super_cell(primitive, [2, 1, 1])
    # This study's comparison grid is a frozen input, not a PySCF mesh heuristic.
    cell.mesh = np.array(FROZEN_GRID_MESH, dtype=np.int32)
    return cell


def _relative_error(reference, candidate):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    difference = candidate - reference
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_frobenius": float(np.linalg.norm(difference) / denominator),
    }


def _residual_for_fixed_pivots(diagonal, column, pivots, *, rcond=1e-12):
    """Evaluate an exact metric residual after a prescribed pivot order."""
    residual = np.maximum(np.asarray(diagonal, dtype=np.float64), 0.0).copy()
    maximum = float(np.max(residual))
    threshold = rcond * maximum
    factor = np.zeros((residual.size, len(pivots)), dtype=np.complex128)
    realized = 0
    for step, pivot in enumerate(np.asarray(pivots, dtype=np.int64)):
        if residual[pivot] <= threshold:
            break
        col = np.asarray(column(int(pivot)), dtype=np.complex128)
        update = factor[:, :step] @ factor[pivot, :step].conj() if step else 0.0
        vector = (col - update) / np.sqrt(residual[pivot])
        factor[:, step] = vector
        residual = np.maximum(residual - np.abs(vector) ** 2, 0.0)
        realized += 1
    return {
        "realized_rank": realized,
        "max_abs": float(np.max(residual)),
        "relative_to_initial_max": float(np.max(residual) / maximum),
    }


def same_grid_reciprocal_pivots(cell, canonical_kpts, rank):
    """Choose pivots from the Phase-R0 same-grid reciprocal approximation."""
    mesh = np.asarray(cell.mesh)
    coords = cell.get_uniform_grids(mesh)
    direct = np.asarray(
        cell.pbc_eval_gto("GTOval", coords, kpts=list(canonical_kpts)), dtype=np.complex128,
    )
    n_primitive_ao = cell.nao_nr() // 2
    translation = cell.atom_coords()[2] - cell.atom_coords()[0]
    generated = np.asarray([
        reciprocal_translate_bloch_ao(
            direct[k, :, :n_primitive_ao], coords, mesh, kpt, translation,
            cell.get_Gv(mesh),
        )
        for k, kpt in enumerate(canonical_kpts)
    ])
    n_kpts = len(canonical_kpts)

    def groups():
        return (direct[:, :, :n_primitive_ao], generated)

    diagonal = sum(
        np.sum(np.abs(group) ** 2, axis=(0, 2)) for group in groups()
    ) ** 2 / n_kpts

    def column(pivot):
        accumulator = sum(
            np.einsum("ka,kga->g", group[:, pivot, :].conj(), group, optimize=True)
            for group in groups()
        )
        return (np.abs(accumulator) ** 2 / n_kpts).astype(np.complex128)

    pivots, _, realized = pivoted_cholesky_hermitian(diagonal, column, rank=rank)
    return pivots, realized, {
        "ao_max_abs_vs_exact": _relative_error(direct[:, :, n_primitive_ao:], generated)["max_abs"],
        "ao_relative_frobenius_vs_exact": _relative_error(
            direct[:, :, n_primitive_ao:], generated,
        )["relative_frobenius"],
    }


def _select_exact_pivots(cell, canonical_kpts, grid_coords, block_size, rank):
    started = time.perf_counter()
    diag, column, _ = build_cached_periodic_pivot_oracle(
        cell, canonical_kpts, grid_coords, block_size,
    )
    pivots, _, realized = pivoted_cholesky_hermitian(diag, column, rank=rank)
    return pivots, realized, diag, column, time.perf_counter() - started


def _compare_cached_and_streamed_exact(cell, canonical_kpts, grid_coords, block_size, rank):
    cached_pivots, cached_count, _, _, cached_seconds = _select_exact_pivots(
        cell, canonical_kpts, grid_coords, block_size, rank,
    )
    started = time.perf_counter()
    streamed_diag, streamed_column = build_periodic_pivot_oracle(
        cell, canonical_kpts, grid_coords, block_size,
    )
    streamed_pivots, _, streamed_count = pivoted_cholesky_hermitian(
        streamed_diag, streamed_column, rank=rank,
    )
    return {
        "rank": rank,
        "cached_seconds": cached_seconds,
        "streamed_seconds": time.perf_counter() - started,
        "cached_realized_rank": cached_count,
        "streamed_realized_rank": streamed_count,
        "pivots_identical": bool(np.array_equal(cached_pivots, streamed_pivots)),
    }


def _reference(cell, kpts):
    mf = _configure_scf(scf.KRHF(cell, kpts))
    mf.verbose = 0
    mf.with_df = FFTDF(cell, kpts)
    total = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("Frozen FFTDF KRHF reference did not converge.")
    started = time.perf_counter()
    correlation, _ = mp.KMP2(mf).kernel()
    return mf, {
        "krhf_total": total,
        "kmp2_correlation": float(correlation),
        "kmp2_total": total + float(correlation),
        "mp2_seconds": time.perf_counter() - started,
        "cycles": int(mf.cycles),
    }


def _adapter_mo_eri_audit(adapter, mf, built, fftdf):
    """Audit the adapter's PySCF ao2mo convention on a nontrivial quartet."""
    canonical = built["mesh_obj"].canonical_kpts
    kconserv = build_kconserv(adapter.cell, canonical)
    k1, k2, k3 = 0, 1, 2
    k4 = int(kconserv[k1, k2, k3])
    mo_coeffs = [mf.mo_coeff[index] for index in (k1, k2, k3, k4)]
    expected, actual_k4 = coulomb.get_mo_eri(
        built["inpv_kpt"], built["coul_kpt"], kconserv, mo_coeffs, k1, k2, k3,
    )
    if actual_k4 != k4:
        raise RuntimeError("Direct MO-ERI momentum reconstruction disagrees with kconserv.")
    adapter_eri = adapter.ao2mo(
        mo_coeffs, canonical[[k1, k2, k3, k4]], compact=False,
    ).reshape(expected.shape)
    reference_mo = np.asarray(
        fftdf.ao2mo(
            mo_coeffs,
            kpts=[canonical[index] for index in (k1, k2, k3, k4)],
            compact=False,
        )
    ).reshape(expected.shape)
    return {
        "k_indices": [k1, k2, k3, k4],
        "adapter_return_shape": list(adapter_eri.shape),
        "adapter_vs_direct_build_get_mo_eri": _relative_error(expected, adapter_eri),
        "adapter_vs_fftdf_mo_eri": _relative_error(reference_mo, adapter_eri),
    }


def _run_fixed_pivot_path(cell, kpts, pivots, block_size, reference_mf, reference_values):
    rank = int(len(pivots))
    started = time.perf_counter()
    built = coulomb.build(
        cell, kpts, rank=rank, block_size=block_size, rtol=RTOL,
        retention_mode=RETENTION_MODE,
        selection_mode="streamed", fixed_pivots=pivots,
    )
    build_seconds = time.perf_counter() - started
    dm = reference_mf.make_rdm1()
    vj_ref, vk_ref = reference_mf.get_jk(dm_kpts=dm)
    vj = coulomb.get_j(cell, dm, kpts)
    vk = coulomb.get_k(
        dm, built["inpv_kpt"], built["coul_kpt"], built["mesh_obj"].phase,
        exxdiv=reference_mf.exxdiv, cell=cell, kpts=kpts, neg=built["mesh_obj"].neg,
    )
    kconserv = build_kconserv(cell, built["mesh_obj"].canonical_kpts)
    k1, k2, k3 = 0, 1, 2
    eri, k4 = coulomb.get_ao_eri(
        built["inpv_kpt"], built["coul_kpt"], kconserv, k1, k2, k3,
    )
    fftdf = FFTDF(cell)
    reference_eri = np.asarray(
        fftdf.get_eri([
            built["mesh_obj"].canonical_kpts[index] for index in (k1, k2, k3, k4)
        ], compact=False)
    ).reshape((cell.nao_nr(),) * 4)
    adapter = coulomb.ISDFDF(
        cell, kpts, rank=rank, block_size=block_size, rtol=RTOL,
        retention_mode=RETENTION_MODE,
        selection_mode="streamed", fixed_pivots=pivots,
    )
    adapter._built = built
    mf = _configure_scf(scf.KRHF(cell, kpts))
    mf.verbose = 0
    mf.with_df = adapter
    started = time.perf_counter()
    total = float(mf.kernel())
    scf_seconds = time.perf_counter() - started
    if mf.converged:
        ao2mo_calls_before_kmp2 = adapter._ao2mo_call_count
        started = time.perf_counter()
        correlation, _ = mp.KMP2(mf).kernel()
        mp2_seconds = time.perf_counter() - started
        ao2mo_calls_after_kmp2 = adapter._ao2mo_call_count
        if ao2mo_calls_after_kmp2 <= ao2mo_calls_before_kmp2:
            raise RuntimeError("KMP2 completed without consuming ISDFDF.ao2mo.")
        correlation = float(correlation)
        mp2_record = {
            "status": "completed",
            "ao2mo_calls_during_kmp2": (
                ao2mo_calls_after_kmp2 - ao2mo_calls_before_kmp2
            ),
            "correlation": correlation,
            "total": total + correlation,
            "correlation_per_atom_error": abs(
                correlation - reference_values["kmp2_correlation"]
            ) / cell.natm,
            "total_per_atom_error": abs(
                total + correlation - reference_values["kmp2_total"]
            ) / cell.natm,
        }
    else:
        mp2_seconds = 0.0
        mp2_record = {
            "status": "not_run",
            "reason": "ISDF KRHF did not converge; no MP2 proxy was substituted.",
            "ao2mo_calls_during_kmp2": 0,
        }
    adapter_audit = _adapter_mo_eri_audit(adapter, mf, built, fftdf)
    adapter_audit["ao2mo_calls_total_after_audit"] = adapter._ao2mo_call_count
    return {
        "requested_rank": rank,
        "realized_rank": int(built["n_selected"]),
        "build_seconds": build_seconds,
        "scf_seconds": scf_seconds,
        "mp2_seconds": mp2_seconds,
        "jk_vs_fftdf": {
            "j": {
                **_relative_error(vj_ref, vj),
                "interpretation": "zero by construction: J delegates to FFTDF, not ISDF",
            },
            "k": _relative_error(vk_ref, vk),
        },
        "deterministic_ao_eri_vs_fftdf": {
            "k_indices": [k1, k2, k3, int(k4)],
            **_relative_error(reference_eri, eri),
        },
        "krhf": {
            "converged": bool(mf.converged),
            "cycles": int(mf.cycles),
            "total": total,
            "per_atom_error": abs(total - reference_values["krhf_total"]) / cell.natm,
        },
        "kmp2": mp2_record,
        "adapter_ao2mo_audit": adapter_audit,
        "retained_mode_warnings": [
            info.get("retention_warning") for info in built["solve_infos"]
            if info.get("retention_warning") is not None
        ],
        "selection_provenance": built["selection_provenance"],
    }


def _first_rank_meeting(records, selector, metric):
    for record in records:
        result = record[selector]
        if metric == "krhf":
            if result["krhf"]["converged"] and result["krhf"]["per_atom_error"] <= 1e-4:
                return record["rank"]
        elif (
            result["kmp2"]["status"] == "completed"
            and result["kmp2"]["correlation_per_atom_error"] <= 1e-4
            and result["kmp2"]["total_per_atom_error"] <= 1e-4
        ):
            return record["rank"]
    return None


def run_study(block_size=256):
    """Run the frozen R1 paired CPU study and return JSON-ready evidence."""
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    source_commit = _git_revision()
    cell = diamond_211()
    kpts = cell.make_kpts([3, 2, 1], wrap_around=False)
    mesh = canonicalize_kpts(cell, kpts)
    grid = cell.get_uniform_grids(cell.mesh)
    n_ao = cell.nao_nr()
    ranks = [multiplier * n_ao for multiplier in RANK_MULTIPLIERS]
    reference_mf, reference_values = _reference(cell, kpts)
    cached_streamed = _compare_cached_and_streamed_exact(
        cell, mesh.canonical_kpts, grid, block_size, max(ranks),
    )
    if not cached_streamed["pivots_identical"]:
        raise RuntimeError("Exact cached and streamed pivot orders differ; reference is not validated.")
    exact_pivots, exact_count, diagonal, column, exact_seconds = _select_exact_pivots(
        cell, mesh.canonical_kpts, grid, block_size, max(ranks),
    )
    approximate_pivots, approximate_count, approximate_ao_error = same_grid_reciprocal_pivots(
        cell, mesh.canonical_kpts, max(ranks),
    )
    records = []
    for rank in ranks:
        exact_prefix = exact_pivots[:rank]
        approximate_prefix = approximate_pivots[:rank]
        if len(exact_prefix) != rank or len(approximate_prefix) != rank:
            break
        exact_residual = _residual_for_fixed_pivots(diagonal, column, exact_prefix)
        approximate_residual = _residual_for_fixed_pivots(diagonal, column, approximate_prefix)
        exact_path = _run_fixed_pivot_path(
            cell, kpts, exact_prefix, block_size, reference_mf, reference_values,
        )
        approximate_path = _run_fixed_pivot_path(
            cell, kpts, approximate_prefix, block_size, reference_mf, reference_values,
        )
        overlap = len(set(exact_prefix.tolist()) & set(approximate_prefix.tolist()))
        mp2_complete = (
            exact_path["kmp2"]["status"] == "completed"
            and approximate_path["kmp2"]["status"] == "completed"
        )
        records.append({
            "rank": rank,
            "pivot_overlap_count": overlap,
            "pivot_overlap_fraction": overlap / rank,
            "same_position_count": int(np.count_nonzero(exact_prefix == approximate_prefix)),
            "exact_metric_residual": exact_residual,
            "fft_selected_exact_metric_residual": approximate_residual,
            "residual_ratio_fft_over_exact": (
                approximate_residual["max_abs"] / max(exact_residual["max_abs"], np.finfo(float).tiny)
            ),
            "exact_selected": exact_path,
            "fft_selected": approximate_path,
            "paired_delta_per_atom": {
                "krhf_total": (
                    approximate_path["krhf"]["total"] - exact_path["krhf"]["total"]
                ) / cell.natm,
                "kmp2_status": "completed" if mp2_complete else "not_available",
                "kmp2_correlation": (
                    approximate_path["kmp2"]["correlation"] - exact_path["kmp2"]["correlation"]
                ) / cell.natm if mp2_complete else None,
                "kmp2_total": (
                    approximate_path["kmp2"]["total"] - exact_path["kmp2"]["total"]
                ) / cell.natm if mp2_complete else None,
            },
        })
    decision = {
        "targets_hartree_per_atom": {"krhf": 1e-4, "kmp2_correlation": 1e-4, "kmp2_total": 1e-4},
        "exact_selected_first_rank_meeting_krhf_target": _first_rank_meeting(
            records, "exact_selected", "krhf",
        ),
        "fft_selected_first_rank_meeting_krhf_target": _first_rank_meeting(
            records, "fft_selected", "krhf",
        ),
        "exact_selected_first_rank_meeting_both_kmp2_targets": _first_rank_meeting(
            records, "exact_selected", "kmp2",
        ),
        "fft_selected_first_rank_meeting_both_kmp2_targets": _first_rank_meeting(
            records, "fft_selected", "kmp2",
        ),
        "additional_pivots_needed": "bounded measurement through cIP=14; no extension beyond cIP=14 is authorized without consultation",
        "equal_rank_integral_error_ratios_fft_over_exact": [
            record["fft_selected"]["deterministic_ao_eri_vs_fftdf"]["relative_frobenius"]
            / max(
                record["exact_selected"]["deterministic_ao_eri_vs_fftdf"]["relative_frobenius"],
                np.finfo(float).tiny,
            )
            for record in records
        ],
        "interpretation": "Report equal-rank residual, ERI, and energy ratios directly; no unannounced materiality threshold is imposed.",
    }
    return {
        "schema": "pytc-periodic-reciprocal-selection-study/v1",
        "scope": "CPU-only experimental pivot-selection comparison; approximate same-grid reciprocal AOs select indices only.",
        "input": {
            "system": "diamond 211 supercell",
            "basis": "gth-dzvp",
            "pseudo": "gth-pbe",
            "ke_cutoff_hartree": 30.0,
            "grid_mesh": [int(value) for value in cell.mesh],
            "realized_mesh_override": {
                "mesh": list(FROZEN_GRID_MESH),
                "applied_before_fftdf_and_isdf": True,
            },
            "k_mesh": [3, 2, 1],
            "n_ao_supercell": n_ao,
            "rank_multipliers": list(RANK_MULTIPLIERS),
            "requested_ranks": ranks,
            "block_size": block_size,
            "rtol": RTOL,
            "retention_mode": RETENTION_MODE,
            "scf_controls": SCF_CONTROLS,
        },
        "provenance": {
            "started_at_utc": started_at,
            "source_commit": source_commit,
            "runner_commit": source_commit,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pyscf_version": pyscf.__version__,
            "jax_version": jax.__version__,
            "jax_x64_enabled": bool(jax.config.jax_enable_x64),
            "jax_backend": jax.default_backend(),
            "selection_isolation": "Both paths pass only fixed indices to an exact PySCF-AO build; approximate AO arrays never enter Pi, eta, kernels, ERIs, SCF, or MP2.",
        },
        "reference_fftdf": reference_values,
        "exact_cached_vs_streamed": cached_streamed,
        "exact_cached_selection_seconds": exact_seconds,
        "exact_cached_realized_rank": exact_count,
        "same_grid_reciprocal_realized_rank": approximate_count,
        "same_grid_reciprocal_ao_error": approximate_ao_error,
        "records": records,
        "decision": decision,
    }


def _write_json_atomically(path, record):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".reciprocal-selection-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON result path; overwritten atomically.")
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args(argv)
    record = run_study(block_size=args.block_size)
    record["provenance"]["command"] = (
        "python -m pytc.pbc.df.reciprocal_selection_study "
        f"--output {args.output} --block-size {args.block_size}"
    )
    _write_json_atomically(args.output, record)


if __name__ == "__main__":
    main()
