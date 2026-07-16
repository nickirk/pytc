"""Frozen CPU benchmark for the non-default reciprocal same-grid selector.

The reciprocal selector changes only pivot indices.  Every downstream build
uses the existing exact PySCF AO evaluation through ``fixed_pivots``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import resource
import tempfile
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from pytc.pbc import coulomb
from pytc.pbc.df.isdf import (
    build_cached_periodic_pivot_oracle,
    build_periodic_pivot_oracle,
    pivoted_cholesky_hermitian,
    select_jax_cached_matrix_free,
)
from pytc.pbc.df.kpts import canonicalize_kpts
from pytc.pbc.df.reciprocal_ao_pilot import (
    ReciprocalOrbitPartition,
    reciprocal_same_grid_byte_model,
    select_reciprocal_same_grid,
)
from pytc.pbc.df.reciprocal_selection_study import (
    RTOL,
    _reference,
    _residual_for_fixed_pivots,
    _run_fixed_pivot_path,
    diamond_211,
)


FROZEN_RANK = 624
FROZEN_BLOCK_SIZE = 256
WARM_REPEATS = 5


def frozen_211_partition(cell):
    """Return the validated two-cell bulk orbit for the frozen diamond case."""
    return ReciprocalOrbitPartition(
        seed_shell_slice=(0, 6),
        replica_shell_slices=((6, 12),),
        replica_translations=np.asarray([cell.atom_coords()[2] - cell.atom_coords()[0]]),
        supercell_matrix=np.diag([2, 1, 1]),
    )


def _rss_bytes():
    # macOS reports ru_maxrss in bytes; Linux reports KiB. This benchmark is
    # CPU-only and records the unit-normalized peak rather than inferring HBM.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _timed_call(callable_):
    before = _rss_bytes()
    started = time.perf_counter()
    value = callable_()
    elapsed = time.perf_counter() - started
    return value, {"seconds": elapsed, "rss_peak_bytes": max(before, _rss_bytes())}


def _selection_runner(mode, cell, mesh_obj, grid_coords, rank, *, peak_max_bytes):
    if mode == "reciprocal_same_grid":
        return lambda: select_reciprocal_same_grid(
            cell, mesh_obj.canonical_kpts, grid_coords, rank, frozen_211_partition(cell),
            selection_peak_max_bytes=peak_max_bytes,
        )
    if mode == "jax_cached_matrix_free":
        def run():
            _, _, cache = build_cached_periodic_pivot_oracle(
                cell, mesh_obj.canonical_kpts, grid_coords, FROZEN_BLOCK_SIZE,
            )
            return select_jax_cached_matrix_free(
                cache, rank, selection_peak_max_bytes=peak_max_bytes,
            )
        return run
    if mode == "streamed":
        def run():
            diagonal, column = build_periodic_pivot_oracle(
                cell, mesh_obj.canonical_kpts, grid_coords, FROZEN_BLOCK_SIZE,
            )
            pivots, factor, count = pivoted_cholesky_hermitian(diagonal, column, rank=rank)
            return pivots, factor, count, {"mode": "streamed"}
        return run
    raise ValueError(f"unknown selector mode {mode!r}.")


def _measure_selection(mode, cell, mesh_obj, grid_coords, rank, *, peak_max_bytes):
    runner = _selection_runner(mode, cell, mesh_obj, grid_coords, rank, peak_max_bytes=peak_max_bytes)
    cold_value, cold = _timed_call(runner)
    raw = []
    warm_values = []
    for _ in range(WARM_REPEATS):
        value, measurement = _timed_call(runner)
        warm_values.append(value)
        raw.append(measurement)
    pivots, _, n_selected, provenance = warm_values[-1]
    if not all(np.array_equal(pivots, value[0]) and n_selected == value[2] for value in warm_values):
        raise RuntimeError(f"{mode} produced non-deterministic warm pivots.")
    return {
        "mode": mode,
        "pivots": np.asarray(pivots, dtype=np.int64),
        "n_selected": int(n_selected),
        "cold": cold,
        "warm_raw": raw,
        "warm_median_seconds": float(np.median([item["seconds"] for item in raw])),
        "warm_peak_rss_bytes": max(item["rss_peak_bytes"] for item in raw),
        "device_peak": "not_measured_cpu",
        "selection_provenance": provenance,
    }


def _projection(record, *, n_ao_bulk):
    reciprocal = record["reciprocal_same_grid"]
    base_seconds_per_group = reciprocal["warm_median_seconds"] / max(
        reciprocal["selection_provenance"].get("reconstruction_count", 0), 1,
    )
    records = {}
    for label, volume in (("222", 8), ("333", 27), ("444", 64)):
        scale = volume / 2
        n_grid = int(round(record["n_grid"] * scale))
        rank = int(round(record["rank"] * scale))
        model = reciprocal_same_grid_byte_model(
            record["n_kpts"], n_grid, n_ao_bulk, 0, volume, rank,
            selection_peak_max_bytes=record["selection_peak_max_bytes"],
            peak_safety_factor=record["peak_safety_factor"],
            process_pipeline_allowance_bytes=record["process_pipeline_allowance_bytes"],
        )
        reconstructions = (volume - 1) * (rank + 1)
        records[label] = {
            "projected_only_not_capacity_evidence": True,
            "byte_model": model,
            "reconstruction_count": reconstructions,
            "fft_count": 2 * record["n_kpts"] * reconstructions,
            "estimated_seconds_from_measured_per_group": base_seconds_per_group * reconstructions,
        }
    return records


def run_benchmark(*, rank=FROZEN_RANK, peak_max_bytes=24 * 2**30,
                  peak_safety_factor=1.10, process_pipeline_allowance_bytes=0,
                  run_downstream=True):
    """Run the frozen R2 CPU selector and exact-downstream evidence package."""
    if rank != FROZEN_RANK:
        raise ValueError(f"R2 benchmark is frozen at rank={FROZEN_RANK}.")
    cell = diamond_211()
    kpts = cell.make_kpts([3, 2, 1], wrap_around=False)
    mesh_obj = canonicalize_kpts(cell, kpts)
    grid_coords = cell.get_uniform_grids(cell.mesh)
    modes = ("reciprocal_same_grid", "jax_cached_matrix_free", "streamed")
    selection = {
        mode: _measure_selection(
            mode, cell, mesh_obj, grid_coords, rank, peak_max_bytes=peak_max_bytes,
        )
        for mode in modes
    }
    exact_diagonal, exact_column, _ = build_cached_periodic_pivot_oracle(
        cell, mesh_obj.canonical_kpts, grid_coords, FROZEN_BLOCK_SIZE,
    )
    exact_pivots = selection["jax_cached_matrix_free"]["pivots"]
    for mode, result in selection.items():
        result["pivot_overlap_with_jax_cached"] = int(np.intersect1d(
            exact_pivots, result["pivots"], assume_unique=True,
        ).size)
        result["same_position_count_with_jax_cached"] = int(np.count_nonzero(
            exact_pivots == result["pivots"],
        ))
        result["exact_metric_residual"] = _residual_for_fixed_pivots(
            exact_diagonal, exact_column, result["pivots"],
        )
        result["pivots"] = result["pivots"].tolist()

    downstream = {}
    if run_downstream:
        reference_mf, reference = _reference(cell, kpts)
        for mode, result in selection.items():
            path = _run_fixed_pivot_path(
                cell, kpts, np.asarray(result["pivots"], dtype=np.int64),
                FROZEN_BLOCK_SIZE, reference_mf, reference,
            )
            if not path["krhf"]["converged"]:
                raise RuntimeError(f"{mode} exact-downstream KRHF did not converge.")
            if path["krhf"]["per_atom_error"] > 1e-4:
                raise RuntimeError(f"{mode} failed the frozen KRHF chemistry gate.")
            if path["kmp2"]["status"] != "completed":
                raise RuntimeError(f"{mode} exact-downstream KMP2 did not complete.")
            if max(path["kmp2"]["correlation_per_atom_error"], path["kmp2"]["total_per_atom_error"]) > 1e-4:
                raise RuntimeError(f"{mode} failed the frozen KMP2 chemistry gate.")
            downstream[mode] = path

    record = {
        "schema": "pytc-periodic-reciprocal-same-grid-r2/v1",
        "scope": "CPU-only non-default selector benchmark; all downstream AOs remain exact PySCF values.",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": {
            "system": "diamond 211 supercell",
            "basis": "gth-dzvp",
            "pseudo": "gth-pbe",
            "ke_cutoff_hartree": 30.0,
            "mesh": [27, 13, 13],
            "k_mesh": [3, 2, 1],
            "rtol": RTOL,
            "rank": rank,
            "block_size": FROZEN_BLOCK_SIZE,
            "warm_repeats": WARM_REPEATS,
        },
        "n_grid": int(grid_coords.shape[0]),
        "n_kpts": int(mesh_obj.n_kpts),
        "selection_peak_max_bytes": int(peak_max_bytes),
        "peak_safety_factor": float(peak_safety_factor),
        "process_pipeline_allowance_bytes": int(process_pipeline_allowance_bytes),
        "selection": selection,
        "downstream_exact_ao": downstream,
    }
    record["projected_222_333_444"] = _projection(record, n_ao_bulk=26)
    return record


def write_record(path, record):
    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot serialize {type(value).__name__} in benchmark record")

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".reciprocal-r2-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-only", action="store_true")
    options = parser.parse_args(argv)
    write_record(options.output, run_benchmark(run_downstream=not options.selection_only))


if __name__ == "__main__":
    main()
