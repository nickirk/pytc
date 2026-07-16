"""R3 selector-only 222 benchmark: streamed, reciprocal, and cached exact.

Each mode is run in a fresh process through ``--mode``.  The companion
``--summarize`` operation validates the exact-selector identity gate and
records reciprocal selection quality without treating it as exact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import resource
import tempfile
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from pyscf.pbc import gto, tools

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
from pytc.pbc.df.reciprocal_selection_study import RTOL, _residual_for_fixed_pivots


FROZEN_222_MESH = (27, 27, 27)
FROZEN_KMESH = (3, 2, 1)
# The R2 per-NAO rank projection is 2496, but the conservative streamed
# time model rejects it.  All R3 modes therefore use this documented lower
# matched rank.
FROZEN_RANK = 624
MAX_MODELED_RANK = 2496
BLOCK_SIZE = 256
PROCESS_PIPELINE_ALLOWANCE_BYTES = 384 * 2**20
SELECTION_PEAK_MAX_BYTES = 24 * 2**30
R2_211_NGRID = 4563
R2_211_NAO = 52
R3_222_NAO = 208
R2_211_STREAMED_RANK300_SECONDS = 42.73
R2_211_RECIPROCAL_WARM_SECONDS = 8.323912666004617
R2_211_RECIPROCAL_RECONSTRUCTIONS = 625


def diamond_222():
    """Frozen R3 diamond input, extending R1's 211 grid density to 222."""
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
    cell = tools.super_cell(primitive, [2, 2, 2])
    cell.mesh = np.asarray(FROZEN_222_MESH, dtype=np.int32)
    return cell


def frozen_222_partition(cell):
    """The eight internal primitive-cell replicas in the frozen 222 cell."""
    shells_per_primitive = 6
    if cell.nbas != 8 * shells_per_primitive:
        raise RuntimeError("frozen 222 shell layout no longer has eight primitive groups.")
    atom_coords = np.asarray(cell.atom_coords(), dtype=np.float64)
    translations = np.asarray([
        atom_coords[2 * group] - atom_coords[0] for group in range(1, 8)
    ])
    return ReciprocalOrbitPartition(
        seed_shell_slice=(0, shells_per_primitive),
        replica_shell_slices=tuple(
            (group * shells_per_primitive, (group + 1) * shells_per_primitive)
            for group in range(1, 8)
        ),
        replica_translations=translations,
        supercell_matrix=np.diag([2, 2, 2]),
    )


def _rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _timed(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, {
        "seconds": time.perf_counter() - started,
        "rss_peak_bytes": _rss_bytes(),
    }


def _runner(mode, cell, mesh_obj, coords, rank):
    if mode == "streamed":
        def run():
            diagonal, column = build_periodic_pivot_oracle(
                cell, mesh_obj.canonical_kpts, coords, BLOCK_SIZE,
            )
            pivots, _, count = pivoted_cholesky_hermitian(diagonal, column, rank=rank)
            return pivots, count, {"mode": mode}
        return run
    if mode == "reciprocal_same_grid":
        def run():
            pivots, _, count, provenance = select_reciprocal_same_grid(
                cell, mesh_obj.canonical_kpts, coords, rank, frozen_222_partition(cell),
                selection_peak_max_bytes=SELECTION_PEAK_MAX_BYTES,
                process_pipeline_allowance_bytes=PROCESS_PIPELINE_ALLOWANCE_BYTES,
            )
            return pivots, count, provenance
        return run
    if mode == "jax_cached_matrix_free":
        def run():
            _, _, cache = build_cached_periodic_pivot_oracle(
                cell, mesh_obj.canonical_kpts, coords, BLOCK_SIZE,
            )
            pivots, _, count, provenance = select_jax_cached_matrix_free(
                cache, rank, selection_peak_max_bytes=SELECTION_PEAK_MAX_BYTES,
            )
            return pivots, count, provenance
        return run
    raise ValueError(f"unsupported R3 mode {mode!r}")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_record(path, record):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".r3-selector-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def time_model(*, rank=FROZEN_RANK):
    """Conservative pre-submission model, not a capacity or runtime claim."""
    if rank <= 0 or rank > MAX_MODELED_RANK:
        raise ValueError(f"rank must be in [1,{MAX_MODELED_RANK}].")
    n_grid = int(np.prod(FROZEN_222_MESH))
    grid_factor = n_grid / R2_211_NGRID
    fft_size_factor = (n_grid * math.log(n_grid)) / (R2_211_NGRID * math.log(R2_211_NGRID))
    reciprocal_reconstructions = 7 * (rank + 1)
    nao_factor = R3_222_NAO / R2_211_NAO
    streamed_seconds = (
        R2_211_STREAMED_RANK300_SECONDS * nao_factor * grid_factor * rank / 300
    )
    reciprocal_seconds = (
        R2_211_RECIPROCAL_WARM_SECONDS
        * reciprocal_reconstructions / R2_211_RECIPROCAL_RECONSTRUCTIONS
        * fft_size_factor
    )
    projected_selector_seconds = 1.15 * (4 * streamed_seconds + 6 * reciprocal_seconds)
    return {
        "kind": "conservative_pre_submission_time_model_not_measurement",
        "rank": rank,
        "n_grid": n_grid,
        "streamed_nao_factor": nao_factor,
        "streamed_single_run_seconds": streamed_seconds,
        "reciprocal_single_run_seconds": reciprocal_seconds,
        "cached_exact_single_run_seconds": "not extrapolated; anchor-only and expected inexpensive",
        "reciprocal_reconstructions": reciprocal_reconstructions,
        "reciprocal_fft_count": 2 * 6 * reciprocal_reconstructions,
        "planned_runs": {
            "streamed": {"cold": 1, "warm": 3},
            "reciprocal_same_grid": {"cold": 1, "warm": 5},
            "jax_cached_matrix_free": {"cold": 1, "warm": 5},
        },
        "walltime_envelope_seconds": 5 * 3600,
        "projected_selector_seconds_with_15pct_overhead": projected_selector_seconds,
        "fits_envelope_by_model": projected_selector_seconds <= 5 * 3600,
        "rank_binding": (
            "R3 runs rank=624 for every mode; 1248 and 2496 are rejected by this model."
        ),
    }


def run_mode(mode, *, rank=FROZEN_RANK, warm_repeats=None):
    if mode not in {"streamed", "reciprocal_same_grid", "jax_cached_matrix_free"}:
        raise ValueError("mode must be streamed, reciprocal_same_grid, or jax_cached_matrix_free.")
    if rank != FROZEN_RANK:
        raise ValueError(f"R3 execution is frozen at matched rank={FROZEN_RANK}.")
    if warm_repeats is None:
        warm_repeats = 3 if mode == "streamed" else 5
    if warm_repeats < 1:
        raise ValueError("warm_repeats must be positive.")
    cell = diamond_222()
    kpts = cell.make_kpts(FROZEN_KMESH, wrap_around=False)
    mesh_obj = canonicalize_kpts(cell, kpts)
    coords = cell.get_uniform_grids(cell.mesh)
    runner = _runner(mode, cell, mesh_obj, coords, rank)
    cold, cold_measurement = _timed(runner)
    warm = [_timed(runner) for _ in range(warm_repeats)]
    pivots, count, provenance = warm[-1][0]
    all_runs = (cold,) + tuple(value for value, _ in warm)
    if not all(np.array_equal(pivots, value[0]) and count == value[1] for value in all_runs):
        raise RuntimeError(f"{mode} failed deterministic within-mode pivot gate.")
    return {
        "schema": "pytc-periodic-r3-222-selector-mode/v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode,
        "input": {
            "supercell": [2, 2, 2], "mesh": list(FROZEN_222_MESH),
            "k_mesh": list(FROZEN_KMESH), "ke_cutoff_hartree": 30.0,
            "basis": "gth-dzvp", "pseudo": "gth-pbe", "rtol": RTOL,
            "rank": rank, "block_size": BLOCK_SIZE,
        },
        "cold": cold_measurement,
        "warm_raw": [measurement for _, measurement in warm],
        "warm_median_seconds": float(np.median([measurement["seconds"] for _, measurement in warm])),
        "pivots": np.asarray(pivots, dtype=np.int64),
        "n_selected": int(count),
        "selection_provenance": provenance,
        "time_model": time_model(rank=rank),
    }


def summarize(records):
    """Apply the amended R3 exactness and approximate-selector quality gates."""
    by_mode = {record["mode"]: record for record in records}
    required = {"streamed", "reciprocal_same_grid", "jax_cached_matrix_free"}
    if set(by_mode) != required:
        raise ValueError(f"summary requires exactly {sorted(required)}.")
    streamed = np.asarray(by_mode["streamed"]["pivots"], dtype=np.int64)
    cached = np.asarray(by_mode["jax_cached_matrix_free"]["pivots"], dtype=np.int64)
    reciprocal = np.asarray(by_mode["reciprocal_same_grid"]["pivots"], dtype=np.int64)
    if not np.array_equal(streamed, cached):
        raise RuntimeError("R3 exact-selector gate failed: streamed and cached pivots differ.")
    cell = diamond_222()
    kpts = cell.make_kpts(FROZEN_KMESH, wrap_around=False)
    mesh_obj = canonicalize_kpts(cell, kpts)
    coords = cell.get_uniform_grids(cell.mesh)
    diagonal, column, _ = build_cached_periodic_pivot_oracle(
        cell, mesh_obj.canonical_kpts, coords, BLOCK_SIZE,
    )
    reciprocal_time = by_mode["reciprocal_same_grid"]["warm_median_seconds"]
    streamed_time = by_mode["streamed"]["warm_median_seconds"]
    return {
        "schema": "pytc-periodic-r3-222-selector-summary/v1",
        "modes": by_mode,
        "exact_selector_identity": True,
        "reciprocal_vs_exact": {
            "pivot_overlap_count": int(np.intersect1d(reciprocal, streamed, assume_unique=True).size),
            "same_position_count": int(np.count_nonzero(reciprocal == streamed)),
            "exact_metric_residual": _residual_for_fixed_pivots(diagonal, column, reciprocal),
        },
        "pairwise_warm_time_ratios": {
            "reciprocal_over_streamed": reciprocal_time / streamed_time,
            "reciprocal_over_cached": reciprocal_time / by_mode["jax_cached_matrix_free"]["warm_median_seconds"],
            "cached_over_streamed": by_mode["jax_cached_matrix_free"]["warm_median_seconds"] / streamed_time,
        },
        "decision": {
            "reciprocal_vs_streamed": "terminal_negative" if reciprocal_time >= streamed_time else "reciprocal_faster_than_streamed",
            "scope": "selector-only timing and selection-quality evidence; no downstream chemistry or production promotion",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("streamed", "reciprocal_same_grid", "jax_cached_matrix_free"))
    parser.add_argument("--summarize", nargs=3, metavar="MODE_RECORD")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=FROZEN_RANK)
    parser.add_argument("--warm-repeats", type=int)
    parser.add_argument("--time-model", action="store_true")
    options = parser.parse_args(argv)
    if options.time_model:
        if options.mode or options.summarize:
            parser.error("--time-model cannot be combined with --mode or --summarize.")
        record = time_model(rank=options.rank)
    elif options.mode:
        record = run_mode(options.mode, rank=options.rank, warm_repeats=options.warm_repeats)
    elif options.summarize:
        with_records = [json.load(open(path, encoding="utf-8")) for path in options.summarize]
        record = summarize(with_records)
    else:
        parser.error("one of --time-model, --mode, or --summarize is required.")
    write_record(options.output, record)


if __name__ == "__main__":
    main()
