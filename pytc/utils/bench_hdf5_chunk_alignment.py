"""Micro-benchmark for the ovvv/vovv HDF5 chunk-alignment fix.

This script mirrors ONLY the HDF5 write path of ``_compute_large_blocks``
in ``pytc/solver/xtc_ccsd.py`` — no GPUs, no XTC, no DF.  It creates an
``ovvv``-shaped dataset with two alternative chunk shapes and writes the
same sequence of tile slabs to each, timing the writes.

Why this isolates the fix
--------------------------
The only thing changed for the large-blocks pipeline write path is the
HDF5 chunk shape on axis 2:

  DEFAULT (h5py auto-chunking, ``chunks=True``): h5py's own heuristic
      picks a chunk shape from the dataset shape/dtype alone, with no
      knowledge of ``panel_blk`` -- the actual write-slab size the
      pipeline uses.
  ALIGNED (this work): axis-2 chunk size set equal to ``panel_blk``.

Each tile is written as ``ovvv[:, :, r0:r0+panel_blk, :]``.  Whenever the
auto-picked axis-2 chunk size does not evenly divide ``panel_blk`` (the
general case), each write slab straddles chunk boundaries and HDF5 must
read-modify-write the boundary chunks.  The ALIGNED layout picks the
axis-2 chunk equal to ``panel_blk`` so every write slab lands exactly on
a chunk boundary -- no RMW.

This compares against h5py's real auto-chunking (matching the "aligned
vs default auto-chunking" claim in the manuscript), not a hand-picked
"old" shape -- see the git history around commit 5366fed for the
previous (hand-shape-vs-hand-shape) version of this benchmark, which
measured a different comparison.

Run
---
    python -m pytc.utils.bench_hdf5_chunk_alignment

Scale down ``nvir`` if disk space is tight.
"""

import argparse
import os
import tempfile
import time

import h5py
import numpy as np


def run_write_bench(path, shape, chunks, panel_blk, label):
    """Write the full ovvv tensor in tiles of panel_blk slabs on axis 2."""
    nocc, nvir_1, nvir_2, nvir_3 = shape
    tile = np.random.default_rng(0).standard_normal(
        (nocc, nvir_1, panel_blk, nvir_3), dtype=np.float64
    )

    # Create a fresh file for each run so OS page cache does not favour run 2.
    if os.path.exists(path):
        os.remove(path)

    t_open = time.perf_counter()
    with h5py.File(path, "w") as f:
        dset = f.create_dataset("ovvv", shape, dtype="f8", chunks=chunks)
        actual_chunks = dset.chunks
        t_create = time.perf_counter()

        n_tiles = -(-nvir_2 // panel_blk)  # ceil
        t_write_start = time.perf_counter()
        for i in range(n_tiles):
            r0 = i * panel_blk
            r1 = min(r0 + panel_blk, nvir_2)
            if r1 - r0 == panel_blk:
                dset[:, :, r0:r1, :] = tile
            else:
                dset[:, :, r0:r1, :] = tile[:, :, : r1 - r0, :]
        # Ensure all writes hit disk before we stop the clock.
        f.flush()
        t_end = time.perf_counter()

    file_bytes = os.path.getsize(path)

    bytes_written = np.prod(shape) * 8
    print(f"  [{label}]")
    print(f"    chunks requested   : {chunks}")
    print(f"    chunks actual      : {actual_chunks}")
    print(f"    chunk bytes        : {np.prod(actual_chunks) * 8 / 1e6:.2f} MB")
    print(f"    n tiles            : {n_tiles}")
    print(f"    dataset create     : {t_create - t_open:.3f} s")
    print(f"    write + flush      : {t_end - t_write_start:.3f} s")
    print(f"    effective throughput: {bytes_written / (t_end - t_write_start) / 1e9:.2f} GB/s")
    print(f"    file size on disk  : {file_bytes / 1e9:.2f} GB")
    print()
    return t_end - t_write_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nocc", type=int, default=21,
                        help="Occupied dimension (default 21 -- benzene/cc-pCV5Z scale)")
    parser.add_argument("--nvir", type=int, default=400,
                        help="Virtual dimension (default 400; production is ~1179)")
    parser.add_argument("--panel-blk", type=int, default=22,
                        help="r-slab tile size (default 22 -- mimics resolve_v3o_panel_block_size)")
    parser.add_argument("--tmpdir", default=None,
                        help="Directory for temp HDF5 files (default system tmp)")
    args = parser.parse_args()

    nocc, nvir, panel_blk = args.nocc, args.nvir, args.panel_blk
    shape = (nocc, nvir, nvir, nvir)

    aligned_chunks = (nocc, min(64, nvir), min(panel_blk, nvir), min(64, nvir))

    total_bytes = np.prod(shape) * 8
    print(f"Benchmark: ovvv shape={shape}  ({total_bytes / 1e9:.2f} GB on disk)")
    print(f"           panel_blk={panel_blk}  tiles={-(-nvir // panel_blk)}")
    print()

    tmpdir = args.tmpdir or tempfile.gettempdir()
    path_default = os.path.join(tmpdir, "bench_ovvv_default.h5")
    path_aligned = os.path.join(tmpdir, "bench_ovvv_aligned.h5")

    t_default = run_write_bench(path_default, shape, True, panel_blk,
                                "DEFAULT (h5py auto-chunk, chunks=True)")
    t_aligned = run_write_bench(path_aligned, shape, aligned_chunks, panel_blk,
                                "ALIGNED (panel_blk on axis-2)")

    print("-" * 60)
    print(f"  speedup (DEFAULT / ALIGNED): {t_default / t_aligned:.2f}x")
    print("-" * 60)

    # Clean up so repeated runs don't leave stale multi-GB files around.
    for p in (path_default, path_aligned):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
