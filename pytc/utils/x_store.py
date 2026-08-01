"""X-factor store layout conversion: rank-innermost -> rank-major.

The ISDF stores keep the VVVV X factor as ``(nmo, nmo, rank)`` (rank
innermost).  The streamed/pipelined contraction consumes rank panels, and
in that layout one panel is ~nmo^2 strided chunks -- the measured tier-3
read floor at the 1200 deck is ~103 MiB/s effective (JID 20633292), two
orders of magnitude below sequential NVMe.  In the rank-major layout
``(rank, nmo, nmo)`` a panel is one contiguous block (and already the
rank-leading shape the panel kernels consume, so no transpose either).

This module converts an existing store once; jobs then reuse the
converted artifact exactly like the original store.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


def convert_x_to_rank_major(src, dst, *, dataset="X", row_block=8):
    """Convert one X dataset to rank-major layout, chunked over an nmo axis.

    ``src``/``dst`` are paths or open ``h5py.File`` objects.  The loop
    materializes one ``(nmo, row_block, rank)`` slab at a time, so peak
    host memory is ``nmo * row_block * rank * 8`` bytes (~1.6 GiB at the
    1200 deck with the default block).  Returns the destination path or
    file unchanged.  The destination dataset is written contiguous (no
    HDF5 chunking) so panel reads stay single sequential extents.
    """

    close_src = not isinstance(src, h5py.File)
    close_dst = not isinstance(dst, h5py.File)
    src_f = h5py.File(src, "r") if close_src else src
    dst_f = h5py.File(dst, "w") if close_dst else dst
    try:
        _convert_x_dataset(src_f, dst_f, dataset, row_block)
        return dst
    finally:
        if close_src:
            src_f.close()
        if close_dst:
            dst_f.close()


def _convert_x_dataset(src_f, dst_f, dataset, row_block, out_name=None):
    if int(row_block) < 1:
        raise ValueError(f"row_block must be >= 1; got {row_block}")
    x_in = src_f[dataset]
    if x_in.ndim != 3 or x_in.shape[0] != x_in.shape[1]:
        raise ValueError(
            f"source dataset {dataset!r} must be (nmo, nmo, rank); "
            f"got {x_in.shape}")
    nmo, _, rank = x_in.shape
    x_out = dst_f.create_dataset(out_name or dataset,
                                 shape=(rank, nmo, nmo), dtype=np.float64)
    for key, val in x_in.attrs.items():
        x_out.attrs[key] = val
    x_out.attrs["x_layout"] = "rank_major"
    for j0 in range(0, nmo, int(row_block)):
        j1 = min(j0 + int(row_block), nmo)
        slab = np.asarray(x_in[j0:j1, :, :], dtype=np.float64)
        x_out[:, j0:j1, :] = np.ascontiguousarray(
            slab.transpose(2, 0, 1))


def convert_store_to_rank_major(src, dst, *, x_dataset="X", row_block=8):
    """Whole-store conversion: every dataset copied, X converted rank-major.

    The ISDF store carries more than X (K1/K3/D kernels, metadata); a
    store the driver can actually load needs all of it.  Top-level
    datasets and file attributes are copied verbatim; only ``x_dataset``
    changes layout (and gains the ``x_layout=rank_major`` attribute the
    solver's layout detection reads).

    Writes go to ``<dst>.tmp`` and are atomically renamed into place, so a
    killed or timed-out conversion never leaves a corrupt partial file at
    the production path (JID 20771447 timed out at 82% and left exactly
    that trap).
    """

    tmp = f"{dst}.tmp"
    with h5py.File(src, "r") as src_f, h5py.File(tmp, "w") as dst_f:
        for key, val in src_f.attrs.items():
            dst_f.attrs[key] = val
        for key in src_f:
            if key == x_dataset:
                _convert_x_dataset(src_f, dst_f, x_dataset, row_block)
            else:
                src_f.copy(key, dst_f)
    os.replace(tmp, dst)
    return dst


def add_rank_major(store, *, x_dataset="X", out_dataset="X_rm", row_block=8):
    """Append a rank-major copy of X to an existing store, in place.

    The store keeps ``x_dataset`` (innermost) untouched for legacy
    consumers (eris build, fingerprint manifest); the factorized
    contraction reads ``out_dataset`` instead when present (see
    ``isdf_xtc_ccsd._factorized_state``).  The two layout families want
    opposite axis orders -- (r,s)-tiles are cheap on innermost, rank
    panels only on rank-major -- and carrying both is cheaper than
    rewriting every eris-side consumer (JID 20803176 crashed in
    ``get_delta_h`` on a rank-major-only store).  If ``out_dataset``
    already exists it is shape/attr-verified and kept, so re-running is
    cheap.
    """

    with h5py.File(store, "r+") as fh:
        x_in = fh[x_dataset]
        if x_in.ndim != 3 or x_in.shape[0] != x_in.shape[1]:
            raise ValueError(
                f"{x_dataset!r} must be (nmo, nmo, rank); got {x_in.shape}")
        nmo, _, rank = x_in.shape
        if out_dataset in fh:
            x_rm = fh[out_dataset]
            if (tuple(x_rm.shape) != (rank, nmo, nmo)
                    or x_rm.attrs.get("x_layout") != "rank_major"):
                raise ValueError(
                    f"{out_dataset!r} exists but is not a valid rank-major X: "
                    f"shape {x_rm.shape}, attrs {dict(x_rm.attrs)}")
            return store
        # Write to a temp dataset and rename into place: an interrupted
        # conversion must never leave a correctly-shaped but partially
        # written X_rm behind (review finding).
        tmp_name = out_dataset + ".tmp"
        if tmp_name in fh:
            del fh[tmp_name]
        _convert_x_dataset(fh, fh, x_dataset, row_block, out_name=tmp_name)
        fh.move(tmp_name, out_dataset)
    return store


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", help="source store path (rank-innermost X)")
    parser.add_argument("dst", nargs="?", default=None,
                        help="destination path (not used by --add-rank-major)")
    parser.add_argument("--dataset", default="X",
                        help="dataset name in both files (default: X)")
    parser.add_argument("--row-block", type=int, default=8,
                        help="nmo rows converted per slab (default: 8)")
    parser.add_argument("--whole-store", action="store_true",
                        help="copy all datasets/attrs, converting only "
                             "--dataset (a driver-loadable store)")
    parser.add_argument("--add-rank-major", action="store_true",
                        help="append X_rm (rank-major) to the src store IN "
                             "PLACE, leaving X innermost untouched")
    args = parser.parse_args(argv)
    if args.add_rank_major:
        add_rank_major(args.src, x_dataset=args.dataset,
                       row_block=args.row_block)
    else:
        if args.dst is None:
            parser.error("dst is required unless --add-rank-major is given")
        if args.whole_store:
            convert_store_to_rank_major(
                args.src, args.dst, x_dataset=args.dataset,
                row_block=args.row_block)
        else:
            convert_x_to_rank_major(
                args.src, args.dst, dataset=args.dataset,
                row_block=args.row_block)


if __name__ == "__main__":
    main()
