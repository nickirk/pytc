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
        x_in = src_f[dataset]
        if x_in.ndim != 3 or x_in.shape[0] != x_in.shape[1]:
            raise ValueError(
                f"source dataset {dataset!r} must be (nmo, nmo, rank); "
                f"got {x_in.shape}")
        nmo, _, rank = x_in.shape
        x_out = dst_f.create_dataset(dataset, shape=(rank, nmo, nmo),
                                     dtype=np.float64)
        x_out.attrs["x_layout"] = "rank_major"
        for j0 in range(0, nmo, int(row_block)):
            j1 = min(j0 + int(row_block), nmo)
            slab = np.asarray(x_in[j0:j1, :, :], dtype=np.float64)
            x_out[:, j0:j1, :] = np.ascontiguousarray(
                slab.transpose(2, 0, 1))
        return dst
    finally:
        if close_src:
            src_f.close()
        if close_dst:
            dst_f.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", help="source store path (rank-innermost X)")
    parser.add_argument("dst", help="destination path (rank-major X)")
    parser.add_argument("--dataset", default="X",
                        help="dataset name in both files (default: X)")
    parser.add_argument("--row-block", type=int, default=8,
                        help="nmo rows converted per slab (default: 8)")
    args = parser.parse_args(argv)
    convert_x_to_rank_major(args.src, args.dst, dataset=args.dataset,
                            row_block=args.row_block)


if __name__ == "__main__":
    main()
