"""
Canonical GPU memory model for ISDF/XTC tile operations.

Both the build-phase estimators in ``gpu_memory.py`` and the runtime
memory guard in ``xtc._assemble_2b_tile`` import from this single module
so their formulas can never silently diverge.

Usage
-----
>>> from pytc.utils.tile_memory import isdf_tile_peak_bytes, find_max_blksize
"""

__all__ = ["isdf_tile_peak_bytes", "find_max_blksize"]


def isdf_tile_peak_bytes(Np, Nq, Nr, Ns, n_fused, *, include_d=False):
    """Peak device memory for one balanced direct Delta-U tile kernel call.

    The direct contraction kernel computes::

        out[p, q, r, s] = sum_{mu, nu} C_pq[p, q, mu] * D[mu, nu] * C_rs[r, s, nu]

    where the C panels are built on-the-fly from ``phi`` (shape: nmo × n_fused).
    Peak memory is the sum of all simultaneously live buffers:

    =========  =========  ===================================
    Symbol     Shape      Description
    =========  =========  ===================================
    D          Nf × Nf    interaction kernel (persistent)
    X_sliced   Nr×Ns×Nf   pre-sliced X = C_rs input
    C_pq       Np×Nq×Nf   left orbital panel (used twice)
    C_rs       Nr×Ns×Nf   right orbital panel
    out        Np×Nq×Nr×Ns  output (allocated + accumulation)
    =========  =========  ===================================

    Parameters
    ----------
    Np, Nq : int
        First orbital-pair dimensions (after any JIT padding).
    Nr, Ns : int
        Second orbital-pair dimensions (after any JIT padding).
    n_fused : int
        ISDF rank (number of interpolating points / fused basis functions).
    include_d : bool
        Include the D matrix in the estimate.  Pass ``False`` (default)
        when D is already a persistent on-device resident that has been
        subtracted from the available budget before calling this function.

    Returns
    -------
    int
        Peak bytes required on the device.
    """
    if n_fused == 0:
        return 0  # no ISDF computation — no memory allocated by this kernel
    B = 8  # bytes per element: the model assumes float64 throughout
    d_bytes   = int(n_fused ** 2 * B) if include_d else 0
    x_bytes   = int(Nr * Ns * n_fused * B)   # X_sliced / C_rs input
    cpq_bytes = int(Np * Nq * n_fused * B)   # C_pq panel (used twice)
    crs_bytes = x_bytes                       # C_rs = same shape as X_sliced
    out_bytes = int(Np * Nq * Nr * Ns * B)   # one output tile
    # Peak: D + X_sliced + C_pq + C_rs + C_pq (second use) + 2 × out
    return d_bytes + x_bytes + cpq_bytes + crs_bytes + cpq_bytes + 2 * out_bytes


def find_max_blksize(tile_bytes_fn, lo, hi, gpu_target,
                     host_target=None, host_bytes_fn=None):
    """Binary-search for the largest integer ``blk`` in ``[lo, hi]``
    whose tile fits within both GPU and host memory budgets.

    Parameters
    ----------
    tile_bytes_fn : callable(int) -> int
        Peak GPU bytes as a function of tile size.
    lo, hi : int
        Search interval (both inclusive).  Must satisfy ``lo <= hi``.
    gpu_target : int
        Maximum allowed GPU bytes (should already account for persistent
        residents and the desired safety factor).
    host_target : int or None
        Maximum allowed host bytes.  Ignored when ``None``.
    host_bytes_fn : callable(int) -> int or None
        Peak host bytes as a function of tile size.  Required when
        ``host_target`` is not ``None``.

    Returns
    -------
    int
        Best block size found; equals ``lo`` when nothing larger fits.
    """
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        fits_gpu = tile_bytes_fn(mid) <= gpu_target
        fits_host = (
            True
            if host_target is None or host_bytes_fn is None
            else host_bytes_fn(mid) <= host_target
        )
        if fits_gpu and fits_host:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best
