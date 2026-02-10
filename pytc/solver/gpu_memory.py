"""
Centralized GPU memory budgeting utilities for XTC-CCSD solvers.

Replaces the fragile pattern of querying ``jax.devices()[0].memory_stats()``
and applying a fixed 0.3 fraction.  Instead, every call site declares which
computation *phase* it is about to enter, and this module returns a safe
block-size based on:

1. ``gpu_max_memory`` — **authoritative** user override (MB).  When set
   (and > 0) this is the assumed total GPU budget; no runtime query is done.
2. Analytically computed persistent residents (t1, t2, tau, ERIs …).
3. Phase-specific workspace formulas so we know *exactly* how much VRAM
   a single block of a given size will consume.

Usage
-----
>>> from pytc.solver.gpu_memory import estimate_blksize
>>> blk, budget = estimate_blksize(nocc, nvir, 'ovvv',
...                                 gpu_max_memory_mb=24000)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent GPU residents
# ---------------------------------------------------------------------------

def estimate_persistent_gpu_bytes(nocc, nvir, include_eris=True,
                                  include_accumulators=False):
    """Analytically estimate bytes occupied by tensors known to live on GPU.

    Parameters
    ----------
    nocc, nvir : int
        Number of occupied / virtual orbitals.
    include_eris : bool
        Include the 7 ERI blocks that ``_update_amps`` loads.
    include_accumulators : bool
        Include the 4 W / tmp accumulators kept on GPU when
        ``use_gpu_acc=True``.

    Returns
    -------
    int
        Estimated bytes (float64).
    """
    B = 8  # bytes per float64 element
    O, V = nocc, nvir

    mem = 0
    # Amplitudes
    mem += O * V * B                   # t1
    mem += O * O * V * V * B           # t2
    mem += O * O * V * V * B           # tau

    # Fock / diag vectors (small)
    nmo = O + V
    mem += nmo * nmo * B               # fock_jax
    mem += O * B + V * B               # mo_e_o, mo_e_v

    if include_eris:
        mem += (O * V) ** 2 * B        # ovov
        mem += (O * V) ** 2 * B        # ovvo
        mem += O * O * V * V * B       # oovv
        mem += O * V * O * O * B       # ovoo
        mem += O ** 4 * B              # oooo
        mem += V * O ** 3 * B          # vooo
        mem += (V * O) ** 2 * B        # vovo

    if include_accumulators:
        mem += V * O * O * V * B       # Wvoov_acc
        mem += V * O * V * O * B       # Wvovo_acc
        mem += O * V * O * O * B       # tmp_a_acc
        mem += O * V * O * O * B       # tmp_b_acc

    return mem


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_gpu_budget_bytes(gpu_max_memory_mb=None):
    """Return the total usable GPU memory budget in bytes.

    If *gpu_max_memory_mb* is provided and > 0 it is treated as the
    authoritative limit (no runtime query).  Otherwise we try
    ``jax.devices()[0].memory_stats()`` and fall back to a conservative
    16 GiB default.

    Returns
    -------
    int
        Total GPU budget in bytes.
    """
    if gpu_max_memory_mb is not None and gpu_max_memory_mb > 0:
        return int(gpu_max_memory_mb * 1024 ** 2)

    try:
        import jax
        stats = jax.devices()[0].memory_stats()
        return int(stats['bytes_limit'])
    except Exception:
        return 16 * 1024 ** 3  # 16 GiB fallback


def estimate_blksize(nocc, nvir, phase, *,
                     gpu_max_memory_mb=None,
                     host_max_memory_mb=None,
                     include_accumulators=False,
                     naux=None):
    """Compute a safe block-size for *phase* from first-principles memory accounting.

    Parameters
    ----------
    nocc, nvir : int
        Occupied / virtual counts.
    phase : str
        One of ``'ovvv'``, ``'vovv'``, ``'vvvv'``, ``'acc_decision'``,
        ``'ovvv_eri_build'``, ``'vovv_eri_build'``.
    gpu_max_memory_mb : float | None
        User override for total GPU memory (MB).  **Authoritative** when set.
    host_max_memory_mb : float | None
        Host memory budget (MB).  Used to apply ``min(host, gpu)`` capping.
    include_accumulators : bool
        Whether the 4 accumulator tensors are present on GPU at this point.
    naux : int | None
        DF auxiliary basis size (needed for ``'vvvv'`` phase with DF).

    Returns
    -------
    blksize : int
        Safe block-size (≥ 1).
    budget_bytes : int
        The total GPU budget used for this estimate.
    """
    O, V, B = nocc, nvir, 8

    # 1. Total budget ----------------------------------------------------------
    gpu_budget = get_gpu_budget_bytes(gpu_max_memory_mb)
    if host_max_memory_mb is not None and host_max_memory_mb > 0:
        host_budget = int(host_max_memory_mb * 1e6)  # PySCF convention: MB → bytes
        budget = min(gpu_budget, host_budget)
    else:
        budget = gpu_budget

    # 2. Subtract persistent residents -----------------------------------------
    persistent = estimate_persistent_gpu_bytes(
        nocc, nvir, include_eris=True,
        include_accumulators=include_accumulators)
    available = max(budget - persistent, 0)

    # 3. Phase-specific workspace per unit of blk -----------------------------
    if phase == 'ovvv':
        # ovvv_blk: (O, V, blk, V) loaded to GPU
        # kernel outputs: t1_upd(O,blk) + Lvv_blk(blk,V) + Wvoov_blk(blk,O,O,V)
        #   + Wvovo_blk(blk,O,V,O) + tmp_a_blk(O,blk,O,O) + tmp_b_blk(O,blk,O,O)
        per_blk = (O * V * V            # input ovvv slice per unit blk
                   + O                   # t1_upd [:,blk] per unit
                   + V                   # Lvv_blk [blk,:]
                   + O * O * V           # Wvoov_blk
                   + O * V * O           # Wvovo_blk
                   + O * O * O           # tmp_a_blk
                   + O * O * O           # tmp_b_blk
                   ) * B
        # Also count JIT intermediates (generous 2x factor)
        per_blk *= 2

    elif phase == 'vovv':
        # vovv_slice: (blk, O, V, V) + tmp2: (blk, V, O, V) + term: (O, O, blk, V)
        per_blk = (O * V * V + V * O * V + O * O * V) * B
        per_blk *= 2

    elif phase == 'vvvv':
        # vvvv_block: (blk, V, V, V) from get_2b + std_block: (blk, V, V, V)
        # + L_vv_full: (V, V, naux) if DF — counted as overhead, not per-blk
        per_blk = 2 * V * V * V * B
        # L_vv_full overhead (if DF)
        if naux is not None and naux > 0:
            overhead = V * V * naux * B
            available = max(available - overhead, 0)
        per_blk *= 2  # JIT intermediates

    elif phase == 'acc_decision':
        # This phase is just deciding whether accumulators fit on GPU.
        # Return the accumulator size directly.
        acc_bytes = (V * O * O * V * 2 + O * V * O * O * 2) * B
        can_fit = available * 0.8 > acc_bytes
        return int(can_fit), budget

    elif phase in ('ovvv_eri_build', 'vovv_eri_build'):
        # During _compute_large_blocks: similar to ovvv/vovv but the
        # persistent residents are different (no t2/tau/ERIs on GPU yet).
        persistent_build = (O + V) ** 2 * B  # only fock-like small arrays
        available = max(budget - persistent_build, 0)
        per_blk = O * V * V * B * 2  # tc_blk + std_blk
        per_blk *= 2

    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    # 4. Safety margin (20%) ---------------------------------------------------
    usable = available * 0.8
    blksize = max(1, int(usable / per_blk))
    blksize = min(nvir, blksize)

    logger.debug(
        "estimate_blksize(phase=%s): budget=%.2f GB, persistent=%.2f GB, "
        "available=%.2f GB, per_blk=%.2f MB → blksize=%d",
        phase, budget / 1e9, persistent / 1e9,
        available / 1e9, per_blk / 1e6, blksize)

    return blksize, budget


def adaptive_rank_block_size(Np, Nq, N_fused, *,
                              gpu_max_memory_mb=None,
                              min_block=64, max_block=2048):
    """Compute the largest safe ``rank_block_size`` for ISDF scan contractions.

    The peak intermediate per scan step in ``contract_K1_isdf_jit`` is::

        W: (Np, N_fused, rank_block_size) × 8 bytes
        T: (Np, Nq, rank_block_size) × 8 bytes

    and in ``_contract_delta_U_kernels_jit`` (D-term)::

        W: (N_fused, rank_block_size, Nq) × 8 bytes

    This function picks the largest power-of-2 block size that keeps the
    peak intermediate within 50% of the available GPU budget.

    Parameters
    ----------
    Np, Nq : int
        Orbital slice sizes for the bra / ket indices.
    N_fused : int
        ISDF rank (dimension being scanned over).
    gpu_max_memory_mb : float | None
        User override for total GPU memory.
    min_block, max_block : int
        Clamps on the returned value.

    Returns
    -------
    int
        Power-of-2 block size.
    """
    budget = get_gpu_budget_bytes(gpu_max_memory_mb)
    B = 8  # float64

    # Worst-case peak: max of K1-style and delta_U-style intermediates
    peak_per_rank_K1 = (Np * N_fused + Np * Nq) * B
    peak_per_rank_DU = (N_fused * Nq + Np * Nq) * B
    peak_per_rank = max(peak_per_rank_K1, peak_per_rank_DU, 1)

    # Use 50% of budget (the rest is for the accumulator + other tensors)
    max_rank_block = max(min_block, int(budget * 0.5 / peak_per_rank))

    # Round down to power of 2
    import math
    max_rank_block = 2 ** int(math.log2(max(max_rank_block, 1)))
    max_rank_block = min(max_rank_block, max_block, N_fused)
    max_rank_block = max(max_rank_block, min_block)

    logger.debug(
        "adaptive_rank_block_size(Np=%d, Nq=%d, N_fused=%d): "
        "peak_per_rank=%.2f MB → rank_block_size=%d",
        Np, Nq, N_fused, peak_per_rank / 1e6, max_rank_block)

    return max_rank_block
