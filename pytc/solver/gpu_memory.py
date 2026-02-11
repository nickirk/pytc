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


def _get_gpu_physical_bytes():
    """Return the physical GPU memory capacity in bytes.

    Queries the actual hardware limit (total device memory), which is
    independent of JAX's pre-allocation fraction.
    """
    try:
        import jax
        stats = jax.devices()[0].memory_stats()
        # bytes_limit is JAX's pool limit (fraction of physical).
        # For the *physical* capacity we want the pool limit divided by
        # the pre-allocation fraction — but that's tricky to get.
        # Instead, return bytes_limit as a conservative proxy.
        return int(stats['bytes_limit'])
    except Exception:
        return 80 * 1024 ** 3  # 80 GiB fallback (A100)


def _get_gpu_free_bytes():
    """Return the *currently free* GPU memory inside JAX's pre-allocated pool.

    This accounts for JAX's pre-allocation fraction (default 75% of physical)
    AND any tensors that are already resident on the device.  Much safer than
    ``bytes_limit`` which ignores current allocations.

    Falls back to ``_get_gpu_physical_bytes() * 0.75`` if stats are unavailable.
    """
    try:
        import jax
        stats = jax.devices()[0].memory_stats()
        pool_limit = int(stats['bytes_limit'])
        in_use = int(stats.get('bytes_in_use', 0))
        free = pool_limit - in_use
        logger.debug(
            "_get_gpu_free_bytes: pool_limit=%.2f GB, in_use=%.2f GB, free=%.2f GB",
            pool_limit / 1e9, in_use / 1e9, free / 1e9)
        return max(free, 0)
    except Exception:
        return int(_get_gpu_physical_bytes() * 0.75)


def estimate_blksize(nocc, nvir, phase, *,
                     gpu_max_memory_mb=None,
                     host_max_memory_mb=None,
                     include_accumulators=False,
                     naux=None,
                     n_fused=None):
    """Compute a safe block-size for *phase* from first-principles memory accounting.

    Parameters
    ----------
    nocc, nvir : int
        Occupied / virtual counts.
    phase : str
        One of ``'ovvv'``, ``'vovv'``, ``'vvvv'``, ``'vvvv_gpu'``,
        ``'acc_decision'``, ``'ovvv_eri_build'``, ``'vovv_eri_build'``.
        ``'vvvv'`` is for the host (NumPy) contraction path in xtc_ccsd;
        ``'vvvv_gpu'`` is for the GPU (JAX) contraction path in
        jax_xtc_ccsd, where L_vv_full, t2, and the contraction kernel
        all reside on device.
    gpu_max_memory_mb : float | None
        User override for total GPU memory (MB).  **Authoritative** when set.
    host_max_memory_mb : float | None
        Host memory budget (MB).  Used to apply ``min(host, gpu)`` capping.
    include_accumulators : bool
        Whether the 4 accumulator tensors are present on GPU at this point.
    naux : int | None
        DF auxiliary basis size.  Used for host-side ``L_vv_full`` overhead
        in the ``'vvvv'`` phase, and as ISDF N_fused rank for
        ``'ovvv_eri_build'``/``'vovv_eri_build'`` phases when *n_fused*
        is not provided.
    n_fused : int | None
        ISDF fused rank (``phi_isdf.shape[1]``).  When provided, used for
        GPU scan workspace estimation instead of *naux*.  This is typically
        larger than *naux* and is the quantity that actually determines the
        scan memory footprint.

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

    elif phase in ('vvvv', 'vvvv_gpu'):
        # Two variants:
        # - 'vvvv': xtc_ccsd.py — contraction on HOST (NumPy).  GPU only
        #   runs get_2b, then result is transferred to host.
        # - 'vvvv_gpu': jax_xtc_ccsd.py — contraction on GPU (JAX).
        #   After get_2b, the result stays on GPU and contract_block_kernel
        #   runs tensordot + einsum, requiring additional GPU memory.
        gpu_contraction = (phase == 'vvvv_gpu')

        # _contract_vvvv_t2: working arrays live on HOST (NumPy) for 'vvvv',
        #   or on GPU (JAX) for 'vvvv_gpu'.
        # With in-place addition (vvvv_block += std_block), at most 2 copies
        # of (blk,V,V,V) coexist.  After the add, std_block is deleted.
        # xtc_obj.get_2b() produces the output on GPU first (the
        # ISDF scan carry is (blk,V,V,V)), so the GPU must fit that tensor.

        # --- Host constraint (only binding for 'vvvv') ---
        if not gpu_contraction:
            # For the host path, use host_budget directly (not GPU-capped 'available')
            if host_max_memory_mb is not None and host_max_memory_mb > 0:
                host_avail = int(host_max_memory_mb * 1e6)
            else:
                host_avail = available  # fall back to whatever budget we have
            
            # Subtract persistent host residents + vvvv-specific overhead
            host_avail = max(host_avail - persistent, 0)
            
            # Additional constant host overhead for this phase: L_vv_full
            constant_host = 0
            # naux here is the DF auxiliary basis size, not ISDF N_fused
            if naux is not None and naux > 0:
                constant_host += V * V * naux * B     # L_vv_full (V,V,naux)
            host_avail = max(host_avail - constant_host, 0)

            # Peak per-blk: 2× (blk,V,V,V) — vvvv_block + std_block coexist
            # briefly. The einsum('abcd,ijcd->ijab', vvvv_block, t2) may
            # also create temporaries.  Use 4× to match the previous safe estimate.
            host_per_blk = int(4 * V * V * V * B)
            host_blk = max(1, int(host_avail * 0.8 / host_per_blk))
        else:
            host_blk = nvir  # not the binding constraint for GPU path

        # --- GPU constraint ---
        # get_2b runs ISDF scan with carry + contribution = 2×(blk,V,V,V)
        # plus W workspace (constant, independent of blk).
        # The scan workspace depends on the ISDF N_fused rank (n_fused),
        # NOT the DF auxiliary basis size (naux).  n_fused is typically
        # much larger (e.g. 5050 vs ~500 for naux).
        gpu_free = _get_gpu_free_bytes()

        # Constant GPU residents for the GPU contraction path:
        #   L_vv_full_jax (V,V,naux_DF), t2_jax (O,O,V,V)
        constant_gpu = 0
        if gpu_contraction:
            if naux is not None and naux > 0:
                constant_gpu += V * V * naux * B     # L_vv_full_jax on GPU
            constant_gpu += O * O * V * V * B        # t2_jax on GPU
        gpu_free = max(gpu_free - constant_gpu, 0)

        _N_fused = n_fused if n_fused is not None else (naux if naux is not None else 0)
        if _N_fused > 0:
            # Estimate scan workspace (W intermediate) same as eri_build
            _peak_per_rank = max(V * _N_fused + V * V,
                                 _N_fused * V + V * V, 1) * B
            import math
            _rank_blk = max(64, int(gpu_free * 0.5 / _peak_per_rank))
            _rank_blk = 2 ** int(math.log2(max(_rank_blk, 1)))
            _rank_blk = min(_rank_blk, 2048, _N_fused)
            _rank_blk = max(_rank_blk, 64)
            scan_workspace = _N_fused * _rank_blk * V * B
        else:
            scan_workspace = 0

        if gpu_contraction:
            # Inside contract_block_kernel (all on GPU simultaneously):
            #   xtc_block          (blk,V,V,V) — from get_2b, stays on GPU
            #   tensordot result   (blk,V,V,V) — L_ab_sub @ L_vv_full
            #   sum                (blk,V,V,V) — xtc_block + tensordot
            #   einsum result      (O,O,blk,V) — output
            # XLA may fuse some of these, but peak ≈ 3× (blk,V,V,V).
            # Plus the get_2b scan needs carry+contribution = 2× (blk,V,V,V),
            # but those are freed before the kernel runs.
            # Conservative: 3× (blk,V,V,V) for the kernel.
            gpu_per_blk = V * V * V * B * 3
        else:
            gpu_per_blk = V * V * V * B * 2   # carry + contribution from get_2b scan

        gpu_available = max(gpu_free - scan_workspace, 0)
        gpu_blk = max(1, int(gpu_available * 0.65 / gpu_per_blk))

        blksize = min(host_blk, gpu_blk, nvir)
        logger.debug(
            "estimate_blksize(phase=%s): host_blk=%d, "
            "gpu_free=%.2f GB (after constant_gpu=%.2f GB), "
            "N_fused=%d, scan_ws=%.2f GB, gpu_per_blk=%.2f MB, "
            "gpu_blk=%d → blksize=%d",
            phase, host_blk,
            gpu_free / 1e9, constant_gpu / 1e9,
            _N_fused, scan_workspace / 1e9, gpu_per_blk / 1e6,
            gpu_blk, blksize)
        return blksize, budget

    elif phase == 'acc_decision':
        # This phase is just deciding whether accumulators fit on GPU.
        # Return the accumulator size directly.
        acc_bytes = (V * O * O * V * 2 + O * V * O * O * 2) * B
        can_fit = available * 0.8 > acc_bytes
        return int(can_fit), budget

    elif phase in ('ovvv_eri_build', 'vovv_eri_build'):
        # Host side: std_blk + tc_blk per iteration.
        # L_vv_full / Lov_reshaped are already allocated — not subtracted.
        if host_max_memory_mb is not None and host_max_memory_mb > 0:
            host_budget = int(host_max_memory_mb * 1e6)
        else:
            host_budget = budget  # fall back to GPU budget as proxy
        host_per_blk = O * V * V * B * 2  # std_blk + tc_blk

        # GPU side: query *actually free* memory in JAX's pool.
        # This already accounts for the pre-allocation fraction (default 75%)
        # AND any resident tensors (ISDF kernels, phi_isdf, etc.).
        gpu_free = _get_gpu_free_bytes()

        # ----- Scan workspace overhead (constant, independent of blk) -----
        # Inside each ISDF scan JIT (contract_K1_minus_K2_isdf_jit, etc.)
        # the dominant intermediate is:
        #   W: (N_fused, rank_block_size, max(Np,Nq)) × 8 bytes
        # This must be subtracted from the GPU budget *before* we divide
        # by the per-blk cost (carry + contribution).
        #
        # Compute rank_block_size the same way adaptive_rank_block_size does:
        N_fused = n_fused if n_fused is not None else (naux if naux is not None else 0)
        if N_fused > 0:
            _Np = O  # bra dimension (nocc for ovvv/vovv)
            _Nq = V  # ket dimension
            _peak_per_rank = max(_Np * N_fused + _Np * _Nq,
                                 N_fused * _Nq + _Np * _Nq, 1) * B
            import math
            _rank_blk = max(64, int(gpu_free * 0.5 / _peak_per_rank))
            _rank_blk = 2 ** int(math.log2(max(_rank_blk, 1)))
            _rank_blk = min(_rank_blk, 2048, N_fused)
            _rank_blk = max(_rank_blk, 64)
            # W intermediate: (N_fused, rank_blk, max(Np,Nq))
            scan_workspace = N_fused * _rank_blk * max(_Np, _Nq) * B
        else:
            scan_workspace = 0

        # Per-blk cost: carry + contribution inside lax.scan
        #   carry:        (Np, Nq, blk, Ns) — persists across iterations
        #   contribution: (Np, Nq, blk, Ns) — recomputed each step
        gpu_per_blk = O * V * V * B * 2   # carry + contribution per unit blk

        # Available GPU after subtracting scan workspace overhead
        gpu_available = max(gpu_free - scan_workspace, 0)

        # Blksize is the min of host-derived and GPU-derived limits.
        # 0.65 safety factor on gpu_available to cover C_rs, scan
        # double-buffering, and XLA scratch/fragmentation.
        host_blk = max(1, int(host_budget * 0.8 / host_per_blk))
        gpu_blk = max(1, int(gpu_available * 0.65 / gpu_per_blk))
        blksize = min(host_blk, gpu_blk, nvir)

        logger.debug(
            "estimate_blksize(phase=%s): host_budget=%.2f GB, gpu_free=%.2f GB, "
            "scan_workspace=%.2f GB, gpu_available=%.2f GB, "
            "host_blk=%d, gpu_blk=%d → blksize=%d",
            phase, host_budget / 1e9, gpu_free / 1e9,
            scan_workspace / 1e9, gpu_available / 1e9,
            host_blk, gpu_blk, blksize)
        return blksize, budget

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
