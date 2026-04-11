"""
Centralized GPU memory budgeting utilities.

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
>>> from pytc.utils.gpu_memory import estimate_blksize
>>> blk, budget = estimate_blksize(nocc, nvir, 'ovvv',
...                                 gpu_max_memory_mb=24000)
"""

import logging
import numpy as np

from pytc.utils.tile_memory import isdf_tile_peak_bytes, find_max_blksize  # noqa: F401 (re-exported)

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
    authoritative limit (no runtime query).  Otherwise the minimum
    ``bytes_limit`` across all local devices is used, falling back to a
    conservative 16 GiB default.

    Returns
    -------
    int
        Total GPU budget in bytes.
    """
    if gpu_max_memory_mb is not None and gpu_max_memory_mb > 0:
        return int(gpu_max_memory_mb * 1024 ** 2)

    try:
        import jax
        limits = [int(d.memory_stats()['bytes_limit'])
                  for d in jax.local_devices()]
        return min(limits) if limits else 16 * 1024 ** 3
    except Exception:
        return 16 * 1024 ** 3  # 16 GiB fallback


def _get_gpu_physical_bytes():
    """Return the minimum GPU pool limit across all local devices.

    Uses ``bytes_limit`` (JAX's pre-allocation pool) as a conservative proxy
    for physical capacity.  Taking the minimum ensures that block-size
    estimates are safe for the most memory-constrained device.
    """
    try:
        import jax
        limits = [int(d.memory_stats()['bytes_limit'])
                  for d in jax.local_devices()]
        return min(limits) if limits else 80 * 1024 ** 3
    except Exception:
        return 80 * 1024 ** 3  # 80 GiB fallback (A100)


def _get_gpu_free_bytes():
    """Return the *currently free* GPU memory of the most-constrained local device.

    Takes the minimum across all local devices so that tile sizing is
    conservative for every GPU in the job, not just device 0.

    Falls back to ``_get_gpu_physical_bytes() * 0.75`` if stats are unavailable.
    """
    try:
        import jax
        min_free = None
        for d in jax.local_devices():
            stats = d.memory_stats()
            pool_limit = int(stats['bytes_limit'])
            in_use = int(stats.get('bytes_in_use', 0))
            free = max(pool_limit - in_use, 0)
            logger.debug(
                "_get_gpu_free_bytes: device=%s pool_limit=%.2f GB, "
                "in_use=%.2f GB, free=%.2f GB",
                getattr(d, 'id', repr(d)),
                pool_limit / 1e9, in_use / 1e9, free / 1e9)
            if min_free is None or free < min_free:
                min_free = free
        return min_free if min_free is not None else int(_get_gpu_physical_bytes() * 0.75)
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
            # ----- TC scan workspace -----
            _peak_per_rank = max(V * _N_fused + V * V,
                                 _N_fused * V + V * V, 1) * B
            _rank_blk = max(64, int(gpu_free * 0.5 / _peak_per_rank))
            _rank_blk = 2 ** int(np.log2(max(_rank_blk, 1)))
            _rank_blk = min(_rank_blk, 2048, _N_fused)
            _rank_blk = max(_rank_blk, 64)
            tc_scan_ws = _N_fused * _rank_blk * V * B

            # ----- Delta_U constant overhead -----
            # get_2b calls ISDFTC.get_2b (TC) then get_delta_U (delta_U)
            # sequentially.  The binding constraint is the phase that
            # needs more GPU memory.
            #
            # For vvvv/vvvv_gpu, ranges=(blk, V, V, V):
            #   delta_U: Np=blk, Nq=V, Nr=V, Ns=V
            #   X_sliced = (V, V, N_fused) — CONSTANT, ~46 GB for large systems
            #   D = (N_fused, N_fused) — constant
            #   Peak: X_sliced(input+padded) + D + 3×carry (term_d+carry+contrib)
            x_constant = V * V * _N_fused * B * 2   # input + padded copy
            d_constant = _N_fused * _N_fused * B
            du_constant = x_constant + d_constant

            # TC phase: tc_scan_ws + 2× carry per blk
            tc_per_blk = V * V * V * B * 2
            # Delta_U phase: 3× carry per blk (term_d + carry + contribution)
            du_per_blk = V * V * V * B * 3
        else:
            tc_scan_ws = 0
            du_constant = 0
            tc_per_blk = V * V * V * B * 2
            du_per_blk = V * V * V * B * 2

        if gpu_contraction:
            # Inside contract_block_kernel (all on GPU simultaneously):
            #   xtc_block          (blk,V,V,V) — from get_2b, stays on GPU
            #   tensordot result   (blk,V,V,V) — L_ab_sub @ L_vv_full
            #   sum                (blk,V,V,V) — xtc_block + tensordot
            #   einsum result      (O,O,blk,V) — output
            # XLA may fuse some of these, but peak ≈ 3× (blk,V,V,V).
            # Conservative: max of contraction kernel and delta_U.
            kernel_per_blk = V * V * V * B * 3
            # Contraction kernel also needs the xtc_block on GPU, so
            # delta_U constant overhead doesn't apply during this phase.
            # But get_2b's delta_U is the binding constraint for production.
            gpu_per_blk = max(kernel_per_blk, du_per_blk)
        else:
            gpu_per_blk = du_per_blk

        # GPU blksize = min of TC-limited and delta_U-limited
        tc_avail = max(gpu_free - tc_scan_ws, 0)
        du_avail = max(gpu_free - du_constant, 0)
        gpu_blk_tc = max(1, int(tc_avail * 0.65 / tc_per_blk))
        gpu_blk_du = max(1, int(du_avail * 0.45 / du_per_blk))
        gpu_blk = min(gpu_blk_tc, gpu_blk_du)

        scan_workspace = tc_scan_ws  # for logging

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

        # Per-blk cost: std_blk + tc_blk on host
        host_per_blk = O * V * V * B * 2  # std_blk + tc_blk

        # GPU side: query *actually free* memory in JAX's pool.
        gpu_free = _get_gpu_free_bytes()

        N_fused = n_fused if n_fused is not None else (naux if naux is not None else 0)
        if N_fused > 0:
            # ----- TC scan workspace (constant overhead during TC phase) -----
            if phase == 'vovv_eri_build':
                _Np_tc = nvir   # worst-case Np for TC scan
                _Nq_tc = max(O, V)
            else:  # ovvv_eri_build
                _Np_tc = O
                _Nq_tc = V
            _peak = max(_Np_tc * N_fused + _Np_tc * _Nq_tc,
                        N_fused * _Nq_tc + _Np_tc * _Nq_tc, 1) * B
            _rblk = max(64, int(gpu_free * 0.5 / _peak))
            _rblk = 2 ** int(np.log2(max(_rblk, 1)))
            _rblk = min(_rblk, 2048, N_fused)
            _rblk = max(_rblk, 64)
            tc_scan_ws = N_fused * _rblk * max(_Np_tc, _Nq_tc) * B

            # TC phase GPU constraint: scan_workspace + 2× carry per blk
            tc_constant = tc_scan_ws
            tc_per_blk = O * V * V * B * 2

            # ----- Delta_U phase GPU overhead -----
            # _contract_delta_U_kernels_jit runs TWO sequential scans:
            #   1. D-scan: accumulates term_d (Np,Nq,Nr,Ns)
            #   2. X-scan: accumulates term_x, with term_d still alive
            # Peak during X-scan = X_sliced + D + 3× carry_size
            #   (term_d + term_x carry + contribution)
            #
            # X_sliced = X[slice_r, slice_s] is transferred to GPU as a
            # JIT argument and stays resident.  Inside the JIT, X_padded
            # (a zero-padded copy, ~14% larger) is created.  Both coexist
            # briefly.  To be safe, budget ~2× X_sliced for this overlap.
            #
            # Dimension mapping:
            #   ovvv: ranges=(O,V,blk,V) → Np=O, Nq=V, Nr=blk, Ns=V
            #     X_sliced = (blk, V, N_fused) → **proportional to blk**
            #   vovv: ranges=(blk,O,V,V) → Np=blk, Nq=O, Nr=V, Ns=V
            #     X_sliced = (V, V, N_fused) → **constant** (huge!)
            d_overhead = N_fused * N_fused * B  # D matrix on GPU

            if phase == 'vovv_eri_build':
                # X_sliced constant: (V, V, N_fused).  Budget 2× for
                # input + padded copy inside JIT.
                x_constant = V * V * N_fused * B * 2
                du_constant = x_constant + d_overhead
                # 3× carry: term_d + carry + contribution
                du_per_blk = O * V * V * B * 3
            else:  # ovvv_eri_build
                # X_sliced per-blk: (blk, V, N_fused).  Budget 2× for
                # input + padded copy.
                du_constant = d_overhead
                du_per_blk = (3 * O * V * V + 2 * V * N_fused) * B

            # GPU blksize = min of TC-limited and delta_U-limited
            tc_avail = max(gpu_free - tc_constant, 0)
            du_avail = max(gpu_free - du_constant, 0)

            gpu_blk_tc = max(1, int(tc_avail * 0.65 / tc_per_blk))
            gpu_blk_du = max(1, int(du_avail * 0.45 / du_per_blk))
            gpu_blk = min(gpu_blk_tc, gpu_blk_du)

            # For logging, record the binding workspace
            scan_workspace = tc_scan_ws  # TC scan workspace (for log)
        else:
            scan_workspace = 0
            gpu_blk = max(1, int(gpu_free * 0.65 / (O * V * V * B * 2)))

        host_blk = max(1, int(host_budget * 0.65 / host_per_blk))
        blksize = min(host_blk, gpu_blk, nvir)

        logger.debug(
            "estimate_blksize(phase=%s): host_budget=%.2f GB, gpu_free=%.2f GB, "
            "scan_workspace=%.2f GB, "
            "host_blk=%d, gpu_blk=%d → blksize=%d",
            phase, host_budget / 1e9, gpu_free / 1e9,
            scan_workspace / 1e9,
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


def _budget_for_tile_sizing(nocc, nvir, gpu_max_memory_mb,
                            include_eris, include_accumulators,
                            n_fused, safety_factor):
    """Return ``(usable, gpu_target, resident_gb_parts)`` for panel estimators.

    Shared boilerplate extracted from ``estimate_vvvv_panel_blksize`` and
    ``estimate_v3o_panel_blksize`` so the two never drift apart.

    Returns
    -------
    usable : int
        Bytes available after subtracting persistent CCSD residents.
    gpu_target : int
        ``(usable - resident_isdf_constants) * safety_factor`` — the budget
        a single tile must fit within.
    log_parts : dict
        Intermediate values for debug logging (resident breakdown).
    """
    O, V, B = nocc, nvir, 8
    Nf = n_fused if n_fused is not None else 0

    gpu_budget = get_gpu_budget_bytes(gpu_max_memory_mb)
    persistent = estimate_persistent_gpu_bytes(
        nocc, nvir,
        include_eris=include_eris,
        include_accumulators=include_accumulators,
    )
    gpu_free = _get_gpu_free_bytes()
    usable = max(min(max(gpu_budget - persistent, 0), gpu_free), 0)

    nmo = O + V
    d_bytes   = Nf * Nf * B
    tc_bytes  = 4 * Nf * Nf * B if Nf > 0 else 0   # K1, K3, D, X kernel matrices
    phi_bytes = 4 * nmo * Nf * B if Nf > 0 else 0   # phi panels (4 copies)
    resident_isdf = d_bytes + tc_bytes + phi_bytes

    gpu_target = max(int(max(usable - resident_isdf, 0) * safety_factor), 0)

    log_parts = dict(
        usable=usable,
        resident_isdf=resident_isdf,
        d_bytes=d_bytes,
        tc_bytes=tc_bytes,
        phi_bytes=phi_bytes,
        gpu_target=gpu_target,
        Nf=Nf,
    )
    return usable, gpu_target, log_parts


def estimate_vvvv_panel_blksize(nocc, nvir, *,
                                gpu_max_memory_mb=None,
                                include_eris=False,
                                include_accumulators=False,
                                naux=None,
                                n_fused=None,
                                safety_factor=0.5):
    """Estimate a safe square ``(p, r)`` tile size for panelised VVVV work.

    Tile shape is ``(blk, nvir, blk, nvir)`` — both the p and r virtual
    indices are sliced.  Peak device memory per tile:

    - ISDF kernel cost for tile ``(blk, V, blk, V)`` via
      :func:`~pytc.utils.tile_memory.isdf_tile_peak_bytes`
    - DF operands ``L_p`` and ``L_r``: ``2 × blk × V × naux``
    - CCSD contraction output ``t2new[:, :, blk, blk]``: ``O² × blk²``
    """
    O, V, B = nocc, nvir, 8
    Nf = n_fused if n_fused is not None else 0

    usable, gpu_target, lp = _budget_for_tile_sizing(
        nocc, nvir, gpu_max_memory_mb,
        include_eris, include_accumulators, n_fused, safety_factor,
    )

    def tile_bytes(blk):
        isdf   = isdf_tile_peak_bytes(blk, V, blk, V, Nf) if Nf > 0 else 2 * blk * V * blk * V * B
        df     = 2 * blk * V * naux * B if (naux is not None and naux > 0) else 0
        ccsd   = O * O * blk * blk * B   # t2new slice on GPU
        return isdf + df + ccsd

    best = find_max_blksize(tile_bytes, lo=1, hi=max(1, nvir),
                            gpu_target=gpu_target)
    best = max(1, min(best, nvir))

    logger.debug(
        "estimate_vvvv_panel_blksize: usable=%.2f GB, resident=%.2f GB "
        "(D=%.2f GB, TC=%.2f GB, phi=%.2f GB), gpu_target=%.2f GB, "
        "naux=%s, n_fused=%s -> blk=%d (tile=%.2f GB)",
        usable / 1e9, lp["resident_isdf"] / 1e9,
        lp["d_bytes"] / 1e9, lp["tc_bytes"] / 1e9, lp["phi_bytes"] / 1e9,
        lp["gpu_target"] / 1e9, naux, n_fused, best, tile_bytes(best) / 1e9,
    )
    return best, usable


def resolve_vvvv_panel_block_sizes(nocc, nvir, *,
                                   p_block_size=None,
                                   r_block_size=None,
                                   gpu_max_memory_mb=None,
                                   include_eris=False,
                                   include_accumulators=False,
                                   naux=None,
                                   n_fused=None):
    """Resolve one square VVVV panel size from overrides or a VRAM estimate."""
    auto_blk, _ = estimate_vvvv_panel_blksize(
        nocc, nvir,
        gpu_max_memory_mb=gpu_max_memory_mb,
        include_eris=include_eris,
        include_accumulators=include_accumulators,
        naux=naux,
        n_fused=n_fused,
    )
    if p_block_size is not None and r_block_size is not None and int(p_block_size) != int(r_block_size):
        raise ValueError(
            "Balanced VVVV tiling requires vvvv_p_block_size == vvvv_r_block_size. "
            "Set only one override or use the same value for both."
        )

    panel_blk = p_block_size or r_block_size or auto_blk
    panel_blk = max(1, min(int(panel_blk), nvir))
    logger.debug(
        "Resolved square VVVV panel block: panel_blk=%d (auto=%d)",
        panel_blk, auto_blk,
    )
    return panel_blk, panel_blk


def estimate_v3o_panel_blksize(nocc, nvir, *,
                               gpu_max_memory_mb=None,
                               host_max_memory_mb=None,
                               include_eris=False,
                               include_accumulators=False,
                               naux=None,
                               n_fused=None,
                               safety_factor=0.5):
    """Estimate a safe virtual tile size for balanced 3V1O (ovvv/vovv) builds.

    Both ovvv and vovv tiles are padded to shape ``(nvir, ps, ps, nvir)``
    where ``ps = max(nocc, blk)``.  Peak device memory per tile:

    - ISDF kernel cost for ``(V, ps, ps, V)`` via
      :func:`~pytc.utils.tile_memory.isdf_tile_peak_bytes`
    - DF L_vv slabs: ``2 × blk × V × naux``

    The minimum viable ``blk`` is ``nocc`` because ``panel_size`` is floored
    at ``nocc`` in ``_compute_large_blocks``.
    """
    O, V, B = nocc, nvir, 8
    Nf = n_fused if n_fused is not None else 0

    usable, gpu_target, lp = _budget_for_tile_sizing(
        nocc, nvir, gpu_max_memory_mb,
        include_eris, include_accumulators, n_fused, safety_factor,
    )

    host_target = None
    if host_max_memory_mb is not None and host_max_memory_mb > 0:
        host_budget = int(host_max_memory_mb * 1e6)
        host_target = int(host_budget * 0.25)

    def tile_bytes(blk):
        # panel_size is the actual JIT-compiled shape: max(nocc, blk).
        # Both ovvv (layout="pr") and vovv (layout="qr") tiles are padded to
        # (V, ps, ps, V) — see _compute_large_blocks.
        ps = max(nocc, blk)
        isdf = isdf_tile_peak_bytes(V, ps, ps, V, Nf) if Nf > 0 else 2 * V * ps * ps * V * B
        df   = 2 * blk * V * naux * B if (naux is not None and naux > 0) else 0
        return isdf + df

    def host_slab_bytes(blk):
        # CPU slab written to HDF5 per tile: (nvir, nocc, blk, nvir)
        return V * O * blk * V * B

    # Start search at nocc: panel_size = max(nocc, blk) is flat below nocc.
    best = find_max_blksize(
        tile_bytes,
        lo=nocc, hi=max(nocc, nvir),
        gpu_target=gpu_target,
        host_target=host_target,
        host_bytes_fn=host_slab_bytes if host_target is not None else None,
    )
    best = max(nocc, min(best, nvir))

    if gpu_target > 0 and tile_bytes(nocc) > gpu_target:
        logger.warning(
            "estimate_v3o_panel_blksize: minimum tile (blk=%d) needs %.2f GB "
            "but gpu_target=%.2f GB — returning minimum anyway. "
            "Consider raising gpu_max_memory or reducing system size.",
            nocc, tile_bytes(nocc) / 1e9, gpu_target / 1e9,
        )

    logger.debug(
        "estimate_v3o_panel_blksize: usable=%.2f GB, resident=%.2f GB "
        "(D=%.2f GB, TC=%.2f GB, phi=%.2f GB), gpu_target=%.2f GB, "
        "host_target=%s GB, naux=%s, n_fused=%s -> blk=%d (tile=%.2f GB)",
        usable / 1e9, lp["resident_isdf"] / 1e9,
        lp["d_bytes"] / 1e9, lp["tc_bytes"] / 1e9, lp["phi_bytes"] / 1e9,
        lp["gpu_target"] / 1e9,
        "None" if host_target is None else f"{host_target / 1e9:.2f}",
        naux, n_fused, best, tile_bytes(best) / 1e9,
    )
    return best, usable


def resolve_v3o_panel_block_size(nocc, nvir, *,
                                 block_size=None,
                                 gpu_max_memory_mb=None,
                                 host_max_memory_mb=None,
                                 include_eris=False,
                                 include_accumulators=False,
                                 naux=None,
                                 n_fused=None):
    """Resolve one square virtual panel size for balanced 3V1O tile builds."""
    auto_blk, _ = estimate_v3o_panel_blksize(
        nocc, nvir,
        gpu_max_memory_mb=gpu_max_memory_mb,
        host_max_memory_mb=host_max_memory_mb,
        include_eris=include_eris,
        include_accumulators=include_accumulators,
        naux=naux,
        n_fused=n_fused,
    )
    panel_blk = block_size or auto_blk
    panel_blk = max(1, min(int(panel_blk), nvir))
    logger.debug(
        "Resolved square V3O panel block: panel_blk=%d (auto=%d)",
        panel_blk, auto_blk,
    )
    return panel_blk


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
    max_rank_block = 2 ** int(np.log2(max(max_rank_block, 1)))
    max_rank_block = min(max_rank_block, max_block, N_fused)
    max_rank_block = max(max_rank_block, min_block)

    logger.debug(
        "adaptive_rank_block_size(Np=%d, Nq=%d, N_fused=%d): "
        "peak_per_rank=%.2f MB → rank_block_size=%d",
        Np, Nq, N_fused, peak_per_rank / 1e6, max_rank_block)

    return max_rank_block


# ---------------------------------------------------------------------------
# XLA persistent compilation cache
# ---------------------------------------------------------------------------

_XLA_CACHE_ENABLED = False


def enable_xla_compilation_cache(cache_dir=None):
    """Enable the JAX/XLA persistent compilation cache.

    Persists compiled HLO programs to disk so that subsequent runs (or
    CCSD iterations after the first) skip costly XLA compilation.  This
    is particularly valuable for the ISDF-XTC workflow where multiple
    distinct ``(Np, Nq, Nr, Ns)`` shapes trigger separate compilations
    during the ovvv / vovv / vvvv ERI phases.

    Parameters
    ----------
    cache_dir : str or None
        Directory for the cache.  Defaults to ``~/.cache/jax_xla``.
        The directory is created automatically by JAX if needed.

    Notes
    -----
    Safe to call multiple times — subsequent calls are no-ops.
    Must be called *before* the first ``jax.jit`` invocation to have
    full effect (JAX ignores late cache-dir changes for already-traced
    functions).
    """
    global _XLA_CACHE_ENABLED
    if _XLA_CACHE_ENABLED:
        return

    import os
    import jax

    # Only enable persistent cache when a GPU backend is available.
    # CPU-only AOT caches are not portable across machines with different
    # instruction-set features (e.g. AVX-512 vs. no AVX-512) and cause
    # "Loading XLA:CPU AOT result ... not supported on the host machine"
    # errors at load time.
    try:
        gpu_devices = jax.devices("gpu")
        has_gpu = len(gpu_devices) > 0
    except RuntimeError:
        has_gpu = False

    if not has_gpu:
        _XLA_CACHE_ENABLED = True  # mark as handled (no-op on re-call)
        logger.info("XLA persistent compilation cache SKIPPED (CPU-only; "
                     "cache is not portable across CPU micro-architectures)")
        return

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "jax_xla")

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Cache every compiled program, regardless of size or compile time.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    _XLA_CACHE_ENABLED = True
    logger.info("XLA persistent compilation cache enabled → %s", cache_dir)
