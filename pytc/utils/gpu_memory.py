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

Runtime memory knobs
--------------------
GPU tile sizes auto-fit at dispatch time: ``_assemble_delta_u_tile`` probes
free memory just before issuing each tile pair and shrinks ``panel_size`` if
the auto-sized tile would not fit.  A normal run should never need any of
the env vars below; omit them to get fully automatic behaviour.

Three environment variables are provided as expert escape hatches for
advanced tuning or diagnostics.

PYTC_GPU_MAX_MEMORY_MB
    Authoritative total-GPU budget in MiB.  When set, ``adaptive_rank_block_size``
    uses this as the assumed device capacity instead of querying XLA.
    Example: ``PYTC_GPU_MAX_MEMORY_MB=40000`` for an 80 GB A100 with
    ~40 GB reserved for XLA caches and resident kernels.

PYTC_PANEL_BLK
    Hard cap (integer ≥ 1) on the K-stream ``panel_size`` and
    ``rank_block_size`` in ``tc.py``.  Use when the ISDF K-stream
    pre-allocation would exceed available HBM.
    Example: ``PYTC_PANEL_BLK=64``.

PYTC_SOLVER_BLK
    Hard cap (integer ≥ 1) on the initial CCSD tile ``panel_blk`` returned by
    ``resolve_vvvv_panel_block_sizes`` (vvvv path) and
    ``resolve_v3o_panel_block_size`` (v3o / large-blocks path).  Useful to
    pre-set a known-good size and avoid JAX recompiles from repeated auto-shrink.
    Example: ``PYTC_SOLVER_BLK=130`` for nkeep=300 on an 80 GB A100.

PYTC_XLA_CACHE_DIR
    Absolute directory for the optional persistent JAX/XLA compilation cache.
    Persistent caching is disabled unless this is set (or a cache directory is
    passed directly to :func:`enable_xla_compilation_cache`).  This avoids an
    implicit cache write to a user's home directory on shared systems.

Usage
-----
>>> from pytc.utils.gpu_memory import estimate_blksize
>>> blk, budget = estimate_blksize(nocc, nvir, 'ovvv',
...                                 gpu_max_memory_mb=24000)
"""

import logging
import os
import numpy as np

from pytc.utils.tile_memory import isdf_tile_peak_bytes, find_max_blksize  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tile absolute size ceiling (asymmetric-tile phases only)
# ---------------------------------------------------------------------------
# Raising ``gpu_max_memory`` to help one phase (e.g. VVVV, tile ~10 GB
# symmetric shape) can push another phase's tile across a cuBLAS / XLA
# autotuning cliff.  Empirically observed on B200 (HBM=180 GB) at benzene
# cc-pCV5Z (n_fused=25961, nvir=1179):
#
#   OVVV (asymmetric ``(blk, V, V, O)`` output fusion):
#     blksize=89  → tile ≈ 21 GB → succeeds
#     blksize=193 → tile ≈ 45 GB → "No valid config found!"  (autotuner
#                                  cannot find a cuBLAS config whose
#                                  workspace fits alongside the tile).
#
#   VVVV / V3O (symmetric ``(blk, V, blk, V)`` / ``(V, ps, ps, V)``):
#     tile can exceed 50 GB without hitting any autotune cliff.
#
# So the cliff is specific to the ASYMMETRIC tile shapes handled by the
# generic ``estimate_blksize`` path (OVVV / VOVV / similar).  The ceiling
# below is applied ONLY in that generic path.  ``estimate_vvvv_panel_blksize``
# and ``estimate_v3o_panel_blksize`` deliberately do NOT cap themselves
# with this constant — their symmetric tile shapes do not trigger the
# cuBLAS autotuner cliff at the sizes currently seen in benchmarks.
SAFE_TILE_BYTES_CEILING = 25 * 10 ** 9  # ~25 GB per tile

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
    B = 8
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


# ---------------------------------------------------------------------------
# Generic memory-constrained tile-size solver (used by K-kernel construction)
# ---------------------------------------------------------------------------

# Reserve this fraction of the probed free budget for XLA workspace, alignment
# padding, and BFC allocator fragmentation overhead.  Raise if OOMs persist
# despite the analytical model saying otherwise; lower if the model is proven
# too conservative.  Documented here (not at the call site) so there is one
# place to audit the slack.
K_KERNEL_SAFETY_FRACTION = 0.30

# Largest fraction of the budget that any single replicated/partially-sharded
# *input* tensor is allowed to occupy on a device.  Without this, a huge
# replicated ``xi_phi_r2`` can eat the budget and the peak-memory solver will
# still say "it fits" — but one of the K1 carry transients then has nowhere
# to land.  Enforcing a ceiling on the biggest input forces host tiling when
# replication gets pathological, independent of the specific system size.
#
# 0.20 is a physically motivated choice: XLA's BFC allocator observably eats
# 20–30 % of a pool near full occupancy via fragmentation and alignment, so
# no single input should claim more than about one fifth of the budget —
# otherwise the remaining headroom is too small to absorb one K1-carry
# transient (~ 3 n² / m_k * 8 bytes) on top of everything else.
K_KERNEL_SINGLE_INPUT_MAX_FRACTION = 0.20


def solve_tile_sizes(
    *,
    budget_bytes: int,
    fixed_bytes: int,
    r1_elem_bytes_per_unit: float,
    r2_elem_bytes_per_unit: float,
    r1_upper: int,
    r2_upper: int,
    r2_single_input_bytes_per_unit: float = None,
    safety_fraction: float = K_KERNEL_SAFETY_FRACTION,
    r2_single_input_max_fraction: float = K_KERNEL_SINGLE_INPUT_MAX_FRACTION,
    r1_floor: int = 1024,
    r2_floor: int = 1024,
):
    """Solve for r1/r2 host tile sizes under an explicit peak-memory model.

    The caller declares:

    * ``budget_bytes`` — total per-device budget (usually the probed free pool).
    * ``fixed_bytes`` — sum of resident + transient per-device tensors whose
      size does *not* depend on the tile sizes (carries, scratch, psum bufs).
    * ``r1_elem_bytes_per_unit`` — how many bytes of per-device memory grow
      by 1 when ``B_r1`` grows by 1 (typically ``4 * n_rank / m_k * 8`` for
      K-kernels: xi_phi_r1 + 3 * xi_grad_r1 components, k-sharded).
    * ``r2_elem_bytes_per_unit`` — same for ``B_r2``.
    * ``r1_upper``, ``r2_upper`` — absolute caps (usually ``n_grid``).
    * ``r2_single_input_bytes_per_unit`` — if provided, cap B_r2 so the
      single biggest r2-side input (e.g. xi_phi_r2 shard) occupies at
      most ``r2_single_input_max_fraction * budget_bytes``.

    Returns
    -------
    dict with ``B_r1``, ``B_r2``, and a ``predicted_peak_bytes`` estimate
    that is ``fixed_bytes + B_r1 * r1_elem_bytes_per_unit + B_r2 *
    r2_elem_bytes_per_unit``.  The returned tile sizes satisfy
    ``predicted_peak_bytes <= budget_bytes * (1 - safety_fraction)``.

    Raises ``RuntimeError`` if ``fixed_bytes`` alone already exceeds the
    safety-adjusted budget.
    """
    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
    usable = budget_bytes * (1.0 - safety_fraction)
    if fixed_bytes >= usable:
        raise RuntimeError(
            f"solve_tile_sizes: fixed-cost {fixed_bytes / 1024 ** 3:.2f} GiB "
            f"already exceeds safety-adjusted budget "
            f"{usable / 1024 ** 3:.2f} GiB "
            f"(budget {budget_bytes / 1024 ** 3:.2f} GiB, "
            f"safety {safety_fraction:.0%}). "
            "Use more devices, increase m_k, or lower safety_fraction."
        )

    remaining = usable - fixed_bytes

    # r1 gets a fixed small slice (it is typically tiny compared to r2).  Start
    # with a generous r1 cap; r2 gets everything left.
    # We bound r1 by the remaining headroom / 5 to leave most for r2.
    r1_frac = 0.2
    r1_cap_bytes = remaining * r1_frac
    B_r1 = int(r1_cap_bytes / max(r1_elem_bytes_per_unit, 1.0))
    B_r1 = max(min(B_r1, r1_upper), r1_floor)
    # Clamp B_r1 to upper since that is n_grid; never need more.
    B_r1 = min(B_r1, r1_upper)

    r2_avail = remaining - B_r1 * r1_elem_bytes_per_unit
    B_r2 = int(r2_avail / max(r2_elem_bytes_per_unit, 1.0))

    if r2_single_input_bytes_per_unit is not None:
        single_input_cap_bytes = budget_bytes * r2_single_input_max_fraction
        B_r2_input_cap = int(single_input_cap_bytes
                             / max(r2_single_input_bytes_per_unit, 1.0))
        B_r2 = min(B_r2, B_r2_input_cap)

    B_r2 = max(min(B_r2, r2_upper), r2_floor)

    predicted_peak = (
        fixed_bytes
        + B_r1 * r1_elem_bytes_per_unit
        + B_r2 * r2_elem_bytes_per_unit
    )
    # The r1/r2 floors can pull B_r1/B_r2 back up after they were sized
    # against `remaining`; in tight-memory cases that silently violates the
    # budget contract.  Fail fast instead of returning oversize tiles that
    # OOM later in K-kernel construction.
    if predicted_peak > usable:
        raise RuntimeError(
            f"solve_tile_sizes: no feasible tile fits the safety-adjusted "
            f"budget — even at the floors B_r1={B_r1}, B_r2={B_r2} the "
            f"predicted peak {predicted_peak / 1024 ** 3:.2f} GiB exceeds "
            f"usable {usable / 1024 ** 3:.2f} GiB "
            f"(budget {budget_bytes / 1024 ** 3:.2f} GiB, fixed "
            f"{fixed_bytes / 1024 ** 3:.2f} GiB, safety {safety_fraction:.0%}). "
            "Use more devices, increase m_k, lower the floors, or lower safety_fraction."
        )
    return {
        "B_r1": int(B_r1),
        "B_r2": int(B_r2),
        "predicted_peak_bytes": int(predicted_peak),
        "usable_bytes": int(usable),
    }


def choose_orb_block_size(
    n_orb: int,
    n_fused: int,
    *,
    bytes_per_elem: int = 8,
    budget_fraction: float = 0.15,
    min_block: int = 1,
    max_block: int = 256,
) -> int:
    """Choose ``orb_block_size`` so one HDF5 chunk of ``X`` fits a device.

    ``get_delta_h`` (in :mod:`pytc.xtc`) streams ``X`` from HDF5 in chunks of
    shape ``(orb_block_size, n_orb, n_fused)`` and then does einsums that
    materialise the chunk on device. Per-chunk bytes are
    ``orb_block_size * n_orb * n_fused * bytes_per_elem`` and grow linearly
    with ``n_fused``. For modest bases this is a few GiB; for cc-pCV5Z-class
    n_fused (~25k) the default ``orb_block_size=128`` alone asks for ~30 GiB
    per chunk — larger than any single A100 partition.

    Rather than hard-coding a per-system block, solve for it: read the probed
    free memory, apportion ``budget_fraction`` of it to X_chunk, and floor.

    Parameters
    ----------
    n_orb, n_fused : int
        Shapes of the chunk's non-block axes.
    bytes_per_elem : int
        8 for float64 (default), 4 for float32.
    budget_fraction : float
        Fraction of the probed free memory that X_chunk is allowed to
        consume. 0.15 leaves room for P_phi (~n_fused^2 * 8), einsum
        scratch, other resident tensors, and XLA fragmentation.
    min_block, max_block : int
        Clamp the chosen block to this range.

    Returns
    -------
    int
        Chosen ``orb_block_size``, clamped to
        ``[min_block, min(n_orb, max_block)]`` for the normal path.
        Degenerate inputs (``n_orb <= 0`` or ``n_fused <= 0``) bypass the
        clamp and return ``max_block`` — the caller's chunk loop won't
        iterate anyway, so the value is a safe no-op default.
    """
    if n_orb <= 0 or n_fused <= 0:
        return max_block
    free = _get_gpu_free_bytes()
    budget = max(int(free * budget_fraction), 0)
    per_block_bytes = n_orb * n_fused * bytes_per_elem
    max_block_for_mem = max(1, budget // per_block_bytes)
    cap = min(n_orb, max_block)
    return int(max(min_block, min(cap, max_block_for_mem)))


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
        # Return whether the accumulators fit, as a 0/1 flag.
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

    # 4. Safety margin (20%) + per-tile ceiling -------------------------------
    usable = available * 0.8
    # Absolute per-tile cap — protects against pushing small-per-blk phases
    # past the cuBLAS/XLA autotuning cliff when gpu_max_memory is big.
    capped_usable = min(usable, SAFE_TILE_BYTES_CEILING)
    blksize = max(1, int(capped_usable / per_blk))
    blksize = min(nvir, blksize)

    bound_by = "ceiling" if capped_usable < usable else "budget"
    logger.debug(
        "estimate_blksize(phase=%s): budget=%.2f GB, persistent=%.2f GB, "
        "available=%.2f GB, per_blk=%.2f MB, ceiling=%.2f GB "
        "→ blksize=%d (bound by %s)",
        phase, budget / 1e9, persistent / 1e9,
        available / 1e9, per_blk / 1e6,
        SAFE_TILE_BYTES_CEILING / 1e9, blksize, bound_by)

    return blksize, budget


def _budget_for_tile_sizing(nocc, nvir, gpu_max_memory_mb,
                            include_eris, include_accumulators,
                            n_fused, safety_factor,
                            tc_resident=None):
    """Return ``(usable, gpu_target, resident_gb_parts)`` for panel estimators.

    Shared boilerplate extracted from ``estimate_vvvv_panel_blksize`` and
    ``estimate_v3o_panel_blksize`` so the two never drift apart.

    Parameters
    ----------
    tc_resident : bool or None
        Whether the TC kernels (K1 + K3, combined size ``4·Nf²·8``) sit
        resident on device throughout the phase.  ``True`` adds them to
        the resident budget subtraction (tile gets less room).  ``False``
        assumes they are streamed in panels per tile call (tile gets more
        room).  ``None`` auto-detects: K1+K3 resident if they fit in 15 %
        of probed free memory, streamed otherwise — matching the
        ``_choose_tc_kernel_strategy`` rule used by the per-device TC
        cache in ``pytc.tc``.

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
    tc_full_bytes = 4 * Nf * Nf * B if Nf > 0 else 0   # K1 (3·Nf²) + K3 (Nf²)

    if tc_resident is None:
        # Auto: streaming if K1+K3 don't fit resident at the same 15 %
        # threshold used by ``_choose_tc_kernel_strategy``.
        tc_resident = tc_full_bytes <= int(gpu_free * 0.15) if Nf > 0 else True

    tc_bytes = tc_full_bytes if tc_resident else 0
    phi_bytes = 4 * nmo * Nf * B if Nf > 0 else 0   # phi panels (4 copies)
    resident_isdf = d_bytes + tc_bytes + phi_bytes

    gpu_target = max(int(max(usable - resident_isdf, 0) * safety_factor), 0)

    log_parts = dict(
        usable=usable,
        resident_isdf=resident_isdf,
        d_bytes=d_bytes,
        tc_bytes=tc_bytes,
        tc_resident=tc_resident,
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
                                extra_tile_bytes_fn=None,
                                safety_factor=0.5):
    """Estimate a safe square ``(p, r)`` tile size for panelised VVVV work.

    Tile shape is ``(blk, nvir, blk, nvir)`` — both the p and r virtual
    indices are sliced.  Peak device memory per tile:

    - ISDF kernel cost for tile ``(blk, V, blk, V)`` via
      :func:`~pytc.utils.tile_memory.isdf_tile_peak_bytes`
    - DF operands ``L_p`` and ``L_r``: ``2 × blk × V × naux``
    - CCSD contraction output ``t2new[:, :, blk, blk]``: ``O² × blk²``

    Parameters
    ----------
    extra_tile_bytes_fn : callable or None
        Optional ``(blk) -> int`` that returns additional bytes that will be
        live concurrently with the tile computation (e.g. a tc_tile output
        and a sum result for ISDF-XTC, each ``blk² × nvir² × 8`` bytes).
        Pass ``None`` (default) when no extra concurrent buffers exist.
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
        extra  = extra_tile_bytes_fn(blk) if extra_tile_bytes_fn is not None else 0
        return isdf + df + ccsd + extra

    # VVVV tile is a symmetric (blk, V, blk, V) shape that empirically does
    # not hit the cuBLAS autotune cliff even when total tile bytes exceed
    # SAFE_TILE_BYTES_CEILING — so this path is NOT capped by the ceiling.
    # (The ceiling is only applied in the generic estimate_blksize path,
    # which handles the asymmetric OVVV tile that DID hit the cliff.)
    best = find_max_blksize(tile_bytes, lo=1, hi=max(1, nvir),
                            gpu_target=gpu_target)
    best = max(1, min(best, nvir))

    # find_max_blksize returns ``lo`` (=1 here) even when the minimum tile
    # already exceeds gpu_target — warn so OOMs in compute_vvvv aren't
    # surprises.  Mirrors the same guard in estimate_v3o_panel_blksize.
    if gpu_target > 0 and tile_bytes(best) > gpu_target:
        logger.warning(
            "estimate_vvvv_panel_blksize: minimum tile (blk=%d) needs %.2f GB "
            "but gpu_target=%.2f GB — returning minimum anyway. "
            "Consider raising gpu_max_memory or reducing system size.",
            best, tile_bytes(best) / 1e9, gpu_target / 1e9,
        )

    logger.debug(
        "estimate_vvvv_panel_blksize: usable=%.2f GB, resident=%.2f GB "
        "(D=%.2f GB, TC=%.2f GB [resident=%s], phi=%.2f GB), "
        "gpu_target=%.2f GB, naux=%s, n_fused=%s -> blk=%d (tile=%.2f GB)",
        usable / 1e9, lp["resident_isdf"] / 1e9,
        lp["d_bytes"] / 1e9, lp["tc_bytes"] / 1e9, lp["tc_resident"],
        lp["phi_bytes"] / 1e9,
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
                                   n_fused=None,
                                   extra_tile_bytes_fn=None):
    """Resolve one square VVVV panel size from overrides or a VRAM estimate."""
    auto_blk, _ = estimate_vvvv_panel_blksize(
        nocc, nvir,
        gpu_max_memory_mb=gpu_max_memory_mb,
        include_eris=include_eris,
        include_accumulators=include_accumulators,
        naux=naux,
        n_fused=n_fused,
        extra_tile_bytes_fn=extra_tile_bytes_fn,
    )
    if p_block_size is not None and r_block_size is not None and int(p_block_size) != int(r_block_size):
        raise ValueError(
            "Balanced VVVV tiling requires vvvv_p_block_size == vvvv_r_block_size. "
            "Set only one override or use the same value for both."
        )

    panel_blk = p_block_size or r_block_size or auto_blk
    panel_blk = max(1, min(int(panel_blk), nvir))

    # Honor PYTC_SOLVER_BLK as a hard cap on the CCSD vvvv tile panel_size.
    # Useful when the delta_U direct-tile pre-flight is borderline (e.g. 100 MB
    # short): reducing blk from nvir to a smaller value shrinks the tile O(blk²)
    # without changing correctness.
    sb = os.environ.get("PYTC_SOLVER_BLK")
    if sb is not None and sb.strip():
        try:
            solver_blk = int(sb)
            if solver_blk >= 1:
                panel_blk = min(panel_blk, solver_blk)
                logger.debug(
                    "PYTC_SOLVER_BLK=%d capping vvvv panel_blk=%d (auto=%d)",
                    solver_blk, panel_blk, auto_blk,
                )
        except ValueError:
            pass

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
    # V3O tiles are symmetric (V, ps, ps, V) — same rationale as VVVV for
    # not applying SAFE_TILE_BYTES_CEILING here.
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
        "(D=%.2f GB, TC=%.2f GB [resident=%s], phi=%.2f GB), "
        "gpu_target=%.2f GB, host_target=%s GB, naux=%s, "
        "n_fused=%s -> blk=%d (tile=%.2f GB)",
        usable / 1e9, lp["resident_isdf"] / 1e9,
        lp["d_bytes"] / 1e9, lp["tc_bytes"] / 1e9, lp["tc_resident"],
        lp["phi_bytes"] / 1e9,
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

    # Honor PYTC_SOLVER_BLK as a hard cap (same knob as resolve_vvvv_panel_block_sizes).
    sb = os.environ.get("PYTC_SOLVER_BLK")
    if sb is not None and sb.strip():
        try:
            solver_blk = int(sb)
            if solver_blk >= 1:
                panel_blk = min(panel_blk, solver_blk)
                logger.debug(
                    "PYTC_SOLVER_BLK=%d capping v3o panel_blk=%d (auto=%d)",
                    solver_blk, panel_blk, auto_blk,
                )
        except ValueError:
            pass

    logger.debug(
        "Resolved square V3O panel block: panel_blk=%d (auto=%d)",
        panel_blk, auto_blk,
    )
    return panel_blk


def estimate_tc_contract_resident_bytes(n_orb, n_fused, *,
                                        k_stream_panel=None,
                                        bytes_per_elem=8):
    """Bytes resident on one device during a TC contract_K1/K3 JIT call,
    excluding the W scan intermediate (which is what :func:`adaptive_rank_block_size`
    is sizing).

    Line items:
      * D                 : n_fused²
      * phi_isdf          : n_orb * n_fused
      * grad_phi_isdf     : 3 * n_orb * n_fused
      * tile accumulator  : small (≤ n_orb² × a few KB worst case) — use an
                            n_orb² × 1024 upper bound
      * K1 / K3 effective : resident or streamed + padded/scannable overhead.
        Observed on A100 grace runs at n_fused ≈ 26k: XLA keeps roughly 1×
        input + 1× scannable copy of each, i.e. a 2× multiplier on the input
        bytes. We use 2.0 to be safe against small per-call allocator
        transients.

    ``k_stream_panel=None`` means K1/K3 stay fully resident across tiles;
    otherwise they are slabbed per tile at ``k_stream_panel`` rank columns.
    """
    B = int(bytes_per_elem)
    d_bytes = n_fused * n_fused * B
    phi_bytes = n_orb * n_fused * B
    grad_phi_bytes = 3 * n_orb * n_fused * B
    # Generous bound on tile output accumulator without needing panel size.
    tile_bytes = n_orb * n_orb * 1024 * B

    if k_stream_panel is None:
        # K1 (n_fused² * 3) + K3 (n_fused²) = 4 n_fused².
        k_input_bytes = 4 * n_fused * n_fused * B
    else:
        k_input_bytes = 4 * n_fused * int(k_stream_panel) * B

    # Padded/scannable overhead inside the JIT; observed ≈ 1× input on top.
    k_overhead_bytes = k_input_bytes

    return (d_bytes + phi_bytes + grad_phi_bytes + tile_bytes
            + k_input_bytes + k_overhead_bytes)


def adaptive_rank_block_size(Np, Nq, N_fused, *,
                              resident_bytes=0,
                              budget_bytes=None,
                              gpu_max_memory_mb=None,
                              budget_fraction=0.25,
                              min_block=4, max_block=2048):
    """Compute the largest safe ``rank_block_size`` for ISDF scan contractions.

    The peak intermediate per scan step in ``contract_K1_isdf_jit`` is::

        W: (Np, N_fused, rank_block_size) × 8 bytes
        T: (Np, Nq, rank_block_size) × 8 bytes

    and in ``_contract_delta_U_kernels_jit`` (D-term)::

        W: (N_fused, rank_block_size, Nq) × 8 bytes

    Returned rank_block_size satisfies::

        peak_W_bytes ≤ budget_fraction × (budget - resident_bytes)

    Parameters
    ----------
    Np, Nq : int
        Orbital slice sizes for the bra / ket indices. ``Np`` drives the W
        peak for K1 contractions; ``Nq`` drives it for the D-term pattern.
    N_fused : int
        ISDF rank (dimension being scanned over).
    resident_bytes : int, optional
        Sum of per-device bytes that will coexist with W during the scan
        (D, phi, grad_phi, K1/K3 or their panels with XLA overhead, tile
        accumulator). Default 0 reproduces the old "pool is entirely
        free" assumption — caller should supply a real estimate whenever
        possible. :func:`estimate_tc_contract_resident_bytes` helps.
    budget_bytes : int, optional
        Explicit budget override (bytes). If not given, falls back to
        ``gpu_max_memory_mb`` if supplied, otherwise to the runtime-probed
        ``_get_gpu_free_bytes()``.
    gpu_max_memory_mb : float, optional
        Legacy override in MiB; used only when ``budget_bytes`` is not set.
    budget_fraction : float
        Fraction of the *available* budget (``budget − resident``) allowed
        for the W intermediate. 0.25 leaves three quarters for XLA workspace,
        BFC fragmentation, and scan-carry transients — observed to be a
        safe margin on A100 at n_fused ≈ 26k.
    min_block, max_block : int
        Clamps on the returned value.

    Returns
    -------
    int
        Power-of-2 block size, clamped to ``[min_block, min(N_fused, max_block)]``.
    """
    if budget_bytes is None:
        if gpu_max_memory_mb is not None and gpu_max_memory_mb > 0:
            budget_bytes = int(gpu_max_memory_mb * 1024 ** 2)
        else:
            try:
                budget_bytes = int(_get_gpu_free_bytes())
            except Exception:
                budget_bytes = int(get_gpu_budget_bytes())

    available = max(int(budget_bytes) - int(resident_bytes), 0)
    B = 8  # float64

    # Worst-case peak per unit of rank_block_size: max of K1-style and
    # delta_U-style intermediates.
    peak_per_rank_K1 = (Np * N_fused + Np * Nq) * B
    peak_per_rank_DU = (N_fused * Nq + Np * Nq) * B
    peak_per_rank = max(peak_per_rank_K1, peak_per_rank_DU, 1)

    max_rank_block = max(min_block, int(available * budget_fraction / peak_per_rank))

    max_rank_block = 2 ** int(np.log2(max(max_rank_block, 1)))
    max_rank_block = min(max_rank_block, max_block, N_fused)
    max_rank_block = max(max_rank_block, min_block)

    logger.debug(
        "adaptive_rank_block_size(Np=%d, Nq=%d, N_fused=%d): "
        "budget=%.2f GiB, resident=%.2f GiB, peak_per_rank=%.2f MB, "
        "fraction=%.2f → rank_block_size=%d",
        Np, Nq, N_fused,
        budget_bytes / (1024 ** 3),
        resident_bytes / (1024 ** 3),
        peak_per_rank / 1e6,
        budget_fraction,
        max_rank_block)

    return max_rank_block


# ---------------------------------------------------------------------------
# XLA persistent compilation cache
# ---------------------------------------------------------------------------

_XLA_CACHE_ENABLED = False
_XLA_CACHE_DIR_ENV = "PYTC_XLA_CACHE_DIR"


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
        Absolute directory for the cache.  When omitted, reads
        ``PYTC_XLA_CACHE_DIR``.  Persistent caching is disabled when neither
        is supplied.  The directory is created before JAX is configured.

    Notes
    -----
    Safe to call multiple times after the cache is enabled — subsequent calls
    are no-ops.
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
        cache_dir = os.environ.get(_XLA_CACHE_DIR_ENV)

    if not cache_dir:
        logger.info("XLA persistent compilation cache disabled; set %s to "
                    "an absolute approved directory to enable it",
                    _XLA_CACHE_DIR_ENV)
        return

    cache_dir = os.fspath(cache_dir)
    if not os.path.isabs(cache_dir):
        raise ValueError(
            f"{_XLA_CACHE_DIR_ENV} must be an absolute path; got {cache_dir!r}"
        )
    os.makedirs(cache_dir, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    _XLA_CACHE_ENABLED = True
    logger.info("XLA persistent compilation cache enabled → %s", cache_dir)
