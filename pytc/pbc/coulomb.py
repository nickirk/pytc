"""PeriodicFFTISDF consumer: wires the S1-S4 pipeline in
pytc.pbc.df.{kpts,isdf} into build(), plus get_k/get_j and the THC-ERI
(get_ao_eri/get_mo_eri) and pyscf with_df (ISDFDF) interfaces.
See design doc §2, §4, §7-§8.
"""

from __future__ import annotations

import gc
import logging
import os

import numpy as np

import jax
import jax.numpy as jnp

from pytc.pbc.df.isdf import (
    build_pi_kern_p_blocked,
    RawKernelProvider,
    build_cached_periodic_bpc_gemm_oracle,
    build_periodic_batched_pivot_oracle,
    build_coul_kpt_device,
    build_coul_kpt_host,
    build_periodic_pivot_oracle,
    build_pi_eta,
    build_pi_eta_staged,
    explicit_candidate_identity,
    full_grid_candidate_identity,
    pivoted_cholesky_hermitian,
    pivoted_cholesky_batched_hermitian,
    stream_ao_blocks,
)
from pytc.pbc.df.kpts import canonicalize_kpts, check_time_reversal_residual, kpt_to_spc, spc_to_kpt

logger = logging.getLogger(__name__)


# Public selection surface: selector x storage. The mode strings below are the
# flattened product and remain the accepted spelling; retired experiments map to
# their surviving replacement so callers get a directive error, not a KeyError.
SELECTOR_STORAGE = {
    "bpc_cached_gemm": ("bpc", "cached"),
    "bpc_streamed": ("bpc", "streamed"),
    "bpc_auto": ("bpc", "auto"),
    "streamed": ("exact", "streamed"),
    "fixed_pivots": ("fixed", None),
}
# Ceiling on the cached AO feature matrix. Cached storage allocates
# n_grid x (Nk*nao) complex128 in one block with no gate today: 0.45 GB at
# diamond-222, 4.3 GB at 333/Gamma, 24.3 GB at 444/Gamma. Large but not absurd,
# which is exactly why it needs a predicted-byte decision rather than an
# assumption in either direction.
CACHED_AO_MAX_BYTES = 16 * 2**30


def predicted_cached_ao_bytes(n_grid, n_kpts, n_ao):
    """Bytes the cached AO feature matrix would allocate, before allocating it."""
    return int(n_grid) * int(n_kpts) * int(n_ao) * 16
# Frozen BPC policy accepted by the campaign; the only validated tuning.
FROZEN_BPC_POLICY = {
    "bpc_batch_size": 64,
    "bpc_min_separation": 2.0,
    "bpc_candidate_oversampling": 4,
    "bpc_n_topup": 16,
}
# Default selector: BPC with the frozen policy, storage chosen by predicted bytes.
# Promoted 2026-07-29 on owner instruction, after the two blockers were removed.
# It was previously held at the exact streamed oracle because FROZEN_BPC_POLICY is the
# only validated tuning and was not size-universal (n_topup=16 refused at rank<16), and
# because cached storage allocated the full AO feature matrix with no capacity gate.
# n_topup is now clamped to rank and recorded, and bpc_auto picks cached vs streamed on
# predicted bytes, so both objections are answered rather than waived.
#
# The TUNING moves with the mode deliberately. Defaulting BPC while leaving the generic
# batch=16/oversampling=1/topup=0 values would ship a configuration nothing has validated
# -- which is the failure the previous comment existed to prevent.
DEFAULT_SELECTION_MODE = "bpc_auto"

RETIRED_SELECTION_MODES = {
    "jax_cached_matrix_free": "bpc_cached_gemm",
    "jax_translation_matrix_free": "bpc_cached_gemm",
    "cached_full": "streamed",
    "bpc_cached_full": "bpc_cached_gemm",
    "panel_dense": "bpc_cached_gemm",
    "panel_oracle": "bpc_cached_gemm",
    "reciprocal_same_grid": "bpc_cached_gemm",
}


def validate_option_compatibility(*, p_block_rows=None, kern_blocking=None,
                                 stage_eta_root=None, solve_backend="device",
                                 jitter_rcond=None, retention_mode="single",
                                 rtol=None, n_retained_pin=None,
                                 target_truncation_residual=None):
    """Refuse statically-knowable option combinations.

    Placement is the point. These checks previously lived only inside build(), which
    reaches them AFTER pivot selection -- so a combination knowable in microseconds
    killed a 444 production run ~3 h in. ISDFDF.__init__ calls this too, so the
    failure is immediate.

    ONE function called from both sites rather than duplicated: a second copy of a
    refusal rule drifts, and then the two disagree about what is legal.
    """
    if solve_backend not in ("device", "host"):
        raise ValueError(
            f"solve_backend must be 'device' or 'host', got {solve_backend!r}. "
            f"'host' is a reference path for accuracy work, not a performance path."
        )
    if p_block_rows is not None and kern_blocking is not None:
        raise ValueError(
            "p_block_rows and kern_blocking are alternative memory levers, "
            "not composable: the panel-blocked path forms kern directly, so "
            "kern_blocking's rq staging would never run. Choose one."
        )
    if p_block_rows is not None and stage_eta_root is not None:
        raise ValueError(
            "p_block_rows and stage_eta_root are alternative memory levers, "
            "not composable: the panel-blocked path never materialises eta, "
            "so there is nothing to stage. Choose one."
        )
    if solve_backend == "host":
        for name, value in (("kern_blocking", kern_blocking),
                            ("p_block_rows", p_block_rows)):
            if value is not None:
                raise ValueError(
                    f"solve_backend='host' does not implement {name}; it is a "
                    f"reference path, not a performance path."
                )
    elif jitter_rcond is not None:
        raise ValueError(
            "jitter_rcond requires solve_backend='host'; the device path does "
            "not implement retention_mode='cholesky_jitter'."
        )
    # These were statically knowable and yet failed only after pivot selection,
    # because the first version of this validator took neither retention_mode nor
    # rtol. Fixing placement for SOME options and not others is not a fix.
    if retention_mode == "cholesky_jitter":
        if solve_backend != "host":
            raise ValueError(
                f"retention_mode='cholesky_jitter' requires solve_backend='host'; "
                f"the device path refuses the mode. Got {solve_backend!r}."
            )
        if rtol is not None:
            raise ValueError(
                "rtol does not apply to retention_mode='cholesky_jitter': it is a "
                "spectral truncation threshold and this mode does not truncate. "
                "Pass jitter_rcond instead."
            )
        for name, value in (("n_retained_pin", n_retained_pin),
                            ("target_truncation_residual", target_truncation_residual)):
            if value is not None:
                raise ValueError(
                    f"{name} does not apply to retention_mode='cholesky_jitter': "
                    f"the mode regularizes rather than truncating, so it has no "
                    f"retained set."
                )
    elif jitter_rcond is not None:
        raise ValueError(
            f"jitter_rcond applies only to retention_mode='cholesky_jitter', got "
            f"{retention_mode!r}."
        )


def build(cell, kpts, *, rank, block_size, rtol=None, retention_mode="single",
          provider_cls=RawKernelProvider, selection_mode=None,
          fixed_pivots=None, on_selection=None,
          bpc_batch_size=FROZEN_BPC_POLICY["bpc_batch_size"],
          bpc_min_separation=FROZEN_BPC_POLICY["bpc_min_separation"],
          bpc_candidate_oversampling=FROZEN_BPC_POLICY["bpc_candidate_oversampling"],
          bpc_n_topup=FROZEN_BPC_POLICY["bpc_n_topup"], reuse_ao_cache_for_eta=True,
          stage_eta_root=None, stage_eta_block=4096, kern_blocking=None,
          n_retained_pin=None, convolve_device=False, p_block_rows=None,
          solve_backend="device", jitter_rcond=None, cached_ao_max_bytes=None):
    """Build the periodic FFT-ISDF interpolation-point factor and solved
    kernel for one (cell, k-mesh) system, wiring S1-S4 end to end.

    Args:
        kpts: (Nk,3) absolute k-points (canonicalized internally).
        rank: requested interpolation-point rank.
        block_size: grid points per streamed AO block.
        on_selection: optional callable invoked immediately after pivot
            selection with ``(pivots, selection_provenance)``. Exists so a
            caller can persist pivots at the moment they are produced;
            without it they are unreachable until the entire build returns,
            so any interruption after selection discards hours of work. Feed
            what it receives back through ``fixed_pivots`` to resume. A raising
            callback is logged and recorded in the provenance, not propagated.
        rtol: forwarded to the S4 Hermitian sandwich solve; None means the
            1e-4 default, and must be None when n_retained_pin is given.
        retention_mode: "single" or "pairwise" -- forwarded to the S4
            Hermitian sandwich solve. See hermitian_sandwich_solve's
            docstring (pytc/df/solvers.py) for the two modes.
        provider_cls: KernelProvider for S4 (default RawKernelProvider).
        n_retained_pin: optional int K or length-Nk sequence, forwarded
            per-q to the S4 solve (fixed effective rank; mutually
            exclusive with rtol).

    Returns:
        dict: mesh_obj (KptsMesh), inpv_kpt (Nk,Nip,Nao) complex128,
        coul_kpt / kern_kpt (Nk,Nip,Nip) complex128, n_selected (may be
        < rank if the pivot metric exhausts), n_pipeline_calls,
        solve_infos (length-Nk list).
    """
    validate_option_compatibility(
        p_block_rows=p_block_rows, kern_blocking=kern_blocking,
        stage_eta_root=stage_eta_root, solve_backend=solve_backend,
        jitter_rcond=jitter_rcond, retention_mode=retention_mode, rtol=rtol,
        n_retained_pin=n_retained_pin,
        target_truncation_residual=None)
    if selection_mode == "fixed_pivots" and fixed_pivots is None:
        raise ValueError(
            "selection_mode='fixed_pivots' requires an explicit fixed_pivots array."
        )
    if selection_mode is None:
        selection_mode = "fixed_pivots" if fixed_pivots is not None else DEFAULT_SELECTION_MODE
    if fixed_pivots is not None:
        if selection_mode not in {"streamed", "fixed_pivots"}:
            raise ValueError(
                "fixed_pivots is only compatible with exact 'streamed' "
                "selection provenance."
            )
        selection_mode = "fixed_pivots"
    if selection_mode in RETIRED_SELECTION_MODES:
        raise ValueError(
            f"selection_mode={selection_mode!r} was retired; use "
            f"{RETIRED_SELECTION_MODES[selection_mode]!r}."
        )
    if selection_mode not in SELECTOR_STORAGE:
        raise ValueError(
            "selection_mode must be one of "
            f"{sorted(SELECTOR_STORAGE)} -- selector x storage."
        )
    selector, storage = SELECTOR_STORAGE[selection_mode]
    mesh_obj = canonicalize_kpts(cell, kpts)
    grid_coords = cell.get_uniform_grids(cell.mesh)

    # Predicted-byte gate, evaluated BEFORE the cached allocation exists.
    cache_ceiling = CACHED_AO_MAX_BYTES if cached_ao_max_bytes is None else int(cached_ao_max_bytes)
    predicted_cache_bytes = predicted_cached_ao_bytes(
        grid_coords.shape[0], mesh_obj.n_kpts, cell.nao)
    cache_gate = {
        "predicted_cached_ao_bytes": predicted_cache_bytes,
        "cached_ao_max_bytes": cache_ceiling,
    }
    if storage == "auto":
        # 'auto' CHOOSES; it never fails, because streamed is always available.
        storage = "cached" if predicted_cache_bytes <= cache_ceiling else "streamed"
        selection_mode = "bpc_cached_gemm" if storage == "cached" else "bpc_streamed"
        cache_gate["auto_resolved_to"] = storage
    elif storage == "cached" and predicted_cache_bytes > cache_ceiling:
        # Explicitly requested: refuse rather than OOM mid-selection.
        raise ValueError(
            f"selection_mode={selection_mode!r} would allocate a cached AO feature "
            f"matrix of {predicted_cache_bytes / 2**30:.2f} GiB, above the "
            f"{cache_ceiling / 2**30:.2f} GiB ceiling. Use 'bpc_auto' to pick "
            f"automatically, 'bpc_streamed' to force streaming, or raise "
            f"cached_ao_max_bytes deliberately."
        )

    ao_stats = {"pbc_eval_calls": 0, "grid_points": 0}
    selection_provenance = {
        "mode": selection_mode,
        "selector": selector,
        "storage": storage,
        "candidate_rule": "all_grid_points_v1",
        "candidate_count": int(grid_coords.shape[0]),
        "candidate_identity": full_grid_candidate_identity(grid_coords.shape[0]),
        "cache_bytes": 0,
        "panel_bytes": 0,
        "ao_dtype": np.dtype(np.complex128).name,
        **cache_gate,
    }
    cached_ao = None
    translation_cache = None
    translation_representation = None
    selector_ao_calls = 0
    selector_ao_grid_points = 0
    if selection_mode == "fixed_pivots":
        pivots = np.asarray(fixed_pivots)
        if pivots.ndim != 1 or not np.issubdtype(pivots.dtype, np.integer):
            raise ValueError("fixed_pivots must be a one-dimensional integer array.")
        pivots = pivots.astype(np.int64, copy=False)
        if pivots.size != rank:
            raise ValueError(
                f"fixed_pivots must contain exactly rank={rank} entries, got {pivots.size}."
            )
        if np.any(pivots < 0) or np.any(pivots >= grid_coords.shape[0]):
            raise ValueError("fixed_pivots contains an out-of-range grid index.")
        if np.unique(pivots).size != pivots.size:
            raise ValueError("fixed_pivots must be unique.")
        n_selected = int(pivots.size)
        selection_provenance.update({
            "mode": "fixed_pivots_experimental",
            "candidate_rule": "externally_fixed_pivots_v1",
            "candidate_count": int(pivots.size),
            "candidate_identity": explicit_candidate_identity(pivots),
        })
    elif selection_mode == "bpc_streamed":
        diag, col_batch_eval = build_periodic_batched_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size, stats=ao_stats,
        )
    elif selection_mode == "bpc_cached_gemm":
        diag, col_batch_eval, cached_ao = build_cached_periodic_bpc_gemm_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size, stats=ao_stats,
        )
    else:
        diag, col_eval = build_periodic_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size, stats=ao_stats,
        )
    if selection_mode == "fixed_pivots":
        pass
    elif selection_mode == "streamed":
        pivots, _, n_selected = pivoted_cholesky_hermitian(diag, col_eval, rank=rank)
    else:
        # n_topup > rank is refused downstream. That is the ONLY size-dependent
        # constraint in this policy -- verified by isolation: batch_size=999 and
        # candidate_oversampling=99 both pass at rank=12, because batch_size
        # self-limits via retain_count = min(batch_size, rank - len(pivots)).
        # The clamp is recorded rather than silent, so a run reports what it used.
        # Validate BEFORE converting. int() silently accepts True as 1 and 3.7 as 3,
        # and the clamp then records the coerced value as though it had been asked for.
        if isinstance(bpc_n_topup, bool) or not isinstance(bpc_n_topup, (int, np.integer)):
            raise ValueError(
                f"bpc_n_topup must be a non-negative integer, got "
                f"{bpc_n_topup!r} of type {type(bpc_n_topup).__name__}."
            )
        if int(bpc_n_topup) < 0:
            raise ValueError(f"bpc_n_topup must be non-negative, got {bpc_n_topup}.")
        n_topup_eff = min(int(bpc_n_topup), int(rank))
        pivots, _, n_selected, rounds = pivoted_cholesky_batched_hermitian(
            diag, col_batch_eval, rank=rank, mesh=cell.mesh,
            batch_size=bpc_batch_size, min_separation=bpc_min_separation,
            candidate_oversampling=bpc_candidate_oversampling,
            n_topup=n_topup_eff,
        )
        selection_provenance.update({
            "bpc_batch_size": int(bpc_batch_size),
            "bpc_min_separation_grid_units": float(bpc_min_separation),
            "bpc_candidate_oversampling": int(bpc_candidate_oversampling),
            "bpc_n_topup_requested": int(bpc_n_topup),
            "bpc_n_topup": n_topup_eff,
            "bpc_n_topup_clamped_to_rank": n_topup_eff != int(bpc_n_topup),
            "bpc_rounds": rounds,
            "bpc_joint_within_batch_exact_pivoting": True,
        })
        if selection_mode == "bpc_cached_gemm":
            selection_provenance["cache_bytes"] = int(cached_ao.nbytes)

    selector_ao_calls = ao_stats["pbc_eval_calls"]
    selector_ao_grid_points = ao_stats["grid_points"]
    ao_stats = {"pbc_eval_calls": 0, "grid_points": 0}
    selection_provenance["ao_calls_selection"] = selector_ao_calls
    selection_provenance["ao_grid_points_selection"] = selector_ao_grid_points

    # Selection is complete here, and its result is otherwise unreachable until
    # the whole call returns -- by which point the build and solve have also run.
    # A stage costing hours whose output only escapes at the end of the enclosing
    # call is unrecoverable by construction: any cancellation, OOM or wall-timeout
    # downstream discards it, and the retry pays for it again. The hook gives the
    # pivots an exit at the moment they exist, so a caller can checkpoint them and
    # resume through fixed_pivots. A raising callback must not corrupt the build,
    # so it is contained and reported rather than propagated.
    if on_selection is not None:
        try:
            on_selection(np.array(pivots, copy=True), dict(selection_provenance))
        except Exception as exc:                                   # noqa: BLE001
            logger.warning(
                "on_selection callback raised %r; the build continues, but the "
                "pivots were NOT checkpointed and a retry will repeat selection.",
                exc,
            )
            selection_provenance["on_selection_error"] = repr(exc)
    inpv_kpt = np.asarray(
        cell.pbc_eval_gto("GTOval", grid_coords[pivots], kpts=list(mesh_obj.canonical_kpts)),
        dtype=np.complex128,
    )
    ao_stats["pbc_eval_calls"] += 1
    ao_stats["grid_points"] += int(pivots.size)
    ao_tr_residual = check_time_reversal_residual(inpv_kpt, mesh_obj.neg)

    if not reuse_ao_cache_for_eta and cached_ao is not None:
        # Free the AO cache -- and the selector closure that also holds it -- so
        # the selection and build peaks do not overlap. Costs one extra AO pass.
        cached_ao = None
        col_batch_eval = None
        diag = None
        gc.collect()

    if cached_ao is not None:
        # The bpc_cached_gemm oracle caches AO features in the contiguous 2-D
        # (Ng, Nk*Nao) layout its threaded candidate GEMM needs
        # (build_cached_periodic_bpc_gemm_oracle), whereas build_pi_eta consumes
        # the 3-D (Nk, Ng, Nao) blocks the pivot-oracle caches produce. Recover
        # the 3-D block by inverting that oracle's exact pack
        # (ao_block.transpose(1, 0, 2).reshape(g, -1)) so the AO-reuse benefit is
        # kept for eta instead of re-streaming the AOs.
        if cached_ao.ndim == 2:
            n_kpts_eta = len(mesh_obj.canonical_kpts)
            n_ao_eta = cell.nao_nr()
            ao_blocks_for_eta = cached_ao.reshape(
                cached_ao.shape[0], n_kpts_eta, n_ao_eta
            ).transpose(1, 0, 2)
        else:
            ao_blocks_for_eta = cached_ao
    else:
        ao_blocks_for_eta = (
            blk for _, _, blk in stream_ao_blocks(
                cell, mesh_obj.canonical_kpts, grid_coords, block_size, stats=ao_stats,
            )
        )

    # One sweep counts stats; later sweeps re-evaluate the same AOs.
    _ao_sweeps = []

    def _ao_block_factory():
        """A FRESH iterable per call, which the panel-blocked build needs: each
        panel takes its own sweep. The single-use generator above cannot serve
        it -- a second panel would silently see an exhausted iterator.

        Stats are collected on the first sweep only; later sweeps re-evaluate the
        same AOs and would double-count."""
        if cached_ao is not None:
            # ONE block containing every grid point. ao_blocks_for_eta is a single
            # (Nk, Ngrid, Nao) array here, so iter() over it would yield Nk slices
            # of shape (Ngrid, Nao) -- 2-D, which pair_convolve rejects. Latent
            # until BPC+cached became the default and the panel path could reach it.
            return iter([ao_blocks_for_eta])
        first = not _ao_sweeps
        _ao_sweeps.append(1)
        return (
            blk for _, _, blk in stream_ao_blocks(
                cell, mesh_obj.canonical_kpts, grid_coords, block_size,
                stats=ao_stats if first else None,
            )
        )
    # With stage_eta_root set, eta is staged to a memmap: per-q reads off a
    # C-order file are contiguous, so only one q is resident. None keeps in-RAM.
    eta = None
    kern_p_blocked = None
    staged_path = None
    eta_staging_stats = None
    try:
        if p_block_rows is not None:
            # Panel-blocked: kern is built directly and eta never exists. The
            # panel knob trades storage against regeneration; Pi and kern are
            # untouched by it (see p_blocked_peak_bytes).
            neg_arr = np.asarray(mesh_obj.neg)
            Pi, kern_p_blocked = build_pi_kern_p_blocked(
                inpv_kpt, _ao_block_factory, mesh_obj.phase, mesh_obj.neg,
                provider_cls(cell=cell, canonical_kpts=mesh_obj.canonical_kpts,
                             grid_mesh=cell.mesh),
                grid_coords, panel_rows=int(p_block_rows),
                convolve_device=convolve_device,
                self_paired=lambda q: int(neg_arr[q]) == q,
            )
        elif stage_eta_root is not None:
            staged_path = os.path.join(
                stage_eta_root, f"isdf_eta_stage_{os.getpid()}.dat")
            Pi, eta, eta_staging_stats = build_pi_eta_staged(
                inpv_kpt, ao_blocks_for_eta, mesh_obj.phase, mesh_obj.neg,
                staging_path=staged_path, n_grid=int(grid_coords.shape[0]),
                staging_block=stage_eta_block,
                # the blocked solve stages a per-q rq alongside eta
                additional_reserve_bytes=(
                    int(inpv_kpt.shape[1]) * int(grid_coords.shape[0]) * 16
                    if kern_blocking is not None else 0),
                convolve_device=convolve_device,
            )
        else:
            Pi, eta = build_pi_eta(
                inpv_kpt, ao_blocks_for_eta, mesh_obj.phase, mesh_obj.neg,
                convolve_device=convolve_device)

        provider = provider_cls(
            cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
        )
        if solve_backend == "host":
            # Reference path. Exists so an accuracy question can be settled without
            # first porting a solver to the device path; refuses the device-only
            # levers rather than silently ignoring them, since a lever that is
            # accepted and dropped reads as "measured, no effect".
            coul_kpt, kern_kpt, solve_infos, n_pipeline_calls = build_coul_kpt_host(
                cell, Pi, eta, grid_coords, mesh_obj, rtol=rtol,
                retention_mode=retention_mode, jitter_rcond=jitter_rcond,
                n_retained_pin=n_retained_pin,
            )
        else:
            coul_kpt, kern_kpt, solve_infos, n_pipeline_calls = build_coul_kpt_device(
                provider, Pi, eta, grid_coords, mesh_obj, rtol=rtol,
                kern=kern_p_blocked,
                retention_mode=retention_mode, kern_blocking=kern_blocking,
                n_retained_pin=n_retained_pin,
            )
    finally:
        # Never strand the staging file, on success or failure.
        if staged_path is not None:
            eta = None
            gc.collect()
            try:
                os.unlink(staged_path)
            except FileNotFoundError:
                pass

    return {
        "mesh_obj": mesh_obj,
        "inpv_kpt": inpv_kpt,
        "coul_kpt": coul_kpt,
        "kern_kpt": kern_kpt,
        "n_selected": n_selected,
        "n_pipeline_calls": n_pipeline_calls,
        "solve_infos": solve_infos,
        "eta_staging": eta_staging_stats,
        "ao_tr_residual": ao_tr_residual,
        "selection_provenance": {
            **selection_provenance,
            "pivot_indices": pivots.tolist(),
            "n_selected": n_selected,
            "ao_calls_through_eta": selector_ao_calls + ao_stats["pbc_eval_calls"],
            "ao_grid_points_through_eta": selector_ao_grid_points + ao_stats["grid_points"],
        },
    }


def get_k(dm_kpts, inpv_kpt, coul_kpt, phase, *, exxdiv=None, cell=None, kpts=None, neg=None):
    """Periodic THC-ISDF exchange matrix (design doc §4, §7).

    Composition (per density-matrix set):
        rho_kpt[k] = inpv_kpt[k] @ dm_kpt[k] @ inpv_kpt[k].conj().T / Nk
        rho_spc    = kpt_to_spc(rho_kpt, phase).transpose(0,2,1)
        coul_spc   = kpt_to_spc(coul_kpt, phase) * sqrt(Nk)
        v_spc      = coul_spc * rho_spc          -- elementwise (Hadamard)
        v_kpt      = spc_to_kpt(v_spc, phase)
        vk_kpt     = conj( inpv_kpt.transpose(0,2,1) @ v_kpt @ inpv_kpt.conj() )

    exxdiv is applied HERE, after the bare vk_kpt -- never inside a
    KernelProvider. Only None and "ewald" are supported.

    Args:
        dm_kpts: (nset, Nk, Nao, Nao) or (Nk, Nao, Nao) complex128.
        inpv_kpt: (Nk, Nip, Nao) complex128.
        coul_kpt: (Nk, Nip, Nip) complex128.
        phase: (Nk, Nk) unitary matrix (KptsMesh.phase).
        cell, kpts: required when exxdiv="ewald".
        neg: (Nk,) int array (KptsMesh.neg), optional. When given,
            rho_kpt is symmetrized exactly by construction before
            kpt_to_spc (a real SCF density satisfies
            rho_kpt[neg[k]]=conj(rho_kpt[k]) only to floating-point
            precision, which trips kpt_to_spc's imag_tol gate). Without
            neg, the prior gate-armed, unsymmetrized behavior applies.

    Returns:
        vk_kpts: (nset, Nk, Nao, Nao) complex128.
    """
    inpv_kpt = np.asarray(inpv_kpt, dtype=np.complex128)
    coul_kpt = np.asarray(coul_kpt, dtype=np.complex128)
    n_k, n_ip, n_ao = inpv_kpt.shape
    if coul_kpt.shape != (n_k, n_ip, n_ip):
        raise ValueError(
            f"coul_kpt must have shape ({n_k},{n_ip},{n_ip}) matching inpv_kpt, got "
            f"{coul_kpt.shape}."
        )
    if exxdiv not in (None, "ewald"):
        raise ValueError(f"exxdiv must be None or 'ewald', got {exxdiv!r}.")
    if exxdiv == "ewald" and (cell is None or kpts is None):
        raise ValueError("exxdiv='ewald' requires both cell and kpts.")

    dm_kpts = np.asarray(dm_kpts, dtype=np.complex128)
    single_set = dm_kpts.ndim == 3
    if single_set:
        dm_kpts = dm_kpts[None, ...]
    n_set = dm_kpts.shape[0]
    if dm_kpts.shape != (n_set, n_k, n_ao, n_ao):
        raise ValueError(
            f"dm_kpts must have shape (nset,{n_k},{n_ao},{n_ao}) or ({n_k},{n_ao},{n_ao}), "
            f"got {dm_kpts.shape}."
        )

    if neg is not None:
        neg = np.asarray(neg)
        if neg.shape != (n_k,):
            raise ValueError(f"neg must have shape ({n_k},), got {neg.shape}.")
    coul_spc = kpt_to_spc(coul_kpt, phase) * np.sqrt(n_k)

    vk_kpts = np.empty((n_set, n_k, n_ao, n_ao), dtype=np.complex128)
    for i in range(n_set):
        dm_kpt = dm_kpts[i]
        rho_kpt = (inpv_kpt @ dm_kpt @ inpv_kpt.conj().transpose(0, 2, 1)) / n_k
        if neg is not None:
            # Symmetrize rho_kpt exactly by construction: a real SCF dm
            # satisfies rho_kpt[neg[k]]=conj(rho_kpt[k]) only to roundoff,
            # which trips kpt_to_spc's imag_tol gate; both pair members
            # encode the same physics, so deriving one loses nothing.
            rho_kpt = rho_kpt.copy()
            visited = np.zeros(n_k, dtype=bool)
            for k in range(n_k):
                if visited[k]:
                    continue
                nk = int(neg[k])
                if nk == k:
                    rho_kpt[k] = rho_kpt[k].real.astype(np.complex128)
                elif not visited[nk]:
                    rho_kpt[nk] = rho_kpt[k].conj()
                visited[k] = True
                visited[nk] = True
        rho_spc = kpt_to_spc(rho_kpt, phase).transpose(0, 2, 1)

        v_spc = coul_spc * rho_spc
        v_kpt = spc_to_kpt(v_spc, phase)

        vk_kpt = inpv_kpt.transpose(0, 2, 1) @ v_kpt @ inpv_kpt.conj()
        vk_kpts[i] = vk_kpt.conj()

    if exxdiv == "ewald":
        from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

        for i in range(n_set):
            _ewald_exxdiv_for_G0(cell, kpts, dm_kpts[i][None], vk_kpts[i][None])

    if single_set:
        vk_kpts = vk_kpts[0]
    return vk_kpts


def _relative_imag_norm(z, axes):
    norm_im = jnp.sqrt(jnp.sum(jnp.square(jnp.imag(z)), axis=axes))
    norm_total = jnp.sqrt(jnp.sum(jnp.square(jnp.abs(z)), axis=axes))
    return jnp.where(
        norm_total > 0,
        norm_im / jnp.where(norm_total > 0, norm_total, 1.0),
        norm_im,
    )


@jax.jit
def _get_k_bare_device_preflight(dm_sets, inpv_kpt, coul_kpt, phase, neg):
    """Build gated supercell quantities without host transfers."""
    n_k = dm_sets.shape[1]
    rho = jnp.einsum(
        "kia,skab,kjb->skij", inpv_kpt, dm_sets, jnp.conj(inpv_kpt),
        optimize=True,
    ) / n_k
    indices = jnp.arange(n_k)
    rho = jnp.where(
        (indices <= neg)[None, :, None, None], rho, jnp.conj(rho[:, neg]),
    )
    rho = jnp.where(
        (indices == neg)[None, :, None, None],
        jnp.real(rho).astype(jnp.complex128),
        rho,
    )
    rho_spc_complex = jnp.einsum("rk,skij->srij", phase, rho, optimize=True)
    coul_spc_complex = jnp.sqrt(n_k) * jnp.einsum(
        "rk,kij->rij", phase, coul_kpt, optimize=True,
    )
    rho_ratio = _relative_imag_norm(rho_spc_complex, axes=(1, 2, 3))
    coul_ratio = _relative_imag_norm(coul_spc_complex, axes=(0, 1, 2))
    return rho_spc_complex, coul_spc_complex, rho_ratio, coul_ratio


@jax.jit
def _get_k_bare_device_core(rho_spc_complex, coul_spc_complex, inpv_kpt, phase):
    """Contract validated bare exchange entirely on device."""
    rho_spc = jnp.swapaxes(jnp.real(rho_spc_complex), -1, -2)
    v_spc = jnp.real(coul_spc_complex)[None, :, :, :] * rho_spc
    v_kpt = jnp.einsum("rk,srij->skij", jnp.conj(phase), v_spc, optimize=True)
    return jnp.conj(jnp.einsum(
        "kia,skij,kjb->skab", inpv_kpt, v_kpt, jnp.conj(inpv_kpt), optimize=True,
    ))


def _validate_bare_k_device_inputs(dm_kpts, inpv_kpt, coul_kpt, phase, neg):
    """Validate the small host boundary for the private c128 device lane."""
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("get_k_bare_device requires jax_enable_x64=True for complex128.")

    def shape_dtype(value, name):
        try:
            return tuple(value.shape), np.dtype(value.dtype)
        except (AttributeError, TypeError):
            raise ValueError(f"{name} must provide shape and dtype metadata.") from None

    inpv_shape, inpv_dtype = shape_dtype(inpv_kpt, "inpv_kpt")
    if len(inpv_shape) != 3:
        raise ValueError(f"inpv_kpt must have shape (Nk,Nip,Nao), got {inpv_shape}.")
    n_k, n_ip, n_ao = inpv_shape
    if any(size <= 0 for size in inpv_shape) or inpv_dtype != np.dtype(np.complex128):
        raise TypeError("inpv_kpt must be nonempty complex128 with shape (Nk,Nip,Nao).")

    coul_shape, coul_dtype = shape_dtype(coul_kpt, "coul_kpt")
    if coul_shape != (n_k, n_ip, n_ip) or coul_dtype != np.dtype(np.complex128):
        raise ValueError(
            f"coul_kpt must be complex128 with shape ({n_k},{n_ip},{n_ip}), got "
            f"shape {coul_shape}, dtype {coul_dtype}."
        )
    phase_shape, phase_dtype = shape_dtype(phase, "phase")
    if phase_shape != (n_k, n_k) or phase_dtype != np.dtype(np.complex128):
        raise ValueError(
            f"phase must be complex128 with shape ({n_k},{n_k}), got "
            f"shape {phase_shape}, dtype {phase_dtype}."
        )

    dm_shape, dm_dtype = shape_dtype(dm_kpts, "dm_kpts")
    single_set = len(dm_shape) == 3
    if single_set:
        expected_dm_shape = (n_k, n_ao, n_ao)
    elif len(dm_shape) == 4:
        expected_dm_shape = (dm_shape[0], n_k, n_ao, n_ao)
    else:
        expected_dm_shape = None
    if dm_dtype != np.dtype(np.complex128) or dm_shape != expected_dm_shape:
        raise ValueError(
            f"dm_kpts must be complex128 with shape ({n_k},{n_ao},{n_ao}) or "
            f"(Nset,{n_k},{n_ao},{n_ao}), got shape {dm_shape}, dtype {dm_dtype}."
        )

    neg_np = np.asarray(neg)
    if neg_np.shape != (n_k,) or not np.issubdtype(neg_np.dtype, np.integer):
        raise ValueError(f"neg must be an integer array with shape ({n_k},), got {neg_np.shape}.")
    if np.any(neg_np < 0) or np.any(neg_np >= n_k):
        raise ValueError("neg must contain only in-range k-point indices.")
    if not np.array_equal(neg_np[neg_np], np.arange(n_k)):
        raise ValueError("neg must be an involution: neg[neg[k]] == k.")
    return single_set, neg_np


def _get_k_bare_device(dm_kpts, inpv_kpt, coul_kpt, phase, *, neg, imag_tol=1e-10):
    """Private c128 bare-K CPU/JAX parity boundary; no Ewald or PySCF adapter."""
    single_set, neg_np = _validate_bare_k_device_inputs(
        dm_kpts, inpv_kpt, coul_kpt, phase, neg,
    )
    if not isinstance(imag_tol, (int, float)) or imag_tol < 0:
        raise ValueError("imag_tol must be a nonnegative real scalar.")
    dm_sets = jnp.asarray(dm_kpts, dtype=jnp.complex128)
    inpv_device = jnp.asarray(inpv_kpt, dtype=jnp.complex128)
    coul_device = jnp.asarray(coul_kpt, dtype=jnp.complex128)
    phase_device = jnp.asarray(phase, dtype=jnp.complex128)
    if single_set:
        dm_sets = dm_sets[None, :, :, :]
    rho_spc, coul_spc, rho_ratio, coul_ratio = _get_k_bare_device_preflight(
        dm_sets, inpv_device, coul_device, phase_device, jnp.asarray(neg_np),
    )
    rho_ratio_host = np.asarray(rho_ratio)
    coul_ratio_host = float(np.asarray(coul_ratio))
    invalid_sets = np.flatnonzero(rho_ratio_host > imag_tol)
    if invalid_sets.size:
        raise ValueError(
            "get_k_bare_device: kpt_to_spc imaginary gate failed for density sets "
            f"{invalid_sets.tolist()} at imag_tol={imag_tol:.1e}."
        )
    if coul_ratio_host > imag_tol:
        raise ValueError(
            "get_k_bare_device: kpt_to_spc imaginary gate failed for coul_kpt at "
            f"imag_tol={imag_tol:.1e}."
        )
    vk_sets = _get_k_bare_device_core(rho_spc, coul_spc, inpv_device, phase_device)
    return vk_sets[0] if single_set else vk_sets


def _get_k_bare_device_adapter(
    dm_kpts, inpv_kpt, coul_kpt, phase, *, exxdiv=None, cell=None, kpts=None, neg,
):
    """Private host adapter: one result transfer, then unchanged host Ewald."""
    if exxdiv not in (None, "ewald"):
        raise ValueError(f"exxdiv must be None or 'ewald', got {exxdiv!r}.")
    if exxdiv == "ewald" and (cell is None or kpts is None):
        raise ValueError("exxdiv='ewald' requires both cell and kpts.")

    vk_device = _get_k_bare_device(
        dm_kpts, inpv_kpt, coul_kpt, phase, neg=neg,
    )
    vk_host = np.asarray(vk_device)  # The adapter's sole device-to-host transfer.
    if exxdiv != "ewald":
        return vk_host
    vk_host = vk_host.copy()  # PySCF applies the Ewald correction in place.

    dm_host = np.asarray(dm_kpts, dtype=np.complex128)
    single_set = dm_host.ndim == 3
    dm_sets = dm_host[None, ...] if single_set else dm_host
    vk_sets = vk_host[None, ...] if single_set else vk_host
    from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

    for i in range(dm_sets.shape[0]):
        _ewald_exxdiv_for_G0(cell, kpts, dm_sets[i][None], vk_sets[i][None])
    return vk_sets[0] if single_set else vk_sets


def get_ao_eri(inpv_kpt, coul_kpt, kconserv, k1, k2, k3):
    """AO-basis THC-ERI block (a^k1 b^k2 | c^k3 d^k4), one block at a time.
    Standard pyscf/chemist convention: a,c conjugated; k4 fixed by
    momentum conservation -- directly comparable to FFTDF.get_eri with no
    axis relabeling. See design doc §2.

        Q  = kconserv[k2, k1, 0]   -- NOTE the (k2, k1, 0) argument order,
             not (k1, k2, 0): this index enters via coul_kpt's Hermiticity.
        k4 = kconserv[k1, k2, k3]
        (a^k1 b^k2 | c^k3 d^k4)_{abcd} =
            sum_IJ conj(X[k1]_Ia) X[k2]_Ib * coul_kpt[Q]_IJ *
                   conj(X[k3]_Jc) X[k4]_Jd

    Returns:
        (eri_block, k4): eri_block (Nao, Nao, Nao, Nao) complex128.
    """
    inpv_kpt = np.asarray(inpv_kpt, dtype=np.complex128)
    coul_kpt = np.asarray(coul_kpt, dtype=np.complex128)
    kconserv = np.asarray(kconserv)
    n_k, n_ip, n_ao = inpv_kpt.shape
    if coul_kpt.shape != (n_k, n_ip, n_ip):
        raise ValueError(
            f"coul_kpt must have shape ({n_k},{n_ip},{n_ip}) matching inpv_kpt, got "
            f"{coul_kpt.shape}."
        )
    if kconserv.shape != (n_k, n_k, n_k):
        raise ValueError(f"kconserv must have shape ({n_k},{n_k},{n_k}), got {kconserv.shape}.")
    for name, k in (("k1", k1), ("k2", k2), ("k3", k3)):
        if not (0 <= int(k) < n_k):
            raise ValueError(f"{name}={k} out of range for n_k={n_k}.")
    k1, k2, k3 = int(k1), int(k2), int(k3)

    Q = int(kconserv[k2, k1, 0])
    k4 = int(kconserv[k1, k2, k3])

    X1, X2, X3, X4 = inpv_kpt[k1], inpv_kpt[k2], inpv_kpt[k3], inpv_kpt[k4]
    W = coul_kpt[Q]
    rho_ab = np.einsum("Ia,Ib->Iab", X1.conj(), X2, optimize=True)
    rho_cd = np.einsum("Ic,Id->Icd", X3.conj(), X4, optimize=True)
    eri_block = np.einsum("Iab,IJ,Jcd->abcd", rho_ab, W, rho_cd, optimize=True)
    return eri_block, k4


def get_mo_eri(inpv_kpt, coul_kpt, kconserv, mo_coeff_kpts, k1, k2, k3):
    """MO-basis THC-ERI block: get_ao_eri transformed per k-point.

    The AO convention is ``(a* b | c* d)``, so the first and third
    MO coefficient matrices are conjugated in the transformation.

    Args:
        mo_coeff_kpts: length-4 sequence (C1, C2, C3, C4), each
            (Nao, n_i) complex128. k4 is derived internally, but the
            caller MUST supply C4 already selected for whatever k4 turns
            out to be.

    Returns:
        (eri_mo, k4): eri_mo (n1, n2, n3, n4) complex128.
    """
    if len(mo_coeff_kpts) != 4:
        raise ValueError(f"mo_coeff_kpts must have exactly 4 entries, got {len(mo_coeff_kpts)}.")
    C1, C2, C3, C4 = (np.asarray(C, dtype=np.complex128) for C in mo_coeff_kpts)

    eri_ao, k4 = get_ao_eri(inpv_kpt, coul_kpt, kconserv, k1, k2, k3)
    eri_mo = np.einsum(
        "abcd,ai,bj,ck,dl->ijkl", eri_ao, C1.conj(), C2, C3.conj(), C4,
        optimize=True,
    )
    return eri_mo, k4


def get_j(cell, dm_kpts, kpts):
    """Periodic Coulomb (J) matrix -- delegates entirely to pyscf grid-J
    via a real FFTDF object (never ISDF-factorized; design doc §6).

    Returns:
        vj_kpts: same leading shape convention as dm_kpts.
    """
    from pyscf.pbc.df import FFTDF
    from pyscf.pbc.df.fft_jk import get_j_kpts

    return get_j_kpts(FFTDF(cell), dm_kpts, kpts=np.asarray(kpts))


class ISDFDF:
    """Thin pyscf-compatible `with_df` adapter (design doc §8, V4 gate):
    wraps build()/get_k/get_j behind the get_jk interface pyscf's
    KRHF/KRKS call on mf.with_df. Gate-thin: one-time build()
    memoization only; kpts_band and omega are NOT supported.

    Args:
        kpts: (Nk,3) absolute k-points, e.g. cell.make_kpts(kmesh).
        rank, block_size, rtol, retention_mode: forwarded to build().
    """

    def __init__(self, cell, kpts, *, rank, block_size, rtol=None, retention_mode="single",
                 selection_mode=None, fixed_pivots=None,
                 bpc_batch_size=FROZEN_BPC_POLICY["bpc_batch_size"],
                 bpc_min_separation=FROZEN_BPC_POLICY["bpc_min_separation"],
                 bpc_candidate_oversampling=FROZEN_BPC_POLICY["bpc_candidate_oversampling"],
                 bpc_n_topup=FROZEN_BPC_POLICY["bpc_n_topup"], reuse_ao_cache_for_eta=True,
                 stage_eta_root=None, stage_eta_block=4096, kern_blocking=None,
                 n_retained_pin=None, convolve_device=False, p_block_rows=None,
                 solve_backend="device", jitter_rcond=None, cached_ao_max_bytes=None):
        self.cell = cell
        self.kpts = np.asarray(kpts, dtype=np.float64)
        self.rank = rank
        self.block_size = block_size
        self.rtol = rtol
        self.retention_mode = retention_mode
        self.solve_backend = solve_backend
        self.jitter_rcond = jitter_rcond
        self.cached_ao_max_bytes = cached_ao_max_bytes
        self.selection_mode = selection_mode
        self.fixed_pivots = None if fixed_pivots is None else np.asarray(fixed_pivots)
        self.bpc_batch_size = bpc_batch_size
        self.bpc_min_separation = bpc_min_separation
        self.bpc_candidate_oversampling = bpc_candidate_oversampling
        self.bpc_n_topup = bpc_n_topup
        self.reuse_ao_cache_for_eta = reuse_ao_cache_for_eta
        self.convolve_device = convolve_device
        self.p_block_rows = p_block_rows
        # Fail here, not hours later inside build() after pivot selection.
        validate_option_compatibility(
            p_block_rows=p_block_rows, kern_blocking=kern_blocking,
            stage_eta_root=stage_eta_root, solve_backend=solve_backend,
            jitter_rcond=jitter_rcond, retention_mode=retention_mode, rtol=rtol,
            n_retained_pin=n_retained_pin)
        self.stage_eta_root = stage_eta_root
        self.stage_eta_block = stage_eta_block
        self.kern_blocking = kern_blocking
        self.n_retained_pin = n_retained_pin
        self._built = None
        self._ao2mo_call_count = 0
        # get_pp/get_nuc (core-Hamiltonian integrals, unrelated to the J/K
        # factorization) delegate to a real FFTDF instance.
        from pyscf.pbc.df import FFTDF

        self._core_df = FFTDF(cell, kpts)

    def get_pp(self, kpts=None):
        return self._core_df.get_pp(self.kpts if kpts is None else kpts)

    def get_nuc(self, kpts=None):
        return self._core_df.get_nuc(self.kpts if kpts is None else kpts)

    def build(self):
        """Run S1-S4 once and cache; the build artifacts are
        density-independent."""
        if self._built is None:
            self._built = build(
                self.cell, self.kpts, rank=self.rank, block_size=self.block_size,
                rtol=self.rtol, retention_mode=self.retention_mode,
                solve_backend=self.solve_backend, jitter_rcond=self.jitter_rcond,
                cached_ao_max_bytes=self.cached_ao_max_bytes,
                selection_mode=self.selection_mode,
                fixed_pivots=self.fixed_pivots,
                bpc_batch_size=self.bpc_batch_size,
                bpc_min_separation=self.bpc_min_separation,
                bpc_candidate_oversampling=self.bpc_candidate_oversampling,
                bpc_n_topup=self.bpc_n_topup,
                reuse_ao_cache_for_eta=self.reuse_ao_cache_for_eta,
                stage_eta_root=self.stage_eta_root,
                stage_eta_block=self.stage_eta_block,
                kern_blocking=self.kern_blocking,
                n_retained_pin=self.n_retained_pin,
                convolve_device=self.convolve_device,
                p_block_rows=self.p_block_rows,
            )
        return self._built

    def ao2mo(self, mo_coeffs, kpts, compact=False):
        """Return one momentum-conserving MO ERI block for periodic MP2.

        This experimental adapter path deliberately reuses the already-built
        fixed-pivot ISDF factorization.  PySCF's KMP2 requests
        ``compact=False`` blocks, so packed output is intentionally not
        implemented.
        """
        if compact:
            raise NotImplementedError("ISDFDF.ao2mo supports compact=False only.")
        if len(mo_coeffs) != 4:
            raise ValueError("mo_coeffs must contain exactly four k-point coefficient arrays.")
        kpts = np.asarray(kpts, dtype=np.float64)
        if kpts.shape != (4, 3):
            raise ValueError("kpts must have shape (4, 3).")
        built = self.build()
        canonical = built["mesh_obj"].canonical_kpts
        indices = []
        for kpt in kpts:
            matches = np.flatnonzero(np.all(np.isclose(canonical, kpt, atol=1e-8), axis=1))
            if matches.size != 1:
                raise ValueError("ao2mo kpts must belong uniquely to this adapter's canonical mesh.")
            indices.append(int(matches[0]))
        from pytc.pbc.df.kpts import build_kconserv

        kconserv = build_kconserv(self.cell, canonical)
        eri, k4 = get_mo_eri(
            built["inpv_kpt"], built["coul_kpt"], kconserv, mo_coeffs,
            indices[0], indices[1], indices[2],
        )
        if k4 != indices[3]:
            raise ValueError("ao2mo kpts violate momentum conservation for this mesh.")
        self._ao2mo_call_count += 1
        return eri.reshape(-1)

    def get_jk(self, dm_kpts, hermi=1, kpts=None, kpts_band=None, with_j=True,
               with_k=True, omega=None, exxdiv=None):
        if omega is not None:
            raise NotImplementedError(
                "ISDFDF.get_jk: omega (range-separated hybrids) is out of scope for "
                "the V4 gate-thin adapter."
            )
        if kpts_band is not None:
            raise NotImplementedError(
                "ISDFDF.get_jk: kpts_band (band-structure evaluation) is out of "
                "scope for the V4 gate-thin adapter."
            )
        kpts = self.kpts if kpts is None else np.asarray(kpts, dtype=np.float64)
        if kpts.shape != self.kpts.shape or not np.allclose(kpts, self.kpts):
            raise ValueError(
                "ISDFDF.get_jk: kpts passed by the caller do not match the kpts "
                "this adapter was built with -- the cached build() artifact is only "
                "valid for the ORIGINAL kpts."
            )

        vj = None
        vk = None
        if with_j:
            from pyscf.pbc.df.fft_jk import get_j_kpts

            vj = get_j_kpts(self._core_df, dm_kpts, kpts=kpts)
        if with_k:
            built = self.build()
            vk = get_k(
                dm_kpts, built["inpv_kpt"], built["coul_kpt"], built["mesh_obj"].phase,
                exxdiv=exxdiv, cell=self.cell, kpts=kpts, neg=built["mesh_obj"].neg,
            )
        return vj, vk
