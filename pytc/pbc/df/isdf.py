"""Periodic ISDF fit machinery: matrix-free Hermitian-PSD pivoted Cholesky
selector, Pi^q/eta^q builders, kernel-apply-and-solve (NumPy oracle and
device/KernelProvider paths), and staging policy. See design doc §3-§7.
Deliberately independent of pytc.df.pivots (see design doc §3).
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


# The exact E1 selector keeps the complete Bloch AO cache on the JAX device.
# This is intentionally a bounded, fail-closed baseline.  Any distinct
# localized backend is separately held until measured capacity evidence and
# owner direction justify it; no fallback changes this selector's physical
# candidate set.
def pivoted_cholesky_hermitian(diag, col_eval, rank, *, rcond=1e-12, ramp_scale=1e-12):
    """Matrix-free greedy pivoted (partial) Cholesky for an implicit N x N
    Hermitian PSD matrix M given diag(M) and a column oracle
    col_eval(j) -> M[:, j] of the ORIGINAL M (shape (N,), complex128).

    Tie-break: a tiny increasing ramp `ramp_scale * arange(n) * max(diag)`
    is added to the argmax score, biasing exact ties toward the higher
    index (matches pytc.df.pivots's convention; copied, not imported).

    Returns:
        (pivots, L, n_selected): pivots (n_selected,) int64; L
        (n, n_selected) complex128 partial Cholesky factor; n_selected
        <= rank (fewer if the Schur diagonal exhausts below
        rcond*max(diag) first).
    """
    diag = np.asarray(diag, dtype=np.float64)
    if diag.ndim != 1:
        raise ValueError(f"diag must be 1-D, got shape {diag.shape}.")
    n = diag.shape[0]
    if n == 0:
        raise ValueError("diag must be nonempty.")
    if not np.all(np.isfinite(diag)):
        raise ValueError("diag must be finite.")

    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError(f"rank must be an integer, got {rank!r}.")
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    if rank > n:
        raise ValueError(f"rank={rank} exceeds n={n} -- cannot select more pivots than rows.")

    max_diag = float(np.max(diag)) if n > 0 else 0.0
    if max_diag <= 0.0:
        raise ValueError("diag is entirely non-positive -- M appears to be the zero matrix.")

    neg_floor = -rcond * max_diag
    if np.any(diag < neg_floor):
        raise ValueError(
            f"diag contains an entry below -{rcond:.1e}*max(diag)={neg_floor:.3e} -- "
            f"M does not appear to be PSD within the declared rcond."
        )
    diag = np.maximum(diag, 0.0)

    ramp = ramp_scale * np.arange(n, dtype=np.float64) * max_diag
    threshold = rcond * max_diag

    L = np.zeros((n, rank), dtype=np.complex128)
    pivots = np.zeros(rank, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)

    n_selected = 0
    for t in range(rank):
        score = np.where(selected, -np.inf, diag + ramp)
        j = int(np.argmax(score))
        if diag[j] <= threshold:
            break

        col = np.asarray(col_eval(j))
        if col.shape != (n,):
            raise ValueError(f"col_eval({j}) must return shape ({n},), got {col.shape}.")
        col = col.astype(np.complex128)

        if t > 0:
            update = L[:, :t] @ L[j, :t].conj()
        else:
            update = 0.0
        l_t = (col - update) / np.sqrt(diag[j])
        L[:, t] = l_t

        diag = diag - np.abs(l_t) ** 2
        diag = np.maximum(diag, 0.0)

        pivots[t] = j
        selected[j] = True
        n_selected += 1

    return pivots[:n_selected], L[:, :n_selected], n_selected


def _batch_candidates(diag, selected, batch_size, mesh, min_separation, ramp):
    """Choose a deterministic, minimum-image-separated stale-diagonal batch."""
    score = np.where(selected, -np.inf, diag + ramp)
    order = np.argsort(score, kind="stable")[::-1]
    accepted = []
    coordinates = []
    mesh = np.asarray(mesh, dtype=np.int64)
    for index in order:
        if not np.isfinite(score[index]):
            break
        coordinate = np.asarray(np.unravel_index(int(index), tuple(mesh)), dtype=np.int64)
        if coordinates:
            delta = np.abs(np.asarray(coordinates) - coordinate)
            delta = np.minimum(delta, mesh - delta)
            if np.any(np.linalg.norm(delta, axis=1) < min_separation):
                continue
        accepted.append(int(index))
        coordinates.append(coordinate)
        if len(accepted) == batch_size:
            break
    return np.asarray(accepted, dtype=np.int64)


# Task #107: the selection factor lives on device (task #113's decision, verified
# by job 59987566 -- donated updates held peak growth at 0.00 GB against a
# control that grew a full extra copy, at 4.1x the speed and bit-identical).
#
# WIDTH BUCKET. The projection reads `factor[:, :p0]`, and p0 advances every
# round, so a literal slice hands jit a NEW SHAPE each time: 580 compilations at
# 444/cc-pvtz. Padding to the full rank instead keeps one shape but does ~2x the
# projection flops on average, and projection is 36.6% of selection. Bucketing the
# width to a multiple of 4096 gives 11 shapes at 1.14x flops -- columns between p0
# and the bucket edge are still zero, so they contribute nothing, and correctness
# follows from the zero-initialisation rather than from explicit masking.
_SELECTION_WIDTH_BUCKET = 4096


def _settle(value):
    """Force a dispatched device computation to finish, so the timer that ends
    here measures it rather than the timer that happens to contain the next
    host read.

    Costs nothing measurable in this loop: nothing independent sits between
    dispatch and the following read, so there is no overlap to lose. Measured at
    0.99x the no-sync wall, inside a 5.6% noise band.
    """
    ready = getattr(value, "block_until_ready", None)
    return ready() if ready is not None else value


# Task #127 A/B knob. OFF by default and inert when off -- the bucketed return
# below is byte-for-byte the previous behaviour, so a flag-off run is the old
# code path.
#
# WHY IT EXISTS: at a bucketed width, `dynamic_slice` in _selection_projection
# MATERIALISES the [n_grid x width] prefix; only at width == rank does XLA alias
# the argument and the copy vanish. Job 60129052 put projection at 62.2% of
# selection, and the aliasing boundary in its own records puts ~64% of that in
# the copy. Passing rank trades the copy for arithmetic on zero columns.
#
# NUMERICALLY IDENTICAL either way: columns in [p0, rank) are exactly zero, which
# is the premise bucketing already relies on. If pivots move, the premise is
# false and the timing is irrelevant -- the run gates on that.
_SELECTION_FULL_WIDTH = False


def _bucketed_width(p0, rank):
    if p0 <= 0:
        return 0
    if _SELECTION_FULL_WIDTH:
        return int(rank)
    return int(min(rank, -(-p0 // _SELECTION_WIDTH_BUCKET) * _SELECTION_WIDTH_BUCKET))


@partial(jax.jit, static_argnames=("width",))
def _selection_projection(factor, retained_idx, block, width):
    """block - L @ conj(L[retained]).T for L = factor[:, :width], on device.

    WARNING: this slice IS materialised. Slicing under trace rather than in
    Python was intended to let XLA fold it into the dot operand; XLA's own
    memory accounting says it does not, at the production shape
    (n_grid=328509, rank=37120):

        width   4096   slice 10.8 GB   temp 10.8 GB
        width  18432   slice 48.4 GB   temp 48.5 GB
        width  37120   slice 97.6 GB   temp  0.0 GB   <- width == rank

    Temp tracks the slice exactly until the slice becomes the identity, at
    which point XLA aliases the argument and it is free. So the copy is real
    and avoidable: the columns in [p0, rank) are exactly zero -- the same
    premise the bucket already relies on -- so passing the full factor is
    numerically identical and costs no temp, at the price of GEMM arithmetic on
    zero columns (~2x FLOPs). Whether that trade pays is being measured; do not
    assume it either way.

    `width` is static so the traced shape is fixed; bucketing keeps the number
    of distinct widths at 11 rather than 580.

    Task #113 existed to eliminate per-round copies of this array and verified
    donation prevents them in the WRITE. This line reintroduced them in the
    READ, where that verification never looked.
    """
    left = jax.lax.dynamic_slice(factor, (0, 0), (factor.shape[0], width))
    return block - left @ jnp.conj(left[retained_idx]).T


@partial(jax.jit, donate_argnums=(0,))
def _selection_write(factor, block_factor, p0):
    """Write a retained block into the factor IN PLACE.

    donate_argnums=(0,) is load-bearing, not an optimisation: without it every
    round copies the whole [n_grid x rank] factor -- 97.6 GB at 444/cc-pvtz, 581
    times. Measured in job 59987566: donated 0.00 GB growth, undonated a full
    extra copy. The caller must not use the old buffer afterwards.
    """
    return jax.lax.dynamic_update_slice(factor, block_factor, (0, p0))


def _write_factor_block(factor, block_factor, p0, *, check=True):
    """Donated in-place write, with donation ASSERTED rather than assumed.

    JAX deletes a buffer it has donated, so `is_deleted()` reports directly
    whether the write reused the input or silently copied it. At 444/cc-pvtz the
    factor is 97.6 GB and there are ~596 rounds, so a donation that quietly
    stops firing costs ~58 TB of copying -- a performance failure that looks
    exactly like working code, and which no small-scale test can reach: donation
    verified in situ at 3 GB (16/16 rounds donated) while job 59992249's
    selection footprint was 255.5 GB against an expected 117.7 GB, i.e. the
    shape of a second factor.

    Raising here converts that silent failure into a named one, in the log of
    whichever run hits it.
    """
    out = _selection_write(factor, block_factor, p0)
    if check and not factor.is_deleted():
        out.block_until_ready()
        if not factor.is_deleted():
            raise RuntimeError(
                "donation did not fire on the selection factor write "
                f"(shape {tuple(factor.shape)}, {factor.nbytes/1e9:.1f} GB): the "
                "input buffer survived, so this round COPIED the factor instead "
                "of updating it in place. Every round would pay that copy."
            )
    return out


def pivoted_cholesky_batched_hermitian(
    diag, col_batch_eval, rank, *, mesh, batch_size, min_separation=2.0,
    candidate_oversampling=1, n_topup=0, rcond=1e-12, ramp_scale=1e-12,
    stage_stats=None, blocked_projection=False,
):
    """Approximate greedy selection with exact batched columns and updates.

    A stale-diagonal candidate pool of ``candidate_oversampling * batch_size``
    members is chosen, but only ``batch_size`` members are retained after exact
    within-pool re-pivoting.  Columns are exact, so oversampling changes only
    the cheap per-round column GEMM, not the streamed AO-sweep count.
    Optionally, the final ``n_topup`` pivots are selected as exact greedy
    singleton batches; this repairs final-residual order-statistic error
    without changing the preceding batched rounds.
    """
    diag = np.asarray(diag, dtype=np.float64)
    mesh = tuple(int(value) for value in mesh)
    if diag.ndim != 1 or np.prod(mesh) != diag.size:
        raise ValueError("mesh must be a positive grid shape matching diag.")
    if (
        rank <= 0 or rank > diag.size or batch_size <= 0 or min_separation < 0
        or isinstance(candidate_oversampling, bool)
        or not isinstance(candidate_oversampling, (int, np.integer))
        or candidate_oversampling <= 0
        or isinstance(n_topup, bool) or not isinstance(n_topup, (int, np.integer))
        or n_topup < 0 or n_topup > rank
    ):
        raise ValueError(
            "invalid rank, batch_size, min_separation, candidate_oversampling, or n_topup."
        )
    if stage_stats is not None and not isinstance(stage_stats, list):
        raise ValueError("stage_stats must be a list when supplied.")
    candidate_oversampling = int(candidate_oversampling)
    n_topup = int(n_topup)
    initial_max = float(np.max(diag))
    if initial_max <= 0:
        raise ValueError("diag is entirely non-positive.")
    threshold = rcond * initial_max
    ramp = ramp_scale * np.arange(diag.size, dtype=np.float64) * initial_max
    # factor L follows the metric's dtype (real-f64-L): the periodic selection
    # metric M = |S|^2/Nk is real, so its factor is float64 and this [Ng x rank]
    # array -- the dominant selection-phase memory term -- halves; a general
    # complex Hermitian metric keeps a complex128 factor unchanged. Allocated
    # lazily on the first column batch, once columns.dtype is known.
    factor = None
    selected = np.zeros(diag.size, dtype=bool)
    pivots = []
    rounds = []
    batched_rank = rank - n_topup
    while len(pivots) < batched_rank:
        retain_count = min(batch_size, batched_rank - len(pivots))
        requested = min(candidate_oversampling * retain_count, diag.size - len(pivots))
        candidates = _batch_candidates(
            diag, selected, requested, mesh, min_separation, ramp,
        )
        if candidates.size == 0 or diag[candidates[0]] <= threshold:
            break
        candidate_started = time.perf_counter()
        columns = np.asarray(col_batch_eval(candidates))
        candidate_seconds = time.perf_counter() - candidate_started
        if columns.shape != (diag.size, candidates.size):
            raise ValueError("col_batch_eval must return shape (n_grid, n_batch).")
        if factor is None:
            # Device-resident for the blocked path (task #113). The legacy
            # branch below writes columns in place with numpy assignment, so it
            # keeps a host array -- porting it is not in scope and pretending
            # otherwise would silently break the reference path.
            if blocked_projection:
                factor = jnp.zeros((diag.size, rank), dtype=columns.dtype)
            else:
                factor = np.zeros((diag.size, rank), dtype=columns.dtype)
        projection_started = time.perf_counter()
        residual_batch = columns[candidates]
        if pivots:
            # Gather the candidate ROWS, then slice columns -- not the reverse.
            # `factor[:, :p]` builds an [n_grid x p] intermediate, and on a
            # device array that is a materialised copy (up to 97.6 GB per round
            # at 444/cc-pvtz) of which only these `batch` rows are ever read.
            # Measured 130x on JAX and 1.0x on numpy (job 60001324) -- the
            # asymmetry is why this line was free before the port and expensive
            # after it, and why the baseline ran in 1,877 s.
            existing_c = factor[candidates, :len(pivots)]
            residual_batch = residual_batch - existing_c @ existing_c.conj().T
        # Settle HERE so `projection_seconds` means the projection.
        #
        # Without this the gather and GEMM above are merely DISPATCHED, and
        # their execution is paid at the first host read -- `np.diag` below,
        # inside `materialisation_seconds`. That made the bucket named for the
        # device read actually hold "GEMM execution + host reduction", so a
        # materialisation-dominated profile could not distinguish "the read is
        # the regression" from "the projection GEMM is the regression". Given
        # the GEMM is the leading suspect (130x on JAX vs 1.0x on numpy, above),
        # the ambiguity sat exactly on the hypothesis under test. Found by
        # @Woke reviewing the profiling driver before its job ran.
        residual_batch = _settle(residual_batch)
        projection_seconds = time.perf_counter() - projection_started
        # `projection_seconds` above is DISPATCH ONLY when blocked_projection is
        # on: `factor` is then a jnp array, so the gather and the GEMM are both
        # async and neither has run yet. The next line is the first host read of
        # `residual_batch`, so it is where that work is actually paid for.
        #
        # Timing it separately costs nothing -- the sync already happened here.
        #
        # An earlier revision of this comment justified NOT calling
        # `block_until_ready()` above by claiming a forced sync "would serialise
        # dispatch". That was a cost claim asserted from reading the code, and
        # a standalone overlap probe refutes it (agent workspace, not in this
        # repo -- ask for `batched_overlap_probe.py` if you want to rerun it):
        # forcing the sync
        # measured 0.99x the no-sync wall, inside a 5.6% noise band. Nothing
        # independent sits between the dispatch and this read, so there is no
        # overlap for a sync to destroy. Both spellings cost the same.
        #
        # The real reason to time this line instead is narrower: the read
        # already happens here, so it needs no added call, and this way the
        # bucket also captures the host-side `diag`/`real`/`maximum` reduction
        # rather than only the device wait.
        #
        # Without this bucket the seconds land between two `perf_counter` calls
        # and are charged to NO stage: a local reproduction of this exact shape
        # (`timer_blindness_probe.py`, checksum-matched arms; kept in the agent
        # workspace, not in this repo -- ask if you want to rerun it) put
        # 47-66% of the real projection cost in that untimed gap, varying run to
        # run. Any attribution built on `projection_seconds` alone is unfounded.
        materialisation_started = time.perf_counter()
        local_diag = np.maximum(np.real(np.diag(residual_batch)), 0.0)
        materialisation_seconds = time.perf_counter() - materialisation_started
        within_batch_started = time.perf_counter()
        local_pivots, _, local_count = pivoted_cholesky_hermitian(
            local_diag, lambda index: residual_batch[:, index],
            rank=min(candidates.size, retain_count), rcond=rcond,
            ramp_scale=ramp_scale,
        )
        within_batch_seconds = time.perf_counter() - within_batch_started
        retained = []
        factor_update_seconds = 0.0
        if blocked_projection:
            # The sequential per-pivot arithmetic below, as three level-3 ops:
            #   (a) pre-round-L correction: block = columns - L @ conj(L[idx]).T
            #   (b) round-mate orthogonalization: block = V R => V = block R^-1,
            #       R being the within-batch Cholesky (R^H R = the retained
            #       sub-Gram, the factorization the sequential loop performs
            #       implicitly);
            #   (c) blocked diagonal update, diag -= sum(|V|^2, axis=1), summing
            #       retained columns in the sequential loop's order.
            # A reordering of identical arithmetic: local_pivots and the
            # candidate residual Gram are byte-identical, and the only fp delta
            # is R's Gram recursion vs sequential coefficient accumulation.
            from scipy.linalg import solve_triangular
            from jax.scipy.linalg import solve_triangular as jax_solve_triangular
            retained_local = []
            for local_index in local_pivots[:local_count]:
                index = int(candidates[local_index])
                if selected[index] or diag[index] <= threshold:
                    continue
                retained_local.append(int(local_index))
                selected[index] = True
                pivots.append(index)
                retained.append(index)
                if len(pivots) == rank:
                    break
            m = len(retained_local)
            if m:
                p0 = len(pivots) - m
                retained_local = np.asarray(retained_local, dtype=np.int64)
                retained_idx = candidates[retained_local]
                projection_started = time.perf_counter()
                block = jnp.asarray(columns[:, retained_local])
                if p0:
                    # Bucketed width, not `:p0`: see _SELECTION_WIDTH_BUCKET.
                    # Columns in [p0, width) are still zero and contribute nothing.
                    width = _bucketed_width(p0, rank)
                    block = _selection_projection(
                        factor, retained_idx, block, width)
                # Same reason as the first segment: `_selection_projection` is
                # jitted, so without settling here its execution would be paid
                # at the `jnp.sum` sync below and charged to `factor_update`,
                # which would then not be a write bucket either.
                block = _settle(block)
                projection_seconds += time.perf_counter() - projection_started
                factor_update_started = time.perf_counter()
                gram = residual_batch[np.ix_(retained_local, retained_local)]
                upper_R = np.linalg.cholesky(gram).conj().T
                block_factor = jax_solve_triangular(
                    jnp.asarray(upper_R.T), block.T, lower=True,
                ).T
                # Donated: the old buffer must not be used after this call.
                factor = _write_factor_block(factor, block_factor, p0)
                # Settle the WRITE inside this window. The `jnp.sum` sync below
                # depends on `block_factor`, not on `factor`, so without this the
                # write stays pending and is paid at the NEXT round's
                # `factor[candidates, :p]` gather -- inflating that round's
                # `projection` and deflating this one's `factor_update`, across
                # rounds where it is hardest to notice.
                factor = _settle(factor)
                # diag stays on host -- it drives the loop's control flow and is
                # only [n_grid] float64 (2.6 MB), so the downdate contribution is
                # the one small array that comes back each round.
                diag = np.maximum(
                    diag - np.asarray(
                        jnp.sum(jnp.abs(block_factor) ** 2, axis=1)), 0.0,
                )
                factor_update_seconds += time.perf_counter() - factor_update_started
        else:
            # Full [n_grid x p] is genuinely needed here: the sequential branch
            # forms `existing @ existing[index].conj()` over every grid row.
            # `factor` is numpy on this branch, so the slice is a view and free.
            existing = factor[:, :len(pivots)]
            for local_index in local_pivots[:local_count]:
                index = int(candidates[local_index])
                if selected[index] or diag[index] <= threshold:
                    continue
                projection_started = time.perf_counter()
                correction = existing @ existing[index].conj() if pivots else 0.0
                projection_seconds += time.perf_counter() - projection_started
                factor_update_started = time.perf_counter()
                vector = (columns[:, local_index] - correction) / np.sqrt(diag[index])
                factor[:, len(pivots)] = vector
                diag = np.maximum(diag - np.abs(vector) ** 2, 0.0)
                factor_update_seconds += time.perf_counter() - factor_update_started
                selected[index] = True
                pivots.append(index)
                retained.append(index)
                existing = factor[:, :len(pivots)]
                if len(pivots) == rank:
                    break
        rounds.append({
            "requested_candidates": candidates.tolist(),
            "within_batch_pivots": [int(candidates[index]) for index in local_pivots[:local_count]],
            "retained_pivots": retained,
        })
        if stage_stats is not None:
            stage_stats.append({
                "stage": "batched", "round_index": len(rounds) - 1,
                "candidate_count": int(candidates.size), "retained_count": len(retained),
                "candidate_eval_seconds": candidate_seconds,
                "projection_seconds": projection_seconds,
                "materialisation_seconds": materialisation_seconds,
                "within_batch_pivot_seconds": within_batch_seconds,
                "factor_update_seconds": factor_update_seconds,
            })
        if not retained:
            break
    topup_start_max_index = None
    topup_start_was_last_round_rejected = None
    if len(pivots) < rank and n_topup:
        topup_start_max_index = int(np.argmax(np.where(selected, -np.inf, diag + ramp)))
        if rounds:
            last_round = rounds[-1]
            topup_start_was_last_round_rejected = (
                topup_start_max_index in last_round["requested_candidates"]
                and topup_start_max_index not in last_round["retained_pivots"]
            )
    topup_pivots = []
    while len(pivots) < rank:
        score = np.where(selected, -np.inf, diag + ramp)
        index = int(np.argmax(score))
        if diag[index] <= threshold:
            break
        candidate_started = time.perf_counter()
        column = np.asarray(col_batch_eval(np.asarray([index], dtype=np.int64)))
        candidate_seconds = time.perf_counter() - candidate_started
        if column.shape != (diag.size, 1):
            raise ValueError("col_batch_eval singleton must return shape (n_grid, 1).")
        if factor is None:
            factor = np.zeros((diag.size, rank), dtype=column.dtype)
        projection_started = time.perf_counter()
        # `factor[:, :p]` is the SAME full-width column slice 897b346 removed
        # from the batched rounds: on a jnp factor it is a materialised copy of
        # an [n_grid x p] prefix. It used to sit ABOVE `projection_started`, so
        # its cost was charged to no stage at all. Timed here, not hoisted --
        # the slice is part of the projection, not free setup.
        #
        # It is NOT the same waste as the batched case, which copied [n_grid x p]
        # to read `batch` rows: this matvec genuinely reads every row. The cost
        # is the extra copy, and by top-up p is already close to `rank`, so
        # padding to full width to read `factor` in place should be nearly free.
        # Untested at 444 -- proposed, not applied.
        existing = factor[:, :len(pivots)]
        correction = existing @ existing[index].conj() if pivots else 0.0
        projection_seconds = time.perf_counter() - projection_started
        # Force the read HERE so the split is exact rather than approximate.
        #
        # Different spelling from the batched round above, same reason. Neither
        # path has anything independent between the dispatch and the host read,
        # so a forced sync destroys no overlap in either (measured for the
        # batched case: 0.99x wall, inside a 5.6% noise band). The paths differ
        # only in WHERE the natural read lands. Above, it is the very next line,
        # so timing that line is enough. Here it is buried inside the factor
        # update, several statements later and mixed with the write -- so the
        # read is pulled forward to a named point. `vector` DEPENDS on
        # `correction`, so nothing could have proceeded without it anyway:
        # this moves where the seconds are recorded, not when work happens.
        #
        # This corrects a claim I committed in e34476d and had NOT measured --
        # that top-up's projection cost is charged to `factor_update_seconds`,
        # recorded as a hardcoded 0.0. @Woke flagged it as unverified and he was
        # right to: a standalone `topup_accounting_probe.py` (agent workspace,
        # not in this repo -- ask if you want to rerun it), checksum-matched
        # arms, measured the OPPOSITE split -- the majority already landed in
        # `projection_seconds` (~70-80% across runs) and only the rest leaked
        # onward. A large jnp column slice does not dispatch freely the way the
        # batched gather does. The hardcoded zero was wrong in both directions:
        # it claimed a leak that is mostly not there, and hid the part that is.
        #
        # Only the DIRECTION of that split transfers -- the probe runs on a
        # loaded laptop at reduced shapes and its absolute times swung several
        # fold between runs. Do not quote its magnitudes as cluster numbers.
        materialisation_started = time.perf_counter()
        if pivots:
            correction = np.asarray(correction)
        materialisation_seconds = time.perf_counter() - materialisation_started
        factor_update_started = time.perf_counter()
        vector = (column[:, 0] - correction) / np.sqrt(diag[index])
        if isinstance(factor, jnp.ndarray):
            # The top-up runs AFTER the batched rounds and writes one column at
            # a time, so it inherits whichever factor those rounds produced. It
            # is NOT gated on blocked_projection, which is why scoping the port
            # to the blocked branch alone left this in-place write reachable --
            # caught by test_isdf_selector, not by inspection.
            factor = _write_factor_block(
                factor, jnp.asarray(vector)[:, None], len(pivots))
            diag = np.maximum(diag - np.abs(np.asarray(vector)) ** 2, 0.0)
        else:
            factor[:, len(pivots)] = vector
            diag = np.maximum(diag - np.abs(vector) ** 2, 0.0)
        factor_update_seconds = time.perf_counter() - factor_update_started
        selected[index] = True
        pivots.append(index)
        topup_pivots.append(index)
        if stage_stats is not None:
            stage_stats.append({
                "stage": "topup", "round_index": len(topup_pivots) - 1,
                "candidate_count": 1, "retained_count": 1,
                "candidate_eval_seconds": candidate_seconds,
                "projection_seconds": projection_seconds,
                "materialisation_seconds": materialisation_seconds,
                "within_batch_pivot_seconds": 0.0,
                "factor_update_seconds": factor_update_seconds,
            })
    if n_topup:
        rounds.append({
            "mode": "exact_topup",
            "n_requested": n_topup,
            "topup_pivots": topup_pivots,
            "pre_topup_max_index": topup_start_max_index,
            "pre_topup_max_was_last_round_rejected": topup_start_was_last_round_rejected,
        })
    if factor is None:
        # No column was ever evaluated -> nothing selected (empty first batch /
        # exhausted diag). No metric dtype was observed, so don't guess one:
        # return an empty (Ng, 0) factor, matching the pre-real-f64-L empty
        # return exactly. len(pivots) is 0 here.
        return (np.asarray(pivots, dtype=np.int64),
                np.zeros((diag.size, 0), dtype=np.complex128), len(pivots), rounds)
    return np.asarray(pivots, dtype=np.int64), factor[:, :len(pivots)], len(pivots), rounds


def _pair_convolve():
    """The convolve. There is exactly one, and it is the device path.

    Owner directive (2026-08-12): "remove the numpy path, since we want the same
    code to be run on GPU in the future" -- and then, on the vestigial selector
    this function replaced: "if numpy is removed, why do we still need this
    keyword?" Correct: a switch with one position is not a switch.

    Measured on the real 222 build before removal (job 59928607), the numpy path
    cost 243.7 s at 24 threads against 38.0 s here, because it put a second thread
    pool (OpenBLAS) alongside XLA's -- and XLA's is sized from the cpuset and
    ignores OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS. At 333:
    1903.0 s -> 470.2 s, a 4.05x cut with no environment tuning.

    ``pair_convolve`` (numpy) still exists in kpts.py as the reference the device
    path is gated against to 8.4e-16 -- the tests call it directly, by name, so
    the build needs no keyword to reach it. Deleting the oracle would delete the
    proof that this path is correct.
    """
    from pytc.pbc.df.kpts import pair_convolve_device
    return pair_convolve_device


def build_pi_eta(X, ao_blocks, phase, neg, *, imag_tol=1e-10):
    """Build Pi^q = pair_convolve(X, X)[q] and eta^q = pair_convolve(X, AO)[q].
    eta is accumulated block-by-block so one pair_convolve call holds only
    one block of AO data. See design doc §4-§5.

    Alg. 1's convolution and Eq. 4/5's defining equations for Pi/eta agree
    only up to a q<->-q relabeling, so both outputs' q-axis is relabeled with
    neg once after construction. Every consumer then receives the physical
    Pi^q/eta^q and needs no convention logic of its own. The relabeling is
    deliberately not inside pair_convolve, which is shared and correct as is.
    The offset is invisible at self-paired q, and invisible in Pi against a
    transpose-based oracle check, so it must be preserved by construction
    rather than by test.

    Args:
        X: (Nk, Nip, Nao) complex128 across the canonical k-mesh.
        ao_blocks: (Nk, Ng, Nao) complex128 array, or iterable of
            (Nk, blk_i, Nao) blocks on the SAME canonical k-mesh.
        phase: (Nk, Nk) unitary matrix (KptsMesh.phase).
        neg: (Nk,) int array (KptsMesh.neg), used to relabel both
            outputs' q-axis.

    Returns:
        (Pi, eta): (Nk, Nip, Nip) and (Nk, Nip, Ng) complex128.
    """
    # Local import: kpts.py stays a leaf.
    pair_convolve = _pair_convolve()

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (Nk, Nip, Nao), got shape {X.shape}.")
    neg = np.asarray(neg)
    if neg.shape != (X.shape[0],):
        raise ValueError(f"neg must have shape ({X.shape[0]},), got {neg.shape}.")

    Pi = pair_convolve(X, X, phase, imag_tol=imag_tol)[neg]

    if isinstance(ao_blocks, np.ndarray):
        ao_blocks = [ao_blocks]
    else:
        ao_blocks = list(ao_blocks)
    if not ao_blocks:
        raise ValueError("ao_blocks must be nonempty.")

    # Preallocate and fill rather than concatenate. The previous form,
    #   np.concatenate([...], axis=2)[neg]
    # held the whole chunk list, the concatenate's full copy, and the
    # fancy-index's full copy simultaneously -- a 3x eta transient that OOM'd
    # a 333 build at 622 GiB before it could reach the third copy. Filling in
    # place holds one eta plus one block.
    #
    # [neg] is applied per block because it commutes with the concatenation:
    # it permutes axis 0 (k) while blocks concatenate along axis 2 (grid).
    # Same reasoning build_pi_eta_staged relies on.
    n_grid_total = sum(int(np.asarray(block).shape[1]) for block in ao_blocks)
    eta = None
    col = 0
    for block in ao_blocks:
        Z = pair_convolve(X, np.asarray(block), phase, imag_tol=imag_tol)[neg]
        if eta is None:
            eta = np.empty((Z.shape[0], Z.shape[1], n_grid_total), dtype=Z.dtype)
        eta[:, :, col:col + Z.shape[2]] = Z
        col += int(Z.shape[2])
        del Z
    if col != n_grid_total:
        raise ValueError(
            f"eta covered {col} grid points, expected {n_grid_total} -- the AO "
            f"block stream did not span the grid."
        )
    return Pi, eta


class StagedEta:
    """Per-q view over a grid-major staged eta file.

    INVARIANT: the file holds flush records back to back, each record the
    C-order (Nk, Nip, ncols) block for one grid range. A flush is therefore a
    single sequential append, and one q's slab within a record is contiguous.
    ``self[q]`` assembles the (Nip, Ng) slab the solve expects.

    COST: the q-major layout this replaces made a flush a last-axis slice --
    Nk*Nip runs of ncols*itemsize scattered over the whole file. Measured on
    NFS at the 333 geometry (155 GiB extent, 2.83 MiB stride, 64 KiB runs),
    same node and same bytes, only the access pattern differing:
    43.5 MiB/s q-major against 807.2 MiB/s appended whole -- 18.6x. The cost
    tracks stride distance and file extent, not run length: sizing the probe
    file to the bytes written shrinks the stride, the runs coalesce, and the
    effect disappears.
    """

    def __init__(self, path, n_kpts, n_ip, n_grid, record_cols):
        self.path = str(path)
        self.shape = (int(n_kpts), int(n_ip), int(n_grid))
        self.record_cols = [int(c) for c in record_cols]
        if sum(self.record_cols) != int(n_grid):
            raise ValueError(
                f"staged records cover {sum(self.record_cols)} grid points, "
                f"expected {n_grid}."
            )
        self.dtype = np.complex128

    def __getitem__(self, q):
        n_kpts, n_ip, n_grid = self.shape
        q = int(q)
        if not 0 <= q < n_kpts:
            raise IndexError(f"q index {q} out of range for {n_kpts} k-points.")
        flat = np.memmap(self.path, dtype=self.dtype, mode="r")
        out = np.empty((n_ip, n_grid), dtype=self.dtype)
        elem, col = 0, 0
        for ncols in self.record_cols:
            start = elem + q * n_ip * ncols
            out[:, col:col + ncols] = flat[start:start + n_ip * ncols].reshape(n_ip, ncols)
            elem += n_kpts * n_ip * ncols
            col += ncols
        del flat
        return out


def build_pi_eta_staged(X, ao_blocks, phase, neg, *, staging_path, n_grid,
                        staging_block=4096, imag_tol=1e-10,
                        free_bytes_safety=1.25, additional_reserve_bytes=0,
                        ):
    """build_pi_eta with eta written to a (Nk, Nip, Ng) C-order memmap rather
    than held in RAM, so only one q's contiguous slab need be resident.

    The q axis is not separable (pair_convolve couples all k), so eta is still
    produced all-q-per-grid-block; this stages the assembled array without
    restructuring the math. Writes are buffered to `staging_block` grid points,
    decoupling the write run length from the AO block size.

    The caller owns the staged file and must unlink it. Returns
    (Pi, eta_memmap, stats). Costs and design:
    docs/isdf-periodic/task46_phaseB_eta_staging_spec.md.
    """
    pair_convolve = _pair_convolve()

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (Nk, Nip, Nao), got shape {X.shape}.")
    neg = np.asarray(neg)
    if neg.shape != (X.shape[0],):
        raise ValueError(f"neg must have shape ({X.shape[0]},), got {neg.shape}.")
    n_kpts, n_ip = int(X.shape[0]), int(X.shape[1])
    n_grid = int(n_grid)
    if n_grid <= 0 or int(staging_block) <= 0:
        raise ValueError("n_grid and staging_block must be positive.")

    Pi = pair_convolve(X, X, phase, imag_tol=imag_tol)[neg]

    # Fail closed before writing: never start a stage we cannot finish.
    # Reserve the concurrent peak so the refusal precedes the large write.
    predicted = n_kpts * n_ip * n_grid * np.dtype(np.complex128).itemsize
    required = predicted + int(additional_reserve_bytes)
    staging_dir = os.path.dirname(os.path.abspath(staging_path)) or "."
    stat = os.statvfs(staging_dir)
    free = stat.f_bavail * stat.f_frsize
    if free < required * float(free_bytes_safety):
        raise OSError(
            f"eta staging REFUSED: {staging_dir} has {free / 2**30:.1f} GiB free, "
            f"needs {required * float(free_bytes_safety) / 2**30:.1f} GiB "
            f"(eta {predicted / 2**30:.1f} + reserved "
            f"{int(additional_reserve_bytes) / 2**30:.1f} GiB, x{free_bytes_safety})."
        )

    # One 3-D array is a single block; an iterable is streamed, never list()-ed.
    if isinstance(ao_blocks, np.ndarray):
        ao_blocks = [ao_blocks]

    # Grid-major: each flush is appended whole, so the write is sequential.
    # The q-major layout this replaces turned every flush into a last-axis
    # slice -- Nk*Nip small runs sprayed across the entire file.
    handle = open(staging_path, "wb")
    record_cols = []
    pending, pending_cols, col0 = [], 0, 0
    write_seconds, bytes_written = 0.0, 0

    def _flush(pending, pending_cols, col0):
        if not pending:
            return col0, 0.0, 0
        chunk = pending[0] if len(pending) == 1 else np.concatenate(pending, axis=2)
        chunk = np.ascontiguousarray(chunk)
        started = time.perf_counter()
        chunk.tofile(handle)
        record_cols.append(int(pending_cols))
        return col0 + pending_cols, time.perf_counter() - started, chunk.nbytes

    n_flushes = 0
    stage_started = time.perf_counter()
    for block in ao_blocks:
        # [neg] commutes with the concatenation: it permutes axis 0.
        Z = pair_convolve(X, np.asarray(block), phase, imag_tol=imag_tol)[neg]
        pending.append(Z)
        pending_cols += int(Z.shape[2])
        if pending_cols >= int(staging_block):
            col0, dt, nb = _flush(pending, pending_cols, col0)
            write_seconds += dt
            bytes_written += nb
            n_flushes += 1
            pending, pending_cols = [], 0
            # Progress marker: without it this stage is silent for hours and a
            # slow run is indistinguishable from a hung one.
            logger.info(
                "eta staging: %d/%d grid points (%.1f%%), %.2f GB written, "
                "%.1f MiB/s inst, %.1f MiB/s cumulative",
                col0, n_grid, 100.0 * col0 / n_grid, bytes_written / 1e9,
                (nb / 2**20 / dt) if dt > 0 else float("nan"),
                (bytes_written / 2**20 / write_seconds) if write_seconds > 0 else float("nan"),
            )
    col0, dt, nb = _flush(pending, pending_cols, col0)
    write_seconds += dt
    bytes_written += nb

    if col0 != n_grid:
        raise ValueError(
            f"staged eta covered {col0} grid points, expected {n_grid} -- the AO "
            f"block stream did not span the grid."
        )
    started = time.perf_counter()
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    write_seconds += time.perf_counter() - started

    eta = StagedEta(staging_path, n_kpts, n_ip, n_grid, record_cols)

    stats = {
        "staging_path": str(staging_path),
        "staged_bytes": int(bytes_written),
        "staging_block": int(staging_block),
        # A flush is now appended whole, so the contiguous run is the entire
        # record, not one grid-block row.
        "write_run_bytes": int(n_kpts) * int(n_ip) * int(staging_block)
        * np.dtype(np.complex128).itemsize,
        "layout": "grid_major_records",
        "record_cols": list(record_cols),
        "write_seconds": float(write_seconds),
        # Sequential appends plus an explicit fsync at close, so this is closer
        # to real throughput than the memmap slice-assignment timing it
        # replaces -- but allocated-blocks-over-time (du) remains the only
        # cache-proof measure.
        "write_gb_per_s": (float(bytes_written) / 1e9 / write_seconds
                           if write_seconds > 0 else None),
        "n_flushes": int(n_flushes),
        "stage_wall_seconds": float(time.perf_counter() - stage_started),
        "free_bytes_before": int(free),
    }
    return Pi, eta, stats


def apply_raw_kernel_and_solve(
    Pi_q, eta_q, *, cell, q_kpt, grid_coords, grid_mesh, rtol=None, self_paired=False,
    n_retained_pin=None, retention_mode="single", jitter_rcond=None,
):
    """Apply the "raw" (bare 4pi/G^2, exx=False) periodic Coulomb kernel to
    eta^q over the spatial grid, contract to (Nip, Nip), and solve the
    Hermitian sandwich for W^q. Plain NumPy, single q -- the CPU oracle for
    the device KernelProvider path. See design doc §5, §7.

        lq     = eta_q * exp(-1j * grid_coords @ q_kpt)   # Bloch phase
        wq     = FFT(lq, grid_mesh)
        vq     = coulG(q, exx=False) * vol / Ng
        rq     = conj(IFFT(wq * vq, grid_mesh))
        kern_q = lq @ rq.T / sqrt(Ng)
        W_q    = sqrt(Ng) * hermitian_sandwich_solve(Pi_q, kern_q)[0]

    The final sqrt(Ng) rescale exactly cancels kern_q's 1/sqrt(Ng)
    (paper Eq. 10 factor placements; verified in the V2 reference-replay
    test). exxdiv is NEVER applied here -- it is owned by a later get_k
    post-processing step.

    Args:
        Pi_q: (Nip, Nip) complex128 metric.
        eta_q: (Nip, Ng) complex128 RHS.
        q_kpt: (3,) absolute k-vector for this q.
        grid_coords: (Ng, 3), same flattened order as eta_q's grid axis.
        grid_mesh: (3,) positive ints, real-space integration mesh
            (distinct from the k-point mesh); prod must equal Ng.
        rtol: forwarded to hermitian_sandwich_solve.
        self_paired: True when neg[q]==q. Physics requires both Pi_q and
            kern_q real for such q, but the complex intermediates leave
            floating-point imaginary noise that the near-singular solve
            amplifies; when True, Pi_q.real and kern_q.real are taken
            BEFORE the solve (noise projection, not a loosened gate). See
            design doc §5.

    Returns:
        (W_q, kern_q, solve_info): W_q (Nip, Nip) complex128; kern_q is
        the raw contracted kernel before the solve; solve_info is
        hermitian_sandwich_solve's info dict.
    """
    from pyscf.pbc import tools as pbctools

    from pytc.df.solvers import hermitian_sandwich_solve

    Pi_q = np.asarray(Pi_q)
    eta_q = np.asarray(eta_q)
    n_ip = Pi_q.shape[0]
    if Pi_q.shape != (n_ip, n_ip):
        raise ValueError(f"Pi_q must be square, got shape {Pi_q.shape}.")
    if eta_q.ndim != 2 or eta_q.shape[0] != n_ip:
        raise ValueError(f"eta_q must have shape ({n_ip},Ng), got {eta_q.shape}.")
    n_grid = eta_q.shape[1]
    if self_paired:
        Pi_q = Pi_q.real.astype(np.complex128)

    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords.shape != (n_grid, 3):
        raise ValueError(
            f"grid_coords must have shape ({n_grid},3) matching eta_q's grid axis, "
            f"got {grid_coords.shape}."
        )
    grid_mesh_t = _require_mesh3(grid_mesh)
    if int(np.prod(grid_mesh_t)) != n_grid:
        raise ValueError(
            f"prod(grid_mesh)={int(np.prod(grid_mesh_t))} != eta_q's grid size {n_grid}."
        )

    q_kpt = np.asarray(q_kpt, dtype=np.float64)
    if q_kpt.shape != (3,):
        raise ValueError(f"q_kpt must have shape (3,), got {q_kpt.shape}.")

    phase = np.exp(-1j * (grid_coords @ q_kpt))
    lq = eta_q * phase[None, :]

    lq_mesh = lq.reshape((n_ip,) + grid_mesh_t)
    wq_mesh = np.fft.fftn(lq_mesh, axes=(1, 2, 3), norm="backward")

    Gv = cell.get_Gv(list(grid_mesh_t))
    vq = pbctools.get_coulG(cell, k=q_kpt, exx=False, Gv=Gv, mesh=list(grid_mesh_t))
    vq = vq * (cell.vol / n_grid)
    vq_mesh = vq.reshape(grid_mesh_t)

    rq_mesh = np.fft.ifftn(wq_mesh * vq_mesh[None, :, :, :], axes=(1, 2, 3), norm="backward")
    rq = rq_mesh.reshape(n_ip, n_grid).conj()

    kern_q = (lq @ rq.T) / np.sqrt(n_grid)
    kern_q = np.asarray(kern_q, dtype=np.complex128)
    if self_paired:
        kern_q = kern_q.real.astype(np.complex128)

    if retention_mode == "cholesky_jitter":
        # This mode has no spectral cutoff, so rtol is rejected rather than dropped:
        # a caller sweeping rtol over a mode that ignores it gets identical runs and
        # reads them as insensitivity to rtol.
        if rtol is not None:
            raise ValueError(
                "rtol does not apply to retention_mode='cholesky_jitter'; pass "
                "jitter_rcond to set the jitter scale."
            )
        if n_retained_pin is not None:
            raise ValueError(
                "n_retained_pin does not apply to retention_mode='cholesky_jitter': "
                "the mode regularizes rather than truncating, so it has no retained set."
            )
        W_q_unscaled, solve_info = hermitian_sandwich_solve(
            Pi_q, kern_q, retention_mode=retention_mode, jitter_rcond=jitter_rcond
        )
    else:
        if jitter_rcond is not None:
            raise ValueError(
                f"jitter_rcond applies only to retention_mode='cholesky_jitter', got "
                f"{retention_mode!r}."
            )
        W_q_unscaled, solve_info = hermitian_sandwich_solve(
            Pi_q, kern_q, rtol=rtol, retention_mode=retention_mode,
            n_retained_pin=n_retained_pin
        )
    # sqrt(Ng) rescale cancels kern_q's own 1/sqrt(Ng) (Eq. 10 factor placement).
    W_q = np.sqrt(n_grid) * W_q_unscaled
    return W_q, kern_q, solve_info


# ---------------------------------------------------------------------------
# Device path, KernelProvider protocol (design doc §7).
#
# KernelProvider.apply(q_index, lq) is a LINEAR q-momentum kernel operator on
# a PRE-PHASED (Nip, Ng) slab, returning v_q BEFORE the final conjugate; the
# Bloch-phase multiply and outer conjugate are pipeline glue, not part of the
# contract. vol/Ng normalization stays INSIDE the provider; exxdiv stays
# OUTSIDE (owned by get_k post-processing).
#
# Dagger law at this seam: apply(neg[q], conj(l_q)) == conj(apply(q, l_q)),
# given l_q[neg[q]] = conj(l_q[q]). Verified in test_raw_kernel_apply_dagger_law.


# WHY THESE ARE SEPARATE jit UNITS AND MUST STAY THAT WAY.
#
# Written as one traced region, XLA fuses the elementwise work into the FFT's
# loop nest and the fused kernel loses the threaded FFT path. Measured on 24
# cores at n_ip=256, mesh 57^3 (task #103, job 59921330):
#
#     fft + ifft alone                       0.239 s at 12.45 cores
#     + ONE elementwise multiply between     1.161 s at  2.46 cores
#     the whole thing fused (as shipped)     2.354 s at  1.26 cores
#     the pointwise algebra alone, no FFT    0.117 s at  3.52 cores
#
# Transforms 0.24 s plus pointwise 0.12 s, but 2.35 s together -- 6.5x the sum
# of the parts. Nothing got more expensive; only the COMBINING did. Splitting the
# stages into separate jit units forces materialisation at the boundaries and
# lets each run at its own ceiling: 7.46x at n_ip=2048 (job 59923770), output
# bit-identical (rel_err 0.0), and the win grows with panel size.
#
# `lax.optimization_barrier` does NOT work here (1.99 s vs 1.83 s): it constrains
# the optimiser without forcing a buffer boundary. Only separate compiled units do.
#
# NO MEMORY IS SAVED BY FUSING, which is what the previous docstring assumed.
# Peak RSS is identical across fused, split, and every row-chunk size tested --
# 28.79 GiB in all five cases -- because the transform already allocates
# full-size arrays internally, so the temporary the fusion "avoided" was being
# allocated anyway. Row-chunking to bound it is therefore unnecessary AND slower
# (1.8-2.1x), since it starves the transform of the batch it parallelises over.
#
# If you re-fuse these for readability, you will silently give back ~7x.


# No grid_mesh argument: these act on an already-reshaped 4-D array, so the mesh
# is carried by its shape, which jit already keys on.
@jax.jit
def _fft_fwd(lq_mesh):
    return jnp.fft.fftn(lq_mesh, axes=(1, 2, 3))


@jax.jit
def _fft_inv(wq_mesh):
    return jnp.fft.ifftn(wq_mesh, axes=(1, 2, 3))


@partial(jax.jit, static_argnames=("grid_mesh",))
def _apply_coulg(wq_mesh, coulG_scaled, grid_mesh):
    vq = jnp.asarray(coulG_scaled, dtype=wq_mesh.dtype).reshape(grid_mesh)
    return wq_mesh * vq[None, :, :, :]


def _raw_kernel_apply_core(lq, coulG_scaled, grid_mesh):
    """Core of the "raw" provider: v_q = IFFT(coulG_scaled * FFT(lq)).

    No validation (host wrapper's job), no phase multiply, no outer conjugate.

    NOT one jit: see the note above. The stages are separate compiled units on
    purpose, and re-fusing them costs ~7x.
    """
    n_ip = lq.shape[0]
    lq_mesh = lq.reshape((n_ip,) + grid_mesh)
    wq_mesh = _fft_fwd(lq_mesh)
    vq_mesh = _apply_coulg(wq_mesh, coulG_scaled, grid_mesh)
    rq_mesh = _fft_inv(vq_mesh)
    return rq_mesh.reshape(n_ip, -1)


@partial(jax.jit, static_argnames=("grid_mesh",))
def _rf_pre(eta, gphase, grid_mesh):
    return (eta * gphase[None, :]).reshape((eta.shape[0],) + grid_mesh)


@jax.jit
def _rf_post(rq_mesh, gphase):
    # n_ip comes from the array's own leading axis, not an argument: a traced
    # int cannot be a reshape dimension.
    return jnp.conj(rq_mesh.reshape(rq_mesh.shape[0], -1)) * gphase[None, :]


def _raw_right_factor_core(eta, coulG_scaled, gphase, grid_mesh):
    """``right_q(eta) = conj(apply(q, eta*g)) * g``, as SEPARATE jit units.

    NOT one traced region -- see the note above ``_fft_fwd``. The previous
    revision fused the whole sequence deliberately, to avoid materialising
    ``lq = eta*g`` as a panel-sized array. **That saving does not exist**: peak
    RSS is identical fused and split (28.79 GiB in every variant measured,
    job 59923770), because the transform allocates full-size arrays regardless.
    The fusion bought no memory and cost 6.5x in wall.

    Numerically this is NOT required to match the fused composition bitwise --
    XLA may reassociate either way. In practice the split path measured
    ``rel_err = 0.0`` against the fused one at every shape tested; the
    pre-registered gate still holds it to a c128 bound against the generic
    ``apply``-composed path rather than to bitwise identity.
    """
    lq_mesh = _rf_pre(eta, gphase, grid_mesh)
    wq_mesh = _fft_fwd(lq_mesh)
    vq_mesh = _apply_coulg(wq_mesh, coulG_scaled, grid_mesh)
    rq_mesh = _fft_inv(vq_mesh)
    return _rf_post(rq_mesh, gphase)


def raw_kernel_apply(lq, *, cell, q_kpt, grid_mesh):
    """Host-side wrapper: validate, compute coulG(q)*vol/Ng via pyscf (not
    jittable), dispatch to the jitted core.

    Args:
        lq: (Nip, Ng) complex128, ALREADY Bloch-phase-corrected.
        grid_mesh: (3,) positive ints; prod must equal Ng.

    Returns:
        v_q: (Nip, Ng) complex128 jax array, BEFORE the outer conjugate.
    """
    from pyscf.pbc import tools as pbctools

    lq_np = np.asarray(lq)
    if lq_np.ndim != 2:
        raise ValueError(f"lq must be 2-D (Nip, Ng), got shape {lq_np.shape}.")
    n_ip, n_grid = lq_np.shape

    grid_mesh_t = _require_mesh3(grid_mesh)
    if int(np.prod(grid_mesh_t)) != n_grid:
        raise ValueError(f"prod(grid_mesh)={int(np.prod(grid_mesh_t))} != lq's grid size {n_grid}.")

    q_kpt_np = np.asarray(q_kpt, dtype=np.float64)
    if q_kpt_np.shape != (3,):
        raise ValueError(f"q_kpt must have shape (3,), got {q_kpt_np.shape}.")

    Gv = cell.get_Gv(list(grid_mesh_t))
    coulG = pbctools.get_coulG(cell, k=q_kpt_np, exx=False, Gv=Gv, mesh=list(grid_mesh_t))
    coulG_scaled = np.asarray(coulG, dtype=np.float64) * (cell.vol / n_grid)

    lq_jnp = jnp.asarray(lq_np, dtype=jnp.complex128)
    if lq_jnp.dtype != jnp.complex128:
        logger.warning(
            f"raw_kernel_apply: resolved dtype is {lq_jnp.dtype}, not complex128 -- JAX "
            f"defaults to complex64 SILENTLY unless the caller has enabled "
            f"jax.config.update('jax_enable_x64', True). This device path is defined only "
            f"at the c128 parity tier (design v2.1 section 1); verify x64 is enabled "
            f"before trusting production numbers from this path."
        )
    return _raw_kernel_apply_core(lq_jnp, jnp.asarray(coulG_scaled), grid_mesh_t)


def precompute_coulG_all_q(cell, canonical_kpts, grid_mesh):
    """Precompute coulG(q)*vol/Ng for every q at once (host-only pyscf
    calls are not jittable; the result is a per-q constant that can then
    be threaded into a jitted core as a traced argument).

    Returns:
        coulG_all: (Nk, Ng) float64 jax array.
    """
    from pyscf.pbc import tools as pbctools

    canonical_kpts_np = np.asarray(canonical_kpts, dtype=np.float64)
    if canonical_kpts_np.ndim != 2 or canonical_kpts_np.shape[1] != 3:
        raise ValueError(
            f"canonical_kpts must have shape (Nk,3), got {canonical_kpts_np.shape}."
        )
    grid_mesh_t = _require_mesh3(grid_mesh)

    n_grid = int(np.prod(grid_mesh_t))
    Gv = cell.get_Gv(list(grid_mesh_t))
    n_kpts = canonical_kpts_np.shape[0]
    coulG_all = np.empty((n_kpts, n_grid), dtype=np.float64)
    for q in range(n_kpts):
        coulG = pbctools.get_coulG(
            cell, k=canonical_kpts_np[q], exx=False, Gv=Gv, mesh=list(grid_mesh_t)
        )
        coulG_all[q] = np.asarray(coulG, dtype=np.float64) * (cell.vol / n_grid)
    return jnp.asarray(coulG_all)


@partial(jax.jit, static_argnames=("grid_mesh", "self_paired", "retention_mode"))
def _fused_apply_kernel_and_solve_core(
    Pi_q, eta_q, phase_q, coulG_scaled_q, grid_mesh, rtol, self_paired,
    retention_mode="single", n_retained_pin=-1
):
    """Fully fused single-jax.jit per-q hot path: phase-multiply -> raw
    kernel apply -> conjugate -> ZGEMM -> Hermitian sandwich solve, one XLA
    graph with no host round trips. Reachable only via a provider exposing
    fused_apply_and_solve. See design doc §6. n_retained_pin is a traced
    scalar forwarded to the solve core (K >= 0 pins, K < 0 disables)."""
    n_grid = eta_q.shape[1]
    lq = eta_q * phase_q[None, :]
    v_q = _raw_kernel_apply_core(lq, coulG_scaled_q, grid_mesh)
    rq = jnp.conj(v_q)
    kern_q = (lq @ rq.T) / jnp.sqrt(n_grid)
    if self_paired:
        kern_q = kern_q.real.astype(jnp.complex128)

    from pytc.df.solvers import _hermitian_sandwich_solve_core

    (
        W, n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
        v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
        s_first_discarded,
    ) = _hermitian_sandwich_solve_core(Pi_q, kern_q, rtol, retention_mode, n_retained_pin)

    return (
        W, kern_q, n_retained, s_max, s_min_retained,
        pi_anti_hermitian_residual, v_anti_hermitian_residual,
        retained_solve_residual, truncation_residual, s_first_discarded,
    )


@dataclasses.dataclass(frozen=True)
class RawKernelProvider:
    """The "raw" (bare 4pi/G^2, exx=False) KernelProvider (design doc §7):
    apply(q_index, lq) -> v_q plus provenance(). q_index resolves the
    absolute k-vector from canonical_kpts internally.

    ``is_self_adjoint_per_q`` is True because the bare kernel is real and
    diagonal in G, so lq V lq^dagger is Hermitian at fixed q. Consumers that
    mirror a transposed block rely on this; a provider without the attribute is
    treated as not self-adjoint.

    fused_apply_and_solve is an OPTIONAL fast-path hook that
    apply_kernel_and_solve_device prefers when present; providers without
    it fall back to the eager per-stage path.

    Args:
        canonical_kpts: (Nk, 3) float64, e.g. KptsMesh.canonical_kpts.
        grid_mesh: (3,) positive ints, real-space integration mesh.
    """

    is_self_adjoint_per_q = True
    cell: object
    canonical_kpts: object
    grid_mesh: tuple

    def __post_init__(self):
        canonical_kpts = np.asarray(self.canonical_kpts, dtype=np.float64)
        if canonical_kpts.ndim != 2 or canonical_kpts.shape[1] != 3:
            raise ValueError(
                f"canonical_kpts must have shape (Nk,3), got {canonical_kpts.shape}."
            )
        grid_mesh = _require_mesh3(self.grid_mesh)
        object.__setattr__(self, "canonical_kpts", canonical_kpts)
        object.__setattr__(self, "grid_mesh", grid_mesh)
        object.__setattr__(
            self, "coulG_all", precompute_coulG_all_q(self.cell, canonical_kpts, grid_mesh)
        )

    def apply(self, q_index, lq):
        n_kpts = self.canonical_kpts.shape[0]
        if not (0 <= q_index < n_kpts):
            raise ValueError(f"q_index={q_index} out of range for {n_kpts} k-points.")
        return raw_kernel_apply(
            lq, cell=self.cell, q_kpt=self.canonical_kpts[q_index], grid_mesh=self.grid_mesh
        )

    def apply_right_factor(self, q_index, eta_q, gphase):
        """Fused ``conj(apply(q, eta*g)) * g`` — the optional hook _right_factor prefers.

        Uses the per-q ``coulG_all`` precomputed in __post_init__, so unlike
        ``apply`` this needs no host pyscf call and the whole sequence stays in
        one traced region. Providers without this method fall back to the generic
        composition, which is the reference the fused path is gated against.
        """
        n_kpts = self.canonical_kpts.shape[0]
        if not (0 <= q_index < n_kpts):
            raise ValueError(f"q_index={q_index} out of range for {n_kpts} k-points.")
        eta_j = jnp.asarray(eta_q, dtype=jnp.complex128)
        if eta_j.ndim != 2:
            raise ValueError(f"eta_q must be 2-D (Nip, Ng), got shape {eta_j.shape}.")
        gphase_j = jnp.asarray(gphase, dtype=jnp.complex128)
        if gphase_j.shape != (eta_j.shape[1],):
            raise ValueError(
                f"gphase must have shape (Ng,)={(eta_j.shape[1],)}, got {gphase_j.shape}."
            )
        return _raw_right_factor_core(
            eta_j, self.coulG_all[q_index], gphase_j, self.grid_mesh
        )

    def fused_apply_and_solve(
        self, q_index, Pi_q, eta_q, phase_q, rtol, self_paired, retention_mode="single",
        n_retained_pin=-1
    ):
        n_kpts = self.canonical_kpts.shape[0]
        if not (0 <= q_index < n_kpts):
            raise ValueError(f"q_index={q_index} out of range for {n_kpts} k-points.")
        return _fused_apply_kernel_and_solve_core(
            Pi_q, eta_q, phase_q, self.coulG_all[q_index], self.grid_mesh, rtol, self_paired,
            retention_mode, n_retained_pin,
        )

    def provenance(self):
        return {
            "kernel_name": "raw",
            "kernel_version": 1,
            "g0_convention": "pyscf_get_coulG_exx_false",
            "grid_mesh": self.grid_mesh,
            "normalization": "vol_over_ng_inside_provider",
            "exxdiv": "owned_by_get_k_postprocessing_not_this_provider",
        }


@jax.jit
def _precompute_phase_all_q_core(grid_coords, canonical_kpts):
    """Jitted batched core: per-q Bloch phase exp(-1j * grid_coords @ q_kpt)
    for every q from a single grid_coords upload."""
    return jnp.exp(-1j * (grid_coords @ canonical_kpts.T)).T


def precompute_phase_all_q(grid_coords, canonical_kpts):
    """Compute the per-q Bloch phase for every q in one batched jitted
    call; slice per-q into apply_kernel_and_solve_device's phase_q.

    Returns:
        phase_all: (Nk, Ng) complex128 jax array.
    """
    grid_coords_np = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords_np.ndim != 2 or grid_coords_np.shape[1] != 3:
        raise ValueError(f"grid_coords must have shape (Ng,3), got {grid_coords_np.shape}.")
    canonical_kpts_np = np.asarray(canonical_kpts, dtype=np.float64)
    if canonical_kpts_np.ndim != 2 or canonical_kpts_np.shape[1] != 3:
        raise ValueError(
            f"canonical_kpts must have shape (Nk,3), got {canonical_kpts_np.shape}."
        )
    return _precompute_phase_all_q_core(
        jnp.asarray(grid_coords_np), jnp.asarray(canonical_kpts_np)
    )


def apply_kernel_and_solve_device(
    provider, q_index, Pi_q, eta_q, *, grid_coords=None, phase_q=None, rtol=None,
    retained_solve_residual_gate=1e-10, self_paired=False, retention_mode="single",
    kern_blocking=None, n_retained_pin=None, kern_q=None, jitter_rcond=None,
):
    """S4 pipeline glue, device-resident, provider-agnostic: phase multiply
    -> provider.apply -> conjugate -> ZGEMM -> device Hermitian sandwich
    solve. Matches apply_raw_kernel_and_solve's math for RawKernelProvider.
    See design doc §6.

    Args:
        provider: object exposing .apply(q_index, lq) -> v_q (Nip, Ng).
        Pi_q: (Nip, Nip) complex128 metric.
        eta_q: (Nip, Ng) complex128 RHS, NOT yet phase-corrected.
        grid_coords: (Ng, 3); required only when phase_q is not given.
        phase_q: (Ng,) complex128 precomputed
            exp(-1j * grid_coords @ canonical_kpts[q_index]); callers
            looping over q should precompute via precompute_phase_all_q.
            Exactly one of grid_coords/phase_q must be given.
        rtol: forwarded to the device sandwich solve; None means the 1e-4
            default, and must be None when n_retained_pin is given.
        retained_solve_residual_gate: HARD host-side gate on
            solve_info["retained_solve_residual"]; also hard-fails on
            n_retained == 0 (the jitted solve cannot raise on traced
            values, so degradation is turned into an error here).
        self_paired: True when neg[q]==q; Pi_q.real and kern_q.real are
            taken before the solve (see apply_raw_kernel_and_solve).
        retention_mode: "single" (default) or "pairwise" -- forwarded to
            hermitian_sandwich_solve_device / the fused core. See
            hermitian_sandwich_solve's docstring for the two modes.
        n_retained_pin: optional int K in [1, Nip], "single" mode only;
            retain exactly the K largest-eigenvalue modes of Pi_q
            regardless of rtol (fixed effective rank; mutually exclusive
            with rtol).

    Returns:
        (W_q, kern_q, solve_info): W_q (Nip, Nip) complex128 jax array;
        kern_q the raw contracted kernel; solve_info the solve info dict.
    """
    from pytc.df.solvers import _solve_info_from_core_output, hermitian_sandwich_solve_device

    if kern_q is not None:
        # eta is absent by construction on this path; both dimensions come from
        # the kern and the per-q phase vector instead.
        n_ip = int(np.asarray(kern_q).shape[0])
        n_grid = int(np.asarray(phase_q).shape[0])
    else:
        n_ip, n_grid = eta_q.shape

    if n_retained_pin is not None:
        if retention_mode != "single":
            raise ValueError("n_retained_pin is only supported for retention_mode='single'.")
        if rtol is not None:
            raise ValueError(
                "n_retained_pin is mutually exclusive with rtol; leave rtol=None."
            )
        if isinstance(n_retained_pin, bool) or not isinstance(
            n_retained_pin, (int, np.integer)
        ):
            raise ValueError(
                f"n_retained_pin must be an integer in [1, {n_ip}], got {n_retained_pin!r}."
            )
        n_retained_pin = int(n_retained_pin)
        if not 1 <= n_retained_pin <= n_ip:
            raise ValueError(f"n_retained_pin must be in [1, {n_ip}], got {n_retained_pin}.")
    rtol_eff = 1e-4 if rtol is None else rtol
    # cholesky_jitter regularizes instead of truncating and refuses rtol;
    # passing the defaulted rtol_eff down would trip that refusal.
    _rtol_arg = (None if (n_retained_pin is not None
                 or retention_mode == "cholesky_jitter") else rtol_eff)

    if phase_q is None and grid_coords is None:
        raise ValueError("apply_kernel_and_solve_device: give one of grid_coords/phase_q.")
    if phase_q is not None:
        phase = jnp.asarray(phase_q, dtype=jnp.complex128)
        if phase.shape != (n_grid,):
            raise ValueError(
                f"phase_q must have shape ({n_grid},) matching eta_q's grid axis, "
                f"got {phase.shape}."
            )
    else:
        q_kpt = provider.canonical_kpts[q_index]
        grid_coords_np = np.asarray(grid_coords, dtype=np.float64)
        if grid_coords_np.shape != (n_grid, 3):
            raise ValueError(
                f"grid_coords must have shape ({n_grid},3) matching eta_q's grid axis, "
                f"got {grid_coords_np.shape}."
            )
        phase = jnp.exp(-1j * (jnp.asarray(grid_coords_np) @ jnp.asarray(q_kpt)))

    eta_q_jnp = (None if kern_q is not None
                 else jnp.asarray(eta_q, dtype=jnp.complex128))
    Pi_q_jnp = jnp.asarray(Pi_q, dtype=jnp.complex128)
    # Fail closed at the boundary common to BOTH the fused and eager solve
    # paths: with jax_enable_x64 off, JAX silently downcasts the
    # complex128 casts above to complex64, the device solve runs in single
    # precision (~1e-5 accuracy), and W is silently corrupted -- surfacing only
    # as an opaque trip of the 1e-10 machine-tier retained-solve gate below.
    # The fused path never reaches hermitian_sandwich_solve_device's guard, so
    # the check must live here.
    if ((eta_q_jnp is not None and eta_q_jnp.dtype != jnp.complex128)
            or Pi_q_jnp.dtype != jnp.complex128):
        raise ValueError(
            f"apply_kernel_and_solve_device: q_index={q_index} resolved to dtype "
            f"{Pi_q_jnp.dtype}, not complex128 -- jax_enable_x64 is off, so JAX "
            f"silently downcast to complex64 and the device solve would run in single "
            f"precision (~1e-5 accuracy), corrupting W and tripping the 1e-10 "
            f"machine-tier gate downstream. Call jax.config.update('jax_enable_x64', "
            f"True) before building (design v2.1 section 1 fixes c128 as the only tier "
            f"with defined 1e-6-class gates)."
        )
    if self_paired:
        Pi_q_jnp = Pi_q_jnp.real.astype(jnp.complex128)

    # Providers with a fused fast path run the whole chain as one jax.jit
    # graph; others fall back to the eager per-stage path below.
    if kern_q is not None:
        # Precomputed by a caller that already formed kern without eta (the
        # panel-blocked builder). Routed through this same solve deliberately:
        # a second solve path would be where the residual gate, retention mode
        # and pin quietly diverge.
        kern_q = jnp.asarray(kern_q, dtype=jnp.complex128)
        if self_paired:
            kern_q = kern_q.real.astype(jnp.complex128)
        W_q_unscaled, solve_info = hermitian_sandwich_solve_device(
            Pi_q_jnp, kern_q,
            rtol=_rtol_arg,
            retention_mode=retention_mode,
            n_retained_pin=n_retained_pin, jitter_rcond=jitter_rcond,
        )
    elif kern_blocking is not None:
        # Bypasses the fused graph, which assumes a resident (Nip, Ng).
        kern_q = jnp.asarray(
            build_kern_q_blocked(
                provider, q_index, np.asarray(eta_q), np.asarray(phase),
                self_paired=self_paired, **kern_blocking),
            dtype=jnp.complex128)
        W_q_unscaled, solve_info = hermitian_sandwich_solve_device(
            Pi_q_jnp, kern_q,
            rtol=_rtol_arg,
            retention_mode=retention_mode, n_retained_pin=n_retained_pin,
            jitter_rcond=jitter_rcond)
    elif (retention_mode != "cholesky_jitter"
          # The provider's fused hook takes no jitter, so this mode uses the
          # unfused path rather than silently solving at the wrong jitter.
          and (fused := getattr(provider, "fused_apply_and_solve", None)) is not None):
        (
            W_q_unscaled, kern_q, n_retained, s_max, s_min_retained,
            pi_anti_hermitian_residual, v_anti_hermitian_residual,
            retained_solve_residual, truncation_residual, s_first_discarded,
        ) = fused(q_index, Pi_q_jnp, eta_q_jnp, phase, rtol_eff, self_paired, retention_mode,
                  n_retained_pin if n_retained_pin is not None else -1)

        solve_info = _solve_info_from_core_output(
            n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
            v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
            s_first_discarded, n_ip, W_q_unscaled.dtype,
            None if n_retained_pin is not None else rtol_eff,
            caller="apply_kernel_and_solve_device[fused]",
            retention_mode=retention_mode,
            n_retained_pin=n_retained_pin,
        )
    else:
        lq = eta_q_jnp * phase[None, :]

        v_q = provider.apply(q_index, lq)
        rq = jnp.conj(v_q)

        kern_q = (lq @ rq.T) / jnp.sqrt(n_grid)
        if self_paired:
            kern_q = kern_q.real.astype(jnp.complex128)

        W_q_unscaled, solve_info = hermitian_sandwich_solve_device(
            Pi_q_jnp, kern_q,
            rtol=_rtol_arg,
            retention_mode=retention_mode, n_retained_pin=n_retained_pin,
            jitter_rcond=jitter_rcond,
        )

    # Host-side gate: the jitted solve cannot raise on a traced value, so
    # degeneracy becomes a precise, q-indexed error here (not a silent W_q=0).
    if solve_info["n_retained"] == 0:
        raise ValueError(
            f"apply_kernel_and_solve_device: q_index={q_index} retained ZERO modes of "
            f"Pi_q in the device sandwich solve (Pi_q is non-PSD, the zero matrix, or "
            f"rtol={rtol_eff} is too large) -- W_q would be silently zero; refusing to "
            f"proceed. Validate Pi_q against the NumPy oracle (hermitian_sandwich_solve) "
            f"for a precise diagnosis."
        )
    # The gate is calibrated on the truncating branch's arithmetic; this mode
    # regularizes instead, so its residual is legitimately large and reported
    # rather than gated (owner decision 2026-08-10).
    if (retention_mode != "cholesky_jitter"
            and solve_info["retained_solve_residual"] > retained_solve_residual_gate):
        raise ValueError(
            f"apply_kernel_and_solve_device: q_index={q_index} retained-space solve "
            f"residual {solve_info['retained_solve_residual']:.3e} exceeds the hard "
            f"machine-tier gate {retained_solve_residual_gate:.1e} (design v2.1 section "
            f"5) -- this is a numerical sanity check on the eigendecomposition/solve "
            f"arithmetic itself, not the (separately reported, ungated here) truncation "
            f"residual; something is wrong with this q's Pi_q/kern_q inputs or dtype."
        )

    W_q = jnp.sqrt(n_grid) * W_q_unscaled
    return W_q, kern_q, solve_info


def build_kern_q_blocked(provider, q_index, eta_q, phase, *, staging_root,
                         row_block=2048, grid_chunk=4096, self_paired=False):
    """kern_q without holding a full (Nip, Ng) array.

    Pass 1 row-blocks the transform (the FFT's axis 0 is a batch axis) and stages
    rq; pass 2 accumulates kern_q over grid chunks. Peak is
    max(2*row_block*Ng, 2*Nip*grid_chunk) + Nip^2. Design and costs:
    docs/isdf-periodic/task46_phaseB_solve_blocking_spec.md.
    """
    n_ip, n_grid = int(eta_q.shape[0]), int(eta_q.shape[1])
    phase = np.asarray(phase, dtype=np.complex128)
    itemsize = np.dtype(np.complex128).itemsize
    predicted = n_ip * n_grid * itemsize
    stat = os.statvfs(staging_root)
    free = stat.f_bavail * stat.f_frsize
    if free < predicted * 1.25:
        raise OSError(
            f"rq staging REFUSED for q={int(q_index)}: {staging_root} has "
            f"{free / 2**30:.1f} GiB free, needs {predicted * 1.25 / 2**30:.1f} GiB."
        )

    rq_path = os.path.join(
        staging_root, f"isdf_rq_q{int(q_index)}_{os.getpid()}.dat")
    rq = None
    try:
        rq = np.memmap(rq_path, dtype=np.complex128, mode="w+",
                       shape=(n_ip, n_grid))
        for r0 in range(0, n_ip, int(row_block)):
            r1 = min(r0 + int(row_block), n_ip)
            lq_rows = np.asarray(eta_q[r0:r1], dtype=np.complex128) * phase[None, :]
            rq[r0:r1] = np.conj(np.asarray(provider.apply(q_index, lq_rows)))
        rq.flush()

        kern = np.zeros((n_ip, n_ip), dtype=np.complex128)
        for g0 in range(0, n_grid, int(grid_chunk)):
            g1 = min(g0 + int(grid_chunk), n_grid)
            lq_c = (np.asarray(eta_q[:, g0:g1], dtype=np.complex128)
                    * phase[None, g0:g1])
            kern += lq_c @ np.asarray(rq[:, g0:g1]).T
        kern /= np.sqrt(n_grid)
        if self_paired:
            kern = kern.real.astype(np.complex128)
        return kern
    finally:
        rq = None
        gc.collect()
        try:
            os.unlink(rq_path)
        except FileNotFoundError:
            pass


def _require_mesh3(mesh, name="grid_mesh"):
    """Coerce to a 3-tuple of positive ints, or raise."""
    t = tuple(int(x) for x in mesh)
    if len(t) != 3 or any(m <= 0 for m in t):
        raise ValueError(f"{name} must be 3 positive ints, got {t}.")
    return t


def _normalize_n_retained_pin(n_retained_pin, n_kpts):
    """Broadcast a pin to one entry per q, or None when unpinned."""
    if n_retained_pin is None:
        return None
    if isinstance(n_retained_pin, (list, tuple, np.ndarray)):
        if len(n_retained_pin) != n_kpts:
            raise ValueError(
                f"n_retained_pin sequence must have length {n_kpts}, got "
                f"{len(n_retained_pin)}."
            )
        return list(n_retained_pin)
    return [n_retained_pin] * n_kpts


def build_coul_kpt_device(provider, Pi, eta, grid_coords, mesh_obj, *, rtol=None,
                          kern=None,
                           retained_solve_residual_gate=1e-10, retention_mode="single",
                           kern_blocking=None, n_retained_pin=None, jitter_rcond=None):
    """S4 orchestration: build coul_kpt (Nk, Nip, Nip) with one
    apply_kernel_and_solve_device call per unique {q, neg[q]} pair; the
    partner is set by exact conjugation (W[neg[q]] = conj(W[q]),
    kern[neg[q]] = conj(kern[q])). See design doc §6; verified against
    independent neg[q] builds in
    test_build_coul_kpt_device_conjugate_shortcut_matches_independent_build.

    Args:
        provider: KernelProvider built against mesh_obj.canonical_kpts.
        Pi: (Nk, Nip, Nip) complex128.
        eta: (Nk, Nip, Ng) complex128.
        grid_coords: (Ng, 3).
        mesh_obj: KptsMesh (uses .neg, .n_kpts).
        rtol: forwarded to apply_kernel_and_solve_device; None means the
            1e-4 default, and must be None when n_retained_pin is given.
        n_retained_pin: optional int K or length-Nk sequence of ints,
            forwarded per-q to apply_kernel_and_solve_device; a q solved
            via the conjugate shortcut inherits its partner's pin.

    Returns:
        (coul_kpt, kern_kpt, infos, n_pipeline_calls): (Nk, Nip, Nip) jax
        arrays; length-Nk info list (a conjugated q shares its partner's
        dict); number of q's that actually ran the pipeline.
    """
    n_kpts = mesh_obj.n_kpts
    Pi = np.asarray(Pi)
    # eta may be a StagedEta view over a grid-major file, which materialises
    # one q-slab per __getitem__; asarray would collapse it to a 0-d object.
    if kern is not None:
        # kern was formed without eta (panel-blocked build); eta is then unused
        # and validating it would demand an object the caller deliberately
        # never materialised.
        kern = np.asarray(kern)
        if kern.shape[0] != n_kpts:
            raise ValueError(
                f"kern.shape[0]={kern.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")
    elif not isinstance(eta, StagedEta):
        eta = np.asarray(eta)
    if Pi.shape[0] != n_kpts:
        raise ValueError(f"Pi.shape[0]={Pi.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")
    if kern is None and eta.shape[0] != n_kpts:
        raise ValueError(f"eta.shape[0]={eta.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")

    pin_per_q = _normalize_n_retained_pin(n_retained_pin, n_kpts)

    # Precompute every q's Bloch phase once: per-q constant within one build.
    phase_all = precompute_phase_all_q(grid_coords, mesh_obj.canonical_kpts)

    neg = mesh_obj.neg
    coul_kpt = [None] * n_kpts
    kern_kpt = [None] * n_kpts
    infos = [None] * n_kpts
    done = [False] * n_kpts
    n_pipeline_calls = 0

    for q in range(n_kpts):
        if done[q]:
            continue
        nq = int(neg[q])
        W_q, kern_q, info_q = apply_kernel_and_solve_device(
            provider, q, Pi[q],
            None if kern is not None else eta[q],
            kern_q=None if kern is None else kern[q],
            phase_q=phase_all[q], rtol=rtol,
            retained_solve_residual_gate=retained_solve_residual_gate,
            jitter_rcond=jitter_rcond,
            self_paired=(nq == q), retention_mode=retention_mode,
            kern_blocking=kern_blocking,
            n_retained_pin=None if pin_per_q is None else pin_per_q[q],
        )
        coul_kpt[q] = W_q
        kern_kpt[q] = kern_q
        infos[q] = info_q
        done[q] = True
        n_pipeline_calls += 1

        if nq != q and not done[nq]:
            coul_kpt[nq] = jnp.conj(W_q)
            kern_kpt[nq] = jnp.conj(kern_q)
            infos[nq] = info_q
            done[nq] = True

    return jnp.stack(coul_kpt, axis=0), jnp.stack(kern_kpt, axis=0), infos, n_pipeline_calls


def build_coul_kpt_host(cell, Pi, eta, grid_coords, mesh_obj, *, rtol=None,
                        retention_mode="single", jitter_rcond=None,
                        n_retained_pin=None):
    """Host reference mirror of build_coul_kpt_device.

    Exists so an accuracy question can be answered without first porting a solver
    to the device path: the device path is production, but nothing in this
    assembly is device-specific. Reference-grade, NOT a performance path -- it
    runs the NumPy per-q solve and materializes kern_q densely.

    The conjugate shortcut, the self_paired flag at nq == q, and the per-q
    ordering are mirrored from the device loop deliberately; if they drift apart
    the two paths stop being comparable, which is the whole point of the mirror.

    Returns:
        (coul_kpt, kern_kpt, infos, n_pipeline_calls): matching
        build_coul_kpt_device's return signature.
    """
    n_kpts = mesh_obj.n_kpts
    neg = mesh_obj.neg
    pin_per_q = _normalize_n_retained_pin(n_retained_pin, n_kpts)

    coul_kpt = [None] * n_kpts
    kern_kpt = [None] * n_kpts
    infos = [None] * n_kpts
    done = [False] * n_kpts
    # Counted, not assumed: the conjugate shortcut solves one q per {q, neg[q]}
    # pair, so a hard-coded n_kpts overstates the work at any mesh with pairs.
    n_pipeline_calls = 0

    for q in range(n_kpts):
        if done[q]:
            continue
        nq = int(neg[q])
        W_q, kern_q, info_q = apply_raw_kernel_and_solve(
            Pi[q], eta[q], cell=cell, q_kpt=mesh_obj.canonical_kpts[q],
            grid_coords=grid_coords, grid_mesh=cell.mesh, rtol=rtol,
            self_paired=(nq == q), retention_mode=retention_mode,
            jitter_rcond=jitter_rcond,
            n_retained_pin=None if pin_per_q is None else pin_per_q[q],
        )
        coul_kpt[q] = W_q
        kern_kpt[q] = kern_q
        infos[q] = info_q
        done[q] = True
        n_pipeline_calls += 1

        if nq != q and not done[nq]:
            coul_kpt[nq] = np.conj(W_q)
            kern_kpt[nq] = np.conj(kern_q)
            infos[nq] = info_q
            done[nq] = True

    return (np.stack(coul_kpt, axis=0), np.stack(kern_kpt, axis=0), infos,
            n_pipeline_calls)


# ---------------------------------------------------------------------------
# S1/S2 streaming (design v2.1 section 6): AO evaluation and the periodic
# pivot-selection metric oracle, both grid-block-streamed so host memory for
# either stays bounded by one block regardless of the full grid size Ng.


def stream_ao_blocks(cell, kpts, grid_coords, block_size, *, stats=None):
    """S1: stream AO values at kpts over grid_coords in blocks of
    block_size grid points; host memory stays bounded by one block.

    Yields:
        (g0, g1, ao_block): grid-index bounds [g0,g1) and the
        (Nk, g1-g0, Nao) complex128 block.
    """
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError(f"grid_coords must have shape (Ng,3), got {grid_coords.shape}.")
    n_grid = grid_coords.shape[0]
    if n_grid == 0:
        raise ValueError("grid_coords must be nonempty.")

    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    kpts_list = list(np.asarray(kpts, dtype=np.float64))

    for g0 in range(0, n_grid, block_size):
        g1 = min(g0 + block_size, n_grid)
        ao_block = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[g0:g1], kpts=kpts_list), dtype=np.complex128,
        )
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + (g1 - g0)
        yield g0, g1, ao_block


def build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """S2 periodic pivot-selection metric oracle (design doc §3): a
    (diag, col_eval) pair for the reference-cell pair-density Gram matrix
    M[r,r'] = |sum_{k,mu} conj(AO_k(r,mu)) AO_k(r',mu)|^2 / Nk, never
    materialized. Each col_eval(j) costs a full streamed AO-grid sweep,
    so selecting `rank` pivots costs `rank` sweeps -- callers should
    account for this traffic.

    Returns:
        (diag, col_eval): diag (Ng,) float64; col_eval(j) -> (Ng,)
        complex128 (M is real-valued; complex128 only to match
        pivoted_cholesky_hermitian's contract).
    """
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    n_grid = grid_coords.shape[0]
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_kpts = kpts_np.shape[0]

    diag = np.empty(n_grid, dtype=np.float64)
    for g0, g1, ao_block in stream_ao_blocks(
        cell, kpts_np, grid_coords, block_size, stats=stats,
    ):
        pooled = np.sum(np.abs(ao_block) ** 2, axis=(0, 2))  # (blk,), sum_{k,mu} |AO_k(r,mu)|^2
        diag[g0:g1] = pooled ** 2 / n_kpts

    def col_eval(j):
        ao_j_block = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[j:j + 1], kpts=list(kpts_np)),
            dtype=np.complex128,
        )
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + 1
        ao_j = ao_j_block[:, 0, :]  # (Nk, Nao)

        col = np.empty(n_grid, dtype=np.complex128)
        for g0, g1, ao_block in stream_ao_blocks(
            cell, kpts_np, grid_coords, block_size, stats=stats,
        ):
            gram = np.einsum("km,krm->r", ao_j.conj(), ao_block, optimize=True)
            col[g0:g1] = (np.abs(gram) ** 2 / n_kpts).astype(np.complex128)
        return col

    return diag, col_eval


def build_periodic_batched_pivot_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """Exact streamed periodic metric with one full AO sweep per column batch."""
    diagonal, _ = build_periodic_pivot_oracle(
        cell, kpts, grid_coords, block_size, stats=stats,
    )
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_grid = len(grid_coords)
    n_kpts = len(kpts_np)

    def col_batch_eval(indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= n_grid):
            raise ValueError("indices must be a nonempty in-range integer vector.")
        pivot_ao = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[indices], kpts=list(kpts_np)),
            dtype=np.complex128,
        )
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + int(indices.size)
        columns = np.empty((n_grid, indices.size), dtype=np.complex128)
        for g0, g1, ao_block in stream_ao_blocks(
            cell, kpts_np, grid_coords, block_size, stats=stats,
        ):
            gram = np.einsum("kbm,krm->br", pivot_ao.conj(), ao_block, optimize=True)
            columns[g0:g1] = (np.abs(gram) ** 2 / n_kpts).T
        return columns

    return diagonal, col_batch_eval


def periodic_metric_column_from_ao(ao, index):
    """Return one periodic-metric column without materializing the metric."""
    ao = np.asarray(ao, dtype=np.complex128)
    if ao.ndim != 3:
        raise ValueError(f"ao must have shape (Nk,Npanel,Nao), got {ao.shape}.")
    n_kpts, n_panel, _ = ao.shape
    if n_kpts == 0 or n_panel == 0:
        raise ValueError("ao must have nonempty k and panel axes.")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer.")
    if not 0 <= index < n_panel:
        raise ValueError(f"index={index} is outside panel size {n_panel}.")
    gram = np.einsum("km,krm->r", ao[:, index, :].conj(), ao, optimize=True)
    return (np.abs(gram) ** 2 / n_kpts).astype(np.complex128)


def full_grid_candidate_identity(n_grid):
    """Return a compact identity for the complete grid candidate set."""
    if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)) or n_grid <= 0:
        raise ValueError("n_grid must be a positive integer.")
    return {"kind": "range", "start": 0, "stop": int(n_grid), "step": 1}


def explicit_candidate_identity(indices):
    """Return the persisted identity for a bounded explicit candidate panel."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array.")
    return {"kind": "explicit_indices", "indices": indices.tolist()}


def build_cached_periodic_pivot_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """Full-cache oracle carrying the same metric as the streamed path.

    Reference-only test oracle; intentionally no production caller. It is the
    independent implementation the BPC equivalence tests measure against, so it
    must not be inlined into those tests -- correlated implementations would
    weaken the gate.
    """
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_grid = len(grid_coords)
    n_kpts = len(kpts_np)
    cache = None
    for g0, g1, ao_block in stream_ao_blocks(
        cell, kpts_np, grid_coords, block_size, stats=stats,
    ):
        if cache is None:
            cache = np.empty((n_kpts, n_grid, ao_block.shape[2]), dtype=np.complex128)
        cache[:, g0:g1] = ao_block
    pooled = np.sum(np.abs(cache) ** 2, axis=(0, 2))
    diag = pooled ** 2 / n_kpts
    return diag, lambda j: periodic_metric_column_from_ao(cache, j), cache


def periodic_metric_columns_from_ao(ao, indices):
    """Exact periodic-metric columns for a bounded batch from a cached AO tensor.

    Reference-only test oracle; intentionally no production caller (see
    build_cached_periodic_pivot_oracle).
    """
    ao = np.asarray(ao, dtype=np.complex128)
    indices = np.asarray(indices, dtype=np.int64)
    if ao.ndim != 3 or indices.ndim != 1 or indices.size == 0:
        raise ValueError("ao must be (Nk,Ng,Nao) and indices must be nonempty 1-D.")
    if np.any(indices < 0) or np.any(indices >= ao.shape[1]):
        raise ValueError("indices are outside the AO grid.")
    pivot_ao = ao[:, indices, :]
    gram = np.einsum("kbm,krm->br", pivot_ao.conj(), ao, optimize=True)
    # M = |gram|^2/Nk is real, nonnegative; return float64 (not the historical
    # interface-convenience complex128) so the pivoted-Cholesky factor L it feeds
    # is stored real.
    return (np.abs(gram) ** 2 / ao.shape[0]).T.astype(np.float64)


def build_cached_periodic_bpc_gemm_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """Opt-in contiguous feature cache for BPC's threaded candidate GEMM."""
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_grid = len(grid_coords)
    n_kpts = len(kpts_np)
    features = None
    for g0, g1, ao_block in stream_ao_blocks(cell, kpts_np, grid_coords, block_size, stats=stats):
        if features is None:
            features = np.empty((n_grid, n_kpts * ao_block.shape[2]), dtype=np.complex128)
        features[g0:g1] = ao_block.transpose(1, 0, 2).reshape(g1 - g0, -1)
    pooled = np.sum(np.abs(features) ** 2, axis=1)
    diag = pooled ** 2 / n_kpts

    def col_batch_eval(indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= n_grid):
            raise ValueError("indices must be a nonempty in-range integer vector.")
        gram = features[indices].conj() @ features.T
        # real M -> float64 columns so the BPC factor L is stored real (real-f64-L).
        return (np.abs(gram) ** 2 / n_kpts).T.astype(np.float64)

    return diag, col_batch_eval, features


# ---------------------------------------------------------------------------
# Staging-policy layer (design doc §6): predicted-byte-model-driven selection
# among ram/memmap/recompute eta-store policies, plus the mechanics.
# Byte model is PREDICTED ONLY; observed fields exist in the schema as
# None/"unmeasured" so a later calibration pass backfills without a schema
# change. Real resource queries live only in query_host_resources.


def predicted_byte_model(n_kpts, n_ip, n_grid, n_ao, block_size, *, itemsize=16):
    """Closed-form predicted byte counts for one build (c128, itemsize=16);
    nothing here is measured. Returns a dict of the per-component byte
    terms plus total_predicted_bytes (eta_store + selection_traffic).
    See design doc §6."""
    for name, value in (
        ("n_kpts", n_kpts), ("n_ip", n_ip), ("n_grid", n_grid),
        ("n_ao", n_ao), ("block_size", block_size),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    ao_grid_block_bytes = n_kpts * block_size * n_ao * itemsize
    eta_store_bytes = n_kpts * n_ip * n_grid * itemsize
    double_buffer_bytes = 2 * n_ip * block_size * itemsize
    fft_workspace_bytes = 2 * n_ip * n_grid * itemsize
    pi_v_w_workspace_bytes = (n_kpts * n_ip * n_ip + n_ip * n_ip) * itemsize
    selection_traffic_bytes = n_ip * n_kpts * n_grid * n_ao * itemsize
    total_predicted_bytes = eta_store_bytes + selection_traffic_bytes

    return {
        "ao_grid_block_bytes": ao_grid_block_bytes,
        "eta_store_bytes": eta_store_bytes,
        "double_buffer_bytes": double_buffer_bytes,
        "fft_workspace_bytes": fft_workspace_bytes,
        "pi_v_w_workspace_bytes": pi_v_w_workspace_bytes,
        "selection_traffic_bytes": selection_traffic_bytes,
        "total_predicted_bytes": total_predicted_bytes,
    }


def choose_staging_policy(byte_model, *, available_host_bytes, available_disk_bytes,
                           ram_headroom_fraction=0.5, disk_headroom_fraction=0.9):
    """Select the eta-store staging policy from a predicted byte model and
    caller-supplied resource numbers (never queried internally, so this
    stays synthetic-input testable).

    Rule: "ram" if eta_store_bytes <= ram_headroom_fraction *
    available_host_bytes; else "memmap" if it fits the disk headroom;
    else "recompute".

    Returns a provenance dict; observed_peak_host_bytes/observed_status
    are always None/"unmeasured" here so a later calibration pass
    backfills the SAME schema.
    """
    eta_bytes = byte_model["eta_store_bytes"]
    if eta_bytes < 0:
        raise ValueError(f"byte_model['eta_store_bytes'] must be non-negative, got {eta_bytes}.")
    for name, value in (
        ("available_host_bytes", available_host_bytes),
        ("available_disk_bytes", available_disk_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    for name, value in (
        ("ram_headroom_fraction", ram_headroom_fraction),
        ("disk_headroom_fraction", disk_headroom_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < value <= 1.0):
            raise ValueError(f"{name} must be in (0,1], got {value!r}.")

    ram_headroom_bytes = int(ram_headroom_fraction * available_host_bytes)
    disk_headroom_bytes = int(disk_headroom_fraction * available_disk_bytes)

    if eta_bytes <= ram_headroom_bytes:
        policy = "ram"
    elif eta_bytes <= disk_headroom_bytes:
        policy = "memmap"
    else:
        policy = "recompute"

    return {
        "policy": policy,
        "eta_store_bytes": eta_bytes,
        "ram_headroom_bytes": ram_headroom_bytes,
        "disk_headroom_bytes": disk_headroom_bytes,
        "ram_headroom_fraction": ram_headroom_fraction,
        "disk_headroom_fraction": disk_headroom_fraction,
        "available_host_bytes": available_host_bytes,
        "available_disk_bytes": available_disk_bytes,
        "observed_peak_host_bytes": None,
        "observed_status": "unmeasured",
    }


def query_host_resources(scratch_dir="."):
    """The one place this module reads real host/disk resource numbers
    (psutil / shutil.disk_usage); pass the result to choose_staging_policy.

    Returns:
        (available_host_bytes, available_disk_bytes): both int.
    """
    import shutil

    import psutil

    available_host_bytes = int(psutil.virtual_memory().available)
    available_disk_bytes = int(shutil.disk_usage(scratch_dir).free)
    return available_host_bytes, available_disk_bytes


def jit_memory_analysis_smoke(jitted_fn, *args):
    """Compile jitted_fn against args (shapes/dtypes only; not executed)
    and return its jax CompiledMemoryStats as a dict, labeled by the
    actual backend (CPU stats are a structural smoke, not HBM data).
    See design doc §6."""
    backend = jax.default_backend()
    label = (
        "cpu_backend_structural_smoke_not_hbm"
        if backend == "cpu"
        else "gpu_backend_compiled_memory_stats"
    )
    stats = jitted_fn.lower(*args).compile().memory_analysis()
    return {
        "backend": backend,
        "label": label,
        "generated_code_size_in_bytes": stats.generated_code_size_in_bytes,
        "argument_size_in_bytes": stats.argument_size_in_bytes,
        "output_size_in_bytes": stats.output_size_in_bytes,
        "alias_size_in_bytes": stats.alias_size_in_bytes,
        "temp_size_in_bytes": stats.temp_size_in_bytes,
        "host_generated_code_size_in_bytes": stats.host_generated_code_size_in_bytes,
        "host_argument_size_in_bytes": stats.host_argument_size_in_bytes,
        "host_output_size_in_bytes": stats.host_output_size_in_bytes,
        "host_alias_size_in_bytes": stats.host_alias_size_in_bytes,
        "host_temp_size_in_bytes": stats.host_temp_size_in_bytes,
    }


def stage_eta_memmap(eta_chunks_iter, shape, memmap_path):
    """memmap staging (policy 2): write streamed eta chunks into an
    np.memmap at memmap_path without holding the full (Nk,Nip,Ng) eta in
    RAM.

    Args:
        eta_chunks_iter: iterable of (g0, g1, chunk), chunk
            (Nk,Nip,g1-g0) complex128, covering [0,Ng) in order.
        shape: (Nk,Nip,Ng).

    Returns:
        np.memmap, dtype complex128, flushed to disk.
    """
    shape_t = tuple(int(x) for x in shape)
    if len(shape_t) != 3 or any(s <= 0 for s in shape_t):
        raise ValueError(f"shape must be 3 positive ints, got {shape_t}.")

    mm = np.memmap(memmap_path, dtype=np.complex128, mode="w+", shape=shape_t)
    for g0, g1, chunk in eta_chunks_iter:
        mm[:, :, g0:g1] = chunk
    mm.flush()
    return mm


def stage_eta_recompute_tile(X, ao_block_source, phase, neg, q_slice=None):
    """recompute staging (policy 3): rebuild eta on demand via build_pi_eta,
    with ao_block_source() returning a FRESH iterable of (Nk,blk,Nao)
    blocks on every call. q_slice is applied to the leading (Nk) axis
    AFTER the full build (the complete pass still runs each call).

    Returns:
        (Pi, eta): same as build_pi_eta, optionally sliced by q_slice.
    """
    Pi, eta = build_pi_eta(X, ao_block_source(), phase, neg)
    if q_slice is not None:
        return Pi[q_slice], eta[q_slice]
    return Pi, eta


def p_blocked_peak_bytes(n_kpts, n_ip, n_grid, panel_rows, *,
                         ao_block_cols, n_ao):
    """Honest peak-byte model for build_pi_kern_p_blocked.

    Counts every resident term, not just the panels -- the omission that made an
    earlier docstring describe an algorithm that had not been written. Returns a
    dict so a caller can see which term dominates rather than a single number.

    Off-diagonal work holds TWO eta panels, so that term carries a factor 2.
    ``Pi`` and ``kern`` are each (Nk, Nip, Nip) and are resident for the whole
    call; at large Nip they dominate and no panel knob reduces them.

    NOT AN UPPER BOUND. Measured against tracemalloc at Nk=2, Nip=48, Ng=3375,
    the ratio of real peak to this model ran 0.89 / 1.23 / 1.25 at panel_rows
    48 / 16 / 8: it over-predicts for a single panel and UNDER-predicts by about
    a quarter once panels are small, because per-pair GEMM outputs and the
    provider's own temporaries are not counted. **A fail-closed preflight must
    apply a safety factor** -- 1.25, matching ``free_bytes_safety`` on the
    staging path -- rather than treating this as a bound.
    """
    c16 = 16
    panels = 2 * int(n_kpts) * int(panel_rows) * int(n_grid) * c16
    square = int(n_kpts) * int(n_ip) * int(n_ip) * c16
    terms = {
        "eta_panels": panels,
        "Pi": square,
        "kern": square,
        "grid_phases": int(n_kpts) * int(n_grid) * c16,
        # lq_i, lq_j and rq_j for one q at a time.
        "per_q_temporaries": 3 * int(panel_rows) * int(n_grid) * c16,
        "one_ao_block": int(n_kpts) * int(ao_block_cols) * int(n_ao) * c16,
    }
    terms["total"] = sum(terms.values())
    return terms


def _eta_rows_streamed(X_rows, ao_blocks, phase, neg, n_grid, *,
                       imag_tol=1e-10, on_block=None):
    """eta for a slab of pivot rows, holding one AO block at a time.

    ``build_pi_eta`` cannot be reused here: it does ``list(ao_blocks)`` to learn
    the total grid width before allocating, which materialises the entire AO
    stream. Taking ``n_grid`` from the caller removes that need, so the stream
    stays a stream -- the difference between reducing peak memory and increasing
    it. It also skips ``Pi``, which the panel loop builds once and would
    otherwise recompute and discard per panel.

    ``on_block`` is called with each block index for instrumentation; the tests
    use it to assert that only one block is ever live.
    """
    X_rows = np.asarray(X_rows)
    n_kpts, n_rows = int(X_rows.shape[0]), int(X_rows.shape[1])
    pair_convolve = _pair_convolve()
    eta = np.empty((n_kpts, n_rows, int(n_grid)), dtype=np.complex128)
    col = 0
    index = 0
    # NOT enumerate(): CPython reuses its result tuple, which keeps the previously
    # yielded block alive for one extra iteration. That retention is invisible to
    # any equivalence test and defeats the point of streaming.
    for block in ao_blocks:
        Z = pair_convolve(X_rows, np.asarray(block), phase, imag_tol=imag_tol)[neg]
        width = int(Z.shape[2])
        if col + width > int(n_grid):
            raise ValueError(
                f"AO blocks span more than n_grid={int(n_grid)} columns.")
        eta[:, :, col:col + width] = Z
        col += width
        if on_block is not None:
            on_block(index)
        index += 1
        del Z, block
    if col != int(n_grid):
        raise ValueError(
            f"AO blocks spanned {col} columns, expected n_grid={int(n_grid)}.")
    return eta


_RIGHT_FACTOR_PATH_LOGGED = False


def _right_factor(provider, q, eta_q, gphase):
    """The right factor of one panel pair: conj(apply(q, eta*g)) * g.

    ONE semantic operation, so the phase/conjugation convention lives in a single
    place rather than being duplicated across the mirror branches -- a convention
    error here would otherwise have to be made identically twice to be caught.

    The trailing phase is what lets the caller pass ``eta`` itself as the LEFT
    operand: gphase multiplies along the grid axis that both factors share, so
        (eta_i*g) @ conj(apply(eta_j*g)).T  ==  eta_i @ (conj(apply(eta_j*g))*g).T
    exactly. No ``lq_i``-shaped array is built anywhere in the common path.

    A provider may implement ``apply_right_factor(q, eta, phase)`` to fuse the
    whole sequence device-side; this is the generic fallback that composes it
    from ``apply``. The two are gated against each other rather than assumed
    equivalent: fusion is free to reassociate, so they are held to a numerical
    bound, not to bitwise identity.
    """
    # ENGAGEMENT RECEIPT. A lever that "did nothing" is indistinguishable from a
    # lever that never fired, so WHICH path ran is logged rather than inferred
    # from a timing. Logged once per process, not per pair. Yesterday a
    # monkeypatch silently targeted the wrong namespace and recorded nothing;
    # this is the cheap guard against reading that as "no effect".
    global _RIGHT_FACTOR_PATH_LOGGED
    fused = getattr(provider, "apply_right_factor", None)
    if not _RIGHT_FACTOR_PATH_LOGGED:
        logger.info("right_factor: %s path active (provider=%s)",
                    "FUSED" if fused is not None else "FALLBACK",
                    type(provider).__name__)
        _RIGHT_FACTOR_PATH_LOGGED = True
    if fused is not None:
        # DEVICE-RESIDENT (task #116). This used to end `np.asarray(...)`, which
        # pulled a whole right-factor panel to host -- 32.5 GB at 444/cc-pvtz
        # with P=6, and 65.0 GB at the P that OOM-killed job 59952219, where the
        # failing allocation was exactly one panel. The caller now contracts on
        # device and materialises only the (block_rows x block_rows) result, so
        # the panel-sized host copy disappears and the block GEMM stops being a
        # numpy/OpenBLAS island inside an otherwise-XLA loop.
        return fused(q, eta_q, gphase)
    lq = eta_q * gphase[None, :]
    rq = jnp.conj(jnp.asarray(provider.apply(q, lq)))
    rq = rq * gphase[None, :]
    return rq


@jax.jit
def _panel_block(eta_iq, rq_j, inv_sqrt_grid):
    """One panel pair's kernel block: (eta_i @ rq_j.T) * inv_sqrt(n_grid).

    Kept on device deliberately. The operands are panel-sized; the result is
    (block_rows x block_rows), roughly 0.6 GB at 444/cc-pvtz against a 32.5 GB
    operand, so materialising the OUTPUT costs ~2% of materialising an INPUT.
    """
    return (eta_iq @ rq_j.T) * inv_sqrt_grid


def build_pi_kern_p_blocked(X, ao_block_factory, phase, neg, provider,
                            grid_coords, *,
                            panel_rows, imag_tol=1e-10,
                            self_paired=None,
                            on_block=None, on_panel=None):
    """kern for every q without holding a full eta.

    AO blocks are streamed one at a time via ``_eta_rows_streamed``; the residency
    is asserted by test, not assumed.

    The interpolation-point axis is a pure batch axis from the AO evaluation
    through the Coulomb apply; only the final ``kern = lq rq^T`` couples P with
    P'. So a panel of pivot rows can be built, have the kernel applied, and be
    discarded.

    REMAINING LIMITS (independent review of f2aff84, task #66) -- still not wired
    into ``build``/``ISDFDF``:

    * **Peak is two panels, not one**, because off-diagonal work holds ``eta_i``
      and ``eta_j`` at once. ``p_blocked_peak_bytes`` models every resident term.
      **``Pi`` and ``kern`` are each (Nk, Nip, Nip) and no panel knob reduces
      them** -- at 444/k222 that pair alone is 353 GB, so the knob cannot take
      this configuration under a 300 GB target however far it is turned.
    * **``panel_rows`` is a schedule knob, not a resident cache.** The loop keeps
      one outer panel and regenerates the inner one; it does not retain c panels.
      A real cache would cost ``(c+1)b`` and update c block rows per sweep.
    * **The Hermitian mirror assumes a self-adjoint provider.** It is certified
      against ``RawKernelProvider`` only; linearity plus the q/-q dagger law does
      not imply self-adjointness within one q.
    * **``self_paired`` projects ``kern`` but not ``Pi``.** The dense solve
      projects both, so the caller must project ``Pi`` or treat this as pre-solve.

    COST: with P panels the schedule performs P(P+1)/2 panel builds, i.e.
    ``(P+1)/2`` full-build equivalents -- 1x at P=1. Unequal final panels need a
    row-weighted count.

    PROGRESS: because that cost is set by ``panel_rows`` and is invisible from
    outside, each COMPLETED ``(i,j)`` pair is logged at INFO with the count and the
    elapsed loop time. ``on_panel(pairs_done, n_pairs, elapsed_s)`` is the
    structured form; the log fires whether or not it is supplied, since the
    omission being fixed here was a caller that passed no hook.

    The count advances only after a pair's kernel block is written, and the clock
    starts after the fixed setup, so reported elapsed covers exactly the completed
    work -- it neither omits the current pair's kernel term nor amortises an
    allocation cost over a growing denominator.

    NO projected total is reported, deliberately. Extrapolating a mean over this
    schedule is not sound without diagonal/off-diagonal and short-panel weights:
    with ``mirror`` False the off-diagonal branch does twice the diagonal work and
    the mix shifts as ``i`` grows, and the final panel is short. A progress figure
    that licenses killing a run must err pessimistically, and an unweighted mean
    does not. Callers wanting a projection should calibrate against measured pairs
    and state an uncertainty band.

    ``ao_block_factory`` must return a FRESH iterable per call.

    Returns (Pi, kern) with kern (Nk, Nip, Nip) complex128.
    """
    X = np.asarray(X)
    n_kpts, n_ip = int(X.shape[0]), int(X.shape[1])
    if isinstance(panel_rows, bool) or not isinstance(panel_rows, (int, np.integer)):
        raise ValueError(f"panel_rows must be an integer, got {type(panel_rows)}.")
    if int(panel_rows) <= 0:
        raise ValueError(f"panel_rows must be positive, got {panel_rows}.")
    pair_convolve = _pair_convolve()

    # Pi needs no AO stream -- it is X against itself -- so it is built once and
    # never regenerated, whatever the panel schedule does.
    Pi = pair_convolve(X, X, phase, imag_tol=imag_tol)[neg]

    # The mirror needs kern Hermitian, which needs the provider self-adjoint at
    # fixed q. Linearity and the q/-q dagger law do not imply it, so it is opt-in
    # capability rather than assumption; without it the transposed panel is built.
    mirror = bool(getattr(provider, "is_self_adjoint_per_q", False))

    rows = int(panel_rows)
    panels = [(p0, min(p0 + rows, n_ip)) for p0 in range(0, n_ip, rows)]
    # Progress is reported by DEFAULT, not only when a caller opts in: a production
    # run spent 18.9 h in this loop and printed nothing between the enclosing stage
    # markers, so the panel_rows price could not be read from outside. The unit is a
    # COMPLETED (i,j) pair, counted after its kernel block, so reported elapsed
    # covers only finished work.
    #
    # Deliberately NO projected total. An earlier revision extrapolated a mean over
    # panel builds and was wrong in the unsafe direction: the count advanced at the
    # eta build, before that pair's kernel work, so every line omitted a kernel term
    # -- at the first line the whole of it. Against a deterministic P=2 model the
    # projections read 3.0/4.5/6.0 for an actual 7.0, still short at the last line.
    # An honest projection needs diagonal/off-diagonal and short-panel weights or
    # measured calibration with an uncertainty band; until then this reports what
    # HAPPENED and the caller does its own arithmetic. Progress that licenses a kill
    # decision has to be right in the pessimistic direction, and this was not.
    n_pairs = len(panels) * (len(panels) + 1) // 2
    pairs_done = 0

    def _eta_panel(p0, p1):
        """eta rows [p0:p1) for every q, one AO block resident at a time."""
        return _eta_rows_streamed(
            X[:, p0:p1, :], ao_block_factory(), phase, neg, n_grid_total,
            imag_tol=imag_tol,
            on_block=on_block)

    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    q_kpts = np.asarray(provider.canonical_kpts, dtype=np.float64)
    # Known up front, so eta slabs preallocate and the AO stream is never listed.
    n_grid_total = int(grid_coords.shape[0])
    # Precomputed once: the contraction scales by 1/sqrt(n_grid), and passing
    # it in keeps _panel_block's signature free of a traced-vs-static split.
    inv_sqrt_grid = 1.0 / np.sqrt(n_grid_total)
    gphases = np.exp(-1j * (grid_coords @ q_kpts.T)).T          # (Nk, Ng)
    kern = np.zeros((n_kpts, n_ip, n_ip), dtype=np.complex128)
    # After the fixed setup above, so elapsed is loop work and not an allocation
    # term amortised over a growing denominator.
    loop_started = time.perf_counter()
    for i, (i0, i1) in enumerate(panels):
        # HOISTED (task #116 follow-up). eta_i is reused by every j >= i, so it
        # is staged on device ONCE PER PANEL rather than once per pair. Inside
        # the j loop it cost P(P+1)/2 conversions instead of P -- 21 vs 6 at
        # P=6, i.e. 488 GB of avoidable host->device traffic at 444/cc-pvtz,
        # which cancelled most of what removing the rq materialisation saved.
        #
        # The host array is released immediately: everything below consumes the
        # device version, so keeping both would hold two panels (65 GB) where
        # one will do.
        eta_i_host = _eta_panel(i0, i1)
        eta_i = jnp.asarray(eta_i_host, dtype=jnp.complex128)
        del eta_i_host
        for j in range(i, len(panels)):
            j0, j1 = panels[j]
            if j == i:
                eta_j = eta_i
            else:
                eta_j_host = _eta_panel(j0, j1)
                eta_j = jnp.asarray(eta_j_host, dtype=jnp.complex128)
                del eta_j_host
            for q in range(n_kpts):
                # gphase multiplies along the shared grid axis, so
                #   (eta_i*g) @ conj(apply(eta_j*g)).T
                # = eta_i @ (conj(apply(eta_j*g)) * g).T
                # exactly, and the left factor needs no separate array.
                gphase = gphases[q]
                # jnp, not np: the contraction below runs on device, so the
                # operands are staged once here instead of a panel-sized array
                # crossing the boundary per pair (task #116).
                # Device slices of already-device panels: no host transfer.
                eta_iq = eta_i[q]
                eta_jq = eta_iq if j == i else eta_j[q]
                rq_j = _right_factor(provider, q, eta_jq, gphase)
                block = np.asarray(_panel_block(eta_iq, rq_j, inv_sqrt_grid))
                kern[q, i0:i1, j0:j1] = block
                if j != i and not mirror:
                    # No self-adjointness guarantee: compute the transposed panel
                    # instead of mirroring it. Correct for any provider, at twice
                    # the off-diagonal work.
                    rq_i = _right_factor(provider, q, eta_iq, gphase)
                    kern[q, j0:j1, i0:i1] = np.asarray(
                        _panel_block(eta_jq, rq_i, inv_sqrt_grid))
                    del rq_i
                elif j != i:
                    kern[q, j0:j1, i0:i1] = np.conj(block).T
                del rq_j
                # Do not let the final q slices extend into the next panel pair.
                del eta_iq, eta_jq, block
            if j != i:
                # Only when it is a distinct buffer -- on the diagonal eta_j
                # ALIASES eta_i, and dropping it there would free the panel the
                # remaining j iterations still need.
                del eta_j
            # Counted HERE: the pair's kernel work is finished, so elapsed contains
            # it. Advancing at the eta build instead is what made the previous
            # revision optimistic.
            pairs_done += 1
            elapsed = time.perf_counter() - loop_started
            logger.info(
                "p_blocked: pair %d/%d done (i=%d rows[%d:%d], j=%d rows[%d:%d]) "
                "elapsed %.1fs",
                pairs_done, n_pairs, i, i0, i1, j, j0, j1, elapsed,
            )
            if on_panel is not None:
                on_panel(pairs_done, n_pairs, elapsed)
        del eta_i

    if self_paired is not None:
        # The dense solve projects BOTH Pi_q and kern_q at self-paired q
        # (isdf.py:675 and :712). Projecting only kern would hand the solve a
        # complex Pi it would have made real.
        Pi = np.array(Pi, dtype=np.complex128, copy=True)
        for q in range(n_kpts):
            if self_paired(q):
                Pi[q] = Pi[q].real.astype(np.complex128)
                kern[q] = kern[q].real.astype(np.complex128)
    return Pi, kern
