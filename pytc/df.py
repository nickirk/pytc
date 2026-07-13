"""JAX implementation of Density Fitting / ISDF."""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from functools import partial
import os
import logging
import time
import h5py
import uuid
import gc

logger = logging.getLogger(__name__)


@jax.jit
def solve_normal_equations_batch(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                   phi_p_batch: jnp.ndarray, phi_q_batch: jnp.ndarray,
                                   rcond: float = 1e-14) -> jnp.ndarray:
    """Fast solver using LU decomposition for structured least-squares.
    
    Solves min ||C*X - B||² where:
      C[pq, m] = phi_piv_p[p, m] * phi_piv_q[q, m]
      B[pq, g] = phi_p_batch[p, g] * phi_q_batch[q, g]
    
    This exploits the separable structure to avoid O(N^2) intermediates:
    (C^T B)[m, g] = (sum_p phi_piv_p[p,m]*phi_p_batch[p,g]) * (sum_q phi_piv_q[q,m]*phi_q_batch[q,g])
    
    Args:
        phi_piv_p: (n_orb, n_fused) first factor of pivots
        phi_piv_q: (n_orb, n_fused) second factor of pivots
        phi_p_batch: (n_orb, batch_size) first factor of target
        phi_q_batch: (n_orb, batch_size) second factor of target
        rcond: Relative regularization strength (default 1e-14)
        
    Returns:
        X: (n_fused, batch_size) solutions
    """
    # Compute A^T A efficiently using the Kronecker-like structure
    gram_p = phi_piv_p.T @ phi_piv_p  # (n_fused, n_fused)
    gram_q = phi_piv_q.T @ phi_piv_q  # (n_fused, n_fused)
    ATA = gram_p * gram_q  # Element-wise product
    
    # Compute A^T B efficiently using separable structure
    term_p = jnp.matmul(phi_piv_p.T, phi_p_batch)  # (n_fused, batch_size)
    term_q = jnp.matmul(phi_piv_q.T, phi_q_batch)  # (n_fused, batch_size)
    ATB = term_p * term_q  # (n_fused, batch_size)
    
    # Use LU solve (jnp.linalg.solve) with Tikhonov regularization
    diag_mean = jnp.mean(jnp.diag(ATA))
    jitter = diag_mean * rcond
    ATA_reg = ATA + jitter * jnp.eye(ATA.shape[0])
    X = jnp.linalg.solve(ATA_reg, ATB)
    
    return X


solve_normal_equations_batch = jax.jit(solve_normal_equations_batch, static_argnames=['rcond'])


@jax.jit
def _build_normal_matrix(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray) -> jnp.ndarray:
    """Build unregularized normal-equation matrix for structured LS."""
    gram_p = phi_piv_p.T @ phi_piv_p
    gram_q = phi_piv_q.T @ phi_piv_q
    return gram_p * gram_q


def prepare_normal_equations_solver(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                    rcond: float = 1e-14,
                                    max_jitter_tries: int = 8,
                                    jitter_growth: float = 10.0):
    """Prepare robust Cholesky factor for repeated batched solves.

    Uses adaptive jitter escalation to guarantee numerically SPD matrices.
    """
    ata = _build_normal_matrix(phi_piv_p, phi_piv_q)
    ata = 0.5 * (ata + ata.T)
    diag_mean = float(jnp.mean(jnp.diag(ata)))
    eps_scale = float(jnp.finfo(ata.dtype).eps) * max(diag_mean, 1.0)
    base_jitter = max(diag_mean * rcond, eps_scale)
    eye = jnp.eye(ata.shape[0], dtype=ata.dtype)

    last_chol = None
    for attempt in range(max_jitter_tries):
        jitter = base_jitter * (jitter_growth ** attempt)
        chol, lower = jsp_linalg.cho_factor(ata + jitter * eye, lower=True)
        if bool(jnp.all(jnp.isfinite(chol))):
            if attempt > 0:
                logger.warning(
                    "Cholesky jitter escalated: base=%.3e final=%.3e tries=%d",
                    base_jitter, jitter, attempt + 1
                )
            return chol, bool(lower)
        last_chol = chol

    raise np.linalg.LinAlgError(
        f"Adaptive Cholesky failed after {max_jitter_tries} tries; "
        f"base_jitter={base_jitter:.3e}, last_nonfinite={bool(jnp.any(jnp.isnan(last_chol)))}"
    )


@partial(jax.jit, static_argnames=('lower',))
def solve_normal_equations_batch_prepared(chol: jnp.ndarray, lower: bool,
                                          phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                          phi_p_batch: jnp.ndarray, phi_q_batch: jnp.ndarray) -> jnp.ndarray:
    """Solve batched normal equations using precomputed Cholesky factor."""
    term_p = jnp.matmul(phi_piv_p.T, phi_p_batch)
    term_q = jnp.matmul(phi_piv_q.T, phi_q_batch)
    atb = term_p * term_q
    return jsp_linalg.cho_solve((chol, lower), atb)


@partial(jax.jit, static_argnames=('n_rank', 'track_effective_rank', 'effective_rank_rtol'))
def _pivoted_cholesky_pair_pivots_core(factor_p_weighted, factor_q_weighted, n_rank, shift,
                                        track_effective_rank=False, effective_rank_rtol=1e-6):
    """Matrix-free pivoted-Cholesky selection of grid/interpolation points
    that best span the pair-product space ``A[g,(p,q)] = factor_p[p,g] *
    factor_q[q,g]``, without ever forming ``A`` (task #5,
    isdf-coulomb-cuda: jastrow-independent shared primitive both the TC
    pipeline and the Coulomb path import).

    Exploits the same algebraic identity the original TC-specific
    ``_pivoted_cholesky_phi``/``_pivoted_cholesky_grad`` functions each
    hard-coded separately: the Gram matrix of the pair-product matrix
    factors as ``(A A^T)[g,g'] = (sum_p factor_p[p,g] factor_p[p,g']) *
    (sum_q factor_q[q,g] factor_q[q,g'])`` -- a product of the two
    factors' own (n_grid, n_grid) Gram matrices, each computable directly
    from the (n_feature, n_grid) inputs with no O(n_grid * n_p * n_q)
    intermediate. Passing the SAME array for both factors reproduces
    ``_pivoted_cholesky_phi``'s original single-Gram-squared case (the
    "oo"/"vv" sector case, same MO subset on both sides of the pair);
    passing two DIFFERENT arrays reproduces ``_pivoted_cholesky_grad``'s
    original two-different-Grams case (the "ov" sector case, or TC's
    phi/grad_phi pairing when grad_phi is pre-flattened to 2-D by the
    caller -- e.g. ``grad_phi_weighted.transpose(0,2,1).reshape(n_orb*3,
    n_grid)`` collapses the (orb, xyz) axes together, so the dot-product
    reduction still sums over exactly what the original per-component
    ``for c in range(3)`` loop summed over).

    Args:
        factor_p_weighted: (n_p, n_grid) weighted values (e.g. occupied
            MO values on a grid, sqrt(weight)-scaled).
        factor_q_weighted: (n_q, n_grid) weighted values for the pair's
            other factor. Pass the same array as factor_p_weighted for a
            same-set pair space (TC's phi decomposition; the Coulomb
            path's "oo"/"vv" sectors); pass a different array for a
            mixed pair space (the Coulomb path's "ov" sector; TC's
            phi/grad_phi pairing, factor_q pre-flattened to 2-D).
        n_rank: Number of interpolation points (pivots) to select. Must be
            <= n_grid (validated).
        shift: Tikhonov-style regularization added to the diagonal
            before each pivot selection (numerically pins near-zero/
            near-degenerate residuals to exactly zero rather than
            leaving them as noise-dominated candidates).
        track_effective_rank: STATIC (compile-time) flag. False (default)
            compiles the legacy TC-production code path -- ramp baked
            into diag_err, no effective-rank bookkeeping, same cost as
            Round-1's dedup fix and nothing more. True (Coulomb path
            only, via pivot_selection.select_sector_pivots) compiles the
            effective-rank-tracking path described below. The two are
            DIFFERENT XLA graphs (plain Python ``if`` on a static arg,
            not jax.lax.cond) so the False path never pays for the True
            path's extra bookkeeping (Felix's architect ranking on
            Alice's re-review, 2026-07-12).
        effective_rank_rtol: Only used when track_effective_rank=True.
            Relative tolerance (fraction of the raw diagonal's own max)
            for counting a selection as carrying real signal. Default
            1e-6, not 1e-10: measured on the H2O/cc-pVDZ ov-sector
            reproduction (true rank 95), the raw residual beyond the
            true rank decays SLOWLY (not to a sharp zero) -- 1e-10 and
            1e-8 both left it above threshold out to n_rank=130 (a false
            effective_rank=130, violating the analytic rank<=95 bound);
            1e-6 correctly stays at or below 95 (measured 94, a safe
            1-point margin) and 5e-7 hits exactly 95. 1e-6 is the
            conservative round-number choice. This is a genuine
            system-dependent numerical-noise-floor calibration, not an
            arbitrary constant -- override per-system if a different
            matrix's tail decays differently.

    Returns:
        (pivots, effective_rank): pivots is (n_rank,) selected grid-point
        indices, in selection order, GUARANTEED unique regardless of
        track_effective_rank (see Round-1 fix below). effective_rank is
        a JAX scalar when track_effective_rank=True: how many of those
        selections had real numerical signal before the residual was
        numerically exhausted -- pivots[:effective_rank] is a
        GUARANTEED-VALID prefix slice (see Round-3 fix below), steps
        beyond it are unique grid indices (never duplicates) but pick
        among residual-exhausted candidates rather than genuine
        interpolation-quality points. When track_effective_rank=False,
        effective_rank is not meaningful (not computed) -- callers on
        that path must not use it.

    Round-1 fix (Alice, 2026-07-12): the original per-step zeroing
    (``diag_err.at[pivot].set(0.0)`` only inside the ``is_small``
    branch) relied on the Cholesky update's OWN arithmetic driving the
    just-selected index's residual to exactly zero -- which
    floating-point rounding doesn't guarantee after many iterations, so
    once the residual was numerically exhausted, ``argmax`` could
    re-select an EARLIER pivot instead of a fresh index (measured:
    requesting 300 pivots on a 95-dim H2O/cc-pVDZ pair space returned
    only 274 unique). Fixed with an explicit selected-mask instead of
    relying on the residual reaching exact zero, so duplicates are now
    impossible by construction -- this fix is UNCONDITIONAL (applies on
    both the track_effective_rank True/False paths), since it's a
    genuine correctness fix for TC's production path too, not just the
    Coulomb path's diagnostic.

    Round-2 fix (Alice's 1st re-review, 2026-07-12), superseded by
    Round-3 below: an initial effective_rank implementation counted
    ``diag_err[pivot] >= 1e-12`` against ``diag_err`` itself (shift +
    ramp included) and used a SEPARATE parallel L_raw/raw_diag_err
    Cholesky pair to work around the ramp contamination -- doubling
    memory/compute on every call, including TC's production path (which
    doesn't even use effective_rank). Superseded entirely by Round 3.

    Round-3 fix (Alice's 2nd re-review, 2026-07-12) -- two issues:
    (1) PREFIX-SEMANTICS BUG: Round 2's ramp (``1e-12 * arange(n_grid) *
    max_diag``) has a total span growing with n_grid, which at large
    n_grid can exceed effective_rank_rtol's threshold, and Round 2's
    ``n_effective`` was a raw COUNT of how many selections individually
    passed the raw-residual check -- not necessarily a contiguous prefix
    of the returned pivots. Alice's synthetic repro (n_grid=1000, one
    strong signal + one weak-but-effective signal + noise-only points
    elsewhere) selected [0, 999, 998], reporting effective_rank=1 and
    completely missing the real weak-but-effective signal at index 1.
    (2) MEMORY REGRESSION: Round 2's L_raw/raw_diag_err pair doubled
    memory/compute unconditionally, including on TC's production path.

    Fixed by: (a) making effective-rank tracking a STATIC opt-in
    (track_effective_rank) so the legacy path is bit-structurally
    identical to Round 1 -- zero regression BY CONSTRUCTION, fixing
    issue 2 (the parallel L_raw/raw_diag_err pair from Round 2 is kept,
    but now only allocated on the opt-in path); (b) PREFIX-BY-
    CONSTRUCTION accounting: effective_rank is redefined as the length
    of the LEADING CONTIGUOUS RUN of raw-effective selections (a
    ``still_effective`` flag latches False the instant one selection's
    raw residual falls below eff_tol, and never counts again after
    that), not a total count -- this makes ``pivots[:effective_rank]``
    a valid prefix BY DEFINITION, regardless of whether a later
    noise-level selection happens to read as effective again; (c) the
    tie-break ramp is normalized (divided by n_grid-1) on the opt-in
    path so its span stays ~1e-12*max_diag regardless of n_grid.

    An earlier version of this fix additionally RESTRICTED pivot
    selection itself to the raw-effective candidate pool (once
    available) -- verified EMPIRICALLY (not assumed) that this actively
    HURTS convergence quality on real data: on the H2O/cc-pVDZ ov-sector
    reproduction, restricting the pool inflated effective_rank from the
    correct 95 to 119, because the ramp's role isn't merely tie-breaking
    -- it numerically steers the greedy Cholesky descent to a clean
    collapse (residual -> exactly 0) at the true rank boundary, and
    restricting candidates before that collapse completes disrupts it.
    Selection is therefore UNRESTRICTED on both paths (always argmax
    over the full unselected pool via the official ramped/shifted
    diag_err) -- only the accounting differs.

    Round-4 fix (Alice's 3rd re-review, 2026-07-12): the tracked branch's
    internal numerical-safety guards (``is_small``/``is_small_raw``, used
    to protect ``rsqrt`` from a near-zero pivot) were still ABSOLUTE
    (``pivot_val < 1e-12``) even though effective_rank itself is defined
    by the SCALE-RELATIVE ``eff_tol = effective_rank_rtol * max_diag`` --
    rescaling the factor matrices by a positive constant (same
    mathematical row space, the Gram merely scales) shifts max_diag by
    the same factor but left the absolute guards fixed. A small enough
    global rescale (Alice's repro: 1e-4, one-feature rank-1 factors with
    two identical nonzero columns) pushed EVERY pivot_val below the
    absolute 1e-12 floor, permanently disabling the Cholesky deflation
    update entirely -- a duplicate/correlated column was then never
    deflated after its twin was selected, and was counted as a SECOND
    independent effective signal for an analytically rank-1 problem.
    Fixed with two DIFFERENT scale-relative criteria (only on the
    track_effective_rank=True path -- the legacy path's absolute 1e-12
    is untouched, matching Felix's "bit-identical to Round 1" mandate):
    ``is_small`` (official L, numerical-safety concern) now uses
    ``100 * eps * max_diag`` (machine-epsilon-relative, as tight as
    numerically defensible); ``is_small_raw`` (the L_raw diagnostic)
    reuses ``eff_tol`` itself -- once a selection's raw residual falls
    below eff_tol, ``still_effective`` has already latched False and no
    further raw-track precision is scientifically needed; above that
    threshold, the raw pivot is ALWAYS genuinely deflated regardless of
    how small it looks in absolute terms.
    """
    n_grid = factor_p_weighted.shape[1]
    if n_rank > n_grid:
        raise ValueError(
            f"n_rank={n_rank} exceeds n_grid={n_grid} -- cannot select "
            f"more interpolation points than there are candidate grid "
            f"points."
        )

    A_diag = jnp.sum(factor_p_weighted**2, axis=0)
    B_diag = jnp.sum(factor_q_weighted**2, axis=0)
    raw_diag = A_diag * B_diag
    max_diag = jnp.max(jnp.abs(raw_diag))
    diag_err = raw_diag + shift

    # track_effective_rank is a STATIC arg (Felix's architect ranking,
    # 2026-07-12): the two branches below compile to DIFFERENT XLA
    # graphs at trace time (plain Python ``if``, not jax.lax.cond) --
    # when False (the default, TC's _pivoted_cholesky_phi/_grad never
    # pass True), the compiled graph is exactly Round-1's dedup-fixed
    # single-factor code with the ramp baked into diag_err as originally
    # designed: zero added memory/compute vs that baseline BY
    # CONSTRUCTION, not by hoping the extra work is cheap enough to not
    # matter. Only the Coulomb path (pivot_selection.select_sector_pivots)
    # opts in with True.
    # Selection ALWAYS uses this single "official" diag_err (shift + tie-
    # break ramp baked in) -- empirically, the ramp is not just a
    # cosmetic tie-breaker: it numerically steers the greedy Cholesky
    # descent to a clean collapse (residual -> exactly 0) at the true
    # rank boundary. A prior attempt restricted pivot SELECTION itself to
    # a "raw-effective-only" candidate pool once the true rank was
    # approached; verified empirically on real H2O/cc-pVDZ data that this
    # DISRUPTS that steering (effective_rank inflated from 95 to 119) --
    # reverted in favor of the streak-based accounting below, which
    # guarantees prefix validity without touching selection at all.
    if track_effective_rank:
        eff_tol = effective_rank_rtol * max_diag
        # Numerical-safety floor for rsqrt (distinct from eff_tol's
        # SCIENTIFIC-significance threshold): scale-relative to max_diag,
        # using machine epsilon with a 100x safety margin, not the
        # absolute ``1e-12`` the legacy path below uses. A rescaled
        # problem (e.g. factor matrices multiplied by 1e-4, same
        # mathematical row space) shifts max_diag by the same factor, so
        # this floor tracks it -- the absolute 1e-12 version did not,
        # and could classify EVERY pivot as "small" on a rescaled
        # problem, permanently disabling the Cholesky deflation entirely
        # and letting duplicate/correlated signal go undetected (Alice's
        # 3rd re-review, 2026-07-12: one-feature rank-1 factors with two
        # identical nonzero columns, rescaled by 1e-4, wrongly reported
        # effective_rank=2 for an analytically rank-1 problem).
        small_eps = 100.0 * jnp.finfo(diag_err.dtype).eps * max_diag
        # Tie-break ramp span normalized to stay ~1e-12*max_diag
        # REGARDLESS of n_grid (divide by n_grid-1) -- the previous
        # unnormalized ``1e-12 * arange(n_grid)`` had a span growing with
        # n_grid that could exceed effective_rank_rtol's threshold at
        # large n_grid (Alice's re-review, 2026-07-12). Only applied on
        # this opt-in path -- the legacy path below keeps the exact
        # original formula, unchanged.
        diag_err = diag_err + ((1e-12 / jnp.maximum(n_grid - 1, 1))
                                * jnp.arange(n_grid, dtype=diag_err.dtype) * max_diag)
    else:
        eff_tol = 0.0
        small_eps = 0.0
        diag_err = diag_err + 1e-12 * jnp.arange(n_grid, dtype=diag_err.dtype) * max_diag

    L = jnp.zeros((n_grid, n_rank))
    pivots = jnp.zeros(n_rank, dtype=int)
    selected_mask = jnp.zeros(n_grid, dtype=bool)

    if track_effective_rank:
        # Separate, UNSHIFTED/UNRAMPED raw residual + its own Cholesky
        # factor, tracked in parallel purely for the effective-rank
        # diagnostic -- only allocated on this opt-in path (Felix's
        # architect ranking, 2026-07-12: legacy TC callers pay nothing).
        raw_diag_err0 = raw_diag
        L_raw0 = jnp.zeros((n_grid, n_rank))
        n_effective0 = jnp.array(0, dtype=int)
        still_effective0 = jnp.array(True)

        def body_fn(step, state):
            (diag_err, L, raw_diag_err, L_raw, pivots, selected_mask,
             n_effective, still_effective) = state
            unselected = jnp.logical_not(selected_mask)
            # Force already-selected indices to -inf before argmax --
            # duplicates impossible by construction (Round-1 fix).
            score = jnp.where(unselected, diag_err, -jnp.inf)
            pivot = jnp.argmax(score)
            pivots = pivots.at[step].set(pivot)
            selected_mask = selected_mask.at[pivot].set(True)
            pivot_val = diag_err[pivot]

            # PREFIX-BY-CONSTRUCTION (Alice's re-review, 2026-07-12):
            # effective_rank is the length of the LEADING CONTIGUOUS RUN
            # of raw-effective selections, not a total count -- the
            # instant a selection's raw residual falls below eff_tol,
            # still_effective latches False permanently, so no LATER
            # selection (even a noise-level false positive) can ever be
            # counted again. This makes ``pivots[:effective_rank]`` a
            # valid prefix by DEFINITION, independent of whether
            # selection order and raw-effectiveness order coincide
            # exactly (they empirically do here, but this doesn't rely
            # on hoping that holds).
            pivot_val_raw = raw_diag_err[pivot]
            was_effective = pivot_val_raw >= eff_tol
            still_effective = jnp.logical_and(still_effective, was_effective)
            n_effective = n_effective + jnp.where(still_effective, 1, 0)

            A_col = jnp.dot(factor_p_weighted.T, factor_p_weighted[:, pivot])
            B_col = jnp.dot(factor_q_weighted.T, factor_q_weighted[:, pivot])
            S_col = A_col * B_col
            S_col_shifted = S_col.at[pivot].add(shift)

            dot_prod = jnp.dot(L, L[pivot])
            # Scale-relative numerical-safety floor (see small_eps's
            # definition above) -- NOT the legacy path's absolute 1e-12.
            is_small = pivot_val < small_eps
            safe_pivot = jnp.where(is_small, 1.0, pivot_val)
            inv_sqrt_pivot = jax.lax.rsqrt(safe_pivot)
            l_col = (S_col_shifted - dot_prod) * inv_sqrt_pivot
            l_col = jnp.where(is_small, 0.0, l_col)
            L = L.at[:, step].set(l_col)
            diag_err = jnp.maximum(diag_err - l_col**2, 0.0)
            diag_err = diag_err.at[pivot].set(0.0)

            # Parallel unshifted/unramped Cholesky update (diagnostic
            # only, never fed back into selection or the L above).
            # Latch-consistent skip criterion (Alice's re-review,
            # 2026-07-12): once pivot_val_raw < eff_tol, still_effective
            # has ALREADY closed (or is closing this step) and no
            # further raw update is scientifically needed. Below that,
            # the raw pivot must be genuinely divided/updated even when
            # its ABSOLUTE magnitude is tiny (e.g. a globally rescaled
            # problem) -- reusing eff_tol here (not a separate absolute
            # constant) keeps this consistent with what "effective"
            # means, and is always safely above small_eps's
            # machine-epsilon floor since effective_rank_rtol is many
            # orders larger than eps.
            dot_prod_raw = jnp.dot(L_raw, L_raw[pivot])
            is_small_raw = pivot_val_raw < eff_tol
            safe_pivot_raw = jnp.where(is_small_raw, 1.0, pivot_val_raw)
            inv_sqrt_pivot_raw = jax.lax.rsqrt(safe_pivot_raw)
            l_col_raw = (S_col - dot_prod_raw) * inv_sqrt_pivot_raw
            l_col_raw = jnp.where(is_small_raw, 0.0, l_col_raw)
            L_raw = L_raw.at[:, step].set(l_col_raw)
            raw_diag_err = jnp.maximum(raw_diag_err - l_col_raw**2, 0.0)
            raw_diag_err = raw_diag_err.at[pivot].set(0.0)

            return (diag_err, L, raw_diag_err, L_raw, pivots, selected_mask,
                    n_effective, still_effective)

        _, _, _, _, final_pivots, _, final_n_effective, _ = jax.lax.fori_loop(
            0, n_rank, body_fn,
            (diag_err, L, raw_diag_err0, L_raw0, pivots, selected_mask,
             n_effective0, still_effective0))
        return final_pivots, final_n_effective

    n_effective = jnp.array(0, dtype=int)

    def body_fn(step, state):
        diag_err, L, pivots, selected_mask, n_effective = state
        unselected = jnp.logical_not(selected_mask)
        # Force already-selected indices to -inf before argmax --
        # duplicates are now impossible by construction, not dependent
        # on the residual reaching exactly zero (Round-1 fix, kept
        # unconditionally since it benefits TC's production path too).
        score = jnp.where(unselected, diag_err, -jnp.inf)
        pivot = jnp.argmax(score)
        pivots = pivots.at[step].set(pivot)
        selected_mask = selected_mask.at[pivot].set(True)
        pivot_val = diag_err[pivot]

        A_col = jnp.dot(factor_p_weighted.T, factor_p_weighted[:, pivot])
        B_col = jnp.dot(factor_q_weighted.T, factor_q_weighted[:, pivot])
        S_col = A_col * B_col
        S_col = S_col.at[pivot].add(shift)

        dot_prod = jnp.dot(L, L[pivot])
        is_small = pivot_val < 1e-12
        safe_pivot = jnp.where(is_small, 1.0, pivot_val)
        inv_sqrt_pivot = jax.lax.rsqrt(safe_pivot)

        l_col = (S_col - dot_prod) * inv_sqrt_pivot
        l_col = jnp.where(is_small, 0.0, l_col)
        L = L.at[:, step].set(l_col)
        diag_err = jnp.maximum(diag_err - l_col**2, 0.0)
        diag_err = diag_err.at[pivot].set(0.0)

        return diag_err, L, pivots, selected_mask, n_effective

    _, _, final_pivots, _, final_n_effective = jax.lax.fori_loop(
        0, n_rank, body_fn, (diag_err, L, pivots, selected_mask, n_effective))
    return final_pivots, final_n_effective


def pivoted_cholesky_pair_pivots(factor_p_weighted, factor_q_weighted, n_rank, shift,
                                  track_effective_rank=False, effective_rank_rtol=1e-6):
    """Host-level wrapper around ``_pivoted_cholesky_pair_pivots_core``.

    The core is ``@jax.jit``-compiled and can only return JAX arrays (a
    traced ``effective_rank`` can't be converted with Python's ``int()``
    inside a jitted function -- that raises ConcretizationTypeError under
    tracing). This wrapper calls the jitted core, then does the host-side
    ``int()`` conversion on the (by then concrete) result -- only when
    track_effective_rank=True, since it's not meaningful otherwise (see
    the core's docstring for the algorithm, bug history, and full
    return-value semantics).

    Args:
        track_effective_rank: forwarded to the core -- STATIC opt-in for
            effective-rank tracking. False (default) is the zero-
            overhead legacy path TC's production wrappers use; True is
            for the Coulomb path (pivot_selection.select_sector_pivots),
            which needs a real effective_rank.
        effective_rank_rtol: forwarded to the core -- relative tolerance
            (fraction of the unregularized diagonal's own max) for
            counting a selection as carrying real signal. Ignored when
            track_effective_rank=False.

    Returns:
        (pivots, effective_rank): pivots is a (n_rank,) JAX array of
        selected grid-point indices. effective_rank is a plain Python
        int when track_effective_rank=True, else None (not computed --
        callers on the legacy path must not use it).
    """
    final_pivots, final_n_effective = _pivoted_cholesky_pair_pivots_core(
        factor_p_weighted, factor_q_weighted, n_rank, shift,
        track_effective_rank, effective_rank_rtol)
    if not track_effective_rank:
        return final_pivots, None
    return final_pivots, int(final_n_effective)


def _pivoted_cholesky_phi(phi_weighted, n_rank, shift):
    """Specialized pivoted Cholesky for phi decomposition.

    Thin wrapper: the "same factor on both sides" case of
    ``pivoted_cholesky_pair_pivots`` (see its docstring). Kept as a
    distinct name at TC's existing call site rather than inlining, so
    that site's intent stays self-documenting.

    Discards the ``effective_rank`` half of the shared primitive's
    ``(pivots, effective_rank)`` return to preserve this wrapper's
    pre-existing single-array-return contract with ``isdf_decompose``.
    """
    pivots, _effective_rank = pivoted_cholesky_pair_pivots(phi_weighted, phi_weighted, n_rank, shift)
    return pivots


def _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank, shift):
    """Specialized pivoted Cholesky for gradient decomposition.

    Thin wrapper: the "two different factors" case of
    ``pivoted_cholesky_pair_pivots`` (see its docstring), with
    grad_phi_weighted's (n_orb, n_grid, 3) shape flattened to the 2-D
    (n_orb*3, n_grid) form the shared primitive expects -- the
    transpose-then-reshape collapses (orb, xyz) into one feature axis
    so the dot-product reduction still sums over exactly what the
    original per-component ``for c in range(3)`` loop summed over.

    Discards the ``effective_rank`` half of the shared primitive's
    ``(pivots, effective_rank)`` return to preserve this wrapper's
    pre-existing single-array-return contract with ``isdf_decompose``.
    """
    n_orb, n_grid, _ = grad_phi_weighted.shape
    grad_flat = grad_phi_weighted.transpose(0, 2, 1).reshape(n_orb * 3, n_grid)
    pivots, _effective_rank = pivoted_cholesky_pair_pivots(phi_weighted, grad_flat, n_rank, shift)
    return pivots






def isdf_decompose(phi, grad_phi, n_rank_phi, n_rank_grad, weights=None,
                   grid_batch_size=4096, rcond=1e-14,
                   is_incore=False, save_path=None, fixed_pivots=None):
    """Perform ISDF decomposition of orbitals and their gradients.
    
    Memory-efficient implementation using SVD-based solver to avoid
    materializing large C matrices (n_orb² × n_fused).
    
    Args:
        phi: Orbitals on grid (n_orb, n_grid)
        grad_phi: Orbital gradients on grid (n_orb, n_grid, 3)
        n_rank_phi: Rank for phi decomposition
        n_rank_grad: Rank for gradient decomposition
        weights: Optional (n_grid,) array of integration weights. 
                 If provided, pivot selection is weighted by these weights.
        grid_batch_size: Number of grid points to process in each batch
        rcond: Relative condition number cutoff for SVD pseudoinverse (default 1e-14).
               Smaller values retain more singular values (more accurate but less stable).
        
    Returns:
        phi_piv: (N_orb, N_fused)
        xi_phi: (N_fused, N_grid)
        grad_phi_piv: (N_orb, N_fused, 3)
        xi_grad: (N_fused, N_grid, 3)
        pivots: (N_fused,)
    """
    if save_path is not None and os.path.exists(save_path) and fixed_pivots is None:
        try:
            with h5py.File(save_path, 'r') as f:
                if all(k in f for k in ['xi_phi', 'xi_grad', 'pivots', 'phi_isdf', 'grad_phi_isdf']):
                    logger.info(f"Loading ISDF decomposition from {save_path}")
                    pivots = jnp.array(f['pivots'][:])
                    phi_piv = jnp.array(f['phi_isdf'][:])
                    grad_phi_piv = jnp.array(f['grad_phi_isdf'][:])
                    
                    if is_incore:
                        cpu_device = jax.devices("cpu")[0]
                        xi_phi = jax.device_put(f['xi_phi'][:], cpu_device)
                        xi_grad = jax.device_put(f['xi_grad'][:], cpu_device)
                    else:
                        xi_phi = None
                        xi_grad = None
                        
                    return phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots, save_path
        except Exception as e:
            logger.warning(f"Failed to load ISDF from {save_path}: {e}. Recomputing...")

    n_orb, n_grid = phi.shape
    
    if weights is None:
        w_sqrt = jnp.ones(n_grid)
    else:
        w_sqrt = jnp.sqrt(jnp.abs(weights))  # Use abs to avoid NaN
        
    start_time = time.perf_counter()
    logger.info(f"Starting ISDF decomposition with n_orb={n_orb}, n_grid={n_grid}, n_rank_phi={n_rank_phi}, n_rank_grad={n_rank_grad}")
    if weights is not None:
        logger.info(f"  Using integration weights (min={jnp.min(weights):.3e}, max={jnp.max(weights):.3e})")

    # --- 1. Phi Decomposition ---
    t0 = time.perf_counter()
    
    # Apply weights to orbitals for pivot selection
    # The Gram matrix is (phi^T W phi)(phi^T W phi) where W = diag(weights)
    # Equivalently: (sqrt(W) phi)^T (sqrt(W) phi) squared
    phi_weighted = phi * w_sqrt  # (n_orb, n_grid)
        
    # Pre-compute diagonal for phi to calculate shift
    orb_sq = jnp.sum(phi_weighted**2, axis=0)  # Weighted orbital norms
    diag_phi = orb_sq**2
    shift_phi = 1e-12 * jnp.max(jnp.abs(diag_phi))

    pivots_phi = _pivoted_cholesky_phi(phi_weighted, n_rank_phi, shift_phi)
    t1 = time.perf_counter()
    logger.debug(f"Phi decomposition completed in {t1 - t0:.4f} s")

    # --- 2. Gradient Decomposition ---
    t0 = time.perf_counter()
    
    # Apply weights to gradients for pivot selection
    grad_phi_weighted = grad_phi * w_sqrt[:, None]  # (n_orb, n_grid, 3)
    
    # Pre-compute diagonal for grad to calculate shift
    A_diag = jnp.sum(phi_weighted**2, axis=0)
    B_diag = jnp.sum(jnp.sum(grad_phi_weighted**2, axis=2), axis=0)
    diag_grad = A_diag * B_diag
    shift_grad = 1e-12 * jnp.max(jnp.abs(diag_grad))

    pivots_grad = _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank_grad, shift_grad)
    t1 = time.perf_counter()
    logger.debug(f"Grad decomposition completed in {t1 - t0:.4f} s")
    
    # --- 3. Fuse pivots ---
    t0 = time.perf_counter()
    
    # Use numpy for unique to avoid JAX dynamic shape overhead
    pivots_all = np.concatenate([np.array(pivots_phi), np.array(pivots_grad)])
    pivots = jnp.array(np.unique(pivots_all))
    n_fused = pivots.shape[0]
    t1 = time.perf_counter()
    logger.info(f"Pivots fused: {pivots_phi.shape[0]} + {pivots_grad.shape[0]} -> {n_fused} in {t1 - t0:.4f} s")

    # Experiment hook (Task B precision investigation): override the device-selected
    # pivots with an externally-supplied fused-pivot set. Used to force CPU-selected
    # pivots onto the GPU interpolation so we can isolate whether the GPU/CPU
    # isdf_dU_err gap comes from pivot SELECTION (gap collapses) or downstream
    # numerics (gap persists). No effect on the default path (fixed_pivots=None).
    if fixed_pivots is not None:
        fp = np.asarray(fixed_pivots)
        if fp.ndim != 1 or not np.issubdtype(fp.dtype, np.integer):
            raise ValueError("fixed_pivots must be a 1-D integer array of grid indices")
        if np.unique(fp).shape[0] != fp.shape[0]:
            raise ValueError("fixed_pivots must be unique")
        if fp.size == 0 or fp.min() < 0 or fp.max() >= n_grid:
            raise ValueError(f"fixed_pivots out of range [0, {n_grid})")
        pivots = jnp.asarray(fp, dtype=pivots.dtype)
        n_fused = int(pivots.shape[0])
        logger.info(f"isdf_decompose: overriding with {n_fused} externally-supplied fixed pivots")
    
    # --- 4. Extract pivot values ---
    t0 = time.perf_counter()
    
    phi_piv = phi[:, pivots]  # (n_orb, n_fused)
    grad_phi_piv = grad_phi[:, pivots, :]  # (n_orb, n_fused, 3)
    
    t1 = time.perf_counter()
    logger.debug(f"Pivot values extracted in {t1 - t0:.4f} s")
    
    # --- 5. Solve for xi_phi and xi_grad using fast normal equations solver ---
    t0 = time.perf_counter()
    logger.info("Using fast normal equations solver")

    # Solve for xi_phi and xi_grad
    cpu_device = jax.devices("cpu")[0]
    grid_batch_size = min(grid_batch_size, n_grid)
    n_batches = (n_grid + grid_batch_size - 1) // grid_batch_size if grid_batch_size > 0 else 0
    
    # Pre-factor normal-equation matrices once and reuse for all grid batches.
    # This avoids rebuilding/re-factorizing ATA in every batch.
    phi_chol, phi_lower = prepare_normal_equations_solver(phi_piv, phi_piv, rcond=rcond)
    grad_chol = []
    grad_lower = []
    for c in range(3):
        chol_c, lower_c = prepare_normal_equations_solver(grad_phi_piv[:, :, c], phi_piv, rcond=rcond)
        grad_chol.append(chol_c)
        grad_lower.append(lower_c)

    # Multi-device: shard the grid axis of each batch across local devices;
    # replicate factors. Use local_devices() so this is safe under multi-process
    # JAX (arrays can only be placed on devices visible to this process).
    local_devices = jax.local_devices()
    n_devices = len(local_devices)
    use_sharding = n_devices > 1
    if use_sharding:
        mesh = Mesh(np.array(local_devices), ('g',))
        grid_shard = NamedSharding(mesh, P(None, 'g'))
        repl = NamedSharding(mesh, P())
        phi_chol = jax.device_put(phi_chol, repl)
        phi_piv_d = jax.device_put(phi_piv, repl)
        grad_chol = [jax.device_put(c, repl) for c in grad_chol]
        grad_phi_piv_d = jax.device_put(grad_phi_piv, repl)
        logger.info(f"  Multi-device sharding enabled across {n_devices} devices (grid axis)")
    else:
        phi_piv_d = phi_piv
        grad_phi_piv_d = grad_phi_piv
    
    # Setup storage
    h5_file = None
    if is_incore:
        logger.info(f"  Processing {n_batches} batches of size {grid_batch_size} (In-core)")
        xi_phi_storage = np.zeros((n_fused, n_grid), dtype=phi.dtype)
        xi_grad_storage = np.zeros((n_fused, n_grid, 3), dtype=phi.dtype)
    else:
        if save_path is None:
            save_path = f"isdf_temp_{uuid.uuid4().hex[:8]}.h5"
            logger.info(f"  No save_path provided, creating temporary HDF5: {save_path}")
        
        h5_file = h5py.File(save_path, 'a')
        logger.info(f"  Processing {n_batches} batches of size {grid_batch_size} (HDF5: {save_path})")
        
        # Create/Reset datasets
        for name, shape in [('xi_phi', (n_fused, n_grid)), ('xi_grad', (n_fused, n_grid, 3))]:
            if name in h5_file: del h5_file[name]
            h5_file.create_dataset(name, shape=shape, dtype=phi.dtype)
        
        for name, data in [('pivots', pivots), ('phi_isdf', phi_piv), ('grad_phi_isdf', grad_phi_piv)]:
            if name in h5_file: del h5_file[name]
            h5_file.create_dataset(name, data=np.array(data))
        
        xi_phi_storage = h5_file['xi_phi']
        xi_grad_storage = h5_file['xi_grad']

    # Warm up JIT so the sharded-program compile cost doesn't dominate short loops
    # (on a 7-batch benzene-5Z run the sharded compile otherwise ate the steady-state
    # speedup from parallel GPUs).
    if n_batches > 1:
        t_warm = time.perf_counter()
        warm_batch = jnp.zeros((n_orb, grid_batch_size), dtype=phi.dtype)
        if use_sharding:
            warm_batch = jax.device_put(warm_batch, grid_shard)
        warm = solve_normal_equations_batch_prepared(
            phi_chol, phi_lower, phi_piv_d, phi_piv_d, warm_batch, warm_batch
        )
        jax.block_until_ready(warm)
        for c in range(3):
            warm = solve_normal_equations_batch_prepared(
                grad_chol[c], grad_lower[c], grad_phi_piv_d[:, :, c], phi_piv_d,
                warm_batch, warm_batch
            )
            jax.block_until_ready(warm)
        del warm, warm_batch
        logger.debug(f"  Solve warmup (JIT compile) took {time.perf_counter() - t_warm:.2f} s")

    try:
        t_batch_start = time.perf_counter()
        for batch_idx in range(n_batches):
            g_start = batch_idx * grid_batch_size
            g_end = min(g_start + grid_batch_size, n_grid)
            bs = g_end - g_start

            # Pad batch width to a multiple of n_devices so the grid axis shards evenly.
            pad = (-bs) % n_devices if use_sharding else 0

            # 1. Xi_phi
            phi_batch = phi[:, g_start:g_end]
            if pad:
                phi_batch = jnp.pad(phi_batch, ((0, 0), (0, pad)))
            if use_sharding:
                phi_batch = jax.device_put(phi_batch, grid_shard)
            xi_phi_batch = solve_normal_equations_batch_prepared(
                phi_chol, phi_lower, phi_piv_d, phi_piv_d, phi_batch, phi_batch
            )
            if pad:
                xi_phi_batch = xi_phi_batch[:, :bs]
            xi_phi_storage[:, g_start:g_end] = np.array(xi_phi_batch)

            # 2. Xi_grad
            for c in range(3):
                grad_phi_batch_c = grad_phi[:, g_start:g_end, c]
                if pad:
                    grad_phi_batch_c = jnp.pad(grad_phi_batch_c, ((0, 0), (0, pad)))
                if use_sharding:
                    grad_phi_batch_c = jax.device_put(grad_phi_batch_c, grid_shard)
                xi_grad_batch = solve_normal_equations_batch_prepared(
                    grad_chol[c], grad_lower[c], grad_phi_piv_d[:, :, c], phi_piv_d,
                    grad_phi_batch_c, phi_batch
                )
                if pad:
                    xi_grad_batch = xi_grad_batch[:, :bs]
                xi_grad_storage[:, g_start:g_end, c] = np.array(xi_grad_batch)
            
            if batch_idx % 4 == 0 and batch_idx > 0:
                elapsed = time.perf_counter() - t_batch_start
                rate = batch_idx / elapsed
                eta = (n_batches - batch_idx) / rate if rate > 0 else 0
                logger.debug(f"Batch {batch_idx}/{n_batches} ({rate:.1f} batch/s, ETA: {eta:.1f}s)")
        
        # Load into JAX CPU RAM if requested
        if is_incore:
            xi_phi = jax.device_put(xi_phi_storage[:], cpu_device)
            xi_grad = jax.device_put(xi_grad_storage[:], cpu_device)
        else:
            xi_phi = None
            xi_grad = None
        
        # Explicitly delete storage to save RAM
        if is_incore:
            del xi_phi_storage, xi_grad_storage
            gc.collect()
        
    finally:
        if h5_file is not None:
            h5_file.close()
        gc.collect()
    
    total_time = time.perf_counter() - start_time
    logger.debug(f"Total fused ranks = {n_fused}")
    logger.info(f"ISDF decomposition total time: {total_time:.4f} s")
    
    return phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots, save_path
