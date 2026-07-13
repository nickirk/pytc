"""Matrix-free pivoted-Cholesky pivot selection -- model-agnostic
(pytc/df/ package reorganization, task #8, isdf-coulomb-cuda,
2026-07-12). Both TC's own phi/gradient decomposition
(pytc.df.isdf.isdf_decompose) and the Coulomb path's sector-aware pivot
selection (pytc.coulomb.pivot_selection) share this module -- see
pivoted_cholesky_pair_pivots's docstring for the algebraic identity that
makes it matrix-free, and the extensive review-round history for how
its correctness/scale-invariance guarantees were established.
"""
import jax
import jax.numpy as jnp
from functools import partial


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

    Round-5 fix (Alice's 4th re-review, 2026-07-12): Round 4's
    ``small_eps = 100 * eps(dtype) * max_diag`` can EXCEED the
    scientific ``eff_tol = effective_rank_rtol * max_diag`` in float32
    (100*eps ~= 1.2e-5, larger than the default rtol=1e-6) -- a pivot
    can be scientifically effective (raw residual >= eff_tol) while the
    OFFICIAL L update still calls it "small" and skips deflation. The
    test module's module-level x64 config hides this in float64 (where
    100*eps ~= 2.2e-14, always far below any reasonable rtol), so this
    needs an explicit float32 regression to stay caught.
    INVARIANT: the numerical threshold must never exceed the scientific
    one, else effective pivots skip deflation and the prefix breaks in
    low precision. Fixed with ``small_eps = min(100*eps(dtype),
    0.1*effective_rank_rtol) * max_diag`` -- the min() with a c=0.1
    factor guarantees small_eps < eff_tol always, regardless of dtype
    or rtol choice. Also fixed a related dtype-propagation gap exposed
    while reproducing this in float32: ``L``/``L_raw`` were allocated
    via bare ``jnp.zeros(...)`` (no explicit dtype), silently following
    JAX's ambient x64-flag default rather than the actual input dtype --
    now explicitly ``dtype=diag_err.dtype``.
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
        # INVARIANT: the numerical threshold must never exceed the
        # scientific one, else a pivot judged effective by eff_tol gets
        # skipped by the official L update and never deflated -- its
        # duplicate/correlated twin then looks like fresh signal, and
        # the prefix latch closes too early, silently dropping later
        # genuinely-independent signal (Alice's 4th re-review,
        # 2026-07-12: 100*eps is ~1.2e-5 in float32, LARGER than the
        # default effective_rank_rtol=1e-6 -- the x64-enabled test
        # module hid this path entirely). min() with c=0.1 guarantees
        # small_eps < eff_tol always, regardless of dtype or rtol.
        small_eps = jnp.minimum(100.0 * jnp.finfo(diag_err.dtype).eps,
                                 0.1 * effective_rank_rtol) * max_diag
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

    # Explicit dtype (not JAX's ambient x64-flag default) so mixed-
    # precision callers (e.g. float32 inputs under a process that has
    # jax_enable_x64 on for other code) get a float32 L/L_raw trajectory
    # matching diag_err's own dtype, not a silently-upcast float64 one.
    L = jnp.zeros((n_grid, n_rank), dtype=diag_err.dtype)
    pivots = jnp.zeros(n_rank, dtype=int)
    selected_mask = jnp.zeros(n_grid, dtype=bool)

    if track_effective_rank:
        # Separate, UNSHIFTED/UNRAMPED raw residual + its own Cholesky
        # factor, tracked in parallel purely for the effective-rank
        # diagnostic -- only allocated on this opt-in path (Felix's
        # architect ranking, 2026-07-12: legacy TC callers pay nothing).
        raw_diag_err0 = raw_diag
        L_raw0 = jnp.zeros((n_grid, n_rank), dtype=diag_err.dtype)
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
