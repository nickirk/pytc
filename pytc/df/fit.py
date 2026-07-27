"""Kernel-agnostic LS-THC/ISDF core-fit algebra -- model-agnostic.

pair_collocation_at_pivots, compute_Z,
compute_Z_cross, and reconstruct_eri_block all operate purely on (P, C)
pair-collocation/DF-contraction matrices with zero Coulomb-specific
content -- under the kernel-policy design (decision 001), the Poisson
builder and any future TC channel consume this identically, which is
precisely why it lives here rather than in pytc/integrals/coulomb.py.
What STAYS in pytc/integrals/coulomb.py: compute_C_streamed (the
DF-B-tensor route -- MolecularDFReference policy specifically) and the
gpu4pyscf adapter.

Notation: pair-collocation matrix
A[g,a] = sqrt(w_g) phi_p(r_g) phi_q(r_g) for pair index a=(p,q);
P[mu,a] = A[r_mu,a] (pair collocation AT the interpolation points mu
selected by pivot selection); S = P P^dagger; with orthonormalized DF
factor V = B^dagger B (B = the Cholesky-factorized cderi blocks a
kernel policy's own streaming yields, in MO-pair basis after
transform): C = P B^dagger, Z = S^-1 C C^dagger S^-1. ERIs are then
approximated as V ~= P^dagger Z P -- the only unavoidably dense object
is the (n_pivots, n_pivots) Z core, never the full (n_pair, n_pair) or
(n_aux, n_pair) tensors.
"""

import logging

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from .solvers import prepare_spd_cholesky, _CHOLESKY_BACKWARD_ERROR_TOL

logger = logging.getLogger(__name__)


def pair_collocation_at_pivots(factor_p_at_pivots, factor_q_at_pivots):
    """Build P[mu, (p,q)] = factor_p_at_pivots[p,mu] * factor_q_at_pivots[q,mu]
    -- the pair-collocation matrix evaluated ONLY at the (already-
    selected) interpolation points, not the full grid.

    Callers must pass RAW (unweighted) MO values here, not the
    sqrt(weight)-scaled values used for pivot SELECTION -- matching
    pytc.df.isdf_decompose's own convention (``phi_piv = phi[:,
    pivots]``, raw phi, even though pivots were chosen via
    ``phi_weighted``). Weighting is a numerical device for the
    pivoted-Cholesky selection step only; ISDF's actual interpolation
    formula operates on the real orbital values at the interpolation
    points, and the ERI reconstruction (compute_Z/reconstruct_eri_block)
    downstream of this P must reproduce the true (unweighted) integral
    INVARIANT: P is built from RAW values. Passing weighted values here
    makes compute_Z's rcond default silently depend on the grid
    quadrature's weight scale/level -- a portability bug.

    Args:
        factor_p_at_pivots: (n_p, n_pivots) RAW (unweighted) values at
            the pivot points (e.g. mo_occ_values[:, pivots], not
            occ_weighted[:, pivots]).
        factor_q_at_pivots: (n_q, n_pivots) RAW values at the pivot
            points for the pair's other factor.

    Returns:
        P: (n_pivots, n_p * n_q), row-major flattening of the (p, q)
            pair axis (matches np.reshape(..., (n_p, n_q)) on the
            trailing axis of any array built the same way).
    """
    factor_p_at_pivots = np.asarray(factor_p_at_pivots)
    factor_q_at_pivots = np.asarray(factor_q_at_pivots)
    n_p, n_pivots = factor_p_at_pivots.shape
    n_q = factor_q_at_pivots.shape[0]
    # P[mu, p, q] = factor_p[p, mu] * factor_q[q, mu]
    P = np.einsum("pu,qu->upq", factor_p_at_pivots, factor_q_at_pivots)
    return P.reshape(n_pivots, n_p * n_q)


_RESIDUAL_WARN_THRESHOLD = 1e-10  # design doc v1.1 §4: acceptance = 1e-10 fit residual
_RESIDUAL_NORM_CONVENTION_SAME = "||S Z S - C C^dagger|| / ||C C^dagger||"
_RESIDUAL_NORM_CONVENTION_CROSS = "||S_A Z_AB S_B - C_A C_B^dagger|| / ||C_A C_B^dagger||"
_CHOLESKY_JITTER_MAX_TRIES = 7  # design doc v1.1 §4: up to 6 retries (1 initial + 6)
_DEFAULT_RESIDUAL_N_PROBES = 32
_DEFAULT_RESIDUAL_SEED = 0


def _two_sided_residual(S_A, Z, S_B, M):
    """Solver-residual convention: the FULL two-sided residual on the actual returned Z, not a
    one-sided check of only the first S^-1 solve -- ``||S_A Z S_B -
    M|| / ||M||`` (same-sector: S_A=S_B=S, matches ``||S Z S -
    C C^dagger||/||C C^dagger||`` exactly). EXACT, O(n^3): forms the
    dense (n, n) products directly -- fine for reference/small-medium
    builds, but at production N_mu this is comparable to or larger than
    the solve itself; see
    _two_sided_residual_sampled for the O(n^2 * n_probes) alternative."""
    residual_num = float(np.linalg.norm(S_A @ Z @ S_B - M))
    residual_den = float(np.linalg.norm(M))
    return residual_num / residual_den if residual_den > 0.0 else 0.0


def _two_sided_residual_sampled(S_A, Z, S_B, M, n_probes=_DEFAULT_RESIDUAL_N_PROBES,
                                 seed=_DEFAULT_RESIDUAL_SEED):
    """Hutchinson stochastic-trace estimate of the same two-sided
    residual _two_sided_residual computes exactly, at O(n^2 * n_probes)
    instead of O(n^3). For
    A = S_A Z S_B - M, ||A||_F^2 = E[||A v||^2] for random v with
    i.i.d. mean-zero unit-variance entries (Rademacher here); average
    n_probes samples of the numerator and denominator separately and
    take the ratio of square roots. Never forms the dense (n, n)
    products S_A @ Z @ S_B -- only n_probes matrix-VECTOR products
    through S_A, Z, S_B, M, batched into ONE matmul chain each
    (S_A @ (Z @ (S_B @ V)) with V the (n, n_probes) probe matrix) and
    kept entirely on-device via jax.random/jnp -- an earlier version
    called np.asarray on the full dense (n, n) matrices first, forcing
    a multi-GB device->host transfer at production N_mu regardless of
    the reduced FLOP count.

    Returns (estimate, relative_standard_error): the SEM is a rough,
    first-cut uncertainty from the per-probe spread of the numerator
    term only (den_sq is treated as a fixed normalizer here, not itself
    variance-propagated) -- NOT a rigorous ratio-estimator confidence
    interval. NOT yet a validated production estimator: the probe-
    count/variance tradeoff at production N_mu (~25k) hasn't been
    characterized against the exact residual on real data -- this is a
    first cut at the right asymptotic complexity AND device placement.
    Diagnostic-only: unlike the exact residual, this NEVER drives the
    automatic TSVD fallback (see _cholesky_jitter_sandwich) until that
    validation exists. Use residual_mode="exact" for anything
    decision-grade.
    """
    if n_probes < 1:
        raise ValueError(
            f"n_probes must be >= 1, got {n_probes} -- n_probes=0 would silently "
            f"produce an empty probe matrix and a misleading zero residual."
        )
    S_A = jnp.asarray(S_A)
    S_B = jnp.asarray(S_B)
    Z = jnp.asarray(Z)
    M = jnp.asarray(M)
    n = M.shape[1]
    probe_dtype = M.dtype if jnp.iscomplexobj(M) else S_A.dtype
    key = jax.random.PRNGKey(seed)
    V = jax.random.rademacher(key, (n, n_probes)).astype(probe_dtype)  # (n, n_probes)
    AV = S_A @ (Z @ (S_B @ V)) - M @ V  # one batched matmul chain, O(n^2 * n_probes)
    MV = M @ V
    per_probe_num_sq = jnp.sum(jnp.abs(AV) ** 2, axis=0)  # (n_probes,)
    per_probe_den_sq = jnp.sum(jnp.abs(MV) ** 2, axis=0)  # (n_probes,)
    num_sq = float(jnp.sum(per_probe_num_sq))
    den_sq = float(jnp.sum(per_probe_den_sq))
    if den_sq <= 0.0:
        return 0.0, None
    estimate = float(np.sqrt(num_sq / den_sq))
    if n_probes > 1:
        num_mean = num_sq / n_probes
        num_std = float(jnp.std(per_probe_num_sq, ddof=1))
        rel_sem = (num_std / np.sqrt(n_probes)) / num_mean if num_mean > 0.0 else 0.0
    else:
        rel_sem = None
    return estimate, rel_sem


def _compute_residual(S_A, Z, S_B, M, residual_mode, n_probes, seed):
    if residual_mode == "exact":
        return _two_sided_residual(S_A, Z, S_B, M), {"residual_mode": "exact"}
    elif residual_mode == "sampled":
        r, rel_sem = _two_sided_residual_sampled(S_A, Z, S_B, M, n_probes=n_probes, seed=seed)
        return r, {
            "residual_mode": "sampled",
            "residual_n_probes": n_probes,
            "residual_seed": seed,
            "residual_relative_standard_error": rel_sem,
        }
    else:
        raise ValueError(f"residual_mode must be 'exact' or 'sampled', got {residual_mode!r}")


def _cholesky_jitter_sandwich(S_A, S_B, M, jitter_rcond, same_sector,
                               tsvd_rcond=None,
                               backward_error_mode="finite_only",
                               backward_error_tol=_CHOLESKY_BACKWARD_ERROR_TOL,
                               residual_mode="exact",
                               residual_n_probes=_DEFAULT_RESIDUAL_N_PROBES,
                               residual_seed=_DEFAULT_RESIDUAL_SEED):
    """S_A^-1 M S_B^-1 via pytc.df.solvers.prepare_spd_cholesky. Cholesky
    with adaptive diagonal jitter is the production solver for S^-1-type
    applications -- the same idiom TC's own orbital fitting uses, not a
    parallel implementation. Jitter schedule matches the doc's adaptive rule
    exactly: lambda_0 = jitter_rcond * trace(S)/N_mu (default 1e-14),
    x10 growth, up to 6 retries (7 total attempts).

    same_sector must be passed EXPLICITLY by the caller, who already
    knows this fact from how S_A/S_B were constructed -- no longer
    inferred via a dense jnp.array_equal(S_A, S_B) comparison, which
    synchronizes and scans the full (n, n) matrices.

    Two-tier acceptance: prepare_spd_cholesky itself gates jitter escalation on
    the REGULARIZED-solve backward error via backward_error_mode (is
    the factor actually a good factorization of mat+jitter*I? -- more
    jitter can fix this; "finite_only" default keeps this O(1) at
    production scale, "exact" is a dense O(n^3)-per-retry reference/
    small-system diagnostic, see prepare_spd_cholesky's docstring).
    This function separately checks the UNREGULARIZED bias on the real
    right-hand side M (is Z actually a good fit to the true,
    unregularized problem? -- more jitter CANNOT fix this, since jitter
    is exactly what causes the bias) and, if residual_mode="exact" and
    that check fails, falls back to solver="tsvd" rather than
    escalating jitter further -- previously this was warn-only with no
    corrective action. The fallback uses tsvd_rcond (an INDEPENDENT
    singular-value cutoff, default None -> TSVD's own machine-epsilon-
    scaled default), never jitter_rcond -- an earlier version reused
    jitter_rcond as the TSVD cutoff, which for a caller-forced
    jitter_rcond=1.0 (a valid jitter *scale* but a nonsensical
    singular-value *cutoff*) made TSVD retain ZERO modes and return
    Z=0, the fallback silently making things worse instead of better
    (measured: fallback
    residual 1.0 vs 8.4e-14 with TSVD's own default cutoff).

    When residual_mode="sampled", the fallback gate is SKIPPED entirely
    (fit_residual is still computed and reported for diagnostics) --
    the sampled estimator is not yet calibrated against the exact
    residual, so an elevated sampled value must not silently trigger a
    solver switch off an uncharacterized false-positive rate.

    Provenance labels this "unscaled_cholesky_jitter", not
    "cholesky_jitter" matching the design doc's rule exactly, since
    there is no row equilibration yet.

    Returns (Z, provenance) -- provenance carries exactly the
    solver-level fields compute_Z/compute_Z_cross can actually observe
    (solver, jitter/retries, backward-error mode/tolerance, dtype,
    two-sided fit residual + its norm convention + acceptance
    threshold, row-scaling, fallback status); fields that need
    caller-side context (kernel policy, upstream SCF/grid provenance)
    are NOT compute_Z's to fabricate -- a higher-level build_core
    wrapper assembles those -- part of the core-build contract, not
    deferred to the CCSD step.
    """
    S_A = jnp.asarray(S_A)
    S_B = jnp.asarray(S_B)
    M = jnp.asarray(M)
    if S_A.dtype in (jnp.float32, jnp.complex64):
        logger.warning(
            f"compute_Z (unscaled_cholesky_jitter): resolved dtype is {S_A.dtype} -- JAX "
            f"defaults to float32 SILENTLY unless the caller has enabled "
            f"jax.config.update('jax_enable_x64', True), downcasting float64 numpy inputs "
            f"without any error. Measured impact on the H2O/cc-pVDZ ov-sector checkpoint: "
            f"ERI relative error 0.51% (float32) vs 2.0e-4 (float64), a ~25x precision "
            f"loss -- verify x64 is enabled before trusting production numbers from this path."
        )

    chol_A, lower_A, jitter_A, tries_A = prepare_spd_cholesky(
        S_A, rcond=jitter_rcond, max_jitter_tries=_CHOLESKY_JITTER_MAX_TRIES,
        backward_error_mode=backward_error_mode, backward_error_tol=backward_error_tol)
    if same_sector:
        chol_B, lower_B, jitter_B, tries_B = chol_A, lower_A, jitter_A, tries_A
    else:
        chol_B, lower_B, jitter_B, tries_B = prepare_spd_cholesky(
            S_B, rcond=jitter_rcond, max_jitter_tries=_CHOLESKY_JITTER_MAX_TRIES,
            backward_error_mode=backward_error_mode, backward_error_tol=backward_error_tol)

    X = jsp_linalg.cho_solve((chol_A, lower_A), M)  # S_A^-1 M
    Z = jsp_linalg.cho_solve((chol_B, lower_B), X.conj().T).conj().T  # (S_A^-1 M) S_B^-1
    Z_np = np.asarray(Z)

    if residual_mode == "sampled":
        # S_A/S_B/M are already jnp (converted at function entry) and Z
        # is a fresh jnp array from cho_solve above -- pass them through
        # UNCONVERTED so the sampled estimator's on-device batched
        # matvecs never see a host round trip. An earlier version
        # called np.asarray on all four here unconditionally, so sampled
        # mode inherited a device->host->device round trip through
        # _two_sided_residual_sampled's own jnp.asarray -- exactly the
        # transfer the on-device rewrite was meant to remove.
        fit_residual, residual_meta = _compute_residual(
            S_A, Z, S_B, M, residual_mode, residual_n_probes, residual_seed)
    else:
        fit_residual, residual_meta = _compute_residual(
            np.asarray(S_A), Z_np, np.asarray(S_B), np.asarray(M),
            residual_mode, residual_n_probes, residual_seed)

    provenance = {
        "solver": "unscaled_cholesky_jitter",
        "cutoff": jitter_rcond,
        "jitter_used": (jitter_A, jitter_A) if same_sector else (jitter_A, jitter_B),
        "n_tries": (tries_A, tries_A) if same_sector else (tries_A, tries_B),
        "backward_error_mode": backward_error_mode,
        "backward_error_tol": backward_error_tol if backward_error_mode == "exact" else None,
        "retained_singular_value_range": None,  # not applicable to this solver
        "dtype": str(Z_np.dtype),
        "fit_residual": fit_residual,
        "residual_norm_convention": _RESIDUAL_NORM_CONVENTION_SAME if same_sector else _RESIDUAL_NORM_CONVENTION_CROSS,
        "residual_warn_threshold": _RESIDUAL_WARN_THRESHOLD,
        "row_scaling": "identity",
        "fallback_triggered": False,
        "fallback_gating_applicable": residual_mode == "exact",
        **residual_meta,
    }

    if fit_residual > _RESIDUAL_WARN_THRESHOLD:
        if residual_mode != "exact":
            logger.warning(
                f"compute_Z (unscaled_cholesky_jitter): {residual_mode} fit residual "
                f"{fit_residual:.3e} exceeds acceptance threshold {_RESIDUAL_WARN_THRESHOLD:.0e} "
                f"-- NOT triggering the TSVD fallback because residual_mode={residual_mode!r} "
                f"is diagnostic-only and not yet calibrated for fallback gating; only "
                f"residual_mode='exact' drives the automatic fallback."
            )
            return Z_np, provenance
        logger.warning(
            f"compute_Z (unscaled_cholesky_jitter): two-sided fit residual {fit_residual:.3e} "
            f"exceeds acceptance threshold {_RESIDUAL_WARN_THRESHOLD:.0e} (jitter_A={jitter_A:.3e}, "
            f"tries_A={tries_A}) -- this reflects UNREGULARIZED bias, not a factorization "
            f"failure (the regularized-solve backward error is separately gated inside "
            f"prepare_spd_cholesky's own retry loop); more jitter cannot reduce this bias, so "
            f"falling back to solver='tsvd' (independent tsvd_rcond={tsvd_rcond}) rather than "
            f"escalating jitter further."
        )
        tsvd_Z, tsvd_provenance = _tsvd_sandwich(
            S_A, S_B, M, tsvd_rcond, same_sector,
            residual_mode=residual_mode, residual_n_probes=residual_n_probes,
            residual_seed=residual_seed)
        tsvd_provenance["fallback_triggered"] = True
        tsvd_provenance["fallback_reason"] = (
            f"unscaled_cholesky_jitter unregularized-bias residual {fit_residual:.3e} exceeded "
            f"acceptance threshold {_RESIDUAL_WARN_THRESHOLD:.0e}"
        )
        tsvd_provenance["preceding_cholesky_jitter_used"] = provenance["jitter_used"]
        tsvd_provenance["preceding_cholesky_fit_residual"] = fit_residual
        return tsvd_Z, tsvd_provenance

    return Z_np, provenance


def _tsvd_sandwich(S_A, S_B, M, tsvd_rcond, same_sector,
                    residual_mode="exact",
                    residual_n_probes=_DEFAULT_RESIDUAL_N_PROBES,
                    residual_seed=_DEFAULT_RESIDUAL_SEED):
    """S_A^+ M S_B^+ via an EXPLICIT truncated-SVD pseudoinverse (not
    np.linalg.pinv's black box) so the retained singular-value range can
    be reported in provenance -- the diagnostics/fallback solver mode
    per design doc §4 (production default is cholesky_jitter). Also
    serves as _cholesky_jitter_sandwich's automatic fallback when the
    unregularized-bias check fails there, using its OWN independent
    tsvd_rcond (never the Cholesky jitter_rcond -- see
    _cholesky_jitter_sandwich's docstring).

    same_sector must be passed EXPLICITLY by the caller (see
    _cholesky_jitter_sandwich's docstring -- no dense
    np.array_equal(S_A, S_B) inference).
    """
    S_A = np.asarray(S_A)
    S_B = np.asarray(S_B)
    M = np.asarray(M)
    rcond = tsvd_rcond

    def _tsvd_pinv_and_range(S):
        s_vals, U = np.linalg.eigh(0.5 * (S + S.conj().T))  # Hermitian PSD: eigh == SVD up to sign
        s_vals = np.clip(s_vals, 0.0, None)
        cutoff = (rcond if rcond is not None else np.finfo(S.dtype).eps * max(S.shape)) * s_vals.max()
        keep = s_vals > cutoff
        inv_vals = np.where(keep, 1.0 / np.where(keep, s_vals, 1.0), 0.0)
        S_inv = (U * inv_vals) @ U.conj().T
        retained = s_vals[keep]
        sv_range = (float(retained.min()), float(retained.max())) if retained.size else (0.0, 0.0)
        return S_inv, sv_range, int(keep.sum())

    S_A_inv, sv_range_A, n_retained_A = _tsvd_pinv_and_range(S_A)
    if same_sector:
        S_B_inv, sv_range_B, n_retained_B = S_A_inv, sv_range_A, n_retained_A
    else:
        S_B_inv, sv_range_B, n_retained_B = _tsvd_pinv_and_range(S_B)

    Z = S_A_inv @ M @ S_B_inv
    fit_residual, residual_meta = _compute_residual(
        S_A, Z, S_B, M, residual_mode, residual_n_probes, residual_seed)
    if fit_residual > _RESIDUAL_WARN_THRESHOLD:
        logger.warning(
            f"compute_Z (tsvd): two-sided fit residual {fit_residual:.3e} exceeds the "
            f"design-doc acceptance threshold {_RESIDUAL_WARN_THRESHOLD:.0e} -- this is a "
            f"WARNING, not a hard rejection (full acceptance also requires the downstream "
            f"ERI/energy spot-check, which this function cannot see); downstream results "
            f"should be treated with extra suspicion."
        )

    provenance = {
        "solver": "tsvd",
        "cutoff": rcond,
        "retained_singular_value_range": (sv_range_A, sv_range_A) if same_sector else (sv_range_A, sv_range_B),
        "n_retained": (n_retained_A, n_retained_A) if same_sector else (n_retained_A, n_retained_B),
        "dtype": str(Z.dtype),
        "fit_residual": fit_residual,
        "residual_norm_convention": _RESIDUAL_NORM_CONVENTION_SAME if same_sector else _RESIDUAL_NORM_CONVENTION_CROSS,
        "residual_warn_threshold": _RESIDUAL_WARN_THRESHOLD,
        "row_scaling": "identity",
        "fallback_triggered": False,
        **residual_meta,
    }
    return Z, provenance


def compute_Z(P, C, rcond=None, solver="cholesky_jitter",
              tsvd_rcond=None,
              backward_error_mode="finite_only",
              backward_error_tol=_CHOLESKY_BACKWARD_ERROR_TOL,
              residual_mode="exact",
              residual_n_probes=_DEFAULT_RESIDUAL_N_PROBES,
              residual_seed=_DEFAULT_RESIDUAL_SEED):
    """Z = S^-1 C C^dagger S^-1, S = P P^dagger.

    Production solver is Cholesky with adaptive diagonal jitter
    -- decided on measured evidence, not assumed: at production core sizes an SVD is a
    GPU non-starter, pivot budgets are analytically pre-capped so S
    enters the solve at (near-)full rank by construction, and the
    apparent earlier need for delicate pinv-cutoff tuning was itself an
    artifact of a since-fixed bug (weighted, not raw, values in P --
    see pair_collocation_at_pivots's docstring). solver="tsvd" is kept
    as the small/medium-system diagnostic oracle and fallback -- an
    EXPLICIT truncated-SVD pseudoinverse (not np.linalg.pinv's black
    box) so the retained singular-value range can be reported.

    History: this function originally took WEIGHTED (sqrt(w_g)-scaled)
    values into pair_collocation_at_pivots's P, and needed a hand-tuned,
    NON-MONOTONIC rcond sweet spot (default 1e-6) to avoid a real
    catastrophic failure mode -- numpy's own default pinv rcond gave
    39x relative error (nonsense) on an H2O/cc-pVDZ ov-sector test
    (n_pivots=300, n_pair=95, cond(S)~2e24), and rcond tightened much
    past ~3e-7 made it WORSE again (readmitted noise). Root cause: P
    must be built from RAW (unweighted) values -- weighting is a
    pivot-SELECTION device, not part of the interpolation formula. With that fix, S on the same test has rank EXACTLY equal
    to n_pair=95 (no numerical rank inflation from the weight scaling)
    and the rcond sweep becomes well-behaved.

    Args:
        P: (n_pivots, n_pair) pair-collocation matrix at the pivots
            (RAW/unweighted values -- see pair_collocation_at_pivots).
        C: (n_pivots, n_aux) from compute_C_streamed.
        rcond: Solver-specific meaning -- for "cholesky_jitter", the
            STARTING relative jitter scale (forwarded to
            prepare_spd_cholesky, default 1e-14, auto-escalated if
            needed); for "tsvd", the relative singular-value cutoff
            (None uses a machine-epsilon-scaled default, matching
            numpy's own pinv convention).
        solver: "cholesky_jitter" (default, production) or "tsvd"
            (diagnostic/fallback).
        tsvd_rcond: INDEPENDENT singular-value cutoff used only if
            solver="cholesky_jitter" triggers an automatic fallback to
            TSVD (see _cholesky_jitter_sandwich's docstring for why this
            must NOT reuse rcond -- they are different quantities:
            jitter scale vs. singular-value cutoff). None (default)
            uses TSVD's own machine-epsilon-scaled default. Ignored
            when solver="tsvd" is requested directly -- there, rcond
            itself is the cutoff, unchanged from before.
        backward_error_mode, backward_error_tol: forwarded to
            prepare_spd_cholesky when solver="cholesky_jitter" -- see
            its docstring. Default "finite_only" is production-safe
            (O(1)); "exact" is a dense O(n^3)-per-retry reference/
            small-system diagnostic.
        residual_mode: "exact" (default, O(n^3) dense two-sided
            residual, drives the automatic TSVD fallback) or "sampled"
            (O(n^2 * residual_n_probes) on-device Hutchinson estimate --
            diagnostic-only, NEVER drives the fallback, NOT yet
            validated for decision-grade use, see
            _two_sided_residual_sampled's docstring).
        residual_n_probes, residual_seed: forwarded to the sampled
            estimator when residual_mode="sampled"; ignored otherwise.

    Returns:
        (Z, provenance): Z is (n_pivots, n_pivots); provenance is a
        dict with the solver-level fields from design doc §4 this
        function can observe directly (solver, jitter/cutoff + retry
        history or retained singular-value range, backward-error mode,
        fit residual, row-scaling, fallback status, residual_mode).
        Fields needing caller-side context (kernel policy, upstream
        SCF/grid provenance, pivot-index hashes) are NOT fabricated
        here -- assemble those at the call site that actually has them.
    """
    P = np.asarray(P)
    C = np.asarray(C)
    S = P @ P.conj().T
    M = C @ C.conj().T
    # compute_Z is always the same-sector case by construction (one P/C
    # pair fitted against itself) -- same_sector=True is a fact, not an
    # inference.
    if solver == "cholesky_jitter":
        rcond_eff = 1e-14 if rcond is None else rcond
        return _cholesky_jitter_sandwich(S, S, M, rcond_eff, True,
                                          tsvd_rcond=tsvd_rcond,
                                          backward_error_mode=backward_error_mode,
                                          backward_error_tol=backward_error_tol,
                                          residual_mode=residual_mode,
                                          residual_n_probes=residual_n_probes,
                                          residual_seed=residual_seed)
    elif solver == "tsvd":
        return _tsvd_sandwich(S, S, M, rcond, True,
                               residual_mode=residual_mode,
                               residual_n_probes=residual_n_probes,
                               residual_seed=residual_seed)
    else:
        raise ValueError(f"solver must be 'cholesky_jitter' or 'tsvd', got {solver!r}")


def compute_Z_cross(P_A, C_A, P_B, C_B, rcond=None, solver="cholesky_jitter",
                     same_sector=False,
                     tsvd_rcond=None,
                     backward_error_mode="finite_only",
                     backward_error_tol=_CHOLESKY_BACKWARD_ERROR_TOL,
                     residual_mode="exact",
                     residual_n_probes=_DEFAULT_RESIDUAL_N_PROBES,
                     residual_seed=_DEFAULT_RESIDUAL_SEED):
    """Z_AB = S_A^-1 C_A C_B^dagger S_B^-1, S_A = P_A P_A^dagger, S_B = P_B
    P_B^dagger -- the cross-sector generalization of compute_Z, needed
    when the ERI block's bra and ket pair indices come from DIFFERENT
    MO-pair sectors with their OWN independently-selected pivot sets
    (e.g. CCSD's oo|vv and ov|vv blocks: sector A's pivots need not
    equal, or even overlap with, sector B's pivots -- see
    pivot_selection.select_pivots_oo_ov_vv, which selects oo/ov/vv
    pivots independently).

    compute_Z(P, C, rcond, solver) is exactly this function's
    same-sector special case (P_A=P_B=P, C_A=C_B=C); kept as a separate
    simpler entry point since same-sector Z is CCSD's most common need
    (oo|oo, ov|ov, vv|vv) and callers there shouldn't have to pass every
    argument twice. See compute_Z's docstring for the solver choice and
    provenance fields -- identical here.

    Derivation: with V_AB the exact (n_pair_A, n_pair_B) ERI block
    between sectors A and B, and B_A/B_B the DF Cholesky factors
    restricted to each sector's MO-pair space (both built from the SAME
    3-center integrals via compute_C_streamed, just different
    mo_coeff_p/mo_coeff_q), C_A = P_A B_A^dagger and C_B = P_B
    B_B^dagger give C_A C_B^dagger = P_A (B_A^dagger B_B) P_B^dagger =
    P_A V_AB P_B^dagger -- a pure algebraic identity, independent of
    any ISDF approximation quality (mirrors compute_Z's own
    C C^dagger = P V P^dagger identity, the basis of this module's
    test_C_streamed_matches_direct_PVPdagger regression test).

    Args:
        P_A: (n_pivots_A, n_pair_A) pair-collocation matrix at sector
            A's pivots (RAW factors, see pair_collocation_at_pivots).
        C_A: (n_pivots_A, n_aux) from compute_C_streamed for sector A.
        P_B: (n_pivots_B, n_pair_B) pair-collocation matrix at sector
            B's pivots.
        C_B: (n_pivots_B, n_aux) from compute_C_streamed for sector B
            -- must share the SAME n_aux axis as C_A (same mf, same
            auxbasis).
        rcond, solver, tsvd_rcond, backward_error_mode, backward_error_tol,
        residual_mode, residual_n_probes, residual_seed: see compute_Z's
            docstring.
        same_sector: Whether sector A and sector B are the SAME sector
            (same pivots/P/C, just passed twice) -- an EXPLICIT fact
            the caller must supply. Defaults to False (the safe,
            always-correct choice: independent factorization of S_A and
            S_B) rather than guessing -- compute_Z already covers the
            common same-sector convenience path, so pass same_sector=
            True here only when the caller genuinely knows P_A/C_A and
            P_B/C_B come from the same pivot set (a prior
            ``same_sector=None`` default
            fell back to a ``P_A is P_B`` identity check performed
            AFTER ``np.asarray`` conversion -- for JAX/CuPy array
            inputs, np.asarray creates a NEW host object on every call,
            so two calls passing the literal same underlying device
            array never satisfied ``is`` post-conversion, silently
            defeating the fast path for exactly the array type this
            pipeline is built on).

    Returns:
        (Z_AB, provenance): Z_AB is (n_pivots_A, n_pivots_B); provenance
        as in compute_Z, with jitter/retained-range/n_retained fields
        as (A, B) pairs when sectors A and B are genuinely different.
    """
    P_A = np.asarray(P_A)
    C_A = np.asarray(C_A)
    P_B = np.asarray(P_B)
    C_B = np.asarray(C_B)
    S_A = P_A @ P_A.conj().T
    S_B = P_B @ P_B.conj().T
    M = C_A @ C_B.conj().T
    if solver == "cholesky_jitter":
        rcond_eff = 1e-14 if rcond is None else rcond
        return _cholesky_jitter_sandwich(S_A, S_B, M, rcond_eff, same_sector,
                                          tsvd_rcond=tsvd_rcond,
                                          backward_error_mode=backward_error_mode,
                                          backward_error_tol=backward_error_tol,
                                          residual_mode=residual_mode,
                                          residual_n_probes=residual_n_probes,
                                          residual_seed=residual_seed)
    elif solver == "tsvd":
        return _tsvd_sandwich(S_A, S_B, M, rcond, same_sector,
                               residual_mode=residual_mode,
                               residual_n_probes=residual_n_probes,
                               residual_seed=residual_seed)
    else:
        raise ValueError(f"solver must be 'cholesky_jitter' or 'tsvd', got {solver!r}")


def reconstruct_eri_block(P_row, Z, P_col):
    """V ~= P_row^dagger Z P_col.

    Z must match P_row's and P_col's pivot sets: for a SAME-sector
    block (e.g. ov|ov, both bra and ket from the "ov" sector's own
    pivots), pass P_row=P_col=that sector's P and Z=compute_Z(P, C).
    For a CROSS-sector block (e.g. oo|vv, ov|vv -- bra and ket from
    two INDEPENDENTLY pivoted sectors, see
    pivot_selection.select_pivots_oo_ov_vv), pass P_row from sector A,
    P_col from sector B, and Z=compute_Z_cross(P_A, C_A, P_B, C_B) --
    a same-sector Z (square, built from one sector's own P/C) is NOT
    interchangeable with a cross-sector one (rectangular in general,
    built from both sectors' P/C together); passing mismatched P_row/
    P_col/Z shapes will fail at the matmul: a same-sector Z does not
    serve an arbitrary P_row/P_col pairing.

    Args:
        P_row: (n_pivots_A, n_pair_row) pair-collocation at sector A's
            pivots, for the ERI's bra pair index.
        Z: (n_pivots_A, n_pivots_B) from compute_Z (n_pivots_A ==
            n_pivots_B, same-sector) or compute_Z_cross (general case).
        P_col: (n_pivots_B, n_pair_col) pair-collocation at sector B's
            pivots, for the ERI's ket pair index.

    Returns:
        (n_pair_row, n_pair_col) reconstructed ERI block (flattened
        pair axis -- reshape to the caller's own (p, q) shape).
    """
    P_row = np.asarray(P_row)
    P_col = np.asarray(P_col)
    return P_row.conj().T @ Z @ P_col
