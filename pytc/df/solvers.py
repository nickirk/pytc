"""Generic, model-agnostic linear-solve primitives for structured
least-squares and SPD systems, shared by orbital fitting and the
Coulomb path's S-solve."""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from functools import partial
import logging

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
    (C^dagger B)[m, g] = (sum_p conj(phi_piv_p[p,m])*phi_p_batch[p,g])
                        * (sum_q conj(phi_piv_q[q,m])*phi_q_batch[q,g])

    Complex-correct: least squares needs the CONJUGATE transpose C^dagger,
    not the plain transpose C^T -- an earlier version of this function used
    `.T` throughout, which is only correct for real inputs and silently
    gives a wrong (non-Hermitian, not-necessarily-PSD normal-equation
    matrix) answer for complex ones (measured against a dense
    C[(p,q),mu]/B[(p,q),g] + np.linalg.lstsq oracle: relative error ~2.0
    with plain `.T`, 6.4e-10 with `.conj().T`). For real inputs `.conj()`
    is a no-op, so every existing real-valued caller (TC/xTC's own ISDF
    fitting) is bit-identically unaffected by this fix.

    Args:
        phi_piv_p: (n_orb, n_fused) first factor of pivots
        phi_piv_q: (n_orb, n_fused) second factor of pivots
        phi_p_batch: (n_orb, batch_size) first factor of target
        phi_q_batch: (n_orb, batch_size) second factor of target
        rcond: Relative regularization strength (default 1e-14)

    Returns:
        X: (n_fused, batch_size) solutions
    """
    # Compute C^dagger C efficiently using the Kronecker-like structure
    gram_p = phi_piv_p.conj().T @ phi_piv_p  # (n_fused, n_fused)
    gram_q = phi_piv_q.conj().T @ phi_piv_q  # (n_fused, n_fused)
    ATA = gram_p * gram_q  # Element-wise product -- Hermitian PSD (Hadamard product of two Hermitian PSD matrices)

    # Compute C^dagger B efficiently using separable structure
    term_p = jnp.matmul(phi_piv_p.conj().T, phi_p_batch)  # (n_fused, batch_size)
    term_q = jnp.matmul(phi_piv_q.conj().T, phi_q_batch)  # (n_fused, batch_size)
    ATB = term_p * term_q  # (n_fused, batch_size)

    # Use LU solve (jnp.linalg.solve) with Tikhonov regularization.
    # ATA's diagonal is exactly real in exact arithmetic (Hermitian PSD);
    # .real guards against a spurious ~1e-16-scale imaginary rounding
    # residual feeding into the (real-valued-by-construction) jitter scale.
    diag_mean = jnp.mean(jnp.diag(ATA)).real
    jitter = diag_mean * rcond
    ATA_reg = ATA + jitter * jnp.eye(ATA.shape[0])
    X = jnp.linalg.solve(ATA_reg, ATB)

    return X


solve_normal_equations_batch = jax.jit(solve_normal_equations_batch, static_argnames=['rcond'])


@jax.jit
def _build_normal_matrix(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray) -> jnp.ndarray:
    """Build unregularized normal-equation matrix C^dagger C for structured
    LS -- conjugate transpose, not plain transpose (see
    solve_normal_equations_batch's docstring for the complex-correctness
    fix this is part of; real inputs are unaffected, .conj() is a no-op)."""
    gram_p = phi_piv_p.conj().T @ phi_piv_p
    gram_q = phi_piv_q.conj().T @ phi_piv_q
    return gram_p * gram_q


_CHOLESKY_BACKWARD_ERROR_TOL = 1e-6  # relative Frobenius residual, regularized system


def _cholesky_backward_error(chol, lower, mat_reg):
    """Relative Frobenius residual of the Cholesky factor against the
    REGULARIZED matrix it was asked to factor -- ||L L^dagger - mat_reg||
    / ||mat_reg|| (or U^dagger U for upper). LAPACK's cho_factor is
    backward-stable by construction whenever it returns without error,
    so this is a modest additional safety net against a factor that is
    finite but numerically garbage (e.g. under catastrophic cancellation
    at extreme ill-conditioning), not a substitute for the isfinite
    check. This is the REGULARIZED-solve half and it gates jitter
    escalation; the separate UNREGULARIZED
    bias check lives downstream in df/fit.py's _cholesky_jitter_sandwich,
    since only that caller has the actual right-hand side M needed to
    measure it, and it must NEVER be "fixed" by more jitter here (more
    jitter only increases unregularized bias)."""
    if lower:
        tri = jnp.tril(chol)
        recon = tri @ tri.conj().T
    else:
        tri = jnp.triu(chol)
        recon = tri.conj().T @ tri
    num = jnp.linalg.norm(recon - mat_reg)
    den = jnp.linalg.norm(mat_reg)
    den_f = float(den)
    return float(num / den) if den_f > 0.0 else float(num)


def prepare_spd_cholesky(matrix: jnp.ndarray, rcond: float = 1e-14,
                          max_jitter_tries: int = 8, jitter_growth: float = 10.0,
                          backward_error_mode: str = "finite_only",
                          backward_error_tol: float = _CHOLESKY_BACKWARD_ERROR_TOL):
    """Adaptive-jitter Cholesky factorization for a symmetric/Hermitian
    positive-SEMIdefinite matrix -- the shared "Cholesky + adaptive
    diagonal jitter" primitive both TC's own fitting (via
    prepare_normal_equations_solver, now a thin wrapper around this) and
    the Coulomb path's S-solve use -- decided on measured evidence,
    not a default.

    Starts from a small base jitter (max of a diag-mean*rcond estimate
    and a PURELY matrix-relative machine-epsilon floor) and
    geometrically escalates it until the Cholesky factor passes the
    requested acceptance check -- handles matrices that are SPD in
    exact arithmetic but numerically indefinite/near-singular (e.g. a
    near-full-rank pivot-selection Gram matrix) without ever going
    through an SVD, which is a non-starter at production core sizes on
    GPU.

    backward_error_mode="finite_only" (default, production-safe): O(1)
    isfinite check on the factor only -- matches the original b59c6ce
    escalation rule. "exact" additionally reconstructs L L^dagger (or
    U^dagger U) and checks its Frobenius residual against the
    regularized matrix -- a dense O(n^3) operation PER RETRY ATTEMPT,
    which at production N_mu recreates exactly the production-scalable
    residual-estimation cost problem at the fit layer; use "exact" only
    for small/reference-system diagnostics, never at production scale.

    The unregularized-bias check lives in _cholesky_jitter_sandwich, which
    has the fit context this generic primitive lacks. There is no TSVD
    fallback any more: it was removed 2026-07-29 after being measured
    harmful on real periodic data, so a failed bias check is reported and
    not acted on. This solver's provenance label
    is "unscaled_cholesky_jitter" (not "cholesky_jitter" matching the
    doc rule exactly) until row equilibration exists. Production
    default is declared only after the task-#3-mandated benchmark at
    representative N_mu.

    Args:
        matrix: (n, n) symmetric/Hermitian PSD matrix.
        rcond: Relative jitter scale (fraction of the matrix's own
            diagonal mean) used as the STARTING jitter before escalation.
        max_jitter_tries: Escalation attempts before giving up.
        jitter_growth: Geometric growth factor per escalation attempt.
        backward_error_mode: "finite_only" (default) or "exact" -- see
            above. Callers that need this in their own provenance
            (e.g. df/fit.py) must record it themselves; it is not
            returned here (the caller already knows what it asked for).
        backward_error_tol: Relative Frobenius residual tolerance used
            only when backward_error_mode="exact".

    Returns:
        (chol, lower, jitter_used, n_tries): chol/lower are cho_factor's
        own outputs (pass to jsp_linalg.cho_solve); jitter_used is the
        final (possibly escalated) jitter value actually applied;
        n_tries is how many attempts it took (1 = no escalation needed)
        -- both are provenance fields for callers that need to record
        the solver's own diagnostics (Coulomb path's compute_Z).

    Raises:
        ValueError: matrix's diagonal mean is not finite, or
            backward_error_mode is neither "finite_only" nor "exact".
        numpy.linalg.LinAlgError: matrix's diagonal mean is
            non-positive (not PSD -- previously masked by an absolute
            eps*max(diag_mean, 1.0) floor that injected eps-scale
            jitter regardless of the matrix's own scale),
            or the factor never passes the requested check within
            max_jitter_tries attempts.
    """
    if backward_error_mode not in ("finite_only", "exact"):
        raise ValueError(
            f"backward_error_mode must be 'finite_only' or 'exact', got {backward_error_mode!r}"
        )
    mat = 0.5 * (matrix + matrix.conj().T)
    diag_mean = float(jnp.mean(jnp.real(jnp.diag(mat))))
    if not np.isfinite(diag_mean):
        raise ValueError(
            f"prepare_spd_cholesky: matrix diagonal mean is not finite ({diag_mean}); "
            f"cannot form a scale-relative jitter."
        )
    if diag_mean <= 0.0:
        raise np.linalg.LinAlgError(
            f"prepare_spd_cholesky: matrix diagonal mean is non-positive ({diag_mean:.3e}); "
            f"matrix is not PSD, cannot form a scale-relative jitter."
        )
    eps = float(jnp.finfo(mat.dtype).eps)
    eps_scale = eps * diag_mean  # PURELY matrix-relative -- no absolute 1.0 floor
    base_jitter = max(diag_mean * rcond, eps_scale)
    eye = jnp.eye(mat.shape[0], dtype=mat.dtype)

    last_chol = None
    last_backward_error = None
    for attempt in range(max_jitter_tries):
        jitter = base_jitter * (jitter_growth ** attempt)
        mat_reg = mat + jitter * eye
        chol, lower = jsp_linalg.cho_factor(mat_reg, lower=True)
        if bool(jnp.all(jnp.isfinite(chol))):
            if backward_error_mode == "exact":
                backward_error = _cholesky_backward_error(chol, lower, mat_reg)
                accepted = backward_error <= backward_error_tol
                last_backward_error = backward_error
            else:
                accepted = True
            if accepted:
                if attempt > 0:
                    logger.warning(
                        "Cholesky jitter escalated: base=%.3e final=%.3e tries=%d "
                        "backward_error_mode=%s",
                        base_jitter, jitter, attempt + 1, backward_error_mode
                    )
                return chol, bool(lower), float(jitter), attempt + 1
        last_chol = chol

    raise np.linalg.LinAlgError(
        f"Adaptive Cholesky failed after {max_jitter_tries} tries; "
        f"base_jitter={base_jitter:.3e}, last_nonfinite={bool(jnp.any(jnp.isnan(last_chol)))}, "
        f"backward_error_mode={backward_error_mode}, last_backward_error={last_backward_error}"
    )


def prepare_normal_equations_solver(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                    rcond: float = 1e-14,
                                    max_jitter_tries: int = 8,
                                    jitter_growth: float = 10.0,
                                    return_info: bool = False):
    """Prepare robust Cholesky factor for repeated batched solves.

    Thin wrapper around prepare_spd_cholesky (TC's original entry point,
    preserved with its existing (chol, lower)-only return contract --
    see prepare_spd_cholesky's docstring for the shared jitter-escalation
    algorithm and its own 4-tuple return, used by callers that need the
    extra provenance fields).

    Args:
        return_info: False (default) -- backward-compatible (chol, lower)
            2-tuple, unchanged. True -- returns the full
            (chol, lower, jitter_used, n_tries) 4-tuple prepare_spd_cholesky
            itself produces, instead of discarding the last two (added for
            provenance requirements -- a caller building a
            provenance-carrying artifact needs the ACTUAL jitter/retry
            facts, not just the usable factor).
    """
    ata = _build_normal_matrix(phi_piv_p, phi_piv_q)
    chol, lower, jitter_used, n_tries = prepare_spd_cholesky(
        ata, rcond=rcond, max_jitter_tries=max_jitter_tries, jitter_growth=jitter_growth)
    if return_info:
        return chol, lower, jitter_used, n_tries
    return chol, lower


@partial(jax.jit, static_argnames=('lower',))
def solve_normal_equations_batch_prepared(chol: jnp.ndarray, lower: bool,
                                          phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                          phi_p_batch: jnp.ndarray, phi_q_batch: jnp.ndarray) -> jnp.ndarray:
    """Solve batched normal equations using precomputed Cholesky factor.

    Conjugate transpose (`.conj().T`), not plain transpose -- see
    solve_normal_equations_batch's docstring for the complex-correctness
    fix this is part of; real inputs are unaffected (.conj() is a no-op).
    """
    term_p = jnp.matmul(phi_piv_p.conj().T, phi_p_batch)
    term_q = jnp.matmul(phi_piv_q.conj().T, phi_q_batch)
    atb = term_p * term_q
    return jsp_linalg.cho_solve((chol, lower), atb)


_RETENTION_MARGINAL_NEAR_CUTOFF_FACTOR = 10.0
_RETENTION_MARGINAL_COND_THRESHOLD = 1e3


def _check_retention_marginal(s_max, s_min_retained, threshold, rtol, *, caller,
                              s_first_discarded=None, pinned=False):
    """Warn loudly when a solve's retained subspace is near the danger
    regime: flagged when the smallest retained eigenvalue is within 10x of
    the retention cutoff AND the retained condition number exceeds 1e3.
    The cutoff is rtol*s_max, or -- when pinned -- the first discarded
    eigenvalue s_{K+1} (the spectral gap at the pin); a pinned solve with
    nothing discarded has no edge and cannot be marginal. See design doc §5.

    Returns:
        (retention_marginal, cond_pi): cond_pi is None when
        s_min_retained is None (n_retained == 0).
    """
    if s_min_retained is None or s_min_retained <= 0.0:
        return False, None
    cond_pi = s_max / s_min_retained
    if pinned:
        if s_first_discarded is None or s_first_discarded <= 0.0:
            return False, cond_pi
        near_cutoff = (
            s_min_retained < _RETENTION_MARGINAL_NEAR_CUTOFF_FACTOR * s_first_discarded
        )
    else:
        near_cutoff = (
            s_min_retained < _RETENTION_MARGINAL_NEAR_CUTOFF_FACTOR * threshold
        )
    high_cond = cond_pi > _RETENTION_MARGINAL_COND_THRESHOLD
    marginal = bool(near_cutoff and high_cond)
    if marginal and pinned:
        logger.warning(
            f"{caller}: retention_marginal -- pinned retention edge "
            f"s_K={s_min_retained:.3e} is within "
            f"{_RETENTION_MARGINAL_NEAR_CUTOFF_FACTOR:.0f}x of the first discarded "
            f"eigenvalue s_(K+1)={s_first_discarded:.3e} (a narrow spectral gap at "
            f"the pin), and the retained subspace's condition number ({cond_pi:.3e}) "
            f"exceeds {_RETENTION_MARGINAL_COND_THRESHOLD:.0e} -- the pin sits in a "
            f"dense, ill-conditioned region of the spectrum."
        )
    elif marginal:
        logger.warning(
            f"{caller}: retention_marginal -- smallest retained eigenvalue "
            f"{s_min_retained:.3e} is within {_RETENTION_MARGINAL_NEAR_CUTOFF_FACTOR:.0f}x "
            f"of the rtol={rtol:.1e} cutoff ({threshold:.3e}), and the retained "
            f"subspace's condition number ({cond_pi:.3e}) exceeds "
            f"{_RETENTION_MARGINAL_COND_THRESHOLD:.0e} -- this solve is in the danger "
            f"regime the old, over-loose rtol default used to blow up in silently; "
            f"consider a tighter rtol for this system if downstream results look off."
        )
    return marginal, cond_pi


def _solve_info_from_core_output(
    n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
    v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
    s_first_discarded, n, dtype, rtol, *, caller, retention_mode="single",
    n_retained_pin=None,
):
    """Build the device-solve info dict from _hermitian_sandwich_solve_core's
    raw (traced) return values. Shared by hermitian_sandwich_solve_device and
    apply_kernel_and_solve_device's fused path -- both wrap the same core.
    rtol is None exactly when n_retained_pin is set (mutually exclusive)."""
    n_retained_i = int(n_retained)
    s_max_f = float(s_max)
    s_min_retained_f = None if n_retained_i == 0 else float(s_min_retained)
    s_first_discarded_f = None
    if n_retained_pin is not None and n_retained_pin < n:
        s_first_discarded_f = float(s_first_discarded)
    threshold = None if rtol is None else rtol * s_max_f
    retention_marginal, cond_pi = _check_retention_marginal(
        s_max_f, s_min_retained_f, threshold, rtol, caller=caller,
        s_first_discarded=s_first_discarded_f, pinned=n_retained_pin is not None,
    )
    return {
        "n_retained": n_retained_i,
        "n_discarded": n - n_retained_i,
        "s_max": s_max_f,
        "s_min_retained": s_min_retained_f,
        "pi_anti_hermitian_residual": float(pi_anti_hermitian_residual),
        "v_anti_hermitian_residual": float(v_anti_hermitian_residual),
        "retained_solve_residual": float(retained_solve_residual),
        "truncation_residual": float(truncation_residual),
        "rtol": rtol,
        "retention_mode": retention_mode,
        "adaptive_retention_used": False,
        "target_truncation_residual": None,
        "n_retained_pin": n_retained_pin,
        "retention_marginal": retention_marginal,
        "cond_pi_retained": cond_pi,
        "dtype": str(dtype),
        "backend": "jax",
    }


_RESIDUAL_WARN_THRESHOLD = 1e-10  # design doc v1.1 §4: acceptance = 1e-10 fit residual
_RESIDUAL_NORM_CONVENTION_SAME = "||S Z S - C C^dagger|| / ||C C^dagger||"
_RESIDUAL_NORM_CONVENTION_CROSS = "||S_A Z_AB S_B - C_A C_B^dagger|| / ||C_A C_B^dagger||"
_CHOLESKY_JITTER_MAX_TRIES = 7  # design doc v1.1 §4: up to 6 retries (1 initial + 6)
# Jitter SCALE (lambda_0 = _DEFAULT_JITTER_RCOND * trace(S)/N). Deliberately NOT
# reachable from hermitian_sandwich_solve's spectral `rtol`: an earlier revision
# passed rtol (1e-4) in as the jitter scale, which is the exact conflation
# _cholesky_jitter_sandwich's docstring already records for tsvd_rcond. The two
# are different quantities with different units and differ by ten orders of
# magnitude at their defaults.
_DEFAULT_JITTER_RCOND = 1e-14
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
    Diagnostic-only, and NOT decision-grade. No residual mode drives a
    solver switch any more -- the automatic TSVD fallback was removed. The
    calibration gap is now quantified rather than merely warned about:
    measured bias is ~+9% and does NOT vanish with probe count (spread
    falls as 1/sqrt(m), the bias does not), so this must never stand in
    for an acceptance gate. Use residual_mode="exact" for anything
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



# ---------------------------------------------------------------------------
# Two-sided regularised sandwich S_A^-1 M S_B^-1, lifted from df/fit.py so the
# periodic Coulomb core and the molecular fit share ONE implementation. A second
# copy in this module previously reintroduced a parameter conflation that fit.py
# had already found and documented. Behaviour is unchanged: compute_Z's outputs
# and provenance are byte-identical across the move.
# ---------------------------------------------------------------------------



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
                               residual_seed=_DEFAULT_RESIDUAL_SEED,
                               caller_label="compute_Z"):
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
    # REJECTED, not ignored. tsvd_rcond only ever fed the automatic fallback; with
    # that gone it has no effect, and silently accepting a caller's singular-value
    # cutoff while doing nothing with it is worse than refusing it. Callers wanting
    # truncation should select solver="tsvd" explicitly, where the cutoff is live.
    if tsvd_rcond is not None:
        raise ValueError(
            "tsvd_rcond has no effect on the Cholesky path: the automatic TSVD "
            "fallback was removed, so nothing consumes a singular-value cutoff "
            "here. Select solver='tsvd' explicitly if truncation is wanted."
        )
    S_A = jnp.asarray(S_A)
    S_B = jnp.asarray(S_B)
    M = jnp.asarray(M)
    if S_A.dtype in (jnp.float32, jnp.complex64):
        logger.warning(
            f"{caller_label} (unscaled_cholesky_jitter): resolved dtype is {S_A.dtype} -- JAX "
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
        **residual_meta,
    }

    if fit_residual > _RESIDUAL_WARN_THRESHOLD:
        # WARN ONLY -- no automatic solver switch. Removed on owner instruction
        # 2026-07-29 after it was measured harmful on real periodic data: at k112
        # rank 85 the fallback produced dE/atom 1.763e-03 with a diverged SCF,
        # while letting Cholesky finish gave 1.513e-05 and converged. It never
        # helped in any measured case.
        #
        # The reason it hurt is specific and worth keeping: _tsvd_sandwich
        # truncates at its own machine-epsilon tsvd_rcond, which is deliberately
        # NOT jitter_rcond (reusing that was a real bug -- a caller-forced
        # jitter_rcond=1.0 made TSVD retain zero modes). But it is also not the
        # caller's rtol. A caller arriving with an explicit retention policy had
        # it silently replaced by machine epsilon, so the fallback kept exactly
        # the near-null directions that policy exists to discard.
        #
        # The bias residual is still computed and reported; only the automatic
        # action is gone. _tsvd_sandwich remains available as an explicitly
        # selected solver.
        logger.warning(
            f"{caller_label} (unscaled_cholesky_jitter): two-sided fit residual "
            f"{fit_residual:.3e} exceeds acceptance threshold "
            f"{_RESIDUAL_WARN_THRESHOLD:.0e} (jitter_A={jitter_A:.3e}, "
            f"tries_A={tries_A}, residual_mode={residual_mode!r}) -- this reflects "
            f"UNREGULARIZED bias, not a factorization failure, and more jitter "
            f"cannot reduce it. Reported, not acted on: select solver='tsvd' "
            f"explicitly if truncation is wanted."
        )

    return Z_np, provenance


def _tsvd_sandwich(S_A, S_B, M, tsvd_rcond, same_sector,
                    residual_mode="exact",
                    residual_n_probes=_DEFAULT_RESIDUAL_N_PROBES,
                    residual_seed=_DEFAULT_RESIDUAL_SEED,
                    caller_label="compute_Z"):
    """S_A^+ M S_B^+ via an EXPLICIT truncated-SVD pseudoinverse (not
    np.linalg.pinv's black box) so the retained singular-value range can
    be reported in provenance -- the EXPLICITLY selected diagnostic solver
    per design doc §4 (production default is cholesky_jitter).

    NO LONGER a fallback. It was _cholesky_jitter_sandwich's automatic
    corrective action until 2026-07-29; that was removed after being
    measured harmful on real periodic data, and this function is now
    reachable only by a caller choosing solver="tsvd". Its cutoff comes
    from that caller's rcond; the old tsvd_rcond parameter affected
    nothing and is refused at the public boundary.

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
            f"{caller_label} (tsvd): two-sided fit residual {fit_residual:.3e} exceeds the "
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




def _cholesky_jitter_entry(Pi, V, *, jitter_rcond, rtol, n_retained_pin,
                           target_truncation_residual):
    """Route Pi W Pi = V through the shared _cholesky_jitter_sandwich.

    INTERIM BIAS POLICY, and its precondition. When the unregularized bias
    exceeds _RESIDUAL_WARN_THRESHOLD this warns and proceeds -- nothing fails.
    Warning-only is NOT a defensible production policy on its own: a log line
    nobody reads is indistinguishable from no check. It is acceptable here ONLY
    because the mode is confined to the host reference path, which structurally
    cannot run a configuration needing the production memory lever:

      device + cholesky_jitter        -> refused (device has no such mode)
      host   + p_block_rows          -> refused (host is reference-grade)
      host   + cholesky + p_block    -> refused (both of the above)

    So this mode cannot execute any P-blocked run. Precisely stated: it is not
    unreachable from "production" in the abstract -- a small run needing no
    panel blocking could use it -- it is unreachable from any configuration
    that needs P-blocking, which is every run large enough for the bias to
    matter at scale.

    THEREFORE: settling the bias policy (fail-closed vs a calibrated guard
    band, decided against measured energy) is a PREREQUISITE for implementing
    the device path, not a follow-up to it. Removing either refusal above
    without settling it would turn warning-only into silent acceptance.

    Pi appears on BOTH sides, so this is the same_sector case: one
    factorization, reused. Regularizes instead of truncating, so there is no
    spectrum and therefore no n_retained, s_min_retained, truncation_residual
    or retained_solve_residual. Those keys are ABSENT rather than filled with
    placeholders -- a caller gating on retained_solve_residual should raise
    KeyError here, not silently gate on a fabricated number. Acceptance for
    this mode is the energy gate.
    """
    if rtol is not None:
        raise ValueError(
            "rtol does not apply to retention_mode='cholesky_jitter': it is a "
            "spectral truncation threshold and this mode does not truncate. Pass "
            "jitter_rcond to set the jitter scale (default "
            f"{_DEFAULT_JITTER_RCOND:g}); the two differ by ten orders of magnitude "
            "at their defaults and are not interchangeable."
        )
    for name, value in (("n_retained_pin", n_retained_pin),
                        ("target_truncation_residual", target_truncation_residual)):
        if value is not None:
            raise ValueError(
                f"{name} does not apply to retention_mode='cholesky_jitter': the "
                f"mode regularizes rather than truncating, so it has no retained set."
            )
    if jitter_rcond is None:
        jitter_rcond = _DEFAULT_JITTER_RCOND
    elif isinstance(jitter_rcond, bool) or not isinstance(jitter_rcond, (int, float)):
        raise ValueError(f"jitter_rcond must be a finite positive float, got {jitter_rcond!r}.")
    else:
        jitter_rcond = float(jitter_rcond)
        if not np.isfinite(jitter_rcond) or jitter_rcond <= 0.0:
            raise ValueError(f"jitter_rcond must be a finite positive float, got {jitter_rcond!r}.")

    # Checks the RESOLVED dtype rather than jax.config, because the failure mode
    # is the silent downcast itself: c128 in with x64 disabled yields c64 here.
    # The shared helper only warns, since compute_Z's contract predates this; a
    # production periodic solve has no reason to accept the ~25x precision loss.
    resolved = jnp.asarray(Pi).dtype
    if resolved in (jnp.float32, jnp.complex64):
        raise ValueError(
            f"retention_mode='cholesky_jitter' resolved Pi to {resolved}. If Pi was "
            f"float64/complex128, JAX downcast it because x64 is disabled: call "
            f"jax.config.update('jax_enable_x64', True) before solving. Refusing "
            f"rather than warning -- single precision costs ~25x accuracy here."
        )

    tiny = np.finfo(np.result_type(np.asarray(Pi).dtype,
                                   np.asarray(V).dtype, np.complex128)).tiny
    Pi_herm = (Pi + Pi.conj().T) / 2
    V_herm = (V + V.conj().T) / 2
    # Named so the warnings do not claim compute_Z ran. Defaulting the parameter to
    # "compute_Z" keeps every molecular message byte-identical; a note telling readers
    # to reinterpret a false label is not a fix.
    W, provenance = _cholesky_jitter_sandwich(
        Pi_herm, Pi_herm, V_herm, jitter_rcond, same_sector=True,
        caller_label="hermitian_sandwich_solve")

    info = dict(provenance)
    # No fallback exists, so there is no branch: the solver is always Cholesky and
    # there is never a singular-value range. Keeping an `if fell_back` here would be
    # unreachable code implying a second outcome that cannot occur.
    if provenance.get("fallback_triggered", False):
        raise AssertionError(
            "cholesky_jitter reported fallback_triggered=True, but the automatic "
            "TSVD fallback was removed. Provenance and control flow disagree."
        )
    info.pop("retained_singular_value_range", None)
    info.update(
        retention_mode="cholesky_jitter",
        rtol=None,
        jitter_rcond=jitter_rcond,
        # One outcome, one label. Two prior labels here were false in sequence;
        # the regression instruments np.linalg.eigh/svd rather than comparing this
        # string to another hand-written string.
        backend="jax_cho_solve",
        # The helper's own string spells the molecular fit problem
        # (||S Z S - C C^dagger||). The arithmetic is identical, but the periodic
        # caller solves Pi W Pi = V, and a label naming the wrong operands is a
        # false statement about which quantity was measured.
        residual_norm_convention="||Pi W Pi - V|| / ||V||",
        pi_anti_hermitian_residual=float(np.linalg.norm(Pi - Pi.conj().T))
        / max(float(np.linalg.norm(Pi)), tiny),
        v_anti_hermitian_residual=float(np.linalg.norm(V - V.conj().T))
        / max(float(np.linalg.norm(V)), tiny),
    )
    return np.asarray(W), info


def hermitian_sandwich_solve(
    Pi, V, *, rtol=None, retention_mode="single", target_truncation_residual=None,
    n_retained_pin=None, jitter_rcond=None,
):
    """Two-sided Hermitian sandwich solve for W in Pi W Pi ~= V. Pi and V
    are Hermitized on entry; their anti-Hermitian residuals are recorded.
    See design doc §5.

    The four modes differ in KIND, not just in tolerance, and the info schema
    differs with them (see Returns):

      - "single" builds a truncated PSEUDO-INVERSE (a retained subspace with a
        projector). rtol is RELATIVE (rtol * s_max).
      - "pairwise" has NO retained subspace and NO projector -- only a retained
        PAIR SET, so it is not a pseudo-inverse. rtol is RELATIVE.
      - "svd_lstsq" is fftisdf's own lstsq formula; rtol is ABSOLUTE, not scaled
        by s_max, because this mode claims to BE that formula rather than to be
        scale-invariant.
      - "cholesky_jitter" does not truncate at all: it REGULARIZES, rejects rtol,
        and takes jitter_rcond instead.

    Which schema you get is identified by retention_mode for the three truncating
    modes. For "cholesky_jitter" it is NOT knowable in advance -- the mode may
    fall back to TSVD -- so there, and only there, read info["solver"].

    Four retention modes:
      "single" (default): threshold = rtol * s_max (scale-invariant).
        Mode i retained iff s_i > threshold; W's (i,j) term is nonzero
        only when BOTH i and j pass -- equivalent to Pi^+_r V Pi^+_r
        with Pi^+_r's single-mode truncated pseudo-inverse.
      "pairwise": threshold = rtol * s_max (scale-invariant). Pair
        (i,j) retained iff s_i*s_j > threshold**2 -- strictly more
        permissive than "single" (keeps weak-strong couplings a
        single-mode AND excludes). Uses eigh (Pi is Hermitian PSD).
      "svd_lstsq": fftisdf's OWN formula (fft/isdf.py's lstsq),
        transplanted structurally, not just its mask shape -- SVD (not
        eigh) of Pi: Pi = U diag(s) Vh; T = U^H V_herm U; T[i,j] /=
        s_i*s_j where s_i*s_j > rtol**2, else 0; W = V T Vh (V=Vh^H).
        U and V are NOT assumed equal even though Pi is Hermitian (the
        reference never assumes this either). rtol is used as an
        ABSOLUTE threshold here (fftisdf's own tol, default 1e-8), NOT
        scaled by s_max -- this mode does not claim scale-invariance,
        it claims to BE fftisdf's formula. Task #25/C2 item 2b: with
        eta's q-labeling fixed (see build_pi_eta), Pi and kern now match
        fftisdf's own arrays to machine precision, and feeding them
        through fftisdf's own lstsq exactly reproduces fftisdf's own W
        (relerr ~1e-11) -- this mode ports that same computation so
        pytc's own pipeline reaches the same accuracy without depending
        on the external reference at runtime.

      "cholesky_jitter": does NOT truncate -- routes to the shared
        _cholesky_jitter_sandwich (Cholesky with adaptive diagonal
        jitter, same_sector). Having no spectrum, it returns none of
        n_retained/s_min_retained/truncation_residual/
        retained_solve_residual, and reports the helper's fit_residual
        instead; acceptance for this mode is the energy gate. Takes
        jitter_rcond, and REJECTS rtol.

    The three truncating modes report two residuals SEPARATELY:
    retained_solve_residual (machine-tier arithmetic sanity check, ~0 regardless
    of rtol) and truncation_residual (controlled by rtol; reported, not gated
    here). "cholesky_jitter" reports NEITHER -- having no retained set, it
    reports a single fit_residual instead.

    Args:
        Pi: (n,n), Hermitian PSD expected.
        V: (n,n).
        rtol: spectral retention threshold; None means the 1e-4 default
            (design doc §5 -- 1e-8 was far too loose at over-complete rank).
            RELATIVE (rtol * s_max) for "single"/"pairwise"; ABSOLUTE for
            "svd_lstsq"; REJECTED by "cholesky_jitter", which has no spectral
            cutoff. Must be None when n_retained_pin is given.
        retention_mode: "single", "pairwise", "svd_lstsq" or "cholesky_jitter".
        target_truncation_residual: optional, "single" mode only; if
            given, additional modes are retained (decreasing eigenvalue
            order) until the truncation residual meets this target or
            all n modes are retained (adaptive_retention_used=True).
        n_retained_pin: optional int K in [1, n], "single" mode only;
            retain exactly the K largest-eigenvalue modes regardless of
            rtol. Mutually exclusive with rtol/target_truncation_residual
            (both must be left None).

    Args (cholesky_jitter only):
        jitter_rcond: jitter SCALE, lambda_0 = jitter_rcond * trace(Pi)/n,
            default 1e-14. NOT a spectral cutoff and NOT interchangeable with
            rtol, which this mode rejects. Rejected by the other modes.

    Returns:
        (W, info). TWO SCHEMAS. The truncating modes are identified by
        retention_mode and carry NO "solver" key -- reading info["solver"] on
        them raises KeyError. "solver" exists only under "cholesky_jitter",
        where it is always "unscaled_cholesky_jitter":

        Truncating modes ("single"/"pairwise"/"svd_lstsq"): n_retained,
        n_discarded, s_max, s_min_retained (None if n_retained==0),
        pi/v_anti_hermitian_residual, retained_solve_residual,
        truncation_residual, rtol (None when pinned), retention_mode,
        adaptive_retention_used, target_truncation_residual, n_retained_pin,
        retention_marginal, cond_pi_retained, dtype, backend ("numpy").

        "cholesky_jitter", solver="unscaled_cholesky_jitter": NONE of the
        retained-set keys above exist -- there is no spectrum, so a caller
        gating on retained_solve_residual raises rather than gating on a
        fabricated value. Instead: jitter_used, n_tries, cutoff, fit_residual,
        residual_norm_convention, backward_error_mode/tol, fallback_triggered,
        row_scaling, dtype, jitter_rcond, backend ("jax_cho_solve"),
        pi/v_anti_hermitian_residual.

        There is no third schema. The automatic TSVD fallback was removed
        (owner instruction, measured harmful on real periodic data), so
        "cholesky_jitter" cannot return solver="tsvd" -- fallback_triggered is
        reported and is always False. A high unregularized bias is warned about
        and left in fit_residual for the caller to act on; _tsvd_sandwich is
        still reachable by selecting solver="tsvd" explicitly, which is a
        different entry point with its own cutoff.
    """
    Pi = np.asarray(Pi)
    V = np.asarray(V)
    if Pi.ndim != 2 or Pi.shape[0] != Pi.shape[1]:
        raise ValueError(f"Pi must be a square 2-D array, got shape {Pi.shape}.")
    n = Pi.shape[0]
    if n == 0:
        raise ValueError("Pi must be nonempty.")
    if V.shape != (n, n):
        raise ValueError(f"V must have shape {(n, n)} matching Pi, got {V.shape}.")
    if not np.all(np.isfinite(Pi)) or not np.all(np.isfinite(V)):
        raise ValueError("Pi and V must be finite.")

    if retention_mode not in ("single", "pairwise", "svd_lstsq", "cholesky_jitter"):
        raise ValueError(
            f"retention_mode must be 'single', 'pairwise', 'svd_lstsq' or "
            f"'cholesky_jitter', got {retention_mode!r}."
        )
    # Dispatched BEFORE the eigh below, which is unconditional: an earlier revision
    # placed this branch after it, so every "Cholesky" timing on this path had in
    # fact paid for a full eigendecomposition first.
    if retention_mode == "cholesky_jitter":
        return _cholesky_jitter_entry(
            Pi, V, jitter_rcond=jitter_rcond, rtol=rtol,
            n_retained_pin=n_retained_pin,
            target_truncation_residual=target_truncation_residual,
        )
    if jitter_rcond is not None:
        raise ValueError(
            f"jitter_rcond applies only to retention_mode='cholesky_jitter', got "
            f"retention_mode={retention_mode!r}. The truncating modes are controlled "
            f"by rtol, a spectral cutoff, which is not a jitter scale."
        )
    if n_retained_pin is not None:
        if retention_mode != "single":
            raise ValueError("n_retained_pin is only supported for retention_mode='single'.")
        if rtol is not None:
            raise ValueError(
                "n_retained_pin is mutually exclusive with rtol; leave rtol=None."
            )
        if target_truncation_residual is not None:
            raise ValueError(
                "n_retained_pin is mutually exclusive with target_truncation_residual."
            )
        if isinstance(n_retained_pin, bool) or not isinstance(
            n_retained_pin, (int, np.integer)
        ):
            raise ValueError(
                f"n_retained_pin must be an integer in [1, {n}], got {n_retained_pin!r}."
            )
        n_retained_pin = int(n_retained_pin)
        if not 1 <= n_retained_pin <= n:
            raise ValueError(f"n_retained_pin must be in [1, {n}], got {n_retained_pin}.")
    if rtol is None:
        rtol_eff = 1e-4
    else:
        if isinstance(rtol, bool) or not isinstance(rtol, (int, float)):
            raise ValueError(f"rtol must be a finite positive float, got {rtol!r}.")
        rtol_eff = float(rtol)
        if not np.isfinite(rtol_eff) or rtol_eff <= 0.0:
            raise ValueError(f"rtol must be a finite positive float, got {rtol!r}.")
    rtol = None if n_retained_pin is not None else rtol_eff

    if retention_mode != "single" and target_truncation_residual is not None:
        raise ValueError("target_truncation_residual is only supported for retention_mode='single'.")

    if target_truncation_residual is not None:
        if isinstance(target_truncation_residual, bool) or not isinstance(
            target_truncation_residual, (int, float)
        ):
            raise ValueError(
                f"target_truncation_residual must be None or a finite non-negative "
                f"float, got {target_truncation_residual!r}."
            )
        target_truncation_residual = float(target_truncation_residual)
        if not np.isfinite(target_truncation_residual) or target_truncation_residual < 0.0:
            raise ValueError(
                f"target_truncation_residual must be None or a finite non-negative "
                f"float, got {target_truncation_residual!r}."
            )

    tiny = np.finfo(np.result_type(Pi.dtype, V.dtype, np.complex128)).tiny

    Pi_herm = (Pi + Pi.conj().T) / 2
    V_herm = (V + V.conj().T) / 2
    pi_anti_hermitian_residual = float(np.linalg.norm(Pi - Pi.conj().T)) / max(
        float(np.linalg.norm(Pi)), tiny
    )
    v_anti_hermitian_residual = float(np.linalg.norm(V - V.conj().T)) / max(
        float(np.linalg.norm(V)), tiny
    )

    # eigh returns eigenvalues ASCENDING; reverse to descending so
    # "the first n_retained" are the largest, and adaptive retention can
    # grow n_retained by simply extending the slice.
    eigvals_asc, eigvecs_asc = np.linalg.eigh(Pi_herm)
    order = np.argsort(eigvals_asc)[::-1]
    eigvals = eigvals_asc[order]
    eigvecs = eigvecs_asc[:, order]

    s_max = float(eigvals[0]) if n > 0 else 0.0
    if s_max <= 0.0:
        raise ValueError(
            "Pi's largest eigenvalue is non-positive after Hermitization -- Pi does "
            "not appear to be PSD (or is the zero matrix)."
        )
    threshold = rtol_eff * s_max
    v_norm = max(float(np.linalg.norm(V_herm)), tiny)
    adaptive_retention_used = False
    s_first_discarded = None

    if retention_mode == "svd_lstsq":
        # fftisdf's literal lstsq formula: SVD (not eigh), rtol used as an
        # ABSOLUTE threshold (not scaled by s_max -- this mode does not
        # claim scale-invariance, it claims to BE fftisdf's formula).
        threshold = rtol
        u, s_svd, vh = np.linalg.svd(Pi_herm)
        v = vh.conj().T
        s_max = float(s_svd[0]) if n > 0 else 0.0
        s_outer = s_svd[:, None] * s_svd[None, :]
        pair_mask = np.abs(s_outer) > threshold**2
        n_retained = int(np.sum(s_svd > threshold))
        n_discarded = n - n_retained
        s_min_retained = float(np.min(s_svd[s_svd > threshold])) if n_retained > 0 else None

        if n_retained > 0:
            M = u.conj().T @ V_herm @ u
            safe_s_outer = np.where(pair_mask, s_outer, 1.0)
            T = np.where(pair_mask, M / safe_s_outer, 0.0)
            W = v @ T @ vh
            W = (W + W.conj().T) / 2
            diff_uv = u.conj().T @ (Pi_herm @ W @ Pi_herm - V_herm) @ v
            retained_solve = np.where(pair_mask, diff_uv, 0.0)
            retained_target = np.where(pair_mask, M, 0.0)
            retained_solve_residual = float(np.linalg.norm(retained_solve)) / max(
                float(np.linalg.norm(retained_target)), tiny
            )
            truncation_mass = np.where(pair_mask, 0.0, M)
            truncation_residual = float(np.linalg.norm(truncation_mass)) / v_norm
        else:
            W = np.zeros((n, n), dtype=np.result_type(Pi_herm.dtype, V_herm.dtype))
            retained_solve_residual = 0.0
            truncation_residual = 1.0
    elif retention_mode == "pairwise":
        s_outer = eigvals[:, None] * eigvals[None, :]
        pair_mask = s_outer > threshold**2
        touched_mask = np.any(pair_mask, axis=1)
        n_retained = int(np.sum(touched_mask))
        n_discarded = n - n_retained
        s_min_retained = float(np.min(eigvals[touched_mask])) if n_retained > 0 else None

        if n_retained > 0:
            M = eigvecs.conj().T @ V_herm @ eigvecs
            safe_s_outer = np.where(pair_mask, s_outer, 1.0)
            T = np.where(pair_mask, M / safe_s_outer, 0.0)
            W = eigvecs @ T @ eigvecs.conj().T
            W = (W + W.conj().T) / 2
            # Pairwise retention has no retained SUBSPACE (no projector) --
            # only a retained PAIR SET. Diagnostics must be ELEMENTWISE in
            # Pi's eigenbasis: Pi W Pi equals V exactly on retained pairs
            # (a pure arithmetic identity, T=M/s_outer there), so the
            # solve-residual check is a sanity check on that identity, not
            # a subspace-projected accuracy measure -- a union-of-touched-
            # modes projector wrongly counts deliberately-zeroed pairs as
            # error.
            diff_eigenbasis = eigvecs.conj().T @ (Pi_herm @ W @ Pi_herm - V_herm) @ eigvecs
            retained_solve = np.where(pair_mask, diff_eigenbasis, 0.0)
            retained_target = np.where(pair_mask, M, 0.0)
            retained_solve_residual = float(np.linalg.norm(retained_solve)) / max(
                float(np.linalg.norm(retained_target)), tiny
            )
            truncation_mass = np.where(pair_mask, 0.0, M)
            truncation_residual = float(np.linalg.norm(truncation_mass)) / v_norm
        else:
            W = np.zeros((n, n), dtype=np.result_type(Pi_herm.dtype, V_herm.dtype))
            retained_solve_residual = 0.0
            truncation_residual = 1.0
    else:
        if n_retained_pin is not None:
            n_retained = n_retained_pin
        else:
            n_retained = int(np.sum(eigvals > threshold))

        def _truncation_residual(k):
            U_r = eigvecs[:, :k]
            proj = U_r @ U_r.conj().T
            return float(np.linalg.norm(V_herm - proj @ V_herm @ proj)) / v_norm

        if target_truncation_residual is not None:
            current = _truncation_residual(n_retained) if n_retained > 0 else 1.0
            while current > target_truncation_residual and n_retained < n:
                n_retained += 1
                adaptive_retention_used = True
                current = _truncation_residual(n_retained)

        n_discarded = n - n_retained
        U_r = eigvecs[:, :n_retained]
        sigma_r = eigvals[:n_retained]
        s_min_retained = float(np.min(sigma_r)) if n_retained > 0 else None
        if n_retained_pin is not None and n_retained < n:
            s_first_discarded = float(eigvals[n_retained])

        if n_retained > 0:
            # Column scaling, not a GEMM: U_r @ diag(d) scales column j by d[j],
            # which broadcasting does with the same multiplications and none of the
            # n*k*k accumulation.
            Pi_pinv_r = (U_r * (1.0 / sigma_r)[None, :]) @ U_r.conj().T
            W = Pi_pinv_r @ V_herm @ Pi_pinv_r
            W = (W + W.conj().T) / 2
            proj_r = U_r @ U_r.conj().T
            retained_target = proj_r @ V_herm @ proj_r
            retained_solve = proj_r @ (Pi_herm @ W @ Pi_herm - V_herm) @ proj_r
            retained_solve_residual = float(np.linalg.norm(retained_solve)) / max(
                float(np.linalg.norm(retained_target)), tiny
            )
            # _truncation_residual(n_retained) recomputes U_r @ U_r^H and
            # proj @ V @ proj -- both already in hand as proj_r and retained_target.
            # Three n^3-class GEMMs, ~0.8 h each at 444, for values we hold. The
            # helper stays: the adaptive-retention loop above calls it at varying k.
            truncation_residual = float(np.linalg.norm(V_herm - retained_target)) / v_norm
        else:
            W = np.zeros((n, n), dtype=np.result_type(Pi_herm.dtype, V_herm.dtype))
            retained_solve_residual = 0.0
            truncation_residual = 1.0

    retention_marginal, cond_pi = _check_retention_marginal(
        s_max, s_min_retained, threshold, rtol, caller="hermitian_sandwich_solve",
        s_first_discarded=s_first_discarded, pinned=n_retained_pin is not None,
    )

    info = {
        "n_retained": n_retained,
        "n_discarded": n_discarded,
        "s_max": s_max,
        "s_min_retained": s_min_retained,
        "pi_anti_hermitian_residual": pi_anti_hermitian_residual,
        "v_anti_hermitian_residual": v_anti_hermitian_residual,
        "retained_solve_residual": retained_solve_residual,
        "truncation_residual": truncation_residual,
        "rtol": rtol,
        "retention_mode": retention_mode,
        "adaptive_retention_used": adaptive_retention_used,
        "target_truncation_residual": target_truncation_residual,
        "n_retained_pin": n_retained_pin,
        "retention_marginal": retention_marginal,
        "cond_pi_retained": cond_pi,
        "dtype": str(W.dtype),
        "backend": "numpy",
    }
    return W, info


@partial(jax.jit, static_argnames=("retention_mode",))
def _hermitian_sandwich_solve_core(Pi, V, rtol, retention_mode="single", n_retained_pin=-1):
    """Fixed-shape, jitted, device-resident core of
    hermitian_sandwich_solve_device (design v2.1 sections 5+6/7).
    Reproduces hermitian_sandwich_solve's math exactly (both retention
    modes -- see hermitian_sandwich_solve's docstring), restructured so
    retained-rank truncation is a boolean MASK over the full n-dimensional
    eigenbasis rather than a dynamic-size slice, which is required for a
    static-shape jax.jit graph -- masked-out modes/pairs contribute
    exactly 0, which is mathematically identical to slicing them away.

    n_retained_pin: traced scalar, "single" mode only. K >= 0 retains the
    top-K eigenmodes via a positional mask (eigenvalues are sorted
    descending); K < 0 disables the pin (rtol thresholding applies).
    """
    n = Pi.shape[0]
    dtype = jnp.result_type(Pi.dtype, V.dtype, jnp.complex128)
    tiny = jnp.finfo(dtype).tiny

    Pi_herm = (Pi + Pi.conj().T) / 2
    V_herm = (V + V.conj().T) / 2
    pi_anti_hermitian_residual = jnp.linalg.norm(Pi - Pi.conj().T) / jnp.maximum(
        jnp.linalg.norm(Pi), tiny
    )
    v_anti_hermitian_residual = jnp.linalg.norm(V - V.conj().T) / jnp.maximum(
        jnp.linalg.norm(V), tiny
    )

    eigvals_asc, eigvecs_asc = jnp.linalg.eigh(Pi_herm)
    order = jnp.argsort(eigvals_asc)[::-1]
    eigvals = eigvals_asc[order]
    eigvecs = eigvecs_asc[:, order]

    s_max = eigvals[0]
    threshold = rtol * s_max
    v_norm = jnp.maximum(jnp.linalg.norm(V_herm), tiny)
    s_first_discarded = jnp.zeros((), dtype)

    if retention_mode == "svd_lstsq":
        # fftisdf's literal lstsq formula: SVD (not eigh), rtol used as an
        # ABSOLUTE threshold (not scaled by s_max -- see hermitian_
        # sandwich_solve's docstring).
        threshold = rtol
        u, s_svd, vh = jnp.linalg.svd(Pi_herm)
        v = vh.conj().T
        s_max = s_svd[0]
        s_outer = s_svd[:, None] * s_svd[None, :]
        pair_mask = jnp.abs(s_outer) > threshold**2
        n_retained = jnp.sum(s_svd > threshold)
        has_retained = n_retained > 0

        M = u.conj().T @ V_herm @ u
        safe_s_outer = jnp.where(pair_mask, s_outer, 1.0)
        T = jnp.where(pair_mask, M / safe_s_outer, 0.0)
        W_full = v @ T @ vh
        W_full = (W_full + W_full.conj().T) / 2

        s_min_retained = jnp.min(jnp.where(s_svd > threshold, s_svd, jnp.inf))

        diff_uv = u.conj().T @ (Pi_herm @ W_full @ Pi_herm - V_herm) @ v
        retained_solve = jnp.where(pair_mask, diff_uv, 0.0)
        retained_target = jnp.where(pair_mask, M, 0.0)
        retained_solve_residual_full = jnp.linalg.norm(retained_solve) / jnp.maximum(
            jnp.linalg.norm(retained_target), tiny
        )
        truncation_mass = jnp.where(pair_mask, 0.0, M)
        truncation_residual_full = jnp.linalg.norm(truncation_mass) / v_norm
    elif retention_mode == "pairwise":
        s_outer = eigvals[:, None] * eigvals[None, :]
        pair_mask = s_outer > threshold**2
        touched_mask = jnp.any(pair_mask, axis=1)
        n_retained = jnp.sum(touched_mask)
        has_retained = n_retained > 0

        M = eigvecs.conj().T @ V_herm @ eigvecs
        safe_s_outer = jnp.where(pair_mask, s_outer, 1.0)
        T = jnp.where(pair_mask, M / safe_s_outer, 0.0)
        W_full = eigvecs @ T @ eigvecs.conj().T
        W_full = (W_full + W_full.conj().T) / 2

        s_min_retained = jnp.min(jnp.where(touched_mask, eigvals, jnp.inf))

        # Pairwise retention has no retained SUBSPACE (no projector) --
        # only a retained PAIR SET. Diagnostics must be ELEMENTWISE in
        # Pi's eigenbasis: Pi W Pi equals V exactly on retained pairs (a
        # pure arithmetic identity, T=M/s_outer there), so the solve-
        # residual check is a sanity check on that identity, not a
        # subspace-projected accuracy measure -- a union-of-touched-modes
        # projector wrongly counts deliberately-zeroed pairs as error.
        diff_eigenbasis = eigvecs.conj().T @ (Pi_herm @ W_full @ Pi_herm - V_herm) @ eigvecs
        retained_solve = jnp.where(pair_mask, diff_eigenbasis, 0.0)
        retained_target = jnp.where(pair_mask, M, 0.0)
        retained_solve_residual_full = jnp.linalg.norm(retained_solve) / jnp.maximum(
            jnp.linalg.norm(retained_target), tiny
        )
        truncation_mass = jnp.where(pair_mask, 0.0, M)
        truncation_residual_full = jnp.linalg.norm(truncation_mass) / v_norm
    else:
        K = n_retained_pin
        pin_mask = jnp.arange(n) < jnp.maximum(K, 0)
        mask = jnp.where(K >= 0, pin_mask, eigvals > threshold)
        n_retained = jnp.sum(mask)
        has_retained = n_retained > 0
        s_first_discarded = jnp.where(
            (K >= 0) & (K < n), eigvals[jnp.clip(K, 0, n - 1)], 0.0
        )

        safe_eigvals = jnp.where(mask, eigvals, 1.0)
        inv_eigvals = jnp.where(mask, 1.0 / safe_eigvals, 0.0)
        Pi_pinv_r = (eigvecs * inv_eigvals[None, :]) @ eigvecs.conj().T
        W_full = Pi_pinv_r @ V_herm @ Pi_pinv_r
        W_full = (W_full + W_full.conj().T) / 2

        mask_c = mask.astype(eigvecs.dtype)
        s_min_retained = jnp.min(jnp.where(mask, eigvals, jnp.inf))

        proj_r = (eigvecs * mask_c[None, :]) @ eigvecs.conj().T
        retained_target = proj_r @ V_herm @ proj_r
        retained_solve = proj_r @ (Pi_herm @ W_full @ Pi_herm - V_herm) @ proj_r
        retained_solve_residual_full = jnp.linalg.norm(retained_solve) / jnp.maximum(
            jnp.linalg.norm(retained_target), tiny
        )
        truncation_residual_full = jnp.linalg.norm(V_herm - retained_target) / v_norm

    W = jnp.where(has_retained, W_full, jnp.zeros_like(W_full))
    retained_solve_residual = jnp.where(has_retained, retained_solve_residual_full, 0.0)
    truncation_residual = jnp.where(has_retained, truncation_residual_full, 1.0)

    return (
        W, n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
        v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
        s_first_discarded,
    )


def hermitian_sandwich_solve_device(Pi, V, *, rtol=None, retention_mode="single",
                                    n_retained_pin=None):
    """Device (JAX, fixed-shape, jitted) counterpart of
    hermitian_sandwich_solve. See design doc §5-§7.

    Deliberate simplifications vs the NumPy oracle:
      - No hard PSD validation: the jitted graph cannot raise on a traced
        value, so a non-PSD/zero Pi degrades to n_retained=0 (W=0)
        instead of raising.
      - No adaptive-retention mode (variable iteration count does not fit
        a fixed-shape jitted graph).

    rtol: None means the 1e-4 default; must be None when n_retained_pin is
    given. n_retained_pin: optional int K in [1, n], "single" mode only --
    retain exactly the K largest-eigenvalue modes (positional mask inside
    the jitted core; K is a traced scalar, so changing K does not
    recompile). Mutually exclusive with rtol.

    Returns:
        (W, info): W is (n,n) complex128 jax array; info has the same
        keys as hermitian_sandwich_solve's, with backend="jax",
        adaptive_retention_used=False, target_truncation_residual=None.
    """
    Pi_np = np.asarray(Pi)
    V_np = np.asarray(V)
    if Pi_np.ndim != 2 or Pi_np.shape[0] != Pi_np.shape[1]:
        raise ValueError(f"Pi must be a square 2-D array, got shape {Pi_np.shape}.")
    n = Pi_np.shape[0]
    if n == 0:
        raise ValueError("Pi must be nonempty.")
    if V_np.shape != (n, n):
        raise ValueError(f"V must have shape {(n, n)} matching Pi, got {V_np.shape}.")
    if not np.all(np.isfinite(Pi_np)) or not np.all(np.isfinite(V_np)):
        raise ValueError("Pi and V must be finite.")
    if retention_mode not in ("single", "pairwise", "svd_lstsq"):
        raise ValueError(
            f"retention_mode must be 'single', 'pairwise' or 'svd_lstsq', "
            f"got {retention_mode!r}. 'cholesky_jitter' is NOT available on the\n"
            f"DEVICE path: this path once accepted it, ran eig, and returned info\n"
            f"labelled Cholesky. The host hermitian_sandwich_solve supports it by\n"
            f"delegating to _cholesky_jitter_sandwich in this module; routing the\n"
            f"device path through the same helper is separate, unstarted work."
        )
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
                f"n_retained_pin must be an integer in [1, {n}], got {n_retained_pin!r}."
            )
        n_retained_pin = int(n_retained_pin)
        if not 1 <= n_retained_pin <= n:
            raise ValueError(f"n_retained_pin must be in [1, {n}], got {n_retained_pin}.")
    if rtol is None:
        rtol_eff = 1e-4
    else:
        if isinstance(rtol, bool) or not isinstance(rtol, (int, float)):
            raise ValueError(f"rtol must be a finite positive float, got {rtol!r}.")
        rtol_eff = float(rtol)
        if not np.isfinite(rtol_eff) or rtol_eff <= 0.0:
            raise ValueError(f"rtol must be a finite positive float, got {rtol!r}.")

    Pi_jnp = jnp.asarray(Pi_np, dtype=jnp.complex128)
    V_jnp = jnp.asarray(V_np, dtype=jnp.complex128)
    if Pi_jnp.dtype != jnp.complex128:
        # Hard-fail at the boundary: JAX silently downcast the complex128 cast
        # because jax_enable_x64 is off, so the eigh/solve would run in single
        # precision (~1e-5 accuracy) and silently corrupt W -- surfacing only
        # three stages downstream as an opaque trip of the 1e-10 machine-tier
        # retained-solve gate. A warn-only guard in front of a hard
        # gate is incoherent; fail closed here with an actionable message.
        raise ValueError(
            f"hermitian_sandwich_solve_device: resolved dtype is {Pi_jnp.dtype}, not "
            f"complex128 -- jax_enable_x64 is off, so JAX silently downcast the "
            f"complex128 cast and the device solve would run in single precision "
            f"(~1e-5 accuracy), corrupting W and tripping the 1e-10 machine-tier gate "
            f"downstream. Call jax.config.update('jax_enable_x64', True) before building "
            f"(design v2.1 section 1 fixes c128 as the only tier with defined "
            f"1e-6-class gates)."
        )

    (
        W, n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
        v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
        s_first_discarded,
    ) = _hermitian_sandwich_solve_core(
        Pi_jnp, V_jnp, rtol_eff, retention_mode,
        n_retained_pin if n_retained_pin is not None else -1,
    )

    info = _solve_info_from_core_output(
        n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
        v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
        s_first_discarded, n, W.dtype,
        None if n_retained_pin is not None else rtol_eff,
        caller="hermitian_sandwich_solve_device",
        retention_mode=retention_mode,
        n_retained_pin=n_retained_pin,
    )
    return W, info
