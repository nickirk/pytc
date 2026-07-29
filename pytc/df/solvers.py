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

    The unregularized-bias check and the TSVD fallback live in
    df/fit.py's _cholesky_jitter_sandwich since they need the fit context
    this generic primitive doesn't have. This solver's provenance label
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


def hermitian_sandwich_solve(
    Pi, V, *, rtol=None, retention_mode="single", target_truncation_residual=None,
    n_retained_pin=None,
):
    """Two-sided Hermitian sandwich solve for W in Pi W Pi ~= V via a
    truncated pseudo-inverse of Pi. Pi and V are Hermitized on entry;
    their anti-Hermitian residuals are recorded. See design doc §5.

    Three retention modes:
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

    Two residuals are reported SEPARATELY: retained_solve_residual
    (machine-tier arithmetic sanity check, ~0 regardless of rtol) and
    truncation_residual (controlled by rtol; reported, not gated here).

    Args:
        Pi: (n,n), Hermitian PSD expected.
        V: (n,n).
        rtol: relative spectral retention threshold; None means the 1e-4
            default (design doc §5 -- 1e-8 was far too loose at
            over-complete rank). Must be None when n_retained_pin is given.
        retention_mode: "single" or "pairwise".
        target_truncation_residual: optional, "single" mode only; if
            given, additional modes are retained (decreasing eigenvalue
            order) until the truncation residual meets this target or
            all n modes are retained (adaptive_retention_used=True).
        n_retained_pin: optional int K in [1, n], "single" mode only;
            retain exactly the K largest-eigenvalue modes regardless of
            rtol. Mutually exclusive with rtol/target_truncation_residual
            (both must be left None).

    Returns:
        (W, info): info keys are n_retained, n_discarded, s_max,
        s_min_retained (None if n_retained==0), pi/v_anti_hermitian_residual,
        retained_solve_residual, truncation_residual, rtol (None when
        pinned), retention_mode, adaptive_retention_used,
        target_truncation_residual, n_retained_pin,
        retention_marginal, cond_pi_retained, dtype, backend ("numpy").
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
            f"retention_mode must be 'single', 'pairwise', 'svd_lstsq', or "
            f"'cholesky_jitter', got {retention_mode!r}."
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

    if retention_mode == "cholesky_jitter":
        # The molecular path's primitive, transplanted: REGULARISE rather than
        # truncate. prepare_spd_cholesky's docstring already records why an SVD
        # is "a non-starter at production core sizes"; eigh is no better --
        # measured 52x slower than Cholesky at n=2000, and the periodic solve is
        # ~16 h of a projected 23 h at 444/Gamma.
        #
        # W = Pi_reg^-1 V Pi_reg^-1 with Pi_reg = L L^H, via four triangular
        # solves. There is no spectrum here, so n_retained and s_min_retained do
        # not exist: this mode trades the retained-space diagnostics for the
        # cost and must be judged on the ENERGY gate instead.
        from scipy.linalg import solve_triangular

        chol, lower, jitter_used, n_tries = prepare_spd_cholesky(
            Pi_herm, rcond=rtol_eff)
        L = np.asarray(chol) if lower else np.asarray(chol).conj().T
        A = solve_triangular(L, V_herm, lower=True)
        B = solve_triangular(L, A.conj().T, lower=True).conj().T
        C = solve_triangular(L.conj().T, B, lower=False)
        W = solve_triangular(L.conj().T, C.conj().T, lower=False).conj().T
        W = (W + W.conj().T) / 2

        resid = float(np.linalg.norm(Pi_herm @ W @ Pi_herm - V_herm)) / v_norm
        info = {
            "n_retained": None, "n_discarded": None,
            "s_max": s_max, "s_min_retained": None,
            "pi_anti_hermitian_residual": pi_anti_hermitian_residual,
            "v_anti_hermitian_residual": v_anti_hermitian_residual,
            # No truncation, so the machine-tier retained-solve residual has no
            # analogue; this is the actual Pi W Pi vs V residual, jitter bias
            # included.
            "retained_solve_residual": None,
            "truncation_residual": resid,
            "rtol": rtol_eff, "retention_mode": retention_mode,
            "adaptive_retention_used": False,
            "target_truncation_residual": target_truncation_residual,
            "n_retained_pin": None, "retention_marginal": False,
            "cond_pi_retained": None,
            "dtype": str(W.dtype), "backend": "numpy",
            "jitter_used": float(jitter_used), "jitter_tries": int(n_tries),
        }
        return W, info

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
            Pi_pinv_r = U_r @ np.diag(1.0 / sigma_r) @ U_r.conj().T
            W = Pi_pinv_r @ V_herm @ Pi_pinv_r
            W = (W + W.conj().T) / 2
            proj_r = U_r @ U_r.conj().T
            retained_target = proj_r @ V_herm @ proj_r
            retained_solve = proj_r @ (Pi_herm @ W @ Pi_herm - V_herm) @ proj_r
            retained_solve_residual = float(np.linalg.norm(retained_solve)) / max(
                float(np.linalg.norm(retained_target)), tiny
            )
            truncation_residual = _truncation_residual(n_retained)
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
    if retention_mode not in ("single", "pairwise", "svd_lstsq", "cholesky_jitter"):
        raise ValueError(
            f"retention_mode must be 'single', 'pairwise', 'svd_lstsq', or "
            f"'cholesky_jitter', got {retention_mode!r}."
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
