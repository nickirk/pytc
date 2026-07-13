"""Generic linear-solve primitives for structured least-squares and SPD
systems -- model-agnostic (pytc/df/ package reorganization, task #8,
isdf-coulomb-cuda, 2026-07-12: TC's own orbital fitting and the Coulomb
path's S-solve share this module, not a parallel implementation)."""
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
    matrix) answer for complex ones (Alice's review, task #15 design,
    2026-07-13: independently reproduced via an explicit dense
    C[(p,q),mu]/B[(p,q),g] + np.linalg.lstsq oracle -- relative error ~2.0
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
    check -- gap 2's REGULARIZED-solve half (Alice's task #8 review,
    2026-07-12): this gates jitter escalation; the separate UNREGULARIZED
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
    the Coulomb path's S-solve use (isdf-coulomb-cuda design doc §4,
    2026-07-12: "Cholesky with adaptive diagonal jitter is the
    production solver for S⁻¹-type applications" -- decided on measured
    evidence, not a default; see the doc for the rcond-sweep history
    that motivated it).

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
    which at production N_mu recreates exactly the cost problem gap 3
    (production-scalable residual estimation) was trying to solve at
    the fit layer; use "exact" only for small/reference-system
    diagnostics, never at production scale (Alice's task #8 re-review,
    2026-07-12: an earlier version of this function made the exact
    check unconditional).

    STATUS (Alice's re-review of commit 7064a7b, task #8 commit 3,
    2026-07-12): this fixes gap 1 (scale-relative jitter floor) and the
    regularized-solve half of gap 2 (backward-error gating jitter
    escalation, now mode-gated so it doesn't reintroduce an O(n^3) cost
    at the default production setting); the unregularized-bias check /
    TSVD-fallback half of gap 2, plus gaps 3 (production-scalable
    residual) and 4 (explicit same-sector), live in df/fit.py's
    _cholesky_jitter_sandwich since they need the actual fit context
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
            jitter regardless of the matrix's own scale, Alice's gap 1),
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
    eps_scale = eps * diag_mean  # PURELY matrix-relative -- no absolute 1.0 floor (gap 1 fix)
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
            task #15's provenance requirements -- a caller building a
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


def hermitian_sandwich_solve(Pi, V, *, rtol=1e-8, target_truncation_residual=None):
    """Two-sided Hermitian sandwich solve for W in Pi W Pi ~= V, via a
    relative-spectral-threshold truncated pseudo-inverse of Pi (task #21,
    #proj-isdf-periodic, design v2.1 section 5). NEW function, added
    beside the existing SVD/eigh-based solvers above -- reuses NumPy's
    eigh, not previously-existing code.

    Both Pi and V are Hermitized on entry (their own anti-Hermitian
    residuals are recorded, not silently discarded). Pi is eigendecomposed
    (Hermitian, PSD expected); modes are retained by the RELATIVE
    threshold s_i > rtol * s_max (scale-invariant -- rescaling Pi/V by a
    constant does not change which modes are retained). Let U_r be the
    retained eigenvectors and Sigma_r their eigenvalues; the truncated
    pseudo-inverse is Pi^+_r = U_r Sigma_r^-1 U_r^dagger, and
    W = Pi^+_r V Pi^+_r, Hermitized.

    Because U_r consists of Pi's OWN eigenvectors, Pi @ Pi^+_r =
    Pi^+_r @ Pi = Proj_r (the retained-subspace projector) EXACTLY (to
    the precision of the eigendecomposition) -- so Pi W Pi = Proj_r V
    Proj_r exactly on the retained subspace BY CONSTRUCTION, and the
    total error against the full V is entirely the DISCARDED-space part
    of V. Two residuals are reported SEPARATELY rather than one combined
    number, so this cannot be hidden:
        (i)  retained-space solve residual (machine-tier; should be ~0
             regardless of rtol -- a numerical sanity check on the
             eigendecomposition/solve arithmetic, not a scientific gate):
             ||Proj_r (Pi W Pi - V) Proj_r||_F / ||Proj_r V Proj_r||_F
        (ii) truncation residual (controlled by rtol/mode retention;
             REPORTED, not gated here -- its physical consequence is
             gated downstream at consumer parity):
             ||V - Proj_r V Proj_r||_F / ||V||_F

    Args:
        Pi: (n,n) array, Hermitian PSD expected (Hermitized internally
            regardless).
        V: (n,n) array, Hermitized internally.
        rtol: relative spectral retention threshold (default 1e-8).
        target_truncation_residual: optional. If given, after the
            rtol-based retention, ADDITIONAL modes (in decreasing
            eigenvalue order) are retained one at a time until the
            truncation residual meets this target or all n modes are
            retained -- recorded via adaptive_retention_used=True and
            the realized target. The default rtol makes NO promise
            about the truncation residual; this is how a caller opts
            into one.

    Returns:
        (W, info) where info is a dict with keys: n_retained,
        n_discarded, s_max, s_min_retained (None if n_retained==0),
        pi_anti_hermitian_residual, v_anti_hermitian_residual,
        retained_solve_residual, truncation_residual, rtol,
        adaptive_retention_used, target_truncation_residual (the
        realized target, or None), dtype, backend ("numpy").

    Raises:
        ValueError: Pi/V are not square/matching-shape 2-D arrays, or
            rtol/target_truncation_residual are not finite positive
            numbers.
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

    if isinstance(rtol, bool) or not isinstance(rtol, (int, float)):
        raise ValueError(f"rtol must be a finite positive float, got {rtol!r}.")
    rtol = float(rtol)
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be a finite positive float, got {rtol!r}.")

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
    threshold = rtol * s_max
    n_retained = int(np.sum(eigvals > threshold))

    v_norm = max(float(np.linalg.norm(V_herm)), tiny)
    adaptive_retention_used = False

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
        "adaptive_retention_used": adaptive_retention_used,
        "target_truncation_residual": target_truncation_residual,
        "dtype": str(W.dtype),
        "backend": "numpy",
    }
    return W, info


@jax.jit
def _hermitian_sandwich_solve_core(Pi, V, rtol):
    """Fixed-shape, jitted, device-resident core of
    hermitian_sandwich_solve_device (design v2.1 sections 5+6/7).
    Reproduces hermitian_sandwich_solve's math exactly, restructured so
    retained-rank truncation is a boolean MASK over the full n-dimensional
    eigenbasis rather than a dynamic-size slice (eigvecs[:, :n_retained]),
    which is required for a static-shape jax.jit graph -- masked-out modes
    contribute exactly 0 to both the pseudo-inverse and the retained-space
    projector, which is mathematically identical to slicing them away.
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
    mask = eigvals > threshold
    n_retained = jnp.sum(mask)
    has_retained = n_retained > 0

    safe_eigvals = jnp.where(mask, eigvals, 1.0)
    inv_eigvals = jnp.where(mask, 1.0 / safe_eigvals, 0.0)
    Pi_pinv_r = (eigvecs * inv_eigvals[None, :]) @ eigvecs.conj().T
    W_full = Pi_pinv_r @ V_herm @ Pi_pinv_r
    W_full = (W_full + W_full.conj().T) / 2

    mask_c = mask.astype(eigvecs.dtype)
    proj_r = (eigvecs * mask_c[None, :]) @ eigvecs.conj().T
    retained_target = proj_r @ V_herm @ proj_r
    retained_solve = proj_r @ (Pi_herm @ W_full @ Pi_herm - V_herm) @ proj_r
    v_norm = jnp.maximum(jnp.linalg.norm(V_herm), tiny)
    retained_solve_residual_full = jnp.linalg.norm(retained_solve) / jnp.maximum(
        jnp.linalg.norm(retained_target), tiny
    )
    truncation_residual_full = jnp.linalg.norm(V_herm - retained_target) / v_norm

    W = jnp.where(has_retained, W_full, jnp.zeros_like(W_full))
    retained_solve_residual = jnp.where(has_retained, retained_solve_residual_full, 0.0)
    truncation_residual = jnp.where(has_retained, truncation_residual_full, 1.0)
    s_min_retained = jnp.min(jnp.where(mask, eigvals, jnp.inf))

    return (
        W, n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
        v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
    )


def hermitian_sandwich_solve_device(Pi, V, *, rtol=1e-8):
    """Device (JAX, fixed-shape, jitted) counterpart of
    hermitian_sandwich_solve (design v2.1 sections 5+6/7): the
    same two-sided Hermitian sandwich solve, restructured to avoid
    dynamic-shape slicing so the whole computation is one fixed-shape
    jax.jit graph, entirely device-resident -- see
    _hermitian_sandwich_solve_core's docstring for the masking rewrite.

    Two known simplifications vs the NumPy oracle, both deliberate scope
    reductions for a first device implementation, not silent bugs:
      - No hard PSD validation. Raising a Python ValueError from inside a
        jax.jit graph on a TRACED value (s_max) is not a plain `if`/raise
        the way host-side NumPy code can do it. A genuinely non-PSD or
        zero Pi degrades to n_retained=0 (W=0), matching the NumPy
        oracle's OWN n_retained==0 fallback branch, rather than raising
        the oracle's separate "Pi not PSD" ValueError. Callers on the
        device path are expected to have already validated Pi's
        PSD-ness via the NumPy oracle during development/testing.
      - No adaptive-retention mode (target_truncation_residual): that is
        inherently a variable-iteration-count algorithm, which does not
        fit a fixed-shape jitted graph. It remains a NumPy-host-only
        diagnostic feature.

    Args:
        Pi: (n,n) array, Hermitian PSD expected (Hermitized internally).
        V: (n,n) array, Hermitized internally.
        rtol: relative spectral retention threshold (default 1e-8).

    Returns:
        (W, info): W is (n,n) complex128 jax array. info is a dict with
        the same keys as hermitian_sandwich_solve's, plus
        backend="jax"; adaptive_retention_used is always False and
        target_truncation_residual is always None (see above).

    Raises:
        ValueError: Pi/V are not square/matching-shape 2-D arrays,
            not finite, or rtol is not a finite positive number (all
            host-side checks on the untraced inputs).
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
    if isinstance(rtol, bool) or not isinstance(rtol, (int, float)):
        raise ValueError(f"rtol must be a finite positive float, got {rtol!r}.")
    rtol = float(rtol)
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be a finite positive float, got {rtol!r}.")

    Pi_jnp = jnp.asarray(Pi_np, dtype=jnp.complex128)
    V_jnp = jnp.asarray(V_np, dtype=jnp.complex128)
    if Pi_jnp.dtype != jnp.complex128:
        logger.warning(
            f"hermitian_sandwich_solve_device: resolved dtype is {Pi_jnp.dtype}, not "
            f"complex128 -- verify jax.config.update('jax_enable_x64', True) is set "
            f"before trusting production numbers from this path (design v2.1 section 1 "
            f"fixes c128 as the only tier with defined 1e-6-class gates)."
        )

    (
        W, n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
        v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
    ) = _hermitian_sandwich_solve_core(Pi_jnp, V_jnp, rtol)

    n_retained_i = int(n_retained)
    info = {
        "n_retained": n_retained_i,
        "n_discarded": n - n_retained_i,
        "s_max": float(s_max),
        "s_min_retained": None if n_retained_i == 0 else float(s_min_retained),
        "pi_anti_hermitian_residual": float(pi_anti_hermitian_residual),
        "v_anti_hermitian_residual": float(v_anti_hermitian_residual),
        "retained_solve_residual": float(retained_solve_residual),
        "truncation_residual": float(truncation_residual),
        "rtol": rtol,
        "adaptive_retention_used": False,
        "target_truncation_residual": None,
        "dtype": str(W.dtype),
        "backend": "jax",
    }
    return W, info
