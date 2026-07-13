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
                          max_jitter_tries: int = 8, jitter_growth: float = 10.0):
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
    geometrically escalates it until the Cholesky factor is both finite
    AND passes a regularized-solve backward-error check -- handles
    matrices that are SPD in exact arithmetic but numerically
    indefinite/near-singular (e.g. a near-full-rank pivot-selection
    Gram matrix) without ever going through an SVD, which is a
    non-starter at production core sizes on GPU.

    STATUS (Alice's audit of commit b59c6ce, task #8 commit 3,
    2026-07-12): the diagnostic-candidate label from commit b59c6ce
    still applies -- this fixes gap 1 (jitter floor) and half of gap 2
    (regularized-solve backward-error gating jitter escalation, added
    here) of Alice's 4-item review; the unregularized-bias check /
    TSVD-fallback half of gap 2, plus gaps 3 (production-scalable
    residual) and 4 (cheap same-sector detection), live in
    df/fit.py's _cholesky_jitter_sandwich since they need the actual
    fit context this generic primitive doesn't have. This solver's
    provenance label is "unscaled_cholesky_jitter" (not "cholesky_jitter"
    matching the doc rule exactly) until row equilibration exists.
    Production default is declared only after the task-#3-mandated
    benchmark at representative N_mu.

    Args:
        matrix: (n, n) symmetric/Hermitian PSD matrix.
        rcond: Relative jitter scale (fraction of the matrix's own
            diagonal mean) used as the STARTING jitter before escalation.
        max_jitter_tries: Escalation attempts before giving up.
        jitter_growth: Geometric growth factor per escalation attempt.

    Returns:
        (chol, lower, jitter_used, n_tries): chol/lower are cho_factor's
        own outputs (pass to jsp_linalg.cho_solve); jitter_used is the
        final (possibly escalated) jitter value actually applied;
        n_tries is how many attempts it took (1 = no escalation needed)
        -- both are provenance fields for callers that need to record
        the solver's own diagnostics (Coulomb path's compute_Z).

    Raises:
        ValueError: matrix's diagonal mean is not finite.
        numpy.linalg.LinAlgError: matrix's diagonal mean is
            non-positive (not PSD -- previously masked by an absolute
            eps*max(diag_mean, 1.0) floor that injected eps-scale
            jitter regardless of the matrix's own scale, Alice's gap 1),
            or the factor never passes both checks within
            max_jitter_tries attempts.
    """
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
            backward_error = _cholesky_backward_error(chol, lower, mat_reg)
            if backward_error <= _CHOLESKY_BACKWARD_ERROR_TOL:
                if attempt > 0:
                    logger.warning(
                        "Cholesky jitter escalated: base=%.3e final=%.3e tries=%d "
                        "backward_error=%.3e",
                        base_jitter, jitter, attempt + 1, backward_error
                    )
                return chol, bool(lower), float(jitter), attempt + 1
            last_backward_error = backward_error
        last_chol = chol

    raise np.linalg.LinAlgError(
        f"Adaptive Cholesky failed after {max_jitter_tries} tries; "
        f"base_jitter={base_jitter:.3e}, last_nonfinite={bool(jnp.any(jnp.isnan(last_chol)))}, "
        f"last_backward_error={last_backward_error}"
    )


def prepare_normal_equations_solver(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                    rcond: float = 1e-14,
                                    max_jitter_tries: int = 8,
                                    jitter_growth: float = 10.0):
    """Prepare robust Cholesky factor for repeated batched solves.

    Thin wrapper around prepare_spd_cholesky (TC's original entry point,
    preserved with its existing (chol, lower)-only return contract --
    see prepare_spd_cholesky's docstring for the shared jitter-escalation
    algorithm and its own 4-tuple return, used by callers that need the
    extra provenance fields).
    """
    ata = _build_normal_matrix(phi_piv_p, phi_piv_q)
    chol, lower, _jitter_used, _n_tries = prepare_spd_cholesky(
        ata, rcond=rcond, max_jitter_tries=max_jitter_tries, jitter_growth=jitter_growth)
    return chol, lower


@partial(jax.jit, static_argnames=('lower',))
def solve_normal_equations_batch_prepared(chol: jnp.ndarray, lower: bool,
                                          phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                          phi_p_batch: jnp.ndarray, phi_q_batch: jnp.ndarray) -> jnp.ndarray:
    """Solve batched normal equations using precomputed Cholesky factor."""
    term_p = jnp.matmul(phi_piv_p.T, phi_p_batch)
    term_q = jnp.matmul(phi_piv_q.T, phi_q_batch)
    atb = term_p * term_q
    return jsp_linalg.cho_solve((chol, lower), atb)
