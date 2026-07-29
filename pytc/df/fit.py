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

from .solvers import (
    prepare_spd_cholesky,
    _CHOLESKY_BACKWARD_ERROR_TOL,
    _CHOLESKY_JITTER_MAX_TRIES,
    _DEFAULT_RESIDUAL_N_PROBES,
    _DEFAULT_RESIDUAL_SEED,
    _RESIDUAL_NORM_CONVENTION_CROSS,
    _RESIDUAL_NORM_CONVENTION_SAME,
    _RESIDUAL_WARN_THRESHOLD,
    _cholesky_jitter_sandwich,
    _two_sided_residual,
    _two_sided_residual_sampled,
    _compute_residual,
    _tsvd_sandwich,
)

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
