"""MolecularDFReference kernel-policy Z core (task #6, isdf-coulomb-cuda
decision 001): "Parity oracle vs gpu4pyscf DF-MP2/DF-CCSD. Streams
analytic 3-center batches, contracts into C = P B†, discards; never
stores the full 3-index MO tensor. Z = S⁻¹ C C† S⁻¹."

Notation (Alice's spec, decision 001): pair-collocation matrix
A[g,a] = sqrt(w_g) phi_p(r_g) phi_q(r_g) for pair index a=(p,q);
P[mu,a] = A[r_mu,a] (pair collocation AT the interpolation points mu
selected by pivot_selection.py); S = P P^dagger; with orthonormalized
DF factor V = B^dagger B (B = the Cholesky-factorized cderi blocks
pytc.coulomb.gpu4pyscf_adapter.stream_df_cderi_blocks yields, in MO-pair
basis after transform): C = P B^dagger, Z = S^-1 C C^dagger S^-1. ERIs
are then approximated as V ~= P^dagger Z P -- the only unavoidably
dense object is the (n_pivots, n_pivots) Z core, never the full
(n_pair, n_pair) or (n_aux, n_pair) tensors.
"""

import logging

import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from pyscf import lib

from pytc.coulomb.gpu4pyscf_adapter import stream_df_cderi_blocks
from pytc.df import prepare_spd_cholesky

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
    (Alice's task #6 review, 2026-07-12: passing weighted values here
    instead made compute_Z's rcond default silently depend on the grid
    quadrature's weight scale/level, a portability bug).

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


def compute_C_streamed(mf, P, mo_coeff_p, mo_coeff_q, auxbasis="weigend", blksize=None):
    """Stream analytic 3-center DF batches, AO->MO transform each to the
    (p, q) pair space, contract against P, and accumulate C = P B^dagger
    -- never storing the full (n_aux, n_pair) B tensor at once
    (task #6's "contract-discard" requirement).

    Args:
        mf: Converged mean-field object (pyscf or gpu4pyscf RHF) --
            forwarded to stream_df_cderi_blocks.
        P: (n_pivots, n_p * n_q) pair-collocation matrix at the pivots
            (from pair_collocation_at_pivots), for the SAME (p, q)
            orbital sets as mo_coeff_p/mo_coeff_q.
        mo_coeff_p: (n_ao, n_p) MO coefficients for the pair's first
            index set (e.g. occupied).
        mo_coeff_q: (n_ao, n_q) MO coefficients for the pair's second
            index set (e.g. virtual).
        auxbasis, blksize: forwarded to stream_df_cderi_blocks.

    Returns:
        C: (n_pivots, n_aux) host-numpy array.
    """
    mo_coeff_p = np.asarray(mo_coeff_p)
    mo_coeff_q = np.asarray(mo_coeff_q)
    n_ao = mo_coeff_p.shape[0]
    n_p = mo_coeff_p.shape[1]
    n_q = mo_coeff_q.shape[1]

    c_chunks = []
    for block in stream_df_cderi_blocks(mf, auxbasis=auxbasis, blksize=blksize):
        naux_block = block.shape[0]
        # block is packed lower-triangular (naux_block, nao_pair) AO cderi.
        block_ao = lib.unpack_tril(block).reshape(naux_block, n_ao, n_ao)
        # AO -> MO pair-space transform, one aux index at a time avoided
        # via a single batched einsum (still only THIS block's worth of
        # memory, discarded once this loop iteration ends).
        block_mo = np.einsum("up,auv,vq->apq", mo_coeff_p, block_ao, mo_coeff_q,
                              optimize=True)
        block_mo_flat = block_mo.reshape(naux_block, n_p * n_q)
        c_chunks.append(P @ block_mo_flat.T)  # (n_pivots, naux_block)

    if not c_chunks:
        raise ValueError("stream_df_cderi_blocks yielded no blocks -- empty DF object?")
    return np.concatenate(c_chunks, axis=1)


_RESIDUAL_WARN_THRESHOLD = 1e-10  # design doc v1.1 §4: acceptance = 1e-10 fit residual
_RESIDUAL_NORM_CONVENTION_SAME = "||S Z S - C C^dagger|| / ||C C^dagger||"
_RESIDUAL_NORM_CONVENTION_CROSS = "||S_A Z_AB S_B - C_A C_B^dagger|| / ||C_A C_B^dagger||"
_CHOLESKY_JITTER_MAX_TRIES = 7  # design doc v1.1 §4: up to 6 retries (1 initial + 6)


def _two_sided_residual(S_A, Z, S_B, M):
    """Alice's solver-residual convention (task #6/#3 review, 2026-07-12):
    the FULL two-sided residual on the actual returned Z, not a
    one-sided check of only the first S^-1 solve -- ``||S_A Z S_B -
    M|| / ||M||`` (same-sector: S_A=S_B=S, matches ``||S Z S -
    C C^dagger||/||C C^dagger||`` exactly)."""
    residual_num = float(np.linalg.norm(S_A @ Z @ S_B - M))
    residual_den = float(np.linalg.norm(M))
    return residual_num / residual_den if residual_den > 0.0 else 0.0


def _warn_if_residual_large(fit_residual, solver_label, extra=""):
    if fit_residual > _RESIDUAL_WARN_THRESHOLD:
        logger.warning(
            f"compute_Z ({solver_label}): two-sided fit residual {fit_residual:.3e} "
            f"exceeds the design-doc acceptance threshold {_RESIDUAL_WARN_THRESHOLD:.0e}"
            f"{extra} -- this is a WARNING, not a hard rejection (full acceptance also "
            f"requires the downstream ERI/energy spot-check, which this function cannot "
            f"see); downstream results should be treated with extra suspicion."
        )


def _cholesky_jitter_sandwich(S_A, S_B, M, rcond):
    """S_A^-1 M S_B^-1 via pytc.df.prepare_spd_cholesky (design doc §4,
    isdf-coulomb-cuda, 2026-07-12: "Cholesky with adaptive diagonal
    jitter is the production solver for S^-1-type applications" -- the
    same idiom TC's own orbital fitting uses, not a parallel
    implementation). S_A is S_B (same object) for the same-sector case
    (compute_Z); different for cross-sector (compute_Z_cross). Jitter
    schedule matches the doc's adaptive rule exactly: lambda_0 =
    rcond * trace(S)/N_mu (rcond default 1e-14), x10 growth, up to 6
    retries (7 total attempts).

    Returns (Z, provenance) -- provenance carries exactly the
    solver-level fields compute_Z/compute_Z_cross can actually observe
    (solver, jitter/retries, dtype, two-sided fit residual + its norm
    convention + acceptance threshold, row-scaling); fields that need
    caller-side context (kernel policy, upstream SCF/grid provenance)
    are NOT compute_Z's to fabricate -- a higher-level build_core
    wrapper assembles those (Alice/Felix, 2026-07-12, part of phase 1's
    core-build contract, not deferred to CCSD step 3).
    """
    S_A = jnp.asarray(S_A)
    S_B = jnp.asarray(S_B)
    M = jnp.asarray(M)
    if S_A.dtype in (jnp.float32, jnp.complex64):
        logger.warning(
            f"compute_Z (cholesky_jitter): resolved dtype is {S_A.dtype} -- JAX defaults "
            f"to float32 SILENTLY unless the caller has enabled "
            f"jax.config.update('jax_enable_x64', True), downcasting float64 numpy inputs "
            f"without any error. Measured impact on the H2O/cc-pVDZ ov-sector checkpoint: "
            f"ERI relative error 0.51% (float32) vs 2.0e-4 (float64), a ~25x precision "
            f"loss -- verify x64 is enabled before trusting production numbers from this path."
        )
    same_sector = S_B is S_A or (S_A.shape == S_B.shape and bool(jnp.array_equal(S_A, S_B)))

    chol_A, lower_A, jitter_A, tries_A = prepare_spd_cholesky(
        S_A, rcond=rcond, max_jitter_tries=_CHOLESKY_JITTER_MAX_TRIES)
    if same_sector:
        chol_B, lower_B, jitter_B, tries_B = chol_A, lower_A, jitter_A, tries_A
    else:
        chol_B, lower_B, jitter_B, tries_B = prepare_spd_cholesky(
            S_B, rcond=rcond, max_jitter_tries=_CHOLESKY_JITTER_MAX_TRIES)

    X = jsp_linalg.cho_solve((chol_A, lower_A), M)  # S_A^-1 M
    Z = jsp_linalg.cho_solve((chol_B, lower_B), X.conj().T).conj().T  # (S_A^-1 M) S_B^-1
    Z_np = np.asarray(Z)

    fit_residual = _two_sided_residual(np.asarray(S_A), Z_np, np.asarray(S_B), np.asarray(M))
    _warn_if_residual_large(
        fit_residual, "cholesky_jitter",
        extra=f" (jitter_A={jitter_A:.3e}, tries_A={tries_A})")

    provenance = {
        "solver": "cholesky_jitter",
        "cutoff": rcond,
        "jitter_used": (jitter_A, jitter_A) if same_sector else (jitter_A, jitter_B),
        "n_tries": (tries_A, tries_A) if same_sector else (tries_A, tries_B),
        "retained_singular_value_range": None,  # not applicable to this solver
        "dtype": str(Z_np.dtype),
        "fit_residual": fit_residual,
        "residual_norm_convention": _RESIDUAL_NORM_CONVENTION_SAME if same_sector else _RESIDUAL_NORM_CONVENTION_CROSS,
        "residual_warn_threshold": _RESIDUAL_WARN_THRESHOLD,
        "row_scaling": "identity",
    }
    return Z_np, provenance


def _tsvd_sandwich(S_A, S_B, M, rcond):
    """S_A^+ M S_B^+ via an EXPLICIT truncated-SVD pseudoinverse (not
    np.linalg.pinv's black box) so the retained singular-value range can
    be reported in provenance -- the diagnostics/fallback solver mode
    per design doc §4 (production default is cholesky_jitter).
    """
    S_A = np.asarray(S_A)
    S_B = np.asarray(S_B)
    M = np.asarray(M)
    same_sector = S_B is S_A or (S_A.shape == S_B.shape and np.array_equal(S_A, S_B))

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
    fit_residual = _two_sided_residual(S_A, Z, S_B, M)
    _warn_if_residual_large(fit_residual, "tsvd")

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
    }
    return Z, provenance


def compute_Z(P, C, rcond=None, solver="cholesky_jitter"):
    """Z = S^-1 C C^dagger S^-1, S = P P^dagger.

    Production solver is Cholesky with adaptive diagonal jitter
    (isdf-coulomb-cuda design doc §4, 2026-07-12) -- decided on measured
    evidence, not assumed: at production core sizes an SVD is a
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
    past ~3e-7 made it WORSE again (readmitted noise). Alice's task #6
    review (2026-07-12, blocker item 1) identified the root cause: P
    should be built from RAW (unweighted) values -- weighting is a
    pivot-SELECTION device, not part of the actual interpolation
    formula. With that fix, S on the same test has rank EXACTLY equal
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

    Returns:
        (Z, provenance): Z is (n_pivots, n_pivots); provenance is a
        dict with the solver-level fields from design doc §4 this
        function can observe directly (solver, jitter/cutoff + retry
        history or retained singular-value range, fit residual,
        row-scaling). Fields needing caller-side context (kernel
        policy, upstream SCF/grid provenance, pivot-index hashes) are
        NOT fabricated here -- assemble those at the call site that
        actually has them.
    """
    P = np.asarray(P)
    C = np.asarray(C)
    S = P @ P.conj().T
    M = C @ C.conj().T
    if solver == "cholesky_jitter":
        rcond_eff = 1e-14 if rcond is None else rcond
        return _cholesky_jitter_sandwich(S, S, M, rcond_eff)
    elif solver == "tsvd":
        return _tsvd_sandwich(S, S, M, rcond)
    else:
        raise ValueError(f"solver must be 'cholesky_jitter' or 'tsvd', got {solver!r}")


def compute_Z_cross(P_A, C_A, P_B, C_B, rcond=None, solver="cholesky_jitter"):
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
        rcond, solver: see compute_Z's docstring.

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
        return _cholesky_jitter_sandwich(S_A, S_B, M, rcond_eff)
    elif solver == "tsvd":
        return _tsvd_sandwich(S_A, S_B, M, rcond)
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
    P_col/Z shapes will fail at the matmul (Alice's task #6 review,
    2026-07-12 -- this docstring previously implied a same-sector Z
    could serve any P_row/P_col pairing).

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
