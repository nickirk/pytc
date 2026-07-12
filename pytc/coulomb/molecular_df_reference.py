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

import numpy as np
from pyscf import lib

from pytc.coulomb.gpu4pyscf_adapter import stream_df_cderi_blocks


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


def compute_Z(P, C, rcond=None):
    """Z = S^-1 C C^dagger S^-1, S = P P^dagger.

    Uses the Moore-Penrose pseudoinverse (not a direct solve) since S
    can be singular/ill-conditioned -- decision 001 itself calls this
    the "least-squares core," and pytc.coulomb.pivot_selection's own
    reconstruction-error tests measured real rank-deficiency and large
    condition numbers on Gram matrices of this same algebraic form
    (task #5, 2026-07-12), so treating S as exactly invertible would be
    the wrong default here.

    History: this function originally took WEIGHTED (sqrt(w_g)-scaled)
    values into pair_collocation_at_pivots's P, and needed a hand-tuned,
    NON-MONOTONIC rcond sweet spot (default 1e-6) to avoid a real
    catastrophic failure mode -- numpy's own default pinv rcond gave
    39x relative error (nonsense) on an H2O/cc-pVDZ ov-sector test
    (n_pivots=300, n_pair=95, cond(S)~2e24), and rcond tightened much
    past ~3e-7 made it WORSE again (readmitted noise). Alice's task #6
    review (2026-07-12, blocker item 1) identified the root cause: P
    should be built from RAW (unweighted) values (see
    pair_collocation_at_pivots's docstring) -- weighting is a pivot-
    SELECTION device, not part of the actual interpolation formula.
    With that fix, S on the same test has rank EXACTLY equal to
    n_pair=95 (no numerical rank inflation from the weight scaling) and
    the rcond sweep becomes well-behaved: error falls MONOTONICALLY as
    rcond shrinks from 1e-4 (23%) through 1e-6 (2.5%) down to a
    ~1e-6-relative-error plateau at rcond<=3e-10 (no readmitted-noise
    regime observed down to 1e-15) -- numpy's own default rcond
    (~6.7e-14 for this matrix size) already sits on that plateau,
    measured 1.2e-6 relative ERI error. So the numpy default is now the
    right default; rcond is still exposed for callers who need to
    re-tune per system (e.g. much larger/differently-conditioned S).

    Args:
        P: (n_pivots, n_pair) pair-collocation matrix at the pivots
            (RAW/unweighted values -- see pair_collocation_at_pivots).
        C: (n_pivots, n_aux) from compute_C_streamed.
        rcond: Relative singular-value cutoff for S's pseudoinverse.
            None (default) uses numpy's own pinv default.

    Returns:
        Z: (n_pivots, n_pivots).
    """
    P = np.asarray(P)
    C = np.asarray(C)
    S = P @ P.conj().T
    S_inv = np.linalg.pinv(S, rcond=rcond)
    return S_inv @ (C @ C.conj().T) @ S_inv


def compute_Z_cross(P_A, C_A, P_B, C_B, rcond=None):
    """Z_AB = S_A^-1 C_A C_B^dagger S_B^-1, S_A = P_A P_A^dagger, S_B = P_B
    P_B^dagger -- the cross-sector generalization of compute_Z, needed
    when the ERI block's bra and ket pair indices come from DIFFERENT
    MO-pair sectors with their OWN independently-selected pivot sets
    (e.g. CCSD's oo|vv and ov|vv blocks: sector A's pivots need not
    equal, or even overlap with, sector B's pivots -- see
    pivot_selection.select_pivots_oo_ov_vv, which selects oo/ov/vv
    pivots independently).

    compute_Z(P, C, rcond) is exactly this function's same-sector
    special case (P_A=P_B=P, C_A=C_B=C); kept as a separate simpler
    entry point since same-sector Z is CCSD's most common need (oo|oo,
    ov|ov, vv|vv) and callers there shouldn't have to pass every
    argument twice.

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
        rcond: Relative singular-value cutoff for S_A's and S_B's
            pseudoinverses. None (default) uses numpy's own pinv
            default (see compute_Z's docstring on why this is now the
            right default, once P is built from raw/unweighted values).

    Returns:
        Z_AB: (n_pivots_A, n_pivots_B).
    """
    P_A = np.asarray(P_A)
    C_A = np.asarray(C_A)
    P_B = np.asarray(P_B)
    C_B = np.asarray(C_B)
    S_A = P_A @ P_A.conj().T
    S_B = P_B @ P_B.conj().T
    S_A_inv = np.linalg.pinv(S_A, rcond=rcond)
    S_B_inv = np.linalg.pinv(S_B, rcond=rcond)
    return S_A_inv @ (C_A @ C_B.conj().T) @ S_B_inv


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
