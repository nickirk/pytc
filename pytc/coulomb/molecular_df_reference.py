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

    Args:
        factor_p_at_pivots: (n_p, n_pivots) weighted values at the
            pivot points (e.g. occ_weighted[:, pivots]).
        factor_q_at_pivots: (n_q, n_pivots) weighted values at the
            pivot points for the pair's other factor.

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


def compute_Z(P, C, rcond=1e-6):
    """Z = S^-1 C C^dagger S^-1, S = P P^dagger.

    Uses the Moore-Penrose pseudoinverse (not a direct solve) since S
    can be singular/ill-conditioned -- decision 001 itself calls this
    the "least-squares core," and pytc.coulomb.pivot_selection's own
    reconstruction-error tests measured real rank-deficiency and large
    condition numbers on Gram matrices of this same algebraic form
    (task #5, 2026-07-12), so treating S as exactly invertible would be
    the wrong default here.

    rcond matters a lot here, not just as a safety margin: on an H2O/
    cc-pVDZ ov-sector test (n_pivots=300, n_pair=95), S measured
    rank~76-95 out of 300 but condition number ~2e24 -- numpy's default
    pinv rcond (~1e-15, tuned for "ordinary" ill-conditioning) retains
    orders of magnitude too many near-zero singular values as if they
    were real signal, giving reconstruction errors >>1 (nonsense) instead
    of the correct sub-percent error. Swept rcond empirically on that
    same test: default/1e-12 -> 39x relative error (nonsense), 1e-6 ->
    1.4e-3 (good), continuing to shrink rcond past ~3e-7 makes it worse
    again (readmits noise). Default here (1e-6) is the empirically-
    verified sweet spot for that test, not a generic numpy default --
    expose it so callers can re-tune per system if needed.

    Args:
        P: (n_pivots, n_pair) pair-collocation matrix at the pivots.
        C: (n_pivots, n_aux) from compute_C_streamed.
        rcond: Relative singular-value cutoff for S's pseudoinverse.

    Returns:
        Z: (n_pivots, n_pivots).
    """
    P = np.asarray(P)
    C = np.asarray(C)
    S = P @ P.conj().T
    S_inv = np.linalg.pinv(S, rcond=rcond)
    return S_inv @ (C @ C.conj().T) @ S_inv


def reconstruct_eri_block(P_row, Z, P_col):
    """V ~= P^dagger Z P for a given pair of (possibly different) sectors'
    pair-collocation matrices -- e.g. P_row from the "ov" sector and
    P_col from the "ov" sector again gives the (ia|jb) block MP2 needs.

    Args:
        P_row: (n_pivots, n_pair_row) pair-collocation at the pivots
            for the ERI's bra pair index.
        Z: (n_pivots, n_pivots) from compute_Z.
        P_col: (n_pivots, n_pair_col) for the ERI's ket pair index.

    Returns:
        (n_pair_row, n_pair_col) reconstructed ERI block (flattened
        pair axis -- reshape to the caller's own (p, q) shape).
    """
    P_row = np.asarray(P_row)
    P_col = np.asarray(P_col)
    return P_row.conj().T @ Z @ P_col
