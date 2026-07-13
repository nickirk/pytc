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

pair_collocation_at_pivots/compute_Z/compute_Z_cross/reconstruct_eri_block
moved to pytc.df.fit in the pytc/df/ package reorganization (task #8,
2026-07-12) -- kernel-agnostic LS-THC core-fit algebra with zero
Coulomb-specific content, shared by the future Poisson builder and any
TC channel. Re-exported here for backward compatibility (every
importer of this module keeps working unchanged). compute_C_streamed
stays here: it is MolecularDFReference-policy-specific (streams via
pytc.coulomb.gpu4pyscf_adapter.stream_df_cderi_blocks).
"""

import numpy as np
from pyscf import lib

from pytc.coulomb.gpu4pyscf_adapter import stream_df_cderi_blocks
from pytc.df.fit import (
    pair_collocation_at_pivots,
    compute_Z,
    compute_Z_cross,
    reconstruct_eri_block,
)

__all__ = [
    "pair_collocation_at_pivots",
    "compute_C_streamed",
    "compute_Z",
    "compute_Z_cross",
    "reconstruct_eri_block",
]


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
