"""Sector-aware, fixed-rank interpolation-point (pivot) selection for the
ISDF/LS-THC Coulomb-integral pipeline (task #5, isdf-coulomb-cuda decision
001: "pair-collocation + pivoted-Cholesky pivot selection... sector-aware
(oo/ov/vv), fixed-rank, matrix-free").

Built on pytc.df.pivoted_cholesky_pair_pivots, the jastrow-independent
primitive both the TC pipeline (pytc/df.py's own _pivoted_cholesky_phi/
_pivoted_cholesky_grad, now thin wrappers around it) and this module
import -- no duplicated pivot-selection algorithm (Felix's refactor
upgrade over "port", 2026-07-12).

Notation matches Alice's interface spec (isdf-coulomb-cuda decision 001):
pair-collocation A[g,a] = sqrt(w_g) * phi_p(r_g) * phi_q(r_g) for pair
index a=(p,q); interpolation points are a rank-limited subset of grid
points g selected via pivoted Cholesky on A's Gram matrix, without ever
materializing A itself -- see pivoted_cholesky_pair_pivots's docstring
for the algebraic identity that makes this matrix-free.
"""

import logging

import jax.numpy as jnp
import numpy as np

from pytc.df import pivoted_cholesky_pair_pivots

logger = logging.getLogger(__name__)


def weight_mo_values(mo_values, weights):
    """Apply sqrt(weight) scaling for pivot selection, matching pytc.df's
    own convention (isdf_decompose: ``phi_weighted = phi * w_sqrt``).

    Args:
        mo_values: (n_mo, n_grid) MO values on a grid (e.g. from
            transforming coulomb.gpu4pyscf_adapter's AO values by
            mo_coeff).
        weights: (n_grid,) integration weights.

    Returns:
        (n_mo, n_grid) weighted MO values.
    """
    w_sqrt = jnp.sqrt(jnp.abs(jnp.asarray(weights)))
    return jnp.asarray(mo_values) * w_sqrt[None, :]


def select_sector_pivots(factor_p_weighted, factor_q_weighted, n_rank, shift=None,
                          on_over_rank="truncate"):
    """Select interpolation points for one MO-pair sector, at most n_rank
    of them.

    Args:
        factor_p_weighted: (n_p, n_grid) weighted MO values for the
            pair's first index set (e.g. occupied MOs for "oo"/"ov").
        factor_q_weighted: (n_q, n_grid) weighted MO values for the
            pair's second index set. Pass factor_p_weighted itself for
            a same-set sector ("oo", "vv"); pass a different array for
            a mixed sector ("ov").
        n_rank: Requested rank (number of interpolation points) for this
            sector -- phase-1 uses one rank per sector, not an
            adaptive/error-driven stopping rule (validation ladder step
            1 measures how error actually falls with rank; that data
            informs later rank choices, not this call).
        shift: Tikhonov-style diagonal regularization. None (default)
            auto-derives it the same way pytc.df.isdf_decompose does:
            1e-12 * max(|diag_err|), computed from the actual inputs
            rather than a fixed constant, so it scales with the
            sector's own numerical range.
        on_over_rank: What to do if n_rank exceeds the sector's true
            numerical rank (pytc.df.pivoted_cholesky_pair_pivots's
            effective_rank) -- "truncate" (default) returns only the
            effective_rank genuinely meaningful pivots, logging a
            warning; "raise" raises ValueError instead. Either way,
            arbitrary residual-exhausted pivots are never silently
            returned as if they were production-quality interpolation
            points (Alice's task #6 re-review, 2026-07-12, blocker 3).

    Returns:
        pivots: (k,) selected grid-point indices, k = min(n_rank,
        effective_rank).
    """
    if on_over_rank not in ("truncate", "raise"):
        raise ValueError(f"on_over_rank must be 'truncate' or 'raise', got {on_over_rank!r}")
    factor_p_weighted = jnp.asarray(factor_p_weighted)
    factor_q_weighted = jnp.asarray(factor_q_weighted)
    if shift is None:
        diag_err = jnp.sum(factor_p_weighted**2, axis=0) * jnp.sum(factor_q_weighted**2, axis=0)
        shift = 1e-12 * jnp.max(jnp.abs(diag_err))
    pivots, effective_rank = pivoted_cholesky_pair_pivots(factor_p_weighted, factor_q_weighted, n_rank, shift)
    if effective_rank < n_rank:
        if on_over_rank == "raise":
            raise ValueError(
                f"n_rank={n_rank} exceeds this sector's true numerical rank "
                f"(effective_rank={effective_rank}) -- request a smaller "
                f"n_rank or pass on_over_rank='truncate'."
            )
        logger.warning(
            f"select_sector_pivots: n_rank={n_rank} requested but only "
            f"effective_rank={effective_rank} pivots carry real numerical "
            f"signal -- truncating to {effective_rank} pivots rather than "
            f"returning residual-exhausted, numerically-arbitrary padding."
        )
        pivots = pivots[:effective_rank]
    return pivots


def select_pivots_oo_ov_vv(mo_values, n_occ, weights, n_rank_oo, n_rank_ov, n_rank_vv, shift=None):
    """Select interpolation points for the oo, ov, and vv pair sectors.

    "No occupied-only shortcut" (decision 001's interface spec) -- CCSD
    needs all three MO pair spaces, so this always computes all three
    sectors' pivots rather than defaulting to an occupied-only subset.

    Args:
        mo_values: (n_mo, n_grid) MO values on a grid, occupied MOs
            first (indices [0, n_occ)), virtuals after ([n_occ, n_mo)).
        n_occ: Number of occupied MOs.
        weights: (n_grid,) integration weights.
        n_rank_oo, n_rank_ov, n_rank_vv: Fixed rank per sector.
        shift: Forwarded to select_sector_pivots for all three sectors
            (None auto-derives per-sector, see its docstring).

    Returns:
        dict with keys "oo", "ov", "vv", each an (n_rank_*,) array of
        selected grid-point indices for that sector.
    """
    mo_values = jnp.asarray(mo_values)
    n_mo = mo_values.shape[0]
    if not (0 < n_occ < n_mo):
        raise ValueError(f"n_occ={n_occ} must be strictly between 0 and n_mo={n_mo}.")

    mo_weighted = weight_mo_values(mo_values, weights)
    occ_weighted = mo_weighted[:n_occ]
    virt_weighted = mo_weighted[n_occ:]

    return {
        "oo": select_sector_pivots(occ_weighted, occ_weighted, n_rank_oo, shift),
        "ov": select_sector_pivots(occ_weighted, virt_weighted, n_rank_ov, shift),
        "vv": select_sector_pivots(virt_weighted, virt_weighted, n_rank_vv, shift),
    }


def pair_collocation_reconstruction_error(factor_p_weighted, factor_q_weighted, pivots):
    """Relative Frobenius-norm error of reconstructing the FULL
    pair-product Gram matrix ``G = (Phi_p^T Phi_p) * (Phi_q^T Phi_q)``
    (the same (n_grid, n_grid) object pivoted_cholesky_pair_pivots
    implicitly factors) from only the selected pivot rows/columns via
    least-squares interpolation -- ``G_hat = G[:, piv] @ pinv(G[piv,
    piv]) @ G[piv, :]``, the standard ISDF reconstruction check.

    This is a SMALL-SYSTEM correctness/diagnostic tool -- it explicitly
    materializes the (n_grid, n_grid) Gram matrix, which the pivot
    SELECTION itself (pivoted_cholesky_pair_pivots) never does. Intended
    for the rank-curve harness (validation ladder step 1: "ERI-block
    errors vs exact DF blocks, rank-convergence curves"), not for
    production-scale use.

    Args:
        factor_p_weighted: (n_p, n_grid) weighted MO values.
        factor_q_weighted: (n_q, n_grid) weighted MO values.
        pivots: (n_rank,) selected grid-point indices.

    Returns:
        Relative Frobenius-norm error ||G - G_hat||_F / ||G||_F.
    """
    factor_p_weighted = np.asarray(factor_p_weighted)
    factor_q_weighted = np.asarray(factor_q_weighted)
    pivots = np.asarray(pivots)

    gram_p = factor_p_weighted.T @ factor_p_weighted
    gram_q = factor_q_weighted.T @ factor_q_weighted
    gram = gram_p * gram_q

    gram_piv_cols = gram[:, pivots]              # G[:, piv], (n_grid, n_rank)
    gram_piv_rows = gram[pivots, :]              # G[piv, :], (n_rank, n_grid)
    gram_piv_piv = gram[np.ix_(pivots, pivots)]  # G[piv, piv], (n_rank, n_rank)
    # Standard Nystrom approximation: G_hat = G[:,piv] @ pinv(G[piv,piv]) @ G[piv,:].
    # At full rank (pivots == all indices) this reduces to G @ pinv(G) @ G == G
    # exactly (a Moore-Penrose pseudoinverse identity, holds regardless of
    # rank), so the error must vanish there -- the earlier version of this
    # function used gram_piv_cols[pivots, :] (== gram_piv_piv again) instead
    # of the actual G[piv,:] row-slice, degenerating coeffs into an identity
    # and breaking the shape/correctness of the full-rank case.
    coeffs = np.linalg.pinv(gram_piv_piv) @ gram_piv_rows
    gram_hat = gram_piv_cols @ coeffs

    denom = np.linalg.norm(gram)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(gram - gram_hat) / denom)


def rank_curve(factor_p, factor_q, weights, ranks, shift=None):
    """Rank-convergence curve stub for validation ladder step 1: for
    each candidate rank, select that many pivots and report the
    pair-collocation reconstruction error AND cond(S) (S = P P^dagger
    at that rank).

    cond(S) is included alongside the error deliberately (Felix,
    2026-07-12, following task #6's rcond debugging): rank
    recommendations that only report reconstruction error can pick a
    rank that happens to work numerically by accident of pivot
    selection, without flagging that S is already astronomically
    ill-conditioned at that rank (task #6 measured cond(S)~2e24 at
    rank=300 for a 95-dim H2O/cc-pVDZ pair space) -- production rank
    sizing should stay meaningfully below pair-space saturation, and
    this number is how a caller would notice they're not.

    Takes RAW (unweighted) factors + weights, not pre-weighted arrays
    (Alice's task #6 review, 2026-07-12): pivot SELECTION and the
    reconstruction-error diagnostic both legitimately operate in
    weighted space (that's the space pivoted_cholesky_pair_pivots
    itself factors), but cond(S) must characterize the SAME S that
    compute_Z will actually invert in production -- which is built from
    RAW pair-collocation values (see pair_collocation_at_pivots's
    docstring). Weighting only the selection step, not the P/cond(S)
    computation, keeps this diagnostic representative of what compute_Z
    will see.

    Small-system diagnostic tool (see pair_collocation_reconstruction_error's
    docstring on its (n_grid, n_grid) materialization) -- not intended
    for production-scale rank selection, only for characterizing how
    error falls with rank on representative small/medium test systems
    to inform fixed-rank choices elsewhere.

    Args:
        factor_p, factor_q: (n_p/n_q, n_grid) RAW (unweighted) MO
            values defining the sector (pass the same array twice for
            "oo"/"vv"; different arrays for "ov").
        weights: (n_grid,) integration weights, used only for pivot
            selection and the weighted-space reconstruction-error
            diagnostic.
        ranks: Iterable of candidate n_rank values, ascending.
        shift: Forwarded to select_sector_pivots (None auto-derives).

    Returns:
        List of (rank, relative_error, cond_S) tuples, one per input rank.
    """
    from pytc.coulomb.molecular_df_reference import pair_collocation_at_pivots

    factor_p = np.asarray(factor_p)
    factor_q = np.asarray(factor_q)
    factor_p_weighted = np.asarray(weight_mo_values(factor_p, weights))
    factor_q_weighted = np.asarray(weight_mo_values(factor_q, weights))
    results = []
    for n_rank in ranks:
        pivots = np.asarray(select_sector_pivots(factor_p_weighted, factor_q_weighted, n_rank, shift))
        error = pair_collocation_reconstruction_error(factor_p_weighted, factor_q_weighted, pivots)
        P = pair_collocation_at_pivots(factor_p[:, pivots], factor_q[:, pivots])
        S = P @ P.conj().T
        cond_S = float(np.linalg.cond(S))
        results.append((n_rank, error, cond_S))
    return results
