"""ISDF/LS-THC Coulomb-integral factorization for post-HF correlation
(pytc/integrals/coulomb.py, task #14 corrective reorganization,
isdf-coulomb-cuda, 2026-07-13) -- canonical peer module to
pytc/integrals/{tc,xtc}.py, consolidating the former separate Coulomb
subpackage's four files (gpu4pyscf_adapter, pivot_selection,
molecular_df_reference, build_core) into one, per Ke's ruling that
there is no separate Coulomb subpackage. Design decisions live in the
isdf-coulomb-cuda repo (github.com/nickirk/isdf-coulomb-cuda,
docs/decisions/) -- that repo is the docs/decision-record store only,
this file is the code.

Sections below, in dependency-safe order (each only calls into
sections above it):
  1. gpu4pyscf adapter -- extracts SCF quantities (mo_coeff, grid AO
     values/weights, streamed DF cderi blocks) from a converged
     mean-field object, backend-agnostic (plain pyscf or gpu4pyscf).
  2. Pivot / interpolation-point selection -- sector-aware (oo/ov/vv),
     fixed-rank, built on pytc.df.pivoted_cholesky_pair_pivots.
  3. MolecularDFReference kernel policy -- compute_C_streamed (streams
     analytic 3-center DF batches, contract-discards into
     C = P B^dagger) plus pair_collocation_at_pivots/compute_Z/
     compute_Z_cross/reconstruct_eri_block, imported and re-exported
     from the kernel-agnostic pytc.df.fit for backward-compatible
     single-import access.
  4. build_core orchestration -- SectorFit/CoreArtifact typed,
     immutable, provenance-carrying artifacts; build_sector/build_core
     assemble a complete, reproducible LS-THC/ISDF core for one sector
     (same-sector) or a pair of sectors (cross-sector).
"""

import dataclasses
import hashlib
import logging
import types

import jax.numpy as jnp
import numpy as np
import pyscf
from pyscf import dft, df, lib

from pytc.df import pivoted_cholesky_pair_pivots
from pytc.df.fit import (
    pair_collocation_at_pivots,
    compute_Z,
    compute_Z_cross,
    reconstruct_eri_block,
)

logger = logging.getLogger(__name__)

__all__ = [
    "get_mo_coeff",
    "get_grid_ao_values_and_weights",
    "get_naux",
    "stream_df_cderi_blocks",
    "weight_mo_values",
    "select_sector_pivots",
    "select_pivots_oo_ov_vv",
    "pair_collocation_reconstruction_error",
    "rank_curve",
    "pair_collocation_at_pivots",
    "compute_C_streamed",
    "compute_Z",
    "compute_Z_cross",
    "reconstruct_eri_block",
    "SectorFit",
    "CoreArtifact",
    "build_sector",
    "build_core",
]


# ===========================================================================
# 1. gpu4pyscf adapter
# ===========================================================================
"""Adapter extracting SCF quantities from a converged mean-field object
for the ISDF/LS-THC Coulomb-integral-factorization pipeline (task #4,
isdf-coulomb-cuda decision 001: "gpu4pyscf owns SCF; this project
consumes mo_coeff and AO values/weights on a grid as its upstream
interface").

Backend-agnostic by design: every function here accepts either a plain
pyscf.scf.hf.RHF (CPU, testable without CUDA) or a gpu4pyscf.scf.hf.RHF
(GPU) and returns host numpy regardless. gpu4pyscf keeps mo_coeff/grid
data on-device as cupy arrays until explicitly converted (gpu4pyscf's
own gpu4pyscf/lib/utils.py:to_cpu uses the same isinstance(val,
cupy.ndarray) + .get() idiom _to_host mirrors here, without importing
cupy so this module stays importable on CPU-only hosts).
"""


def _to_host(x):
    """Convert a possibly-cupy array to host numpy; numpy arrays and
    plain Python scalars pass through via np.asarray unchanged."""
    get = getattr(x, "get", None)
    return get() if callable(get) else np.asarray(x)


def get_mo_coeff(mf):
    """Host-numpy MO coefficients, shape (n_ao, n_mo).

    Args:
        mf: A converged mean-field object (pyscf or gpu4pyscf RHF).

    Raises:
        ValueError: if mf.mo_coeff is None (mf.kernel() not yet run).
    """
    if mf.mo_coeff is None:
        raise ValueError("mf.mo_coeff is None -- run mf.kernel() first.")
    return _to_host(mf.mo_coeff)


def get_grid_ao_values_and_weights(mf, grid_lvl=2, deriv=0):
    """DFT-grid AO values, weights, and coordinates.

    Follows pytc's own from_pyscf convention exactly
    (pytc/integrals/tc.py's TC.from_pyscf): dft.gen_grid.Grids(mol)
    built at `grid_lvl`, dft.numint.eval_ao at the resulting coords.

    Deliberately uses PLAIN pyscf.dft here, never gpu4pyscf.dft, even
    when `mf` came from gpu4pyscf -- gpu4pyscf.dft.gen_grid pads the
    grid with zero-weight ghost points and re-sorts points into atomic
    groups for its own GPU integration scheme (an internal performance
    detail), so its grid point count/order does not match plain
    pyscf's Grids at the same level. Using pyscf.dft keeps the grid
    backend-independent and directly comparable to pytc's existing
    dense/DF reference paths (this step is CPU-bound regardless of
    which backend ran the SCF, so there's no performance reason to use
    the GPU grid here).

    Args:
        mf: A converged mean-field object (pyscf or gpu4pyscf RHF).
            Only mf.mol is used.
        grid_lvl: dft.gen_grid.Grids level (pytc's default: 2).
        deriv: AO derivative order forwarded to dft.numint.eval_ao.
            0 = values only, shape (n_grid, n_ao). >0 = shape
            (comp, n_grid, n_ao) (see pyscf's eval_ao for the component
            ordering convention).

    Returns:
        (ao_values, weights, coords).
    """
    mol = mf.mol
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_lvl
    grids.build()
    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)
    ao_values = np.asarray(dft.numint.eval_ao(mol, coords, deriv=deriv))
    return ao_values, weights, coords


def get_naux(with_df):
    """Number of auxiliary (density-fitting) basis functions, backend-
    agnostic.

    gpu4pyscf.df.df.DF exposes a `.naux` attribute (set during
    `.build()`); plain pyscf.df.df.DF exposes `.get_naoaux()` instead
    and has no `.naux` attribute -- this checks for the attribute
    first so a gpu4pyscf DF is never made to instantiate a method it
    doesn't have.
    """
    naux = getattr(with_df, "naux", None)
    if naux is not None:
        return int(naux)
    return int(with_df.get_naoaux())


def stream_df_cderi_blocks(mf, auxbasis="weigend", blksize=None):
    """Yield Cholesky-factorized density-fitting blocks L_P^{pq} one
    aux-index block at a time, instead of materializing the full
    (naux, nao_pair) tensor.

    Backend dispatch: gpu4pyscf.df.df.DF.loop() yields cupy
    (unpacked_tensor, packed_slab) tuples; plain pyscf.df.df.DF.loop()
    yields a packed numpy block directly. Both are converted through
    _to_host so callers see numpy regardless of backend -- the same
    "loop over aux blocks, consume packed slab" shape pytc's own DF-CCSD
    path already relies on (pytc/solver/xtc_ccsd.py's _init_df_eris).

    Reuses mf.with_df if the mean-field already carries one (same
    fallback convention as pytc/solver/xtc_ccsd.py's density_fit());
    otherwise builds a fresh DF object with `auxbasis`, dispatched to
    the SAME backend as `mf` -- a gpu4pyscf mf gets a
    gpu4pyscf.df.df.DF, not pyscf's, since gpu4pyscf's DF assumes a
    cupy-resident integral setup and isn't a drop-in substitute.

    Args:
        mf: A converged mean-field object (pyscf or gpu4pyscf RHF).
        auxbasis: Auxiliary basis name, used only when `mf` has no
            existing with_df.
        blksize: Forwarded to with_df.loop(); None lets the backend's
            own memory-based heuristic choose.

    Yields:
        (naux_block, nao_pair) host-numpy packed lower-triangular
        cderi blocks.
    """
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        if type(mf).__module__.startswith("gpu4pyscf"):
            from gpu4pyscf import df as gpu4pyscf_df
            with_df = gpu4pyscf_df.df.DF(mf.mol, auxbasis=auxbasis)
        else:
            with_df = df.DF(mf.mol, auxbasis=auxbasis)
        with_df.build()

    for block in with_df.loop(blksize):
        if isinstance(block, tuple):
            # gpu4pyscf.df.df.DF.loop(): (unpacked_cupy, packed_slab)
            _, packed = block
            yield _to_host(packed)
        else:
            # pyscf.df.df.DF.loop(): packed numpy block directly
            yield _to_host(block)


# ===========================================================================
# 2. Pivot / interpolation-point selection
# ===========================================================================
"""Sector-aware, fixed-rank interpolation-point (pivot) selection for the
ISDF/LS-THC Coulomb-integral pipeline (task #5, isdf-coulomb-cuda decision
001: "pair-collocation + pivoted-Cholesky pivot selection... sector-aware
(oo/ov/vv), fixed-rank, matrix-free").

Built on pytc.df.pivoted_cholesky_pair_pivots, the jastrow-independent
primitive both the TC pipeline (pytc/df/isdf.py's own
_pivoted_cholesky_phi/_pivoted_cholesky_grad, thin wrappers around it)
and this section import -- no duplicated pivot-selection algorithm
(Felix's refactor upgrade over "port", 2026-07-12).

Notation matches Alice's interface spec (isdf-coulomb-cuda decision 001):
pair-collocation A[g,a] = sqrt(w_g) * phi_p(r_g) * phi_q(r_g) for pair
index a=(p,q); interpolation points are a rank-limited subset of grid
points g selected via pivoted Cholesky on A's Gram matrix, without ever
materializing A itself -- see pivoted_cholesky_pair_pivots's docstring
for the algebraic identity that makes this matrix-free.
"""


def weight_mo_values(mo_values, weights):
    """Apply sqrt(weight) scaling for pivot selection, matching pytc.df's
    own convention (isdf_decompose: ``phi_weighted = phi * w_sqrt``).

    Args:
        mo_values: (n_mo, n_grid) MO values on a grid (e.g. from
            transforming this module's own adapter-section AO values by
            mo_coeff).
        weights: (n_grid,) integration weights.

    Returns:
        (n_mo, n_grid) weighted MO values.
    """
    w_sqrt = jnp.sqrt(jnp.abs(jnp.asarray(weights)))
    return jnp.asarray(mo_values) * w_sqrt[None, :]


def select_sector_pivots(factor_p_weighted, factor_q_weighted, n_rank, shift=None,
                          on_over_rank="truncate", same_factor=False, effective_rank_rtol=1e-6,
                          return_provenance=False):
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
        same_factor: True for a symmetric sector ("oo", "vv" -- same MO
            subset on both sides of the pair). Symmetric sectors have a
            pair-product-space rank of AT MOST n*(n+1)/2, not n**2 --
            phi_p*phi_q == phi_q*phi_p as functions on the grid, so
            (p,q) and (q,p) are literally identical columns of the pair
            matrix. This is an UPPER bound, not necessarily the exact
            rank (additional grid/orbital degeneracies can reduce it
            further, which is exactly why the numerical effective_rank
            check still runs even after this analytic pre-cap -- Alice's
            re-review, 2026-07-12). False (default) uses the generic
            n_p*n_q upper bound.
        effective_rank_rtol: Forwarded to pivoted_cholesky_pair_pivots --
            relative tolerance (fraction of the raw diagonal's own max)
            for counting a selection as carrying real signal. Default
            1e-6 matches the library default (see that function's
            docstring for the empirical calibration); expose here so
            callers can re-tune per system without reaching past this
            wrapper.
        return_provenance: False (default, unchanged return type) or
            True to additionally return a provenance dict distinguishing
            requested_rank, analytic_rank_bound, n_rank_capped,
            rank_exhausted, numerical_rank, numerical_rank_lower_bound,
            and n_pivots (len of the final, possibly truncated, pivots
            array) -- these are DISTINCT concepts a caller assembling a
            full provenance record (e.g. this module's own build_core
            orchestration section, below) must not conflate; in
            particular n_pivots must never be presented as
            "effective/numerical rank" on its own (Alice's task #8
            build_core API review, 2026-07-12).

            numerical_rank semantics (Alice's SECOND build_core review,
            2026-07-12): pivoted_cholesky_pair_pivots only ever observes
            a PREFIX of length n_rank_capped. If every candidate pivot
            in that capped prefix remains "effective" (no truncation),
            that proves only rank >= n_rank_capped -- NOT that
            n_rank_capped is the true rank (independent repro: random
            3x12/3x12 factors, requested_rank=2 gave a capped run
            reporting "numerical_rank=2" while the array's TRUE rank
            was 9). So numerical_rank is the EXACT measured rank
            (=effective_rank) ONLY when rank_exhausted is True
            (effective_rank < n_rank_capped, i.e. the algorithm actually
            ran out of real signal within the capped pool); otherwise
            numerical_rank is None and only numerical_rank_lower_bound
            (=effective_rank, which then equals n_rank_capped) is known.

    Returns:
        pivots: (k,) selected grid-point indices, k <= n_rank (bounded
        by both the analytic pair-rank cap and effective_rank). If
        return_provenance=True, returns (pivots, provenance) instead.
    """
    if on_over_rank not in ("truncate", "raise"):
        raise ValueError(f"on_over_rank must be 'truncate' or 'raise', got {on_over_rank!r}")
    if not (0.0 < effective_rank_rtol < 1.0):
        raise ValueError(
            f"effective_rank_rtol must satisfy 0 < rtol < 1, got {effective_rank_rtol!r} "
            f"-- it is a fraction of the raw diagonal's own max (see "
            f"pytc.df.pivoted_cholesky_pair_pivots's docstring), so 0 admits pure noise "
            f"as 'effective' and >=1 rejects all real signal (Alice's task #6 re-review, "
            f"2026-07-12, non-blocking hardening item)."
        )
    factor_p_weighted = jnp.asarray(factor_p_weighted)
    factor_q_weighted = jnp.asarray(factor_q_weighted)

    n_p = factor_p_weighted.shape[0]
    n_q = factor_q_weighted.shape[0]
    analytic_rank_bound = n_p * (n_p + 1) // 2 if same_factor else n_p * n_q
    n_rank_capped = min(n_rank, analytic_rank_bound)
    if n_rank_capped < n_rank:
        logger.info(
            f"select_sector_pivots: n_rank={n_rank} exceeds the sector's "
            f"analytic pair-rank upper bound ({analytic_rank_bound}"
            f"{'=n*(n+1)/2, symmetric sector' if same_factor else '=n_p*n_q'}) "
            f"-- pre-capping the request to {n_rank_capped} before pivot "
            f"selection rather than asking the numerical primitive to "
            f"discover an already-known bound."
        )

    if shift is None:
        diag_err = jnp.sum(factor_p_weighted**2, axis=0) * jnp.sum(factor_q_weighted**2, axis=0)
        shift = 1e-12 * jnp.max(jnp.abs(diag_err))
    pivots, effective_rank = pivoted_cholesky_pair_pivots(
        factor_p_weighted, factor_q_weighted, n_rank_capped, shift,
        track_effective_rank=True, effective_rank_rtol=effective_rank_rtol)
    if effective_rank < n_rank_capped:
        if on_over_rank == "raise":
            raise ValueError(
                f"n_rank={n_rank} exceeds this sector's true numerical rank "
                f"(effective_rank={effective_rank}) -- request a smaller "
                f"n_rank or pass on_over_rank='truncate'."
            )
        logger.warning(
            f"select_sector_pivots: n_rank_capped={n_rank_capped} requested but "
            f"only effective_rank={effective_rank} pivots carry real numerical "
            f"signal -- truncating to {effective_rank} pivots rather than "
            f"returning residual-exhausted, numerically-arbitrary padding."
        )
        pivots = pivots[:effective_rank]
    elif n_rank_capped < n_rank and on_over_rank == "raise":
        raise ValueError(
            f"n_rank={n_rank} exceeds this sector's analytic pair-rank "
            f"upper bound ({analytic_rank_bound}) -- request a smaller n_rank "
            f"or pass on_over_rank='truncate'."
        )
    if not return_provenance:
        return pivots
    rank_exhausted = bool(effective_rank < n_rank_capped)
    provenance = {
        "requested_rank": int(n_rank),
        "analytic_rank_bound": int(analytic_rank_bound),
        "n_rank_capped": int(n_rank_capped),
        "rank_exhausted": rank_exhausted,
        "numerical_rank": int(effective_rank) if rank_exhausted else None,
        "numerical_rank_lower_bound": int(effective_rank),
        "n_pivots": int(len(pivots)),
    }
    return pivots, provenance


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
        "oo": select_sector_pivots(occ_weighted, occ_weighted, n_rank_oo, shift, same_factor=True),
        "ov": select_sector_pivots(occ_weighted, virt_weighted, n_rank_ov, shift),
        "vv": select_sector_pivots(virt_weighted, virt_weighted, n_rank_vv, shift, same_factor=True),
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


# ===========================================================================
# 3. MolecularDFReference kernel policy
# ===========================================================================
"""MolecularDFReference kernel-policy Z core (task #6, isdf-coulomb-cuda
decision 001): "Parity oracle vs gpu4pyscf DF-MP2/DF-CCSD. Streams
analytic 3-center batches, contracts into C = P B†, discards; never
stores the full 3-index MO tensor. Z = S⁻¹ C C† S⁻¹."

Notation (Alice's spec, decision 001): pair-collocation matrix
A[g,a] = sqrt(w_g) phi_p(r_g) phi_q(r_g) for pair index a=(p,q);
P[mu,a] = A[r_mu,a] (pair collocation AT the interpolation points mu
selected by the pivot-selection section above); S = P P^dagger; with
orthonormalized DF factor V = B^dagger B (B = the Cholesky-factorized
cderi blocks stream_df_cderi_blocks, the adapter section above, yields,
in MO-pair basis after transform): C = P B^dagger, Z = S^-1 C C^dagger
S^-1. ERIs are then approximated as V ~= P^dagger Z P -- the only
unavoidably dense object is the (n_pivots, n_pivots) Z core, never the
full (n_pair, n_pair) or (n_aux, n_pair) tensors.

pair_collocation_at_pivots/compute_Z/compute_Z_cross/reconstruct_eri_block
live in pytc.df.fit (the pytc/df/ package reorganization, task #8,
2026-07-12) -- kernel-agnostic LS-THC core-fit algebra with zero
Coulomb-specific content, shared by the future Poisson builder and any
TC channel. Imported and re-exported at the top of this file for
backward-compatible single-import access (e.g.
``from pytc.integrals.coulomb import compute_Z``). compute_C_streamed
stays here: it is MolecularDFReference-policy-specific (streams via
stream_df_cderi_blocks, the adapter section above).
"""


def compute_C_streamed(mf, P, mo_coeff_p, mo_coeff_q, auxbasis="weigend", blksize=None,
                        return_provenance=False):
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
        auxbasis, blksize: forwarded to stream_df_cderi_blocks. NOTE:
            stream_df_cderi_blocks REUSES mf.with_df whenever it is
            already present, in which case `auxbasis` here is silently
            IGNORED -- see return_provenance's effective_auxbasis/
            reused_existing_with_df fields, which record what was
            actually used, not merely what was requested (Alice's
            build_core review, 2026-07-12: a caller-side fingerprint
            built only from the requested auxbasis STRING can diverge
            from the bytes actually streamed).
        return_provenance: False (default, unchanged return type) or
            True to additionally return a provenance dict with
            df_factor_sha256 (an incremental SHA-256 over the raw
            packed DF blocks' bytes AS THEY STREAM, in row order, before
            any AO->MO transform -- the definitive proof that two calls
            consumed the SAME ordered auxiliary factor, regardless of
            what auxbasis string was requested or whether mf.with_df
            was reused), requested_auxbasis, effective_auxbasis (read
            from mf.with_df.auxbasis when an existing with_df was
            reused; equals requested_auxbasis when a fresh DF object
            was built for this call), and reused_existing_with_df.
            df_factor_sha256 is INVARIANT to blksize: the loop only
            ever hashes each block's raw row-major bytes in stream
            order (sequential SHA-256 updates compose identically
            regardless of how the same total byte stream is chunked);
            dtype/packed-column-count/total-auxiliary-row-count are
            folded in ONCE at finalization, not per block, so a
            performance-tuning blksize choice can never change the hash
            (Alice's build_core review, 2026-07-13).

    Returns:
        C: (n_pivots, n_aux) host-numpy array. If return_provenance=True,
        returns (C, provenance) instead.
    """
    mo_coeff_p = np.asarray(mo_coeff_p)
    mo_coeff_q = np.asarray(mo_coeff_q)
    n_ao = mo_coeff_p.shape[0]
    n_p = mo_coeff_p.shape[1]
    n_q = mo_coeff_q.shape[1]

    existing_with_df = getattr(mf, "with_df", None)
    reused_existing_with_df = existing_with_df is not None
    hasher = hashlib.sha256() if return_provenance else None
    total_naux = 0
    nao_pair = None
    block_dtype = None

    c_chunks = []
    for block in stream_df_cderi_blocks(mf, auxbasis=auxbasis, blksize=blksize):
        naux_block = block.shape[0]
        if hasher is not None:
            contiguous_block = np.ascontiguousarray(block)
            hasher.update(contiguous_block.tobytes())
            total_naux += naux_block
            nao_pair = contiguous_block.shape[1]
            block_dtype = contiguous_block.dtype
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
    C = np.concatenate(c_chunks, axis=1)
    if not return_provenance:
        return C

    hasher.update(str(block_dtype).encode())
    hasher.update(str(nao_pair).encode())
    hasher.update(str(total_naux).encode())
    effective_auxbasis = (
        getattr(existing_with_df, "auxbasis", None) if reused_existing_with_df else auxbasis
    )
    provenance = {
        "df_factor_sha256": hasher.hexdigest(),
        "requested_auxbasis": auxbasis,
        "effective_auxbasis": effective_auxbasis,
        "reused_existing_with_df": reused_existing_with_df,
    }
    return C, provenance


# ===========================================================================
# 4. build_core orchestration
# ===========================================================================
"""build_core: assembles a complete, reproducible LS-THC/ISDF core
artifact for one MO-pair sector (same-sector) or a pair of independently
-pivoted sectors (cross-sector), per the isdf-coulomb-cuda design doc
§4's "provenance fields REQUIRED on every core build" list -- pivot
selection, P/C construction, and Z computation, plus the upstream/grid/
pivot provenance compute_Z/compute_Z_cross cannot fabricate themselves
(Alice/Felix's ruling, 2026-07-12: "a Z artifact without grid/pivot/
kernel provenance is irreproducible before any consumer touches it").

Location: pytc/integrals/coulomb.py, NOT pytc/df/ -- this section
CHOOSES the MolecularDFReference kernel policy (calls
compute_C_streamed, the section above), while pytc/df/fit.py stays
kernel-agnostic (Alice, 2026-07-12).

API, per two rounds of Alice's review (2026-07-12):

Round 1 (initial proposal -> approved amendments):
1. Weighted MO-value factors are now DERIVED INTERNALLY from raw
   values + grid_weights (via the pivot-selection section's
   weight_mo_values), not taken as a separate explicit parameter --
   Alice's simplification over her own original "pass both explicitly"
   ask: since grid_weights is already a required parameter (for
   hashing), deriving removes the raw/weighted consistency-mismatch
   class entirely rather than requiring validation of it.
2. Two-stage, typed API: build_sector(...) -> SectorFit(P, C, pivots,
   compatibility_key, provenance) is a reusable artifact; build_core
   (left, right=None) -> CoreArtifact(Z, provenance) -- right=None is
   same-sector (compute_Z), right=<SectorFit> is cross-sector
   (compute_Z_cross). Cross-sector is NOT deferred: conventional CCSD's
   approved contract needs oo/ov/vv same-sector cores AND their cross
   cores (oo|vv, ov|vv) from the start.

Round 2 (re-review of round-1 implementation -> 3 more fixes):
3. numerical_rank is only reported as an EXACT value when
   pivoted_cholesky_pair_pivots actually exhausted the effective prefix
   within the analytically-capped candidate pool (rank_exhausted=True)
   -- if every candidate pivot in a capped selection remained
   "effective", that only proves rank >= n_rank_capped, NOT that
   n_rank_capped IS the true rank (Alice's independent repro: random
   3x12/3x12 factors, requested_rank=2 -> a capped run reported
   numerical_rank=2 while the TRUE rank was 9). numerical_rank=None +
   numerical_rank_lower_bound otherwise.
4. Cross-sector joins now require a matching compatibility_key --
   see round 3 below for its final (corrected) definition.
5. SectorFit/CoreArtifact are genuinely immutable, not just
   shallow-frozen dataclasses: P/C/pivots are DEFENSIVE COPIES with
   the numpy write flag cleared (never the caller's own array object,
   so construction never side-effects the caller's arrays), and
   provenance dicts are recursively frozen (dict -> MappingProxyType,
   list/tuple -> tuple, set -> frozenset) at every nesting level, not
   just the top level -- a shallow top-level MappingProxyType still
   lets nested keys like provenance['upstream_provenance']['x'] = ...
   mutate an already-built artifact.

Round 3 (re-review of round-2 implementation -> 2 more fixes):
6. compatibility_key originally fingerprinted the REQUESTED auxbasis
   argument via pyscf.df.addons.make_auxmol -- but
   stream_df_cderi_blocks silently REUSES mf.with_df whenever already
   present, in which case the requested auxbasis is IGNORED entirely.
   Independently reproduced: an mf with a pre-existing with_df
   (auxbasis "cc-pvdz-jkfit") streamed BIT-IDENTICAL blocks when called
   with two different, both-ignored, requested auxbasis strings, yet
   the two calls' make_auxmol-based keys DIFFERED -- a false rejection
   of a physically identical join. Fixed at the source: compute_C_streamed
   now optionally returns df_factor_sha256, an incremental SHA-256 over
   the ACTUAL packed DF block bytes AS THEY STREAM (blksize-invariant --
   see its docstring) -- the definitive proof two builds consumed the
   same ordered auxiliary factor. compatibility_key now incorporates
   this actual-factor hash instead of an independently-derived auxmol
   fingerprint of the (possibly-ignored) requested auxbasis.
7. _deep_freeze claimed to leave numpy arrays "handled separately" but
   never actually froze arrays nested inside e.g. upstream_provenance
   (only the top-level P/C/pivots went through _readonly_copy) --
   mutating a caller-supplied array nested in upstream_provenance after
   build silently leaked into the already-built artifact. Fixed: an
   explicit np.ndarray branch in _deep_freeze routes through the same
   _readonly_copy used for P/C/pivots.

Also (round 3): explicit ndim/shape validation before any .shape[1]
indexing (grid_coords must be (n_grid, 3); grid_weights (n_grid,);
factors and MO coefficients 2-D) so malformed inputs raise the
documented ValueError rather than an incidental IndexError.
"""

_SUPPORTED_KERNEL_POLICIES = ("MolecularDFReference",)


def _canonical_sha256(*arrays):
    """SHA-256 over one or more arrays' canonicalized bytes (C-contiguous,
    shape/dtype folded into the digest) -- hashlib, never Python's
    built-in hash() (per-process salted, not reproducible across runs).
    """
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(repr(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _readonly_copy(a):
    """Defensive COPY (never the caller's own array object) with the
    numpy write flag cleared. np.asarray on an input that is already an
    ndarray of matching dtype can return the SAME object with no copy --
    calling setflags(write=False) on that result would silently mark
    the CALLER's own array read-only as a side effect. Copy first,
    always (Alice's build_core review, 2026-07-12)."""
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


_DEEP_FREEZE_IMMUTABLE_SCALAR_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _deep_freeze(obj, _path="<root>"):
    """Recursively convert dict -> types.MappingProxyType, list/tuple ->
    tuple, set/frozenset -> frozenset, and numpy arrays/scalars ->
    read-only copies / .item(), at every nesting level -- a shallow
    top-level MappingProxyType still lets nested keys (e.g.
    provenance['upstream_provenance']['x'] = ...) mutate an
    already-built artifact, and arrays nested inside e.g.
    upstream_provenance were previously left untouched entirely,
    silently mutable via any alias the caller kept (Alice's build_core
    review, 2026-07-12, three rounds: shallow top-level freeze, then
    missing array handling).

    CLOSED schema (Alice's 4th round, 2026-07-13): any object that
    isn't one of the above containers, an immutable scalar (None, bool,
    int, float, complex, str, bytes), or a numpy array/scalar raises
    TypeError -- an earlier version silently returned unknown objects
    unchanged, so an arbitrary mutable object (e.g. a caller's own
    class instance) passed via upstream_provenance stayed aliased and
    mutable despite the "genuinely immutable" contract (repro: a plain
    Box object with a mutable .value attribute, passed through
    unchanged, `frozen['box'] is box` and later mutation leaked
    through). Deep-copying unknown objects instead was considered and
    rejected: a copy still would not guarantee immutability (the copy
    itself could contain further-nested mutable state) and would
    silently conceal non-serializable provenance rather than surfacing
    it. Callers must normalize custom objects (e.g. pathlib.Path,
    version objects) to strings before passing them in
    upstream_provenance.

    3 more closures (Alice's 5th round, 2026-07-13 -- the schema was
    still open at edges the container-level checks didn't reach):
    - Mapping KEYS were copied unchanged, only values were frozen -- a
      mutable-but-hashable custom key object stayed aliased (and could
      even corrupt dict lookup if its hash changed after the fact).
      All provenance keys are our own string literals internally, so
      keys are now required to be str (raises TypeError otherwise) --
      free to enforce, closes the hole.
    - np.ndarray with dtype.hasobject (an object-dtype array) was made
      read-only only at the CONTAINER level -- setflags(write=False)
      blocks reassigning array ELEMENTS, but each element is itself a
      reference to an arbitrary Python object, and mutating THAT
      object's own state (Alice's repro: a Box stored in an object
      array) still leaked through despite the array being "read-only".
      Rejected outright (TypeError) rather than attempting to
      recursively normalize each element -- rejection is safer for a
      provenance schema than a partial/best-effort per-element freeze.
    - np.generic (numpy scalar) fed its .item() result straight back to
      the caller without re-validating it -- for plain numeric scalars
      .item() gives an already-immutable Python float/int/bool, but for
      STRUCTURED numpy scalars .item() can return a tuple or an object
      that itself needs recursive validation. Now recurses through
      _deep_freeze(obj.item(), _path) instead of returning it directly.
    """
    if isinstance(obj, types.MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        frozen = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"_deep_freeze: unsupported mapping key at {_path}: "
                    f"{type(k).__name__} ({k!r}) -- provenance mapping keys must "
                    f"be str (all internal provenance keys already are; a "
                    f"non-str/mutable key would stay aliased to the caller)."
                )
            frozen[k] = _deep_freeze(v, f"{_path}[{k!r}]")
        return types.MappingProxyType(frozen)
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v, f"{_path}[{i}]") for i, v in enumerate(obj))
    if isinstance(obj, (set, frozenset)):
        return frozenset(_deep_freeze(v, _path) for v in obj)
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError(
                f"_deep_freeze: unsupported object-dtype array at {_path} "
                f"(dtype={obj.dtype}) -- setflags(write=False) only blocks "
                f"reassigning array elements, not mutating the arbitrary Python "
                f"objects those elements reference. Normalize to a non-object "
                f"dtype or a plain list/tuple of already-immutable values before "
                f"passing it in upstream_provenance."
            )
        return _readonly_copy(obj)
    if isinstance(obj, np.generic):
        return _deep_freeze(obj.item(), _path)
    if isinstance(obj, _DEEP_FREEZE_IMMUTABLE_SCALAR_TYPES):
        return obj
    raise TypeError(
        f"_deep_freeze: unsupported object at {_path}: {type(obj).__name__} "
        f"({obj!r}) -- provenance values must be a Mapping (str keys)/list/"
        f"tuple/set/frozenset/np.ndarray (non-object dtype)/np.generic, or an "
        f"immutable scalar (None, bool, int, float, complex, str, bytes). "
        f"Normalize custom objects (e.g. pathlib.Path, version objects) to "
        f"strings before passing them in upstream_provenance."
    )


def _kernel_compatibility_key(mf, kernel_policy, df_factor_sha256):
    """Fingerprint of everything that must match for two SectorFits' C
    columns to represent the SAME ordered DF factor/kernel instance:
    molecule identity (geometry/AO basis/charge/spin) as human-legible
    supporting metadata, PLUS df_factor_sha256 -- an incremental hash of
    the ACTUAL packed DF block bytes streamed for this build
    (compute_C_streamed's return_provenance option) -- which is the
    DEFINITIVE proof of factor identity, since stream_df_cderi_blocks
    silently reuses mf.with_df whenever present and ignores any
    requested `auxbasis` argument in that case (Alice's build_core
    review, 2026-07-13: a key built only from the REQUESTED auxbasis
    label -- even via a realized-auxmol fingerprint -- can diverge from
    what was actually streamed, in either direction). Also folds in
    kernel_policy and the pyscf version."""
    mol = mf.mol
    h = hashlib.sha256()
    h.update(repr(mol.atom).encode())
    h.update(repr(mol.basis).encode())
    h.update(str(mol.charge).encode())
    h.update(str(mol.spin).encode())
    h.update(_canonical_sha256(mol.atom_coords()).encode())
    h.update(df_factor_sha256.encode())
    h.update(kernel_policy.encode())
    h.update(pyscf.__version__.encode())
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class SectorFit:
    """One MO-pair sector's reusable, IMMUTABLE ISDF/LS-THC fit
    artifact: pivots selected, P and C constructed. Reusable across
    multiple build_core joins (e.g. the same "vv" SectorFit feeds both
    the vv|vv same-sector core and the oo|vv / ov|vv cross-sector
    cores) without recomputing pivot selection or re-streaming the DF
    integrals. P/C/pivots are read-only array copies; provenance is
    recursively frozen -- see _readonly_copy/_deep_freeze."""
    P: np.ndarray
    C: np.ndarray
    pivots: np.ndarray
    compatibility_key: str
    provenance: types.MappingProxyType


@dataclasses.dataclass(frozen=True)
class CoreArtifact:
    """A completed Z core for one sector (same-sector) or one sector
    pair (cross-sector), with full, IMMUTABLE provenance."""
    Z: np.ndarray
    provenance: types.MappingProxyType


def build_sector(mf, factor_p_raw, factor_q_raw, mo_coeff_p, mo_coeff_q,
                  grid_coords, grid_weights, requested_rank,
                  *, same_factor=False, effective_rank_rtol=1e-6, shift=None,
                  on_over_rank="truncate", kernel_policy="MolecularDFReference",
                  auxbasis="weigend", blksize=None, upstream_provenance=None):
    """Select pivots and build P, C for ONE MO-pair sector (e.g. "oo",
    "ov", "vv").

    Args:
        mf: Converged mean-field object -- forwarded to compute_C_streamed
            (kernel_policy="MolecularDFReference"'s DF-B-tensor route)
            and fingerprinted into compatibility_key.
        factor_p_raw, factor_q_raw: (n_p/n_q, n_grid) RAW (unweighted)
            MO values -- used for P construction via
            pair_collocation_at_pivots AND as the source for pivot-
            selection weighting (weight_mo_values(factor, grid_weights)
            is applied internally; passing an independently-supplied
            "weighted" array was removed as an explicit parameter since
            it could drift from raw*sqrt(|weights|) with nothing to
            catch the mismatch).
        mo_coeff_p, mo_coeff_q: (n_ao, n_p/n_q) MO coefficients for this
            sector's two pair indices -- forwarded to compute_C_streamed
            and hashed into provenance. Column count must match
            factor_p_raw/factor_q_raw's row count.
        grid_coords: (n_grid, 3) grid point coordinates -- hashed into
            provenance (grid_sha256) together with grid_weights.
        grid_weights: (n_grid,) integration weights -- used both to
            derive the internal weighted factors and hashed into
            provenance alongside grid_coords.
        requested_rank: Requested pivot-selection rank for this sector.
        same_factor: True for a symmetric sector (oo, vv) -- forwarded
            to select_sector_pivots. Requires factor_p_raw ==
            factor_q_raw and mo_coeff_p == mo_coeff_q (validated,
            raises ValueError otherwise): the n*(n+1)/2 triangular
            analytic rank cap assumes phi_p*phi_q == phi_q*phi_p as
            functions on the grid, which only holds when p and q are
            literally the same orbital set.
        effective_rank_rtol, shift, on_over_rank: forwarded to
            select_sector_pivots.
        kernel_policy: Recorded in provenance and fingerprinted into
            compatibility_key; only "MolecularDFReference" is
            implemented so far (compute_C_streamed's DF-B-tensor route).
        auxbasis, blksize: forwarded to compute_C_streamed. NOTE:
            auxbasis is silently IGNORED by compute_C_streamed whenever
            mf.with_df already exists -- see the requested_auxbasis/
            effective_auxbasis/reused_existing_with_df provenance
            fields, which record what was actually used. Not
            fingerprinted into compatibility_key directly; the ACTUAL
            streamed DF factor bytes (df_factor_sha256) are, which is
            correct regardless of whether auxbasis was honored.
        upstream_provenance: Caller-supplied dict of facts this function
            cannot infer safely from `mf` alone -- gpu4pyscf version,
            the ACTUAL grid settings used to build grid_coords/
            grid_weights/factor_* (level, pruning, atom-grid overrides,
            radii scheme). None becomes an empty dict, recorded as-is
            (not validated -- this function trusts the caller's own
            record; it does NOT participate in compatibility_key, which
            only covers facts this function can independently verify).

    Returns:
        SectorFit(P, C, pivots, compatibility_key, provenance) -- P, C,
        pivots are read-only array copies; provenance is a recursively
        frozen (nested MappingProxyType) dict.

    Raises:
        ValueError: unsupported kernel_policy; grid/factor/mo_coeff
            shape mismatch; same_factor=True with unequal
            factor_p_raw/factor_q_raw or mo_coeff_p/mo_coeff_q.
    """
    if kernel_policy not in _SUPPORTED_KERNEL_POLICIES:
        raise ValueError(
            f"kernel_policy={kernel_policy!r} not implemented -- only "
            f"{_SUPPORTED_KERNEL_POLICIES!r} exist so far (compute_C_streamed's "
            f"DF-B-tensor route)."
        )
    factor_p_raw = np.asarray(factor_p_raw)
    factor_q_raw = np.asarray(factor_q_raw)
    mo_coeff_p = np.asarray(mo_coeff_p)
    mo_coeff_q = np.asarray(mo_coeff_q)
    grid_coords = np.asarray(grid_coords)
    grid_weights = np.asarray(grid_weights)

    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError(f"grid_coords must be (n_grid, 3), got shape {grid_coords.shape}.")
    if grid_weights.ndim != 1:
        raise ValueError(f"grid_weights must be 1-D (n_grid,), got shape {grid_weights.shape}.")
    if factor_p_raw.ndim != 2:
        raise ValueError(f"factor_p_raw must be 2-D (n_p, n_grid), got shape {factor_p_raw.shape}.")
    if factor_q_raw.ndim != 2:
        raise ValueError(f"factor_q_raw must be 2-D (n_q, n_grid), got shape {factor_q_raw.shape}.")
    if mo_coeff_p.ndim != 2:
        raise ValueError(f"mo_coeff_p must be 2-D (n_ao, n_p), got shape {mo_coeff_p.shape}.")
    if mo_coeff_q.ndim != 2:
        raise ValueError(f"mo_coeff_q must be 2-D (n_ao, n_q), got shape {mo_coeff_q.shape}.")

    n_grid = grid_coords.shape[0]
    if grid_weights.shape[0] != n_grid:
        raise ValueError(
            f"grid_weights length ({grid_weights.shape[0]}) != grid_coords length "
            f"({n_grid})."
        )
    if factor_p_raw.shape[1] != n_grid:
        raise ValueError(
            f"factor_p_raw grid axis ({factor_p_raw.shape[1]}) != grid_coords length "
            f"({n_grid})."
        )
    if factor_q_raw.shape[1] != n_grid:
        raise ValueError(
            f"factor_q_raw grid axis ({factor_q_raw.shape[1]}) != grid_coords length "
            f"({n_grid})."
        )
    if mo_coeff_p.shape[1] != factor_p_raw.shape[0]:
        raise ValueError(
            f"mo_coeff_p has {mo_coeff_p.shape[1]} MO columns but factor_p_raw has "
            f"{factor_p_raw.shape[0]} rows -- these must describe the same orbital set."
        )
    if mo_coeff_q.shape[1] != factor_q_raw.shape[0]:
        raise ValueError(
            f"mo_coeff_q has {mo_coeff_q.shape[1]} MO columns but factor_q_raw has "
            f"{factor_q_raw.shape[0]} rows -- these must describe the same orbital set."
        )
    if same_factor:
        if factor_p_raw.shape != factor_q_raw.shape or not np.array_equal(factor_p_raw, factor_q_raw):
            raise ValueError(
                "same_factor=True requires factor_p_raw and factor_q_raw to be the "
                "SAME orbital set (phi_p*phi_q == phi_q*phi_p as functions on the grid "
                "-- the assumption behind the n*(n+1)/2 triangular analytic rank cap) "
                "-- got numerically different arrays."
            )
        if mo_coeff_p.shape != mo_coeff_q.shape or not np.array_equal(mo_coeff_p, mo_coeff_q):
            raise ValueError(
                "same_factor=True requires mo_coeff_p and mo_coeff_q to be the SAME "
                "MO coefficients, consistent with factor_p_raw/factor_q_raw."
            )

    factor_p_weighted = weight_mo_values(factor_p_raw, grid_weights)
    factor_q_weighted = weight_mo_values(factor_q_raw, grid_weights)

    pivots, pivot_provenance = select_sector_pivots(
        factor_p_weighted, factor_q_weighted, requested_rank, shift=shift,
        on_over_rank=on_over_rank, same_factor=same_factor,
        effective_rank_rtol=effective_rank_rtol, return_provenance=True)
    pivots = np.asarray(pivots)

    P = pair_collocation_at_pivots(factor_p_raw[:, pivots], factor_q_raw[:, pivots])
    C, c_provenance = compute_C_streamed(
        mf, P, mo_coeff_p, mo_coeff_q, auxbasis=auxbasis, blksize=blksize,
        return_provenance=True)

    compatibility_key = _kernel_compatibility_key(
        mf, kernel_policy, c_provenance["df_factor_sha256"])

    provenance = {
        "kernel_policy": kernel_policy,
        "kernel_policy_params": {"auxbasis": auxbasis, "blksize": blksize},
        "same_factor": same_factor,
        "effective_rank_rtol": effective_rank_rtol,
        **pivot_provenance,  # requested_rank, analytic_rank_bound, n_rank_capped,
                              # rank_exhausted, numerical_rank, numerical_rank_lower_bound, n_pivots
        **c_provenance,  # df_factor_sha256, requested_auxbasis, effective_auxbasis,
                          # reused_existing_with_df
        "pivot_indices_sha256": _canonical_sha256(pivots),
        "grid_sha256": _canonical_sha256(grid_coords, grid_weights),
        "mo_coeff_p_sha256": _canonical_sha256(mo_coeff_p),
        "mo_coeff_q_sha256": _canonical_sha256(mo_coeff_q),
        "factor_p_raw_sha256": _canonical_sha256(factor_p_raw),
        "factor_q_raw_sha256": _canonical_sha256(factor_q_raw),
        "compatibility_key": compatibility_key,
        "upstream_provenance": dict(upstream_provenance) if upstream_provenance else {},
    }

    return SectorFit(
        P=_readonly_copy(P),
        C=_readonly_copy(C),
        pivots=_readonly_copy(pivots),
        compatibility_key=compatibility_key,
        provenance=_deep_freeze(provenance),
    )


def build_core(left, right=None, rcond=None, solver="cholesky_jitter", **solver_kwargs):
    """Compute a Z core from one (same-sector) or two (cross-sector)
    SectorFit artifacts.

    Args:
        left: SectorFit for the sector (same-sector) or sector A
            (cross-sector).
        right: None (same-sector: calls compute_Z(left.P, left.C)) or a
            SectorFit for sector B (cross-sector: calls
            compute_Z_cross(left.P, left.C, right.P, right.C,
            same_sector=False) -- two independently-built SectorFit
            artifacts are never the literal same sector by construction
            of this API, so same_sector=False is always correct here.
            Use build_core(sf) (right=None), not build_core(sf, sf),
            for the same-sector case). Rejected with ValueError unless
            right.compatibility_key == left.compatibility_key -- same
            n_aux is NOT sufficient evidence that C_left C_right^dagger
            is physically meaningful (they could come from different
            molecules/auxbases/kernel policies that happen to realize
            the same auxiliary dimension).
        rcond, solver, **solver_kwargs: forwarded to compute_Z/
            compute_Z_cross (tsvd_rcond, backward_error_mode,
            backward_error_tol, residual_mode, residual_n_probes,
            residual_seed).

    Returns:
        CoreArtifact(Z, provenance) -- provenance merges compute_Z's/
        compute_Z_cross's own solver-level dict with each side's
        SectorFit provenance (key "sector" when same-sector, keys
        "left_sector"/"right_sector" when cross-sector), recursively
        frozen.

    Raises:
        ValueError: right is not None and
            right.compatibility_key != left.compatibility_key.
    """
    if right is None:
        Z, solver_provenance = compute_Z(
            left.P, left.C, rcond=rcond, solver=solver, **solver_kwargs)
        provenance = {**solver_provenance, "sector": left.provenance}
    else:
        if left.compatibility_key != right.compatibility_key:
            raise ValueError(
                f"Cross-sector join rejected: left.compatibility_key "
                f"({left.compatibility_key[:12]}...) != right.compatibility_key "
                f"({right.compatibility_key[:12]}...) -- these SectorFits do not "
                f"share the same molecule/basis/auxbasis/kernel-policy/pyscf-version "
                f"identity, so C_left C_right^dagger would not represent a physically "
                f"meaningful cross-sector ERI block even if their auxiliary "
                f"dimensions happen to match."
            )
        Z, solver_provenance = compute_Z_cross(
            left.P, left.C, right.P, right.C, rcond=rcond, solver=solver,
            same_sector=False, **solver_kwargs)
        provenance = {
            **solver_provenance,
            "left_sector": left.provenance,
            "right_sector": right.provenance,
        }
    return CoreArtifact(Z=_readonly_copy(Z), provenance=_deep_freeze(provenance))
