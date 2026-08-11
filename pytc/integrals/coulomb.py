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
  5. Free-space Poisson solver -- isolated/free-space FFT Poisson
     backend on a uniform Cartesian mesh (task #13, isdf-coulomb-cuda,
     P2a). Kernel-independent, low-level: no mf/DF/CCSD dependency,
     pure array-in-array-out. A SEPARATE, independent mesh type from
     the DFT quadrature grid used in section 1 -- both are "grids" in
     this file but serve unrelated purposes (irregular pruned
     quadrature vs. a uniform Cartesian FFT mesh).
  6. Poisson interpolation-vector Z-core assembly (task #15,
     isdf-coulomb-cuda, P2b) -- builds raw physical interpolation
     vectors Theta on section 5's mesh (reusing pytc/df/solvers.py's
     structured normal-equations solver, the SAME primitive
     pytc/df/isdf.py:isdf_decompose already uses for TC's own ISDF
     fitting -- no duplicated selection/solver logic) and assembles
     Z = dV * Theta^dagger V via section 5's approved Poisson backend.
     A genuinely different, more direct kernel-policy route to the
     same Z-core object build_core's MolecularDFReference policy
     (section 4) produces via DF blocks -- no S^-1 regularization step
     needed here, since the free-space Poisson operator gives the
     Coulomb kernel's action directly.
"""

import dataclasses
import hashlib
import logging
import math
import types

import jax
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
from pytc.df.solvers import (
    prepare_normal_equations_solver,
    solve_normal_equations_batch_prepared,
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
    "FreeSpacePoissonMesh",
    "FreeSpacePoissonKernel",
    "build_free_space_poisson_kernel",
    "solve_free_space_poisson",
    "free_space_poisson_direct_sum_oracle",
    "PoissonInterpolationSector",
    "PoissonCoreArtifact",
    "build_poisson_interpolation_sector",
    "poisson_core",
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


# ===========================================================================
# 5. Free-space Poisson solver
# ===========================================================================
"""Isolated/free-space FFT Poisson backend on a uniform Cartesian mesh
(task #13, isdf-coulomb-cuda, P2a, 2026-07-13 -- 3 review rounds with
Alice before implementation, see the design-proposal thread for the
full derivation history).

Solves the free-space Poisson equation on a uniform Cartesian mesh,
equivalently v_i = dV * sum_j rho_j / |r_i - r_j| under the documented
finite-cell discretization (dV = dx*dy*dz). This is a genuinely
SEPARATE mesh type from the DFT quadrature grid used in section 1
(get_grid_ao_values_and_weights) -- both are "grids" in this file but
serve unrelated purposes (irregular pruned quadrature vs. this
uniform Cartesian FFT mesh). Zero mf/DF/CCSD dependency: pure
array-in-array-out, a reusable low-level backend.

Zero-padding for LINEAR (not circular/periodic) convolution: rho is
placed at the padded array's origin corner (no shift), padded_shape >=
2*shape-1 per axis (pad_factor=2 default). Cropping the result back to
v_padded[...,:Nx,:Ny,:Nz] recovers the exact open-boundary sum: for any
i,j in the physical region, |i-j| < N <= P/2, so (i-j) mod P never
aliases (verified both algebraically and via a 1D numerical toy problem
during design).

Self-cell (r=0) term: two schemes.
- "rectangular_cell" (default, production): the EXACT closed-form
  Newton potential of a rectangular prism (the mesh cell itself)
  evaluated at its own center -- see _rectangular_cell_self_potential's
  docstring for the derivation/verification. Numerically stable for any
  valid (positive) cell dimensions, no branching needed.
- "equivalent_sphere" (diagnostic/comparison only, NOT the default):
  models the cell's charge as uniformly spread over a sphere of equal
  volume -- a real, shape-dependent APPROXIMATION whose error grows
  sharply with cell anisotropy (measured: +1.59% at an isotropic cube,
  +23.96% at 1:1:4, +66.74% at 1:1:10 -- see
  test_free_space_poisson.py's rectangular-cell-vs-equivalent-sphere
  test for the permanent regression check). Kept only because it is a
  useful independent cross-check, not because it is accurate for
  anisotropic meshes.

Kernel construction uses the FFT-standard *wrapped* integer coordinate
convention (_wrapped_integer_offsets) so the padded real-space kernel
represents the correct (non-aliased) physical separation for every
pair of points within the zero-padded linear-convolution regime.

Typed, immutable, provenance-carrying artifacts:
- FreeSpacePoissonMesh: pure geometry (shape/spacing/origin/
  padded_shape/self_cell_scheme/fft_normalization), fully validated.
  origin is provenance-only metadata -- it does NOT enter the
  translation-invariant kernel itself.
- FreeSpacePoissonKernel: the cached, expensive-to-build spectrum
  artifact -- ONE per (mesh, fft_kind, backend, dtype). fft_kind is
  explicit ("rfft" for real densities, "fft" for complex) because
  rfftn/fftn produce genuinely different spectrum shapes; separate
  cached artifacts are built for each, never guessed/shared. Backend-
  preserving: a NumPy spectrum is a defensive read-only copy; a JAX
  spectrum stays an untouched (already-immutable) jax.Array on its
  actual device -- never forced through a NumPy-only helper, which
  would silently transfer/duplicate the full padded spectrum on host.
  input_dtype/spectrum_dtype are recorded as distinct concepts (e.g.
  rfft(float64) legitimately produces a complex128 spectrum -- a
  different, expected dtype, not a promotion failure) -- both are the
  REALIZED facts (verified after construction), not merely requested
  strings; JAX silently truncating a requested float64/complex128 to
  float32/complex64 because jax_enable_x64 is disabled raises
  ValueError immediately rather than returning a downgraded-precision
  kernel with only a JAX UserWarning to notice it by.

build_free_space_poisson_kernel does the expensive one-time spectrum
build; solve_free_space_poisson batches many density solves against an
already-built kernel with zero rebuild cost, rejecting fft_kind/shape/
backend/dtype mismatches explicitly rather than silently coercing.
free_space_poisson_direct_sum_oracle is an O(N_g^2) NumPy-only
brute-force reference for small-system correctness testing, sharing
the exact same scalar self-cell-aware kernel-value helper
(_coulomb_kernel_values) as the FFT path, so the two agree on the
self-cell convention by construction, not coincidence -- never intended
for production use.
"""

_FREE_SPACE_POISSON_SOLVER_VERSION = "1"

_REAL_COMPUTE_DTYPE_FOR_INPUT = {
    np.dtype(np.float32): np.dtype(np.float32),
    np.dtype(np.complex64): np.dtype(np.float32),
    np.dtype(np.float64): np.dtype(np.float64),
    np.dtype(np.complex128): np.dtype(np.float64),
}


def _wrapped_integer_offsets(P):
    """Pure-integer FFT wrapped-coordinate offsets -- NOT
    (np.fft.fftfreq(P)*P).astype(int), which risks float-roundoff
    truncation (Alice's review, task #13 design round 2, 2026-07-13).
    idx[i] = i for i <= (P-1)//2, else i - P. Verified identical to the
    float version for P in {1,2,7,8,9,16,17} (odd/even/small/edge
    cases) -- this version does zero floating-point arithmetic, so
    there is no roundoff-truncation risk at any P."""
    idx = np.arange(P)
    return np.where(idx > (P - 1) // 2, idx - P, idx)


def _coulomb_kernel_values(offset_x, offset_y, offset_z, K_self):
    """Vectorized scalar Coulomb kernel: 1/|offset| everywhere except
    exactly at the zero offset, where the resolved self-cell value
    K_self is used instead. Shared by BOTH the padded FFT kernel
    builder and free_space_poisson_direct_sum_oracle, so the two paths
    agree on the self-cell convention by construction, not by
    coincidence."""
    self_mask = (offset_x == 0) & (offset_y == 0) & (offset_z == 0)
    r2 = offset_x**2 + offset_y**2 + offset_z**2
    r_safe = np.where(self_mask, 1.0, np.sqrt(r2))
    return np.where(self_mask, K_self, 1.0 / r_safe)


def _rectangular_cell_self_potential(dx, dy, dz):
    """K_self via the EXACT closed-form Newton potential of a
    rectangular prism (the classical gravity/magnetics "prism
    potential" antiderivative), evaluated at the prism's own center.

    Derivation (task #13 design round 3, 2026-07-13 -- symbolically
    derived and verified via `uv run --with sympy`, not transcribed
    from memory): with half-widths x=dx/2, y=dy/2, z=dz/2 and
    R=sqrt(x^2+y^2+z^2),

        F(x,y,z) = x*y*asinh(z/sqrt(x^2+y^2)) + y*z*asinh(x/sqrt(y^2+z^2))
                   + z*x*asinh(y/sqrt(z^2+x^2))
                   - x^2/2*atan(y*z/(x*R)) - y^2/2*atan(z*x/(y*R))
                   - z^2/2*atan(x*y/(z*R))

    satisfies d^3F/dx dy dz = 1/R EXACTLY (sympy simplifies the
    residual to 0 -- verified, not assumed). The box self-potential
    integral, evaluated via the standard 8-corner inclusion-exclusion
    over this triple antiderivative, has EVERY corner term with at
    least one zero coordinate vanish exactly (also sympy-verified via
    proper limits, not epsilon substitution) -- collapsing to a single
    non-singular evaluation:

        self_cell_integral I = 8 * F(dx/2, dy/2, dz/2)

    Since dx,dy,dz > 0 always (mesh spacing is validated positive), F
    is evaluated ONLY at a fully generic, non-singular point --
    unconditionally numerically stable, no piecewise/degenerate-corner
    branching needed for any valid mesh.

    Cross-validated against an independent scipy adaptive quadrature
    (singularity excluded as a tiny ball, added back analytically) for
    an isotropic cube and two anisotropic ratios (1:1:4, 1:1:10) --
    exact agreement to quadrature precision -- plus an exact scale-
    homogeneity identity I(s*dx,s*dy,s*dz) = s^2*I(dx,dy,dz) (matches
    to 1.8e-15, machine precision).

    Returns (self_cell_integral I, K_self = I/dV) -- K_self is the
    volume-AVERAGED value that belongs in the kernel's own origin
    entry (the solver's separate explicit dV factor then makes a
    single occupied cell's self-potential contribution I*rho, not
    dV*K_self*rho*dV -- do not double up the volume factor)."""
    x, y, z = dx / 2.0, dy / 2.0, dz / 2.0
    R = math.sqrt(x * x + y * y + z * z)
    F = (x * y * math.asinh(z / math.sqrt(x * x + y * y))
         + y * z * math.asinh(x / math.sqrt(y * y + z * z))
         + z * x * math.asinh(y / math.sqrt(z * z + x * x))
         - x * x / 2 * math.atan(y * z / (x * R))
         - y * y / 2 * math.atan(z * x / (y * R))
         - z * z / 2 * math.atan(x * y / (z * R)))
    I = 8.0 * F
    dV = dx * dy * dz
    return I, I / dV


def _equivalent_sphere_self_potential(dx, dy, dz):
    """K_self via the equivalent-sphere approximation: models the
    cell's own charge as uniformly spread over a sphere of the SAME
    VOLUME as the cell, using the exact potential such a uniform
    sphere produces at its own center (3Q/(2R), a standard
    electrostatics closed form, verified during design against a
    direct numerical integration of the uniform-ball potential to
    1e-16).

    This is a documented, real, SHAPE-DEPENDENT APPROXIMATION to the
    true rectangular-cell self-term, not an alternative exact scheme --
    measured error vs _rectangular_cell_self_potential: +1.59% at an
    isotropic cube, growing to +23.96% at a 1:1:4 anisotropic cell and
    +66.74% at 1:1:10 (task #13 design round 3, 2026-07-13). Kept as an
    explicit diagnostic/comparison scheme only -- NOT the production
    default (see _rectangular_cell_self_potential).

    Returns (self_cell_integral I, K_self, equivalent_sphere_radius R).
    """
    dV = dx * dy * dz
    R = (3.0 * dV / (4.0 * math.pi)) ** (1.0 / 3.0)
    K_self = 3.0 / (2.0 * R)
    I = K_self * dV
    return I, K_self, R


def _self_cell_values(self_cell_scheme, dx, dy, dz):
    """Dispatch to the requested self-cell scheme. Returns
    (self_cell_integral, K_self, equivalent_sphere_radius) --
    equivalent_sphere_radius is None for "rectangular_cell" (that
    scheme has no such radius; never fabricate a stale/foreign value)."""
    if self_cell_scheme == "rectangular_cell":
        I, K_self = _rectangular_cell_self_potential(dx, dy, dz)
        return I, K_self, None
    if self_cell_scheme == "equivalent_sphere":
        return _equivalent_sphere_self_potential(dx, dy, dz)
    raise ValueError(f"Unsupported self_cell_scheme={self_cell_scheme!r}.")


def _validate_positive_int(name, value):
    """Reject bool (isinstance(True, int) is True in Python -- an
    explicit exclusion, not an oversight) and non-integral floats
    rather than silently truncating them -- int(3.7)==3 and
    int(True)==1 would otherwise coerce malformed input into a
    plausible-looking but wrong value instead of rejecting it (Alice's
    review, task #13, 2026-07-13: directly reproduced shape=(3.7,4,5)
    -> (3,4,5) and pad_factor=2.7 -> 2 silently accepted)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool {value!r}.")
    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
    elif (isinstance(value, (float, np.floating)) and math.isfinite(value)
          and float(value).is_integer()):
        ivalue = int(value)
    else:
        raise ValueError(f"{name} must be an integer (or integral-valued float), got {value!r}.")
    if ivalue <= 0:
        raise ValueError(f"{name} must be positive, got {ivalue}.")
    return ivalue


def _as_length_tuple(name, values, length=3):
    """Coerce to a tuple of the given length, converting a bare
    TypeError from a non-iterable/scalar input (e.g. spacing=1.0) into
    the documented ValueError instead of letting it leak (Alice's
    review, task #13, 2026-07-13: directly reproduced scalar
    spacing=1.0/origin=0.0 raising 'float object is not iterable')."""
    try:
        values = tuple(values)
    except TypeError:
        raise ValueError(f"{name} must be an iterable of {length} values, got scalar {values!r}.")
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {values!r}.")
    return values


def _validate_positive_int_tuple(name, values, length=3):
    values = _as_length_tuple(name, values, length)
    return tuple(_validate_positive_int(f"{name}[{i}]", v) for i, v in enumerate(values))


def _kernel_spec_sha256(fields):
    """SHA-256 over a canonical repr of small scalar/string kernel-
    specification fields only (mesh geometry, scheme, resolved
    self-cell values, fft_kind, backend, versions, dtypes) -- deliberately
    never touches the full spectrum array, which may live on a JAX
    device (hashing/copying it would force an unwanted host transfer,
    defeating device placement -- Alice's review, task #13 design round
    3, 2026-07-13)."""
    h = hashlib.sha256()
    for key in sorted(fields):
        h.update(key.encode())
        h.update(repr(fields[key]).encode())
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class FreeSpacePoissonMesh:
    """Pure geometry for a free-space Poisson solve on a fixed uniform
    Cartesian mesh -- no spectrum/dtype/backend content (that lives on
    FreeSpacePoissonKernel). shape/padded_shape are corner-anchored:
    grid index (i,j,k) sits at physical coordinate
    origin + (i*dx, j*dy, k*dz). origin is PROVENANCE ONLY -- it does
    not enter the translation-invariant kernel itself."""
    shape: tuple
    spacing: tuple
    origin: tuple
    padded_shape: tuple
    self_cell_scheme: str
    fft_normalization: str

    def __post_init__(self):
        shape = _validate_positive_int_tuple("shape", self.shape)
        padded_shape = _validate_positive_int_tuple("padded_shape", self.padded_shape)

        spacing = _as_length_tuple("spacing", self.spacing)
        spacing = tuple(float(s) for s in spacing)
        if any((not math.isfinite(d)) or d <= 0.0 for d in spacing):
            raise ValueError(f"spacing must be 3 finite positive floats, got {spacing}.")

        origin = _as_length_tuple("origin", self.origin)
        origin = tuple(float(o) for o in origin)
        if any(not math.isfinite(o) for o in origin):
            raise ValueError(f"origin must be 3 finite floats, got {origin}.")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "padded_shape", padded_shape)

        for p, n in zip(padded_shape, shape):
            if p < 2 * n - 1:
                raise ValueError(
                    f"padded_shape entry {p} < 2*{n}-1={2 * n - 1} -- insufficient "
                    f"zero-padding for a correct linear (non-wrapping) convolution."
                )
        if self.self_cell_scheme not in ("rectangular_cell", "equivalent_sphere"):
            raise ValueError(f"Unsupported self_cell_scheme={self.self_cell_scheme!r}.")
        if self.fft_normalization != "backward":
            raise ValueError(
                f"Only fft_normalization='backward' is supported, got "
                f"{self.fft_normalization!r}."
            )


@dataclasses.dataclass(frozen=True)
class FreeSpacePoissonKernel:
    """Cached, expensive-to-build spectrum artifact for a FIXED
    (mesh, fft_kind, backend, dtype) -- built once via
    build_free_space_poisson_kernel, reused across many
    solve_free_space_poisson calls with zero rebuild cost.

    spectrum: NumPy backend -> defensive read-only copy (via this
    file's _readonly_copy). JAX backend -> the untouched (already-
    immutable) jax.Array, left on its actual device -- never forced
    through a NumPy-only helper.
    input_dtype/spectrum_dtype: distinct, REALIZED (verified after
    construction) facts, not merely requested strings -- e.g.
    rfft(float64) legitimately produces a complex128 spectrum.
    equivalent_sphere_radius: only meaningful for
    self_cell_scheme="equivalent_sphere"; None for "rectangular_cell"
    (never a fabricated/foreign value)."""
    mesh: object
    spectrum: object
    fft_kind: str
    backend: str
    input_dtype: str
    spectrum_dtype: str
    device: str
    numpy_version: str
    jax_version: object
    self_cell_integral: float
    K_self: float
    equivalent_sphere_radius: object
    kernel_spec_sha256: str
    solver_version: str

    def __post_init__(self):
        if not isinstance(self.mesh, FreeSpacePoissonMesh):
            raise TypeError(f"mesh must be a FreeSpacePoissonMesh, got {type(self.mesh).__name__}.")
        if self.fft_kind not in ("rfft", "fft"):
            raise ValueError(f"Unsupported fft_kind={self.fft_kind!r}.")
        if self.backend not in ("numpy", "jax"):
            raise ValueError(f"Unsupported backend={self.backend!r}.")

        # Backend/spectrum-type consistency, then make the artifact
        # ACTUALLY immutable here (not merely by builder convention) --
        # Alice's review, task #13, 2026-07-13: directly constructed a
        # nominal kernel with a mutable NumPy spectrum and mutated it
        # successfully through the public dataclass.
        if self.backend == "numpy":
            if not isinstance(self.spectrum, np.ndarray):
                raise TypeError(
                    f"backend='numpy' requires spectrum to be a numpy.ndarray, got "
                    f"{type(self.spectrum).__name__}."
                )
            object.__setattr__(self, "spectrum", _readonly_copy(self.spectrum))
        else:
            if not isinstance(self.spectrum, jax.Array):
                raise TypeError(
                    f"backend='jax' requires spectrum to be a jax.Array, got "
                    f"{type(self.spectrum).__name__}."
                )

        expected_shape = (
            (*self.mesh.padded_shape[:-1], self.mesh.padded_shape[-1] // 2 + 1)
            if self.fft_kind == "rfft" else self.mesh.padded_shape
        )
        if tuple(self.spectrum.shape) != expected_shape:
            raise ValueError(
                f"spectrum shape {tuple(self.spectrum.shape)} does not match the "
                f"expected {self.fft_kind} shape {expected_shape} for padded_shape "
                f"{self.mesh.padded_shape}."
            )

        declared_spectrum_dtype = np.dtype(self.spectrum_dtype)
        if np.dtype(self.spectrum.dtype) != declared_spectrum_dtype:
            raise ValueError(
                f"spectrum.dtype={self.spectrum.dtype} != declared spectrum_dtype="
                f"{declared_spectrum_dtype}."
            )
        if not np.issubdtype(declared_spectrum_dtype, np.complexfloating):
            raise ValueError(
                f"spectrum_dtype must be complex (rfftn/fftn spectra are always "
                f"complex-valued), got {declared_spectrum_dtype}."
            )

        input_dtype = np.dtype(self.input_dtype)
        if input_dtype not in _REAL_COMPUTE_DTYPE_FOR_INPUT:
            raise ValueError(
                f"input_dtype must be one of float32/float64/complex64/complex128, "
                f"got {input_dtype}."
            )
        if self.fft_kind == "rfft" and not np.issubdtype(input_dtype, np.floating):
            raise ValueError(f"fft_kind='rfft' requires a real input_dtype, got {input_dtype}.")
        if self.fft_kind == "fft" and not np.issubdtype(input_dtype, np.complexfloating):
            raise ValueError(f"fft_kind='fft' requires a complex input_dtype, got {input_dtype}.")

        # Provenance closure (Alice's review, task #13, 2026-07-13):
        # `dataclasses.replace` on a genuine builder-produced kernel can
        # still construct a nominally-typed but internally INCONSISTENT
        # artifact (self_cell_integral/K_self disagreeing with each
        # other and with mesh geometry; device="gpu999"; a stale/
        # mismatched numpy_version/jax_version/solver_version; a
        # kernel_spec_sha256 that doesn't match its own declared
        # fields) -- all independently reproduced. Recompute every one
        # of these from first principles (the mesh, the realized
        # spectrum/backend, and library __version__ strings) and
        # require exact/tight-tolerance agreement, rather than trusting
        # any of them as passed in.
        I_expected, K_self_expected, R_expected = _self_cell_values(
            self.mesh.self_cell_scheme, *self.mesh.spacing)
        if not math.isclose(self.self_cell_integral, I_expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"self_cell_integral={self.self_cell_integral!r} does not match the "
                f"value recomputed from mesh.self_cell_scheme/spacing ({I_expected!r})."
            )
        if not math.isclose(self.K_self, K_self_expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"K_self={self.K_self!r} does not match the value recomputed from "
                f"mesh.self_cell_scheme/spacing ({K_self_expected!r})."
            )
        if R_expected is None:
            if self.equivalent_sphere_radius is not None:
                raise ValueError(
                    "equivalent_sphere_radius must be None for self_cell_scheme="
                    "'rectangular_cell' (that scheme has no such radius -- never a "
                    "fabricated/foreign value)."
                )
        else:
            r = self.equivalent_sphere_radius
            if (r is None or not math.isfinite(r) or r <= 0.0
                    or not math.isclose(r, R_expected, rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError(
                    f"equivalent_sphere_radius={r!r} does not match the value "
                    f"recomputed from mesh.spacing ({R_expected!r})."
                )

        if self.backend == "numpy":
            if self.device != "cpu":
                raise ValueError(f"device must be 'cpu' for backend='numpy', got {self.device!r}.")
            if self.jax_version is not None:
                raise ValueError(
                    f"jax_version must be None for backend='numpy', got {self.jax_version!r}."
                )
        else:
            realized_device = str(self.spectrum.device)
            if self.device != realized_device:
                raise ValueError(
                    f"device={self.device!r} != the realized spectrum's actual device "
                    f"{realized_device!r}."
                )
            if self.jax_version != jax.__version__:
                raise ValueError(
                    f"jax_version={self.jax_version!r} != the running jax.__version__ "
                    f"{jax.__version__!r}."
                )
        if self.numpy_version != np.__version__:
            raise ValueError(
                f"numpy_version={self.numpy_version!r} != the running numpy.__version__ "
                f"{np.__version__!r}."
            )
        if self.solver_version != _FREE_SPACE_POISSON_SOLVER_VERSION:
            raise ValueError(
                f"solver_version={self.solver_version!r} != "
                f"{_FREE_SPACE_POISSON_SOLVER_VERSION!r}."
            )

        recomputed_hash = _kernel_spec_sha256({
            "shape": self.mesh.shape, "spacing": self.mesh.spacing, "origin": self.mesh.origin,
            "padded_shape": self.mesh.padded_shape,
            "self_cell_scheme": self.mesh.self_cell_scheme,
            "fft_normalization": self.mesh.fft_normalization,
            "self_cell_integral": self.self_cell_integral, "K_self": self.K_self,
            "equivalent_sphere_radius": self.equivalent_sphere_radius,
            "fft_kind": self.fft_kind, "backend": self.backend,
            "input_dtype": self.input_dtype, "spectrum_dtype": self.spectrum_dtype,
            "numpy_version": self.numpy_version, "jax_version": self.jax_version,
            "solver_version": self.solver_version,
        })
        if recomputed_hash != self.kernel_spec_sha256:
            raise ValueError(
                "kernel_spec_sha256 does not match the canonical digest recomputed from "
                "this artifact's own declared fields (never hashes spectrum bytes/JAX data)."
            )


def build_free_space_poisson_kernel(shape, spacing, origin=(0.0, 0.0, 0.0), *,
                                     pad_factor=2, self_cell_scheme="rectangular_cell",
                                     fft_kind="rfft", backend="numpy", dtype=None):
    """Build (once per fixed mesh) the padded reciprocal-space
    convolution kernel -- the expensive part, done once and reused
    across every density solve via solve_free_space_poisson.

    Args:
        shape: (Nx, Ny, Nz) physical (unpadded) mesh, positive integers.
        spacing: (dx, dy, dz), finite positive floats -- anisotropic
            supported, no isotropy assumption anywhere in the math.
        origin: (x0, y0, z0), physical coordinates of grid index
            (0,0,0) -- does not affect the kernel VALUES (the kernel
            is translation-invariant), but is included in
            kernel_spec_sha256 since two kernels built for different
            physical placements are distinct provenance records even
            when numerically identical.
        pad_factor: padded_shape = pad_factor * shape per axis. Must be
            >= 2 (zero-padding sufficient for a correct linear, non-
            wrapping convolution).
        self_cell_scheme: "rectangular_cell" (default, exact closed
            form) or "equivalent_sphere" (diagnostic-only approximation,
            see _equivalent_sphere_self_potential).
        fft_kind: "rfft" (default, for real densities) or "fft" (for
            complex densities) -- explicit, since rfftn/fftn produce
            different spectrum shapes; build a SEPARATE kernel for each
            density family, never shared/guessed.
        backend: "numpy" or "jax".
        dtype: density family dtype. None defaults to float64 for
            fft_kind="rfft", complex128 for fft_kind="fft". Must match
            fft_kind's real/complex family. For backend="jax", raises
            ValueError immediately if a requested float64/complex128
            dtype would be silently truncated to float32/complex64
            because jax_enable_x64 is disabled (JAX itself only warns).

    Returns:
        FreeSpacePoissonKernel.
    """
    if backend not in ("numpy", "jax"):
        raise ValueError(f"Unsupported backend={backend!r}, must be 'numpy' or 'jax'.")
    if fft_kind not in ("rfft", "fft"):
        raise ValueError(f"Unsupported fft_kind={fft_kind!r}, must be 'rfft' or 'fft'.")

    pad_factor = _validate_positive_int("pad_factor", pad_factor)
    if pad_factor < 2:
        raise ValueError(
            f"pad_factor={pad_factor} must be >= 2 -- zero-padding sufficient for a "
            f"correct linear (non-wrapping) convolution."
        )
    shape = _validate_positive_int_tuple("shape", shape)
    padded_shape = tuple(pad_factor * s for s in shape)

    # Construct+validate geometry FIRST -- before computing the self-cell
    # term or anything else that could raise an incidental, uncontracted
    # exception on malformed spacing/origin (Alice's review, task #13,
    # 2026-07-13: directly reproduced spacing=(0,1,1) raising a bare
    # ZeroDivisionError from inside the self-cell formula instead of the
    # documented ValueError, because the self term was computed before
    # the mesh -- and thus the spacing -- was ever validated).
    mesh = FreeSpacePoissonMesh(
        shape=shape, spacing=spacing, origin=origin, padded_shape=padded_shape,
        self_cell_scheme=self_cell_scheme, fft_normalization="backward",
    )
    dx, dy, dz = mesh.spacing

    dtype = (np.dtype(dtype) if dtype is not None
              else (np.dtype(np.float64) if fft_kind == "rfft" else np.dtype(np.complex128)))
    if fft_kind == "rfft" and not np.issubdtype(dtype, np.floating):
        raise ValueError(f"fft_kind='rfft' requires a real dtype, got {dtype}.")
    if fft_kind == "fft" and not np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"fft_kind='fft' requires a complex dtype, got {dtype}.")
    if dtype not in _REAL_COMPUTE_DTYPE_FOR_INPUT:
        raise ValueError(
            f"Unsupported dtype={dtype} -- must be one of float32/float64/"
            f"complex64/complex128."
        )
    real_compute_dtype = _REAL_COMPUTE_DTYPE_FOR_INPUT[dtype]

    if backend == "jax":
        needs_64bit = dtype in (np.dtype(np.float64), np.dtype(np.complex128))
        if needs_64bit and not jax.config.jax_enable_x64:
            # Checked proactively via jax.config.jax_enable_x64 -- NOT by
            # probing with jnp.zeros(dtype=...) and observing whether it
            # got silently downgraded, which fires JAX's own UserWarning
            # even though this function raises immediately afterward
            # (Alice's review, task #13, 2026-07-13: a clean ValueError
            # is preferable to a warning-then-exception).
            raise ValueError(
                f"Requested dtype {dtype} requires jax_enable_x64, which is "
                f"currently disabled -- call jax.config.update('jax_enable_x64', "
                f"True) before requesting a float64/complex128 kernel, or request "
                f"a 32-bit dtype explicitly."
            )

    self_cell_integral, K_self, equivalent_sphere_radius = _self_cell_values(
        self_cell_scheme, dx, dy, dz)

    Px, Py, Pz = mesh.padded_shape
    ix = _wrapped_integer_offsets(Px)
    iy = _wrapped_integer_offsets(Py)
    iz = _wrapped_integer_offsets(Pz)
    OX = (ix[:, None, None] * dx).astype(real_compute_dtype)
    OY = (iy[None, :, None] * dy).astype(real_compute_dtype)
    OZ = (iz[None, None, :] * dz).astype(real_compute_dtype)
    K_padded = _coulomb_kernel_values(OX, OY, OZ, K_self).astype(real_compute_dtype)

    if backend == "jax":
        K_padded_b = jnp.asarray(K_padded)
        fft_ns = jnp.fft
    else:
        K_padded_b = K_padded
        fft_ns = np.fft

    if fft_kind == "rfft":
        spectrum = fft_ns.rfftn(K_padded_b, s=mesh.padded_shape, axes=(0, 1, 2), norm="backward")
    else:
        K_padded_b = K_padded_b.astype(dtype)
        spectrum = fft_ns.fftn(K_padded_b, s=mesh.padded_shape, axes=(0, 1, 2), norm="backward")

    # NOTE: the spectrum is NOT defensively copied here -- FreeSpacePoissonKernel's
    # own __post_init__ now does that unconditionally for backend="numpy" (Alice's
    # review, task #13, 2026-07-13: the builder must not be the SOLE guard of
    # immutability for a publicly-constructible type), so copying here too would
    # just double the allocation for no benefit.
    device = "cpu" if backend == "numpy" else str(spectrum.device)

    spectrum_dtype = str(spectrum.dtype)
    jax_version_str = jax.__version__ if backend == "jax" else None

    kernel_spec_sha256 = _kernel_spec_sha256({
        "shape": mesh.shape, "spacing": mesh.spacing, "origin": mesh.origin,
        "padded_shape": mesh.padded_shape,
        "self_cell_scheme": self_cell_scheme, "fft_normalization": mesh.fft_normalization,
        "self_cell_integral": self_cell_integral, "K_self": K_self,
        "equivalent_sphere_radius": equivalent_sphere_radius,
        "fft_kind": fft_kind, "backend": backend,
        "input_dtype": str(dtype), "spectrum_dtype": spectrum_dtype,
        "numpy_version": np.__version__, "jax_version": jax_version_str,
        "solver_version": _FREE_SPACE_POISSON_SOLVER_VERSION,
    })

    return FreeSpacePoissonKernel(
        mesh=mesh, spectrum=spectrum, fft_kind=fft_kind, backend=backend,
        input_dtype=str(dtype), spectrum_dtype=spectrum_dtype, device=device,
        numpy_version=np.__version__, jax_version=jax_version_str,
        self_cell_integral=float(self_cell_integral), K_self=float(K_self),
        equivalent_sphere_radius=(
            float(equivalent_sphere_radius) if equivalent_sphere_radius is not None else None
        ),
        kernel_spec_sha256=kernel_spec_sha256,
        solver_version=_FREE_SPACE_POISSON_SOLVER_VERSION,
    )


def solve_free_space_poisson(rho, kernel):
    """Solve the free-space Poisson equation for a (batched) density
    against an already-built FreeSpacePoissonKernel -- no kernel
    rebuild, batched over arbitrary leading axes.

    Args:
        rho: (..., Nx, Ny, Nz) array matching kernel.mesh.shape, real
            (for an fft_kind="rfft" kernel) or complex (for "fft").
            Input dtype must exactly match kernel.input_dtype and the
            array's backend (NumPy/JAX) must match kernel.backend --
            both are validated explicitly, never silently coerced.
        kernel: FreeSpacePoissonKernel from build_free_space_poisson_kernel.

    Returns:
        v: potential array of the same shape/dtype as rho.

    Raises:
        TypeError: kernel is not a FreeSpacePoissonKernel.
        ValueError: backend, shape, dtype, or real/complex-kind mismatch.
    """
    if not isinstance(kernel, FreeSpacePoissonKernel):
        raise TypeError(
            "kernel must be a FreeSpacePoissonKernel from build_free_space_poisson_kernel."
        )

    is_jax_array = isinstance(rho, jax.Array)
    is_numpy_array = isinstance(rho, np.ndarray)
    if kernel.backend == "numpy" and is_jax_array:
        raise ValueError(
            "kernel.backend='numpy' but rho is a JAX array -- build a backend='jax' "
            "kernel for JAX inputs."
        )
    if kernel.backend == "jax" and is_numpy_array:
        raise ValueError(
            "kernel.backend='jax' but rho is a NumPy array -- build a backend='numpy' "
            "kernel, or pass a JAX array."
        )

    xp = jnp if kernel.backend == "jax" else np
    fft_ns = jnp.fft if kernel.backend == "jax" else np.fft
    rho = xp.asarray(rho)

    input_dtype = np.dtype(kernel.input_dtype)
    if rho.dtype != input_dtype:
        raise ValueError(f"rho.dtype={rho.dtype} != kernel.input_dtype={input_dtype}.")

    mesh = kernel.mesh
    Nx, Ny, Nz = mesh.shape
    Px, Py, Pz = mesh.padded_shape
    if tuple(rho.shape[-3:]) != (Nx, Ny, Nz):
        raise ValueError(
            f"rho's trailing 3 axes {tuple(rho.shape[-3:])} != mesh.shape {(Nx, Ny, Nz)}."
        )

    is_complex_kernel = kernel.fft_kind == "fft"
    rho_is_complex = np.issubdtype(rho.dtype, np.complexfloating)
    if is_complex_kernel != rho_is_complex:
        raise ValueError(
            f"kernel.fft_kind={kernel.fft_kind!r} but rho.dtype={rho.dtype} "
            f"({'complex' if rho_is_complex else 'real'}) -- fft_kind='rfft' needs real "
            f"rho, fft_kind='fft' needs complex rho."
        )

    batch_shape = tuple(rho.shape[:-3])
    padded_shape_full = batch_shape + (Px, Py, Pz)

    if kernel.backend == "jax":
        rho_padded = jnp.zeros(padded_shape_full, dtype=rho.dtype)
        rho_padded = rho_padded.at[..., :Nx, :Ny, :Nz].set(rho)
    else:
        rho_padded = np.zeros(padded_shape_full, dtype=rho.dtype)
        rho_padded[..., :Nx, :Ny, :Nz] = rho

    if kernel.fft_kind == "rfft":
        rho_hat = fft_ns.rfftn(rho_padded, s=(Px, Py, Pz), axes=(-3, -2, -1), norm="backward")
        v_padded = fft_ns.irfftn(
            rho_hat * kernel.spectrum, s=(Px, Py, Pz), axes=(-3, -2, -1), norm="backward")
    else:
        rho_hat = fft_ns.fftn(rho_padded, s=(Px, Py, Pz), axes=(-3, -2, -1), norm="backward")
        v_padded = fft_ns.ifftn(
            rho_hat * kernel.spectrum, s=(Px, Py, Pz), axes=(-3, -2, -1), norm="backward")

    dV = mesh.spacing[0] * mesh.spacing[1] * mesh.spacing[2]
    v = v_padded[..., :Nx, :Ny, :Nz] * dV
    return v.astype(rho.dtype)


def free_space_poisson_direct_sum_oracle(rho, mesh):
    """O(N_g^2) NumPy-only brute-force reference sum, using the SAME
    scalar kernel-value helper (_coulomb_kernel_values, same
    self_cell_scheme branch) as build_free_space_poisson_kernel --
    small systems only, for correctness testing, never production.

    Args:
        rho: (..., Nx, Ny, Nz) array matching mesh.shape, real or complex.
        mesh: FreeSpacePoissonMesh (geometry + self_cell_scheme only --
            no kernel/spectrum needed for this direct-sum path).

    Returns:
        v: potential array of the same shape as rho.
    """
    if not isinstance(mesh, FreeSpacePoissonMesh):
        raise TypeError("mesh must be a FreeSpacePoissonMesh.")
    rho = np.asarray(rho)
    Nx, Ny, Nz = mesh.shape
    if tuple(rho.shape[-3:]) != (Nx, Ny, Nz):
        raise ValueError(
            f"rho's trailing 3 axes {tuple(rho.shape[-3:])} != mesh.shape {(Nx, Ny, Nz)}."
        )

    dx, dy, dz = mesh.spacing
    _, K_self, _ = _self_cell_values(mesh.self_cell_scheme, dx, dy, dz)

    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    coords = np.stack([ii.ravel() * dx, jj.ravel() * dy, kk.ravel() * dz], axis=-1)  # (Ng,3)
    diff = coords[:, None, :] - coords[None, :, :]  # (Ng,Ng,3)
    K = _coulomb_kernel_values(diff[..., 0], diff[..., 1], diff[..., 2], K_self)  # (Ng,Ng)

    Ng = coords.shape[0]
    dV = dx * dy * dz
    batch_shape = rho.shape[:-3]
    rho_flat = rho.reshape(batch_shape + (Ng,))
    v_flat = dV * np.einsum("...j,ij->...i", rho_flat, K)
    return v_flat.reshape(rho.shape)


# ===========================================================================
# 6. Poisson interpolation-vector Z-core assembly
# ===========================================================================
"""Builds raw physical interpolation vectors Theta on a FreeSpacePoissonMesh
and assembles the Coulomb Z-core Z = dV * Theta^dagger V via section 5's
approved Poisson backend (task #15, isdf-coulomb-cuda, P2b, 2026-07-13 --
2 review rounds with Alice before implementation).

Theta[mu,g] construction reuses pytc/df/solvers.py's structured normal-
equations solver directly -- the SAME primitive pytc/df/isdf.py:
isdf_decompose already uses for TC's own single-orbital ISDF fitting
(xi_phi/xi_grad), applied here with the Coulomb path's own pair factors.
No new solver logic: prepare_normal_equations_solver + repeated
solve_normal_equations_batch_prepared calls, batched over the grid axis.

RAW (unweighted) factor values throughout -- factor_p_raw/factor_q_raw at
BOTH the pivots and the grid batch, matching pair_collocation_at_pivots's
already-established convention (section 4), NOT the isdf-coulomb-cuda
decision doc's original Theta=A P^dagger S^-1 formula (which uses
WEIGHTED A=sqrt(w)*phi_p*phi_q). That weighted formula is exactly what
Alice's own task #6 review moved the codebase away from ("passing
weighted values here... made compute_Z's rcond silently depend on the
grid quadrature's weight scale, a portability bug") -- the decision doc
was never updated after that fix. Selection-weighted collocation
(weight_mo_values) is used ONLY by select_sector_pivots to choose pivots
in the first place, never touches the Theta fit itself.

Z assembly is a DIRECT bilinear contraction, not a compute_Z-style S^-1
regularized solve: Z_mu,nu = <Theta_mu|1/r12|Theta_nu> = integral
Theta_mu(r) V_nu(r) dr, discretized as dV * sum_g conj(Theta_mu(r_g)) *
V_nu(r_g), where V_nu = K Theta_nu (section 5's solve_free_space_poisson).
TWO separate dV factors legitimately appear here -- one already inside
solve_free_space_poisson's own internal step (section 5, unchanged), one
in this section's own outer sum -- these are two DISTINCT nested-integral
discretizations, not a double-count of the same quantity. W = dV*I is
trivial on this uniform mesh (every cell has the same volume, unlike the
irregular DFT grid in section 1) -- no separate weight array is stored.

Mesh-geometry identity (what Theta depends on: mesh.shape/spacing/origin)
is kept SEPARATE from kernel identity (padded_shape/self_cell_scheme/
fft_kind -- a Poisson-SOLVE-only concern Theta never touches):
PoissonInterpolationSector records only geometry, no kernel reference at
all, so one Theta sector legitimately serves multiple kernel choices
(rfft vs fft, differing self_cell_scheme/padding) without rebuilding.
PoissonCoreArtifact records the specific kernel_spec_sha256 used for ITS
Poisson solves.

Explicit incore baseline (storage_mode="incore_full_theta", stated as
such, not implied away): the coupled normal-equations solve returns ALL
n_fused rows for any given grid batch (the dense (n_fused,n_fused) normal
matrix couples every interpolation point), so a sector's Theta cannot be
built in mu-blocks without either recomputing the full solve per block or
an out-of-core store (future work, not this task). grid_batch_size only
bounds the TRANSIENT least-squares output width during construction.
poisson_core's mu_block_size/nu_block_size instead bound the GEMM
contraction slice widths and the Poisson-solve batch width respectively,
tiling the O(N_g*N_mu^2) core-assembly contraction and the
O(N_g*N_mu*log(N_g)) Poisson-solve cost for peak memory, not total FLOPs.
"""

_SUPPORTED_POISSON_STORAGE_MODES = ("incore_full_theta",)
_POISSON_INTERPOLATION_SOLVER_VERSION = "1"


def _validate_sha256_hex(name, value):
    """Closed syntax validation for a caller-supplied or locally-computed
    SHA-256 hex digest -- exactly 64 lowercase hex characters, nothing
    else. Used both for locally-computed digests (sanity check) and for
    caller-attested upstream digests on the JAX path, where this is the
    ONLY validation performed -- syntax, not content (Alice's review,
    task #15, 2026-07-13: "validate closed key names/type/64-hex syntax
    and record that trust boundary... do not claim the JAX digest was
    verified from device bytes")."""
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            f"{name} must be a 64-character lowercase hex SHA-256 string, got {value!r}."
        )


@dataclasses.dataclass(frozen=True)
class PoissonInterpolationSector:
    """One MO-pair sector's raw physical interpolation vectors Theta,
    built once via build_poisson_interpolation_sector and reusable across
    multiple poisson_core calls (same-sector AND cross-sector, against
    any kernel sharing this sector's mesh geometry) without rebuilding.

    Theta: (n_pivots, N_g), the natural solver-output orientation
    (matching pytc.df.isdf.isdf_decompose's xi_phi convention exactly)
    -- backend-preserving: NumPy gets a defensive read-only copy; JAX
    stays an untouched (already-immutable) jax.Array on its actual
    device.
    mesh_shape/spacing/origin: GEOMETRY ONLY (no padded_shape/
    self_cell_scheme/fft_normalization -- those are FreeSpacePoissonKernel
    concerns Theta never touches).
    storage_mode: "incore_full_theta" (only supported value -- an
    honest, explicit label, not an implied-away detail).
    factor_identity_source: "computed_content_hash" (NumPy -- this
    function independently hashed factor_p_raw/factor_q_raw's actual
    bytes) or "caller_attested" (JAX -- the caller supplied
    upstream_provenance['factor_p_sha256'/'factor_q_sha256'], validated
    only for syntax, never independently verified against device bytes).
    """
    Theta: object
    pivots: object
    mesh_shape: tuple
    mesh_spacing: tuple
    mesh_origin: tuple
    same_factor: bool
    storage_mode: str
    backend: str
    device: str
    realized_dtype: str
    grid_batch_size: int
    rcond: float
    jitter_used: float
    n_tries: int
    factor_identity_source: str
    pivots_sha256: str
    factor_p_sha256: str
    factor_q_sha256: str
    sector_spec_sha256: str
    solver_version: str
    provenance: object

    def __post_init__(self):
        mesh_shape = _validate_positive_int_tuple("mesh_shape", self.mesh_shape)
        mesh_spacing = _as_length_tuple("mesh_spacing", self.mesh_spacing)
        mesh_spacing = tuple(float(s) for s in mesh_spacing)
        if any((not math.isfinite(d)) or d <= 0.0 for d in mesh_spacing):
            raise ValueError(f"mesh_spacing must be 3 finite positive floats, got {mesh_spacing}.")
        mesh_origin = _as_length_tuple("mesh_origin", self.mesh_origin)
        mesh_origin = tuple(float(o) for o in mesh_origin)
        if any(not math.isfinite(o) for o in mesh_origin):
            raise ValueError(f"mesh_origin must be 3 finite floats, got {mesh_origin}.")
        object.__setattr__(self, "mesh_shape", mesh_shape)
        object.__setattr__(self, "mesh_spacing", mesh_spacing)
        object.__setattr__(self, "mesh_origin", mesh_origin)

        if not isinstance(self.same_factor, bool):
            raise TypeError(f"same_factor must be bool, got {type(self.same_factor).__name__}.")
        if self.storage_mode not in _SUPPORTED_POISSON_STORAGE_MODES:
            raise ValueError(f"Unsupported storage_mode={self.storage_mode!r}.")
        if self.backend not in ("numpy", "jax"):
            raise ValueError(f"Unsupported backend={self.backend!r}.")

        if self.backend == "numpy":
            if not isinstance(self.Theta, np.ndarray):
                raise TypeError("backend='numpy' requires Theta to be a numpy.ndarray.")
            object.__setattr__(self, "Theta", _readonly_copy(self.Theta))
            if self.device != "cpu":
                raise ValueError(f"device must be 'cpu' for backend='numpy', got {self.device!r}.")
        else:
            if not isinstance(self.Theta, jax.Array):
                raise TypeError("backend='jax' requires Theta to be a jax.Array.")
            realized_device = str(self.Theta.device)
            if self.device != realized_device:
                raise ValueError(
                    f"device={self.device!r} != the realized Theta array's actual device "
                    f"{realized_device!r}."
                )

        n_g = 1
        for s in mesh_shape:
            n_g *= s

        pivots_np = np.asarray(self.pivots)
        if pivots_np.ndim != 1 or not np.issubdtype(pivots_np.dtype, np.integer):
            raise ValueError("pivots must be a 1-D integer array.")
        if np.unique(pivots_np).shape[0] != pivots_np.shape[0]:
            raise ValueError("pivots must contain unique indices.")
        if pivots_np.size == 0 or pivots_np.min() < 0 or pivots_np.max() >= n_g:
            raise ValueError(f"pivots out of range [0, {n_g}).")
        n_fused = int(pivots_np.shape[0])
        object.__setattr__(
            self, "pivots",
            _readonly_copy(pivots_np) if self.backend == "numpy" else jnp.asarray(pivots_np)
        )

        expected_theta_shape = (n_fused, n_g)
        if tuple(self.Theta.shape) != expected_theta_shape:
            raise ValueError(
                f"Theta.shape {tuple(self.Theta.shape)} != expected (n_pivots, prod(mesh_shape)) "
                f"{expected_theta_shape}."
            )

        declared_dtype = np.dtype(self.realized_dtype)
        if np.dtype(self.Theta.dtype) != declared_dtype:
            raise ValueError(
                f"Theta.dtype={self.Theta.dtype} != declared realized_dtype={declared_dtype}."
            )

        grid_batch_size = _validate_positive_int("grid_batch_size", self.grid_batch_size)
        object.__setattr__(self, "grid_batch_size", grid_batch_size)

        if not (math.isfinite(self.rcond) and self.rcond > 0.0):
            raise ValueError(f"rcond must be a finite positive float, got {self.rcond!r}.")
        if not math.isfinite(self.jitter_used) or self.jitter_used < 0.0:
            raise ValueError(
                f"jitter_used must be a finite non-negative float, got {self.jitter_used!r}."
            )
        n_tries = _validate_positive_int("n_tries", self.n_tries)
        object.__setattr__(self, "n_tries", n_tries)

        if self.factor_identity_source not in ("computed_content_hash", "caller_attested"):
            raise ValueError(f"Unsupported factor_identity_source={self.factor_identity_source!r}.")
        # Backend/trust-boundary pairing is a hard invariant, not just a
        # supported-value check -- a manually constructed artifact must
        # not be able to claim NumPy content-hash trust for JAX-sourced
        # factors or vice versa (Alice's review, task #15, 2026-07-13).
        expected_source = "computed_content_hash" if self.backend == "numpy" else "caller_attested"
        if self.factor_identity_source != expected_source:
            raise ValueError(
                f"factor_identity_source={self.factor_identity_source!r} is inconsistent with "
                f"backend={self.backend!r} -- expected {expected_source!r}."
            )

        for name, value in [
            ("pivots_sha256", self.pivots_sha256),
            ("factor_p_sha256", self.factor_p_sha256),
            ("factor_q_sha256", self.factor_q_sha256),
        ]:
            _validate_sha256_hex(name, value)

        # same_factor=True asserts factor_p_raw and factor_q_raw were the
        # SAME array -- their recorded identity digests must agree,
        # regardless of trust boundary (reproduced: JAX same_factor=True
        # accepted contradictory caller-attested factor_p_sha256 !=
        # factor_q_sha256 -- Alice's review, task #15, 2026-07-13).
        if self.same_factor and self.factor_p_sha256 != self.factor_q_sha256:
            raise ValueError(
                "same_factor=True requires factor_p_sha256 == factor_q_sha256 "
                f"(got {self.factor_p_sha256!r} != {self.factor_q_sha256!r})."
            )

        # Pivot identity must be bound to the ACTUAL pivot array, not
        # merely a syntactically-valid digest string -- recompute and
        # require equality (reproduced: dataclasses.replace with a
        # reordered pivots array kept the stale digest -- Alice's
        # review, task #15, 2026-07-13).
        recomputed_pivots_sha256 = _canonical_sha256(pivots_np)
        if recomputed_pivots_sha256 != self.pivots_sha256:
            raise ValueError(
                "pivots_sha256 does not match the digest recomputed from the actual "
                "pivots array -- pivots may have been reordered or replaced without "
                "updating the digest."
            )

        if self.solver_version != _POISSON_INTERPOLATION_SOLVER_VERSION:
            raise ValueError(
                f"solver_version={self.solver_version!r} != "
                f"{_POISSON_INTERPOLATION_SOLVER_VERSION!r}."
            )

        recomputed_hash = _kernel_spec_sha256({
            "mesh_shape": mesh_shape, "mesh_spacing": mesh_spacing, "mesh_origin": mesh_origin,
            "same_factor": self.same_factor, "storage_mode": self.storage_mode,
            "backend": self.backend, "realized_dtype": self.realized_dtype,
            "grid_batch_size": grid_batch_size, "rcond": self.rcond,
            "jitter_used": self.jitter_used, "n_tries": n_tries,
            "factor_identity_source": self.factor_identity_source,
            "pivots_sha256": self.pivots_sha256, "factor_p_sha256": self.factor_p_sha256,
            "factor_q_sha256": self.factor_q_sha256, "solver_version": self.solver_version,
        })
        _validate_sha256_hex("sector_spec_sha256", self.sector_spec_sha256)
        if recomputed_hash != self.sector_spec_sha256:
            raise ValueError(
                "sector_spec_sha256 does not match the canonical digest recomputed from "
                "this artifact's own declared fields."
            )

        object.__setattr__(
            self, "provenance",
            _deep_freeze(dict(self.provenance) if self.provenance else {})
        )


@dataclasses.dataclass(frozen=True)
class PoissonCoreArtifact:
    """A completed Z core for one sector (same-sector) or one sector pair
    (cross-sector), assembled via poisson_core. Explicit typed fields
    (not a nested provenance dict) so __post_init__ can recompute/
    validate every one of them directly against the actual Z array and
    the sectors/kernel that produced it -- mirroring
    FreeSpacePoissonKernel's provenance-closure pattern (section 5)."""
    Z: object
    left_sector_spec_sha256: str
    right_sector_spec_sha256: str
    left_n_fused: int
    right_n_fused: int
    kernel_spec_sha256: str
    mu_block_size: int
    nu_block_size: int
    normalization: str
    backend: str
    device: str
    realized_dtype: str
    solver_version: str
    core_spec_sha256: str

    def __post_init__(self):
        if self.backend not in ("numpy", "jax"):
            raise ValueError(f"Unsupported backend={self.backend!r}.")

        if self.backend == "numpy":
            if not isinstance(self.Z, np.ndarray):
                raise TypeError("backend='numpy' requires Z to be a numpy.ndarray.")
            object.__setattr__(self, "Z", _readonly_copy(self.Z))
            if self.device != "cpu":
                raise ValueError(f"device must be 'cpu' for backend='numpy', got {self.device!r}.")
        else:
            if not isinstance(self.Z, jax.Array):
                raise TypeError("backend='jax' requires Z to be a jax.Array.")
            realized_device = str(self.Z.device)
            if self.device != realized_device:
                raise ValueError(
                    f"device={self.device!r} != the realized Z array's actual device "
                    f"{realized_device!r}."
                )

        left_n_fused = _validate_positive_int("left_n_fused", self.left_n_fused)
        right_n_fused = _validate_positive_int("right_n_fused", self.right_n_fused)
        object.__setattr__(self, "left_n_fused", left_n_fused)
        object.__setattr__(self, "right_n_fused", right_n_fused)
        expected_shape = (left_n_fused, right_n_fused)
        if tuple(self.Z.shape) != expected_shape:
            raise ValueError(
                f"Z.shape {tuple(self.Z.shape)} != expected (left_n_fused, right_n_fused) "
                f"{expected_shape}."
            )

        declared_dtype = np.dtype(self.realized_dtype)
        if np.dtype(self.Z.dtype) != declared_dtype:
            raise ValueError(f"Z.dtype={self.Z.dtype} != declared realized_dtype={declared_dtype}.")

        mu_block_size = _validate_positive_int("mu_block_size", self.mu_block_size)
        nu_block_size = _validate_positive_int("nu_block_size", self.nu_block_size)
        object.__setattr__(self, "mu_block_size", mu_block_size)
        object.__setattr__(self, "nu_block_size", nu_block_size)

        if self.normalization != "dV":
            raise ValueError(f"Only normalization='dV' is supported, got {self.normalization!r}.")

        for name, value in [
            ("left_sector_spec_sha256", self.left_sector_spec_sha256),
            ("right_sector_spec_sha256", self.right_sector_spec_sha256),
            ("kernel_spec_sha256", self.kernel_spec_sha256),
        ]:
            _validate_sha256_hex(name, value)

        if self.solver_version != _POISSON_INTERPOLATION_SOLVER_VERSION:
            raise ValueError(
                f"solver_version={self.solver_version!r} != "
                f"{_POISSON_INTERPOLATION_SOLVER_VERSION!r}."
            )

        # Close the core's own digest, not just the kernel/sector digests
        # it references -- otherwise ANY single typed field (kernel_spec_
        # sha256 included) can be tampered with independently and still
        # look internally consistent (reproduced: dataclasses.replace
        # with kernel_spec_sha256="0"*64 was accepted -- Alice's review,
        # task #15, 2026-07-13).
        recomputed_core_hash = _kernel_spec_sha256({
            "left_sector_spec_sha256": self.left_sector_spec_sha256,
            "right_sector_spec_sha256": self.right_sector_spec_sha256,
            "left_n_fused": left_n_fused, "right_n_fused": right_n_fused,
            "kernel_spec_sha256": self.kernel_spec_sha256,
            "mu_block_size": mu_block_size, "nu_block_size": nu_block_size,
            "normalization": self.normalization, "backend": self.backend,
            "device": self.device,
            "realized_dtype": self.realized_dtype, "solver_version": self.solver_version,
        })
        _validate_sha256_hex("core_spec_sha256", self.core_spec_sha256)
        if recomputed_core_hash != self.core_spec_sha256:
            raise ValueError(
                "core_spec_sha256 does not match the canonical digest recomputed from "
                "this artifact's own declared fields."
            )


def build_poisson_interpolation_sector(factor_p_raw, factor_q_raw, pivots, mesh, *,
                                        same_factor=False, grid_batch_size=None,
                                        rcond=1e-14, upstream_provenance=None):
    """Build the raw physical interpolation vectors Theta for one MO-pair
    sector on a FreeSpacePoissonMesh, via pytc/df/solvers.py's structured
    normal-equations solver (the same primitive
    pytc.df.isdf.isdf_decompose already uses for TC's own ISDF fitting).

    Args:
        factor_p_raw, factor_q_raw: (n_p/n_q, N_g) RAW (unweighted) MO
            values sampled at mesh's physical grid points, in mesh's
            flattened row-major order. Must share dtype and backend
            (both numpy.ndarray or both jax.Array). No hidden
            conjugation is applied to either channel here -- whatever
            conjugation convention a caller's pair-product formula needs
            must already be baked into these arrays.
        pivots: (n_fused,) 1-D integer grid-point indices into
            [0, prod(mesh.shape)), unique, order preserved exactly as
            given (never resorted/deduplicated by this function).
        mesh: FreeSpacePoissonMesh -- only shape/spacing/origin
            (geometry) are used; padded_shape/self_cell_scheme/
            fft_normalization are kernel-construction concerns this
            function never touches.
        same_factor: True for a symmetric sector (oo, vv) -- validated
            by ACTUAL equality (jnp.array_equal on JAX, transferring
            only the resulting scalar to host; np.array_equal on NumPy),
            never merely trusted as a flag.
        grid_batch_size: bounds the TRANSIENT least-squares solve output
            width during construction only (Theta itself is always
            fully retained incore -- see storage_mode="incore_full_theta"
            in the class docstring for why mu-blocking isn't possible
            here). None resolves to N_g (single batch); the realized
            value is recorded.
        rcond: forwarded to prepare_normal_equations_solver.
        upstream_provenance: for backend="numpy", optional extra caller
            metadata (frozen and stored, does not affect
            sector_spec_sha256). For backend="jax", MUST additionally
            contain 'factor_p_sha256'/'factor_q_sha256' string digests
            (caller-attested identity for factor_p_raw/factor_q_raw --
            this function never runs NumPy content hashing over full
            device arrays) -- raises if missing.

    Returns:
        PoissonInterpolationSector.
    """
    if not isinstance(mesh, FreeSpacePoissonMesh):
        raise TypeError("mesh must be a FreeSpacePoissonMesh.")

    is_p_jax = isinstance(factor_p_raw, jax.Array)
    is_q_jax = isinstance(factor_q_raw, jax.Array)
    is_p_numpy = isinstance(factor_p_raw, np.ndarray)
    is_q_numpy = isinstance(factor_q_raw, np.ndarray)
    if not ((is_p_jax and is_q_jax) or (is_p_numpy and is_q_numpy)):
        raise ValueError(
            "factor_p_raw and factor_q_raw must both be numpy.ndarray or both be jax.Array."
        )
    backend = "jax" if is_p_jax else "numpy"
    xp = jnp if backend == "jax" else np

    factor_p_raw = xp.asarray(factor_p_raw)
    factor_q_raw = xp.asarray(factor_q_raw)
    if factor_p_raw.ndim != 2 or factor_q_raw.ndim != 2:
        raise ValueError("factor_p_raw/factor_q_raw must be 2-D (n_orb, N_g).")
    if factor_p_raw.dtype != factor_q_raw.dtype:
        raise ValueError(
            f"factor_p_raw.dtype={factor_p_raw.dtype} != factor_q_raw.dtype={factor_q_raw.dtype}."
        )

    n_g = 1
    for s in mesh.shape:
        n_g *= s
    if factor_p_raw.shape[1] != n_g or factor_q_raw.shape[1] != n_g:
        raise ValueError(
            f"factor_p_raw/factor_q_raw grid axis must equal prod(mesh.shape)={n_g}, "
            f"got {factor_p_raw.shape[1]}/{factor_q_raw.shape[1]}."
        )

    pivots_np = np.asarray(pivots)
    if pivots_np.ndim != 1 or not np.issubdtype(pivots_np.dtype, np.integer):
        raise ValueError("pivots must be a 1-D integer array.")
    if np.unique(pivots_np).shape[0] != pivots_np.shape[0]:
        raise ValueError("pivots must contain unique indices.")
    if pivots_np.size == 0 or pivots_np.min() < 0 or pivots_np.max() >= n_g:
        raise ValueError(f"pivots out of range [0, {n_g}).")
    pivots_backend = xp.asarray(pivots_np)

    if same_factor:
        if backend == "jax":
            equal = bool(jnp.array_equal(factor_p_raw, factor_q_raw))
        else:
            equal = bool(np.array_equal(factor_p_raw, factor_q_raw))
        if not equal:
            raise ValueError(
                "same_factor=True requires factor_p_raw and factor_q_raw to be the SAME "
                "array (validated by actual equality, not merely trusted)."
            )

    if not (math.isfinite(rcond) and rcond > 0.0):
        raise ValueError(f"rcond must be a finite positive float, got {rcond!r}.")

    grid_batch_size_realized = (
        _validate_positive_int("grid_batch_size", grid_batch_size)
        if grid_batch_size is not None else n_g
    )

    factor_p_piv = factor_p_raw[:, pivots_backend]
    factor_q_piv = factor_q_raw[:, pivots_backend]
    n_fused = int(pivots_np.shape[0])

    chol, lower, jitter_used, n_tries = prepare_normal_equations_solver(
        factor_p_piv, factor_q_piv, rcond=rcond, return_info=True)

    theta_chunks = []
    for g_start in range(0, n_g, grid_batch_size_realized):
        g_end = min(g_start + grid_batch_size_realized, n_g)
        theta_batch = solve_normal_equations_batch_prepared(
            chol, lower, factor_p_piv, factor_q_piv,
            factor_p_raw[:, g_start:g_end], factor_q_raw[:, g_start:g_end])
        theta_chunks.append(theta_batch)
    Theta = (jnp.concatenate if backend == "jax" else np.concatenate)(theta_chunks, axis=1)

    device = "cpu" if backend == "numpy" else str(Theta.device)
    realized_dtype = str(Theta.dtype)

    pivots_sha256 = _canonical_sha256(pivots_np)

    upstream_provenance = dict(upstream_provenance) if upstream_provenance else {}
    if backend == "numpy":
        factor_p_sha256 = _canonical_sha256(np.asarray(factor_p_raw))
        factor_q_sha256 = _canonical_sha256(np.asarray(factor_q_raw))
        factor_identity_source = "computed_content_hash"
    else:
        if "factor_p_sha256" not in upstream_provenance or "factor_q_sha256" not in upstream_provenance:
            raise ValueError(
                "backend='jax': upstream_provenance must supply 'factor_p_sha256' and "
                "'factor_q_sha256' -- this function never runs NumPy content hashing over "
                "full device arrays (would force an unwanted host transfer). These are "
                "CALLER-ATTESTED identities, validated only for syntax, never independently "
                "verified against device bytes."
            )
        factor_p_sha256 = upstream_provenance.pop("factor_p_sha256")
        factor_q_sha256 = upstream_provenance.pop("factor_q_sha256")
        _validate_sha256_hex("upstream_provenance['factor_p_sha256']", factor_p_sha256)
        _validate_sha256_hex("upstream_provenance['factor_q_sha256']", factor_q_sha256)
        factor_identity_source = "caller_attested"

    if same_factor and factor_p_sha256 != factor_q_sha256:
        raise ValueError(
            "same_factor=True requires factor_p_sha256 == factor_q_sha256 -- got "
            f"{factor_p_sha256!r} != {factor_q_sha256!r} (on the JAX path this is a "
            "caller-attested identity, but a contradictory pair is still rejected: "
            "same_factor already asserts factor_p_raw and factor_q_raw are the SAME "
            "array, verified by actual equality above)."
        )

    sector_spec_sha256 = _kernel_spec_sha256({
        "mesh_shape": mesh.shape, "mesh_spacing": mesh.spacing, "mesh_origin": mesh.origin,
        "same_factor": same_factor, "storage_mode": "incore_full_theta",
        "backend": backend, "realized_dtype": realized_dtype,
        "grid_batch_size": grid_batch_size_realized, "rcond": rcond,
        "jitter_used": float(jitter_used), "n_tries": int(n_tries),
        "factor_identity_source": factor_identity_source,
        "pivots_sha256": pivots_sha256, "factor_p_sha256": factor_p_sha256,
        "factor_q_sha256": factor_q_sha256,
        "solver_version": _POISSON_INTERPOLATION_SOLVER_VERSION,
    })

    return PoissonInterpolationSector(
        Theta=Theta, pivots=pivots_backend,
        mesh_shape=mesh.shape, mesh_spacing=mesh.spacing, mesh_origin=mesh.origin,
        same_factor=same_factor, storage_mode="incore_full_theta",
        backend=backend, device=device, realized_dtype=realized_dtype,
        grid_batch_size=grid_batch_size_realized, rcond=rcond,
        jitter_used=float(jitter_used), n_tries=int(n_tries),
        factor_identity_source=factor_identity_source,
        pivots_sha256=pivots_sha256, factor_p_sha256=factor_p_sha256,
        factor_q_sha256=factor_q_sha256, sector_spec_sha256=sector_spec_sha256,
        solver_version=_POISSON_INTERPOLATION_SOLVER_VERSION,
        provenance=_deep_freeze(upstream_provenance),
    )


def poisson_core(left, right=None, *, kernel, mu_block_size=None, nu_block_size=None):
    """Assemble a Z core from one (same-sector) or two (cross-sector)
    PoissonInterpolationSector artifacts, via Z = dV * Theta_L^dagger V_R
    where V_R = K Theta_R (section 5's solve_free_space_poisson, batched
    over nu-blocks). Right-nu-outer/left-mu-inner loop: V is solved
    EXACTLY ONCE per nu-block, never recomputed inside the mu-loop.

    Args:
        left: PoissonInterpolationSector for the sector (same-sector) or
            sector A (cross-sector).
        right: None (same-sector: right_sector = left, literally the
            same Theta object, no duplication) or a PoissonInterpolationSector
            for sector B (cross-sector).
        kernel: FreeSpacePoissonKernel, REQUIRED. Its mesh geometry
            (shape/spacing/origin) must match BOTH sectors' recorded
            geometry exactly; its input_dtype must exactly match both
            sectors' realized Theta dtype (no implicit promotion).
        mu_block_size: bounds the left-side GEMM contraction slice width.
            None resolves to left's full n_fused (single block); the
            realized value is recorded.
        nu_block_size: bounds the right-side Poisson-solve batch width
            (and the resulting GEMM slice width). None resolves to the
            right sector's full n_fused; the realized value is recorded.

    Returns:
        PoissonCoreArtifact.

    Raises:
        TypeError: left/right not a PoissonInterpolationSector, or
            kernel not a FreeSpacePoissonKernel.
        ValueError: mesh-geometry mismatch, backend mismatch, dtype
            mismatch, or invalid block sizes.
    """
    if not isinstance(left, PoissonInterpolationSector):
        raise TypeError("left must be a PoissonInterpolationSector.")
    if right is not None and not isinstance(right, PoissonInterpolationSector):
        raise TypeError("right must be a PoissonInterpolationSector or None.")
    if not isinstance(kernel, FreeSpacePoissonKernel):
        raise TypeError("kernel must be a FreeSpacePoissonKernel.")

    right_sector = left if right is None else right

    kernel_geometry = (kernel.mesh.shape, kernel.mesh.spacing, kernel.mesh.origin)
    for sector, label in ((left, "left"), (right_sector, "right")):
        sector_geometry = (sector.mesh_shape, sector.mesh_spacing, sector.mesh_origin)
        if sector_geometry != kernel_geometry:
            raise ValueError(
                f"{label} sector's mesh geometry {sector_geometry} does not match "
                f"kernel.mesh's geometry {kernel_geometry}."
            )

    if left.backend != right_sector.backend:
        raise ValueError(
            f"left.backend={left.backend!r} != right.backend={right_sector.backend!r}."
        )
    if left.backend != kernel.backend:
        raise ValueError(
            f"sector backend={left.backend!r} != kernel.backend={kernel.backend!r}."
        )

    input_dtype = np.dtype(kernel.input_dtype)
    for sector, label in ((left, "left"), (right_sector, "right")):
        if np.dtype(sector.realized_dtype) != input_dtype:
            raise ValueError(
                f"{label} sector's Theta dtype {sector.realized_dtype} != "
                f"kernel.input_dtype {input_dtype} -- no implicit promotion; build the "
                f"sector's factors at the exact dtype the kernel expects."
            )

    n_mu_left = left.Theta.shape[0]
    n_mu_right = right_sector.Theta.shape[0]
    mu_block_size_realized = (
        _validate_positive_int("mu_block_size", mu_block_size)
        if mu_block_size is not None else n_mu_left
    )
    nu_block_size_realized = (
        _validate_positive_int("nu_block_size", nu_block_size)
        if nu_block_size is not None else n_mu_right
    )

    xp = jnp if kernel.backend == "jax" else np
    mesh_shape = kernel.mesh.shape
    dV = kernel.mesh.spacing[0] * kernel.mesh.spacing[1] * kernel.mesh.spacing[2]

    z_row_blocks = []
    for nu_start in range(0, n_mu_right, nu_block_size_realized):
        nu_end = min(nu_start + nu_block_size_realized, n_mu_right)
        b_nu = nu_end - nu_start
        theta_nu_block = right_sector.Theta[nu_start:nu_end, :]
        rho_batch = theta_nu_block.reshape((b_nu,) + mesh_shape)
        v_batch = solve_free_space_poisson(rho_batch, kernel)
        v_flat = v_batch.reshape(b_nu, -1)

        mu_blocks = []
        for mu_start in range(0, n_mu_left, mu_block_size_realized):
            mu_end = min(mu_start + mu_block_size_realized, n_mu_left)
            theta_mu_block = left.Theta[mu_start:mu_end, :]
            mu_blocks.append(dV * (theta_mu_block.conj() @ v_flat.T))  # (b_mu, b_nu)
        z_row_blocks.append(xp.concatenate(mu_blocks, axis=0))  # (n_mu_left, b_nu)

    Z = xp.concatenate(z_row_blocks, axis=1)  # (n_mu_left, n_mu_right)

    device = "cpu" if kernel.backend == "numpy" else str(Z.device)
    realized_dtype = str(Z.dtype)

    core_spec_sha256 = _kernel_spec_sha256({
        "left_sector_spec_sha256": left.sector_spec_sha256,
        "right_sector_spec_sha256": right_sector.sector_spec_sha256,
        "left_n_fused": n_mu_left, "right_n_fused": n_mu_right,
        "kernel_spec_sha256": kernel.kernel_spec_sha256,
        "mu_block_size": mu_block_size_realized, "nu_block_size": nu_block_size_realized,
        "normalization": "dV", "backend": kernel.backend, "device": device,
        "realized_dtype": realized_dtype, "solver_version": _POISSON_INTERPOLATION_SOLVER_VERSION,
    })

    return PoissonCoreArtifact(
        Z=Z,
        left_sector_spec_sha256=left.sector_spec_sha256,
        right_sector_spec_sha256=right_sector.sector_spec_sha256,
        left_n_fused=n_mu_left, right_n_fused=n_mu_right,
        kernel_spec_sha256=kernel.kernel_spec_sha256,
        mu_block_size=mu_block_size_realized, nu_block_size=nu_block_size_realized,
        normalization="dV", backend=kernel.backend, device=device, realized_dtype=realized_dtype,
        solver_version=_POISSON_INTERPOLATION_SOLVER_VERSION,
        core_spec_sha256=core_spec_sha256,
    )
