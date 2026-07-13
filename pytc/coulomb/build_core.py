"""build_core: assembles a complete, reproducible LS-THC/ISDF core
artifact for one MO-pair sector (same-sector) or a pair of independently
-pivoted sectors (cross-sector), per the isdf-coulomb-cuda design doc
§4's "provenance fields REQUIRED on every core build" list -- pivot
selection, P/C construction, and Z computation, plus the upstream/grid/
pivot provenance compute_Z/compute_Z_cross cannot fabricate themselves
(Alice/Felix's ruling, 2026-07-12: "a Z artifact without grid/pivot/
kernel provenance is irreproducible before any consumer touches it").

Location: pytc/coulomb/, NOT pytc/df/ -- this module CHOOSES the
MolecularDFReference kernel policy (calls compute_C_streamed), while
pytc/df/fit.py stays kernel-agnostic (Alice, 2026-07-12).

API (Alice's amendment to the initial proposal, 2026-07-12):
1. Both RAW and selection-WEIGHTED MO-value factors must be passed
   explicitly -- pivot SELECTION uses weighted values, P construction
   uses RAW values (task #6's original bug: reusing weighted values for
   P made compute_Z's rcond default silently depend on the grid
   quadrature's weight scale).
2. Two-stage, typed API: build_sector(...) -> SectorFit(P, C, pivots,
   provenance) is a reusable artifact; build_core(left, right=None)
   -> CoreArtifact(Z, provenance) -- right=None is same-sector
   (compute_Z), right=<SectorFit> is cross-sector (compute_Z_cross).
   Cross-sector is NOT deferred: conventional CCSD's already-approved
   contract needs oo/ov/vv same-sector cores AND their cross cores
   (oo|vv, ov|vv) from the start.
3. requested_rank, n_pivots (selected count), and numerical_rank
   (pivoted_cholesky_pair_pivots's own Gram-spectrum-measured
   effective_rank) are recorded as THREE DISTINCT provenance fields --
   n_pivots is never presented as "effective rank" on its own.
4. TWO separate canonical SHA-256 hashes: pivot_indices_sha256 (exact
   integer index array) and grid_sha256 (canonicalized coords AND
   weights together) -- grid_lvl + basis + geometry alone does not pin
   the realized grid (pruning, atom-grid overrides, radii scheme, pyscf
   version, and point ordering can all vary it). MO coefficients are
   canonicalized (contiguous array + dtype/shape) before hashing too.
   hashlib.sha256 throughout, never Python's built-in hash() (unstable
   across processes/runs). upstream_provenance (pyscf/gpu4pyscf
   versions, the actual realized grid settings) is caller-supplied
   explicitly, never inferred via `mf` type inspection -- for the same
   reason grid_sha256 exists: inference cannot see pruning/version/
   ordering effects.
"""

import dataclasses
import hashlib
import logging

import numpy as np

from pytc.coulomb.pivot_selection import select_sector_pivots
from pytc.coulomb.molecular_df_reference import compute_C_streamed
from pytc.df.fit import pair_collocation_at_pivots, compute_Z, compute_Z_cross

logger = logging.getLogger(__name__)

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


@dataclasses.dataclass(frozen=True)
class SectorFit:
    """One MO-pair sector's reusable ISDF/LS-THC fit artifact: pivots
    selected, P and C constructed. Reusable across multiple build_core
    joins (e.g. the same "vv" SectorFit feeds both the vv|vv same-sector
    core and the oo|vv / ov|vv cross-sector cores) without recomputing
    pivot selection or re-streaming the DF integrals."""
    P: np.ndarray
    C: np.ndarray
    pivots: np.ndarray
    provenance: dict


@dataclasses.dataclass(frozen=True)
class CoreArtifact:
    """A completed Z core for one sector (same-sector) or one sector
    pair (cross-sector), with full provenance."""
    Z: np.ndarray
    provenance: dict


def build_sector(mf, factor_p_raw, factor_q_raw, factor_p_weighted, factor_q_weighted,
                  mo_coeff_p, mo_coeff_q, grid_coords, grid_weights, requested_rank,
                  *, same_factor=False, effective_rank_rtol=1e-6, shift=None,
                  on_over_rank="truncate", kernel_policy="MolecularDFReference",
                  auxbasis="weigend", blksize=None, upstream_provenance=None):
    """Select pivots and build P, C for ONE MO-pair sector (e.g. "oo",
    "ov", "vv").

    Args:
        mf: Converged mean-field object -- forwarded to compute_C_streamed
            (kernel_policy="MolecularDFReference"'s DF-B-tensor route).
        factor_p_raw, factor_q_raw: (n_p/n_q, n_grid) RAW (unweighted)
            MO values -- used for P construction via
            pair_collocation_at_pivots, NEVER for pivot selection.
        factor_p_weighted, factor_q_weighted: (n_p/n_q, n_grid)
            sqrt(weight)-scaled MO values (pytc.coulomb.pivot_selection.
            weight_mo_values's convention) -- used for pivot SELECTION
            only. Passing weighted values into P construction instead
            was task #6's original bug (see pair_collocation_at_pivots's
            docstring) -- both factor kinds are required explicitly here
            so that mistake cannot recur inside this wrapper.
        mo_coeff_p, mo_coeff_q: (n_ao, n_p/n_q) MO coefficients for this
            sector's two pair indices -- forwarded to compute_C_streamed
            and hashed into provenance.
        grid_coords: (n_grid, 3) grid point coordinates -- hashed into
            provenance (grid_sha256) together with grid_weights; not
            otherwise used numerically here (the factor_* arrays already
            encode the grid's effect on the MO values).
        grid_weights: (n_grid,) integration weights -- hashed alongside
            grid_coords.
        requested_rank: Requested pivot-selection rank for this sector.
        same_factor: True for a symmetric sector (oo, vv) -- forwarded
            to select_sector_pivots.
        effective_rank_rtol, shift, on_over_rank: forwarded to
            select_sector_pivots.
        kernel_policy: Recorded in provenance; only "MolecularDFReference"
            is implemented so far (compute_C_streamed's DF-B-tensor route).
        auxbasis, blksize: forwarded to compute_C_streamed.
        upstream_provenance: Caller-supplied dict of facts this function
            cannot infer safely from `mf` alone -- pyscf/gpu4pyscf
            versions, the ACTUAL grid settings used to build grid_coords/
            grid_weights/factor_* (level, pruning, atom-grid overrides,
            radii scheme), basis name. Alice's ruling, 2026-07-12:
            type-inspecting `mf` for this is unreliable, since none of
            those realized-grid details are visible from `mf`'s type
            alone. None becomes an empty dict, recorded as-is (not
            validated -- this function trusts the caller's own record).

    Returns:
        SectorFit(P, C, pivots, provenance).
    """
    if kernel_policy not in _SUPPORTED_KERNEL_POLICIES:
        raise ValueError(
            f"kernel_policy={kernel_policy!r} not implemented -- only "
            f"{_SUPPORTED_KERNEL_POLICIES!r} exist so far (compute_C_streamed's "
            f"DF-B-tensor route)."
        )
    factor_p_raw = np.asarray(factor_p_raw)
    factor_q_raw = np.asarray(factor_q_raw)
    factor_p_weighted = np.asarray(factor_p_weighted)
    factor_q_weighted = np.asarray(factor_q_weighted)
    mo_coeff_p = np.asarray(mo_coeff_p)
    mo_coeff_q = np.asarray(mo_coeff_q)
    grid_coords = np.asarray(grid_coords)
    grid_weights = np.asarray(grid_weights)

    pivots, pivot_provenance = select_sector_pivots(
        factor_p_weighted, factor_q_weighted, requested_rank, shift=shift,
        on_over_rank=on_over_rank, same_factor=same_factor,
        effective_rank_rtol=effective_rank_rtol, return_provenance=True)
    pivots = np.asarray(pivots)

    P = pair_collocation_at_pivots(factor_p_raw[:, pivots], factor_q_raw[:, pivots])
    C = compute_C_streamed(mf, P, mo_coeff_p, mo_coeff_q, auxbasis=auxbasis, blksize=blksize)

    provenance = {
        "kernel_policy": kernel_policy,
        "kernel_policy_params": {"auxbasis": auxbasis, "blksize": blksize},
        "same_factor": same_factor,
        "effective_rank_rtol": effective_rank_rtol,
        **pivot_provenance,  # requested_rank, analytic_rank_bound, n_rank_capped, numerical_rank, n_pivots
        "pivot_indices_sha256": _canonical_sha256(pivots),
        "grid_sha256": _canonical_sha256(grid_coords, grid_weights),
        "mo_coeff_p_sha256": _canonical_sha256(mo_coeff_p),
        "mo_coeff_q_sha256": _canonical_sha256(mo_coeff_q),
        "upstream_provenance": dict(upstream_provenance) if upstream_provenance else {},
    }
    return SectorFit(P=P, C=C, pivots=pivots, provenance=provenance)


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
            for the same-sector case.
        rcond, solver, **solver_kwargs: forwarded to compute_Z/
            compute_Z_cross (tsvd_rcond, backward_error_mode,
            backward_error_tol, residual_mode, residual_n_probes,
            residual_seed).

    Returns:
        CoreArtifact(Z, provenance) -- provenance merges compute_Z's/
        compute_Z_cross's own solver-level dict with each side's
        SectorFit provenance (key "sector" when same-sector, keys
        "left_sector"/"right_sector" when cross-sector).
    """
    if right is None:
        Z, solver_provenance = compute_Z(
            left.P, left.C, rcond=rcond, solver=solver, **solver_kwargs)
        provenance = {**solver_provenance, "sector": left.provenance}
    else:
        Z, solver_provenance = compute_Z_cross(
            left.P, left.C, right.P, right.C, rcond=rcond, solver=solver,
            same_sector=False, **solver_kwargs)
        provenance = {
            **solver_provenance,
            "left_sector": left.provenance,
            "right_sector": right.provenance,
        }
    return CoreArtifact(Z=Z, provenance=provenance)
