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

API, per two rounds of Alice's review (2026-07-12):

Round 1 (initial proposal -> approved amendments):
1. Weighted MO-value factors are now DERIVED INTERNALLY from raw
   values + grid_weights (via pivot_selection.weight_mo_values), not
   taken as a separate explicit parameter -- Alice's simplification
   over her own original "pass both explicitly" ask: since grid_weights
   is already a required parameter (for hashing), deriving removes the
   raw/weighted consistency-mismatch class entirely rather than
   requiring validation of it.
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

import dataclasses
import hashlib
import logging
import types

import numpy as np
import pyscf

from pytc.coulomb.pivot_selection import select_sector_pivots, weight_mo_values
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
    """
    if isinstance(obj, types.MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        return types.MappingProxyType({
            k: _deep_freeze(v, f"{_path}[{k!r}]") for k, v in obj.items()
        })
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v, f"{_path}[{i}]") for i, v in enumerate(obj))
    if isinstance(obj, (set, frozenset)):
        return frozenset(_deep_freeze(v, _path) for v in obj)
    if isinstance(obj, np.ndarray):
        return _readonly_copy(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, _DEEP_FREEZE_IMMUTABLE_SCALAR_TYPES):
        return obj
    raise TypeError(
        f"_deep_freeze: unsupported object at {_path}: {type(obj).__name__} "
        f"({obj!r}) -- provenance values must be a Mapping/list/tuple/set/"
        f"frozenset/np.ndarray/np.generic, or an immutable scalar (None, bool, "
        f"int, float, complex, str, bytes). Normalize custom objects (e.g. "
        f"pathlib.Path, version objects) to strings before passing them in "
        f"upstream_provenance."
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
