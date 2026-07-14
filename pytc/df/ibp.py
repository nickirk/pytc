"""Direct irregular-grid single-IBP Coulomb primitives (atom-centered
grids), moved here from ``pytc/utils/atom_centered_single_ibp_benchmark.py``
so the canonical implementation lives in ``pytc/df/`` alongside the other
model-agnostic DF/ISDF machinery (task #2, #proj-isdf-ibp-coulomb).
``kernel`` is the main-entrance single-IBP primitive (PySCF-style naming,
per Ke); ``naive_coulomb_kernel`` is the diagnostic comparison quadrature.

Unlike a uniform grid, a PySCF/TC atom-centered Becke grid is not
translationally invariant, so the ``rhat`` action here is evaluated by
blocked real-space summation, not FFT:

    (ia|jb) = -1/2 sum_g w_g grad(phi_i phi_a)_g .
                         sum_h w_h rhat_(g-h) (phi_j phi_b)_h.

Coincident left/right points are assigned the centered-point convention
``rhat=0`` explicitly (not skipped or NaN-guarded implicitly) --
``coincident_pairs`` is returned by both primitives so a caller cannot
silently forget which diagonal convention was used.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import time
import types
import typing

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .solvers import (
    prepare_normal_equations_solver,
    solve_normal_equations_batch_prepared,
)


def _validate_grid_inputs(density, coords, weights, name):
    density = np.asarray(density)
    coords = np.asarray(coords)
    weights = np.asarray(weights)
    if density.ndim != 2:
        raise ValueError(f"{name}_density must have shape (n_function,n_grid)")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{name}_coords must have shape (n_grid,3)")
    if weights.ndim != 1 or weights.shape[0] != coords.shape[0]:
        raise ValueError(f"{name}_weights must have shape (n_grid,)")
    if density.shape[1] != coords.shape[0]:
        raise ValueError(f"{name}_density grid axis does not match {name}_coords")
    if not np.issubdtype(density.dtype, np.inexact):
        raise ValueError(f"{name}_density must use a floating or complex dtype")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} coordinates and weights must be finite")
    return density, coords, weights


def kernel(
    gradient_density_left,
    density_right,
    coords_left,
    weights_left,
    *,
    coords_right=None,
    weights_right=None,
    eval_block_size=128,
    source_block_size=4096,
):
    """Blocked one-sided single-IBP core on one or two irregular grids.

    The returned matrix is deliberately unsymmetrized.  Coincident left/right
    points receive ``rhat=0``; ``coincident_pairs`` is returned so a caller
    cannot silently forget which diagonal convention was used.
    """
    gradient = np.asarray(gradient_density_left)
    density_right = np.asarray(density_right)
    if gradient.ndim != 3 or gradient.shape[1] != 3:
        raise ValueError("gradient_density_left must have shape (n_left,3,n_grid_left)")
    dummy_left = np.empty((gradient.shape[0], gradient.shape[2]), dtype=gradient.dtype)
    _, coords_left, weights_left = _validate_grid_inputs(
        dummy_left, coords_left, weights_left, "left"
    )
    if coords_right is None:
        coords_right = coords_left
    if weights_right is None:
        weights_right = weights_left
    density_right, coords_right, weights_right = _validate_grid_inputs(
        density_right, coords_right, weights_right, "right"
    )
    if gradient.dtype != density_right.dtype:
        raise ValueError("gradient_density_left and density_right dtype must match")
    eval_block_size = int(eval_block_size)
    source_block_size = int(source_block_size)
    if eval_block_size <= 0 or source_block_size <= 0:
        raise ValueError("eval_block_size and source_block_size must be positive")

    n_left = gradient.shape[0]
    n_right = density_right.shape[0]
    result = np.zeros((n_left, n_right), dtype=density_right.dtype)
    coincident_pairs = 0
    for i0 in range(0, coords_left.shape[0], eval_block_size):
        i1 = min(i0 + eval_block_size, coords_left.shape[0])
        n_eval = i1 - i0
        vector = np.zeros((n_right, 3, n_eval), dtype=density_right.dtype)
        for j0 in range(0, coords_right.shape[0], source_block_size):
            j1 = min(j0 + source_block_size, coords_right.shape[0])
            diff = coords_left[i0:i1, None, :] - coords_right[None, j0:j1, :]
            radius = np.linalg.norm(diff, axis=-1)
            coincident_pairs += int(np.count_nonzero(radius == 0.0))
            rhat = np.divide(
                diff, radius[..., None], out=np.zeros_like(diff),
                where=radius[..., None] != 0.0,
            )
            weighted_density = density_right[:, j0:j1] * weights_right[j0:j1]
            vector += np.einsum("nj,ijc->nci", weighted_density, rhat, optimize=True)
        result += -0.5 * np.einsum(
            "mci,nci,i->mn", gradient[:, :, i0:i1].conj(), vector,
            weights_left[i0:i1], optimize=True,
        )
    return result, coincident_pairs


def naive_coulomb_kernel(
    density_left,
    density_right,
    coords_left,
    weights_left,
    *,
    coords_right=None,
    weights_right=None,
    eval_block_size=128,
    source_block_size=4096,
):
    """Diagnostic ``1/r`` quadrature with coincident terms set to zero.

    This is *not* a controlled production self-cell prescription.  It exists
    to quantify how much the bounded single-IBP kernel helps relative to the
    naive singular grid sum on the same atom-centered points.
    """
    density_left, coords_left, weights_left = _validate_grid_inputs(
        density_left, coords_left, weights_left, "left"
    )
    if coords_right is None:
        coords_right = coords_left
    if weights_right is None:
        weights_right = weights_left
    density_right, coords_right, weights_right = _validate_grid_inputs(
        density_right, coords_right, weights_right, "right"
    )
    if density_left.dtype != density_right.dtype:
        raise ValueError("density_left and density_right dtype must match")
    eval_block_size = int(eval_block_size)
    source_block_size = int(source_block_size)
    if eval_block_size <= 0 or source_block_size <= 0:
        raise ValueError("eval_block_size and source_block_size must be positive")

    result = np.zeros((density_left.shape[0], density_right.shape[0]),
                      dtype=density_right.dtype)
    coincident_pairs = 0
    for i0 in range(0, coords_left.shape[0], eval_block_size):
        i1 = min(i0 + eval_block_size, coords_left.shape[0])
        potential = np.zeros((density_right.shape[0], i1 - i0), dtype=density_right.dtype)
        for j0 in range(0, coords_right.shape[0], source_block_size):
            j1 = min(j0 + source_block_size, coords_right.shape[0])
            diff = coords_left[i0:i1, None, :] - coords_right[None, j0:j1, :]
            radius = np.linalg.norm(diff, axis=-1)
            coincident_pairs += int(np.count_nonzero(radius == 0.0))
            inv_r = np.divide(
                1.0, radius, out=np.zeros_like(radius), where=radius != 0.0
            )
            weighted_density = density_right[:, j0:j1] * weights_right[j0:j1]
            potential += weighted_density @ inv_r.T
        result += (density_left[:, i0:i1].conj() * weights_left[i0:i1]) @ potential.T
    return result, coincident_pairs


# ---------------------------------------------------------------------------
# Typed, immutable, provenance-carrying artifacts (task #5, #proj-isdf-ibp-
# coulomb). Mirrors the closed-provenance pattern established in
# pytc.integrals.coulomb (FreeSpacePoissonMesh/FreeSpacePoissonKernel,
# PoissonInterpolationSector/PoissonCoreArtifact): every dataclass is
# frozen=True, __post_init__ RECOMPUTES every derived/hashed field from the
# artifact's own declared fields and requires exact agreement rather than
# trusting the builder -- closing the dataclasses.replace(...) tampering
# hole for anyone constructing the type directly, not just through the
# builder function. These helpers are intentionally self-contained (not
# imported from pytc.integrals.coulomb): ibp.py is the lower DF-layer
# module coulomb.py may later depend on, not the reverse.
# ---------------------------------------------------------------------------

_IBP_GRID_SCHEMA_VERSION = "1"
_IBP_OPERATOR_PLAN_VERSION = "1"
_SUPPORTED_IBP_BACKENDS = ("numpy", "jax")
_SUPPORTED_COINCIDENT_POINT_POLICIES = ("centered_zero",)
_SUPPORTED_IBP_OPERATOR_METHODS = ("direct",)

_DEEP_FREEZE_IMMUTABLE_SCALAR_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _validate_positive_int(name, value):
    """Reject bool (isinstance(True, int) is True) and non-integral floats
    rather than silently truncating them into a plausible-looking but wrong
    value -- mirrors pytc.integrals.coulomb's identically-named helper."""
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


def _canonical_sha256(array):
    """SHA-256 over an array's canonicalized bytes (C-contiguous, shape/dtype
    folded into the digest) -- hashlib, never Python's built-in hash()
    (per-process salted, not reproducible across runs). NumPy-only: never
    called on a JAX array (that would force an unwanted host transfer --
    see IBPGrid's docstring for the caller-attested-identity alternative)."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(repr(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _validate_sha256_hex(name, value):
    """Closed syntax validation for a caller-supplied or locally-computed
    SHA-256 hex digest -- exactly 64 lowercase hex characters, nothing else.
    For a caller-attested JAX-backend identity this is the ONLY validation
    performed -- syntax, not content; never claim a JAX digest was verified
    from device bytes."""
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            f"{name} must be a 64-character lowercase hex SHA-256 string, got {value!r}."
        )


def _readonly_copy(a):
    """Defensive COPY (never the caller's own array object) with the numpy
    write flag cleared -- np.asarray on an already-matching ndarray can
    return the SAME object, so copy first, always."""
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _deep_freeze(obj, _path="<root>"):
    """Recursively convert dict -> types.MappingProxyType, list/tuple ->
    tuple, set/frozenset -> frozenset, numpy arrays/scalars -> read-only
    copies/.item(), at every nesting level. CLOSED schema: any object that
    isn't one of these containers, an immutable scalar (None/bool/int/
    float/complex/str/bytes), or a numpy array/scalar raises TypeError --
    silently passing through an unknown mutable object would break the
    "genuinely immutable" contract. Mapping keys must be str. Object-dtype
    numpy arrays are rejected outright (an array being read-only only
    blocks reassigning elements, not mutating the arbitrary objects those
    elements reference)."""
    if isinstance(obj, types.MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        frozen = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"_deep_freeze: unsupported mapping key at {_path}: "
                    f"{type(k).__name__} ({k!r}) -- provenance mapping keys must be str."
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
                f"(dtype={obj.dtype}) -- normalize to a non-object dtype or a plain "
                f"list/tuple of already-immutable values first."
            )
        return _readonly_copy(obj)
    if isinstance(obj, np.generic):
        return _deep_freeze(obj.item(), _path)
    if isinstance(obj, _DEEP_FREEZE_IMMUTABLE_SCALAR_TYPES):
        return obj
    raise TypeError(
        f"_deep_freeze: unsupported object at {_path}: {type(obj).__name__} ({obj!r}) -- "
        f"provenance values must be a Mapping (str keys)/list/tuple/set/frozenset/"
        f"np.ndarray (non-object dtype)/np.generic, or an immutable scalar."
    )


def _tlv(tag, payload):
    """Tag-length-value: a 1-byte type tag, an 8-byte big-endian payload
    length, then the payload itself. Self-delimiting -- concatenating
    any sequence of TLV-encoded nodes is unambiguous/injective
    regardless of what bytes appear inside a payload (Alice's review,
    task #5, 2026-07-13, round 2: the prior delimiter-separated encoder
    let a string CONTAINING a literal delimiter collide with an
    unrelated sibling structure -- construction_metadata={"seq":
    ("a,str:b",)} hashed identically to {"seq": ("a", "b")}, since both
    serialized to the same delimiter-joined bytes. A length-prefixed
    payload makes that impossible: the reader always knows exactly how
    many bytes belong to this node, so it never needs to interpret
    bytes inside the payload as structure.)"""
    return tag + len(payload).to_bytes(8, "big") + payload


def _canonical_encode_node(obj):
    """Return the complete, self-delimiting TLV bytes for obj (already
    routed through _deep_freeze, or a plain scalar) -- for provenance
    hashing.

    Deliberately NOT repr()-based (Alice's review, task #5,
    2026-07-13, round 1): repr(frozenset(...)) iterates in the set's
    internal hash-table order, which depends on PYTHONHASHSEED for str
    elements (randomized per process by default) -- the same logical
    construction_metadata could hash differently across runs, and
    Alice's repro even showed builder vs __post_init__ disagreeing
    WITHIN one construction (re-freezing a frozenset rebuilds its
    internal table). repr() on a large ndarray also silently truncates
    (numpy's summarized repr). This encoder instead: sorts dict keys
    (str, per _deep_freeze's closed schema); preserves tuple/list
    element order; for frozenset/set, encodes each element
    independently (each already self-delimiting) and sorts the
    resulting byte strings (well-defined regardless of hash-seed-
    dependent iteration order); and hashes an ndarray's FULL raw bytes
    (never a summarized repr) -- each as its own length-prefixed field
    so dtype/shape/bytes can never bleed into one another either."""
    if isinstance(obj, types.MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        parts = []
        for k in sorted(obj):
            if not isinstance(k, str):
                raise TypeError(f"_canonical_encode_node: mapping key must be str, got {k!r}.")
            parts.append(_canonical_encode_node(k))
            parts.append(_canonical_encode_node(obj[k]))
        return _tlv(b"D", b"".join(parts))
    if isinstance(obj, (list, tuple)):
        return _tlv(b"L", b"".join(_canonical_encode_node(v) for v in obj))
    if isinstance(obj, (set, frozenset)):
        encoded_elements = sorted(_canonical_encode_node(v) for v in obj)
        return _tlv(b"S", b"".join(encoded_elements))
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError(f"_canonical_encode_node: unsupported object-dtype array {obj.dtype}.")
        a = np.ascontiguousarray(obj)
        payload = (
            _tlv(b"t", str(a.dtype).encode())
            + _tlv(b"h", repr(a.shape).encode())
            + _tlv(b"b", a.tobytes())
        )
        return _tlv(b"A", payload)
    if isinstance(obj, np.generic):
        return _canonical_encode_node(obj.item())
    if obj is None:
        return _tlv(b"n", b"")
    if isinstance(obj, bool):
        return _tlv(b"b", b"1" if obj else b"0")
    if isinstance(obj, int):
        return _tlv(b"i", repr(obj).encode())
    if isinstance(obj, float):
        return _tlv(b"f", repr(obj).encode())
    if isinstance(obj, complex):
        return _tlv(b"c", repr(obj).encode())
    if isinstance(obj, str):
        return _tlv(b"s", obj.encode("utf-8", errors="surrogatepass"))
    if isinstance(obj, bytes):
        return _tlv(b"y", obj)
    raise TypeError(
        f"_canonical_encode_node: unsupported type {type(obj).__name__} for provenance "
        f"hashing ({obj!r})."
    )


def _canonical_spec_sha256(fields):
    """SHA-256 over a canonical, self-delimiting TLV encoding of
    specification fields (see _canonical_encode_node) -- deterministic
    across processes/hash seeds, never truncates large array data, and
    is injective (no delimiter-collision between distinct structures)."""
    h = hashlib.sha256()
    for key in sorted(fields):
        h.update(_canonical_encode_node(key))
        h.update(_canonical_encode_node(fields[key]))
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class IBPGrid:
    """Immutable atom-centered (or otherwise irregular) real-space
    quadrature grid: coordinates, physical weights, backend/device/dtype,
    coincident-point policy, and closed provenance. Independent of any
    orbital sector -- reusable across every IBPInterpolationSector/
    IBPOperatorPlan/IBPCoreArtifact built on it (later tasks).

    coords/weights: backend-preserving, mirroring
    pytc.integrals.coulomb.FreeSpacePoissonKernel's spectrum handling --
    NumPy gets a defensive read-only copy; JAX stays the caller's untouched
    (already-immutable) jax.Array on its actual device, never forced
    through a NumPy-only helper (which would silently transfer/duplicate
    the array on host).

    coords_sha256/weights_sha256: for backend="numpy", a locally computed
    SHA-256 over the actual array bytes, verified against the realized
    array in __post_init__ (closes the dataclasses.replace(...) tampering
    hole). For backend="jax", a CALLER-ATTESTED identity string -- device-
    resident arrays are never hashed on the fly (that would force an
    unwanted host transfer); __post_init__ can only validate its syntax
    (64-hex SHA-256 shape), not its content, exactly as
    pytc.integrals.coulomb.PoissonInterpolationSector's factor_p_sha256/
    factor_q_sha256 already document for the same reason.

    coincident_point_policy: the odd-kernel convention used wherever this
    grid's own points coincide with themselves or another grid's points
    (this module's kernel/naive_coulomb_kernel both use "centered_zero":
    r=0 -> rhat=0). Only "centered_zero" is currently supported; recorded
    explicitly here so a downstream consumer can never silently assume it.

    construction_metadata: arbitrary caller-supplied facts about how this
    grid was built (e.g. mol identity, grid_level, prune scheme) -- deep-
    frozen at every nesting level, never merely shallow-frozen.
    """
    coords: object
    weights: object
    n_grid: int
    backend: str
    device: str
    dtype: str
    coincident_point_policy: str
    coords_sha256: str
    weights_sha256: str
    numpy_version: str
    jax_version: object
    construction_metadata: object
    schema_version: str
    grid_spec_sha256: str

    def __post_init__(self):
        if self.backend not in _SUPPORTED_IBP_BACKENDS:
            raise ValueError(
                f"Unsupported backend={self.backend!r}, must be one of "
                f"{_SUPPORTED_IBP_BACKENDS}."
            )
        if self.coincident_point_policy not in _SUPPORTED_COINCIDENT_POINT_POLICIES:
            raise ValueError(
                f"Unsupported coincident_point_policy={self.coincident_point_policy!r}, "
                f"must be one of {_SUPPORTED_COINCIDENT_POINT_POLICIES}."
            )

        if self.backend == "numpy":
            if not isinstance(self.coords, np.ndarray) or not isinstance(self.weights, np.ndarray):
                raise TypeError(
                    f"backend='numpy' requires coords/weights to be numpy.ndarray, got "
                    f"{type(self.coords).__name__}/{type(self.weights).__name__}."
                )
            object.__setattr__(self, "coords", _readonly_copy(self.coords))
            object.__setattr__(self, "weights", _readonly_copy(self.weights))
        else:
            if not isinstance(self.coords, jax.Array) or not isinstance(self.weights, jax.Array):
                raise TypeError(
                    f"backend='jax' requires coords/weights to be jax.Array, got "
                    f"{type(self.coords).__name__}/{type(self.weights).__name__}."
                )

        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (n_grid,3), got {tuple(self.coords.shape)}.")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.coords.shape[0]:
            raise ValueError(
                f"weights must have shape (n_grid,) matching coords, got "
                f"{tuple(self.weights.shape)} vs coords {tuple(self.coords.shape)}."
            )
        if self.coords.shape[0] == 0:
            raise ValueError("coords/weights must contain at least one grid point.")
        if self.n_grid != self.coords.shape[0]:
            raise ValueError(f"n_grid={self.n_grid} does not match coords.shape[0]={self.coords.shape[0]}.")

        coords_dtype = np.dtype(self.coords.dtype) if self.backend == "numpy" else self.coords.dtype
        weights_dtype = np.dtype(self.weights.dtype) if self.backend == "numpy" else self.weights.dtype
        if not np.issubdtype(coords_dtype, np.floating) or not np.issubdtype(weights_dtype, np.floating):
            raise ValueError(
                f"coords/weights must use a real floating dtype (grid geometry/weights "
                f"are always real), got {coords_dtype}/{weights_dtype}."
            )
        if coords_dtype != weights_dtype:
            raise ValueError(f"coords dtype {coords_dtype} must match weights dtype {weights_dtype}.")
        if str(coords_dtype) != self.dtype:
            raise ValueError(f"dtype={self.dtype!r} does not match the realized coords dtype {coords_dtype}.")

        if self.backend == "numpy":
            if not np.all(np.isfinite(self.coords)) or not np.all(np.isfinite(self.weights)):
                raise ValueError("coords and weights must be finite.")
            if self.device != "cpu":
                raise ValueError(f"device must be 'cpu' for backend='numpy', got {self.device!r}.")
            if self.jax_version is not None:
                raise ValueError(f"jax_version must be None for backend='numpy', got {self.jax_version!r}.")
            recomputed_coords_sha256 = _canonical_sha256(self.coords)
            recomputed_weights_sha256 = _canonical_sha256(self.weights)
            if self.coords_sha256 != recomputed_coords_sha256:
                raise ValueError(
                    "coords_sha256 does not match the digest recomputed from the realized "
                    "coords array."
                )
            if self.weights_sha256 != recomputed_weights_sha256:
                raise ValueError(
                    "weights_sha256 does not match the digest recomputed from the realized "
                    "weights array."
                )
        else:
            if not bool(jnp.all(jnp.isfinite(self.coords))) or not bool(jnp.all(jnp.isfinite(self.weights))):
                raise ValueError("coords and weights must be finite.")
            coords_device = str(self.coords.device)
            weights_device = str(self.weights.device)
            if coords_device != weights_device:
                raise ValueError(
                    f"coords and weights must reside on the same device, got "
                    f"{coords_device!r} vs {weights_device!r}."
                )
            if self.device != coords_device:
                raise ValueError(f"device={self.device!r} != the realized coords device {coords_device!r}.")
            if self.jax_version != jax.__version__:
                raise ValueError(
                    f"jax_version={self.jax_version!r} != the running jax.__version__ "
                    f"{jax.__version__!r}."
                )
            # Caller-attested identity: syntax-only validation, see class docstring.
            _validate_sha256_hex("coords_sha256", self.coords_sha256)
            _validate_sha256_hex("weights_sha256", self.weights_sha256)

        if self.numpy_version != np.__version__:
            raise ValueError(
                f"numpy_version={self.numpy_version!r} != the running numpy.__version__ "
                f"{np.__version__!r}."
            )
        if self.schema_version != _IBP_GRID_SCHEMA_VERSION:
            raise ValueError(f"schema_version={self.schema_version!r} != {_IBP_GRID_SCHEMA_VERSION!r}.")

        object.__setattr__(
            self, "construction_metadata",
            _deep_freeze(dict(self.construction_metadata) if self.construction_metadata else {})
        )

        recomputed_spec = _canonical_spec_sha256({
            "n_grid": self.n_grid, "backend": self.backend, "device": self.device,
            "dtype": self.dtype, "coincident_point_policy": self.coincident_point_policy,
            "coords_sha256": self.coords_sha256, "weights_sha256": self.weights_sha256,
            "numpy_version": self.numpy_version, "jax_version": self.jax_version,
            "schema_version": self.schema_version,
            "construction_metadata": self.construction_metadata,
        })
        if recomputed_spec != self.grid_spec_sha256:
            raise ValueError(
                "grid_spec_sha256 does not match the canonical digest recomputed from this "
                "artifact's own declared fields."
            )


def build_ibp_grid(coords, weights, *, backend="numpy",
                    coincident_point_policy="centered_zero",
                    coords_identity=None, weights_identity=None,
                    construction_metadata=None):
    """Build an immutable IBPGrid from raw coordinates/weights.

    Args:
        coords: (n_grid,3) real coordinates.
        weights: (n_grid,) real physical quadrature weights.
        backend: "numpy" (default) or "jax".
        coincident_point_policy: odd-kernel convention for coincident
            points; only "centered_zero" is currently supported (matches
            this module's kernel/naive_coulomb_kernel r=0 -> rhat=0
            convention).
        coords_identity/weights_identity: REQUIRED, 64-hex SHA-256 caller-
            attested digests, for backend="jax" only (device-resident
            arrays are never hashed on the fly). Must be omitted (None)
            for backend="numpy", where the digest is always computed
            locally from the actual array bytes.
        construction_metadata: optional dict of caller-supplied facts
            about how this grid was built (e.g. mol identity, grid_level)
            -- deep-frozen into the returned artifact's provenance.

    Returns:
        IBPGrid.
    """
    if backend not in _SUPPORTED_IBP_BACKENDS:
        raise ValueError(f"Unsupported backend={backend!r}, must be one of {_SUPPORTED_IBP_BACKENDS}.")
    if coincident_point_policy not in _SUPPORTED_COINCIDENT_POINT_POLICIES:
        raise ValueError(
            f"Unsupported coincident_point_policy={coincident_point_policy!r}, must be one "
            f"of {_SUPPORTED_COINCIDENT_POINT_POLICIES}."
        )

    if backend == "numpy":
        if coords_identity is not None or weights_identity is not None:
            raise ValueError(
                "coords_identity/weights_identity are only accepted for backend='jax' -- "
                "numpy digests are always computed locally from the actual array bytes."
            )
        coords_arr = np.asarray(coords)
        weights_arr = np.asarray(weights)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
            raise ValueError(f"coords must have shape (n_grid,3), got {coords_arr.shape}.")
        if weights_arr.ndim != 1 or weights_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError(
                f"weights must have shape (n_grid,) matching coords, got {weights_arr.shape} "
                f"vs coords {coords_arr.shape}."
            )
        if coords_arr.shape[0] == 0:
            raise ValueError("coords/weights must contain at least one grid point.")
        if not np.issubdtype(coords_arr.dtype, np.floating) or not np.issubdtype(weights_arr.dtype, np.floating):
            raise ValueError("coords and weights must use a real floating dtype.")
        if coords_arr.dtype != weights_arr.dtype:
            raise ValueError(f"coords dtype {coords_arr.dtype} must match weights dtype {weights_arr.dtype}.")
        if not np.all(np.isfinite(coords_arr)) or not np.all(np.isfinite(weights_arr)):
            raise ValueError("coords and weights must be finite.")
        device = "cpu"
        jax_version = None
        dtype_str = str(coords_arr.dtype)
        n_grid = coords_arr.shape[0]
        coords_sha256 = _canonical_sha256(coords_arr)
        weights_sha256 = _canonical_sha256(weights_arr)
        coords_final, weights_final = coords_arr, weights_arr
    else:
        if not isinstance(coords, jax.Array) or not isinstance(weights, jax.Array):
            raise TypeError(
                f"backend='jax' requires coords/weights to be jax.Array, got "
                f"{type(coords).__name__}/{type(weights).__name__}."
            )
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (n_grid,3), got {tuple(coords.shape)}.")
        if weights.ndim != 1 or weights.shape[0] != coords.shape[0]:
            raise ValueError(
                f"weights must have shape (n_grid,) matching coords, got {tuple(weights.shape)} "
                f"vs coords {tuple(coords.shape)}."
            )
        if coords.shape[0] == 0:
            raise ValueError("coords/weights must contain at least one grid point.")
        if not jnp.issubdtype(coords.dtype, jnp.floating) or not jnp.issubdtype(weights.dtype, jnp.floating):
            raise ValueError("coords and weights must use a real floating dtype.")
        if coords.dtype != weights.dtype:
            raise ValueError(f"coords dtype {coords.dtype} must match weights dtype {weights.dtype}.")
        if not bool(jnp.all(jnp.isfinite(coords))) or not bool(jnp.all(jnp.isfinite(weights))):
            raise ValueError("coords and weights must be finite.")
        if coords_identity is None or weights_identity is None:
            raise ValueError(
                "backend='jax' requires caller-attested coords_identity/weights_identity "
                "SHA-256 hex digests -- device-resident arrays are never hashed on the fly "
                "(that would force an unwanted host transfer)."
            )
        _validate_sha256_hex("coords_identity", coords_identity)
        _validate_sha256_hex("weights_identity", weights_identity)
        coords_device = str(coords.device)
        weights_device = str(weights.device)
        if coords_device != weights_device:
            raise ValueError(
                f"coords and weights must reside on the same device, got {coords_device!r} "
                f"vs {weights_device!r}."
            )
        device = coords_device
        jax_version = jax.__version__
        dtype_str = str(coords.dtype)
        n_grid = coords.shape[0]
        coords_sha256 = coords_identity
        weights_sha256 = weights_identity
        coords_final, weights_final = coords, weights

    frozen_metadata = _deep_freeze(dict(construction_metadata) if construction_metadata else {})
    grid_spec_sha256 = _canonical_spec_sha256({
        "n_grid": n_grid, "backend": backend, "device": device, "dtype": dtype_str,
        "coincident_point_policy": coincident_point_policy,
        "coords_sha256": coords_sha256, "weights_sha256": weights_sha256,
        "numpy_version": np.__version__, "jax_version": jax_version,
        "schema_version": _IBP_GRID_SCHEMA_VERSION,
        "construction_metadata": frozen_metadata,
    })

    return IBPGrid(
        coords=coords_final, weights=weights_final, n_grid=n_grid, backend=backend,
        device=device, dtype=dtype_str, coincident_point_policy=coincident_point_policy,
        coords_sha256=coords_sha256, weights_sha256=weights_sha256,
        numpy_version=np.__version__, jax_version=jax_version,
        construction_metadata=frozen_metadata, schema_version=_IBP_GRID_SCHEMA_VERSION,
        grid_spec_sha256=grid_spec_sha256,
    )


@dataclasses.dataclass(frozen=True)
class IBPOperatorPlan:
    """Immutable, grid-bound execution plan for the single-IBP Coulomb
    operator: method, realized block sizes, requested tolerance, backend/
    device/dtype (always matching the bound IBPGrid exactly -- a plan
    never silently retargets a different backend/device than the grid it
    was built for), method version, and closed provenance.

    Deliberately holds NO array data of its own -- in particular, never a
    dense (3, N_grid, N_grid)-shaped kernel; the "direct" method tiles
    coordinate pairs on the fly at core-build time (task #7, later), using
    only this plan's block sizes, exactly like this module's own kernel().

    Only method="direct" (the exact reference/production path moved in
    task #2) is implemented; "hierarchical"/"nufft" are reserved
    identifiers for the later accelerator bake-off (M3) and are rejected
    explicitly rather than silently falling through to "direct".
    """
    grid: object
    method: str
    eval_block_size: int
    source_block_size: int
    tolerance: object
    backend: str
    device: str
    dtype: str
    method_version: str
    provenance: object
    operator_spec_sha256: str

    def __post_init__(self):
        if not isinstance(self.grid, IBPGrid):
            raise TypeError(f"grid must be an IBPGrid, got {type(self.grid).__name__}.")
        if self.method not in _SUPPORTED_IBP_OPERATOR_METHODS:
            raise ValueError(
                f"Unsupported method={self.method!r}, must be one of "
                f"{_SUPPORTED_IBP_OPERATOR_METHODS} in this milestone."
            )
        eval_block_size = _validate_positive_int("eval_block_size", self.eval_block_size)
        source_block_size = _validate_positive_int("source_block_size", self.source_block_size)
        object.__setattr__(self, "eval_block_size", eval_block_size)
        object.__setattr__(self, "source_block_size", source_block_size)

        if self.method == "direct" and self.tolerance is not None:
            raise ValueError(
                "tolerance must be None for method='direct' (an exact reference method has "
                "no approximation tolerance to request)."
            )

        if self.backend != self.grid.backend:
            raise ValueError(
                f"backend={self.backend!r} does not match the bound grid's backend "
                f"{self.grid.backend!r} -- a plan is grid-bound and must not silently "
                f"retarget a different backend."
            )
        if self.device != self.grid.device:
            raise ValueError(
                f"device={self.device!r} does not match the bound grid's device "
                f"{self.grid.device!r}."
            )
        if self.dtype != self.grid.dtype:
            raise ValueError(
                f"dtype={self.dtype!r} does not match the bound grid's dtype {self.grid.dtype!r}."
            )
        if self.method_version != _IBP_OPERATOR_PLAN_VERSION:
            raise ValueError(
                f"method_version={self.method_version!r} != {_IBP_OPERATOR_PLAN_VERSION!r}."
            )

        object.__setattr__(
            self, "provenance", _deep_freeze(dict(self.provenance) if self.provenance else {})
        )

        recomputed_spec = _canonical_spec_sha256({
            "grid_grid_spec_sha256": self.grid.grid_spec_sha256,
            "method": self.method, "eval_block_size": self.eval_block_size,
            "source_block_size": self.source_block_size, "tolerance": self.tolerance,
            "backend": self.backend, "device": self.device, "dtype": self.dtype,
            "method_version": self.method_version, "provenance": self.provenance,
        })
        if recomputed_spec != self.operator_spec_sha256:
            raise ValueError(
                "operator_spec_sha256 does not match the canonical digest recomputed from "
                "this artifact's own declared fields."
            )


def build_ibp_operator_plan(grid, *, method="direct", tolerance=None,
                             eval_block_size=128, source_block_size=4096,
                             upstream_provenance=None):
    """Build an IBPOperatorPlan bound to an already-built IBPGrid.

    Args:
        grid: IBPGrid this plan is bound to.
        method: "direct" (default; only implemented option in this
            milestone).
        tolerance: must be None for method="direct".
        eval_block_size/source_block_size: positive ints, the same tiling
            knobs kernel()/naive_coulomb_kernel() already accept --
            recorded here so a later core build (task #7) can reuse this
            plan's realized values instead of re-deciding them.
        upstream_provenance: optional caller-supplied dict of extra facts,
            deep-frozen into the returned plan's provenance.

    Returns:
        IBPOperatorPlan.
    """
    if not isinstance(grid, IBPGrid):
        raise TypeError(f"grid must be an IBPGrid, got {type(grid).__name__}.")
    if method not in _SUPPORTED_IBP_OPERATOR_METHODS:
        raise ValueError(
            f"Unsupported method={method!r}, must be one of "
            f"{_SUPPORTED_IBP_OPERATOR_METHODS} in this milestone."
        )
    if method == "direct" and tolerance is not None:
        raise ValueError(
            "tolerance must be None for method='direct' (an exact reference method has no "
            "approximation tolerance to request)."
        )
    eval_block_size = _validate_positive_int("eval_block_size", eval_block_size)
    source_block_size = _validate_positive_int("source_block_size", source_block_size)

    upstream_provenance = dict(upstream_provenance) if upstream_provenance else {}
    provenance = {
        "grid_grid_spec_sha256": grid.grid_spec_sha256,
        "grid_construction_metadata": dict(grid.construction_metadata),
        "upstream_provenance": upstream_provenance,
    }
    frozen_provenance = _deep_freeze(provenance)

    operator_spec_sha256 = _canonical_spec_sha256({
        "grid_grid_spec_sha256": grid.grid_spec_sha256,
        "method": method, "eval_block_size": eval_block_size,
        "source_block_size": source_block_size, "tolerance": tolerance,
        "backend": grid.backend, "device": grid.device, "dtype": grid.dtype,
        "method_version": _IBP_OPERATOR_PLAN_VERSION, "provenance": frozen_provenance,
    })

    return IBPOperatorPlan(
        grid=grid, method=method, eval_block_size=eval_block_size,
        source_block_size=source_block_size, tolerance=tolerance,
        backend=grid.backend, device=grid.device, dtype=grid.dtype,
        method_version=_IBP_OPERATOR_PLAN_VERSION,
        provenance=frozen_provenance, operator_spec_sha256=operator_spec_sha256,
    )


# ---------------------------------------------------------------------------
# Fixed-pivot interpolation sectors (task #6).
# ---------------------------------------------------------------------------

_IBP_INTERPOLATION_SOLVER_VERSION = "1"
_SUPPORTED_IBP_STORAGE_MODES = ("incore_full_theta_gradient",)
_IBP_FACTOR_IDENTITY_KEYS = (
    "factor_p_sha256",
    "factor_q_sha256",
    "gradient_p_sha256",
    "gradient_q_sha256",
)
_IBP_PIVOT_PROVENANCE_KEYS = frozenset({
    "requested_rank",
    "analytic_rank_bound",
    "n_rank_capped",
    "rank_exhausted",
    "numerical_rank",
    "numerical_rank_lower_bound",
    "n_pivots",
})


def _validate_nonnegative_finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")
    return value


def _validate_pivot_provenance(record, *, n_p, n_q, same_factor, selected_rank):
    """Validate the exact record returned by select_sector_pivots(...,
    return_provenance=True), without importing the higher integrals layer.

    pytc.df.ibp is intentionally below pytc.integrals.coulomb in the package
    graph.  Consuming and checking the selector's closed record preserves that
    layering while preventing requested, selected, and numerical ranks from
    being silently conflated.
    """
    if record is None:
        raise ValueError(
            "pivot_provenance is required and must be the record returned by "
            "select_sector_pivots(..., return_provenance=True)."
        )
    try:
        record = dict(record)
    except (TypeError, ValueError) as exc:
        raise TypeError("pivot_provenance must be a mapping.") from exc
    keys = frozenset(record)
    if keys != _IBP_PIVOT_PROVENANCE_KEYS:
        missing = sorted(_IBP_PIVOT_PROVENANCE_KEYS - keys)
        extra = sorted(keys - _IBP_PIVOT_PROVENANCE_KEYS)
        raise ValueError(
            f"pivot_provenance has a non-closed schema; missing={missing}, extra={extra}."
        )

    requested_rank = _validate_positive_int("requested_rank", record["requested_rank"])
    analytic_rank_bound = (
        n_p * (n_p + 1) // 2 if same_factor else n_p * n_q
    )
    if record["analytic_rank_bound"] != analytic_rank_bound:
        raise ValueError(
            f"pivot_provenance analytic_rank_bound={record['analytic_rank_bound']!r} "
            f"does not match the factor shapes ({analytic_rank_bound})."
        )
    n_rank_capped = min(requested_rank, analytic_rank_bound)
    if record["n_rank_capped"] != n_rank_capped:
        raise ValueError(
            f"pivot_provenance n_rank_capped={record['n_rank_capped']!r} != "
            f"min(requested_rank, analytic_rank_bound)={n_rank_capped}."
        )
    if record["n_pivots"] != selected_rank:
        raise ValueError(
            f"pivot_provenance n_pivots={record['n_pivots']!r} != actual pivot count "
            f"{selected_rank}."
        )
    if not isinstance(record["rank_exhausted"], bool):
        raise TypeError("pivot_provenance rank_exhausted must be bool.")
    rank_exhausted = record["rank_exhausted"]
    numerical_rank_lower_bound = _validate_positive_int(
        "numerical_rank_lower_bound", record["numerical_rank_lower_bound"]
    )
    if numerical_rank_lower_bound != selected_rank:
        raise ValueError(
            "pivot_provenance numerical_rank_lower_bound must equal the selected pivot "
            f"count ({selected_rank}), got {numerical_rank_lower_bound}."
        )
    numerical_rank = record["numerical_rank"]
    if rank_exhausted:
        numerical_rank = _validate_positive_int("numerical_rank", numerical_rank)
        if numerical_rank != selected_rank or selected_rank >= n_rank_capped:
            raise ValueError(
                "rank_exhausted=True requires numerical_rank == selected_rank < "
                "n_rank_capped."
            )
    else:
        if numerical_rank is not None or selected_rank != n_rank_capped:
            raise ValueError(
                "rank_exhausted=False requires numerical_rank=None and "
                "selected_rank == n_rank_capped."
            )

    return {
        "requested_rank": requested_rank,
        "analytic_rank_bound": analytic_rank_bound,
        "n_rank_capped": n_rank_capped,
        "rank_exhausted": rank_exhausted,
        "numerical_rank": numerical_rank,
        "numerical_rank_lower_bound": numerical_rank_lower_bound,
        "n_pivots": selected_rank,
    }


@dataclasses.dataclass(frozen=True)
class IBPInterpolationSector:
    """One fixed-pivot orbital-pair interpolation sector on an IBPGrid.

    ``P`` is the raw pair collocation at the selected points.  Same-factor
    sectors use packed-lower pair order (the layout required by the later
    PySCF ``loop()`` bridge); mixed sectors use full row-major ``(p,q)``
    order. ``Theta[mu,g]`` and ``grad_Theta[mu,3,g]`` are built with one
    prepared normal-equation factorization.  Gradient construction applies
    the product rule using those exact pivots/factorization; it never runs a
    second pivot selection.

    NumPy outputs are defensive read-only copies. JAX outputs remain on the
    realized device. Large JAX inputs are identified by caller-attested
    SHA-256 strings; only the small pivot vector is transferred to the host
    for canonical validation/hashing.
    """
    P: object
    Theta: object
    grad_Theta: object
    pivots: object
    grid: object
    n_orbital_p: int
    n_orbital_q: int
    n_pair: int
    n_grid: int
    same_factor: bool
    pair_layout: str
    requested_rank: int
    analytic_rank_bound: int
    n_rank_capped: int
    selected_rank: int
    rank_exhausted: bool
    numerical_rank: object
    numerical_rank_lower_bound: int
    storage_mode: str
    backend: str
    device: str
    realized_dtype: str
    grid_batch_size: int
    rcond: float
    jitter_used: float
    n_tries: int
    factor_identity_source: str
    factor_p_sha256: str
    factor_q_sha256: str
    gradient_p_sha256: str
    gradient_q_sha256: str
    pivots_sha256: str
    output_identity_source: str
    p_sha256: object
    theta_sha256: object
    grad_theta_sha256: object
    solver_version: str
    provenance: object
    sector_spec_sha256: str

    def __post_init__(self):
        if not isinstance(self.grid, IBPGrid):
            raise TypeError(f"grid must be an IBPGrid, got {type(self.grid).__name__}.")
        if not isinstance(self.same_factor, bool):
            raise TypeError("same_factor must be bool.")
        if self.backend not in _SUPPORTED_IBP_BACKENDS:
            raise ValueError(f"Unsupported backend={self.backend!r}.")
        if self.backend != self.grid.backend:
            raise ValueError("sector backend must match the bound IBPGrid backend.")
        if self.storage_mode not in _SUPPORTED_IBP_STORAGE_MODES:
            raise ValueError(f"Unsupported storage_mode={self.storage_mode!r}.")

        arrays = (self.P, self.Theta, self.grad_Theta, self.pivots)
        if self.backend == "numpy":
            if not all(isinstance(a, np.ndarray) for a in arrays):
                raise TypeError("backend='numpy' requires P/Theta/grad_Theta/pivots as ndarray.")
            for name in ("P", "Theta", "grad_Theta", "pivots"):
                object.__setattr__(self, name, _readonly_copy(getattr(self, name)))
            if self.device != "cpu" or self.grid.device != "cpu":
                raise ValueError("NumPy sectors and their grid must use device='cpu'.")
        else:
            if not all(isinstance(a, jax.Array) for a in arrays):
                raise TypeError("backend='jax' requires P/Theta/grad_Theta/pivots as jax.Array.")
            devices = {str(a.device) for a in arrays}
            if devices != {self.device} or self.device != self.grid.device:
                raise ValueError(
                    f"JAX sector arrays and grid must share device={self.device!r}, got {devices}."
                )

        n_p = _validate_positive_int("n_orbital_p", self.n_orbital_p)
        n_q = _validate_positive_int("n_orbital_q", self.n_orbital_q)
        n_grid = _validate_positive_int("n_grid", self.n_grid)
        selected_rank = _validate_positive_int("selected_rank", self.selected_rank)
        if n_grid != self.grid.n_grid:
            raise ValueError(f"n_grid={n_grid} != bound grid.n_grid={self.grid.n_grid}.")
        expected_layout = "packed_lower" if self.same_factor else "full_row_major"
        expected_n_pair = n_p * (n_p + 1) // 2 if self.same_factor else n_p * n_q
        if self.same_factor and n_p != n_q:
            raise ValueError("same_factor=True requires n_orbital_p == n_orbital_q.")
        if self.pair_layout != expected_layout or self.n_pair != expected_n_pair:
            raise ValueError(
                f"pair layout/count must be {expected_layout!r}/{expected_n_pair}, got "
                f"{self.pair_layout!r}/{self.n_pair}."
            )
        if self.P.shape != (selected_rank, expected_n_pair):
            raise ValueError(
                f"P must have shape {(selected_rank, expected_n_pair)}, got {self.P.shape}."
            )
        if self.Theta.shape != (selected_rank, n_grid):
            raise ValueError(
                f"Theta must have shape {(selected_rank, n_grid)}, got {self.Theta.shape}."
            )
        if self.grad_Theta.shape != (selected_rank, 3, n_grid):
            raise ValueError(
                f"grad_Theta must have shape {(selected_rank, 3, n_grid)}, "
                f"got {self.grad_Theta.shape}."
            )
        if self.pivots.shape != (selected_rank,) or not np.issubdtype(
            np.dtype(self.pivots.dtype), np.integer
        ):
            raise ValueError("pivots must be a 1-D integer array of selected_rank entries.")

        data_dtypes = {str(a.dtype) for a in (self.P, self.Theta, self.grad_Theta)}
        if data_dtypes != {self.realized_dtype}:
            raise ValueError(
                f"P/Theta/grad_Theta dtype must equal realized_dtype={self.realized_dtype!r}, "
                f"got {data_dtypes}."
            )
        data_dtype = np.dtype(self.realized_dtype)
        if not np.issubdtype(data_dtype, np.inexact):
            raise ValueError("sector data must use a floating or complex dtype.")
        real_dtype = np.empty((), dtype=data_dtype).real.dtype
        if str(real_dtype) != self.grid.dtype:
            raise ValueError(
                f"sector real precision {real_dtype} does not match grid dtype {self.grid.dtype}."
            )
        if self.backend == "numpy":
            finite = all(np.all(np.isfinite(a)) for a in (self.P, self.Theta, self.grad_Theta))
        else:
            finite = all(bool(jnp.all(jnp.isfinite(a))) for a in (self.P, self.Theta, self.grad_Theta))
        if not finite:
            raise ValueError("P, Theta, and grad_Theta must be finite.")

        pivots_np = np.asarray(self.pivots, dtype=np.int64)
        if np.unique(pivots_np).size != selected_rank:
            raise ValueError("pivots must be unique.")
        if pivots_np.min() < 0 or pivots_np.max() >= n_grid:
            raise ValueError(f"pivots must lie in [0, {n_grid}).")
        recomputed_pivots_sha256 = _canonical_sha256(pivots_np)
        if self.pivots_sha256 != recomputed_pivots_sha256:
            raise ValueError("pivots_sha256 does not match the canonical pivot indices.")

        rank_record = _validate_pivot_provenance(
            {
                "requested_rank": self.requested_rank,
                "analytic_rank_bound": self.analytic_rank_bound,
                "n_rank_capped": self.n_rank_capped,
                "rank_exhausted": self.rank_exhausted,
                "numerical_rank": self.numerical_rank,
                "numerical_rank_lower_bound": self.numerical_rank_lower_bound,
                "n_pivots": selected_rank,
            },
            n_p=n_p, n_q=n_q, same_factor=self.same_factor,
            selected_rank=selected_rank,
        )
        for key, value in rank_record.items():
            if key != "n_pivots":
                object.__setattr__(self, key, value)

        grid_batch_size = _validate_positive_int("grid_batch_size", self.grid_batch_size)
        if grid_batch_size > n_grid:
            raise ValueError("grid_batch_size must be the realized value and cannot exceed n_grid.")
        object.__setattr__(self, "grid_batch_size", grid_batch_size)
        if (
            isinstance(self.rcond, bool)
            or not isinstance(self.rcond, (int, float, np.number))
            or not math.isfinite(self.rcond)
            or self.rcond <= 0.0
        ):
            raise ValueError(f"rcond must be finite and positive, got {self.rcond!r}.")
        jitter_used = _validate_nonnegative_finite("jitter_used", self.jitter_used)
        object.__setattr__(self, "jitter_used", jitter_used)
        n_tries = _validate_positive_int("n_tries", self.n_tries)
        object.__setattr__(self, "n_tries", n_tries)
        if self.solver_version != _IBP_INTERPOLATION_SOLVER_VERSION:
            raise ValueError(
                f"solver_version={self.solver_version!r} != "
                f"{_IBP_INTERPOLATION_SOLVER_VERSION!r}."
            )

        expected_identity_source = (
            "computed_content_hash" if self.backend == "numpy" else "caller_attested"
        )
        if self.factor_identity_source != expected_identity_source:
            raise ValueError(
                f"factor_identity_source must be {expected_identity_source!r} for "
                f"backend={self.backend!r}."
            )
        for name in _IBP_FACTOR_IDENTITY_KEYS + ("pivots_sha256",):
            _validate_sha256_hex(name, getattr(self, name))
        if self.same_factor and (
            self.factor_p_sha256 != self.factor_q_sha256
            or self.gradient_p_sha256 != self.gradient_q_sha256
        ):
            raise ValueError("same_factor=True requires identical factor and gradient identities.")

        expected_output_source = (
            "computed_content_hash"
            if self.backend == "numpy"
            else "derived_from_attested_inputs_unverified"
        )
        if self.output_identity_source != expected_output_source:
            raise ValueError(
                f"output_identity_source must be {expected_output_source!r} for "
                f"backend={self.backend!r}."
            )
        if self.backend == "numpy":
            output_hashes = {
                "p_sha256": _canonical_sha256(self.P),
                "theta_sha256": _canonical_sha256(self.Theta),
                "grad_theta_sha256": _canonical_sha256(self.grad_Theta),
            }
            for name, recomputed in output_hashes.items():
                _validate_sha256_hex(name, getattr(self, name))
                if getattr(self, name) != recomputed:
                    raise ValueError(f"{name} does not match the realized NumPy output.")
        elif any(
            value is not None
            for value in (self.p_sha256, self.theta_sha256, self.grad_theta_sha256)
        ):
            raise ValueError(
                "JAX output content hashes must be None: the builder does not copy large "
                "derived arrays to the host merely to hash them; their trust boundary is "
                "recorded by output_identity_source."
            )

        object.__setattr__(
            self, "provenance", _deep_freeze(dict(self.provenance) if self.provenance else {})
        )
        recomputed_spec = _canonical_spec_sha256({
            "grid_spec_sha256": self.grid.grid_spec_sha256,
            "n_orbital_p": n_p, "n_orbital_q": n_q, "n_pair": expected_n_pair,
            "n_grid": n_grid, "same_factor": self.same_factor,
            "pair_layout": self.pair_layout,
            "requested_rank": self.requested_rank,
            "analytic_rank_bound": self.analytic_rank_bound,
            "n_rank_capped": self.n_rank_capped,
            "selected_rank": selected_rank,
            "rank_exhausted": self.rank_exhausted,
            "numerical_rank": self.numerical_rank,
            "numerical_rank_lower_bound": self.numerical_rank_lower_bound,
            "storage_mode": self.storage_mode, "backend": self.backend,
            "device": self.device, "realized_dtype": self.realized_dtype,
            "grid_batch_size": grid_batch_size, "rcond": float(self.rcond),
            "jitter_used": jitter_used, "n_tries": n_tries,
            "factor_identity_source": self.factor_identity_source,
            "factor_p_sha256": self.factor_p_sha256,
            "factor_q_sha256": self.factor_q_sha256,
            "gradient_p_sha256": self.gradient_p_sha256,
            "gradient_q_sha256": self.gradient_q_sha256,
            "pivots_sha256": self.pivots_sha256,
            "output_identity_source": self.output_identity_source,
            "p_sha256": self.p_sha256, "theta_sha256": self.theta_sha256,
            "grad_theta_sha256": self.grad_theta_sha256,
            "solver_version": self.solver_version,
            "provenance": self.provenance,
        })
        if recomputed_spec != self.sector_spec_sha256:
            raise ValueError(
                "sector_spec_sha256 does not match the canonical digest recomputed from "
                "this artifact's own declared fields."
            )


def build_ibp_interpolation_sector(
    factor_p_raw,
    factor_q_raw,
    gradient_p_raw,
    gradient_q_raw,
    pivots,
    grid,
    *,
    pivot_provenance,
    same_factor=False,
    grid_batch_size=None,
    rcond=1e-14,
    upstream_provenance=None,
):
    """Build raw fixed-pivot pair interpolation vectors and derivatives.

    ``pivot_provenance`` must be the exact closed record returned by the
    canonical higher-layer selector's ``return_provenance=True`` mode. The
    builder validates it but deliberately does not select or reorder pivots:
    the undifferentiated selection is a separate step, and gradients must
    never trigger a second selection.
    """
    if not isinstance(grid, IBPGrid):
        raise TypeError(f"grid must be an IBPGrid, got {type(grid).__name__}.")
    if not isinstance(same_factor, bool):
        raise TypeError("same_factor must be bool.")

    inputs = (factor_p_raw, factor_q_raw, gradient_p_raw, gradient_q_raw)
    all_numpy = all(isinstance(a, np.ndarray) for a in inputs)
    all_jax = all(isinstance(a, jax.Array) for a in inputs)
    if not (all_numpy or all_jax):
        raise ValueError(
            "factor and gradient inputs must all be numpy.ndarray or all be jax.Array."
        )
    backend = "jax" if all_jax else "numpy"
    if backend != grid.backend:
        raise ValueError("factor/gradient backend must match the bound IBPGrid backend.")
    xp = jnp if backend == "jax" else np
    factor_p_raw, factor_q_raw, gradient_p_raw, gradient_q_raw = (
        xp.asarray(a) for a in inputs
    )
    if factor_p_raw.ndim != 2 or factor_q_raw.ndim != 2:
        raise ValueError("factor_p_raw/factor_q_raw must be 2-D (n_orbital,n_grid).")
    if gradient_p_raw.shape != (3,) + factor_p_raw.shape:
        raise ValueError("gradient_p_raw must have shape (3,) + factor_p_raw.shape.")
    if gradient_q_raw.shape != (3,) + factor_q_raw.shape:
        raise ValueError("gradient_q_raw must have shape (3,) + factor_q_raw.shape.")
    if factor_p_raw.shape[0] == 0 or factor_q_raw.shape[0] == 0:
        raise ValueError("factor arrays must contain at least one orbital.")
    if factor_p_raw.shape[1] != grid.n_grid or factor_q_raw.shape[1] != grid.n_grid:
        raise ValueError(
            f"factor grid axes must match grid.n_grid={grid.n_grid}, got "
            f"{factor_p_raw.shape[1]}/{factor_q_raw.shape[1]}."
        )
    dtypes = {str(a.dtype) for a in (factor_p_raw, factor_q_raw, gradient_p_raw, gradient_q_raw)}
    if len(dtypes) != 1:
        raise ValueError(f"all factors and gradients must share one dtype, got {dtypes}.")
    realized_dtype = dtypes.pop()
    dtype = np.dtype(realized_dtype)
    if not np.issubdtype(dtype, np.inexact):
        raise ValueError("factor and gradient inputs must use a floating or complex dtype.")
    if dtype in (np.dtype(np.float64), np.dtype(np.complex128)) and not jax.config.jax_enable_x64:
        raise ValueError(
            f"Requested sector dtype {dtype} requires jax_enable_x64 because the shared "
            "prepared normal-equation solver is JAX-backed. Call "
            "jax.config.update('jax_enable_x64', True) before building, or supply explicit "
            "float32/complex64 inputs; silent solver downcasting is not allowed."
        )
    if str(np.empty((), dtype=dtype).real.dtype) != grid.dtype:
        raise ValueError(
            f"factor real precision does not match bound grid dtype {grid.dtype!r}."
        )
    if backend == "numpy":
        finite = all(np.all(np.isfinite(a)) for a in inputs)
        device = "cpu"
    else:
        finite = all(bool(jnp.all(jnp.isfinite(a))) for a in inputs)
        devices = {str(a.device) for a in inputs}
        if devices != {grid.device}:
            raise ValueError(
                f"all JAX factors/gradients must share grid device {grid.device!r}, got {devices}."
            )
        device = grid.device
    if not finite:
        raise ValueError("factor and gradient inputs must be finite.")
    if same_factor:
        array_equal = jnp.array_equal if backend == "jax" else np.array_equal
        if not bool(array_equal(factor_p_raw, factor_q_raw)):
            raise ValueError("same_factor=True requires factor_p_raw == factor_q_raw.")
        if not bool(array_equal(gradient_p_raw, gradient_q_raw)):
            raise ValueError("same_factor=True requires gradient_p_raw == gradient_q_raw.")

    pivots_np_raw = np.asarray(pivots)
    if pivots_np_raw.ndim != 1 or not np.issubdtype(pivots_np_raw.dtype, np.integer):
        raise ValueError("pivots must be a 1-D integer array.")
    pivots_np = np.asarray(pivots_np_raw, dtype=np.int64)
    selected_rank = int(pivots_np.size)
    if selected_rank == 0 or np.unique(pivots_np).size != selected_rank:
        raise ValueError("pivots must be nonempty and unique.")
    if pivots_np.min() < 0 or pivots_np.max() >= grid.n_grid:
        raise ValueError(f"pivots must lie in [0, {grid.n_grid}).")
    rank_record = _validate_pivot_provenance(
        pivot_provenance,
        n_p=factor_p_raw.shape[0], n_q=factor_q_raw.shape[0],
        same_factor=same_factor, selected_rank=selected_rank,
    )
    if backend == "jax":
        pivot_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
        pivots_backend = jnp.asarray(pivots_np, dtype=pivot_dtype)
    else:
        pivots_backend = pivots_np

    batch = (
        _validate_positive_int("grid_batch_size", grid_batch_size)
        if grid_batch_size is not None else grid.n_grid
    )
    batch = min(batch, grid.n_grid)
    if isinstance(rcond, bool) or not isinstance(rcond, (int, float, np.number)):
        raise ValueError(f"rcond must be finite and positive, got {rcond!r}.")
    rcond = float(rcond)
    if not math.isfinite(rcond) or rcond <= 0.0:
        raise ValueError(f"rcond must be finite and positive, got {rcond!r}.")

    factor_p_piv = factor_p_raw[:, pivots_backend]
    factor_q_piv = factor_q_raw[:, pivots_backend]
    chol, lower, jitter_used, n_tries = prepare_normal_equations_solver(
        factor_p_piv, factor_q_piv, rcond=rcond, return_info=True
    )

    theta_chunks = []
    gradient_chunks = [[] for _ in range(3)]
    for start in range(0, grid.n_grid, batch):
        stop = min(start + batch, grid.n_grid)
        fp = factor_p_raw[:, start:stop]
        fq = factor_q_raw[:, start:stop]
        theta_batch = solve_normal_equations_batch_prepared(
            chol, lower, factor_p_piv, factor_q_piv, fp, fq
        )
        if backend == "numpy":
            theta_batch = np.asarray(theta_batch)
        theta_chunks.append(theta_batch)
        for axis in range(3):
            left = solve_normal_equations_batch_prepared(
                chol, lower, factor_p_piv, factor_q_piv,
                gradient_p_raw[axis, :, start:stop], fq,
            )
            right = solve_normal_equations_batch_prepared(
                chol, lower, factor_p_piv, factor_q_piv,
                fp, gradient_q_raw[axis, :, start:stop],
            )
            value = left + right
            if backend == "numpy":
                value = np.asarray(value)
            gradient_chunks[axis].append(value)
    concatenate = jnp.concatenate if backend == "jax" else np.concatenate
    stack = jnp.stack if backend == "jax" else np.stack
    Theta = concatenate(theta_chunks, axis=1)
    grad_Theta = stack(
        [concatenate(component, axis=1) for component in gradient_chunks], axis=1
    )

    if same_factor:
        pair_p_np, pair_q_np = np.tril_indices(factor_p_raw.shape[0])
        if backend == "jax":
            pair_p = jnp.asarray(pair_p_np, dtype=pivots_backend.dtype)
            pair_q = jnp.asarray(pair_q_np, dtype=pivots_backend.dtype)
        else:
            pair_p, pair_q = pair_p_np, pair_q_np
        P = (
            factor_p_piv[pair_p, :] * factor_q_piv[pair_q, :]
        ).T
        pair_layout = "packed_lower"
    else:
        P = xp.einsum("pu,qu->upq", factor_p_piv, factor_q_piv).reshape(
            selected_rank, factor_p_raw.shape[0] * factor_q_raw.shape[0]
        )
        pair_layout = "full_row_major"
    if backend == "numpy":
        P = np.asarray(P)

    upstream = dict(upstream_provenance) if upstream_provenance else {}
    if backend == "numpy":
        reserved = sorted(set(upstream).intersection(_IBP_FACTOR_IDENTITY_KEYS))
        if reserved:
            raise ValueError(
                f"NumPy identities are computed from content; remove reserved upstream keys {reserved}."
            )
        identities = {
            "factor_p_sha256": _canonical_sha256(factor_p_raw),
            "factor_q_sha256": _canonical_sha256(factor_q_raw),
            "gradient_p_sha256": _canonical_sha256(gradient_p_raw),
            "gradient_q_sha256": _canonical_sha256(gradient_q_raw),
        }
        identity_source = "computed_content_hash"
    else:
        missing = [key for key in _IBP_FACTOR_IDENTITY_KEYS if key not in upstream]
        if missing:
            raise ValueError(
                "JAX sectors require caller-attested input identities in upstream_provenance; "
                f"missing {missing}."
            )
        identities = {key: upstream.pop(key) for key in _IBP_FACTOR_IDENTITY_KEYS}
        for key, value in identities.items():
            _validate_sha256_hex(f"upstream_provenance[{key!r}]", value)
        identity_source = "caller_attested"
    if same_factor and (
        identities["factor_p_sha256"] != identities["factor_q_sha256"]
        or identities["gradient_p_sha256"] != identities["gradient_q_sha256"]
    ):
        raise ValueError("same_factor=True requires identical factor and gradient identities.")

    provenance = _deep_freeze({
        "pivot_provenance": rank_record,
        "upstream_provenance": upstream,
    })
    pivots_sha256 = _canonical_sha256(pivots_np)
    n_pair = (
        factor_p_raw.shape[0] * (factor_p_raw.shape[0] + 1) // 2
        if same_factor else factor_p_raw.shape[0] * factor_q_raw.shape[0]
    )
    if backend == "numpy":
        output_identity_source = "computed_content_hash"
        p_sha256 = _canonical_sha256(P)
        theta_sha256 = _canonical_sha256(Theta)
        grad_theta_sha256 = _canonical_sha256(grad_Theta)
    else:
        output_identity_source = "derived_from_attested_inputs_unverified"
        p_sha256 = theta_sha256 = grad_theta_sha256 = None
    spec_fields = {
        "grid_spec_sha256": grid.grid_spec_sha256,
        "n_orbital_p": int(factor_p_raw.shape[0]),
        "n_orbital_q": int(factor_q_raw.shape[0]),
        "n_pair": int(n_pair), "n_grid": int(grid.n_grid),
        "same_factor": same_factor, "pair_layout": pair_layout,
        "requested_rank": rank_record["requested_rank"],
        "analytic_rank_bound": rank_record["analytic_rank_bound"],
        "n_rank_capped": rank_record["n_rank_capped"],
        "selected_rank": selected_rank,
        "rank_exhausted": rank_record["rank_exhausted"],
        "numerical_rank": rank_record["numerical_rank"],
        "numerical_rank_lower_bound": rank_record["numerical_rank_lower_bound"],
        "storage_mode": "incore_full_theta_gradient", "backend": backend,
        "device": device, "realized_dtype": str(Theta.dtype),
        "grid_batch_size": batch, "rcond": rcond,
        "jitter_used": float(jitter_used), "n_tries": int(n_tries),
        "factor_identity_source": identity_source,
        **identities, "pivots_sha256": pivots_sha256,
        "output_identity_source": output_identity_source,
        "p_sha256": p_sha256, "theta_sha256": theta_sha256,
        "grad_theta_sha256": grad_theta_sha256,
        "solver_version": _IBP_INTERPOLATION_SOLVER_VERSION,
        "provenance": provenance,
    }
    sector_spec_sha256 = _canonical_spec_sha256(spec_fields)
    return IBPInterpolationSector(
        P=P, Theta=Theta, grad_Theta=grad_Theta, pivots=pivots_backend, grid=grid,
        n_orbital_p=int(factor_p_raw.shape[0]),
        n_orbital_q=int(factor_q_raw.shape[0]), n_pair=int(n_pair),
        n_grid=int(grid.n_grid), same_factor=same_factor, pair_layout=pair_layout,
        requested_rank=rank_record["requested_rank"],
        analytic_rank_bound=rank_record["analytic_rank_bound"],
        n_rank_capped=rank_record["n_rank_capped"], selected_rank=selected_rank,
        rank_exhausted=rank_record["rank_exhausted"],
        numerical_rank=rank_record["numerical_rank"],
        numerical_rank_lower_bound=rank_record["numerical_rank_lower_bound"],
        storage_mode="incore_full_theta_gradient", backend=backend, device=device,
        realized_dtype=str(Theta.dtype), grid_batch_size=batch, rcond=rcond,
        jitter_used=float(jitter_used), n_tries=int(n_tries),
        factor_identity_source=identity_source,
        factor_p_sha256=identities["factor_p_sha256"],
        factor_q_sha256=identities["factor_q_sha256"],
        gradient_p_sha256=identities["gradient_p_sha256"],
        gradient_q_sha256=identities["gradient_q_sha256"],
        pivots_sha256=pivots_sha256,
        output_identity_source=output_identity_source,
        p_sha256=p_sha256, theta_sha256=theta_sha256,
        grad_theta_sha256=grad_theta_sha256,
        solver_version=_IBP_INTERPOLATION_SOLVER_VERSION,
        provenance=provenance, sector_spec_sha256=sector_spec_sha256,
    )


# ---------------------------------------------------------------------------
# Same/cross-sector Z cores (task #7).
# ---------------------------------------------------------------------------

_IBP_CORE_VERSION = "2"  # v2: JAX backend + deterministic spec (timing/memory excluded)
_SUPPORTED_IBP_CORE_OUTPUT_IDENTITY_SOURCES = (
    "computed_content_hash", "derived_from_attested_inputs_unverified",
)
_SUPPORTED_IBP_CORE_SYMMETRY_MODES = ("two_sided_average", "one_sided")
_SUPPORTED_PEAK_HOST_BYTES_STATUS = ("unmeasured_cpu_oracle", "unmeasured_jax_device")
_DEFAULT_PSD_RTOL = 1e-10
_PSD_HERMITICITY_TOL = 1e-10


def _residual_from_norms(numerator, denom):
    """Shared 0/0->0, nonzero/0->raise, finite/non-negative logic for a
    normalized residual given already-computed (host scalar) norms."""
    numerator = float(numerator)
    denom = float(denom)
    if denom == 0.0:
        if numerator == 0.0:
            return 0.0
        raise ValueError(
            "normalized Frobenius residual is undefined: nonzero difference against a "
            "zero-norm reference."
        )
    result = numerator / denom
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"normalized Frobenius residual is not finite and non-negative: {result!r}.")
    return result


def _normalized_frobenius_residual(a, b):
    """||a - b||_F / ||a||_F. Only 0/0 maps to 0 (an exact all-zero reference);
    a nonzero difference against a zero-norm reference is undefined and raises
    rather than being silently reported as exact."""
    a = np.asarray(a)
    b = np.asarray(b)
    return _residual_from_norms(np.linalg.norm(a - b), np.linalg.norm(a))


def _normalized_frobenius_residual_jax(a, b):
    """Device ||a-b||_F / ||a||_F: norms reduce on device and only the two
    rank-0 scalars cross to the host (through _ibp_sync_scalars) before the
    shared 0/0 / nonzero-0 / finite logic; no large array is transferred."""
    s = _ibp_sync_scalars(num=jnp.linalg.norm(a - b), den=jnp.linalg.norm(a))
    return _residual_from_norms(s["num"], s["den"])


def _packed_pair_residual_rank_space_jax(z_forward, p):
    """Same-sector packed-pair Hermiticity residual computed in rank space,
    never forming P^dagger z P (n_pair x n_pair). With G = P P^dagger,
    ||P^dagger A P||_F^2 = Re tr(A^dagger G A G); numerator uses A = z - z^H,
    denominator A = z. Only the two rank-0 norms cross to the host."""
    g = p @ p.conj().T                                  # (rank, rank)
    diff = z_forward - z_forward.conj().T
    num_sq = _frob_sq_in_rank_space(diff, g)
    den_sq = _frob_sq_in_rank_space(z_forward, g)
    s = _ibp_sync_scalars(num_sq=jnp.maximum(num_sq, 0.0), den_sq=jnp.maximum(den_sq, 0.0))
    return _residual_from_norms(math.sqrt(s["num_sq"]), math.sqrt(s["den_sq"]))


class PSDFactorization(typing.NamedTuple):
    """Result of psd_factorize: the compact Hermitian PSD factor plus the
    mandatory diagnostics. Immutable; task #8 (AO-pair provider) reuses this
    same helper on Z_AO rather than re-deriving the gate."""
    factor: object                    # W, shape (n, retained_rank), read-only
    factor_sha256: str
    rtol: float
    raw_min_eigenvalue: float
    spectral_scale: float
    negative_mode_count: int
    clipped_mode_count: int
    clipped_absolute_weight: float
    retained_rank: int
    reconstruction_residual: float


def psd_factorize(z_hermitian, *, rtol=_DEFAULT_PSD_RTOL):
    """Factor a Hermitian core Z_H ~= W W^dagger within a roundoff band.

    Diagonalizes Z_H, HARD-FAILS if any eigenvalue is below
    ``-rtol * max|eig|`` (a material negative mode is a discretization
    failure and must never be silently repaired), clips only the negative
    modes that lie inside the roundoff band to zero, and forms the COMPACT
    factor ``W = U[:, positive] sqrt(lambda_clipped[positive])`` with no
    zero columns (so W.shape[1] is the retained rank task #8 streams).

    Assumes Z_H is Hermitian (numpy.linalg.eigh reads the lower triangle);
    the caller passes the two-sided-averaged same-sector core, which is
    Hermitian by construction. Zero Z_H yields scale/tolerance/residual 0,
    retained rank 0, and W.shape == (n, 0)."""
    Z = np.asarray(z_hermitian)
    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError(f"psd_factorize requires a square 2-D array, got shape {Z.shape}.")
    if not np.all(np.isfinite(Z)):
        raise ValueError("psd_factorize: Z_H has non-finite entries.")
    if isinstance(rtol, bool) or not isinstance(rtol, (int, float, np.integer, np.floating)):
        raise ValueError(f"psd_factorize: rtol must be a real, non-bool number, got {rtol!r}.")
    rtol = float(rtol)
    if not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"psd_factorize: rtol must be a finite non-negative float, got {rtol!r}.")

    # Hermiticity gate: eigh reads only one triangle, so a non-Hermitian input
    # would be silently "factorized" against its Hermitian completion. An exact
    # two-sided-averaged core is Hermitian to roundoff.
    z_norm = float(np.linalg.norm(Z))
    if z_norm > 0.0:
        hermiticity_residual = float(np.linalg.norm(Z - Z.conj().T)) / z_norm
        if hermiticity_residual > _PSD_HERMITICITY_TOL:
            raise ValueError(
                f"psd_factorize: input is not Hermitian (residual {hermiticity_residual!r} > "
                f"{_PSD_HERMITICITY_TOL!r}); PSD factorization requires a Hermitian core."
            )

    eigvals, eigvecs = np.linalg.eigh(Z)  # ascending real eigenvalues
    spectral_scale = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    raw_min_eigenvalue = float(eigvals[0]) if eigvals.size else 0.0
    threshold = -rtol * spectral_scale

    if np.any(eigvals < threshold):
        offending = float(eigvals[eigvals < threshold].min())
        raise ValueError(
            f"psd_factorize: material negative eigenvalue {offending!r} is below the "
            f"roundoff band threshold {threshold!r} (rtol={rtol!r}, spectral_scale="
            f"{spectral_scale!r}); a materially indefinite core is a hard compatibility "
            f"failure and is never repaired by clipping."
        )

    negative = eigvals < 0.0
    negative_mode_count = int(np.count_nonzero(negative))
    clipped_absolute_weight = (
        float(np.sum(np.abs(eigvals[negative]))) if negative_mode_count else 0.0
    )
    lambda_clipped = np.where(negative, 0.0, eigvals)
    positive = lambda_clipped > 0.0
    retained_rank = int(np.count_nonzero(positive))
    W = eigvecs[:, positive] * np.sqrt(lambda_clipped[positive])[None, :]
    W = _readonly_copy(W)

    reconstruction = W @ W.conj().T
    reconstruction_residual = _normalized_frobenius_residual(Z, reconstruction)

    return PSDFactorization(
        factor=W,
        factor_sha256=_canonical_sha256(W),
        rtol=rtol,
        raw_min_eigenvalue=raw_min_eigenvalue,
        spectral_scale=spectral_scale,
        negative_mode_count=negative_mode_count,
        clipped_mode_count=negative_mode_count,  # every negative here is within-band
        clipped_absolute_weight=clipped_absolute_weight,
        retained_rank=retained_rank,
        reconstruction_residual=reconstruction_residual,
    )


@dataclasses.dataclass(frozen=True)
class IBPCoreArtifact:
    """A completed Z core for one sector (same-sector) or one sector pair
    (cross-sector), built via ibp_core using this module's own kernel()
    primitive (task #2) applied to sector Theta/grad_Theta arrays instead
    of raw AO factors -- Z_AB[mu,nu] = -1/2 sum_g w_g grad(Theta_A[mu,g])^*
    . sum_h w_h rhat(r_g-r_h) Theta_B[nu,h] is exactly kernel()'s formula.

    The blocked real-space IBP kernel is NOT manifestly Hermitian the way
    the free-space Poisson kernel is (that one is an exact FFT convolution;
    this one is a one-sided discrete quadrature evaluated with an explicit
    coincident-point convention) -- so, per the architecture's mathematical
    contract, the raw one-sided orientation(s) and their dagger residual
    are ALWAYS retained regardless of symmetry_mode, so quadrature bias can
    never be silently hidden behind an averaged production Z:
    raw_dagger_residual is the NORMALIZED Frobenius residual
    ||z_forward - dagger||_F / ||z_forward||_F (0/0 -> 0):
    - same-sector: z_forward_one_sided is the sole raw computation;
      z_reverse_one_sided is None (the "reverse" orientation is just its
      conjugate transpose, not a second kernel() evaluation); the dagger is
      z_forward_one_sided^dagger. Same-sector cores also record
      raw_packed_pair_metric_dagger_residual, the normalized Frobenius
      departure from Hermiticity of the packed AO-pair metric
      P^dagger z_forward P (the quantity the AO-pair gate is stated against).
    - cross-sector: BOTH orientations are evaluated (z_forward_one_sided =
      Z_AB, z_reverse_one_sided = Z_BA); the dagger is
      z_reverse_one_sided^dagger; raw_packed_pair_metric_dagger_residual is
      None (a cross core is not a square Hermitian pair metric).

    symmetry_mode="two_sided_average" (production default) sets
    Z = (z_forward_one_sided + dagger(reverse-or-self))/2.
    symmetry_mode="one_sided" (diagnostic only) sets Z = z_forward_one_sided
    unaveraged.

    Backend scope: only backend="numpy" is implemented here. kernel() is a
    NumPy-only primitive (task #2) -- routing a JAX-backend sector/operator
    through it would silently force a host transfer via np.asarray(),
    exactly the hazard IBPGrid/IBPOperatorPlan's own JAX paths were built to
    avoid. A tiled, device-resident JAX execution path is task #10's
    explicit scope, not this one; ibp_core rejects backend="jax" plans
    with a clear NotImplementedError rather than silently doing the wrong
    thing.
    """
    Z: object
    same_sector: bool
    symmetry_mode: str
    z_forward_one_sided: object
    z_reverse_one_sided: object
    z_sha256: str
    z_forward_sha256: str
    z_reverse_sha256: object
    # "computed_content_hash" (numpy: z_*/psd hashes are verified device-free
    # content hashes) or "derived_from_attested_inputs_unverified" (jax: those
    # hashes are None; identity is inherited from attested sector/operator
    # inputs, never a device content hash).
    output_identity_source: str
    raw_dagger_residual: float
    raw_packed_pair_metric_dagger_residual: object
    coincident_pairs: int
    n_mu: int
    n_nu: int
    # PSD factorization (task #7): populated only for a same-sector,
    # two-sided-averaged (Hermitian) core; psd_status="not_applicable" with
    # every psd_* field None for cross-sector or one-sided artifacts.
    psd_status: str
    psd_rtol: object
    psd_factor: object
    psd_factor_sha256: object
    psd_raw_min_eigenvalue: object
    psd_spectral_scale: object
    psd_negative_mode_count: object
    psd_clipped_mode_count: object
    psd_clipped_absolute_weight: object
    psd_retained_rank: object
    psd_reconstruction_residual: object
    left_sector_spec_sha256: str
    right_sector_spec_sha256: str
    operator_spec_sha256: str
    mu_block_size: int
    nu_block_size: int
    backend: str
    device: str
    realized_dtype: str
    build_wall_time_seconds: float
    peak_host_bytes: object
    peak_host_bytes_status: str
    solver_version: str
    provenance: object
    core_spec_sha256: str

    def __post_init__(self):
        if self.backend not in _SUPPORTED_IBP_BACKENDS:
            raise ValueError(
                f"Unsupported backend={self.backend!r} -- must be one of "
                f"{_SUPPORTED_IBP_BACKENDS}."
            )
        if self.output_identity_source not in _SUPPORTED_IBP_CORE_OUTPUT_IDENTITY_SOURCES:
            raise ValueError(
                f"Unsupported output_identity_source={self.output_identity_source!r}, must "
                f"be one of {_SUPPORTED_IBP_CORE_OUTPUT_IDENTITY_SOURCES}."
            )
        if self.backend == "jax":
            self._post_init_jax()
            return
        # ---- NumPy path (numerically and structurally unchanged) ----
        if self.output_identity_source != "computed_content_hash":
            raise ValueError(
                "backend='numpy' requires output_identity_source='computed_content_hash'."
            )
        if self.device != "cpu":
            raise ValueError(f"device must be 'cpu' for backend='numpy', got {self.device!r}.")
        if not isinstance(self.same_sector, bool):
            raise TypeError("same_sector must be bool.")
        if self.symmetry_mode not in _SUPPORTED_IBP_CORE_SYMMETRY_MODES:
            raise ValueError(
                f"Unsupported symmetry_mode={self.symmetry_mode!r}, must be one of "
                f"{_SUPPORTED_IBP_CORE_SYMMETRY_MODES}."
            )

        n_mu = _validate_positive_int("n_mu", self.n_mu)
        n_nu = _validate_positive_int("n_nu", self.n_nu)

        if not isinstance(self.Z, np.ndarray):
            raise TypeError("Z must be a numpy.ndarray for backend='numpy'.")
        if not isinstance(self.z_forward_one_sided, np.ndarray):
            raise TypeError("z_forward_one_sided must be a numpy.ndarray.")
        object.__setattr__(self, "Z", _readonly_copy(self.Z))
        object.__setattr__(self, "z_forward_one_sided", _readonly_copy(self.z_forward_one_sided))
        if self.Z.shape != (n_mu, n_nu):
            raise ValueError(f"Z.shape must be {(n_mu, n_nu)}, got {self.Z.shape}.")
        if self.z_forward_one_sided.shape != (n_mu, n_nu):
            raise ValueError(
                f"z_forward_one_sided.shape must be {(n_mu, n_nu)}, got "
                f"{self.z_forward_one_sided.shape}."
            )

        if self.same_sector:
            if self.z_reverse_one_sided is not None:
                raise ValueError("z_reverse_one_sided must be None for same_sector=True.")
            if self.left_sector_spec_sha256 != self.right_sector_spec_sha256:
                raise ValueError(
                    "same_sector=True requires left_sector_spec_sha256 == "
                    "right_sector_spec_sha256."
                )
            if n_mu != n_nu:
                raise ValueError("same_sector=True requires n_mu == n_nu.")
            dagger = self.z_forward_one_sided.conj().T
        else:
            if not isinstance(self.z_reverse_one_sided, np.ndarray):
                raise TypeError(
                    "z_reverse_one_sided must be a numpy.ndarray for same_sector=False."
                )
            object.__setattr__(
                self, "z_reverse_one_sided", _readonly_copy(self.z_reverse_one_sided)
            )
            if self.z_reverse_one_sided.shape != (n_nu, n_mu):
                raise ValueError(
                    f"z_reverse_one_sided.shape must be {(n_nu, n_mu)}, got "
                    f"{self.z_reverse_one_sided.shape}."
                )
            dagger = self.z_reverse_one_sided.conj().T

        if not np.all(np.isfinite(self.z_forward_one_sided)):
            raise ValueError("z_forward_one_sided has non-finite entries.")
        if self.z_reverse_one_sided is not None and not np.all(
            np.isfinite(self.z_reverse_one_sided)
        ):
            raise ValueError("z_reverse_one_sided has non-finite entries.")
        if not np.all(np.isfinite(self.Z)):
            raise ValueError("Z has non-finite entries.")

        recomputed_residual = _normalized_frobenius_residual(self.z_forward_one_sided, dagger)
        if not math.isclose(self.raw_dagger_residual, recomputed_residual,
                            rel_tol=1e-9, abs_tol=1e-14):
            raise ValueError(
                f"raw_dagger_residual={self.raw_dagger_residual!r} does not match the "
                f"normalized-Frobenius value recomputed from the artifact's own raw "
                f"orientation(s) ({recomputed_residual!r})."
            )

        expected_Z = (
            self.z_forward_one_sided if self.symmetry_mode == "one_sided"
            else (self.z_forward_one_sided + dagger) / 2
        )
        if not np.array_equal(self.Z, expected_Z):
            raise ValueError(
                "Z does not match the value recomputed from this artifact's own raw "
                "orientation(s) and symmetry_mode."
            )

        # Bind array CONTENT into the spec: recomputing Z from the raw
        # orientation(s) keeps a coherent multi-array sign flip self-consistent,
        # so a scalar-only digest cannot detect it -- only hashing the bytes can.
        for name in ("z_sha256", "z_forward_sha256"):
            _validate_sha256_hex(name, getattr(self, name))
        recomputed_z_sha = _canonical_sha256(self.Z)
        if self.z_sha256 != recomputed_z_sha:
            raise ValueError("z_sha256 does not match the digest recomputed from Z's bytes.")
        recomputed_zf_sha = _canonical_sha256(self.z_forward_one_sided)
        if self.z_forward_sha256 != recomputed_zf_sha:
            raise ValueError(
                "z_forward_sha256 does not match the digest recomputed from "
                "z_forward_one_sided's bytes."
            )
        if self.same_sector:
            if self.z_reverse_sha256 is not None:
                raise ValueError("z_reverse_sha256 must be None for same_sector=True.")
        else:
            _validate_sha256_hex("z_reverse_sha256", self.z_reverse_sha256)
            recomputed_zr_sha = _canonical_sha256(self.z_reverse_one_sided)
            if self.z_reverse_sha256 != recomputed_zr_sha:
                raise ValueError(
                    "z_reverse_sha256 does not match the digest recomputed from "
                    "z_reverse_one_sided's bytes."
                )

        # Production-facing packed-pair metric dagger residual: same-sector
        # only (cross cores are not square/Hermitian in pair space); recorded,
        # not recomputed here, because the sector P is not carried on the core.
        if self.same_sector:
            if self.raw_packed_pair_metric_dagger_residual is None:
                raise ValueError(
                    "raw_packed_pair_metric_dagger_residual is required for same_sector=True."
                )
            _validate_nonnegative_finite(
                "raw_packed_pair_metric_dagger_residual",
                self.raw_packed_pair_metric_dagger_residual,
            )
        elif self.raw_packed_pair_metric_dagger_residual is not None:
            raise ValueError(
                "raw_packed_pair_metric_dagger_residual must be None for cross-sector cores."
            )

        self._validate_psd_fields()

        if isinstance(self.coincident_pairs, bool) or not isinstance(self.coincident_pairs, int):
            raise ValueError(f"coincident_pairs must be a non-negative int, got {self.coincident_pairs!r}.")
        if self.coincident_pairs < 0:
            raise ValueError(f"coincident_pairs must be non-negative, got {self.coincident_pairs!r}.")

        data_dtype = np.dtype(self.realized_dtype)
        if not np.issubdtype(data_dtype, np.inexact):
            raise ValueError("core data must use a floating or complex dtype.")
        if str(self.Z.dtype) != self.realized_dtype:
            raise ValueError(
                f"realized_dtype={self.realized_dtype!r} does not match Z.dtype "
                f"{self.Z.dtype}."
            )

        mu_block_size = _validate_positive_int("mu_block_size", self.mu_block_size)
        nu_block_size = _validate_positive_int("nu_block_size", self.nu_block_size)
        object.__setattr__(self, "mu_block_size", mu_block_size)
        object.__setattr__(self, "nu_block_size", nu_block_size)

        for name in ("left_sector_spec_sha256", "right_sector_spec_sha256", "operator_spec_sha256"):
            _validate_sha256_hex(name, getattr(self, name))

        build_wall_time_seconds = _validate_nonnegative_finite(
            "build_wall_time_seconds", self.build_wall_time_seconds
        )
        object.__setattr__(self, "build_wall_time_seconds", build_wall_time_seconds)
        if self.peak_host_bytes is not None:
            if isinstance(self.peak_host_bytes, bool) or not isinstance(self.peak_host_bytes, int):
                raise ValueError(
                    f"peak_host_bytes must be None or a non-negative int, got "
                    f"{self.peak_host_bytes!r}."
                )
            if self.peak_host_bytes < 0:
                raise ValueError(
                    f"peak_host_bytes must be non-negative, got {self.peak_host_bytes!r}."
                )
        if self.peak_host_bytes_status not in _SUPPORTED_PEAK_HOST_BYTES_STATUS:
            raise ValueError(
                f"peak_host_bytes_status={self.peak_host_bytes_status!r} must be one of "
                f"{_SUPPORTED_PEAK_HOST_BYTES_STATUS}."
            )
        if self.peak_host_bytes_status == "unmeasured_cpu_oracle" and self.peak_host_bytes is not None:
            raise ValueError(
                "peak_host_bytes must be None when peak_host_bytes_status is "
                "'unmeasured_cpu_oracle'."
            )

        if self.solver_version != _IBP_CORE_VERSION:
            raise ValueError(f"solver_version={self.solver_version!r} != {_IBP_CORE_VERSION!r}.")

        object.__setattr__(
            self, "provenance", _deep_freeze(dict(self.provenance) if self.provenance else {})
        )

        recomputed_spec = _canonical_spec_sha256(self._core_spec_fields(n_mu, n_nu))
        if recomputed_spec != self.core_spec_sha256:
            raise ValueError(
                "core_spec_sha256 does not match the canonical digest recomputed from "
                "this artifact's own declared fields."
            )

    def _core_spec_fields(self, n_mu, n_nu):
        """The v2 deterministic identity fields -- excludes the nondeterministic
        build_wall_time_seconds / peak_host_bytes execution metadata so the same
        inputs always digest identically. The builder digests the identical
        field set (kept in lock-step; any drift fails the __post_init__ recompute
        immediately)."""
        return {
            "same_sector": self.same_sector, "symmetry_mode": self.symmetry_mode,
            "z_sha256": self.z_sha256, "z_forward_sha256": self.z_forward_sha256,
            "z_reverse_sha256": self.z_reverse_sha256,
            "output_identity_source": self.output_identity_source,
            "raw_dagger_residual": self.raw_dagger_residual,
            "raw_packed_pair_metric_dagger_residual": self.raw_packed_pair_metric_dagger_residual,
            "coincident_pairs": self.coincident_pairs, "n_mu": n_mu, "n_nu": n_nu,
            "psd_status": self.psd_status, "psd_rtol": self.psd_rtol,
            "psd_factor_sha256": self.psd_factor_sha256,
            "psd_raw_min_eigenvalue": self.psd_raw_min_eigenvalue,
            "psd_spectral_scale": self.psd_spectral_scale,
            "psd_negative_mode_count": self.psd_negative_mode_count,
            "psd_clipped_mode_count": self.psd_clipped_mode_count,
            "psd_clipped_absolute_weight": self.psd_clipped_absolute_weight,
            "psd_retained_rank": self.psd_retained_rank,
            "psd_reconstruction_residual": self.psd_reconstruction_residual,
            "left_sector_spec_sha256": self.left_sector_spec_sha256,
            "right_sector_spec_sha256": self.right_sector_spec_sha256,
            "operator_spec_sha256": self.operator_spec_sha256,
            "mu_block_size": self.mu_block_size, "nu_block_size": self.nu_block_size,
            "backend": self.backend, "device": self.device,
            "realized_dtype": self.realized_dtype,
            "solver_version": self.solver_version, "provenance": self.provenance,
        }

    def _validate_psd_fields(self):
        """Close the PSD factorization fields. For a factorized (same-sector,
        two-sided) core, RECOMPUTE the whole factorization from this
        artifact's own Z and require the stored factor/diagnostics to match
        exactly -- so a tampered Z (or tampered PSD field) is rejected, and a
        Z edited to be materially indefinite hard-fails here just as it would
        in the builder. For every other core, enforce the all-None invariant."""
        psd_scalar_fields = (
            "psd_rtol", "psd_factor", "psd_factor_sha256", "psd_raw_min_eigenvalue",
            "psd_spectral_scale", "psd_negative_mode_count", "psd_clipped_mode_count",
            "psd_clipped_absolute_weight", "psd_retained_rank", "psd_reconstruction_residual",
        )
        applicable = self.same_sector and self.symmetry_mode == "two_sided_average"
        if self.psd_status == "not_applicable":
            if applicable:
                raise ValueError(
                    "psd_status='not_applicable' is invalid for a same-sector, "
                    "two-sided-averaged core, which must be factorized."
                )
            for name in psd_scalar_fields:
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{name} must be None when psd_status='not_applicable'."
                    )
            return
        if self.psd_status != "factorized":
            raise ValueError(
                f"psd_status={self.psd_status!r} must be 'factorized' or 'not_applicable'."
            )
        if not applicable:
            raise ValueError(
                "psd_status='factorized' is only valid for a same-sector, "
                "two-sided-averaged core."
            )
        if not isinstance(self.psd_factor, np.ndarray):
            raise TypeError("psd_factor must be a numpy.ndarray for a factorized core.")
        object.__setattr__(self, "psd_factor", _readonly_copy(self.psd_factor))
        if not np.all(np.isfinite(self.psd_factor)):
            raise ValueError("psd_factor has non-finite entries.")
        _validate_sha256_hex("psd_factor_sha256", self.psd_factor_sha256)
        if (isinstance(self.psd_rtol, bool) or not isinstance(self.psd_rtol, float)
                or not math.isfinite(self.psd_rtol) or self.psd_rtol < 0.0):
            raise ValueError(
                f"psd_rtol must be a finite non-negative float for a factorized core, "
                f"got {self.psd_rtol!r}."
            )

        # Bind the STORED factor's own bytes -- a re-diagonalization gives a
        # gauge-equivalent but generally different W (eigenvector sign/phase and
        # degenerate-subspace freedom), so the factor is validated by its own
        # content hash and by whether it actually reconstructs Z, while the
        # gauge-INVARIANT eigenvalue diagnostics are recomputed from Z.
        if _canonical_sha256(self.psd_factor) != self.psd_factor_sha256:
            raise ValueError(
                "psd_factor_sha256 does not match the stored factor's own bytes."
            )
        if str(self.psd_factor.dtype) != self.realized_dtype:
            raise ValueError(
                f"psd_factor.dtype {self.psd_factor.dtype} does not match the core dtype "
                f"{self.realized_dtype}."
            )

        eigvals = np.linalg.eigvalsh(self.Z)  # gauge-invariant, ascending
        spectral_scale = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        raw_min_eigenvalue = float(eigvals[0]) if eigvals.size else 0.0
        threshold = -self.psd_rtol * spectral_scale
        if np.any(eigvals < threshold):
            offending = float(eigvals[eigvals < threshold].min())
            raise ValueError(
                f"Z is materially indefinite (eigenvalue {offending!r} below "
                f"{threshold!r}); a factorized core must have no material negative mode."
            )
        negative = eigvals < 0.0
        negative_mode_count = int(np.count_nonzero(negative))
        clipped_absolute_weight = (
            float(np.sum(np.abs(eigvals[negative]))) if negative_mode_count else 0.0
        )
        retained_rank = int(np.count_nonzero(np.where(negative, 0.0, eigvals) > 0.0))

        float_checks = {
            "psd_rtol": self.psd_rtol,  # identity: bound above and folded into the spec
            "psd_raw_min_eigenvalue": raw_min_eigenvalue,
            "psd_spectral_scale": spectral_scale,
            "psd_clipped_absolute_weight": clipped_absolute_weight,
        }
        for name, expected in float_checks.items():
            actual = getattr(self, name)
            if not isinstance(actual, float) or not math.isfinite(actual):
                raise ValueError(f"{name} must be a finite float, got {actual!r}.")
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(
                    f"{name}={actual!r} does not match the value recomputed from Z "
                    f"({expected!r})."
                )
        int_checks = {
            "psd_negative_mode_count": negative_mode_count,
            "psd_clipped_mode_count": negative_mode_count,
            "psd_retained_rank": retained_rank,
        }
        for name, expected in int_checks.items():
            actual = getattr(self, name)
            if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
                raise ValueError(f"{name} must be a non-negative int, got {actual!r}.")
            if actual != expected:
                raise ValueError(
                    f"{name}={actual!r} does not match the value recomputed from Z "
                    f"({expected!r})."
                )

        # The stored factor must have the retained-rank width and must actually
        # reconstruct Z to the recorded residual (catches a wrong-magnitude or
        # otherwise non-reconstructing factor; a pure sign flip is already
        # caught by the content-hash bind above).
        if self.psd_factor.shape != (self.n_mu, retained_rank):
            raise ValueError(
                f"psd_factor.shape {self.psd_factor.shape} must be "
                f"{(self.n_mu, retained_rank)} (n_mu, retained_rank)."
            )
        actual_reconstruction = _normalized_frobenius_residual(
            self.Z, self.psd_factor @ self.psd_factor.conj().T
        )
        if not isinstance(self.psd_reconstruction_residual, float) or not math.isfinite(
            self.psd_reconstruction_residual
        ):
            raise ValueError(
                f"psd_reconstruction_residual must be a finite float, got "
                f"{self.psd_reconstruction_residual!r}."
            )
        if not math.isclose(self.psd_reconstruction_residual, actual_reconstruction,
                            rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"psd_reconstruction_residual={self.psd_reconstruction_residual!r} does not "
                f"match the residual of the STORED factor's own reconstruction "
                f"({actual_reconstruction!r})."
            )

    def _post_init_jax(self):
        """Validate a device-resident JAX core. Mirrors the NumPy validation
        but every large array stays a device jax.Array on self.device; only
        rank-0 reduced scalars cross to the host (through _ibp_sync_scalars).
        The z_*/psd_factor content hashes are None (caller-attested-input trust
        boundary); identity is bound through the recomputed spec digest and the
        gauge-invariant scalar diagnostics + W W^dagger reconstruction residual,
        never an exact eigenvector-matrix comparison."""
        if self.output_identity_source != "derived_from_attested_inputs_unverified":
            raise ValueError(
                "backend='jax' requires output_identity_source="
                "'derived_from_attested_inputs_unverified'."
            )
        if not isinstance(self.same_sector, bool):
            raise TypeError("same_sector must be bool.")
        if self.symmetry_mode not in _SUPPORTED_IBP_CORE_SYMMETRY_MODES:
            raise ValueError(
                f"Unsupported symmetry_mode={self.symmetry_mode!r}, must be one of "
                f"{_SUPPORTED_IBP_CORE_SYMMETRY_MODES}."
            )
        n_mu = _validate_positive_int("n_mu", self.n_mu)
        n_nu = _validate_positive_int("n_nu", self.n_nu)

        def _require_device_array(name, arr, shape):
            if not isinstance(arr, jax.Array):
                raise TypeError(f"{name} must be a jax.Array for backend='jax'.")
            if str(arr.device) != self.device:
                raise ValueError(
                    f"{name} device {str(arr.device)!r} does not match core device "
                    f"{self.device!r}."
                )
            if arr.shape != shape:
                raise ValueError(f"{name}.shape must be {shape}, got {arr.shape}.")

        _require_device_array("Z", self.Z, (n_mu, n_nu))
        _require_device_array("z_forward_one_sided", self.z_forward_one_sided, (n_mu, n_nu))

        if self.same_sector:
            if self.z_reverse_one_sided is not None:
                raise ValueError("z_reverse_one_sided must be None for same_sector=True.")
            if self.left_sector_spec_sha256 != self.right_sector_spec_sha256:
                raise ValueError(
                    "same_sector=True requires left_sector_spec_sha256 == "
                    "right_sector_spec_sha256."
                )
            if n_mu != n_nu:
                raise ValueError("same_sector=True requires n_mu == n_nu.")
            dagger = self.z_forward_one_sided.conj().T
        else:
            _require_device_array("z_reverse_one_sided", self.z_reverse_one_sided, (n_nu, n_mu))
            dagger = self.z_reverse_one_sided.conj().T

        checks = _ibp_sync_scalars(
            zf=jnp.all(jnp.isfinite(self.z_forward_one_sided)),
            z=jnp.all(jnp.isfinite(self.Z)),
            zr=(jnp.all(jnp.isfinite(self.z_reverse_one_sided))
                if self.z_reverse_one_sided is not None else jnp.asarray(True)),
        )
        if not checks["zf"]:
            raise ValueError("z_forward_one_sided has non-finite entries.")
        if not checks["z"]:
            raise ValueError("Z has non-finite entries.")
        if self.z_reverse_one_sided is not None and not checks["zr"]:
            raise ValueError("z_reverse_one_sided has non-finite entries.")

        recomputed_residual = _normalized_frobenius_residual_jax(self.z_forward_one_sided, dagger)
        if not math.isclose(self.raw_dagger_residual, recomputed_residual,
                            rel_tol=1e-9, abs_tol=1e-14):
            raise ValueError(
                f"raw_dagger_residual={self.raw_dagger_residual!r} does not match the value "
                f"recomputed from the artifact's own raw orientation(s) "
                f"({recomputed_residual!r})."
            )
        expected_Z = (
            self.z_forward_one_sided if self.symmetry_mode == "one_sided"
            else (self.z_forward_one_sided + dagger) / 2
        )
        if not _ibp_sync_scalars(m=jnp.all(self.Z == expected_Z))["m"]:
            raise ValueError(
                "Z does not match the value recomputed from this artifact's own raw "
                "orientation(s) and symmetry_mode."
            )

        for name in ("z_sha256", "z_forward_sha256", "z_reverse_sha256"):
            if getattr(self, name) is not None:
                raise ValueError(
                    f"{name} must be None for backend='jax': device bytes are never hashed "
                    f"on the fly; identity is inherited from the attested inputs."
                )

        if self.same_sector:
            if self.raw_packed_pair_metric_dagger_residual is None:
                raise ValueError(
                    "raw_packed_pair_metric_dagger_residual is required for same_sector=True."
                )
            _validate_nonnegative_finite(
                "raw_packed_pair_metric_dagger_residual",
                self.raw_packed_pair_metric_dagger_residual,
            )
        elif self.raw_packed_pair_metric_dagger_residual is not None:
            raise ValueError(
                "raw_packed_pair_metric_dagger_residual must be None for cross-sector cores."
            )

        self._validate_psd_fields_jax()

        if isinstance(self.coincident_pairs, bool) or not isinstance(self.coincident_pairs, int):
            raise ValueError(f"coincident_pairs must be a non-negative int, got {self.coincident_pairs!r}.")
        if self.coincident_pairs < 0:
            raise ValueError(f"coincident_pairs must be non-negative, got {self.coincident_pairs!r}.")

        data_dtype = np.dtype(self.realized_dtype)
        if not np.issubdtype(data_dtype, np.inexact):
            raise ValueError("core data must use a floating or complex dtype.")
        if str(self.Z.dtype) != self.realized_dtype:
            raise ValueError(
                f"realized_dtype={self.realized_dtype!r} does not match Z.dtype {self.Z.dtype}."
            )

        mu_block_size = _validate_positive_int("mu_block_size", self.mu_block_size)
        nu_block_size = _validate_positive_int("nu_block_size", self.nu_block_size)
        object.__setattr__(self, "mu_block_size", mu_block_size)
        object.__setattr__(self, "nu_block_size", nu_block_size)

        for name in ("left_sector_spec_sha256", "right_sector_spec_sha256", "operator_spec_sha256"):
            _validate_sha256_hex(name, getattr(self, name))

        build_wall_time_seconds = _validate_nonnegative_finite(
            "build_wall_time_seconds", self.build_wall_time_seconds
        )
        object.__setattr__(self, "build_wall_time_seconds", build_wall_time_seconds)
        if self.peak_host_bytes is not None:
            raise ValueError("peak_host_bytes must be None for the JAX device backend.")
        if self.peak_host_bytes_status != "unmeasured_jax_device":
            raise ValueError(
                f"peak_host_bytes_status={self.peak_host_bytes_status!r} must be "
                f"'unmeasured_jax_device' for backend='jax'."
            )
        if self.solver_version != _IBP_CORE_VERSION:
            raise ValueError(f"solver_version={self.solver_version!r} != {_IBP_CORE_VERSION!r}.")

        object.__setattr__(
            self, "provenance", _deep_freeze(dict(self.provenance) if self.provenance else {})
        )
        if _canonical_spec_sha256(self._core_spec_fields(n_mu, n_nu)) != self.core_spec_sha256:
            raise ValueError(
                "core_spec_sha256 does not match the canonical digest recomputed from this "
                "artifact's own declared fields."
            )

    def _validate_psd_fields_jax(self):
        """Device-resident PSD validation mirroring _validate_psd_fields, but
        gauge-robust: validate the stored device factor's shape/device/dtype,
        the gauge-invariant scalar diagnostics recomputed from Z on device, and
        the W W^dagger reconstruction residual -- never an exact eigenvector
        comparison (phase/degenerate-subspace ambiguity), and never a device
        content hash (psd_factor_sha256 stays None)."""
        psd_scalar_fields = (
            "psd_rtol", "psd_factor", "psd_factor_sha256", "psd_raw_min_eigenvalue",
            "psd_spectral_scale", "psd_negative_mode_count", "psd_clipped_mode_count",
            "psd_clipped_absolute_weight", "psd_retained_rank", "psd_reconstruction_residual",
        )
        applicable = self.same_sector and self.symmetry_mode == "two_sided_average"
        if self.psd_status == "not_applicable":
            if applicable:
                raise ValueError(
                    "psd_status='not_applicable' is invalid for a same-sector, "
                    "two-sided-averaged core, which must be factorized."
                )
            for name in psd_scalar_fields:
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} must be None when psd_status='not_applicable'.")
            return
        if self.psd_status != "factorized":
            raise ValueError(
                f"psd_status={self.psd_status!r} must be 'factorized' or 'not_applicable'."
            )
        if self.psd_factor_sha256 is not None:
            raise ValueError("psd_factor_sha256 must be None for backend='jax'.")
        if not isinstance(self.psd_factor, jax.Array):
            raise TypeError("psd_factor must be a jax.Array for backend='jax'.")
        if str(self.psd_factor.device) != self.device:
            raise ValueError(
                f"psd_factor device {str(self.psd_factor.device)!r} does not match core "
                f"device {self.device!r}."
            )
        if str(self.psd_factor.dtype) != self.realized_dtype:
            raise ValueError(
                f"psd_factor.dtype {self.psd_factor.dtype} does not match the core dtype "
                f"{self.realized_dtype}."
            )

        eigvals = jnp.linalg.eigvalsh(self.Z)  # gauge-invariant, ascending, device
        threshold = -self.psd_rtol * jnp.max(jnp.abs(eigvals))
        negative = eigvals < 0.0
        diag = _ibp_sync_scalars(
            spectral_scale=jnp.max(jnp.abs(eigvals)),
            raw_min=eigvals[0],
            material_negative=jnp.any(eigvals < threshold),
            offending=jnp.min(jnp.where(eigvals < threshold, eigvals, jnp.inf)),
            negative_mode_count=jnp.sum(negative),
            clipped_absolute_weight=jnp.sum(jnp.where(negative, jnp.abs(eigvals), 0.0)),
            retained=jnp.sum(eigvals > 0.0),
        )
        if diag["material_negative"]:
            raise ValueError(
                f"Z is materially indefinite (eigenvalue {diag['offending']!r} below the "
                f"roundoff band); a factorized core must have no material negative mode."
            )
        float_checks = {
            "psd_raw_min_eigenvalue": diag["raw_min"],
            "psd_spectral_scale": diag["spectral_scale"],
            "psd_clipped_absolute_weight": diag["clipped_absolute_weight"],
        }
        for name, expected in float_checks.items():
            actual = getattr(self, name)
            if not isinstance(actual, float) or not math.isfinite(actual):
                raise ValueError(f"{name} must be a finite float, got {actual!r}.")
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(
                    f"{name}={actual!r} does not match the value recomputed from Z ({expected!r})."
                )
        if not isinstance(self.psd_rtol, float) or not math.isfinite(self.psd_rtol):
            raise ValueError(f"psd_rtol must be a finite float, got {self.psd_rtol!r}.")
        int_checks = {
            "psd_negative_mode_count": int(diag["negative_mode_count"]),
            "psd_clipped_mode_count": int(diag["negative_mode_count"]),
            "psd_retained_rank": int(diag["retained"]),
        }
        for name, expected in int_checks.items():
            actual = getattr(self, name)
            if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
                raise ValueError(f"{name} must be a non-negative int, got {actual!r}.")
            if actual != expected:
                raise ValueError(
                    f"{name}={actual!r} does not match the value recomputed from Z ({expected!r})."
                )
        retained_rank = int(diag["retained"])
        if self.psd_factor.shape != (self.n_mu, retained_rank):
            raise ValueError(
                f"psd_factor.shape {self.psd_factor.shape} must be "
                f"{(self.n_mu, retained_rank)} (n_mu, retained_rank)."
            )
        actual_reconstruction = _normalized_frobenius_residual_jax(
            self.Z, self.psd_factor @ self.psd_factor.conj().T
        )
        if not isinstance(self.psd_reconstruction_residual, float) or not math.isfinite(
            self.psd_reconstruction_residual
        ):
            raise ValueError(
                f"psd_reconstruction_residual must be a finite float, got "
                f"{self.psd_reconstruction_residual!r}."
            )
        if not math.isclose(self.psd_reconstruction_residual, actual_reconstruction,
                            rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"psd_reconstruction_residual={self.psd_reconstruction_residual!r} does not "
                f"match the STORED factor's own reconstruction ({actual_reconstruction!r})."
            )


# ---------------------------------------------------------------------------
# Tiled, device-resident JAX direct backend (task #10). The whole orientation
# is built inside ONE jax.jit via lax.fori_loop over eval/nu/source/mu tiles,
# so short/padded blocks never trigger shape-dependent recompiles and the
# jaxpr/HLO allocation audit sees the complete builder. Only bounded tiles
# (rhat (E,S,3), field V (Nn,3,E)) are formed -- never an (Ng,Ng), (3,Ng,Ng),
# or full V(nu,3,Ng) array. The field tile V is mu-independent and reused
# across mu tiles. Every large array stays a device jax.Array; only rank-0
# reduced scalars ever cross to the host, through _ibp_sync_scalars.
# ---------------------------------------------------------------------------


def _ibp_sync_scalars(**device_values):
    """The single auditable device->host synchronization boundary for the JAX
    backend. Each value must already be a rank-0 (scalar) device array -- a
    reduced diagnostic, never a large array. Returns a dict of Python scalars.
    Transferring a non-scalar here is a programming error and is rejected."""
    out = {}
    for name, value in device_values.items():
        arr = jnp.asarray(value)
        if arr.ndim != 0:
            raise ValueError(
                f"_ibp_sync_scalars: {name!r} must be a rank-0 scalar to cross to host, "
                f"got shape {arr.shape} -- large arrays must stay device-resident."
            )
        if jnp.issubdtype(arr.dtype, jnp.bool_):
            out[name] = bool(arr)
        elif jnp.issubdtype(arr.dtype, jnp.integer):
            out[name] = int(arr)
        else:
            out[name] = float(arr)
    return out


def _pad_to_multiple(n, block):
    return ((n + block - 1) // block) * block


@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10))
def _ibp_orientation_jax(grad, theta, coords, weights, valid,
                         E, S, Nm, Nn, n_mu, n_nu):
    """Whole-orientation one-sided Z built inside one jit. grad
    (n_mu_pad,3,Ng_pad); theta (n_nu_pad,Ng_pad); coords (Ng_pad,3);
    weights/valid (Ng_pad,). Padded grid points carry valid=0 (so weight 0),
    padded orbital rows are zero; the (n_mu,n_nu) slice is returned."""
    ng_pad = coords.shape[0]
    n_mu_pad = grad.shape[0]
    n_nu_pad = theta.shape[0]
    n_eval = ng_pad // E
    n_src = ng_pad // S
    n_mut = n_mu_pad // Nm
    n_nut = n_nu_pad // Nn
    w_eff = weights * valid                      # zero weight on padded grid points

    def eval_body(et, Z):
        i0 = et * E
        coords_e = lax.dynamic_slice(coords, (i0, 0), (E, 3))
        we = lax.dynamic_slice(w_eff, (i0,), (E,))
        grad_e = lax.dynamic_slice(grad, (0, 0, i0), (n_mu_pad, 3, E))

        def nu_body(nt, Z):
            n0 = nt * Nn
            theta_n = lax.dynamic_slice(theta, (n0, 0), (Nn, ng_pad))

            def src_body(st, V):
                j0 = st * S
                coords_s = lax.dynamic_slice(coords, (j0, 0), (S, 3))
                we_s = lax.dynamic_slice(w_eff, (j0,), (S,))
                theta_ns = lax.dynamic_slice(theta_n, (0, j0), (Nn, S))
                diff = coords_e[:, None, :] - coords_s[None, :, :]   # (E,S,3)
                rad2 = jnp.sum(diff * diff, axis=-1)                 # (E,S)
                nz = rad2 > 0
                safe = jnp.where(nz, jnp.sqrt(rad2), 1.0)            # safe denominator
                rhat = jnp.where(nz[..., None], diff / safe[..., None], 0.0)
                wd = theta_ns * we_s[None, :]                        # (Nn,S)
                return V + jnp.einsum("nj,ijc->nci", wd, rhat)       # (Nn,3,E)

            V = lax.fori_loop(0, n_src, src_body, jnp.zeros((Nn, 3, E), grad.dtype))

            def mu_body(mt, Z):
                m0 = mt * Nm
                grad_mt = lax.dynamic_slice(grad_e, (m0, 0, 0), (Nm, 3, E))
                block = -0.5 * jnp.einsum("mci,nci,i->mn", grad_mt.conj(), V, we)
                cur = lax.dynamic_slice(Z, (m0, n0), (Nm, Nn))
                return lax.dynamic_update_slice(Z, cur + block, (m0, n0))

            return lax.fori_loop(0, n_mut, mu_body, Z)

        return lax.fori_loop(0, n_nut, nu_body, Z)

    Z = lax.fori_loop(0, n_eval, eval_body,
                      jnp.zeros((n_mu_pad, n_nu_pad), grad.dtype))
    return lax.dynamic_slice(Z, (0, 0), (n_mu, n_nu))


@functools.partial(jax.jit, static_argnums=(2, 3))
def _ibp_coincident_jax(coords, valid, E, S):
    """Total ordered coincident grid-pair count (matching NumPy's per-pass
    count), reduced to a rank-0 device scalar over valid eval/source pairs --
    no (Ng,Ng) array is retained (each tile reduces to a scalar)."""
    ng_pad = coords.shape[0]
    n_eval = ng_pad // E
    n_src = ng_pad // S

    def eval_body(et, acc):
        i0 = et * E
        coords_e = lax.dynamic_slice(coords, (i0, 0), (E, 3))
        valid_e = lax.dynamic_slice(valid, (i0,), (E,))

        def src_body(st, acc):
            j0 = st * S
            coords_s = lax.dynamic_slice(coords, (j0, 0), (S, 3))
            valid_s = lax.dynamic_slice(valid, (j0,), (S,))
            diff = coords_e[:, None, :] - coords_s[None, :, :]
            rad2 = jnp.sum(diff * diff, axis=-1)
            vp = (valid_e[:, None] > 0) & (valid_s[None, :] > 0)
            return acc + jnp.sum(jnp.where((rad2 == 0) & vp, 1, 0))

        return lax.fori_loop(0, n_src, src_body, acc)

    return lax.fori_loop(0, n_eval, eval_body, jnp.array(0, jnp.int64))


def _ibp_one_sided_block_jax(grad_theta_source, theta_source, grid, *,
                             eval_block_size, source_block_size,
                             mu_block, nu_block):
    """Device-resident one-sided Z for the JAX backend. grad_theta_source/
    theta_source/grid.coords/grid.weights are jax.Array on grid.device. Tile
    sizes E/S/mu/nu are the realized (clamped) block sizes. Returns the device
    Z (jax.Array) and a rank-0 device coincident-count scalar."""
    dtype = grad_theta_source.dtype
    if dtype in (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128)) and not jax.config.jax_enable_x64:
        raise ValueError(
            "JAX float64/complex128 ibp_core requires jax_enable_x64=True; call "
            "jax.config.update('jax_enable_x64', True) before building, or supply "
            "float32/complex64 inputs."
        )
    n_mu = grad_theta_source.shape[0]
    n_nu = theta_source.shape[0]
    ng = grid.coords.shape[0]
    g = math.lcm(int(eval_block_size), int(source_block_size))
    ng_pad = _pad_to_multiple(ng, g)
    nm_pad = _pad_to_multiple(n_mu, mu_block)
    nn_pad = _pad_to_multiple(n_nu, nu_block)
    # Grid weights are real; they multiply complex theta/grad inside the einsums
    # (JAX promotes). The validity mask matches the weights' real dtype.
    weight_dtype = grid.weights.dtype

    grad = jnp.zeros((nm_pad, 3, ng_pad), dtype).at[:n_mu, :, :ng].set(grad_theta_source)
    theta = jnp.zeros((nn_pad, ng_pad), dtype).at[:n_nu, :ng].set(theta_source)
    coords = jnp.zeros((ng_pad, 3), grid.coords.dtype).at[:ng].set(grid.coords)
    weights = jnp.zeros((ng_pad,), weight_dtype).at[:ng].set(grid.weights)
    valid = jnp.zeros((ng_pad,), weight_dtype).at[:ng].set(jnp.asarray(1, weight_dtype))

    Z = _ibp_orientation_jax(
        grad, theta, coords, weights, valid,
        int(eval_block_size), int(source_block_size),
        int(mu_block), int(nu_block), n_mu, n_nu,
    )
    coincident = _ibp_coincident_jax(coords, valid, int(eval_block_size), int(source_block_size))
    return Z, coincident


def _frob_sq_in_rank_space(a, g):
    """Re tr(a^dagger g a g) == ||P^dagger a P||_F^2 when g = P P^dagger --
    the rank-space Frobenius norm-squared of the packed-pair projection,
    never forming the (n_pair, n_pair) matrix. Backend-agnostic (a/g are
    jax.Array or ndarray)."""
    xp = jnp if isinstance(a, jax.Array) else np
    m = a.conj().T @ g @ a @ g
    return xp.real(xp.trace(m))


def psd_factorize_jax(z_hermitian, *, rtol=_DEFAULT_PSD_RTOL):
    """Device-resident Hermitian PSD factorization mirroring psd_factorize:
    device eigh, Hermiticity gate, material-negative hard-fail, roundoff
    clip, compact device W, zero-core -> rank 0. Only rank-0 reduced scalars
    cross to the host (through _ibp_sync_scalars); the eigenvalue vector and
    W stay device-resident."""
    Z = z_hermitian
    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError(f"psd_factorize_jax requires a square 2-D array, got shape {Z.shape}.")
    if isinstance(rtol, bool) or not isinstance(rtol, (int, float, np.integer, np.floating)):
        raise ValueError(f"psd_factorize_jax: rtol must be a real, non-bool number, got {rtol!r}.")
    rtol = float(rtol)
    if not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"psd_factorize_jax: rtol must be a finite non-negative float, got {rtol!r}.")
    n = Z.shape[0]
    z_norm = jnp.linalg.norm(Z)
    herm = jnp.where(z_norm > 0, jnp.linalg.norm(Z - Z.conj().T) / jnp.where(z_norm > 0, z_norm, 1.0), 0.0)
    finite = jnp.all(jnp.isfinite(Z))
    eigvals, eigvecs = jnp.linalg.eigh(Z)            # ascending real eigenvalues, device
    spectral_scale = jnp.max(jnp.abs(eigvals)) if n else jnp.asarray(0.0, eigvals.dtype)
    raw_min = eigvals[0] if n else jnp.asarray(0.0, eigvals.dtype)
    threshold = -rtol * spectral_scale
    material_negative = jnp.any(eigvals < threshold)
    offending = jnp.min(jnp.where(eigvals < threshold, eigvals, jnp.inf))
    negative = eigvals < 0.0
    negative_mode_count = jnp.sum(negative)
    clipped_absolute_weight = jnp.sum(jnp.where(negative, jnp.abs(eigvals), 0.0))
    retained = jnp.sum(eigvals > 0.0)
    diag = _ibp_sync_scalars(
        finite=finite, hermiticity=herm, spectral_scale=spectral_scale,
        raw_min=raw_min, material_negative=material_negative, offending=offending,
        negative_mode_count=negative_mode_count,
        clipped_absolute_weight=clipped_absolute_weight, retained=retained,
    )
    if not diag["finite"]:
        raise ValueError("psd_factorize_jax: Z_H has non-finite entries.")
    if diag["hermiticity"] > _PSD_HERMITICITY_TOL:
        raise ValueError(
            f"psd_factorize_jax: input is not Hermitian (residual {diag['hermiticity']!r} > "
            f"{_PSD_HERMITICITY_TOL!r}); PSD factorization requires a Hermitian core."
        )
    if diag["material_negative"]:
        raise ValueError(
            f"psd_factorize_jax: material negative eigenvalue {diag['offending']!r} is below "
            f"the roundoff band threshold {(-rtol * diag['spectral_scale'])!r} (rtol={rtol!r}, "
            f"spectral_scale={diag['spectral_scale']!r}); a materially indefinite core is a "
            f"hard compatibility failure and is never repaired by clipping."
        )
    retained_rank = int(diag["retained"])
    # eigh is ascending: the strictly-positive eigenvalues are the last
    # retained_rank; clipped negatives contribute nothing. W stays device-side.
    if retained_rank:
        pos_vals = eigvals[n - retained_rank:]
        W = eigvecs[:, n - retained_rank:] * jnp.sqrt(pos_vals)[None, :]
    else:
        W = eigvecs[:, n:]  # (n, 0)
    recon = W @ W.conj().T
    z_norm_h = jnp.linalg.norm(Z)
    recon_residual = jnp.where(z_norm_h > 0, jnp.linalg.norm(Z - recon) / jnp.where(z_norm_h > 0, z_norm_h, 1.0), 0.0)
    recon_residual = _ibp_sync_scalars(v=recon_residual)["v"]
    return PSDFactorization(
        factor=W, factor_sha256=None, rtol=rtol,
        raw_min_eigenvalue=diag["raw_min"], spectral_scale=diag["spectral_scale"],
        negative_mode_count=int(diag["negative_mode_count"]),
        clipped_mode_count=int(diag["negative_mode_count"]),
        clipped_absolute_weight=diag["clipped_absolute_weight"],
        retained_rank=retained_rank, reconstruction_residual=recon_residual,
    )


def _ibp_one_sided_block(grad_theta_source, theta_source, grid, *,
                         mu_block, nu_block, eval_block_size, source_block_size):
    """Assemble the full (n_mu, n_nu) one-sided Z via kernel(), tiled over
    the pivot (mu/nu) axes at this level -- kernel() itself only tiles over
    the GRID axis (eval_block_size/source_block_size), never the orbital/
    pivot axis, so a large pivot count needs this outer blocking to bound
    the size of kernel()'s internal (n_nu, 3, n_eval)-shaped intermediate.
    Not yet memory-optimal (each (mu,nu) block pair re-walks the grid
    quadrature independently) -- task #10's tiled JAX backend is where
    genuine shared-intermediate reuse belongs; this NumPy oracle path
    favors correctness and a simple, auditable loop structure."""
    n_mu_total = grad_theta_source.shape[0]
    n_nu_total = theta_source.shape[0]
    row_blocks = []
    coincident_pairs = None
    for mu_start in range(0, n_mu_total, mu_block):
        mu_end = min(mu_start + mu_block, n_mu_total)
        col_blocks = []
        for nu_start in range(0, n_nu_total, nu_block):
            nu_end = min(nu_start + nu_block, n_nu_total)
            z_block, block_coincident = kernel(
                grad_theta_source[mu_start:mu_end], theta_source[nu_start:nu_end],
                grid.coords, grid.weights,
                eval_block_size=eval_block_size, source_block_size=source_block_size,
            )
            col_blocks.append(z_block)
            coincident_pairs = block_coincident
        row_blocks.append(np.concatenate(col_blocks, axis=1))
    return np.concatenate(row_blocks, axis=0), coincident_pairs


def ibp_core(left, right=None, *, operator, symmetry_mode="two_sided_average",
             mu_block_size=None, nu_block_size=None, psd_rtol=_DEFAULT_PSD_RTOL,
             upstream_provenance=None):
    """Build an IBPCoreArtifact from one (same-sector) or two (cross-sector)
    IBPInterpolationSector artifacts, via this module's own kernel()
    primitive applied to sector Theta/grad_Theta arrays.

    Args:
        left: IBPInterpolationSector for the sector (same-sector) or
            sector A (cross-sector).
        right: None (same-sector: right_sector = left, the literal same
            Theta/grad_Theta arrays, no duplicated computation) or an
            IBPInterpolationSector for sector B (cross-sector).
        operator: IBPOperatorPlan, REQUIRED. Its bound grid must match
            both sectors' bound grid exactly (grid_spec_sha256 equality);
            only method="direct" is implemented (matching this module's
            only implemented method); backend must be "numpy".
        symmetry_mode: "two_sided_average" (default, production) or
            "one_sided" (diagnostic only, unaveraged raw Z).
        mu_block_size/nu_block_size: bound the pivot-axis blocking of the
            left/right sectors respectively. None resolves to each
            sector's full selected_rank (single block); the realized
            value is recorded.
        upstream_provenance: optional caller-supplied dict of extra facts,
            deep-frozen into the returned artifact's provenance.

    Returns:
        IBPCoreArtifact.
    """
    if not isinstance(left, IBPInterpolationSector):
        raise TypeError(f"left must be an IBPInterpolationSector, got {type(left).__name__}.")
    if right is not None and not isinstance(right, IBPInterpolationSector):
        raise TypeError(f"right must be an IBPInterpolationSector or None, got {type(right).__name__}.")
    if not isinstance(operator, IBPOperatorPlan):
        raise TypeError(f"operator must be an IBPOperatorPlan, got {type(operator).__name__}.")
    if symmetry_mode not in _SUPPORTED_IBP_CORE_SYMMETRY_MODES:
        raise ValueError(
            f"Unsupported symmetry_mode={symmetry_mode!r}, must be one of "
            f"{_SUPPORTED_IBP_CORE_SYMMETRY_MODES}."
        )
    if operator.method != "direct":
        raise ValueError(f"Unsupported operator.method={operator.method!r} for ibp_core.")
    backend = operator.backend
    if backend not in _SUPPORTED_IBP_BACKENDS:
        raise NotImplementedError(
            f"ibp_core supports backend in {_SUPPORTED_IBP_BACKENDS}, got "
            f"operator.backend={backend!r}."
        )

    same_sector = right is None
    right_sector = left if same_sector else right

    for sector, label in ((left, "left"), (right_sector, "right")):
        if sector.backend != backend:
            raise ValueError(
                f"{label} sector.backend={sector.backend!r} must match "
                f"operator.backend={backend!r} -- a core cannot mix backends."
            )
        if sector.grid.grid_spec_sha256 != operator.grid.grid_spec_sha256:
            raise ValueError(
                f"{label} sector's bound grid does not match operator.grid "
                f"(grid_spec_sha256 mismatch) -- a core cannot join sectors/operators "
                f"built on different grids."
            )

    n_mu = left.selected_rank
    n_nu = right_sector.selected_rank
    # Clamp the requested pivot block to the realized rank: a block larger
    # than the rank produces exactly one rank-sized block, so recording the
    # oversized request would misreport the realized tiling.
    mu_block = (
        min(_validate_positive_int("mu_block_size", mu_block_size), n_mu)
        if mu_block_size is not None else n_mu
    )
    nu_block = (
        min(_validate_positive_int("nu_block_size", nu_block_size), n_nu)
        if nu_block_size is not None else n_nu
    )

    start = time.perf_counter()
    if backend == "numpy":
        z_forward, coincident_pairs = _ibp_one_sided_block(
            left.grad_Theta, right_sector.Theta, operator.grid,
            mu_block=mu_block, nu_block=nu_block,
            eval_block_size=operator.eval_block_size, source_block_size=operator.source_block_size,
        )
    else:
        z_forward, coincident_dev = _ibp_one_sided_block_jax(
            left.grad_Theta, right_sector.Theta, operator.grid,
            eval_block_size=operator.eval_block_size, source_block_size=operator.source_block_size,
            mu_block=mu_block, nu_block=nu_block,
        )
        coincident_pairs = _ibp_sync_scalars(c=coincident_dev)["c"]
    if same_sector:
        z_reverse = None
        dagger = z_forward.conj().T
    else:
        if backend == "numpy":
            z_reverse, reverse_coincident = _ibp_one_sided_block(
                right_sector.grad_Theta, left.Theta, operator.grid,
                mu_block=nu_block, nu_block=mu_block,
                eval_block_size=operator.eval_block_size, source_block_size=operator.source_block_size,
            )
        else:
            z_reverse, reverse_dev = _ibp_one_sided_block_jax(
                right_sector.grad_Theta, left.Theta, operator.grid,
                eval_block_size=operator.eval_block_size, source_block_size=operator.source_block_size,
                mu_block=nu_block, nu_block=mu_block,
            )
            reverse_coincident = _ibp_sync_scalars(c=reverse_dev)["c"]
        if reverse_coincident != coincident_pairs:
            raise ValueError(
                "forward/reverse orientation coincident-pair counts disagree "
                f"({coincident_pairs} vs {reverse_coincident}) -- both orientations use "
                "the identical bound grid, so this indicates a structural inconsistency."
            )
        dagger = z_reverse.conj().T
    build_wall_time_seconds = time.perf_counter() - start

    # Reject non-finite raw orientations immediately after assembly, before any
    # norm/eigendecomposition would otherwise propagate nan/inf silently. For
    # JAX the finiteness reduces on device; only the rank-0 bool crosses.
    if backend == "numpy":
        forward_finite = bool(np.all(np.isfinite(z_forward)))
        reverse_finite = z_reverse is None or bool(np.all(np.isfinite(z_reverse)))
    else:
        checks = _ibp_sync_scalars(
            f=jnp.all(jnp.isfinite(z_forward)),
            r=jnp.all(jnp.isfinite(z_reverse)) if z_reverse is not None else jnp.asarray(True),
        )
        forward_finite = checks["f"]
        reverse_finite = z_reverse is None or checks["r"]
    if not forward_finite:
        raise ValueError("ibp_core: z_forward one-sided orientation has non-finite entries.")
    if not reverse_finite:
        raise ValueError("ibp_core: z_reverse one-sided orientation has non-finite entries.")

    if backend == "numpy":
        raw_dagger_residual = _normalized_frobenius_residual(z_forward, dagger)
    else:
        raw_dagger_residual = _normalized_frobenius_residual_jax(z_forward, dagger)
    Z = z_forward if symmetry_mode == "one_sided" else (z_forward + dagger) / 2

    # Production-facing packed-pair metric dagger residual (same-sector only):
    # project the raw one-sided core back to the packed AO-pair space through
    # the sector's interpolation matrix P (shape (rank, packed_pair)) and
    # measure that pair metric's departure from Hermiticity -- this is the
    # quantity task #8's <=1e-3 gate is stated against.
    if same_sector:
        P = left.P
        if backend == "numpy":
            m_raw = P.conj().T @ z_forward @ P
            raw_packed_pair_metric_dagger_residual = _normalized_frobenius_residual(
                m_raw, m_raw.conj().T
            )
        else:
            # Rank-space diagnostic: never form the (n_pair, n_pair) matrix.
            raw_packed_pair_metric_dagger_residual = (
                _packed_pair_residual_rank_space_jax(z_forward, P)
            )
    else:
        raw_packed_pair_metric_dagger_residual = None

    # PSD factorization: only for a same-sector, two-sided-averaged (Hermitian)
    # core. The corrective factor is the one task #8 will stream; material
    # negative modes hard-fail here rather than being repaired.
    if same_sector and symmetry_mode == "two_sided_average":
        psd = psd_factorize(Z, rtol=psd_rtol) if backend == "numpy" else psd_factorize_jax(Z, rtol=psd_rtol)
        psd_status = "factorized"
        psd_rtol_value = psd.rtol
        psd_factor = psd.factor
        psd_factor_sha256 = psd.factor_sha256
        psd_raw_min_eigenvalue = psd.raw_min_eigenvalue
        psd_spectral_scale = psd.spectral_scale
        psd_negative_mode_count = psd.negative_mode_count
        psd_clipped_mode_count = psd.clipped_mode_count
        psd_clipped_absolute_weight = psd.clipped_absolute_weight
        psd_retained_rank = psd.retained_rank
        psd_reconstruction_residual = psd.reconstruction_residual
    else:
        psd_status = "not_applicable"
        psd_rtol_value = None
        psd_factor = None
        psd_factor_sha256 = None
        psd_raw_min_eigenvalue = None
        psd_spectral_scale = None
        psd_negative_mode_count = None
        psd_clipped_mode_count = None
        psd_clipped_absolute_weight = None
        psd_retained_rank = None
        psd_reconstruction_residual = None

    if backend == "numpy":
        # NumPy: computed content hashes over the realized host arrays.
        output_identity_source = "computed_content_hash"
        z_sha256 = _canonical_sha256(Z)
        z_forward_sha256 = _canonical_sha256(z_forward)
        z_reverse_sha256 = None if same_sector else _canonical_sha256(z_reverse)
        device = "cpu"
    else:
        # JAX: the arrays are device-resident and derived from caller-attested
        # sector/operator inputs (task #6 trust boundary); their identity is
        # inherited from those attested upstream identities, not a device
        # content hash. Never hash device bytes on the fly.
        output_identity_source = "derived_from_attested_inputs_unverified"
        z_sha256 = None
        z_forward_sha256 = None
        z_reverse_sha256 = None
        device = str(Z.device)
        if str(operator.grid.device) != device:
            raise ValueError(
                f"JAX core output device {device!r} does not match the bound grid device "
                f"{operator.grid.device!r}."
            )

    # A truthful build-scoped host-memory peak is not available for this
    # oracle (ru_maxrss is process-lifetime; tracemalloc misses NumPy C-level
    # and device allocations). Record an explicit measured-absence.
    peak_host_bytes = None
    peak_host_bytes_status = "unmeasured_cpu_oracle" if backend == "numpy" else "unmeasured_jax_device"

    upstream = dict(upstream_provenance) if upstream_provenance else {}
    provenance = _deep_freeze({
        "left_sector_provenance": dict(left.provenance),
        "right_sector_provenance": dict(right_sector.provenance),
        "operator_provenance": dict(operator.provenance),
        "upstream_provenance": upstream,
    })

    realized_dtype = str(Z.dtype)
    # Core spec v2: the identity digest is DETERMINISTIC -- nondeterministic
    # execution measurements (build_wall_time_seconds, peak_host_bytes) are
    # excluded and carried only as execution metadata, so the same inputs
    # always yield the same core_spec_sha256 regardless of timing/host memory.
    spec_fields = {
        "same_sector": same_sector, "symmetry_mode": symmetry_mode,
        "z_sha256": z_sha256, "z_forward_sha256": z_forward_sha256,
        "z_reverse_sha256": z_reverse_sha256,
        "output_identity_source": output_identity_source,
        "raw_dagger_residual": raw_dagger_residual,
        "raw_packed_pair_metric_dagger_residual": raw_packed_pair_metric_dagger_residual,
        "coincident_pairs": coincident_pairs, "n_mu": n_mu, "n_nu": n_nu,
        "psd_status": psd_status, "psd_rtol": psd_rtol_value,
        "psd_factor_sha256": psd_factor_sha256,
        "psd_raw_min_eigenvalue": psd_raw_min_eigenvalue,
        "psd_spectral_scale": psd_spectral_scale,
        "psd_negative_mode_count": psd_negative_mode_count,
        "psd_clipped_mode_count": psd_clipped_mode_count,
        "psd_clipped_absolute_weight": psd_clipped_absolute_weight,
        "psd_retained_rank": psd_retained_rank,
        "psd_reconstruction_residual": psd_reconstruction_residual,
        "left_sector_spec_sha256": left.sector_spec_sha256,
        "right_sector_spec_sha256": right_sector.sector_spec_sha256,
        "operator_spec_sha256": operator.operator_spec_sha256,
        "mu_block_size": mu_block, "nu_block_size": nu_block,
        "backend": backend, "device": device, "realized_dtype": realized_dtype,
        "solver_version": _IBP_CORE_VERSION, "provenance": provenance,
    }
    core_spec_sha256 = _canonical_spec_sha256(spec_fields)

    return IBPCoreArtifact(
        Z=Z, same_sector=same_sector, symmetry_mode=symmetry_mode,
        z_forward_one_sided=z_forward, z_reverse_one_sided=z_reverse,
        z_sha256=z_sha256, z_forward_sha256=z_forward_sha256,
        z_reverse_sha256=z_reverse_sha256,
        output_identity_source=output_identity_source,
        raw_dagger_residual=raw_dagger_residual,
        raw_packed_pair_metric_dagger_residual=raw_packed_pair_metric_dagger_residual,
        coincident_pairs=coincident_pairs,
        n_mu=n_mu, n_nu=n_nu,
        psd_status=psd_status, psd_rtol=psd_rtol_value,
        psd_factor=psd_factor, psd_factor_sha256=psd_factor_sha256,
        psd_raw_min_eigenvalue=psd_raw_min_eigenvalue,
        psd_spectral_scale=psd_spectral_scale,
        psd_negative_mode_count=psd_negative_mode_count,
        psd_clipped_mode_count=psd_clipped_mode_count,
        psd_clipped_absolute_weight=psd_clipped_absolute_weight,
        psd_retained_rank=psd_retained_rank,
        psd_reconstruction_residual=psd_reconstruction_residual,
        left_sector_spec_sha256=left.sector_spec_sha256,
        right_sector_spec_sha256=right_sector.sector_spec_sha256,
        operator_spec_sha256=operator.operator_spec_sha256,
        mu_block_size=mu_block, nu_block_size=nu_block,
        backend=backend, device=device, realized_dtype=realized_dtype,
        build_wall_time_seconds=build_wall_time_seconds, peak_host_bytes=peak_host_bytes,
        peak_host_bytes_status=peak_host_bytes_status,
        solver_version=_IBP_CORE_VERSION, provenance=provenance,
        core_spec_sha256=core_spec_sha256,
    )
