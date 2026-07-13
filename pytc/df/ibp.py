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
import hashlib
import math
import types

import jax
import jax.numpy as jnp
import numpy as np


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


def _canonical_encode(obj, buf):
    """Append a deterministic, type-tagged canonical byte encoding of obj
    (already routed through _deep_freeze, or a plain scalar) to buf, a
    list of bytes fragments -- for provenance hashing.

    Deliberately NOT repr()-based (Alice's review, task #5,
    2026-07-13): repr(frozenset(...)) iterates in the set's internal
    hash-table order, which depends on PYTHONHASHSEED for str elements
    (randomized per process by default) -- the same logical
    construction_metadata could hash differently across runs, and
    Alice's repro even showed builder vs __post_init__ disagreeing
    WITHIN one construction (re-freezing a frozenset rebuilds its
    internal table). repr() on a large ndarray also silently truncates
    (numpy's summarized repr), so two arrays differing only in their
    untruncated middle content could hash identically. This encoder
    instead: sorts dict keys (str, per _deep_freeze's closed schema);
    preserves tuple/list element order; for frozenset/set, encodes each
    element independently and sorts the resulting BYTE strings (well-
    defined regardless of hash-seed-dependent iteration order); and
    hashes an ndarray's FULL raw bytes (never a summarized repr)."""
    if isinstance(obj, types.MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        buf.append(b"dict{")
        for k in sorted(obj):
            if not isinstance(k, str):
                raise TypeError(f"_canonical_encode: mapping key must be str, got {k!r}.")
            buf.append(b"k:")
            _canonical_encode(k, buf)
            buf.append(b"=")
            _canonical_encode(obj[k], buf)
            buf.append(b";")
        buf.append(b"}")
        return
    if isinstance(obj, (list, tuple)):
        buf.append(b"seq[")
        for v in obj:
            _canonical_encode(v, buf)
            buf.append(b",")
        buf.append(b"]")
        return
    if isinstance(obj, (set, frozenset)):
        encoded_elements = []
        for v in obj:
            sub = []
            _canonical_encode(v, sub)
            encoded_elements.append(b"".join(sub))
        encoded_elements.sort()
        buf.append(b"set{")
        for e in encoded_elements:
            buf.append(e)
            buf.append(b",")
        buf.append(b"}")
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError(f"_canonical_encode: unsupported object-dtype array {obj.dtype}.")
        a = np.ascontiguousarray(obj)
        buf.append(b"ndarray(" + str(a.dtype).encode() + b"," + repr(a.shape).encode() + b")")
        buf.append(a.tobytes())
        return
    if isinstance(obj, np.generic):
        _canonical_encode(obj.item(), buf)
        return
    if obj is None:
        buf.append(b"none")
        return
    if isinstance(obj, bool):
        buf.append(b"bool:" + (b"1" if obj else b"0"))
        return
    if isinstance(obj, (int, float, complex)):
        buf.append(type(obj).__name__.encode() + b":" + repr(obj).encode())
        return
    if isinstance(obj, str):
        buf.append(b"str:" + obj.encode("utf-8", errors="surrogatepass"))
        return
    if isinstance(obj, bytes):
        buf.append(b"bytes:" + obj)
        return
    raise TypeError(
        f"_canonical_encode: unsupported type {type(obj).__name__} for provenance hashing "
        f"({obj!r})."
    )


def _canonical_spec_sha256(fields):
    """SHA-256 over a canonical, type-tagged encoding of specification
    fields (see _canonical_encode) -- deterministic across processes/
    hash seeds and never truncates large array data."""
    h = hashlib.sha256()
    for key in sorted(fields):
        h.update(key.encode())
        buf = []
        _canonical_encode(fields[key], buf)
        h.update(b"".join(buf))
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
