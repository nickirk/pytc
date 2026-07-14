"""Direct irregular-grid single-IBP Coulomb primitives (atom-centered grids)
plus the typed, immutable numerical artifacts built on top of them.

``kernel`` is the main single-IBP primitive (PySCF-style naming). Unlike a
uniform grid, a PySCF/TC atom-centered Becke grid is not translationally
invariant, so the ``rhat`` action is evaluated by blocked real-space
summation, not FFT:

    (ia|jb) = -1/2 sum_g w_g grad(phi_i phi_a)_g .
                         sum_h w_h rhat_(g-h) (phi_j phi_b)_h.

Coincident left/right points are assigned the centered-point convention
``rhat=0`` explicitly (not skipped or NaN-guarded implicitly); the primitive
returns ``coincident_pairs`` so a caller cannot silently forget which diagonal
convention was used.

The artifacts (IBPGrid, IBPOperatorPlan, IBPInterpolationSector,
IBPCoreArtifact) are frozen dataclasses carrying only the numerical state
their consumers need. ``ibp_core`` assembles the same/cross-sector Z core; a
same-sector two-sided-averaged core is passed through the Hermitian PSD gate
(``psd_factorize``/``psd_factorize_jax``). Both a NumPy CPU oracle backend and
a tiled, device-resident JAX backend are implemented.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import typing

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .solvers import (
    prepare_normal_equations_solver,
    solve_normal_equations_batch_prepared,
)

_SUPPORTED_IBP_BACKENDS = ("numpy", "jax")
_SUPPORTED_COINCIDENT_POINT_POLICIES = ("centered_zero",)
_SUPPORTED_IBP_OPERATOR_METHODS = ("direct",)
_SUPPORTED_IBP_STORAGE_MODES = ("incore_full_theta_gradient",)
_SUPPORTED_IBP_CORE_SYMMETRY_MODES = ("two_sided_average", "one_sided")
_DEFAULT_PSD_RTOL = 1e-10
_PSD_HERMITICITY_TOL = 1e-10


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

    The returned matrix is deliberately unsymmetrized. Coincident left/right
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


# ---------------------------------------------------------------------------
# Typed, immutable numerical artifacts.
# ---------------------------------------------------------------------------


def _validate_positive_int(name, value):
    """Reject bool (isinstance(True, int) is True) and non-integral floats
    rather than silently truncating them into a plausible-looking but wrong
    value."""
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


def _validate_nonnegative_finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")
    return value


def _readonly_copy(a):
    """Defensive COPY (never the caller's own array object) with the numpy
    write flag cleared -- np.asarray on an already-matching ndarray can return
    the SAME object, so copy first, always."""
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


@dataclasses.dataclass(frozen=True)
class IBPGrid:
    """Immutable atom-centered (or otherwise irregular) real-space quadrature
    grid: coordinates, physical weights, backend/device/dtype, and the
    coincident-point policy. Independent of any orbital sector -- reusable
    across every sector/plan/core built on it.

    coords/weights are backend-preserving: NumPy gets a defensive read-only
    copy; a JAX grid keeps the caller's (already-immutable) jax.Array on its
    device, never forced through a NumPy-only host transfer.

    coincident_point_policy records the odd-kernel convention used wherever
    this grid's points coincide (only "centered_zero": r=0 -> rhat=0).
    """
    coords: object
    weights: object
    n_grid: int
    backend: str
    device: str
    dtype: str
    coincident_point_policy: str

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


def build_ibp_grid(coords, weights, *, backend="numpy",
                   coincident_point_policy="centered_zero"):
    """Build an immutable IBPGrid from raw coordinates/weights.

    backend="numpy" (default) or "jax". A JAX grid keeps coords/weights on
    their realized device. All structural validation happens in IBPGrid.
    """
    if backend not in _SUPPORTED_IBP_BACKENDS:
        raise ValueError(f"Unsupported backend={backend!r}, must be one of {_SUPPORTED_IBP_BACKENDS}.")
    if backend == "numpy":
        coords = np.asarray(coords)
        weights = np.asarray(weights)
        device = "cpu"
    else:
        if not isinstance(coords, jax.Array) or not isinstance(weights, jax.Array):
            raise TypeError(
                f"backend='jax' requires coords/weights to be jax.Array, got "
                f"{type(coords).__name__}/{type(weights).__name__}."
            )
        device = str(coords.device)
    return IBPGrid(
        coords=coords, weights=weights, n_grid=coords.shape[0], backend=backend,
        device=device, dtype=str(coords.dtype),
        coincident_point_policy=coincident_point_policy,
    )


@dataclasses.dataclass(frozen=True)
class IBPOperatorPlan:
    """Immutable, grid-bound execution plan for the single-IBP Coulomb
    operator: method, realized block sizes, requested tolerance, and
    backend/device/dtype (always matching the bound IBPGrid exactly).

    Holds NO array data of its own -- the "direct" method tiles coordinate
    pairs on the fly at core-build time using only these block sizes. Only
    method="direct" is implemented.
    """
    grid: object
    method: str
    eval_block_size: int
    source_block_size: int
    tolerance: object
    backend: str
    device: str
    dtype: str

    def __post_init__(self):
        if not isinstance(self.grid, IBPGrid):
            raise TypeError(f"grid must be an IBPGrid, got {type(self.grid).__name__}.")
        if self.method not in _SUPPORTED_IBP_OPERATOR_METHODS:
            raise ValueError(
                f"Unsupported method={self.method!r}, must be one of "
                f"{_SUPPORTED_IBP_OPERATOR_METHODS}."
            )
        object.__setattr__(self, "eval_block_size",
                           _validate_positive_int("eval_block_size", self.eval_block_size))
        object.__setattr__(self, "source_block_size",
                           _validate_positive_int("source_block_size", self.source_block_size))
        if self.method == "direct" and self.tolerance is not None:
            raise ValueError(
                "tolerance must be None for method='direct' (an exact reference method has "
                "no approximation tolerance to request)."
            )
        if self.backend != self.grid.backend:
            raise ValueError(
                f"backend={self.backend!r} does not match the bound grid's backend "
                f"{self.grid.backend!r}."
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


def build_ibp_operator_plan(grid, *, method="direct", tolerance=None,
                            eval_block_size=128, source_block_size=4096):
    """Build an IBPOperatorPlan bound to an already-built IBPGrid."""
    if not isinstance(grid, IBPGrid):
        raise TypeError(f"grid must be an IBPGrid, got {type(grid).__name__}.")
    return IBPOperatorPlan(
        grid=grid, method=method, eval_block_size=eval_block_size,
        source_block_size=source_block_size, tolerance=tolerance,
        backend=grid.backend, device=grid.device, dtype=grid.dtype,
    )


# ---------------------------------------------------------------------------
# Fixed-pivot interpolation sectors.
# ---------------------------------------------------------------------------

_IBP_PIVOT_PROVENANCE_KEYS = frozenset({
    "requested_rank",
    "analytic_rank_bound",
    "n_rank_capped",
    "rank_exhausted",
    "numerical_rank",
    "numerical_rank_lower_bound",
    "n_pivots",
})


def _validate_pivot_provenance(record, *, n_p, n_q, same_factor, selected_rank):
    """Validate the exact record returned by the canonical higher-layer
    selector's ``return_provenance=True`` mode, so requested, selected, and
    numerical ranks can never be silently conflated. Returns the extracted
    requested_rank / numerical_rank (the sector stores only these).

    pytc.df.ibp is intentionally below pytc.integrals.coulomb in the package
    graph, so the selector's closed record is consumed and checked here rather
    than importing the higher layer.
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
    analytic_rank_bound = n_p * (n_p + 1) // 2 if same_factor else n_p * n_q
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
    return requested_rank, numerical_rank


@dataclasses.dataclass(frozen=True)
class IBPInterpolationSector:
    """One fixed-pivot orbital-pair interpolation sector on an IBPGrid.

    ``P`` is the raw pair collocation at the selected points. Same-factor
    sectors use packed-lower pair order (the layout required by the PySCF
    ``loop()`` bridge); mixed sectors use full row-major ``(p,q)`` order.
    ``Theta[mu,g]`` and ``grad_Theta[mu,3,g]`` are built with one prepared
    normal-equation factorization; the gradient applies the product rule
    using those exact pivots/factorization (it never runs a second pivot
    selection). NumPy outputs are defensive read-only copies; JAX outputs
    remain on the realized device.
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
    pair_layout: str
    requested_rank: int
    selected_rank: int
    numerical_rank: object
    storage_mode: str
    backend: str
    device: str
    realized_dtype: str

    def __post_init__(self):
        if not isinstance(self.grid, IBPGrid):
            raise TypeError(f"grid must be an IBPGrid, got {type(self.grid).__name__}.")
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
        _validate_positive_int("requested_rank", self.requested_rank)
        if n_grid != self.grid.n_grid:
            raise ValueError(f"n_grid={n_grid} != bound grid.n_grid={self.grid.n_grid}.")

        if self.pair_layout == "packed_lower":
            if n_p != n_q:
                raise ValueError("packed_lower layout requires n_orbital_p == n_orbital_q.")
            expected_n_pair = n_p * (n_p + 1) // 2
        elif self.pair_layout == "full_row_major":
            expected_n_pair = n_p * n_q
        else:
            raise ValueError(f"Unsupported pair_layout={self.pair_layout!r}.")
        if self.n_pair != expected_n_pair:
            raise ValueError(f"n_pair must be {expected_n_pair}, got {self.n_pair}.")

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
        pivots_np = np.asarray(self.pivots, dtype=np.int64)
        if np.unique(pivots_np).size != selected_rank:
            raise ValueError("pivots must be unique.")
        if pivots_np.min() < 0 or pivots_np.max() >= n_grid:
            raise ValueError(f"pivots must lie in [0, {n_grid}).")

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
    requested_rank, numerical_rank = _validate_pivot_provenance(
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
    chol, lower, _jitter, _n_tries = prepare_normal_equations_solver(
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
        P = (factor_p_piv[pair_p, :] * factor_q_piv[pair_q, :]).T
        pair_layout = "packed_lower"
        n_pair = factor_p_raw.shape[0] * (factor_p_raw.shape[0] + 1) // 2
    else:
        P = xp.einsum("pu,qu->upq", factor_p_piv, factor_q_piv).reshape(
            selected_rank, factor_p_raw.shape[0] * factor_q_raw.shape[0]
        )
        pair_layout = "full_row_major"
        n_pair = factor_p_raw.shape[0] * factor_q_raw.shape[0]
    if backend == "numpy":
        P = np.asarray(P)

    return IBPInterpolationSector(
        P=P, Theta=Theta, grad_Theta=grad_Theta, pivots=pivots_backend, grid=grid,
        n_orbital_p=int(factor_p_raw.shape[0]),
        n_orbital_q=int(factor_q_raw.shape[0]), n_pair=int(n_pair),
        n_grid=int(grid.n_grid), pair_layout=pair_layout,
        requested_rank=requested_rank, selected_rank=selected_rank,
        numerical_rank=numerical_rank, storage_mode="incore_full_theta_gradient",
        backend=backend, device=device, realized_dtype=str(Theta.dtype),
    )


# ---------------------------------------------------------------------------
# Same/cross-sector Z cores.
# ---------------------------------------------------------------------------


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
    rank-0 scalars cross to the host (through _ibp_sync_scalars)."""
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
    mandatory diagnostics."""
    factor: object                    # W, shape (n, retained_rank), read-only
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
    ``-rtol * max|eig|`` (a material negative mode is a discretization failure
    and must never be silently repaired), clips only the negative modes that
    lie inside the roundoff band to zero, and forms the COMPACT factor
    ``W = U[:, positive] sqrt(lambda_clipped[positive])`` with no zero columns.

    numpy.linalg.eigh reads the lower triangle; the caller passes the
    two-sided-averaged same-sector core, which is Hermitian by construction.
    Zero Z_H yields scale/tolerance/residual 0, retained rank 0, W.shape (n,0).
    """
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
    # would be silently "factorized" against its Hermitian completion.
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
    (cross-sector), built via ibp_core using kernel() applied to sector
    Theta/grad_Theta arrays:
    Z_AB[mu,nu] = -1/2 sum_g w_g grad(Theta_A[mu,g])^* .
                       sum_h w_h rhat(r_g-r_h) Theta_B[nu,h].

    The blocked real-space IBP kernel is NOT manifestly Hermitian, so the raw
    one-sided orientation(s) and their dagger residual are ALWAYS retained
    regardless of symmetry_mode. raw_dagger_residual is the normalized
    Frobenius residual ||z_forward - dagger||_F / ||z_forward||_F (0/0 -> 0):
    - same-sector: z_forward_one_sided is the sole raw computation;
      z_reverse_one_sided is None; the dagger is z_forward_one_sided^dagger.
      Same-sector cores also record raw_packed_pair_metric_dagger_residual,
      the normalized Frobenius departure from Hermiticity of the packed
      AO-pair metric P^dagger z_forward P.
    - cross-sector: BOTH orientations are evaluated (z_forward = Z_AB,
      z_reverse = Z_BA); the dagger is z_reverse_one_sided^dagger;
      raw_packed_pair_metric_dagger_residual is None.

    symmetry_mode="two_sided_average" (production default) sets
    Z = (z_forward + dagger)/2; "one_sided" (diagnostic) sets Z = z_forward.

    A same-sector two-sided-averaged core is passed through the PSD gate;
    psd_status="not_applicable" (with every psd_* field None) otherwise.
    backend="numpy" keeps host arrays; backend="jax" keeps Z, the raw
    orientation(s), and the PSD factor as device jax.Array on the bound device.
    """
    Z: object
    same_sector: bool
    symmetry_mode: str
    z_forward_one_sided: object
    z_reverse_one_sided: object
    raw_dagger_residual: float
    raw_packed_pair_metric_dagger_residual: object
    coincident_pairs: int
    n_mu: int
    n_nu: int
    psd_status: str
    psd_rtol: object
    psd_factor: object
    psd_raw_min_eigenvalue: object
    psd_spectral_scale: object
    psd_negative_mode_count: object
    psd_clipped_mode_count: object
    psd_clipped_absolute_weight: object
    psd_retained_rank: object
    psd_reconstruction_residual: object
    mu_block_size: int
    nu_block_size: int
    eval_block_size: int
    source_block_size: int
    backend: str
    device: str
    realized_dtype: str

    _PSD_SCALAR_FIELDS = (
        "psd_rtol", "psd_factor", "psd_raw_min_eigenvalue", "psd_spectral_scale",
        "psd_negative_mode_count", "psd_clipped_mode_count",
        "psd_clipped_absolute_weight", "psd_retained_rank", "psd_reconstruction_residual",
    )

    def __post_init__(self):
        if self.backend not in _SUPPORTED_IBP_BACKENDS:
            raise ValueError(
                f"Unsupported backend={self.backend!r} -- must be one of "
                f"{_SUPPORTED_IBP_BACKENDS}."
            )
        is_jax = self.backend == "jax"
        if not isinstance(self.same_sector, bool):
            raise TypeError("same_sector must be bool.")
        if self.symmetry_mode not in _SUPPORTED_IBP_CORE_SYMMETRY_MODES:
            raise ValueError(
                f"Unsupported symmetry_mode={self.symmetry_mode!r}, must be one of "
                f"{_SUPPORTED_IBP_CORE_SYMMETRY_MODES}."
            )
        n_mu = _validate_positive_int("n_mu", self.n_mu)
        n_nu = _validate_positive_int("n_nu", self.n_nu)

        if is_jax:
            self._require_device_array("Z", self.Z, (n_mu, n_nu))
            self._require_device_array("z_forward_one_sided", self.z_forward_one_sided, (n_mu, n_nu))
        else:
            if self.device != "cpu":
                raise ValueError(f"device must be 'cpu' for backend='numpy', got {self.device!r}.")
            for name in ("Z", "z_forward_one_sided"):
                if not isinstance(getattr(self, name), np.ndarray):
                    raise TypeError(f"{name} must be a numpy.ndarray for backend='numpy'.")
                object.__setattr__(self, name, _readonly_copy(getattr(self, name)))
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
            if n_mu != n_nu:
                raise ValueError("same_sector=True requires n_mu == n_nu.")
        else:
            if is_jax:
                self._require_device_array("z_reverse_one_sided", self.z_reverse_one_sided, (n_nu, n_mu))
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

        _validate_nonnegative_finite("raw_dagger_residual", self.raw_dagger_residual)

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

        for name in ("mu_block_size", "nu_block_size", "eval_block_size", "source_block_size"):
            object.__setattr__(self, name, _validate_positive_int(name, getattr(self, name)))

        self._validate_psd_fields(n_mu, is_jax)

    def _require_device_array(self, name, arr, shape):
        if not isinstance(arr, jax.Array):
            raise TypeError(f"{name} must be a jax.Array for backend='jax'.")
        if str(arr.device) != self.device:
            raise ValueError(
                f"{name} device {str(arr.device)!r} does not match core device {self.device!r}."
            )
        if arr.shape != shape:
            raise ValueError(f"{name}.shape must be {shape}, got {arr.shape}.")

    def _validate_psd_fields(self, n_mu, is_jax):
        """Structural validation of the PSD fields. The eigenvalue science (the
        material-negative hard failure, clipping, and reconstruction residual)
        is enforced in the builder via psd_factorize/psd_factorize_jax; here we
        only check the stored factor's shape/dtype/device and the all-None
        invariant for a non-factorized core."""
        applicable = self.same_sector and self.symmetry_mode == "two_sided_average"
        if self.psd_status == "not_applicable":
            if applicable:
                raise ValueError(
                    "psd_status='not_applicable' is invalid for a same-sector, "
                    "two-sided-averaged core, which must be factorized."
                )
            for name in self._PSD_SCALAR_FIELDS:
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} must be None when psd_status='not_applicable'.")
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
        if (isinstance(self.psd_rtol, bool) or not isinstance(self.psd_rtol, float)
                or not math.isfinite(self.psd_rtol) or self.psd_rtol < 0.0):
            raise ValueError(
                f"psd_rtol must be a finite non-negative float for a factorized core, "
                f"got {self.psd_rtol!r}."
            )
        retained_rank = self.psd_retained_rank
        if isinstance(retained_rank, bool) or not isinstance(retained_rank, int) or retained_rank < 0:
            raise ValueError(f"psd_retained_rank must be a non-negative int, got {retained_rank!r}.")
        if is_jax:
            if not isinstance(self.psd_factor, jax.Array):
                raise TypeError("psd_factor must be a jax.Array for backend='jax'.")
            if str(self.psd_factor.device) != self.device:
                raise ValueError(
                    f"psd_factor device {str(self.psd_factor.device)!r} does not match core "
                    f"device {self.device!r}."
                )
        else:
            if not isinstance(self.psd_factor, np.ndarray):
                raise TypeError("psd_factor must be a numpy.ndarray for a factorized core.")
            object.__setattr__(self, "psd_factor", _readonly_copy(self.psd_factor))
            if not np.all(np.isfinite(self.psd_factor)):
                raise ValueError("psd_factor has non-finite entries.")
        if str(self.psd_factor.dtype) != self.realized_dtype:
            raise ValueError(
                f"psd_factor.dtype {self.psd_factor.dtype} does not match the core dtype "
                f"{self.realized_dtype}."
            )
        if self.psd_factor.shape != (n_mu, retained_rank):
            raise ValueError(
                f"psd_factor.shape {self.psd_factor.shape} must be "
                f"{(n_mu, retained_rank)} (n_mu, retained_rank)."
            )


# ---------------------------------------------------------------------------
# Tiled, device-resident JAX direct backend. The whole orientation is built
# inside ONE jax.jit via lax.fori_loop over eval/nu/source/mu tiles, so
# short/padded blocks never trigger shape-dependent recompiles. Only bounded
# tiles (rhat (E,S,3), field V (Nn,3,E)) are formed -- never an (Ng,Ng),
# (3,Ng,Ng), or full V(nu,3,Ng) array. Every large array stays a device
# jax.Array; only rank-0 reduced scalars cross to the host, through
# _ibp_sync_scalars.
# ---------------------------------------------------------------------------


def _ibp_sync_scalars(**device_values):
    """The single auditable device->host synchronization boundary for the JAX
    backend. Each value must already be a rank-0 (scalar) device array -- a
    reduced diagnostic, never a large array. Returns a dict of Python scalars."""
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

        def nu_body(nt, Z):
            n0 = nt * Nn

            def src_body(st, V):
                j0 = st * S
                coords_s = lax.dynamic_slice(coords, (j0, 0), (S, 3))
                we_s = lax.dynamic_slice(w_eff, (j0,), (S,))
                # Slice theta DIRECTLY to (Nn,S) -- never materialize (Nn,ng_pad).
                theta_ns = lax.dynamic_slice(theta, (n0, j0), (Nn, S))
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
                # Slice grad DIRECTLY to (Nm,3,E) -- never materialize (n_mu_pad,3,E).
                grad_mt = lax.dynamic_slice(grad, (m0, 0, i0), (Nm, 3, E))
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
    never forming the (n_pair, n_pair) matrix. Backend-agnostic."""
    xp = jnp if isinstance(a, jax.Array) else np
    m = a.conj().T @ g @ a @ g
    return xp.real(xp.trace(m))


def psd_factorize_jax(z_hermitian, *, rtol=_DEFAULT_PSD_RTOL):
    """Device-resident Hermitian PSD factorization mirroring psd_factorize:
    device eigh, Hermiticity gate, material-negative hard-fail, roundoff clip,
    compact device W, zero-core -> rank 0. Only rank-0 reduced scalars cross to
    the host (through _ibp_sync_scalars); the eigenvalue vector and W stay
    device-resident."""
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
        factor=W, rtol=rtol,
        raw_min_eigenvalue=diag["raw_min"], spectral_scale=diag["spectral_scale"],
        negative_mode_count=int(diag["negative_mode_count"]),
        clipped_mode_count=int(diag["negative_mode_count"]),
        clipped_absolute_weight=diag["clipped_absolute_weight"],
        retained_rank=retained_rank, reconstruction_residual=recon_residual,
    )


def _ibp_one_sided_block(grad_theta_source, theta_source, grid, *,
                         mu_block, nu_block, eval_block_size, source_block_size):
    """Assemble the full (n_mu, n_nu) one-sided Z via kernel(), tiled over the
    pivot (mu/nu) axes at this level -- kernel() itself only tiles over the GRID
    axis, so a large pivot count needs this outer blocking to bound the size of
    kernel()'s internal (n_nu, 3, n_eval)-shaped intermediate."""
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


def _grids_equal(g1, g2):
    """Numerical grid equality (no hashes): same backend/device/dtype/n_grid/
    coincident-point policy and array-equal coords/weights, so independently
    built grids with identical numerical data are compatible while a
    one-element mismatch is not. JAX arrays reduce equality on device to a
    single rank-0 bool through the scalar-sync boundary."""
    if g1 is g2:
        return True
    for attr in ("backend", "device", "dtype", "n_grid", "coincident_point_policy"):
        if getattr(g1, attr) != getattr(g2, attr):
            return False
    if g1.backend == "numpy":
        return (np.array_equal(g1.coords, g2.coords)
                and np.array_equal(g1.weights, g2.weights))
    if g1.coords.shape != g2.coords.shape or g1.weights.shape != g2.weights.shape:
        return False
    eq = _ibp_sync_scalars(
        c=jnp.all(g1.coords == g2.coords),
        w=jnp.all(g1.weights == g2.weights),
    )
    return eq["c"] and eq["w"]


def ibp_core(left, right=None, *, operator, symmetry_mode="two_sided_average",
             mu_block_size=None, nu_block_size=None, psd_rtol=_DEFAULT_PSD_RTOL):
    """Build an IBPCoreArtifact from one (same-sector) or two (cross-sector)
    IBPInterpolationSector artifacts, via kernel() applied to sector
    Theta/grad_Theta arrays.

    Args:
        left: sector for the same-sector core, or sector A for a cross core.
        right: None (same-sector: right_sector = left, no duplicated compute)
            or sector B (cross-sector).
        operator: IBPOperatorPlan, REQUIRED. Its bound grid must be the same
            grid object both sectors were built on; only method="direct" is
            implemented. The operator and both sectors must share one backend.
        symmetry_mode: "two_sided_average" (default) or "one_sided" (diagnostic).
        mu_block_size/nu_block_size: bound the pivot-axis blocking; None
            resolves to each sector's full selected_rank.
        psd_rtol: roundoff band for the same-sector two-sided PSD gate.
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

    same_sector = right is None
    right_sector = left if same_sector else right

    for sector, label in ((left, "left"), (right_sector, "right")):
        if sector.backend != backend:
            raise ValueError(
                f"{label} sector.backend={sector.backend!r} must match "
                f"operator.backend={backend!r} -- a core cannot mix backends."
            )
        if not _grids_equal(sector.grid, operator.grid):
            raise ValueError(
                f"{label} sector's bound grid does not match the operator's grid -- a core "
                f"cannot join sectors/operators built on different grids."
            )

    n_mu = left.selected_rank
    n_nu = right_sector.selected_rank
    # Clamp the requested pivot block to the realized rank: a block larger than
    # the rank produces exactly one rank-sized block, so recording the oversized
    # request would misreport the realized tiling.
    mu_block = (
        min(_validate_positive_int("mu_block_size", mu_block_size), n_mu)
        if mu_block_size is not None else n_mu
    )
    nu_block = (
        min(_validate_positive_int("nu_block_size", nu_block_size), n_nu)
        if nu_block_size is not None else n_nu
    )

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

    # Reject non-finite raw orientations immediately after assembly, before any
    # norm/eigendecomposition would otherwise propagate nan/inf silently.
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
    # the sector's interpolation matrix P and measure that pair metric's
    # departure from Hermiticity.
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
    # core. Material negative modes hard-fail here rather than being repaired.
    if same_sector and symmetry_mode == "two_sided_average":
        psd = psd_factorize(Z, rtol=psd_rtol) if backend == "numpy" else psd_factorize_jax(Z, rtol=psd_rtol)
        psd_status = "factorized"
        psd_rtol_value = psd.rtol
        psd_factor = psd.factor
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
        psd_raw_min_eigenvalue = None
        psd_spectral_scale = None
        psd_negative_mode_count = None
        psd_clipped_mode_count = None
        psd_clipped_absolute_weight = None
        psd_retained_rank = None
        psd_reconstruction_residual = None

    if backend == "numpy":
        device = "cpu"
    else:
        device = str(Z.device)
        if str(operator.grid.device) != device:
            raise ValueError(
                f"JAX core output device {device!r} does not match the bound grid device "
                f"{operator.grid.device!r}."
            )

    return IBPCoreArtifact(
        Z=Z, same_sector=same_sector, symmetry_mode=symmetry_mode,
        z_forward_one_sided=z_forward, z_reverse_one_sided=z_reverse,
        raw_dagger_residual=raw_dagger_residual,
        raw_packed_pair_metric_dagger_residual=raw_packed_pair_metric_dagger_residual,
        coincident_pairs=coincident_pairs, n_mu=n_mu, n_nu=n_nu,
        psd_status=psd_status, psd_rtol=psd_rtol_value, psd_factor=psd_factor,
        psd_raw_min_eigenvalue=psd_raw_min_eigenvalue,
        psd_spectral_scale=psd_spectral_scale,
        psd_negative_mode_count=psd_negative_mode_count,
        psd_clipped_mode_count=psd_clipped_mode_count,
        psd_clipped_absolute_weight=psd_clipped_absolute_weight,
        psd_retained_rank=psd_retained_rank,
        psd_reconstruction_residual=psd_reconstruction_residual,
        mu_block_size=mu_block, nu_block_size=nu_block,
        eval_block_size=operator.eval_block_size,
        source_block_size=operator.source_block_size,
        backend=backend, device=device, realized_dtype=str(Z.dtype),
    )
