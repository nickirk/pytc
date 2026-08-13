"""Separable periodic Boys--Handy gradient channels.

The decomposition is exact on a uniform Gamma mesh.  It keeps every nuclear
centre as a distinct envelope label, groups only coefficients belonging to
the same atom-type/radial-power family, and compresses each realized
coefficient matrix by an explicit singular-value decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn

from .fft_tc import _mesh_tuple, _source_indices, _validate_uniform_grid


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoysHandyFamily:
    """One radial family and its realized envelope coefficient matrix."""

    atom_type: int
    radial_power: int
    coefficient: np.ndarray
    algebraic_rank: int
    numerical_rank: int
    envelope_numerical_rank: int
    rank_threshold: float
    smallest_kept_singular: float
    largest_discarded_singular: float
    kept_to_discarded_gap: float
    envelope_rank_threshold: float
    envelope_smallest_kept_singular: float
    envelope_largest_discarded_singular: float
    envelope_kept_to_discarded_gap: float


@dataclass(frozen=True)
class _GradientChannel:
    kind: str
    family: tuple[int, int]
    coefficient_scale: float
    left: jax.Array
    right: jax.Array
    radial: jax.Array


@dataclass(frozen=True)
class BoysHandyChannelPlan:
    """Prepared coefficient/radial channels for one fixed mesh and Jastrow."""

    mesh: tuple[int, int, int]
    n_grid: int
    channels: tuple[_GradientChannel, ...]
    families: tuple[BoysHandyFamily, ...]
    kernel_hats: dict


def channel_plan_nbytes(plan):
    """Return unique array storage referenced by a prepared plan."""
    arrays = {}
    for channel in plan.channels:
        for array in (channel.left, channel.right, channel.radial):
            arrays[id(array)] = array
    for array in plan.kernel_hats.values():
        arrays[id(array)] = array
    return sum(array.size * array.dtype.itemsize for array in arrays.values())


def _envelope_labels(jastrow_factor):
    labels = [(None, 0)]
    for atom in range(jastrow_factor.natom):
        atom_type = int(jastrow_factor.atom_type_map[atom])
        mask = np.asarray(jastrow_factor._cusp_mask[atom_type])
        for term, is_cusp in enumerate(mask):
            if is_cusp:
                continue
            for power in (
                int(jastrow_factor._term_m[atom_type, term]),
                int(jastrow_factor._term_n[atom_type, term]),
            ):
                label = (atom, power)
                if power and label not in labels:
                    labels.append(label)
    return tuple(labels)


def _singular_value_receipt(singular_values, rank_rtol):
    singular_values = np.abs(np.asarray(singular_values, dtype=float))
    scale = float(np.max(singular_values, initial=0.0))
    threshold = rank_rtol * scale
    keep = singular_values > threshold
    kept = singular_values[keep]
    discarded = singular_values[~keep]
    smallest_kept = float(np.min(kept, initial=np.inf))
    largest_discarded = float(np.max(discarded, initial=0.0))
    gap = (
        float(smallest_kept / largest_discarded)
        if largest_discarded > 0.0
        else float("inf")
    )
    return {
        "rank": int(np.count_nonzero(keep)),
        "threshold": threshold,
        "smallest_kept": smallest_kept,
        "largest_discarded": largest_discarded,
        "gap": gap,
    }


def boys_handy_coefficient_families(jastrow_factor, params, *, rank_rtol=1e-12):
    """Return exact coefficient matrices grouped by ``(atom_type, o)``."""
    if not 0 <= rank_rtol < 1:
        raise ValueError("rank_rtol must lie in [0, 1)")
    labels = _envelope_labels(jastrow_factor)
    label_index = {label: idx for idx, label in enumerate(labels)}
    c_raw = np.asarray(params["c_raw"], dtype=float)
    d = np.asarray(nn.softplus(params["d_raw"]), dtype=float)
    matrices = {}

    for atom in range(jastrow_factor.natom):
        atom_type = int(jastrow_factor.atom_type_map[atom])
        for term in range(jastrow_factor.n_terms):
            m = int(jastrow_factor._term_m[atom_type, term])
            n = int(jastrow_factor._term_n[atom_type, term])
            o = int(jastrow_factor._term_o[atom_type, term])
            key = (atom_type, o)
            matrix = matrices.setdefault(key, np.zeros((len(labels), len(labels))))
            if bool(jastrow_factor._cusp_mask[atom_type, term]):
                # delta=1/2, c=1/(2*N), cusp value=2*k/d.
                matrix[0, 0] += 0.5 / (jastrow_factor.natom * d[atom_type])
                continue
            weight = float(jastrow_factor._delta_factor[atom_type, term]) * c_raw[
                atom_type, term
            ]
            i = label_index[(atom, m)] if m else 0
            j = label_index[(atom, n)] if n else 0
            matrix[i, j] += weight
            matrix[j, i] += weight

    families = []
    for (atom_type, power), matrix in sorted(matrices.items()):
        algebraic_rank = int(np.linalg.matrix_rank(matrix))
        rank_receipt = _singular_value_receipt(
            np.linalg.svd(matrix, compute_uv=False), rank_rtol
        )
        envelope_matrix = matrix.copy()
        envelope_matrix[0, :] = 0.0
        envelope_receipt = _singular_value_receipt(
            np.linalg.svd(envelope_matrix, compute_uv=False), rank_rtol
        )
        families.append(
            BoysHandyFamily(
                atom_type=atom_type,
                radial_power=power,
                coefficient=matrix,
                algebraic_rank=algebraic_rank,
                numerical_rank=rank_receipt["rank"],
                envelope_numerical_rank=envelope_receipt["rank"],
                rank_threshold=rank_receipt["threshold"],
                smallest_kept_singular=rank_receipt["smallest_kept"],
                largest_discarded_singular=rank_receipt[
                    "largest_discarded"
                ],
                kept_to_discarded_gap=rank_receipt["gap"],
                envelope_rank_threshold=envelope_receipt["threshold"],
                envelope_smallest_kept_singular=envelope_receipt[
                    "smallest_kept"
                ],
                envelope_largest_discarded_singular=envelope_receipt[
                    "largest_discarded"
                ],
                envelope_kept_to_discarded_gap=envelope_receipt["gap"],
            )
        )
    return labels, tuple(families)


def _envelopes(jastrow_factor, params, grid, labels):
    b = nn.softplus(params["b_raw"])
    values = []
    gradients = []
    for atom, power in labels:
        if atom is None:
            values.append(jnp.ones(grid.shape[0]))
            gradients.append(jnp.zeros((grid.shape[0], 3)))
            continue
        atom_type = jastrow_factor.atom_type_map[atom]
        nucleus = jastrow_factor.nuclear_pos[atom]

        def envelope(point):
            return jastrow_factor._scaled_r_en(point, nucleus, b[atom_type]) ** power

        values.append(jax.vmap(envelope)(grid))
        gradients.append(jax.vmap(jax.grad(envelope))(grid))
    return jnp.stack(values), jnp.stack(gradients)


def _radial_kernel(jastrow_factor, params, grid, mesh, atom_type, power):
    d = nn.softplus(params["d_raw"])[atom_type]
    source = jnp.asarray(_source_indices(0, mesh))
    left = grid[0]
    right = grid[source]

    def radial(point, other):
        return jastrow_factor._scaled_r_ee(point, other, d) ** power

    values = jax.vmap(radial, in_axes=(None, 0))(left, right)
    gradients = jax.vmap(jax.grad(radial), in_axes=(None, 0))(left, right)
    return values.reshape(mesh), gradients.reshape(mesh + (3,))


def _factor_matrix(matrix, rank_rtol):
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    scale = float(np.max(np.abs(singular_values), initial=0.0))
    keep = np.abs(singular_values) > rank_rtol * scale
    return (
        left[:, keep] * singular_values[keep],
        right_h[keep],
        singular_values[keep],
    )


def _channels(
    jastrow_factor,
    params,
    grid,
    mesh,
    *,
    rank_rtol,
):
    labels, families = boys_handy_coefficient_families(
        jastrow_factor, params, rank_rtol=rank_rtol
    )
    envelopes, envelope_gradients = _envelopes(
        jastrow_factor, params, grid, labels
    )
    channels = []
    for family in families:
        family_key = (family.atom_type, family.radial_power)
        radial_scalar, radial_vector = _radial_kernel(
            jastrow_factor,
            params,
            grid,
            mesh,
            family.atom_type,
            family.radial_power,
        )

        # The constant envelope has identically zero gradient.  Factoring the
        # corresponding row-zeroed matrix separately avoids emitting exact
        # zero channels while retaining every same-centre coefficient.
        envelope_matrix = family.coefficient.copy()
        envelope_matrix[0, :] = 0.0
        left_coefficients, right_coefficients, singular_values = _factor_matrix(
            envelope_matrix, rank_rtol
        )
        for left_coefficient, right_coefficient, singular_value in zip(
            left_coefficients.T, right_coefficients, singular_values
        ):
            left_vector = jnp.einsum(
                "i,igc->gc", jnp.asarray(left_coefficient), envelope_gradients
            )
            right = jnp.einsum(
                "i,ig->g", jnp.asarray(right_coefficient), envelopes
            )
            channels.append(
                _GradientChannel(
                    "envelope",
                    family_key,
                    float(singular_value),
                    left_vector,
                    right,
                    radial_scalar,
                )
            )

        # d(s_0)/dx is exactly zero, so o=0 has no radial-gradient channels.
        if family.radial_power == 0:
            continue
        left_coefficients, right_coefficients, singular_values = _factor_matrix(
            family.coefficient, rank_rtol
        )
        for left_coefficient, right_coefficient, singular_value in zip(
            left_coefficients.T, right_coefficients, singular_values
        ):
            left_scalar = jnp.einsum(
                "i,ig->g", jnp.asarray(left_coefficient), envelopes
            )
            right = jnp.einsum(
                "i,ig->g", jnp.asarray(right_coefficient), envelopes
            )
            channels.append(
                _GradientChannel(
                    "radial",
                    family_key,
                    float(singular_value),
                    left_scalar,
                    right,
                    radial_vector,
                )
            )
    return tuple(channels), families


def _scalar_kernel_hat(kernel, mesh):
    axes = (-3, -2, -1)
    return jnp.fft.fftn(kernel, axes=axes)


def _vector_kernel_hat(kernel, mesh):
    axes = (-3, -2, -1)
    return jnp.fft.fftn(jnp.moveaxis(kernel, -1, 0), axes=axes)


def _convolve_scalar_hat(kernel_hat, rhs, mesh):
    axes = (-3, -2, -1)
    rhs_hat = jnp.fft.fftn(rhs.reshape((-1,) + mesh), axes=axes)
    return jnp.fft.ifftn(kernel_hat[None, ...] * rhs_hat, axes=axes).reshape(
        rhs.shape
    )


def _convolve_vector_hat(kernel_hat, rhs, mesh):
    axes = (-3, -2, -1)
    rhs_hat = jnp.fft.fftn(rhs.reshape((-1,) + mesh), axes=axes)
    result = jnp.fft.ifftn(
        kernel_hat[:, None, ...] * rhs_hat[None, ...], axes=axes
    )
    return jnp.moveaxis(result.reshape((3,) + rhs.shape), 0, -1)


def _get_kernel_hat(cache, key, compute):
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def _prepare_kernel_hats(channels, mesh):
    kernel_hats = {}
    for channel in channels:
        if channel.kind == "envelope":
            key = ("gradient", "envelope", channel.family)
            _get_kernel_hat(
                kernel_hats,
                key,
                lambda: _scalar_kernel_hat(channel.radial, mesh),
            )
        else:
            key = ("gradient", "radial", channel.family)
            _get_kernel_hat(
                kernel_hats,
                key,
                lambda: _vector_kernel_hat(channel.radial, mesh),
            )

    for q, left in enumerate(channels):
        for r in range(q, len(channels)):
            other = channels[r]
            if left.kind == "envelope" and other.kind == "envelope":
                key = (
                    "k3",
                    "envelope-envelope",
                    tuple(sorted((left.family, other.family))),
                )
                _get_kernel_hat(
                    kernel_hats,
                    key,
                    lambda: _scalar_kernel_hat(
                        left.radial * other.radial, mesh
                    ),
                )
            elif left.kind == "radial" and other.kind == "radial":
                key = (
                    "k3",
                    "radial-radial",
                    tuple(sorted((left.family, other.family))),
                )
                _get_kernel_hat(
                    kernel_hats,
                    key,
                    lambda: _scalar_kernel_hat(
                        jnp.sum(left.radial * other.radial, axis=-1), mesh
                    ),
                )
            else:
                envelope = left if left.kind == "envelope" else other
                radial = other if left.kind == "envelope" else left
                key = (
                    "k3",
                    "envelope-radial",
                    envelope.family,
                    radial.family,
                )
                _get_kernel_hat(
                    kernel_hats,
                    key,
                    lambda: _vector_kernel_hat(
                        envelope.radial[..., None] * radial.radial, mesh
                    ),
                )
    return kernel_hats


def prepare_boys_handy_channels(
    grid_points,
    weights,
    mesh,
    jastrow_factor,
    jastrow_params,
    *,
    rank_rtol=1e-12,
):
    """Prepare bounded channel data without constructing a grid-pair tensor."""
    mesh = _validate_uniform_grid(grid_points, weights, mesh)
    grid = jnp.asarray(grid_points)
    channels, families = _channels(
        jastrow_factor, jastrow_params, grid, mesh, rank_rtol=rank_rtol
    )
    for family in families:
        logger.info(
            "Boys-Handy channel rank: atom_type=%d radial_power=%d "
            "full=%d threshold=%.3e smallest_kept=%.3e "
            "largest_discarded=%.3e gap=%.3e; envelope=%d "
            "threshold=%.3e smallest_kept=%.3e largest_discarded=%.3e "
            "gap=%.3e",
            family.atom_type,
            family.radial_power,
            family.numerical_rank,
            family.rank_threshold,
            family.smallest_kept_singular,
            family.largest_discarded_singular,
            family.kept_to_discarded_gap,
            family.envelope_numerical_rank,
            family.envelope_rank_threshold,
            family.envelope_smallest_kept_singular,
            family.envelope_largest_discarded_singular,
            family.envelope_kept_to_discarded_gap,
        )
    return BoysHandyChannelPlan(
        mesh=mesh,
        n_grid=grid.shape[0],
        channels=channels,
        families=families,
        kernel_hats=_prepare_kernel_hats(channels, mesh),
    )


def apply_boys_handy_channel_plan(
    plan,
    weights,
    right_functions,
    *,
    squared_gradient=False,
    k3_rank_rtol=0.0,
):
    """Apply a prepared channel plan to right-grid functions."""
    if not 0 <= k3_rank_rtol < 1:
        raise ValueError("k3_rank_rtol must lie in [0, 1)")
    mesh = plan.mesh
    weights = jnp.asarray(weights)
    right = jnp.asarray(right_functions)
    if right.ndim == 1:
        right = right[None, :]
    if right.ndim != 2 or right.shape[1] != plan.n_grid:
        raise ValueError("right_functions must have shape (n_rhs, n_grid)")
    if weights.shape != (plan.n_grid,):
        raise ValueError("weights must have shape (n_grid,)")
    channels = plan.channels
    families = plan.families
    kernel_hats = plan.kernel_hats
    used_kernel_keys = set()

    if not squared_gradient:
        result = jnp.zeros((right.shape[0], plan.n_grid, 3), dtype=right.dtype)
        for channel in channels:
            rhs = right * weights[None, :] * channel.right[None, :]
            if channel.kind == "envelope":
                key = ("gradient", "envelope", channel.family)
                used_kernel_keys.add(key)
                kernel_hat = kernel_hats[key]
                convolved = _convolve_scalar_hat(kernel_hat, rhs, mesh)
                result += convolved[..., None] * channel.left[None, ...]
            else:
                key = ("gradient", "radial", channel.family)
                used_kernel_keys.add(key)
                kernel_hat = kernel_hats[key]
                convolved = _convolve_vector_hat(kernel_hat, rhs, mesh)
                result += convolved * channel.left[None, ..., None]
    else:
        result = jnp.zeros((right.shape[0], plan.n_grid), dtype=right.dtype)
        max_scale = max(
            (channel.coefficient_scale for channel in channels), default=0.0
        )
        cross_threshold = k3_rank_rtol * max_scale * max_scale
        retained_cross_count = 0
        for q, left in enumerate(channels):
            for r in range(q, len(channels)):
                other = channels[r]
                if (
                    left.coefficient_scale * other.coefficient_scale
                    <= cross_threshold
                ):
                    continue
                retained_cross_count += 1
                rhs = right * weights[None, :] * (left.right * other.right)[None, :]
                symmetry = 1.0 if q == r else 2.0
                if left.kind == "envelope" and other.kind == "envelope":
                    key = (
                        "k3",
                        "envelope-envelope",
                        tuple(sorted((left.family, other.family))),
                    )
                    used_kernel_keys.add(key)
                    kernel_hat = kernel_hats[key]
                    convolved = _convolve_scalar_hat(kernel_hat, rhs, mesh)
                    left_factor = jnp.sum(left.left * other.left, axis=-1)
                    result += symmetry * convolved * left_factor[None, :]
                elif left.kind == "radial" and other.kind == "radial":
                    key = (
                        "k3",
                        "radial-radial",
                        tuple(sorted((left.family, other.family))),
                    )
                    used_kernel_keys.add(key)
                    kernel_hat = kernel_hats[key]
                    convolved = _convolve_scalar_hat(kernel_hat, rhs, mesh)
                    result += symmetry * convolved * (left.left * other.left)[None, :]
                else:
                    envelope = left if left.kind == "envelope" else other
                    radial = other if left.kind == "envelope" else left
                    key = (
                        "k3",
                        "envelope-radial",
                        envelope.family,
                        radial.family,
                    )
                    used_kernel_keys.add(key)
                    kernel_hat = kernel_hats[key]
                    convolved = _convolve_vector_hat(kernel_hat, rhs, mesh)
                    left_factor = envelope.left * radial.left[:, None]
                    result += symmetry * jnp.sum(
                        convolved * left_factor[None, ...], axis=-1
                    )

    if not jnp.iscomplexobj(right):
        result = result.real
    diagnostics = {
        "n_families": len(families),
        "n_gradient_channels": len(channels),
        "raw_k3_cross_count": len(channels) * (len(channels) + 1) // 2,
        "retained_k3_cross_count": (
            retained_cross_count if squared_gradient else None
        ),
        "k3_rank_rtol": k3_rank_rtol,
        "n_radial_kernel_transforms": len(used_kernel_keys),
        "family_ranks": tuple(
            (
                family.atom_type,
                family.radial_power,
                family.envelope_numerical_rank,
                family.numerical_rank if family.radial_power else 0,
            )
            for family in families
        ),
        "family_rank_receipts": tuple(
            (
                family.atom_type,
                family.radial_power,
                family.rank_threshold,
                family.smallest_kept_singular,
                family.largest_discarded_singular,
                family.kept_to_discarded_gap,
                family.envelope_rank_threshold,
                family.envelope_smallest_kept_singular,
                family.envelope_largest_discarded_singular,
                family.envelope_kept_to_discarded_gap,
            )
            for family in families
        ),
    }
    return result, diagnostics


def apply_boys_handy_channels(
    grid_points,
    weights,
    mesh,
    jastrow_factor,
    jastrow_params,
    right_functions,
    *,
    squared_gradient=False,
    rank_rtol=1e-12,
    k3_rank_rtol=0.0,
):
    """Apply ``grad_1 u`` or ``|grad_1 u|^2`` without a grid-pair tensor."""
    plan = prepare_boys_handy_channels(
        grid_points,
        weights,
        mesh,
        jastrow_factor,
        jastrow_params,
        rank_rtol=rank_rtol,
    )
    return apply_boys_handy_channel_plan(
        plan,
        weights,
        right_functions,
        squared_gradient=squared_gradient,
        k3_rank_rtol=k3_rank_rtol,
    )
