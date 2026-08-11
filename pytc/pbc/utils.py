"""JAX lattice helpers for periodic integral oracles."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import spglib


_NEIGHBOR_SHIFTS = jnp.asarray(
    np.stack(
        np.meshgrid(
            np.arange(-1, 2),
            np.arange(-1, 2),
            np.arange(-1, 2),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
)


def reduce_lattice(lattice):
    """Return a Niggli-reduced basis for the same translation lattice."""
    lattice = np.asarray(lattice, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("lattice must be a finite 3x3 array")
    if abs(np.linalg.det(lattice)) <= np.finfo(np.float64).eps:
        raise ValueError("lattice must be nonsingular")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Set OLD_ERROR_HANDLING to false.*",
            category=DeprecationWarning,
        )
        reduced = spglib.niggli_reduce(lattice)
    if reduced is None:
        raise ValueError("Niggli reduction failed for the supplied lattice")

    transform = reduced @ np.linalg.inv(lattice)
    integer_transform = np.rint(transform)
    if not np.allclose(transform, integer_transform, atol=1e-8, rtol=0.0):
        raise ValueError("reduced basis does not preserve the translation lattice")
    if not np.isclose(abs(np.linalg.det(integer_transform)), 1.0, atol=1e-8):
        raise ValueError("reduced basis transformation is not unimodular")
    return reduced


@jax.custom_jvp
def _wrap_to_half(frac):
    return frac - jnp.round(frac)


@_wrap_to_half.defjvp
def _wrap_to_half_jvp(primals, tangents):
    (frac,) = primals
    (tangent,) = tangents
    return _wrap_to_half(frac), tangent


@jax.custom_jvp
def mic_displacement(r1, r2, lattice):
    """Return the nearest image of ``r1 - r2`` in a reduced basis."""
    frac = (r1 - r2) @ jnp.linalg.inv(lattice)
    wrapped = _wrap_to_half(frac)
    candidates = (wrapped[..., None, :] - _NEIGHBOR_SHIFTS) @ lattice
    squared_distances = jnp.sum(candidates * candidates, axis=-1)
    nearest = jnp.argmin(squared_distances, axis=-1)
    return jnp.take_along_axis(
        candidates, nearest[..., None, None], axis=-2
    )[..., 0, :]


@mic_displacement.defjvp
def _mic_displacement_jvp(primals, tangents):
    r1, r2, lattice = primals
    tangent_r1, tangent_r2, _ = tangents
    return mic_displacement(r1, r2, lattice), tangent_r1 - tangent_r2


def mic_distance(r1, r2, lattice):
    """Return the minimum-image distance between two positions."""
    return jnp.linalg.norm(mic_displacement(r1, r2, lattice), axis=-1)
