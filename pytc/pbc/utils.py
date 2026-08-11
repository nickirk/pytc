"""JAX lattice helpers for periodic integral oracles."""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def _wrap_to_half(frac):
    return frac - jnp.round(frac)


@_wrap_to_half.defjvp
def _wrap_to_half_jvp(primals, tangents):
    (frac,) = primals
    (tangent,) = tangents
    return _wrap_to_half(frac), tangent


def mic_displacement(r1, r2, lattice):
    """Return the minimum-image representative of ``r1 - r2``."""
    frac = (r1 - r2) @ jnp.linalg.inv(lattice)
    return _wrap_to_half(frac) @ lattice


def mic_distance(r1, r2, lattice):
    """Return the minimum-image distance between two positions."""
    return jnp.linalg.norm(mic_displacement(r1, r2, lattice), axis=-1)
