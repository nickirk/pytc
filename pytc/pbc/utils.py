"""Lattice-derived primitives for periodic boundary conditions.

All functions take ``lattice`` as a ``(3, 3)`` array whose **rows** are the
lattice vectors ``a1, a2, a3``, matching the convention of
``pyscf.pbc.gto.Cell.lattice_vectors()``. A fractional coordinate ``f`` and
Cartesian coordinate ``r`` are related by ``r = f @ lattice``.

These primitives are pure JAX and may be called from inside ``jit`` / ``grad``.
"""

import numpy as np
import jax
import jax.numpy as jnp


def wrap(positions, lattice):
    """Fold Cartesian positions into the primitive cell.

    The wrapped positions have fractional coordinates in ``[0, 1)`` along each
    lattice direction. Works for any leading batch shape.

    Args:
        positions: Cartesian coordinates, shape ``(..., 3)``.
        lattice: Lattice matrix with rows ``a1, a2, a3``, shape ``(3, 3)``.

    Returns:
        Wrapped Cartesian coordinates, same shape as ``positions``.
    """
    inv_lat = jnp.linalg.inv(lattice)
    frac = positions @ inv_lat
    frac = frac - jnp.floor(frac)
    return frac @ lattice


@jax.custom_jvp
def _wrap_to_half(frac):
    """Map fractional coordinates to ``[-0.5, 0.5)`` via the nearest integer.

    Wrapped in ``custom_jvp`` because:
      * The function is piecewise the identity (gradient is 1 almost
        everywhere; the discontinuity set has measure zero in 3D).
      * ``jnp.round`` is not in folx's forward-Laplacian registry, which
        would otherwise trigger a full-Hessian fallback. With a custom
        JVP the round op is opaque to higher-order autodiff frameworks.
    """
    return frac - jnp.round(frac)


@_wrap_to_half.defjvp
def _wrap_to_half_jvp(primals, tangents):
    (x,) = primals
    (tx,) = tangents
    return _wrap_to_half(x), tx


def mic_displacement(r1, r2, lattice):
    """Minimum-image displacement vector ``r1 - r2``.

    Returns the representative of ``r1 - r2`` whose fractional coordinates
    lie in ``[-0.5, 0.5)``. For non-orthogonal cells the minimum-image
    convention is exact only when the cutoff radius is below the inscribed
    sphere of the Wigner-Seitz cell; callers requiring full accuracy at
    larger separations should sum over image vectors explicitly.

    Args:
        r1, r2: Cartesian coordinates with broadcast-compatible shapes
            ending in ``(..., 3)``.
        lattice: Lattice matrix, shape ``(3, 3)``.

    Returns:
        Displacement vector with same broadcast shape, ending in ``(..., 3)``.
    """
    inv_lat = jnp.linalg.inv(lattice)
    frac = (r1 - r2) @ inv_lat
    return _wrap_to_half(frac) @ lattice


def mic_distance(r1, r2, lattice):
    """Minimum-image distance ``|r1 - r2|``.

    Convenience wrapper around :func:`mic_displacement`.

    Args:
        r1, r2: Cartesian coordinates ending in ``(..., 3)``.
        lattice: Lattice matrix, shape ``(3, 3)``.

    Returns:
        Distances with shape equal to the broadcast of the leading axes.
    """
    return jnp.linalg.norm(mic_displacement(r1, r2, lattice), axis=-1)


def _perpendicular_widths(lattice):
    """Perpendicular spacing between adjacent replicas along each lattice direction.

    Returns ``h_i = V / |a_j x a_k|`` for ``(i, j, k)`` cyclic.
    """
    a1, a2, a3 = lattice[0], lattice[1], lattice[2]
    volume = jnp.abs(jnp.dot(a1, jnp.cross(a2, a3)))
    h = jnp.array([
        volume / jnp.linalg.norm(jnp.cross(a2, a3)),
        volume / jnp.linalg.norm(jnp.cross(a3, a1)),
        volume / jnp.linalg.norm(jnp.cross(a1, a2)),
    ])
    return h


def generate_images(lattice, rcut, include_origin=True):
    """Enumerate lattice translation vectors with ``|T| <= rcut``.

    Used for summing Bloch-summed orbital tails, nucleus image potentials,
    and the real-space part of Ewald sums.

    Computed eagerly with numpy (the result is a small static table whose
    contents and length depend on data, so it cannot live inside JIT).

    Args:
        lattice: Lattice matrix, shape ``(3, 3)``, with rows ``a1, a2, a3``.
            May be a JAX or NumPy array.
        rcut: Cutoff radius. Translation vectors with ``|T| > rcut`` are
            dropped.
        include_origin: Whether to include the zero vector in the output.

    Returns:
        ``(n_images, 3)`` NumPy array of Cartesian translation vectors,
        sorted by ascending ``|T|``.
    """
    lat = np.asarray(lattice)
    h = np.array([
        np.abs(np.linalg.det(lat)) / np.linalg.norm(np.cross(lat[1], lat[2])),
        np.abs(np.linalg.det(lat)) / np.linalg.norm(np.cross(lat[2], lat[0])),
        np.abs(np.linalg.det(lat)) / np.linalg.norm(np.cross(lat[0], lat[1])),
    ])
    n_max = np.ceil(rcut / h).astype(int)

    n1 = np.arange(-n_max[0], n_max[0] + 1)
    n2 = np.arange(-n_max[1], n_max[1] + 1)
    n3 = np.arange(-n_max[2], n_max[2] + 1)
    grid = np.stack(np.meshgrid(n1, n2, n3, indexing='ij'), axis=-1).reshape(-1, 3)
    images = grid @ lat
    norms = np.linalg.norm(images, axis=-1)

    mask = norms <= rcut
    if not include_origin:
        mask &= norms > 0
    images = images[mask]
    norms = norms[mask]
    order = np.argsort(norms)
    return images[order]
