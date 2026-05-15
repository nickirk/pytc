"""Ewald summation for the periodic Coulomb energy.

Provides JAX-traceable, JIT/grad-friendly Ewald sums used by the PBC
local-energy evaluator. Two primitives are exposed:

- :func:`ewald_self_energy` — Coulomb energy of a *single* charge set
  in a uniform neutralizing background (matches ``pyscf.pbc.gto.Cell.ewald``
  when applied to the cell's nuclei).
- :func:`ewald_cross_energy` — Coulomb interaction between two *distinct*
  charge sets, no self-correction or neutralizing background.

For a charge-neutral molecule (electrons + nuclei), the total potential
energy is conveniently computed as

.. code-block:: python

    V = ewald_self_energy(all_positions, all_charges, ewald)

where ``all_*`` concatenates electrons (charge -1 each) and nuclei.

The :class:`EwaldParams` carries the (real-space, reciprocal-space)
image tables and the splitting parameter ``alpha``. It is built eagerly
(numpy) from a lattice so the cutoff tables are static across JIT calls.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc
from flax import struct

from ..utils import generate_images


@struct.dataclass
class EwaldParams:
    """Static Ewald summation tables for a fixed cell.

    Fields:
        alpha: Splitting parameter (Bohr^-1).
        real_images: ``(n_real, 3)`` lattice translations T with |T| within
            the real-space cutoff. Includes T = 0.
        recip_vectors: ``(n_recip, 3)`` reciprocal lattice vectors G with
            |G| within the cutoff. Excludes G = 0.
        volume: Cell volume in Bohr^3.
    """

    alpha: float = struct.field(pytree_node=False)
    real_images: jax.Array
    recip_vectors: jax.Array
    volume: float = struct.field(pytree_node=False)


def _reciprocal_lattice(lattice):
    """Reciprocal lattice rows (with the 2π factor).

    For row-vector lattice ``A``, returns ``B = 2π (A^{-1})^T`` so that
    ``a_i · b_j = 2π δ_{ij}``.
    """
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice)).T


def make_ewald_params(lattice, alpha: float = None, precision: float = 1e-10) -> EwaldParams:
    """Build static Ewald tables for a cell.

    Args:
        lattice: ``(3, 3)`` lattice matrix with rows ``a1, a2, a3``.
        alpha: Splitting parameter. If ``None``, defaults to
            ``sqrt(pi) / V^{1/3}``, which balances real-space and
            reciprocal-space convergence for a roughly cubic cell.
        precision: Target precision; controls both the real-space and
            reciprocal-space cutoffs.

    Returns:
        :class:`EwaldParams`.
    """
    lattice_np = np.asarray(lattice)
    volume = float(np.abs(np.linalg.det(lattice_np)))

    if alpha is None:
        alpha = float(np.sqrt(np.pi) / volume ** (1.0 / 3.0))
    alpha = float(alpha)

    log_eps = -np.log(precision)

    # Real-space cutoff: erfc(α r)/r drops below ``precision`` for r
    # exceeding sqrt(log_eps)/α; add the cell circumscribed radius so the
    # worst-case in-cell separation is covered.
    rcut = float(np.sqrt(log_eps) / alpha) + float(np.linalg.norm(lattice_np.sum(axis=0)))
    real_images = generate_images(lattice_np, rcut, include_origin=True)

    # Reciprocal-space cutoff: exp(-G²/(4α²)) drops below precision for
    # G² > 4 α² log_eps.
    gcut = 2.0 * alpha * float(np.sqrt(log_eps))
    B = _reciprocal_lattice(lattice_np)
    recip_vectors = generate_images(B, gcut, include_origin=False)

    return EwaldParams(
        alpha=alpha,
        real_images=jnp.asarray(real_images),
        recip_vectors=jnp.asarray(recip_vectors),
        volume=volume,
    )


def _real_space_self(positions, charges, ewald: EwaldParams):
    """Real-space contribution for a single charge set.

    Returns ``(1/2) sum_{i,j,T; (i,j,T) ≠ (j,j,0)} q_i q_j erfc(α |r_ij - T|) / |r_ij - T|``.
    """
    n = positions.shape[0]
    r_ij = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    diag = jnp.eye(n, dtype=bool)

    def per_image(T):
        d = r_ij + T  # (n, n, 3)
        r2 = jnp.sum(d * d, axis=-1)
        T_is_zero = jnp.sum(T * T) < 1e-20
        # Mask out the (i=j, T=0) pair — would be divergent.
        skip = T_is_zero & diag
        r_safe = jnp.where(skip, 1.0, jnp.sqrt(r2 + 1e-30))
        v = jnp.where(skip, 0.0, erfc(ewald.alpha * r_safe) / r_safe)
        return v

    per_T = jax.vmap(per_image)(ewald.real_images)  # (n_T, n, n)
    qq = charges[:, None] * charges[None, :]  # (n, n)
    return 0.5 * jnp.sum(per_T * qq[None, :, :])


def _recip_space_self(positions, charges, ewald: EwaldParams):
    """Reciprocal-space contribution for a single charge set.

    Returns ``(2π / V) sum_{G ≠ 0} exp(-G²/(4α²)) |F(G)|² / G²`` where
    ``F(G) = sum_i q_i exp(iG·r_i)``.
    """
    G = ewald.recip_vectors  # (nG, 3)
    G_sq = jnp.sum(G * G, axis=-1)
    G_dot_r = G @ positions.T  # (nG, n)
    cos_F = jnp.cos(G_dot_r) @ charges  # (nG,)
    sin_F = jnp.sin(G_dot_r) @ charges
    F_sq = cos_F ** 2 + sin_F ** 2
    return (2.0 * jnp.pi / ewald.volume) * jnp.sum(
        jnp.exp(-G_sq / (4.0 * ewald.alpha ** 2)) / G_sq * F_sq
    )


def ewald_self_energy(positions, charges, ewald: EwaldParams):
    """Periodic Coulomb energy of a single charge set in a neutralizing background.

    Computes the Ewald decomposition of

    .. math::

        (1/2) \\sum_{(i,j,T) \\ne (j,j,0)} q_i q_j / |r_{ij} - T|

    For ``sum q_i ≠ 0`` the result is the energy in a uniform background
    that neutralises the cell (the implicit ``G = 0`` term is subtracted
    via the charge-neutrality correction).

    Args:
        positions: ``(n, 3)`` Cartesian positions.
        charges: ``(n,)`` charges. Floats; nuclei are positive, electrons
            negative.
        ewald: Cached Ewald tables for the cell.

    Returns:
        Scalar energy.
    """
    real = _real_space_self(positions, charges, ewald)
    recip = _recip_space_self(positions, charges, ewald)
    self_correction = -(ewald.alpha / jnp.sqrt(jnp.pi)) * jnp.sum(charges ** 2)
    total_charge_sq = jnp.sum(charges) ** 2
    neutral_correction = -jnp.pi * total_charge_sq / (2.0 * ewald.alpha ** 2 * ewald.volume)
    return real + recip + self_correction + neutral_correction


def ewald_cross_energy(r_a, q_a, r_b, q_b, ewald: EwaldParams):
    """Periodic Coulomb interaction between two distinct charge sets.

    Computes the Ewald decomposition of

    .. math::

        \\sum_{i,j,T} q^a_i q^b_j / |r^a_i - r^b_j - T|.

    No self-correction (the sets are assumed disjoint, so no point coincides).
    A ``G = 0`` cross-charge correction ``-π (Σ q_a)(Σ q_b) / (α² V)`` is
    applied so the cross energy is consistent with two
    :func:`ewald_self_energy` calls treating each set in a neutralizing
    background.

    Args:
        r_a, r_b: ``(Na, 3)`` and ``(Nb, 3)`` Cartesian positions.
        q_a, q_b: ``(Na,)`` and ``(Nb,)`` charges.
        ewald: Cached Ewald tables.

    Returns:
        Scalar cross-set energy.
    """
    r_ij = r_a[:, None, :] - r_b[None, :, :]  # (Na, Nb, 3)

    def per_image(T):
        d = r_ij + T
        r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)
        return erfc(ewald.alpha * r) / r

    per_T = jax.vmap(per_image)(ewald.real_images)  # (n_T, Na, Nb)
    qq = q_a[:, None] * q_b[None, :]
    real = jnp.sum(per_T * qq[None, :, :])

    G = ewald.recip_vectors
    G_sq = jnp.sum(G * G, axis=-1)
    G_dot_ra = G @ r_a.T
    G_dot_rb = G @ r_b.T
    cos_a = jnp.cos(G_dot_ra) @ q_a
    sin_a = jnp.sin(G_dot_ra) @ q_a
    cos_b = jnp.cos(G_dot_rb) @ q_b
    sin_b = jnp.sin(G_dot_rb) @ q_b
    cross_FF = cos_a * cos_b + sin_a * sin_b
    recip = (4.0 * jnp.pi / ewald.volume) * jnp.sum(
        jnp.exp(-G_sq / (4.0 * ewald.alpha ** 2)) / G_sq * cross_FF
    )

    total_charge_cross = jnp.sum(q_a) * jnp.sum(q_b)
    neutral_correction = -jnp.pi * total_charge_cross / (ewald.alpha ** 2 * ewald.volume)
    return real + recip + neutral_correction


def total_coulomb_energy(electron_positions, nuclear_positions, nuclear_charges, ewald: EwaldParams):
    """Total periodic Coulomb energy of electrons (charge -1) and nuclei.

    Convenience wrapper around :func:`ewald_self_energy` that builds a
    combined charge set. For a charge-neutral molecule (``n_electrons ==
    sum nuclear_charges``) the result is independent of any background
    convention.

    Args:
        electron_positions: ``(n_electrons, 3)``.
        nuclear_positions: ``(n_atoms, 3)``.
        nuclear_charges: ``(n_atoms,)``; will be cast to float.
        ewald: Cached Ewald tables.
    """
    n_elec = electron_positions.shape[0]
    all_pos = jnp.concatenate([electron_positions, nuclear_positions], axis=0)
    all_q = jnp.concatenate([
        -jnp.ones(n_elec, dtype=electron_positions.dtype),
        jnp.asarray(nuclear_charges, dtype=electron_positions.dtype),
    ])
    return ewald_self_energy(all_pos, all_q, ewald)
