"""Bloch-summed Cartesian GTO evaluator at a k-point mesh.

Extends :class:`pytc.pbc.ansatz.gto.GTO` (Gamma-only) to a general
k-point mesh. Each image translation T in the real-space lattice sum is
weighted by ``exp(+i k · T)`` so the resulting orbital satisfies the
Bloch condition

.. math::

    \\chi_k(r + a) = e^{i k \\cdot a} \\chi_k(r)

for any lattice vector ``a``. Phases are determined per (k-point, image)
pair. At k = 0 all phases are 1 and the evaluator reduces to the
Gamma-only :class:`GTO`.

The data layout matches :class:`GTO` (centers, ijk, expts, coeffs,
images) plus an extra ``kpts`` field of shape ``(Nk, 3)``. The returned
AO array carries a leading k-axis of size ``Nk``.

Sign convention: PySCF's ``pyscf.pbc.dft.numint.eval_ao_kpts`` uses
``e^{+i k T} \\chi(r - R - T)``. In our kernel ``ctr_xyz = (xyz -
center) + image``, so the effective translation evaluated is
``T = -image`` and the per-image phase is ``e^{-i k image}``. Tests in
test_kgto verify the empirical match against PySCF.
"""

import logging

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from pytc.ansatz.gto import MolGTO

from ..utils import generate_images
from .gto import default_rcut

logger = logging.getLogger(__name__)


def _cartesian_gto_kpts(centers, ijk, expts, coeffs, images, kpts, xyz):
    """Bloch-summed Cartesian GTO at a single field point.

    Args:
        centers: ``(n_ao, 3)`` orbital centers.
        ijk: ``(n_ao, 3)`` Cartesian angular-momentum exponents.
        expts: ``(n_ao, n_prim)`` primitive exponents (zero-padded).
        coeffs: ``(n_ao, n_prim)`` contraction coefficients (zero-padded).
        images: ``(n_image, 3)`` real-space lattice translations.
        kpts: ``(Nk, 3)`` k-points in Cartesian units (Bohr^-1).
        xyz: ``(3,)`` field point.

    Returns:
        ``(Nk, n_ao)`` complex array.
    """
    centers2d = jnp.atleast_2d(centers)
    ctr_xyz_first = xyz[jnp.newaxis, :] - centers2d                       # (n_ao, 3)
    ctr_xyz = ctr_xyz_first[:, jnp.newaxis, :] + images                   # (n_ao, n_image, 3)

    xyz_pow = ctr_xyz ** ijk[:, jnp.newaxis, :]
    xyz_ijk = jnp.prod(xyz_pow, axis=-1)                                  # (n_ao, n_image)

    r2 = jnp.sum(ctr_xyz ** 2, axis=-1)                                   # (n_ao, n_image)
    gauss = jnp.exp(-expts[:, jnp.newaxis, :] * r2[:, :, jnp.newaxis])    # (n_ao, n_image, n_prim)

    all_prod = coeffs[:, jnp.newaxis, :] * gauss * xyz_ijk[:, :, jnp.newaxis]
    per_image = jnp.sum(all_prod, axis=2)                                  # (n_ao, n_image)

    # Phase factor for each (k, image): exp(-i k . image). With T = -image
    # being the effective lattice translation, this realises the
    # +i k . T convention used by PySCF.
    k_dot_T = kpts @ images.T                                              # (Nk, n_image)
    phase = jnp.exp(-1j * k_dot_T)                                         # (Nk, n_image)

    # Bloch sum over images: (Nk, n_ao)
    return jnp.einsum('ki,ni->kn', phase, per_image.astype(phase.dtype))


@struct.dataclass
class KGTO:
    """Cartesian GTO evaluator with Bloch sums at a k-point mesh.

    All ``eval_*`` standalone functions below operate on this dataclass.
    At ``kpts = [[0, 0, 0]]`` the evaluator reduces to the Gamma-only
    :class:`GTO` (with a leading k-axis of size 1).
    """

    centers: jax.Array
    ijk: jax.Array
    expts: jax.Array
    coeffs: jax.Array
    images: jax.Array
    kpts: jax.Array
    cart: bool = struct.field(pytree_node=False, default=True)

    @classmethod
    def from_cell(cls, cell, kpts, rcut: float = None, precision: float = 1e-8):
        """Construct from a built ``pyscf.pbc.gto.Cell`` and a k-point mesh.

        Args:
            cell: Built Cell with ``cart=True``.
            kpts: ``(Nk, 3)`` array of k-points in Bohr^-1. Typically
                obtained from ``cell.make_kpts([n1, n2, n3])``.
            rcut: Image cutoff in Bohr. Defaults to :func:`default_rcut`.
            precision: Tolerance used to derive ``rcut`` when not given.
        """
        if not cell.cart:
            raise ValueError(
                "KGTO currently only supports Cartesian basis sets. "
                "Build the cell with cell.cart = True before .build()."
            )

        centers, ijk, expts, coeffs, _ = MolGTO._extract_params(cell)

        if rcut is None:
            rcut = default_rcut(cell, precision=precision)

        lattice = np.asarray(cell.lattice_vectors())
        images = generate_images(lattice, rcut)
        kpts_arr = np.atleast_2d(np.asarray(kpts))
        logger.debug(
            "KGTO: rcut=%.3f Bohr, %d images, %d k-points",
            rcut, images.shape[0], kpts_arr.shape[0]
        )

        return cls(
            centers=centers,
            ijk=ijk,
            expts=expts,
            coeffs=coeffs,
            images=jnp.asarray(images),
            kpts=jnp.asarray(kpts_arr),
            cart=cell.cart,
        )


def eval_gto(kgto: KGTO, xyz: jax.Array) -> jax.Array:
    """Evaluate Bloch AOs at a single point.

    Returns ``(Nk, n_ao)`` complex.
    """
    return _cartesian_gto_kpts(
        kgto.centers, kgto.ijk, kgto.expts, kgto.coeffs,
        kgto.images, kgto.kpts, xyz,
    )


def eval_gto_grad(kgto: KGTO, xyz: jax.Array) -> jax.Array:
    """Gradient of Bloch AOs w.r.t. the field point.

    Returns ``(Nk, n_ao, 3)`` complex.
    """
    return jax.jacfwd(lambda x: eval_gto(kgto, x))(xyz)


def eval_gto_lap(kgto: KGTO, xyz: jax.Array) -> jax.Array:
    """Laplacian of Bloch AOs w.r.t. the field point.

    Returns ``(Nk, n_ao)`` complex.
    """
    hess = jax.jacfwd(lambda x: eval_gto_grad(kgto, x))(xyz)
    # hess shape: (Nk, n_ao, 3, 3) — trace over the last two axes.
    return jnp.trace(hess, axis1=-2, axis2=-1)


def eval_ao(kgto: KGTO, pos: jax.Array, deriv: int = 0):
    """Batched Bloch AO evaluation.

    Args:
        kgto: Bloch AO evaluator.
        pos: ``(..., 3)`` field points.
        deriv: 0 = values only, 1 = values + gradient, 2 = values +
            gradient + laplacian.

    Returns:
        For deriv=0: array of shape ``(..., Nk, n_ao)`` complex.
        For deriv=1: ``(vals, grads)`` with shapes
            ``(..., Nk, n_ao)`` and ``(..., Nk, n_ao, 3)``.
        For deriv=2: ``(vals, grads, laps)`` with shapes as above and
            ``(..., Nk, n_ao)`` for the Laplacian.
    """
    batch_shape = pos.shape[:-1]
    pos_flat = pos.reshape(-1, 3)

    if deriv == 0:
        vals = jax.vmap(lambda x: eval_gto(kgto, x))(pos_flat)
        return vals.reshape(batch_shape + vals.shape[1:])

    if deriv == 1:
        def _val_and_grad(x):
            return eval_gto(kgto, x), eval_gto_grad(kgto, x)
        vals, grads = jax.vmap(_val_and_grad)(pos_flat)
        return (
            vals.reshape(batch_shape + vals.shape[1:]),
            grads.reshape(batch_shape + grads.shape[1:]),
        )

    if deriv == 2:
        def _all(x):
            return eval_gto(kgto, x), eval_gto_grad(kgto, x), eval_gto_lap(kgto, x)
        vals, grads, laps = jax.vmap(_all)(pos_flat)
        return (
            vals.reshape(batch_shape + vals.shape[1:]),
            grads.reshape(batch_shape + grads.shape[1:]),
            laps.reshape(batch_shape + laps.shape[1:]),
        )

    raise ValueError(f"Unsupported derivative order: {deriv}")
