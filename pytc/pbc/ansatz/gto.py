"""Bloch-summed Cartesian Gaussian-type orbitals at the Gamma point.

The molecular ``_cartesian_gto`` already implements a sum over an
``images`` array of center translations,

.. math::

    \\varphi(r) = \\sum_T \\varphi_\\mathrm{atom}(r - R_\\mathrm{atom} - T),

with the molecular case using a single zero image. For PBC at Gamma we
populate ``images`` with all lattice translations whose Cartesian length
is below a cutoff chosen so that the omitted tails of the most diffuse
primitive fall below ``precision``.

Only Cartesian basis sets are supported, matching the constraint in the
molecular :class:`pytc.ansatz.gto.MolGTO`.
"""

import logging

import numpy as np
import jax.numpy as jnp
from flax import struct

from pytc.ansatz.gto import MolGTO

from ..utils import generate_images

logger = logging.getLogger(__name__)


def default_rcut(cell, precision: float = 1e-8) -> float:
    """Conservative image-summation cutoff for a given cell and precision.

    The most diffuse Gaussian primitive in the basis has exponent
    ``alpha_min``; its amplitude drops below ``precision`` at radius
    ``r_atom = sqrt(-log(precision) / alpha_min)``. Field points lie
    inside the primitive cell, so the worst case requires keeping image
    centers at most ``r_atom`` from the *farthest* in-cell point — adding
    a circumscribed-sphere buffer gives a safe envelope.

    Args:
        cell: ``pyscf.pbc.gto.Cell``.
        precision: Target amplitude tolerance for omitted images.

    Returns:
        Suggested cutoff radius in Bohr.
    """
    alpha_min = float(min(min(p[0] for p in shell[1:]) for atom in cell._basis.values() for shell in atom))
    r_atom = float(np.sqrt(-np.log(precision) / alpha_min))
    lattice = np.asarray(cell.lattice_vectors())
    # Circumscribed sphere of the cell (over-estimate of in-cell distance).
    cell_diam = float(np.linalg.norm(lattice.sum(axis=0)))
    return r_atom + cell_diam


@struct.dataclass
class GTO(MolGTO):
    """Cartesian GTO evaluator with image-summed centers.

    Same data layout as :class:`MolGTO`; only ``images`` is populated
    differently. All evaluator functions in :mod:`pytc.ansatz.gto`
    (``eval_gto``, ``eval_ao``, gradient / laplacian helpers) work
    unchanged because the image sum is already inside ``_cartesian_gto``.
    """

    @classmethod
    def from_cell(cls, cell, rcut: float = None, precision: float = 1e-8):
        """Construct a PBC GTO evaluator from a ``pyscf.pbc.gto.Cell``.

        Args:
            cell: A built Cell with ``cart=True``.
            rcut: Image cutoff in Bohr. Defaults to :func:`default_rcut`.
            precision: Tolerance used to derive ``rcut`` when not given.
        """
        if not cell.cart:
            raise ValueError(
                "GTO currently only supports Cartesian basis sets. "
                "Build the cell with cell.cart = True before .build()."
            )

        centers, ijk, expts, coeffs, _ = MolGTO._extract_params(cell)

        if rcut is None:
            rcut = default_rcut(cell, precision=precision)

        lattice = np.asarray(cell.lattice_vectors())
        images = generate_images(lattice, rcut)
        logger.debug(
            "GTO: rcut=%.3f Bohr, %d image translations summed", rcut, images.shape[0]
        )

        return cls(
            centers=centers,
            ijk=ijk,
            expts=expts,
            coeffs=coeffs,
            images=jnp.asarray(images),
            cart=cell.cart,
        )
