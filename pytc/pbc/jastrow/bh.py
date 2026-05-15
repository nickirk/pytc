"""Boys-Handy Jastrow with minimum-image e-n and e-e distances.

The molecular :class:`pytc.jastrow.BoysHandy` reads only ``atom_coords``,
``atom_charges``, and ``nelectron`` from the input system, all of which a
``pyscf.pbc.gto.Cell`` exposes identically. Spline-free construction is
fully reused.

The only physical change is that the two scaled-distance helpers
``_scaled_r_en`` and ``_scaled_r_ee`` use minimum-image displacements so
electrons near a cell face see the closest image of each nucleus / of the
other electron.
"""

from dataclasses import fields

import jax
import jax.numpy as jnp
from flax import struct

from pytc.jastrow.bh import BoysHandy as MolBoysHandy

from ..utils import mic_displacement


@struct.dataclass
class BoysHandy(MolBoysHandy):
    """Periodic Boys-Handy Jastrow.

    Adds a ``lattice`` field. Overrides the two distance helpers so that
    ``r_electron - r_nuclear`` and ``r1 - r2`` are evaluated via
    :func:`pytc.pbc.utils.mic_displacement`. All other methods —
    ``_compute_forward``, ``_compute``, ``init_params``, parameter
    flatten/unflatten, ``get_log_grads_r1``/``r2`` inherited from the
    :class:`Jastrow` base — are inherited and pick up the periodic
    distances through dynamic dispatch on ``self._scaled_r_en`` and
    ``self._scaled_r_ee``.
    """

    lattice: jax.Array = None

    @classmethod
    def create(cls, cell, terms_per_nucleus=None, epsilon=1e-16, name=None):
        """Construct a periodic Boys-Handy Jastrow from a built cell."""
        mol_bh = MolBoysHandy.create(
            cell,
            terms_per_nucleus=terms_per_nucleus,
            epsilon=epsilon,
            name=name,
        )
        return cls(
            lattice=jnp.asarray(cell.lattice_vectors()),
            **{f.name: getattr(mol_bh, f.name) for f in fields(mol_bh)},
        )

    def _scaled_r_en(self, r_electron, r_nuclear, b):
        dr = mic_displacement(r_electron, r_nuclear, self.lattice)
        r = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + self.epsilon)
        return r * b / (1.0 + r * b)

    def _scaled_r_ee(self, r1, r2, d):
        dr = mic_displacement(r1, r2, self.lattice)
        r = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + self.epsilon)
        return r * d / (1.0 + r * d)
