"""Minimum-image periodic Boys--Handy Jastrow factor."""

from dataclasses import fields

import jax
import jax.numpy as jnp
from flax import struct

from pytc.jastrow.bh import BoysHandy as MolecularBoysHandy

from ..utils import mic_displacement


@struct.dataclass
class BoysHandy(MolecularBoysHandy):
    """Boys--Handy Jastrow with periodic electron distances."""

    lattice: jax.Array = None

    @classmethod
    def create(cls, cell, terms_per_nucleus=None, epsilon=1e-16, name=None):
        molecular = MolecularBoysHandy.create(
            cell,
            terms_per_nucleus=terms_per_nucleus,
            epsilon=epsilon,
            name=name,
        )
        return cls(
            lattice=jnp.asarray(cell.lattice_vectors()),
            **{field.name: getattr(molecular, field.name) for field in fields(molecular)},
        )

    def _scaled_r_en(self, r_electron, r_nuclear, b):
        displacement = mic_displacement(r_electron, r_nuclear, self.lattice)
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1) + self.epsilon)
        return distance * b / (1.0 + distance * b)

    def _scaled_r_ee(self, r1, r2, d):
        displacement = mic_displacement(r1, r2, self.lattice)
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1) + self.epsilon)
        return distance * d / (1.0 + distance * d)
