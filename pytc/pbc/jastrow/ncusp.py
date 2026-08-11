"""Minimum-image periodic nuclear-cusp Jastrow factor."""

from dataclasses import fields

import jax
import jax.numpy as jnp
from flax import struct

from pytc.jastrow.ncusp import NuclearCusp as MolecularNuclearCusp

from ..utils import mic_displacement


@struct.dataclass
class NuclearCusp(MolecularNuclearCusp):
    """Nuclear-cusp correction with periodic electron--nucleus distances."""

    lattice: jax.Array = None

    @classmethod
    def create(cls, cell, name=None, n_radial=1000):
        molecular = MolecularNuclearCusp.create(cell, name=name, n_radial=n_radial)
        return cls(
            lattice=jnp.asarray(cell.lattice_vectors()),
            **{field.name: getattr(molecular, field.name) for field in fields(molecular)},
        )

    def _compute_inner(self, r1, r2, clipped_params, poly_coeffs):
        del r2

        def contribution(nucleus_index):
            displacement = mic_displacement(
                r1, self.coords[nucleus_index], self.lattice
            )
            distance = jnp.sqrt(jnp.sum(displacement**2) + 1e-16)
            charge = self.charges[nucleus_index]
            type_index = self.Z_to_idx[charge.astype(jnp.int32)]
            cutoff_radius = clipped_params["rc"][type_index]
            coefficients = poly_coeffs[type_index]
            polynomial = jnp.where(
                distance <= cutoff_radius,
                self._eval_poly(distance, coefficients),
                0.0,
            )
            phi_s = jnp.where(
                distance <= cutoff_radius,
                self.eval_mo_at_r(nucleus_index, distance),
                1.0,
            )
            log_term = polynomial - jnp.log(jnp.maximum(jnp.abs(phi_s), 1e-16))
            cutoff = self._cutoff_function(distance, cutoff_radius)
            return jnp.where(distance <= cutoff_radius, log_term * cutoff, 0.0)

        values = jax.vmap(contribution)(jnp.arange(self.n_nuclei))
        return jnp.sum(values) / (self.nelectron - 1)
