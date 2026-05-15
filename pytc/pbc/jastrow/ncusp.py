"""Nuclear cusp Jastrow with minimum-image electron-nucleus distance.

The molecular :class:`pytc.jastrow.NuclearCusp` reads only ``atom_coords``,
``atom_charges``, ``nelectron``, and s-type AO values on a radial grid via
``mol.eval_gto('GTOval_sph', ...)``. A built ``pyscf.pbc.gto.Cell``
exposes the same interface and returns identical atomic s-AO values, so
the entire spline construction logic is reused by calling
``MolNuclearCusp.create(cell)``.

The only physical change is the per-electron distance to each nucleus:
the molecular evaluator uses ``r1 - nucleus_coord``; the PBC version uses
:func:`pytc.pbc.utils.mic_displacement` so an electron near one face of
the cell sees the closest image of each nucleus.
"""

from dataclasses import fields

import jax
import jax.numpy as jnp
from flax import struct

from pytc.jastrow.ncusp import NuclearCusp as MolNuclearCusp

from ..utils import mic_displacement


@struct.dataclass
class NuclearCusp(MolNuclearCusp):
    """Periodic nuclear cusp Jastrow.

    Adds a ``lattice`` field carrying the cell's lattice vectors (rows).
    Overrides :meth:`_compute_inner` so the e-n displacement uses the
    minimum-image convention. All other parent methods —
    :meth:`_compute`, :meth:`get_log_grads_r1`, :meth:`get_log_grads_r2`,
    :meth:`init_params`, the spline machinery — are inherited and call
    the overridden method automatically through dynamic dispatch.
    """

    lattice: jax.Array = None

    @classmethod
    def create(cls, cell, name=None, n_radial=1000):
        """Construct a PBC :class:`NuclearCusp` from a ``pyscf.pbc.gto.Cell``.

        Args:
            cell: A built periodic cell with ``cart=True`` (the cusp
                correction does not depend on the angular basis, but the
                rest of the ansatz typically requires Cartesian).
            name: Optional instance name.
            n_radial: Radial grid resolution for the s-AO spline.

        Returns:
            A periodic :class:`NuclearCusp`.
        """
        mol_cusp = MolNuclearCusp.create(cell, name=name, n_radial=n_radial)
        return cls(
            lattice=jnp.asarray(cell.lattice_vectors()),
            **{f.name: getattr(mol_cusp, f.name) for f in fields(mol_cusp)},
        )

    def _compute_inner(self, r1, r2, clipped_params, poly_coeffs):
        """Sum of per-nucleus contributions with MIC e-n displacement.

        Identical to the molecular version except that the electron-to-
        nucleus displacement is the minimum-image one.
        """

        def compute_nucleus_contribution(nucleus_idx):
            dr = mic_displacement(r1, self.coords[nucleus_idx], self.lattice)
            r = jnp.sqrt(jnp.sum(dr ** 2) + 1e-16)

            Z = self.charges[nucleus_idx]
            Z_idx = self.Z_to_idx[Z.astype(jnp.int32)]
            rc = clipped_params['rc'][Z_idx]
            coeffs = poly_coeffs[Z_idx]

            poly_val = jnp.where(r <= rc, self._eval_poly(r, coeffs), 0.0)
            phi_s = jnp.where(r <= rc, self.eval_mo_at_r(nucleus_idx, r), 1.0)
            safe_phi_s = jnp.maximum(jnp.abs(phi_s), 1e-16)
            log_term = poly_val - jnp.log(safe_phi_s)

            cutoff = self._cutoff_function(r, rc)
            return jnp.where(r <= rc, log_term * cutoff, 0.0)

        contributions = jax.vmap(compute_nucleus_contribution)(jnp.arange(self.n_nuclei))
        total = jnp.sum(contributions)
        return total / (self.nelectron - 1)
