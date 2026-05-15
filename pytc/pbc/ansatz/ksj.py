"""k-point Slater-Jastrow.

The molecular :class:`pytc.ansatz.sj.SlaterJastrow` evaluator calls
``value_and_grad(det, walker)`` directly, which hardcodes the molecular
``eval_det_value_and_grad`` path and reads ``det.eval_ao_func``. That
field doesn't exist on :class:`KSlaterDet`. To stay PySCF-style
("KRHF subclasses RHF, overriding only what's truly different"), we
keep the molecular :class:`SlaterJastrow` as the parent and provide a
thin :class:`KSlaterJastrow` that overrides ``__call__`` to route the
det evaluation through ``det(walker, params)`` — picking up KSlaterDet's
own evaluator via dynamic dispatch.

All Jastrow infrastructure (``compute_jastrow_log_value``,
``update_jastrow_one_electron``) is reused unchanged because it only
touches ``self.jastrow``, which we don't fork.
"""

import jax
import jax.numpy as jnp
from flax import struct

from pytc.ansatz.sj import SlaterJastrow, compute_jastrow_log_value


@struct.dataclass
class KSlaterJastrow(SlaterJastrow):
    """SlaterJastrow whose determinant evaluation honours each det's __call__.

    Drop-in replacement when ``dets`` contains :class:`KSlaterDet`
    instances; ``isinstance(KSlaterJastrow, SlaterJastrow)`` is True so
    the PBC move dispatch still routes via the SlaterJastrow branch.
    """

    @classmethod
    def create(cls, supercell, jastrow, dets):
        """Build a KSlaterJastrow from a supercell, Jastrow factor, and dets.

        Args:
            supercell: ``pyscf.pbc.gto.Cell`` (the Nk-replicated supercell
                for k-point VMC, or the cell itself at Gamma).
            jastrow: A PBC Jastrow factor.
            dets: List of :class:`KSlaterDet` instances.
        """
        # Reuse SlaterJastrow.create to populate atom_coords / charges /
        # ion_ion_potential. The ion_ion field is the molecular bare-1/r
        # ion sum, which is *not* the Madelung Ewald value. For the PBC
        # local-energy paths (compute_single_walker_energy with the
        # Ewald total Coulomb), this field is intentionally unused.
        base = SlaterJastrow.create(supercell, jastrow, dets)
        return cls(
            dets=base.dets,
            atom_coords=base.atom_coords,
            atom_charges=base.atom_charges,
            ion_ion_potential=base.ion_ion_potential,
            jastrow=base.jastrow,
        )

    def __call__(self, walker, params):
        return eval_ksj(self, walker, params)


def eval_ksj(ksj: KSlaterJastrow, walker, params):
    """Single-walker wavefunction value with cached walker fields.

    Mirrors :func:`pytc.ansatz.sj.eval_sj` but evaluates each det via
    ``det(walker, params)`` so :class:`KSlaterDet`'s complex Slater /
    inverse / gradient / Laplacian fields populate the walker through
    its own ``eval_kdet_value_and_grad`` path. Single-determinant only
    for now (matches the closed-shell case in step 2).
    """
    jastrow_params, linear_coeffs = params

    log_jastrow_val = compute_jastrow_log_value(ksj, walker.positions, jastrow_params)

    if len(ksj.dets) != 1:
        raise NotImplementedError(
            "KSlaterJastrow currently supports a single determinant"
        )

    det_val, updated_walker = ksj.dets[0](walker, params)
    det_sign, det_logabs = det_val

    # Complex det_sign × real linear coefficient → complex phase.
    linear_combo_sign = jnp.sign(linear_coeffs[0]) * det_sign
    linear_combo_logabs = jnp.log(jnp.abs(linear_coeffs[0])) + det_logabs

    psi_sign = linear_combo_sign
    psi_logabs = log_jastrow_val + linear_combo_logabs

    updated_walker = updated_walker.replace(
        log_psi=psi_logabs,
        psi_sign=psi_sign,
        log_jastrow=log_jastrow_val,
    )
    return (psi_sign, psi_logabs), updated_walker
