"""Local energy for periodic systems.

The kinetic and Jastrow-derivative machinery in
:mod:`pytc.vmc.hamiltonian` is coordinate-agnostic: it works on
``walker.lap_up``, ``walker.grad_up`` etc., which are populated by the
PBC :class:`SlaterDet`, and on ``compute_jastrow_terms`` which calls
``sj.jastrow.get_log_grads_r1`` — themselves PBC-aware via dynamic
dispatch on the periodic Jastrow subclasses.

The only PBC-specific piece is the Coulomb potential: bare ``1/r`` over
distinct charges is replaced by the Ewald-summed periodic Coulomb energy
of the combined electron + nuclear charge set. The N-N piece is folded
into this Ewald total — :attr:`sj.ion_ion_potential` (which the
molecular factory still computes as a bare 1/r sum) is not consulted.
"""

import jax
import jax.numpy as jnp

from pytc.vmc.hamiltonian import compute_jastrow_terms

from ..ansatz.kdet import KSlaterDet
from .ewald import total_coulomb_energy


def compute_single_walker_energy(sj, walker, jastrow_params, ewald):
    """Local energy for one walker under periodic boundary conditions.

    Args:
        sj: A :class:`SlaterJastrow` whose Jastrows and SlaterDet are
            PBC-aware (Bloch-summed orbitals, MIC distances).
        walker: Walker with cached Slater matrices, gradients, and
            Laplacians (populated by ``eval_det_value_and_grad`` on the
            PBC SlaterDet).
        jastrow_params: Jastrow parameter tree.
        ewald: Cached :class:`EwaldParams` for the cell.

    Returns:
        Scalar real-valued local energy.
    """
    n_alpha = sj.dets[0].n_alpha

    grad_J_over_J, lap_J_over_J = compute_jastrow_terms(
        sj, walker.positions, jastrow_params
    )

    grad_J_alpha = grad_J_over_J[:n_alpha]
    grad_J_beta = grad_J_over_J[n_alpha:]
    lap_J_alpha = lap_J_over_J[:n_alpha]
    lap_J_beta = lap_J_over_J[n_alpha:]

    B_kin_alpha = -0.5 * (
        walker.lap_up
        + 2 * jnp.einsum('ik,ijk->ij', grad_J_alpha, walker.grad_up)
        + jnp.multiply(lap_J_alpha[:, None], walker.slater_up)
    )
    B_kin_beta = -0.5 * (
        walker.lap_down
        + 2 * jnp.einsum('ik,ijk->ij', grad_J_beta, walker.grad_down)
        + jnp.multiply(lap_J_beta[:, None], walker.slater_down)
    )

    E_kinetic = (
        jnp.trace(walker.inv_up @ B_kin_alpha)
        + jnp.trace(walker.inv_down @ B_kin_beta)
    )

    V_coulomb = total_coulomb_energy(
        walker.positions, sj.atom_coords, sj.atom_charges, ewald
    )

    return jnp.real(E_kinetic + V_coulomb)


def compute_single_walker_energy_kpts(kdet: KSlaterDet, walker, ewald):
    """Local energy for one walker with a bare k-point :class:`KSlaterDet`.

    No Jastrow. The kinetic-energy expression collapses to

    .. math::

        E_\\mathrm{kin} = -\\tfrac{1}{2} \\mathrm{Tr}[S_\\uparrow^{-1} \\nabla^2 S_\\uparrow]
                       - \\tfrac{1}{2} \\mathrm{Tr}[S_\\downarrow^{-1} \\nabla^2 S_\\downarrow],

    evaluated on the complex Slater matrices cached on ``walker``. The
    potential part is the Ewald-summed total Coulomb energy of the
    electron and nuclear charges, identical to the existing PBC path.
    Both pieces are complex (or partially complex); ``Re[...]`` at the
    end gives the real local energy.

    Args:
        kdet: PBC k-point SlaterDet (provides ``atom_coords`` /
            ``atom_charges`` for the Ewald sum).
        walker: Walker with cached complex ``inv_up/down`` and ``lap_up/down``.
        ewald: Cached :class:`EwaldParams` for the supercell.

    Returns:
        Scalar real-valued local energy.
    """
    B_kin_alpha = -0.5 * walker.lap_up
    B_kin_beta = -0.5 * walker.lap_down
    E_kinetic = (
        jnp.trace(walker.inv_up @ B_kin_alpha)
        + jnp.trace(walker.inv_down @ B_kin_beta)
    )
    V_coulomb = total_coulomb_energy(
        walker.positions, kdet.atom_coords, kdet.atom_charges, ewald
    )
    return jnp.real(E_kinetic + V_coulomb)


def eval_local_energy_kpts(kdet: KSlaterDet, walker, ewald):
    """Convenience wrapper analogous to :func:`eval_local_energy`.

    Returns ``(energy, walker)`` — the walker is passed through unchanged.
    """
    return compute_single_walker_energy_kpts(kdet, walker, ewald), walker


def eval_local_energy(sj, walker, params, ewald):
    """Evaluate the periodic local energy for a SlaterJastrow ansatz.

    Mirrors :func:`pytc.vmc.hamiltonian.eval_local_energy` but threads
    the :class:`EwaldParams` through. The returned walker is unchanged
    (energy evaluation is read-only).

    Args:
        sj: PBC-aware SlaterJastrow.
        walker: Walker with cached Slater quantities.
        params: ``(jastrow_params, linear_coeffs)`` tuple.
        ewald: Cached Ewald tables.

    Returns:
        ``(energy, walker)``.
    """
    jastrow_params, _ = params
    energy = compute_single_walker_energy(sj, walker, jastrow_params, ewald)
    return energy, walker
