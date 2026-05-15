"""Slater determinant for periodic systems at the Gamma point.

The molecular :class:`pytc.ansatz.SlaterDet` is reused unchanged. A
``pyscf.pbc.gto.Cell`` exposes the same interface as ``pyscf.gto.Mole``
for everything ``SlaterDet.create`` reads (``cart``, ``nelec``,
``atom_coords()``, ``atom_charges()``, ``_basis``, ``atom_symbol``,
``atom_pure_symbol``, ``natm``), so a PBC determinant is constructed by
calling the molecular factory and replacing the orbital evaluator with a
:class:`GTO` whose ``images`` are populated from the lattice.

Downstream functions in :mod:`pytc.ansatz.det` (``eval_det_value``,
``eval_det_value_and_grad``, ``rank1_update_one_electron``,
``eval_single_electron_ao``) remain valid because they only touch
``mol_gto`` through the stored ``eval_ao_func``, which is itself unchanged
between molecular and PBC Cartesian basis sets.
"""

from pytc.ansatz.det import SlaterDet

from .gto import GTO

__all__ = ['SlaterDet', 'create_slater_det']


def create_slater_det(
    cell,
    mo_coeff=None,
    nelec=None,
    excitations=None,
    rcut: float = None,
    precision: float = 1e-8,
) -> SlaterDet:
    """Build a Gamma-point :class:`SlaterDet` from a periodic cell.

    Args:
        cell: A built ``pyscf.pbc.gto.Cell`` with ``cart=True``.
        mo_coeff: MO coefficient matrix from a Gamma-only PBC mean field
            (e.g. ``pyscf.pbc.scf.RHF(cell).mo_coeff``). Real-valued.
            Can be a single ``(nao, nmo)`` array (restricted) or a pair
            of arrays for unrestricted references.
        nelec: Optional ``(n_alpha, n_beta)``. Defaults to ``cell.nelec``.
        excitations: Optional ``(alpha_exc, beta_exc)`` excitations passed
            through to the molecular factory.
        rcut: Image-summation cutoff for the PBC GTO. Defaults to
            :func:`pytc.pbc.ansatz.gto.default_rcut` evaluated at
            ``precision``.
        precision: Tolerance used to derive ``rcut`` when not given.

    Returns:
        A :class:`SlaterDet` whose ``mol_gto`` is a :class:`GTO`.
    """
    det = SlaterDet.create(
        cell, mo_coeff=mo_coeff, nelec=nelec, excitations=excitations
    )
    pbc_gto = GTO.from_cell(cell, rcut=rcut, precision=precision)
    return det.replace(mol_gto=pbc_gto)
