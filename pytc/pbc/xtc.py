"""Periodic xTC (extended transcorrelated) integral corrections.

The molecular :class:`pytc.xtc.XTC` extends :class:`pytc.tc.TC` with
``mo_occ`` and ``energy_nuc`` and adds the methods that compute the
xTC corrections to one-body, two-body, and constant terms (``get_1b``,
``get_2b``, ``get_const``, ``get_delta_U``, ``get_delta_h``). All of
those are pure JAX operations on ``phi``, ``grad_phi``, ``weights``,
``grid_points``, ``jastrow_factor``, ``mo_occ``, and ``energy_nuc`` —
they never read back into the molecule or the ``_eri`` blob.

So the periodic port reduces to a factory that:

* Builds the underlying TC grid via :func:`pytc.pbc.tc.create_tc`.
* Attaches ``mo_occ`` from the PBC mean-field.
* Attaches the periodic nuclear-repulsion energy, which
  ``pyscf.pbc.scf.RHF.energy_nuc()`` already returns as the
  Madelung-summed Ewald value (it dispatches to ``cell.ewald()``).

The standard 2e ERI tensor needed to build ``ChemistsERIs`` for CCSD
is NOT computed here — that's a separate pipeline (PBC density-fitted
ERIs, MP2 integrals, etc.) and will be wired in later. This module
exposes only the xTC *corrections*, which are the novel part of xTC
and are what we want to validate first.
"""

import logging

import jax.numpy as jnp

from pytc.xtc import XTC

from .tc import create_tc

logger = logging.getLogger(__name__)


def create_xtc(mf, jastrow_factor, mo_coeff=None, grid_lvl: int = 2) -> XTC:
    """Build a periodic :class:`XTC` from a Gamma-only PBC mean-field.

    Args:
        mf: A ``pyscf.pbc.scf.RHF``-like object. Its ``cell``, ``mo_coeff``,
            ``mo_occ``, and ``energy_nuc()`` are read.
        jastrow_factor: Periodic Jastrow factor (e.g. built via
            :func:`pytc.pbc.jastrow.BoysHandy.create`).
        mo_coeff: ``(n_ao, n_mo)`` MO coefficients. Defaults to ``mf.mo_coeff``.
        grid_lvl: Becke-grid level for the integration grid.

    Returns:
        A :class:`pytc.xtc.XTC` populated with periodic grid arrays, the
        Madelung nuclear-repulsion energy, and the supplied periodic
        Jastrow factor. All ``get_*`` methods on the returned object
        (``get_const``, ``get_1b``, ``get_2b``, ``get_delta_U``,
        ``get_delta_h``) are PBC-correct without further changes.
    """
    tc_obj = create_tc(mf, jastrow_factor, mo_coeff=mo_coeff, grid_lvl=grid_lvl)

    mo_occ = jnp.asarray(mf.mo_occ)
    # For pyscf.pbc.scf, energy_nuc() returns cell.ewald() — the Madelung
    # sum, not the bare 1/r nuclear-nuclear term.
    energy_nuc = float(mf.energy_nuc())

    return XTC(
        grid_points=tc_obj.grid_points,
        weights=tc_obj.weights,
        phi=tc_obj.phi,
        grad_phi=tc_obj.grad_phi,
        n_orb=tc_obj.n_orb,
        grid_lvl=tc_obj.grid_lvl,
        jastrow_factor=tc_obj.jastrow_factor,
        mo_coeff=tc_obj.mo_coeff,
        nocc=tc_obj.nocc,
        mo_occ=mo_occ,
        energy_nuc=energy_nuc,
    )
