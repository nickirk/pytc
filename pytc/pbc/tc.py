"""Periodic transcorrelated (TC) integral grid.

The molecular :class:`pytc.tc.TC` stores a real-space DFT grid plus the
MO values and gradients evaluated on it. The downstream K-matrix
builders in :mod:`pytc.kmat` consume these arrays without ever looking
back at the ``mol`` object — they are purely grid-and-Jastrow driven.

So the periodic port reduces to a single factory:

* Build a periodic Becke grid via :class:`pyscf.pbc.dft.gen_grid.BeckeGrids`.
* Evaluate AOs (and gradients) on that grid using PySCF's PBC numint at
  the Gamma point. Equivalent to our :class:`GTO` evaluator (we
  verified they agree to machine precision), but PySCF's path is the
  reference implementation and avoids re-tuning rcut for the
  integration grid.
* Hand the AO grid + supplied ``mo_coeff`` to the existing molecular
  :class:`TC` constructor.

All Jastrow-gradient operations in the K-matrix path go through the
Jastrow's ``_compute`` / ``grad_r`` methods, which our PBC Jastrows
(:class:`pytc.pbc.jastrow.NuclearCusp`, :class:`pytc.pbc.jastrow.BoysHandy`)
override to use minimum-image distances. So feeding a PBC Jastrow into
this factory yields a fully periodic TC integral kernel without any
further changes to :mod:`pytc.kmat`.
"""

import logging
import time

import numpy as np
import jax.numpy as jnp
from pyscf.pbc import dft as pbcdft

from pytc.tc import TC

logger = logging.getLogger(__name__)


def create_tc(mf, jastrow_factor, mo_coeff=None, grid_lvl: int = 2) -> TC:
    """Build a :class:`TC` from a Gamma-only PBC mean-field.

    Args:
        mf: A ``pyscf.pbc.scf.RHF``-like mean-field whose ``cell``
            exposes Cartesian basis functions (``cell.cart == True``).
        jastrow_factor: A periodic Jastrow factor (e.g. one built via
            :func:`pytc.pbc.jastrow.BoysHandy.create`). The K-matrix
            builders read this object's ``grad_r`` / ``grad_r_batch``,
            which inherit minimum-image behaviour from the periodic
            ``_compute`` override.
        mo_coeff: ``(n_ao, n_mo)`` molecular orbital coefficients.
            Defaults to ``mf.mo_coeff``.
        grid_lvl: Becke-grid level passed to
            :class:`pyscf.pbc.dft.gen_grid.BeckeGrids`. Higher = denser.

    Returns:
        A :class:`pytc.tc.TC` populated with periodic grid quantities.
    """
    cell = mf.cell
    if not cell.cart:
        raise ValueError(
            "pytc.pbc.tc.create_tc currently requires a Cartesian basis "
            "(set cell.cart = True before cell.build())."
        )

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    n_orb = mo_coeff.shape[1]
    nocc = int(np.sum(mf.mo_occ > 0))

    logger.info("PBC TC: Building Becke grid (level %d)", grid_lvl)
    t0 = time.perf_counter()
    grids = pbcdft.gen_grid.BeckeGrids(cell)
    grids.level = grid_lvl
    grids.build()
    logger.debug("PBC TC: Grid built in %.3f s (%d points)",
                 time.perf_counter() - t0, grids.coords.shape[0])

    grid_points = jnp.asarray(grids.coords)
    weights = jnp.asarray(grids.weights)

    logger.info("PBC TC: Evaluating Gamma-point AOs on grid")
    t0 = time.perf_counter()
    # eval_ao_kpts returns a list of arrays, one per k-point. At kpts=[0,0,0]
    # we get a single Gamma-point block of shape (deriv+1, n_grid, n_ao).
    ao_kpts = pbcdft.numint.eval_ao_kpts(
        cell, np.asarray(grids.coords), kpts=np.zeros((1, 3)), deriv=1
    )
    ao = np.asarray(ao_kpts[0])
    # ao has shape (4, n_grid, n_ao): [value, d/dx, d/dy, d/dz]
    ao_values = ao[0].T.real           # (n_ao, n_grid)
    ao_gradients = ao[1:4].transpose(2, 1, 0).real  # (n_ao, n_grid, 3)
    logger.debug("PBC TC: AOs evaluated in %.3f s",
                 time.perf_counter() - t0)

    mo_coeff_jax = jnp.asarray(mo_coeff)
    ao_values_jax = jnp.asarray(ao_values)
    ao_gradients_jax = jnp.asarray(ao_gradients)

    # phi[i, g] = sum_a C[a, i] * ao[a, g]
    phi = jnp.matmul(mo_coeff_jax.T, ao_values_jax)

    n_grid = grid_points.shape[0]
    n_ao = mo_coeff.shape[0]
    grad_phi = jnp.matmul(
        mo_coeff_jax.T, ao_gradients_jax.reshape(n_ao, -1)
    ).reshape(n_orb, n_grid, 3)

    return TC(
        grid_points=grid_points,
        weights=weights,
        phi=phi,
        grad_phi=grad_phi,
        n_orb=n_orb,
        grid_lvl=grid_lvl,
        jastrow_factor=jastrow_factor,
        mo_coeff=mo_coeff_jax,
        nocc=nocc,
    )
