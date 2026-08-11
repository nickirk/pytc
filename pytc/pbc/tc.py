"""Small-cell Gamma-point periodic TC oracle."""

import logging
import time

import jax.numpy as jnp
import numpy as np
from pyscf.pbc import dft as periodic_dft

from pytc.integrals.tc import TC

logger = logging.getLogger(__name__)


def _require_gamma_real(mf, mo_coeff):
    kpoint = np.asarray(getattr(mf, "kpt", np.zeros(3)))
    if not np.allclose(kpoint, 0.0, atol=1e-12):
        raise NotImplementedError("periodic TC currently supports Gamma only")
    coefficients = np.asarray(mo_coeff)
    if np.iscomplexobj(coefficients):
        if not np.allclose(coefficients.imag, 0.0, atol=1e-12):
            raise NotImplementedError("periodic TC currently requires real orbitals")
        coefficients = coefficients.real
    return coefficients


def create_tc(mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
    """Build the direct pair-loop TC oracle for a real Gamma calculation."""
    cell = mf.cell
    if not cell.cart:
        raise ValueError("periodic TC requires a Cartesian basis")
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    mo_coeff = _require_gamma_real(mf, mo_coeff)

    logger.info("PBC TC: building level-%d Becke grid", grid_lvl)
    start = time.perf_counter()
    grids = periodic_dft.gen_grid.BeckeGrids(cell)
    grids.level = grid_lvl
    grids.build()
    logger.debug("PBC TC: grid built in %.3f s", time.perf_counter() - start)

    ao_kpoints = periodic_dft.numint.eval_ao_kpts(
        cell,
        np.asarray(grids.coords),
        kpts=np.zeros((1, 3)),
        deriv=1,
    )
    ao = np.asarray(ao_kpoints[0])
    ao_values = ao[0].T.real
    ao_gradients = ao[1:4].transpose(2, 1, 0).real

    coefficients = jnp.asarray(mo_coeff)
    n_ao, n_orb = coefficients.shape
    n_grid = grids.coords.shape[0]
    phi = coefficients.T @ jnp.asarray(ao_values)
    grad_phi = (
        coefficients.T @ jnp.asarray(ao_gradients).reshape(n_ao, -1)
    ).reshape(n_orb, n_grid, 3)

    return TC(
        grid_points=jnp.asarray(grids.coords),
        weights=jnp.asarray(grids.weights),
        phi=phi,
        grad_phi=grad_phi,
        n_orb=n_orb,
        grid_lvl=grid_lvl,
        jastrow_factor=jastrow_factor,
        mo_coeff=coefficients,
        nocc=int(np.sum(mf.mo_occ > 0)),
    )
