"""Small-cell Gamma-point periodic xTC oracle."""

import jax.numpy as jnp

from pytc.integrals.xtc import XTC

from .tc import create_tc


def create_xtc(mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
    """Build periodic xTC corrections using the direct TC grid oracle."""
    tc = create_tc(
        mf,
        jastrow_factor=jastrow_factor,
        mo_coeff=mo_coeff,
        grid_lvl=grid_lvl,
    )
    return XTC(
        grid_points=tc.grid_points,
        weights=tc.weights,
        phi=tc.phi,
        grad_phi=tc.grad_phi,
        n_orb=tc.n_orb,
        grid_lvl=tc.grid_lvl,
        jastrow_factor=tc.jastrow_factor,
        mo_coeff=tc.mo_coeff,
        nocc=tc.nocc,
        mo_occ=jnp.asarray(mf.mo_occ),
        energy_nuc=float(mf.energy_nuc()),
    )
