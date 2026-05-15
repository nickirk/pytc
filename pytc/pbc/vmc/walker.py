"""PBC walker initialization.

Reuses :class:`pytc.vmc.walker.Walker` and the unchanged
:func:`pytc.vmc.walker.initialize_walker_state`. Only the initial position
sampling is overridden so that electrons are placed inside the primitive
cell (atoms outside the cell are first wrapped, then electrons sampled
around them are also wrapped).
"""

import time

import jax.numpy as jnp
from jax import random

from pytc.vmc.walker import Walker, initialize_walker_state
from pytc.vmc.mcmc_utils import init_electron_configs

from ..utils import wrap


def initialize_walkers(
    ansatz,
    cell,
    n_walkers,
    initial_walkers=None,
    key=None,
    log_init: bool = True,
):
    """Initialize walker configurations for a periodic system.

    Electrons are first placed around the atomic positions using the same
    pairing-based distribution as the molecular code, then folded into the
    primitive cell. Atomic positions themselves are wrapped before the
    placement so electrons cluster around the in-cell atom image.

    Args:
        ansatz: Wavefunction object with ``atom_coords``, ``atom_charges``,
            ``n_electrons``, ``n_alpha`` attributes (e.g. a PBC SlaterDet).
        cell: ``pyscf.pbc.gto.Cell`` providing ``lattice_vectors()``.
        n_walkers: Number of parallel walkers.
        initial_walkers: Optional pre-built ``Walker`` or position array.
            If a position array is provided, it is wrapped into the cell.
        key: PRNG key. Defaults to a fresh key seeded from wall clock.
        log_init: Forwarded to :func:`init_electron_configs`.

    Returns:
        :class:`Walker` initialized with positions inside the cell.
    """
    if key is None:
        key = random.PRNGKey(int(time.time()))

    lattice = jnp.asarray(cell.lattice_vectors())

    if isinstance(initial_walkers, Walker):
        return initial_walkers.replace(
            positions=wrap(initial_walkers.positions, lattice)
        )

    if initial_walkers is not None:
        positions = wrap(jnp.asarray(initial_walkers), lattice)
        return initialize_walker_state(ansatz, positions)

    atom_coords = wrap(jnp.asarray(ansatz.atom_coords), lattice)
    atom_charges = ansatz.atom_charges
    n_electrons = ansatz.n_electrons
    n_alpha = ansatz.n_alpha

    key, subkey = random.split(key)
    positions = init_electron_configs(
        atom_coords, atom_charges, n_electrons, n_walkers, subkey,
        n_alpha=n_alpha, log_init=log_init,
    )
    positions = wrap(positions, lattice)
    return initialize_walker_state(ansatz, positions)
