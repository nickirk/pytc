"""Periodic boundary conditions support for pytc.

This subpackage mirrors the molecular pytc layout but provides PBC-aware
versions of the walker, moves, ansatz, Jastrow, and Hamiltonian components.
Modules that are coordinate-agnostic (optimizer, sampling driver, blocking,
loss) are re-used from the molecular implementation rather than duplicated.

Convention: arrays carry a leading k-axis of size 1 at Gamma so that the
multi-kpoint extension reduces to relaxing that dimension.
"""

from . import utils
from . import tc
from . import xtc

__all__ = ['utils', 'tc', 'xtc']
