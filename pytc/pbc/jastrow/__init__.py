"""PBC-aware Jastrow factors.

Each PBC Jastrow is a thin subclass of its molecular counterpart that:
  * stores the lattice as a struct field, and
  * overrides per-pair distance computations to use the minimum-image
    convention.

All construction logic, spline tables, parameter initialisation, and
gradient/Laplacian plumbing are inherited unchanged.
"""

from .ncusp import NuclearCusp
from .bh import BoysHandy

__all__ = ['NuclearCusp', 'BoysHandy']
