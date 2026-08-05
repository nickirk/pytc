"""Shared cells for the periodic test suites.

Built fresh per call rather than cached: tests mutate the cell they are
handed, and a shared instance would leak state between them.
"""

from pyscf.pbc.gto import Cell

__all__ = ["diamond_111"]


def diamond_111(*, ke_cutoff=20.0, mesh=None, basis="gth-dzvp", pseudo="gth-pbe"):
    """Two-atom diamond cell. Pass ``mesh`` to pin the grid instead of the cutoff."""
    cell = Cell()
    cell.atom = "C 0 0 0; C .8917 .8917 .8917"
    cell.a = "0 1.7834 1.7834\n1.7834 0 1.7834\n1.7834 1.7834 0"
    cell.unit = "A"
    cell.basis = basis
    cell.pseudo = pseudo
    if mesh is not None:
        cell.mesh = mesh
    else:
        cell.ke_cutoff = ke_cutoff
    cell.verbose = 0
    cell.build()
    return cell
