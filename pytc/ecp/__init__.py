"""Effective core potential (ECP) support for pytc VMC.

See `docs/design_ecp_vmc.md` for the theory and design.

This module currently exposes the pure-data layer:

    parse_pyscf_ecp(mol)   -> EcpData    (parser.py)
    eval_v_loc, eval_v_nl, find_nonlocal_cutoff   (radial.py)
    get_grid(name) -> AngularGrid                  (quadrature.py)

Integration with the SlaterJastrow ansatz and the VMC local-energy
evaluator is staged separately (see design doc §8 steps 5-9).
"""

from pytc.ecp.energy import compute_nonlocal_ecp_energy
from pytc.ecp.parser import EcpData, parse_pyscf_ecp
from pytc.ecp.quadrature import AngularGrid, get_grid, icosahedral_12, lebedev_26
from pytc.ecp.radial import (
    eval_radial_channel,
    eval_v_loc,
    eval_v_nl,
    find_nonlocal_cutoff,
)

__all__ = [
    "EcpData",
    "parse_pyscf_ecp",
    "AngularGrid",
    "get_grid",
    "icosahedral_12",
    "lebedev_26",
    "eval_radial_channel",
    "eval_v_loc",
    "eval_v_nl",
    "find_nonlocal_cutoff",
    "compute_nonlocal_ecp_energy",
]
