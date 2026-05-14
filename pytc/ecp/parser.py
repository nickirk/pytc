"""Parse PySCF ECP data (`mol._ecp`) into JAX-friendly padded arrays.

PySCF stores parsed ECP data as

    mol._ecp[symbol] = [
        n_core,
        [
            [l, [          # l = -1 for local, 0,1,2,... for non-local channels
                [             # one entry per radial power n (list index = n)
                    [zeta_k, c_k], ...
                ],
                ...
            ]],
            ...
        ]
    ]

The radial form per channel is

    V(r) = sum_k c_k * r^(n_k - 2) * exp(-zeta_k * r^2),

where the list index `n` in the parsed structure is exactly the `n_k` used
in the formula above.

This module produces a single `EcpData` dataclass holding padded arrays
indexed by atom (covering both ECP and non-ECP atoms uniformly, with all
coefficients zero for non-ECP atoms).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from flax import struct

from pytc.ecp import radial as _radial


@struct.dataclass
class EcpData:
    """Padded ECP parameters covering every atom in the molecule.

    Atoms without an ECP are represented by all-zero entries and
    `has_ecp[A] = False`, so a single kernel can process the full system.
    """

    has_ecp: jnp.ndarray         # (N_atoms,) bool
    n_core: jnp.ndarray          # (N_atoms,) int  (0 for non-ECP)
    # Local channel (one per atom), padded over primitives:
    loc_n: jnp.ndarray           # (N_atoms, K_loc) int
    loc_zeta: jnp.ndarray        # (N_atoms, K_loc) float
    loc_c: jnp.ndarray           # (N_atoms, K_loc) float
    # Non-local channels, padded over both l (0..L) and primitives:
    nl_n: jnp.ndarray            # (N_atoms, L_plus_1, K_nl) int
    nl_zeta: jnp.ndarray         # (N_atoms, L_plus_1, K_nl) float
    nl_c: jnp.ndarray            # (N_atoms, L_plus_1, K_nl) float
    # Per-atom mask telling which l-channels actually carry weight:
    l_mask: jnp.ndarray          # (N_atoms, L_plus_1) bool
    # Per-atom spatial cutoff for the non-local part (see design doc §4.6):
    r_cut: jnp.ndarray           # (N_atoms,) float

    @property
    def n_atoms(self) -> int:
        return int(self.has_ecp.shape[0])

    @property
    def l_max(self) -> int:
        """Largest non-local angular momentum present across atoms."""
        return int(self.nl_n.shape[1]) - 1

    @property
    def any_ecp(self) -> bool:
        """True iff at least one atom carries an ECP."""
        return bool(jnp.any(self.has_ecp))


def _lookup_ecp_entry(mol, atom_index: int):
    """Return mol._ecp entry for atom_index, or None if the atom has no ECP."""
    if not getattr(mol, "_ecp", None):
        return None
    # PySCF allows both labelled symbols ('C1') and pure symbols ('C').
    for key in (mol.atom_symbol(atom_index), mol.atom_pure_symbol(atom_index)):
        if key in mol._ecp:
            return mol._ecp[key]
    return None


def _collect_local(channel_terms) -> List[Tuple[int, float, float]]:
    """Flatten the local channel into a list of (n, zeta, c)."""
    out: List[Tuple[int, float, float]] = []
    for n, prims in enumerate(channel_terms):
        for zeta, c in prims:
            out.append((n, float(zeta), float(c)))
    return out


def _collect_nonlocal(
    channels,
) -> Tuple[int, List[List[Tuple[int, float, float]]]]:
    """Flatten non-local channels.

    Returns (l_plus_1, per_l_terms) where per_l_terms[l] is a list of
    (n, zeta, c) for that l. Missing l-channels are represented by [].
    """
    by_l: dict[int, List[Tuple[int, float, float]]] = {}
    for entry in channels:
        l, terms = entry[0], entry[1]
        if l < 0:
            continue
        flat: List[Tuple[int, float, float]] = []
        for n, prims in enumerate(terms):
            for zeta, c in prims:
                flat.append((n, float(zeta), float(c)))
        if flat:
            by_l[l] = flat
    if not by_l:
        return 0, []
    l_max = max(by_l)
    per_l = [by_l.get(l, []) for l in range(l_max + 1)]
    return l_max + 1, per_l


def parse_pyscf_ecp(
    mol,
    *,
    nl_cutoff_tol: float = 1.0e-5,
    nl_cutoff_rmax: float = 10.0,
    nl_cutoff_ngrid: int = 4096,
) -> EcpData:
    """Parse `mol._ecp` into padded JAX arrays for VMC use.

    Args:
        mol: a PySCF `Mole` object.
        nl_cutoff_tol: tolerance |V_l(r)| < tol used to define r_cut^A
            (default 1e-5 Ha, matching QMCPACK).
        nl_cutoff_rmax: outer radius for the cutoff scan.
        nl_cutoff_ngrid: number of grid points for the cutoff scan.

    Returns:
        EcpData. If no atom carries an ECP, returns a structure with
        `has_ecp` all False and minimum-shape arrays (K_loc = K_nl = 1,
        L_plus_1 = 1) so downstream kernels still have a well-defined shape.
    """
    n_atoms = mol.natm

    # 1. Walk atoms, collect per-atom local and non-local term lists.
    locals_per_atom: List[List[Tuple[int, float, float]]] = []
    nonlocals_per_atom: List[List[List[Tuple[int, float, float]]]] = []
    has_ecp_list: List[bool] = []
    n_core_list: List[int] = []
    l_plus_1_per_atom: List[int] = []

    for a in range(n_atoms):
        entry = _lookup_ecp_entry(mol, a)
        if entry is None:
            has_ecp_list.append(False)
            n_core_list.append(0)
            locals_per_atom.append([])
            nonlocals_per_atom.append([])
            l_plus_1_per_atom.append(0)
            continue

        n_core, channels = entry[0], entry[1]
        local_terms: List[Tuple[int, float, float]] = []
        for ch in channels:
            l = ch[0]
            if l == -1:
                local_terms.extend(_collect_local(ch[1]))
        l_plus_1, nl_terms = _collect_nonlocal(channels)

        has_ecp_list.append(True)
        n_core_list.append(int(n_core))
        locals_per_atom.append(local_terms)
        nonlocals_per_atom.append(nl_terms)
        l_plus_1_per_atom.append(l_plus_1)

    # 2. Decide padding sizes.
    k_loc_max = max((len(t) for t in locals_per_atom), default=0)
    k_loc_max = max(k_loc_max, 1)  # avoid zero-dim arrays
    l_plus_1_max = max(l_plus_1_per_atom + [0])
    l_plus_1_max = max(l_plus_1_max, 1)
    k_nl_max = 1
    for nl in nonlocals_per_atom:
        for terms in nl:
            if len(terms) > k_nl_max:
                k_nl_max = len(terms)

    # 3. Allocate and fill.
    loc_n = np.zeros((n_atoms, k_loc_max), dtype=np.int32)
    loc_zeta = np.ones((n_atoms, k_loc_max), dtype=np.float64)
    loc_c = np.zeros((n_atoms, k_loc_max), dtype=np.float64)
    nl_n = np.zeros((n_atoms, l_plus_1_max, k_nl_max), dtype=np.int32)
    nl_zeta = np.ones((n_atoms, l_plus_1_max, k_nl_max), dtype=np.float64)
    nl_c = np.zeros((n_atoms, l_plus_1_max, k_nl_max), dtype=np.float64)
    l_mask = np.zeros((n_atoms, l_plus_1_max), dtype=bool)

    for a, terms in enumerate(locals_per_atom):
        for k, (n, zeta, c) in enumerate(terms):
            loc_n[a, k] = n
            loc_zeta[a, k] = zeta
            loc_c[a, k] = c

    for a, nl in enumerate(nonlocals_per_atom):
        for l, terms in enumerate(nl):
            if not terms:
                continue
            l_mask[a, l] = True
            for k, (n, zeta, c) in enumerate(terms):
                nl_n[a, l, k] = n
                nl_zeta[a, l, k] = zeta
                nl_c[a, l, k] = c

    has_ecp = jnp.asarray(has_ecp_list)
    n_core = jnp.asarray(n_core_list, dtype=jnp.int32)

    loc_n_j = jnp.asarray(loc_n)
    loc_zeta_j = jnp.asarray(loc_zeta)
    loc_c_j = jnp.asarray(loc_c)
    nl_n_j = jnp.asarray(nl_n)
    nl_zeta_j = jnp.asarray(nl_zeta)
    nl_c_j = jnp.asarray(nl_c)
    l_mask_j = jnp.asarray(l_mask)

    # 4. Per-atom non-local cutoff.
    r_cut = _radial.find_nonlocal_cutoff(
        nl_n_j, nl_zeta_j, nl_c_j,
        tol=nl_cutoff_tol,
        r_max=nl_cutoff_rmax,
        n_grid=nl_cutoff_ngrid,
    )

    return EcpData(
        has_ecp=has_ecp,
        n_core=n_core,
        loc_n=loc_n_j,
        loc_zeta=loc_zeta_j,
        loc_c=loc_c_j,
        nl_n=nl_n_j,
        nl_zeta=nl_zeta_j,
        nl_c=nl_c_j,
        l_mask=l_mask_j,
        r_cut=r_cut,
    )
