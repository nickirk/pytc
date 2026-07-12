"""ISDF/LS-THC Coulomb-integral factorization for post-HF correlation.

Design decisions live in the isdf-coulomb-cuda repo
(github.com/nickirk/isdf-coulomb-cuda, docs/decisions/) -- this
subpackage is the code; that repo is the docs/decision-record store
only (Ke's direction, 2026-07-12).
"""

from .gpu4pyscf_adapter import (
    get_mo_coeff,
    get_grid_ao_values_and_weights,
    get_naux,
    stream_df_cderi_blocks,
)

__all__ = [
    "get_mo_coeff",
    "get_grid_ao_values_and_weights",
    "get_naux",
    "stream_df_cderi_blocks",
]
