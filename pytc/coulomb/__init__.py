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
from .pivot_selection import (
    weight_mo_values,
    select_sector_pivots,
    select_pivots_oo_ov_vv,
    pair_collocation_reconstruction_error,
    rank_curve,
)
from .molecular_df_reference import (
    pair_collocation_at_pivots,
    compute_C_streamed,
    compute_Z,
    compute_Z_cross,
    reconstruct_eri_block,
)

__all__ = [
    "get_mo_coeff",
    "get_grid_ao_values_and_weights",
    "get_naux",
    "stream_df_cderi_blocks",
    "weight_mo_values",
    "select_sector_pivots",
    "select_pivots_oo_ov_vv",
    "pair_collocation_reconstruction_error",
    "rank_curve",
    "pair_collocation_at_pivots",
    "compute_C_streamed",
    "compute_Z",
    "compute_Z_cross",
    "reconstruct_eri_block",
]
