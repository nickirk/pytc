"""Generic density-fitting/ISDF numerical machinery -- model-agnostic,
shared by TC/xTC and the Coulomb path.

API: this __init__ provides ONLY compatibility re-exports -- every name previously importable as ``pytc.df.X`` when this was
a single module stays importable the same way. No substantive
implementation lives here -- see pivots.py, solvers.py, isdf.py.
"""
from .pivots import (
    pivoted_cholesky_pair_pivots,
    _pivoted_cholesky_pair_pivots_core,
)
from .solvers import (
    solve_normal_equations_batch,
    _build_normal_matrix,
    prepare_spd_cholesky,
    prepare_normal_equations_solver,
    solve_normal_equations_batch_prepared,
)
from .isdf import (
    isdf_decompose,
    _pivoted_cholesky_phi,
    _pivoted_cholesky_grad,
)
from .fit import (
    pair_collocation_at_pivots,
    compute_Z,
    compute_Z_cross,
    reconstruct_eri_block,
)

__all__ = [
    "pivoted_cholesky_pair_pivots",
    "solve_normal_equations_batch",
    "prepare_spd_cholesky",
    "prepare_normal_equations_solver",
    "solve_normal_equations_batch_prepared",
    "isdf_decompose",
    "pair_collocation_at_pivots",
    "compute_Z",
    "compute_Z_cross",
    "reconstruct_eri_block",
]
