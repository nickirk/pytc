"""Legacy NumPy-based implementations of TC/XTC.

This module contains the original NumPy implementations of transcorrelated
integrals. It is maintained for backward compatibility, testing, and rapid
prototyping. For production use, prefer the JAX-based autodiff implementations
in the main pytc package.
"""

from .df import isdf_decompose_multi, test_accuracy
from .kmat import calc_K1, calc_K2, calc_K3, calc_K1_isdf, calc_K2_isdf, calc_K3_isdf
from .lmat import calc_L
from .tc import TC
from .xtc import XTC

__all__ = [
    'isdf_decompose_multi', 'test_accuracy',
    'calc_K1', 'calc_K2', 'calc_K3', 
    'calc_K1_isdf', 'calc_K2_isdf', 'calc_K3_isdf',
    'calc_L', 'TC', 'XTC'
]
