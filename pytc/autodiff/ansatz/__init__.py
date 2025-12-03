"""Ansatz module for quantum many-body wavefunctions."""

from .sj import make_slater_jastrow
from .det import get_hf_det

__all__ = ['make_slater_jastrow', 'get_hf_det']
