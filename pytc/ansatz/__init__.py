"""Ansatz module for quantum many-body wavefunctions."""

from .sj import SlaterJastrow
from .det import SlaterDet

__all__ = ['SlaterJastrow', 'SlaterDet']
