"""Helper functions for TC/XTC calculations."""

import numpy as np
from pyscf import ao2mo

def get_eri(mf, mo_coeff=None):
    """Get two-body integrals in MO basis.
    
    Args:
        mf: PySCF mean-field object
        mo_coeff: Molecular orbital coefficients. If None, use mf.mo_coeff
        
    Returns:
        numpy.ndarray: Two-body integrals (chemists' notation)
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
        
    mo_coeff = np.asarray(mo_coeff, dtype=np.double)
    
    # Compute ERI using PySCF
    eri = ao2mo.incore.full(mf._eri, mo_coeff, compact=False)
    eri = ao2mo.restore(1, eri, mo_coeff.shape[1])
    return eri

def get_hcore(mf, mo_coeff=None):
    """Get core Hamiltonian in MO basis.
    
    Args:
        mf: PySCF mean-field object
        mo_coeff: Molecular orbital coefficients. If None, use mf.mo_coeff
        
    Returns:
        numpy.ndarray: Core Hamiltonian matrix
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
        
    mo_coeff = np.asarray(mo_coeff, dtype=np.double)
    
    h1e = mf.get_hcore()
    h1e = np.dot(mo_coeff.T, np.dot(h1e, mo_coeff))
    return h1e
