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
        
    # PySCF may not retain the AO integral cache for larger molecules or on
    # memory-constrained runners. Fall back to the molecule-backed transform.
    ao_eri = getattr(mf, "_eri", None)
    if ao_eri is None:
        eri = ao2mo.kernel(mf.mol, mo_coeff, compact=False)
    else:
        eri = ao2mo.incore.full(ao_eri, mo_coeff, compact=False)
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
        
    h1e = mf.get_hcore()
    h1e = np.dot(mo_coeff.T, np.dot(h1e, mo_coeff))
    return h1e
