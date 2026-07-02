import numpy as np

from pyscf import gto, scf, lib

from pytc import xtc, jastrow
from pytc.solver import jax_xtc_ccsd

lib.num_threads(1)
mol = gto.M(atom='C 0 0 0; O 0 0 -5.63; O 0 0 5.63', basis='aug-ccpvtz', unit='Bohr')
mf = scf.RHF(mol)
mf.kernel()

my_jastrow = jastrow.REXP()
jastrow_params = my_jastrow.init_params(alpha=1.4)
my_xtc = xtc.XTC.from_pyscf(mf, my_jastrow, grid_lvl=2)

print("Making eris")
eris = my_xtc.make_eris(mf, jastrow_params)

print("Running CCSD")
# Use pytc's own JAX-native xTC-CCSD solver, which correctly handles the
# non-Hermitian transcorrelated integrals (PySCF's stock RCCSD assumes ERI
# symmetries that don't hold here and silently gives the wrong energy).
lib.num_threads(1)
mycc = jax_xtc_ccsd.RCCSD(mf, my_xtc, jastrow_params)
tc_e_corr, t1, t2 = mycc.kernel(eris=eris)
                                                                                
# Calculating HF energy with xtc integrals
# PySCF internally will use the bare Coulomb integrals for the HF energy, 
# so we need to calculate the HF energy manually with the xtc integrals
no = mycc.nocc                                                                  
e_hf = 2*np.einsum('ii->', eris.fock[:no,:no])                                  
e_hf -= 2*np.einsum('iijj ->', eris.oooo)                                       
e_hf += np.einsum('ijji ->', eris.oooo)                                         
e_hf += eris.e_core                                                             
                                                                                
print("E_XTC_CCSD = ", tc_e_corr + e_hf)  