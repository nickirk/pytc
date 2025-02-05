import numpy as np                                                              
                                                                                
from pyscf import gto, scf, cc, lib                                             
                                                                                
from pytc import xtc, jastrow                                                   
                                                                                
lib.num_threads(1)                                                             
mol = gto.M(atom='C 0 0 0; O 0 0 -5.63; O 0 0 5.63', basis='aug-ccpvtz', unit='Bohr')
mf = scf.RHF(mol)                                                               
mf.kernel()                                                                     
                                                                                
my_jastrow = jastrow.SimpleJastrow([1.4])                                       
my_xtc = xtc.XTC(mf, my_jastrow, grid_lvl=2)                                    
                                                                                
print("Making eris")                                                            
eris = my_xtc.make_eris()                                                       
                                                                                
print("Running CCSD")                                                           
# Use modified rccsd from https://github.com/nickirk/pyscf/tree/tc-ccsd    
# Which can handle non-hermitian integrals   
lib.num_threads(1)                                                             
mycc = cc.rccsd.RCCSD(mf)                                                       
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