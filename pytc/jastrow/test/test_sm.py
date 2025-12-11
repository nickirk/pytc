import unittest
import numpy as np
from pyscf import gto, scf, cc
from pytc.xtc import XTC
from pytc.jastrow import SM7, SM17

def get_he_ccpvtz():
    """Return a He atom with cc-pVTZ basis for testing."""
    mol = gto.M(atom='Be 0 0 0', basis='ccpvtz', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

class TestSM(unittest.TestCase):
    """Test SM7 and SM17 Jastrow factors with XTC."""

    def test_sm7(self):
        """Test SM7 Jastrow factor."""
        mol, mf = get_he_ccpvtz()
        jastrow = SM7(atom='Be')
        xtc = XTC(mf, jastrow, grid_lvl=2)
        
        # Run CCSD with XTC integrals
        mycc = cc.rccsd.RCCSD(mf)
        eris = xtc.make_eris()
        
        
        # Let's try packing the last two dimensions: (nocc, nvir, nvir*nvir)
        nocc, nvir, _, _ = eris.ovvv.shape
        
        e_corr, t1, t2 = mycc.kernel(eris=eris)
        
        no = mycc.nocc
        tc_h1e = xtc.get_1b()
        tc_e_hf = 2. * np.einsum('ii->', tc_h1e[:no, :no])
        tc_e_dir = 2. * np.einsum('jjii->', eris.oooo)
        tc_e_ex = -1. * np.einsum('ijji->', eris.oooo)

        tc_e_hf += (tc_e_dir + tc_e_ex) + eris.e_core 
        total_energy = e_corr + tc_e_hf
        
        print(f"SM7 Total Energy: {total_energy}")
        self.assertAlmostEqual(total_energy, -14.6591301503, places=6)

    def test_sm17(self):
        """Test SM17 Jastrow factor."""
        mol, mf = get_he_ccpvtz()
        jastrow = SM17(atom='Be')
        xtc = XTC(mf, jastrow, grid_lvl=2)
        
        # Run CCSD with XTC integrals
        mycc = cc.rccsd.RCCSD(mf)
        eris = xtc.make_eris()
        e_corr, t1, t2 = mycc.kernel(eris=eris)
        
        # Calculate total energy
        no = mycc.nocc
        tc_h1e = xtc.get_1b()
        tc_e_hf = 2. * np.einsum('ii->', tc_h1e[:no, :no])
        tc_e_dir = 2. * np.einsum('jjii->', eris.oooo)
        tc_e_ex = -1. * np.einsum('ijji->', eris.oooo)

        tc_e_hf += (tc_e_dir + tc_e_ex) + eris.e_core 
        total_energy = e_corr + tc_e_hf
        
        print(f"SM17 Total Energy: {total_energy}")
        self.assertAlmostEqual(total_energy, -14.6677960658, places=6)

if __name__ == '__main__':
    unittest.main()
