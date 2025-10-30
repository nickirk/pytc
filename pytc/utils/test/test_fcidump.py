import unittest
import os
import numpy as np
from pyscf import gto, scf
from pytc.utils import fcidump
from pytc.xtc import XTC
from pytc.jastrow import REXP

class TestFCIDump(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom='He 0 0 0', basis='ccpvdz')
        self.filename = 'test_fcidump.txt'

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    # Example function to use the write_fcidump & read_fcidump function
    def test_fcidump(self):
        mf = scf.RHF(self.mol)
        mf.kernel()
        my_jastrow = REXP([1.4])
        my_xtc = XTC(mf, my_jastrow, grid_lvl=2)
        
        h1e_xtc = my_xtc.get_1b()
        h2e_xtc = my_xtc.get_2b()
        ecore_xtc = my_xtc.get_const()
        n_orb_xtc = h1e_xtc.shape[0]
        n_elec_xtc = self.mol.nelectron
        
        fcidump.write(self.filename, h1e_xtc, h2e_xtc, ecore_xtc, n_orb_xtc, n_elec_xtc)
        n_orb_fcidump, n_elec_fcidump, ecore_fcidump, h1e_fcidump, h2e_fcidump = fcidump.read(self.filename)
        
        self.assertEqual(n_orb_xtc, n_orb_fcidump)
        self.assertEqual(n_elec_xtc, n_elec_fcidump)
        self.assertAlmostEqual(ecore_xtc, ecore_fcidump)
        self.assertTrue(np.allclose(h1e_xtc, h1e_fcidump))
        self.assertTrue(np.allclose(h2e_xtc, h2e_fcidump))



if __name__ == '__main__':
    unittest.main()
