import unittest
import os
import numpy as np
from pyscf import gto, scf
from pytc.utils.fcidump import write_fcidump
from pytc.xtc import XTC
from pytc.jastrow import SimpleJastrow

class TestFCIDump(unittest.TestCase):

    def setUp(self):
        self.mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
        self.filename = 'test_fcidump.txt'

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_write_fcidump(self):
        mf = scf.RHF(self.mol)
        mf.kernel()
        my_jastrow = SimpleJastrow([1.4])
        my_xtc = XTC(mf, my_jastrow, grid_lvl=2)
        
        h1e = my_xtc.get_1b()
        h2e = my_xtc.get_2b()
        ecore = my_xtc.get_const()
        n_orb = h1e.shape[0]
        n_elec = self.mol.nelectron
        
        write_fcidump(self.filename, h1e, h2e, ecore, n_orb, n_elec)
        self.assertTrue(os.path.exists(self.filename))

    def test_fcidump_content(self):
        mf = scf.RHF(self.mol)
        mf.kernel()
        my_jastrow = SimpleJastrow([1.4])
        my_xtc = XTC(mf, my_jastrow, grid_lvl=2)
        
        h1e = my_xtc.get_1b()
        h2e = my_xtc.get_2b()
        ecore = my_xtc.get_const()
        n_orb = h1e.shape[0]
        n_elec = self.mol.nelectron
        
        write_fcidump(self.filename, h1e, h2e, ecore, n_orb, n_elec)
        with open(self.filename, 'r') as f:
            content = f.read()
        
        # Check for header
        self.assertIn('&FCI', content)
        self.assertIn('NORB=', content)
        self.assertIn('NELEC=', content)
        self.assertIn('&END', content)

        # Check for 1-body integrals
        self.assertIn('0 0 0 0', content)

        # Check for 2-body integrals
        self.assertIn('0 0 0 0', content)

        # Check for core energy
        self.assertIn('0 0 0 0', content)

    def test_extraction_of_integrals_and_ecore(self):
        mf = scf.RHF(self.mol)
        mf.kernel()
        my_jastrow = SimpleJastrow([1.4])
        my_xtc = XTC(mf, my_jastrow, grid_lvl=2)
        
        h1e = my_xtc.get_1b()
        h2e = my_xtc.get_2b()
        ecore = my_xtc.get_const()
        
        self.assertIsInstance(h1e, np.ndarray)
        self.assertIsInstance(h2e, np.ndarray)
        self.assertIsInstance(ecore, float)

    def test_read_fcidump(self):
        def read_fcidump(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            h1e = []
            h2e = []
            ecore = 0.0
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    value = float(parts[0])
                    i, j, k, l = map(int, parts[1:])
                    if i == 0 and j == 0 and k == 0 and l == 0:
                        ecore = value
                    elif k == 0 and l == 0:
                        h1e.append((i, j, value))
                    else:
                        h2e.append((i, j, k, l, value))
            return h1e, h2e, ecore

        mf = scf.RHF(self.mol)
        mf.kernel()
        my_jastrow = SimpleJastrow([1.4])
        my_xtc = XTC(mf, my_jastrow, grid_lvl=2)
        
        h1e = my_xtc.get_1b()
        h2e = my_xtc.get_2b()
        ecore = my_xtc.get_const()
        n_orb = h1e.shape[0]
        n_elec = self.mol.nelectron
        
        write_fcidump(self.filename, h1e, h2e, ecore, n_orb, n_elec)
        h1e_read, h2e_read, ecore_read = read_fcidump(self.filename)
        
        self.assertIsInstance(h1e_read, list)
        self.assertIsInstance(h2e_read, list)
        self.assertIsInstance(ecore_read, float)

if __name__ == '__main__':
    unittest.main()
