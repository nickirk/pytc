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

    def test_fcidump(self):
        mf = scf.RHF(self.mol)
        mf.kernel()
        my_jastrow = REXP()
        params = my_jastrow.init_params(alpha=1.4)
        my_xtc = XTC.from_pyscf(mf, my_jastrow, grid_lvl=2)
        
        h1e_xtc = my_xtc.get_1b(params)
        h2e_xtc = my_xtc.get_2b(params)
        ecore_xtc = my_xtc.get_const(params)
        n_orb_xtc = h1e_xtc.shape[0]
        n_elec_xtc = self.mol.nelectron
        
        fcidump.write(self.filename, h1e_xtc, h2e_xtc, ecore_xtc, n_orb_xtc, n_elec_xtc)
        n_orb_fcidump, n_elec_fcidump, ecore_fcidump, h1e_fcidump, h2e_fcidump = fcidump.read(self.filename)
        
        self.assertEqual(n_orb_xtc, n_orb_fcidump)
        self.assertEqual(n_elec_xtc, n_elec_fcidump)
        self.assertAlmostEqual(ecore_xtc, ecore_fcidump)
        self.assertTrue(np.allclose(h1e_xtc, h1e_fcidump))
        self.assertTrue(np.allclose(h2e_xtc, h2e_fcidump))



class TestFCIDumpGolden(unittest.TestCase):
    """Byte-identity golden-file test for fcidump.write vectorization."""

    # Synthetic fixed data (no PySCF dependency → fully deterministic)
    N_ORB = 4
    N_ELEC = 4

    def _make_data(self):
        n = self.N_ORB
        h1e = np.zeros((n, n))
        h1e[0, 0] = 1.0; h1e[0, 1] = 0.5; h1e[1, 0] = 0.5; h1e[1, 1] = 2.0
        h1e[2, 2] = 1.5; h1e[2, 3] = 0.3; h1e[3, 2] = 0.3; h1e[3, 3] = 3.0
        h1e[0, 3] = 0.1; h1e[3, 0] = 0.1

        h2e = np.zeros((n, n, n, n))
        h2e[0, 0, 0, 0] = 1.234567890123456e-1
        h2e[1, 1, 1, 1] = 2.5e-1
        h2e[0, 1, 0, 1] = 3.141592653589793e-2
        h2e[1, 0, 1, 0] = 3.141592653589793e-2
        h2e[2, 2, 3, 3] = 5.0e-2
        h2e[3, 3, 2, 2] = 5.0e-2
        h2e[0, 2, 0, 2] = 1e-16  # below threshold, should be skipped

        ecore = -5.4321
        return h1e, h2e, ecore

    GOLDEN = (
        "&FCI NORB=   4,NELEC= 4,MS2=0,\n"
        "  ORBSYM=1,1,1,1,\n"
        "  ISYM=1,\n"
        " &END\n"
        " 1.234567890123456E-01   1   1   1   1\n"
        " 3.141592653589793E-02   1   2   1   2\n"
        " 3.141592653589793E-02   2   1   2   1\n"
        " 2.500000000000000E-01   2   2   2   2\n"
        " 5.000000000000000E-02   3   3   4   4\n"
        " 1.000000000000000E+00   1   1   0   0\n"
        " 5.000000000000000E-01   1   2   0   0\n"
        " 1.000000000000000E-01   1   4   0   0\n"
        " 2.000000000000000E+00   2   2   0   0\n"
        " 1.500000000000000E+00   3   3   0   0\n"
        " 3.000000000000000E-01   3   4   0   0\n"
        " 3.000000000000000E+00   4   4   0   0\n"
        "-5.432100000000000E+00   0   0   0   0\n"
    )

    def test_byte_identity(self):
        h1e, h2e, ecore = self._make_data()
        fname = 'test_golden_fcidump.txt'
        try:
            fcidump.write(fname, h1e, h2e, ecore, self.N_ORB, self.N_ELEC)
            with open(fname) as f:
                output = f.read()
            self.assertEqual(output, self.GOLDEN,
                             "fcidump.write output does not match golden reference")
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def test_roundtrip_synthetic(self):
        h1e, h2e, ecore = self._make_data()
        fname = 'test_golden_fcidump.txt'
        try:
            fcidump.write(fname, h1e, h2e, ecore, self.N_ORB, self.N_ELEC)
            n, nelec, ec, h1, h2 = fcidump.read(fname)
            self.assertEqual(n, self.N_ORB)
            self.assertEqual(nelec, self.N_ELEC)
            self.assertAlmostEqual(ec, ecore)
            self.assertTrue(np.allclose(h1, h1e))
            self.assertTrue(np.allclose(h2, h2e))
        finally:
            if os.path.exists(fname):
                os.remove(fname)


if __name__ == '__main__':
    unittest.main()
