import unittest
import numpy as np
import jax
import jax.numpy as jnp
import time
from pyscf import gto, scf
from pytc.autodiff.tc import TC, ISDFTC
from pytc.autodiff.jastrow import REXP

jax.config.update("jax_enable_x64", True)

class TestTCBlock(unittest.TestCase):
    def setUp(self):
        self.mol = gto.Mole()
        self.mol.atom = 'H 0 0 0; O 0 0 1; H 0 0 2'
        self.mol.basis = 'ccpvdz'
        self.mol.build()
        
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        self.jastrow = REXP()
        self.jastrow_params = {"alpha": jnp.array([1.0])}
        
        self.tc = TC.from_pyscf(self.mf, self.jastrow)
        self.nocc = self.tc.nocc
        self.n_orb = self.tc.n_orb
        
    def test_full_vs_block_oooo(self):
        print("Testing oooo block...")
        full_2b = self.tc.get_2b(self.jastrow_params)
        block_2b = self.tc.get_2b(self.jastrow_params, block_str='oooo')
        
        slice_o = slice(0, self.nocc)
        expected = full_2b[slice_o, slice_o, slice_o, slice_o]
        
        # Check shapes
        self.assertEqual(block_2b.shape, expected.shape)
        
        # Check values - get_2b now handles symmetrization
        np.testing.assert_allclose(block_2b, expected, atol=1e-8)

    def test_full_vs_block_oovv(self):
        print("Testing oovv block...")
        full_2b = self.tc.get_2b(self.jastrow_params)
        block_2b = self.tc.get_2b(self.jastrow_params, block_str='oovv')
        
        slice_o = slice(0, self.nocc)
        slice_v = slice(self.nocc, self.n_orb)
        
        expected = full_2b[slice_o, slice_o, slice_v, slice_v]
        
        np.testing.assert_allclose(block_2b, expected, atol=1e-8)

    def test_ranges_custom(self):
        print("Testing custom ranges...")
        # Test arbitrary ranges (slices)
        range_p = slice(0, 4)
        range_q = slice(1, 5)
        range_r = slice(0, 7)
        range_s = slice(1, 10)
        
        # ranges are (p, q, r, s)
        block_2b = self.tc.get_2b(self.jastrow_params, ranges=(range_p, range_q, range_r, range_s))
        full_2b = self.tc.get_2b(self.jastrow_params)
        
        # full_2b indices are (p, q, r, s)
        expected = full_2b[range_p, range_q, range_r, range_s]
        
        np.testing.assert_allclose(block_2b, expected, atol=1e-8)

    def test_recompilation(self):
        print("Testing recompilation behavior...")
        # First call triggers compilation
        start = time.time()
        _ = self.tc.get_2b(self.jastrow_params, block_str='oooo')
        first_time = time.time() - start
        print(f"First call time: {first_time:.4f}s")
        
        # Second call should be faster (no recompilation)
        start = time.time()
        _ = self.tc.get_2b(self.jastrow_params, block_str='oooo')
        second_time = time.time() - start
        print(f"Second call time: {second_time:.4f}s")
        
        # Check if second call is significantly faster
        # Note: on CPU/small system, compilation might be fast, but usually distinguishable.
        self.assertLess(second_time, first_time)

if __name__ == "__main__":
    import time
    unittest.main()
