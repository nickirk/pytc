"""Tests for XTC block integral calculations."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.autodiff.xtc import XTC
from pytc.autodiff.jastrow import REXP

# Enable float64
jax.config.update("jax_enable_x64", True)

def get_h2o_ccpvdz():
    """Return H2O molecule with cc-pVDZ basis."""
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='cc-pvdz',
        unit='Angstrom',
        verbose=0
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

class TestXTCBlock(unittest.TestCase):
    """Test XTC block integral calculations."""
    
    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf = get_h2o_ccpvdz()
        # Use coarse grid for speed
        cls.jastrow = REXP(epsilon=1e-8)
        cls.xtc = XTC.from_pyscf(cls.mf, cls.jastrow, grid_lvl=1)
        cls.jastrow_params = cls.jastrow.init_params(alpha=1.0)
        
        cls.nocc = np.sum(cls.mf.mo_occ > 0)
        cls.n_orb = cls.mf.mo_coeff.shape[1]
        
    def test_full_vs_block_oooo(self):
        """Test if 'oooo' block matches full tensor slice."""
        print(f"\nTesting XTC oooo block... nocc={self.nocc}")
        
        # Compute full tensor
        full_2b = self.xtc.get_2b(self.jastrow_params)
        
        # Compute block
        block_2b = self.xtc.get_2b(self.jastrow_params, block_str='oooo')
        
        slice_o = slice(0, self.nocc)
        expected = full_2b[slice_o, slice_o, slice_o, slice_o]
        
        self.assertEqual(block_2b.shape, expected.shape)
        np.testing.assert_allclose(block_2b, expected, atol=1e-8)
        
    def test_full_vs_block_oovv(self):
        """Test if 'oovv' block matches full tensor slice."""
        print("\nTesting XTC oovv block...")
        
        # Compute full tensor
        full_2b = self.xtc.get_2b(self.jastrow_params)
        
        # Compute block
        block_2b = self.xtc.get_2b(self.jastrow_params, block_str='oovv')
        
        slice_o = slice(0, self.nocc)
        slice_v = slice(self.nocc, self.n_orb)
        
        expected = full_2b[slice_o, slice_o, slice_v, slice_v]
        
        self.assertEqual(block_2b.shape, expected.shape)
        np.testing.assert_allclose(block_2b, expected, atol=1e-8)
        
    def test_ranges_custom(self):
        """Test custom ranges."""
        print("\nTesting XTC custom ranges...")
        
        # Define arbitrary ranges
        range_p = slice(0, 6)
        range_q = slice(1, 4)
        range_r = slice(2, 5)
        range_s = slice(0, 8)
        
        # ranges are (p, q, r, s)
        block_2b = self.xtc.get_2b(self.jastrow_params, ranges=(range_p, range_q, range_r, range_s))
        full_2b = self.xtc.get_2b(self.jastrow_params)
        
        # full_2b indices are (p, q, r, s)
        expected = full_2b[range_p, range_q, range_r, range_s]
        
        np.testing.assert_allclose(block_2b, expected, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
