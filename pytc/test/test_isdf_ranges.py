import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.tc import TC, ISDFTC
from pytc.xtc import XTC, ISDFXTC
from pytc.jastrow.rexp import REXP

jax.config.update("jax_enable_x64", True)

class TestISDFRanges(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='ccpvdz', verbose=0)
        self.mf = scf.RHF(self.mol).run()
        
        self.jastrow_jax = REXP()
        self.jastrow_params_jax = {'alpha': jnp.array([1.0])}
        
        self.tc_jax = TC.from_pyscf(self.mf, self.jastrow_jax, grid_lvl=1)
        self.xtc_jax = XTC.from_pyscf(self.mf, self.jastrow_jax, grid_lvl=1)
        
        self.n_rank = 200
        self.isdf_tc = ISDFTC.from_tc(self.tc_jax, n_rank=self.n_rank)
        self.isdf_xtc = ISDFXTC.from_xtc(self.xtc_jax, n_rank=self.n_rank)
        
        self.nocc = self.isdf_tc.nocc
        self.n_orb = self.isdf_tc.n_orb

    def test_isdf_tc_get_2b_ranges(self):
        """Test ISDFTC.get_2b with ranges and block_str."""
        k2b_full = self.isdf_tc.get_2b(self.jastrow_params_jax)
        
        k2b_oooo = self.isdf_tc.get_2b(self.jastrow_params_jax, block_str='oooo')
        ref_oooo = k2b_full[:self.nocc, :self.nocc, :self.nocc, :self.nocc]
        np.testing.assert_allclose(np.array(k2b_oooo), np.array(ref_oooo), atol=1e-10)
        
        k2b_oovv = self.isdf_tc.get_2b(self.jastrow_params_jax, block_str='oovv')
        ref_oovv = k2b_full[:self.nocc, :self.nocc, self.nocc:, self.nocc:]
        np.testing.assert_allclose(np.array(k2b_oovv), np.array(ref_oovv), atol=1e-10)
        
        ranges = (slice(0, 2), slice(2, 4), slice(0, 2), slice(2, 4))
        k2b_ranges = self.isdf_tc.get_2b(self.jastrow_params_jax, ranges=ranges)
        ref_ranges = k2b_full[0:2, 2:4, 0:2, 2:4]
        np.testing.assert_allclose(np.array(k2b_ranges), np.array(ref_ranges), atol=1e-10)

    def test_isdf_xtc_get_delta_U_ranges(self):
        """Test ISDFXTC.get_delta_U with ranges and block_str."""
        dU_full = self.isdf_xtc.get_delta_U(self.jastrow_params_jax)
        
        dU_oooo = self.isdf_xtc.get_delta_U(self.jastrow_params_jax, block_str='oooo')
        ref_oooo = dU_full[:self.nocc, :self.nocc, :self.nocc, :self.nocc]
        np.testing.assert_allclose(np.array(dU_oooo), np.array(ref_oooo), atol=1e-10)
        
        dU_oovv = self.isdf_xtc.get_delta_U(self.jastrow_params_jax, block_str='oovv')
        ref_oovv = dU_full[:self.nocc, :self.nocc, self.nocc:, self.nocc:]
        np.testing.assert_allclose(np.array(dU_oovv), np.array(ref_oovv), atol=1e-10)
        
        ranges = (slice(0, 2), slice(2, 4), slice(0, 2), slice(2, 4))
        dU_ranges = self.isdf_xtc.get_delta_U(self.jastrow_params_jax, ranges=ranges)
        ref_ranges = dU_full[0:2, 2:4, 0:2, 2:4]
        np.testing.assert_allclose(np.array(dU_ranges), np.array(ref_ranges), atol=1e-10)

    def test_isdf_xtc_get_2b_ranges(self):
        """Test ISDFXTC.get_2b with ranges and block_str."""
        k2b_full = self.isdf_xtc.get_2b(self.jastrow_params_jax)
        
        k2b_oooo = self.isdf_xtc.get_2b(self.jastrow_params_jax, block_str='oooo')
        ref_oooo = k2b_full[:self.nocc, :self.nocc, :self.nocc, :self.nocc]
        np.testing.assert_allclose(np.array(k2b_oooo), np.array(ref_oooo), atol=1e-10)
        
        k2b_oovv = self.isdf_xtc.get_2b(self.jastrow_params_jax, block_str='oovv')
        ref_oovv = k2b_full[:self.nocc, :self.nocc, self.nocc:, self.nocc:]
        np.testing.assert_allclose(np.array(k2b_oovv), np.array(ref_oovv), atol=1e-10)

if __name__ == '__main__':
    unittest.main()
