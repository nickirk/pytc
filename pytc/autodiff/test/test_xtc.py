"""Tests for JAX implementation of Extended Transcorrelated methods."""

import unittest
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from pyscf import gto, scf

from pytc.xtc import XTC as XTC_numpy
from pytc.jastrow import REXP as REXP_np
from pytc.autodiff.xtc import XTC as XTC_jax
from pytc.autodiff.jastrow import REXP as REXP_jax
from pytc.autodiff import tc_helper

# Enable float64 support
jax.config.update("jax_enable_x64", True)

def get_be_ccpvdz():
    """Return a Be atom with cc-pVDZ basis for testing."""
    mol = gto.M(atom='Be 0 0 0', basis='ccpvdz', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

class TestXTC(unittest.TestCase):
    """Test JAX implementation of Extended Transcorrelated methods."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case using Be atom."""
        # Get mean-field data
        _, cls.mf = get_be_ccpvdz()
        
        # Create simple Jastrow factors for both implementations
        cls.params_jax = {'alpha': jnp.array([1.0])}
        cls.params_numpy = np.array([1.0])
        
        cls.jastrow_jax = REXP_jax()  # Remove params from constructor
        cls.jastrow_numpy = REXP_np(cls.params_numpy)  # Keep numpy version unchanged
        
        # Initialize XTC calculators with low grid level for testing
        cls.xtc_jax = XTC_jax.from_pyscf(cls.mf, cls.jastrow_jax, grid_lvl=2)
        cls.xtc_numpy = XTC_numpy(cls.mf, cls.jastrow_numpy, grid_lvl=2)
        
    def test_grid_initialization(self):
        """Test grid initialization and conversion to JAX arrays."""
        np.testing.assert_allclose(
            np.asarray(self.xtc_jax.grid_points),
            self.xtc_numpy.grid_points,
            rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(self.xtc_jax.weights),
            self.xtc_numpy.weights,
            rtol=1e-7, atol=1e-7
        )

    def test_basis_evaluation(self):
        """Test basis function evaluation on grid."""
        phi_jax = self.xtc_jax.phi
        rho_numpy, _ = self.xtc_numpy._eval_basis_on_grid()
        
        np.testing.assert_allclose(
            np.asarray(phi_jax),
            rho_numpy,
            rtol=1e-5, atol=1e-5
        )

    def test_v_vector(self):
        """Test v_vector calculation."""
        # Get orbital values on grid
        mo_values_jax = self.xtc_jax.phi
        mo_values_numpy, _ = self.xtc_numpy._eval_basis_on_grid()
        
        # Prepare paired indices
        phi_paired_jax = jnp.einsum('in,jn->ijn', 
                                   mo_values_jax, 
                                   mo_values_jax).reshape(-1, len(self.xtc_jax.weights))
        phi_paired_numpy = np.einsum('in,jn->ijn', 
                                    mo_values_numpy, 
                                    mo_values_numpy).reshape(-1, len(self.xtc_numpy.weights))
        
        v_vector_jax = self.xtc_jax._calc_v_batch(
            self.xtc_jax.grid_points,
            self.xtc_jax.phi,
            self.xtc_jax.weights,
            self.params_jax,
            batch_size=len(self.xtc_jax.grid_points)
        )
        v_vector_numpy = self.xtc_numpy._calc_v_vector(phi_paired_numpy)
        
        np.testing.assert_allclose(
            np.asarray(v_vector_jax),
            v_vector_numpy,
            rtol=1e-5, atol=1e-5
        )
        
    def test_delta_U(self):
        """Test delta_U calculation."""
        # First test delta_U matrices
        dm1_jax = self.xtc_jax._get_mf_dm()
        # dm1_numpy = self.xtc_numpy._get_mf_dm() # Not needed if we trust JAX dm1
        
        delta_U_jax = self.xtc_jax.get_delta_U(self.params_jax, dm1_jax)  # Add params
        delta_U_numpy = self.xtc_numpy.get_delta_U()
        
        np.testing.assert_allclose(
            np.asarray(delta_U_jax),
            delta_U_numpy,
            rtol=1e-6, atol=1e-6
        )
        
        # Test symmetry property
        np.testing.assert_allclose(
            np.asarray(delta_U_jax),
            np.asarray(delta_U_jax.transpose(2,3,0,1)),
            rtol=1e-7, atol=1e-7
        )
        
        # Test gradient calculations with defined points
        test_r1 = np.array([[0.0, 0.0, 0.0]])
        test_r2 = np.array([[0.0, 0.0, 1.0]])
        
        grad_jax = self.xtc_jax.jastrow_factor.grad_r(test_r1, test_r2, self.params_jax)
        grad_numpy = self.jastrow_numpy.grad(test_r1, test_r2)
        
        np.testing.assert_allclose(
            np.asarray(grad_jax[:, None, :]),  # Add middle dimension to match numpy shape
            grad_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Gradients don't match"
        )

    def test_delta_h(self):
        """Test delta_h calculation."""
        delta_h_jax = self.xtc_jax.get_delta_h(self.params_jax)  # Add params
        delta_h_numpy = self.xtc_numpy._calc_delta_h()
        
        np.testing.assert_allclose(
            np.asarray(delta_h_jax),
            delta_h_numpy,
            rtol=1e-5, atol=1e-5
        )
        
        # Test hermiticity
        np.testing.assert_allclose(
            np.asarray(delta_h_jax),
            np.asarray(delta_h_jax.T.conj()),
            rtol=1e-7, atol=1e-7
        )

    def test_one_body(self):
        """Test one-body operator calculation."""
        # JAX returns only delta_h
        delta_h_jax = self.xtc_jax.get_1b(self.params_jax)
        
        # Get standard h1e
        h1e_std = tc_helper.get_hcore(self.mf)
        
        # Combine
        h1e_jax = h1e_std + delta_h_jax
        
        # Numpy returns full h1e
        h1e_numpy = self.xtc_numpy.get_1b()
        
        np.testing.assert_allclose(
            np.asarray(h1e_jax),
            h1e_numpy,
            rtol=1e-5, atol=1e-5
        )

    def test_two_body(self):
        """Test two-body term calculation."""
        # JAX returns correction + delta_U
        correction_jax = self.xtc_jax.get_2b(self.params_jax)
        
        # Get standard ERI
        eri_std = tc_helper.get_eri(self.mf)
        
        # Combine
        v2e_jax = eri_std + correction_jax
        
        # Numpy returns full effective ERI
        v2e_numpy = self.xtc_numpy.get_2b()
        
        np.testing.assert_allclose(
            np.asarray(v2e_jax),
            v2e_numpy,
            rtol=1e-5, atol=1e-5
        )

    def test_nhccsd(self):
        from pyscf.cc import rccsd, CCSD
        from pyscf import ao2mo
        import numpy as np
        from functools import reduce

        mycc = CCSD(self.mf)
        e_corr, t1, t2 = mycc.kernel()
        mycc.verbose = 5
        print("E_CCSD = ", e_corr)
        #self.assertAlmostEqual(e_corr, -0.04503138331130402, places=6)
        t = mycc.amplitudes_to_vector(t1, t2)
        print("|t2| = ", np.linalg.norm(t2))
        print("|t1+t2| = ", np.linalg.norm(t))
        eri1 = ao2mo.incore.full(self.xtc_jax.mf._eri if hasattr(self.xtc_jax, 'mf') else self.mf._eri, 
                                 self.xtc_jax.mo_coeff, compact=False)
        eri1 = ao2mo.restore(1, eri1, self.xtc_jax.mo_coeff.shape[1])
        h1e = mycc._scf.get_hcore()
        h1e = reduce(np.dot, (self.xtc_jax.mo_coeff.T, h1e, self.xtc_jax.mo_coeff))
        
        e_hf_0 = 2. * np.einsum('ii->', h1e[:mycc.nocc, :mycc.nocc])
        e_dir = 2. * np.einsum('jjii->', eri1[:mycc.nocc, :mycc.nocc, :mycc.nocc, :mycc.nocc])
        e_ex = -1. * np.einsum('ijji->', eri1[:mycc.nocc, :mycc.nocc, :mycc.nocc, :mycc.nocc])
        e_hf_0 += (e_dir + e_ex) + mycc._scf.energy_nuc()
        print("Check e_hf = ",  e_hf_0)
        self.assertAlmostEqual(e_hf_0, -14.57233763095337, places=6)

        myrcc = rccsd.RCCSD(self.mf)
        #myrcc.verbose = 5
        eris = self.xtc_jax.make_eris(self.mf, self.params_jax)  # Pass mf and params
        tc_e_corr, t1, t2 = myrcc.kernel(eris=eris)
        t = myrcc.amplitudes_to_vector(t1, t2)
        print("|t2| = ", np.linalg.norm(t2))
        print("|t1+t2| = ", np.linalg.norm(t))
        print("corr E_XTC_CCSD = ", myrcc.e_corr)
        self.assertAlmostEqual(tc_e_corr, -0.03271941719618445, places=6)
        # get the hf energy using fock and eris
        no = myrcc.nocc
        
        # Reconstruct full h1e for checking
        tc_h1e_corr = self.xtc_jax.get_1b(self.params_jax)
        h1e_std = tc_helper.get_hcore(self.mf, self.xtc_jax.mo_coeff)
        tc_h1e = h1e_std + tc_h1e_corr
        
        tc_e_hf = 2. * np.einsum('ii->', tc_h1e[:no, :no])
        tc_e_dir = 2. * np.einsum('jjii->', eris.oooo)
        tc_e_ex = -1. * np.einsum('ijji->', eris.oooo)

        tc_e_hf += (tc_e_dir + tc_e_ex) + eris.e_core 
        print("E_XTC_CCSD = ", myrcc.e_corr + tc_e_hf)
        self.assertAlmostEqual(tc_e_hf + tc_e_corr, -14.65640358838446, places=6)


if __name__ == '__main__':
    unittest.main()
