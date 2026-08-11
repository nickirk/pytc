"""Tests for JAX implementation of Transcorrelated method."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.legacy.tc import TC as TC_numpy
from pytc.integrals.tc import TC as TC_jax
from pytc.jastrow import Poly
from pytc import tc_helper

jax.config.update("jax_enable_x64", True)

class TestTC(unittest.TestCase):
    """Test JAX implementation of TC method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mol = gto.M(atom='He 0 0 0', basis='sto-3g')
        self.mf = scf.RHF(self.mol)
        self.mf.kernel()
        
        self.params = jnp.array([1.0])
        self.jastrow_jax = Poly()  # No params in constructor
        
        class PolyNumpy:
            """Numpy version of Poly for comparison."""
            def __init__(self, params):
                self.params = params

            def __call__(self, r1, r2):
                r12 = np.sqrt(np.sum((r1 - r2)**2, axis=-1))
                return self.params[0] * r12

            def grad(self, r1, r2):
                """Combined gradient calculation for comparison."""
                diff = r1[:, None, :] - r2[None, :, :]
                r12_sq = np.sum(diff * diff, axis=-1)
                r12 = np.sqrt(r12_sq + 1e-10)  # Add epsilon for stability
                cutoff = 1.0 / (1.0 + np.exp(-(r12 - 1e-5) * 1e6))  # Sigmoid cutoff
                grad = np.where(r12_sq[..., None] > 1e-10,
                               diff * self.params[0] * cutoff[..., None] / r12[..., None],
                               np.zeros_like(diff))
                return grad
        self.jastrow_numpy = PolyNumpy(self.params)
        
        self.tc_jax = TC_jax.from_pyscf(self.mf, self.jastrow_jax)
        self.tc_numpy = TC_numpy(self.mf, self.jastrow_numpy)
        
    def test_grid_initialization(self):
        """Test grid initialization and conversion to JAX arrays."""
        self.assertIsNotNone(self.tc_jax.grid_points)
        self.assertIsNotNone(self.tc_jax.weights)
        self.assertTrue(isinstance(self.tc_jax.grid_points, jnp.ndarray))
        self.assertTrue(isinstance(self.tc_jax.weights, jnp.ndarray))
        
    def test_basis_evaluation(self):
        """Test basis function evaluation on grid."""
        phi_jax = self.tc_jax.phi
        grad_phi_jax = self.tc_jax.grad_phi
        
        rho_numpy, nabla_rho_numpy = self.tc_numpy._eval_basis_on_grid()
        
        np.testing.assert_allclose(
            np.asarray(phi_jax), rho_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="JAX and numpy basis evaluations don't match"
        )
        np.testing.assert_allclose(
            np.asarray(grad_phi_jax), nabla_rho_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="JAX and numpy basis gradients don't match"
        )

    def test_from_pyscf_grid_chunking_matches_unchunked(self):
        """Chunked and effectively-unchunked from_pyscf must agree exactly.

        Guards the grid-chunked AO->MO transform (task: H50 2xA100 OOM fix)
        against regression: a tiny explicit grid_chunk_size forces many
        chunks, an oversized one forces a single chunk (the old,
        unchunked behavior); both must produce identical phi/grad_phi.
        """
        tc_unchunked = TC_jax.from_pyscf(self.mf, self.jastrow_jax, grid_chunk_size=10**9)
        tc_chunked = TC_jax.from_pyscf(self.mf, self.jastrow_jax, grid_chunk_size=3)

        np.testing.assert_array_equal(
            np.asarray(tc_chunked.phi), np.asarray(tc_unchunked.phi),
            err_msg="Chunked phi doesn't exactly match unchunked phi"
        )
        np.testing.assert_array_equal(
            np.asarray(tc_chunked.grad_phi), np.asarray(tc_unchunked.grad_phi),
            err_msg="Chunked grad_phi doesn't exactly match unchunked grad_phi"
        )

    def test_from_pyscf_grid_chunk_size_validation(self):
        """Non-positive grid_chunk_size raises a clear error, not a confusing one."""
        with self.assertRaises(ValueError):
            TC_jax.from_pyscf(self.mf, self.jastrow_jax, grid_chunk_size=0)
        with self.assertRaises(ValueError):
            TC_jax.from_pyscf(self.mf, self.jastrow_jax, grid_chunk_size=-5)

    def test_get_2b_against_numpy(self):
        """Test two-body term calculation against numpy version."""
        correction_jax = self.tc_jax.get_2b(self.params)
        
        eri1 = tc_helper.get_eri(self.mf)
        
        result_jax = eri1 + correction_jax
        
        # Numpy version returns full effective ERI
        result_numpy = self.tc_numpy.get_2b()
        
        result_jax = np.asarray(result_jax)
        
        np.testing.assert_allclose(
            result_jax, result_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="JAX and numpy two-body terms don't match"
        )

    def test_get_eri_without_incore_cache(self):
        """Test AO-to-MO transformation when PySCF does not cache AO ERIs."""
        expected = tc_helper.get_eri(self.mf)
        self.mf._eri = None

        actual = tc_helper.get_eri(self.mf)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    
    def test_mo_coeff_handling(self):
        """Test handling of molecular orbital coefficients."""
        new_mo = self.mf.mo_coeff + 0.1
        
        tc_jax_new = TC_jax.from_pyscf(self.mf, self.jastrow_jax, mo_coeff=new_mo)
        tc_numpy_new = TC_numpy(self.mf, self.jastrow_numpy, mo_coeff=new_mo)
        
        correction_jax = tc_jax_new.get_2b(self.params)
        eri1 = tc_helper.get_eri(self.mf, mo_coeff=new_mo)
        result_jax = eri1 + correction_jax
        
        result_numpy = tc_numpy_new.get_2b()
        
        np.testing.assert_allclose(
            np.asarray(result_jax), result_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Results don't match with explicit mo_coeff"
        )

    def test_two_body_terms(self):
        """Test two-body term calculation."""
        r1 = np.array([[0.0, 0.0, 0.0]])
        r2 = np.array([[0.0, 0.0, 1.0]])
        grad_jax = self.jastrow_jax.grad_r(r1, r2, self.params)[:, None, :]
        grad_numpy = self.jastrow_numpy.grad(r1, r2)
        
        np.testing.assert_allclose(
            grad_jax, grad_numpy,
            rtol=1e-5, atol=1e-5,
            err_msg="Gradients don't match"
        )

if __name__ == '__main__':
    unittest.main()
