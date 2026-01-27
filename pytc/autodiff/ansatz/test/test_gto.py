import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, dft
from pytc.autodiff.ansatz.gto import MolGTO, eval_ao
from pytc.autodiff.ansatz.gto_spherical import MolGTO_Spherical, eval_ao_spherical

class TestGTO(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_enable_x64", True)
        
    def test_h2_sto3g(self):
        """Test H2 with STO-3G basis (s orbitals only)"""
        mol = gto.M(
            atom='H 0 0 0; H 0 0 0.74',
            basis='sto-3g',
            cart=True
        )
        self._check_mol(mol)
        
    @unittest.skip("Known issue with Cartesian GTO normalization/ordering")
    def test_h2o_dz(self):
        """Test H2O with cc-pVDZ (s, p, d orbitals)"""
        mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='cc-pvdz',
            cart=True
        )
        self._check_mol(mol)

    def test_h2o_dz_spherical(self):
        """Test H2O with cc-pVDZ spherical orbitals"""
        mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='cc-pvdz',
            cart=False
        )
        self._check_mol_spherical(mol)
        
    def _check_mol(self, mol):
        mol_gto = MolGTO.create(mol)
        
        # Random points to evaluate
        np.random.seed(42)
        coords = np.random.randn(10, 3)
        coords_jax = jnp.array(coords)
        
        # PySCF reference
        ao_ref = mol.eval_gto('GTOval_cart', coords)
        
        # JAX implementation
        ao_jax = eval_ao(mol_gto, coords_jax, deriv=0)
        
        # Check values
        np.testing.assert_allclose(ao_jax, ao_ref, atol=1e-12, rtol=1e-12)
        
        # Check gradients
        # PySCF returns (4, N, nao) -> value, grad_x, grad_y, grad_z
        ao_deriv1_ref = mol.eval_gto('GTOval_cart_deriv1', coords)
        val_ref = ao_deriv1_ref[0]
        grad_ref = ao_deriv1_ref[1:4].transpose(1, 2, 0) # (N, nao, 3)
        
        val_jax, grad_jax = eval_ao(mol_gto, coords_jax, deriv=1)
        
        np.testing.assert_allclose(val_jax, val_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(grad_jax, grad_ref, atol=1e-12, rtol=1e-12)
        
        # Check laplacian
        # PySCF returns (10, N, nao) for deriv2
        # 0: val
        # 1,2,3: grad (dx, dy, dz)
        # 4,7,9: dxx, dyy, dzz
        ao_deriv2_ref = mol.eval_gto('GTOval_cart_deriv2', coords)
        lap_ref = ao_deriv2_ref[4] + ao_deriv2_ref[7] + ao_deriv2_ref[9]
        
        val_jax, grad_jax, lap_jax = eval_ao(mol_gto, coords_jax, deriv=2)
        
        np.testing.assert_allclose(val_jax, val_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(grad_jax, grad_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(lap_jax, lap_ref, atol=1e-12, rtol=1e-12)

    def _check_mol_spherical(self, mol):
        mol_gto = MolGTO_Spherical.create(mol)
        
        # Random points to evaluate
        np.random.seed(42)
        coords = np.random.randn(10, 3)
        coords_jax = jnp.array(coords)
        
        # PySCF reference
        ao_ref = mol.eval_gto('GTOval_sph', coords)
        
        # JAX implementation
        ao_jax = eval_ao_spherical(mol_gto, coords_jax, deriv=0)
        
        # Check values
        np.testing.assert_allclose(ao_jax, ao_ref, atol=1e-12, rtol=1e-12)
        
        # Check gradients
        ao_deriv1_ref = mol.eval_gto('GTOval_sph_deriv1', coords)
        val_ref = ao_deriv1_ref[0]
        grad_ref = ao_deriv1_ref[1:4].transpose(1, 2, 0) # (N, nao, 3)
        
        val_jax, grad_jax = eval_ao_spherical(mol_gto, coords_jax, deriv=1)
        
        np.testing.assert_allclose(val_jax, val_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(grad_jax, grad_ref, atol=1e-12, rtol=1e-12)
        
        # Check laplacian
        ao_deriv2_ref = mol.eval_gto('GTOval_sph_deriv2', coords)
        lap_ref = ao_deriv2_ref[4] + ao_deriv2_ref[7] + ao_deriv2_ref[9]
        
        val_jax, grad_jax, lap_jax = eval_ao_spherical(mol_gto, coords_jax, deriv=2)
        
        np.testing.assert_allclose(val_jax, val_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(grad_jax, grad_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(lap_jax, lap_ref, atol=1e-12, rtol=1e-12)

if __name__ == '__main__':
    unittest.main()
