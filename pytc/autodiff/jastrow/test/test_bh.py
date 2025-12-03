"""Tests for Boys-Handy Jastrow implementation."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from pyscf import gto

from pytc.autodiff.jastrow.bh import make_bh_jastrow, BHTerm
from pytc.jastrow.sm7 import SM7
from pytc.jastrow.sm17 import SM17

# Enable float64 support
jax.config.update("jax_enable_x64", True)

class MockMol:
    """Mock molecule for make_bh_jastrow."""
    def __init__(self, atoms, electrons):
        self.atoms = atoms # list of (symbol, coords) or similar
        self.nelectron = sum(electrons)
        self._atom_coords = [a[1] for a in atoms]
        self._atom_charges = [gto.charge(a[0]) for a in atoms]

    def atom_coords(self):
        return self._atom_coords
    
    def atom_charges(self):
        return self._atom_charges

def get_h2_molecule(bond_length=1.4):
    """Create H2 molecule mock."""
    atoms = [('H', (0, 0, 0)), ('H', (0, 0, bond_length))]
    electrons = (1, 1)
    return MockMol(atoms, electrons)

def get_atom_molecule(atom_symbol):
    """Create single atom molecule at origin."""
    atoms = [(atom_symbol, (0, 0, 0))]
    # Determine electrons for neutral atom
    charge = gto.charge(atom_symbol)
    n_alpha = (charge + 1) // 2
    n_beta = charge - n_alpha
    electrons = (n_alpha, n_beta)
    return MockMol(atoms, electrons)

def sm_coeffs_to_bh_terms(sm_class, atom_symbol):
    """Convert SM coefficients to BH terms."""
    sm_coeffs = sm_class._coeff_table[atom_symbol]
    terms = []
    for (m, n, o), coeff in sm_coeffs.items():
        terms.append(BHTerm(m, n, o, coeff))
    return terms

def compute_distances(r1, r2, atom_coords):
    """Compute r_ee and r_ae for two electrons and given atoms."""
    # r1, r2: (3,)
    # atom_coords: list of (3,) or (N, 3) array
    
    # r_ee
    d12 = jnp.linalg.norm(r1 - r2)
    r_ee = jnp.array([[0.0, d12], [d12, 0.0]])
    
    # r_ae
    pos = jnp.stack([r1, r2]) # (2, 3)
    atoms = jnp.array(atom_coords) # (natom, 3)
    
    # (2, 1, 3) - (1, natom, 3) -> (2, natom, 3)
    diff = pos[:, None, :] - atoms[None, :, :]
    r_ae = jnp.linalg.norm(diff, axis=-1) # (2, natom)
    
    return r_ee, r_ae

class TestBoysHandyVsSMBase(unittest.TestCase):
    """Base class for testing Boys-Handy against SM reference."""
    
    sm_class = None # Set in subclass
    
    def setUp(self):
        """Set up test cases."""
        self.key = random.PRNGKey(42)
        
    def _setup_atom_comparison(self, atom_symbol):
        """Set up BH and SM for a given atom."""
        # Create molecule with atom at origin
        mol = get_atom_molecule(atom_symbol)
        
        # Get SM coefficients and convert to BH terms
        bh_terms = sm_coeffs_to_bh_terms(self.sm_class, atom_symbol)
        
        # Create BH Jastrow
        # terms_per_nucleus expects list of lists (per atom type)
        # Here we have 1 atom type.
        bh_init, bh_apply = make_bh_jastrow(mol, terms_per_nucleus=[bh_terms])
        
        # Initialize parameters
        bh_params = bh_init()
        
        # SM uses fixed scaling r/(1+r), which corresponds to b=1, d=1.
        # BH uses softplus(param) for b and d.
        # We need to set params such that softplus(param) = 1.0.
        # param = log(exp(1) - 1)
        inv_softplus_1 = np.log(np.exp(1.0) - 1.0)
        
        bh_params['b_raw'] = jnp.full_like(bh_params['b_raw'], inv_softplus_1)
        bh_params['d_raw'] = jnp.full_like(bh_params['d_raw'], inv_softplus_1)
        
        # Create SM Jastrow
        sm = self.sm_class(atom=atom_symbol)
        
        return bh_apply, bh_params, sm, mol
    
    def _test_function_evaluation(self, atom_symbol):
        """Compare BH and SM function values."""
        bh_apply, bh_params, sm, mol = self._setup_atom_comparison(atom_symbol)
        
        # Generate random electron positions
        key1, key2 = random.split(self.key)
        r1 = random.normal(key1, (3,)) * 2.0
        r2 = random.normal(key2, (3,)) * 2.0
        
        # Compute BH value
        r_ee, r_ae = compute_distances(r1, r2, mol.atom_coords())
        bh_value = bh_apply(r_ee, bh_params, r_ae=r_ae)
        
        # Compute SM value
        sm_output = sm(np.array(r1), np.array(r2))
        sm_value = float(np.atleast_1d(sm_output).flat[0])
        
        # Compare values
        np.testing.assert_allclose(
            float(bh_value), sm_value, rtol=1e-5, atol=1e-8,
            err_msg=f"BH vs {self.sm_class.__name__} mismatch for {atom_symbol} at r1={r1}, r2={r2}"
        )

    def _test_function_evaluation_multiple_points(self, atom_symbol):
        """Test with multiple random positions."""
        bh_apply, bh_params, sm, mol = self._setup_atom_comparison(atom_symbol)
        
        n_tests = 10
        for i in range(n_tests):
            key_i = random.fold_in(self.key, i)
            key1, key2 = random.split(key_i)
            r1 = random.normal(key1, (3,)) * 2.0
            r2 = random.normal(key2, (3,)) * 2.0
            
            r_ee, r_ae = compute_distances(r1, r2, mol.atom_coords())
            bh_value = bh_apply(r_ee, bh_params, r_ae=r_ae)
            
            sm_output = sm(np.array(r1), np.array(r2))
            sm_value = float(np.atleast_1d(sm_output).flat[0])
            
            np.testing.assert_allclose(
                float(bh_value), sm_value, rtol=1e-5, atol=1e-8,
                err_msg=f"Test {i}: BH vs {self.sm_class.__name__} mismatch for {atom_symbol}"
            )

    def _test_gradient_r1(self, atom_symbol):
        """Compare gradients with respect to r1."""
        bh_apply, bh_params, sm, mol = self._setup_atom_comparison(atom_symbol)
        
        key1, key2 = random.split(self.key)
        r1 = random.normal(key1, (3,)) * 2.0
        r2 = random.normal(key2, (3,)) * 2.0
        
        def compute_wrapper(r1_val):
            r_ee, r_ae = compute_distances(r1_val, r2, mol.atom_coords())
            return bh_apply(r_ee, bh_params, r_ae=r_ae)

        bh_grad_fn = jax.grad(compute_wrapper)
        bh_grad = bh_grad_fn(r1)
        
        sm_grad = sm.grad(np.array(r1), np.array(r2))[0, 0, :]
        
        np.testing.assert_allclose(
            np.array(bh_grad), sm_grad, rtol=1e-4, atol=1e-7,
            err_msg=f"BH vs {self.sm_class.__name__} gradient mismatch for {atom_symbol} at r1={r1}, r2={r2}"
        )

class TestBoysHandyVsSM7(TestBoysHandyVsSMBase):
    """Test Boys-Handy implementation against SM7 reference."""
    sm_class = SM7
    
    def test_he_function_evaluation(self):
        self._test_function_evaluation('He')
        
    def test_be_function_evaluation(self):
        self._test_function_evaluation('Be')
        
    def test_he_function_evaluation_multiple_points(self):
        self._test_function_evaluation_multiple_points('He')
        
    def test_be_function_evaluation_multiple_points(self):
        self._test_function_evaluation_multiple_points('Be')
        
    def test_he_gradient_r1(self):
        self._test_gradient_r1('He')
        
    def test_be_gradient_r1(self):
        self._test_gradient_r1('Be')

class TestBoysHandyVsSM17(TestBoysHandyVsSMBase):
    """Test Boys-Handy implementation against SM17 reference."""
    sm_class = SM17
    
    def test_he_function_evaluation(self):
        self._test_function_evaluation('He')
        
    def test_be_function_evaluation(self):
        self._test_function_evaluation('Be')
        
    def test_he_function_evaluation_multiple_points(self):
        self._test_function_evaluation_multiple_points('He')
        
    def test_be_function_evaluation_multiple_points(self):
        self._test_function_evaluation_multiple_points('Be')
        
    def test_he_gradient_r1(self):
        self._test_gradient_r1('He')
        
    def test_be_gradient_r1(self):
        self._test_gradient_r1('Be')

class TestBoysHandy(unittest.TestCase):

    """Test Boys-Handy Jastrow implementation."""
    
    def setUp(self):
        self.mol = get_h2_molecule()
        self.key = random.PRNGKey(0)
        
        # Create Boys-Handy Jastrow with custom terms
        terms = [
            [BHTerm(0, 0, 1, 0.5),  # e-e cusp term
             BHTerm(1, 0, 0, -0.1), # e-n term (attractive)
             BHTerm(2, 0, 0, -0.1)] # higher order term (attractive)
        ]
        self.bh_init, self.bh_apply = make_bh_jastrow(self.mol, terms_per_nucleus=terms)
        self.params = self.bh_init()

    def test_init(self):
        """Test initialization."""
        # Test default initialization
        bh_init, _ = make_bh_jastrow(self.mol)
        params = bh_init()
        
        # Check parameter structure
        self.assertIn('b_raw', params)
        self.assertIn('d_raw', params)
        self.assertIn('c_raw', params)
        
        # Check shapes - H2 has 2 atoms but only 1 atom type (H)
        self.assertEqual(params['b_raw'].shape, (1,))  # 1 atom type
        self.assertEqual(params['d_raw'].shape, (1,))  # 1 atom type
        # Default has 17 terms per atom type
        self.assertEqual(params['c_raw'].shape[0], 1)  # 1 atom type
        self.assertEqual(params['c_raw'].shape[1], 17)  # 17 default terms

    def test_electron_symmetry(self):
        """Test symmetry with respect to electron exchange."""
        r1 = jnp.array([0., 0., 0.])
        r2 = jnp.array([0., 0., 1.0])
        
        r_ee1, r_ae1 = compute_distances(r1, r2, self.mol.atom_coords())
        value1 = self.bh_apply(r_ee1, self.params, r_ae=r_ae1)
        
        r_ee2, r_ae2 = compute_distances(r2, r1, self.mol.atom_coords())
        value2 = self.bh_apply(r_ee2, self.params, r_ae=r_ae2)
        
        np.testing.assert_allclose(value1, value2, rtol=1e-7)

    def test_electron_cusp(self):
        """Test electron-electron cusp condition."""
        r1 = jnp.array([0., 0., 0.])
        eps = 1e-5
        r2 = jnp.array([eps, 0., 0.])
        
        # Compute numerical gradient at small separation
        def compute_wrapper(r2_val):
            r_ee, r_ae = compute_distances(r1, r2_val, self.mol.atom_coords())
            return self.bh_apply(r_ee, self.params, r_ae=r_ae)

        grad_fn = jax.grad(compute_wrapper)
        grad_val = grad_fn(r2)[0]  # x-component of gradient at r2=(eps,0,0)
        
        self.assertGreater(grad_val, 0.0, "Gradient should be positive at small separation")
        self.assertLess(grad_val, 1.0, "Gradient should be less than 1.0")

    def test_nuclear_decay(self):
        """Test decay of correlation with nuclear distance."""
        r1_near = jnp.array([0., 0., 0.1])
        r1_far = jnp.array([0., 0., 5.0])
        r2 = jnp.array([0., 0., -0.5])
        
        r_ee_near, r_ae_near = compute_distances(r1_near, r2, self.mol.atom_coords())
        value_near = self.bh_apply(r_ee_near, self.params, r_ae=r_ae_near)
        
        r_ee_far, r_ae_far = compute_distances(r1_far, r2, self.mol.atom_coords())
        value_far = self.bh_apply(r_ee_far, self.params, r_ae=r_ae_far)
        
        self.assertTrue(jnp.isfinite(value_near), "Value at near distance should be finite")
        self.assertTrue(jnp.isfinite(value_far), "Value at far distance should be finite")

if __name__ == '__main__':
    unittest.main()
