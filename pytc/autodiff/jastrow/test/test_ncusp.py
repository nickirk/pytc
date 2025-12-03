import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto

# Import the factory function
from pytc.autodiff.jastrow.ncusp import make_ncusp_jastrow

class TestNuclearCuspJastrow(unittest.TestCase):
    """Test cases for NuclearCuspJastrow functional interface."""
    
    def setUp(self):
        """Set up H2O molecule."""
        self.mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pvdz')
        self.init, self.apply = make_ncusp_jastrow(self.mol, n_radial=1000)
        
    def test_param_initialization(self):
        """Test parameter initialization."""
        params = self.init()
        
        # Check keys
        self.assertIn('rc', params)
        self.assertIn('X4', params)
        
        # Check shapes
        # H2O has 2 types of atoms: H (Z=1) and O (Z=8)
        # unique_Z should be [1, 8]
        unique_Z = np.unique(self.mol.atom_charges())
        n_types = len(unique_Z)
        
        self.assertEqual(params['rc'].shape, (n_types,))
        self.assertEqual(params['X4'].shape, (n_types,))
        
        # Check rc initialization (approx 1/Z)
        for i, Z in enumerate(unique_Z):
            self.assertAlmostEqual(params['rc'][i], 1.0/Z, delta=0.1)

    def test_apply_finite_values(self):
        """Test that apply returns finite values."""
        params = self.init()
        
        # Create random electron positions (nelec, natom)
        # We need r_ae, which is distance from each electron to each nucleus
        nelec = self.mol.nelec[0] + self.mol.nelec[1]
        natom = self.mol.natm
        
        # Random positions
        key = jax.random.PRNGKey(42)
        elec_pos = jax.random.normal(key, (nelec, 3))
        atom_pos = jnp.array(self.mol.atom_coords())
        
        # Compute r_ae
        diff = elec_pos[:, None, :] - atom_pos[None, :, :]
        r_ae = jnp.linalg.norm(diff, axis=-1)
        
        # Apply Jastrow
        val = self.apply(r_ae, params)
        
        # Check finite
        self.assertTrue(jnp.isfinite(val))
        
        # Check shape (scalar)
        self.assertEqual(val.shape, ())

    def test_cusp_behavior(self):
        """Test behavior near nucleus (cusp condition)."""
        params = self.init()
        
        # Pick the first nucleus (Oxygen, at 0,0,0)
        nucleus_idx = 0
        Z = self.mol.atom_charges()[nucleus_idx]
        
        # Define a function to evaluate Jastrow for a single electron at position r
        def eval_single_elec(r):
            # Construct r_ae for this single electron
            # Other electrons are far away (dummy)
            nelec = self.mol.nelec[0] + self.mol.nelec[1]
            
            # r is (3,)
            # atom_pos is (natom, 3)
            atom_pos = jnp.array(self.mol.atom_coords())
            
            # Distance to all nuclei
            diff = r[None, :] - atom_pos
            dists = jnp.linalg.norm(diff, axis=-1) # (1, natom)
            
            # Create full r_ae matrix (nelec, natom)
            # We put the probe electron at index 0, others far away (distance 100.0)
            r_ae_full = jnp.ones((nelec, self.mol.natm)) * 100.0
            r_ae_full = r_ae_full.at[0, :].set(dists)
            
            return self.apply(r_ae_full, params)

        # Compute gradient at small r from nucleus
        r_eval = jnp.array([1e-6, 0.0, 0.0]) # Slightly offset from 0,0,0
        
        grad_fun = jax.grad(eval_single_elec)
        grad = grad_fun(r_eval)
        
        # Radial derivative d/dr = grad . r_hat
        r_hat = r_eval / jnp.linalg.norm(r_eval)
        d_dr = jnp.dot(grad, r_hat)
        
        # Cusp condition: d ln Psi / dr = -Z
        self.assertAlmostEqual(d_dr, -Z, delta=0.5) # Relaxed tolerance due to spline approx

    def test_gradients_finite(self):
        """Test that gradients w.r.t parameters are finite."""
        params = self.init()
        
        nelec = self.mol.nelec[0] + self.mol.nelec[1]
        key = jax.random.PRNGKey(0)
        elec_pos = jax.random.normal(key, (nelec, 3))
        atom_pos = jnp.array(self.mol.atom_coords())
        diff = elec_pos[:, None, :] - atom_pos[None, :, :]
        r_ae = jnp.linalg.norm(diff, axis=-1)
        
        jax.debug.print("r_ae: {}", r_ae) # Added jax.debug.print here
        
        def loss(p):
            return self.apply(r_ae, p)
            
        grads = jax.grad(loss)(params)
        
        for k, g in grads.items():
            self.assertTrue(jnp.all(jnp.isfinite(g)), f"Gradient for {k} contains NaNs/Infs")

if __name__ == '__main__':
    unittest.main()
