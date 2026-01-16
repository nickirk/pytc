import unittest
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.autodiff.df import isdf_decompose
from pytc.autodiff.tc import TC

jax.config.update("jax_enable_x64", True)

class TestISDFReconstruction(unittest.TestCase):
    def setUp(self):
        # Setup H4 chain system with ccpvtz basis
        atom = []
        for i in range(4):
            atom.append(f'H 0 0 {i*1.4}')
        self.mol = gto.M(atom=atom, basis='ccpvtz', unit='Bohr', verbose=0)
        self.mf = scf.RHF(self.mol).run()
        
        # Initialize TC object to get orbitals and gradients on grid
        # We use a dummy Jastrow factor as we only need phi and grad_phi
        from pytc.autodiff.jastrow.rexp import REXP
        self.tc = TC.from_pyscf(self.mf, REXP(), grid_lvl=1)
        
        self.phi = self.tc.phi
        self.grad_phi = self.tc.grad_phi
        self.weights = self.tc.weights
        
        print(f"\nSystem: H4, Basis: ccpvdz, Grid size: {self.phi.shape[1]}")

    def test_reconstruction_convergence(self):
        """Verify that phi and grad_phi overlaps are reconstructed accurately and converge with rank."""
        n_orb, n_grid = self.phi.shape
        
        # Exact overlaps on grid
        # S_ij = sum_g w_g phi_i(g) phi_j(g)
        phi_weighted = self.phi * self.weights[None, :]
        S_exact = jnp.dot(phi_weighted, self.phi.T)
        
        # G_ij,c = sum_g w_g grad_phi_i,c(g) phi_j(g)
        G_exact = jnp.einsum('igc,g,jg->ijc', self.grad_phi, self.weights, self.phi)
        
        ranks = [400, 600, 800]
        
        print(f"\n{'Rank':<10} {'S Rel Error':<15} {'S Max Abs':<15} {'G Rel Error':<15} {'G Max Abs':<15}")
        print("-" * 75)
        
        prev_S_error = float('inf')
        prev_G_error = float('inf')
        
        for n_rank in ranks:
            # Perform ISDF decomposition
            phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots = isdf_decompose(
                self.phi, self.grad_phi, n_rank, n_rank, weights=self.weights, use_iterative=True, rcond=1e-16
            )
            
            # Reconstruct overlaps
            # S_reconst_ij = sum_m C_phi_ij,m * (sum_g xi_phi_m,g * w_g)
            # C_phi_ij,m = phi_i,m * phi_j,m
            xi_phi_weighted_sum = jnp.dot(xi_phi, self.weights)
            C_phi = jnp.einsum('pm,qm->pqm', phi_piv, phi_piv).reshape(-1, len(pivots))
            S_reconst = jnp.dot(C_phi, xi_phi_weighted_sum).reshape(n_orb, n_orb)
            
            # G_reconst_ij,c = sum_m C_grad_ij,m,c * (sum_g xi_grad_m,g,c * w_g)
            # xi_grad is (n_fused, n_grid, 3)
            xi_grad_weighted_sum = jnp.einsum('mgc,g->mc', xi_grad, self.weights)
            # C_grad is (n_orb^2, n_fused, 3)
            # C_grad_ij,m,c = grad_phi_i,m,c * phi_j,m
            C_grad = jnp.einsum('pmc,qm->pqmc', grad_phi_piv, phi_piv).reshape(-1, len(pivots), 3)
            G_reconst = jnp.einsum('nmc,mc->nc', C_grad, xi_grad_weighted_sum).reshape(n_orb, n_orb, 3)
            
            # Compute relative errors
            S_error = jnp.linalg.norm(S_reconst - S_exact) / jnp.linalg.norm(S_exact)
            G_error = jnp.linalg.norm(G_reconst - G_exact) / jnp.linalg.norm(G_exact)
            
            # Compute max absolute errors
            S_max_abs = jnp.max(jnp.abs(S_reconst - S_exact))
            G_max_abs = jnp.max(jnp.abs(G_reconst - G_exact))
            
            print(f"{n_rank:<10} {S_error:<15.2e} {S_max_abs:<15.2e} {G_error:<15.2e} {G_max_abs:<15.2e}")
            
            # Check for convergence
            if n_rank > ranks[0]:
                if S_error > 1e-12:
                    self.assertLess(S_error, prev_S_error, f"S error did not decrease at rank {n_rank}")
                if G_error > 1e-12:
                    self.assertLess(G_error, prev_G_error, f"G error did not decrease at rank {n_rank}")
            
            prev_S_error = S_error
            prev_G_error = G_error
            
        # Final accuracy check
        self.assertLess(S_error, 1e-4, f"Final S reconstruction error {S_error} is too high")
        self.assertLess(G_error, 1e-3, f"Final G reconstruction error {G_error} is too high")

if __name__ == '__main__':
    unittest.main()
