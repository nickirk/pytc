"""Implementation of the quantum many-body wavefunction ansatz."""

import jax
import jax.numpy as jnp
from typing import List, Any

class SlaterJastrow:
    """Quantum many-body wavefunction ansatz combining Jastrow factor with Slater determinants."""
    
    def __init__(self, mol, jastrow, dets: List[Any], linear_coeffs: jnp.ndarray):
        """Initialize the ansatz.
        
        Args:
            jastrow: Jastrow factor object
            determinants: List of Slater determinant objects
            coefficients: Array of linear coefficients for determinants
        """
        self.mol = mol
        self.jastrow = jastrow
        self.dets = dets
        self.linear_coeffs = jnp.asarray(linear_coeffs, dtype=jnp.float64)
        
    def __call__(self, elec_coords):
        """Evaluate wavefunction at given electron positions."""
        
        # Vectorize Jastrow computation over all pairs
        vmap_single = jax.vmap(self.jastrow._compute, in_axes=(None, 0, None))
        vmap_all = jax.vmap(vmap_single, in_axes=(0, None, None))
        
        # Compute all pairwise values at once
        all_pairs = vmap_all(elec_coords, elec_coords, self.jastrow.params)
        
        # Sum upper triangle (excluding diagonal) for total exponent
        mask = jnp.triu(jnp.ones_like(all_pairs), k=1)
        jastrow_val = jnp.exp(jnp.sum(all_pairs * mask))
        
        # Evaluate determinants (already vectorized internally)
        det_vals = jnp.array([det.value(elec_coords) for det in self.dets])
        linear_combo = jnp.sum(self.linear_coeffs * det_vals)
        
        return jastrow_val * linear_combo

    def update_jastrow(self, new_jastrow_params):
        """Update Jastrow parameters.
        
        Args:
            new_jastrow_params: New parameters for Jastrow factor
            
        Returns:
            New Ansatz instance with updated Jastrow
        """
        new_jastrow = self.jastrow.update(new_jastrow_params)
        return SlaterJastrow(new_jastrow, self.dets, self.linear_coeffs)
    
    def update_coefficients(self, new_coefficients):
        """Update linear coefficients.
        
        Args:
            new_coefficients: New linear coefficients for determinants
            
        Returns:
            New Ansatz instance with updated coefficients
        """
        return SlaterJastrow(self.jastrow, self.dets, new_coefficients)

    def _compute_jastrow_terms(self, elec_coords):
        """Compute ∇J/J and ∇²J/J for all electrons using vmap."""
        n_electrons = elec_coords.shape[0]
        
        # Vectorize gradient and laplacian computation
        vmap_grads = jax.vmap(self.jastrow.get_log_grads, in_axes=(None, 0))
        vmap_all_grads = jax.vmap(vmap_grads, in_axes=(0, None))
        
        # Compute all pairs once
        # TODO: could be optimized to compute only upper triangle
        all_grads, all_laps = vmap_all_grads(elec_coords, elec_coords)  # (n,n,3), (n,n)
        
        # Use transpose for opposite ordering
        # For gradients: need to negate when swapping i,j due to chain rule
        all_grads_T = jnp.transpose(all_grads, (1, 0, 2))  
        
        # Create mask for upper and lower triangles
        upper_mask = jnp.triu(jnp.ones((n_electrons, n_electrons)), k=1)[..., None]  # for j>i
        lower_mask = jnp.tril(jnp.ones((n_electrons, n_electrons)), k=-1)[..., None]  # for j<i
        
        # Sum contributions properly respecting order
        grad_J_over_J = (jnp.sum(all_grads * upper_mask, axis=1) + 
                        jnp.sum(all_grads_T * lower_mask, axis=1))
        
        # Laplacian contributions (symmetric part)
        lap_J_over_J = (jnp.sum(all_laps * upper_mask[..., 0], axis=1) + 
                       jnp.sum(all_laps * lower_mask[..., 0], axis=1))
        
        # Add (∇u)² term
        lap_J_over_J = lap_J_over_J + jnp.sum(grad_J_over_J**2, axis=1)
        
        return grad_J_over_J, lap_J_over_J

    def _compute_kinetic_matrix(self, elec_coords, grad_J_over_J, lap_J_over_J):
        """Compute kinetic energy part of B matrix."""
        # Get Slater matrices and their gradients/laplacians
        slater_up, slater_down = self.dets[0].matrix(elec_coords)
        grad_up, grad_down = self.dets[0].grad(elec_coords)  # shape: (n_up/down, n_up/down, 3)
        lap_up, lap_down = self.dets[0].laplacian(elec_coords)
        
        n_up = self.dets[0].n_alpha
        
        # Slice gradients and laplacians for up/down electrons
        grad_J_up = grad_J_over_J[:n_up]      # shape: (n_up, 3)
        grad_J_down = grad_J_over_J[n_up:]    # shape: (n_down, 3)
        lap_J_up = lap_J_over_J[:n_up]        # shape: (n_up,)
        lap_J_down = lap_J_over_J[n_up:]      # shape: (n_down,)
        
        # Build inverses
        inv_up = jnp.linalg.inv(slater_up)
        inv_down = jnp.linalg.inv(slater_down)
        
        # Compute kinetic terms using vmap
        def compute_row(grad_J, lap_J, orb_vals, grad_orb, lap_orb):
            """Compute single row of kinetic matrix."""
            return -0.5 * (lap_orb + 
                          2 * jnp.sum(grad_J[:, None, :] * grad_orb, axis=-1) +
                          lap_J[:, None] * orb_vals)
        
        # Vectorize over electrons
        B_up = jax.vmap(compute_row)(grad_J_up, lap_J_up, slater_up, grad_up, lap_up)
        B_down = jax.vmap(compute_row)(grad_J_down, lap_J_down, slater_down, grad_down, lap_up)
        
        return inv_up, inv_down, B_up, B_down

    def _compute_potential_matrix(self, elec_coords, slater_up, slater_down):
        """Compute potential energy part of B matrix using vmap."""
        n_up = self.dets[0].n_alpha
        
        # Electron-nuclear potential
        atom_coords = self.mol.atom_coords()
        atom_charges = self.mol.atom_charges()
        
        # Vectorize e-n potential computation
        def e_n_potential(r):
            """Compute electron-nuclear potential for one electron."""
            dists = jnp.linalg.norm(r - atom_coords, axis=1)
            return jnp.sum(-atom_charges / dists)
            
        V_en_up = jax.vmap(e_n_potential)(elec_coords[:n_up])
        V_en_down = jax.vmap(e_n_potential)(elec_coords[n_up:])
        
        # Electron-electron potential
        def e_e_potential(r, other_coords):
            """Compute e-e potential for one electron with all others."""
            dists = jnp.linalg.norm(r - other_coords, axis=1)
            mask = jnp.ones_like(dists)  # Mask to exclude self-interaction
            return jnp.sum(mask / (dists + 1e-10))  # Add small constant for stability
            
        # Vectorize e-e potential computation
        V_ee_up = jax.vmap(e_e_potential, in_axes=(0, None))(
            elec_coords[:n_up], elec_coords[:n_up])
        V_ee_down = jax.vmap(e_e_potential, in_axes=(0, None))(
            elec_coords[n_up:], elec_coords[n_up:])
        
        # Combine potentials with orbital values
        B_up = (V_en_up[:, None] + V_ee_up[:, None]) * slater_up
        B_down = (V_en_down[:, None] + V_ee_down[:, None]) * slater_down
        
        return B_up, B_down

    def local_energy(self, elec_coords):
        """Compute local energy E_L = ℋΨ/Ψ. 
        See "Simple formalism for eﬃcient derivatives and multi-determinant expansions
            in quantum Monte Carlo" for details.
        """
        # Get Jastrow contributions
        grad_J_over_J, lap_J_over_J = self._compute_jastrow_terms(elec_coords)
        
        # Compute kinetic energy matrices
        inv_up, inv_down, B_kin_up, B_kin_down = self._compute_kinetic_matrix(
            elec_coords, grad_J_over_J, lap_J_over_J)
        
        # Get Slater matrices for potential energy
        slater_up, slater_down = self.dets[0].matrix(elec_coords)
        
        # Compute potential energy matrices
        B_pot_up, B_pot_down = self._compute_potential_matrix(
            elec_coords, slater_up, slater_down)
        
        # Combine kinetic and potential terms
        E_L = (jnp.trace(inv_up @ (B_kin_up + B_pot_up)) + 
               jnp.trace(inv_down @ (B_kin_down + B_pot_down)))
        
        return jnp.real(E_L)  # Ensure real value
