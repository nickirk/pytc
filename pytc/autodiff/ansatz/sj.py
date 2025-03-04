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
        
        # Create a mask to exclude diagonal elements (no self-interaction)
        n_electrons = elec_coords.shape[0]
        diag_mask = 1.0 - jnp.eye(n_electrons)
        
        # Use redundant summation form (multiply by 1/2)
        jastrow_val = jnp.exp(0.5 * jnp.sum(all_pairs * diag_mask))
        
        # Evaluate determinants (already vectorized internally)
        det_vals = jnp.array([det.value(elec_coords) for det in self.dets])
        linear_combo = jnp.sum(self.linear_coeffs * det_vals)
        
        return jastrow_val * linear_combo

    def update_jastrow(self, new_jastrow_params):
        """Update Jastrow parameters."""
        new_jastrow = self.jastrow.update(new_jastrow_params)
        return SlaterJastrow(self.mol, new_jastrow, self.dets, self.linear_coeffs)
    
    def update_coefficients(self, new_coefficients):
        """Update linear coefficients.
        
        Args:
            new_coefficients: New linear coefficients for determinants
            
        Returns:
            New Ansatz instance with updated coefficients
        """
        return SlaterJastrow(self.jastrow, self.dets, new_coefficients)

    def _compute_jastrow_terms(self, elec_coords):
        """Compute ∇J/J and ∇²J/J for all electrons using redundant summation form.
        
        In the redundant summation form J = exp(0.5*∑ᵢⱼ u(rᵢ,rⱼ)), we have:
        ∇ᵢJ/J = ∑ⱼ≠ᵢ ∇ᵢu(rᵢ,rⱼ)/2 + ∑ⱼ≠ᵢ ∇ᵢu(rⱼ,rᵢ)/2
        
        For a symmetric u function where u(rᵢ,rⱼ) = u(rⱼ,rᵢ), this simplifies to:
        ∇ᵢJ/J = ∑ⱼ≠ᵢ ∇ᵢu(rᵢ,rⱼ)
        """
        n_electrons = elec_coords.shape[0]
        
        # Vectorize gradient and laplacian computation over all pairs
        # First vmap over r2, keeping r1 fixed
        vmap_grads = jax.vmap(self.jastrow.get_log_grads, in_axes=(None, 0))
        # Then vmap over r1, broadcasting r2
        vmap_all_grads = jax.vmap(vmap_grads, in_axes=(0, None))
        
        # Compute all pairs at once
        all_grads, all_laps = vmap_all_grads(elec_coords, elec_coords)

        # The shape of all_grads is (n_electrons, n_electrons, 3)
        # To symmetrize, we need to swap the first two dimensions (electron indices)
        # not the last two dimensions (which would mix spatial coordinates with electron indices)
        #all_grads = 0.5 * (all_grads + jnp.swapaxes(all_grads, 0, 1))
        #all_laps = 0.5 * (all_laps + jnp.swapaxes(all_laps, 0, 1))

        
        # Create a mask to exclude diagonal elements (no self-interaction)
        # TODO: check if it is needed, since when ri=rj, the value is zero
        diag_mask = 1.0 - jnp.eye(n_electrons)
        diag_mask_3d = diag_mask[..., None]  # Add dimension for xyz coordinates
        
        # For gradients: since we use redundant summation, each gradient 
        # contribution is already counted correctly when we sum
        grad_J_over_J = jnp.sum(all_grads * diag_mask_3d, axis=1)
        
        # For laplacian: contributions are summed with the same mask
        lap_J_over_J = jnp.sum(all_laps * diag_mask, axis=1)
        
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
        
        # Compute kinetic terms directly without vmap
        B_up = -0.5 * (
            lap_up +  # (n_up, n_up)
            2 * jnp.einsum('ik,ijk->ij', grad_J_up, grad_up) +  # sum over spatial dimensions
            jnp.multiply(lap_J_up[:, None], slater_up)  # broadcast laplacian
        )
        
        B_down = -0.5 * (
            lap_down +  # (n_down, n_down)
            2 * jnp.einsum('ik,ijk->ij', grad_J_down, grad_down) +  # sum over spatial dimensions
            jnp.multiply(lap_J_down[:, None], slater_down)  # broadcast laplacian
        )
        
        return inv_up, inv_down, B_up, B_down

    def _compute_potential_matrix(self, elec_coords, slater_up, slater_down):
        """Compute potential energy part of B matrix using vmap.
        
        Efficiently computes both electron-nuclear and electron-electron
        potential energy interactions, with careful handling to avoid
        double-counting or self-interactions.
        """
        n_up = self.dets[0].n_alpha
        n_electrons = len(elec_coords)
        
        # Electron-nuclear potential with regularization
        atom_coords = self.mol.atom_coords()
        atom_charges = self.mol.atom_charges()
        
        def e_n_potential(r):
            """Compute electron-nuclear potential for one electron with regularization."""
            dists = jnp.linalg.norm(r - atom_coords, axis=1)
            # Add small regularization parameter to avoid numerical instability
            return -jnp.sum(atom_charges / (dists + 1e-10))
            
        # Use JAX-friendly slicing with jnp.take to avoid potential issues with direct slicing
        up_coords = jnp.take(elec_coords, jnp.arange(n_up), axis=0)
        down_coords = jnp.take(elec_coords, jnp.arange(n_up, n_electrons), axis=0)
        
        # Calculate electron-nuclear potentials
        V_en_up = jax.vmap(e_n_potential)(up_coords)[:, None]
        V_en_down = jax.vmap(e_n_potential)(down_coords)[:, None]
        
        # Electron-electron potential with efficient mask creation
        def pairwise_potential(coords):
            """Calculate electron-electron potential energy efficiently.
            
            Uses a redundant sum approach (similar to Jastrow) and avoids
            double-counting and self-interactions.
            """
            # Calculate all pairwise distances
            n = coords.shape[0]
            # Create expanded arrays: (n,1,3) and (1,n,3)
            ri = coords[:, None, :]
            rj = coords[None, :, :]
            
            # Calculate 1/r_ij for all pairs
            diff = ri - rj
            dist = jnp.sqrt(jnp.sum(diff**2, axis=2) + 1e-10)
            
            # Create mask to exclude self-interaction (diagonal elements)
            mask = 1.0 - jnp.eye(n)
            
            # Calculate potential for each electron with all others
            # Using redundant sum approach (will multiply by 0.5 later)
            pot = jnp.sum(mask / dist, axis=1)
            
            return pot
        
        # Calculate e-e potential for all electrons
        all_ee_pot = pairwise_potential(elec_coords)
        
        # Extract up and down electron potentials
        V_ee_up = all_ee_pot[:n_up, None]
        V_ee_down = all_ee_pot[n_up:, None]
        
        # Combine potentials with orbital values
        B_up = (V_en_up + V_ee_up) * slater_up
        B_down = (V_en_down + V_ee_down) * slater_down
        
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
