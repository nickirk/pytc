"""Implementation of the quantum many-body wavefunction ansatz."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Any
from functools import partial

from pytc.autodiff.ansatz.det import value, grad, laplacian, matrix


class SlaterJastrow:
    """Quantum many-body wavefunction ansatz combining Jastrow factor with Slater determinants."""
    
    def __init__(self, mol, jastrow, dets: List[Any]):
        """Initialize the ansatz without storing optimizable parameters."""
        self.mol = mol
        self.jastrow = jastrow
        self.dets = dets
        self._ion_ion_potential = self._compute_ion_ion_potential()

    def _compute_ion_ion_potential(self):
        """Calculate the ion-ion repulsion energy (nuclear-nuclear Coulomb interaction).
        
        This is a constant term that depends only on the molecular geometry.
        
        Returns:
            float: The ion-ion potential energy
        """
        atom_coords = self.mol.atom_coords()
        atom_charges = self.mol.atom_charges()
        n_atoms = len(atom_charges)
        
        # Calculate ion-ion potential energy
        v_ion_ion = 0.0
        
        R_diff = atom_coords[:, None, :] - atom_coords[None, :, :]
        R_dist = jnp.linalg.norm(R_diff, axis=-1)
        charge_products = jnp.outer(atom_charges, atom_charges)

        mask = 1-jnp.eye(n_atoms)
        v_ion_ion = jnp.sum(charge_products * mask / (R_dist+1e-10)) / 2.0
        
        return v_ion_ion
    
    @property
    def ion_ion_potential(self):
        """Return the ion-ion potential energy (nuclear-nuclear repulsion)."""
        return self._ion_ion_potential
   
    def __call__(self, elec_coords_batch, jastrow_params, linear_coeffs):
        """Evaluate wavefunction with explicit parameters."""
        jastrow_vals = jax.vmap(lambda x: self._compute_jastrow_value(x, jastrow_params))(elec_coords_batch)
        
        det_vals = []
        for det in self.dets:
            det_batch_vals = value(det, elec_coords_batch)
            det_vals.append(det_batch_vals)
        
        det_vals_array = jnp.array(det_vals).transpose()
        linear_combo = jnp.sum(linear_coeffs * det_vals_array, axis=1)
        
        return jastrow_vals * linear_combo

    @partial(jax.jit, static_argnums=(0,))
    def _compute_jastrow_value(self, elec_coords, jastrow_params):
        """Compute Jastrow factor value with explicit parameters."""
        vmap_single = jax.vmap(self.jastrow._compute, in_axes=(None, 0, None))
        vmap_all = jax.vmap(vmap_single, in_axes=(0, None, None))
        
        all_pairs = vmap_all(elec_coords, elec_coords, jastrow_params)
        
        n_electrons = elec_coords.shape[0]
        diag_mask = 1.0 - jnp.eye(n_electrons)
        
        return jnp.exp(1./2. * jnp.sum(all_pairs * diag_mask))
    
    @property
    def n_electrons(self):
        """Return the number of electrons."""
        return self.dets[0].n_electrons


    @partial(jax.jit, static_argnums=(0,))
    def _compute_jastrow_terms(self, elec_coords, jastrow_params):
        """Compute ∇J/J and ∇²J/J with explicit parameters."""
        n_electrons = elec_coords.shape[0]
        
        vmap_grads = jax.vmap(
            lambda r1, r2: self.jastrow.get_log_grads(r1, r2, jastrow_params),
            in_axes=(None, 0)
        )
        vmap_all_grads = jax.vmap(vmap_grads, in_axes=(0, None))
        
        all_grads, all_laps = vmap_all_grads(elec_coords, elec_coords)
        
        diag_mask = 1.0 - jnp.eye(n_electrons)
        diag_mask_3d = diag_mask[..., None]
        
        grad_J_over_J = jnp.sum(all_grads * diag_mask_3d, axis=1)
        lap_sum = jnp.sum(all_laps * diag_mask, axis=1)
        grad_squared = jnp.sum(grad_J_over_J**2, axis=1)
        lap_J_over_J = lap_sum + grad_squared
        
        return grad_J_over_J, lap_J_over_J
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_potential_matrix(self, elec_coords, slater_alpha, slater_beta):
        """Compute potential energy part of B matrix using vmap.
        
        Efficiently computes both electron-nuclear and electron-electron
        potential energy interactions, with careful handling to avoid
        double-counting or self-interactions.
        """
        n_alpha = self.dets[0].n_alpha
        n_electrons = len(elec_coords)
        
        # Electron-nuclear potential with regularization
        atom_coords = self.mol.atom_coords()
        atom_charges = self.mol.atom_charges()
        
        def e_n_potential(r):
            """Compute electron-nuclear potential for one electron with regularization."""
            # Fix broadcasting issue by using proper reshaping
            # Reshape r to (n_electrons, 1, 3) and atom_coords to (1, n_atoms, 3)
            # for proper broadcasting
            r_reshaped = r[:, jnp.newaxis, :]  # Shape: (n_elec, 1, 3)
            diff = r_reshaped - atom_coords[jnp.newaxis, :, :]  # Shape: (n_elec, n_atoms, 3)
            dists = jnp.linalg.norm(diff, axis=2)  # Shape: (n_elec, n_atoms)
            # Compute Coulomb potentials with small regularization
            potentials = -atom_charges[jnp.newaxis, :] / (dists + 1e-10)  # Shape: (n_elec, n_atoms)
            # Sum over all atoms for each electron
            return jnp.sum(potentials, axis=1)  # Shape: (n_elec,)
        
        # Use JAX-friendly slicing with jnp.take to avoid potential issues with direct slicing
        alpha_coords = jnp.take(elec_coords, jnp.arange(n_alpha), axis=0)
        beta_coords = jnp.take(elec_coords, jnp.arange(n_alpha, n_electrons), axis=0)
        
        # Calculate electron-nuclear potentials directly without using vmap
        V_en_alpha = e_n_potential(alpha_coords)
        V_en_beta = e_n_potential(beta_coords)
        
        # More efficient electron-electron potential using vmap
        def pairwise_distance(r_i, r_j):
            """Compute 1/|r_i - r_j| with regularization."""
            diff = r_i - r_j
            dist = jnp.sqrt(jnp.sum(diff**2) + 1e-10)
            return 1.0 / dist
        
        # Map over all electron pairs
        ee_vmap_inner = jax.vmap(pairwise_distance, in_axes=(None, 0))
        ee_vmap_outer = jax.vmap(ee_vmap_inner, in_axes=(0, None))
        
        # Compute all pairwise interactions at once
        all_e_e_pot = jax.jit(ee_vmap_outer)(elec_coords, elec_coords)
        
        # Remove self-interactions
        mask = 1.0 - jnp.eye(n_electrons)
        all_e_e_pot = all_e_e_pot * mask
        
        # Sum interactions for each electron
        e_e_pot = 0.5 * jnp.sum(all_e_e_pot, axis=1)
        
        # Extract alpha and beta electron potentials
        V_ee_alpha = e_e_pot[:n_alpha, None]
        V_ee_beta = e_e_pot[n_alpha:, None]
        
        # Combine potentials with orbital values
        B_alpha = (V_en_alpha + V_ee_alpha) * slater_alpha
        B_beta = (V_en_beta + V_ee_beta) * slater_beta
        
        return B_alpha, B_beta


    @property
    def n_alpha(self):
        """Number of alpha-spin electrons."""
        return self.dets[0].n_alpha if self.dets else 0

    @property
    def n_beta(self):
        """Number of beta-spin electrons."""
        return self.n_electrons - self.n_alpha
    
    def local_energy(self, elec_coords_batch, jastrow_params, linear_coeffs):
        """Compute local energy for a batch of electron configurations.
        
        Args:
            elec_coords_batch: Array with shape (n_walkers, n_electrons, 3)
                            or (n_electrons, 3) for a single walker
            jastrow_params: Jastrow parameters
            linear_coeffs: Linear coefficients for determinants
                            
        Returns:
            Array of local energy values with shape (n_walkers,)
            or a single value for a single walker
        """
        import psutil
        
        
        # Use vmap to compute Jastrow terms for all walkers
        grad_J_over_J_batch, lap_J_over_J_batch = jax.vmap(lambda x: self._compute_jastrow_terms(x, jastrow_params))(elec_coords_batch)
        
        # Get batched matrices, gradients, and laplacians using JAX wrappers
        det = self.dets[0]  # Using the first determinant (assuming single-determinant for now)
        slater_alpha_batch, slater_beta_batch = matrix(det, elec_coords_batch)
        grad_alpha_batch, grad_beta_batch = grad(det, elec_coords_batch)
        lap_alpha_batch, lap_beta_batch = laplacian(det, elec_coords_batch)

        energies = jax.vmap(self._compute_single_walker_energy)(
            elec_coords_batch, grad_J_over_J_batch, lap_J_over_J_batch,
            slater_alpha_batch, slater_beta_batch, grad_alpha_batch, grad_beta_batch,
            lap_alpha_batch, lap_beta_batch
        )

        return energies
            
    @partial(jax.jit, static_argnums=(0,))
    def _compute_single_walker_energy(self, coords, grad_J_over_J, lap_J_over_J,
                                     slater_alpha, slater_beta, grad_alpha, grad_beta,
                                     lap_alpha, lap_beta):
        """Compute energy for a single walker with pre-computed quantities."""
        n_alpha = self.dets[0].n_alpha
        
        # Slice gradients and laplacians for alpha/beta electrons
        grad_J_alpha = grad_J_over_J[:n_alpha]      # shape: (n_alpha, 3)
        grad_J_beta = grad_J_over_J[n_alpha:]    # shape: (n_beta, 3)
        lap_J_alpha = lap_J_over_J[:n_alpha]        # shape: (n_alpha,)
        lap_J_beta = lap_J_over_J[n_alpha:]      # shape: (n_beta,)
        
        # Build inverses
        inv_alpha = jnp.linalg.inv(slater_alpha)
        inv_beta = jnp.linalg.inv(slater_beta)
        
        # Compute kinetic terms
        B_kin_alpha = -0.5 * (
            lap_alpha +
            2 * jnp.einsum('ik,ijk->ij', grad_J_alpha, grad_alpha) +
            jnp.multiply(lap_J_alpha[:, None], slater_alpha)
        )
        
        B_kin_beta = -0.5 * (
            lap_beta + 
            2 * jnp.einsum('ik,ijk->ij', grad_J_beta, grad_beta) +
            jnp.multiply(lap_J_beta[:, None], slater_beta)
        )
        
        # Compute potential energy matrices
        B_pot_alpha, B_pot_beta = self._compute_potential_matrix(
            coords, slater_alpha, slater_beta)
        
        # Combine kinetic and potential terms
        E_L = (jnp.trace(inv_alpha @ (B_kin_alpha + B_pot_alpha)) + 
              jnp.trace(inv_beta @ (B_kin_beta + B_pot_beta)))
        
        # Add ion-ion potential energy (constant term)
        E_L = E_L + self._ion_ion_potential
        
        return jnp.real(E_L)  # Ensure real value

    def quantum_force(self, elec_coords, jastrow_params, linear_coeffs, cutoff=1.0):
        """Compute quantum force (2∇ψ/ψ) for importance sampling with magnitude clipping.
        
        Args:
            elec_coords_batch: Array with shape (n_walkers, n_electrons, 3)
                           or (n_electrons, 3) for a single walker
            jastrow_params: Jastrow parameters
            linear_coeffs: Linear coefficients for determinants
            cutoff: Maximum allowed magnitude for quantum forces
                           
        Returns:
            Array of quantum forces with shape (n_walkers, n_electrons, 3)
            or shape (n_electrons, 3) for a single walker
        """
        # Handle single walker case by adding a batch dimension
        
        
        # Compute Jastrow gradient contributions
        grad_J_over_J_batch = jax.vmap(lambda coords: self._compute_jastrow_terms(coords, jastrow_params)[0])(elec_coords)
        
        # Get determinant gradient contributions using JAX wrappers
        det = self.dets[0]  # Using the first determinant
        slater_alpha_batch, slater_beta_batch = matrix(det, elec_coords)
        grad_alpha_batch, grad_beta_batch = grad(det, elec_coords)
        
        # Build quantum forces with vmap
        forces = jax.vmap(self._compute_quantum_force)(
            grad_J_over_J_batch,
            slater_alpha_batch,
            slater_beta_batch,
            grad_alpha_batch,
            grad_beta_batch
        )

        # Apply cutoff to quantum forces while maintaining direction
        # Calculate force magnitudes (shape: batch, n_electrons)
        force_magnitudes = jnp.linalg.norm(forces, axis=-1)
        
        # Create scaling factors: min(1.0, cutoff/magnitude)
        # This preserves direction while limiting magnitude
        scaling_factors = jnp.minimum(1.0, cutoff / (force_magnitudes + 1e-10))
        
        # Reshape for broadcasting (add dimension for x,y,z components)
        scaling_factors = scaling_factors[..., jnp.newaxis]
        
        # Apply scaling to forces
        clipped_forces = forces * scaling_factors
        
        # Return single value if input was a single walker
        return clipped_forces
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_quantum_force(self, grad_J_over_J, slater_alpha, slater_beta, grad_alpha, grad_beta):
        """Compute quantum force for a single configuration."""
        n_alpha = self.n_alpha
        
        # Slice gradient contributions for alpha/beta electrons
        grad_J_alpha = grad_J_over_J[:n_alpha]      # shape: (n_alpha, 3)
        grad_J_beta = grad_J_over_J[n_alpha:]    # shape: (n_beta, 3)
        
        # Build inverses
        inv_alpha = jnp.linalg.inv(slater_alpha)
        inv_beta = jnp.linalg.inv(slater_beta)
        
        # Compute gradient of log determinant part: ∇ln|D|/D
        grad_logD_alpha = jnp.einsum('ij,ijk->ik', inv_alpha, grad_alpha)
        grad_logD_beta = jnp.einsum('ij,ijk->ik', inv_beta, grad_beta)
        
        # Combine gradient contributions: 2∇ψ/ψ = 2(∇J/J + ∇D/D)
        quantum_force_alpha = 2.0 * (grad_J_alpha + grad_logD_alpha)
        quantum_force_beta = 2.0 * (grad_J_beta + grad_logD_beta)
        
        # Combine and return
        return jnp.concatenate([quantum_force_alpha, quantum_force_beta], axis=0)
