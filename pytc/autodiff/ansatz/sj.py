"""Implementation of the quantum many-body wavefunction ansatz."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Any
from functools import partial

from pytc.autodiff.ansatz.det import  value_and_grad, grad 


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
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, walker, params):
        """Evaluate wavefunction for a single walker with explicit parameters.
        
        Args:
            walker: Walker dataclass with single walker (positions shape: (n_electrons, 3))
                   For batches, use vmap externally.
            params: (jastrow_params, linear_coeffs)
            
        Returns:
            Tuple of (psi_values, updated_walker) where:
                psi_values: Tuple of (sign, log|psi|) for numerical stability (scalars)
                updated_walker: Walker with updated determinant matrices
        """
        jastrow_params, linear_coeffs = params
        
        # Compute Jastrow value for single walker (returns log|J|, Jastrow is always positive)
        log_jastrow_val = self._compute_jastrow_log_value(walker.positions, jastrow_params)
        
        # Compute determinant values with efficient updates (returns (sign, log|det|) tuples)
        # For single determinant case (most common), avoid list accumulation
        if len(self.dets) == 1:
            det_val, final_updated_walker = value_and_grad(self.dets[0], walker)
            # det_val is (sign, log|det|) tuple for single walker
            det_sign, det_logabs = det_val
            linear_combo_sign = jnp.sign(linear_coeffs[0]) * det_sign
            linear_combo_logabs = jnp.log(jnp.abs(linear_coeffs[0])) + det_logabs
        else:
            # Multiple determinants: accumulate in log space
            det_vals_list = []
            final_updated_walker = None
            
            for i, det in enumerate(self.dets):
                det_val, updated_walker = value_and_grad(det, walker)
                # Convert (sign, log|det|) back to regular values for linear combination
                # TODO: Implement proper log-space linear combination
                det_sign, det_logabs = det_val
                det_val_scalar = det_sign * jnp.exp(det_logabs)
                det_vals_list.append(det_val_scalar)
                # Only keep the first walker
                if i == 0:
                    final_updated_walker = updated_walker
            
            det_vals_array = jnp.array(det_vals_list)
            linear_combo = jnp.sum(linear_coeffs * det_vals_array)
            # Convert back to (sign, log) format
            linear_combo_sign = jnp.sign(linear_combo)
            linear_combo_logabs = jnp.log(jnp.abs(linear_combo) + 1e-100)
        
        # Combine Jastrow and determinant in log space
        # ψ = J × D  =>  log|ψ| = log|J| + log|D|, sign(ψ) = sign(D) (J always positive)
        psi_sign = linear_combo_sign
        psi_logabs = log_jastrow_val + linear_combo_logabs
        psi_values = (psi_sign, psi_logabs)
        
        # Return psi_values tuple and updated walker (no need to store psi_values in walker)
        return psi_values, final_updated_walker

    @partial(jax.jit, static_argnums=(0,))
    def _compute_jastrow_log_value(self, elec_coords, jastrow_params):
        """Compute Jastrow factor in log space for numerical stability.
        
        Returns:
            log|J|: Logarithm of Jastrow factor (Jastrow is always positive, so no sign needed)
        """
        vmap_single = jax.vmap(self.jastrow._compute, in_axes=(None, 0, None))
        vmap_all = jax.vmap(vmap_single, in_axes=(0, None, None))
        
        all_pairs = vmap_all(elec_coords, elec_coords, jastrow_params)
        
        n_electrons = elec_coords.shape[0]
        diag_mask = 1.0 - jnp.eye(n_electrons)
        
        # Return log(J) instead of J = exp(sum/2)
        # J = exp(sum/2) => log(J) = sum/2
        return 0.5 * jnp.sum(all_pairs * diag_mask)
    
    @property
    def n_electrons(self):
        """Return the number of electrons."""
        return self.mol.nelectron

    @partial(jax.jit, static_argnums=(0,))
    def _compute_jastrow_terms(self, elec_coords, jastrow_params):
        """Compute ∇J/J and ∇²J/J with explicit parameters."""
        n_electrons = elec_coords.shape[0]
        
        vmap_grads_r1 = jax.vmap(
            lambda r1, r2: self.jastrow.get_log_grads_r1(r1, r2, jastrow_params),
            in_axes=(None, 0)
        )
        vmap_all_grads_r1 = jax.vmap(vmap_grads_r1, in_axes=(0, None))
        
        all_grads_r1, all_laps_r1 = vmap_all_grads_r1(elec_coords, elec_coords)

        vmap_grads_r2 = jax.vmap(
            lambda r1, r2: self.jastrow.get_log_grads_r2(r1, r2, jastrow_params),
            in_axes=(None, 0)
        )
        vmap_all_grads_r2 = jax.vmap(vmap_grads_r2, in_axes=(0, None))
        
        all_grads_r2, all_laps_r2 = vmap_all_grads_r2(elec_coords, elec_coords)
        
        diag_mask = 1.0 - jnp.eye(n_electrons)
        diag_mask_3d = diag_mask[..., None]
        
        grad_J_over_J = jnp.sum(all_grads_r1 * diag_mask_3d, axis=1)
        grad_J_over_J += jnp.sum(all_grads_r2 * diag_mask_3d, axis=0)
        grad_J_over_J /= 2.0
        lap_sum = jnp.sum(all_laps_r1 * diag_mask, axis=1)
        lap_sum += jnp.sum(all_laps_r2 * diag_mask, axis=0)
        lap_sum /= 2.0
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
        return self.mol.nelec[0]

    @property
    def n_beta(self):
        """Number of beta-spin electrons."""
        return self.n_electrons - self.n_alpha
    
    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, walker, params):
        """Compute local energy for a single electron configuration using Walker.
        
        Uses the walker's cached Slater matrices, gradients, and laplacians which were
        updated by the most recent __call__() invocation.
        
        Args:
            walker: Walker dataclass with positions and cached matrices/gradients/laplacians
                   Should be a single walker (n_electrons, 3). For batches, use vmap externally.
            params: Tuple of parameters (jastrow_params, linear_coeffs)
                            
        Returns:
            Tuple of (energy, walker) where:
                energy: Scalar local energy value
                walker: Unchanged walker (no updates needed)
        """
        jastrow_params, linear_coeffs = params
        
        # Compute Jastrow terms for single walker
        grad_J_over_J, lap_J_over_J = self._compute_jastrow_terms(walker.positions, jastrow_params)
        
        energy = self._compute_single_walker_energy(
            walker.positions, grad_J_over_J, lap_J_over_J,
            walker.slater_up, walker.slater_down, walker.inv_up, walker.inv_down,
            walker.grad_up, walker.grad_down,
            walker.lap_up, walker.lap_down
        )
        return energy, walker
            
    @partial(jax.jit, static_argnums=(0,))
    def _compute_single_walker_energy(self, coords, grad_J_over_J, lap_J_over_J,
                                     slater_alpha, slater_beta, inv_alpha, inv_beta, 
                                     grad_alpha, grad_beta,
                                     lap_alpha, lap_beta):
        """Compute energy for a single walker with pre-computed quantities."""
        n_alpha = self.dets[0].n_alpha
        
        # Slice gradients and laplacians for alpha/beta electrons
        grad_J_alpha = grad_J_over_J[:n_alpha]      # shape: (n_alpha, 3)
        grad_J_beta = grad_J_over_J[n_alpha:]    # shape: (n_beta, 3)
        lap_J_alpha = lap_J_over_J[:n_alpha]        # shape: (n_alpha,)
        lap_J_beta = lap_J_over_J[n_alpha:]      # shape: (n_beta,)
        
        # Build inverses
        #inv_alpha = jnp.linalg.inv(slater_alpha)
        #inv_beta = jnp.linalg.inv(slater_beta)
        
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

    @partial(jax.jit, static_argnums=(0,))
    def quantum_force(self, walker, params, cutoff=5.0):
        """Compute quantum force (2∇ψ/ψ) for importance sampling with magnitude clipping.
        
        Args:
            walker: Walker object with single walker (positions shape: (n_electrons, 3))
                   For batches, use vmap externally.
            params: Tuple of parameters (jastrow_params, linear_coeffs)
            cutoff: Maximum allowed magnitude for quantum forces
                           
        Returns:
            Array of quantum forces with shape (n_electrons, 3)
        """
        jastrow_params, linear_coeffs = params 
        elec_coords = walker.positions
        
        # Compute Jastrow gradient contributions for single walker
        grad_J_over_J = self._compute_jastrow_terms(elec_coords, jastrow_params)[0]
        
        # Get determinant gradient contributions using JAX wrappers
        det = self.dets[0]  # Using the first determinant
        slater_alpha, slater_beta, grad_alpha, grad_beta = grad(det, walker)
        
        # Build quantum force for single walker
        forces = self._compute_quantum_force(
            grad_J_over_J,
            slater_alpha,
            slater_beta,
            grad_alpha,
            grad_beta
        )

        # Apply cutoff to quantum forces while maintaining direction
        # Calculate force magnitudes (shape: n_electrons)
        force_magnitudes = jnp.linalg.norm(forces, axis=-1)
        
        # Create scaling factors: min(1.0, cutoff/magnitude)
        # This preserves direction while limiting magnitude
        scaling_factors = jnp.minimum(1.0, cutoff / (force_magnitudes + 1e-10))
        
        # Reshape for broadcasting (add dimension for x,y,z components)
        scaling_factors = scaling_factors[..., jnp.newaxis]
        
        # Apply scaling to forces
        clipped_forces = forces * scaling_factors
        
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
