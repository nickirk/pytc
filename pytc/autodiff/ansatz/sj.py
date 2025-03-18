"""Implementation of the quantum many-body wavefunction ansatz."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Any
from functools import partial


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
        
        # Calculate and store ion-ion repulsion energy (constant for fixed geometry)
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
   
    def __call__(self, elec_coords_batch):
        """Evaluate wavefunction for a batch of electron configurations.
        
        Args:
            elec_coords_batch: Array with shape (n_walkers, n_electrons, 3)
                           or (n_electrons, 3) for a single walker
                           
        Returns:
            Array of wavefunction values with shape (n_walkers,)
            or a single value for a single walker
        """
        # Handle single walker case by adding a batch dimension
        single_walker = False
        if len(elec_coords_batch.shape) == 2:  # (n_electrons, 3)
            elec_coords_batch = elec_coords_batch[None, ...]  # Add batch dimension
            single_walker = True
        
        # Vectorize Jastrow calculation over batch dimension
        jastrow_vals = jax.jit(jax.vmap(self._compute_jastrow_value))(elec_coords_batch)
        
        # Convert to NumPy for determinant calculations
        elec_coords_batch_np = np.array(elec_coords_batch)
        
        # Use new batched determinant evaluation
        det_vals = []
        for det in self.dets:
            det_batch_vals = det.value(elec_coords_batch_np)  # Now returns values for all walkers at once
            det_vals.append(det_batch_vals)
            
        # Combine determinant values using linear coefficients
        det_vals_array = np.array(det_vals).transpose()  # Shape (n_walkers, n_dets)
        linear_combo = np.sum(np.array(self.linear_coeffs) * det_vals_array, axis=1)
        
        # Convert to JAX array
        det_linear_combos = jnp.array(linear_combo)
        
        # Multiply Jastrow and determinant parts
        psi_vals = jastrow_vals * det_linear_combos
        
        # Return single value if input was a single walker
        if single_walker:
            return psi_vals[0]
        else:
            return psi_vals

    # Use partial with static_argnums to specify that 'self' is static
    @partial(jax.jit, static_argnums=(0,))
    def _compute_jastrow_value(self, elec_coords):
        """Compute Jastrow factor value."""
        # Vectorize Jastrow computation over all pairs
        vmap_single = jax.vmap(self.jastrow._compute, in_axes=(None, 0, None))
        vmap_all = jax.vmap(vmap_single, in_axes=(0, None, None))
        
        # Compute all pairwise values at once
        all_pairs = vmap_all(elec_coords, elec_coords, self.jastrow.params)
        
        # Create a mask to exclude diagonal elements (no self-interaction)
        n_electrons = elec_coords.shape[0]
        diag_mask = 1.0 - jnp.eye(n_electrons)
        
        # Use redundant summation form (multiply by 1/2)
        return jnp.exp(1./2. * jnp.sum(all_pairs * diag_mask))
    
    @property
    def n_electrons(self):
        """Return the number of electrons."""
        return self.dets[0].n_electrons
        
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

    @partial(jax.jit, static_argnums=(0,))
    def _compute_jastrow_terms(self, elec_coords):
        """Compute ∇J/J and ∇²J/J for all electrons using redundant summation form.
        
        In the redundant summation form J = exp(0.5*∑ᵢⱼ u(rᵢ,rⱼ)), we have:
        ∇ᵢJ/J = ∑ⱼ≠ᵢ ∇ᵢu(rᵢ,rⱼ)/2 + ∑ⱼ≠ᵢ ∇ᵢu(rⱼ,rᵢ)/2
        
        For a symmetric u function where u(rᵢ,rⱼ) = -u(rⱼ,rⱼ), this simplifies to:
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
        #all_grads =  0.5 * (all_grads - jnp.swapaxes(all_grads, 0, 1))

        
        # Create a mask to exclude diagonal elements (no self-interaction)
        # TODO: check if it is needed, since when ri=rj, the value is zero
        diag_mask = 1.0 - jnp.eye(n_electrons)
        diag_mask_3d = diag_mask[..., None]  # Add dimension for xyz coordinates
        
        # For gradients: since we use redundant summation, each gradient 
        # contribution is already counted correctly when we sum
        grad_J_over_J = jnp.sum(all_grads * diag_mask_3d, axis=1)
        
        # For laplacian: first sum the laplacian terms
        lap_sum = jnp.sum(all_laps * diag_mask, axis=1) 
        
        # Then add the squared gradient term (∇u)²
        # Square the gradients and sum over spatial dimensions for each electron
        grad_squared = jnp.sum(grad_J_over_J**2, axis=1)
        
        # Complete Laplacian expression: ∇²J/J = ∇²u + (∇u)²
        lap_J_over_J = lap_sum + grad_squared
        
        return grad_J_over_J, lap_J_over_J
    
    @partial(jax.jit, static_argnums=(0,))
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
        up_coords = jnp.take(elec_coords, jnp.arange(n_up), axis=0)
        down_coords = jnp.take(elec_coords, jnp.arange(n_up, n_electrons), axis=0)
        
        # Calculate electron-nuclear potentials directly without using vmap
        V_en_up = e_n_potential(up_coords)
        V_en_down = e_n_potential(down_coords)
        
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
        
        # Extract up and down electron potentials
        V_ee_up = e_e_pot[:n_up, None]
        V_ee_down = e_e_pot[n_up:, None]
        
        # Combine potentials with orbital values
        B_up = (V_en_up + V_ee_up) * slater_up
        B_down = (V_en_down + V_ee_down) * slater_down
        
        return B_up, B_down

    # Create wrappers for matrix, grad, laplacian that use NumPy conversion
    def _get_matrices(self, elec_coords_batch):
        """Get Slater matrices with NumPy conversion.
        
        Args:
            elec_coords_batch: Array of shape (n_walkers, n_electrons, 3)
                              or (n_electrons, 3) for a single walker
        
        Returns:
            tuple: (slater_up, slater_down) with appropriate batch dimensions
        """
        # Ensure batch dimension
        single_walker = False
        if len(elec_coords_batch.shape) == 2:
            elec_coords_batch = elec_coords_batch[None, ...]
            single_walker = True
            
        # Convert to NumPy
        elec_coords_batch_np = np.array(elec_coords_batch)
        
        # Use batched matrix calculation
        slater_up, slater_down = self.dets[0].matrix(elec_coords_batch_np)
        
        # Convert to JAX arrays
        slater_up_jax = jnp.array(slater_up)
        slater_down_jax = jnp.array(slater_down)
        
        # Return appropriately based on input shape
        if single_walker:
            return slater_up_jax[0], slater_down_jax[0]  # Extract single walker data
        else:
            return slater_up_jax, slater_down_jax
    
    def _get_gradients(self, elec_coords_batch):
        """Get gradients of Slater matrices with NumPy conversion.
        
        Args:
            elec_coords_batch: Array of shape (n_walkers, n_electrons, 3)
                              or (n_electrons, 3) for a single walker
        """
        # Ensure batch dimension
        single_walker = False
        if len(elec_coords_batch.shape) == 2:
            elec_coords_batch = elec_coords_batch[None, ...]
            single_walker = True
            
        # Convert to NumPy
        elec_coords_batch_np = np.array(elec_coords_batch)
        
        # Use batched gradient calculation
        grad_up, grad_down = self.dets[0].grad(elec_coords_batch_np)
        
        # Convert to JAX arrays
        grad_up_jax = jnp.array(grad_up)
        grad_down_jax = jnp.array(grad_down)
        
        # Return appropriately based on input shape
        if single_walker:
            return grad_up_jax[0], grad_down_jax[0]  # Extract single walker data
        else:
            return grad_up_jax, grad_down_jax
    
    def _get_laplacians(self, elec_coords_batch):
        """Get laplacians of Slater matrices with NumPy conversion.
        
        Args:
            elec_coords_batch: Array of shape (n_walkers, n_electrons, 3)
                              or (n_electrons, 3) for a single walker
        """
        # Ensure batch dimension
        single_walker = False
        if len(elec_coords_batch.shape) == 2:
            elec_coords_batch = elec_coords_batch[None, ...]
            single_walker = True
            
        # Convert to NumPy
        elec_coords_batch_np = np.array(elec_coords_batch)
        
        # Use batched laplacian calculation
        lap_up, lap_down = self.dets[0].laplacian(elec_coords_batch_np)
        
        # Convert to JAX arrays
        lap_up_jax = jnp.array(lap_up)
        lap_down_jax = jnp.array(lap_down)
        
        # Return appropriately based on input shape
        if single_walker:
            return lap_up_jax[0], lap_down_jax[0]  # Extract single walker data
        else:
            return lap_up_jax, lap_down_jax

    def _get_matrices_pure(self, elec_coords_batch):
        """Wrapper for _get_matrices using pure_callback for JIT compatibility."""
        # Determine output shapes based on input
        batch_size = elec_coords_batch.shape[0] if len(elec_coords_batch.shape) > 2 else 1
        n_up = self.dets[0].n_alpha
        n_down = self.n_electrons - n_up
        
        # Define shapes and dtypes for the expected outputs
        up_shape = (batch_size, n_up, n_up) if batch_size > 1 else (n_up, n_up)
        down_shape = (batch_size, n_down, n_down) if batch_size > 1 else (n_down, n_down)
        
        # Define the function to pass to pure_callback
        def get_matrices_callback(coords):
            return self._get_matrices(coords)
        
        # Use pure_callback to isolate the non-traceable computation
        slater_up, slater_down = jax.pure_callback(
            get_matrices_callback,
            (jax.ShapeDtypeStruct(up_shape, jnp.float64), 
             jax.ShapeDtypeStruct(down_shape, jnp.float64)),
            elec_coords_batch
        )
        
        return slater_up, slater_down
    
    def _get_gradients_pure(self, elec_coords_batch):
        """Wrapper for _get_gradients using pure_callback for JIT compatibility."""
        # Determine output shapes based on input
        batch_size = elec_coords_batch.shape[0] if len(elec_coords_batch.shape) > 2 else 1
        n_up = self.dets[0].n_alpha
        n_down = self.n_electrons - n_up
        
        # Define shapes and dtypes for the expected outputs
        up_shape = (batch_size, n_up, n_up, 3) if batch_size > 1 else (n_up, n_up, 3)
        down_shape = (batch_size, n_down, n_down, 3) if batch_size > 1 else (n_down, n_down, 3)
        
        # Define the function to pass to pure_callback
        def get_gradients_callback(coords):
            return self._get_gradients(coords)
        
        # Use pure_callback to isolate the non-traceable computation
        grad_up, grad_down = jax.pure_callback(
            get_gradients_callback,
            (jax.ShapeDtypeStruct(up_shape, jnp.float64), 
             jax.ShapeDtypeStruct(down_shape, jnp.float64)),
            elec_coords_batch
        )
        
        return grad_up, grad_down
    
    def _get_laplacians_pure(self, elec_coords_batch):
        """Wrapper for _get_laplacians using pure_callback for JIT compatibility."""
        # Determine output shapes based on input
        batch_size = elec_coords_batch.shape[0] if len(elec_coords_batch.shape) > 2 else 1
        n_up = self.dets[0].n_alpha
        n_down = self.n_electrons - n_up
        
        # Define shapes and dtypes for the expected outputs
        up_shape = (batch_size, n_up, n_up) if batch_size > 1 else (n_up, n_up)
        down_shape = (batch_size, n_down, n_down) if batch_size > 1 else (n_down, n_down)
        
        # Define the function to pass to pure_callback
        def get_laplacians_callback(coords):
            return self._get_laplacians(coords)
        
        # Use pure_callback to isolate the non-traceable computation
        lap_up, lap_down = jax.pure_callback(
            get_laplacians_callback,
            (jax.ShapeDtypeStruct(up_shape, jnp.float64), 
             jax.ShapeDtypeStruct(down_shape, jnp.float64)),
            elec_coords_batch
        )
        
        return lap_up, lap_down

    @property
    def n_up(self):
        """Number of up-spin electrons."""
        return self.dets[0].n_alpha if self.dets else 0

    @property
    def n_down(self):
        """Number of down-spin electrons."""
        return self.n_electrons - self.n_up
    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, elec_coords_batch):
        """Compute local energy for a batch of electron configurations.
        
        Args:
            elec_coords_batch: Array with shape (n_walkers, n_electrons, 3)
                            or (n_electrons, 3) for a single walker
                            
        Returns:
            Array of local energy values with shape (n_walkers,)
            or a single value for a single walker
        """
        # Handle single walker case by adding a batch dimension
        single_walker = False
        if len(elec_coords_batch.shape) == 2:  # (n_electrons, 3)
            elec_coords_batch = elec_coords_batch[None, ...]  # Add batch dimension
            single_walker = True
        
        # Use vmap to compute Jastrow terms for all walkers
        grad_J_over_J_batch, lap_J_over_J_batch = jax.vmap(self._compute_jastrow_terms)(elec_coords_batch)
        
        # Get batched matrices, gradients, and laplacians using pure_callback wrappers
        slater_up_batch, slater_down_batch = self._get_matrices_pure(elec_coords_batch)
        grad_up_batch, grad_down_batch = self._get_gradients_pure(elec_coords_batch)
        lap_up_batch, lap_down_batch = self._get_laplacians_pure(elec_coords_batch)
        
        # Now we'll use vmap to process all walkers at once
        energies = jax.vmap(self._compute_single_walker_energy)(
            elec_coords_batch,
            grad_J_over_J_batch,
            lap_J_over_J_batch,
            slater_up_batch, 
            slater_down_batch,
            grad_up_batch,
            grad_down_batch,
            lap_up_batch,
            lap_down_batch
        )
        
        # Return single value if input was a single walker
        if single_walker:
            return energies[0]
        else:
            return energies
            
    @partial(jax.jit, static_argnums=(0,))
    def _compute_single_walker_energy(self, coords, grad_J_over_J, lap_J_over_J,
                                     slater_up, slater_down, grad_up, grad_down,
                                     lap_up, lap_down):
        """Compute energy for a single walker with pre-computed quantities."""
        n_up = self.dets[0].n_alpha
        
        # Slice gradients and laplacians for up/down electrons
        grad_J_up = grad_J_over_J[:n_up]      # shape: (n_up, 3)
        grad_J_down = grad_J_over_J[n_up:]    # shape: (n_down, 3)
        lap_J_up = lap_J_over_J[:n_up]        # shape: (n_up,)
        lap_J_down = lap_J_over_J[n_up:]      # shape: (n_down,)
        
        # Build inverses
        inv_up = jnp.linalg.inv(slater_up)
        inv_down = jnp.linalg.inv(slater_down)
        
        # Compute kinetic terms
        B_kin_up = -0.5 * (
            lap_up +
            2 * jnp.einsum('ik,ijk->ij', grad_J_up, grad_up) +
            jnp.multiply(lap_J_up[:, None], slater_up)
        )
        
        B_kin_down = -0.5 * (
            lap_down + 
            2 * jnp.einsum('ik,ijk->ij', grad_J_down, grad_down) +
            jnp.multiply(lap_J_down[:, None], slater_down)
        )
        
        # Compute potential energy matrices
        B_pot_up, B_pot_down = self._compute_potential_matrix(
            coords, slater_up, slater_down)
        
        # Combine kinetic and potential terms
        E_L = (jnp.trace(inv_up @ (B_kin_up + B_pot_up)) + 
              jnp.trace(inv_down @ (B_kin_down + B_pot_down)))
        
        # Add ion-ion potential energy (constant term)
        E_L = E_L + self._ion_ion_potential
        
        return jnp.real(E_L)  # Ensure real value

    def quantum_force(self, elec_coords_batch, cutoff=1.0):
        """Compute quantum force (2∇ψ/ψ) for importance sampling with magnitude clipping.
        
        Args:
            elec_coords_batch: Array with shape (n_walkers, n_electrons, 3)
                           or (n_electrons, 3) for a single walker
            cutoff: Maximum allowed magnitude for quantum forces
                           
        Returns:
            Array of quantum forces with shape (n_walkers, n_electrons, 3)
            or shape (n_electrons, 3) for a single walker
        """
        # Handle single walker case by adding a batch dimension
        single_walker = False
        if len(elec_coords_batch.shape) == 2:  # (n_electrons, 3)
            elec_coords_batch = elec_coords_batch[None, ...]  # Add batch dimension
            single_walker = True
        
        # Compute Jastrow gradient contributions
        grad_J_over_J_batch = jax.jit(jax.vmap(lambda coords: self._compute_jastrow_terms(coords)[0]))(elec_coords_batch)
        
        # Get determinant gradient contributions
        slater_up_batch, slater_down_batch = self._get_matrices(elec_coords_batch)
        grad_up_batch, grad_down_batch = self._get_gradients(elec_coords_batch)
        
        # Build quantum forces with vmap
        forces = jax.jit(jax.vmap(self._compute_quantum_force))(
            elec_coords_batch, 
            grad_J_over_J_batch,
            slater_up_batch,
            slater_down_batch,
            grad_up_batch,
            grad_down_batch
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
        if single_walker:
            return clipped_forces[0]
        else:
            return clipped_forces
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_quantum_force(self, coords, grad_J_over_J, slater_up, slater_down, grad_up, grad_down):
        """Compute quantum force for a single configuration."""
        n_up = self.n_up
        
        # Slice gradient contributions for up/down electrons
        grad_J_up = grad_J_over_J[:n_up]      # shape: (n_up, 3)
        grad_J_down = grad_J_over_J[n_up:]    # shape: (n_down, 3)
        
        # Build inverses
        inv_up = jnp.linalg.inv(slater_up)
        inv_down = jnp.linalg.inv(slater_down)
        
        # Compute gradient of log determinant part: ∇ln|D|/D
        grad_logD_up = jnp.einsum('ij,ijk->ik', inv_up, grad_up)
        grad_logD_down = jnp.einsum('ij,ijk->ik', inv_down, grad_down)
        
        # Combine gradient contributions: 2∇ψ/ψ = 2(∇J/J + ∇D/D)
        quantum_force_up = 2.0 * (grad_J_up + grad_logD_up)
        quantum_force_down = 2.0 * (grad_J_down + grad_logD_down)
        
        # Combine and return
        return jnp.concatenate([quantum_force_up, quantum_force_down], axis=0)
