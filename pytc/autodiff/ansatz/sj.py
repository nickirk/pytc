"""Implementation of the quantum many-body wavefunction ansatz."""

import jax
import jax.numpy as jnp
from typing import List, Any

class SlaterJastrow:
    """Quantum many-body wavefunction ansatz combining Jastrow factor with Slater determinants."""
    
    def __init__(self, jastrow, dets: List[Any], linear_coeffs: jnp.ndarray):
        """Initialize the ansatz.
        
        Args:
            jastrow: Jastrow factor object
            determinants: List of Slater determinant objects
            coefficients: Array of linear coefficients for determinants
        """
        self.jastrow = jastrow
        self.dets = dets
        self.linear_coeffs = jnp.asarray(linear_coeffs, dtype=jnp.float64)
        
    def __call__(self, elec_coords):
        """Evaluate wavefunction at given electron positions.
        
        Args:
            electron_positions: Array of shape (N_electrons, 3) where first n_alpha 
                              positions are spin-up electrons
        
        Returns:
            Complex value of wavefunction
        """
        # TODO: Current Jastrow implementation treats all electron pairs equally
        # Future improvement: Differentiate between same-spin and opposite-spin pairs
        
        # Create vectorized Jastrow evaluation for one electron against all others
        vmap_jastrow = jax.vmap(self.jastrow, in_axes=(None, 0))
        
        # Further vectorize over the first electron position
        # This gives us all pairs of electrons
        vmap_jastrow_all = jax.vmap(vmap_jastrow, in_axes=(0, None))
        
        # Compute all pairwise Jastrow values in one shot
        # Result shape is (n_electrons, n_electrons)
        all_pairs = vmap_jastrow_all(elec_coords, elec_coords)
        
        # Remove extra dimensions from all_pairs
        all_pairs = jnp.squeeze(all_pairs)  # Should now be shape (2,2)
        
        # Create upper triangular mask (excluding diagonal)
        ones = jnp.ones_like(all_pairs)  # Will inherit correct shape (2,2)
        mask = jnp.triu(ones, k=1)
        
        # Multiply Jastrow values where mask is 1, ignore others
        jastrow_val = jnp.exp(jnp.sum(all_pairs * mask))
        
        # Evaluate each determinant and combine with coefficients
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
    