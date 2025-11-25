"""Implementation of the quantum many-body wavefunction ansatz."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Any
from functools import partial
from flax import struct

from pytc.autodiff.ansatz.det import SlaterDet, value_and_grad, grad 

@struct.dataclass
class SlaterJastrow:
    """Quantum many-body wavefunction ansatz combining Jastrow factor with Slater determinants."""
    dets: List[SlaterDet]
    atom_coords: jax.Array
    atom_charges: jax.Array
    ion_ion_potential: jax.Array
    jastrow: Any 

    @property
    def n_electrons(self):
        return self.dets[0].n_electrons

    @property
    def n_alpha(self):
        return self.dets[0].n_alpha

    @property
    def n_beta(self):
        return self.dets[0].n_beta

    @classmethod
    def create(cls, mol, jastrow, dets: List[SlaterDet]):
        """Initialize the ansatz without storing optimizable parameters."""
        atom_coords = jnp.array(mol.atom_coords())
        atom_charges = jnp.array(mol.atom_charges())
        
        # Calculate ion-ion potential energy
        n_atoms = len(atom_charges)
        R_diff = atom_coords[:, None, :] - atom_coords[None, :, :]
        R_dist = jnp.linalg.norm(R_diff, axis=-1)
        charge_products = jnp.outer(atom_charges, atom_charges)
        mask = 1-jnp.eye(n_atoms)
        v_ion_ion = jnp.sum(charge_products * mask / (R_dist+1e-10)) / 2.0
        
        return cls(
            dets=dets,
            atom_coords=atom_coords,
            atom_charges=atom_charges,
            ion_ion_potential=v_ion_ion,
            jastrow=jastrow
        )

    # Compatibility method for __call__
    def __call__(self, walker, params):
        return eval_sj(self, walker, params)

    def local_energy(self, walker, params):
        return eval_local_energy(self, walker, params)
    
    def quantum_force(self, walker, params, cutoff=5.0):
        return eval_sj_quantum_force(self, walker, params, cutoff)


# Standalone functions

def compute_jastrow_log_value(sj: SlaterJastrow, elec_coords, jastrow_params):
    """Compute Jastrow factor in log space for numerical stability.
    
    Assumes unbatched elec_coords with shape (n_electrons, 3).
    Use vmap for batched processing.
    """
    vmap_single = jax.vmap(sj.jastrow._compute, in_axes=(None, 0, None))
    vmap_all = jax.vmap(vmap_single, in_axes=(0, None, None))
    
    all_pairs = vmap_all(elec_coords, elec_coords, jastrow_params)
    
    n_electrons = elec_coords.shape[0]
    diag_mask = 1.0 - jnp.eye(n_electrons)
    
    return 0.5 * jnp.sum(all_pairs * diag_mask)

def eval_sj(sj: SlaterJastrow, walker, params):
    """Evaluate wavefunction for a single walker with explicit parameters."""
    jastrow_params, linear_coeffs = params
    
    # Compute Jastrow value for single walker
    log_jastrow_val = compute_jastrow_log_value(sj, walker.positions, jastrow_params)
    
    # Compute determinant values
    if len(sj.dets) == 1:
        det_val, final_updated_walker = value_and_grad(sj.dets[0], walker)
        det_sign, det_logabs = det_val
        linear_combo_sign = jnp.sign(linear_coeffs[0]) * det_sign
        linear_combo_logabs = jnp.log(jnp.abs(linear_coeffs[0])) + det_logabs
    else:
        det_vals_list = []
        final_updated_walker = None
        
        for i, det in enumerate(sj.dets):
            det_val, updated_walker = value_and_grad(det, walker)
            det_sign, det_logabs = det_val
            det_val_scalar = det_sign * jnp.exp(det_logabs)
            det_vals_list.append(det_val_scalar)
            if i == 0:
                final_updated_walker = updated_walker
        
        det_vals_array = jnp.array(det_vals_list)
        linear_combo = jnp.sum(linear_coeffs * det_vals_array)
        linear_combo_sign = jnp.sign(linear_combo)
        linear_combo_logabs = jnp.log(jnp.abs(linear_combo) + 1e-100)
    
    psi_sign = linear_combo_sign
    psi_logabs = log_jastrow_val + linear_combo_logabs
    psi_values = (psi_sign, psi_logabs)
    
    return psi_values, final_updated_walker

from pytc.autodiff.vmc.hamiltonian import (
    eval_local_energy,
    compute_jastrow_terms
)

def compute_quantum_force(sj, grad_J_over_J, slater_alpha, slater_beta, grad_alpha, grad_beta):
    """Compute quantum force for a single configuration."""
    n_alpha = sj.n_alpha
    
    grad_J_alpha = grad_J_over_J[:n_alpha]
    grad_J_beta = grad_J_over_J[n_alpha:]
    
    inv_alpha = jnp.linalg.inv(slater_alpha)
    inv_beta = jnp.linalg.inv(slater_beta)
    
    grad_logD_alpha = jnp.einsum('ij,ijk->ik', inv_alpha, grad_alpha)
    grad_logD_beta = jnp.einsum('ij,ijk->ik', inv_beta, grad_beta)
    
    quantum_force_alpha = 2.0 * (grad_J_alpha + grad_logD_alpha)
    quantum_force_beta = 2.0 * (grad_J_beta + grad_logD_beta)
    
    return jnp.concatenate([quantum_force_alpha, quantum_force_beta], axis=0)

def eval_sj_quantum_force(sj: SlaterJastrow, walker, params, cutoff=5.0):
    jastrow_params, linear_coeffs = params 
    elec_coords = walker.positions
    
    grad_J_over_J = compute_jastrow_terms(sj, elec_coords, jastrow_params)[0]
    
    det = sj.dets[0]
    slater_alpha, slater_beta, grad_alpha, grad_beta = grad(det, walker)
    
    forces = compute_quantum_force(
        sj,
        grad_J_over_J,
        slater_alpha,
        slater_beta,
        grad_alpha,
        grad_beta
    )

    force_magnitudes = jnp.linalg.norm(forces, axis=-1)
    scaling_factors = jnp.minimum(1.0, cutoff / (force_magnitudes + 1e-10))
    scaling_factors = scaling_factors[..., jnp.newaxis]
    clipped_forces = forces * scaling_factors
    
    return clipped_forces
