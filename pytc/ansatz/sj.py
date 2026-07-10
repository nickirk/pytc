"""Implementation of the quantum many-body wavefunction ansatz."""

import jax
import jax.numpy as jnp
from typing import List, Any
from flax import struct

from pytc.ansatz.det import SlaterDet, value_and_grad, grad 

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

    def local_energy(self, walker, params, jastrow_terms_impl="pairwise"):
        return eval_local_energy(self, walker, params, jastrow_terms_impl=jastrow_terms_impl)
    
    def quantum_force(self, walker, params, cutoff=5.0):
        return eval_sj_quantum_force(self, walker, params, cutoff)

    def init_params(self, key):
        """Initialize parameters for the ansatz."""
        jastrow_params = self.jastrow.init_params()
        linear_coeffs = jnp.ones(len(self.dets))
        return [jastrow_params, linear_coeffs]


# Standalone functions

def compute_jastrow_log_value(sj: SlaterJastrow, elec_coords, jastrow_params):
    """Compute Jastrow factor in log space for numerical stability.
    
    Assumes unbatched elec_coords with shape (n_electrons, 3).
    Use vmap for batched processing.
    """
    elec_coords = jnp.asarray(elec_coords)
    n_electrons = elec_coords.shape[0]
    
    # Create indices for unique pairs (i < j)
    # We use triu_indices to get the upper triangle indices
    rows, cols = jnp.triu_indices(n_electrons, k=1)
    
    # Pre-bind the compute function to avoid overhead
    compute_fn = sj.jastrow._compute
    
    def scan_body(carry, pair_idx):
        i, j = pair_idx
        r1 = elec_coords[i]
        r2 = elec_coords[j]
        
        val = compute_fn(r1, r2, jastrow_params)
        return carry + val, None

    # Scan over all unique pairs
    # We stack rows and cols to scan over them together
    pair_indices = jnp.stack([rows, cols], axis=1)
    
    log_j_val, _ = jax.lax.scan(scan_body, 0.0, pair_indices)
            
    return log_j_val


def update_jastrow_one_electron(sj: SlaterJastrow, old_positions, new_positions,
                                 electron_idx, jastrow_params, old_log_jastrow):
    """Update Jastrow log-value after moving one electron.

    Recomputes only the N-1 pairs involving ``electron_idx`` instead of
    all N(N-1)/2 pairs:
        new_log_J = old_log_J + sum_{j != k} [u_new(k,j) - u_old(k,j)]

    Respects the argument order convention of compute_jastrow_log_value
    (sum_{i<j} u(r_i, r_j)), which matters when u is asymmetric.

    Args:
        sj: SlaterJastrow ansatz
        old_positions: (n_electrons, 3)
        new_positions: (n_electrons, 3)
        electron_idx: int
        jastrow_params: Jastrow parameters
        old_log_jastrow: scalar

    Returns:
        new_log_jastrow: scalar
    """
    n_electrons = old_positions.shape[0]
    compute_fn = sj.jastrow._compute

    r_k_old = old_positions[electron_idx]
    r_k_new = new_positions[electron_idx]

    other_indices = jnp.arange(n_electrons)

    def scan_body(carry, j):
        r_j = old_positions[j]
        j_less_than_k = (j < electron_idx)
        # j < k: pair was u(r_j, r_k); j > k: pair was u(r_k, r_j)
        val_new = jnp.where(j_less_than_k,
                            compute_fn(r_j, r_k_new, jastrow_params),
                            compute_fn(r_k_new, r_j, jastrow_params))
        val_old = jnp.where(j_less_than_k,
                            compute_fn(r_j, r_k_old, jastrow_params),
                            compute_fn(r_k_old, r_j, jastrow_params))
        is_self = (j == electron_idx)
        delta = jnp.where(is_self, 0.0, val_new - val_old)
        return carry + delta, None

    delta_log_j, _ = jax.lax.scan(scan_body, 0.0, other_indices)

    return old_log_jastrow + delta_log_j


def _combine_multi_dets(dets, walker, linear_coeffs):
    """Combine multiple determinants with linear coefficients.

    Evaluates each determinant, converts to scalar values, and computes
    the signed linear combination in log space.

    Returns:
        ((sign, logabs), walker_from_first_det)
    """
    det_vals_list = []
    final_updated_walker = None

    for i, det in enumerate(dets):
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

    return (linear_combo_sign, linear_combo_logabs), final_updated_walker


def eval_sj(sj: SlaterJastrow, walker, params):
    """Evaluate wavefunction for a single walker with explicit parameters."""
    jastrow_params, linear_coeffs = params

    log_jastrow_val = compute_jastrow_log_value(sj, walker.positions, jastrow_params)

    if len(sj.dets) == 1:
        det_val, final_updated_walker = value_and_grad(sj.dets[0], walker)
        det_sign, det_logabs = det_val
        linear_combo_sign = jnp.sign(linear_coeffs[0]) * det_sign
        linear_combo_logabs = jnp.log(jnp.abs(linear_coeffs[0])) + det_logabs
    else:
        (linear_combo_sign, linear_combo_logabs), final_updated_walker = (
            _combine_multi_dets(sj.dets, walker, linear_coeffs))

    psi_sign = linear_combo_sign
    psi_logabs = log_jastrow_val + linear_combo_logabs
    psi_values = (psi_sign, psi_logabs)

    final_updated_walker = final_updated_walker.replace(
        log_psi=psi_logabs,
        psi_sign=psi_sign,
        log_jastrow=log_jastrow_val,
    )

    return psi_values, final_updated_walker

from pytc.vmc.hamiltonian import (
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
