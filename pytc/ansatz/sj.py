"""Implementation of the quantum many-body wavefunction ansatz."""

import jax
import jax.numpy as jnp
from typing import List, Any
from flax import struct

from pytc.ansatz.det import SlaterDet, value_and_grad, grad


@struct.dataclass
class MultiSlaterRef:
    """Linear combination of Slater determinants, WITHOUT a Jastrow factor.

    Used as the **fixed-form reference distribution** for VMC sampling of
    multi-determinant CSFs (e.g. singlet CIS states built as
    (1/sqrt(2)) (D_{ia->aa} + D_{ib->ab})).

    The walker distribution is the correct quantum density
        |Psi_ref|^2 = |sum_i c_i D_i(R)|^2
    of the multi-det reference, including the inter-determinant
    interference term Re(D_i^* D_j) — which the single-det
    ``ansatz.dets[0]`` reference would have missed.

    The linear coefficients c_i are read from ``params[1]`` at call time
    (so they update continuously alongside the Jastrow during state-averaged
    optimization). The Jastrow params ``params[0]`` are accepted from the
    call signature but ignored: this object intentionally has no Jastrow.

    For a single-determinant case (len(dets) == 1) this reduces to the
    standard ``SlaterDet`` reference up to the trivial scalar factor c_0.
    """
    dets: List[SlaterDet]

    @classmethod
    def from_dets(cls, dets):
        return cls(dets=list(dets))

    # Attributes that walker init / metropolis machinery reads off the ref:
    @property
    def n_electrons(self):
        return self.dets[0].n_electrons

    @property
    def n_alpha(self):
        return self.dets[0].n_alpha

    @property
    def n_beta(self):
        return self.dets[0].n_beta

    @property
    def atom_coords(self):
        return self.dets[0].atom_coords

    @property
    def atom_charges(self):
        return self.dets[0].atom_charges

    @property
    def mol_gto(self):
        return self.dets[0].mol_gto

    @property
    def eval_ao_func(self):
        return self.dets[0].eval_ao_func

    @property
    def alpha_occ(self):
        return self.dets[0].alpha_occ

    @property
    def beta_occ(self):
        return self.dets[0].beta_occ

    @property
    def unrestricted(self):
        return self.dets[0].unrestricted

    def __call__(self, walker, params):
        return eval_multi_slater_ref(self, walker, params)


def eval_multi_slater_ref(ms_ref, walker, params):
    """Evaluate Psi_ref(R) = sum_i c_i * D_i(R) for a single walker.

    Mirrors ``eval_sj`` minus the Jastrow contribution. Returns the same
    ((sign, log|Psi|), updated_walker) tuple the metropolis kernel expects,
    with ``log_jastrow`` set to 0 on the returned walker (no Jastrow here).

    Args:
        ms_ref: ``MultiSlaterRef`` ansatz.
        walker: single Walker dataclass instance.
        params: ``[jastrow_params, linear_coeffs]``. ``jastrow_params`` is
            accepted but ignored; ``linear_coeffs`` (length == len(dets))
            sets the linear combination.

    Returns:
        ((psi_sign, psi_logabs), updated_walker)
    """
    _jastrow_params, linear_coeffs = params

    if len(ms_ref.dets) == 1:
        det_val, final_walker = value_and_grad(ms_ref.dets[0], walker)
        det_sign, det_logabs = det_val
        psi_sign = jnp.sign(linear_coeffs[0]) * det_sign
        psi_logabs = jnp.log(jnp.abs(linear_coeffs[0])) + det_logabs
    else:
        det_vals_list = []
        final_walker = None
        for i, det in enumerate(ms_ref.dets):
            det_val, uw = value_and_grad(det, walker)
            det_sign, det_logabs = det_val
            det_vals_list.append(det_sign * jnp.exp(det_logabs))
            if i == 0:
                final_walker = uw

        det_vals_array = jnp.array(det_vals_list)
        linear_combo = jnp.sum(linear_coeffs * det_vals_array)
        psi_sign = jnp.sign(linear_combo)
        psi_logabs = jnp.log(jnp.abs(linear_combo) + 1e-100)

    final_walker = final_walker.replace(
        log_psi=psi_logabs,
        psi_sign=psi_sign,
        log_jastrow=jnp.zeros_like(psi_logabs),  # no Jastrow in the reference
    )
    return (psi_sign, psi_logabs), final_walker

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
    
    # Cache psi values and Jastrow in the walker for MCMC reuse
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
