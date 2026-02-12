import numpy as np
from functools import partial
from typing import Union, Callable, Any, Tuple
import jax
import jax.numpy as jnp
from flax import struct
from pytc.ansatz.gto import MolGTO, eval_ao
from pytc.ansatz.gto_spherical import MolGTO_Spherical, eval_ao_spherical

@struct.dataclass
class SlaterDet:
    """
    Slater determinant ansatz.
    Stores configuration and parameters as a PyTree.
    """
    mo_coeff_alpha_occ: jax.Array
    mo_coeff_beta_occ: jax.Array
    mol_gto: Union[MolGTO, MolGTO_Spherical]
    n_alpha: int = struct.field(pytree_node=False)
    n_beta: int = struct.field(pytree_node=False)
    alpha_occ: Tuple[int] = struct.field(pytree_node=False)
    beta_occ: Tuple[int] = struct.field(pytree_node=False)
    atom_coords: jax.Array = struct.field(pytree_node=False)
    atom_charges: jax.Array = struct.field(pytree_node=False)
    eval_ao_func: Callable = struct.field(pytree_node=False)
    unrestricted: bool = struct.field(pytree_node=False, default=False)

    @property
    def n_electrons(self):
        return self.n_alpha + self.n_beta

    @classmethod
    def create(cls, mol, mo_coeff=None, nelec=None, excitations=None):
        """
        Factory method to create a SlaterDet instance.
        """
        if nelec is None:
            n_alpha, n_beta = mol.nelec 
        else:
            n_alpha, n_beta = nelec
            
        # Initialize the appropriate GTO evaluator
        if mol.cart:
            mol_gto = MolGTO.create(mol)
            eval_ao_func = eval_ao
        else:
            mol_gto = MolGTO_Spherical.create(mol)
            eval_ao_func = eval_ao_spherical
    
        # Detect if mo_coeff is restricted or unrestricted
        if isinstance(mo_coeff, (list, tuple)):
            # mo_coeff[0] = alpha, mo_coeff[1] = beta
            mo_coeff_alpha = mo_coeff[0]
            mo_coeff_beta = mo_coeff[1]
        else:
            # Single set of coefficients, treat as RHF
            mo_coeff_alpha = mo_coeff
            mo_coeff_beta = mo_coeff
    
        # Default occupied orbitals (HF reference)
        alpha_occ = list(range(n_alpha)) 
        beta_occ = list(range(n_beta))  

        # Apply excitations if specified
        if excitations is not None:
            alpha_exc, beta_exc = excitations
            
            # Handle alpha excitations
            if alpha_exc and len(alpha_exc) == 2:
                from_idx, to_idx = alpha_exc
                if len(from_idx) != len(to_idx):
                    raise ValueError("Number of occupied and virtual orbitals must match for alpha excitations")
                for i, a in zip(from_idx, to_idx):
                    if i not in alpha_occ:
                        raise ValueError(f"Cannot remove electron from unoccupied alpha orbital {i}")
                    if a in alpha_occ:
                        raise ValueError(f"Cannot add electron to already occupied alpha orbital {a}")
                    alpha_occ.remove(i)  
                    alpha_occ.append(a)  
                alpha_occ.sort()
                
            # Handle beta excitations
            if beta_exc and len(beta_exc) == 2:
                from_idx, to_idx = beta_exc
                if len(from_idx) != len(to_idx):
                    raise ValueError("Number of occupied and virtual orbitals must match for beta excitations")
                for i, a in zip(from_idx, to_idx):
                    if i not in beta_occ:
                        raise ValueError(f"Cannot remove electron from unoccupied beta orbital {i}")
                    if a in beta_occ:
                        raise ValueError(f"Cannot add electron to already occupied beta orbital {a}")
                    beta_occ.remove(i)  
                    beta_occ.append(a)  
                beta_occ.sort()

        # Extract occupied MO coefficients
        mo_coeff_alpha_occ = jnp.array(mo_coeff_alpha[:, alpha_occ])
        mo_coeff_beta_occ = jnp.array(mo_coeff_beta[:, beta_occ])

        atom_coords = jnp.array(mol.atom_coords())
        atom_charges = jnp.array(mol.atom_charges())
        unrestricted = isinstance(mo_coeff, (list, tuple))
        
        return cls(
            mo_coeff_alpha_occ=mo_coeff_alpha_occ,
            mo_coeff_beta_occ=mo_coeff_beta_occ,
            mol_gto=mol_gto,
            n_alpha=n_alpha,
            n_beta=n_beta,
            alpha_occ=tuple(alpha_occ),
            beta_occ=tuple(beta_occ),
            unrestricted=unrestricted,
            atom_coords=atom_coords,
            atom_charges=atom_charges,
            eval_ao_func=eval_ao_func
        )
    
    # Compatibility method for tests that call det.matrix()
    def matrix(self, coords):
        return eval_det_matrix(self, coords)
        
    def grad(self, walker):
        return eval_det_grad(self, walker)

    def __call__(self, walker, params=None):
        return eval_det_value_and_grad(self, walker)

def eval_det_value(det: SlaterDet, walker):
    """
    Compute determinant values and update walker.
    """
    positions = walker.positions
    is_batched = positions.ndim == 3
    
    # Evaluate AOs for all electrons
    ao_vals = det.eval_ao_func(det.mol_gto, positions, deriv=0) 
    
    # Split into alpha and beta
    if is_batched:
        ao_alpha = ao_vals[:, :det.n_alpha, :]
        ao_beta = ao_vals[:, det.n_alpha:, :]
        einsum_str = 'bix,xj->bij'
    else:
        ao_alpha = ao_vals[:det.n_alpha, :]
        ao_beta = ao_vals[det.n_alpha:, :]
        einsum_str = 'ix,xj->ij'
    
    # Compute Slater matrices
    slater_up = jnp.einsum(einsum_str, ao_alpha, det.mo_coeff_alpha_occ)
    slater_down = jnp.einsum(einsum_str, ao_beta, det.mo_coeff_beta_occ)
    
    # Compute determinants and inverses
    sign_up, logdet_up = jnp.linalg.slogdet(slater_up)
    sign_down, logdet_down = jnp.linalg.slogdet(slater_down)
    
    inv_up = jnp.linalg.inv(slater_up)
    inv_down = jnp.linalg.inv(slater_down)
    
    det_sign = sign_up * sign_down
    det_logabs = logdet_up + logdet_down
    
    updated_walker = walker.replace(
        slater_up=slater_up,
        slater_down=slater_down,
        inv_up=inv_up,
        inv_down=inv_down,
        det_up=(sign_up, logdet_up),
        det_down=(sign_down, logdet_down),
        log_psi=det_logabs,
        psi_sign=det_sign,
    )
    
    return (det_sign, det_logabs), updated_walker

def eval_det_value_and_grad(det: SlaterDet, walker):
    """
    Compute determinant values, gradients, and laplacians.
    """
    positions = walker.positions
    is_batched = positions.ndim == 3
    
    ao_vals, ao_grad, ao_lap = det.eval_ao_func(det.mol_gto, positions, deriv=2)
    
    if is_batched:
        ao_alpha = ao_vals[:, :det.n_alpha, :]
        ao_beta = ao_vals[:, det.n_alpha:, :]
        ao_grad_alpha = ao_grad[:, :det.n_alpha, :, :]
        ao_grad_beta = ao_grad[:, det.n_alpha:, :, :]
        ao_lap_alpha = ao_lap[:, :det.n_alpha, :]
        ao_lap_beta = ao_lap[:, det.n_alpha:, :]
        einsum_str_val = 'bix,xj->bij'
        einsum_str_grad = 'bixd,xj->bijd'
    else:
        ao_alpha = ao_vals[:det.n_alpha, :]
        ao_beta = ao_vals[det.n_alpha:, :]
        ao_grad_alpha = ao_grad[:det.n_alpha, :, :]
        ao_grad_beta = ao_grad[det.n_alpha:, :, :]
        ao_lap_alpha = ao_lap[:det.n_alpha, :]
        ao_lap_beta = ao_lap[det.n_alpha:, :]
        einsum_str_val = 'ix,xj->ij'
        einsum_str_grad = 'ixd,xj->ijd'
    
    slater_up = jnp.einsum(einsum_str_val, ao_alpha, det.mo_coeff_alpha_occ)
    slater_down = jnp.einsum(einsum_str_val, ao_beta, det.mo_coeff_beta_occ)
    
    grad_up = jnp.einsum(einsum_str_grad, ao_grad_alpha, det.mo_coeff_alpha_occ)
    grad_down = jnp.einsum(einsum_str_grad, ao_grad_beta, det.mo_coeff_beta_occ)
    
    lap_up = jnp.einsum(einsum_str_val, ao_lap_alpha, det.mo_coeff_alpha_occ)
    lap_down = jnp.einsum(einsum_str_val, ao_lap_beta, det.mo_coeff_beta_occ)
    
    sign_up, logdet_up = jnp.linalg.slogdet(slater_up)
    sign_down, logdet_down = jnp.linalg.slogdet(slater_down)
    
    inv_up = jnp.linalg.inv(slater_up)
    inv_down = jnp.linalg.inv(slater_down)
    
    det_sign = sign_up * sign_down
    det_logabs = logdet_up + logdet_down
    
    updated_walker = walker.replace(
        slater_up=slater_up,
        slater_down=slater_down,
        inv_up=inv_up,
        inv_down=inv_down,
        det_up=(sign_up, logdet_up),
        det_down=(sign_down, logdet_down),
        grad_up=grad_up,
        grad_down=grad_down,
        lap_up=lap_up,
        lap_down=lap_down,
        log_psi=det_logabs,
        psi_sign=det_sign,
    )
    
    return (det_sign, det_logabs), updated_walker

def eval_det_grad(det: SlaterDet, walker):
    """
    Compute gradients only.
    """
    positions = walker.positions
    is_batched = positions.ndim == 3
    
    ao_vals, ao_grad = det.eval_ao_func(det.mol_gto, positions, deriv=1)
    
    if is_batched:
        ao_alpha = ao_vals[:, :det.n_alpha, :]
        ao_beta = ao_vals[:, det.n_alpha:, :]
        ao_grad_alpha = ao_grad[:, :det.n_alpha, :, :]
        ao_grad_beta = ao_grad[:, det.n_alpha:, :, :]
        einsum_str_val = 'bix,xj->bij'
        einsum_str_grad = 'bixd,xj->bijd'
    else:
        ao_alpha = ao_vals[:det.n_alpha, :]
        ao_beta = ao_vals[det.n_alpha:, :]
        ao_grad_alpha = ao_grad[:det.n_alpha, :, :]
        ao_grad_beta = ao_grad[det.n_alpha:, :, :]
        einsum_str_val = 'ix,xj->ij'
        einsum_str_grad = 'ixd,xj->ijd'
    
    slater_up = jnp.einsum(einsum_str_val, ao_alpha, det.mo_coeff_alpha_occ)
    slater_down = jnp.einsum(einsum_str_val, ao_beta, det.mo_coeff_beta_occ)
    
    grad_up = jnp.einsum(einsum_str_grad, ao_grad_alpha, det.mo_coeff_alpha_occ)
    grad_down = jnp.einsum(einsum_str_grad, ao_grad_beta, det.mo_coeff_beta_occ)
    
    return (slater_up, slater_down, grad_up, grad_down)

def eval_det_laplacian(det: SlaterDet, walker):
    """
    Compute laplacians only (and values/grads as needed).
    """
    (det_sign, det_logabs), updated_walker = eval_det_value_and_grad(det, walker)
    
    return ((updated_walker.slater_up, updated_walker.slater_down),
            (updated_walker.grad_up, updated_walker.grad_down),
            (updated_walker.lap_up, updated_walker.lap_down),
            updated_walker)

def eval_det_matrix(det: SlaterDet, coords):
    """
    Compute Slater matrices for given coordinates.
    """
    is_batched = coords.ndim == 3
    ao_vals = det.eval_ao_func(det.mol_gto, coords, deriv=0)
    
    if is_batched:
        ao_alpha = ao_vals[:, :det.n_alpha, :]
        ao_beta = ao_vals[:, det.n_alpha:, :]
        einsum_str = 'bix,xj->bij'
    else:
        ao_alpha = ao_vals[:det.n_alpha, :]
        ao_beta = ao_vals[det.n_alpha:, :]
        einsum_str = 'ix,xj->ij'
    
    slater_up = jnp.einsum(einsum_str, ao_alpha, det.mo_coeff_alpha_occ)
    slater_down = jnp.einsum(einsum_str, ao_beta, det.mo_coeff_beta_occ)
    
    return slater_up, slater_down

# =====================================================================
# Sherman-Morrison rank-1 update functions for single-electron moves
# =====================================================================

def eval_single_electron_ao(det: SlaterDet, pos_single):
    """Evaluate AOs (value, gradient, laplacian) for a single electron position.

    Args:
        det: SlaterDet object
        pos_single: shape (3,)

    Returns:
        ao_val:  (nao,), ao_grad: (nao, 3), ao_lap: (nao,)
    """
    pos_expand = pos_single[None, :]  # (1, 3)
    ao_val, ao_grad, ao_lap = det.eval_ao_func(det.mol_gto, pos_expand, deriv=2)
    return ao_val[0], ao_grad[0], ao_lap[0]


def compute_new_row(det: SlaterDet, ao_val, ao_grad, ao_lap, is_alpha):
    """Compute new Slater row, gradient row, and Laplacian row for one electron.

    Args:
        det: SlaterDet object
        ao_val:  (nao,)
        ao_grad: (nao, 3)
        ao_lap:  (nao,)
        is_alpha: bool

    Returns:
        new_row: (n_occ,), new_grad_row: (n_occ, 3), new_lap_row: (n_occ,)
    """
    mo_coeff = det.mo_coeff_alpha_occ if is_alpha else det.mo_coeff_beta_occ
    new_row = ao_val @ mo_coeff
    new_grad_row = jnp.einsum('xd,xj->jd', ao_grad, mo_coeff)
    new_lap_row = ao_lap @ mo_coeff
    return new_row, new_grad_row, new_lap_row


def compute_det_ratio_from_row(new_row, inv, row_idx):
    """Compute det(S')/det(S) for a rank-1 row update.

    With inv = S^{-1} and S' differing from S only in row ``row_idx``:
        ratio = new_row @ inv[:, row_idx]

    Follows from cofactor expansion: cofactor(S,k,j)/det(S) = (S^{-1})[j,k].

    Args:
        new_row: (n_occ,)
        inv:     (n_occ, n_occ) — S^{-1}
        row_idx: int

    Returns:
        ratio: scalar
    """
    return new_row @ inv[:, row_idx]


def update_inverse_sherman_morrison(inv, new_row, old_row, row_idx, ratio):
    """Update the inverse matrix via Sherman-Morrison after a rank-1 row update.

    Using S @ inv = I, we have old_row @ inv = e_k^T, so:
        inv' = inv - outer(inv[:, k], new_row @ inv - e_k^T) / ratio

    Args:
        inv:     (n_occ, n_occ) — current S^{-1}
        new_row: (n_occ,)
        old_row: (n_occ,) — unused, kept for API clarity
        row_idx: int
        ratio:   scalar — det(S')/det(S)

    Returns:
        inv': (n_occ, n_occ)
    """
    col_k = inv[:, row_idx]
    row_update = new_row @ inv
    row_update = row_update.at[row_idx].add(-1.0)
    inv_new = inv - jnp.outer(col_k, row_update) / ratio
    return inv_new


def rank1_update_one_electron(det: SlaterDet, walker, electron_idx):
    """Rank-1 update of determinant quantities after a single-electron move.

    Given a walker whose ``positions[electron_idx]`` has already been set to
    the new location, updates slater rows, inverse, grad/lap rows, and
    log-determinant/sign.  O(N²) instead of O(N³) full recomputation.

    Args:
        det: SlaterDet object
        walker: Walker (unbatched)
        electron_idx: int — index of the moved electron

    Returns:
        total_ratio: scalar — det(S')/det(S) (product over spin channels)
        det_logabs_new: scalar — updated log|det|
        det_sign_new: scalar — updated sign
        updated_walker: Walker with updated fields
    """
    n_alpha = det.n_alpha

    # Evaluate AOs at the new electron position
    new_pos = walker.positions[electron_idx]
    ao_val, ao_grad, ao_lap = eval_single_electron_ao(det, new_pos)

    # Determine spin and local row index
    is_alpha = electron_idx < n_alpha
    local_idx = jnp.where(is_alpha, electron_idx, electron_idx - n_alpha)

    # Compute new row for both spin channels (only the affected one is used)
    new_row_up, new_grad_row_up, new_lap_row_up = compute_new_row(
        det, ao_val, ao_grad, ao_lap, is_alpha=True)
    new_row_dn, new_grad_row_dn, new_lap_row_dn = compute_new_row(
        det, ao_val, ao_grad, ao_lap, is_alpha=False)

    # Det ratio for affected spin channel
    ratio_up = jnp.where(
        is_alpha,
        compute_det_ratio_from_row(new_row_up, walker.inv_up, local_idx),
        1.0)
    ratio_dn = jnp.where(
        is_alpha, 1.0,
        compute_det_ratio_from_row(new_row_dn, walker.inv_down, local_idx))

    # Sherman-Morrison inverse update
    old_row_up = walker.slater_up[local_idx]
    inv_up_new = jnp.where(
        is_alpha,
        update_inverse_sherman_morrison(
            walker.inv_up, new_row_up, old_row_up, local_idx, ratio_up),
        walker.inv_up)
    old_row_dn = walker.slater_down[local_idx]
    inv_dn_new = jnp.where(
        is_alpha, walker.inv_down,
        update_inverse_sherman_morrison(
            walker.inv_down, new_row_dn, old_row_dn, local_idx, ratio_dn))

    # Update Slater matrix row
    slater_up_new = jnp.where(
        is_alpha, walker.slater_up.at[local_idx].set(new_row_up), walker.slater_up)
    slater_dn_new = jnp.where(
        is_alpha, walker.slater_down,
        walker.slater_down.at[local_idx].set(new_row_dn))

    # Update gradient and laplacian rows
    grad_up_new = jnp.where(
        is_alpha, walker.grad_up.at[local_idx].set(new_grad_row_up), walker.grad_up)
    grad_dn_new = jnp.where(
        is_alpha, walker.grad_down,
        walker.grad_down.at[local_idx].set(new_grad_row_dn))
    lap_up_new = jnp.where(
        is_alpha, walker.lap_up.at[local_idx].set(new_lap_row_up), walker.lap_up)
    lap_dn_new = jnp.where(
        is_alpha, walker.lap_down,
        walker.lap_down.at[local_idx].set(new_lap_row_dn))

    # Update log-determinant and sign
    sign_up_old, logdet_up_old = walker.det_up
    sign_dn_old, logdet_dn_old = walker.det_down

    sign_up_new = jnp.where(is_alpha, sign_up_old * jnp.sign(ratio_up), sign_up_old)
    logdet_up_new = jnp.where(is_alpha, logdet_up_old + jnp.log(jnp.abs(ratio_up)), logdet_up_old)
    sign_dn_new = jnp.where(is_alpha, sign_dn_old, sign_dn_old * jnp.sign(ratio_dn))
    logdet_dn_new = jnp.where(is_alpha, logdet_dn_old, logdet_dn_old + jnp.log(jnp.abs(ratio_dn)))

    det_sign_new = sign_up_new * sign_dn_new
    det_logabs_new = logdet_up_new + logdet_dn_new

    updated_walker = walker.replace(
        slater_up=slater_up_new,
        slater_down=slater_dn_new,
        inv_up=inv_up_new,
        inv_down=inv_dn_new,
        det_up=(sign_up_new, logdet_up_new),
        det_down=(sign_dn_new, logdet_dn_new),
        grad_up=grad_up_new,
        grad_down=grad_dn_new,
        lap_up=lap_up_new,
        lap_down=lap_dn_new,
    )

    total_ratio = ratio_up * ratio_dn
    return total_ratio, det_logabs_new, det_sign_new, updated_walker


# Aliases for compatibility
value_and_grad = eval_det_value_and_grad
grad = eval_det_grad
laplacian = eval_det_laplacian
value = eval_det_value
matrix = eval_det_matrix
