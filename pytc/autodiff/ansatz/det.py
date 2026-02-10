import numpy as np
from functools import partial
from typing import Union, Callable, Any, Tuple
import jax
import jax.numpy as jnp
from flax import struct
from pytc.autodiff.ansatz.gto import MolGTO, eval_ao
from pytc.autodiff.ansatz.gto_spherical import MolGTO_Spherical, eval_ao_spherical

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

# Aliases for compatibility
value_and_grad = eval_det_value_and_grad
grad = eval_det_grad
laplacian = eval_det_laplacian
value = eval_det_value
matrix = eval_det_matrix
