import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from pytc.autodiff.ansatz.gto import MolGTO, eval_ao
from pytc.autodiff.ansatz.gto_spherical import MolGTO_Spherical, eval_ao_spherical

class SlaterDet:
    def __init__(self, mol, mo_coeff=None, nelec=None, excitations=None):
        """
        Args:
        mol: A PySCF mol object (provides integrals, eval_gto, etc.)
        mo_coeff: Either a single np.ndarray of shape (nAOs, nMOs) for RHF,
        or a tuple/list [mo_coeff_alpha, mo_coeff_beta] each of
        shape (nAOs, nMOs) for UHF.
        nelec: Number of electrons as a tuple (n_alpha, n_beta).
        excitations: Tuple of (alpha_excitations, beta_excitations) where each is a tuple of
                    (from_indices, to_indices) specifying which orbitals to remove and add.
        """
        self.mol = mol
        if nelec is None:
            self.n_alpha, self.n_beta = mol.nelec 
        else:
            self.n_alpha, self.n_beta = nelec
            
        # Initialize the appropriate GTO evaluator
        if self.mol.cart:
            self.mol_gto = MolGTO(self.mol)
            self.eval_ao_func = eval_ao
        else:
            self.mol_gto = MolGTO_Spherical(self.mol)
            self.eval_ao_func = eval_ao_spherical
    
        # Detect if mo_coeff is restricted or unrestricted:
        if isinstance(mo_coeff, (list, tuple)):
            # mo_coeff[0] = alpha, mo_coeff[1] = beta
            self.mo_coeff_alpha = mo_coeff[0]
            self.mo_coeff_beta = mo_coeff[1]
            self.unrestricted = True
        else:
            # Single set of coefficients, treat as RHF
            self.mo_coeff_alpha = mo_coeff
            self.mo_coeff_beta = mo_coeff  # identical for spin up/down
            self.unrestricted = False
    
        # Default occupied orbitals (HF reference)
        self.alpha_occ = list(range(self.n_alpha)) 
        self.beta_occ = list(range(self.n_beta))  

        # Apply excitations if specified
        if excitations is not None:
            alpha_exc, beta_exc = excitations
            
            # Handle alpha excitations
            if alpha_exc and len(alpha_exc) == 2:
                from_idx, to_idx = alpha_exc
                if len(from_idx) != len(to_idx):
                    raise ValueError("Number of occupied and virtual orbitals must match for alpha excitations")
                for i, a in zip(from_idx, to_idx):
                    if i not in self.alpha_occ:
                        raise ValueError(f"Cannot remove electron from unoccupied alpha orbital {i}")
                    if a in self.alpha_occ:
                        raise ValueError(f"Cannot add electron to already occupied alpha orbital {a}")
                    self.alpha_occ.remove(i)  
                    self.alpha_occ.append(a)  
                self.alpha_occ.sort()
                
            # Handle beta excitations
            if beta_exc and len(beta_exc) == 2:
                from_idx, to_idx = beta_exc
                if len(from_idx) != len(to_idx):
                    raise ValueError("Number of occupied and virtual orbitals must match for beta excitations")
                for i, a in zip(from_idx, to_idx):
                    if i not in self.beta_occ:
                        raise ValueError(f"Cannot remove electron from unoccupied beta orbital {i}")
                    if a in self.beta_occ:
                        raise ValueError(f"Cannot add electron to already occupied beta orbital {a}")
                    self.beta_occ.remove(i)  
                    self.beta_occ.append(a)  
                self.beta_occ.sort()

        # Store the occupied MO coefficients
        self.mo_coeff_alpha_occ = self.mo_coeff_alpha[:, self.alpha_occ]
        self.mo_coeff_beta_occ = self.mo_coeff_beta[:, self.beta_occ]
    
        # Stored determinant values and coordinates
        self.det_up = None
        self.det_down = None
        self.last_positions = None

    @property
    def n_electrons(self):
        """Return the total number of electrons."""
        return self.n_alpha + self.n_beta

    @partial(jax.jit, static_argnums=(0,))
    def value(self, walker):
        """
        Compute determinant values and update walker.
        Uses full recomputation in JAX.
        """
        positions = walker.positions # (batch, nelec, 3) or (nelec, 3)
        is_batched = positions.ndim == 3
        
        # Evaluate AOs for all electrons
        ao_vals = self.eval_ao_func(self.mol_gto, positions, deriv=0) 
        
        # Split into alpha and beta
        if is_batched:
            ao_alpha = ao_vals[:, :self.n_alpha, :] # (batch, n_alpha, nao)
            ao_beta = ao_vals[:, self.n_alpha:, :]  # (batch, n_beta, nao)
            einsum_str = 'bix,xj->bij'
        else:
            ao_alpha = ao_vals[:self.n_alpha, :] # (n_alpha, nao)
            ao_beta = ao_vals[self.n_alpha:, :]  # (n_beta, nao)
            einsum_str = 'ix,xj->ij'
        
        # Compute Slater matrices
        mo_alpha = jnp.array(self.mo_coeff_alpha_occ)
        mo_beta = jnp.array(self.mo_coeff_beta_occ)
        
        slater_up = jnp.einsum(einsum_str, ao_alpha, mo_alpha)
        slater_down = jnp.einsum(einsum_str, ao_beta, mo_beta)
        
        # Compute determinants and inverses
        sign_up, logdet_up = jnp.linalg.slogdet(slater_up)
        sign_down, logdet_down = jnp.linalg.slogdet(slater_down)
        
        inv_up = jnp.linalg.inv(slater_up)
        inv_down = jnp.linalg.inv(slater_down)
        
        # Total determinant
        det_sign = sign_up * sign_down
        det_logabs = logdet_up + logdet_down
        
        # Update walker
        updated_walker = walker.replace(
            slater_up=slater_up,
            slater_down=slater_down,
            inv_up=inv_up,
            inv_down=inv_down,
            det_up=(sign_up, logdet_up),
            det_down=(sign_down, logdet_down)
        )
        
        return (det_sign, det_logabs), updated_walker

    @partial(jax.jit, static_argnums=(0,))
    def value_and_grad(self, walker):
        """
        Compute determinant values, gradients, and laplacians.
        Uses full recomputation in JAX.
        """
        positions = walker.positions
        is_batched = positions.ndim == 3
        
        # Evaluate AOs, grads, laps
        ao_vals, ao_grad, ao_lap = self.eval_ao_func(self.mol_gto, positions, deriv=2)
        
        # Split alpha/beta
        if is_batched:
            ao_alpha = ao_vals[:, :self.n_alpha, :]
            ao_beta = ao_vals[:, self.n_alpha:, :]
            
            ao_grad_alpha = ao_grad[:, :self.n_alpha, :, :]
            ao_grad_beta = ao_grad[:, self.n_alpha:, :, :]
            
            ao_lap_alpha = ao_lap[:, :self.n_alpha, :]
            ao_lap_beta = ao_lap[:, self.n_alpha:, :]
            
            einsum_str_val = 'bix,xj->bij'
            einsum_str_grad = 'bixd,xj->bijd'
        else:
            ao_alpha = ao_vals[:self.n_alpha, :]
            ao_beta = ao_vals[self.n_alpha:, :]
            
            ao_grad_alpha = ao_grad[:self.n_alpha, :, :]
            ao_grad_beta = ao_grad[self.n_alpha:, :, :]
            
            ao_lap_alpha = ao_lap[:self.n_alpha, :]
            ao_lap_beta = ao_lap[self.n_alpha:, :]
            
            einsum_str_val = 'ix,xj->ij'
            einsum_str_grad = 'ixd,xj->ijd'
        
        mo_alpha = jnp.array(self.mo_coeff_alpha_occ)
        mo_beta = jnp.array(self.mo_coeff_beta_occ)
        
        # Slater matrices
        slater_up = jnp.einsum(einsum_str_val, ao_alpha, mo_alpha)
        slater_down = jnp.einsum(einsum_str_val, ao_beta, mo_beta)
        
        # Gradients
        grad_up = jnp.einsum(einsum_str_grad, ao_grad_alpha, mo_alpha)
        grad_down = jnp.einsum(einsum_str_grad, ao_grad_beta, mo_beta)
        
        # Laplacians
        lap_up = jnp.einsum(einsum_str_val, ao_lap_alpha, mo_alpha)
        lap_down = jnp.einsum(einsum_str_val, ao_lap_beta, mo_beta)
        
        # Determinants and inverses
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
            lap_down=lap_down
        )
        
        return (det_sign, det_logabs), updated_walker

    @partial(jax.jit, static_argnums=(0,))
    def grad(self, walker):
        """
        Compute gradients only.
        """
        positions = walker.positions
        is_batched = positions.ndim == 3
        
        ao_vals, ao_grad = self.eval_ao_func(self.mol_gto, positions, deriv=1)
        
        if is_batched:
            ao_alpha = ao_vals[:, :self.n_alpha, :]
            ao_beta = ao_vals[:, self.n_alpha:, :]
            ao_grad_alpha = ao_grad[:, :self.n_alpha, :, :]
            ao_grad_beta = ao_grad[:, self.n_alpha:, :, :]
            einsum_str_val = 'bix,xj->bij'
            einsum_str_grad = 'bixd,xj->bijd'
        else:
            ao_alpha = ao_vals[:self.n_alpha, :]
            ao_beta = ao_vals[self.n_alpha:, :]
            ao_grad_alpha = ao_grad[:self.n_alpha, :, :]
            ao_grad_beta = ao_grad[self.n_alpha:, :, :]
            einsum_str_val = 'ix,xj->ij'
            einsum_str_grad = 'ixd,xj->ijd'
        
        mo_alpha = jnp.array(self.mo_coeff_alpha_occ)
        mo_beta = jnp.array(self.mo_coeff_beta_occ)
        
        slater_up = jnp.einsum(einsum_str_val, ao_alpha, mo_alpha)
        slater_down = jnp.einsum(einsum_str_val, ao_beta, mo_beta)
        
        grad_up = jnp.einsum(einsum_str_grad, ao_grad_alpha, mo_alpha)
        grad_down = jnp.einsum(einsum_str_grad, ao_grad_beta, mo_beta)
        
        return (slater_up, slater_down, grad_up, grad_down)

    @partial(jax.jit, static_argnums=(0,))
    def laplacian(self, walker):
        """
        Compute laplacians only (and values/grads as needed).
        """
        (det_sign, det_logabs), updated_walker = self.value_and_grad(walker)
        
        return ((updated_walker.slater_up, updated_walker.slater_down),
                (updated_walker.grad_up, updated_walker.grad_down),
                (updated_walker.lap_up, updated_walker.lap_down),
                updated_walker)

    @partial(jax.jit, static_argnums=(0,))
    def matrix(self, coords):
        """
        Compute Slater matrices for given coordinates.
        Args:
            coords: (batch, nelec, 3) or (nelec, 3)
        """
        is_batched = coords.ndim == 3
        ao_vals = self.eval_ao_func(self.mol_gto, coords, deriv=0)
        
        if is_batched:
            ao_alpha = ao_vals[:, :self.n_alpha, :]
            ao_beta = ao_vals[:, self.n_alpha:, :]
            einsum_str = 'bix,xj->bij'
        else:
            ao_alpha = ao_vals[:self.n_alpha, :]
            ao_beta = ao_vals[self.n_alpha:, :]
            einsum_str = 'ix,xj->ij'
        
        mo_alpha = jnp.array(self.mo_coeff_alpha_occ)
        mo_beta = jnp.array(self.mo_coeff_beta_occ)
        
        slater_up = jnp.einsum(einsum_str, ao_alpha, mo_alpha)
        slater_down = jnp.einsum(einsum_str, ao_beta, mo_beta)
        
        return slater_up, slater_down

    def __call__(self, walker, params=None):
        return self.value_and_grad(walker)

# Standalone wrappers for compatibility
def value_and_grad(det, walker):
    return det.value_and_grad(walker)

def grad(det, walker):
    return det.grad(walker)

def laplacian(det, walker):
    return det.laplacian(walker)

def value(det, walker):
    return det.value(walker)
