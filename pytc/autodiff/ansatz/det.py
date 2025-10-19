import numpy as np
from functools import partial
import jax
import jax.numpy as jnp

from pyscf.dft import numint
from pytc.lib import np_helper

einsum = partial(np.einsum, optimize=True)

@partial(jax.jit, static_argnums=(0,))
def value(det, coords, move_mask=None):
    """JAX-compatible wrapper for SlaterDet.value()
    
    Args:
        det: SlaterDet object (static)
        coords: JAX array of shape (n_walkers, n_electrons, 3)
        move_mask: Optional boolean mask of shape (n_walkers, n_electrons)
                  indicating which electrons moved
    Returns:
        JAX array of shape (n_walkers,)
    """
    def value_callback(coords_np, mask_np=None):
        return np.asarray(det.value(np.array(coords_np), mask_np))
    
    result_shape = jax.ShapeDtypeStruct((coords.shape[0],), jnp.float64)
    return jax.pure_callback(value_callback, result_shape, coords, move_mask)

@partial(jax.jit, static_argnums=(0,))
def grad(det, coords):
    """JAX-compatible wrapper for SlaterDet.grad()
    
    Args:
        det: SlaterDet object (static)
        coords: JAX array of shape (n_walkers, n_electrons, 3)
        
    Returns:
        Tuple of JAX arrays for (grad_up, grad_down) with shapes:
        ((n_walkers, n_alpha, n_alpha, 3), (n_walkers, n_beta, n_beta, 3))
    """
    def grad_callback(coords_np):
        matrix_out, grad_out = det.grad(np.array(coords_np))
        return (np.asarray(matrix_out[0]), np.asarray(matrix_out[1]),
                np.asarray(grad_out[0]), np.asarray(grad_out[1]))
    
    matrix_up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha), jnp.float64)
    matrix_down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta), jnp.float64)
    grad_up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha, 3), jnp.float64)
    grad_down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta, 3), jnp.float64)
    
    return jax.pure_callback(grad_callback, 
                           (matrix_up_shape, matrix_down_shape, 
                            grad_up_shape, grad_down_shape), 
                           coords)

@partial(jax.jit, static_argnums=(0,))
def laplacian(det, coords):
    """JAX-compatible wrapper for SlaterDet.laplacian()
    
    Args:
        det: SlaterDet object (static)
        coords: JAX array of shape (n_walkers, n_electrons, 3)
        
    Returns:
        Tuple of JAX arrays for (lap_up, lap_down) with shapes:
        ((n_walkers, n_alpha, n_alpha), (n_walkers, n_beta, n_beta))
    """
    def laplacian_callback(coords_np):
        matrix_out, grad_out, lap_out = det.laplacian(np.array(coords_np))
        return (np.asarray(matrix_out[0]), np.asarray(matrix_out[1]),
                np.asarray(grad_out[0]), np.asarray(grad_out[1]),
                np.asarray(lap_out[0]), np.asarray(lap_out[1]))
    
    matrix_up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha), jnp.float64)
    matrix_down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta), jnp.float64)
    grad_up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha, 3), jnp.float64)
    grad_down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta, 3), jnp.float64)
    lap_up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha), jnp.float64)
    lap_down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta), jnp.float64)
    
    return jax.pure_callback(laplacian_callback,
                           (matrix_up_shape, matrix_down_shape,
                            grad_up_shape, grad_down_shape,
                            lap_up_shape, lap_down_shape),
                           coords)

@partial(jax.jit, static_argnums=(0,))
def matrix(det, coords):
    """JAX-compatible wrapper for SlaterDet.matrix()
    
    Args:
        det: SlaterDet object (static)
        coords: JAX array of shape (n_walkers, n_electrons, 3)
        
    Returns:
        Tuple of JAX arrays for (slater_up, slater_down) with shapes:
        ((n_walkers, n_alpha, n_alpha), (n_walkers, n_beta, n_beta))
    """
    def matrix_callback(coords_np):
        # Call SlaterDet method directly - it handles batching
        slater_up, slater_down = det.matrix(np.array(coords_np))
        return (np.asarray(slater_up), np.asarray(slater_down))
    
    up_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_alpha, det.n_alpha), jnp.float64)
    down_shape = jax.ShapeDtypeStruct((coords.shape[0], det.n_beta, det.n_beta), jnp.float64)
    
    return jax.pure_callback(matrix_callback, (up_shape, down_shape), coords)

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
                    Example: (([0,1], [5,6]), ([], [])) means:
                    - For alpha: remove electrons from orbitals 0,1 and add to orbitals 5,6
                    - For beta: no excitations (regular HF reference)
        """
        self.mol = mol
        if nelec is None:
            self.n_alpha, self.n_beta = mol.nelec 
        else:
            self.n_alpha, self.n_beta = nelec
    
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
                # Validate excitation indices
                if len(from_idx) != len(to_idx):
                    raise ValueError("Number of occupied and virtual orbitals must match for alpha excitations")
                
                # Apply excitations
                for i, a in zip(from_idx, to_idx):
                    if i not in self.alpha_occ:
                        raise ValueError(f"Cannot remove electron from unoccupied alpha orbital {i}")
                    if a in self.alpha_occ:
                        raise ValueError(f"Cannot add electron to already occupied alpha orbital {a}")
                    self.alpha_occ.remove(i)  
                    self.alpha_occ.append(a)  
                self.alpha_occ.sort()  # Keep indices sorted
                
            # Handle beta excitations
            if beta_exc and len(beta_exc) == 2:
                from_idx, to_idx = beta_exc
                # Validate excitation indices
                if len(from_idx) != len(to_idx):
                    raise ValueError("Number of occupied and virtual orbitals must match for beta excitations")
                
                # Apply excitations
                for i, a in zip(from_idx, to_idx):
                    if i not in self.beta_occ:
                        raise ValueError(f"Cannot remove electron from unoccupied beta orbital {i}")
                    if a in self.beta_occ:
                        raise ValueError(f"Cannot add electron to already occupied beta orbital {a}")
                    self.beta_occ.remove(i)  
                    self.beta_occ.append(a)  
                self.beta_occ.sort()  # Keep indices sorted

        # Store the occupied MO coefficients
        self.mo_coeff_alpha_occ = self.mo_coeff_alpha[:, self.alpha_occ]
        self.mo_coeff_beta_occ = self.mo_coeff_beta[:, self.beta_occ]
    
        # Stored determinant values and coordinates
        self.det_up = None
        self.det_down = None
        self.last_positions = None

    def __call__(self, coords, params=None, move_mask=None):
        """
        Convenience method to call value on a set of coordinates.
    
        Args:
            coords: (n_up + n_down, 3) electron positions
        Returns:
            float: The product of alpha and beta determinants
        """
        return self.value(coords, move_mask=move_mask)
    
    @property
    def n_electrons(self):
        """Return the total number of electrons."""
        return self.n_alpha + self.n_beta

    def value_and_grad(self, coords):
        """Compute both value and gradient of Slater determinant.
        
        Args:
            coords: (n_up + n_down, 3) electron positions
            
        Returns:
            tuple: (value, grad) where
                  value is the determinant value (float)
                  grad will be implemented later
        """
        # For now, just return value and None for gradient
        return self.value(coords), None

    def ao2mo(self, ao_vals, mo_coeff_alpha, mo_coeff_beta, n_walkers, n_electrons):
        """Transform AO values to MO basis for both spins.
        
        Args:
            ao_vals: Array of AO values
            mo_coeff_alpha/beta: MO coefficients for each spin
            n_walkers: Number of walkers
            n_electrons: Total number of electrons
            
        Returns:
            Tuple of (alpha_vals, beta_vals) in MO basis
        """
        if not self.unrestricted:
            mo_vals = np.dot(ao_vals, mo_coeff_alpha).reshape(n_walkers, n_electrons, -1)
            return (mo_vals[:, :self.n_alpha, :self.n_alpha], 
                    mo_vals[:, self.n_alpha:, :self.n_beta])
        else:
            ao_up = ao_vals.reshape(n_walkers, n_electrons, -1)[:, :self.n_alpha].reshape(-1, ao_vals.shape[-1])
            ao_down = ao_vals.reshape(n_walkers, n_electrons, -1)[:, self.n_alpha:].reshape(-1, ao_vals.shape[-1])
            return (np.dot(ao_up, mo_coeff_alpha).reshape(n_walkers, self.n_alpha, self.n_alpha),
                    np.dot(ao_down, mo_coeff_beta).reshape(n_walkers, self.n_beta, self.n_beta))

    def matrix(self, coords):
        """
        Build the alpha/beta Slater matrices for all electrons given coords.
    
        Args:
            coords: (n_up + n_down, 3) array of electron coordinates
                   or (n_walkers, n_up + n_down, 3) for batched evaluation
                   
        Returns:
            slater_up, slater_down: the Slater matrices for alpha and beta spins
                                  If input is batched, returns arrays of shape:
                                  (n_walkers, n_up, n_up), (n_walkers, n_down, n_down)
        """
        coords_batch, is_single = self._ensure_batch(coords)
        n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
        
        flat_coords = coords_batch.reshape(-1, 3)
        ao_vals_all = numint.eval_ao(self.mol, flat_coords, deriv=0)
        
        # Direct computation without caching
        if not self.unrestricted:
            slater_batch = np.dot(ao_vals_all, self.mo_coeff_alpha_occ).reshape(n_walkers, n_electrons, -1)
            up = slater_batch[:,:self.n_alpha, :self.n_alpha]
            down = slater_batch[:, self.n_alpha:, :self.n_beta]
        else:
            ao_vals_all = ao_vals_all.reshape(n_walkers, n_electrons, -1)
            ao_up = ao_vals_all[:, :self.n_alpha].reshape(-1, ao_vals_all.shape[-1])
            ao_down = ao_vals_all[:, self.n_alpha:].reshape(-1, ao_vals_all.shape[-1])
            up = np.dot(ao_up, self.mo_coeff_alpha_occ).reshape(n_walkers, self.n_alpha, self.n_alpha)
            down = np.dot(ao_down, self.mo_coeff_beta_occ).reshape(n_walkers, self.n_beta, self.n_beta)
            
        return up, down

    def grad(self, coords):
        """Compute the gradient and matrix of the Slater determinant.

        Args:
            coords: (n_up + n_down, 3) electron positions
                  or (n_walkers, n_up + n_down, 3) for batched evaluation
                  
        Returns:
            tuple: ((matrix_up, matrix_down), (grad_up, grad_down))
                 If input is batched:
                 matrix_up/down: (n_walkers, n_up/down, n_up/down)
                 grad_up/down: (n_walkers, n_up/down, n_up/down, 3)
        """
        coords_batch, is_single = self._ensure_batch(coords)
        n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
        
        flat_coords = coords_batch.reshape(-1, 3)
        ao_vals_deriv = numint.eval_ao(self.mol, flat_coords, deriv=1)
        
        # Direct computation
        matrix_up, matrix_down = self.ao2mo(ao_vals_deriv[0],
                                          self.mo_coeff_alpha_occ,
                                          self.mo_coeff_beta_occ,
                                          n_walkers, n_electrons)
        
        grad_up = np.zeros((n_walkers, self.n_alpha, self.n_alpha, 3))
        grad_down = np.zeros((n_walkers, self.n_beta, self.n_beta, 3))
        
        ao_grads = ao_vals_deriv[1:].transpose(1, 2, 0)
        for d in range(3):
            gup, gdown = self.ao2mo(ao_grads[..., d],
                                  self.mo_coeff_alpha_occ,
                                  self.mo_coeff_beta_occ,
                                  n_walkers, n_electrons)
            grad_up[..., d] = gup
            grad_down[..., d] = gdown
            
        return (matrix_up, matrix_down), (grad_up, grad_down)

    def laplacian(self, coords):
        """Compute laplacian, gradient and matrix of the Slater determinant.
                
        Args:
            coords: (n_up + n_down, 3) electron positions
                  or (n_walkers, n_up + n_down, 3) for batched evaluation
                  
        Returns:
            tuple: ((matrix_up, matrix_down), 
                    (grad_up, grad_down),
                    (lap_up, lap_down))
        """
        coords_batch, is_single = self._ensure_batch(coords)
        n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
        
        # Direct computation without caching
        flat_coords = coords_batch.reshape(-1, 3)
        ao_vals_deriv = numint.eval_ao(self.mol, flat_coords, deriv=2)
        
        # Get matrices
        matrix_up, matrix_down = self.ao2mo(ao_vals_deriv[0],
                                          self.mo_coeff_alpha_occ,
                                          self.mo_coeff_beta_occ,
                                          n_walkers, n_electrons)
        
        # Get gradients
        ao_grads = ao_vals_deriv[1:4].transpose(1, 2, 0)
        grad_up = np.zeros((n_walkers, self.n_alpha, self.n_alpha, 3))
        grad_down = np.zeros((n_walkers, self.n_beta, self.n_beta, 3))
        
        for d in range(3):
            gup_d, gdown_d = self.ao2mo(ao_grads[..., d],
                                  self.mo_coeff_alpha_occ,
                                  self.mo_coeff_beta_occ,
                                  n_walkers, n_electrons)
            grad_up[..., d] = gup_d
            grad_down[..., d] = gdown_d
            
        # Get laplacians
        ao_lapls = ao_vals_deriv[[4, 7, 9]].sum(axis=0)
        lap_up, lap_down = self.ao2mo(ao_lapls,
                                    self.mo_coeff_alpha_occ,
                                    self.mo_coeff_beta_occ,
                                    n_walkers, n_electrons)
        
        return (matrix_up, matrix_down), (grad_up, grad_down), (lap_up, lap_down)


    def value(self, coords, move_mask=None):
        """Compute determinant value (pure function)"""
        coords_batch, is_single = self._ensure_batch(coords)
        
        # Always compute full determinant
        slater_up, slater_down = self.matrix(coords_batch)
        det_up = np_helper.batched_det(slater_up)
        det_down = np_helper.batched_det(slater_down)
        values = det_up * det_down
        
        return float(values[0]) if is_single else values

    def _ensure_batch(self, coords):
        """Ensure coords are in batched format.
    
        Args:
            coords: Array of shape (n_electrons, 3) or (n_walkers, n_electrons, 3)
        
        Returns:
            tuple: (batched_coords, is_single) where
                   batched_coords has shape (n_walkers, n_electrons, 3)
                   is_single is True if original input was single walker
        """
        if len(coords.shape) == 3:
            return coords, False
        else:
            return coords[np.newaxis, :, :], True