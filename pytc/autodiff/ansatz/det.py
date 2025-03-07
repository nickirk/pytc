import numpy as np
from functools import partial

from pyscf.dft import numint

einsum = partial(np.einsum, optimize=True)

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
    
        # Internal placeholders for (inverse) Slater matrices
        self.inv_up = None
        self.inv_down = None
    
        # Stored determinant values and coordinates
        self.det_up = None
        self.det_down = None
        self.last_positions = None
    
    def __call__(self, coords):
        """
        Convenience method to call value on a set of coordinates.
    
        Args:
            coords: (n_up + n_down, 3) electron positions
        Returns:
            float: The product of alpha and beta determinants
        """
        return self.value(coords)
    
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

    def grad(self, coords):
        """Compute the gradient of the Slater determinant.
        
        Args:
            coords: (n_up + n_down, 3) electron positions
                  or (n_walkers, n_up + n_down, 3) for batched evaluation
                  
        Returns:
            tuple: (grad_up, grad_down)
                 If input is batched, returns arrays of shape:
                 (n_walkers, n_up, n_up, 3), (n_walkers, n_down, n_down, 3)
        """
        coords_batch, is_single = self._ensure_batch(coords)
        n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
        
        # Reshape to (n_walkers * n_electrons, 3) for eval_ao
        flat_coords = coords_batch.reshape(-1, 3)
        
        # Call eval_ao once for all walkers/electrons
        ao_vals_deriv = numint.eval_ao(self.mol, flat_coords, deriv=1)
        ao_vals = ao_vals_deriv[0]
        ao_grads = ao_vals_deriv[1:].transpose(1, 2, 0)  # Reshape to (n_walkers*n_electrons, nAOs, 3)
        
        
        if not self.unrestricted:
            grad_batch = np.zeros((n_walkers, self.n_electrons, self.mo_coeff_alpha_occ.shape[-1], 3))

            # Vectorized computation for gradients in each direction
            for d in range(3):
                # Batch matrix multiplication using einsum
                # 'wij,jk->wik': w=walker index, i=electron index, j=AO index, k=orbital index
                grad_batch[..., d] = np.dot(ao_grads[..., d], self.mo_coeff_alpha_occ).reshape(n_walkers, self.n_electrons, -1)
            
            grad_up_batch = grad_batch[:, :self.n_alpha]
            grad_down_batch = grad_batch[:, self.n_alpha:]
            
        else:
            # Reshape back to batch form
            ao_vals = ao_vals.reshape(n_walkers, n_electrons, -1)
            ao_grads = ao_grads.reshape(n_walkers, n_electrons, -1, 3)
            # Split by spin
            ao_grads_up = ao_grads[:, :self.n_alpha]      # shape (n_walkers, n_up, nAOs, 3)
            ao_grads_down = ao_grads[:, self.n_alpha:]    # shape (n_walkers, n_down, nAOs, 3)
        
            # Initialize output arrays
            grad_up_batch = np.zeros((n_walkers, self.n_alpha, self.n_alpha, 3))
            grad_down_batch = np.zeros((n_walkers, self.n_beta, self.n_beta, 3))
        
            # Vectorized computation for gradients in each direction
            for d in range(3):
                # Batch matrix multiplication using einsum
                # 'wij,jk->wik': w=walker index, i=electron index, j=AO index, k=orbital index
                grad_up_batch[..., d] = einsum('wij,jk->wik', ao_grads_up[..., d], self.mo_coeff_alpha_occ)
                grad_down_batch[..., d] = einsum('wij,jk->wik', ao_grads_down[..., d], self.mo_coeff_beta_occ)
        
        #grad_up_batch = einsum('wijc,jk->wikc', ao_grads_up, self.mo_coeff_alpha_occ)
        #grad_down_batch = einsum('wijc,jk->wikc', ao_grads_down, self.mo_coeff_beta_occ)
                
        # Return single matrices if input was single walker
        if is_single:
            return grad_up_batch[0], grad_down_batch[0]
        else:
            return grad_up_batch, grad_down_batch
    
    def laplacian(self, coords):
        """Compute the Laplacian of the Slater determinant.
        
        Args:
            coords: (n_up + n_down, 3) electron positions
                  or (n_walkers, n_up + n_down, 3) for batched evaluation
                  
        Returns:
            tuple: (lap_up, lap_down)
                 If input is batched, returns arrays of shape:
                 (n_walkers, n_up, n_up), (n_walkers, n_down, n_down)
        """
        coords_batch, is_single = self._ensure_batch(coords)
        n_walkers, n_electrons = coords_batch.shape[0], coords_batch.shape[1]
        
        # Reshape to (n_walkers * n_electrons, 3) for eval_ao
        flat_coords = coords_batch.reshape(-1, 3)
        
        # Call eval_ao once for all walkers/electrons
        # deriv=2 returns (10, N, nAOs) array with ao values, first, second derivatives
        # val, x, y, z, xx, xy, xz, yy, yz, zz
        ao_vals_deriv = numint.eval_ao(self.mol, flat_coords, deriv=2)
        
        ao_lapls = ao_vals_deriv[[4, 7, 9]].sum(axis=0)  # Sum only diagonal terms for Laplacian
        
        # Reshape back to batch form
        ao_lapls = ao_lapls.reshape(n_walkers, n_electrons, -1)
        
        # Split by spin
        ao_lapls_up = ao_lapls[:, :self.n_alpha].reshape(-1, ao_lapls.shape[-1])      # shape (n_walkers * n_up, nAOs)
        ao_lapls_down = ao_lapls[:, self.n_alpha:].reshape(-1, ao_lapls.shape[-1])    # shape (n_walkers * n_down, nAOs)
        
        # Vectorized matrix multiplication for all walkers
        # 'wij,jk->wik': w=walker index, i=electron index, j=AO index, k=orbital index
        #lap_up_batch = einsum('wij,jk->wik', ao_lapls_up, self.mo_coeff_alpha_occ)
        #lap_down_batch = einsum('wij,jk->wik', ao_lapls_down, self.mo_coeff_beta_occ)
        lap_up_batch = np.dot(ao_lapls_up, self.mo_coeff_alpha_occ).reshape(n_walkers, self.n_alpha, self.n_alpha)
        lap_down_batch = np.dot(ao_lapls_down, self.mo_coeff_beta_occ).reshape(n_walkers, self.n_beta, self.n_beta)
                
        # Return single matrices if input was single walker
        if is_single:
            return lap_up_batch[0], lap_down_batch[0]
        else:
            return lap_up_batch, lap_down_batch
    
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
        
        # Reshape to (n_walkers * n_electrons, 3) for eval_ao
        flat_coords = coords_batch.reshape(-1, 3)
        
        # Call eval_ao once for all walkers/electrons
        ao_vals_all = numint.eval_ao(self.mol, flat_coords, deriv=0)
        
        # Reshape back to (n_walkers, n_electrons, n_aos)
        #ao_vals_all = ao_vals_all.reshape(n_walkers, n_electrons, -1)
        
        
        # Vectorized matrix multiplication for all walkers at once
        # We need to use einsum for batch matrix multiplication
        # 'wij,jk->wik': w=walker index, i=electron index, j=AO index, k=orbital index
        if not self.unrestricted:
            slater_batch = np.dot(ao_vals_all, self.mo_coeff_alpha_occ).reshape(n_walkers, n_electrons, -1)

            slater_up_batch = slater_batch[:,:self.n_alpha]
            slater_down_batch = slater_batch[:, self.n_alpha:]

        else:
            # Split AO values by spin - shape: (n_walkers, n_up, nAOs) and (n_walkers, n_down, nAOs)
            ao_vals_all = ao_vals_all.reshape(n_walkers, n_electrons, -1)
            ao_up = ao_vals_all[:, :self.n_alpha].reshape(-1, ao_vals_all.shape[-1])
            ao_down = ao_vals_all[:, self.n_alpha:].reshape(-1, ao_vals_all.shape[-1])
            slater_up_batch = np.dot(ao_up, self.mo_coeff_alpha_occ).reshape(n_walkers, self.n_alpha, self.n_alpha)
            slater_down_batch = np.dot(ao_down, self.mo_coeff_beta_occ).reshape(n_walkers, self.n_beta, self.n_beta)
        
        # Return single matrices if input was single walker
        if is_single:
            return slater_up_batch[0], slater_down_batch[0]
        else:
            return slater_up_batch, slater_down_batch
    
    def value(self, coords):
        """
        Full evaluation of the Slater determinant for the entire electron configuration.
    
        Args:
            coords: (n_up + n_down, 3) electron positions
                  or (n_walkers, n_up + n_down, 3) for batched evaluation
                  
        Returns:
            float or array: determinant product(s)
                          If input is batched, returns array of shape (n_walkers,)
        """
        coords_batch, is_single = self._ensure_batch(coords)
        
        # Get Slater matrices for all walkers
        slater_up_batch, slater_down_batch = self.matrix(coords_batch)
        
        # Compute determinants
        if is_single:
            det_up = np.linalg.det(slater_up_batch)
            det_down = np.linalg.det(slater_down_batch)
            return det_up * det_down
        else:
            # Compute determinants for each walker
            n_walkers = coords_batch.shape[0]
            
            # NumPy can compute determinants of batched matrices using a list comprehension
            # but we'll avoid loops by using built-in vectorization
            # Use optimized batched determinant calculation
            det_up_batch = np.linalg.det(slater_up_batch)
            det_down_batch = np.linalg.det(slater_down_batch)
            
            # Multiply the determinants
            values = det_up_batch * det_down_batch
            return values
    
    def init_inverse(self, coords):
        """
        Compute and store the inverse Slater matrices for the current coords,
        as well as the determinant for alpha/beta. This is needed for fast updates.
    
        Args:
            coords: (n_up + n_down, 3)
        """
        slater_up, slater_down = self.matrix(coords)
        self.inv_up = np.linalg.inv(slater_up)
        self.inv_down = np.linalg.inv(slater_down)
        self.det_up = np.linalg.det(slater_up)
        self.det_down = np.linalg.det(slater_down)
        self.last_positions = coords.copy()
    
    def update(self, e_idx, new_pos):
        """
        Perform a rank-1 update of the inverse Slater matrix for a single electron move.
    
        Args:
            e_idx: int, index of electron that moved (0..n_up-1 for alpha,
                   n_up..n_up+n_down-1 for beta)
            new_pos: (3,) new coordinates
        Returns:
            ratio: float, ratio of new determinant to old determinant for that spin block
        """
        # Figure out if alpha or beta
        is_alpha = (e_idx < self.n_alpha)
        spin_inv = self.inv_up if is_alpha else self.inv_down
        old_det = self.det_up if is_alpha else self.det_down
    
        # Evaluate AO for the new position
        ao_new = numint.eval_ao(self.mol, new_pos.reshape(1, 3)).squeeze(axis=0)
    
        # Build new row using pre-computed occupied MO coefficients
        mo_coeff_spin_occ = self.mo_coeff_alpha_occ if is_alpha else self.mo_coeff_beta_occ
    
        # Build the row (shape (n_spin,))
        new_row = ao_new @ mo_coeff_spin_occ
    
        # local index within that spin
        local_idx = e_idx if is_alpha else e_idx - self.n_alpha
    
        # ratio = new_row dot (spin_inv column local_idx)
        ratio = new_row @ spin_inv[:, local_idx]
    
        # Sherman-Morrison update for inverse
        c = spin_inv @ new_row
        factor = 1.0 / c[local_idx]  # same as 1.0 / ratio
        for j in range(n_spin := (self.n_alpha if is_alpha else self.n_beta)):
            spin_inv[:, j] -= c * factor * spin_inv[local_idx, j]
    
        # Update the stored determinant
        new_det = old_det * ratio
        if is_alpha:
            self.det_up = new_det
        else:
            self.det_down = new_det
    
        return ratio
    
    def total_value(self):
        """
        Return the product of the current alpha and beta determinants
        from the stored inverses. Valid after init_inverse or partial updates.
    
        Returns:
            float: det_up * det_down
        """
        return self.det_up * self.det_down

    def _is_batched(self, coords):
        """Detect if coordinates are batched.
        
        Args:
            coords: Array of shape (n_electrons, 3) or (n_walkers, n_electrons, 3)
            
        Returns:
            bool: True if coords is batched, False otherwise
        """
        return len(coords.shape) == 3
    
    def _get_batch_dims(self, coords):
        """Get batch dimensions from coordinates.
        
        Args:
            coords: Array of shape (n_electrons, 3) or (n_walkers, n_electrons, 3)
            
        Returns:
            tuple: (n_walkers, n_electrons) or (1, n_electrons)
        """
        if self._is_batched(coords):
            return coords.shape[0], coords.shape[1]
        else:
            return 1, coords.shape[0]
            
    def _ensure_batch(self, coords):
        """Ensure coords are in batched format.
        
        Args:
            coords: Array of shape (n_electrons, 3) or (n_walkers, n_electrons, 3)
            
        Returns:
            tuple: (batched_coords, is_single) where
                   batched_coords has shape (n_walkers, n_electrons, 3)
                   is_single is True if original input was single walker
        """
        if self._is_batched(coords):
            return coords, False
        else:
            return coords[np.newaxis, :, :], True