import numpy as np
from pyscf.dft import numint

class SlaterDet:
    def __init__(self, mol, mo_coeff=None, nelec=None):
        """
        Args:
        mol: A PySCF mol object (provides integrals, eval_gto, etc.)
        mo_coeff: Either a single np.ndarray of shape (nAOs, nMOs) for RHF,
        or a tuple/list [mo_coeff_alpha, mo_coeff_beta] each of
        shape (nAOs, nMOs) for UHF.
        nelec: Number of electrons as a tuple (n_alpha, n_beta).
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
            
        Returns:
            grad_up, grad_down: tuple ((n_up, n_up, 3), (n_down, n_down, 3))
                Each element (i,j,k) represents ∇ᵢϕⱼ(rᵢ) in direction k
        """
        # Get AO values and gradients using numint (returns tuple (value, grad))
        ao_grads = numint.eval_ao(self.mol, coords, deriv=1)[1:].transpose(1, 2, 0)
        
        # Split into up and down electron parts
        ao_grads_up = ao_grads[:self.n_alpha]      # shape (n_up, nAOs, 3)
        ao_grads_down = ao_grads[self.n_alpha:]    # shape (n_down, nAOs, 3)
        
        # Contract with MO coefficients to get gradients of molecular orbitals
        # For each spatial direction, multiply ao_grad by mo_coeff
        grad_up = np.zeros((self.n_alpha, self.n_alpha, 3))
        grad_down = np.zeros((self.n_beta, self.n_beta, 3))
        
        # Handle each spatial direction
        for d in range(3):
            grad_up[..., d] = ao_grads_up[..., d] @ self.mo_coeff_alpha[:, :self.n_alpha]
            grad_down[..., d] = ao_grads_down[..., d] @ self.mo_coeff_beta[:, :self.n_beta]
            
        return grad_up, grad_down
    
    def laplacian(self, coords):
        """Compute the Laplacian of the Slater determinant."""
        # Get AO values, gradients, and laplacians using numint
        ao_vals = numint.eval_ao(self.mol, coords, deriv=2)
        
        # PySCF returns a list where ao_vals[4:] contains the laplacian components
        # Need to reshape and combine the xx, yy, zz components
        ao_lapls = ao_vals[4:].sum(axis=0)  # Sum the diagonal terms
        
        # Split laplacians into up and down electron parts
        ao_lapls_up = ao_lapls[:self.n_alpha]      # shape (n_up, nAOs)
        ao_lapls_down = ao_lapls[self.n_alpha:]    # shape (n_down, nAOs)
        
        # Contract with MO coefficients to get laplacians of molecular orbitals
        # Only use coefficients up to number of electrons
        lapl_up = ao_lapls_up @ self.mo_coeff_alpha[:, :self.n_alpha]    # shape (n_up, n_up)
        lapl_down = ao_lapls_down @ self.mo_coeff_beta[:, :self.n_beta]  # shape (n_down, n_down)
            
        return lapl_up, lapl_down
    
    def matrix(self, coords):
        """
        Build the alpha/beta Slater matrices for all electrons given coords.
    
        Args:
            coords: (n_up + n_down, 3) array of electron coordinates
                    with spin-up electrons in the first n_up rows,
                    spin-down in the last n_down rows.
        Returns:
            slater_up, slater_down: the Slater matrices for alpha and beta spins
        """
        # Get AO values using numint
        ao_vals_all = numint.eval_ao(self.mol, coords, deriv=0)
    
        # Partition the AO values by spin
        ao_up = ao_vals_all[:self.n_alpha]      # shape (n_up, nAOs)
        ao_down = ao_vals_all[self.n_alpha:]    # shape (n_down, nAOs)
    
        # Multiply AO by mo_coeff
        mat_up = ao_up @ self.mo_coeff_alpha     # shape (n_up, nMOs_alpha)
        mat_down = ao_down @ self.mo_coeff_beta  # shape (n_down, nMOs_beta)
    
        # Select only the orbitals up to number of electrons for each spin
        slater_up = mat_up[:, :self.n_alpha]        # shape (n_up, n_up)
        slater_down = mat_down[:, :self.n_beta]  # shape (n_down, n_down)
    
        return slater_up, slater_down
    
    def value(self, coords):
        """
        Full evaluation of the Slater determinant for the entire electron configuration.
    
        Args:
            coords: (n_up + n_down, 3) electron positions
        Returns:
            float: The product of alpha and beta determinants
        """
        slater_up, slater_down = self.matrix(coords)
        det_up = np.linalg.det(slater_up)
        det_down = np.linalg.det(slater_down)
        return det_up * det_down
    
    
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
    
        # Build new row. If alpha spin, use mo_coeff_alpha, else mo_coeff_beta
        mo_coeff_spin = self.mo_coeff_alpha if is_alpha else self.mo_coeff_beta
        n_spin = self.n_alpha if is_alpha else self.n_beta
    
        # Build the row (shape (n_spin,))
        new_row = ao_new @ mo_coeff_spin
        new_row = new_row[:n_spin]
    
        # local index within that spin
        local_idx = e_idx if is_alpha else e_idx - self.n_alpha
    
        # ratio = new_row dot (spin_inv column local_idx)
        ratio = new_row @ spin_inv[:, local_idx]
    
        # Sherman-Morrison update for inverse
        c = spin_inv @ new_row
        factor = 1.0 / c[local_idx]  # same as 1.0 / ratio
        for j in range(n_spin):
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