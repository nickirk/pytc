import numpy as np
from pytcint.tc import TC

class XTC(TC):
    """Extended Transcorrelated class that handles density matrices."""
    
    def __init__(self, mf, mo_coeff=None, grid_lvl=2):
        """Initialize XTC object.
        
        Args:
            mf: PySCF mean-field object
            mo_coeff: Optional molecular orbital coefficients
            grid_lvl: Grid level for numerical integration
        """
        super().__init__(mf, mo_coeff, grid_lvl)
    
    def _get_mf_dm(self):
        """Get mean-field 1-body density matrix for closed shell system.
        
        Returns:
            numpy.ndarray: Diagonal density matrix with 2.0 for occupied orbitals
        """
        nelec = self.mol.nelectron
        nocc = nelec // 2
        dm1 = np.zeros((self.n_orb, self.n_orb))
        np.fill_diagonal(dm1[:nocc, :nocc], 2.0)
        return dm1
    
    def get_1b(self, dm1=None, dm2=None):
        """Compute one-body integrals.
        
        Args:
            dm1: One-body density matrix. If None, uses mean-field density.
            dm2: Two-body density matrix (not used for 1-body terms).
            
        Returns:
            Float: One-body energy contribution.
        """
        if dm1 is None:
            dm1 = self._get_mf_dm()
        raise NotImplementedError("Implementation pending")
    
    def get_2b(self, dm1=None, dm2=None):
        """Compute two-body integrals.
        
        Args:
            dm1: One-body density matrix. If None, uses mean-field density.
            dm2: Two-body density matrix. If None, constructs from dm1.
            
        Returns:
            Float: Two-body energy contribution.
        """
        if dm1 is None:
            dm1 = self._get_mf_dm()
        raise NotImplementedError("Implementation pending")
    
    def get_const(self, dm1=None, dm2=None, delta_U=None, delta_h=None):
        """Compute constant contribution: const = -1/3 * δh^q_p * γ^p_q"""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if delta_h is None:
            delta_h = self._calc_delta_h(delta_U, dm1)
            
        # Calculate const = -1/3 * δh^q_p * γ^p_q
        const = -1/3 * np.einsum('qp,pq->', delta_h, dm1)
        return const

    def _calc_delta_h(self, delta_U=None, dm1=None):
        """Calculate δh using δU and density matrix.
        
        δh^q_p = -1/2 * (δU^{qs}_{pr} - δU^{sq}_{pr}) * γ^r_s
        """
        if delta_U is None:
            delta_U = self._calc_delta_U()  # To be implemented
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Calculate δh using einstein summation
        term1 = np.einsum('qspr,rs->qp', delta_U, dm1)
        term2 = np.einsum('sqpr,rs->qp', delta_U, dm1)
        delta_h = -0.5 * (term1 - term2)
        return delta_h

    def _validate_and_reshape(self, v_vector, rho_paired):
        """Validate inputs and reshape to (Nb, Nb, N_grid, ...) format.
        
        Args:
            v_vector: Array of shape (Nb*Nb, N_grid, 3)
            rho_paired: Array of shape (Nb*Nb, N_grid)
            
        Returns:
            Tuple of reshaped arrays
        """
        Nb = int(np.sqrt(v_vector.shape[0]))
        if Nb * Nb != v_vector.shape[0] or Nb != self.n_orb:
            raise ValueError(f"Invalid v_vector shape: {v_vector.shape}")
        if rho_paired.shape[0] != Nb * Nb:
            raise ValueError(f"Invalid rho_paired shape: {rho_paired.shape}")
            
        v_reshaped = v_vector.reshape(Nb, Nb, -1, 3)
        rho_reshaped = rho_paired.reshape(Nb, Nb, -1)
        return v_reshaped, rho_reshaped
        
    def _calc_delta_U(self, v_vector=None, rho_paired=None, dm1=None):
        """Calculate ΔU^{QS}_{PR} using intermediates.
        
        Args:
            v_vector: Optional array of shape (Nb*Nb, N_grid, 3)
            rho_paired: Optional array of shape (Nb*Nb, N_grid)
            dm1: Optional density matrix of shape (Nb, Nb)
            
        Returns:
            Array of shape (Nb, Nb, Nb, Nb)
        """
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if v_vector is None or rho_paired is None:
            raise NotImplementedError("Auto-calculation of v_vector not yet implemented")
            
        # Reshape inputs
        V, rho = self._validate_and_reshape(v_vector, rho_paired)
        # weight the rho using self.weights
        rho_weighted = rho * self.weights[None, None, :]
        # Calculate intermediates
        W = np.einsum('tuix,ut->ix', V, dm1)  # (N_grid, 3)
        Vbar = np.einsum('ix,srix->sri', W, V)  # (Nb, Nb, N_grid, 3)
        
        X = np.einsum('stix,tu->suix', V, dm1)  # (Nb, N_grid, 3)
        Zbar = np.einsum('urid,suid->sri', V, X)  # (Nb, Nb, N_grid, 3)
        
        Wbar = np.einsum('uti,tu->i', rho_weighted, dm1)  # (N_grid,)
        Y = np.einsum('urix,tu->trix', V, dm1)  # (Nb, Nb, N_grid, 3)
        G = (np.einsum('uri,suix->srix', rho_weighted, X) + 
             np.einsum('trix,sti->srix', Y, rho_weighted))  # (Nb, Nb, N_grid, 3)
        
        # Compute A and B using Zbar instead for A
        A = Vbar - Zbar  # (Nb, Nb, N_grid, 3)
        B = 0.5 * Wbar[None, None, :, None] * V - G  # (Nb, Nb, N_grid, 3)
        
        # Final contraction
        term1 = np.einsum('qpi,sri->qspr', rho_weighted, A)
        term2 = np.einsum('qpix,srix->qspr', V, B)
        
        result = -(term1 + term2)
        # Add permutation P^{PQ}_{SR}
        final = result + result.transpose(2,3,0,1)
        
        return final
