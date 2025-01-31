import numpy as np
from functools import partial, reduce
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pytc.tc import TC
from pytc.lmat import calc_v_vector

einsum = partial(np.einsum, optimize='optimal')
class XTC(TC):
    """Extended Transcorrelated class that handles density matrices."""
    
    def __init__(self, mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
        """Initialize XTC object.
        
        Args:
            mf: PySCF mean-field object
            mo_coeff: Optional molecular orbital coefficients
            grid_lvl: Grid level for numerical integration
        """
        super().__init__(mf, jastrow_factor, mo_coeff, grid_lvl)
        self._delta_U = None  # Cache for delta_U
        self._delta_h = None  # Cache for delta_h
    
    def get_delta_U(self, dm1=None):
        """Get or compute delta_U with caching."""
        if self._delta_U is None:
            if dm1 is None:
                dm1 = self._get_mf_dm()
            
            # Get orbital values on grid
            rho, _ = self._get_intermediates()
            rho_paired = einsum('in,jn->ijn', rho, rho).reshape(-1, rho.shape[1])
            
            # Check if ISDF results are available
            if self._isdf_results is not None:
                # Use ISDF method
                self._delta_U = self._calc_delta_U_isdf(
                    self._isdf_results['C_rho'],
                    self._isdf_results['xi_rho'],
                    self.jastrow_factor,
                    self.grid_points,
                    self.weights
                )
            else:
                # Use original method
                v_vector = calc_v_vector(rho_paired, self.jastrow_factor, 
                                       self.grid_points, self.weights)
                self._delta_U = self._calc_delta_U(v_vector, rho_paired, dm1)
        
        return self._delta_U

    def _get_mf_dm(self):
        """Get mean-field 1-body density matrix for closed shell system.
        
        Returns:
            numpy.ndarray: Diagonal density matrix with 2.0 for occupied orbitals
        """
        #nelec = self.mol.nelectron
        #nocc = nelec // 2
        #dm1 = np.zeros((self.n_orb, self.n_orb))
        #np.fill_diagonal(dm1[:nocc, :nocc], 2.0)
        dm1 = np.diag(self.mf.mo_occ)
        return dm1

    def get_delta_h(self, dm1=None):
        """Get or compute delta_h with caching."""
        if self._delta_h is None:
            if dm1 is None:
                dm1 = self._get_mf_dm()
            delta_U = self.get_delta_U(dm1)
            self._delta_h = self._calc_delta_h(delta_U, dm1)
        return self._delta_h

    def get_1b(self, dm1=None, dm2=None):
        """Get one-body operator h1e = T + V + δh."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Get core Hamiltonian and transform to MO basis
        h1e = self.mf.get_hcore()
        h1e = reduce(np.dot, (self.mo_coeff.T, h1e, self.mo_coeff))
        
        # Add delta_h using cached value
        h1e += self.get_delta_h(dm1)
        
        return h1e

    def get_2b(self, dm1=None, dm2=None):
        """Compute two-body integrals."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        
        # Get TC's two-body contribution
        result = super().get_2b()
        
        # Add cached delta_U
        result += self.get_delta_U(dm1)
        return result
        
    
    def get_const(self, dm1=None, dm2=None):
        """Compute constant contribution: const = -1/3 * δh^q_p * γ^p_q"""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        
        delta_h = self.get_delta_h(dm1)
        const = -2/3 * einsum('qp,pq->', delta_h, dm1)
        const += self.mf.energy_nuc()
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
        term1 = 2*einsum('qpsr,rs->qp', delta_U, dm1)
        term2 = einsum('spqr,rs->qp', delta_U, dm1)
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
        start_time = time.time()
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if v_vector is None or rho_paired is None:
            raise NotImplementedError("Auto-calculation of v_vector not yet implemented")
            
        # Reshape inputs
        V, rho = self._validate_and_reshape(v_vector, rho_paired)
        nb = V.shape[0]
        n_grid = V.shape[2]

        # weight the rho using self.weights
        rho_weighted = rho * self.weights[None, None, :]
        # Calculate intermediates
        W = 2*einsum('utix,tu->ix', V, dm1)  # (N_grid, 3)
        Vbar = einsum('ix,srix->sri', W, V)  # (nb, Nb, N_grid, 3)
        
        X = einsum('stix,tu->suix', V, dm1)  # (Nb, N_grid, 3)
        
        # Compute Zbar using parallel processing with progress tracking
        Zbar = _parallel_over_i(
            lambda i: _process_Zbar_i(i, V, X),
            output_shape=(nb, nb, n_grid),
            n_i=n_grid,
            desc="Computing Zbar"
        )
        
        Wbar = 2*einsum('uti,tu->i', rho_weighted, dm1)  # (N_grid,)
        Y = einsum('urix,tu->trix', V, dm1)  # (Nb, Nb, N_grid, 3)

        # Compute G using parallel processing with progress tracking
        G = _parallel_over_i(
            lambda i: _process_G_i(i, rho_weighted, X, Y),
            output_shape=(nb, nb, n_grid, 3),
            n_i=n_grid,
            desc="Computing G tensor"
        )

        #G = (einsum('uri,suix->srix', rho_weighted, X) + 
        #     einsum('trix,sti->srix', Y, rho_weighted))  # (Nb, Nb, N_grid, 3)
        
        # Compute A and B using Zbar instead for A
        A = Vbar - Zbar  # (Nb, Nb, N_grid, 3)
        B = 0.5 * Wbar[None, None, :, None] * V - G  # (Nb, Nb, N_grid, 3)
        
        # Final contraction
        term1 = einsum('qpi,sri->qpsr', rho_weighted, A)
        term2 = einsum('qpix,srix->qpsr', V, B)
        
        result = -(term1 + term2)
        # Add permutation of two electrons
        final = result + result.transpose(2,3,0,1)
        
        end_time = time.time()
        print(f"Delta U calculation took {end_time - start_time:.2f} seconds")
        return final
    
    def _calc_delta_U_isdf(self, C_rho, xi_rho, jastrow_factor, grid_points, weights, dm1=None, batch_size=None):
        """Calculate ΔU^{QS}_{PR} using ISDF intermediates with batched processing.
        
        Args:
            C_rho: (Nb^2, n_fused) Selected columns for rho
            xi_rho: (n_fused, N_grid) Interpolation coeffs for rho
            jastrow_factor: Jastrow instance for computing gradients
            grid_points: (N_grid, 3) Grid points
            weights: (N_grid,) Grid weights
            batch_size: Optional batch size for r2 coordinate
        """
        from pytc.kmat import _get_safe_batch_size

        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        N_grid = len(grid_points)
        Nb = int(np.sqrt(C_rho.shape[0]))
        N_rank = C_rho.shape[1]
        C_rho = C_rho.reshape(Nb, Nb, N_rank)
        
        if batch_size is None:
            batch_size = _get_safe_batch_size(N_grid, Nb**2)
        
        # Initialize accumulator for M tensor
        M = np.zeros((N_rank, N_rank, N_rank))
        #weighted_xi_rho = xi_rho * weights[None, :]
        #G = np.zeros((N_rank, N_grid, 3))
        
        # Process r2 points in batches
        for i in range(0, N_grid, batch_size):
            i_end = min(i + batch_size, N_grid)
            
            # Get Jastrow gradients for this batch
            u_grad_batch = jastrow_factor.grad(grid_points[i:i_end], grid_points)  # (batch, N_grid,  3)
            G = einsum('j,bj,ijc->bic', weights, xi_rho, u_grad_batch)  # (Nr, N_grid, 3)
        
            K = einsum('bic,dic->bdi', G, G)  # (N_grid, Nr, Nr)
        
            # Compute weighted xi_rho for the chunk
            weighted_xi = xi_rho[:,i:i_end] * weights[None, i:i_end]  # (Nr, chunk_size)
        
            # Contract to get chunk contribution
            M += einsum('ai,bdi->abd', weighted_xi, K)
        
        # Step 3: Form G(b) = C_rho(t,u,b) * dm(t,u)
        Gb = einsum('tub,tu->b', C_rho, dm1)  # (N_rank,)
        
        # Step 4: Form GM(a,d) = G(b) * M(a,b,d)
        GM = einsum('b,abd->ad', Gb, M)  # (N_rank, N_rank)
        
        # Step 5: Form T(p,q,d) = C_rho(p,q,a) * GM(a,d)
        T = einsum('pqa,ad->pqd', C_rho, GM)  # (Nb, Nb, N_rank)
        
        # Step 6: Final contraction for term1
        term1 = 2 * einsum('pqd,rsd->pqrs', T, C_rho)  # (Nb, Nb, Nb, Nb)
        
        # Term 2: -C_rho(p,q,a)C_rho(r,u,b)C_rho(t,s,c)dm(t,u)M(a,b,c)
        # Step 1: L(u,s,c) = dm(t,u)C_rho(t,s,c)
        L = einsum('tu,tsc->usc', dm1, C_rho)  # (Nb, Nb, N_rank)
        
        # Step 2: T(u,s,a,b) = M(a,b,c)L(u,s,c)
        T = einsum('abc,usc->usab', M, L)  # (Nb, Nb, N_rank, N_rank)
        
        # Step 3: Q(r,s,a) = T(u,s,a,b)C_rho(r,u,b)
        Q = einsum('usab,rub->rsa', T, C_rho)  # (Nb, Nb, N_rank)
        
        # Step 4: Final contraction for term2
        term2 = -einsum('pqa,rsa->pqrs', C_rho, Q)  # (Nb, Nb, Nb, Nb)
        
        #term 3
        # T(u,s,a,b) = M(b,a,c)L(u,s,c)
        Mt = M + M.transpose(2,1,0)
        T = einsum('cab,usc->usab', Mt, L)  # (Nb, Nb, N_rank, N_rank)
        # Step 3: Q(r,s,a) = T(u,s,a,b)C_rho(r,u,b)
        Q = einsum('usab,rub->rsa', T, C_rho)  # (Nb, Nb, N_rank)
        term3 = -einsum('pqa,rsa->pqrs', C_rho, Q)  # (Nb, Nb, Nb, Nb)
        # Term 4
        term4 = einsum('b,bac->ac', Gb, M)  # (N_rank, N_rank)
        term4 = einsum('ac,pqa->pqc', term4, C_rho)  # (Nb, Nb, N_rank)
        term4 = einsum('pqc,rsc->pqrs', term4, C_rho)  # (Nb, Nb, Nb, Nb)
        
        # Combine terms
        result = term1 + term2 + term3 + term4
        
        # Add permutation of two electrons
        final = result + result.transpose(2,3,0,1)
        
        return -final

    def make_eris(self):
        from pyscf.cc import rccsd
        mycc = rccsd.RCCSD(self.mf)
        nocc = np.sum(self.mf.mo_occ > 0)
        nmo = mycc.nmo

        eris = rccsd._ChemistsERIs(mycc)
        eris.e_core = self.get_const()
        eris.fock = self.get_1b().copy()
        h2e = self.get_2b()
        eris.fock += 2 * einsum('pqii->pq', h2e[:,:,:nocc,:nocc])
        eris.fock -= einsum('piiq->pq', h2e[:,:nocc,:nocc,:])

        eris.mo_energy = np.diag(eris.fock).copy()
        
        # Slice the full ERI tensor into required blocks
        eris.oooo = h2e[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = h2e[:nocc,nocc:,:nocc,:nocc].copy()
        eris.ooov = h2e[:nocc,:nocc,:nocc,nocc:].copy()
        eris.vooo = h2e[nocc:,:nocc,:nocc,:nocc].copy()
        eris.ovov = h2e[:nocc,nocc:,:nocc,nocc:].copy()
        eris.vovo = h2e[nocc:,:nocc,nocc:,:nocc].copy()
        eris.ovvo = h2e[:nocc,nocc:,nocc:,:nocc].copy()
        eris.voov = h2e[nocc:,:nocc,:nocc,nocc:].copy()
        eris.oovv = h2e[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvv = h2e[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vovv = h2e[nocc:,:nocc,nocc:,nocc:].copy()
        eris.vvov = h2e[nocc:,nocc:,:nocc,nocc:].copy()
        eris.vvvv = h2e[nocc:,nocc:,nocc:,nocc:].copy()

        return eris


def _parallel_compute_G(rho_weighted, X, Y):
    """Compute G tensor using parallel thread processing.
    
    Computes: G[s,r,i,x] = rho[u,r,i] * X[s,u,i,x] + Y[t,r,i,x] * rho[s,t,i]
    """
    # Get dimensions
    S, U_X, I, X_dim = X.shape
    U_rho, R, I_rho = rho_weighted.shape
    assert U_X == U_rho and I == I_rho, "Dimension mismatch"
    
    # Initialize output
    G = np.zeros((S, R, I, X_dim))
    
    def process_i(i):
        # Get views for this i
        X_slice = X[:, :, i, :]  # (S, U, X)
        Y_slice = Y[:, :, i, :]  # (T, R, X)
        rho_slice = rho_weighted[:, :, i]  # (U, R) and (S, T)
        
        # First term: rho[u,r,i] * X[s,u,i,x]
        term1 = np.einsum('ur,sux->srx', rho_slice, X_slice)
        
        # Second term: Y[t,r,i,x] * rho[s,t,i]
        term2 = np.einsum('trx,st->srx', Y_slice, rho_slice)
        
        return i, term1 + term2
    
    # Parallelize over i using threads
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(I):
            futures.append(executor.submit(process_i, i))
        
        for future in futures:
            i, result = future.result()
            G[:, :, i, :] = result
    
    return G

def _parallel_over_i(process_i_func, output_shape, n_i, desc=None):
    """Generic parallel processor for i-index contractions with progress tracking.
    
    Args:
        process_i_func: Function that takes i and returns (i, result)
        output_shape: Shape of output tensor
        n_i: Number of i indices to process
        desc: Optional description for progress bar
        
    Returns:
        Tensor with results assembled
    """
    result = np.zeros(output_shape)
    
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_i_func, i): i 
            for i in range(n_i)
        }
        
        # Process results as they complete with progress bar
        with tqdm(total=n_i, desc=desc or "Processing") as pbar:
            for future in as_completed(futures):
                i, slice_result = future.result()
                if len(output_shape) == 3:
                    result[..., i] = slice_result
                elif len(output_shape) == 4:
                    result[..., i, :] = slice_result
                else:
                    raise ValueError("Invalid output shape")
                pbar.update(1)
    
    return result

def _process_G_i(i, rho_weighted, X, Y):
    """Process i-th slice for G tensor."""
    X_slice = X[:, :, i, :]  # (S, U, X)
    Y_slice = Y[:, :, i, :]  # (T, R, X)
    rho_slice = rho_weighted[:, :, i]  # (U, R) and (S, T)
    
    term1 = np.einsum('ur,sux->srx', rho_slice, X_slice)
    term2 = np.einsum('trx,st->srx', Y_slice, rho_slice)
    
    return i, term1 + term2

def _process_Zbar_i(i, V, X):
    """Process i-th slice for Zbar tensor."""
    V_slice = V[:, :, i, :]  # (U, R, X)
    X_slice = X[:, :, i, :]  # (S, U, X)
    
    result = np.einsum('urx,sux->sr', V_slice, X_slice)
    return i, result
