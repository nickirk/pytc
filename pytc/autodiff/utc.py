"""JAX implementation of Unrestricted Transcorrelated method."""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import scf, dft, ao2mo
from . import kmat as kmat_jax

class UTC:
    """JAX implementation of Unrestricted Transcorrelated method."""
    
    def __init__(self, mf, jastrow_factor, mo_coeff=None, grid_lvl=2):
        """Initialize the UTC object.
        
        Args:
            mf: PySCF unrestricted mean-field object (UHF/UKS)
            jastrow_factor: JAX Jastrow factor instance without parameters
            mo_coeff: Optional MO coefficients (will use mf.mo_coeff if None)
            grid_lvl: Grid level for numerical integration
        """
        # Check that mf is unrestricted  
        # Note: UKS inherits from UHF, so we only need to check for UHF
        if not isinstance(mf, scf.uhf.UHF):
            raise ValueError("UTC requires unrestricted mean-field object (UHF or UKS). "
                           "Use TC class for restricted calculations.")
        
        self.mf = mf
        self.mol = mf.mol
        
        # Extract alpha and beta MO coefficients
        # PySCF stores UHF mo_coeff as array with shape (2, n_ao, n_mo)
        mo = mo_coeff if mo_coeff is not None else mf.mo_coeff
        self.mo_coeff_alpha = mo[0]
        self.mo_coeff_beta = mo[1]
        
        self.n_orb_alpha = self.mo_coeff_alpha.shape[1]
        self.n_orb_beta = self.mo_coeff_beta.shape[1]
        self.verbose = mf.verbose if hasattr(mf, 'verbose') else 0
        self.jastrow_factor = jastrow_factor
        
        # Cache for evaluated quantities (only orbital values, not ERIs)
        self._rho_alpha = None
        self._nabla_rho_alpha = None
        self._rho_beta = None
        self._nabla_rho_beta = None
        
        # Initialize grid (spin-independent)
        self._init_grid(grid_lvl)
        self._eval_basis_on_grid()
    
    def _init_grid(self, grid_lvl=2):
        """Initialize numerical integration grid using PySCF."""
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = grid_lvl
        grids.build()
        
        # Convert to JAX arrays immediately
        self.grid_points = jnp.asarray(grids.coords)
        self.weights = jnp.asarray(grids.weights)
    
    def _eval_basis_on_grid(self):
        """Evaluate basis functions for both spins using PySCF's numint."""
        if (self._rho_alpha is not None and self._nabla_rho_alpha is not None and
            self._rho_beta is not None and self._nabla_rho_beta is not None):
            return
        
        # Use PySCF to evaluate AOs with numpy arrays
        ao = dft.numint.eval_ao(self.mol, jax.device_get(self.grid_points), deriv=1)
        ao_values = ao[0].T  # (N_ao, N_grid)
        ao_gradients = ao[1:4].transpose(2, 1, 0)  # (N_ao, N_grid, 3)
        
        # Transform to MO basis for alpha spin
        mo_values_alpha = np.dot(self.mo_coeff_alpha.T, ao_values)
        mo_gradients_alpha = np.einsum('ji,jnc->inc', self.mo_coeff_alpha, ao_gradients)
        
        # Transform to MO basis for beta spin
        mo_values_beta = np.dot(self.mo_coeff_beta.T, ao_values)
        mo_gradients_beta = np.einsum('ji,jnc->inc', self.mo_coeff_beta, ao_gradients)
        
        # Cache as JAX arrays
        self._rho_alpha = jnp.asarray(mo_values_alpha)
        self._nabla_rho_alpha = jnp.asarray(mo_gradients_alpha)
        self._rho_beta = jnp.asarray(mo_values_beta)
        self._nabla_rho_beta = jnp.asarray(mo_gradients_beta)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_2b_spin_block(self, rho_left, nabla_rho_left, rho_right, nabla_rho_right, jastrow_params):
        """Calculate two-body terms for a specific spin block.
        
        This computes the TC correction for integrals (p_s1 q_s1 | r_s2 s_s2)
        where s1 is the spin for p,q (left) and s2 is the spin for r,s (right).
        
        Args:
            rho_left: Orbital values for left pair (p,q) - shape (n_orb_left, n_grid)
            nabla_rho_left: Orbital gradients for left pair - shape (n_orb_left, n_grid, 3)
            rho_right: Orbital values for right pair (r,s) - shape (n_orb_right, n_grid)
            nabla_rho_right: Orbital gradients for right pair - shape (n_orb_right, n_grid, 3)
            jastrow_params: Parameters for the Jastrow factor
            
        Returns:
            TC correction array of shape (n_orb_left, n_orb_left, n_orb_right, n_orb_right)
        """
        n_orb_left = rho_left.shape[0]
        n_orb_right = rho_right.shape[0]
        
        # Prepare paired quantities for left (pq)
        rho_paired_left = jnp.einsum('in,jn->ijn', rho_left, rho_left).reshape(-1, len(self.weights))
        
        # Prepare paired quantities for right (rs)
        rho_paired_right = jnp.einsum('in,jn->ijn', rho_right, rho_right).reshape(-1, len(self.weights))
        
        # For K1 term: we need ∇_r φ_p(r1) φ_q(r1) and ∇_r u(r1,r2) φ_r(r2) φ_s(r2)
        # The nabla acts on the right pair (r2) for K1
        rho_nabla_rho_paired_right = jnp.einsum('pnd,rn->prnd', nabla_rho_right, rho_right).reshape(-1, len(self.weights), 3)
        
        # Compute K1 term (nabla contribution)
        k_nabla = kmat_jax.calc_K1(
            rho_paired_left, rho_nabla_rho_paired_right,
            self.jastrow_factor, jastrow_params,
            self.grid_points, self.weights
        )
        
        # Compute K3 term (gradient squared contribution)
        k_square = kmat_jax.calc_K3(
            rho_paired_left, self.jastrow_factor, jastrow_params,
            self.grid_points, self.weights
        )
        
        # Reshape results
        k_nabla = k_nabla.reshape(n_orb_left, n_orb_left, n_orb_right, n_orb_right)
        k_square = k_square.reshape(n_orb_left, n_orb_left, n_orb_right, n_orb_right)
        
        # For same-spin blocks, include laplacian term
        # Check if this is a same-spin block by comparing shapes and identity
        # (This is a bit of a hack, but works for our purposes)
        if n_orb_left == n_orb_right:
            # Compute laplacian contribution from both directions
            k_laplacian = -(k_nabla + k_nabla.swapaxes(0, 1))
            result = 0.5 * (k_laplacian + k_square)
        else:
            # For different spin blocks, no laplacian term
            result = 0.5 * k_square
        
        result += k_nabla
        result += result.transpose(2, 3, 0, 1)
        
        return result
    
    @partial(jax.jit, static_argnums=(0,))
    def get_2b(self, jastrow_params, dm1_alpha=None, dm1_beta=None):
        """Calculate two-body terms for all spin blocks.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor
            dm1_alpha: Optional alpha density matrix
            dm1_beta: Optional beta density matrix
            
        Returns:
            Dictionary with keys 'aa', 'ab', 'ba', 'bb' containing TC integrals
        """
        # Get orbital values on grid
        self._eval_basis_on_grid()
        
        # Compute aa block (alpha-alpha)
        tc_aa = self._get_2b_spin_block(
            self._rho_alpha, self._nabla_rho_alpha,
            self._rho_alpha, self._nabla_rho_alpha,
            jastrow_params
        )
        
        # Compute ab block (alpha on left, beta on right)
        tc_ab = self._get_2b_spin_block(
            self._rho_alpha, self._nabla_rho_alpha,
            self._rho_beta, self._nabla_rho_beta,
            jastrow_params
        )
        
        # Compute ba block (beta on left, alpha on right)
        # This is NOT equal to ab due to non-Hermitian nature of TC
        tc_ba = self._get_2b_spin_block(
            self._rho_beta, self._nabla_rho_beta,
            self._rho_alpha, self._nabla_rho_alpha,
            jastrow_params
        )
        
        # Compute bb block (beta-beta)
        tc_bb = self._get_2b_spin_block(
            self._rho_beta, self._nabla_rho_beta,
            self._rho_beta, self._nabla_rho_beta,
            jastrow_params
        )
        
        # Get ERI blocks (computed on the fly, not cached to save VRAM)
        eri_aa, eri_ab, eri_bb = self._get_eri()
        
        # Return dictionary with TC Hamiltonian = ERI - TC_correction
        return {
            'aa': eri_aa - tc_aa,
            'ab': eri_ab - tc_ab,
            'ba': eri_ab - tc_ba,  # Note: standard ERI has ba = ab due to symmetry
            'bb': eri_bb - tc_bb
        }
    
    def _get_eri(self):
        """Compute all ERI blocks (aa, ab, bb) on the fly to save VRAM.
        
        Returns:
            Tuple of (eri_aa, eri_ab, eri_bb) as JAX arrays
        """
        # Alpha-alpha block
        eri_aa = ao2mo.incore.full(self.mf._eri, self.mo_coeff_alpha, compact=False)
        eri_aa = ao2mo.restore(1, eri_aa, self.mo_coeff_alpha.shape[1])
        
        # Alpha-beta block (p_alpha q_alpha | r_beta s_beta)
        eri_ab = ao2mo.general(
            self.mf._eri,
            (self.mo_coeff_alpha, self.mo_coeff_alpha, 
             self.mo_coeff_beta, self.mo_coeff_beta),
            compact=False
        )
        eri_ab = eri_ab.reshape(
            self.n_orb_alpha, self.n_orb_alpha,
            self.n_orb_beta, self.n_orb_beta
        )
        
        # Beta-beta block
        eri_bb = ao2mo.incore.full(self.mf._eri, self.mo_coeff_beta, compact=False)
        eri_bb = ao2mo.restore(1, eri_bb, self.mo_coeff_beta.shape[1])
        
        return jnp.asarray(eri_aa), jnp.asarray(eri_ab), jnp.asarray(eri_bb)
    
    def get_3b(self):
        """Compute all three-body integrals."""
        raise NotImplementedError("JAX implementation pending")
