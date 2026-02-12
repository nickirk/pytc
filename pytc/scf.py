
import numpy as np
import scipy.linalg
from pyscf import scf
from pyscf.scf import hf
import jax
import jax.numpy as jnp
from .tc import TC, ISDFTC

class TCSCF(hf.RHF):
    """
    Transcorrelated Self-Consistent Field (TC-SCF) solver.
    Inherits from PySCF's RHF class to leverage standard integral engines.
    Adds TC effective potentials to H_core and V_eff.
    """
    def __init__(self, mol, jastrow_factor, jastrow_params, tc_obj=None):
        super().__init__(mol)
        self.jastrow_factor = jastrow_factor
        self.jastrow_params = jastrow_params
        
        if tc_obj is None:
            # Initialize TC object in AO basis (mo_coeff=Identity)
            # We use a dummy mean-field to initialize TC from pyscf
            # but we need to ensure mo_coeff is Identity.
            # TC.from_pyscf uses mf.mo_coeff.
            
            # Create a dummy MF with identity coeffs
            dummy_mf = scf.RHF(mol)
            dummy_mf.mo_coeff = np.eye(mol.nao_nr())
            dummy_mf.mo_occ = np.zeros(mol.nao_nr()) # Dummy
            
            # We need to ensure grid level is sufficient
            self.tc_obj = TC.from_pyscf(dummy_mf, jastrow_factor, mo_coeff=dummy_mf.mo_coeff, grid_lvl=3)
        else:
            self.tc_obj = tc_obj
            

    def isdf(self, n_rank=None):
        """Enable ISDF approximation."""
        self.tc_obj = ISDFTC.from_tc(self.tc_obj, n_rank=n_rank)
        # Precompute kernels and L_aux
        self.tc_obj = self.tc_obj.isdf(self.jastrow_params)
        return self

    def _get_init_occ(self, n_orb):
        """Get initial occupation numbers."""
        nocc = self.mol.nelectron // 2
        mo_occ = np.zeros(n_orb)
        mo_occ[:nocc] = 2
        return mo_occ

    def get_hcore(self, mol=None):
        """Get core Hamiltonian including 1-body TC corrections."""
        if mol is None: mol = self.mol
        h_core = super().get_hcore(mol)
        
        # Add TC 1-body correction
        # Note: get_1b_fock returns JAX array, convert to numpy
        h_tc = np.array(self.tc_obj.get_1b_fock(self.jastrow_params))
        
        return h_core + h_tc

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        """Get effective potential including 2-body and 3-body TC corrections."""
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        
        # Standard HF potential (Coulomb + Exchange)
        v_hf = super().get_veff(mol, dm, dm_last, vhf_last, hermi)
        
        # Convert DM to JAX
        dm_jax = jnp.array(dm)
        
        v_tc_2b = np.array(self.tc_obj.get_2b_fock(self.jastrow_params, dm_jax))
        v_tc_3b = np.array(self.tc_obj.get_3b_fock(self.jastrow_params, dm_jax))
        
        return v_hf + v_tc_2b + v_tc_3b

    def eig(self, h, s):
        """Solver for generalized eigenvalue problem F C = S C e.
        Overridden to handle non-Hermitian F and orthonormalize orbitals.
        """
        # Use scipy.linalg.eig for non-Hermitian matrices
        # h is F, s is S
        e, c = scipy.linalg.eig(h, s)
        
        # Sort eigenvalues (real part)
        idx = np.argsort(e.real)
        e = e[idx]
        c = c[:, idx]
        
        # Orthonormalize orbitals with respect to S
        # The TC-SCF procedure requires the density to be built from 
        # orthonormalized orbitals (Gram-Schmidt).
        # We use QR decomposition of S^1/2 C to get orthonormal C_orth.
        # C_orth = S^-1/2 Q
        
        # Compute S^1/2
        # S is Hermitian, so we can use eigh
        s_e, s_v = scipy.linalg.eigh(s)
        # Handle small eigenvalues if any (though S should be positive definite)
        s_e = np.maximum(s_e, 1e-15)
        s_half = s_v @ np.diag(np.sqrt(s_e)) @ s_v.T
        s_inv_half = s_v @ np.diag(1.0/np.sqrt(s_e)) @ s_v.T
        
        # Transform C
        c_trans = s_half @ c
        
        q, r = scipy.linalg.qr(c_trans, mode='economic')
        
        # Transform back
        c_orth = s_inv_half @ q
        
        
        return e, c_orth


    def get_grad(self, mo_coeff, mo_occ, fock=None):
        """
        Gradient of the objective function.
        For non-Hermitian/TC-SCF, this might need adjustment, 
        but for now we rely on the default which checks [F, P].
        """
        return super().get_grad(mo_coeff, mo_occ, fock)
