"""
Systematic Improvable Embedding (SIE)

A quantum embedding framework that combines:
- Intrinsic Atomic Orbitals (IAO) for localization
- Local Natural Orbitals (LNO) from MP2 for compression
- Fragment CCSD for high accuracy

This implementation follows the methodology from embedding.py,
adapted into a class-based interface.

Author: Ke Liao
Date: October 27, 2025
"""

import numpy as np
from pyscf import gto, scf, lo, ao2mo, cc
import logging

logger = logging.getLogger(__name__)


class SIE:
    """
    Systematic Improvable Embedding (SIE) class.
    
    This class implements a quantum embedding scheme that:
    1. Constructs Intrinsic Atomic Orbitals (IAO) from RHF orbitals
    2. Defines local active spaces for each fragment
    3. Compresses external spaces using MP2-LNO
    4. Solves fragment CCSD equations
    5. Computes fragment correlation energies
    
    Attributes
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    iao_coeff : np.ndarray
        IAO coefficients in AO basis
    iao_labels : list
        Labels for each IAO
    local_spaces : list
        Local active space definitions
    local_spaces_lno : list
        Local spaces with LNO compression
    ccsd_results : dict
        Fragment CCSD results
    energy_results : dict
        Fragment energy results
    """
    
    def __init__(self, mol, mf):
        """
        Initialize SIE with molecule and mean-field object.
        
        Parameters
        ----------
        mol : pyscf.gto.Mole
            Molecule object
        mf : pyscf.scf.RHF
            Converged RHF mean-field object
        """
        self.mol = mol
        self.mf = mf
        
        # IAO-related attributes
        self.iao_coeff = None
        self.iao_labels = None
        
        # Local space attributes
        self.local_spaces = None
        self.local_spaces_lno = None
        
        # Results
        self.ccsd_results = None
        self.energy_results = None
        
        logger.info("SIE object initialized")
        logger.info(f"  Molecule: {mol.natm} atoms, {mol.nelectron} electrons")
        logger.info(f"  Basis: {mol.basis}")
        logger.info(f"  RHF energy: {mf.e_tot:.8f} a.u.")
    
    def construct_iaos(self, minao='minao'):
        """
        Construct Intrinsic Atomic Orbitals (IAO).
        
        IAOs are localized orbitals constructed from occupied MOs that
        resemble atomic orbitals. They are orthogonalized using Löwdin method.
        
        Parameters
        ----------
        minao : str, optional
            Minimal basis set for IAO construction (default: 'minao')
        
        Returns
        -------
        iao_coeff : np.ndarray
            IAO coefficients in AO basis (nao, niao)
        iao_labels : list
            Labels for each IAO
        """
        logger.info("\n" + "="*60)
        logger.info("Step 3: Constructing Intrinsic Atomic Orbitals (IAO)")
        logger.info("="*60)
        
        # Construct IAOs using PySCF's lo.iao module
        # IAOs are constructed from occupied orbitals
        nocc = self.mol.nelectron // 2
        mo_coeff_occ = self.mf.mo_coeff[:, :nocc]
        
        logger.info(f"\nIAO construction parameters:")
        logger.info(f"  Minimal basis: {minao}")
        logger.info(f"  Number of occupied MOs: {nocc}")
        
        # Build IAOs - these span the occupied space (non-orthogonal)
        iao_coeff_raw = lo.iao.iao(self.mol, mo_coeff_occ, minao=minao)
        
        niao = iao_coeff_raw.shape[1]
        
        # Orthogonalize IAOs using Löwdin orthogonalization
        S = self.mol.intor('int1e_ovlp')
        iao_overlap = iao_coeff_raw.T @ S @ iao_coeff_raw
        
        from pyscf.lo import orth
        lowdin_transform = orth.lowdin(iao_overlap)
        iao_coeff = iao_coeff_raw @ lowdin_transform
        
        # Verify orthonormality
        S_iao_orth = iao_coeff.T @ S @ iao_coeff
        max_off_diag = np.max(np.abs(S_iao_orth - np.diag(np.diag(S_iao_orth))))
        logger.info(f"  IAO orthogonalization: max off-diagonal = {max_off_diag:.2e}")
        
        # Get IAO labels from reference molecule
        pmol = lo.iao.reference_mol(self.mol, minao=minao)
        iao_labels_raw = pmol.ao_labels()
        
        if len(iao_labels_raw) != niao:
            raise ValueError(f"Mismatch: {niao} IAOs but {len(iao_labels_raw)} labels")
        
        # Parse labels to extract atom index, symbol, and orbital type
        iao_labels = []
        for i, label in enumerate(iao_labels_raw):
            parts = label.split()
            if len(parts) >= 3:
                atom_idx = int(parts[0])
                atom_symbol = parts[1]
                orbital_type = parts[2]
                iao_labels.append(f"{atom_idx} {atom_symbol} {orbital_type}")
            else:
                logger.warning(f"Unexpected label format for IAO {i}: {label}")
                iao_labels.append(label)
        
        logger.info(f"\nIAO construction completed:")
        logger.info(f"  Number of IAOs: {niao}")
        logger.info(f"  IAO dimensions: {iao_coeff.shape}")
        logger.info(f"  IAOs are orthonormalized using Löwdin method")
        
        # Store as instance attributes
        self.iao_coeff = iao_coeff
        self.iao_labels = iao_labels
        
        return iao_coeff, iao_labels
    
    def define_local_spaces(self, fragments, svd_threshold=1e-6):
        """
        Define local active spaces for each fragment using SVD decomposition in MO space.
        
        For each fragment F:
        1. Transform IAOs to MO basis: C_IAO_MO = C_MO^T @ S @ C_IAO
        2. Extract U_occ(F) (N_occ × nLO(F)) and U_virt(F) (N_virt × nLO(F))
        3. Perform SVD: U = W @ diag(s) @ Vh
        4. W columns with s > threshold define internal space (in MO basis)
        5. Remaining W columns (full_matrices=True) define external space
        
        Parameters
        ----------
        fragments : list of list
            List of atom indices for each fragment
            Example: [[0, 1], [2, 3], ...] for pairs of atoms
        svd_threshold : float, optional
            Threshold for singular values in SVD (default: 1e-6)
        
        Returns
        -------
        local_spaces : list of dict
            Local active space definitions for each fragment containing:
            - 'fragment_id': Fragment index
            - 'atoms': List of atom indices
            - 'iao_indices': IAO indices belonging to fragment
            - 'W_occ': Internal occupied orbitals in MO basis (nocc, n_int_occ)
            - 'W_tilde_occ': External occupied orbitals in MO basis (nocc, n_ext_occ)
            - 'W_virt': Internal virtual orbitals in MO basis (nvirt, n_int_virt)
            - 'W_tilde_virt': External virtual orbitals in MO basis (nvirt, n_ext_virt)
            - 'n_int_occ', 'n_ext_occ', 'n_int_virt', 'n_ext_virt': Dimensions
        """
        logger.info("\n" + "="*60)
        logger.info("Step 4: Constructing Local Active Space for Each Fragment")
        logger.info("="*60)
        
        if self.iao_coeff is None or self.iao_labels is None:
            raise ValueError("IAOs not constructed. Call construct_iaos() first.")
        
        nocc = self.mol.nelectron // 2
        nao = self.iao_coeff.shape[0]
        niao = self.iao_coeff.shape[1]
        nvirt = nao - nocc
        
        logger.info(f"\nActive space construction parameters:")
        logger.info(f"  Number of fragments: {len(fragments)}")
        logger.info(f"  SVD threshold: {svd_threshold}")
        logger.info(f"  Number of occupied MOs: {nocc}")
        logger.info(f"  Number of virtual MOs: {nvirt}")
        logger.info(f"  Total AO basis functions: {nao}")
        logger.info(f"  Total IAOs: {niao}")
        
        # Transform IAO coefficients to MO basis
        S_ao = self.mf.get_ovlp()
        C_mo = self.mf.mo_coeff
        C_iao_mo = C_mo.T @ S_ao @ self.iao_coeff  # (nmo, niao)
        
        # Split into occupied and virtual blocks
        C_iao_mo_occ = C_iao_mo[:nocc, :]  # (nocc, niao)
        C_iao_mo_virt = C_iao_mo[nocc:, :]  # (nvirt, niao)
        
        logger.info(f"\nIAO in MO basis:")
        logger.info(f"  C_IAO_MO (occupied): {C_iao_mo_occ.shape}")
        logger.info(f"  C_IAO_MO (virtual): {C_iao_mo_virt.shape}")
        
        local_spaces = []
        
        for frag_idx, frag_atoms in enumerate(fragments):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fragment {frag_idx}: Atoms {frag_atoms}")
            logger.info(f"{'='*60}")
            
            # Get IAO indices for this fragment
            frag_iao_indices = []
            for iatom in frag_atoms:
                for i, label in enumerate(self.iao_labels):
                    parts = label.split()
                    if len(parts) >= 3:
                        label_atom_idx = int(parts[0])
                        if label_atom_idx == iatom:
                            frag_iao_indices.append(i)
            
            nlo_F = len(frag_iao_indices)
            logger.info(f"  Fragment IAO indices: {frag_iao_indices}")
            logger.info(f"  Number of fragment IAOs (nLO(F)): {nlo_F}")
            
            if nlo_F == 0:
                logger.warning(f"  No IAOs found for fragment {frag_idx}, skipping")
                continue
            
            # Extract projection matrices U_occ(F) and U_virt(F)
            U_occ_F = C_iao_mo_occ[:, frag_iao_indices]  # (nocc, nlo_F)
            U_virt_F = C_iao_mo_virt[:, frag_iao_indices]  # (nvirt, nlo_F)
            
            logger.info(f"\nProjection matrices:")
            logger.info(f"  U_occ(F) shape: {U_occ_F.shape} - (N_occ × nlo(F))")
            logger.info(f"  U_virt(F) shape: {U_virt_F.shape} - (N_virt × nlo(F))")
            
            # Perform SVD on occupied space
            # U_occ(F) = W_occ @ diag(s_occ) @ Vh_occ
            # W_occ: (nocc, nocc) - complete orthonormal basis in occupied space
            # s_occ: singular values (length = min(nocc, nlo_F) = nlo_F typically)
            # Vh_occ: (nlo_F, nlo_F)
            if U_occ_F.shape[1] > 0:
                W_occ_full, s_occ, Vh_occ = np.linalg.svd(U_occ_F, full_matrices=True)
                
                logger.info(f"\nOccupied space SVD:")
                logger.info(f"  W_occ_full shape: {W_occ_full.shape} - (N_occ, N_occ)")
                logger.info(f"  Singular values (n={len(s_occ)}): {s_occ}")
                
                # Internal: singular values > threshold
                internal_occ_mask = s_occ > svd_threshold
                n_int_occ = np.sum(internal_occ_mask)
                n_ext_occ = nocc - n_int_occ
                
                # W_occ_MO: internal occupied in MO space (nocc, n_int_occ)
                # W_tilde_occ_MO: external occupied in MO space (nocc, n_ext_occ)
                W_occ_MO = W_occ_full[:, :len(s_occ)][:, internal_occ_mask]
                W_tilde_occ_MO = W_occ_full[:, len(s_occ):]  # All columns after the fragment IAO columns
                
                # If some fragment IAO projections have small singular values, include them in external
                if np.any(~internal_occ_mask):
                    W_tilde_occ_MO = np.hstack([
                        W_occ_full[:, :len(s_occ)][:, ~internal_occ_mask],
                        W_occ_full[:, len(s_occ):]
                    ])
                
                logger.info(f"\n  Occupied space classification:")
                logger.info(f"  W_occ_MO shape: {W_occ_MO.shape}")
                logger.info(f"  W_tilde_occ_MO shape: {W_tilde_occ_MO.shape}")
                logger.info(f"  Internal occupied: {n_int_occ} orbitals (s > {svd_threshold})")
                logger.info(f"  External occupied: {n_ext_occ} orbitals")
            else:
                n_int_occ = 0
                n_ext_occ = nocc
                W_occ_MO = np.zeros((nocc, 0))
                W_tilde_occ_MO = np.eye(nocc)
                logger.info(f"\n  No occupied IAOs - all {nocc} occupied are external")
            
            # Same for virtual space
            if U_virt_F.shape[1] > 0:
                W_virt_full, s_virt, Vh_virt = np.linalg.svd(U_virt_F, full_matrices=True)
                
                logger.info(f"\nVirtual space SVD:")
                logger.info(f"  W_virt_full shape: {W_virt_full.shape} - (N_virt, N_virt)")
                logger.info(f"  Singular values (n={len(s_virt)}): {s_virt}")
                
                internal_virt_mask = s_virt > svd_threshold
                n_int_virt = np.sum(internal_virt_mask)
                n_ext_virt = nvirt - n_int_virt
                
                W_virt_MO = W_virt_full[:, :len(s_virt)][:, internal_virt_mask]
                W_tilde_virt_MO = W_virt_full[:, len(s_virt):]
                
                if np.any(~internal_virt_mask):
                    W_tilde_virt_MO = np.hstack([
                        W_virt_full[:, :len(s_virt)][:, ~internal_virt_mask],
                        W_virt_full[:, len(s_virt):]
                    ])
                
                logger.info(f"\n  Virtual space classification:")
                logger.info(f"  W_virt_MO shape: {W_virt_MO.shape}")
                logger.info(f"  W_tilde_virt_MO shape: {W_tilde_virt_MO.shape}")
                logger.info(f"  Internal virtual: {n_int_virt} orbitals (s > {svd_threshold})")
                logger.info(f"  External virtual: {n_ext_virt} orbitals")
            else:
                n_int_virt = 0
                n_ext_virt = nvirt
                W_virt_MO = np.zeros((nvirt, 0))
                W_tilde_virt_MO = np.eye(nvirt)
                logger.info(f"\n  No virtual IAOs - all {nvirt} virtuals are external")
            
            logger.info(f"\nActive space summary:")
            logger.info(f"  Internal: {n_int_occ} occ + {n_int_virt} virt = {n_int_occ + n_int_virt}")
            logger.info(f"  External: {n_ext_occ} occ + {n_ext_virt} virt = {n_ext_occ + n_ext_virt}")
            logger.info(f"  Total: {nocc} occ + {nvirt} virt = {nocc + nvirt}")
            
            # Store local space information
            local_space = {
                'fragment_id': frag_idx,
                'atoms': frag_atoms,
                'iao_indices': frag_iao_indices,
                'W_occ': W_occ_MO,
                'W_tilde_occ': W_tilde_occ_MO,
                'W_virt': W_virt_MO,
                'W_tilde_virt': W_tilde_virt_MO,
                'n_int_occ': n_int_occ,
                'n_ext_occ': n_ext_occ,
                'n_int_virt': n_int_virt,
                'n_ext_virt': n_ext_virt,
            }
            
            local_spaces.append(local_space)
        
        logger.info("\n" + "="*60)
        logger.info("Local Active Space Construction Summary")
        logger.info("="*60)
        for ls in local_spaces:
            logger.info(f"\nFragment {ls['fragment_id']}:")
            logger.info(f"  Atoms: {ls['atoms']}")
            logger.info(f"  Active space size: {ls['n_int_occ'] + ls['n_int_virt']} orbitals")
            logger.info(f"    ({ls['n_int_occ']} occ + {ls['n_int_virt']} virt)")
        logger.info("="*60 + "\n")
        
        self.local_spaces = local_spaces
        return local_spaces
    
    def _compute_lno_compression(self, density_matrix, eta_threshold, space_name=""):
        """
        Compress a space using Local Natural Orbital (LNO) analysis.
        
        Helper method that diagonalizes a density matrix, sorts eigenvalues,
        and selects LNOs based on a threshold criterion.
        
        Parameters
        ----------
        density_matrix : ndarray
            Density matrix in the external space (n_ext, n_ext)
        eta_threshold : float
            Threshold for selecting LNOs based on eigenvalue magnitude
        space_name : str, optional
            Name for logging (e.g., "occupied" or "virtual")
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'X': Transformation matrix (n_ext, n_ext), sorted by eigenvalue
            - 'Lambda': Eigenvalues array, sorted in descending order
            - 'lno_indices': Indices of selected LNOs passing threshold
            - 'n_lno': Number of selected LNOs
        """
        # Diagonalize density matrix
        Lambda, X = np.linalg.eigh(density_matrix)
        
        # Sort by descending absolute eigenvalue
        idx = np.argsort(-np.abs(Lambda))
        Lambda = Lambda[idx]
        X = X[:, idx]
        
        logger.info(f"    {space_name.capitalize()} eigenvalues shape: {Lambda.shape}")
        logger.info(f"    {space_name.capitalize()} eigenvalues (top 5): {Lambda[:min(5, len(Lambda))]}")
        
        # Select LNOs based on threshold
        lno_indices = np.where(np.abs(Lambda) >= eta_threshold)[0]
        n_lno = len(lno_indices)
        
        return {
            'X': X,
            'Lambda': Lambda,
            'lno_indices': lno_indices,
            'n_lno': n_lno
        }
    
    def _create_empty_lno_result(self, n_ext):
        """
        Create identity/empty LNO result when space is empty or should not be compressed.
        
        Parameters
        ----------
        n_ext : int
            Dimension of the external space
        
        Returns
        -------
        dict
            Dictionary with empty/identity LNO results
        """
        return {
            'X': np.eye(n_ext) if n_ext > 0 else np.zeros((0, 0)),
            'Lambda': np.array([]),
            'lno_indices': np.array([], dtype=int),
            'n_lno': 0
        }
    
    def compute_mp2_lno_compression(self, eta_occ=1e-6, eta_virt=1e-6):
        """
        Compress external spaces using MP2 Local Natural Orbitals (LNOs).
        
        This method implements MP2-LNO compression as described in the paper:
        1. Compute MP1 amplitudes using semi-canonical orbitals
        2. Build MP2 density matrices in external space (occupied and virtual)
        3. Diagonalize density matrices to obtain LNOs
        4. Select active LNOs based on eigenvalue thresholds
        
        The function handles occupied and virtual spaces independently. If one external
        space is empty (n_ext_occ=0 or n_ext_virt=0), compression is still performed
        on the non-empty space.
        
        Parameters
        ----------
        eta_occ : float, optional
            Threshold for occupied LNO eigenvalues (default: 1e-6)
        eta_virt : float, optional
            Threshold for virtual LNO eigenvalues (default: 1e-6)
        
        Returns
        -------
        local_spaces_lno : list of dict
            Updated local spaces with LNO-compressed external spaces
        """
        logger.info("\n" + "="*60)
        logger.info("Step 5: Compressing External Spaces with MP2 LNOs")
        logger.info("="*60)
        
        if self.local_spaces is None:
            raise ValueError("Local spaces not defined. Call define_local_spaces() first.")
        
        nocc = self.mol.nelectron // 2
        nmo = self.mf.mo_coeff.shape[1]
        nvirt = nmo - nocc
        
        logger.info(f"\nMP2-LNO parameters:")
        logger.info(f"  Occupied threshold (eta_occ): {eta_occ}")
        logger.info(f"  Virtual threshold (eta_virt): {eta_virt}")
        logger.info(f"  Number of occupied MOs: {nocc}")
        logger.info(f"  Number of virtual MOs: {nvirt}")
        logger.info(f"  Total MOs: {nmo}")
        
        # Get MO energies
        eps_mo = self.mf.mo_energy
        eps_occ_mo = eps_mo[:nocc]
        eps_virt_mo = eps_mo[nocc:]
        
        logger.info(f"\nMO energies:")
        logger.info(f"  Occupied: {eps_occ_mo.shape}")
        logger.info(f"  Virtual: {eps_virt_mo.shape}")
        
        # Get ERIs in MO basis
        logger.info(f"\nComputing ERIs in MO basis...")
        eri_mo = ao2mo.kernel(self.mol, self.mf.mo_coeff, compact=False)
        eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
        logger.info(f"  ERI shape in MO basis: {eri_mo.shape}")
        
        # Process each fragment
        local_spaces_lno = []
        
        for frag_idx, ls in enumerate(self.local_spaces):
            logger.info(f"\n--- Fragment {frag_idx}: LNO Compression ---")
            
            # Get W transformation matrices
            W_occ = ls['W_occ']  # (nocc, n_int_occ)
            W_tilde_occ = ls['W_tilde_occ']  # (nocc, n_ext_occ)
            W_virt = ls['W_virt']  # (nvirt, n_int_virt)
            W_tilde_virt = ls['W_tilde_virt']  # (nvirt, n_ext_virt)
            
            n_int_occ = W_occ.shape[1] if W_occ.size > 0 else 0
            n_ext_occ = W_tilde_occ.shape[1] if W_tilde_occ.size > 0 else 0
            n_int_virt = W_virt.shape[1] if W_virt.size > 0 else 0
            n_ext_virt = W_tilde_virt.shape[1] if W_tilde_virt.size > 0 else 0
            
            logger.info(f"  Fragment space dimensions:")
            logger.info(f"    Internal: {n_int_occ} occ + {n_int_virt} virt")
            logger.info(f"    External: {n_ext_occ} occ + {n_ext_virt} virt")
            
            # Skip MP2 if no internal occupied orbitals
            if n_int_occ == 0:
                logger.info(f"  Skipping MP2: no internal occupied orbitals")
                ls_lno = ls.copy()
                occ_result = self._create_empty_lno_result(n_ext_occ)
                virt_result = self._create_empty_lno_result(n_ext_virt)
                ls_lno.update({
                    'n_lno_occ': occ_result['n_lno'],
                    'n_lno_virt': virt_result['n_lno'],
                    'lno_occ_indices': occ_result['lno_indices'],
                    'lno_virt_indices': virt_result['lno_indices'],
                    'X_occ': occ_result['X'],
                    'X_virt': virt_result['X'],
                })
                local_spaces_lno.append(ls_lno)
                continue
            
            # Transform Fock matrix to internal occupied basis
            f_mo = np.diag(eps_mo)
            f_int_occ = W_occ.T @ f_mo[:nocc, :nocc] @ W_occ
            eps_int_occ = np.diag(f_int_occ)
            
            logger.info(f"\n  Orbital energies:")
            logger.info(f"    Internal occupied (epsilon_iF): {eps_int_occ.shape}")
            logger.info(f"      Mean: {np.mean(eps_int_occ):.6f}, Min: {np.min(eps_int_occ):.6f}, Max: {np.max(eps_int_occ):.6f}")
            
            # For other indices use MO energies directly
            eps_j = eps_occ_mo
            eps_a = eps_virt_mo
            eps_b = eps_virt_mo
            
            # Check if we can perform MP2
            if n_ext_occ == 0 and n_ext_virt == 0:
                logger.info(f"  Skipping MP2: both external spaces are empty")
                ls_lno = ls.copy()
                occ_result = self._create_empty_lno_result(n_ext_occ)
                virt_result = self._create_empty_lno_result(n_ext_virt)
                ls_lno.update({
                    'n_lno_occ': occ_result['n_lno'],
                    'n_lno_virt': virt_result['n_lno'],
                    'lno_occ_indices': occ_result['lno_indices'],
                    'lno_virt_indices': virt_result['lno_indices'],
                    'X_occ': occ_result['X'],
                    'X_virt': virt_result['X'],
                })
                local_spaces_lno.append(ls_lno)
                continue
            
            # Step 1: Compute MP1 amplitudes t_{iF,a,j,b}
            logger.info(f"\n  Computing MP1 amplitudes...")
            logger.info(f"    Transforming ERIs to mixed basis...")
            
            # Extract occupied-virtual-occupied-virtual block from ERI
            eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]  # (nocc, nvirt, nocc, nvirt)
            
            # Transform first index: (iF a | j b)
            V_Iajb = np.einsum('iI,iajb->Iajb', W_occ, eri_ovov, optimize=True)
            logger.info(f"    Transformed ERI shape: {V_Iajb.shape} = ({n_int_occ}, {nvirt}, {nocc}, {nvirt})")
            
            # Compute energy denominators
            eps_I = eps_int_occ[:, None, None, None]  # (n_int_occ, 1, 1, 1)
            eps_a_arr = eps_a[None, :, None, None]  # (1, nvirt, 1, 1)
            eps_j_arr = eps_j[None, None, :, None]  # (1, 1, nocc, 1)
            eps_b_arr = eps_b[None, None, None, :]  # (1, 1, 1, nvirt)
            
            denom = eps_I + eps_j_arr - eps_a_arr - eps_b_arr
            
            # Compute MP1 amplitudes with safe division
            t_Iajb = np.divide(V_Iajb, denom, where=np.abs(denom) > 1e-10, out=np.zeros_like(V_Iajb))
            
            logger.info(f"    MP1 amplitude shape: {t_Iajb.shape}")
            logger.info(f"    MP1 amplitude norm: {np.linalg.norm(t_Iajb):.6f}")
            logger.info(f"    Max amplitude: {np.max(np.abs(t_Iajb)):.6f}")
            
            # Step 2: Build MP2 density matrices
            logger.info(f"\n  Building MP2 density matrices...")
            
            # Occupied density: D_jj'^(F) in full occupied MO space
            term1_full = 2.0 * np.einsum('Iajb,IaJb->jJ', np.conj(t_Iajb), t_Iajb, optimize=True)
            term2_full = np.einsum('Iajb,IbJa->jJ', np.conj(t_Iajb), t_Iajb, optimize=True)
            D_occ_full = 2.0 * (term1_full - term2_full)
            
            logger.info(f"    D_occ (full MO) shape: {D_occ_full.shape}")
            logger.info(f"    D_occ norm: {np.linalg.norm(D_occ_full):.6f}")
            
            # Virtual density: D_ab^(F) in full virtual MO space
            term1 = np.einsum('Iajc,Ibjc->ab', np.conj(t_Iajb), t_Iajb, optimize=True)
            term2 = np.einsum('Icja,Icjb->ab', np.conj(t_Iajb), t_Iajb, optimize=True)
            term3 = np.einsum('Icja,Ibjc->ab', np.conj(t_Iajb), t_Iajb, optimize=True)
            term4 = np.einsum('Iajc,Icjb->ab', np.conj(t_Iajb), t_Iajb, optimize=True)
            D_virt_full = 2.0 * (term1 + term2) - (term3 + term4)
            
            logger.info(f"    D_virt (full MO) shape: {D_virt_full.shape}")
            logger.info(f"    D_virt norm: {np.linalg.norm(D_virt_full):.6f}")
            
            # Step 3: Project density matrices to external spaces and compress with LNOs
            logger.info(f"\n  Projecting to external spaces and computing LNOs...")
            
            # Handle occupied space compression
            if n_ext_occ > 0:
                D_occ_ext = W_tilde_occ.T @ D_occ_full @ W_tilde_occ
                logger.info(f"    D_occ_ext shape: {D_occ_ext.shape}")
                logger.info(f"\n  Diagonalizing occupied space for LNOs...")
                occ_result = self._compute_lno_compression(D_occ_ext, eta_occ, "occupied")
            else:
                logger.info(f"    Skipping occupied LNO compression (n_ext_occ=0)")
                occ_result = self._create_empty_lno_result(n_ext_occ)
            
            # Handle virtual space compression
            if n_ext_virt > 0:
                D_virt_ext = W_tilde_virt.T @ D_virt_full @ W_tilde_virt
                logger.info(f"    D_virt_ext shape: {D_virt_ext.shape}")
                logger.info(f"\n  Diagonalizing virtual space for LNOs...")
                virt_result = self._compute_lno_compression(D_virt_ext, eta_virt, "virtual")
            else:
                logger.info(f"    Skipping virtual LNO compression (n_ext_virt=0)")
                virt_result = self._create_empty_lno_result(n_ext_virt)
            
            # LNO selection summary
            logger.info(f"\n  LNO selection (threshold eta_occ={eta_occ}, eta_virt={eta_virt}):")
            logger.info(f"    Occupied LNOs selected: {occ_result['n_lno']} / {n_ext_occ}")
            logger.info(f"    Virtual LNOs selected: {virt_result['n_lno']} / {n_ext_virt}")
            if occ_result['n_lno'] > 0:
                logger.info(f"    Selected occ eigenvalues: {occ_result['Lambda'][occ_result['lno_indices']]}")
            if virt_result['n_lno'] > 0:
                logger.info(f"    Selected virt eigenvalues: {virt_result['Lambda'][virt_result['lno_indices']]}")
            
            # Update local space with LNO information
            ls_lno = ls.copy()
            ls_lno.update({
                'X_occ': occ_result['X'],  # LNO transformation matrix (n_ext_occ, n_ext_occ)
                'X_virt': virt_result['X'],  # LNO transformation matrix (n_ext_virt, n_ext_virt)
                'Lambda_occ': occ_result['Lambda'],  # Eigenvalues
                'Lambda_virt': virt_result['Lambda'],
                'lno_occ_indices': occ_result['lno_indices'],  # Indices of selected LNOs
                'lno_virt_indices': virt_result['lno_indices'],
                'n_lno_occ': occ_result['n_lno'],
                'n_lno_virt': virt_result['n_lno'],
            })
            
            local_spaces_lno.append(ls_lno)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("MP2-LNO Compression Summary")
        logger.info("="*60)
        logger.info(f"Total fragments: {len(local_spaces_lno)}")
        
        total_ext_before = 0
        total_ext_after = 0
        max_active_occ = 0
        max_active_virt = 0
        max_active_total = 0
        max_active_frag_id = None
        
        for ls in local_spaces_lno:
            n_int_occ = ls['n_int_occ']
            n_int_virt = ls['n_int_virt']
            n_ext_occ = ls['n_ext_occ']
            n_ext_virt = ls['n_ext_virt']
            n_ext_before = n_ext_occ + n_ext_virt
            n_ext_after = ls['n_lno_occ'] + ls['n_lno_virt']
            total_ext_before += n_ext_before
            total_ext_after += n_ext_after
            
            # Calculate active space size (internal + selected LNOs)
            active_occ = n_int_occ + ls['n_lno_occ']
            active_virt = n_int_virt + ls['n_lno_virt']
            active_total = active_occ + active_virt
            
            # Track maximum active space
            if active_total > max_active_total:
                max_active_total = active_total
                max_active_occ = active_occ
                max_active_virt = active_virt
                max_active_frag_id = ls['fragment_id']
            
            logger.info(f"\nFragment {ls['fragment_id']}:")
            logger.info(f"  Internal: {n_int_occ} occ + {n_int_virt} virt = {n_int_occ + n_int_virt}")
            logger.info(f"  External (before): {n_ext_occ} occ + {n_ext_virt} virt = {n_ext_before}")
            logger.info(f"  External (after LNO): {ls['n_lno_occ']} occ + {ls['n_lno_virt']} virt = {n_ext_after}")
            logger.info(f"  Active space size: {active_occ} occ + {active_virt} virt = {active_total}")
            if n_ext_before > 0:
                compression_ratio = 1 - n_ext_after / n_ext_before
                logger.info(f"  Compression ratio: {compression_ratio:.2%} ({n_ext_after}/{n_ext_before})")
            else:
                logger.info(f"  No external space")
        
        logger.info(f"\nOverall compression:")
        logger.info(f"  Total external orbitals before: {total_ext_before}")
        logger.info(f"  Total external orbitals after LNO: {total_ext_after}")
        if total_ext_before > 0:
            overall_ratio = 1 - total_ext_after / total_ext_before
            logger.info(f"  Overall compression ratio: {overall_ratio:.2%} ({total_ext_after}/{total_ext_before})")
        else:
            logger.info(f"  No external space to compress")
        
        logger.info(f"\nLargest active space (Fragment {max_active_frag_id}):")
        logger.info(f"  Occupied: {max_active_occ}")
        logger.info(f"  Virtual: {max_active_virt}")
        logger.info(f"  Total: {max_active_total}")
        logger.info("="*60 + "\n")
        
        self.local_spaces_lno = local_spaces_lno
        return local_spaces_lno
    
    def construct_local_ham(self, ls):
        """
        Construct local Hamiltonian (Fock and ERIs) in active space for a single fragment.
        
        The active space comprises:
        - Internal orbitals (from SVD with large singular values)
        - Active LNOs (external orbitals selected by MP2-LNO compression)
        
        This follows the paper's orbital rotation scheme with modified Fock matrix
        including frozen orbital contributions (Equation 11).
        
        Parameters
        ----------
        ls : dict
            Local space information for a single fragment from compute_mp2_lno_compression
            
        Returns
        -------
        local_ham : dict
            Dictionary containing:
            - 'f_active': Fock matrix in active space (n_active, n_active)
            - 'eri_active': ERIs in active space (n_active, n_active, n_active, n_active)
            - 'n_active_occ': Number of occupied orbitals in active space
            - 'n_active_virt': Number of virtual orbitals in active space
            - 'U_act_occ': Transformation from MO occ to active occ (nocc, n_active_occ)
            - 'U_act_virt': Transformation from MO virt to active virt (nvirt, n_active_virt)
        """
        logger.info(f"\n  Constructing local Hamiltonian for fragment {ls['fragment_id']}...")
        
        # Extract fragment information
        W_occ = ls['W_occ']  # (nocc, n_int_occ)
        W_tilde_occ = ls['W_tilde_occ']  # (nocc, n_ext_occ)
        W_virt = ls['W_virt']  # (nvirt, n_int_virt)
        W_tilde_virt = ls['W_tilde_virt']  # (nvirt, n_ext_virt)
        
        X_occ = ls['X_occ']  # (n_ext_occ, n_ext_occ)
        X_virt = ls['X_virt']  # (n_ext_virt, n_ext_virt)
        
        lno_occ_indices = ls['lno_occ_indices']  # Indices of selected occ LNOs
        lno_virt_indices = ls['lno_virt_indices']  # Indices of selected virt LNOs
        
        n_int_occ = W_occ.shape[1] if W_occ.size > 0 else 0
        n_int_virt = W_virt.shape[1] if W_virt.size > 0 else 0
        n_lno_occ = len(lno_occ_indices)
        n_lno_virt = len(lno_virt_indices)
        
        n_active_occ = n_int_occ + n_lno_occ
        n_active_virt = n_int_virt + n_lno_virt
        n_active = n_active_occ + n_active_virt
        
        logger.info(f"    Active space dimensions:")
        logger.info(f"      Internal: {n_int_occ} occ + {n_int_virt} virt")
        logger.info(f"      Active LNOs: {n_lno_occ} occ + {n_lno_virt} virt")
        logger.info(f"      Total active: {n_active_occ} occ + {n_active_virt} virt = {n_active}")
        
        # Check if active space is valid
        if n_active_occ == 0 or n_active_virt == 0:
            logger.warning(f"    Invalid active space (need both occ and virt)")
            return None
        
        # Build transformation matrices U_act from MO to active space
        # U_act_occ: (nocc, n_active_occ) = [W_occ | W_tilde_occ @ X_occ[:, lno_occ_indices]]
        # U_act_virt: (nvirt, n_active_virt) = [W_virt | W_tilde_virt @ X_virt[:, lno_virt_indices]]
        
        nocc = self.mol.nelectron // 2
        nvirt = self.mf.mo_coeff.shape[1] - nocc
        
        # Internal part
        U_occ_int = W_occ  # (nocc, n_int_occ)
        U_virt_int = W_virt  # (nvirt, n_int_virt)
        
        # External LNO part - select columns of X corresponding to active LNOs
        if n_lno_occ > 0:
            X_occ_active = X_occ[:, lno_occ_indices]  # (n_ext_occ, n_lno_occ)
            U_occ_ext = W_tilde_occ @ X_occ_active  # (nocc, n_lno_occ)
        else:
            U_occ_ext = np.zeros((nocc, 0))
        
        if n_lno_virt > 0:
            X_virt_active = X_virt[:, lno_virt_indices]  # (n_ext_virt, n_lno_virt)
            U_virt_ext = W_tilde_virt @ X_virt_active  # (nvirt, n_lno_virt)
        else:
            U_virt_ext = np.zeros((nvirt, 0))
        
        # Combine internal and external LNOs: [internal | active_LNOs]
        if n_int_occ > 0 and n_lno_occ > 0:
            U_act_occ = np.hstack([U_occ_int, U_occ_ext])
        elif n_int_occ > 0:
            U_act_occ = U_occ_int
        elif n_lno_occ > 0:
            U_act_occ = U_occ_ext
        else:
            U_act_occ = np.zeros((nocc, 0))
        
        if n_int_virt > 0 and n_lno_virt > 0:
            U_act_virt = np.hstack([U_virt_int, U_virt_ext])
        elif n_int_virt > 0:
            U_act_virt = U_virt_int
        elif n_lno_virt > 0:
            U_act_virt = U_virt_ext
        else:
            U_act_virt = np.zeros((nvirt, 0))
        
        logger.info(f"    Transformation matrices (active only):")
        logger.info(f"      U_act_occ: {U_act_occ.shape}")
        logger.info(f"      U_act_virt: {U_act_virt.shape}")
        
        # Build transformation to "active + ALL external LNOs" basis (including frozen)
        # This is needed to construct the modified Fock with frozen orbital contributions
        n_ext_occ = W_tilde_occ.shape[1]
        n_ext_virt = W_tilde_virt.shape[1]
        n_frozen_occ = n_ext_occ - n_lno_occ  # Frozen occupied LNOs
        n_frozen_virt = n_ext_virt - n_lno_virt  # Frozen virtual LNOs
        
        logger.info(f"    Extended basis (active + frozen LNOs):")
        logger.info(f"      Frozen: {n_frozen_occ} occ + {n_frozen_virt} virt")
        
        # Build full external LNO transformations: [W_tilde @ X]
        # These include both active and frozen LNOs
        if n_ext_occ > 0:
            U_occ_ext_full = W_tilde_occ @ X_occ  # (nocc, n_ext_occ)
        else:
            U_occ_ext_full = np.zeros((nocc, 0))
        
        if n_ext_virt > 0:
            U_virt_ext_full = W_tilde_virt @ X_virt  # (nvirt, n_ext_virt)
        else:
            U_virt_ext_full = np.zeros((nvirt, 0))
        
        # Combine: [internal | all_external_LNOs]
        if n_int_occ > 0 and n_ext_occ > 0:
            U_full_occ = np.hstack([U_occ_int, U_occ_ext_full])
        elif n_int_occ > 0:
            U_full_occ = U_occ_int
        elif n_ext_occ > 0:
            U_full_occ = U_occ_ext_full
        else:
            U_full_occ = np.zeros((nocc, 0))
        
        if n_int_virt > 0 and n_ext_virt > 0:
            U_full_virt = np.hstack([U_virt_int, U_virt_ext_full])
        elif n_int_virt > 0:
            U_full_virt = U_virt_int
        elif n_ext_virt > 0:
            U_full_virt = U_virt_ext_full
        else:
            U_full_virt = np.zeros((nvirt, 0))
        
        n_full_occ = U_full_occ.shape[1]
        n_full_virt = U_full_virt.shape[1]
        n_full = n_full_occ + n_full_virt
        
        logger.info(f"    Full extended basis: {n_full_occ} occ + {n_full_virt} virt = {n_full}")
        
        # Transform core Hamiltonian and ERIs to extended basis
        logger.info(f"    Transforming to extended basis (active + frozen)...")
        
        # Get core Hamiltonian in MO basis
        h_core_ao = self.mol.intor('int1e_kin') + self.mol.intor('int1e_nuc')
        h_mo = self.mf.mo_coeff.T @ h_core_ao @ self.mf.mo_coeff
        
        # Transform core Hamiltonian to extended basis
        h_oo_full = U_full_occ.T @ h_mo[:nocc, :nocc] @ U_full_occ
        h_ov_full = U_full_occ.T @ h_mo[:nocc, nocc:] @ U_full_virt
        h_vo_full = U_full_virt.T @ h_mo[nocc:, :nocc] @ U_full_occ
        h_vv_full = U_full_virt.T @ h_mo[nocc:, nocc:] @ U_full_virt
        
        # Assemble full core Hamiltonian in extended basis
        h_full = np.zeros((n_full, n_full))
        h_full[:n_full_occ, :n_full_occ] = h_oo_full
        h_full[:n_full_occ, n_full_occ:] = h_ov_full
        h_full[n_full_occ:, :n_full_occ] = h_vo_full
        h_full[n_full_occ:, n_full_occ:] = h_vv_full
        
        logger.info(f"      h_full shape: {h_full.shape}, norm: {np.linalg.norm(h_full):.6f}")
        
        # Get ERIs in MO basis and transform to extended basis
        nmo = self.mf.mo_coeff.shape[1]
        eri_mo = ao2mo.kernel(self.mol, self.mf.mo_coeff, compact=False)
        eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
        
        # Build combined transformation matrix for extended basis
        U_full_mo = np.zeros((nmo, n_full))
        U_full_mo[:nocc, :n_full_occ] = U_full_occ
        U_full_mo[nocc:, n_full_occ:] = U_full_virt
        
        # Four-index transformation
        logger.info(f"    Transforming ERIs (this may take a moment)...")
        eri_temp1 = np.einsum('ip,ijkl->pjkl', U_full_mo, eri_mo, optimize=True)
        eri_temp2 = np.einsum('jq,pjkl->pqkl', U_full_mo, eri_temp1, optimize=True)
        eri_temp3 = np.einsum('kr,pqkl->pqrl', U_full_mo, eri_temp2, optimize=True)
        eri_full = np.einsum('ls,pqrl->pqrs', U_full_mo, eri_temp3, optimize=True)
        
        logger.info(f"      ERI_full shape: {eri_full.shape}, norm: {np.linalg.norm(eri_full):.6f}")
        
        # Construct modified Fock matrix using Eq. (11)
        # f_pq^(F) = h_pq + Σ_{i∉A_F} (2*V_{pqii} - V_piiq)
        # where i∉A_F are the frozen occupied orbitals
        logger.info(f"    Constructing modified Fock with frozen orbital contributions...")
        
        f_full = h_full.copy()
        
        # Frozen occupied orbitals are indices [n_int_occ + n_lno_occ : n_full_occ]
        frozen_occ_start = n_int_occ + n_lno_occ
        frozen_occ_end = n_full_occ
        
        if frozen_occ_end > frozen_occ_start:
            logger.info(f"      Adding contributions from {frozen_occ_end - frozen_occ_start} frozen occupied orbitals")
            frozen_indices = slice(frozen_occ_start, frozen_occ_end)
            
            # Coulomb: 2 * Σ_i V[:,:,i,i]
            coulomb = 2.0 * np.sum(eri_full[:, :, frozen_indices, frozen_indices].diagonal(axis1=2, axis2=3), axis=2)
            
            # Exchange: Σ_i V[:,i,i,:]
            exchange = np.sum(eri_full[:, frozen_indices, frozen_indices, :].transpose(0, 3, 1, 2).diagonal(axis1=2, axis2=3), axis=2)
            
            f_full += coulomb - exchange
        else:
            logger.info(f"      No frozen occupied orbitals - using core Hamiltonian only")
        
        logger.info(f"      f_full (modified) shape: {f_full.shape}, norm: {np.linalg.norm(f_full):.6f}")
        
        # Extract active space block from full extended basis
        # Active occupied: [0 : n_int_occ + n_lno_occ]
        # Active virtual: [n_full_occ : n_full_occ + n_int_virt + n_lno_virt]
        active_occ_slice = slice(0, n_active_occ)
        active_virt_slice = slice(n_full_occ, n_full_occ + n_active_virt)
        
        # Extract active blocks
        f_active = np.zeros((n_active, n_active))
        f_active[:n_active_occ, :n_active_occ] = f_full[active_occ_slice, active_occ_slice]
        f_active[:n_active_occ, n_active_occ:] = f_full[active_occ_slice, active_virt_slice]
        f_active[n_active_occ:, :n_active_occ] = f_full[active_virt_slice, active_occ_slice]
        f_active[n_active_occ:, n_active_occ:] = f_full[active_virt_slice, active_virt_slice]
        
        # Extract ERIs for active space
        # Vectorized: use advanced indexing with np.ix_
        active_indices = np.array(list(range(n_active_occ)) + list(range(n_full_occ, n_full_occ + n_active_virt)))
        eri_active = eri_full[np.ix_(active_indices, active_indices, active_indices, active_indices)].copy()
        
        logger.info(f"      f_active (extracted) shape: {f_active.shape}, norm: {np.linalg.norm(f_active):.6f}")
        logger.info(f"      eri_active (extracted) shape: {eri_active.shape}, norm: {np.linalg.norm(eri_active):.6f}")
        
        local_ham = {
            'f_active': f_active,
            'eri_active': eri_active,
            'n_active_occ': n_active_occ,
            'n_active_virt': n_active_virt,
            'U_act_occ': U_act_occ,
            'U_act_virt': U_act_virt,
        }
        return local_ham
    
    def solve_fragment_ccsd(self):
        """
        Solve fragment CCSD equations for all fragments.
        
        For each fragment:
        1. Construct local Hamiltonian with modified Fock matrix
        2. Set up and run CCSD calculation using PySCF
        3. Compute E_{ii'} matrix using Equation 13
        
        Returns
        -------
        ccsd_results : dict
            Dictionary containing:
            - 'fragment_results': List of CCSD results for each fragment
            - 'fragment_hamiltonians': List of local Hamiltonians
        """
        logger.info("\n" + "="*60)
        logger.info("Step 6: Solving Fragment CCSD Equations")
        logger.info("="*60)
        
        if self.local_spaces_lno is None:
            raise ValueError("LNO compression not done. Call compute_mp2_lno_compression() first.")
        
        logger.info(f"\nFragment CCSD setup:")
        logger.info(f"  Number of fragments: {len(self.local_spaces_lno)}")
        
        fragment_results = []
        fragment_hamiltonians = []
        
        for frag_idx, ls in enumerate(self.local_spaces_lno):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fragment {frag_idx}: CCSD Calculation")
            logger.info(f"{'='*60}")
            
            # Construct local Hamiltonian
            local_ham = self.construct_local_ham(ls)
            
            # Check if Hamiltonian construction was successful
            if local_ham is None:
                logger.warning(f"  Skipping fragment {frag_idx} (invalid active space)")
                fragment_results.append(None)
                fragment_hamiltonians.append(None)
                continue
            
            # Extract Hamiltonian components
            f_active = local_ham['f_active']
            eri_active = local_ham['eri_active']
            n_active_occ = local_ham['n_active_occ']
            n_active_virt = local_ham['n_active_virt']
            n_active = n_active_occ + n_active_virt
            
            logger.info(f"\n  Setting up CCSD calculation...")
            logger.info(f"    Active space: {n_active} orbitals ({n_active_occ} occ + {n_active_virt} virt)")
            
            # Create fake molecule and mean-field object for PySCF CCSD
            mol_frag = gto.M()
            mol_frag.nelectron = 2 * n_active_occ
            mol_frag.verbose = 0
            
            mf_frag = scf.RHF(mol_frag)
            mf_frag.mo_coeff = np.eye(n_active)
            mf_frag.mo_energy = np.diag(f_active)
            mf_frag.mo_occ = np.zeros(n_active)
            mf_frag.mo_occ[:n_active_occ] = 2.0
            mf_frag.e_tot = 0.0
            mf_frag.converged = True
            
            # Set h1e (core Hamiltonian) for energy calculation
            mf_frag.get_hcore = lambda *args: f_active
            
            # Override _eri to provide custom ERIs
            mf_frag._eri = eri_active
            
            # Create CCSD object
            mycc = cc.CCSD(mf_frag)
            mycc.verbose = 0
            mycc.max_cycle = 100
            mycc.conv_tol = 1e-7
            
            # Run CCSD
            logger.info(f"  Running CCSD...")
            mycc.kernel()
            
            if mycc.converged:
                logger.info(f"    CCSD converged!")
            else:
                logger.warning(f"    CCSD did not converge!")
            
            logger.info(f"    CCSD correlation energy: {mycc.e_corr:.8f} a.u.")
            
            # Get amplitudes in active space
            t1_act = mycc.t1  # (n_active_occ, n_active_virt)
            t2_act = mycc.t2  # (n_active_occ, n_active_occ, n_active_virt, n_active_virt)
            
            logger.info(f"    T1 amplitude shape: {t1_act.shape}, norm: {np.linalg.norm(t1_act):.6f}")
            logger.info(f"    T2 amplitude shape: {t2_act.shape}, norm: {np.linalg.norm(t2_act):.6f}")
            
            # Compute tau intermediate: τ_{iajb} = t_{iajb} - t_{ia}*t_{jb}
            # Note: t2_act has indices [i,j,a,b] but we need τ with indices [i,a,j,b]
            t1_outer = np.einsum('ia,jb->iajb', t1_act, t1_act, optimize=True)
            tau_act = t2_act.transpose(0, 2, 1, 3) - t1_outer  # Reorder t2 from ijab to iajb
            
            # Compute E_{ii'} using Equation 13:
            # E_{ii'}^{(F)} = Σ_{jab∈A_F} τ_{iajb}^{(F)} (2V_{i'ajb} - V_{i'bja})
            logger.info(f"  Computing energy contribution matrix E_{{ii'}} using Equation 13...")
            
            # Extract occupied-virtual blocks of ERIs
            eri_ovov = eri_active[:n_active_occ, n_active_occ:, :n_active_occ, n_active_occ:]
            
            # E[i, i'] = Σ_{jab} τ[i,a,j,b] * (2*V[i',a,j,b] - V[i',b,j,a])
            coulomb = 2.0 * np.einsum('iajb,Iajb->iI', tau_act, eri_ovov, optimize=True)
            exchange = np.einsum('iajb,Ibja->iI', tau_act, eri_ovov, optimize=True)
            E_ii_prime = coulomb - exchange
            
            logger.info(f"    E_{{ii'}} shape: {E_ii_prime.shape}, norm: {np.linalg.norm(E_ii_prime):.6f}")
            
            # Store results
            frag_result = {
                'fragment_id': frag_idx,
                'n_active_occ': n_active_occ,
                'n_active_virt': n_active_virt,
                'ccsd': mycc,
                't1': t1_act,
                't2': t2_act,
                'tau': tau_act,
                'E_ii_prime': E_ii_prime,
                'e_corr_ccsd': mycc.e_corr,
            }
            
            fragment_results.append(frag_result)
            fragment_hamiltonians.append(local_ham)
        
        logger.info("\n" + "="*60)
        logger.info("Fragment CCSD Completed")
        logger.info("="*60)
        logger.info(f"Successfully computed {len(fragment_results)} fragments")
        logger.info("="*60 + "\n")
        
        results = {
            'fragment_results': fragment_results,
            'fragment_hamiltonians': fragment_hamiltonians,
        }
        
        self.ccsd_results = results
        return results
    
    def compute_fragment_energies(self):
        """
        Compute fragment correlation energies by transforming E_{ii'} to fragment IAO basis.
        
        For each fragment:
        1. Transform E_{ii'} from active space to MO basis
        2. Extract IAO columns for this fragment from C_iao_mo_occ
        3. Transform to fragment IAO basis: E_frag = C_frag^T @ E_mo @ C_frag
        4. Sum diagonal (trace) to get fragment energy
        
        Returns
        -------
        energy_results : dict
            Dictionary containing:
            - 'fragment_energies': List of correlation energies
            - 'total_correlation': Total correlation energy
            - 'total_energy': RHF + correlation energy
        """
        logger.info("\n" + "="*60)
        logger.info("Step 7: Computing Fragment Correlation Energies")
        logger.info("="*60)
        
        if self.ccsd_results is None:
            raise ValueError("CCSD not solved. Call solve_fragment_ccsd() first.")
        
        fragment_results = self.ccsd_results['fragment_results']
        fragment_hamiltonians = self.ccsd_results['fragment_hamiltonians']
        
        # Get overlap and transform IAO to MO basis
        S_ao = self.mf.get_ovlp()
        C_mo = self.mf.mo_coeff
        C_iao_mo = C_mo.T @ S_ao @ self.iao_coeff
        
        nocc = self.mol.nelectron // 2
        C_iao_mo_occ = C_iao_mo[:nocc, :]
        
        logger.info(f"\nTransformation matrices:")
        logger.info(f"  IAO-MO (occupied): {C_iao_mo_occ.shape}")
        
        fragment_energies = []
        
        for frag_idx, (frag_result, local_ham, ls) in enumerate(
            zip(fragment_results, fragment_hamiltonians, self.local_spaces_lno)
        ):
            logger.info(f"\nFragment {frag_idx}:")
            
            if frag_result is None or local_ham is None:
                logger.info(f"  Skipped (no CCSD result)")
                fragment_energies.append(0.0)
                continue
            
            # Get E_{ii'} and transformation matrix
            E_ii_prime = frag_result['E_ii_prime']
            U_act_occ = local_ham['U_act_occ']
            
            # Transform to MO basis
            E_mo = U_act_occ @ E_ii_prime @ U_act_occ.T
            
            logger.info(f"  E_MO shape: {E_mo.shape}, norm: {np.linalg.norm(E_mo):.6f}")
            
            # Get fragment IAO indices
            frag_iao_indices = ls['iao_indices']
            
            # Extract fragment IAO columns
            C_frag = C_iao_mo_occ[:, frag_iao_indices]
            
            logger.info(f"  Fragment IAO indices: {frag_iao_indices}")
            logger.info(f"  C_frag shape: {C_frag.shape}")
            
            # Transform to fragment IAO basis
            E_frag = C_frag.T @ E_mo @ C_frag
            
            logger.info(f"  E_frag shape: {E_frag.shape}, norm: {np.linalg.norm(E_frag):.6f}")
            
            # Sum diagonal to get fragment energy
            fragment_energy = np.trace(E_frag)
            
            logger.info(f"  Fragment correlation energy: {fragment_energy:.8f} a.u.")
            
            fragment_energies.append(fragment_energy)
        
        # Compute total correlation energy
        total_corr = np.sum(fragment_energies)
        total_energy = self.mf.e_tot + total_corr
        
        logger.info("\n" + "="*60)
        logger.info("Fragment Energy Summary")
        logger.info("="*60)
        for i, e in enumerate(fragment_energies):
            logger.info(f"  Fragment {i}: {e:.8f} a.u.")
        logger.info(f"\nTotal correlation energy: {total_corr:.8f} a.u.")
        logger.info(f"RHF energy: {self.mf.e_tot:.8f} a.u.")
        logger.info(f"Total energy: {total_energy:.8f} a.u.")
        logger.info("="*60 + "\n")
        
        energy_results = {
            'fragment_energies': fragment_energies,
            'total_correlation': total_corr,
            'total_energy': total_energy,
        }
        
        self.energy_results = energy_results
        return energy_results
    
    def run(self, fragments, minao='minao', svd_threshold=1e-8,
            eta_occ=1e-6, eta_virt=1e-6):
        """
        Run the complete SIE workflow.
        
        Parameters
        ----------
        fragments : list of list
            Atom indices for each fragment
        minao : str, optional
            Minimal basis for IAO construction
        svd_threshold : float, optional
            Threshold for SVD-based internal/external classification
        eta_occ : float, optional
            LNO threshold for occupied
        eta_virt : float, optional
            LNO threshold for virtual
        
        Returns
        -------
        results : dict
            Complete results dictionary
        """
        logger.info("="*70)
        logger.info("Systematic Improvable Embedding (SIE) Workflow")
        logger.info("="*70)
        
        # Step 1: Construct IAOs
        self.construct_iaos(minao=minao)
        
        # Step 2: Define local spaces
        self.define_local_spaces(
            fragments=fragments,
            svd_threshold=svd_threshold
        )
        
        # Step 3: MP2-LNO compression
        self.compute_mp2_lno_compression(
            eta_occ=eta_occ,
            eta_virt=eta_virt
        )
        
        # Step 4: Solve fragment CCSD
        self.solve_fragment_ccsd()
        
        # Step 5: Compute fragment energies
        self.compute_fragment_energies()
        
        logger.info("\n" + "="*70)
        logger.info("SIE Workflow Completed Successfully!")
        logger.info("="*70)
        
        results = {
            'iao_coeff': self.iao_coeff,
            'iao_labels': self.iao_labels,
            'local_spaces': self.local_spaces,
            'local_spaces_lno': self.local_spaces_lno,
            'ccsd_results': self.ccsd_results,
            'energy_results': self.energy_results,
        }
        
        return results
