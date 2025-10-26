"""
Quantum Embedding Scheme for H10 Chain

This module explores quantum embedding schemes with the following steps:
1. Define a molecule of H10 chain using cc-pvdz basis set
2. Carry out a RHF calculation
3. Use converged molecular orbitals to construct Intrinsic Atomic Orbitals (IAO)
4. Construct Local Active Space for each fragment
5. Compress external spaces using Local Natural Orbitals (LNOs) from MP2
6. Solve fragment CCSD equations and compute correlation energy

Author: Ke Liao
Date: October 21, 2025
"""

import numpy as np
from pyscf import gto, scf, lo, ao2mo, cc
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_mol(separation=1.5, basis='cc-pvdz', unit='Angstrom'):
    """
    Build a linear H10 chain molecule.
    
    Parameters
    ----------
    separation : float, optional
        Distance between adjacent H atoms in the chain (default: 1.5)
    basis : str, optional
        Basis set to use (default: 'cc-pvdz')
    unit : str, optional
        Unit for atomic coordinates, 'Angstrom' or 'Bohr' (default: 'Angstrom')
    
    Returns
    -------
    mol : pyscf.gto.Mole
        The H10 chain molecule object
    """
    logger.info(f"Building H10 chain with separation={separation} {unit}, basis={basis}")
    
    # Create molecule object
    mol = gto.Mole()
    
    # Build the H10 chain along the z-axis
    atom_string = ""
    for i in range(10):
        z_coord = i * separation
        atom_string += f"H 0 0 {z_coord};"
    
    mol.atom = atom_string
    mol.basis = basis
    mol.unit = unit
    mol.spin = 0  # Closed-shell system (10 electrons, 5 pairs)
    mol.charge = 0
    mol.verbose = 4
    mol.build()
    
    logger.info(f"Molecule built successfully:")
    logger.info(f"  Number of atoms: {mol.natm}")
    logger.info(f"  Number of electrons: {mol.nelectron}")
    logger.info(f"  Number of basis functions: {mol.nao}")
    logger.info(f"  Nuclear repulsion energy: {mol.energy_nuc():.6f} a.u.")
    
    return mol


def run_rhf(mol):
    """
    Perform Restricted Hartree-Fock (RHF) calculation.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    
    Returns
    -------
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    """
    logger.info("Starting RHF calculation...")
    
    # Create RHF object
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.max_cycle = 100
    mf.conv_tol = 1e-10
    
    # Run SCF calculation
    energy = mf.kernel()
    
    if mf.converged:
        logger.info(f"RHF calculation converged successfully!")
        logger.info(f"  Total energy: {energy:.8f} a.u.")
        logger.info(f"  Number of occupied orbitals: {mol.nelectron // 2}")
        logger.info(f"  Number of virtual orbitals: {mol.nao - mol.nelectron // 2}")
    else:
        logger.warning("RHF calculation did not converge!")
    
    return mf


def analyze_rhf_solution(mol, mf):
    """
    Analyze the RHF solution and print useful information.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    """
    logger.info("\n" + "="*60)
    logger.info("RHF Solution Analysis")
    logger.info("="*60)
    
    # Orbital energies
    nocc = mol.nelectron // 2
    logger.info(f"\nOccupied orbital energies (HOMO-4 to HOMO):")
    for i in range(max(0, nocc-5), nocc):
        logger.info(f"  MO {i}: {mf.mo_energy[i]:.6f} a.u.")
    
    logger.info(f"\nVirtual orbital energies (LUMO to LUMO+4):")
    for i in range(nocc, min(nocc+5, len(mf.mo_energy))):
        logger.info(f"  MO {i}: {mf.mo_energy[i]:.6f} a.u.")
    
    # HOMO-LUMO gap
    if nocc < len(mf.mo_energy):
        gap = mf.mo_energy[nocc] - mf.mo_energy[nocc-1]
        logger.info(f"\nHOMO-LUMO gap: {gap:.6f} a.u. ({gap*27.2114:.4f} eV)")
    
    # Mulliken population analysis
    logger.info(f"\nMulliken population analysis:")
    pop = mf.mulliken_pop()
    
    logger.info("="*60 + "\n")


def construct_iao(mol, mf, minao='minao'):
    """
    Construct Intrinsic Atomic Orbitals (IAO) from converged RHF molecular orbitals.
    
    IAOs are localized atomic-like orbitals that span the occupied space. This function
    returns orthogonalized IAOs using Löwdin orthogonalization, which are suitable for
    fragment-based quantum embedding calculations.
    
    Note: Raw IAOs are non-orthogonal by design (they preserve maximal atomic character).
    We apply Löwdin orthogonalization: C_orth = C (C^T S C)^{-1/2} to obtain orthonormal
    IAOs while maintaining reasonable localization.
    
    IAO labels include both CORE and VALENCE orbitals:
    - For second-row atoms (C, N, O, etc.): 1s (core), 2s and 2p (valence)
    - For hydrogen: 1s (valence)
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    minao : str, optional
        Minimal basis set for IAO construction (default: 'minao')
    
    Returns
    -------
    iao_coeff : np.ndarray
        Orthogonalized IAO coefficients in AO basis, shape (nao, niao)
    iao_labels : list
        Labels for each IAO in format "atom_idx atom_symbol orbital_type"
        Example: "0 O 1s", "0 O 2s", "1 H 1s"
    """
    logger.info("\n" + "="*60)
    logger.info("Step 3: Constructing Intrinsic Atomic Orbitals (IAO)")
    logger.info("="*60)
    
    # Construct IAOs using PySCF's lo.iao module
    # IAOs are constructed from occupied orbitals
    nocc = mol.nelectron // 2
    mo_coeff_occ = mf.mo_coeff[:, :nocc]
    
    logger.info(f"\nIAO construction parameters:")
    logger.info(f"  Minimal basis: {minao}")
    logger.info(f"  Number of occupied MOs: {nocc}")
    
    # Build IAOs - these span the occupied space (non-orthogonal)
    iao_coeff_raw = lo.iao.iao(mol, mo_coeff_occ, minao=minao)
    
    niao = iao_coeff_raw.shape[1]
    
    # Orthogonalize IAOs using Löwdin orthogonalization
    # C_orth = C (C^T S C)^{-1/2}
    # This maintains reasonable localization while ensuring orthonormality
    S = mol.intor('int1e_ovlp')
    iao_overlap = iao_coeff_raw.T @ S @ iao_coeff_raw
    
    # Import orth module for Löwdin orthogonalization
    from pyscf.lo import orth
    lowdin_transform = orth.lowdin(iao_overlap)
    iao_coeff = iao_coeff_raw @ lowdin_transform
    
    # Verify orthonormality
    S_iao_orth = iao_coeff.T @ S @ iao_coeff
    max_off_diag = np.max(np.abs(S_iao_orth - np.diag(np.diag(S_iao_orth))))
    logger.info(f"  IAO orthogonalization: max off-diagonal = {max_off_diag:.2e}")
    
    # Use reference_mol to get IAO labels directly from PySCF
    # This is more reliable than overlap-based methods
    pmol = lo.iao.reference_mol(mol, minao=minao)
    iao_labels_raw = pmol.ao_labels()
    
    # Validate that the number of IAOs matches the number of labels
    if len(iao_labels_raw) != niao:
        error_msg = (f"Mismatch between number of IAOs ({niao}) and "
                    f"number of labels from reference_mol ({len(iao_labels_raw)})")
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Parse the labels to extract atom index, symbol, and orbital type
    # Format: '0 O 1s' -> atom_idx=0, atom_symbol='O', orbital_type='1s'
    iao_labels = []
    for i, label in enumerate(iao_labels_raw):
        # The label format is: 'atom_idx atom_symbol orbital_type'
        parts = label.split()
        if len(parts) >= 3:
            atom_idx = int(parts[0])
            atom_symbol = parts[1]
            orbital_type = parts[2]
            iao_labels.append(f"{atom_idx} {atom_symbol} {orbital_type}")
        else:
            # Fallback if label format is unexpected
            logger.warning(f"Unexpected label format for IAO {i}: {label}")
            iao_labels.append(label)
    
    logger.info(f"\nIAO construction completed:")
    logger.info(f"  Number of IAOs: {niao}")
    logger.info(f"  IAO dimensions: {iao_coeff.shape}")
    logger.info(f"  IAOs are orthonormalized using Löwdin method")
    logger.info(f"  IAO labels (first 10): {iao_labels[:10]}")
    if niao > 10:
        logger.info(f"  ... (total {niao} IAOs)")
    
    # Log information about core vs valence orbitals
    # Core orbitals: 1s for second-row atoms (C, N, O, F, etc.), not H
    core_count = 0
    for label in iao_labels:
        parts = label.split()
        if len(parts) >= 3:
            atom_symbol = parts[1]
            orbital_type = parts[2]
            # 1s is core for non-hydrogen atoms
            if orbital_type == '1s' and atom_symbol != 'H':
                core_count += 1
    valence_count = niao - core_count
    logger.info(f"  Core orbitals: {core_count}, Valence orbitals: {valence_count}")
    logger.info("="*60 + "\n")
    
    return iao_coeff, iao_labels


def construct_local_active_space(mol, mf, iao_coeff, iao_labels, fragment_size=1, svd_threshold=1e-6):
    """
    Construct Local Active Space for each fragment using SVD decomposition in MO space.
    
    For each fragment F:
    1. Transform IAOs to MO basis: C_IAO_MO = C_MO^T @ S @ C_IAO
    2. Extract U_occ(F) (N_occ × nLO(F)) and U_virt(F) (N_virt × nLO(F))
    3. Perform SVD: U = W @ diag(s) @ Vh
    4. W columns with s > threshold define internal space (in MO basis)
    5. Remaining W columns (full_matrices=True) define external space
    6. Transform W matrices back to AO basis for ERI transformations
    
    Reference: Equation 18 in the quantum embedding paper
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    iao_coeff : np.ndarray
        IAO coefficients in AO basis, shape (nao, niao)
    iao_labels : list
        Labels for each IAO
    fragment_size : int, optional
        Number of atoms per fragment (default: 1 - each atom is a fragment)
    svd_threshold : float, optional
        Threshold for SVD singular values to determine internal space (default: 1e-6)
    
    Returns
    -------
    local_spaces : list of dict
        List of dictionaries, one per fragment, containing:
        - 'atoms': list of atom indices in fragment
        - 'iao_indices': IAO indices belonging to fragment
        - 'W_occ': internal occupied orbitals in AO basis (nao, n_int_occ)
        - 'W_tilde_occ': external occupied orbitals in AO basis (nao, n_ext_occ)
        - 'W_virt': internal virtual orbitals in AO basis (nao, n_int_virt)
        - 'W_tilde_virt': external virtual orbitals in AO basis (nao, n_ext_virt)
        - 'internal_virt': indices of internal virtual orbitals
        - 'external_occ': indices of external occupied orbitals
        - 'W_tilde_virt': external virtual orbitals in AO basis (nao, n_ext_virt)
    """
    logger.info("\n" + "="*60)
    logger.info("Step 4: Constructing Local Active Space for Each Fragment")
    logger.info("="*60)
    
    nocc = mol.nelectron // 2
    nao = iao_coeff.shape[0]
    niao = iao_coeff.shape[1]
    nvirt = nao - nocc  # Total number of virtual AO/MO functions
    
    logger.info(f"\nActive space construction parameters:")
    logger.info(f"  Fragment size: {fragment_size} atom(s)")
    logger.info(f"  SVD threshold: {svd_threshold}")
    logger.info(f"  Number of occupied MOs: {nocc}")
    logger.info(f"  Number of virtual MOs: {nvirt}")
    logger.info(f"  Total AO basis functions: {nao}")
    logger.info(f"  Total IAOs: {niao}")
    
    # Transform IAO coefficients to MO basis
    # C_IAO_MO = C_MO^T @ S @ C_IAO
    S_ao = mf.get_ovlp()
    C_mo = mf.mo_coeff  # (nao, nmo)
    C_iao_mo = C_mo.T @ S_ao @ iao_coeff  # (nmo, niao)
    
    # Split into occupied and virtual blocks
    C_iao_mo_occ = C_iao_mo[:nocc, :]  # (nocc, niao)
    C_iao_mo_virt = C_iao_mo[nocc:, :]  # (nvirt, niao)
    
    logger.info(f"\nIAO in MO basis:")
    logger.info(f"  C_IAO_MO (occupied): {C_iao_mo_occ.shape}")
    logger.info(f"  C_IAO_MO (virtual): {C_iao_mo_virt.shape}")
    
    # Group atoms into fragments
    natoms = mol.natm
    fragments = []
    for i in range(0, natoms, fragment_size):
        fragment_atoms = list(range(i, min(i + fragment_size, natoms)))
        fragments.append(fragment_atoms)
    
    logger.info(f"  Number of fragments: {len(fragments)}")
    
    local_spaces = []
    
    for frag_idx, frag_atoms in enumerate(fragments):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fragment {frag_idx}: Atoms {frag_atoms}")
        logger.info(f"{'='*60}")
        
        # Get IAO indices for this fragment by matching atom indices
        # New label format: "atom_idx atom_symbol orbital_type" (e.g., "0 O 1s")
        frag_iao_indices = []
        for iatom in frag_atoms:
            for i, label in enumerate(iao_labels):
                # Parse label to get atom index
                parts = label.split()
                if len(parts) >= 3:
                    label_atom_idx = int(parts[0])
                    if label_atom_idx == iatom:
                        frag_iao_indices.append(i)
        
        logger.info(f"Fragment IAO indices: {frag_iao_indices}")
        logger.info(f"Number of fragment IAOs: {len(frag_iao_indices)}")
        
        if len(frag_iao_indices) == 0:
            logger.warning(f"No IAOs found for fragment {frag_idx}, skipping")
            continue
        
        # Extract U_occ(F): (nocc, nLO(F)) and U_virt(F): (nvirt, nLO(F))
        # These are the submatrices of C_IAO_MO with columns corresponding to fragment IAOs
        U_occ_F = C_iao_mo_occ[:, frag_iao_indices]  # (nocc, n_frag_iao)
        U_virt_F = C_iao_mo_virt[:, frag_iao_indices]  # (nvirt, n_frag_iao)
        
        logger.info(f"\nU_occ(F) shape: {U_occ_F.shape} - (N_occ × nLO(F))")
        logger.info(f"U_virt(F) shape: {U_virt_F.shape} - (N_virt × nLO(F))")
        
        # Perform SVD on U_occ(F) with full_matrices=True to get complete orthonormal basis
        # U_occ(F) = W_occ @ diag(s_occ) @ Vh_occ
        # W_occ: (nocc, nocc) - complete orthonormal basis in occupied space
        # s_occ: (n_frag_iao,) - singular values
        
        if U_occ_F.shape[1] > 0:
            W_occ_full, s_occ, Vh_occ = np.linalg.svd(U_occ_F, full_matrices=True)
            
            logger.info(f"\nOccupied space SVD:")
            logger.info(f"  W_occ_full shape: {W_occ_full.shape} - (N_occ, N_occ)")
            logger.info(f"  Singular values (nLO(F)={len(s_occ)}): {s_occ}")
            
            # Columns with s > threshold: internal space
            # Remaining columns: external space (orthogonal complement)
            internal_occ_mask = s_occ > svd_threshold
            n_int_occ = np.sum(internal_occ_mask)
            n_ext_occ = nocc - n_int_occ
            
            # W_occ_MO: internal occupied in MO space (nocc, n_int_occ)
            # W_tilde_occ_MO: external occupied in MO space (nocc, n_ext_occ)
            W_occ_MO = W_occ_full[:, :len(s_occ)][:, internal_occ_mask]
            W_tilde_occ_MO = W_occ_full[:, len(s_occ):]  # All columns after the fragment IAO columns
            # Also include external columns from within the fragment IAO subspace
            if np.sum(~internal_occ_mask) > 0:
                W_tilde_occ_MO = np.hstack([
                    W_occ_full[:, :len(s_occ)][:, ~internal_occ_mask],
                    W_occ_full[:, len(s_occ):]
                ])
            
            logger.info(f"  Internal occupied: {n_int_occ} orbitals (s > {svd_threshold})")
            logger.info(f"  External occupied: {n_ext_occ} orbitals")
            logger.info(f"  W_occ_MO shape: {W_occ_MO.shape}")
            logger.info(f"  W_tilde_occ_MO shape: {W_tilde_occ_MO.shape}")
        else:
            # No fragment IAOs - all occupied orbitals are external
            W_occ_MO = np.zeros((nocc, 0))
            W_tilde_occ_MO = np.eye(nocc)
            n_int_occ = 0
            n_ext_occ = nocc
            logger.info(f"\nNo occupied IAOs in fragment - all {nocc} occupied orbitals are external")
        
        # Same for virtual space
        if U_virt_F.shape[1] > 0:
            W_virt_full, s_virt, Vh_virt = np.linalg.svd(U_virt_F, full_matrices=True)
            
            logger.info(f"\nVirtual space SVD:")
            logger.info(f"  W_virt_full shape: {W_virt_full.shape} - (N_virt, N_virt)")
            logger.info(f"  Singular values (nLO(F)={len(s_virt)}): {s_virt}")
            
            internal_virt_mask = s_virt > svd_threshold
            n_int_virt = np.sum(internal_virt_mask)
            n_ext_virt = nvirt - n_int_virt
            
            W_virt_MO = W_virt_full[:, :len(s_virt)][:, internal_virt_mask]
            W_tilde_virt_MO = W_virt_full[:, len(s_virt):]
            if np.sum(~internal_virt_mask) > 0:
                W_tilde_virt_MO = np.hstack([
                    W_virt_full[:, :len(s_virt)][:, ~internal_virt_mask],
                    W_virt_full[:, len(s_virt):]
                ])
            
            logger.info(f"  Internal virtual: {n_int_virt} orbitals (s > {svd_threshold})")
            logger.info(f"  External virtual: {n_ext_virt} orbitals")
            logger.info(f"  W_virt_MO shape: {W_virt_MO.shape}")
            logger.info(f"  W_tilde_virt_MO shape: {W_tilde_virt_MO.shape}")
        else:
            # No fragment IAOs - all virtual orbitals are external
            W_virt_MO = np.zeros((nvirt, 0))
            W_tilde_virt_MO = np.eye(nvirt)
            n_int_virt = 0
            n_ext_virt = nvirt
            logger.info(f"\nNo virtual IAOs in fragment - all {nvirt} virtual orbitals are external")
        
        
        logger.info(f"\nTransformation matrices in AO basis:")
        logger.info(f"  W_occ shape: {W_occ_MO.shape} - (nao, n_int_occ)")
        logger.info(f"  W_tilde_occ shape: {W_tilde_occ_MO.shape} - (nao, n_ext_occ)")
        logger.info(f"  W_virt shape: {W_virt_MO.shape} - (nao, n_int_virt)")
        logger.info(f"  W_tilde_virt shape: {W_tilde_virt_MO.shape} - (nao, n_ext_virt)")

        logger.info(f"\nActive space summary:")
        logger.info(f"  Atoms: {frag_atoms}")
        logger.info(f"  Active space size: {n_int_occ + n_int_virt} orbitals")
        logger.info(f"    ({n_int_occ} occ + {n_int_virt} virt)")
        
        # Determine orbital indices for internal and external spaces
        # Internal occupied: IAO indices with significant overlap (s > threshold)
        # External occupied: all other occupied orbitals
        if U_occ_F.shape[1] > 0:
            internal_occ = [i for i, mask in enumerate(internal_occ_mask) if mask]
            # External occupied orbitals are the complement
            all_occ = list(range(nocc))
            # Note: in IAO basis, we work with IAO indices, not MO indices
            # But for tracking, we use the MO-space indexing
            external_occ = list(range(n_int_occ, nocc))
        else:
            internal_occ = []
            external_occ = list(range(nocc))
        
        # Same for virtual
        if U_virt_F.shape[1] > 0:
            internal_virt = [i for i, mask in enumerate(internal_virt_mask) if mask]
            external_virt = list(range(n_int_virt, nvirt))
        else:
            internal_virt = []
            external_virt = list(range(nvirt))
        
        # Store fragment's local space information
        local_space = {
            'fragment_id': frag_idx,
            'atoms': frag_atoms,
            'iao_indices': frag_iao_indices,
            'n_int_occ': n_int_occ,
            'n_int_virt': n_int_virt,
            'n_ext_occ': n_ext_occ,
            'n_ext_virt': n_ext_virt,
            'internal_occ': internal_occ,  # Indices in fragment local space
            'external_occ': external_occ,  # Indices in fragment local space
            'internal_virt': internal_virt,  # Indices in fragment local space
            'external_virt': external_virt,  # Indices in fragment local space
            'W_occ': W_occ_MO,
            'W_tilde_occ': W_tilde_occ_MO,
            'W_virt': W_virt_MO,
            'W_tilde_virt': W_tilde_virt_MO,
        }
        
        local_spaces.append(local_space)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Local Active Space Construction Summary")
    logger.info(f"{'='*60}")
    for ls in local_spaces:
        logger.info(f"\nFragment {ls['fragment_id']}:")
        logger.info(f"  Atoms: {ls['atoms']}")
        logger.info(f"  Active space size: {ls['n_int_occ'] + ls['n_int_virt']} orbitals")
        logger.info(f"    ({ls['n_int_occ']} occ + {ls['n_int_virt']} virt)")
    logger.info("="*60 + "\n")
    
    return local_spaces


def compute_lno_compression(density_matrix, eta_threshold, space_name=""):
    """
    Compress a space using Local Natural Orbital (LNO) analysis.
    
    This function diagonalizes a density matrix, sorts eigenvalues in descending order,
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


def create_empty_lno_result(n_ext):
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


def compute_mp2_lno_compression(mol, mf, local_spaces, eta_occ=1e-6, eta_virt=1e-6):
    """
    Compress external spaces using MP2 Local Natural Orbitals (LNOs).
    
    This function implements MP2-LNO compression as described in the paper:
    1. Compute MP1 amplitudes using semi-canonical orbitals
    2. Build MP2 density matrices in external space (occupied and virtual)
    3. Diagonalize density matrices to obtain LNOs
    4. Select active LNOs based on eigenvalue thresholds
    
    The function handles occupied and virtual spaces independently. If one external
    space is empty (n_ext_occ=0 or n_ext_virt=0), compression is still performed
    on the non-empty space.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    local_spaces : list of dict
        Local active spaces from construct_local_active_space
    eta_occ : float, optional
        Threshold for occupied LNO eigenvalues (default: 1e-6)
    eta_virt : float, optional
        Threshold for virtual LNO eigenvalues (default: 1e-6)
    
    Returns
    -------
    local_spaces_lno : list of dict
        Updated local spaces with LNO-compressed external spaces, including:
        - 'X_occ', 'X_virt': LNO transformation matrices
        - 'lno_occ_indices', 'lno_virt_indices': indices of selected active LNOs
        - 'n_lno_occ', 'n_lno_virt': number of selected LNOs
    """
    logger.info("\n" + "="*60)
    logger.info("Step 5: Compressing External Spaces with MP2 LNOs")
    logger.info("="*60)
    
    nocc = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    nvirt = nmo - nocc
    
    logger.info(f"\nMP2-LNO parameters:")
    logger.info(f"  Occupied threshold (eta_occ): {eta_occ}")
    logger.info(f"  Virtual threshold (eta_virt): {eta_virt}")
    logger.info(f"  Number of occupied MOs: {nocc}")
    logger.info(f"  Number of virtual MOs: {nvirt}")
    logger.info(f"  Total MOs: {nmo}")
    
    # Get MO energies from mean-field object
    eps_mo = mf.mo_energy  # All MO energies
    eps_occ_mo = eps_mo[:nocc]  # Occupied
    eps_virt_mo = eps_mo[nocc:]  # Virtual
    
    logger.info(f"\nMO energies:")
    logger.info(f"  Occupied: {eps_occ_mo.shape}")
    logger.info(f"  Virtual: {eps_virt_mo.shape}")
    
    # Get ERIs in MO basis using PySCF's ao2mo module
    # Note: PySCF uses chemist's notation (pq|rs)
    logger.info(f"\nComputing ERIs in MO basis...")
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff, compact=False)
    eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
    logger.info(f"  ERI shape in MO basis: {eri_mo.shape}")
    
    # Process each fragment
    local_spaces_lno = []
    
    for frag_idx, ls in enumerate(local_spaces):
        logger.info(f"\n--- Fragment {frag_idx}: LNO Compression ---")
        
        # Get W transformation matrices from MO basis to internal/external spaces
        # W_occ: (nocc, n_int_occ) - transforms MO occupied to internal occupied
        # W_tilde_occ: (nocc, n_ext_occ) - transforms MO occupied to external occupied
        # Similar for virtual
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
        
        # Skip MP2 if no internal occupied orbitals (can't do correlation)
        if n_int_occ == 0:
            logger.info(f"  Skipping MP2: no internal occupied orbitals")
            ls_lno = ls.copy()
            ls_lno.update({
                'n_lno_occ': 0,
                'n_lno_virt': 0,
                'lno_occ_indices': np.array([], dtype=int),
                'lno_virt_indices': np.array([], dtype=int),
                'X_occ': np.eye(n_ext_occ) if n_ext_occ > 0 else np.zeros((0, 0)),
                'X_virt': np.eye(n_ext_virt) if n_ext_virt > 0 else np.zeros((0, 0)),
            })
            local_spaces_lno.append(ls_lno)
            continue
        
        # Compute energies for internal occupied orbitals from Fock matrix
        # Transform Fock to internal occupied basis: epsilon_iF = diag(W_occ^T @ F_MO @ W_occ)
        f_mo = np.diag(eps_mo)  # Fock in MO basis (diagonal for canonical MOs)
        f_int_occ = W_occ.T @ f_mo[:nocc, :nocc] @ W_occ
        eps_int_occ = np.diag(f_int_occ)  # epsilon_iF for internal occupied
        
        logger.info(f"\n  Orbital energies:")
        logger.info(f"    Internal occupied (epsilon_iF): {eps_int_occ.shape}")
        logger.info(f"      Mean: {np.mean(eps_int_occ):.6f}, Min: {np.min(eps_int_occ):.6f}, Max: {np.max(eps_int_occ):.6f}")
        
        # For other indices (i,j,a,b) we use MO energies directly
        eps_j = eps_occ_mo  # All occupied MO energies
        eps_a = eps_virt_mo  # All virtual MO energies
        eps_b = eps_virt_mo  # All virtual MO energies
        
        # Check if we can perform MP2 - need both external occ and virt for MP2 amplitudes
        if n_ext_occ == 0 and n_ext_virt == 0:
            logger.info(f"  Skipping MP2: both external spaces are empty (n_ext_occ={n_ext_occ}, n_ext_virt={n_ext_virt})")
            ls_lno = ls.copy()
            occ_result = create_empty_lno_result(n_ext_occ)
            virt_result = create_empty_lno_result(n_ext_virt)
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
        # t_{iF,a,j,b} = V_{iF,a,j,b} / (epsilon_iF + epsilon_j - epsilon_a - epsilon_b)
        # where V_{iF,a,j,b} is the ERI in mixed basis
        
        logger.info(f"\n  Computing MP1 amplitudes...")
        logger.info(f"    Transforming ERIs to mixed basis...")
        
        # Transform ERIs: V_{iF,a,j,b} = sum_i W_{i,iF} * ERI_{i,a,j,b}
        # ERIs are in chemist notation: (pq|rs) -> eri_mo[p,q,r,s]
        # We need: V_{iF,a,j,b} = (iF a | j b) in chemist notation
        
        # First, transform first occupied index from MO to internal occupied
        # V_{iF,a,j,b} = sum_i W_occ[i, iF] * eri_mo[i, a+nocc, j, b+nocc]
        # Shape: (n_int_occ, nvirt, nocc, nvirt)
        
        # Extract occupied-virtual-occupied-virtual block from ERI
        # eri_mo[i, j, k, l] in chemist notation (ij|kl)
        # We need (ia|jb) = eri_mo[i, a+nocc, j, b+nocc]
        eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]  # (nocc, nvirt, nocc, nvirt)
        
        # Transform first index: (iF a | j b)
        V_Iajb = np.einsum('iI,iajb->Iajb', W_occ, eri_ovov, optimize=True)
        logger.info(f"    Transformed ERI shape: {V_Iajb.shape} = ({n_int_occ}, {nvirt}, {nocc}, {nvirt})")
        
        # Compute energy denominators
        # denom[iF, a, j, b] = eps_iF + eps_j - eps_a - eps_b
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
        # Equation 20a: D_jj'^(F) = 2 * sum_{kF,a,b} t*_{i,a,kF,b} * (2*t_{j,a,kF,b} - t_{j,b,kF,a})
        # Equation 20b: D_ab^(F) = sum_{iF,j,c} 2*(t*_{iF,a,j,c}*t_{iF,b,j,c} + t*_{iF,c,j,a}*t_{iF,c,j,b})
        #                                    - (t*_{iF,c,j,a}*t_{iF,b,j,c} + t*_{iF,a,j,c}*t_{iF,c,j,b})
        
        logger.info(f"\n  Building MP2 density matrices...")
        
        # Occupied density: D_jj'^(F) in full occupied MO space
        # Equation 20a: D_jj' = 2 * sum_{kF,a,b} t*_{kF,a,j,b} * (2*t_{kF,a,j',b} - t_{kF,b,j',a})
        # t_Iajb has shape (n_int_occ, nvirt, nocc, nvirt) with indices [I=kF, a, j, b]
        
        # First term: 2 * sum_{I,a,b} t*[I,a,j,b] * t[I,a,j',b]
        # Both j and j' index into the nocc dimension (index 2 of t_Iajb)
        term1_full = 2.0 * np.einsum('Iajb,IaJb->jJ', np.conj(t_Iajb), t_Iajb, optimize=True)
        
        # Second term: sum_{I,a,b} t*[I,a,j,b] * t[I,b,j',a]
        # Note: we swap a<->b in the second tensor
        term2_full = np.einsum('Iajb,IbJa->jJ', np.conj(t_Iajb), t_Iajb, optimize=True)
        
        D_occ_full = 2.0 * (term1_full - term2_full)
        
        logger.info(f"    D_occ (full MO) shape: {D_occ_full.shape}")
        logger.info(f"    D_occ norm: {np.linalg.norm(D_occ_full):.6f}")
        
        # Virtual density: D_ab^(F) in full virtual MO space
        # Equation 20b: D_ab = sum_{iF,j,c} 2*(t*_{iF,a,j,c}*t_{iF,b,j,c} + t*_{iF,c,j,a}*t_{iF,c,j,b})
        #                                 - (t*_{iF,c,j,a}*t_{iF,b,j,c} + t*_{iF,a,j,c}*t_{iF,c,j,b})
        # t_Iajb has shape (n_int_occ, nvirt, nocc, nvirt) with indices [I=iF, a, j, b]
        
        # First group: 2*(term1 + term2)
        # term1: sum_{I,j,c} t*[I,a,j,c] * t[I,b,j,c]
        term1 = np.einsum('Iajc,Ibjc->ab', np.conj(t_Iajb), t_Iajb, optimize=True)
        # term2: sum_{I,j,c} t*[I,c,j,a] * t[I,c,j,b]
        term2 = np.einsum('Icja,Icjb->ab', np.conj(t_Iajb), t_Iajb, optimize=True)
        
        # Second group: (term3 + term4)
        # term3: sum_{I,j,c} t*[I,c,j,a] * t[I,b,j,c]
        term3 = np.einsum('Icja,Ibjc->ab', np.conj(t_Iajb), t_Iajb, optimize=True)
        # term4: sum_{I,j,c} t*[I,a,j,c] * t[I,c,j,b]
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
            occ_result = compute_lno_compression(D_occ_ext, eta_occ, "occupied")
        else:
            logger.info(f"    Skipping occupied LNO compression (n_ext_occ=0)")
            occ_result = create_empty_lno_result(n_ext_occ)
        
        # Handle virtual space compression
        if n_ext_virt > 0:
            D_virt_ext = W_tilde_virt.T @ D_virt_full @ W_tilde_virt
            logger.info(f"    D_virt_ext shape: {D_virt_ext.shape}")
            logger.info(f"\n  Diagonalizing virtual space for LNOs...")
            virt_result = compute_lno_compression(D_virt_ext, eta_virt, "virtual")
        else:
            logger.info(f"    Skipping virtual LNO compression (n_ext_virt=0)")
            virt_result = create_empty_lno_result(n_ext_virt)
        
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
            compression_ratio = 1-n_ext_after / n_ext_before
            logger.info(f"  Compression ratio: {compression_ratio:.2%} ({n_ext_after}/{n_ext_before})")
        else:
            logger.info(f"  No external space")
    
    logger.info(f"\nOverall compression:")
    logger.info(f"  Total external orbitals before: {total_ext_before}")
    logger.info(f"  Total external orbitals after LNO: {total_ext_after}")
    if total_ext_before > 0:
        overall_ratio = 1-total_ext_after / total_ext_before
        logger.info(f"  Overall compression ratio: {overall_ratio:.2%} ({total_ext_after}/{total_ext_before})")
    else:
        logger.info(f"  No external space to compress")
    
    logger.info(f"\nLargest active space (Fragment {max_active_frag_id}):")
    logger.info(f"  Occupied: {max_active_occ}")
    logger.info(f"  Virtual: {max_active_virt}")
    logger.info(f"  Total: {max_active_total}")
    logger.info("="*60 + "\n")
    
    return local_spaces_lno


def construct_local_ham(mol, mf, local_space_lno):
    """
    Construct local Hamiltonian (Fock and ERIs) in active space for a single fragment.
    
    The active space comprises:
    - Internal orbitals (from SVD with large singular values)
    - Active LNOs (external orbitals selected by MP2-LNO compression)
    
    This follows the paper's orbital rotation scheme:
    - Occupied and virtual projection of local orbitals (internal - red in paper)
    - Occupied and virtual active LNOs (blue in paper)
    - Frozen LNOs are excluded (gray in paper)
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    local_space_lno : dict
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
    logger.info(f"\n  Constructing local Hamiltonian for fragment {local_space_lno['fragment_id']}...")
    
    # Extract fragment information
    W_occ = local_space_lno['W_occ']  # (nocc, n_int_occ)
    W_tilde_occ = local_space_lno['W_tilde_occ']  # (nocc, n_ext_occ)
    W_virt = local_space_lno['W_virt']  # (nvirt, n_int_virt)
    W_tilde_virt = local_space_lno['W_tilde_virt']  # (nvirt, n_ext_virt)
    
    X_occ = local_space_lno['X_occ']  # (n_ext_occ, n_ext_occ)
    X_virt = local_space_lno['X_virt']  # (n_ext_virt, n_ext_virt)
    
    lno_occ_indices = local_space_lno['lno_occ_indices']  # Indices of selected occ LNOs
    lno_virt_indices = local_space_lno['lno_virt_indices']  # Indices of selected virt LNOs
    
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
        logger.warning(f"    Invalid active space (need occ>0 and virt>0)")
        return None
    
    # Build transformation matrices U_act from MO to active space
    # U_act_occ: (nocc, n_active_occ) = [W_occ | W_tilde_occ @ X_occ[:, lno_occ_indices]]
    # U_act_virt: (nvirt, n_active_virt) = [W_virt | W_tilde_virt @ X_virt[:, lno_virt_indices]]
    
    nocc = mol.nelectron // 2
    nvirt = mf.mo_coeff.shape[1] - nocc
    
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
        U_ext_occ_full = W_tilde_occ @ X_occ  # (nocc, n_ext_occ) - all external LNOs
    else:
        U_ext_occ_full = np.zeros((nocc, 0))
    
    if n_ext_virt > 0:
        U_ext_virt_full = W_tilde_virt @ X_virt  # (nvirt, n_ext_virt) - all external LNOs
    else:
        U_ext_virt_full = np.zeros((nvirt, 0))
    
    # Combine: [internal | all_external_LNOs]
    if n_int_occ > 0 and n_ext_occ > 0:
        U_full_occ = np.hstack([U_occ_int, U_ext_occ_full])
    elif n_int_occ > 0:
        U_full_occ = U_occ_int
    elif n_ext_occ > 0:
        U_full_occ = U_ext_occ_full
    else:
        U_full_occ = np.zeros((nocc, 0))
    
    if n_int_virt > 0 and n_ext_virt > 0:
        U_full_virt = np.hstack([U_virt_int, U_ext_virt_full])
    elif n_int_virt > 0:
        U_full_virt = U_virt_int
    elif n_ext_virt > 0:
        U_full_virt = U_ext_virt_full
    else:
        U_full_virt = np.zeros((nvirt, 0))
    
    n_full_occ = U_full_occ.shape[1]
    n_full_virt = U_full_virt.shape[1]
    n_full = n_full_occ + n_full_virt
    
    logger.info(f"    Full extended basis: {n_full_occ} occ + {n_full_virt} virt = {n_full}")
    
    # Transform core Hamiltonian and ERIs to extended basis
    logger.info(f"    Transforming to extended basis (active + frozen)...")
    
    # Get core Hamiltonian in MO basis
    h_core_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    h_mo = mf.mo_coeff.T @ h_core_ao @ mf.mo_coeff
    
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
    nmo = mf.mo_coeff.shape[1]
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff, compact=False)
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
        # Sum over frozen occupied: i_bar in [frozen_occ_start : frozen_occ_end]
        # Vectorized: f_pq += Σ_i_bar (2*V_pqii - V_piiq)
        frozen_indices = slice(frozen_occ_start, frozen_occ_end)
        
        # Coulomb: 2 * Σ_i V[:,:,i,i]
        coulomb = 2.0 * np.sum(eri_full[:, :, frozen_indices, frozen_indices].diagonal(axis1=2, axis2=3), axis=2)
        
        # Exchange: Σ_i V[:,i,i,:] - needs transpose because we want V_piiq
        exchange = np.sum(eri_full[:, frozen_indices, frozen_indices, :].diagonal(axis1=1, axis2=2), axis=2)
        
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


def solve_fragment_ccsd(mol, mf, local_spaces_lno):
    """
    Solve fragment CCSD equations for each fragment.
    
    This function:
    1. Constructs local Hamiltonian for each fragment using construct_local_ham
    2. Sets up and runs CCSD calculation in active space
    3. Computes energy contribution matrix E_{ii'} in active space
    4. Returns results for energy computation in IAO basis (done separately)
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    local_spaces_lno : list of dict
        Local active spaces with LNO compression from compute_mp2_lno_compression
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'fragment_results': list of fragment CCSD results
        - 'fragment_hamiltonians': list of local Hamiltonians
    """
    logger.info("\n" + "="*60)
    logger.info("Step 6: Solving Fragment CCSD Equations")
    logger.info("="*60)
    
    logger.info(f"\nFragment CCSD setup:")
    logger.info(f"  Number of fragments: {len(local_spaces_lno)}")
    
    fragment_results = []
    fragment_hamiltonians = []
    
    for frag_idx, ls in enumerate(local_spaces_lno):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fragment {frag_idx}: CCSD Calculation")
        logger.info(f"{'='*60}")
        
        # Construct local Hamiltonian in active space
        local_ham = construct_local_ham(mol, mf, ls)
        
        if local_ham is None:
            logger.warning(f"  Skipping fragment {frag_idx}: invalid active space")
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
        mf_frag._eri = eri_active
        mf_frag.get_hcore = lambda *args: f_active
        mf_frag.get_ovlp = lambda *args: np.eye(n_active)
        
        # Run CCSD
        logger.info(f"  Running CCSD...")
        mycc = cc.CCSD(mf_frag)
        mycc.verbose = 0
        mycc.kernel()
        
        if mycc.converged:
            logger.info(f"    CCSD converged!")
            logger.info(f"    CCSD correlation energy: {mycc.e_corr:.8f} a.u.")
        else:
            logger.warning(f"    CCSD did not converge!")
        
        # Get amplitudes in active space
        t1_act = mycc.t1  # (n_active_occ, n_active_virt)
        t2_act = mycc.t2  # (n_active_occ, n_active_occ, n_active_virt, n_active_virt)
        
        logger.info(f"    T1 amplitude shape: {t1_act.shape}, norm: {np.linalg.norm(t1_act):.6f}")
        logger.info(f"    T2 amplitude shape: {t2_act.shape}, norm: {np.linalg.norm(t2_act):.6f}")
        
        # Compute tau intermediate: τ_{iajb} = t_{iajb} - t_{ia}*t_{jb}
        # Note: t2_act has indices [i,j,a,b] but we need τ with indices [i,a,j,b]
        # Following Equation 13, we need the difference, not the sum
        # t1_outer[i,a,j,b] = t_ia * t_jb
        t1_outer = np.einsum('ia,jb->iajb', t1_act, t1_act, optimize=True)
        # tau_act[i,a,j,b] = t2_act[i,j,a,b] - t1_outer[i,a,j,b]
        tau_act = t2_act.transpose(0, 2, 1, 3) - t1_outer  # Reorder t2 from ijab to iajb
        
        # Compute energy contribution matrix E_{ii'} using Equation 13:
        # E_{ii'}^{(F)} = Σ_{jab∈A_F} τ_{iajb}^{(F)} (2V_{i'ajb} - V_{i'bja})
        # In chemist notation: V_{i'ajb} = eri[i',a,j,b], V_{i'bja} = eri[i',b,j,a]
        logger.info(f"  Computing energy contribution matrix E_{{ii'}} using Equation 13...")
        
        # Extract occupied-virtual blocks of ERIs in active space
        # eri_active has shape (n_active, n_active, n_active, n_active)
        # We need: occ x virt x occ x virt blocks
        eri_ovov = eri_active[:n_active_occ, n_active_occ:, :n_active_occ, n_active_occ:]  # (nocc, nvirt, nocc, nvirt)
        
        # E[i, i'] = Σ_{jab} τ[i,a,j,b] * (2*V[i',a,j,b] - V[i',b,j,a])
        # Using einsum: 'iajb,Iajb->iI' for coulomb, 'iajb,Ibja->iI' for exchange
        coulomb = 2.0 * np.einsum('iajb,Iajb->iI', tau_act, eri_ovov, optimize=True)
        exchange = np.einsum('iajb,Ibja->iI', tau_act, eri_ovov, optimize=True)
        E_ii_prime = coulomb - exchange
        # also add fock and t1 contributions
        # construct active fock blocks
        #eri_vooo = eri_active[n_active_occ:, :n_active_occ, :n_active_occ, :n_active_occ]  # (nvirt, nocc, nocc, nocc)
        #f_ia = f_active[:n_active_occ, n_active_occ:]
        #f_ia += 2*np.einsum('aijj->ia', eri_vooo)
        #f_ia -= np.einsum('ajji->ia', eri_vooo)

        #E_ii_prime += np.einsum('ia,Ia->iI', t1_act, f_ia, optimize=True)

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
            'E_ii_prime': E_ii_prime,  # Energy matrix in active space
            'e_corr_ccsd': mycc.e_corr,  # CCSD correlation energy in active space
        }
        
        fragment_results.append(frag_result)
        fragment_hamiltonians.append(local_ham)
    
    logger.info("\n" + "="*60)
    logger.info("Fragment CCSD Completed")
    logger.info("="*60)
    logger.info(f"Successfully computed {sum(1 for r in fragment_results if r is not None)} fragments")
    logger.info("="*60 + "\n")
    
    results = {
        'fragment_results': fragment_results,
        'fragment_hamiltonians': fragment_hamiltonians,
    }
    
    return results


def compute_fragment_energies(mol, mf, iao_coeff, local_spaces_lno, ccsd_results):
    """
    Compute fragment correlation energies by transforming E_{ii'} to fragment IAO basis.
    
    For each fragment, the correlation energy is computed by:
    1. Transform E_{ii'} from active space to MO basis using U_act_occ
    2. Extract IAO columns belonging to this fragment from C_iao_mo_occ
    3. Transform E_mo to fragment IAO basis: E_frag = C_frag^T @ E_mo @ C_frag
    4. Sum all diagonal elements (trace): E_F = Tr(E_frag) = Σ_α E_frag[α,α]
    
    This approach correctly contracts over occupied indices i,i' while summing over
    all IAOs α belonging to the fragment.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    iao_coeff : np.ndarray
        IAO coefficients in AO basis (nao, niao)
    local_spaces_lno : list of dict
        Local active spaces with LNO compression
    ccsd_results : dict
        Results from solve_fragment_ccsd containing fragment_results and fragment_hamiltonians
        
    Returns
    -------
    energy_results : dict
        Dictionary containing:
        - 'fragment_energies': list of correlation energy for each fragment
        - 'total_correlation': total correlation energy (sum over fragments)
        - 'total_energy': RHF energy + correlation energy
    """
    logger.info("\n" + "="*60)
    logger.info("Computing Fragment Correlation Energies")
    logger.info("="*60)
    
    fragment_results = ccsd_results['fragment_results']
    fragment_hamiltonians = ccsd_results['fragment_hamiltonians']
    
    # Get overlap matrix and transform IAO to MO basis
    S_ao = mf.get_ovlp()
    C_mo = mf.mo_coeff  # (nao, nmo)
    
    # C_iao_mo = C_mo^T @ S_ao @ C_iao
    C_iao_mo = C_mo.T @ S_ao @ iao_coeff  # (nmo, niao)
    
    nocc = mol.nelectron // 2
    C_iao_mo_occ = C_iao_mo[:nocc, :]  # (nocc, niao)
    
    logger.info(f"\nTransformation matrices:")
    logger.info(f"  IAO-MO (occupied): {C_iao_mo_occ.shape}")
    
    fragment_energies = []
    
    for frag_idx, (frag_result, local_ham, ls) in enumerate(
        zip(fragment_results, fragment_hamiltonians, local_spaces_lno)
    ):
        logger.info(f"\nFragment {frag_idx}:")
        
        if frag_result is None or local_ham is None:
            logger.info(f"  Skipped (no CCSD result)")
            fragment_energies.append(0.0)
            continue
        
        # Get E_{ii'} in active space and transformation matrix
        E_ii_prime = frag_result['E_ii_prime']  # (n_active_occ, n_active_occ)
        U_act_occ = local_ham['U_act_occ']  # (nocc, n_active_occ)
        
        # Transform E_{ii'} from active space to MO occupied space
        # E_MO = U_act_occ @ E_{ii'} @ U_act_occ^T
        E_mo = U_act_occ @ E_ii_prime @ U_act_occ.T  # (nocc, nocc)
        
        logger.info(f"  E_MO shape: {E_mo.shape}, norm: {np.linalg.norm(E_mo):.6f}")
        
        # Get IAO indices for this fragment
        frag_iao_indices = ls['iao_indices']
        
        # Extract the slice of C_iao_mo_occ corresponding to this fragment's IAOs
        # C_iao_mo_occ has shape (nocc, niao), we want columns for this fragment
        C_frag = C_iao_mo_occ[:, frag_iao_indices]  # (nocc, n_frag_iao)
        
        logger.info(f"  Fragment IAO indices: {frag_iao_indices}")
        logger.info(f"  C_frag shape: {C_frag.shape}")
        
        # Transform E_mo to fragment IAO basis:
        # Contract C_frag^T from left and C_frag from right over occupied indices i and i'
        # E_frag[α,β] = Σ_{ii'} C_frag[i,α] * E_mo[i,i'] * C_frag[i',β]
        # This gives E in the IAO basis restricted to this fragment
        E_frag = C_frag.T @ E_mo @ C_frag  # (n_frag_iao, n_frag_iao)
        logger.info(f"  Transforming E_MO to fragment IAO basis...")
        logger.info(E_frag)
        
        logger.info(f"  E_frag shape: {E_frag.shape}, norm: {np.linalg.norm(E_frag):.6f}")
        
        # Sum over all diagonal elements (all IAOs in the fragment)
        # This gives the total correlation energy for this fragment
        fragment_energy = np.trace(E_frag)
        
        logger.info(f"  Fragment correlation energy: {fragment_energy:.8f} a.u.")
        
        fragment_energies.append(fragment_energy)
    
    # Compute total correlation energy
    total_corr = np.sum(fragment_energies)
    total_energy = mf.e_tot + total_corr
    
    logger.info("="*60)
    logger.info("Fragment Energy Summary")
    logger.info("="*60)
    for i, e in enumerate(fragment_energies):
        logger.info(f"  Fragment {i}: {e:.8f} a.u.")
    logger.info(f"\nTotal correlation energy: {total_corr:.8f} a.u.")
    logger.info(f"RHF energy: {mf.e_tot:.8f} a.u.")
    logger.info(f"Total energy: {total_energy:.8f} a.u.")
    logger.info("="*60 + "\n")
    
    energy_results = {
        'fragment_energies': fragment_energies,
        'total_correlation': total_corr,
        'total_energy': total_energy,
    }
    
    return energy_results


def main():
    """
    Main function to run the quantum embedding scheme.
    All steps: Build H10 chain, RHF, IAO, Local Active Space, MP2-LNO, and fragment CCSD.
    """
    print("\n" + "="*70)
    print("Quantum Embedding Scheme for H10 Chain")
    print("Complete workflow: Molecule -> RHF -> IAO -> LAS -> LNO -> CCSD")
    print("="*70 + "\n")
    
    # Step 1: Build H10 chain molecule
    separation = 1.5  # Angstrom, typical H-H distance in molecules
    basis = 'cc-pvdz'
    
    #mol = build_mol(separation=separation, basis=basis)
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="cc-pvdz"
    )
    
    # Step 2: Run RHF calculation
    mf = run_rhf(mol)
    
    # Analyze the solution
    analyze_rhf_solution(mol, mf)
    
    # Step 3: Construct Intrinsic Atomic Orbitals (IAO)
    iao_coeff, iao_labels = construct_iao(mol, mf, minao='minao')
    
    # Transform Fock and ERIs to IAO basis (needed for all subsequent steps)
    logger.info(f"\nTransforming Fock matrix and ERIs to IAO basis...")
    
    
    # Step 4: Construct Local Active Space for each fragment
    # Use fragment_size=1 to treat each H atom as a separate fragment
    # Or fragment_size=2 to pair up neighboring H atoms
    fragment_size = 1  # Each atom is a fragment
    svd_threshold = 1e-8  # Threshold for determining internal space
    
    local_spaces = construct_local_active_space(
        mol, mf, iao_coeff, iao_labels,
        fragment_size=fragment_size, 
        svd_threshold=svd_threshold
    )
    
    # Step 5: Compress external spaces using MP2-LNO
    eta_occ = 1e-2  # Threshold for occupied LNO eigenvalues
    eta_virt = 1e-3  # Threshold for virtual LNO eigenvalues
    
    local_spaces_lno = compute_mp2_lno_compression(
        mol, mf, local_spaces,
        eta_occ=eta_occ,
        eta_virt=eta_virt
    )
    
    # Step 6: Solve fragment CCSD equations
    ccsd_results = solve_fragment_ccsd(mol, mf, local_spaces_lno)
    
    # Step 7: Compute fragment correlation energies in IAO basis
    energy_results = compute_fragment_energies(mol, mf, iao_coeff, local_spaces_lno, ccsd_results)
    
    # Store results for future use
    results = {
        'mol': mol,
        'mf': mf,
        'separation': separation,
        'basis': basis,
        'energy_rhf': mf.e_tot,
        'mo_coeff': mf.mo_coeff,
        'mo_energy': mf.mo_energy,
        'iao_coeff': iao_coeff,
        'iao_labels': iao_labels,
        'local_spaces': local_spaces,
        'local_spaces_lno': local_spaces_lno,
        'fragment_size': fragment_size,
        'svd_threshold': svd_threshold,
        'eta_occ': eta_occ,
        'eta_virt': eta_virt,
        'ccsd_results': ccsd_results,
        'energy_results': energy_results,
        'fragment_energies': energy_results['fragment_energies'],
        'energy_corr': energy_results['total_correlation'],
        'energy_total': energy_results['total_energy'],
    }
    ccsd_full = run_full_ccsd(mol, mf)
    results['ccsd_full'] = ccsd_full

    logger.info("="*70)
    logger.info("All steps completed successfully!")
    logger.info("="*70)
    logger.info("\nWorkflow summary:")
    logger.info(f"  1. Built H10 chain with {mol.natm} atoms")
    logger.info(f"  2. RHF energy: {mf.e_tot:.8f} a.u.")
    logger.info(f"  3. Constructed {iao_coeff.shape[1]} IAOs")
    logger.info(f"  4. Defined {len(local_spaces)} local active spaces")
    logger.info(f"  5. Compressed external spaces with MP2-LNOs")
    logger.info(f"  6. Solved {len(local_spaces_lno)} fragment CCSD equations")
    logger.info(f"  7. Computed fragment energies in IAO basis")
    logger.info(f"\nFinal energies:")
    logger.info(f"  RHF energy:         {mf.e_tot:.8f} a.u.")
    logger.info(f"  Correlation energy: {energy_results['total_correlation']:.8f} a.u.")
    logger.info(f"  Total energy:       {energy_results['total_energy']:.8f} a.u.")
    logger.info(f"  Full CCSD correlation energy:       {ccsd_full['e_corr']:.8f} a.u.")
    logger.info(f"  Full CCSD total energy:       {ccsd_full['e_total']:.8f} a.u.")
    logger.info("="*70 + "\n")
    
    return results


def run_full_ccsd(mol, mf):
    """
    Run full CCSD calculation on the entire system for reference.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'ccsd': CCSD object
        - 'e_corr': correlation energy
        - 'e_total': total energy
    """
    logger.info("\n" + "="*70)
    logger.info("Running Full CCSD Calculation (Reference)")
    logger.info("="*70)
    
    mycc = cc.CCSD(mf)
    mycc.kernel()
    
    logger.info(f"Full CCSD converged: {mycc.converged}")
    logger.info(f"Full CCSD correlation energy: {mycc.e_corr:.8f} a.u.")
    logger.info(f"Full CCSD total energy: {mycc.e_tot:.8f} a.u.")
    
    return {
        'ccsd': mycc,
        'e_corr': mycc.e_corr,
        'e_total': mycc.e_tot
    }


def convergence_test(mol, mf, iao_coeff, local_spaces, eri_mo,
                     eta_occ_values=None, gamma_values=None):
    """
    Test convergence of fragment CCSD with respect to LNO thresholds.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    iao_coeff : np.ndarray
        IAO coefficients
    local_spaces : list
        Local active spaces
    eri_mo : np.ndarray
        Two-electron integrals in MO basis
    eta_occ_values : list, optional
        List of eta_occ thresholds to test (default: [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    gamma_values : list, optional
        List of gamma = eta_occ/eta_virt ratios (default: [1, 2, 5, 10, 20])
    
    Returns
    -------
    results : dict
        Dictionary containing convergence test results
    """
    if eta_occ_values is None:
        eta_occ_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    
    if gamma_values is None:
        gamma_values = [1, 2, 5, 10, 20]
    
    logger.info("\n" + "="*70)
    logger.info("LNO Threshold Convergence Test")
    logger.info("="*70)
    logger.info(f"Testing eta_occ values: {eta_occ_values}")
    logger.info(f"Testing gamma (eta_occ/eta_virt) values: {gamma_values}")
    
    # Run full CCSD for reference
    full_ccsd = run_full_ccsd(mol, mf)
    e_ref = full_ccsd['e_corr']
    
    # Store results
    results = {
        'full_ccsd': full_ccsd,
        'eta_occ_values': eta_occ_values,
        'gamma_values': gamma_values,
        'energies': [],
        'errors': [],
        'n_orbitals': [],
        'compression_ratios': []
    }
    
    # Test each combination
    for eta_occ in eta_occ_values:
        for gamma in gamma_values:
            eta_virt = eta_occ / gamma
            
            logger.info(f"\n--- Testing: eta_occ={eta_occ:.1e}, gamma={gamma}, eta_virt={eta_virt:.1e} ---")
            
            # Compute LNO compression
            local_spaces_lno = compute_mp2_lno_compression(
                mol=mol,
                mf=mf,
                local_spaces=local_spaces,
                eta_occ=eta_occ,
                eta_virt=eta_virt
            )
            
            # Solve fragment CCSD
            ccsd_results = solve_fragment_ccsd(mol, mf, local_spaces_lno)
            
            # Compute fragment energies
            energy_results = compute_fragment_energies(mol, mf, iao_coeff, local_spaces_lno, ccsd_results)
            
            # Compute error
            e_corr = energy_results['total_correlation']
            error = e_corr - e_ref
            
            # Count total orbitals
            n_orb_total = sum(len(ls['internal_occ']) + len(ls['internal_virt']) + 
                             ls['n_lno_occ'] + ls['n_lno_virt']
                             for ls in local_spaces_lno)
            n_orb_before = sum(len(ls['internal_occ']) + len(ls['internal_virt']) +
                              len(ls['external_occ']) + len(ls['external_virt'])
                              for ls in local_spaces)
            compression = n_orb_total / n_orb_before * 100
            
            logger.info(f"  Correlation energy: {e_corr:.8f} a.u.")
            logger.info(f"  Error vs full CCSD: {error:.8f} a.u. ({error*1e3:.4f} mEh)")
            logger.info(f"  Total orbitals: {n_orb_total} ({compression:.2f}% of original)")
            
            results['energies'].append({
                'eta_occ': eta_occ,
                'gamma': gamma,
                'eta_virt': eta_virt,
                'e_corr': e_corr,
                'error': error
            })
            results['errors'].append(abs(error))
            results['n_orbitals'].append(n_orb_total)
            results['compression_ratios'].append(compression)
    
    return results


def plot_convergence(results, filename='lno_convergence.png'):
    """
    Plot convergence test results.
    
    Parameters
    ----------
    results : dict
        Results from convergence_test function
    filename : str
        Output filename for the plot
    """
    import matplotlib.pyplot as plt
    
    eta_occ_values = results['eta_occ_values']
    gamma_values = results['gamma_values']
    e_ref = results['full_ccsd']['e_corr']
    
    # Organize data by gamma
    data_by_gamma = {gamma: {'eta_occ': [], 'error': [], 'n_orb': []} 
                     for gamma in gamma_values}
    
    for entry in results['energies']:
        gamma = entry['gamma']
        data_by_gamma[gamma]['eta_occ'].append(entry['eta_occ'])
        data_by_gamma[gamma]['error'].append(abs(entry['error']) * 1000)  # Convert to mEh
        
    # Count orbitals for each eta_occ (average over gamma)
    n_orb_by_eta = {}
    for entry in results['energies']:
        eta = entry['eta_occ']
        if eta not in n_orb_by_eta:
            n_orb_by_eta[eta] = []
        # Find corresponding orbital count
        idx = results['energies'].index(entry)
        n_orb_by_eta[eta].append(results['n_orbitals'][idx])
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Error vs eta_occ for different gamma values
    colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_values)))
    for idx, gamma in enumerate(gamma_values):
        data = data_by_gamma[gamma]
        ax1.loglog(data['eta_occ'], data['error'], 'o-', 
                   color=colors[idx], label=f'γ = {gamma}', linewidth=2, markersize=8)
    
    ax1.set_xlabel('η_occ (occupied LNO threshold)', fontsize=12)
    ax1.set_ylabel('|Error| (mEh)', fontsize=12)
    ax1.set_title('Convergence vs LNO Threshold', fontsize=14, fontweight='bold')
    ax1.legend(title='γ = η_occ/η_virt', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1 mEh (chemical accuracy)')
    
    # Plot 2: Error vs number of orbitals
    for idx, gamma in enumerate(gamma_values):
        data = data_by_gamma[gamma]
        # Get corresponding orbital counts
        n_orbs = []
        for eta in data['eta_occ']:
            # Find index in results
            for entry in results['energies']:
                if entry['eta_occ'] == eta and entry['gamma'] == gamma:
                    idx_entry = results['energies'].index(entry)
                    n_orbs.append(results['n_orbitals'][idx_entry])
                    break
        
        ax2.semilogy(n_orbs, data['error'], 'o-', 
                    color=colors[idx], label=f'γ = {gamma}', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Total number of orbitals', fontsize=12)
    ax2.set_ylabel('|Error| (mEh)', fontsize=12)
    ax2.set_title('Error vs Computational Cost', fontsize=14, fontweight='bold')
    ax2.legend(title='γ = η_occ/η_virt', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1 mEh')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"\nConvergence plot saved to: {filename}")
    plt.close()


if __name__ == "__main__":
    # Check if we should run convergence test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--convergence':
        # Run convergence test
        logger.info("Running convergence test mode...")
        
        # Build system
        #mol = build_mol(separation=1.5, basis='cc-pvdz')
        mol = gto.M(
            atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
            basis="cc-pvtz"
            )
        mf = run_rhf(mol)
        iao_coeff, iao_labels = construct_iao(mol, mf, minao='minao')
        
        local_spaces = construct_local_active_space(
            mol=mol,
            mf=mf,
            iao_coeff=iao_coeff,
            iao_labels=iao_labels,
            fragment_size=1,
            svd_threshold=1e-6
        )
        
        # Get ERIs
        mo_coeff = mf.mo_coeff
        eri_mo = ao2mo.kernel(mol, mo_coeff, aosym='s4')
        eri_mo = ao2mo.restore(1, eri_mo, mo_coeff.shape[1])
        
        # Run convergence test
        conv_results = convergence_test(
            mol=mol,
            mf=mf,
            iao_coeff=iao_coeff,
            local_spaces=local_spaces,
            eri_mo=eri_mo,
            eta_occ_values=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
            gamma_values=[1, 5, 10]  # Reduced for faster testing
        )
        
        # Plot results
        plot_convergence(conv_results, filename='lno_convergence.png')
        
        logger.info("\nConvergence test completed!")
    elif len(sys.argv) > 1 and sys.argv[1] == '--full-ccsd':
        # Run full CCSD calculation
        logger.info("Running full CCSD calculation mode...")
        mol = build_mol(separation=1.5, basis='cc-pvdz')
        mf = run_rhf(mol)
        full_ccsd_results = run_full_ccsd(mol, mf)
        logger.info("\nFull CCSD calculation completed!")
    else: 
        # Run standard workflow
        results = main()
