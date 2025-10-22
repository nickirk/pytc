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


def build_h10_chain(separation=1.5, basis='cc-pvdz', unit='Angstrom'):
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
    
    IAOs are localized atomic-like orbitals that span the occupied space and are
    orthogonal to each other. They provide a chemically intuitive basis for 
    fragment-based quantum embedding calculations.
    
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
        IAO coefficients in AO basis, shape (nao, niao)
    iao_labels : list
        Labels for each IAO indicating which atom it belongs to
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
    
    # Build IAOs - these span the occupied space
    iao_coeff = lo.iao.iao(mol, mo_coeff_occ, minao=minao)
    
    # For H10, each H atom contributes 1 IAO (from 1s minimal basis)
    # Total of 10 IAOs for 10 H atoms
    # Create labels: first 5 are occupied IAOs, next 5 are virtual IAOs
    # (IAO also constructs virtual IAOs orthogonal to occupied ones)
    niao = iao_coeff.shape[1]
    iao_labels = []
    
    # For each IAO, determine which atom it belongs to by checking overlap
    s = mol.intor('int1e_ovlp')
    for iao_idx in range(niao):
        iao_vec = iao_coeff[:, iao_idx]
        # Compute <iao | S | ao> for each AO
        overlap_with_aos = s @ iao_vec
        
        # Find which atom this IAO is most localized on
        max_overlap_atom = -1
        max_overlap = 0.0
        for iatom in range(mol.natm):
            atom_label = mol.atom_symbol(iatom)
            # Get AOs for this atom
            ao_labels = mol.ao_labels()
            atom_overlap = 0.0
            for ao_idx, ao_label in enumerate(ao_labels):
                if f'{atom_label}{iatom}' in ao_label:
                    atom_overlap += np.abs(iao_vec[ao_idx])**2
            
            if atom_overlap > max_overlap:
                max_overlap = atom_overlap
                max_overlap_atom = iatom
        
        atom_symbol = mol.atom_symbol(max_overlap_atom)
        iao_labels.append(f'{atom_symbol}{max_overlap_atom}_IAO{iao_idx}')
    
    logger.info(f"\nIAO construction completed:")
    logger.info(f"  Number of IAOs: {niao}")
    logger.info(f"  IAO dimensions: {iao_coeff.shape}")
    logger.info(f"  IAO labels: {iao_labels}")
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
        
        # Get IAO indices for this fragment by matching atom labels
        frag_iao_indices = []
        for iatom in frag_atoms:
            atom_label = mol.atom_symbol(iatom)
            for i, label in enumerate(iao_labels):
                if f'{atom_label}{iatom}' in label:
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
        
        # Transform W matrices from MO basis back to AO basis
        # W_AO = C_MO @ W_MO
        C_mo_occ = C_mo[:, :nocc]  # (nao, nocc)
        C_mo_virt = C_mo[:, nocc:]  # (nao, nvirt)
        
        W_occ = C_mo_occ @ W_occ_MO  # (nao, n_int_occ)
        W_tilde_occ = C_mo_occ @ W_tilde_occ_MO  # (nao, n_ext_occ)
        W_virt = C_mo_virt @ W_virt_MO  # (nao, n_int_virt)
        W_tilde_virt = C_mo_virt @ W_tilde_virt_MO  # (nao, n_ext_virt)
        
        logger.info(f"\nTransformation matrices in AO basis:")
        logger.info(f"  W_occ shape: {W_occ.shape} - (nao, n_int_occ)")
        logger.info(f"  W_tilde_occ shape: {W_tilde_occ.shape} - (nao, n_ext_occ)")
        logger.info(f"  W_virt shape: {W_virt.shape} - (nao, n_int_virt)")
        logger.info(f"  W_tilde_virt shape: {W_tilde_virt.shape} - (nao, n_ext_virt)")
        
        logger.info(f"\nActive space summary:")
        logger.info(f"  Atoms: {frag_atoms}")
        logger.info(f"  Active space size: {n_int_occ + n_int_virt} orbitals")
        logger.info(f"    ({n_int_occ} occ + {n_int_virt} virt)")
        
        # Store fragment's local space information
        local_space = {
            'fragment_id': frag_idx,
            'atoms': frag_atoms,
            'iao_indices': frag_iao_indices,
            'n_int_occ': n_int_occ,
            'n_int_virt': n_int_virt,
            'n_ext_occ': n_ext_occ,
            'n_ext_virt': n_ext_virt,
            'W_occ': W_occ,
            'W_tilde_occ': W_tilde_occ,
            'W_virt': W_virt,
            'W_tilde_virt': W_tilde_virt,
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


def compute_mp2_lno_compression(mol, mf, local_spaces, eta_occ=1e-6, eta_virt=1e-6, f_iao=None, eri_iao=None):
    """
    Compress external orbital spaces using Local Natural Orbitals (LNOs) from MP2.
    
    This implements the LNO-based active space compression in IAO basis:
    1. Compute MP1 amplitudes with mixed basis: t_{i_F ajb} where i_F are active occupied, a,j,b in IAO basis
    2. Build MP2 density matrices for each fragment in IAO basis
    3. Project density matrices into external spaces (Eq. 24)
    4. Diagonalize to get LNOs and occupation numbers (Eq. 25)
    5. Truncate based on thresholds eta_occ and eta_virt
    
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
    f_iao : np.ndarray
        Fock matrix in IAO basis, shape (niao, niao)
    eri_iao : np.ndarray
        Two-electron integrals in IAO basis, shape (niao, niao, niao, niao)
    
    Returns
    -------
    local_spaces_lno : list of dict
        Updated local spaces with LNO-compressed external spaces
    """
    logger.info("\n" + "="*60)
    logger.info("Step 5: Compressing External Spaces with MP2 LNOs (IAO Basis)")
    logger.info("="*60)
    
    nocc = mol.nelectron // 2
    niao = f_iao.shape[0]  # Number of IAOs
    nvirt = niao - nocc    # Virtual IAOs
    
    logger.info(f"\nMP2-LNO parameters:")
    logger.info(f"  Occupied threshold (eta_occ): {eta_occ}")
    logger.info(f"  Virtual threshold (eta_virt): {eta_virt}")
    logger.info(f"  Number of occupied IAOs: {nocc}")
    logger.info(f"  Number of virtual IAOs: {nvirt}")
    logger.info(f"  Total IAOs: {niao}")
    
    # Get IAO energies from diagonal of Fock matrix in IAO basis
    eps_iao = np.diag(f_iao)  # Diagonal elements are orbital energies
    eps_occ_iao = eps_iao[:nocc]
    eps_virt_iao = eps_iao[nocc:]
    
    logger.info(f"\nIAO Fock matrix shape: {f_iao.shape}")
    logger.info(f"IAO ERI shape: {eri_iao.shape}")
    
    # Process each fragment
    local_spaces_lno = []
    
    for frag_idx, ls in enumerate(local_spaces):
        logger.info(f"\n--- Fragment {frag_idx}: LNO Compression (W Basis) ---")
        
        # Get W transformation matrices (these are in AO basis, shape: nao × n_orbitals)
        W_occ = ls['W_occ']  # (nao, n_int_occ) - internal occupied orbitals
        W_tilde_occ = ls['W_tilde_occ']  # (nao, n_ext_occ) - external occupied orbitals
        W_virt = ls['W_virt']  # (nao, n_int_virt) - internal virtual orbitals
        W_tilde_virt = ls['W_tilde_virt']  # (nao, n_ext_virt) - external virtual orbitals
        
        n_int_occ = W_occ.shape[1]
        n_ext_occ = W_tilde_occ.shape[1]
        n_int_virt = W_virt.shape[1]
        n_ext_virt = W_tilde_virt.shape[1]
        
        logger.info(f"  W_occ shape: {W_occ.shape} - {n_int_occ} internal occupied")
        logger.info(f"  W_tilde_occ shape: {W_tilde_occ.shape} - {n_ext_occ} external occupied")
        logger.info(f"  W_virt shape: {W_virt.shape} - {n_int_virt} internal virtual")
        logger.info(f"  W_tilde_virt shape: {W_tilde_virt.shape} - {n_ext_virt} external virtual")
        
        # Skip MP2 if no internal occupied orbitals (can't do correlation)
        if n_int_occ == 0:
            logger.info(f"  Skipping MP2: no internal occupied orbitals")
            ls_lno = ls.copy()
            ls_lno.update({
                'n_lno_occ': 0,
                'n_lno_virt': 0,
                'lno_occ_mask': np.array([], dtype=bool),
                'lno_virt_mask': np.array([], dtype=bool),
                'X_occ': np.eye(n_ext_occ) if n_ext_occ > 0 else np.zeros((0, 0)),
                'X_virt': np.eye(n_ext_virt) if n_ext_virt > 0 else np.zeros((0, 0)),
                'Lambda_occ': np.zeros(n_ext_occ),
                'Lambda_virt': np.zeros(n_ext_virt),
            })
            local_spaces_lno.append(ls_lno)
            continue
        
        # Transform Fock matrix to W basis for internal occupied orbitals
        # F_int_occ = W_occ^T @ F_AO @ W_occ
        h_core = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        vhf = mf.get_veff()
        f_ao = h_core + vhf
        
        F_occ_int = W_occ.T @ f_ao @ W_occ  # (n_int_occ, n_int_occ)
        eps_int_occ = np.diag(F_occ_int)  # Internal occupied orbital energies
        
        logger.info(f"  Transformed Fock to internal occupied basis:")
        logger.info(f"    F_occ_int shape: {F_occ_int.shape}")
        logger.info(f"    Internal occupied energies: {eps_int_occ}")
        
        # For external occupied and all virtuals, we need energies in AO basis
        # Get energies by diagonalizing in respective bases
        if n_ext_occ > 0:
            F_ext_occ = W_tilde_occ.T @ f_ao @ W_tilde_occ
            eps_ext_occ = np.diag(F_ext_occ)
        else:
            eps_ext_occ = np.array([])
        
        if n_ext_virt > 0:
            F_ext_virt = W_tilde_virt.T @ f_ao @ W_tilde_virt
            eps_ext_virt = np.diag(F_ext_virt)
        else:
            eps_ext_virt = np.array([])
        
        logger.info(f"  External orbital energies computed")
        
        # Transform ERIs to mixed basis: V_{i_int, a_ext, j_ext, b_ext}
        # This is the key transformation for MP1 amplitudes
        logger.info(f"  Transforming ERIs to mixed basis...")
        
        # Get ERIs in AO basis
        eri_ao = mol.intor('int2e', aosym='s1').reshape(mol.nao, mol.nao, mol.nao, mol.nao)
        
        # Transform: V_{Iajb} = W_occ^T[I] @ eri_ao @ W_tilde_virt[a] @ W_tilde_occ[j] @ W_tilde_virt[b]
        # Do this in stages to save memory
        if n_ext_virt > 0 and n_ext_occ > 0:
            nao = W_occ.shape[0]
            # Stage 1: contract first AO index with W_occ
            # W_occ: (nao, n_int_occ), W_occ.T: (n_int_occ, nao)
            # eri_ao: (nao, nao, nao, nao)
            # Contract eri_ao[i,:,:,:] with W_occ[:,I] -> need W_occ.T[I,i]
            V_temp1 = np.einsum('Ii,ijkl->Ijkl', W_occ.T, eri_ao, optimize=True)  # (n_int_occ, nao, nao, nao)
            logger.debug(f"    After stage 1: V_temp1 shape = {V_temp1.shape}, expected ({n_int_occ}, {nao}, {nao}, {nao})")
            # Stage 2: contract second AO index with W_tilde_virt
            V_temp2 = np.einsum('Ijkl,ja->Iakl', V_temp1, W_tilde_virt, optimize=True)  # (n_int_occ, n_ext_virt, nao, nao)
            logger.debug(f"    After stage 2: V_temp2 shape = {V_temp2.shape}, expected ({n_int_occ}, {n_ext_virt}, {nao}, {nao})")
            # Stage 3: contract third AO index with W_tilde_occ
            V_temp3 = np.einsum('Iakl,kJ->IaJl', V_temp2, W_tilde_occ, optimize=True)  # (n_int_occ, n_ext_virt, n_ext_occ, nao)
            logger.debug(f"    After stage 3: V_temp3 shape = {V_temp3.shape}, expected ({n_int_occ}, {n_ext_virt}, {n_ext_occ}, {nao})")
            # Stage 4: contract fourth AO index with W_tilde_virt
            V_Iajb = np.einsum('IaJl,lb->IaJb', V_temp3, W_tilde_virt, optimize=True)  # (n_int_occ, n_ext_virt, n_ext_occ, n_ext_virt)
            
            logger.info(f"    V_Iajb shape: {V_Iajb.shape}, expected ({n_int_occ}, {n_ext_virt}, {n_ext_occ}, {n_ext_virt})")
            
            # Compute MP1 amplitudes: t_{Iajb} = V_{Iajb} / (eps_I + eps_j - eps_a - eps_b)
            eps_I = eps_int_occ[:, None, None, None]  # (n_int_occ, 1, 1, 1)
            eps_a = eps_ext_virt[None, :, None, None]  # (1, n_ext_virt, 1, 1)
            eps_j = eps_ext_occ[None, None, :, None]  # (1, 1, n_ext_occ, 1)
            eps_b = eps_ext_virt[None, None, None, :]  # (1, 1, 1, n_ext_virt)
            
            denom = eps_I + eps_j - eps_a - eps_b
            t_Iajb = np.divide(V_Iajb, denom, where=np.abs(denom) > 1e-10, out=np.zeros_like(V_Iajb))
            
            logger.info(f"    MP1 amplitude shape: {t_Iajb.shape}")
            logger.info(f"    MP1 amplitude norm: {np.linalg.norm(t_Iajb):.6f}")
        else:
            t_Iajb = np.zeros((n_int_occ, max(n_ext_virt, 1), max(n_ext_occ, 1), max(n_ext_virt, 1)))
            logger.info(f"    Skipping ERI transformation: n_ext_virt={n_ext_virt}, n_ext_occ={n_ext_occ}")
        
        # Build MP2 density matrices
        # D_jj' (occupied): sum over I, a, b of t*_{Iajb} * (2*t_{Iaj'b} - t_{Ibj'a})
        # D_ab (virtual): sum over I, j, c of expressions similar to before
        logger.info(f"  Building MP2 density matrices...")
        
        if n_ext_occ > 0 and n_ext_virt > 0 and n_int_occ > 0:
            # Occupied density: D_jj' in external occupied space
            # t_Iajb shape: (n_int_occ, n_ext_virt, n_ext_occ, n_ext_virt)
            # Index: I=internal occ, a=ext virt, j=ext occ, b=ext virt
            # D[j,j'] = sum_I,a,b conj(t[I,a,j,b]) * (2*t[I,a,j',b] - t[I,b,j',a])
            # First term: 2 * sum_{I,a,b} t*[I,a,j,b] * t[I,a,j',b]
            term1 = np.einsum('Iajb,IaJb->jJ', np.conj(t_Iajb), t_Iajb, optimize=True)
            # Second term: sum_{I,a,b} t*[I,a,j,b] * t[I,b,j',a]
            term2 = np.einsum('Iajb,IbJa->jJ', np.conj(t_Iajb), t_Iajb, optimize=True)
            D_occ = 2.0 * (2.0 * term1 - term2)
            
            # Virtual density: D_ab in external virtual space
            # D[a,a'] = sum_{I,j,b} (2*t*[I,a,j,b]*t[I,a',j,b] - t*[I,b,j,a]*t[I,b,j,a'] 
            #                        - t*[I,b,j,a]*t[I,a',j,b])
            # First term: 2 * sum_{I,j,b} t*[I,a,j,b] * t[I,a',j,b]
            term1 = np.einsum('Iajb,IAjb->aA', np.conj(t_Iajb), t_Iajb, optimize=True)
            # Second term: sum_{I,j,b} t*[I,b,j,a] * t[I,b,j,a']
            term2 = np.einsum('Ibja,IbjA->aA', np.conj(t_Iajb), t_Iajb, optimize=True)
            # Third term: sum_{I,j,b} t*[I,b,j,a] * t[I,a',j,b]
            term3 = np.einsum('Ibja,IAjb->aA', np.conj(t_Iajb), t_Iajb, optimize=True)
            D_virt = 2.0 * (2.0 * term1 - term2 - term3)
            
            logger.info(f"    D_occ shape: {D_occ.shape}, norm: {np.linalg.norm(D_occ):.6f}")
            logger.info(f"    D_virt shape: {D_virt.shape}, norm: {np.linalg.norm(D_virt):.6f}")
        else:
            D_occ = np.zeros((max(n_ext_occ, 1), max(n_ext_occ, 1)))
            D_virt = np.zeros((max(n_ext_virt, 1), max(n_ext_virt, 1)))
            logger.info(f"    Zero density matrices (insufficient orbitals for correlation)")
        
        # Diagonalize to get LNOs
        logger.info(f"  Diagonalizing for LNOs...")
        
        if n_ext_occ > 0:
            Lambda_occ, X_occ = np.linalg.eigh(D_occ)
            idx_occ = np.argsort(-np.abs(Lambda_occ))
            Lambda_occ = Lambda_occ[idx_occ]
            X_occ = X_occ[:, idx_occ]
            logger.info(f"    Occupied eigenvalues (top 5): {Lambda_occ[:min(5, len(Lambda_occ))]}")
        else:
            Lambda_occ = np.array([])
            X_occ = np.zeros((0, 0))
        
        if n_ext_virt > 0:
            Lambda_virt, X_virt = np.linalg.eigh(D_virt)
            idx_virt = np.argsort(-np.abs(Lambda_virt))
            Lambda_virt = Lambda_virt[idx_virt]
            X_virt = X_virt[:, idx_virt]
            logger.info(f"    Virtual eigenvalues (top 5): {Lambda_virt[:min(5, len(Lambda_virt))]}")
        else:
            Lambda_virt = np.array([])
            X_virt = np.zeros((0, 0))
        
        # Select LNOs based on thresholds
        lno_occ_mask = np.abs(Lambda_occ) > eta_occ if len(Lambda_occ) > 0 else np.array([], dtype=bool)
        lno_virt_mask = np.abs(Lambda_virt) > eta_virt if len(Lambda_virt) > 0 else np.array([], dtype=bool)
        
        n_lno_occ = np.sum(lno_occ_mask)
        n_lno_virt = np.sum(lno_virt_mask)
        
        logger.info(f"\n  LNO selection:")
        logger.info(f"    Occupied LNOs selected: {n_lno_occ} / {n_ext_occ}")
        logger.info(f"    Virtual LNOs selected: {n_lno_virt} / {n_ext_virt}")
        
        # Update local space with LNO information
        ls_lno = ls.copy()
        ls_lno.update({
            'Lambda_occ': Lambda_occ,
            'Lambda_virt': Lambda_virt,
            'X_occ': X_occ,
            'X_virt': X_virt,
            'n_lno_occ': n_lno_occ,
            'n_lno_virt': n_lno_virt,
            'lno_occ_mask': lno_occ_mask,
            'lno_virt_mask': lno_virt_mask,
            'eta_occ': eta_occ,
            'eta_virt': eta_virt,
        })
        
        local_spaces_lno.append(ls_lno)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("MP2-LNO Compression Summary")
    logger.info("="*60)
    logger.info(f"Total fragments: {len(local_spaces_lno)}")
    
    total_ext_before = 0
    total_ext_after = 0
    
    for ls in local_spaces_lno:
        n_ext_before = len(ls['external_occ']) + len(ls['external_virt'])
        n_ext_after = ls['n_lno_occ'] + ls['n_lno_virt']
        total_ext_before += n_ext_before
        total_ext_after += n_ext_after
        
        logger.info(f"\nFragment {ls['fragment_id']}:")
        logger.info(f"  Internal: {len(ls['internal_occ'])} occ + {len(ls['internal_virt'])} virt")
        logger.info(f"  External (before): {len(ls['external_occ'])} occ + {len(ls['external_virt'])} virt = {n_ext_before}")
        logger.info(f"  External (LNO): {ls['n_lno_occ']} occ + {ls['n_lno_virt']} virt = {n_ext_after}")
        logger.info(f"  Compression ratio: {n_ext_after/n_ext_before:.2%}" if n_ext_before > 0 else "  No external space")
    
    logger.info(f"\nOverall compression:")
    logger.info(f"  Total external orbitals before: {total_ext_before}")
    logger.info(f"  Total external orbitals after LNO: {total_ext_after}")
    logger.info(f"  Overall compression ratio: {total_ext_after/total_ext_before:.2%}" if total_ext_before > 0 else "  No external space")
    logger.info("="*60 + "\n")
    
    return local_spaces_lno


def solve_fragment_ccsd(mol, mf, local_spaces_lno, f_iao, eri_iao, iao_coeff):
    """
    Solve fragment CCSD equations for each fragment and compute correlation energy.
    
    This implements the local CCSD approach in IAO basis:
    1. Construct local Hamiltonian H^(F) for each fragment in active space (internal + external LNO)
    2. Transform Fock and ERIs from IAO basis to active space
    3. Solve CCSD equations within active space A_F  
    4. Transform amplitudes back to IAO basis
    5. Project onto fragment IAOs for energy contributions
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    mf : pyscf.scf.RHF
        Converged RHF mean-field object
    local_spaces_lno : list of dict
        Local active spaces with LNO compression
    f_iao : np.ndarray
        Fock matrix in IAO basis, shape (niao, niao)
    eri_iao : np.ndarray
        Two-electron integrals in IAO basis, shape (niao, niao, niao, niao)
    iao_coeff : np.ndarray
        IAO coefficients in AO basis, shape (nao, niao)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'fragment_energies': correlation energy for each fragment
        - 'total_correlation': total correlation energy
        - 'fragment_ccsd': CCSD objects for each fragment
        - 'fragment_amplitudes': amplitudes for each fragment
    """
    logger.info("\n" + "="*60)
    logger.info("Step 6: Solving Fragment CCSD Equations (W Orbital Basis)")
    logger.info("="*60)
    
    # Get AO Fock matrix and ERIs
    f_ao = mf.get_fock()
    # Get ERIs in AO basis - need to check if they're already available
    if hasattr(mf, '_eri') and mf._eri is not None:
        if isinstance(mf._eri, np.ndarray) and mf._eri.ndim == 4:
            eri_ao = mf._eri
        else:
            # Reconstruct full 4D ERI tensor
            eri_ao = ao2mo.restore(1, mf._eri, mf.mol.nao_nr())
    else:
        # Compute ERIs from scratch
        eri_ao = mol.intor('int2e', aosym='s1')
    
    nao = f_ao.shape[0]
    
    logger.info(f"\nFragment CCSD setup:")
    logger.info(f"  Number of fragments: {len(local_spaces_lno)}")
    logger.info(f"  AO basis size: {nao}")
    logger.info(f"  ERI tensor shape: {eri_ao.shape}")
    
    fragment_results = []
    fragment_energies = []
    
    for frag_idx, ls in enumerate(local_spaces_lno):
        logger.info(f"\n--- Fragment {frag_idx}: CCSD Calculation (W Basis) ---")
        
        # Get W transformation matrices (in AO basis)
        W_occ = ls['W_occ']  # (nao, n_int_occ)
        W_tilde_occ = ls['W_tilde_occ']  # (nao, n_ext_occ)
        W_virt = ls['W_virt']  # (nao, n_int_virt)
        W_tilde_virt = ls['W_tilde_virt']  # (nao, n_ext_virt)
        
        X_occ = ls['X_occ']  # (n_ext_occ, n_ext_occ) - LNO transformation for occ
        X_virt = ls['X_virt']  # (n_ext_virt, n_ext_virt) - LNO transformation for virt
        lno_occ_mask = ls['lno_occ_mask']
        lno_virt_mask = ls['lno_virt_mask']
        
        n_int_occ = W_occ.shape[1]
        n_int_virt = W_virt.shape[1]
        n_lno_occ = np.sum(lno_occ_mask)
        n_lno_virt = np.sum(lno_virt_mask)
        n_active_occ = n_int_occ + n_lno_occ
        n_active_virt = n_int_virt + n_lno_virt
        n_active = n_active_occ + n_active_virt
        
        logger.info(f"  Active space: {n_active} orbitals ({n_active_occ} occ + {n_active_virt} virt)")
        logger.info(f"    Internal: {n_int_occ} occ + {n_int_virt} virt")
        logger.info(f"    External LNO: {n_lno_occ} occ + {n_lno_virt} virt")
        
        # Skip if active space is too small
        if n_active_occ == 0 or n_active_virt == 0:
            logger.info(f"  Skipping CCSD: active space too small (need occ>0 and virt>0)")
            fragment_results.append(None)
            fragment_energies.append(0.0)
            continue
        
        # Build transformation matrix U_act from AO basis to active space
        # U_act_occ: (nao, n_active_occ) = [W_occ | W_tilde_occ @ X_occ[:, selected]]
        # U_act_virt: (nao, n_active_virt) = [W_virt | W_tilde_virt @ X_virt[:, selected]]
        
        logger.info(f"  Building transformation U_act from AO to active space...")
        
        # Internal part (from SVD) - already in AO basis
        U_occ_int = W_occ  # (nao, n_int_occ)
        U_virt_int = W_virt  # (nao, n_int_virt)
        
        # External LNO part - transform W_tilde with LNO eigenvectors
        if n_lno_occ > 0:
            U_occ_ext = W_tilde_occ @ X_occ[:, lno_occ_mask]  # (nao, n_lno_occ)
        else:
            U_occ_ext = np.zeros((nao, 0))
        
        if n_lno_virt > 0:
            U_virt_ext = W_tilde_virt @ X_virt[:, lno_virt_mask]  # (nao, n_lno_virt)
        else:
            U_virt_ext = np.zeros((nao, 0))
        
        # Combine: [internal | external_LNO]
        if n_int_occ > 0 and n_lno_occ > 0:
            U_act_occ = np.hstack([U_occ_int, U_occ_ext])  # (nao, n_active_occ)
        elif n_int_occ > 0:
            U_act_occ = U_occ_int
        elif n_lno_occ > 0:
            U_act_occ = U_occ_ext
        else:
            U_act_occ = np.zeros((nao, 0))
            
        if n_int_virt > 0 and n_lno_virt > 0:
            U_act_virt = np.hstack([U_virt_int, U_virt_ext])  # (nao, n_active_virt)
        elif n_int_virt > 0:
            U_act_virt = U_virt_int
        elif n_lno_virt > 0:
            U_act_virt = U_virt_ext
        else:
            U_act_virt = np.zeros((nao, 0))
        
        logger.info(f"    U_act_occ shape: {U_act_occ.shape}")
        logger.info(f"    U_act_virt shape: {U_act_virt.shape}")
        
        # Transform Fock matrix from AO basis to active space
        # F_act = U_act^T @ F_AO @ U_act
        logger.info(f"  Transforming Fock matrix to active space...")
        
        f_oo_act = U_act_occ.T @ f_ao @ U_act_occ  # (n_active_occ, n_active_occ)
        f_ov_act = U_act_occ.T @ f_ao @ U_act_virt  # (n_active_occ, n_active_virt)
        f_vo_act = U_act_virt.T @ f_ao @ U_act_occ  # (n_active_virt, n_active_occ)
        f_vv_act = U_act_virt.T @ f_ao @ U_act_virt  # (n_active_virt, n_active_virt)
        
        # Assemble full Fock matrix in active space
        f_active = np.zeros((n_active, n_active))
        f_active[:n_active_occ, :n_active_occ] = f_oo_act
        f_active[:n_active_occ, n_active_occ:] = f_ov_act
        f_active[n_active_occ:, :n_active_occ] = f_vo_act
        f_active[n_active_occ:, n_active_occ:] = f_vv_act
        
        logger.info(f"    F_active shape: {f_active.shape}, norm: {np.linalg.norm(f_active):.6f}")
        
        # Transform ERIs from AO basis to active space
        # ERI_act = sum_{ijkl} U_occ[i,p] * U[j,q] * U[k,r] * U[l,s] * ERI_AO[i,j,k,l]
        # where U for each index can be either U_act_occ or U_act_virt depending on the orbital type
        logger.info(f"  Transforming ERIs to active space...")
        
        # Four-index transformation using einsum (one index at a time)
        # Transform all 4 indices with combined occupied+virtual transformation
        # We need to transform from AO (nao×nao×nao×nao) to active (n_act×n_act×n_act×n_act)
        # Build U_full: (nao, n_active) where occupied come first, then virtual
        U_full = np.hstack([U_act_occ, U_act_virt])  # (nao, n_active)
        
        eri_temp1 = np.einsum('ip,ijkl->pjkl', U_full, eri_ao, optimize=True)
        eri_temp2 = np.einsum('jq,pjkl->pqkl', U_full, eri_temp1, optimize=True)
        eri_temp3 = np.einsum('kr,pqkl->pqrl', U_full, eri_temp2, optimize=True)
        eri_active = np.einsum('ls,pqrl->pqrs', U_full, eri_temp3, optimize=True)
        
        logger.info(f"    ERI_active shape: {eri_active.shape}, norm: {np.linalg.norm(eri_active):.6f}")
        
        # Create fake molecule and mean-field object for PySCF CCSD
        logger.info(f"  Setting up fragment CCSD calculation...")
        
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
        mycc.kernel()
        
        if mycc.converged:
            logger.info(f"    CCSD converged!")
            logger.info(f"    CCSD correlation energy: {mycc.e_corr:.8f} a.u.")
        else:
            logger.warning(f"    CCSD did not converge!")
        
        # Get amplitudes in active space basis
        t1_act = mycc.t1  # (n_active_occ, n_active_virt)
        t2_act = mycc.t2  # (n_active_occ, n_active_occ, n_active_virt, n_active_virt)
        
        logger.info(f"    T1 amplitude shape: {t1_act.shape}, norm: {np.linalg.norm(t1_act):.6f}")
        logger.info(f"    T2 amplitude shape: {t2_act.shape}, norm: {np.linalg.norm(t2_act):.6f}")
        
        # Compute tau intermediate: tau_ijab = t_ijab + t_ia * t_jb
        tau_act = t2_act + np.einsum('ia,jb->ijab', t1_act, t1_act, optimize=True)
        
        logger.info(f"    Tau intermediate norm: {np.linalg.norm(tau_act):.6f}")
        
        # Compute energy matrix E_{ii'} in active occupied space
        # E_{ii'} = sum_{ab} tau_{i'iab} * f_{ab}  (Eq. from paper)
        # Then transform back to IAO basis: E_IAO = U_act_occ @ E_{ii'} @ U_act_occ^T
        logger.info(f"  Computing energy contribution matrix...")
        
        # Extract virtual block of Fock in active space
        f_vv_act_for_energy = f_vv_act  # (n_active_virt, n_active_virt)
        
        # E[i, i'] = sum_a,b tau[i', i, a, b] * f_vv[a, b]
        # tau: (n_active_occ, n_active_occ, n_active_virt, n_active_virt)
        # f_vv: (n_active_virt, n_active_virt)
        E_ii_prime = np.einsum('ijab,ab->ij', tau_act, f_vv_act_for_energy, optimize=True)
        
        logger.info(f"    E_ii' shape: {E_ii_prime.shape}, norm: {np.linalg.norm(E_ii_prime):.6f}")
        
        # Transform back to IAO basis: E_IAO = U_act_occ @ E_{ii'} @ U_act_occ^T
        E_iao = U_act_occ @ E_ii_prime @ U_act_occ.T  # (nocc, nocc)
        
        logger.info(f"    E_IAO shape: {E_iao.shape}, norm: {np.linalg.norm(E_iao):.6f}")
        
        # Project onto fragment IAOs to get fragment energy
        # For fragment F with internal occupied orbitals, sum over those diagonal elements
        # E_F = sum_{i in F} E_IAO[i, i]
        internal_occ_iao_idx = ls['internal_occ']  # IAO indices
        fragment_energy = np.sum(E_iao[i, i] for i in internal_occ_iao_idx)
        
        logger.info(f"  Fragment {frag_idx} correlation energy: {fragment_energy:.8f} a.u.")
        logger.info(f"    Internal occupied IAO indices: {internal_occ_iao_idx}")
        
        # Store results
        frag_result = {
            'fragment_id': frag_idx,
            'n_active_occ': n_active_occ,
            'n_active_virt': n_active_virt,
            'f_active': f_active,
            'eri_active': eri_active,
            'ccsd': mycc,
            't1': t1_act,
            't2': t2_act,
            'tau': tau_act,
            'e_corr': fragment_energy,
            'E_iao': E_iao,
        }
        
        fragment_results.append(frag_result)
        fragment_energies.append(fragment_energy)
    
    # Compute total correlation energy
    total_corr = np.sum(fragment_energies)
    
    logger.info("\n" + "="*60)
    logger.info("Fragment CCSD Results Summary")
    logger.info("="*60)
    logger.info(f"Individual fragment correlation energies:")
    for i, e in enumerate(fragment_energies):
        logger.info(f"  Fragment {i}: {e:.8f} a.u.")
    logger.info(f"\nTotal correlation energy (sum): {total_corr:.8f} a.u.")
    logger.info(f"RHF energy: {mf.e_tot:.8f} a.u.")
    logger.info(f"Estimated total energy: {mf.e_tot + total_corr:.8f} a.u.")
    logger.info("="*60 + "\n")
    
    results = {
        'fragment_results': fragment_results,
        'fragment_energies': fragment_energies,
        'total_correlation': total_corr,
        'total_energy': mf.e_tot + total_corr,
    }
    
    return results


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
    
    mol = build_h10_chain(separation=separation, basis=basis)
    
    # Step 2: Run RHF calculation
    mf = run_rhf(mol)
    
    # Analyze the solution
    analyze_rhf_solution(mol, mf)
    
    # Step 3: Construct Intrinsic Atomic Orbitals (IAO)
    iao_coeff, iao_labels = construct_iao(mol, mf, minao='minao')
    
    # Transform Fock and ERIs to IAO basis (needed for all subsequent steps)
    logger.info(f"\nTransforming Fock matrix and ERIs to IAO basis...")
    
    # Get Fock matrix in AO basis
    h_core = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    s_ao = mol.intor('int1e_ovlp')
    dm = mf.make_rdm1()
    vhf = mf.get_veff(mol, dm)
    f_ao = h_core + vhf
    
    # Transform to IAO basis: F_IAO = C_IAO^T @ F_AO @ C_IAO
    f_iao = iao_coeff.T @ f_ao @ iao_coeff
    logger.info(f"  Fock matrix in IAO basis shape: {f_iao.shape}")
    logger.info(f"  Fock matrix norm: {np.linalg.norm(f_iao):.6f}")
    
    # Get ERIs in AO basis and transform to IAO basis
    logger.info(f"  Transforming ERIs to IAO basis (this may take a moment)...")
    eri_ao = mol.intor('int2e', aosym='s1')
    eri_ao = eri_ao.reshape(mol.nao, mol.nao, mol.nao, mol.nao)
    
    # Transform one index at a time: V_IAO = C^T @ C^T @ C^T @ C^T @ V_AO
    eri_temp1 = np.einsum('ip,ijkl->pjkl', iao_coeff, eri_ao, optimize=True)
    eri_temp2 = np.einsum('jq,pjkl->pqkl', iao_coeff, eri_temp1, optimize=True)
    eri_temp3 = np.einsum('kr,pqkl->pqrl', iao_coeff, eri_temp2, optimize=True)
    eri_iao = np.einsum('ls,pqrl->pqrs', iao_coeff, eri_temp3, optimize=True)
    
    logger.info(f"  ERIs in IAO basis shape: {eri_iao.shape}")
    logger.info(f"  ERIs norm: {np.linalg.norm(eri_iao):.6f}")
    
    # Step 4: Construct Local Active Space for each fragment
    # Use fragment_size=1 to treat each H atom as a separate fragment
    # Or fragment_size=2 to pair up neighboring H atoms
    fragment_size = 1  # Each atom is a fragment
    svd_threshold = 1e-6  # Threshold for determining internal space
    
    local_spaces = construct_local_active_space(
        mol, mf, iao_coeff, iao_labels,
        fragment_size=fragment_size, 
        svd_threshold=svd_threshold
    )
    
    # Step 5: Compress external spaces using MP2-LNO
    eta_occ = 1e-6  # Threshold for occupied LNO eigenvalues
    eta_virt = 1e-6  # Threshold for virtual LNO eigenvalues
    
    local_spaces_lno = compute_mp2_lno_compression(
        mol, mf, local_spaces,
        f_iao=f_iao,
        eri_iao=eri_iao,
        eta_occ=eta_occ,
        eta_virt=eta_virt
    )
    
    # Step 6: Solve fragment CCSD equations
    ccsd_results = solve_fragment_ccsd(mol, mf, local_spaces_lno, f_iao, eri_iao, iao_coeff)
    
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
        'energy_corr': ccsd_results['total_correlation'],
        'energy_total': ccsd_results['total_energy'],
    }
    
    logger.info("\n" + "="*70)
    logger.info("All steps completed successfully!")
    logger.info("="*70)
    logger.info("\nWorkflow summary:")
    logger.info(f"  1. Built H10 chain with {mol.natm} atoms")
    logger.info(f"  2. RHF energy: {mf.e_tot:.8f} a.u.")
    logger.info(f"  3. Constructed {iao_coeff.shape[1]} IAOs")
    logger.info(f"  4. Defined {len(local_spaces)} local active spaces")
    logger.info(f"  5. Compressed external spaces with MP2-LNOs")
    logger.info(f"  6. Solved {len(local_spaces_lno)} fragment CCSD equations")
    logger.info(f"\nFinal energies:")
    logger.info(f"  RHF energy:         {mf.e_tot:.8f} a.u.")
    logger.info(f"  Correlation energy: {ccsd_results['total_correlation']:.8f} a.u.")
    logger.info(f"  Total energy:       {ccsd_results['total_energy']:.8f} a.u.")
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
                eri_mo=eri_mo,
                eta_occ=eta_occ,
                eta_virt=eta_virt
            )
            
            # Solve fragment CCSD
            ccsd_results = solve_fragment_ccsd(mol, mf, local_spaces_lno, eri_mo, iao_coeff)
            
            # Compute error
            e_corr = ccsd_results['total_correlation']
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
        mol = build_h10_chain(separation=1.5, basis='cc-pvdz')
        mf = run_rhf(mol)
        iao_coeff, iao_labels = construct_iao(mol, mf, minao='minao')
        
        local_spaces = construct_local_active_space(
            mol=mol,
            mf=mf,
            iao_coeff=iao_coeff,
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
    else:
        # Run standard workflow
        results = main()
