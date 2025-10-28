"""
Test SIE embedding for Benzene (C6H6) molecule

This test verifies that the SIE class works correctly for benzene molecule.
Due to the high symmetry (D6h), we calculate one C-H fragment and multiply by 6.
"""

import numpy as np
from pyscf import gto, scf, cc
import logging
import sys
import os

from pytc.embedding import SIE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_benzene(basis='sto-3g'):
    """Build benzene (C6H6) molecule with D6h symmetry."""
    logger.info(f"Building benzene molecule with basis={basis}")
    
    mol = gto.Mole()
    mol.atom = '''
        C 2.866 1.0 0
        C 3.7321 0.5 0
        C 2.0 0.5 0
        C 3.7321 -0.5 0
        C 2.0 -0.5 0
        C 2.866 -1.0 0
        H 2.866 1.62 0
        H 4.269 0.81 0
        H 1.4631 0.81 0
        H 4.269 -0.81 0
        H 1.4631 -0.81 0
        H 2.866 -1.62 0
    '''
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 4
    mol.symmetry = False  # Don't use symmetry to get all atoms
    mol.build()
    
    logger.info(f"  Number of atoms: {mol.natm}")
    logger.info(f"  Number of electrons: {mol.nelectron}")
    logger.info(f"  Number of basis functions: {mol.nao}")
    
    return mol


def run_rhf(mol):
    """Run RHF calculation."""
    logger.info("Starting RHF calculation...")
    
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.max_cycle = 100
    mf.conv_tol = 1e-10
    
    energy = mf.kernel()
    
    if mf.converged:
        logger.info(f"RHF converged! Energy: {energy:.8f} a.u.")
    else:
        logger.warning("RHF did not converge!")
    
    return mf


def run_full_ccsd(mol, mf):
    """Run full CCSD calculation for reference."""
    logger.info("\n" + "="*70)
    logger.info("Running Full CCSD for Reference")
    logger.info("="*70)
    
    mycc = cc.CCSD(mf)
    mycc.verbose = 4
    mycc.kernel()
    
    if mycc.converged:
        logger.info(f"Full CCSD converged!")
        logger.info(f"  CCSD correlation energy: {mycc.e_corr:.8f} a.u.")
        logger.info(f"  Total CCSD energy: {mycc.e_tot:.8f} a.u.")
    else:
        logger.warning("Full CCSD did not converge!")
    
    return mycc


def test_benzene_one_ch_fragment():
    """
    Test benzene with one C-H fragment and exploit 6-fold symmetry.
    
    Due to benzene's D6h symmetry, all 6 C-H units are equivalent.
    We calculate one C-H fragment and multiply the correlation energy by 6.
    """
    print("\n" + "="*70)
    print("Test: Benzene with One C-H Fragment (Exploiting 6-fold Symmetry)")
    print("="*70 + "\n")
    
    # Build molecule and run RHF
    mol = build_benzene(basis='ccpvdz')
    mf = run_rhf(mol)
    
    # Run full CCSD for reference
    full_ccsd = run_full_ccsd(mol, mf)
    
    # Initialize SIE
    sie = SIE(mol, mf)
    
    # Define fragment: First C atom (index 0) and its H atom (index 6)
    # This represents one C-H unit in the benzene ring
    fragments = [[0, 6]]
    
    logger.info("\n" + "="*70)
    logger.info("Fragmentation Scheme")
    logger.info("="*70)
    logger.info(f"  Fragment 0: C-H unit (atoms 0 and 6)")
    logger.info(f"  Note: Due to D6h symmetry, this fragment represents all 6 C-H units")
    logger.info(f"  Total fragments calculated: {len(fragments)}")
    
    # Run SIE workflow
    results = sie.run(
        fragments=fragments,
        minao='minao',
        svd_threshold=1e-8,
        eta_occ=1e-5,
        eta_virt=1e-5
    )
    
    # Extract single fragment energy and multiply by 6 for total
    energy_results = results['energy_results']
    single_fragment_energy = energy_results['fragment_energies'][0]
    total_correlation_energy = single_fragment_energy * 6
    total_sie_energy = mf.e_tot + total_correlation_energy
    
    # Print results
    print("\n" + "="*70)
    print("Results Comparison:")
    print("="*70)
    print(f"RHF energy:              {mf.e_tot:.8f} a.u.")
    print(f"\nFull CCSD:")
    print(f"  Correlation energy:    {full_ccsd.e_corr:.8f} a.u.")
    print(f"  Total energy:          {full_ccsd.e_tot:.8f} a.u.")
    print(f"\nSIE Embedding (Single C-H Fragment):")
    print(f"  Fragment energy:       {single_fragment_energy:.8f} a.u.")
    print(f"\nSIE Total (6 × Fragment):")
    print(f"  Correlation energy:    {total_correlation_energy:.8f} a.u.")
    print(f"  Total energy:          {total_sie_energy:.8f} a.u.")
    print(f"\nError (SIE - Full CCSD):")
    error_corr = total_correlation_energy - full_ccsd.e_corr
    error_total = total_sie_energy - full_ccsd.e_tot
    print(f"  Correlation energy:    {error_corr:.8f} a.u. ({error_corr*1000:.4f} mEh)")
    print(f"  Total energy:          {error_total:.8f} a.u. ({error_total*1000:.4f} mEh)")
    
    # Check that error is less than 1 mEh
    error_meh = abs(error_corr * 1000)
    print(f"\nAssertion: Error < 1.0 mEh")
    print(f"  Actual error: {error_meh:.4f} mEh")
    
    if error_meh < 1.0:
        print(f"  ✓ PASSED")
    else:
        print(f"  ✗ FAILED")
        raise AssertionError(f"Error {error_meh:.4f} mEh exceeds threshold of 1.0 mEh")
    
    print("="*70 + "\n")
    
    return results, full_ccsd


def test_benzene_all_ch_fragments():
    """
    Test benzene with all 6 C-H fragments for verification.
    
    This calculates all 6 C-H fragments separately to verify the symmetry assumption.
    """
    print("\n" + "="*70)
    print("Test: Benzene with All 6 C-H Fragments (Verification)")
    print("="*70 + "\n")
    
    # Build molecule and run RHF
    mol = build_benzene(basis='sto-3g')
    mf = run_rhf(mol)
    
    # Run full CCSD for reference
    full_ccsd = run_full_ccsd(mol, mf)
    
    # Initialize SIE
    sie = SIE(mol, mf)
    
    # Define fragments: All 6 C-H units
    # C atoms: 0-5, H atoms: 6-11
    # Pairing: C0-H6, C1-H7, C2-H8, C3-H9, C4-H10, C5-H11
    fragments = [
        [0, 6],   # C-H unit 1
        [1, 7],   # C-H unit 2
        [2, 8],   # C-H unit 3
        [3, 9],   # C-H unit 4
        [4, 10],  # C-H unit 5
        [5, 11]   # C-H unit 6
    ]
    
    logger.info("\n" + "="*70)
    logger.info("Fragmentation Scheme")
    logger.info("="*70)
    logger.info(f"  All 6 C-H fragments calculated")
    logger.info(f"  Total fragments: {len(fragments)}")
    
    # Run SIE workflow
    results = sie.run(
        fragments=fragments,
        minao='minao',
        svd_threshold=1e-8,
        eta_occ=1e-8,
        eta_virt=1e-8
    )
    
    # Print results
    energy_results = results['energy_results']
    print("\n" + "="*70)
    print("Results Comparison:")
    print("="*70)
    print(f"RHF energy:              {mf.e_tot:.8f} a.u.")
    print(f"\nFull CCSD:")
    print(f"  Correlation energy:    {full_ccsd.e_corr:.8f} a.u.")
    print(f"  Total energy:          {full_ccsd.e_tot:.8f} a.u.")
    print(f"\nSIE Embedding (All 6 C-H Fragments):")
    print(f"  Fragment energies:")
    for i, e in enumerate(energy_results['fragment_energies']):
        print(f"    Fragment {i} (C{i}-H{i+6}): {e:.8f} a.u.")
    print(f"  Correlation energy:    {energy_results['total_correlation']:.8f} a.u.")
    print(f"  Total energy:          {energy_results['total_energy']:.8f} a.u.")
    
    # Check symmetry: all fragments should have similar energies
    fragment_energies = energy_results['fragment_energies']
    mean_frag_energy = np.mean(fragment_energies)
    std_frag_energy = np.std(fragment_energies)
    print(f"\nFragment Energy Statistics (verifying symmetry):")
    print(f"  Mean:    {mean_frag_energy:.8f} a.u.")
    print(f"  Std Dev: {std_frag_energy:.8e} a.u.")
    print(f"  Max:     {np.max(fragment_energies):.8f} a.u.")
    print(f"  Min:     {np.min(fragment_energies):.8f} a.u.")
    
    print(f"\nError (SIE - Full CCSD):")
    error_corr = energy_results['total_correlation'] - full_ccsd.e_corr
    error_total = energy_results['total_energy'] - full_ccsd.e_tot
    print(f"  Correlation energy:    {error_corr:.8f} a.u. ({error_corr*1000:.4f} mEh)")
    print(f"  Total energy:          {error_total:.8f} a.u. ({error_total*1000:.4f} mEh)")
    
    # Check that error is less than 1 mEh
    error_meh = abs(error_corr * 1000)
    print(f"\nAssertion: Error < 1.0 mEh")
    print(f"  Actual error: {error_meh:.4f} mEh")
    
    if error_meh < 1.0:
        print(f"  ✓ PASSED")
    else:
        print(f"  ✗ FAILED")
        raise AssertionError(f"Error {error_meh:.4f} mEh exceeds threshold of 1.0 mEh")
    
    print("="*70 + "\n")
    
    return results, full_ccsd


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SIE Embedding Tests for Benzene (C6H6) Molecule")
    print("="*70 + "\n")
    
    # Run tests
    try:
        # Test 1: Single C-H fragment with 6-fold symmetry
        test1_results, full_ccsd1 = test_benzene_one_ch_fragment()
        
        # Test 2: All 6 C-H fragments for verification
        test2_results, full_ccsd2 = test_benzene_all_ch_fragments()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"TEST FAILED WITH ERROR:")
        print(f"{'='*70}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        sys.exit(1)
