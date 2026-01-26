import numpy as np
from pyscf import gto, scf, lib
from pytc import xtc, jastrow
import time

def run_isdf_example():
    """
    Example demonstrating Interpolative Density Fitting (ISDF) for 
    Transcorrelated (TC) integral evaluation.
    
    ISDF approximates orbital products as a linear combination of a 
    reduced number of interpolation points (auxiliary basis).
    This significantly accelerates the computation of two-body 
    TC integrals while maintaining high accuracy.
    """
    # 1. Setup Molecule
    # We use a simple N2 molecule for this example to ensure quick execution.
    # A larger basis set and higher grid level will show more significant speedup with ISDF.
    mol = gto.M(
        atom='N 0 0 0; N 0 0 2.07', 
        basis='cc-pVDZ', 
        unit='Bohr',
        verbose=0
    )
    mf = scf.RHF(mol)
    print("Running reference RHF...")
    mf.kernel()
    print(f"RHF energy: {mf.e_tot:.8f} Hartree")

    # 2. Define Jastrow factor
    # We use a simple REXP (Reduced Exponential) Jastrow factor.
    my_jastrow = jastrow.REXP([1.0])
    
    # 3. Standard XTC calculation (Numerical Integration)
    print("\n--- Standard XTC (Numerical Integration) ---")
    start_time = time.time()
    # Use grid_lvl=2 for a balance between speed and accuracy.
    my_xtc = xtc.XTC(mf, my_jastrow, grid_lvl=2)
    print("Computing standard XTC ERIs...")
    eris_standard = my_xtc.make_eris()
    standard_time = time.time() - start_time
    print(f"Standard XTC ERI computation took: {standard_time:.2f} seconds")

    # 4. ISDF XTC calculation
    print("\n--- ISDF-accelerated XTC ---")
    start_time = time.time()
    my_xtc_isdf = xtc.XTC(mf, my_jastrow, grid_lvl=2)
    
    print("Performing ISDF decomposition...")
    # n_rank controls the number of interpolation points.
    # Setting it to 200 for this small system provides high accuracy.
    # For larger systems, a common choice is 4-10 times the number of MOs.
    my_xtc_isdf.isdf(n_rank=200) 
    
    print("Computing ISDF XTC ERIs...")
    eris_isdf = my_xtc_isdf.make_eris()
    isdf_time = time.time() - start_time
    print(f"ISDF XTC ERI computation took: {isdf_time:.2f} seconds")

    # 5. Accuracy and Performance Comparison
    speedup = standard_time / isdf_time
    print(f"\n--- Comparison ---")
    print(f"Speedup factor: {speedup:.2f}x")

    # Compare (oo|oo) block (occupied-occupied integrals)
    diff_oooo = np.linalg.norm(eris_standard.oooo - eris_isdf.oooo)
    print(f"Frobenius norm of difference in (oo|oo) integrals: {diff_oooo:.2e}")
    
    # Helper to calculate TC-HF energy from ERIs
    def get_tc_hf_energy(eris, nocc):
        # E_HF = 2 * sum(h_ii) + sum(2*(ii|jj) - (ij|ji)) + E_nuc
        # Note: make_eris() already handles delta_h and e_core
        e_hf = 2 * np.einsum('ii->', eris.fock[:nocc, :nocc])
        e_hf -= 2 * np.einsum('iijj ->', eris.oooo)
        e_hf += np.einsum('ijji ->', eris.oooo)
        e_hf += eris.e_core
        return e_hf

    nocc = mol.nelectron // 2
    e_tc_hf_standard = get_tc_hf_energy(eris_standard, nocc)
    e_tc_hf_isdf = get_tc_hf_energy(eris_isdf, nocc)
    
    print(f"Standard TC-HF energy: {e_tc_hf_standard:.8f}")
    print(f"ISDF-TC-HF energy:     {e_tc_hf_isdf:.8f}")
    print(f"Energy difference:      {abs(e_tc_hf_standard - e_tc_hf_isdf):.2e} Hartree")

if __name__ == "__main__":
    run_isdf_example()
