import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
import time
import os
import sys

# 1. Configure JAX for high precision
jax.config.update("jax_enable_x64", True)

from pytc.autodiff.xtc import XTC, ISDFXTC
from pytc.autodiff.jastrow.rexp import REXP

def run_autodiff_isdf_example():
    """
    Example demonstrating the JAX/autodiff implementation of 
    Interpolative Density Fitting (ISDF) for Transcorrelated (TC) methods.
    
    This script showcases:
    1. Using the JAX/autodiff API for TC calculations.
    2. Out-of-core storage of ISDF kernels using HDF5.
    3. RAM and VRAM management using batching and blocking.
    """
    
    # --- 1. Setup Molecule & Mean-Field ---
    # We use H2O with cc-pVDZ basis for a realistic demonstration.
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', 
        basis='cc-pvdz', 
        verbose=0
    )
    mf = scf.RHF(mol)
    print("Running reference RHF...")
    mf.kernel()
    print(f"RHF energy: {mf.e_tot:.8f} Hartree")

    # --- 2. Initialize JAX XTC ---
    # Jastrow factor with initial parameter alpha=1.0
    jastrow = REXP()
    jastrow_params = {'alpha': jnp.array([1.0])}
    
    # Initialize the standard XTC object
    # grid_lvl=2 is a good balance between speed and accuracy
    my_xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=2)
    print(f"Grid size: {len(my_xtc.grid_points)} points")

    # --- 3. ISDF Decomposition (Out-of-Core) ---
    print("\n--- ISDF Decomposition ---")
    n_mo = mf.mo_coeff.shape[1]
    n_rank = 10 * n_mo  # Typical rank: 8-12 times number of orbitals
    
    # Specify a path to save ISDF kernels to disk (HDF5)
    # This enables out-of-core processing for large systems.
    save_path = "h2o_isdf_kernels.h5"
    if os.path.exists(save_path):
        os.remove(save_path) # Start fresh
        
    start_time = time.time()
    # ls_grid_batch_size controls memory usage during ISDF linear solver
    my_xtc_isdf = ISDFXTC.from_xtc(
        my_xtc, 
        n_rank=n_rank, 
        save_path=save_path,
        ls_grid_batch_size=8192  # Reduced to save RAM
    )
    print(f"ISDF decomposition took: {time.time() - start_time:.2f} seconds")

    # --- 4. Compute ISDF Intermediates (RAM Management) ---
    print("\n--- Computing ISDF Kernels (Out-of-Core) ---")
    start_time = time.time()
    
    # RAM and VRAM can be managed via several keyword arguments:
    # - batch_size: number of grid points processed in JAX scans.
    # - orb_block_size: block size for orbital indices in heavy X kernel calculation (VRAM).
    # - host_grid_block_size: block size for grid points when sharding to GPUs (Host RAM).
    my_xtc_isdf = my_xtc_isdf.isdf(
        jastrow_params,
        batch_size=512,            # Lower value saves GPU VRAM
        orb_block_size=32,          # Lower value saves GPU VRAM
        host_grid_block_size=10000  # Lower value saves Host RAM
    )
    print(f"Kernel computation took: {time.time() - start_time:.2f} seconds")

    # --- 5. Compute TC Integrals (ERIs) ---
    print("\n--- Computing ISDF XTC ERIs ---")
    start_time = time.time()
    
    # make_eris returns a ChemistsERIs-like object compatible with CCSD
    # It internally uses the precomputed ISDF kernels
    eris_isdf = my_xtc_isdf.make_eris(mf, jastrow_params)
    
    print(f"ISDF ERI computation took: {time.time() - start_time:.2f} seconds")

    # --- 6. Results Verification ---
    def get_tc_hf_energy(eris, nocc):
        # E_HF = 2 * sum(h_ii) + sum(2*(ii|jj) - (ij|ji)) + E_nuc
        # eris.fock already contains the modified core Hamiltonian + 2J-K
        e_hf = 2 * np.einsum('ii->', eris.fock[:nocc, :nocc])
        e_hf -= 2 * np.einsum('iijj ->', eris.oooo)
        e_hf += np.einsum('ijji ->', eris.oooo)
        e_hf += eris.e_core
        return e_hf

    nocc = mol.nelectron // 2
    e_tc_hf = get_tc_hf_energy(eris_isdf, nocc)
    print(f"\nISDF TC-HF energy: {e_tc_hf:.8f} Hartree")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
    print("\nExample completed successfully.")

if __name__ == "__main__":
    run_autodiff_isdf_example()
