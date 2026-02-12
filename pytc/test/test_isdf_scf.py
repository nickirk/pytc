
import time
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.scf import TCSCF
from pytc.jastrow.rexp import REXP

jax.config.update("jax_enable_x64", True)

def make_h_chain(n, dist=1.4):
    """Create a hydrogen chain of length n."""
    atom_str = ""
    for i in range(n):
        atom_str += f"H 0 0 {i*dist}; "
    return atom_str[:-2]

def run_benchmark():
    print("Running ISDF TCSCF Benchmark on H4 chain...", flush=True)
    
    # System Setup
    mol = gto.M(atom=make_h_chain(2), basis='sto-3g', verbose=5)
    print(f"System: H2, Basis: STO-3G, N_orb: {mol.nao_nr()}", flush=True)
    
    # Jastrow Setup
    jastrow = REXP()
    params = {'alpha': jnp.array([1.0])}
    
    # --- Standard TCSCF ---
    print("\n--- Standard TCSCF ---")
    mf_std = TCSCF(mol, jastrow, params)
    
    start_time = time.time()
    mf_std.kernel()
    end_time = time.time()
    
    time_std = end_time - start_time
    e_std = mf_std.e_tot
    print(f"Total Time: {time_std:.4f} s")
    print(f"Energy: {e_std:.8f} Ha")
    
    # --- ISDF TCSCF ---
    print("\n--- ISDF TCSCF (Rank 100) ---")
    mf_isdf = TCSCF(mol, jastrow, params).isdf(n_rank=400)
    
    start_time = time.time()
    mf_isdf.kernel()
    end_time = time.time()
    
    time_isdf = end_time - start_time
    e_isdf = mf_isdf.e_tot
    print(f"Total Time: {time_isdf:.4f} s")
    print(f"Energy: {e_isdf:.8f} Ha")
    
    # --- Comparison ---
    print("\n--- Results ---")
    print(f"Speedup: {time_std / time_isdf:.2f}x")
    print(f"Energy Diff: {abs(e_std - e_isdf):.2e} Ha")
    
    # Verify accuracy
    if abs(e_std - e_isdf) > 1e-3:
        print("WARNING: Energy difference is large!")
    else:
        print("Accuracy check passed.")

if __name__ == "__main__":
    run_benchmark()
