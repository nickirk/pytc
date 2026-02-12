import time
import numpy as np
import jax
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.xtc import XTC
from pytc.jastrow import REXP
import resource

# Enable float64
jax.config.update("jax_enable_x64", True)

def get_peak_memory_mb():
    """Returns peak memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # on Mac, ru_maxrss is in bytes. On Linux it's in KB.
    # Assuming Mac based on user info.
    return usage.ru_maxrss / 1024 / 1024 

def profile_h4():
    print(f"Initial Peak Memory: {get_peak_memory_mb():.2f} MB")
    print("Setting up H4 chain...")
    # H4 chain with 1.0 Bohr spacing
    atom = []
    for i in range(4):
        atom.append(f'H 0 0 {i*1.0}')
    mol = gto.M(atom=atom, basis='cc-pvtz', unit='Bohr', verbose=3)
    mf = scf.RHF(mol)
    mf.kernel()
    
    print(f"Peak Memory after SCF: {get_peak_memory_mb():.2f} MB")
    
    print("Initializing XTC...")
    # Use a reasonable grid level. Default is 2.
    jastrow = REXP()
    params = {'alpha': jnp.array([1.0])}
    
    start_time = time.time()
    xtc = XTC.from_pyscf(mf, jastrow, grid_lvl=2)
    print(f"XTC initialized in {time.time() - start_time:.2f}s", flush=True)
    print(f"Grid size: {xtc.grid_points.shape}", flush=True)
    print(f"Rho shape: {xtc.rho.shape}", flush=True)
    print(f"Peak Memory after XTC init: {get_peak_memory_mb():.2f} MB")
    
    print("Compiling and running get_2b...", flush=True)
    start_time = time.time()
    # Trigger JIT
    res = xtc.get_2b(params)
    # Block until ready
    res.block_until_ready()
    print(f"get_2b finished in {time.time() - start_time:.2f}s")
    print(f"Peak Memory after get_2b: {get_peak_memory_mb():.2f} MB")
    
    print("Compiling and running make_eris...")
    start_time = time.time()
    eris = xtc.make_eris(mf, params)
    print(f"make_eris finished in {time.time() - start_time:.2f}s")
    print(f"Peak Memory after make_eris: {get_peak_memory_mb():.2f} MB")
    
    print("Done.")

if __name__ == "__main__":
    profile_h4()
