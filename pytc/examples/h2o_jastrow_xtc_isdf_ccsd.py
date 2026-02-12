import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf, cc, lib

from pytc import xtc
from pytc.jastrow import rexp

# Set number of threads for PySCF/NumPy
lib.num_threads(1)

# Define the molecule (H2O with geometry provided by user)
mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='ccpvdz')
mf = scf.RHF(mol)
mf.kernel()

# Initialize Jastrow factor and parameters (Autodiff version)
# u(r) = 0.5 * r * exp(-alpha * r)
my_jastrow = rexp.REXP()
jastrow_params = {'alpha': jnp.array([1.0])}

# Initialize XTC object from PySCF mean-field object
my_xtc = xtc.XTC.from_pyscf(mf, my_jastrow, grid_lvl=2)

print("--- Exact XTC ---")
print("Computing exact delta_U...")
# Get full exact delta_U for comparison
delta_U_exact = np.asarray(my_xtc.get_delta_U(jastrow_params).block_until_ready())

print("Making exact XTC eris...")
eris_exact = my_xtc.make_eris(mf, jastrow_params)

print("Running exact XTC CCSD...")
mycc_exact = cc.rccsd.RCCSD(mf)
tc_e_corr_exact, t1_exact, t2_exact = mycc_exact.kernel(eris=eris_exact)

# Calculating HF energy with xtc integrals manually
no = mycc_exact.nocc
e_hf_exact = 2*np.einsum('ii->', eris_exact.fock[:no,:no])
e_hf_exact -= 2*np.einsum('iijj ->', eris_exact.oooo)
e_hf_exact += np.einsum('ijji ->', eris_exact.oooo)
e_hf_exact += eris_exact.e_core
e_tot_exact = tc_e_corr_exact + e_hf_exact
print(f"E_XTC_CCSD (Exact) = {e_tot_exact:.10f}")

print("\n--- ISDF XTC Convergence Study ---")
factors = [5, 8, 10, 15]
results = []

for factor in factors:
    n_rank = factor * my_xtc.n_orb
    print(f"\nTesting ISDF Rank: {n_rank}")
    
    print("Initializing ISDFXTC...")
    my_isdf_xtc = xtc.ISDFXTC.from_xtc(my_xtc, n_rank=n_rank)

    # Precompute ISDF kernels
    print("Computing ISDF kernels...")
    my_isdf_xtc = my_isdf_xtc.isdf(jastrow_params)

    # Compare delta_U
    print("Comparing delta_U...")
    delta_U_isdf = np.asarray(my_isdf_xtc.get_delta_U(jastrow_params).block_until_ready())
    mae_delta_U = np.max(np.abs(delta_U_isdf - delta_U_exact))
    
    print("Making ISDF XTC eris...")
    eris_isdf = my_isdf_xtc.make_eris(mf, jastrow_params)
    
    # Compare ERI blocks
    mae_oooo = np.max(np.abs(eris_isdf.oooo - eris_exact.oooo))
    mae_oovv = np.max(np.abs(eris_isdf.oovv - eris_exact.oovv))

    print("Running ISDF XTC CCSD...")
    mycc_isdf = cc.rccsd.RCCSD(mf)
    tc_e_corr_isdf, t1_isdf, t2_isdf = mycc_isdf.kernel(eris=eris_isdf)

    # Calculate HF energy for ISDF XTC
    e_hf_isdf = 2*np.einsum('ii->', eris_isdf.fock[:no,:no])
    e_hf_isdf -= 2*np.einsum('iijj ->', eris_isdf.oooo)
    e_hf_isdf += np.einsum('ijji ->', eris_isdf.oooo)
    e_hf_isdf += eris_isdf.e_core
    e_tot_isdf = tc_e_corr_isdf + e_hf_isdf
    
    error_hartree = e_tot_isdf - e_tot_exact
    results.append({
        'factor': factor,
        'e_tot': e_tot_isdf,
        'error_mEh': error_hartree * 1000,
        'mae_delta_U': mae_delta_U,
        'mae_oooo': mae_oooo,
        'mae_oovv': mae_oovv,
    })
    print(f"E_XTC_CCSD (ISDF, factor={factor}) = {e_tot_isdf:.10f}")
    print(f"Energy Error: {error_hartree:.10f} Hartree ({error_hartree * 1000:.6f} mHartree)")
    print(f"MAE delta_U: {mae_delta_U:.2e}")
    print(f"MAE oooo: {mae_oooo:.2e}")
    print(f"MAE oovv: {mae_oovv:.2e}")

print("\n--- Summary of Convergence ---")
header = f"{'Factor':<8} {'Error (mEh)':<15} {'MAE delta_U':<15} {'MAE oooo':<15} {'MAE oovv':<15}"
print(header)
print("-" * len(header))
for r in results:
    print(f"{r['factor']:<8} {r['error_mEh']:<15.6f} {r['mae_delta_U']:<15.2e} {r['mae_oooo']:<15.2e} {r['mae_oovv']:<15.2e}")
