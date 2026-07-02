import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import time

from pyscf import gto, scf

from pytc.vmc import sample, optimize_ref_var
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import CompositeJastrow, NuclearCusp, NeuralEE, NeuralEN, NeuralEEN, REXP 
from pytc.jastrow import BoysHandy



def main():
    """Run optimization test on the specified molecule."""
    # Create molecule
    mol_name = "Be"
    mol = gto.Mole()
    mol.atom = "Be 0 0 0;" 
    mol.basis = 'cc-pVDZ' 
    #mol.unit = 'A'
    mol.spin = 0
    mol.verbose = 5
    mol.build()

    # Run PySCF calculation for reference energy
    mf = scf.RHF(mol)
    mf.verbose = 5
    mf.kernel()
    print(f"Reference HF energy: {mf.e_tot:.6f}")
  

    # Create determinant from HF solution
    det = SlaterDet.create(mol, mf.mo_coeff)

    # Create jastrow factors with names

    # Create neural jastrow factors
    #n_per_layer = 6
    #layer_widths = [n_per_layer] * 3
    #jee = NeuralEE(mol, layer_widths=layer_widths, name="jee")
    #jen = NeuralEN(mol, layer_widths=layer_widths, name="jen")
    #jeen = NeuralEEN(mol, layer_widths=layer_widths, name="jeen")

    # orbital nuclear cusp jastrow is important for noise reduction
    jncusp = NuclearCusp.create(mol, name="ncusp")

    # Boys-Handy jastrow
    jbh = BoysHandy.create(mol, terms_per_nucleus=None, name="bh")

    # Use composite jastrow to glue together the jastrow factors
    jastrow_phase1 = CompositeJastrow.create([jncusp, jbh])
    jastrow_params_phase1 = jastrow_phase1.init_params()

    # Create SlaterJastrow ansatz for phase 1
    linear_coeffs = jnp.ones(1)  # Single determinant

    
    # Phase 1: Optimize jastrow factors
    print("\nPhase 1: Optimizing jastrow factors...")
    
    # Create Slater-Jastrow ansatz for phase 1
    sj_ansatz_phase1 = SlaterJastrow.create(mol, jastrow_phase1, [det])
    params_phase1 = [jastrow_params_phase1, linear_coeffs]
    
    # Settings for phase 1
    n_walkers = 5000  
    n_opt_steps_phase2 = 10 # total optimization steps
    burn_in_steps = 2000 # burn-in steps
    n_steps = 50 # resampling wavefunction every n_steps
    step_size = 0.02 # step size for the random walk
    learning_rate = 0.001 # learning rate for the optimizer
    key = random.PRNGKey(43)

    start_time = time.time() 
    opt_results = optimize_ref_var(
        sj_ansatz_phase1,
        params=params_phase1,
        n_walkers=n_walkers,
        n_steps=n_steps,
        step_size=step_size,
        burn_in_steps=burn_in_steps,
        n_opt_steps=n_opt_steps_phase2,
        optimizer_type='newton',
        learning_rate=learning_rate,
        key=key,
    )

    end_time = time.time()
    ###### save jastrow parameters
    print(f"Total optimization completed in {end_time - start_time:.2f} seconds")

    ## Save jastrow parameters (only the jastrow part)
    jastrow_file = f'{mol_name}_dz_bh.hdf5'
    # Get the last optimized parameters
    opt_params = opt_results["params"][-1]
    # Save only the jastrow parameters
    jastrow_phase1.save_params(opt_params[0], filename=jastrow_file)

    # loaded_jastrow_params = jastrow_phase1.read_params(jastrow_file)
    opt_jastrow_params = opt_results["params"][-1][0] 
    ### Combine with linear coefficients
    linear_coeffs = jnp.ones(1)  # Single determinant
    params_for_sampling= [opt_jastrow_params, linear_coeffs]

    n_samples = 5000
    n_walkers = 5000
    burn_in_steps = 3000
    step_size = 0.02
    samples = sample(
        sj_ansatz_phase1,
        params=params_for_sampling,  # Use explicitly loaded and converted parameters
        n_walkers=n_walkers,
        n_steps=n_samples,
        step_size=step_size,
        burn_in_steps=burn_in_steps,
        use_importance_sampling=False,
        key=key
    )

    ### Save results
    import h5py
    with h5py.File(f"vmc_{mol_name}.hdf5", "w") as f:
        # Save phase 1 results
        f.create_group("phase1")
        for key, value in opt_results.items():
            if isinstance(value, jnp.ndarray):
                f.create_dataset(f"phase1/{key}", data=value)
        # Save sampled energies
        f.create_group("phase2")
        for key, value in samples.items():
            if isinstance(value, jnp.ndarray):
                f.create_dataset(f"phase2/{key}", data=value)

    do_ccsd(mf, jastrow_phase1, opt_params[0])

def do_ccsd(mf, jastrow_factor, params):
    from pyscf.cc import CCSD
    import numpy as np
    from functools import reduce

    from pytc.xtc import XTC
    from pytc.tc_helper import get_eri
    from pytc.solver import jax_xtc_ccsd
    # only need the jastrow params
    xtc = XTC.from_pyscf(mf, jastrow_factor, grid_lvl=2)

    mycc = CCSD(mf)
    e_corr, t1, t2 = mycc.kernel()
    mycc.verbose = 5
    print("E_CCSD = ", e_corr)
    #self.assertAlmostEqual(e_corr, -0.04503138331130402, places=6)
    t = mycc.amplitudes_to_vector(t1, t2)
    print("|t2| = ", np.linalg.norm(t2))
    print("|t1+t2| = ", np.linalg.norm(t))
    eri1 = get_eri(mf, xtc.mo_coeff)
    h1e = mycc._scf.get_hcore()
    h1e = reduce(np.dot, (xtc.mo_coeff.T, h1e, xtc.mo_coeff))
    
    e_hf_0 = 2. * np.einsum('ii->', h1e[:mycc.nocc, :mycc.nocc])
    e_dir = 2. * np.einsum('jjii->', eri1[:mycc.nocc, :mycc.nocc, :mycc.nocc, :mycc.nocc])
    e_ex = -1. * np.einsum('ijji->', eri1[:mycc.nocc, :mycc.nocc, :mycc.nocc, :mycc.nocc])
    e_hf_0 += (e_dir + e_ex) + mycc._scf.energy_nuc()
    print("Check e_hf = ",  e_hf_0)

    # Use pytc's own JAX-native xTC-CCSD solver, which correctly handles the
    # non-Hermitian transcorrelated integrals (PySCF's stock RCCSD assumes
    # ERI symmetries that don't hold here and silently gives the wrong energy).
    myrcc = jax_xtc_ccsd.RCCSD(mf, xtc, params)
    eris = xtc.make_eris(mf, params)
    tc_e_corr, t1, t2 = myrcc.kernel(eris=eris)
    t = myrcc.amplitudes_to_vector(t1, t2)
    print("|t2| = ", np.linalg.norm(t2))
    print("|t1+t2| = ", np.linalg.norm(t))
    print("corr E_XTC_CCSD = ", myrcc.e_corr)
    # get the hf energy using fock and eris
    no = myrcc.nocc
    tc_e_hf = 2. * np.einsum('ii->', eris.fock[:no, :no])
    tc_e_dir = 2. * np.einsum('jjii->', eris.oooo)
    tc_e_ex = -1. * np.einsum('ijji->', eris.oooo)

    tc_e_hf += -(tc_e_dir + tc_e_ex) + eris.e_core 
    print("Check xtc e_hf = ",  tc_e_hf)
    print("E_XTC_CCSD = ", myrcc.e_corr + tc_e_hf)
    print("Ref exact ground state energy =", -14.6673)

if __name__ == "__main__":
    main()
