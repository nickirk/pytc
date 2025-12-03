import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from absl import flags
from absl import logging

# Add ferminet to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../pytc/lib/ferminet')))

from ferminet import constants
from ferminet import hamiltonian
from ferminet import mcmc
from ferminet import networks
from ferminet import train
from ferminet.utils import system

from pytc.autodiff.ansatz.sj import make_slater_jastrow
from pytc.autodiff.ansatz.det import get_hf_det

# Patch constants for serial execution
constants.pmean = lambda x, axis_name=None: x
constants.pmap = lambda f, axis_name=None, **kwargs: jax.jit(f)

FLAGS = flags.FLAGS

if 'batch_size' not in FLAGS:
    flags.DEFINE_integer('batch_size', 4096, 'Number of walkers.')
if 'mcmc_steps' not in FLAGS:
    flags.DEFINE_integer('mcmc_steps', 3000, 'Number of MCMC steps for sampling.')
if 'burn_in' not in FLAGS:
    flags.DEFINE_integer('burn_in', 500, 'Number of burn-in steps.')

def main(argv):
    del argv
    
    logging.info("Starting HF Sampling Verification Test...")
    
    # 1. System Definition: Be atom
    atoms = [system.Atom('Be', (0, 0, 0))]
    electrons = (2, 2) # Be: 1s2 2s2
    
    logging.info(f"Atoms: {atoms}")
    logging.info(f"Electrons: {electrons}")
    
    # 2. Hartree-Fock
    logging.info("Solving Hartree-Fock...")
    hf_solution = get_hf_det(
        molecule=atoms,
        nspins=electrons,
        basis='sto-3g',
        restricted=True
    )
    logging.info("Hartree-Fock solved.")
    
    # 3. Ansatz (Pure HF, no Jastrow)
    # We use make_slater_jastrow but pass None for jastrows
    log_psi = make_slater_jastrow(hf_solution, jastrow_apply=None, ncusp_apply=None)
    
    # Wrapper for signed log_psi (needed for local energy)
    def log_psi_signed(params, pos, spins, atom_coords, charges):
        return log_psi(params, pos, spins, atom_coords, charges)

    # Wrapper for abs log_psi (needed for MCMC)
    def log_psi_abs(params, pos, spins, atom_coords, charges):
        return log_psi(params, pos, spins, atom_coords, charges)[1]

    # Vectorize
    batch_log_psi_signed = jax.vmap(log_psi_signed, in_axes=(None, 0, 0, 0, 0))
    batch_log_psi_abs = jax.vmap(log_psi_abs, in_axes=(None, 0, 0, 0, 0))
    
    # 4. Initialization
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    params = {} # No parameters needed for pure HF
    
    # 5. MCMC Setup
    batch_size = FLAGS.batch_size
    atom_pos = jnp.array([atom.coords for atom in atoms])
    atom_charges = jnp.array([atom.charge for atom in atoms])
    
    pos, spins = train.init_electrons(
        key=subkey,
        molecule=atoms,
        electrons=electrons,
        batch_size=batch_size,
        init_width=0.5
    )
    
    data = networks.FermiNetData(
        positions=pos,
        spins=spins,
        atoms=jnp.tile(atom_pos[None, ...], (batch_size, 1, 1)),
        charges=jnp.tile(atom_charges[None, ...], (batch_size, 1))
    )
    
    mcmc_step = mcmc.make_mcmc_step(
        batch_network=batch_log_psi_abs,
        batch_per_device=batch_size,
        steps=10,
        atoms=atom_pos,
    )
    mcmc_step = jax.jit(mcmc_step)
    
    mcmc_width = jnp.asarray(0.1)
    adapt_frequency = 10
    pmoves = np.zeros(adapt_frequency)
    current_pmove = 0.5
    
    # 6. Burn-in
    logging.info(f"Burning in MCMC for {FLAGS.burn_in} steps...")
    for i in range(FLAGS.burn_in):
        key, subkey = jax.random.split(key)
        data, pmove = mcmc_step(params, data, subkey, width=mcmc_width)
        
        pmoves[i % adapt_frequency] = pmove
        if i > 0 and i % adapt_frequency == 0:
            mcmc_width, pmoves = mcmc.update_mcmc_width(
                i, mcmc_width, adapt_frequency, current_pmove, pmoves
            )
        current_pmove = pmove
    logging.info(f"Burn-in complete. Final pmove={current_pmove:.2f}, width={mcmc_width:.4f}")
    
    # 7. Sampling and Energy Calculation
    local_energy_fn = hamiltonian.local_energy(
        f=log_psi_signed,
        charges=atom_charges,
        nspins=electrons,
        use_scan=False
    )
    batch_local_energy = jax.vmap(local_energy_fn, in_axes=(None, None, 0))
    batch_local_energy = jax.jit(batch_local_energy)
    
    energies = []
    
    logging.info(f"Sampling for {FLAGS.mcmc_steps} steps...")
    for i in range(FLAGS.mcmc_steps):
        key, subkey = jax.random.split(key)
        data, pmove = mcmc_step(params, data, subkey, width=mcmc_width)
        
        # Calculate energy
        key, subkey = jax.random.split(key)
        e_l, _ = batch_local_energy(params, subkey, data)
        energies.append(jnp.mean(e_l))
        
        if i % 100 == 0:
             logging.info(f"Sample step {i}: Current Mean Energy = {jnp.mean(e_l):.6f}")

    energies = np.array(energies)
    n_nan = np.sum(np.isnan(energies))
    if n_nan > 0:
        logging.warning(f"Found {n_nan} NaN values in energy samples. Filtering them out.")
        energies = energies[~np.isnan(energies)]
        
    if len(energies) == 0:
        logging.error("All energy samples are NaN!")
        return

    mean_energy = np.mean(energies)
    std_error = np.std(energies) / np.sqrt(len(energies))
    
    logging.info(f"Final Mean Energy: {mean_energy:.6f} +/- {std_error:.6f} Ha")
    
    # Retrieve HF energy from mean_field
    hf_energy = None
    if hasattr(hf_solution, 'mean_field') and hasattr(hf_solution.mean_field, 'e_tot'):
        hf_energy = hf_solution.mean_field.e_tot
         
    if hf_energy:
        logging.info(f"Reference HF Energy: {hf_energy:.6f} Ha")
        diff = abs(mean_energy - hf_energy)
        logging.info(f"Difference: {diff:.6f} Ha")
        
        if diff < 1e-2:
             logging.info("SUCCESS: VMC energy matches HF energy.")
        else:
             logging.info("WARNING: VMC energy differs from HF energy.")
    else:
        logging.info("Reference HF energy not directly available in hf_solution object.")

if __name__ == '__main__':
    app.run(main)
