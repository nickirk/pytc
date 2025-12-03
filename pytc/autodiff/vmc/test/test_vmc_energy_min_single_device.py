
import sys
import os
# Add ferminet to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pytc/lib/ferminet')))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import optax
from absl import app
from absl import flags
from absl import logging

from ferminet import constants
from ferminet import hamiltonian
from ferminet import jastrows
from ferminet import mcmc
from ferminet import networks
from ferminet import pretrain
from ferminet import train
from ferminet.utils import system
# import kfac_jax # Not needed for serial

from pytc.autodiff.jastrow.ncusp import make_ncusp_jastrow
from pytc.autodiff.jastrow.bh import make_bh_jastrow
from pytc.autodiff.ansatz.sj import make_slater_jastrow
from pytc.autodiff.ansatz.det import get_hf_det

# Patch constants for serial execution
constants.pmean = lambda x, axis_name=None: x
constants.pmap = lambda f, axis_name=None, **kwargs: jax.jit(f)

FLAGS = flags.FLAGS

# Define flags if not already defined (to avoid errors if running multiple times)
if 'batch_size' not in FLAGS:
    flags.DEFINE_integer('batch_size', 4096, 'Number of walkers.')
if 'iterations' not in FLAGS:
    flags.DEFINE_integer('iterations', 1000, 'Number of optimization steps.')
if 'learning_rate' not in FLAGS:
    flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate.')

def main(argv):
  del argv

  logging.info("Starting Be atom VMC test (SERIAL) with Boys-Handy...")

  # 1. System Definition: Be atom
  atoms = [system.Atom('Be', (0, 0, 0))]
  electrons = (2, 2) # Be: 1s2 2s2
  
  logging.info(f"Atoms: {atoms}")
  logging.info(f"Electrons: {electrons}")

  # 2. Hartree-Fock
  logging.info("Solving Hartree-Fock...")
  
  class PytcMolWrapper:
      def __init__(self, atoms, electrons):
          self.atoms = atoms
          self.nelectron = sum(electrons)
          # We need pyscf mol for basis info
          import pyscf
          self.pyscf_mol = pyscf.gto.M(
              atom=[[a.symbol, a.coords] for a in atoms],
              basis='sto-3g',
              unit='bohr',
              spin=electrons[0]-electrons[1],
              charge=0
          )
          self.nbas = self.pyscf_mol.nbas
          
      def atom_coords(self):
          return [a.coords for a in self.atoms]
      
      def atom_charges(self):
          return [a.atomic_number for a in self.atoms]
          
      def bas_atom(self, i):
          return self.pyscf_mol.bas_atom(i)
          
      def bas_angular(self, i):
          return self.pyscf_mol.bas_angular(i)
          
      def eval_gto(self, mode, coords, shls_slice=None):
          # mode is 'GTOval_sph'
          # coords: (n, 3)
          # shls_slice: (start, end)
          return self.pyscf_mol.eval_gto(mode, coords, shls_slice=shls_slice)

  pytc_mol = PytcMolWrapper(atoms, electrons)

  hf_solution = get_hf_det(
      molecule=atoms,
      nspins=electrons,
      basis='sto-3g',
      restricted=True
  )
  logging.info("Hartree-Fock solved.")

  # 3. Jastrows
  # Simple EE - Wrapped to accept extra args
  ee_init, ee_apply_orig = jastrows.make_simple_ee_jastrow()
  def ee_apply(r_ee, params, nspins, **kwargs):
      return ee_apply_orig(r_ee, params, nspins)
  
  # Boys-Handy
  bh_init, bh_apply = make_bh_jastrow(pytc_mol)
  
  # Nuclear Cusp
  ncusp_init, ncusp_apply = make_ncusp_jastrow(pytc_mol)
  
  # 4. Ansatz
  # We can choose to use BH or SimpleEE. Let's use BH.
  log_psi = make_slater_jastrow(hf_solution, bh_apply, ncusp_apply)
  
  # Wrapper for signed log_psi (needed for local energy)
  def log_psi_signed(params, pos, spins, atom_coords, charges):
      return log_psi(params, pos, spins, atom_coords, charges)

  # Wrapper for abs log_psi (needed for loss and MCMC)
  def log_psi_abs(params, pos, spins, atom_coords, charges):
      return log_psi(params, pos, spins, atom_coords, charges)[1]

  # Vectorize
  batch_log_psi = jax.vmap(log_psi, in_axes=(None, 0, 0, 0, 0))
  batch_log_psi_signed = jax.vmap(log_psi_signed, in_axes=(None, 0, 0, 0, 0))
  batch_log_psi_abs = jax.vmap(log_psi_abs, in_axes=(None, 0, 0, 0, 0))

  # 5. Initialization
  key = jax.random.PRNGKey(42)
  key, subkey = jax.random.split(key)
  
  params = {}
  # params['jastrow'] = ee_init() # Using BH instead
  params['jastrow'] = bh_init()
  params['ncusp'] = ncusp_init()
  
  logging.info("Params initialized.")

  # 6. MCMC
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
  
  # JIT compile the MCMC step for burn-in
  mcmc_step_jit = jax.jit(mcmc_step)

  logging.info("Burning in MCMC...")
  for i in range(20):
      print(f"Burn-in step {i+1}/20...")
      key, subkey = jax.random.split(key)
      data, pmove = mcmc_step_jit(params, data, subkey, width=0.1)
  logging.info("Burn-in complete.")

  # 7. Optimization
  optimizer = optax.adam(FLAGS.learning_rate)
  
  local_energy_fn = hamiltonian.local_energy(
      f=log_psi_signed,
      charges=atom_charges,
      nspins=electrons,
      use_scan=False
  )
  
  from ferminet import loss as qmc_loss_functions
  evaluate_loss = qmc_loss_functions.make_loss(
      network=log_psi_abs,
      local_energy=local_energy_fn,
      clip_local_energy=5.0,
      clip_from_median=True,
      center_at_clipped_energy=True
  )

  opt_update_step = train.make_opt_update_step(evaluate_loss, optimizer)
  train_step = train.make_training_step(mcmc_step, opt_update_step)
  # No pmap
  train_step = jax.jit(train_step)
  
  opt_state = optimizer.init(params)
  
  # No replication
  # params = kfac_jax.utils.replicate_all_local_devices(params)
  # opt_state = kfac_jax.utils.replicate_all_local_devices(opt_state)
  
  # No reshaping data
  # data = networks.FermiNetData(...)
  
  mcmc_width = jnp.asarray(0.1)
  adapt_frequency = 10
  pmoves = np.zeros(adapt_frequency)
  
  logging.info("Starting optimization...")
  for t in range(FLAGS.iterations):
      key, subkey = jax.random.split(key)
      
      data, params, opt_state, loss, aux_data, pmove = train_step(
          data, params, opt_state, subkey, mcmc_width
      )
      
      # pmove is scalar in serial
      current_pmove = pmove
      mcmc_width, pmoves = mcmc.update_mcmc_width(
          t, mcmc_width, adapt_frequency, current_pmove, pmoves
      )
      
      if t % 10 == 0:
          logging.info(f"Step {t}: Energy = {loss:.6f}, pmove = {current_pmove:.2f}")

if __name__ == '__main__':
  app.run(main)
