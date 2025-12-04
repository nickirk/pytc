import sys
import os
# Add ferminet to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../pytc/lib/ferminet')))

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
from ferminet.jastrows import make_simple_ee_jastrow
# import kfac_jax # Not needed for serial

from pytc.autodiff.jastrow.ncusp import make_ncusp_jastrow
from pytc.autodiff.jastrow.bh import make_bh_jastrow
from pytc.autodiff.ansatz.sj import make_slater_jastrow
from pytc.autodiff.ansatz.det import get_hf_det
from pytc.autodiff.vmc.loss import make_variance_loss
from pytc.autodiff.vmc.mcmc_utils import create_optimizer
from pytc.utils.checkpoint import Checkpoint

# Patch constants for serial execution
constants.pmean = lambda x, axis_name=None: x
constants.pmap = lambda f, axis_name=None, **kwargs: jax.jit(f)

FLAGS = flags.FLAGS

# Define flags if not already defined (to avoid errors if running multiple times)
if 'batch_size' not in FLAGS:
    flags.DEFINE_integer('batch_size', 8192, 'Number of walkers.')
if 'iterations' not in FLAGS:
    flags.DEFINE_integer('iterations', 10, 'Number of optimization steps.')
if 'learning_rate' not in FLAGS:
    flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate.')
if 'n_opt' not in FLAGS:
    flags.DEFINE_integer('n_opt', 10, 'Number of optimization steps per MCMC step.')
if 'optimizer' not in FLAGS:
    flags.DEFINE_string('optimizer', 'kfac', 'Optimizer type: kfac, adam, etc.')
if 'atoms' not in FLAGS:
    flags.DEFINE_string('atoms', 'Be 0 0 0', 'Specify pyscf geometry')
if 'basis' not in FLAGS:
    flags.DEFINE_string('basis', 'ccpvdz', 'Specify basis')
if 'n_burn_in' not in FLAGS:
    flags.DEFINE_integer('n_burn_in', 1000, 'Number of burn in steps')

if 'max_vmap_batch_size' not in FLAGS:
    flags.DEFINE_integer('max_vmap_batch_size', 0, 'Maximum batch size for vmap. 0 means no batching.')

if 'checkpoint_local_energy' not in FLAGS:
    flags.DEFINE_bool('checkpoint_local_energy', False, 'Whether to checkpoint local energy calculation.')

if 'jastrow_type' not in FLAGS:
    flags.DEFINE_string('jastrow_type', 'bh', 'Jastrow type: bh or simple_ee')

if 'checkpoint_path' not in FLAGS:
    flags.DEFINE_string('checkpoint_path', None, 'Path to save/load checkpoints.')

def main(argv):
  del argv

  logging.info("Starting Be atom Variance Optimization (Reference Det) test...")

  # 1. System Definition
  import pyscf
  
  logging.info(f"Initializing molecule with atoms='{FLAGS.atoms}', basis='{FLAGS.basis}'")
  
  pyscf_mol = pyscf.gto.M(
      atom=FLAGS.atoms,
      basis=FLAGS.basis,
      unit='A',
      charge=0,
      spin=0, 
      verbose=0
  )
  pyscf_mol.build()

  # Derive FermiNet system atoms from PySCF molecule
  atoms = []
  for i in range(pyscf_mol.natm):
      symbol = pyscf_mol.atom_symbol(i)
      coords = pyscf_mol.atom_coord(i)
      atoms.append(system.Atom(symbol, coords))
  
  electrons = pyscf_mol.nelec
  
  logging.info(f"Atoms: {atoms}")
  logging.info(f"Electrons: {electrons}")

  # 2. Hartree-Fock
  logging.info("Solving Hartree-Fock...")

  hf_solution = get_hf_det(
      molecule=atoms,
      nspins=electrons,
      basis=FLAGS.basis,
      restricted=True
  )
  logging.info("Hartree-Fock solved.")

  # 3. Jastrows
  # Boys-Handy
  bh_init, bh_apply = make_bh_jastrow(pyscf_mol)
  
  # Nuclear Cusp
  ncusp_init, ncusp_apply = make_ncusp_jastrow(pyscf_mol)

  simple_ee_init, simple_ee_apply_orig = make_simple_ee_jastrow()
  
  def simple_ee_apply(r_ee, params, nspins, **kwargs):
      return simple_ee_apply_orig(r_ee, params, nspins)
  
  # 4. Ansatz
  # Full Slater-Jastrow Ansatz    # Create the network (ansatz)
  if FLAGS.jastrow_type == 'bh':
      jastrow_apply_fn = bh_apply
      logging.info("Using Boys-Handy Jastrow")
  elif FLAGS.jastrow_type == 'simple_ee':
      jastrow_apply_fn = simple_ee_apply
      logging.info("Using Simple EE Jastrow")
  else:
      raise ValueError(f"Unknown jastrow_type: {FLAGS.jastrow_type}")

  log_psi = make_slater_jastrow(
      hf_solution, 
      jastrow_apply=jastrow_apply_fn,
      ncusp_apply=ncusp_apply
  )
  # Wrapper for signed log_psi (needed for local energy)
  def log_psi_signed(params, pos, spins, atom_coords, charges):
      return log_psi(params, pos, spins, atom_coords, charges)

  # Wrapper for abs log_psi (needed for KFAC/Loss interface)
  def log_psi_abs(params, pos, spins, atom_coords, charges):
      return log_psi(params, pos, spins, atom_coords, charges)[1]

  # Reference Determinant Ansatz (for MCMC sampling)
  def log_det_abs(params, pos, spins, atom_coords, charges):
        # Determine electrons tuple
        if hasattr(hf_solution, 'nelectrons') and hf_solution.nelectrons is not None:
             elec = hf_solution.nelectrons
        elif hasattr(hf_solution, '_mol'):
             elec = hf_solution._mol.nelec
        else:
             elec = electrons

        if pos.ndim > 1:
            pos = pos.reshape(-1)
        
        sign, log_det = hf_solution.eval_slater(pos, elec)
        return log_det

  # Vectorize
  batch_log_psi = jax.vmap(log_psi, in_axes=(None, 0, 0, 0, 0))
  batch_log_psi_signed = jax.vmap(log_psi_signed, in_axes=(None, 0, 0, 0, 0))
  batch_log_psi_abs = jax.vmap(log_psi_abs, in_axes=(None, 0, 0, 0, 0))
  
  # Vectorize Reference Det
  batch_log_det_abs = jax.vmap(log_det_abs, in_axes=(None, 0, 0, 0, 0))

  # 5. Initialization
  key = jax.random.PRNGKey(42)
  key, subkey = jax.random.split(key)
  
  params = {}
  if FLAGS.jastrow_type == 'bh':
      params['jastrow'] = bh_init()
  elif FLAGS.jastrow_type == 'simple_ee':
      params['jastrow'] = simple_ee_init()
  params['ncusp'] = ncusp_init()

  # Load from checkpoint if available
  if FLAGS.checkpoint_path:
      checkpointer = Checkpoint(FLAGS.checkpoint_path)
      loaded_params = checkpointer.load()
      if loaded_params:
          logging.info(f"Loaded parameters from checkpoint {FLAGS.checkpoint_path}")
          # Depending on how params are saved (wrapped or not), unwrap if needed.
          # Our Checkpoint utility saves 'items' which is just params in our case.
          # But if we saved as {'params': params}, we'd need to extract.
          # For now, assuming direct save.
          params = loaded_params
      else:
          logging.info(f"No checkpoint found at {FLAGS.checkpoint_path}, using initial params.")

  
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
  
  # Use Reference Determinant for MCMC
  mcmc_step = mcmc.make_mcmc_step(
      batch_network=batch_log_det_abs, # Sample from Slater determinant
      batch_per_device=batch_size,
      steps=10,
      atoms=atom_pos
  )
  mcmc_step = jax.jit(mcmc_step)
  
  mcmc_width = jnp.asarray(0.1)
  adapt_frequency = 10
  pmoves = np.zeros(adapt_frequency)
  current_pmove = 0.5

  logging.info("Burning in MCMC (Reference Det)...")
  for i in range(FLAGS.n_burn_in):
      if i % 100 == 0:
          logging.info(f"Burn-in step {i}/{FLAGS.n_burn_in}... pmove={current_pmove:.2f}, width={mcmc_width:.4f}")
      key, subkey = jax.random.split(key)
      data, pmove = mcmc_step(params, data, subkey, width=mcmc_width)
      
      pmoves[i % adapt_frequency] = pmove
      # Update MCMC width
      if i > 0 and i % adapt_frequency == 0:
          mcmc_width, pmoves = mcmc.update_mcmc_width(
              i, mcmc_width, adapt_frequency, current_pmove, pmoves
          )
      current_pmove = pmove
  logging.info(f"Burn-in complete. Final pmove={current_pmove:.2f}, width={mcmc_width:.4f}")

  # 7. Optimization
  optimizer_type = FLAGS.optimizer
  learning_rate = FLAGS.learning_rate
  
  local_energy_fn = hamiltonian.local_energy(
      f=log_psi_signed, # Evaluate energy of Full Ansatz
      charges=atom_charges,
      nspins=electrons,
      use_scan=False
  )
  
  # Use Variance Loss
  evaluate_loss = make_variance_loss(
      network=log_psi_abs, # Full ansatz for KFAC registration (if used)
      local_energy=local_energy_fn,
      clip_local_energy=0,
      clip_from_median=True,
      center_at_clipped_energy=True,
      max_vmap_batch_size=FLAGS.max_vmap_batch_size,
      checkpoint_local_energy=FLAGS.checkpoint_local_energy,
  )

  from pytc.autodiff.vmc.optimization import optimize_ref_var

  # Create a simple wrapper for ansatz to satisfy optimization interface
  class AnsatzWrapper:
      def __init__(self, ncusp_apply):
          self.ncusp_apply = ncusp_apply
          # Add other attributes if needed by optimize_ref_var defaults, 
          # but we are passing explicit params/data/mcmc_step so they shouldn't be touched.
          self.dets = [None] # Dummy
          self.jastrow = None # Dummy

  ansatz_wrapper = AnsatzWrapper(ncusp_apply)

  # Run optimization using the refactored function
  results = optimize_ref_var(
      ansatz=ansatz_wrapper,
      params=params,
      data=data,
      mcmc_step=mcmc_step,
      loss_fn=evaluate_loss,
      optimizer_type=optimizer_type,
      learning_rate=learning_rate,
      n_opt_steps=FLAGS.iterations,
      checkpoint_path=FLAGS.checkpoint_path,
      mcmc_width=mcmc_width, # Pass current width
      adapt_frequency=FLAGS.n_opt, # Pass adapt frequency
      burn_in_steps=FLAGS.n_burn_in
  )

if __name__ == '__main__':
  app.run(main)
