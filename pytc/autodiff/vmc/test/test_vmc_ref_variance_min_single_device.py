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
from ferminet.jastrows import make_simple_ee_jastrow
# import kfac_jax # Not needed for serial

from pytc.autodiff.jastrow.ncusp import make_ncusp_jastrow
from pytc.autodiff.jastrow.bh import make_bh_jastrow
from pytc.autodiff.ansatz.sj import make_slater_jastrow
from pytc.autodiff.ansatz.det import get_hf_det
from pytc.autodiff.vmc.loss import make_variance_loss

# Patch constants for serial execution
constants.pmean = lambda x, axis_name=None: x
constants.pmap = lambda f, axis_name=None, **kwargs: jax.jit(f)

FLAGS = flags.FLAGS

# Define flags if not already defined (to avoid errors if running multiple times)
if 'batch_size' not in FLAGS:
    flags.DEFINE_integer('batch_size', 4096, 'Number of walkers.')
if 'iterations' not in FLAGS:
    flags.DEFINE_integer('iterations', 100, 'Number of optimization steps.')
if 'learning_rate' not in FLAGS:
    flags.DEFINE_float('learning_rate', 1e-1, 'Learning rate.')
if 'n_opt' not in FLAGS:
    flags.DEFINE_integer('n_opt', 20, 'Number of optimization steps per MCMC step.')

def main(argv):
  del argv

  logging.info("Starting Be atom Variance Optimization (Reference Det) test...")

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
  # Boys-Handy
  bh_init, bh_apply = make_bh_jastrow(pytc_mol)
  
  # Nuclear Cusp
  ncusp_init, ncusp_apply = make_ncusp_jastrow(pytc_mol)

  simple_ee_init, simple_ee_apply_orig = make_simple_ee_jastrow()
  
  def simple_ee_apply(r_ee, params, nspins, **kwargs):
      return simple_ee_apply_orig(r_ee, params, nspins)
  
  # 4. Ansatz
  # Full Slater-Jastrow Ansatz (for energy evaluation)
  log_psi = make_slater_jastrow(hf_solution, bh_apply, ncusp_apply=None)
  
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
  
  # Use Reference Determinant for MCMC
  mcmc_step = mcmc.make_mcmc_step(
      batch_network=batch_log_det_abs, # Sample from Slater determinant
      batch_per_device=batch_size,
      steps=10,
      atoms=atom_pos,
  )
  mcmc_step = jax.jit(mcmc_step)
  
  mcmc_width = jnp.asarray(0.1)
  adapt_frequency = 10
  pmoves = np.zeros(adapt_frequency)
  current_pmove = 0.5

  logging.info("Burning in MCMC (Reference Det)...")
  for i in range(500):
      if i % 100 == 0:
          logging.info(f"Burn-in step {i}/500... pmove={current_pmove:.2f}, width={mcmc_width:.4f}")
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
  # optimizer = optax.adam(FLAGS.learning_rate)
  import kfac_jax
  
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
      center_at_clipped_energy=True
  )

  # KFAC Optimizer
  learning_rate = FLAGS.learning_rate
  momentum = 0.9
  damping = 1e-3
  optimizer = optax.adam(learning_rate)
  #optimizer = kfac_jax.Optimizer(
  #    value_and_grad_func=jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True),
  #    l2_reg=0.0,
  #    value_func_has_aux=True,
  #    value_func_has_rng=True,
  #    learning_rate_schedule=lambda t: learning_rate,
  #    momentum_schedule=lambda t: momentum,
  #    damping_schedule=lambda t: damping,
  #    norm_constraint=1e-3,
  #    num_burnin_steps=0,
  #    estimation_mode='fisher_exact', # Use exact Fisher for small systems
  #    multi_device=False,
  #    pmap_axis_name=None # Serial execution
  #)
  
  # KFAC training step
  # Note: KFAC handles JIT internally usually, but here we are in serial mode.
  # We can use ferminet.train.make_kfac_training_step but it assumes pmap.
  # Let's use a custom training step for serial KFAC or adapt make_kfac_training_step.
  # Actually, ferminet.train.make_kfac_training_step uses constants.pmap which we patched to jax.jit.
  # However, it also uses kfac_jax.utils.replicate_all_local_devices which might fail or be redundant.
  # Let's try to use the KFAC optimizer directly in a simple step function.
  
  # shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  # shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(damping))
  
  # We need to patch kfac_jax.utils.replicate_all_local_devices if it's used inside optimizer.step
  # But optimizer.step is a method.
  # Let's define a simple step function for serial KFAC.
  
  # Separate MCMC and Optimization steps
  
  # Optimization step
  def opt_step(data, params, state, key):
      loss_key = key
      # KFAC step expects batch to be (data,) or similar depending on loss signature
      # Our loss signature is (params, key, data) -> (loss, aux)
      # KFAC calls value_and_grad_func(params, rng, batch)
      # So batch should be data.
      
      new_params, new_state, stats = optimizer.step(
          params=params,
          state=state,
          rng=loss_key,
          batch=data
      )
      return new_params, new_state, stats['loss'], stats['aux']

  # opt_step = jax.jit(opt_step) # KFAC handles JIT internally

  # Initialize KFAC state
  # KFAC init expects (params, rng, batch)
  
  opt_state = optimizer.init(params)
  #opt_state = optimizer.init(params, subkey, data)

  # opt_update_step = train.make_opt_update_step(evaluate_loss, optimizer)
  # train_step = train.make_training_step(mcmc_step, opt_update_step)
  # train_step = jax.jit(train_step)
  # opt_state = optimizer.init(params)
  
  # mcmc_width = jnp.asarray(0.1)
  # adapt_frequency = 10
  # pmoves = np.zeros(adapt_frequency)
  
  logging.info("Starting optimization...")
  
  for t in range(FLAGS.iterations):
      # MCMC Step
      key, subkey = jax.random.split(key)
      data, pmove = mcmc_step(params, data, subkey, mcmc_width)
      
      # Update MCMC width
      pmoves[t % adapt_frequency] = pmove
      if t > 0 and t % adapt_frequency == 0:
          mcmc_width, pmoves = mcmc.update_mcmc_width(
              t, mcmc_width, adapt_frequency, current_pmove, pmoves
          )
      current_pmove = pmove

      # Optimization Steps
      for _ in range(FLAGS.n_opt):
          key, subkey = jax.random.split(key)
          params, opt_state, loss_val, aux_data = opt_step(data, params, opt_state, subkey)
      
      if t % 1 == 0:
          logging.info(f"Step {t}: Variance = {loss_val:.6f}, Energy = {aux_data.energy:.6f}, pmove = {pmove:.2f}")

if __name__ == '__main__':
  app.run(main)
