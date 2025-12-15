import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 support
import jax.numpy as jnp
import jax.tree_util as jtu
import optax  # JAX's optimization library
from functools import partial
from pytc.autodiff import jastrow
from pytc.autodiff import xtc
from pytc.autodiff import tc_helper
from pyscf import gto, scf

import resource

def get_peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024 / 1024 

def optimize_jastrow(xtc_obj, mf, init_params, n_steps=50, optimizer_name='adam', learning_rate=1e-3, opt_file='opt_data.npz'):
    """Optimize Jastrow parameters using advanced optimizers with adaptive learning rate."""
    params = init_params.copy()
    
    # Precompute standard integrals
    h1e_std = jnp.asarray(tc_helper.get_hcore(mf, xtc_obj.mo_coeff))
    eri_std = jnp.asarray(tc_helper.get_eri(mf, xtc_obj.mo_coeff))
    nocc = int(sum(mf.mo_occ == 2))

    # Add learning rate schedule parameters
    current_lr = learning_rate
    lr_decay_factor = 0.5  # How much to reduce learning rate
    lr_min = 1e-6  # Minimum learning rate
    patience = 10  # How many steps to wait before reducing lr

    # Create optimizer with current learning rate
    def create_optimizer(lr):
        if (optimizer_name == 'adam'):
            return optax.adam(lr)
        elif (optimizer_name == 'adamw'):
            return optax.adamw(lr)
        elif (optimizer_name == 'adagrad'):
            return optax.adagrad(lr)
        elif (optimizer_name == 'rmsprop'):
            return optax.rmsprop(lr)
        else:
            return optax.sgd(lr)
    
    optimizer = create_optimizer(current_lr)
    opt_state = optimizer.init(params)
    
    # Track gradient history for adaptive learning rate
    prev_grad_norm = None
    increasing_count = 0
    
    @jax.jit
    def loss_fn(params):
        # Get corrections
        delta_h = xtc_obj.get_1b(params)
        
        # Compute specific 2-body blocks to save memory
        # V_iajb corresponds to (o, v, o, v)
        delta_ovov = xtc_obj.get_2b(params, block_str='ovov')
        
        # For f_ia terms:
        # term1: sum_j (ia|jj) -> (o, v, o, o)
        delta_ovoo = xtc_obj.get_2b(params, block_str='ovoo')
        
        # term2: sum_j (ij|ja) -> (o, o, o, v)
        delta_ooov = xtc_obj.get_2b(params, block_str='ooov')
        
        # Combine with standard integrals (sliced)
        # h1e_std is (N, N), delta_h is (N, N)
        one_body_ia = h1e_std[:nocc, nocc:] + delta_h[:nocc, nocc:]
        
        # V_iajb = eri_ovov + delta_ovov
        V_iajb = eri_std[:nocc, nocc:, :nocc, nocc:] + delta_ovov
        V_iajb_anti = 2*V_iajb - V_iajb.transpose(0,3,2,1)

        # Build Fock matrix elements
        f_ia = one_body_ia
        
        # Coulomb term: 2 * sum_j (ia|jj)
        # eri_ovoo + delta_ovoo
        term_coulomb = eri_std[:nocc, nocc:, :nocc, :nocc] + delta_ovoo
        f_ia = f_ia + 2. * jnp.einsum('iajj->ia', term_coulomb)
        
        # Exchange term: sum_j (ij|ja)
        # eri_ooov + delta_ooov
        term_exchange = eri_std[:nocc, :nocc, :nocc, nocc:] + delta_ooov
        f_ia = f_ia - jnp.einsum('ijja->ia', term_exchange)

        loss = jnp.sum(f_ia*f_ia) + jnp.sum(V_iajb_anti*V_iajb_anti)
        return loss

    steps = []
    losses = []
    grad_norms = []
    params_bag = []
    
    for step in range(n_steps):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        flat_grads, _ = jtu.tree_flatten(grads)
        grad_norm = jnp.linalg.norm(jnp.concatenate([jnp.ravel(g) for g in flat_grads]))
        
        # Check for NaN gradients using tree flattening
        if any(jnp.any(jnp.isnan(g)) for g in flat_grads):
            print(f"Warning: NaN gradients at step {step}")
            break
            
        # Adaptive learning rate logic
        if prev_grad_norm is not None:
            if grad_norm > prev_grad_norm:
                increasing_count += 1
                if increasing_count >= patience and current_lr > lr_min:
                    # Reduce learning rate
                    current_lr = max(current_lr * lr_decay_factor, lr_min)
                    print(f"\nReducing learning rate to {current_lr}")
                    # Reinitialize optimizer with new learning rate
                    optimizer = create_optimizer(current_lr)
                    opt_state = optimizer.init(params)
                    increasing_count = 0
            else:
                increasing_count = 0
        
        prev_grad_norm = grad_norm
        
        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss_val:.6f}, "
                  f"Grad norm: {grad_norm:.6f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Params: {params}, "
                  f"Peak Mem: {get_peak_memory_mb():.2f} MB")
                  
        if grad_norm < 1e-6:
            print(f"Converged at step {step}")
            break
            
        steps.append(step)
        losses.append(loss_val)
        grad_norms.append(grad_norm)
        params_bag.append(params)
        np.savez(opt_file, steps=steps, losses=losses, 
                 grad_norms=grad_norms, params_bag=params_bag)

    return params


def create_test_system(basis):
    """Create a test He atom system with cc-pVDZ basis."""
    mol = gto.M(atom='He 0 0 0', basis=basis, unit='Bohr')
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def do_ccsd(params, basis):
    # Create new system with cc-pVTZ basis
    mol, mf = create_test_system('ccpvtz')
    
    my_jastrow = jastrow.REXP()  # Remove params from constructor
    myxtc = xtc.XTC.from_pyscf(mf, my_jastrow, grid_lvl=2)
    eris = myxtc.make_eris(mf, params)  # Pass mf and params explicitly
    from pyscf.cc import rccsd
    mycc = rccsd.RCCSD(mf)
    mycc.kernel(eris=eris)
    nocc = int(sum(mf.mo_occ == 2))
    e_hf = 2*np.einsum("ii->", eris.fock[:nocc,:nocc])
    e_hf -= 2*np.einsum("iijj->", eris.oooo) - np.einsum("ijji->", eris.oooo)
    print("HF energy:", e_hf)
    print("CCSD correlation energy:", mycc.e_corr)
    print("Total CCSD energy:", e_hf + mycc.e_corr)
    assert np.isclose(e_hf, -2.9022851504761435, atol=1e-7)
    assert np.isclose(mycc.e_corr, -0.0013057018938958786, atol=1e-7)
    assert np.isclose(e_hf + mycc.e_corr, -2.9035908523700393, atol=1e-7)

    # --- NumPy Verification ---
    print("\n--- NumPy Verification ---")
    from pytc.xtc import XTC as XTC_np
    from pytc.jastrow.rexp import REXP as REXP_np
    
    # Initialize NumPy REXP with optimized parameters
    # Note: REXP_np takes params in __init__
    rexp_np = REXP_np(params=params['alpha'])
    
    # Initialize NumPy XTC
    xtc_np = XTC_np(mf, rexp_np, grid_lvl=2)
    
    # Make ERIs using NumPy implementation
    print("Calculating ERIs using NumPy XTC...")
    eris_np = xtc_np.make_eris()
    
    # Run CCSD with NumPy ERIs
    mycc_np = rccsd.RCCSD(mf)
    mycc_np.kernel(eris=eris_np)
    
    e_hf_np = 2*np.einsum("ii->", eris_np.fock[:nocc,:nocc])
    e_hf_np -= 2*np.einsum("iijj->", eris_np.oooo) - np.einsum("ijji->", eris_np.oooo)
    
    print("NumPy HF energy:", e_hf_np)
    print("NumPy CCSD correlation energy:", mycc_np.e_corr)
    print("NumPy Total CCSD energy:", e_hf_np + mycc_np.e_corr)
    
    # Compare JAX and NumPy results
    print("\n--- Comparison ---")
    print(f"HF Energy Diff: {abs(e_hf - e_hf_np):.2e}")
    print(f"Corr Energy Diff: {abs(mycc.e_corr - mycc_np.e_corr):.2e}")
    
    assert np.isclose(e_hf, e_hf_np, atol=1e-7)
    assert np.isclose(mycc.e_corr, mycc_np.e_corr, atol=1e-7)
    print("Verification Passed!")

def main():
    """Example usage with He atom."""
    # Create test system
    mol, mf = create_test_system('ccpvdz')

    
    my_jastrow = jastrow.REXP()  # Remove params from constructor
    init_params = my_jastrow.init_params()  # Initialize parameters
    
    # Run optimization with smaller learning rate
    myxtc = xtc.XTC.from_pyscf(mf, my_jastrow, grid_lvl=2)
    
    # Try different optimizers
    optimizers_to_try = {
        'rmsprop': 1e-2
    }
    
    for opt_name, lr in optimizers_to_try.items():
        print(f"\nTrying {opt_name} optimizer...")
        optimized_params = optimize_jastrow(myxtc, mf, init_params,
                                          optimizer_name=opt_name,
                                          learning_rate=lr,
                                          n_steps=20)
        print(f"{opt_name} optimized parameters:", optimized_params)
    
    assert np.isclose(optimized_params['alpha'][0], 0.37550687, atol=1e-5)
    
    do_ccsd(optimized_params, 'ccpvtz')



if __name__ == "__main__":
    main()