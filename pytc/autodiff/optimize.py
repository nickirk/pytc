import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 support
import jax.numpy as jnp
import optax  # JAX's optimization library
from functools import partial
from pytc.autodiff import jastrow
from pytc.autodiff import xtc
from pyscf import gto, scf

def optimize_jastrow(xtc, init_params, n_steps=50, optimizer_name='adam', learning_rate=1e-3):
    """Optimize Jastrow parameters using advanced optimizers with adaptive learning rate."""
    params = jnp.asarray(init_params, dtype=jnp.float64)
    
    # Add learning rate schedule parameters
    current_lr = learning_rate
    lr_decay_factor = 0.5  # How much to reduce learning rate
    lr_min = 1e-6  # Minimum learning rate
    patience = 3  # How many steps to wait before reducing lr
    
    # Create optimizer with current learning rate
    def create_optimizer(lr):
        if optimizer_name == 'adam':
            return optax.adam(lr)
        elif optimizer_name == 'adamw':
            return optax.adamw(lr)
        elif optimizer_name == 'adagrad':
            return optax.adagrad(lr)
        elif optimizer_name == 'rmsprop':
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
        # Update the Jastrow parameters in a way that maintains JAX gradients
        xtc.update_jastrow_params(params)  # We need to add this method to XTC class
        one_body = xtc.get_1b()
        two_body = xtc.get_2b()

        # Get number of occupied and virtual orbitals
        nocc = int(sum(xtc.mf.mo_occ == 2))  # Number of occupied orbitals
        nvir = len(xtc.mf.mo_occ) - nocc     # Number of virtual orbitals

        # Slice the tensors for occupied and virtual spaces
        V_ijab = two_body[:nocc,:nocc,nocc:,nocc:]
        V_ijab_anti = 2*V_ijab - V_ijab.transpose(0,1,3,2)
        V_abij = two_body[nocc:,nocc:,:nocc,:nocc]
        V_abij_anti = 2*V_abij - V_abij.transpose(0,1,3,2)

        # Build Fock matrix elements
        f_ia = one_body[:nocc,nocc:]
        f_ia = f_ia + 2.*jnp.einsum('iajj->ia', two_body[:nocc,nocc:,:nocc,:nocc])
        f_ia = f_ia - jnp.einsum('ijja->ia', two_body[:nocc,:nocc,:nocc,nocc:])

        f_ai = one_body[nocc:,:nocc]
        f_ai = f_ai + 2.*jnp.einsum('aijj->ai', two_body[nocc:,:nocc,:nocc,:nocc])
        f_ai = f_ai - jnp.einsum('jiaj->ai', two_body[:nocc,:nocc,nocc:,:nocc])

        loss = jnp.asarray(jnp.einsum('ia,ai->', f_ia, f_ai), dtype=jnp.float64)
        loss = loss + jnp.asarray(jnp.einsum('ijab,abij->', V_ijab_anti, V_abij_anti), dtype=jnp.float64)
        
        return loss

    steps = []
    losses = []
    grad_norms = []
    params_bag = []
    
    for step in range(n_steps):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = jnp.linalg.norm(grads)
        
        # Check for NaN gradients
        if jnp.any(jnp.isnan(grads)):
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
                  f"Params: {params}")
                  
        if grad_norm < 1e-6:
            print(f"Converged at step {step}")
            break
            
        steps.append(step)
        losses.append(loss_val)
        grad_norms.append(grad_norm)
        params_bag.append(params)
        np.savez('Mg_opt_data2.npz', steps=steps, losses=losses, 
                 grad_norms=grad_norms, params_bag=params_bag)

    return params


def create_test_system():
    """Create a test Be atom system with cc-pVDZ basis."""
    mol = gto.M(atom='Mg 0 0 0', basis='ccpvdz', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

def main():
    """Example usage with Be atom."""
    # Create test system
    mol, mf = create_test_system()
    
    # Initialize Jastrow with smaller parameters
    init_params = jnp.array([0.5, 0.1, -0.1, 0.2], dtype=jnp.float64)
    init_params = jnp.array([1.30442946,  0.55080467, -0.20307955, -0.22240826])
    #init_params = jnp.array([ 3.30571992,  0.52066121,  0.3091659,   1.56504699, -0.17847568, -0.0455195 ], dtype=jnp.float64)
    #init_params = jnp.array([3.30992979,  0.57100468,  0.31507011,  1.53111045, -0.22547859, -0.0537576], dtype=jnp.float64)
    my_jastrow = jastrow.SimpleJastrow(init_params)
    
    # Run optimization with smaller learning rate
    myxtc = xtc.XTC(mf, my_jastrow, grid_lvl=1)
    
    # Try different optimizers
    optimizers_to_try = {
        'rmsprop': 5e-2
    }
    
    for opt_name, lr in optimizers_to_try.items():
        print(f"\nTrying {opt_name} optimizer...")
        optimized_params = optimize_jastrow(myxtc, init_params,
                                          optimizer_name=opt_name,
                                          learning_rate=lr,
                                          n_steps=500)
        print(f"{opt_name} optimized parameters:", optimized_params)

if __name__ == "__main__":
    main()