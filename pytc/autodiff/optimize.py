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
    patience = 10  # How many steps to wait before reducing lr
    
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
        V_iajb = two_body[:nocc,nocc:,:nocc,nocc:]
        V_iajb_anti = 2*V_iajb - V_iajb.transpose(0,3,2,1)
        #V_abij = two_body[nocc:,nocc:,:nocc,:nocc]
        #V_abij_anti = 2*V_abij - V_abij.transpose(0,1,3,2)

        # Build Fock matrix elements
        f_ia = one_body[:nocc,nocc:]
        f_ia = f_ia + 2.*jnp.einsum('iajj->ia', two_body[:nocc,nocc:,:nocc,:nocc])
        f_ia = f_ia - jnp.einsum('ijja->ia', two_body[:nocc,:nocc,:nocc,nocc:])

        #f_ai = one_body[nocc:,:nocc]
        #f_ai = f_ai + 2.*jnp.einsum('aijj->ai', two_body[nocc:,:nocc,:nocc,:nocc])
        #f_ai = f_ai - jnp.einsum('jiaj->ai', two_body[:nocc,:nocc,nocc:,:nocc])

        loss = jnp.sum(f_ia*f_ia) + jnp.sum(V_iajb_anti*V_iajb_anti)
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


def create_test_system(basis):
    """Create a test Be atom system with cc-pVDZ basis."""
    mol = gto.M(atom='He 0 0 0', basis=basis, unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

class REXP(jastrow.Jastrow):
    def __init__(self, params, epsilon=1e-12):
        super().__init__(params)
        self.epsilon = epsilon
        
    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
    def _compute(self, r1, r2, params):
        r12 = r1-r2
        r12_norm = self._safe_norm(r12)
        return 0.5*jnp.exp(-params[0] * r12_norm) * r12_norm

    def __call__(self, r1, r2):
        return super().__call__(r1, r2)


def main():
    """Example usage with Be atom."""
    # Create test system
    mol, mf = create_test_system('ccpvtz')
    
    init_params = jnp.array([0.39839], dtype=jnp.float64)
    my_jastrow = REXP(init_params)
    
    # Run optimization with smaller learning rate
    myxtc = xtc.XTC(mf, my_jastrow, grid_lvl=2)
    
    # Try different optimizers
    optimizers_to_try = {
        'rmsprop': 5e-2
    }
    
    for opt_name, lr in optimizers_to_try.items():
        print(f"\nTrying {opt_name} optimizer...")
        optimized_params = optimize_jastrow(myxtc, init_params,
                                          optimizer_name=opt_name,
                                          learning_rate=lr,
                                          n_steps=0)
        print(f"{opt_name} optimized parameters:", optimized_params)

    #mol, mf = create_test_system('ccpvtz') 
    #my_jastrow = REXP(optimized_params)
    #myxtc = xtc.XTC(mf, my_jastrow, grid_lvl=2)

    #eris = myxtc.make_eris()
    #from pyscf.cc import rccsd
    #mycc = rccsd.RCCSD(mf)
    #mycc.kernel(eris=eris)
    #nocc = int(sum(mf.mo_occ == 2))
    #e_hf = 2*np.einsum("ii->", eris.fock[:nocc,:nocc])
    #e_hf -= 2*np.einsum("iijj->", eris.oooo) - np.einsum("ijji->", eris.oooo)
    #print("HF energy:", e_hf)
    #print("CCSD correlation energy:", mycc.e_corr)
    #print("Total CCSD energy:", e_hf + mycc.e_corr)

if __name__ == "__main__":
    main()