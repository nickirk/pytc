import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 support
import jax.numpy as jnp
import optax  # JAX's optimization library
from functools import partial
from pytc.autodiff import jastrow
from pytc.autodiff import xtc
from pyscf import gto, scf

def optimize_jastrow(xtc, init_params, n_steps=50, optimizer_name='adam', learning_rate=1e-3):
    """Optimize Jastrow parameters using advanced optimizers.
    
    Args:
        xtc: XTC instance
        init_params: Initial parameters
        n_steps: Number of optimization steps
        optimizer_name: One of ['adam', 'adamw', 'adagrad', 'rmsprop', 'sgd']
        learning_rate: Learning rate for optimizer
    """
    # Convert initial parameters to float64
    params = jnp.asarray(init_params, dtype=jnp.float64)
    
    # Select optimizer
    if optimizer_name == 'adam':
        optimizer = optax.adam(learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = optax.adamw(learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = optax.adagrad(learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optax.rmsprop(learning_rate)
    else:
        optimizer = optax.sgd(learning_rate)
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    
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

    # Optimization loop with advanced optimizer
    for step in range(n_steps):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        
        # Check for NaN gradients
        if jnp.any(jnp.isnan(grads)):
            print(f"Warning: NaN gradients at step {step}")
            break
        
        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss_val:.6f}, "
                  f"Grad norm: {jnp.linalg.norm(grads):.6f}, "
                  f"Params: {params}")

    return params


def create_test_system():
    """Create a test Be atom system with cc-pVDZ basis."""
    mol = gto.M(atom='Be 0 0 0', basis='sto6g', unit='Bohr')
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf

def main():
    """Example usage with Be atom."""
    # Create test system
    mol, mf = create_test_system()
    
    # Initialize Jastrow with smaller parameters
    init_params = jnp.array([0.5], dtype=jnp.float64)  # Start with smaller initial value
    my_jastrow = jastrow.SimpleJastrow(init_params)
    
    # Run optimization with smaller learning rate
    myxtc = xtc.XTC(mf, my_jastrow, grid_lvl=1)
    
    # Try different optimizers
    optimizers_to_try = {
        'rmsprop': 1e-2
    }
    
    for opt_name, lr in optimizers_to_try.items():
        print(f"\nTrying {opt_name} optimizer...")
        optimized_params = optimize_jastrow(myxtc, init_params,
                                          optimizer_name=opt_name,
                                          learning_rate=lr,
                                          n_steps=100)
        print(f"{opt_name} optimized parameters:", optimized_params)

if __name__ == "__main__":
    main()