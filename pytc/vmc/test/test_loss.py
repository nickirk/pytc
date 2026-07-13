"""Test loss functions from loss.py module."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

from pyscf import gto, scf
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import REXP
from pytc.vmc.walker import initialize_walkers
from pytc.vmc.loss import (
    make_energy_loss,
    make_variance_loss
)


def test_energy_loss():
    """Test energy loss function factory."""
    # Create simple H2 molecule
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'sto-3g'
    mol.build()
    
    # HF calculation
    mf = scf.RHF(mol)
    mf.kernel()
    
    # Create ansatz
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = REXP()
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    
    # Initialize parameters
    jastrow_params = jastrow.init_params()
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]
    
    # Initialize walkers
    key = random.PRNGKey(42)
    walkers = initialize_walkers(ansatz, n_walkers=10, key=key)
    
    # Create energy loss
    loss_fn = make_energy_loss(ansatz, optimizer_type="adam")
    
    # Compute loss (aux is an AuxData namedtuple; first two fields are
    # mean energy and energy std)
    loss, aux = loss_fn(params, walkers)
    mean_e, std_e = aux[0], aux[1]
    
    print(f"Energy loss test:")
    print(f"  Loss: {loss:.6f}")
    print(f"  Mean energy: {mean_e:.6f}")
    print(f"  Energy std: {std_e:.6f}")
    print(f"  HF reference: {mf.e_tot:.6f}")
    
    # Test gradient
    grad_fn = jax.grad(lambda p: loss_fn(p, walkers)[0], argnums=0)
    grads = grad_fn(params)
    
    print(f"  Gradient computed successfully!")
    print(f"  Jastrow grad shape: {jax.tree_util.tree_map(lambda x: x.shape, grads[0])}")
    
    assert isinstance(loss, jax.Array), "Loss should be JAX array"
    assert loss.shape == (), "Loss should be scalar"
    print("✓ Energy loss test passed!\n")


def test_variance_loss():
    """Test variance loss function factory."""
    # Create simple H2 molecule
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'sto-3g'
    mol.build()
    
    # HF calculation
    mf = scf.RHF(mol)
    mf.kernel()
    
    # Create ansatz
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = REXP()
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    
    # Initialize parameters
    jastrow_params = jastrow.init_params()
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]
    
    # Initialize walkers
    key = random.PRNGKey(42)
    walkers = initialize_walkers(ansatz, n_walkers=10, key=key)
    
    # Create variance loss
    loss_fn = make_variance_loss(ansatz, optimizer_type="adam")
    
    # Compute loss
    variance, (mean_e, std_e) = loss_fn(params, walkers)
    
    print(f"Variance loss test:")
    print(f"  Variance: {variance:.6f}")
    print(f"  Mean energy: {mean_e:.6f}")
    print(f"  Energy std: {std_e:.6f}")
    
    # Test gradient
    grad_fn = jax.grad(lambda p: loss_fn(p, walkers)[0], argnums=0)
    grads = grad_fn(params)
    
    print(f"  Gradient computed successfully!")
    
    assert isinstance(variance, jax.Array), "Variance should be JAX array"
    assert variance.shape == (), "Variance should be scalar"
    print("✓ Variance loss test passed!\n")


def test_combined_loss():
    """Test combined loss function factory."""
    # Create simple H2 molecule
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'sto-3g'
    mol.build()
    
    # HF calculation
    mf = scf.RHF(mol)
    mf.kernel()
    
    # Create ansatz
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = REXP()
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    
    # Initialize parameters
    jastrow_params = jastrow.init_params()
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]
    
    # Initialize walkers
    key = random.PRNGKey(42)
    walkers = initialize_walkers(ansatz, n_walkers=10, key=key)
    
    # Create combined loss manually in test
    energy_loss_fn = make_energy_loss(ansatz, optimizer_type="adam")
    variance_loss_fn = make_variance_loss(ansatz, optimizer_type="adam", use_custom_jvp=False)
    
    def loss_fn(params, batch_data):
        e_loss, aux = energy_loss_fn(params, batch_data)
        v_loss, _ = variance_loss_fn(params, batch_data)
        return e_loss + 0.1 * v_loss, aux
    
    # Compute loss
    loss, aux = loss_fn(params, walkers)
    mean_e = aux[0] if isinstance(aux, tuple) else aux.mean_energy
    std_e = aux[1] if isinstance(aux, tuple) else aux.energy_std
    
    print(f"Combined loss test:")
    print(f"  Combined loss: {loss:.6f}")
    print(f"  Mean energy: {mean_e:.6f}")
    print(f"  Energy std: {std_e:.6f}")
    
    # Test gradient
    grad_fn = jax.grad(lambda p: loss_fn(p, walkers)[0], argnums=0)
    grads = grad_fn(params)
    
    print(f"  Gradient computed successfully!")
    
    assert isinstance(loss, jax.Array), "Loss should be JAX array"
    assert loss.shape == (), "Loss should be scalar"
    print("✓ Combined loss test passed!\n")


def test_batched_energy_loss():
    """Test batched energy loss function for memory efficiency."""
    # Create simple H2 molecule
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'sto-3g'
    mol.build()
    
    # HF calculation
    mf = scf.RHF(mol)
    mf.kernel()
    
    # Create ansatz
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = REXP()
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    
    # Initialize parameters
    jastrow_params = jastrow.init_params()
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]
    
    # Initialize walkers
    key = random.PRNGKey(42)
    walkers = initialize_walkers(ansatz, n_walkers=50, key=key)
    
    # Create unbatched loss (max_vmap_batch_size=0 means standard vmap)
    loss_fn_unbatched = make_energy_loss(ansatz, optimizer_type="adam", max_vmap_batch_size=0)
    
    # Create batched loss (max_vmap_batch_size=10 means use folx.batched_vmap)
    loss_fn_batched = make_energy_loss(ansatz, optimizer_type="adam", max_vmap_batch_size=10)
    
    # Compute losses
    loss_unbatched, aux_unbatched = loss_fn_unbatched(params, walkers)
    loss_batched, aux_batched = loss_fn_batched(params, walkers)
    
    # Extract mean and std (namedtuples are indexable)
    mean_e_unbatched, std_e_unbatched = aux_unbatched[0], aux_unbatched[1]
    mean_e_batched, std_e_batched = aux_batched[0], aux_batched[1]
    
    print(f"Batched energy loss test (vmap vs batched_vmap):")
    print(f"  Unbatched - Loss: {loss_unbatched:.6f}, Mean E: {mean_e_unbatched:.6f}")
    print(f"  Batched   - Loss: {loss_batched:.6f}, Mean E: {mean_e_batched:.6f}")
    print(f"  Difference: {abs(loss_unbatched - loss_batched):.10f}")
    
    # Test gradients match
    grad_fn_unbatched = jax.grad(lambda p: loss_fn_unbatched(p, walkers)[0], argnums=0)
    grad_fn_batched = jax.grad(lambda p: loss_fn_batched(p, walkers)[0], argnums=0)
    
    grads_unbatched = grad_fn_unbatched(params)
    grads_batched = grad_fn_batched(params)
    
    # Check gradients are close
    def flatten_pytree(tree):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        return jnp.concatenate([jnp.ravel(x) for x in leaves])
    
    grad_diff = jnp.linalg.norm(
        flatten_pytree(grads_unbatched) - flatten_pytree(grads_batched)
    )
    
    print(f"  Gradient difference norm: {grad_diff:.10f}")
    print(f"  Gradients computed successfully!")
    
    # Verify they're approximately equal
    assert jnp.allclose(loss_unbatched, loss_batched, rtol=1e-10), \
        "Batched and unbatched losses should match"
    assert grad_diff < 1e-8, "Batched and unbatched gradients should match"
    
    print("✓ Batched energy loss test passed!\n")


if __name__ == "__main__":
    print("Testing loss functions...\n")
    test_energy_loss()
    test_variance_loss()
    test_combined_loss()
    test_batched_energy_loss()
    print("All loss tests passed! ✓")
