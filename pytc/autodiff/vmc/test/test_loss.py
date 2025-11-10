"""Test loss functions from loss.py module."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

from pyscf import gto, scf
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pytc.autodiff.ansatz.det import SlaterDet
from pytc.autodiff.jastrow import REXP
from pytc.autodiff.vmc.walker import initialize_walkers
from pytc.autodiff.vmc.loss import (
    make_energy_loss,
    make_variance_loss,
    make_combined_loss
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
    det = SlaterDet(mol, mf.mo_coeff)
    jastrow = REXP()
    ansatz = SlaterJastrow([det], jastrow)
    
    # Initialize parameters
    jastrow_params = jastrow.init_params()
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]
    
    # Initialize walkers
    key = random.PRNGKey(42)
    walkers = initialize_walkers(ansatz, n_walkers=10, key=key)
    
    # Create energy loss
    loss_fn = make_energy_loss(ansatz, optimizer_type="adam")
    
    # Compute loss
    loss, (mean_e, std_e) = loss_fn(params, walkers)
    
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
    det = SlaterDet(mol, mf.mo_coeff)
    jastrow = REXP()
    ansatz = SlaterJastrow([det], jastrow)
    
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
    det = SlaterDet(mol, mf.mo_coeff)
    jastrow = REXP()
    ansatz = SlaterJastrow([det], jastrow)
    
    # Initialize parameters
    jastrow_params = jastrow.init_params()
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]
    
    # Initialize walkers
    key = random.PRNGKey(42)
    walkers = initialize_walkers(ansatz, n_walkers=10, key=key)
    
    # Create combined loss
    loss_fn = make_combined_loss(
        ansatz,
        optimizer_type="adam",
        energy_weight=1.0,
        variance_weight=0.1
    )
    
    # Compute loss
    loss, (mean_e, std_e) = loss_fn(params, walkers)
    
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


if __name__ == "__main__":
    print("Testing loss functions...\n")
    test_energy_loss()
    test_variance_loss()
    test_combined_loss()
    print("All loss tests passed! ✓")
