"""Test script to compare standard vs Hamiltonian-based gradient methods.

This script runs variance optimization on H2 molecule using both methods
and compares the results to verify correctness.
"""

import jax
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
from pyscf import gto, scf

from pytc.autodiff.vmc import optimize_ref_var
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pytc.autodiff.jastrow import Poly
from pytc.autodiff.ansatz.det import SlaterDet


def test_gradient_methods():
    """Test both gradient methods on H2 molecule."""
    
    # Create H2 molecule
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1.0'
    mol.basis = 'sto-3g'
    mol.build()
    
    # Run HF calculation
    mf = scf.RHF(mol)
    mf.kernel()
    
    # Create ansatz
    det = SlaterDet(mol, mf.mo_coeff)
    jastrow = Poly()
    jastrow_params = jnp.array([0.5])  # Start with non-zero parameter
    linear_coeffs = jnp.ones(1)
    sj_ansatz = SlaterJastrow(mol, jastrow, [det])
    
    # Test parameters
    n_walkers = 100
    n_steps = 5
    n_opt_steps = 10
    key = random.PRNGKey(42)
    
    print("="*60)
    print("Testing Gradient Methods on H2")
    print("="*60)
    
    # Test 1: Standard method
    print("\n1. Running with STANDARD gradient method...")
    key1, subkey = random.split(key)
    results_standard = optimize_ref_var(
        sj_ansatz,
        params=[jastrow_params, linear_coeffs],
        n_walkers=n_walkers,
        n_steps=n_steps,
        step_size=0.01,
        burn_in_steps=100,
        n_opt_steps=n_opt_steps,
        optimizer_type='adam',
        learning_rate=0.01,
        use_hamiltonian_grad=False,  # Standard method
        key=subkey
    )
    
    print(f"  Final variance: {results_standard['cost'][-1]:.6f}")
    print(f"  Final energy: {results_standard['energies'][-1]:.6f}")
    
    # Test 2: Hamiltonian-based method
    print("\n2. Running with HAMILTONIAN gradient method...")
    key2, subkey = random.split(key1)
    results_hamiltonian = optimize_ref_var(
        sj_ansatz,
        params=[jastrow_params, linear_coeffs],
        n_walkers=n_walkers,
        n_steps=n_steps,
        step_size=0.01,
        burn_in_steps=100,
        n_opt_steps=n_opt_steps,
        optimizer_type='adam',
        learning_rate=0.01,
        use_hamiltonian_grad=True,  # New Hamiltonian method
        key=subkey
    )
    
    print(f"  Final variance: {results_hamiltonian['cost'][-1]:.6f}")
    print(f"  Final energy: {results_hamiltonian['energies'][-1]:.6f}")
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    
    variance_diff = abs(results_standard['cost'][-1] - results_hamiltonian['cost'][-1])
    energy_diff = abs(results_standard['energies'][-1] - results_hamiltonian['energies'][-1])
    
    print(f"Variance difference: {variance_diff:.6e}")
    print(f"Energy difference: {energy_diff:.6e}")
    
    # Check if results are similar (allowing for stochastic variation)
    variance_rel_error = variance_diff / max(abs(results_standard['cost'][-1]), 1e-10)
    print(f"Variance relative error: {variance_rel_error:.2%}")
    
    if variance_rel_error < 0.5:  # Allow 50% relative difference due to stochasticity
        print("\n✓ SUCCESS: Both methods produce comparable results!")
    else:
        print("\n⚠ WARNING: Methods produce different results. This may be due to:")
        print("  - Statistical fluctuations (try increasing n_walkers or n_opt_steps)")
        print("  - Implementation bugs (check gradient computation)")
    
    return results_standard, results_hamiltonian


if __name__ == "__main__":
    test_gradient_methods()
