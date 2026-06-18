"""End-to-end VMC optimization for N2 with the ccECP pseudopotential.

Motivation: provide a small-molecule VMC datapoint that we can later use
to anchor xTC+ECP validation against Simula, Alavi et al. arXiv:2412.05885v1
(Dec 2024).  Simula et al. report xTC-CCSD(T) atomization energies for
N2/CO/H2O/F2/etc. with eCEPP/ccECP at AVTZ/AVQZ and demonstrate chemical
accuracy via the PP-2 (second-order BCH) commutator treatment.

Their paper reports *atomization* energies and *deviations* from CBS
estimates, not absolute total energies — so the direct comparison is HF
total energy + recovered correlation, not the full xTC-CCSD(T) number.

System: N2 (¹Σ_g⁺) at experimental bond length r = 2.074 Bohr
(≈ 1.09768 Å), closed-shell singlet, RHF.

Basis: ccECP / cc-pVTZ.  ccECP large-core: [He] core removed, so each N
contributes 5 valence electrons; 10 electrons total.  L_max_NL = 1 (s+p
projectors).

Trial wavefunction: JSD = single Slater × (NuclearCusp + BoysHandy).
NuclearCusp is gated off at both N centers by the ECP-aware mask.

Usage:
    python -m pytc.examples.n2_ccecp_vmc_optimization
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from pyscf import gto, scf

from pytc.ansatz import SlaterDet, SlaterJastrow
from pytc.jastrow import BoysHandy, CompositeJastrow, NuclearCusp
from pytc.vmc import optimize, sample


# Experimental bond length r_NN = 1.09768 Å = 2.074 Bohr.
BOND_LENGTH_BOHR = 2.074


def build_n2(basis: str = "ccecp-cc-pvtz"):
    mol = gto.M(
        atom=f"N 0 0 0; N 0 0 {BOND_LENGTH_BOHR}",
        basis=basis,
        ecp="ccecp",
        spin=0,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def main(
    basis: str = "ccecp-cc-pvtz",
    n_walkers: int = 256,
    n_opt_steps: int = 80,
    n_mcmc_per_opt: int = 20,
    burn_in_steps: int = 1000,
    n_sample_steps: int = 4000,
    thinning: int = 10,
    step_size: float = 0.4,
    learning_rate: float = 0.01,
    seed: int = 9091,
):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    jax.config.update("jax_enable_x64", True)

    print("=" * 70)
    print(f"N2 (¹Σ_g⁺) / ccECP / {basis} — VMC end-to-end optimization")
    print("=" * 70)
    mol, mf = build_n2(basis)
    hf_energy = float(mf.e_tot)
    print(f"HF/RHF/ccECP/{basis} = {hf_energy:.6f} Ha")
    print(f"r_NN = {BOND_LENGTH_BOHR} Bohr (experimental)")
    print(f"n_alpha = {mol.nelec[0]}, n_beta = {mol.nelec[1]}, "
          f"n_AO = {mol.nao}")
    print()

    det = SlaterDet.create(mol, mf.mo_coeff)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    bh = BoysHandy.create(mol, terms_per_nucleus=None, name="bh")
    jastrow = CompositeJastrow.create([ncusp, bh])
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    params = [jastrow.init_params(), jnp.ones(1)]

    n_ae = int((~ncusp.has_ecp_per_atom).sum())
    n_ecp = int(ncusp.has_ecp_per_atom.sum())
    print(
        f"Jastrow: NuclearCusp (AE atoms: {n_ae}, ECP atoms gated off: "
        f"{n_ecp}) + BoysHandy ({bh.get_param_count()} params)"
    )
    print()

    key = random.PRNGKey(seed)
    print(
        f"Optimization: {n_opt_steps} opt × {n_mcmc_per_opt} mcmc, "
        f"{n_walkers} walkers, step={step_size}, lr={learning_rate}, adam"
    )
    t0 = time.time()
    opt_results = optimize(
        ansatz, params=params,
        n_walkers=n_walkers, n_steps=n_mcmc_per_opt,
        step_size=step_size, burn_in_steps=burn_in_steps,
        n_opt_steps=n_opt_steps, optimizer_type="adam",
        learning_rate=learning_rate, adaptive_step_size=True,
        key=key,
    )
    print(f"Optimization done in {time.time() - t0:.1f}s")

    energies_opt = np.asarray(opt_results["energies"])
    last_k = max(20, n_opt_steps // 10)
    print(
        f"Energy trace: first 5 = {energies_opt[:5]}, "
        f"last {last_k} mean = {energies_opt[-last_k:].mean():.6f}, "
        f"std = {energies_opt[-last_k:].std():.6f}"
    )
    print()

    opt_params = opt_results["params"][-1]
    print(
        f"Final sampling: {n_walkers} walkers × {n_sample_steps} steps "
        f"(record every {thinning})"
    )
    t0 = time.time()
    samples = sample(
        ansatz, params=opt_params,
        n_walkers=n_walkers, n_steps=n_sample_steps,
        step_size=step_size, thinning=thinning,
        burn_in_steps=burn_in_steps,
        use_importance_sampling=False,
        key=random.PRNGKey(seed + 1),
    )
    print(f"Sampling done in {time.time() - t0:.1f}s")

    energies_2d = np.asarray(samples["energies"]).reshape(-1, n_walkers)
    walker_means = energies_2d.mean(axis=0)
    e_vmc = float(walker_means.mean())
    stderr_bw = float(walker_means.std(ddof=1) / np.sqrt(n_walkers))
    stderr_naive = float(energies_2d.std(ddof=1) / np.sqrt(energies_2d.size))

    print()
    print("=" * 70)
    print(f"N2 / ccECP / {basis} VMC results")
    print("=" * 70)
    print(f"HF                = {hf_energy:.6f} Ha")
    print(f"VMC (pytc, JSD)   = {e_vmc:.6f} ± {stderr_bw:.6f} Ha "
          f"(between-walker, naive {stderr_naive:.6f})")
    print(f"Δ(VMC − HF)       = {e_vmc - hf_energy:+.6f} Ha "
          f"(correlation recovered)")
    print("=" * 70)
    return e_vmc, stderr_bw, hf_energy


if __name__ == "__main__":
    main()
