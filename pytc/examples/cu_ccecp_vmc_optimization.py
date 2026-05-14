"""End-to-end VMC optimization of the Cu atom (²S) with ccECP.

This is a 3d transition-metal validation of the ECP-VMC pipeline:

- Open-shell UHF reference (n_alpha = 10, n_beta = 9; ground-state ²S
  configuration 3s² 3p⁶ 3d¹⁰ 4s¹).
- ccECP large-core: [Ne] removed, 19 valence electrons. ``Z_eff = 19``.
- ccECP for 3d atoms uses L_max_NL = 1 (s and p projectors); the d valence
  shell sees only V_loc.  So this test exercises:
    * the UHF mo_coeff handling (mf.mo_coeff is a 3-D ndarray for UHF),
    * deeper V_loc + larger Z_eff than first-row,
    * heavier-element locality-approximation regime,
  but NOT the ℓ=2 quadrature path (which would need 4d/5d external ccECP).

Trial wavefunction: JSD = single Slater × (NuclearCusp + BoysHandy).
NuclearCusp is gated off at the Cu nucleus by the ECP-aware mask added
earlier on this branch.

Measured result on this branch (256 walkers × 150 opt × 20 mcmc + 4000 sample
steps / thinning 10, adam @ lr=0.01, step_size=0.3 Bohr):

    Basis                n_AO   max_l   HF              VMC                Δ(VMC-HF)
    ccecp-cc-pvdz         38      3    -195.332178    -196.105472(22336)   -773 mHa
    ccecp-cc-pvtz         63      4    -195.337588    -196.101826(22168)   -764 mHa

VTZ doesn't improve on VDZ for VMC even though HF drops by 5 mHa: the
BoysHandy Jastrow has a fixed number of parameters and cannot exploit
the richer orbital basis at fixed Jastrow flexibility (a standard
Jastrow-VMC plateau effect; closing requires a more flexible Jastrow or
a second-order optimizer like Newton or linear method).

Compared to Annaberdiyev et al., JCTC 16, 1482 (2020) Table 17 (using
cc-pCVQZ — a core-valence basis much richer than ours):

    DMC single-det (HF trial)   -196.3178(3) Ha
    DMC multi-det (sCI trial)   -196.353(3)  Ha
    CIPSI "exact"               -196.4038(10) Ha

We recover ~73% of the published correlation energy at the smaller cc-pVDZ
basis.  The ~213 mHa gap to single-det DMC is dominated by basis-set
incompleteness — PySCF's bundled ccECP library only ships valence-only
sets (cc-pVxZ); the literature uses cc-pCVxZ which is not available in
pyscf without external loading from Basis Set Exchange or
pseudopotentiallibrary.org.

NB: Cu/cc-pVTZ has g shells (max_l=4) which were silently zeroed by the
spherical GTO bug fixed in commit 6b1068c.  This re-run is on top of the
fix; the pre-fix Cu/VTZ run was killed before completing so we have no
buggy-vs-fixed comparison number, but the H2O/V5Z analogue (commit 520b3e4)
showed only a 0.13 mHa shift since occupied MOs have tiny g/h coefficients.

Usage:
    python -m pytc.examples.cu_ccecp_vmc_optimization
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


def build_cu():
    mol = gto.M(
        atom="Cu 0 0 0",
        basis="ccecp-cc-pvdz",
        ecp="ccecp",
        spin=1,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.kernel()
    return mol, mf


def main(
    n_walkers: int = 256,
    n_opt_steps: int = 150,
    n_mcmc_per_opt: int = 20,
    burn_in_steps: int = 1500,
    n_sample_steps: int = 4000,
    thinning: int = 10,
    step_size: float = 0.3,
    learning_rate: float = 0.01,
    seed: int = 5102,
):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    jax.config.update("jax_enable_x64", True)

    print("=" * 70)
    print("Cu atom (²S) / ccECP / cc-pVDZ — VMC end-to-end optimization")
    print("=" * 70)
    mol, mf = build_cu()
    hf_energy = float(mf.e_tot)
    print(f"HF/UHF/ccECP/cc-pVDZ = {hf_energy:.6f} Ha")
    print(f"n_alpha = {mol.nelec[0]}, n_beta = {mol.nelec[1]}, "
          f"Z_eff = {int(mol.atom_charges()[0])}, n_AO = {mol.nao}")
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
        f"Jastrow: NuclearCusp (AE atoms: {n_ae}, ECP atoms gated off: {n_ecp}) "
        f"+ BoysHandy ({bh.get_param_count()} params)"
    )
    print(f"Determinant: {det.unrestricted=}; "
          f"alpha {n_ae} active, beta {n_ae} active")
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
        f"last {last_k} mean = {energies_opt[-last_k:].mean():.6f}"
    )
    print()

    opt_params = opt_results["params"][-1]
    print(f"Final sampling: {n_walkers} walkers × {n_sample_steps} steps "
          f"(record every {thinning})")
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
    print("Cu / ccECP VMC results")
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
