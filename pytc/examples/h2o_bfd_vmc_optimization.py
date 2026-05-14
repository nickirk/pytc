"""End-to-end VMC optimization for H2O with the BFD pseudopotential.

Reference value (literature):
    Zen, Sorella, Gillan, Michaelides, Alfè, J. Chem. Theory Comput.
    9, 4332 (2013) — Table 1 "JSD/uncontracted":
        VMC = -17.24820(5) Ha
    Trial wavefunction: single Slater × Jastrow with 1-body (e-I),
    2-body (e-e), and inhomogeneous 2-body (e-e-I, i.e. 3-body) terms.
    Geometry: experimental, r_OH = 0.95721 Å, ∠HOH = 104.522°.
    Pseudopotential: BFD on O (2-electron core), H all-electron.

This script reproduces that setup as closely as the pytc primitives allow:

  - PySCF basis bfd-vdz / bfd-vtz with ecp='bfd' on O.
  - SlaterDet built from RHF orbitals.
  - BoysHandy Jastrow with the default term list, which contains the 1-body
    e-n, 2-body e-e (cusp + polynomial), and 3-body e-e-n components in a
    single unified Padé-times-polynomial expansion — equivalent to Zen
    et al.'s "JSD" form.
  - Adam optimizer, energy minimization (`optimize`).
  - Final sampling at the optimized parameters reports <E_L> with
    between-walker stderr.

Caveats vs the Zen et al. number:

  - We do NOT add the NuclearCusp Jastrow.  The default NuclearCusp uses
    mol.atom_charges() which under BFD returns Z_eff for O (Coulomb-free
    centre) and would impose a spurious cusp at the ECP nucleus.  At the
    H atoms (all-electron) we DO lose the e-n cusp, costing a few mHa.
    An ECP-aware NuclearCusp is a follow-up — see design_ecp_vmc.md §6.
  - Basis-set incompleteness: Zen et al. used an "uncontracted" basis
    closer to the BFD-V5Z limit.  At BFD-VDZ we are well below their
    correlation budget; even a perfect JSD wf cannot reach -17.248 Ha
    with this basis.  BFD-VTZ closes most of the gap; BFD-VQZ closes more.

Usage:
    python -m pytc.examples.h2o_bfd_vmc_optimization
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
from pytc.jastrow import BoysHandy
from pytc.vmc import sample, optimize


# ----- Reference value ------------------------------------------------------
REFERENCE_E_VMC = -17.24820       # Ha, Zen et al. 2013, JSD/uncontracted
REFERENCE_E_VMC_ERR = 5.0e-5      # Ha, their 1-σ statistical uncertainty
REFERENCE_E_EXACT_BFD = -17.276   # approximate BFD-basis-limit total energy


# ----- Geometry: experimental, r_OH = 0.95721 Å, ∠HOH = 104.522° -----------
def water_geometry_bohr():
    r_OH_A = 0.95721
    bohr_per_angstrom = 1.0 / 0.529177210903
    r = r_OH_A * bohr_per_angstrom               # ~1.80916 Bohr
    half_angle = np.deg2rad(104.522 / 2.0)
    x = r * np.sin(half_angle)
    z = r * np.cos(half_angle)
    return f"O 0 0 0; H {x:.6f} 0 {-z:.6f}; H {-x:.6f} 0 {-z:.6f}"


def build_water(basis: str = "bfd-vdz"):
    mol = gto.M(
        atom=water_geometry_bohr(),
        basis={"O": basis, "H": basis},
        ecp={"O": "bfd"},
        spin=0,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


def main(
    basis: str = "bfd-vdz",
    n_walkers: int = 512,
    burn_in_steps: int = 1500,
    n_opt_steps: int = 200,
    n_mcmc_per_opt: int = 25,
    step_size: float = 0.4,
    learning_rate: float = 0.01,
    n_sample_steps: int = 8000,
    thinning: int = 10,
    seed: int = 4321,
):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    jax.config.update("jax_enable_x64", True)

    print(f"{'=' * 70}")
    print(f"H2O / BFD VMC end-to-end optimization (basis = {basis})")
    print(f"{'=' * 70}")
    mol, mf = build_water(basis)
    hf_energy = float(mf.e_tot)
    print(f"HF/{basis}             = {hf_energy:.6f} Ha")
    print(f"Reference (Zen 2013)   = {REFERENCE_E_VMC} ± {REFERENCE_E_VMC_ERR} Ha")
    print(f"Approx. BFD basis lim. = {REFERENCE_E_EXACT_BFD} Ha (FCI-equivalent)")
    print(f"  -> correlation budget = {REFERENCE_E_EXACT_BFD - hf_energy:.3f} Ha")
    print()

    # ----- Ansatz ----------------------------------------------------------
    det = SlaterDet.create(mol, mf.mo_coeff)
    jastrow = BoysHandy.create(mol, terms_per_nucleus=None, name="bh")
    jastrow_params = jastrow.init_params()
    linear_coeffs = jnp.ones(1)
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    params = [jastrow_params, linear_coeffs]

    print(f"BoysHandy param count: {jastrow.get_param_count()}")
    print()

    # ----- Optimize --------------------------------------------------------
    key = random.PRNGKey(seed)
    print(
        f"Optimization: {n_opt_steps} steps × {n_mcmc_per_opt} mcmc/step, "
        f"{n_walkers} walkers, step={step_size}, lr={learning_rate}, adam"
    )
    t0 = time.time()
    opt_results = optimize(
        ansatz,
        params=params,
        n_walkers=n_walkers,
        n_steps=n_mcmc_per_opt,
        step_size=step_size,
        burn_in_steps=burn_in_steps,
        n_opt_steps=n_opt_steps,
        optimizer_type="adam",
        learning_rate=learning_rate,
        adaptive_step_size=True,
        key=key,
    )
    t_opt = time.time() - t0
    print(f"Optimization done in {t_opt:.1f}s")
    print()

    energies_opt = np.asarray(opt_results["energies"])
    last_k = max(20, n_opt_steps // 10)
    e_late = energies_opt[-last_k:]
    print(
        f"Energy trace: first 5 = {energies_opt[:5]}, "
        f"last {last_k} mean = {e_late.mean():.6f}, std = {e_late.std():.6f}"
    )

    # ----- Final sampling at optimized parameters --------------------------
    opt_params = opt_results["params"][-1]
    print(
        f"Final sampling: {n_walkers} walkers × {n_sample_steps} steps "
        f"(record every {thinning})"
    )
    t0 = time.time()
    samples = sample(
        ansatz,
        params=opt_params,
        n_walkers=n_walkers,
        n_steps=n_sample_steps,
        step_size=step_size,
        thinning=thinning,
        burn_in_steps=burn_in_steps,
        use_importance_sampling=False,
        key=random.PRNGKey(seed + 1),
    )
    t_samp = time.time() - t0
    print(f"Sampling done in {t_samp:.1f}s")

    # ``sample`` returns a dict with "energies" of shape (n_recorded, n_walkers).
    energies_2d = np.asarray(samples["energies"])
    if energies_2d.ndim == 1:
        energies_2d = energies_2d.reshape(-1, n_walkers)
    walker_means = energies_2d.mean(axis=0)
    e_vmc = float(walker_means.mean())
    stderr_bw = float(walker_means.std(ddof=1) / np.sqrt(n_walkers))
    stderr_naive = float(energies_2d.std(ddof=1) / np.sqrt(energies_2d.size))

    print()
    print(f"{'=' * 70}")
    print(f"VMC results (basis = {basis})")
    print(f"{'=' * 70}")
    print(f"HF                 = {hf_energy:.6f} Ha")
    print(f"VMC (pytc)         = {e_vmc:.6f} ± {stderr_bw:.6f} Ha "
          f"(between-walker, naive {stderr_naive:.6f})")
    print(f"Reference (Zen)    = {REFERENCE_E_VMC} ± {REFERENCE_E_VMC_ERR} Ha")
    print(f"Δ(VMC - HF)        = {e_vmc - hf_energy:+.6f} Ha  "
          f"(correlation recovered)")
    print(f"Δ(VMC - Reference) = {e_vmc - REFERENCE_E_VMC:+.6f} Ha  "
          f"(basis-set + Jastrow flexibility gap)")
    print(f"{'=' * 70}")

    return e_vmc, stderr_bw


if __name__ == "__main__":
    main()
