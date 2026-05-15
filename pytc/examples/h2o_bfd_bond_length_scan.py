"""H2O / BFD bond-length smoothness scan.

Robustness check on the ECP-VMC kernel: sample <E_L> for five symmetric
O-H bond lengths around equilibrium with the SAME optimized Jastrow at
each geometry, and verify the resulting potential energy curve is smooth.

Sharp dips or spikes at specific r_OH would indicate numerical issues in
the V_loc / V_NL evaluation, the non-local quadrature, the Jastrow form,
or the Slater-Jastrow rank-1 ratio path — all of which depend on the
electron-atom and electron-electron geometry.

Design:

1. Optimize the JSD (NuclearCusp + BoysHandy) Jastrow ONCE at experimental
   equilibrium r_OH = 0.95721 A.  This gives a "transferable" Jastrow.
2. For each r_OH in [0.90, 0.93, 0.95721, 0.99, 1.05] A:
   a. Build the molecule + RHF/BFD.
   b. Construct a fresh SlaterJastrow with the geometry-correct SlaterDet
      (HF MOs at this r) and the SAME BoysHandy term list (parameters
      transferable since they are per-atom-type, not per-atom-position).
      Reload optimized Jastrow params onto it.
   c. Sample <E_L> with the same (seed, n_walkers, n_steps).

This intentionally re-uses the equilibrium-optimized Jastrow at off-
equilibrium points — the resulting energies are NOT a fully variationally
optimal potential energy curve.  The check is on *smoothness*: any
geometry-dependent numerical pathology in the kernel would show up as a
non-smooth feature on top of an otherwise smooth curve.

Usage:
    python -m pytc.examples.h2o_bfd_bond_length_scan
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


BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
HOH_DEG = 104.522  # experimental angle, fixed across the scan


def _geometry(r_OH_A: float) -> str:
    r = r_OH_A * BOHR_PER_ANGSTROM
    half = np.deg2rad(HOH_DEG / 2.0)
    x = r * np.sin(half)
    z = r * np.cos(half)
    return f"O 0 0 0; H {x:.6f} 0 {-z:.6f}; H {-x:.6f} 0 {-z:.6f}"


def _build_mol_hf(r_OH_A: float, basis: str = "bfd-vdz"):
    mol = gto.M(
        atom=_geometry(r_OH_A),
        basis={"O": basis, "H": basis},
        ecp={"O": "bfd"},
        spin=0, unit="Bohr", verbose=0,
    )
    mf = scf.RHF(mol); mf.kernel()
    return mol, mf


def _build_ansatz(mol, mf):
    det = SlaterDet.create(mol, mf.mo_coeff)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    bh = BoysHandy.create(mol, terms_per_nucleus=None, name="bh")
    jastrow = CompositeJastrow.create([ncusp, bh])
    ansatz = SlaterJastrow.create(mol, jastrow, [det])
    init_params = [jastrow.init_params(), jnp.ones(1)]
    return ansatz, init_params


def _between_walker_stderr(samples_dict, n_walkers):
    e_2d = np.asarray(samples_dict["energies"]).reshape(-1, n_walkers)
    wm = e_2d.mean(axis=0)
    return float(wm.mean()), float(wm.std(ddof=1) / np.sqrt(n_walkers))


def main(
    basis: str = "bfd-vdz",
    r_OH_grid_A: tuple = (0.90, 0.93, 0.95721, 0.99, 1.05),
    r_OH_opt_A: float = 0.95721,
    n_walkers: int = 256,
    burn_in_steps: int = 1000,
    n_opt_steps: int = 100,
    n_mcmc_per_opt: int = 20,
    n_sample_steps: int = 4000,
    thinning: int = 10,
    step_size: float = 0.4,
    learning_rate: float = 0.01,
    seed: int = 7777,
):
    logging.basicConfig(level=logging.WARNING)
    jax.config.update("jax_enable_x64", True)

    print(f"{'=' * 78}")
    print(f"H2O / BFD bond-length smoothness scan  (basis = {basis})")
    print(f"{'=' * 78}")
    print(
        f"Optimize at r_OH = {r_OH_opt_A:.5f} A "
        f"({r_OH_opt_A*BOHR_PER_ANGSTROM:.4f} Bohr), "
        f"then sample at {list(r_OH_grid_A)} A with the same params."
    )
    print()

    # ----- 1. Optimize Jastrow at equilibrium ------------------------------
    mol_eq, mf_eq = _build_mol_hf(r_OH_opt_A, basis)
    ansatz_eq, init_params_eq = _build_ansatz(mol_eq, mf_eq)
    print(f"HF/{basis} at r_OH={r_OH_opt_A:.5f}A : {mf_eq.e_tot:.6f} Ha")
    print(
        f"Optimizing Jastrow: {n_opt_steps} opt × {n_mcmc_per_opt} mcmc, "
        f"{n_walkers} walkers, adam @ lr={learning_rate}"
    )
    t0 = time.time()
    opt_results = optimize(
        ansatz_eq, params=init_params_eq,
        n_walkers=n_walkers, n_steps=n_mcmc_per_opt,
        step_size=step_size, burn_in_steps=burn_in_steps,
        n_opt_steps=n_opt_steps, optimizer_type="adam",
        learning_rate=learning_rate, adaptive_step_size=True,
        key=random.PRNGKey(seed),
    )
    print(f"Optimization done in {time.time() - t0:.1f}s\n")
    opt_jastrow_params = opt_results["params"][-1][0]   # the Jastrow leaf

    # ----- 2. Scan geometries ---------------------------------------------
    rows = []
    print(f"  {'r_OH (A)':>10s}  {'r_OH (Bohr)':>11s}  "
          f"{'HF (Ha)':>14s}  {'VMC (Ha)':>16s}  "
          f"{'stderr (mHa)':>13s}  {'Δ(VMC-HF) (mHa)':>17s}")
    for r_A in r_OH_grid_A:
        mol_r, mf_r = _build_mol_hf(r_A, basis)
        ansatz_r, init_r = _build_ansatz(mol_r, mf_r)
        # Transfer optimized Jastrow params; keep linear coeff = 1.
        params_r = [opt_jastrow_params, jnp.ones(1)]

        t0 = time.time()
        samples = sample(
            ansatz_r, params=params_r,
            n_walkers=n_walkers, n_steps=n_sample_steps,
            step_size=step_size, thinning=thinning,
            burn_in_steps=burn_in_steps,
            use_importance_sampling=False,
            key=random.PRNGKey(seed + 1000),  # SAME seed across geometries
        )
        e_vmc, stderr = _between_walker_stderr(samples, n_walkers)
        elapsed = time.time() - t0
        rows.append({
            "r_OH_A": r_A,
            "r_OH_Bohr": r_A * BOHR_PER_ANGSTROM,
            "HF": float(mf_r.e_tot),
            "VMC": e_vmc,
            "stderr": stderr,
            "time": elapsed,
        })
        print(
            f"  {r_A:10.5f}  {r_A*BOHR_PER_ANGSTROM:11.5f}  "
            f"{mf_r.e_tot:14.6f}  {e_vmc:16.6f}  "
            f"{stderr*1e3:13.3f}  {(e_vmc - mf_r.e_tot)*1e3:+17.3f}   "
            f"({elapsed:.1f}s)"
        )

    # ----- 3. Smoothness check --------------------------------------------
    # The "correlation recovered" = VMC - HF should vary smoothly with r_OH;
    # the underlying physics is the basis-set-truncated correlation energy
    # at each geometry, which is a smooth function.  Quantify smoothness by
    # fitting a quadratic to (VMC - HF) vs r_OH and looking at residuals.
    rs = np.array([r["r_OH_A"] for r in rows])
    corrs = np.array([r["VMC"] - r["HF"] for r in rows])
    stderrs = np.array([r["stderr"] for r in rows])
    # Quadratic fit:
    coeffs = np.polyfit(rs, corrs, deg=2)
    fit = np.polyval(coeffs, rs)
    resid = corrs - fit
    print()
    print(f"{'=' * 78}")
    print(f"Smoothness summary")
    print(f"{'=' * 78}")
    print(f"  Correlation energy recovered (VMC - HF) vs r_OH:")
    for r, c, e, f, d in zip(rs, corrs, stderrs, fit, resid):
        print(
            f"    r={r:.5f} A  corr={c*1e3:+8.3f} mHa  "
            f"(±{e*1e3:.3f})  quad_fit={f*1e3:+8.3f}  "
            f"resid={d*1e3:+7.3f} mHa"
        )
    rms_resid = np.sqrt(np.mean(resid ** 2))
    max_stderr = stderrs.max()
    print()
    print(
        f"  RMS residual from quadratic fit = {rms_resid*1e3:.3f} mHa  "
        f"(max per-point stderr = {max_stderr*1e3:.3f} mHa)"
    )
    if rms_resid < 3.0 * max_stderr:
        print(f"  -> RMS within 3σ of MCMC noise: smoothness OK.")
    else:
        print(f"  -> RMS exceeds 3σ of MCMC noise: suspicious! Investigate.")
    print(f"{'=' * 78}")
    return rows


if __name__ == "__main__":
    main()
