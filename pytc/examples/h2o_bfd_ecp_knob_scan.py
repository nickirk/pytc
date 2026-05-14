"""ECP-knob sensitivity scan for the H2O / BFD JSD VMC pipeline.

The locality-approximation non-local energy has two tunable knobs:

1. Angular quadrature grid: 12-point icosahedral (exact through l=5, default)
   vs 26-point Lebedev (exact through l=7).
2. Non-local spatial cutoff tau: |V_l(r)| < tau drops pair contributions
   when r_iA > r_cut^A. Default 1e-5 Ha (QMCPACK convention).

This script optimizes the JSD wavefunction ONCE with default settings, then
samples <E_L> at the optimized parameters under different (grid, tau) combos.
Since the wavefunction is fixed, every difference is a pure ECP-integration
effect.

Setup: H2O at experimental geometry with BFD on O, BFD-VDZ basis on both
species (small fast basis to isolate the ECP effect from basis-set noise).
JSD = NuclearCusp (gated off at O) + BoysHandy default 17 terms.

Measured result on this branch (BFD-VDZ, 256 walkers × 150 opt × 20 mcmc +
4000 sample / thinning 10, single seed, adam lr=0.01):

    grid              tau     r_cut(O)            VMC      Δ vs baseline
    icosahedral_12    1e-5    1.319      -17.212535       baseline
    lebedev_26        1e-5    1.319      -17.212618       -0.084 mHa
    icosahedral_12    1e-3    1.101      -17.212604       -0.069 mHa
    icosahedral_12    1e-7    1.507      -17.212534       +0.000 mHa
    icosahedral_12    1e-9    1.673      -17.212534       +0.000 mHa
    lebedev_26        1e-9    1.673      -17.212618       -0.083 mHa

Stderr per row ~3.55 mHa, so the only consistent signal is the
icosahedral_12 vs lebedev_26 offset of -0.08 mHa (a deterministic
quadrature truncation in the trial wf's l>5 angular content; appears
identically at tau=1e-5 and tau=1e-9 because the walker trajectories are
seed-identical and psi is unchanged across rows).  All other differences
are pure MCMC noise.

Conclusion: the default (icosahedral_12, tau=1e-5) is fully converged
for BFD-on-O.  Higher-l pseudopotentials (transition metals with l=2
non-local channels) might benefit from the 26-point Lebedev grid; for
main-group BFD/ccECP atoms with l<=1 non-local channels, 12-point is
over-specced and the knobs are at the noise floor.

Usage:
    python -m pytc.examples.h2o_bfd_ecp_knob_scan
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
from pytc.ecp.parser import parse_pyscf_ecp
from pytc.jastrow import BoysHandy, CompositeJastrow, NuclearCusp
from pytc.vmc import optimize, sample


def water_geometry_bohr():
    r_OH_A = 0.95721
    bohr = 1.0 / 0.529177210903
    r = r_OH_A * bohr
    half_angle = np.deg2rad(104.522 / 2.0)
    x = r * np.sin(half_angle)
    z = r * np.cos(half_angle)
    return f"O 0 0 0; H {x:.6f} 0 {-z:.6f}; H {-x:.6f} 0 {-z:.6f}"


def build_h2o_bfd(basis="bfd-vdz"):
    mol = gto.M(
        atom=water_geometry_bohr(),
        basis={"O": basis, "H": basis},
        ecp={"O": "bfd"},
        spin=0, unit="Bohr", verbose=0,
    )
    mf = scf.RHF(mol); mf.kernel()
    return mol, mf


def build_jsd_ansatz(mol, mf, *, ecp_quad_grid="icosahedral_12",
                     ecp_nl_cutoff_tol=1.0e-5):
    det = SlaterDet.create(mol, mf.mo_coeff)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    bh = BoysHandy.create(mol, terms_per_nucleus=None, name="bh")
    jastrow = CompositeJastrow.create([ncusp, bh])
    ansatz = SlaterJastrow.create(
        mol, jastrow, [det],
        ecp_nl_cutoff_tol=ecp_nl_cutoff_tol,
        ecp_quad_grid=ecp_quad_grid,
    )
    params = [jastrow.init_params(), jnp.ones(1)]
    return ansatz, params


def sample_walker_mean(ansatz, opt_params, *, n_walkers, n_sample_steps,
                       thinning, burn_in, step_size, key):
    """<E_L> at fixed parameters under the given ansatz; between-walker stderr."""
    samples = sample(
        ansatz,
        params=opt_params,
        n_walkers=n_walkers,
        n_steps=n_sample_steps,
        step_size=step_size,
        thinning=thinning,
        burn_in_steps=burn_in,
        use_importance_sampling=False,
        key=key,
    )
    energies = np.asarray(samples["energies"]).reshape(-1, n_walkers)
    walker_means = energies.mean(axis=0)
    mean = float(walker_means.mean())
    stderr_bw = float(walker_means.std(ddof=1) / np.sqrt(n_walkers))
    return mean, stderr_bw


def main(
    basis="bfd-vdz",
    n_walkers=256,
    n_opt_steps=150,
    n_mcmc_per_opt=20,
    burn_in_steps=1000,
    n_sample_steps=4000,
    thinning=10,
    step_size=0.4,
    learning_rate=0.01,
    seed=2024,
):
    logging.basicConfig(level=logging.WARNING)
    jax.config.update("jax_enable_x64", True)

    print(f"{'=' * 78}")
    print(f"H2O / BFD ECP-knob sensitivity scan ({basis}, fixed JSD wf)")
    print(f"{'=' * 78}")

    mol, mf = build_h2o_bfd(basis)
    hf_energy = float(mf.e_tot)
    print(f"HF/{basis} = {hf_energy:.6f} Ha\n")

    # ----- One optimization with default ECP settings ----------------------
    ansatz0, params0 = build_jsd_ansatz(mol, mf)
    print(
        f"Optimizing JSD with default ECP (12-pt grid, tau=1e-5): "
        f"{n_opt_steps} opt × {n_mcmc_per_opt} mcmc, {n_walkers} walkers"
    )
    t0 = time.time()
    opt_results = optimize(
        ansatz0,
        params=params0,
        n_walkers=n_walkers,
        n_steps=n_mcmc_per_opt,
        step_size=step_size,
        burn_in_steps=burn_in_steps,
        n_opt_steps=n_opt_steps,
        optimizer_type="adam",
        learning_rate=learning_rate,
        adaptive_step_size=True,
        key=random.PRNGKey(seed),
    )
    print(f"Optimization done in {time.time() - t0:.1f}s\n")
    opt_params = opt_results["params"][-1]

    # ----- Knob matrix -----------------------------------------------------
    # The optimized Jastrow params and SlaterDet are reusable; only the ECP
    # integration (parse + r_cut + quad grid) changes.  We rebuild only the
    # SlaterJastrow shell with different EcpData and the same det+jastrow.

    knob_grid = [
        ("icosahedral_12", 1.0e-5),
        ("lebedev_26",     1.0e-5),
        ("icosahedral_12", 1.0e-3),
        ("icosahedral_12", 1.0e-7),
        ("icosahedral_12", 1.0e-9),
        ("lebedev_26",     1.0e-9),
    ]

    results = []
    for grid_name, tau in knob_grid:
        ecp_data = parse_pyscf_ecp(mol, nl_cutoff_tol=tau, quad_grid_name=grid_name)
        ansatz = ansatz0.replace(ecp=ecp_data)
        r_cut = float(np.asarray(ecp_data.r_cut)[0])  # O is atom 0
        t0 = time.time()
        e_vmc, stderr = sample_walker_mean(
            ansatz, opt_params,
            n_walkers=n_walkers,
            n_sample_steps=n_sample_steps,
            thinning=thinning,
            burn_in=burn_in_steps,
            step_size=step_size,
            key=random.PRNGKey(seed + 100),
        )
        elapsed = time.time() - t0
        results.append({
            "grid": grid_name,
            "tau": tau,
            "r_cut_O": r_cut,
            "VMC": e_vmc,
            "stderr": stderr,
            "time": elapsed,
        })
        print(
            f"  grid={grid_name:<16s}  tau={tau:.0e}   r_cut_O={r_cut:.3f} Bohr   "
            f"VMC = {e_vmc:.6f} ± {stderr:.6f} Ha   ({elapsed:.1f}s)"
        )

    # ----- Summary ---------------------------------------------------------
    baseline = results[0]
    print()
    print(f"{'=' * 78}")
    print(f"Summary  (Δ vs baseline icosahedral_12 / tau=1e-5)")
    print(f"{'=' * 78}")
    print(f"  {'grid':<16s}  {'tau':<8s}  {'r_cut(O)':>9s}  "
          f"{'VMC':>13s}  {'Δ vs baseline':>15s}")
    for r in results:
        delta = (r["VMC"] - baseline["VMC"]) * 1e3  # mHa
        marker = "  ← baseline" if r is baseline else ""
        print(
            f"  {r['grid']:<16s}  {r['tau']:.0e}  {r['r_cut_O']:9.3f}  "
            f"{r['VMC']:13.6f}  {delta:+11.3f} mHa{marker}"
        )
    print(f"{'=' * 78}")
    print(
        f"NB: stderr per row is ~{baseline['stderr']*1e3:.2f} mHa; differences "
        "below ~5 mHa are dominated by MCMC noise on a single seed."
    )
    return results


if __name__ == "__main__":
    main()
