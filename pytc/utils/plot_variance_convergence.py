"""Run optimize_ref_var for N steps and plot the reference-variance loss
(and energy) vs optimization step, for Ke's convergence-behavior request
(task #1, #proj-pytc-efficiency-refactor) -- the per-phase timing harness
(profile_vmc_ref_var_phases.py) only measures ONE Newton step, it doesn't
track convergence across steps, so this is a separate script.

Uses optimize_ref_var's own history tracking (result["cost"]/["energies"]/
["stds"]/["acceptance"]) rather than reimplementing a training loop.

Usage:
    python plot_variance_convergence.py --system water --n-water 25 \
        --basis cc-pVTZ --n-walkers 1000 --n-opt-steps 100 \
        --jac-batch-size 64 \
        --out h2o25_convergence.json --plot h2o25_convergence.png
"""
import argparse
import json
import os
import subprocess
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import random
from pyscf import gto, scf

from pytc.vmc import optimize_ref_var
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import NuclearCusp, CompositeJastrow
from pytc.jastrow.bha import BoysHandyAnalytical


def h_chain(n, sep=1.8):
    return "; ".join(f"H 0 0 {i*sep}" for i in range(n))


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return "unknown"


def run_scf(mol, atom, basis, unit, cache_dir):
    """Same caching pattern as profile_vmc_ref_var_phases.py -- large
    systems shouldn't re-pay SCF on every convergence-plot rerun either."""
    mf = scf.RHF(mol).density_fit()
    if not cache_dir:
        t0 = time.time()
        mf.kernel()
        return mf, time.time() - t0, False
    os.makedirs(cache_dir, exist_ok=True)
    import hashlib
    cache_key = hashlib.sha256(f"{atom}|{basis}|{unit}".encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"{cache_key}.h5")
    t0 = time.time()
    if os.path.exists(cache_path):
        loaded = scf.chkfile.load(cache_path, "scf")
        mf.mo_coeff = loaded["mo_coeff"]
        mf.mo_energy = loaded["mo_energy"]
        mf.mo_occ = loaded["mo_occ"]
        mf.e_tot = loaded["e_tot"]
        mf.converged = True
        return mf, time.time() - t0, True
    mf.chkfile = cache_path
    mf.kernel()
    return mf, time.time() - t0, False


def save_figure(fig, path, meta):
    """Save a figure with a provenance footer (commit/system/params) baked
    into the image itself -- no established save_figure() helper existed
    elsewhere in the repo, so this mirrors the JSON provenance fields
    (git_commit(), system, basis, n_walkers, jac_batch_size) already used
    by profile_vmc_ref_var_phases.py, just rendered onto the PNG so the
    plot is self-describing even if separated from its JSON sidecar."""
    footer = (
        f"commit={meta['commit'][:12]}  system={meta['system']}  basis={meta['basis']}  "
        f"W={meta['n_walkers']}  burn_in={meta['burn_in_steps']}  "
        f"jac_batch={meta['jac_batch_size'] or 'full'}"
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7, color="gray")
    fig.savefig(path, dpi=150)


def build_system(args):
    if args.system == "h-chain":
        return h_chain(args.n_atoms), "Bohr", f"H{args.n_atoms}"
    elif args.system == "water":
        from pytc.utils.gen_water_cluster import build_water_cluster
        return build_water_cluster(args.n_water), "Angstrom", f"(H2O){args.n_water}"
    raise ValueError(args.system)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["h-chain", "water"], required=True)
    p.add_argument("--n-atoms", type=int, default=20)
    p.add_argument("--n-water", type=int, default=2)
    p.add_argument("--basis", default="cc-pVTZ")
    p.add_argument("--n-walkers", type=int, default=1000)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--n-opt-steps", type=int, default=50,
                    help="Felix's spec for the first GPU convergence run: 50 steps, "
                         "extend later if the curve looks healthy.")
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--damping", type=float, default=1e-6)
    p.add_argument("--jac-batch-size", type=int, default=0,
                    help="max_vmap_batch_size passed to optimize_ref_var -- unlike "
                         "the timing harness, optimize_ref_var exposes only ONE "
                         "combined batch-size knob shared by burn_in/MCMC/Jacobian, "
                         "so this sizes all three. Match to whatever fits per task "
                         "#6's Wave-2 per-system sizing (e.g. 64 for (H2O)25 at "
                         "moderate W, 32 for W=5000+).")
    p.add_argument("--scf-cache-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scf_cache"),
                    help="Same cache as profile_vmc_ref_var_phases.py; empty string disables.")
    p.add_argument("--out", default=None, help="Write raw history JSON here.")
    p.add_argument("--plot", default=None, help="Write convergence plot PNG here.")
    args = p.parse_args()

    atom, unit, label = build_system(args)
    result_meta = {
        "commit": git_commit(),
        "system": label,
        "basis": args.basis,
        "n_walkers": args.n_walkers,
        "burn_in_steps": args.burn_in,
        "n_opt_steps": args.n_opt_steps,
        "jac_batch_size": args.jac_batch_size,
    }

    mol = gto.M(atom=atom, basis=args.basis, unit=unit, verbose=0)
    mf, scf_time_s, scf_cached = run_scf(mol, atom, args.basis, unit, args.scf_cache_dir)
    result_meta["scf_time_s"] = scf_time_s
    result_meta["scf_cached"] = scf_cached
    result_meta["n_orb"] = int(mol.nao)
    result_meta["n_elec"] = int(mol.nelectron)

    det = SlaterDet.create(mol, mf.mo_coeff)
    bh = BoysHandyAnalytical.create(mol)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    jastrow = CompositeJastrow.create([ncusp, bh])
    jastrow_params = jastrow.init_params()
    sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]

    key = random.PRNGKey(43)

    t0 = time.time()
    opt_result = optimize_ref_var(
        sj_ansatz,
        params=params,
        n_walkers=args.n_walkers,
        n_opt_steps=args.n_opt_steps,
        burn_in_steps=args.burn_in,
        step_size=args.step_size,
        max_vmap_batch_size=args.jac_batch_size,
        optimizer_type="newton",
        learning_rate=args.learning_rate,
        opt_kwargs={"damping": args.damping, "solver": "exact"},
        key=key,
    )
    result_meta["total_optimize_time_s"] = time.time() - t0

    cost = np.asarray(opt_result["cost"])
    energies = np.asarray(opt_result["energies"])
    stds = np.asarray(opt_result["stds"])
    acceptance = np.asarray(opt_result["acceptance"])

    out_payload = {
        **result_meta,
        "cost": cost.tolist(),
        "energies": energies.tolist(),
        "stds": stds.tolist(),
        "acceptance": acceptance.tolist(),
    }
    out_json = json.dumps(out_payload, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out_json)
        print(f"Wrote {args.out}")
    else:
        print(out_json)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.arange(len(cost))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ax1.plot(steps, cost, marker="o", markersize=3)
        ax1.set_ylabel("Reference variance (cost)")
        ax1.set_yscale("log")
        ax1.set_title(f"{label} / {args.basis}, W={args.n_walkers}, "
                       f"jac_batch={args.jac_batch_size or 'full'}")
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, energies, marker="o", markersize=3, color="C1")
        if len(stds) == len(energies):
            ax2.fill_between(steps, energies - stds, energies + stds, alpha=0.2, color="C1")
        ax2.set_xlabel("Optimization step")
        ax2.set_ylabel("Energy (Ha)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0, 0.02, 1, 1))
        save_figure(fig, args.plot, result_meta)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
