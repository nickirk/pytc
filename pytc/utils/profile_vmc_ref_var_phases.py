"""Per-phase timing instrumentation for the optimize_ref_var VMC path.

Unlike a cProfile pass over the fused, JIT-compiled `optimize_ref_var` call
(which misattributes queued-async compute to whatever host call happens to
block first -- see the retracted 89%-in-device_get finding), this harness
calls the SAME building blocks `optimize_ref_var` uses (`burn_in`,
`make_mcmc_step`, the Newton optimizer's Jacobian-build and linear-solve
sub-steps) as separate, explicitly `jax.block_until_ready()`-bounded calls,
so each phase's wall-clock number reflects only that phase's real compute.

Emits one JSON object per run (task #1 in #pro-pytc-efficiency-refactor,
Felix's measurement spec, doc attachment b35f245c).

Usage:
    python profile_vmc_ref_var_phases.py --n-atoms 20 --basis cc-pVTZ \
        --n-walkers 1000 --burn-in 500 --n-newton-steps 10 \
        --system h-chain --out h20_w1000.json
    python profile_vmc_ref_var_phases.py --n-water 4 --basis cc-pVTZ \
        --n-walkers 1000 --system water --out h2o4_w1000.json
"""
import argparse
import json
import subprocess
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from pyscf import gto, scf

from pytc.vmc.sampling import burn_in
from pytc.vmc.metropolis import make_mcmc_step
from pytc.vmc.hamiltonian import eval_local_energy
from pytc.vmc.walker import initialize_walkers
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import BoysHandy, NuclearCusp, CompositeJastrow


def block(x):
    """Recursively block_until_ready() a pytree; return it unchanged."""
    jax.block_until_ready(x)
    return x


def h_chain(n, sep=1.8):
    return "; ".join(f"H 0 0 {i*sep}" for i in range(n))


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=__file__.rsplit("/", 3)[0]
        ).decode().strip()
    except Exception:
        return "unknown"


def build_system(args):
    if args.system == "h-chain":
        atom = h_chain(args.n_atoms)
        unit = "Bohr"
        label = f"H{args.n_atoms}"
    elif args.system == "water":
        from pytc.utils.gen_water_cluster import build_water_cluster
        atom = build_water_cluster(args.n_water)
        unit = "Angstrom"
        label = f"(H2O){args.n_water}"
    else:
        raise ValueError(args.system)
    return atom, unit, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=["h-chain", "water"], required=True)
    p.add_argument("--n-atoms", type=int, default=20, help="H-chain: n H atoms")
    p.add_argument("--n-water", type=int, default=2, help="water: n H2O monomers")
    p.add_argument("--basis", default="cc-pVTZ")
    p.add_argument("--n-walkers", type=int, default=1000)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--n-newton-steps", type=int, default=10)
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--damping", type=float, default=1e-6)
    p.add_argument("--out", default=None, help="Write JSON here (default: stdout)")
    args = p.parse_args()

    atom, unit, label = build_system(args)
    result = {
        "commit": git_commit(),
        "system": label,
        "basis": args.basis,
        "n_walkers": args.n_walkers,
        "burn_in_steps": args.burn_in,
        "n_newton_steps": args.n_newton_steps,
    }

    t0 = time.time()
    mol = gto.M(atom=atom, basis=args.basis, unit=unit, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    result["scf_time_s"] = time.time() - t0
    result["n_orb"] = int(mol.nao)
    result["n_elec"] = int(mol.nelectron)

    det = SlaterDet.create(mol, mf.mo_coeff)
    bh = BoysHandy.create(mol)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    jastrow = CompositeJastrow.create([ncusp, bh])
    jastrow_params = jastrow.init_params()
    sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]

    key = random.PRNGKey(43)
    key, subkey = random.split(key)

    # --- Walker init ---
    t0 = time.time()
    walkers = block(initialize_walkers(sj_ansatz, args.n_walkers, key=subkey))
    result["walker_init_time_s"] = time.time() - t0

    # --- Burn-in ---
    # NOTE: calling burn_in() twice in a row on already-warmed-up walkers hits
    # a vmap shape bug in _warmup_ansatz (sampling.py) unrelated to this
    # profiling harness -- filed separately, not blocking this PR. Reporting
    # one number (total, including one-time JIT compile) rather than a
    # compile/steady-state split until that's fixed.
    key, subkey = random.split(key)
    t0 = time.time()
    walkers, _acc_hist, key, _step_size = burn_in(
        det, walkers, n_steps=args.burn_in, step_size=args.step_size,
        key=subkey, params=params, report_interval=10 ** 9,
    )
    walkers = block(walkers)
    result["burn_in_total_time_s"] = time.time() - t0
    result["burn_in_time_per_step_s"] = result["burn_in_total_time_s"] / args.burn_in

    # --- One MCMC step (compile + steady-state) ---
    mcmc_step = make_mcmc_step(det, args.step_size, move_type="one")
    key, subkey = random.split(key)
    t0 = time.time()
    walkers, acc = mcmc_step(det, walkers, subkey, params)
    walkers, acc = block(walkers), block(acc)
    result["mcmc_step_compile_time_s"] = time.time() - t0

    key, subkey = random.split(key)
    t0 = time.time()
    walkers, acc = mcmc_step(det, walkers, subkey, params)
    walkers, acc = block(walkers), block(acc)
    result["mcmc_step_steady_time_s"] = time.time() - t0
    result["acceptance_rate"] = float(acc)

    # --- E_L + Jacobian build (Gauss-Newton path, matches optimizer.py) ---
    # Use the same vmap pattern as NewtonOptimizer's gauss_newton path.
    el_grad_vmap = jax.vmap(
        lambda w, p: jax.value_and_grad(lambda pp: eval_local_energy(sj_ansatz, w, pp)[0])(p),
        in_axes=(0, None),
    )

    t0 = time.time()
    (energies, grads) = el_grad_vmap(walkers, params)
    energies, grads = block(energies), block(grads)
    result["jacobian_build_compile_time_s"] = time.time() - t0

    t0 = time.time()
    (energies, grads) = el_grad_vmap(walkers, params)
    energies, grads = block(energies), block(grads)
    result["jacobian_build_steady_time_s"] = time.time() - t0

    # --- Newton solve (curvature matrix assembly + linear solve) ---
    grads_flat, treedef = jax.tree_util.tree_flatten(grads)
    jac_mat = jnp.concatenate(
        [jnp.reshape(leaf, (args.n_walkers, -1)) for leaf in grads_flat], axis=1
    )

    def newton_solve(jac_mat, energies, damping):
        jac_centered = jac_mat - jnp.mean(jac_mat, axis=0, keepdims=True)
        e_centered = energies - jnp.mean(energies)
        n_w = jac_mat.shape[0]
        curvature = (2.0 / (n_w - 1)) * (jac_centered.T @ jac_centered)
        grad_var = (2.0 / (n_w - 1)) * (jac_centered.T @ e_centered)
        p = curvature.shape[0]
        damped = curvature + damping * jnp.eye(p)
        delta = jnp.linalg.solve(damped, -grad_var)
        return delta

    newton_solve_jit = jax.jit(newton_solve)
    t0 = time.time()
    delta = block(newton_solve_jit(jac_mat, energies, args.damping))
    result["newton_solve_compile_time_s"] = time.time() - t0

    t0 = time.time()
    delta = block(newton_solve_jit(jac_mat, energies, args.damping))
    result["newton_solve_steady_time_s"] = time.time() - t0

    # --- Peak device memory (GPU only; empty dict on CPU) ---
    dev = jax.local_devices()[0]
    try:
        stats = dev.memory_stats()
        result["peak_device_memory_bytes"] = stats.get("peak_bytes_in_use") if stats else None
    except Exception:
        result["peak_device_memory_bytes"] = None
    result["device_kind"] = dev.device_kind

    out = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"Wrote {args.out}")
    print(out)


if __name__ == "__main__":
    main()
