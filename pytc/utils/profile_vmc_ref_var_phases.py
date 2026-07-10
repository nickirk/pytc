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
import os
import subprocess
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.flatten_util
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
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))
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
    p.add_argument("--jac-batch-size", type=int, default=0,
                    help="If >0, use the batched scan-accumulator Jacobian path "
                         "(optimizer.py:187-297, NewtonOptimizer's max_vmap_batch_size "
                         "branch) instead of a full-batch vmap. Needed for W/N combos "
                         "that OOM unbatched (task #4).")
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
    walkers, _acc_hist, key, adapted_step_size = burn_in(
        det, walkers, n_steps=args.burn_in, step_size=args.step_size,
        key=subkey, params=params, report_interval=10 ** 9,
    )
    walkers = block(walkers)
    result["burn_in_total_time_s"] = time.time() - t0
    result["burn_in_time_per_step_s"] = result["burn_in_total_time_s"] / args.burn_in

    # --- One MCMC step (compile + steady-state, averaged over 20 steps) ---
    # Use the burn-in-adapted step_size, matching production (optimization.py:747-748),
    # so acceptance_rate here is comparable to a real run.
    mcmc_step = make_mcmc_step(det, adapted_step_size, move_type="one")
    key, subkey = random.split(key)
    t0 = time.time()
    walkers, acc = mcmc_step(det, walkers, subkey, params)
    walkers, acc = block(walkers), block(acc)
    result["mcmc_step_compile_time_s"] = time.time() - t0

    n_mcmc_avg = 20
    accs = []
    t0 = time.time()
    for _ in range(n_mcmc_avg):
        key, subkey = random.split(key)
        walkers, acc = mcmc_step(det, walkers, subkey, params)
        walkers, acc = block(walkers), block(acc)
        accs.append(float(acc))
    result["mcmc_step_steady_time_s"] = (time.time() - t0) / n_mcmc_avg
    result["acceptance_rate"] = sum(accs) / len(accs)

    # --- E_L + Jacobian build (Gauss-Newton path, matches optimizer.py) ---
    def single_local_energy_and_grad(w, p):
        return jax.value_and_grad(lambda pp: eval_local_energy(sj_ansatz, w, pp)[0])(p)

    result["jac_batch_size"] = args.jac_batch_size

    if args.jac_batch_size > 0:
        # Batched scan-accumulator path, matching optimizer.py:187-297
        # (NewtonOptimizer's max_vmap_batch_size>0 branch): accumulates
        # sum_e/sum_e2/sum_j/sum_jte/sum_jtj per batch instead of
        # materializing the full (W,P) Jacobian + reverse-mode residuals
        # for all walkers at once -- the latter is what OOMs past ~40
        # electrons at W=1000 on a single A100 (task #4, Wave-1 diagnosis).
        batch_size = min(args.jac_batch_size, args.n_walkers)
        n_batches = (args.n_walkers + batch_size - 1) // batch_size
        padded_n = n_batches * batch_size
        pad_count = padded_n - args.n_walkers
        result["jac_n_batches"] = n_batches

        params_vec, _ = jax.flatten_util.ravel_pytree(params)
        param_dtype = params_vec.dtype

        def pad_walkers(w):
            if pad_count == 0:
                return w, jnp.ones((padded_n,), dtype=bool)
            padded = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(
                    [x, jnp.repeat(x[:1], pad_count, axis=0)], axis=0
                ),
                w,
            )
            mask = jnp.concatenate(
                [jnp.ones((args.n_walkers,), dtype=bool),
                 jnp.zeros((pad_count,), dtype=bool)]
            )
            return padded, mask

        def flatten_jacobian(jac, n):
            jac_flat, _ = jax.tree_util.tree_flatten(jac)
            return jnp.concatenate(
                [jnp.reshape(leaf, (n, -1)) for leaf in jac_flat], axis=1
            )

        def masked_energy_jacobian_batch(batch_walkers, batch_mask):
            energies_batch, jac_batch = jax.vmap(
                single_local_energy_and_grad, in_axes=(0, None)
            )(batch_walkers, params)
            energies_batch = jnp.where(batch_mask, energies_batch, 0.0)
            jac_mat_batch = flatten_jacobian(jac_batch, batch_size)
            jac_mat_batch = jnp.where(batch_mask[:, None], jac_mat_batch, 0.0)
            return energies_batch, jac_mat_batch

        def stats_scan_body(carry, xs):
            sum_e, sum_e2, sum_j, sum_jte, sum_jtj = carry
            batch_walkers, batch_mask = xs
            energies_batch, jac_mat_batch = masked_energy_jacobian_batch(
                batch_walkers, batch_mask
            )
            sum_e = sum_e + jnp.sum(energies_batch)
            sum_e2 = sum_e2 + jnp.sum(energies_batch ** 2)
            sum_j = sum_j + jnp.sum(jac_mat_batch, axis=0)
            sum_jte = sum_jte + jac_mat_batch.T @ energies_batch
            sum_jtj = sum_jtj + jac_mat_batch.T @ jac_mat_batch
            return (sum_e, sum_e2, sum_j, sum_jte, sum_jtj), None

        @jax.jit
        def batched_jacobian_pass(walkers):
            padded_walkers, mask = pad_walkers(walkers)
            batched_walkers = jax.tree_util.tree_map(
                lambda x: x.reshape((n_batches, batch_size) + x.shape[1:]),
                padded_walkers,
            )
            batched_mask = mask.reshape((n_batches, batch_size))
            init_carry = (
                jnp.array(0.0, dtype=param_dtype),
                jnp.array(0.0, dtype=param_dtype),
                jnp.zeros_like(params_vec),
                jnp.zeros_like(params_vec),
                jnp.zeros((params_vec.shape[0], params_vec.shape[0]), dtype=param_dtype),
            )
            return jax.lax.scan(
                stats_scan_body, init_carry, (batched_walkers, batched_mask)
            )[0]

        t0 = time.time()
        sum_e, sum_e2, sum_j, sum_jte, sum_jtj = block(batched_jacobian_pass(walkers))
        result["jacobian_build_compile_time_s"] = time.time() - t0

        t0 = time.time()
        sum_e, sum_e2, sum_j, sum_jte, sum_jtj = block(batched_jacobian_pass(walkers))
        result["jacobian_build_steady_time_s"] = time.time() - t0

        # Same normalization as optimizer.py:292-297 (2/(n-1) for the
        # gradient, 2/n for the curvature matrix -- not a typo, matches
        # production exactly for this path).
        n_w = args.n_walkers
        e_mean = sum_e / n_w
        mean_j = sum_j / n_w
        grads_vec = (2.0 / (n_w - 1)) * (sum_jte - n_w * mean_j * e_mean)
        curvature_mat = (2.0 / n_w) * (sum_jtj - n_w * jnp.outer(mean_j, mean_j))
    else:
        # Unbatched path: jitted as a whole -- production runs this inside
        # the jitted training_step (optimization.py:776); an eager vmap
        # would overstate this phase's cost with per-op dispatch overhead,
        # which is precisely the number R1 hinges on.
        el_grad_jit = jax.jit(jax.vmap(single_local_energy_and_grad, in_axes=(0, None)))

        t0 = time.time()
        (energies, grads) = el_grad_jit(walkers, params)
        energies, grads = block(energies), block(grads)
        result["jacobian_build_compile_time_s"] = time.time() - t0

        t0 = time.time()
        (energies, grads) = el_grad_jit(walkers, params)
        energies, grads = block(energies), block(grads)
        result["jacobian_build_steady_time_s"] = time.time() - t0

        grads_flat, _ = jax.tree_util.tree_flatten(grads)
        jac_mat = jnp.concatenate(
            [jnp.reshape(leaf, (args.n_walkers, -1)) for leaf in grads_flat], axis=1
        )
        jac_centered = jac_mat - jnp.mean(jac_mat, axis=0, keepdims=True)
        e_centered = energies - jnp.mean(energies)
        n_w = jac_mat.shape[0]
        curvature_mat = (2.0 / (n_w - 1)) * (jac_centered.T @ jac_centered)
        grads_vec = (2.0 / (n_w - 1)) * (jac_centered.T @ e_centered)

    # Peak device memory right after the Jacobian phase (GPU only), before
    # the (much smaller) Newton solve allocates anything more.
    dev = jax.local_devices()[0]
    try:
        stats = dev.memory_stats()
        result["jacobian_build_peak_mem_bytes"] = stats.get("peak_bytes_in_use") if stats else None
    except Exception:
        result["jacobian_build_peak_mem_bytes"] = None

    # --- Newton solve (linear solve only; curvature/grad already assembled above) ---
    def solve_delta(curvature_mat, grads_vec, damping):
        p = curvature_mat.shape[0]
        damped = curvature_mat + damping * jnp.eye(p)
        return jnp.linalg.solve(damped, -grads_vec)

    solve_delta_jit = jax.jit(solve_delta)
    t0 = time.time()
    delta = block(solve_delta_jit(curvature_mat, grads_vec, args.damping))
    result["newton_solve_compile_time_s"] = time.time() - t0

    t0 = time.time()
    delta = block(solve_delta_jit(curvature_mat, grads_vec, args.damping))
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
