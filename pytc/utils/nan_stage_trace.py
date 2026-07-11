"""Stagewise NaN trace through the Newton/GN optimizer, matching the
exact config of the broken (H2O)25 variance-convergence run
(#proj-pytc-efficiency-refactor, Felix's spec) -- reports isnan/isinf at
each intermediate stage of the batched Gauss-Newton exact-solver path
(``NewtonOptimizer.step``, curvature_type="gauss_newton",
max_vmap_batch_size>0, ``pytc/vmc/optimizer.py:187-297,338-364``):

    energies (inside the loss) -> jac_mat/sum_jtj -> grads_vec ->
    curvature_mat -> solved delta -> updated params

**Critical structural point (Felix, independently confirmed from code):**
``optimize_ref_var``'s printed "Step 0" is NOT one Newton update. With
the default cadence (``n_mcmc_per_opt``/``n_opt_per_mcmc`` both
unset -> ``n_opt_per_mcmc = n_steps = 20``),
``make_second_order_training_step``'s Pattern 2
(``optimization.py:250-274``) chains 20 sequential ``optimizer.step()``
calls via ``jax.lax.scan`` on the SAME frozen walkers (no MCMC move
between them), params updating each iteration, and only logs the LAST
(20th) iteration's stats. A single-Newton-step trace only tests
iteration 1 -- clean by itself does not mean the whole 20-substep chain
stays clean. This script iterates the full chain and reports magnitudes
(not just isnan) at every iteration, since a huge-but-finite delta at
step 5 can compound into overflow by step 15-20 even though step 1
alone looks fine.

The first iteration whose per-walker E_L or optimizer stage goes
non-finite is the actual break point. This mirrors optimizer.py's math
directly (same formulas, same scan structure, same learning-rate
schedule) rather than monkeypatching production code, so it stays a
read-only diagnostic -- no library changes.

Usage (run AFTER nan_census.py has shown clean E_L through the same
burn-in depth, on the same W/N/basis, so this picks up where that left
off):
    python -m pytc.utils.nan_stage_trace --n-water 25 --basis cc-pVTZ \
        --n-walkers 5000 --burn-in-steps 1000 --jac-batch-size 32 \
        --damping 1e-6 --n-newton-substeps 20
"""
import argparse
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import random
from pyscf import gto, scf

from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import NuclearCusp, CompositeJastrow
from pytc.jastrow.bha import BoysHandyAnalytical
from pytc.vmc.walker import initialize_walkers
from pytc.vmc.sampling import burn_in


def build_system(n_water, basis, scf_cache_dir):
    from pytc.utils.gen_water_cluster import build_water_cluster
    atom = build_water_cluster(n_water)
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).density_fit()
    cache_path = None
    if scf_cache_dir:
        os.makedirs(scf_cache_dir, exist_ok=True)
        import hashlib
        key = hashlib.sha256(f"{atom}|{basis}|Angstrom".encode()).hexdigest()[:16]
        cache_path = os.path.join(scf_cache_dir, f"{key}.h5")
    if cache_path and os.path.exists(cache_path):
        loaded = scf.chkfile.load(cache_path, "scf")
        mf.mo_coeff = loaded["mo_coeff"]
        mf.mo_energy = loaded["mo_energy"]
        mf.mo_occ = loaded["mo_occ"]
        mf.e_tot = loaded["e_tot"]
        mf.converged = True
    else:
        if cache_path:
            mf.chkfile = cache_path
        mf.kernel()
    return mol, mf


def report(name, x):
    x = np.asarray(x)
    n_nan = int(np.isnan(x).sum())
    n_inf = int(np.isinf(x).sum())
    status = "OK" if (n_nan == 0 and n_inf == 0) else "*** NON-FINITE ***"
    print(f"  [{name}] shape={x.shape}  n_nan={n_nan}  n_inf={n_inf}  {status}")
    return n_nan == 0 and n_inf == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-water", type=int, default=25)
    p.add_argument("--basis", default="cc-pVTZ")
    p.add_argument("--n-walkers", type=int, default=5000)
    p.add_argument("--burn-in-steps", type=int, default=1000)
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--vmap-batch-size", type=int, default=256)
    p.add_argument("--jac-batch-size", type=int, default=32,
                    help="max_vmap_batch_size for the Newton step -- default "
                         "matches the broken run's config.")
    p.add_argument("--damping", type=float, default=1e-6)
    p.add_argument("--clip-multiplier", type=float, default=5.0)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--min-learning-rate", type=float, default=0.01,
                    help="NewtonOptimizer's default min_learning_rate floor.")
    p.add_argument("--lr-decay-rate", type=float, default=1.0)
    p.add_argument("--lr-transition-steps", type=int, default=100)
    p.add_argument("--n-newton-substeps", type=int, default=20,
                    help="Number of chained optimizer.step() calls on the SAME "
                         "frozen walkers, mirroring Pattern 2's scan (default "
                         "matches optimize_ref_var's n_opt_per_mcmc=n_steps=20 "
                         "when neither cadence arg is set).")
    p.add_argument("--scf-cache-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scf_cache"))
    args = p.parse_args()

    def newton_lr(step):
        """Mirrors create_optimizer's newton_schedule exactly."""
        base = args.learning_rate / (1.0 + (step / args.lr_transition_steps) * args.lr_decay_rate)
        return max(base, args.min_learning_rate)

    mol, mf = build_system(args.n_water, args.basis, args.scf_cache_dir)
    det = SlaterDet.create(mol, mf.mo_coeff)
    bh = BoysHandyAnalytical.create(mol)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    jastrow = CompositeJastrow.create([ncusp, bh])
    jastrow_params = jastrow.init_params()
    sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    walkers = initialize_walkers(det, args.n_walkers, None, subkey)
    if args.burn_in_steps > 0:
        walkers, _, key, _ = burn_in(
            sj_ansatz, walkers, n_steps=args.burn_in_steps, step_size=args.step_size,
            key=key, params=params, max_vmap_batch_size=args.vmap_batch_size,
        )
    n_walkers = args.n_walkers
    batch_size = args.jac_batch_size
    n_batches = -(-n_walkers // batch_size)
    pad_count = n_batches * batch_size - n_walkers

    def pad(w):
        return jax.tree_util.tree_map(
            lambda x: jnp.concatenate(
                [x, jnp.zeros((pad_count,) + x.shape[1:], dtype=x.dtype)], axis=0
            ) if pad_count > 0 else x,
            w,
        )

    padded_walkers = pad(walkers)
    mask = jnp.concatenate(
        [jnp.ones((n_walkers,), dtype=bool), jnp.zeros((pad_count,), dtype=bool)]
    )
    batched_walkers = jax.tree_util.tree_map(
        lambda x: x.reshape((n_batches, batch_size) + x.shape[1:]), padded_walkers
    )
    batched_mask = mask.reshape((n_batches, batch_size))

    def single_local_energy_and_grad(w, p):
        return jax.value_and_grad(lambda pp: sj_ansatz.local_energy(w, pp)[0])(p)

    def single_local_energy(w, p):
        return sj_ansatz.local_energy(w, p)[0]

    batch_vmap_fn = jax.vmap

    def masked_energy_batch(bw, bm, p):
        e = batch_vmap_fn(single_local_energy, in_axes=(0, None))(bw, p)
        return jnp.where(bm, e, 0.0)

    def masked_energy_jacobian_batch(bw, bm, p, clip_lo, clip_hi):
        e, jac = batch_vmap_fn(single_local_energy_and_grad, in_axes=(0, None))(bw, p)
        e = jnp.where(bm, e, 0.0)
        if clip_lo is not None:
            e = jnp.where(bm, jnp.clip(e, clip_lo, clip_hi), 0.0)
        jac_flat, _ = jax.tree_util.tree_flatten(jac)
        jac_mat = jnp.concatenate([jnp.reshape(leaf, (batch_size, -1)) for leaf in jac_flat], axis=1)
        jac_mat = jnp.where(bm[:, None], jac_mat, 0.0)
        return e, jac_mat

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)

    def component_maxabs(p, name):
        jastrow_params = p[0]
        # jastrow_params is a list of per-component dicts (CompositeJastrow);
        # find the BoysHandyAnalytical component's b_raw/d_raw/c_raw.
        for comp in jastrow_params:
            if isinstance(comp, dict) and name in comp:
                return float(jnp.max(jnp.abs(comp[name])))
        return float("nan")

    def run_substep(p, opt_state_step, verbose_stage_report=False):
        """One Pattern-2 optimizer.step() equivalent on the SAME frozen
        walkers. Returns (new_p, delta_vec, loss, ok)."""
        ok = True

        raw_energies = []
        for i in range(n_batches):
            bw_i = jax.tree_util.tree_map(lambda x: x[i], batched_walkers)
            raw_energies.append(np.asarray(masked_energy_batch(bw_i, batched_mask[i], p)))
        raw_energies = np.concatenate(raw_energies)[:n_walkers]
        if verbose_stage_report:
            ok &= report("stage1_raw_energies", raw_energies)

        e_mean_raw = raw_energies.mean()
        e_std_raw = np.abs(raw_energies - e_mean_raw).mean()
        clip_lo = e_mean_raw - args.clip_multiplier * e_std_raw
        clip_hi = e_mean_raw + args.clip_multiplier * e_std_raw

        params_vec, _ = jax.flatten_util.ravel_pytree(p)
        sum_e = sum_e2 = 0.0
        sum_j = jnp.zeros_like(params_vec)
        sum_jte = jnp.zeros_like(params_vec)
        sum_jtj = jnp.zeros((params_vec.shape[0], params_vec.shape[0]))
        for i in range(n_batches):
            bw = jax.tree_util.tree_map(lambda x: x[i], batched_walkers)
            bm = batched_mask[i]
            e_b, jac_b = masked_energy_jacobian_batch(bw, bm, p, clip_lo, clip_hi)
            if verbose_stage_report:
                ok &= report(f"stage2_batch{i}_energies", e_b)
                ok &= report(f"stage2_batch{i}_jacobian", jac_b)
            sum_e = sum_e + jnp.sum(e_b)
            sum_e2 = sum_e2 + jnp.sum(e_b**2)
            sum_j = sum_j + jnp.sum(jac_b, axis=0)
            sum_jte = sum_jte + jac_b.T @ e_b
            sum_jtj = sum_jtj + jac_b.T @ jac_b

        e_mean = sum_e / n_walkers
        mean_j = sum_j / n_walkers
        loss = (sum_e2 - n_walkers * e_mean**2) / (n_walkers - 1)

        grads_vec = (2.0 / (n_walkers - 1)) * (sum_jte - n_walkers * mean_j * e_mean)
        curvature_mat = (2.0 / n_walkers) * (sum_jtj - n_walkers * jnp.outer(mean_j, mean_j))
        curvature_mat = curvature_mat + args.damping * jnp.eye(curvature_mat.shape[0])
        delta_vec = jax.scipy.linalg.solve(curvature_mat, -grads_vec, assume_a="pos")

        lr = newton_lr(opt_state_step)
        new_params_vec = params_vec + lr * delta_vec
        new_p = unravel_fn(new_params_vec)

        if verbose_stage_report:
            ok &= report("stage3_sum_jtj", sum_jtj)
            ok &= report("stage3_loss", jnp.array([loss]))
            ok &= report("stage4_grads_vec", grads_vec)
            ok &= report("stage5_curvature_mat", curvature_mat)
            ok &= report("stage6_delta_vec", delta_vec)
            ok &= report("stage7_new_params", new_params_vec)

        return new_p, delta_vec, float(loss), lr, ok

    print(f"Chaining {args.n_newton_substeps} optimizer.step() calls on the SAME "
          f"frozen walkers (Pattern 2), params updating each iteration...")
    print()

    cur_params = params
    for substep in range(args.n_newton_substeps):
        verbose = (substep == 0)  # full per-stage report only for iteration 1
        new_params, delta_vec, loss, lr, ok = run_substep(
            cur_params, substep, verbose_stage_report=verbose
        )
        delta_norm = float(jnp.linalg.norm(delta_vec))
        b_max = component_maxabs(new_params, "b_raw")
        d_max = component_maxabs(new_params, "d_raw")
        c_max = component_maxabs(new_params, "c_raw")
        status = "OK" if ok else "*** NON-FINITE (see stage report above) ***"
        print(f"  [substep {substep:2d}] lr={lr:.4f}  loss={loss:.6e}  "
              f"‖delta‖={delta_norm:.6e}  max|b_raw|={b_max:.6e}  "
              f"max|d_raw|={d_max:.6e}  max|c_raw|={c_max:.6e}  {status}")

        if not ok or not np.isfinite(loss) or not np.isfinite(delta_norm):
            print()
            print(f"NON-FINITE DETECTED at substep {substep} -- see the stage "
                  f"report above (only printed for substep 0; rerun with "
                  f"--n-newton-substeps {substep+1} to get the full per-stage "
                  f"breakdown for the failing substep).")
            break

        cur_params = new_params
    else:
        print()
        print("ALL SUBSTEPS FINITE through the full chain")


if __name__ == "__main__":
    main()
