"""Stagewise NaN trace through one Newton/GN optimizer step, matching the
exact config of the broken (H2O)25 variance-convergence run
(#proj-pytc-efficiency-refactor, Felix's spec) -- reports isnan/isinf at
each intermediate stage of the batched Gauss-Newton exact-solver path
(``NewtonOptimizer.step``, curvature_type="gauss_newton",
max_vmap_batch_size>0, ``pytc/vmc/optimizer.py:187-297,338-364``):

    energies (inside the loss) -> jac_mat/sum_jtj -> grads_vec ->
    curvature_mat -> solved delta -> updated params

The first stage that goes non-finite is the actual break point. This
mirrors optimizer.py's math directly (same formulas, same scan structure)
rather than monkeypatching production code, so it stays a read-only
diagnostic -- no library changes.

Usage (run AFTER nan_census.py has shown clean E_L through the same
burn-in depth, on the same W/N/basis, so this picks up where that left
off):
    python -m pytc.utils.nan_stage_trace --n-water 25 --basis cc-pVTZ \
        --n-walkers 5000 --burn-in-steps 1000 --jac-batch-size 32 \
        --damping 1e-6
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
    p.add_argument("--scf-cache-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scf_cache"))
    args = p.parse_args()

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

    def masked_energy_batch(bw, bm):
        e = batch_vmap_fn(single_local_energy, in_axes=(0, None))(bw, params)
        return jnp.where(bm, e, 0.0)

    def masked_energy_jacobian_batch(bw, bm, clip_lo, clip_hi):
        e, jac = batch_vmap_fn(single_local_energy_and_grad, in_axes=(0, None))(bw, params)
        e = jnp.where(bm, e, 0.0)
        if clip_lo is not None:
            e = jnp.where(bm, jnp.clip(e, clip_lo, clip_hi), 0.0)
        jac_flat, _ = jax.tree_util.tree_flatten(jac)
        jac_mat = jnp.concatenate([jnp.reshape(leaf, (batch_size, -1)) for leaf in jac_flat], axis=1)
        jac_mat = jnp.where(bm[:, None], jac_mat, 0.0)
        return e, jac_mat

    ok = True

    # Stage 1: raw energies (unclipped, per-batch scan for the mean/MAD)
    raw_energies = []
    for i in range(n_batches):
        bw_i = jax.tree_util.tree_map(lambda x: x[i], batched_walkers)
        raw_energies.append(np.asarray(masked_energy_batch(bw_i, batched_mask[i])))
    raw_energies = np.concatenate(raw_energies)[:n_walkers]
    ok &= report("stage1_raw_energies", raw_energies)

    e_mean_raw = raw_energies.mean()
    e_std_raw = np.abs(raw_energies - e_mean_raw).mean()
    clip_lo = e_mean_raw - args.clip_multiplier * e_std_raw
    clip_hi = e_mean_raw + args.clip_multiplier * e_std_raw
    print(f"  (clip bounds: [{clip_lo:.4f}, {clip_hi:.4f}], "
          f"raw mean={e_mean_raw:.4f}, raw MAD={e_std_raw:.4f})")

    # Stage 2: energies + Jacobian per batch, accumulated
    params_vec, unravel_fn = jax.flatten_util.ravel_pytree(params)
    sum_e = sum_e2 = 0.0
    sum_j = jnp.zeros_like(params_vec)
    sum_jte = jnp.zeros_like(params_vec)
    sum_jtj = jnp.zeros((params_vec.shape[0], params_vec.shape[0]))
    all_jac_finite = True
    for i in range(n_batches):
        bw = jax.tree_util.tree_map(lambda x: x[i], batched_walkers)
        bm = batched_mask[i]
        e_b, jac_b = masked_energy_jacobian_batch(bw, bm, clip_lo, clip_hi)
        if not report(f"stage2_batch{i}_energies", e_b):
            all_jac_finite = False
        if not report(f"stage2_batch{i}_jacobian", jac_b):
            all_jac_finite = False
        sum_e = sum_e + jnp.sum(e_b)
        sum_e2 = sum_e2 + jnp.sum(e_b**2)
        sum_j = sum_j + jnp.sum(jac_b, axis=0)
        sum_jte = sum_jte + jac_b.T @ e_b
        sum_jtj = sum_jtj + jac_b.T @ jac_b
    ok &= all_jac_finite
    ok &= report("stage3_sum_jtj", sum_jtj)

    e_mean = sum_e / n_walkers
    mean_j = sum_j / n_walkers
    loss = (sum_e2 - n_walkers * e_mean**2) / (n_walkers - 1)
    ok &= report("stage3_loss", jnp.array([loss]))

    grads_vec = (2.0 / (n_walkers - 1)) * (sum_jte - n_walkers * mean_j * e_mean)
    ok &= report("stage4_grads_vec", grads_vec)

    curvature_mat = (2.0 / n_walkers) * (sum_jtj - n_walkers * jnp.outer(mean_j, mean_j))
    curvature_mat = curvature_mat + args.damping * jnp.eye(curvature_mat.shape[0])
    ok &= report("stage5_curvature_mat", curvature_mat)

    delta_vec = jax.scipy.linalg.solve(curvature_mat, -grads_vec, assume_a="pos")
    ok &= report("stage6_delta_vec", delta_vec)

    new_params_vec = params_vec + args.learning_rate * delta_vec
    ok &= report("stage7_new_params", new_params_vec)

    print()
    print("ALL STAGES FINITE" if ok else "NON-FINITE DETECTED -- see the first *** flagged stage above")


if __name__ == "__main__":
    main()
