"""NaN census for the (H2O)25 variance-convergence NaN-from-step-0 bug
(#proj-pytc-efficiency-refactor, Felix's diagnosis request).

Grace's bouchet run showed Var/E as NaN starting at step 0, with sane
acceptance (~0.5) -- suggesting a small number of walkers are individually
poisoned (E_L = NaN/inf for that walker) rather than a systematic math
bug, since a single poisoned walker NaNs the batch mean/variance via plain
reductions.

Leading hypothesis (Felix): Sherman-Morrison degeneracy during burn-in.
``update_inverse_sherman_morrison`` (pytc/ansatz/det.py:328-348) divides
by ``ratio`` (det(S')/det(S)) with no zero-guard -- a walker's proposed
move that lands near a nodal surface (ratio -> 0) blows up ``inv`` to
inf/NaN, permanently poisoning that walker's E_L for the rest of the run
via the rank-1 update path (``rank1_update_one_electron``).

This script: run E_L on (H2O)25/cc-pVTZ, W=500, on (a) freshly
initialized walkers (no burn-in) and (b) after 200/500/1000 burn-in
steps, and report the finite/NaN/inf census at each point.

  (a) finite, (b) NaN count growing with burn-in steps -> confirms the
      Sherman-Morrison hypothesis; fix is periodic exact-inverse
      recomputation + a ratio-magnitude guard.
  (a) already NaN -> a different, non-burn-in-related math bug.

Usage:
    python -m pytc.utils.nan_census --n-water 25 --basis cc-pVTZ \
        --n-walkers 500 --burn-in-checkpoints 0,200,500,1000
"""
import argparse
import os
import time

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
from pytc.vmc.hamiltonian import eval_local_energy


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


def census(energies, label):
    energies = np.asarray(energies)
    n = energies.size
    n_nan = int(np.isnan(energies).sum())
    n_inf = int(np.isinf(energies).sum())
    finite = energies[np.isfinite(energies)]
    if finite.size > 0:
        fmin, fmax, fmean = finite.min(), finite.max(), finite.mean()
    else:
        fmin = fmax = fmean = float("nan")
    print(f"  [{label}] n_walkers={n}  n_nan={n_nan}  n_inf={n_inf}  "
          f"n_finite={finite.size}  finite_min={fmin:.6f}  "
          f"finite_max={fmax:.6f}  finite_mean={fmean:.6f}")
    return n_nan, n_inf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-water", type=int, default=25)
    p.add_argument("--basis", default="cc-pVTZ")
    p.add_argument("--n-walkers", type=int, default=500)
    p.add_argument("--burn-in-checkpoints", default="0,200,500,1000",
                    help="Comma-separated cumulative burn-in step counts to census at.")
    p.add_argument("--step-size", type=float, default=0.02)
    p.add_argument("--vmap-batch-size", type=int, default=256)
    p.add_argument("--scf-cache-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scf_cache"))
    args = p.parse_args()

    checkpoints = [int(x) for x in args.burn_in_checkpoints.split(",")]
    if checkpoints[0] != 0:
        checkpoints = [0] + checkpoints
    # Convert cumulative checkpoints to incremental burn-in step counts.
    increments = [checkpoints[0]] + [
        checkpoints[i] - checkpoints[i - 1] for i in range(1, len(checkpoints))
    ]

    mol, mf = build_system(args.n_water, args.basis, args.scf_cache_dir)
    det = SlaterDet.create(mol, mf.mo_coeff)
    bh = BoysHandyAnalytical.create(mol)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    jastrow = CompositeJastrow.create([ncusp, bh])
    jastrow_params = jastrow.init_params()
    sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
    linear_coeffs = jnp.ones(1)
    params = [jastrow_params, linear_coeffs]

    batch_eval_energy = jax.jit(jax.vmap(
        lambda w, p: eval_local_energy(sj_ansatz, w, p)[0], in_axes=(0, None)
    ))

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    walkers = initialize_walkers(det, args.n_walkers, None, subkey)

    cumulative_steps = 0
    for inc in increments:
        if inc > 0:
            t0 = time.time()
            walkers, _, key, _ = burn_in(
                sj_ansatz, walkers, n_steps=inc, step_size=args.step_size,
                key=key, params=params, max_vmap_batch_size=args.vmap_batch_size,
            )
            cumulative_steps += inc
            print(f"  (burn-in +{inc} steps -> {cumulative_steps} total, "
                  f"{time.time()-t0:.1f}s)")

        energies = batch_eval_energy(walkers, params)
        jax.block_until_ready(energies)
        census(energies, f"burn_in_steps={cumulative_steps}")


if __name__ == "__main__":
    main()
