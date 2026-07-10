"""Profiling utility: optimize_ref_var per-phase timing + peak memory.

Runs a small BoysHandy+NuclearCusp VMC optimization on an H-chain and
profiles it with cProfile (function-level cumulative-time breakdown) plus
peak RSS (resource module). CPU-only; useful as a quick qualitative check
of where wall time goes, but absolute numbers don't transfer to GPU -- see
the efficiency-refactor initiative (task #1, #pro-pytc-efficiency-refactor)
for the real GPU baseline.

Edit N_ATOMS/BASIS/N_WALKERS/BURN_IN/N_OPT_STEPS below to change scale.
"""
import cProfile
import pstats
import io
import resource
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from pyscf import gto, scf

from pytc.vmc import optimize_ref_var
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import BoysHandy, NuclearCusp, CompositeJastrow


def h_chain(n, sep=1.8):
    return "; ".join(f"H 0 0 {i*sep}" for i in range(n))


def main():
    N_ATOMS = 10  # H10 = 10 electrons
    BASIS = "cc-pVDZ"
    N_WALKERS = 1000
    BURN_IN = 200
    N_OPT_STEPS = 3

    print(f"System: H{N_ATOMS}, basis={BASIS}, n_walkers={N_WALKERS}, "
          f"burn_in={BURN_IN}, n_opt_steps={N_OPT_STEPS}")

    t0 = time.time()
    mol = gto.M(atom=h_chain(N_ATOMS), basis=BASIS, unit="Bohr", verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    print(f"SCF done ({time.time()-t0:.1f}s), n_orb={mol.nao}, "
          f"n_elec={mol.nelectron}")

    det = SlaterDet.create(mol, mf.mo_coeff)
    bh = BoysHandy.create(mol)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    jastrow = CompositeJastrow.create([ncusp, bh])
    jastrow_params = jastrow.init_params()
    sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
    linear_coeffs = jnp.ones(1)

    key = random.PRNGKey(43)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3  # GB on macOS (bytes->GB)

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.time()
    result = optimize_ref_var(
        sj_ansatz,
        params=[jastrow_params, linear_coeffs],
        n_walkers=N_WALKERS,
        n_opt_steps=N_OPT_STEPS,
        burn_in_steps=BURN_IN,
        step_size=0.02,
        optimizer_type="newton",
        learning_rate=0.1,
        opt_kwargs={"damping": 1e-6, "solver": "exact"},
        key=key,
    )
    total_time = time.time() - t1
    pr.disable()

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    print(f"\nTotal optimize_ref_var wall time: {total_time:.1f}s")
    print(f"Peak RSS: {rss_after:.2f} GB (delta from before: "
          f"{rss_after - rss_before:.2f} GB)")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    print("\n=== cProfile top 40 by cumulative time ===")
    print(s.getvalue())

    # Save full stats for later inspection
    pr.dump_stats("/tmp/vmc_profile_h20.pstats")
    print("Saved full profile to /tmp/vmc_profile_h20.pstats")


if __name__ == "__main__":
    main()
