"""01: Optimize a Jastrow factor by variational Monte Carlo (VMC).

Two-phase reference-variance optimization (phase A: coarse; phase B:
refined, with the parameter trajectory saved to HDF5) of a BoysHandy +
NuclearCusp Jastrow factor for H2O/cc-pVDZ. This is the same
`optimize_ref_var` two-phase pattern used for the manuscript's production
H-chain and benzene runs (see `isdf-data/hchain/scripts/run_opt.py`), scaled
down to run in a few minutes on a laptop CPU rather than hours on a cluster.

Note: REXP (the simplest Jastrow, used in 03-06) has no electron-nucleus
cusp, and empirically diverges to NaN under this reference-variance VMC
optimizer -- BoysHandy+NuclearCusp (the same combination the production
scripts use) is what's actually being optimized here.

The saved phase-B history (`h2o_phase_b_hist.h5`, NOT committed to git --
see .gitignore) is consumed by 02_load_and_average_jastrow_params.py, which
prints the averaged parameters to hardcode into 03-05.

Run: python 01_vmc_optimize_jastrow.py   (~4 minutes)
"""
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from pyscf import gto, scf

from pytc.vmc import optimize_ref_var
from pytc.ansatz.sj import SlaterJastrow
from pytc.ansatz.det import SlaterDet
from pytc.jastrow import NuclearCusp, CompositeJastrow
from pytc.jastrow.bha import BoysHandyAnalytical


def main():
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
                basis="cc-pVDZ", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"Reference HF energy: {mf.e_tot:.6f}")

    det = SlaterDet.create(mol, mf.mo_coeff)
    # BoysHandy + NuclearCusp (not plain REXP): REXP alone has no
    # electron-nucleus cusp, and empirically the reference-variance Newton
    # optimizer diverges to NaN on H2O with it. 03-06's deterministic
    # (non-sampling) xTC-CCSD calculations use plain REXP just fine -- the
    # instability is specific to VMC sampling near the nuclei.
    # BoysHandyAnalytical: hand-written analytic derivatives, faster than
    # the generic BoysHandy's folx-autodiff path. Construct it directly --
    # BoysHandy.create() no longer implicitly substitutes it.
    bh = BoysHandyAnalytical.create(mol)
    ncusp = NuclearCusp.create(mol, name="ncusp")
    jastrow = CompositeJastrow.create([ncusp, bh])
    jastrow_params = jastrow.init_params()
    sj_ansatz = SlaterJastrow.create(mol, jastrow, [det])
    linear_coeffs = jnp.ones(1)  # single determinant

    # Conservative MCMC step size / learning rate (same order as the
    # production H-chain/benzene settings) -- an overly large step_size or
    # learning_rate here destabilizes the reference-variance optimizer.
    key = random.PRNGKey(43)

    print("\nPhase A: coarse optimization...")
    t0 = time.time()
    phase_a = optimize_ref_var(
        sj_ansatz,
        params=[jastrow_params, linear_coeffs],
        n_walkers=1000,
        n_opt_steps=10,
        burn_in_steps=2000,
        step_size=0.02,
        optimizer_type="newton",
        learning_rate=0.1,
        opt_kwargs={"damping": 1e-6, "solver": "exact"},
        key=key,
    )
    print(f"Phase A done in {time.time() - t0:.1f}s, "
          f"final params: {phase_a['params'][-1][0]}")

    print("\nPhase B: refine + save parameter trajectory...")
    t0 = time.time()
    phase_b_params = phase_a["params"][-1]
    phase_b = optimize_ref_var(
        sj_ansatz,
        params=phase_b_params,
        n_walkers=1000,
        n_opt_steps=20,
        burn_in_steps=2000,
        step_size=0.02,
        optimizer_type="newton",
        learning_rate=0.1,
        opt_kwargs={"damping": 1e-6, "solver": "exact"},
        key=key,
        save_path="h2o_phase_b_hist.h5",
        save_frequency=1,
    )
    print(f"Phase B done in {time.time() - t0:.1f}s")
    print(f"Final Jastrow params: {phase_b['params'][-1][0]}")
    print("Saved phase-B history to h2o_phase_b_hist.h5 "
          "(consumed by 02_load_and_average_jastrow_params.py)")


if __name__ == "__main__":
    main()
