"""02: Load a VMC optimization history and average the Jastrow parameters.

The manuscript's production runs use Polyak-Ruppert averaging: rather than
taking the very last optimization step's (noisy) parameters, average the
last N steps of the phase-B trajectory to cancel out stochastic VMC noise.
This mirrors `load_averaged_jastrow_params()` in
`isdf-data/hchain/scripts/isdf_xtc_fno.py` (production uses the last 500 of
2000 phase-B steps; here we average the last few of our much shorter run).

Requires `h2o_phase_b_hist.h5` from 01_vmc_optimize_jastrow.py to exist in
the working directory.

The printed averaged parameters are what 03/04/05 hardcode directly (rather
than re-loading this HDF5 file at runtime, so those examples run standalone).

Run: python 02_load_and_average_jastrow_params.py
"""
import sys

import jax
import numpy as np

from pytc.vmc.mcmc_utils import load_optimization_history

HIST_PATH = "h2o_phase_b_hist.h5"
AVERAGE_LAST_N = 10  # production uses 500 of 2000 steps; we ran only 20


def main():
    try:
        history = load_optimization_history(HIST_PATH)
    except FileNotFoundError:
        print(f"'{HIST_PATH}' not found -- run 01_vmc_optimize_jastrow.py first.")
        sys.exit(1)

    stacked_params = history["params"]  # PyTree, each leaf stacked along axis 0
    n_saved = jax.tree_util.tree_leaves(stacked_params)[0].shape[0]
    n_use = min(AVERAGE_LAST_N, n_saved)
    print(f"Averaging the last {n_use}/{n_saved} phase-B steps...")

    averaged = jax.tree_util.tree_map(
        lambda x: np.mean(x[-n_use:], axis=0), stacked_params
    )
    jastrow_params, linear_coeffs = averaged

    print("\nAveraged Jastrow parameters (CompositeJastrow([NuclearCusp, BoysHandy])):")
    print(f"  {jastrow_params!r}")
    print(f"\nLinear coefficients: {linear_coeffs!r}")
    print(
        "\nCopy the dict above into 03_dense_xtc_ccsd.py / "
        "04_isdf_xtc_ccsd.py / 05_make_fno_xtc_ccsd.py's JASTROW_PARAMS."
    )


if __name__ == "__main__":
    main()
