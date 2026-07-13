"""06: Deterministic (non-stochastic) Jastrow optimization via 2nd quantization.

DEFERRED / NOT COMMITTED: `pytc.optimize.optimize_jastrow` is currently
broken on main (confirmed by running `python3 -m pytc.optimize` directly) --
see task #49 in #research-pytc. `XTC.get_2b()` materializes a host numpy
copy mid-computation (`xtc.py:715`), which breaks `jax.value_and_grad`
through `get_1b`/`get_2b` regardless of system/Jastrow. Kept locally only,
per Ke's instruction (2026-07-08), until that's fixed.

Alternative to 01's VMC/reference-variance route: rather than sampling
walkers, minimize the ov-block residual ||f_ia||^2 + ||(2V-V^T)_iajb||^2 of
the transcorrelated Hamiltonian directly (JAX autodiff + optax), which
drives the occupied-virtual coupling of the effective one-body/two-body
integrals toward zero -- the same target a Brueckner-orbital-style
optimization aims for, without any MCMC noise. Uses REXP (fine here: this
optimizer never samples walkers near a nucleus, so REXP's missing cusp
isn't destabilizing the way it is in 01's VMC route).

H2O/cc-pVDZ, matching 01-05. Ends with an xTC-CCSD check using the optimized
Jastrow, mirroring `pytc/optimize.py`'s existing He/REXP `do_ccsd` pattern.

Run: python 06_deterministic_2nd_quantized_jastrow_optimization.py   (~20s,
once the underlying bug is fixed)
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from pyscf import gto, scf

from pytc.jastrow import REXP
from pytc.integrals.xtc import XTC
from pytc.optimize import optimize_jastrow
from pytc.solver import jax_xtc_ccsd

EXPECTED_ALPHA = 0.4  # placeholder, see note in main()
ALPHA_ATOL = 0.2
EXPECTED_E_TOT = -76.24  # placeholder, see note in main()
E_TOT_ATOL = 5e-3


def main():
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
                basis="cc-pVDZ", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"Reference HF energy: {mf.e_tot:.6f}")

    jastrow = REXP()
    init_params = jastrow.init_params()
    xtc_obj = XTC.from_pyscf(mf, jastrow, grid_lvl=2)

    print("\nOptimizing Jastrow via 2nd-quantized residual minimization...")
    optimized_params = optimize_jastrow(
        xtc_obj, mf, init_params,
        optimizer_name="rmsprop", learning_rate=1e-2, n_steps=20,
    )
    print(f"\nOptimized alpha: {optimized_params['alpha'][0]:.6f}")
    assert np.isclose(optimized_params["alpha"][0], EXPECTED_ALPHA,
                       atol=ALPHA_ATOL), "optimized alpha outside expected range"

    print("\nRunning xTC-CCSD with the optimized Jastrow...")
    eris = xtc_obj.make_eris(mf, optimized_params)
    mycc = jax_xtc_ccsd.RCCSD(mf, xtc_obj, optimized_params)
    e_corr, t1, t2 = mycc.kernel(eris=eris)

    no = mycc.nocc
    e_hf = 2 * np.einsum("ii->", eris.fock[:no, :no])
    e_hf -= 2 * np.einsum("iijj->", eris.oooo) - np.einsum("ijji->", eris.oooo)
    e_hf += eris.e_core
    e_tot = e_hf + e_corr
    print(f"xTC-CCSD E_tot: {e_tot:.6f}")

    assert np.isclose(e_tot, EXPECTED_E_TOT, atol=E_TOT_ATOL), (
        f"E_tot {e_tot:.6f} deviates from expected {EXPECTED_E_TOT:.6f}"
    )
    print(f"OK: E_tot matches expected value to {E_TOT_ATOL} Ha.")


if __name__ == "__main__":
    main()
