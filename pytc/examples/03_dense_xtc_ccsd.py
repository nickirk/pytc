"""03: Dense (non-ISDF) xTC-CCSD for H2O/cc-pVDZ.

Uses a plain REXP Jastrow (alpha=0.4) -- NOT the BoysHandy+NuclearCusp
factor optimized in 01/02. Empirically, the dense (real-space quadrature)
path costs ~O(N_grid^2) in the pairwise Jastrow evaluation (this is exactly
the cost ISDF exists to eliminate -- see 04), and BoysHandy+NuclearCusp's
richer per-point polynomial terms make that quadrature impractically slow
off-cluster even at this small a grid (21,952 points for H2O/cc-pVDZ).
Plain REXP's much cheaper per-point cost keeps dense mode here to ~2-3
minutes -- still slow, but that slowness *is* the point: 04 solves the
identical system with the ISDF approximation in a fraction of the time,
matching this dense reference to <0.01 mHa.

Run: python 03_dense_xtc_ccsd.py   (~2-3 min)
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import REXP
from pytc.solver import jax_xtc_ccsd

JASTROW_PARAMS = {"alpha": np.array([0.4])}

# Expected E_tot for this exact system/params (self-check tolerance).
EXPECTED_E_TOT = -77.066141
E_TOT_ATOL = 1e-4


def main():
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
                basis="cc-pVDZ", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"Reference HF energy: {mf.e_tot:.6f}")

    jastrow = REXP()
    xtc_obj = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=2)
    eris = xtc_obj.make_eris(mf, JASTROW_PARAMS)

    mycc = jax_xtc_ccsd.RCCSD(mf, xtc_obj, JASTROW_PARAMS)
    e_corr, t1, t2 = mycc.kernel(eris=eris)

    no = mycc.nocc
    e_hf = 2 * np.einsum("ii->", eris.fock[:no, :no])
    e_hf -= 2 * np.einsum("iijj->", eris.oooo) - np.einsum("ijji->", eris.oooo)
    e_hf += eris.e_core
    e_tot = e_hf + e_corr

    print(f"xTC HF energy:   {e_hf:.6f}")
    print(f"xTC-CCSD E_corr: {e_corr:.6f}")
    print(f"xTC-CCSD E_tot:  {e_tot:.6f}")

    assert np.isclose(e_tot, EXPECTED_E_TOT, atol=E_TOT_ATOL), (
        f"E_tot {e_tot:.6f} deviates from expected {EXPECTED_E_TOT:.6f} "
        f"by more than {E_TOT_ATOL} Ha"
    )
    print(f"OK: E_tot matches expected value to {E_TOT_ATOL} Ha.")


if __name__ == "__main__":
    main()
