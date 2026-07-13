"""04: ISDF-approximated xTC-CCSD for H2O/cc-pVDZ, vs. the dense reference.

Same system and Jastrow (REXP, alpha=0.4) as 03_dense_xtc_ccsd.py, but
builds the transcorrelated two-body integrals via the Interpolative
Separable Density Fitting (ISDF) approximation (`ISDFXTC`) instead of the
exact dense grid contraction. At the production rank setting (15x n_orb),
ISDF matches the dense E_tot to <0.01 mHa while running several times
faster (48s vs 03's ~150s here) -- and the gap widens dramatically for
larger/more complex systems, which is the whole point of ISDF (see 03's
docstring for why dense mode doesn't scale).

Run: python 04_isdf_xtc_ccsd.py   (~1 min)
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from pyscf import gto, scf

from pytc.integrals import xtc
from pytc.jastrow import REXP
from pytc.solver import jax_xtc_ccsd

JASTROW_PARAMS = {"alpha": np.array([0.4])}

# 03's dense E_tot -- the reference this example's ISDF result is checked
# against (see 03_dense_xtc_ccsd.py's EXPECTED_E_TOT for provenance).
DENSE_E_TOT = -77.066141
ISDF_ERROR_ATOL_MHA = 0.5  # ISDF error should be well under chemical accuracy


def main():
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
                basis="cc-pVDZ", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    jastrow = REXP()
    xtc_obj = xtc.XTC.from_pyscf(mf, jastrow, grid_lvl=2)
    n_rank = xtc_obj.n_orb * 15  # production ISDF-rank setting
    isdf_xtc = xtc.ISDFXTC.from_xtc(xtc_obj, n_rank=n_rank)
    isdf_xtc = isdf_xtc.isdf(JASTROW_PARAMS)
    print(f"ISDF rank: {n_rank} (n_orb={xtc_obj.n_orb})")

    eris = isdf_xtc.make_eris(mf, JASTROW_PARAMS)
    mycc = jax_xtc_ccsd.RCCSD(mf, isdf_xtc, JASTROW_PARAMS)
    e_corr, t1, t2 = mycc.kernel(eris=eris)

    no = mycc.nocc
    e_hf = 2 * np.einsum("ii->", eris.fock[:no, :no])
    e_hf -= 2 * np.einsum("iijj->", eris.oooo) - np.einsum("ijji->", eris.oooo)
    e_hf += eris.e_core
    e_tot = e_hf + e_corr

    error_mha = (e_tot - DENSE_E_TOT) * 1000
    print(f"ISDF xTC-CCSD E_tot: {e_tot:.6f}")
    print(f"Dense  xTC-CCSD E_tot (03): {DENSE_E_TOT:.6f}")
    print(f"ISDF error: {error_mha:+.4f} mHa")

    assert abs(error_mha) < ISDF_ERROR_ATOL_MHA, (
        f"ISDF error {error_mha:.4f} mHa exceeds {ISDF_ERROR_ATOL_MHA} mHa "
        "tolerance -- unexpected for this rank"
    )
    print(f"OK: ISDF error is within {ISDF_ERROR_ATOL_MHA} mHa of dense.")


if __name__ == "__main__":
    main()
