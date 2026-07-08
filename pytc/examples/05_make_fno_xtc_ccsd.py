"""05: FNO (frozen/truncated natural orbital) active-space scan, xTC-CCSD.

H2O/cc-pVQZ (a larger basis than 01-04's cc-pVDZ, so virtual-space
truncation actually matters) with a plain REXP Jastrow (no VMC-optimized
parameters needed here -- REXP's fixed alpha is fine for a deterministic
ISDF-xTC-CCSD calculation; see 01's docstring for why REXP is VMC-unstable
but fine here). Uses the ISDF-approximated xTC-CCSD path: the DENSE
construction (03) is O(N_grid^2) in the real-space quadrature -- already
impractical at H2O/cc-pVDZ's ~22k grid points (04 demonstrates this exact
contrast), and cc-pVQZ's ~115 orbitals makes dense completely infeasible on
a laptop. ISDF (n_rank = 15*n_orb, the production setting) is what makes
this scan tractable; the n_keep convergence reported below therefore
includes a small, fixed ISDF approximation error alongside the FNO
truncation error (04 already isolates the pure ISDF error separately).

Mirrors `isdf-data/hchain/scripts/isdf_xtc_fno.py`'s FNO-scan pattern at
H2O/cc-pVQZ scale instead of H-chain/cc-pV5Z scale.

Run: python 05_make_fno_xtc_ccsd.py   (~5-10 min -- CCSD cost grows
steeply with n_orb, so the nkeep=40 point dominates the runtime)
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import REXP
from pytc.solver import jax_xtc_ccsd
from pytc.fno import make_fno_mo_coeff

JASTROW_PARAMS = {"alpha": np.array([0.5])}
# The full cc-pVQZ virtual space (n_orb=115) makes CCSD itself too slow for
# a quickstart example (CCSD cost grows steeply with n_orb); nkeep=40 stands
# in as the "largest active space" reference instead of the untruncated
# full space or a much larger nkeep.
N_KEEP_LIST = [10, 20, 30, 40]


def run_one(mf, jastrow, n_keep):
    fno = make_fno_mo_coeff(mf, n_keep=n_keep)
    mf_run = fno.mf

    xtc_obj = xtc.XTC.from_pyscf(mf_run, jastrow, grid_lvl=2)
    n_rank = xtc_obj.n_orb * 15  # production ISDF-rank setting
    isdf_xtc = xtc.ISDFXTC.from_xtc(xtc_obj, n_rank=n_rank)
    isdf_xtc = isdf_xtc.isdf(JASTROW_PARAMS)

    eris = isdf_xtc.make_eris(mf_run, JASTROW_PARAMS)
    mycc = jax_xtc_ccsd.RCCSD(mf_run, isdf_xtc, JASTROW_PARAMS)
    e_corr, t1, t2 = mycc.kernel(eris=eris)

    no = mycc.nocc
    e_hf = 2 * np.einsum("ii->", eris.fock[:no, :no])
    e_hf -= 2 * np.einsum("iijj->", eris.oooo) - np.einsum("ijji->", eris.oooo)
    e_hf += eris.e_core
    return e_hf + e_corr, mf_run.mo_coeff.shape[1]


def main():
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
                basis="cc-pVQZ", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"Reference HF energy: {mf.e_tot:.6f}, n_orb={mol.nao}")

    jastrow = REXP()

    results = []
    for n_keep in N_KEEP_LIST:
        label = f"nkeep={n_keep}"
        e_tot, n_orb = run_one(mf, jastrow, n_keep)
        results.append((label, n_orb, e_tot))
        print(f"{label:>10s}  n_orb={n_orb:3d}  E_tot={e_tot:.6f}")

    e_ref = results[-1][2]  # largest active space (nkeep=40) as reference
    print(f"\n{'label':>10s} {'n_orb':>6s} {'E_tot':>14s} {'dE vs nkeep=40 (mHa)':>22s}")
    for label, n_orb, e_tot in results:
        d_mha = (e_tot - e_ref) * 1000
        print(f"{label:>10s} {n_orb:6d} {e_tot:14.6f} {d_mha:22.4f}")

    # Self-check: successive n_keep steps should get monotonically closer to
    # the largest active space (a convergence trend, not an absolute
    # tolerance -- virtual-space truncation converges slowly, so nkeep=30
    # need not be within chemical accuracy of nkeep=40 for this to be
    # working correctly).
    devs = [abs(e_tot - e_ref) for _, _, e_tot in results[:-1]]
    assert devs == sorted(devs, reverse=True), (
        f"Deviation from the nkeep=40 reference should shrink monotonically "
        f"as n_keep grows, got {devs}"
    )
    print("OK: FNO deviation from the nkeep=40 reference shrinks monotonically.")


if __name__ == "__main__":
    main()
