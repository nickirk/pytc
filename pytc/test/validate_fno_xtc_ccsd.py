"""Task #10 validation: full vs FNO (MP2 natural-orbital) xTC-CCSD.

Runs the existing XTC -> RCCSD pipeline on the full mean-field and on a
sequence of FNO-truncated mean-fields, checking that (a) the pipeline runs
with n_orb < n_ao and (b) the FNO energy converges to the full energy as
n_keep -> nvir (CCSD is invariant under the occupied-preserving virtual
rotation, so the all-virtual limit must reproduce the full energy to
solver tolerance).

Usage:
    python validate_fno_xtc_ccsd.py <n> <basis> <grid_lvl> <csv n_keeps> [alpha] [use_isdf]
e.g.   python validate_fno_xtc_ccsd.py 6 cc-pvdz 0 2,4,999
(n_keep larger than nvir is clamped to nvir == "all virtuals" limit.)
"""
import sys
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pyscf import gto, scf
from pytc.xtc import XTC, ISDFXTC
from pytc.jastrow.rexp import REXP
from pytc.solver import xtc_ccsd
from pytc.fno import make_fno_mo_coeff


def make_h_chain(n, dist=1.4):
    return "; ".join(f"H 0 0 {i*dist}" for i in range(n))


def run_ccsd(mf, jastrow, params, grid_lvl, use_isdf):
    xtc_obj = XTC.from_pyscf(mf, jastrow, grid_lvl=grid_lvl)
    if use_isdf:
        n_rank = xtc_obj.n_orb * 12
        xtc_obj = ISDFXTC.from_xtc(xtc_obj, n_rank=n_rank)
        xtc_obj = xtc_obj.isdf(params)  # preload ISDF kernels (avoids on-the-fly recompute in ao2mo)
    cc = xtc_ccsd.RCCSD(mf, xtc_obj, params)
    cc.kernel()
    return float(cc.e_tot), int(xtc_obj.n_orb)


def main():
    n = int(sys.argv[1])
    basis = sys.argv[2]
    grid_lvl = int(sys.argv[3])
    n_keeps = [int(x) for x in sys.argv[4].split(",")]
    alpha = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    use_isdf = (len(sys.argv) > 6) and (sys.argv[6].lower() in ("1", "isdf", "true"))

    mol = gto.M(atom=make_h_chain(n), basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    nocc = int(np.sum(mf.mo_occ > 0))
    nmo = mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    print(f"=== H{n}/{basis} grid_lvl={grid_lvl} alpha={alpha} isdf={use_isdf} ===",
          flush=True)
    print(f"n_ao={mol.nao_nr()} nocc={nocc} nvir={nvir} nmo={nmo}", flush=True)

    jastrow = REXP()
    params = {"alpha": jnp.array([alpha])}

    print("  [full] building xTC-CCSD ...", flush=True)
    e_full, n_orb_full = run_ccsd(mf, jastrow, params, grid_lvl, use_isdf)
    print(f"  [full] E_tot={e_full:.10f}  n_orb={n_orb_full}", flush=True)

    print(f"\n{'n_keep':>7} {'n_orb':>6} {'E_tot':>16} {'dE vs full':>14} {'n_orb<n_ao':>11}", flush=True)
    print("-" * 60, flush=True)
    last_dE = None
    for k in n_keeps:
        k_eff = min(k, nvir)
        r = make_fno_mo_coeff(mf, n_keep=k_eff)
        e_fno, n_orb_fno = run_ccsd(r.mf, jastrow, params, grid_lvl, use_isdf)
        dE = e_fno - e_full
        last_dE = dE
        flag = "OK" if n_orb_fno < mol.nao_nr() else "FULL"
        print(f"{k_eff:>7} {n_orb_fno:>6} {e_fno:>16.10f} {dE:>+14.2e} {flag:>11}", flush=True)

    # All-virtuals limit must reproduce the full energy (virtual-rotation invariant).
    if nvir in [min(k, nvir) for k in n_keeps]:
        print(f"\nall-virtual limit |dE| = {abs(last_dE):.2e} (expect ~solver tol)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
