"""CBD automerization via XTC-EOM-CCSD.

Demonstrates how to use ``xtc_eom_ccsd.EOMEE`` to find the 1¹B₁g
singlet state of cyclobutadiene at the D₄ₕ geometry as a "de-excitation"
from a closed-shell 1¹A_g RHF reference. The corrected D₄ₕ ground-state
energy is then::

    E(1¹B₁g, D₄ₕ) = E_CC(closed-shell 1¹A_g, D₄ₕ) + ω_EOM

where ``ω_EOM`` is the most negative EOM eigenvalue.

This is what allows direct comparison to Loos 2022's CCSDTQ TBE
(8.93 kcal/mol) and to FermiNet-DMC (~10 kcal/mol), neither of which
suffers from the closed-shell-reference bias that gives ground-state
CCSD a ~21 kcal/mol gap on its own.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import rexp
from pytc.solver import jax_xtc_ccsd
from pytc.solver.xtc_eom_ccsd import EOMEE

jax.config.update("jax_enable_x64", True)

HA2KCAL = 627.5094740631

# Geometries (Bhaskaran-Nair MR-BWCCSD(T)/cc-pVTZ, same as FermiNet repo)
CBD_D2H = """
C    0.0000000000    0.0000000000    0.0000000000
C    1.5640000000    0.0000000000    0.0000000000
C    1.5640000000    1.3540000000    0.0000000000
C    0.0000000000    1.3540000000    0.0000000000
H   -0.7621688203   -0.7637667769    0.0000000000
H    2.3261688203   -0.7637667769    0.0000000000
H    2.3261688203    2.1177667769    0.0000000000
H   -0.7621688203    2.1177667769    0.0000000000
"""

CBD_D4H = """
C    0.0000000000    0.0000000000    0.0000000000
C    1.4510000000    0.0000000000    0.0000000000
C    1.4510000000    1.4510000000    0.0000000000
C    0.0000000000    1.4510000000    0.0000000000
H   -0.7622611101   -0.7622611101    0.0000000000
H    2.2132611101   -0.7622611101    0.0000000000
H    2.2132611101    2.2132611101    0.0000000000
H   -0.7622611101    2.2132611101    0.0000000000
"""


def run(geom, basis, jastrow_params):
    print(f"\n{'='*72}\n  {geom['name']}  /  {basis}\n{'='*72}", flush=True)
    mol = gto.M(atom=geom["xyz"], basis=basis, unit="Angstrom",
                verbose=4, max_memory=20000)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    print(f"  RHF total: {mf.e_tot:.10f}", flush=True)

    jastrow = rexp.REXP()
    my_xtc = xtc.XTC.from_pyscf(mf, jastrow)

    cc = jax_xtc_ccsd.RCCSD(mf, my_xtc, jastrow_params)
    cc.kernel()
    print(f"  XTC-CCSD total: {cc.e_tot:.10f}  E_corr: {cc.e_corr:.10f}",
          flush=True)

    if geom["needs_eom"]:
        eom = EOMEE(cc)
        e, _ = eom.kernel(nroots=4, koopmans=False)  # also seed r2 for doubles
        print(f"  EOM-EE excitation energies (Ha):")
        for k, w in enumerate(np.sort(e)):
            tag = "  <-- de-excitation (1¹B₁g candidate)" if w < 0 else ""
            print(f"    root {k}: ω = {w:+.6f} Ha = {w*HA2KCAL:+.3f} kcal/mol{tag}",
                  flush=True)
        omega_b1g = float(np.min(e))   # most negative root
        e_b1g = cc.e_tot + omega_b1g
        print(f"\n  E(1¹B₁g, D₄ₕ) = E_CC + ω = {e_b1g:.10f}", flush=True)
        return e_b1g
    return cc.e_tot


def main():
    basis = "aug-cc-pvdz"
    # Placeholder Jastrow parameters — for a real run, load from VMC history
    # (see pytc.vmc.mcmc_utils.load_optimization_history).
    jastrow_params = {"alpha": jnp.array([0.5])}

    e_ground = run({"name": "D2h_ground", "xyz": CBD_D2H, "needs_eom": False},
                   basis, jastrow_params)
    e_trans = run({"name": "D4h_trans (via EOM)", "xyz": CBD_D4H, "needs_eom": True},
                  basis, jastrow_params)

    gap = (e_trans - e_ground) * HA2KCAL
    print(f"\n\n{'='*72}\n  Automerization barrier (XTC-EOM-CCSD): {gap:.3f} kcal/mol")
    print(f"  Reference:  Loos CCSDTQ TBE       = 8.93 kcal/mol")
    print(f"              FermiNet-DMC          = ~9.98 kcal/mol")
    print(f"              ground-state CCSD     = ~21 (overshoots due to closed-shell bias)")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
