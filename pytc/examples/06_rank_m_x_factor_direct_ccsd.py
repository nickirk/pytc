"""06: Opt-in rank-M X with factor-direct ISDF xTC-CCSD.

This small H4/STO-3G calculation demonstrates the API rather than a
production accuracy benchmark.  Passing ``n_factor=M`` to ``ISDFXTC.isdf``
replaces dense ``X[r,s,c]`` with the orbital Tucker factors
``X_tucker = {U[r,a], Z[a,b,c]}``, where ``a,b < M``.  The dedicated
factor-direct ISDF CCSD solver contracts T2 directly with U/Z; dense X, a
four-virtual tile, and the full VVVV tensor are never materialized.

Rank-M X remains opt-in.  Its current numerical validation is limited to the
H10/M=80 study reported for PyTC 0.2.1, so choose M and validate energies for
each new production system.

Run: python 06_rank_m_x_factor_direct_ccsd.py   (~20 s on a laptop CPU)
"""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from pyscf import gto, scf

from pytc import xtc
from pytc.jastrow import REXP
from pytc.solver import isdf_xtc_ccsd


JASTROW_PARAMS = {"alpha": np.array([0.4])}
N_FACTOR = 2


def main():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    # The factor-direct solver combines ISDF xTC terms with the ordinary
    # Coulomb contribution through the density-fitting factors.
    mf = scf.RHF(mol).density_fit().run()
    assert mf.converged

    xtc_obj = xtc.XTC.from_pyscf(mf, REXP(), grid_lvl=0)
    n_rank = max(8, 3 * xtc_obj.n_orb)
    isdf_xtc = xtc.ISDFXTC.from_xtc(
        xtc_obj, n_rank=n_rank, is_incore=True,
    )

    # n_factor is the explicit opt-in.  None (the default) builds dense X.
    isdf_xtc = isdf_xtc.isdf(
        JASTROW_PARAMS,
        n_factor=N_FACTOR,
        batch_size=64,
        orb_block_size=2,
        host_grid_block_size=512,
    )
    kernels = isdf_xtc.isdf_kernels
    assert "X" not in kernels
    assert "X_tucker" in kernels
    u = kernels["X_tucker"]["U"]
    z = kernels["X_tucker"]["Z"]
    assert u.shape == (xtc_obj.n_orb, N_FACTOR)
    assert z.shape == (N_FACTOR, N_FACTOR, isdf_xtc.phi_isdf.shape[1])
    print(f"factor-only X: U{u.shape}, Z{z.shape}; dense X absent")

    # This is the dedicated factor-direct solver, not jax_xtc_ccsd.RCCSD's
    # bounded-tile path.  Its VVVV hook contracts T2 directly with K1/K2/K3,
    # D, and the rank-M U/Z factors without constructing a four-virtual tile.
    mycc = isdf_xtc_ccsd.RCCSD(
        mf,
        isdf_xtc,
        JASTROW_PARAMS,
        on_the_fly_vvvv=True,
    )
    eris = mycc.ao2mo()
    try:
        e_corr, _, _ = mycc.kernel(eris=eris)
    finally:
        eris.close()

    assert mycc.converged
    print(f"rank-{N_FACTOR} X factor-direct xTC-CCSD E_corr: {e_corr:.10f}")
    print("OK: factor-only ISDF build and direct T2-U-Z CCSD completed.")


if __name__ == "__main__":
    main()
