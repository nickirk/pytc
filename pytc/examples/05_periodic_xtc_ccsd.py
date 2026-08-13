"""Runnable Gamma periodic FFT-ISDF xTC-RCCSD pipeline.

This correctness fixture is intentionally small.  It exercises every public
stage from periodic RHF orbitals through the factor-direct RCCSD energy while
printing machine-readable stage timings; it is not a production wall-time
benchmark.

Run from the repository root with
``python -m pytc.examples.05_periodic_xtc_ccsd``.
"""

import argparse
import json
import time

import numpy as np
from pyscf.pbc import gto, scf

from pytc.pbc.jastrow import BoysHandy
from pytc.pbc.xtc import create_isdf_xtc_fft
from pytc.solver import isdf_xtc_ccsd


def _timed(timings, name, fn):
    start = time.perf_counter()
    result = fn()
    timings[name] = time.perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scf-mesh", type=int, default=15)
    parser.add_argument("--xtc-mesh", type=int, default=2)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--max-cycle", type=int, default=50)
    args = parser.parse_args()

    timings = {}
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.4"
    cell.basis = "sto-3g"
    cell.a = np.eye(3) * 8.0
    cell.unit = "B"
    cell.cart = True
    cell.mesh = [args.scf_mesh] * 3
    cell.verbose = 0
    _timed(timings, "cell_build_s", cell.build)

    mf = scf.RHF(cell)
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    _timed(timings, "rhf_s", mf.kernel)
    if not mf.converged:
        raise RuntimeError("periodic RHF did not converge")

    jastrow = BoysHandy.create(cell)
    params = jastrow.init_params()
    mesh = (args.xtc_mesh,) * 3
    xtc_obj = _timed(
        timings,
        "fft_isdf_build_s",
        lambda: create_isdf_xtc_fft(
            mf,
            jastrow,
            params,
            mesh=mesh,
            n_rank=args.rank,
            is_incore=True,
        ),
    )

    cc = isdf_xtc_ccsd.RCCSD(
        mf,
        xtc_obj,
        params,
        on_the_fly_vvvv=True,
        max_memory=2000,
        gpu_max_memory=512,
    )
    cc.conv_tol = 1e-9
    cc.max_cycle = args.max_cycle
    eris = _timed(timings, "eris_build_s", cc.ao2mo)
    _timed(timings, "rccsd_s", lambda: cc.kernel(eris=eris))
    if not cc.converged:
        raise RuntimeError("factor-direct xTC-RCCSD did not converge")

    print(
        json.dumps(
            {
                "system": "Gamma H2/sto-3g, 8 bohr cubic cell",
                "scf_mesh": list(cell.mesh),
                "xtc_mesh": list(mesh),
                "n_ao": int(cell.nao_nr()),
                "n_rank": int(xtc_obj.phi_isdf.shape[1]),
                "e_rhf": float(mf.e_tot),
                "e_corr": float(cc.e_corr),
                "e_xtc_ccsd": float(cc.e_tot),
                "converged": bool(cc.converged),
                "timings": timings,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
