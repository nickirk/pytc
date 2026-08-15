"""06: Controlled hierarchical L_aux construction on H2/STO-3G.

This small, deterministic control compares the standard L_aux/H_aux pair
construction with the experimental geometry-hierarchical residual mode.  The
NuclearCusp contribution is recovered analytically; near cluster pairs remain
direct, while admissible far blocks may use a sampled CUR representation.

Run an exact algebra control (the default):

    JAX_ENABLE_X64=1 python 06_hmatrix_laux_h2.py

Set ``--tolerance`` to a positive value to explore the approximation.  That
does not establish a production chemistry tolerance: compare relaxed energies
and same-hardware timings before using it beyond this H2 demonstration.
"""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

from pytc.jastrow import BoysHandy, CompositeJastrow, NuclearCusp
from pytc.tc import ISDFTC, TC


def relative_error(actual, reference) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(np.linalg.norm(actual - reference) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--leaf-size", type=int, default=128)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--max-rank", type=int, default=16)
    args = parser.parse_args()
    if args.tolerance < 0.0 or args.leaf_size < 1 or args.max_rank < 1:
        raise SystemExit("tolerance must be nonnegative; leaf-size/max-rank positive")

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom", verbose=0
    )
    mf = scf.RHF(mol).run()
    cusp = NuclearCusp.create(mol)
    boys_handy = BoysHandy.create(mol)
    jastrow = CompositeJastrow.create([cusp, boys_handy])
    params = [cusp.init_params(), boys_handy.init_params()]

    tc = TC.from_pyscf(mf, jastrow, grid_lvl=0)
    isdf = ISDFTC.from_tc(tc, n_rank=max(8, 3 * tc.n_orb), is_incore=True)
    direct = isdf.isdf(
        params,
        batch_size=32,
        host_grid_block_size=512,
        reuse_aux_kernels=True,
        use_laux_fast_grad=True,
        use_laux_exact_split=True,
    )
    hierarchy = isdf.isdf(
        params,
        batch_size=32,
        host_grid_block_size=512,
        reuse_aux_kernels=True,
        use_laux_fast_grad=True,
        use_laux_exact_split=True,
        use_laux_hmatrix=True,
        laux_hmatrix_leaf_size=args.leaf_size,
        laux_hmatrix_eta=args.eta,
        laux_hmatrix_tolerance=args.tolerance,
        laux_hmatrix_max_rank=args.max_rank,
    )

    print("hierarchical L_aux H2/STO-3G control")
    for key in ("L_aux", "K1_kernel", "K3_kernel"):
        error = relative_error(
            np.asarray(hierarchy.isdf_kernels[key]),
            np.asarray(direct.isdf_kernels[key]),
        )
        print(f"  {key}: relative error {error:.3e}")
        if args.tolerance == 0.0:
            assert error < 2.0e-12, f"exact control failed for {key}: {error:.3e}"

    direct_2b = np.asarray(direct.get_2b(params))
    hierarchy_2b = np.asarray(hierarchy.get_2b(params))
    error_2b = relative_error(hierarchy_2b, direct_2b)
    print(f"  two-body: relative error {error_2b:.3e}")
    if args.tolerance == 0.0:
        assert error_2b < 2.0e-12, f"exact control failed for two-body: {error_2b:.3e}"
        print("OK: exact hierarchy control matches direct L_aux/K1/K3/two-body tensors.")


if __name__ == "__main__":
    main()
