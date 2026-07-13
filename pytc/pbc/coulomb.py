"""PeriodicFFTISDF consumer (design v2.1 section 2): wires the S1-S4
pipeline in pytc.pbc.df.{kpts,isdf} into a single build() entry point
producing the interpolation-point factor and per-q solved kernel.

get_k/get_j and the THC-ERI/ao2mo interface are intentionally NOT
implemented here yet -- their exchange/Coulomb contraction formulas need
their own derivation-and-cross-validation pass (the same rigor already
applied to the KernelProvider seam and the periodic pivot-selection
metric oracle), not a guess from the design doc's prose alone.
"""

from __future__ import annotations

import numpy as np

from pytc.pbc.df.isdf import (
    RawKernelProvider,
    build_coul_kpt_device,
    build_periodic_pivot_oracle,
    build_pi_eta,
    pivoted_cholesky_hermitian,
    stream_ao_blocks,
)
from pytc.pbc.df.kpts import canonicalize_kpts


def build(cell, kpts, *, rank, block_size, rtol=1e-8, provider_cls=RawKernelProvider):
    """Build the periodic FFT-ISDF interpolation-point factor and solved
    kernel for one (cell, k-mesh) system, wiring S1-S4 end to end:
        S1/S2: stream_ao_blocks + build_periodic_pivot_oracle -> pivot
            selection via pivoted_cholesky_hermitian.
        S3: build_pi_eta (Pi^q/eta^q), consuming a streamed AO-block
            generator for the eta RHS.
        S4: build_coul_kpt_device (device kernel-apply + Hermitian
            solve, exploiting the q<->-q conjugate closure).

    Args:
        cell: pyscf.pbc.gto.Cell.
        kpts: (Nk,3) absolute k-points (any order/gauge -- canonicalized
            internally).
        rank: requested interpolation-point rank (forwarded to
            pivoted_cholesky_hermitian).
        block_size: grid points per streamed AO block (S1/S3).
        rtol: forwarded to the device Hermitian sandwich solve.
        provider_cls: KernelProvider implementation used for S4 (default
            RawKernelProvider, the bare 4pi/G^2 kernel).

    Returns:
        dict with keys:
            mesh_obj: pytc.pbc.df.kpts.KptsMesh.
            inpv_kpt: (Nk,Nip,Nao) complex128 -- AO values at the
                selected interpolation points, across all k.
            coul_kpt: (Nk,Nip,Nip) complex128 -- the solved kernel W^q.
            kern_kpt: (Nk,Nip,Nip) complex128 -- the raw contracted
                kernel before the Hermitian sandwich solve.
            n_selected: int, realized interpolation-point rank (may be
                less than the requested rank if the pivot metric is
                numerically exhausted first).
            n_pipeline_calls: int, from build_coul_kpt_device's
                conjugate-shortcut accounting.
            solve_infos: length-Nk list of per-q solve info dicts.

    Raises:
        ValueError: forwarded from canonicalize_kpts, build_periodic_pivot_oracle,
            pivoted_cholesky_hermitian, build_pi_eta, or build_coul_kpt_device
            for malformed inputs.
    """
    mesh_obj = canonicalize_kpts(cell, kpts)
    grid_coords = cell.get_uniform_grids(cell.mesh)

    diag, col_eval = build_periodic_pivot_oracle(
        cell, mesh_obj.canonical_kpts, grid_coords, block_size
    )
    pivots, _, n_selected = pivoted_cholesky_hermitian(diag, col_eval, rank=rank)

    inpv_kpt = np.asarray(
        cell.pbc_eval_gto("GTOval", grid_coords[pivots], kpts=list(mesh_obj.canonical_kpts)),
        dtype=np.complex128,
    )

    ao_blocks_for_eta = (
        blk for _, _, blk in stream_ao_blocks(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size
        )
    )
    Pi, eta = build_pi_eta(inpv_kpt, ao_blocks_for_eta, mesh_obj.kmesh)

    provider = provider_cls(
        cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
    )
    coul_kpt, kern_kpt, solve_infos, n_pipeline_calls = build_coul_kpt_device(
        provider, Pi, eta, grid_coords, mesh_obj, rtol=rtol
    )

    return {
        "mesh_obj": mesh_obj,
        "inpv_kpt": inpv_kpt,
        "coul_kpt": coul_kpt,
        "kern_kpt": kern_kpt,
        "n_selected": n_selected,
        "n_pipeline_calls": n_pipeline_calls,
        "solve_infos": solve_infos,
    }


def get_k(*args, **kwargs):
    """NOT YET IMPLEMENTED. The periodic exchange-matrix contraction
    (density -> K via inpv_kpt/coul_kpt) and the exxdiv/Ewald
    post-processing insertion point (exxdiv is owned HERE, never inside
    a KernelProvider) need their own derivation-and-cross-validation
    pass against a real reference before being implemented -- not
    guessed from the design doc's prose alone.
    """
    raise NotImplementedError(
        "get_k: periodic exchange contraction + exxdiv post-processing not yet "
        "implemented -- needs its own derivation/cross-validation pass, see this "
        "function's docstring."
    )


def get_j(*args, **kwargs):
    """NOT YET IMPLEMENTED -- see get_k's docstring for why."""
    raise NotImplementedError(
        "get_j: periodic Coulomb (J) contraction not yet implemented -- needs its "
        "own derivation/cross-validation pass, see get_k's docstring."
    )
