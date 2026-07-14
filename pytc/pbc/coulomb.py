"""PeriodicFFTISDF consumer (design v2.1 section 2): wires the S1-S4
pipeline in pytc.pbc.df.{kpts,isdf} into a single build() entry point
producing the interpolation-point factor and per-q solved kernel.

The THC-ERI/ao2mo interface is intentionally NOT implemented here yet.

get_k's structural composition (density projection -> k<->supercell
convolution trick -> exchange assembly) was derived by reading an
external reference's own K-build routine for UNDERSTANDING, then
independently reimplemented here using this module's own kpt_to_spc/
spc_to_kpt (design-canonical NumPy-"backward"-FFT convention), never the
reference's phase-matrix machinery. The reference's stated normalization
prefactors (1/Nk on the density projection, sqrt(Nk) on the transformed
kernel) are transcribed faithfully since they are the only concrete
numeric prescription available; this module's own convention has
already been shown to differ from the reference's by a characterized
sqrt(Nk)*conj() factor on Pi^q/eta^q, so get_k's
ABSOLUTE scale is NOT yet independently confirmed here -- only its
Hermiticity and its exact reduction to the Gamma-only (Nk=1) molecular
ISDF-K formula are validated in this module's tests. Numeric validation
against a real periodic FFTDF K matrix is the explicit next step before
any production number from get_k should be trusted.
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
from pytc.pbc.df.kpts import canonicalize_kpts, kpt_to_spc, spc_to_kpt


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


def get_k(dm_kpts, inpv_kpt, coul_kpt, kmesh, *, exxdiv=None, cell=None, kpts=None):
    """Periodic THC-ISDF exchange matrix (design v2.1 section 4/7).

    Composition (per density-matrix set):
        rho_kpt[k] = inpv_kpt[k] @ dm_kpt[k] @ inpv_kpt[k].conj().T / Nk
        rho_spc    = kpt_to_spc(rho_kpt, kmesh).transpose(0,2,1)
        coul_spc   = kpt_to_spc(coul_kpt, kmesh) * sqrt(Nk)
        v_spc      = coul_spc * rho_spc          -- elementwise (Hadamard)
        v_kpt      = spc_to_kpt(v_spc, kmesh)
        vk_kpt     = conj( inpv_kpt.transpose(0,2,1) @ v_kpt @ inpv_kpt.conj() )

    The Hadamard product against the density projected onto interpolation
    points (transposed) is the standard ISDF exchange-build trick that
    avoids ever forming a 4-index ERI tensor; kpt_to_spc/spc_to_kpt
    implement the k<->supercell convolution theorem trick (a sum over
    k' with momentum-transfer indexing V[k-k'] becomes an elementwise
    real-space product) using THIS module's own canonical FFT
    convention throughout, so rho and coul are transformed self-
    consistently even though the ABSOLUTE scale is not yet independently
    confirmed against a real reference (see module docstring).

    exxdiv is applied HERE, after the bare vk_kpt is assembled -- never
    inside a KernelProvider. Only exxdiv=None (bare vk, for FFTDF
    cross-checks) and exxdiv="ewald"
    (pyscf's probe-charge Ewald/Madelung correction) are supported.

    Args:
        dm_kpts: (nset, Nk, Nao, Nao) or (Nk, Nao, Nao) complex128
            density matrices at each canonical k-point.
        inpv_kpt: (Nk, Nip, Nao) complex128, e.g. build()'s inpv_kpt.
        coul_kpt: (Nk, Nip, Nip) complex128, e.g. build()'s coul_kpt.
        kmesh: (3,) positive ints, e.g. KptsMesh.kmesh.
        exxdiv: None or "ewald".
        cell: pyscf.pbc.gto.Cell, required when exxdiv="ewald".
        kpts: (Nk,3) absolute k-points, required when exxdiv="ewald"
            (pyscf's Ewald helper needs the actual k-vectors, not just
            the mesh shape).

    Returns:
        vk_kpts: (nset, Nk, Nao, Nao) complex128 (real-cast when the
        mesh contains only real-valued k-points, matching pyscf's own
        get_k_kpts convention).

    Raises:
        ValueError: malformed shapes, or exxdiv is not None/"ewald", or
            exxdiv="ewald" without cell/kpts.
    """
    inpv_kpt = np.asarray(inpv_kpt, dtype=np.complex128)
    coul_kpt = np.asarray(coul_kpt, dtype=np.complex128)
    n_k, n_ip, n_ao = inpv_kpt.shape
    if coul_kpt.shape != (n_k, n_ip, n_ip):
        raise ValueError(
            f"coul_kpt must have shape ({n_k},{n_ip},{n_ip}) matching inpv_kpt, got "
            f"{coul_kpt.shape}."
        )
    if exxdiv not in (None, "ewald"):
        raise ValueError(f"exxdiv must be None or 'ewald', got {exxdiv!r}.")
    if exxdiv == "ewald" and (cell is None or kpts is None):
        raise ValueError("exxdiv='ewald' requires both cell and kpts.")

    dm_kpts = np.asarray(dm_kpts, dtype=np.complex128)
    single_set = dm_kpts.ndim == 3
    if single_set:
        dm_kpts = dm_kpts[None, ...]
    n_set = dm_kpts.shape[0]
    if dm_kpts.shape != (n_set, n_k, n_ao, n_ao):
        raise ValueError(
            f"dm_kpts must have shape (nset,{n_k},{n_ao},{n_ao}) or ({n_k},{n_ao},{n_ao}), "
            f"got {dm_kpts.shape}."
        )

    coul_spc = kpt_to_spc(coul_kpt, kmesh) * np.sqrt(n_k)

    vk_kpts = np.empty((n_set, n_k, n_ao, n_ao), dtype=np.complex128)
    for i in range(n_set):
        dm_kpt = dm_kpts[i]
        rho_kpt = (inpv_kpt @ dm_kpt @ inpv_kpt.conj().transpose(0, 2, 1)) / n_k
        rho_spc = kpt_to_spc(rho_kpt, kmesh).transpose(0, 2, 1)

        v_spc = coul_spc * rho_spc
        v_kpt = spc_to_kpt(v_spc, kmesh)

        vk_kpt = inpv_kpt.transpose(0, 2, 1) @ v_kpt @ inpv_kpt.conj()
        vk_kpts[i] = vk_kpt.conj()

    if exxdiv == "ewald":
        from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

        for i in range(n_set):
            _ewald_exxdiv_for_G0(cell, kpts, dm_kpts[i][None], vk_kpts[i][None])

    if single_set:
        vk_kpts = vk_kpts[0]
    return vk_kpts


def get_j(cell, dm_kpts, kpts):
    """Periodic Coulomb (J) matrix -- delegates entirely to plain pyscf
    grid-J (never ISDF-factorized), matching the reference's own
    approach (design v2.1 section 6). Only the D2H/H2D accounting around
    this call is a device-pipeline concern, not the J formula itself.
    Constructs a real pyscf.pbc.df.FFTDF object (not a hand-rolled
    stand-in) since get_j_kpts needs its full _numint/grids/aoR_loop
    machinery, not just cell/mesh.

    Args:
        cell: pyscf.pbc.gto.Cell.
        dm_kpts: (nset, Nk, Nao, Nao) or (Nk, Nao, Nao) density matrices.
        kpts: (Nk,3) absolute k-points.

    Returns:
        vj_kpts: same leading shape convention as dm_kpts, from pyscf's
        own get_j_kpts.
    """
    from pyscf.pbc.df import FFTDF
    from pyscf.pbc.df.fft_jk import get_j_kpts

    return get_j_kpts(FFTDF(cell), dm_kpts, kpts=np.asarray(kpts))
