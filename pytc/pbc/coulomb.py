"""PeriodicFFTISDF consumer (design v2.1 section 2): wires the S1-S4
pipeline in pytc.pbc.df.{kpts,isdf} into a single build() entry point
producing the interpolation-point factor and per-q solved kernel.

get_ao_eri/get_mo_eri (THC-ERI/ao2mo) were derived by direct algebra on
get_k's own validated composition, not guessed or copied from an external
convention: expanding get_k's kpt_to_spc/spc_to_kpt/Hadamard chain in
closed form (using the phase-matrix orthogonality relation
sum_R phase[R,k1]*phase[R,k2]*conj(phase[R,k]) = delta(k1+k2-k mod G) /
sqrt(Nk)), then using coul_kpt's own Hermiticity (W^q_IJ = conj(W^q_JI),
design v2.1 section 4) to fold a stray conjugate off the kernel factor,
yields the STANDARD pyscf/chemist-convention THC-ERI in get_ao_eri's
docstring directly (conj on a,c; momentum conservation k1-k2+k3-k4=0 via
pyscf's own kconserv table, no axis relabeling needed by callers) -- an
earlier draft used a self-consistent but nonstandard (a,d)-conjugated
convention; normalized to the standard one before any consumer existed,
per review. Verified by reconstructing get_k's own K matrix from a full
(k1,k2) double loop over get_ao_eri blocks and comparing to a direct
get_k call: agreement to 1.4e-15 (machine precision, he2-cubic-cell
[1,1,3] rank=15) -- this is an algebraic identity, not a numerical-
tolerance gate.

get_k's structural composition (density projection -> k<->supercell
unitary transform -> exchange assembly) was derived by reading an
external reference's own K-build routine for UNDERSTANDING, then
independently reimplemented here using pytc.pbc.df.kpts' own
kpt_to_spc/spc_to_kpt. Those functions use the SAME unitary
transform construction the reference does (a phase matrix built from
the actual canonical k-vectors and pyscf's own real-space translation
vectors, k2gamma.translation_vectors_for_kmesh) -- an earlier ifftn-
reshape-based implementation was found and fixed to be wrong (it
assumed the flat k-index maps onto FFT frequency positions the same way
the physical k-ordering does, which is false in general). Validated
against a real periodic FFTDF K matrix on he2-cubic-cell [1,1,3]: exact
reduction to the Gamma-only (Nk=1) molecular ISDF-K formula, Hermiticity,
and rank-matched parity with an external reference implementation at the
same interpolation-point rank.
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
from pytc.pbc.df.kpts import build_kconserv, canonicalize_kpts, kpt_to_spc, spc_to_kpt


def build(cell, kpts, *, rank, block_size, rtol=1e-4, provider_cls=RawKernelProvider):
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
    Pi, eta = build_pi_eta(inpv_kpt, ao_blocks_for_eta, mesh_obj.phase)

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


def get_k(dm_kpts, inpv_kpt, coul_kpt, phase, *, exxdiv=None, cell=None, kpts=None):
    """Periodic THC-ISDF exchange matrix (design v2.1 section 4/7).

    Composition (per density-matrix set):
        rho_kpt[k] = inpv_kpt[k] @ dm_kpt[k] @ inpv_kpt[k].conj().T / Nk
        rho_spc    = kpt_to_spc(rho_kpt, phase).transpose(0,2,1)
        coul_spc   = kpt_to_spc(coul_kpt, phase) * sqrt(Nk)
        v_spc      = coul_spc * rho_spc          -- elementwise (Hadamard)
        v_kpt      = spc_to_kpt(v_spc, phase)
        vk_kpt     = conj( inpv_kpt.transpose(0,2,1) @ v_kpt @ inpv_kpt.conj() )

    The Hadamard product against the density projected onto interpolation
    points (transposed) is the standard ISDF exchange-build trick that
    avoids ever forming a 4-index ERI tensor; kpt_to_spc/spc_to_kpt
    implement the k<->supercell unitary transform (a sum over k' with
    momentum-transfer indexing V[k-k'] becomes an elementwise real-space
    product) so rho and coul are transformed self-consistently.

    exxdiv is applied HERE, after the bare vk_kpt is assembled -- never
    inside a KernelProvider. Only exxdiv=None (bare vk, for FFTDF
    cross-checks) and exxdiv="ewald"
    (pyscf's probe-charge Ewald/Madelung correction) are supported.

    Args:
        dm_kpts: (nset, Nk, Nao, Nao) or (Nk, Nao, Nao) complex128
            density matrices at each canonical k-point.
        inpv_kpt: (Nk, Nip, Nao) complex128, e.g. build()'s inpv_kpt.
        coul_kpt: (Nk, Nip, Nip) complex128, e.g. build()'s coul_kpt.
        phase: (Nk, Nk) complex128 unitary transform matrix, e.g.
            KptsMesh.phase.
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

    coul_spc = kpt_to_spc(coul_kpt, phase) * np.sqrt(n_k)

    vk_kpts = np.empty((n_set, n_k, n_ao, n_ao), dtype=np.complex128)
    for i in range(n_set):
        dm_kpt = dm_kpts[i]
        rho_kpt = (inpv_kpt @ dm_kpt @ inpv_kpt.conj().transpose(0, 2, 1)) / n_k
        rho_spc = kpt_to_spc(rho_kpt, phase).transpose(0, 2, 1)

        v_spc = coul_spc * rho_spc
        v_kpt = spc_to_kpt(v_spc, phase)

        vk_kpt = inpv_kpt.transpose(0, 2, 1) @ v_kpt @ inpv_kpt.conj()
        vk_kpts[i] = vk_kpt.conj()

    if exxdiv == "ewald":
        from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0

        for i in range(n_set):
            _ewald_exxdiv_for_G0(cell, kpts, dm_kpts[i][None], vk_kpts[i][None])

    if single_set:
        vk_kpts = vk_kpts[0]
    return vk_kpts


def get_ao_eri(inpv_kpt, coul_kpt, kconserv, k1, k2, k3):
    """AO-basis THC-ERI block (design v2.1 section 2, THC-ERI/ao2mo
    interface): (a^k1 b^k2 | c^k3 d^k4), never forming the full 4-index
    tensor beyond this one requested block. Standard pyscf/chemist
    convention throughout: a,c conjugated (bra), b,d not (ket); momentum
    conservation k1-k2+k3-k4=0 (mod G) via pyscf's own kconserv table --
    callers may compare a block directly against
    pyscf.pbc.df.FFTDF(cell).get_eri([kpts[k1],kpts[k2],kpts[k3],kpts[k4]])
    with no axis relabeling.

        Q = kconserv[k2, k1, 0]     -- momentum-transfer index (the
            k-point equal to k2 - k1 mod G); NOTE the argument order
            (k2, k1, 0), not (k1, k2, 0) (see module docstring: this
            index enters the formula via coul_kpt's own Hermiticity).
        k4 = kconserv[k1, k2, k3]  -- standard momentum conservation.
        (a^k1 b^k2 | c^k3 d^k4)_{abcd} =
            sum_IJ conj(X[k1]_Ia) X[k2]_Ib * coul_kpt[Q]_IJ *
                   conj(X[k3]_Jc) X[k4]_Jd

    Args:
        inpv_kpt: (Nk, Nip, Nao) complex128, e.g. build()'s inpv_kpt.
        coul_kpt: (Nk, Nip, Nip) complex128, e.g. build()'s coul_kpt.
        kconserv: (Nk, Nk, Nk) int64, e.g. pytc.pbc.df.kpts.build_kconserv.
        k1, k2, k3: canonical k-point indices into inpv_kpt's axis 0.

    Returns:
        (eri_block, k4): eri_block is (Nao, Nao, Nao, Nao) complex128
        (axes a,b,c,d matching k1,k2,k3,k4); k4 is the int index into
        inpv_kpt's axis 0 fixed by momentum conservation.

    Raises:
        ValueError: malformed shapes or out-of-range k-indices.
    """
    inpv_kpt = np.asarray(inpv_kpt, dtype=np.complex128)
    coul_kpt = np.asarray(coul_kpt, dtype=np.complex128)
    kconserv = np.asarray(kconserv)
    n_k, n_ip, n_ao = inpv_kpt.shape
    if coul_kpt.shape != (n_k, n_ip, n_ip):
        raise ValueError(
            f"coul_kpt must have shape ({n_k},{n_ip},{n_ip}) matching inpv_kpt, got "
            f"{coul_kpt.shape}."
        )
    if kconserv.shape != (n_k, n_k, n_k):
        raise ValueError(f"kconserv must have shape ({n_k},{n_k},{n_k}), got {kconserv.shape}.")
    for name, k in (("k1", k1), ("k2", k2), ("k3", k3)):
        if not (0 <= int(k) < n_k):
            raise ValueError(f"{name}={k} out of range for n_k={n_k}.")
    k1, k2, k3 = int(k1), int(k2), int(k3)

    Q = int(kconserv[k2, k1, 0])
    k4 = int(kconserv[k1, k2, k3])

    X1, X2, X3, X4 = inpv_kpt[k1], inpv_kpt[k2], inpv_kpt[k3], inpv_kpt[k4]
    W = coul_kpt[Q]
    rho_ab = np.einsum("Ia,Ib->Iab", X1.conj(), X2, optimize=True)
    rho_cd = np.einsum("Ic,Id->Icd", X3.conj(), X4, optimize=True)
    eri_block = np.einsum("Iab,IJ,Jcd->abcd", rho_ab, W, rho_cd, optimize=True)
    return eri_block, k4


def get_mo_eri(inpv_kpt, coul_kpt, kconserv, mo_coeff_kpts, k1, k2, k3):
    """MO-basis THC-ERI block: get_ao_eri transformed into an arbitrary
    MO basis per k-point via a one-sided AO->MO contraction on
    inpv_kpt (periodic ISDF only needs this, unlike the molecular
    two-sided P/Z sandwich in pytc.df.fit -- coul_kpt is already in a
    pivot x pivot, not an AO-pair, basis, so there is nothing on the
    kernel side left to transform).

    Args:
        inpv_kpt, coul_kpt, kconserv: as in get_ao_eri.
        mo_coeff_kpts: length-4 sequence (C1, C2, C3, C4), each
            (Nao, n_i) complex128 -- MO coefficients at k1, k2, k3, and
            k4 (k4 is derived internally; the caller does not supply a
            k4 index but MUST supply C4 already selected for whatever
            k4 turns out to be, e.g. via mo_coeff_kpts[kconserv[k2,k1,k3]]
            at the call site).
        k1, k2, k3: as in get_ao_eri.

    Returns:
        (eri_mo, k4): eri_mo is (n1, n2, n3, n4) complex128; k4 as in
        get_ao_eri.

    Raises:
        ValueError: malformed shapes, forwarded from get_ao_eri, or
            mo_coeff_kpts does not have exactly 4 entries.
    """
    if len(mo_coeff_kpts) != 4:
        raise ValueError(f"mo_coeff_kpts must have exactly 4 entries, got {len(mo_coeff_kpts)}.")
    C1, C2, C3, C4 = (np.asarray(C, dtype=np.complex128) for C in mo_coeff_kpts)

    eri_ao, k4 = get_ao_eri(inpv_kpt, coul_kpt, kconserv, k1, k2, k3)
    eri_mo = np.einsum(
        "abcd,ai,bj,ck,dl->ijkl", eri_ao, C1, C2, C3, C4, optimize=True
    )
    return eri_mo, k4


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


class ISDFDF:
    """Thin pyscf-compatible `with_df` adapter (design v2.1 section 8,
    V4 gate): wraps this module's build()/get_k/get_j behind the
    `get_jk(dm, hermi, kpts, kpts_band, with_j, with_k, omega, exxdiv)`
    interface pyscf's KRHF/KRKS classes call on `mf.with_df` -- so V4
    exercises OUR integrals inside PYSCF'S UNMODIFIED SCF machinery
    against FFTDF's integrals in the SAME machinery, not a hand-rolled
    SCF loop (which would entangle a from-scratch SCF implementation
    into the comparison and weaken the gate).

    Gate-thin by design: no caching cleverness beyond the one-time
    build() memoization every df object needs to avoid rebuilding the
    interpolation-point factor every SCF iteration, and no feature
    completeness beyond what KRHF/KRKS actually call (kpts_band and
    omega are explicitly NOT supported -- band-structure evaluation and
    range-separated hybrids are out of scope for this gate).

    Args:
        cell: pyscf.pbc.gto.Cell.
        kpts: (Nk,3) absolute k-points, e.g. cell.make_kpts(kmesh).
        rank: requested interpolation-point rank (forwarded to build()).
        block_size: grid points per streamed AO block (forwarded).
        rtol: forwarded to the device Hermitian sandwich solve.
    """

    def __init__(self, cell, kpts, *, rank, block_size, rtol=1e-4):
        self.cell = cell
        self.kpts = np.asarray(kpts, dtype=np.float64)
        self.rank = rank
        self.block_size = block_size
        self.rtol = rtol
        self._built = None
        # KRHF/KRKS's get_hcore calls with_df.get_pp/get_nuc for the
        # pseudopotential/nuclear-attraction core-Hamiltonian term -- an
        # AO-grid integral unrelated to the J/K Coulomb factorization
        # this adapter exists to gate; delegate to a real FFTDF instance
        # (composition, not reimplementation) rather than reinventing it,
        # same pattern get_j already uses.
        from pyscf.pbc.df import FFTDF

        self._core_df = FFTDF(cell, kpts)

    def get_pp(self, kpts=None):
        return self._core_df.get_pp(self.kpts if kpts is None else kpts)

    def get_nuc(self, kpts=None):
        return self._core_df.get_nuc(self.kpts if kpts is None else kpts)

    def build(self):
        """Runs S1-S4 once and caches the result; a real SCF loop calls
        get_jk every iteration but the interpolation-point factor and
        solved kernel are density-independent, so rebuilding them per
        iteration would be wasted (and wrong-scope) work for this gate.
        """
        if self._built is None:
            self._built = build(
                self.cell, self.kpts, rank=self.rank, block_size=self.block_size,
                rtol=self.rtol,
            )
        return self._built

    def get_jk(self, dm_kpts, hermi=1, kpts=None, kpts_band=None, with_j=True,
               with_k=True, omega=None, exxdiv=None):
        if omega is not None:
            raise NotImplementedError(
                "ISDFDF.get_jk: omega (range-separated hybrids) is out of scope for "
                "the V4 gate-thin adapter."
            )
        if kpts_band is not None:
            raise NotImplementedError(
                "ISDFDF.get_jk: kpts_band (band-structure evaluation) is out of "
                "scope for the V4 gate-thin adapter."
            )
        kpts = self.kpts if kpts is None else np.asarray(kpts, dtype=np.float64)
        if kpts.shape != self.kpts.shape or not np.allclose(kpts, self.kpts):
            raise ValueError(
                "ISDFDF.get_jk: kpts passed by the caller do not match the kpts "
                "this adapter was built with -- the cached build() artifact is only "
                "valid for the ORIGINAL kpts."
            )

        vj = None
        vk = None
        if with_j:
            vj = get_j(self.cell, dm_kpts, kpts)
        if with_k:
            built = self.build()
            vk = get_k(
                dm_kpts, built["inpv_kpt"], built["coul_kpt"], built["mesh_obj"].phase,
                exxdiv=exxdiv, cell=self.cell, kpts=kpts,
            )
        return vj, vk
