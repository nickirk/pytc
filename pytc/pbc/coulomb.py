"""PeriodicFFTISDF consumer: wires the S1-S4 pipeline in
pytc.pbc.df.{kpts,isdf} into build(), plus get_k/get_j and the THC-ERI
(get_ao_eri/get_mo_eri) and pyscf with_df (ISDFDF) interfaces.
See design doc §2, §4, §7-§8.
"""

from __future__ import annotations

import numpy as np

from pytc.pbc.df.isdf import (
    DEFAULT_JAX_CACHED_SELECTOR_CACHE_MAX_BYTES,
    RawKernelProvider,
    build_cached_periodic_pivot_oracle,
    build_coul_kpt_device,
    build_periodic_pivot_oracle,
    build_pi_eta,
    candidate_panel_indices,
    explicit_candidate_identity,
    full_grid_candidate_identity,
    periodic_metric_column_from_ao,
    periodic_metric_from_ao,
    pivoted_cholesky_hermitian,
    jax_cached_matrix_free_byte_model,
    select_jax_cached_matrix_free,
    stream_ao_blocks,
)
from pytc.pbc.df.kpts import canonicalize_kpts, check_time_reversal_residual, kpt_to_spc, spc_to_kpt


def build(cell, kpts, *, rank, block_size, rtol=1e-4, retention_mode="single",
          provider_cls=RawKernelProvider, selection_mode="jax_cached_matrix_free",
          selection_cache_max_bytes=DEFAULT_JAX_CACHED_SELECTOR_CACHE_MAX_BYTES):
    """Build the periodic FFT-ISDF interpolation-point factor and solved
    kernel for one (cell, k-mesh) system, wiring S1-S4 end to end.

    Args:
        kpts: (Nk,3) absolute k-points (canonicalized internally).
        rank: requested interpolation-point rank.
        block_size: grid points per streamed AO block.
        retention_mode: "single" or "pairwise" -- forwarded to the S4
            Hermitian sandwich solve. See hermitian_sandwich_solve's
            docstring (pytc/df/solvers.py) for the two modes.
        provider_cls: KernelProvider for S4 (default RawKernelProvider).

    Returns:
        dict: mesh_obj (KptsMesh), inpv_kpt (Nk,Nip,Nao) complex128,
        coul_kpt / kern_kpt (Nk,Nip,Nip) complex128, n_selected (may be
        < rank if the pivot metric exhausts), n_pipeline_calls,
        solve_infos (length-Nk list).
    """
    valid_selection_modes = {
        "jax_cached_matrix_free", "streamed", "cached_full", "panel_dense", "panel_oracle",
    }
    if selection_mode not in valid_selection_modes:
        raise ValueError(
            "selection_mode must be 'jax_cached_matrix_free', 'streamed', 'cached_full', "
            "'panel_dense', or 'panel_oracle'"
        )
    mesh_obj = canonicalize_kpts(cell, kpts)
    grid_coords = cell.get_uniform_grids(cell.mesh)

    ao_stats = {"pbc_eval_calls": 0, "grid_points": 0}
    selection_provenance = {
        "mode": selection_mode,
        "candidate_rule": "all_grid_points_v1",
        "candidate_count": int(grid_coords.shape[0]),
        "candidate_identity": full_grid_candidate_identity(grid_coords.shape[0]),
        "cache_bytes": 0,
        "panel_bytes": 0,
        "ao_dtype": np.dtype(np.complex128).name,
    }
    cached_ao = None
    if selection_mode == "jax_cached_matrix_free":
        n_ao = int(cell.nao_nr())
        byte_model = jax_cached_matrix_free_byte_model(
            mesh_obj.n_kpts, grid_coords.shape[0], n_ao, rank,
            cache_max_bytes=selection_cache_max_bytes,
        )
        # The exact cache policy is checked before any full-grid AO allocation.
        # select_jax_cached_matrix_free raises the named capacity condition if
        # this record is outside policy; there is deliberately no dense/panel
        # fallback from the production default.
        if byte_model["capacity_condition"] is not None:
            from pytc.pbc.df.isdf import JAXCachedMatrixFreeCapacityError
            raise JAXCachedMatrixFreeCapacityError(
                f"{byte_model['capacity_condition']}: AO cache requires "
                f"{byte_model['ao_cache_complex128_bytes']} bytes, policy allows "
                f"{byte_model['cache_max_bytes']} bytes."
            )
        _, _, cached_ao = build_cached_periodic_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size, stats=ao_stats,
        )
        pivots, _, n_selected, jax_provenance = select_jax_cached_matrix_free(
            cached_ao, rank, cache_max_bytes=selection_cache_max_bytes,
        )
        selection_provenance.update(jax_provenance)
        selection_provenance["cache_bytes"] = int(cached_ao.nbytes)
        selection_provenance["eta_ao_source"] = "same_full_grid_ao_cache"
    elif selection_mode == "cached_full":
        diag, col_eval, cached_ao = build_cached_periodic_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size, stats=ao_stats,
        )
    else:
        diag, col_eval = build_periodic_pivot_oracle(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size, stats=ao_stats,
        )
    if selection_mode == "jax_cached_matrix_free":
        pass
    elif selection_mode == "streamed":
        pivots, _, n_selected = pivoted_cholesky_hermitian(diag, col_eval, rank=rank)
    elif selection_mode == "cached_full":
        pivots, _, n_selected = pivoted_cholesky_hermitian(diag, col_eval, rank=rank)
        selection_provenance["cache_bytes"] = int(cached_ao.nbytes)
    else:
        candidates = candidate_panel_indices(diag, rank)
        panel_ao = np.asarray(cell.pbc_eval_gto(
            "GTOval", grid_coords[candidates], kpts=list(mesh_obj.canonical_kpts)
        ), dtype=np.complex128)
        ao_stats["pbc_eval_calls"] += 1
        ao_stats["grid_points"] += int(candidates.size)
        if selection_mode == "panel_dense":
            panel_metric = periodic_metric_from_ao(panel_ao)
            panel_pivots, _, n_selected = pivoted_cholesky_hermitian(
                panel_metric.real.diagonal(), lambda j: panel_metric[:, j], rank=rank
            )
            selection_provenance["panel_bytes"] = int(panel_ao.nbytes + panel_metric.nbytes)
        else:
            panel_pivots, _, n_selected = pivoted_cholesky_hermitian(
                np.sum(np.abs(panel_ao) ** 2, axis=(0, 2)) ** 2 / panel_ao.shape[0],
                lambda j: periodic_metric_column_from_ao(panel_ao, j), rank=rank,
            )
            selection_provenance["panel_bytes"] = int(panel_ao.nbytes)
        pivots = candidates[panel_pivots]
        selection_provenance.update({
            "candidate_rule": "top_half_effective_diag_plus_stratified_bins_v1",
            "candidate_count": int(candidates.size),
            "candidate_identity": explicit_candidate_identity(candidates),
        })

    selection_provenance["ao_calls_selection"] = ao_stats["pbc_eval_calls"]
    selection_provenance["ao_grid_points_selection"] = ao_stats["grid_points"]
    inpv_kpt = np.asarray(
        cell.pbc_eval_gto("GTOval", grid_coords[pivots], kpts=list(mesh_obj.canonical_kpts)),
        dtype=np.complex128,
    )
    ao_stats["pbc_eval_calls"] += 1
    ao_stats["grid_points"] += int(pivots.size)
    ao_tr_residual = check_time_reversal_residual(inpv_kpt, mesh_obj.neg)

    ao_blocks_for_eta = cached_ao if cached_ao is not None else (
        blk for _, _, blk in stream_ao_blocks(
            cell, mesh_obj.canonical_kpts, grid_coords, block_size, stats=ao_stats,
        )
    )
    Pi, eta = build_pi_eta(inpv_kpt, ao_blocks_for_eta, mesh_obj.phase, mesh_obj.neg)

    provider = provider_cls(
        cell=cell, canonical_kpts=mesh_obj.canonical_kpts, grid_mesh=cell.mesh
    )
    coul_kpt, kern_kpt, solve_infos, n_pipeline_calls = build_coul_kpt_device(
        provider, Pi, eta, grid_coords, mesh_obj, rtol=rtol, retention_mode=retention_mode
    )

    return {
        "mesh_obj": mesh_obj,
        "inpv_kpt": inpv_kpt,
        "coul_kpt": coul_kpt,
        "kern_kpt": kern_kpt,
        "n_selected": n_selected,
        "n_pipeline_calls": n_pipeline_calls,
        "solve_infos": solve_infos,
        "ao_tr_residual": ao_tr_residual,
        "selection_provenance": {
            **selection_provenance,
            "pivot_indices": pivots.tolist(),
            "n_selected": n_selected,
            "ao_calls_through_eta": ao_stats["pbc_eval_calls"],
            "ao_grid_points_through_eta": ao_stats["grid_points"],
        },
    }


def get_k(dm_kpts, inpv_kpt, coul_kpt, phase, *, exxdiv=None, cell=None, kpts=None, neg=None):
    """Periodic THC-ISDF exchange matrix (design doc §4, §7).

    Composition (per density-matrix set):
        rho_kpt[k] = inpv_kpt[k] @ dm_kpt[k] @ inpv_kpt[k].conj().T / Nk
        rho_spc    = kpt_to_spc(rho_kpt, phase).transpose(0,2,1)
        coul_spc   = kpt_to_spc(coul_kpt, phase) * sqrt(Nk)
        v_spc      = coul_spc * rho_spc          -- elementwise (Hadamard)
        v_kpt      = spc_to_kpt(v_spc, phase)
        vk_kpt     = conj( inpv_kpt.transpose(0,2,1) @ v_kpt @ inpv_kpt.conj() )

    exxdiv is applied HERE, after the bare vk_kpt -- never inside a
    KernelProvider. Only None and "ewald" are supported.

    Args:
        dm_kpts: (nset, Nk, Nao, Nao) or (Nk, Nao, Nao) complex128.
        inpv_kpt: (Nk, Nip, Nao) complex128.
        coul_kpt: (Nk, Nip, Nip) complex128.
        phase: (Nk, Nk) unitary matrix (KptsMesh.phase).
        cell, kpts: required when exxdiv="ewald".
        neg: (Nk,) int array (KptsMesh.neg), optional. When given,
            rho_kpt is symmetrized exactly by construction before
            kpt_to_spc (a real SCF density satisfies
            rho_kpt[neg[k]]=conj(rho_kpt[k]) only to floating-point
            precision, which trips kpt_to_spc's imag_tol gate). Without
            neg, the prior gate-armed, unsymmetrized behavior applies.

    Returns:
        vk_kpts: (nset, Nk, Nao, Nao) complex128.
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

    if neg is not None:
        neg = np.asarray(neg)
        if neg.shape != (n_k,):
            raise ValueError(f"neg must have shape ({n_k},), got {neg.shape}.")
    coul_spc = kpt_to_spc(coul_kpt, phase) * np.sqrt(n_k)

    vk_kpts = np.empty((n_set, n_k, n_ao, n_ao), dtype=np.complex128)
    for i in range(n_set):
        dm_kpt = dm_kpts[i]
        rho_kpt = (inpv_kpt @ dm_kpt @ inpv_kpt.conj().transpose(0, 2, 1)) / n_k
        if neg is not None:
            # Symmetrize rho_kpt exactly by construction: a real SCF dm
            # satisfies rho_kpt[neg[k]]=conj(rho_kpt[k]) only to roundoff,
            # which trips kpt_to_spc's imag_tol gate; both pair members
            # encode the same physics, so deriving one loses nothing.
            rho_kpt = rho_kpt.copy()
            visited = np.zeros(n_k, dtype=bool)
            for k in range(n_k):
                if visited[k]:
                    continue
                nk = int(neg[k])
                if nk == k:
                    rho_kpt[k] = rho_kpt[k].real.astype(np.complex128)
                elif not visited[nk]:
                    rho_kpt[nk] = rho_kpt[k].conj()
                visited[k] = True
                visited[nk] = True
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
    """AO-basis THC-ERI block (a^k1 b^k2 | c^k3 d^k4), one block at a time.
    Standard pyscf/chemist convention: a,c conjugated; k4 fixed by
    momentum conservation -- directly comparable to FFTDF.get_eri with no
    axis relabeling. See design doc §2.

        Q  = kconserv[k2, k1, 0]   -- NOTE the (k2, k1, 0) argument order,
             not (k1, k2, 0): this index enters via coul_kpt's Hermiticity.
        k4 = kconserv[k1, k2, k3]
        (a^k1 b^k2 | c^k3 d^k4)_{abcd} =
            sum_IJ conj(X[k1]_Ia) X[k2]_Ib * coul_kpt[Q]_IJ *
                   conj(X[k3]_Jc) X[k4]_Jd

    Returns:
        (eri_block, k4): eri_block (Nao, Nao, Nao, Nao) complex128.
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
    """MO-basis THC-ERI block: get_ao_eri transformed per k-point.

    Args:
        mo_coeff_kpts: length-4 sequence (C1, C2, C3, C4), each
            (Nao, n_i) complex128. k4 is derived internally, but the
            caller MUST supply C4 already selected for whatever k4 turns
            out to be.

    Returns:
        (eri_mo, k4): eri_mo (n1, n2, n3, n4) complex128.
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
    """Periodic Coulomb (J) matrix -- delegates entirely to pyscf grid-J
    via a real FFTDF object (never ISDF-factorized; design doc §6).

    Returns:
        vj_kpts: same leading shape convention as dm_kpts.
    """
    from pyscf.pbc.df import FFTDF
    from pyscf.pbc.df.fft_jk import get_j_kpts

    return get_j_kpts(FFTDF(cell), dm_kpts, kpts=np.asarray(kpts))


class ISDFDF:
    """Thin pyscf-compatible `with_df` adapter (design doc §8, V4 gate):
    wraps build()/get_k/get_j behind the get_jk interface pyscf's
    KRHF/KRKS call on mf.with_df. Gate-thin: one-time build()
    memoization only; kpts_band and omega are NOT supported.

    Args:
        kpts: (Nk,3) absolute k-points, e.g. cell.make_kpts(kmesh).
        rank, block_size, rtol, retention_mode: forwarded to build().
    """

    def __init__(self, cell, kpts, *, rank, block_size, rtol=1e-4, retention_mode="single",
                 selection_mode="streamed"):
        self.cell = cell
        self.kpts = np.asarray(kpts, dtype=np.float64)
        self.rank = rank
        self.block_size = block_size
        self.rtol = rtol
        self.retention_mode = retention_mode
        self.selection_mode = selection_mode
        self._built = None
        # get_pp/get_nuc (core-Hamiltonian integrals, unrelated to the J/K
        # factorization) delegate to a real FFTDF instance.
        from pyscf.pbc.df import FFTDF

        self._core_df = FFTDF(cell, kpts)

    def get_pp(self, kpts=None):
        return self._core_df.get_pp(self.kpts if kpts is None else kpts)

    def get_nuc(self, kpts=None):
        return self._core_df.get_nuc(self.kpts if kpts is None else kpts)

    def build(self):
        """Run S1-S4 once and cache; the build artifacts are
        density-independent."""
        if self._built is None:
            self._built = build(
                self.cell, self.kpts, rank=self.rank, block_size=self.block_size,
                rtol=self.rtol, retention_mode=self.retention_mode,
                selection_mode=self.selection_mode,
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
            from pyscf.pbc.df.fft_jk import get_j_kpts

            vj = get_j_kpts(self._core_df, dm_kpts, kpts=kpts)
        if with_k:
            built = self.build()
            vk = get_k(
                dm_kpts, built["inpv_kpt"], built["coul_kpt"], built["mesh_obj"].phase,
                exxdiv=exxdiv, cell=self.cell, kpts=kpts, neg=built["mesh_obj"].neg,
            )
        return vj, vk
