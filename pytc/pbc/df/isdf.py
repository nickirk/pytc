"""Periodic ISDF fit machinery: matrix-free Hermitian-PSD pivoted Cholesky
selector, Pi^q/eta^q builders, kernel-apply-and-solve (NumPy oracle and
device/KernelProvider paths), and staging policy. See design doc §3-§7.
Deliberately independent of pytc.df.pivots (see design doc §3).
"""

from __future__ import annotations

import dataclasses
import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


# The exact E1 selector keeps the complete Bloch AO cache on the JAX device.
# This is intentionally a bounded, fail-closed baseline; E3 will provide the
# distinct localized large-system algorithm rather than silently changing the
# exact selector's physical candidate set.
DEFAULT_JAX_CACHED_SELECTOR_CACHE_MAX_BYTES = 24 * 2**30


class JAXCachedMatrixFreeCapacityError(RuntimeError):
    """Raised when the exact AO cache exceeds the declared selector policy."""

    condition = "JAX_CACHED_MATRIX_FREE_AO_CACHE_EXCEEDS_POLICY"


def jax_cached_matrix_free_byte_model(
    n_kpts, n_grid, n_ao, rank, *, cache_max_bytes=DEFAULT_JAX_CACHED_SELECTOR_CACHE_MAX_BYTES,
):
    """Return exact selector residency terms without allocating an AO cache.

    ``F`` is the complex128 matrix with shape ``(Nk*Nao, Ng)`` used by the
    device pivot loop.  The Cholesky factor and residual are real float64;
    the metric column is formed transiently from ``F.conj().T @ F[:, pivot]``.
    A cache-policy failure is named in the returned record so callers can
    refuse before evaluating the complete AO grid.
    """
    values = {
        "n_kpts": n_kpts,
        "n_grid": n_grid,
        "n_ao": n_ao,
        "rank": rank,
        "cache_max_bytes": cache_max_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    cache_bytes = int(n_kpts) * int(n_ao) * int(n_grid) * np.dtype(np.complex128).itemsize
    factor_bytes = int(n_grid) * int(rank) * np.dtype(np.float64).itemsize
    gram_work_bytes = int(n_grid) * np.dtype(np.complex128).itemsize
    real_work_bytes = 2 * int(n_grid) * np.dtype(np.float64).itemsize
    selector_bytes = int(n_grid) * np.dtype(np.bool_).itemsize
    pivot_bytes = int(rank) * np.dtype(np.int64).itemsize
    work_bytes = gram_work_bytes + real_work_bytes + selector_bytes + pivot_bytes
    capacity_condition = (
        None
        if cache_bytes <= int(cache_max_bytes)
        else JAXCachedMatrixFreeCapacityError.condition
    )
    return {
        "mode": "jax_cached_matrix_free",
        "ao_cache_layout": "F[Nk*Nao,Ng]",
        "ao_cache_complex128_bytes": cache_bytes,
        "cholesky_real_float64_bytes": factor_bytes,
        "pivot_work_bytes": work_bytes,
        "selection_peak_device_bytes": cache_bytes + factor_bytes + work_bytes,
        "cache_max_bytes": int(cache_max_bytes),
        "capacity_condition": capacity_condition,
        "within_cache_policy": capacity_condition is None,
    }


def select_jax_cached_matrix_free(
    ao_cache, rank, *, rcond=1e-12, ramp_scale=1e-12,
    cache_max_bytes=DEFAULT_JAX_CACHED_SELECTOR_CACHE_MAX_BYTES,
):
    """Select exact full-grid pivots in one JIT/device control-flow loop.

    ``ao_cache`` has shape ``(Nk, Ng, Nao)``.  The returned pivots use the
    same metric, residual update, threshold, and high-index tie convention as
    :func:`pivoted_cholesky_hermitian`; the only change is that every pivot
    column is formed from the already cached AO matrix on the JAX device.
    Host conversion happens only after the complete device loop returns.
    """
    ao_cache = np.asarray(ao_cache, dtype=np.complex128)
    if ao_cache.ndim != 3 or any(size <= 0 for size in ao_cache.shape):
        raise ValueError("ao_cache must have nonempty shape (Nk,Ng,Nao).")
    n_kpts, n_grid, n_ao = ao_cache.shape
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError(f"rank must be an integer, got {rank!r}.")
    rank = int(rank)
    if rank <= 0 or rank > n_grid:
        raise ValueError(f"rank must be in [1,{n_grid}], got {rank}.")
    byte_model = jax_cached_matrix_free_byte_model(
        n_kpts, n_grid, n_ao, rank, cache_max_bytes=cache_max_bytes,
    )
    if byte_model["capacity_condition"] is not None:
        raise JAXCachedMatrixFreeCapacityError(
            f"{byte_model['capacity_condition']}: AO cache requires "
            f"{byte_model['ao_cache_complex128_bytes']} bytes, policy allows "
            f"{byte_model['cache_max_bytes']} bytes."
        )
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "jax_cached_matrix_free requires jax_enable_x64=True for real float64 "
            "residual/L and complex128 AO cache."
        )

    # F is the only AO object used by the pivot loop: rows are flattened
    # (k, AO) channels and columns are physical-grid points.
    f_cache = jnp.asarray(
        np.ascontiguousarray(ao_cache.transpose(0, 2, 1).reshape(n_kpts * n_ao, n_grid)),
        dtype=jnp.complex128,
    )

    @jax.jit
    def _select(f):
        diagonal = jnp.sum(jnp.abs(f) ** 2, axis=0) ** 2 / n_kpts
        max_diagonal = jnp.max(diagonal)
        ramp = ramp_scale * jnp.arange(n_grid, dtype=jnp.float64) * max_diagonal
        threshold = rcond * max_diagonal
        initial = (
            diagonal.astype(jnp.float64),
            jnp.zeros((n_grid, rank), dtype=jnp.float64),
            jnp.full((rank,), -1, dtype=jnp.int64),
            jnp.zeros((n_grid,), dtype=jnp.bool_),
            jnp.array(0, dtype=jnp.int64),
        )

        def body(t, state):
            residual, factor, pivots, selected, count = state
            pivot = jnp.argmax(jnp.where(selected, -jnp.inf, residual + ramp))
            active = residual[pivot] > threshold
            gram = f.conj().T @ f[:, pivot]
            metric_column = (jnp.abs(gram) ** 2 / n_kpts).astype(jnp.float64)
            previous = factor @ factor[pivot, :]
            denominator = jnp.sqrt(jnp.maximum(residual[pivot], jnp.finfo(jnp.float64).tiny))
            new_column = (metric_column - previous) / denominator
            new_column = jnp.where(active, new_column, jnp.zeros_like(new_column))
            factor = factor.at[:, t].set(new_column)
            residual = jnp.where(
                active, jnp.maximum(residual - new_column ** 2, 0.0), residual,
            )
            pivots = pivots.at[t].set(jnp.where(active, pivot, -1))
            selected = jnp.where(active, selected.at[pivot].set(True), selected)
            return residual, factor, pivots, selected, count + active.astype(jnp.int64)

        return jax.lax.fori_loop(0, rank, body, initial)

    _, factor, pivots, _, count = _select(f_cache)
    n_selected = int(np.asarray(count))
    pivots_host = np.asarray(pivots)[:n_selected]
    factor_host = np.asarray(factor)[:, :n_selected]
    provenance = {
        **byte_model,
        "pivot_executor": "jax.jit/lax.fori_loop",
        "pivot_loop_device_resident": True,
        "ao_cache_dtype": str(f_cache.dtype),
        "factor_dtype": str(factor.dtype),
        "device_platforms": sorted({device.platform for device in f_cache.devices()}),
    }
    return pivots_host, factor_host, n_selected, provenance


def pivoted_cholesky_hermitian(diag, col_eval, rank, *, rcond=1e-12, ramp_scale=1e-12):
    """Matrix-free greedy pivoted (partial) Cholesky for an implicit N x N
    Hermitian PSD matrix M given diag(M) and a column oracle
    col_eval(j) -> M[:, j] of the ORIGINAL M (shape (N,), complex128).

    Tie-break: a tiny increasing ramp `ramp_scale * arange(n) * max(diag)`
    is added to the argmax score, biasing exact ties toward the higher
    index (matches pytc.df.pivots's convention; copied, not imported).

    Returns:
        (pivots, L, n_selected): pivots (n_selected,) int64; L
        (n, n_selected) complex128 partial Cholesky factor; n_selected
        <= rank (fewer if the Schur diagonal exhausts below
        rcond*max(diag) first).
    """
    diag = np.asarray(diag, dtype=np.float64)
    if diag.ndim != 1:
        raise ValueError(f"diag must be 1-D, got shape {diag.shape}.")
    n = diag.shape[0]
    if n == 0:
        raise ValueError("diag must be nonempty.")
    if not np.all(np.isfinite(diag)):
        raise ValueError("diag must be finite.")

    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError(f"rank must be an integer, got {rank!r}.")
    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    if rank > n:
        raise ValueError(f"rank={rank} exceeds n={n} -- cannot select more pivots than rows.")

    max_diag = float(np.max(diag)) if n > 0 else 0.0
    if max_diag <= 0.0:
        raise ValueError("diag is entirely non-positive -- M appears to be the zero matrix.")

    neg_floor = -rcond * max_diag
    if np.any(diag < neg_floor):
        raise ValueError(
            f"diag contains an entry below -{rcond:.1e}*max(diag)={neg_floor:.3e} -- "
            f"M does not appear to be PSD within the declared rcond."
        )
    diag = np.maximum(diag, 0.0)

    ramp = ramp_scale * np.arange(n, dtype=np.float64) * max_diag
    threshold = rcond * max_diag

    L = np.zeros((n, rank), dtype=np.complex128)
    pivots = np.zeros(rank, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)

    n_selected = 0
    for t in range(rank):
        score = np.where(selected, -np.inf, diag + ramp)
        j = int(np.argmax(score))
        if diag[j] <= threshold:
            break

        col = np.asarray(col_eval(j))
        if col.shape != (n,):
            raise ValueError(f"col_eval({j}) must return shape ({n},), got {col.shape}.")
        col = col.astype(np.complex128)

        if t > 0:
            update = L[:, :t] @ L[j, :t].conj()
        else:
            update = 0.0
        l_t = (col - update) / np.sqrt(diag[j])
        L[:, t] = l_t

        diag = diag - np.abs(l_t) ** 2
        diag = np.maximum(diag, 0.0)

        pivots[t] = j
        selected[j] = True
        n_selected += 1

    return pivots[:n_selected], L[:, :n_selected], n_selected


def build_pi_eta(X, ao_blocks, phase, neg, *, imag_tol=1e-10):
    """Build Pi^q = pair_convolve(X, X)[q] and eta^q = pair_convolve(X, AO)[q].
    eta is accumulated block-by-block so one pair_convolve call holds only
    one block of AO data. See design doc §4-§5.

    q-labeling fix (task #25/C2 item 2b, 2026-07-14): Alg. 1's convolution
    (which pair_convolve implements) and Eq. 4/5's own defining equations
    for Pi/eta agree only up to a q<->-q relabeling (derived by dummy-
    relabeling Alg. 1's expansion with m=-k). This is INVISIBLE at
    self-paired q (q=neg[q]: real-valued/trivially-conjugate either way)
    and was invisible in Pi's specific case for a second reason -- Pi's
    symmetric X=X call makes pair_convolve's raw output satisfy
    Pi_raw[neg[q]]=conj(Pi_raw[q]) regardless of labeling, so a plain
    transpose relation to an external oracle (task #25/C2's Test D) LOOKED
    clean while still carrying the SAME offset one level down: feeding
    Pi_raw[q] into the Hermitian sandwich solve reproduces the WRONG
    physical W at genuine-pair q (confirmed empirically: relative error
    ~1e4 vs an external oracle, collapsing to ~1e-11 when Pi_raw[neg[q]]
    is used instead) even though Pi_raw[q] itself "matched" the oracle's
    own (equally offset) metric up to transpose. One bug, two faces --
    both eta and Pi carry the identical offset; only eta's showed up as
    an obvious VALUE mismatch, Pi's hid inside a transpose that looked
    like a clean convention difference rather than a shared bug. Fixed
    identically for both (not inside pair_convolve, which stays correct/
    shared/untouched) by relabeling BOTH outputs' q-axis with neg once,
    after construction -- every consumer (solve modes, the device
    pipeline, apply_kernel_and_solve_device) receives the physical Pi^q/
    eta^q and needs no compensating convention logic of its own.

    Args:
        X: (Nk, Nip, Nao) complex128 across the canonical k-mesh.
        ao_blocks: (Nk, Ng, Nao) complex128 array, or iterable of
            (Nk, blk_i, Nao) blocks on the SAME canonical k-mesh.
        phase: (Nk, Nk) unitary matrix (KptsMesh.phase).
        neg: (Nk,) int array (KptsMesh.neg), used to relabel both
            outputs' q-axis.

    Returns:
        (Pi, eta): (Nk, Nip, Nip) and (Nk, Nip, Ng) complex128.
    """
    # Local import: kpts.py stays a leaf.
    from pytc.pbc.df.kpts import pair_convolve

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (Nk, Nip, Nao), got shape {X.shape}.")
    neg = np.asarray(neg)
    if neg.shape != (X.shape[0],):
        raise ValueError(f"neg must have shape ({X.shape[0]},), got {neg.shape}.")

    Pi = pair_convolve(X, X, phase, imag_tol=imag_tol)[neg]

    if isinstance(ao_blocks, np.ndarray):
        ao_blocks = [ao_blocks]
    else:
        ao_blocks = list(ao_blocks)
    if not ao_blocks:
        raise ValueError("ao_blocks must be nonempty.")

    eta_chunks = [
        pair_convolve(X, np.asarray(block), phase, imag_tol=imag_tol)
        for block in ao_blocks
    ]
    eta = np.concatenate(eta_chunks, axis=2)[neg]
    return Pi, eta


def apply_raw_kernel_and_solve(
    Pi_q, eta_q, *, cell, q_kpt, grid_coords, grid_mesh, rtol=1e-4, self_paired=False
):
    """Apply the "raw" (bare 4pi/G^2, exx=False) periodic Coulomb kernel to
    eta^q over the spatial grid, contract to (Nip, Nip), and solve the
    Hermitian sandwich for W^q. Plain NumPy, single q -- the CPU oracle for
    the device KernelProvider path. See design doc §5, §7.

        lq     = eta_q * exp(-1j * grid_coords @ q_kpt)   # Bloch phase
        wq     = FFT(lq, grid_mesh)
        vq     = coulG(q, exx=False) * vol / Ng
        rq     = conj(IFFT(wq * vq, grid_mesh))
        kern_q = lq @ rq.T / sqrt(Ng)
        W_q    = sqrt(Ng) * hermitian_sandwich_solve(Pi_q, kern_q)[0]

    The final sqrt(Ng) rescale exactly cancels kern_q's 1/sqrt(Ng)
    (paper Eq. 10 factor placements; verified in the V2 reference-replay
    test). exxdiv is NEVER applied here -- it is owned by a later get_k
    post-processing step.

    Args:
        Pi_q: (Nip, Nip) complex128 metric.
        eta_q: (Nip, Ng) complex128 RHS.
        q_kpt: (3,) absolute k-vector for this q.
        grid_coords: (Ng, 3), same flattened order as eta_q's grid axis.
        grid_mesh: (3,) positive ints, real-space integration mesh
            (distinct from the k-point mesh); prod must equal Ng.
        rtol: forwarded to hermitian_sandwich_solve.
        self_paired: True when neg[q]==q. Physics requires both Pi_q and
            kern_q real for such q, but the complex intermediates leave
            floating-point imaginary noise that the near-singular solve
            amplifies; when True, Pi_q.real and kern_q.real are taken
            BEFORE the solve (noise projection, not a loosened gate). See
            design doc §5.

    Returns:
        (W_q, kern_q, solve_info): W_q (Nip, Nip) complex128; kern_q is
        the raw contracted kernel before the solve; solve_info is
        hermitian_sandwich_solve's info dict.
    """
    from pyscf.pbc import tools as pbctools

    from pytc.df.solvers import hermitian_sandwich_solve

    Pi_q = np.asarray(Pi_q)
    eta_q = np.asarray(eta_q)
    n_ip = Pi_q.shape[0]
    if Pi_q.shape != (n_ip, n_ip):
        raise ValueError(f"Pi_q must be square, got shape {Pi_q.shape}.")
    if eta_q.ndim != 2 or eta_q.shape[0] != n_ip:
        raise ValueError(f"eta_q must have shape ({n_ip},Ng), got {eta_q.shape}.")
    n_grid = eta_q.shape[1]
    if self_paired:
        Pi_q = Pi_q.real.astype(np.complex128)

    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords.shape != (n_grid, 3):
        raise ValueError(
            f"grid_coords must have shape ({n_grid},3) matching eta_q's grid axis, "
            f"got {grid_coords.shape}."
        )
    grid_mesh_t = tuple(int(x) for x in grid_mesh)
    if len(grid_mesh_t) != 3 or any(m <= 0 for m in grid_mesh_t):
        raise ValueError(f"grid_mesh must be 3 positive ints, got {grid_mesh_t}.")
    if int(np.prod(grid_mesh_t)) != n_grid:
        raise ValueError(
            f"prod(grid_mesh)={int(np.prod(grid_mesh_t))} != eta_q's grid size {n_grid}."
        )

    q_kpt = np.asarray(q_kpt, dtype=np.float64)
    if q_kpt.shape != (3,):
        raise ValueError(f"q_kpt must have shape (3,), got {q_kpt.shape}.")

    phase = np.exp(-1j * (grid_coords @ q_kpt))
    lq = eta_q * phase[None, :]

    lq_mesh = lq.reshape((n_ip,) + grid_mesh_t)
    wq_mesh = np.fft.fftn(lq_mesh, axes=(1, 2, 3), norm="backward")

    Gv = cell.get_Gv(list(grid_mesh_t))
    vq = pbctools.get_coulG(cell, k=q_kpt, exx=False, Gv=Gv, mesh=list(grid_mesh_t))
    vq = vq * (cell.vol / n_grid)
    vq_mesh = vq.reshape(grid_mesh_t)

    rq_mesh = np.fft.ifftn(wq_mesh * vq_mesh[None, :, :, :], axes=(1, 2, 3), norm="backward")
    rq = rq_mesh.reshape(n_ip, n_grid).conj()

    kern_q = (lq @ rq.T) / np.sqrt(n_grid)
    kern_q = np.asarray(kern_q, dtype=np.complex128)
    if self_paired:
        kern_q = kern_q.real.astype(np.complex128)

    W_q_unscaled, solve_info = hermitian_sandwich_solve(Pi_q, kern_q, rtol=rtol)
    # sqrt(Ng) rescale cancels kern_q's own 1/sqrt(Ng) (Eq. 10 factor placement).
    W_q = np.sqrt(n_grid) * W_q_unscaled
    return W_q, kern_q, solve_info


# ---------------------------------------------------------------------------
# Device path, KernelProvider protocol (design doc §7).
#
# KernelProvider.apply(q_index, lq) is a LINEAR q-momentum kernel operator on
# a PRE-PHASED (Nip, Ng) slab, returning v_q BEFORE the final conjugate; the
# Bloch-phase multiply and outer conjugate are pipeline glue, not part of the
# contract. vol/Ng normalization stays INSIDE the provider; exxdiv stays
# OUTSIDE (owned by get_k post-processing).
#
# Dagger law at this seam: apply(neg[q], conj(l_q)) == conj(apply(q, l_q)),
# given l_q[neg[q]] = conj(l_q[q]). Verified in test_raw_kernel_apply_dagger_law.


@partial(jax.jit, static_argnames=("grid_mesh",))
def _raw_kernel_apply_core(lq, coulG_scaled, grid_mesh):
    """Jitted core of the "raw" provider: v_q = IFFT(coulG_scaled * FFT(lq)).
    No validation (host wrapper's job), no phase multiply, no outer
    conjugate."""
    n_ip = lq.shape[0]
    lq_mesh = lq.reshape((n_ip,) + grid_mesh)
    wq_mesh = jnp.fft.fftn(lq_mesh, axes=(1, 2, 3))
    vq_mesh = jnp.asarray(coulG_scaled, dtype=lq.dtype).reshape(grid_mesh)
    vq_mesh = wq_mesh * vq_mesh[None, :, :, :]
    rq_mesh = jnp.fft.ifftn(vq_mesh, axes=(1, 2, 3))
    return rq_mesh.reshape(n_ip, -1)


def raw_kernel_apply(lq, *, cell, q_kpt, grid_mesh):
    """Host-side wrapper: validate, compute coulG(q)*vol/Ng via pyscf (not
    jittable), dispatch to the jitted core.

    Args:
        lq: (Nip, Ng) complex128, ALREADY Bloch-phase-corrected.
        grid_mesh: (3,) positive ints; prod must equal Ng.

    Returns:
        v_q: (Nip, Ng) complex128 jax array, BEFORE the outer conjugate.
    """
    from pyscf.pbc import tools as pbctools

    lq_np = np.asarray(lq)
    if lq_np.ndim != 2:
        raise ValueError(f"lq must be 2-D (Nip, Ng), got shape {lq_np.shape}.")
    n_ip, n_grid = lq_np.shape

    grid_mesh_t = tuple(int(x) for x in grid_mesh)
    if len(grid_mesh_t) != 3 or any(m <= 0 for m in grid_mesh_t):
        raise ValueError(f"grid_mesh must be 3 positive ints, got {grid_mesh_t}.")
    if int(np.prod(grid_mesh_t)) != n_grid:
        raise ValueError(f"prod(grid_mesh)={int(np.prod(grid_mesh_t))} != lq's grid size {n_grid}.")

    q_kpt_np = np.asarray(q_kpt, dtype=np.float64)
    if q_kpt_np.shape != (3,):
        raise ValueError(f"q_kpt must have shape (3,), got {q_kpt_np.shape}.")

    Gv = cell.get_Gv(list(grid_mesh_t))
    coulG = pbctools.get_coulG(cell, k=q_kpt_np, exx=False, Gv=Gv, mesh=list(grid_mesh_t))
    coulG_scaled = np.asarray(coulG, dtype=np.float64) * (cell.vol / n_grid)

    lq_jnp = jnp.asarray(lq_np, dtype=jnp.complex128)
    if lq_jnp.dtype != jnp.complex128:
        logger.warning(
            f"raw_kernel_apply: resolved dtype is {lq_jnp.dtype}, not complex128 -- JAX "
            f"defaults to complex64 SILENTLY unless the caller has enabled "
            f"jax.config.update('jax_enable_x64', True). This device path is defined only "
            f"at the c128 parity tier (design v2.1 section 1); verify x64 is enabled "
            f"before trusting production numbers from this path."
        )
    return _raw_kernel_apply_core(lq_jnp, jnp.asarray(coulG_scaled), grid_mesh_t)


def precompute_coulG_all_q(cell, canonical_kpts, grid_mesh):
    """Precompute coulG(q)*vol/Ng for every q at once (host-only pyscf
    calls are not jittable; the result is a per-q constant that can then
    be threaded into a jitted core as a traced argument).

    Returns:
        coulG_all: (Nk, Ng) float64 jax array.
    """
    from pyscf.pbc import tools as pbctools

    canonical_kpts_np = np.asarray(canonical_kpts, dtype=np.float64)
    if canonical_kpts_np.ndim != 2 or canonical_kpts_np.shape[1] != 3:
        raise ValueError(
            f"canonical_kpts must have shape (Nk,3), got {canonical_kpts_np.shape}."
        )
    grid_mesh_t = tuple(int(x) for x in grid_mesh)
    if len(grid_mesh_t) != 3 or any(m <= 0 for m in grid_mesh_t):
        raise ValueError(f"grid_mesh must be 3 positive ints, got {grid_mesh_t}.")

    n_grid = int(np.prod(grid_mesh_t))
    Gv = cell.get_Gv(list(grid_mesh_t))
    n_kpts = canonical_kpts_np.shape[0]
    coulG_all = np.empty((n_kpts, n_grid), dtype=np.float64)
    for q in range(n_kpts):
        coulG = pbctools.get_coulG(
            cell, k=canonical_kpts_np[q], exx=False, Gv=Gv, mesh=list(grid_mesh_t)
        )
        coulG_all[q] = np.asarray(coulG, dtype=np.float64) * (cell.vol / n_grid)
    return jnp.asarray(coulG_all)


@partial(jax.jit, static_argnames=("grid_mesh", "self_paired", "retention_mode"))
def _fused_apply_kernel_and_solve_core(
    Pi_q, eta_q, phase_q, coulG_scaled_q, grid_mesh, rtol, self_paired, retention_mode="single"
):
    """Fully fused single-jax.jit per-q hot path: phase-multiply -> raw
    kernel apply -> conjugate -> ZGEMM -> Hermitian sandwich solve, one XLA
    graph with no host round trips. Reachable only via a provider exposing
    fused_apply_and_solve. See design doc §6."""
    n_grid = eta_q.shape[1]
    lq = eta_q * phase_q[None, :]
    v_q = _raw_kernel_apply_core(lq, coulG_scaled_q, grid_mesh)
    rq = jnp.conj(v_q)
    kern_q = (lq @ rq.T) / jnp.sqrt(n_grid)
    if self_paired:
        kern_q = kern_q.real.astype(jnp.complex128)

    from pytc.df.solvers import _hermitian_sandwich_solve_core

    (
        W, n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
        v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
    ) = _hermitian_sandwich_solve_core(Pi_q, kern_q, rtol, retention_mode)

    return (
        W, kern_q, n_retained, s_max, s_min_retained,
        pi_anti_hermitian_residual, v_anti_hermitian_residual,
        retained_solve_residual, truncation_residual,
    )


@dataclasses.dataclass(frozen=True)
class RawKernelProvider:
    """The "raw" (bare 4pi/G^2, exx=False) KernelProvider (design doc §7):
    apply(q_index, lq) -> v_q plus provenance(). q_index resolves the
    absolute k-vector from canonical_kpts internally.

    fused_apply_and_solve is an OPTIONAL fast-path hook that
    apply_kernel_and_solve_device prefers when present; providers without
    it fall back to the eager per-stage path.

    Args:
        canonical_kpts: (Nk, 3) float64, e.g. KptsMesh.canonical_kpts.
        grid_mesh: (3,) positive ints, real-space integration mesh.
    """
    cell: object
    canonical_kpts: object
    grid_mesh: tuple

    def __post_init__(self):
        canonical_kpts = np.asarray(self.canonical_kpts, dtype=np.float64)
        if canonical_kpts.ndim != 2 or canonical_kpts.shape[1] != 3:
            raise ValueError(
                f"canonical_kpts must have shape (Nk,3), got {canonical_kpts.shape}."
            )
        grid_mesh = tuple(int(x) for x in self.grid_mesh)
        if len(grid_mesh) != 3 or any(m <= 0 for m in grid_mesh):
            raise ValueError(f"grid_mesh must be 3 positive ints, got {grid_mesh}.")
        object.__setattr__(self, "canonical_kpts", canonical_kpts)
        object.__setattr__(self, "grid_mesh", grid_mesh)
        object.__setattr__(
            self, "coulG_all", precompute_coulG_all_q(self.cell, canonical_kpts, grid_mesh)
        )

    def apply(self, q_index, lq):
        n_kpts = self.canonical_kpts.shape[0]
        if not (0 <= q_index < n_kpts):
            raise ValueError(f"q_index={q_index} out of range for {n_kpts} k-points.")
        return raw_kernel_apply(
            lq, cell=self.cell, q_kpt=self.canonical_kpts[q_index], grid_mesh=self.grid_mesh
        )

    def fused_apply_and_solve(
        self, q_index, Pi_q, eta_q, phase_q, rtol, self_paired, retention_mode="single"
    ):
        n_kpts = self.canonical_kpts.shape[0]
        if not (0 <= q_index < n_kpts):
            raise ValueError(f"q_index={q_index} out of range for {n_kpts} k-points.")
        return _fused_apply_kernel_and_solve_core(
            Pi_q, eta_q, phase_q, self.coulG_all[q_index], self.grid_mesh, rtol, self_paired,
            retention_mode,
        )

    def provenance(self):
        return {
            "kernel_name": "raw",
            "kernel_version": 1,
            "g0_convention": "pyscf_get_coulG_exx_false",
            "grid_mesh": self.grid_mesh,
            "normalization": "vol_over_ng_inside_provider",
            "exxdiv": "owned_by_get_k_postprocessing_not_this_provider",
        }


@jax.jit
def _precompute_phase_all_q_core(grid_coords, canonical_kpts):
    """Jitted batched core: per-q Bloch phase exp(-1j * grid_coords @ q_kpt)
    for every q from a single grid_coords upload."""
    return jnp.exp(-1j * (grid_coords @ canonical_kpts.T)).T


def precompute_phase_all_q(grid_coords, canonical_kpts):
    """Compute the per-q Bloch phase for every q in one batched jitted
    call; slice per-q into apply_kernel_and_solve_device's phase_q.

    Returns:
        phase_all: (Nk, Ng) complex128 jax array.
    """
    grid_coords_np = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords_np.ndim != 2 or grid_coords_np.shape[1] != 3:
        raise ValueError(f"grid_coords must have shape (Ng,3), got {grid_coords_np.shape}.")
    canonical_kpts_np = np.asarray(canonical_kpts, dtype=np.float64)
    if canonical_kpts_np.ndim != 2 or canonical_kpts_np.shape[1] != 3:
        raise ValueError(
            f"canonical_kpts must have shape (Nk,3), got {canonical_kpts_np.shape}."
        )
    return _precompute_phase_all_q_core(
        jnp.asarray(grid_coords_np), jnp.asarray(canonical_kpts_np)
    )


def apply_kernel_and_solve_device(
    provider, q_index, Pi_q, eta_q, *, grid_coords=None, phase_q=None, rtol=1e-4,
    retained_solve_residual_gate=1e-10, self_paired=False, retention_mode="single",
):
    """S4 pipeline glue, device-resident, provider-agnostic: phase multiply
    -> provider.apply -> conjugate -> ZGEMM -> device Hermitian sandwich
    solve. Matches apply_raw_kernel_and_solve's math for RawKernelProvider.
    See design doc §6.

    Args:
        provider: object exposing .apply(q_index, lq) -> v_q (Nip, Ng).
        Pi_q: (Nip, Nip) complex128 metric.
        eta_q: (Nip, Ng) complex128 RHS, NOT yet phase-corrected.
        grid_coords: (Ng, 3); required only when phase_q is not given.
        phase_q: (Ng,) complex128 precomputed
            exp(-1j * grid_coords @ canonical_kpts[q_index]); callers
            looping over q should precompute via precompute_phase_all_q.
            Exactly one of grid_coords/phase_q must be given.
        rtol: forwarded to the device sandwich solve.
        retained_solve_residual_gate: HARD host-side gate on
            solve_info["retained_solve_residual"]; also hard-fails on
            n_retained == 0 (the jitted solve cannot raise on traced
            values, so degradation is turned into an error here).
        self_paired: True when neg[q]==q; Pi_q.real and kern_q.real are
            taken before the solve (see apply_raw_kernel_and_solve).
        retention_mode: "single" (default) or "pairwise" -- forwarded to
            hermitian_sandwich_solve_device / the fused core. See
            hermitian_sandwich_solve's docstring for the two modes.

    Returns:
        (W_q, kern_q, solve_info): W_q (Nip, Nip) complex128 jax array;
        kern_q the raw contracted kernel; solve_info the solve info dict.
    """
    from pytc.df.solvers import _solve_info_from_core_output, hermitian_sandwich_solve_device

    n_ip, n_grid = eta_q.shape

    if phase_q is None and grid_coords is None:
        raise ValueError("apply_kernel_and_solve_device: give one of grid_coords/phase_q.")
    if phase_q is not None:
        phase = jnp.asarray(phase_q, dtype=jnp.complex128)
        if phase.shape != (n_grid,):
            raise ValueError(
                f"phase_q must have shape ({n_grid},) matching eta_q's grid axis, "
                f"got {phase.shape}."
            )
    else:
        q_kpt = provider.canonical_kpts[q_index]
        grid_coords_np = np.asarray(grid_coords, dtype=np.float64)
        if grid_coords_np.shape != (n_grid, 3):
            raise ValueError(
                f"grid_coords must have shape ({n_grid},3) matching eta_q's grid axis, "
                f"got {grid_coords_np.shape}."
            )
        phase = jnp.exp(-1j * (jnp.asarray(grid_coords_np) @ jnp.asarray(q_kpt)))

    eta_q_jnp = jnp.asarray(eta_q, dtype=jnp.complex128)
    Pi_q_jnp = jnp.asarray(Pi_q, dtype=jnp.complex128)
    if self_paired:
        Pi_q_jnp = Pi_q_jnp.real.astype(jnp.complex128)

    # Providers with a fused fast path run the whole chain as one jax.jit
    # graph; others fall back to the eager per-stage path below.
    fused = getattr(provider, "fused_apply_and_solve", None)
    if fused is not None:
        (
            W_q_unscaled, kern_q, n_retained, s_max, s_min_retained,
            pi_anti_hermitian_residual, v_anti_hermitian_residual,
            retained_solve_residual, truncation_residual,
        ) = fused(q_index, Pi_q_jnp, eta_q_jnp, phase, rtol, self_paired, retention_mode)

        solve_info = _solve_info_from_core_output(
            n_retained, s_max, s_min_retained, pi_anti_hermitian_residual,
            v_anti_hermitian_residual, retained_solve_residual, truncation_residual,
            n_ip, W_q_unscaled.dtype, rtol, caller="apply_kernel_and_solve_device[fused]",
            retention_mode=retention_mode,
        )
    else:
        lq = eta_q_jnp * phase[None, :]

        v_q = provider.apply(q_index, lq)
        rq = jnp.conj(v_q)

        kern_q = (lq @ rq.T) / jnp.sqrt(n_grid)
        if self_paired:
            kern_q = kern_q.real.astype(jnp.complex128)

        W_q_unscaled, solve_info = hermitian_sandwich_solve_device(
            Pi_q_jnp, kern_q, rtol=rtol, retention_mode=retention_mode
        )

    # Host-side gate: the jitted solve cannot raise on a traced value, so
    # degeneracy becomes a precise, q-indexed error here (not a silent W_q=0).
    if solve_info["n_retained"] == 0:
        raise ValueError(
            f"apply_kernel_and_solve_device: q_index={q_index} retained ZERO modes of "
            f"Pi_q in the device sandwich solve (Pi_q is non-PSD, the zero matrix, or "
            f"rtol={rtol} is too large) -- W_q would be silently zero; refusing to "
            f"proceed. Validate Pi_q against the NumPy oracle (hermitian_sandwich_solve) "
            f"for a precise diagnosis."
        )
    if solve_info["retained_solve_residual"] > retained_solve_residual_gate:
        raise ValueError(
            f"apply_kernel_and_solve_device: q_index={q_index} retained-space solve "
            f"residual {solve_info['retained_solve_residual']:.3e} exceeds the hard "
            f"machine-tier gate {retained_solve_residual_gate:.1e} (design v2.1 section "
            f"5) -- this is a numerical sanity check on the eigendecomposition/solve "
            f"arithmetic itself, not the (separately reported, ungated here) truncation "
            f"residual; something is wrong with this q's Pi_q/kern_q inputs or dtype."
        )

    W_q = jnp.sqrt(n_grid) * W_q_unscaled
    return W_q, kern_q, solve_info


def build_coul_kpt_device(provider, Pi, eta, grid_coords, mesh_obj, *, rtol=1e-4,
                           retained_solve_residual_gate=1e-10, retention_mode="single"):
    """S4 orchestration: build coul_kpt (Nk, Nip, Nip) with one
    apply_kernel_and_solve_device call per unique {q, neg[q]} pair; the
    partner is set by exact conjugation (W[neg[q]] = conj(W[q]),
    kern[neg[q]] = conj(kern[q])). See design doc §6; verified against
    independent neg[q] builds in
    test_build_coul_kpt_device_conjugate_shortcut_matches_independent_build.

    Args:
        provider: KernelProvider built against mesh_obj.canonical_kpts.
        Pi: (Nk, Nip, Nip) complex128.
        eta: (Nk, Nip, Ng) complex128.
        grid_coords: (Ng, 3).
        mesh_obj: KptsMesh (uses .neg, .n_kpts).

    Returns:
        (coul_kpt, kern_kpt, infos, n_pipeline_calls): (Nk, Nip, Nip) jax
        arrays; length-Nk info list (a conjugated q shares its partner's
        dict); number of q's that actually ran the pipeline.
    """
    n_kpts = mesh_obj.n_kpts
    Pi = np.asarray(Pi)
    eta = np.asarray(eta)
    if Pi.shape[0] != n_kpts:
        raise ValueError(f"Pi.shape[0]={Pi.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")
    if eta.shape[0] != n_kpts:
        raise ValueError(f"eta.shape[0]={eta.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")

    # Precompute every q's Bloch phase once: per-q constant within one build.
    phase_all = precompute_phase_all_q(grid_coords, mesh_obj.canonical_kpts)

    neg = mesh_obj.neg
    coul_kpt = [None] * n_kpts
    kern_kpt = [None] * n_kpts
    infos = [None] * n_kpts
    done = [False] * n_kpts
    n_pipeline_calls = 0

    for q in range(n_kpts):
        if done[q]:
            continue
        nq = int(neg[q])
        W_q, kern_q, info_q = apply_kernel_and_solve_device(
            provider, q, Pi[q], eta[q], phase_q=phase_all[q], rtol=rtol,
            retained_solve_residual_gate=retained_solve_residual_gate,
            self_paired=(nq == q), retention_mode=retention_mode,
        )
        coul_kpt[q] = W_q
        kern_kpt[q] = kern_q
        infos[q] = info_q
        done[q] = True
        n_pipeline_calls += 1

        if nq != q and not done[nq]:
            coul_kpt[nq] = jnp.conj(W_q)
            kern_kpt[nq] = jnp.conj(kern_q)
            infos[nq] = info_q
            done[nq] = True

    return jnp.stack(coul_kpt, axis=0), jnp.stack(kern_kpt, axis=0), infos, n_pipeline_calls


# ---------------------------------------------------------------------------
# S1/S2 streaming (design v2.1 section 6): AO evaluation and the periodic
# pivot-selection metric oracle, both grid-block-streamed so host memory for
# either stays bounded by one block regardless of the full grid size Ng.


def stream_ao_blocks(cell, kpts, grid_coords, block_size, *, stats=None):
    """S1: stream AO values at kpts over grid_coords in blocks of
    block_size grid points; host memory stays bounded by one block.

    Yields:
        (g0, g1, ao_block): grid-index bounds [g0,g1) and the
        (Nk, g1-g0, Nao) complex128 block.
    """
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3:
        raise ValueError(f"grid_coords must have shape (Ng,3), got {grid_coords.shape}.")
    n_grid = grid_coords.shape[0]
    if n_grid == 0:
        raise ValueError("grid_coords must be nonempty.")

    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    kpts_list = list(np.asarray(kpts, dtype=np.float64))

    for g0 in range(0, n_grid, block_size):
        g1 = min(g0 + block_size, n_grid)
        ao_block = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[g0:g1], kpts=kpts_list), dtype=np.complex128,
        )
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + (g1 - g0)
        yield g0, g1, ao_block


def build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """S2 periodic pivot-selection metric oracle (design doc §3): a
    (diag, col_eval) pair for the reference-cell pair-density Gram matrix
    M[r,r'] = |sum_{k,mu} conj(AO_k(r,mu)) AO_k(r',mu)|^2 / Nk, never
    materialized. Each col_eval(j) costs a full streamed AO-grid sweep,
    so selecting `rank` pivots costs `rank` sweeps -- callers should
    account for this traffic.

    Returns:
        (diag, col_eval): diag (Ng,) float64; col_eval(j) -> (Ng,)
        complex128 (M is real-valued; complex128 only to match
        pivoted_cholesky_hermitian's contract).
    """
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    n_grid = grid_coords.shape[0]
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_kpts = kpts_np.shape[0]

    diag = np.empty(n_grid, dtype=np.float64)
    for g0, g1, ao_block in stream_ao_blocks(
        cell, kpts_np, grid_coords, block_size, stats=stats,
    ):
        pooled = np.sum(np.abs(ao_block) ** 2, axis=(0, 2))  # (blk,), sum_{k,mu} |AO_k(r,mu)|^2
        diag[g0:g1] = pooled ** 2 / n_kpts

    def col_eval(j):
        ao_j_block = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[j:j + 1], kpts=list(kpts_np)),
            dtype=np.complex128,
        )
        if stats is not None:
            stats["pbc_eval_calls"] = stats.get("pbc_eval_calls", 0) + 1
            stats["grid_points"] = stats.get("grid_points", 0) + 1
        ao_j = ao_j_block[:, 0, :]  # (Nk, Nao)

        col = np.empty(n_grid, dtype=np.complex128)
        for g0, g1, ao_block in stream_ao_blocks(
            cell, kpts_np, grid_coords, block_size, stats=stats,
        ):
            gram = np.einsum("km,krm->r", ao_j.conj(), ao_block, optimize=True)
            col[g0:g1] = (np.abs(gram) ** 2 / n_kpts).astype(np.complex128)
        return col

    return diag, col_eval


def periodic_metric_from_ao(ao):
    """Materialize the periodic metric for an explicit, bounded AO panel."""
    ao = np.asarray(ao, dtype=np.complex128)
    if ao.ndim != 3:
        raise ValueError(f"ao must have shape (Nk,Npanel,Nao), got {ao.shape}.")
    n_kpts, n_panel, _ = ao.shape
    if n_kpts == 0 or n_panel == 0:
        raise ValueError("ao must have nonempty k and panel axes.")
    flat = ao.transpose(1, 0, 2).reshape(n_panel, -1)
    return (np.abs(flat.conj() @ flat.T) ** 2 / n_kpts).astype(np.complex128)


def periodic_metric_column_from_ao(ao, index):
    """Return one periodic-metric column without materializing the metric."""
    ao = np.asarray(ao, dtype=np.complex128)
    if ao.ndim != 3:
        raise ValueError(f"ao must have shape (Nk,Npanel,Nao), got {ao.shape}.")
    n_kpts, n_panel, _ = ao.shape
    if n_kpts == 0 or n_panel == 0:
        raise ValueError("ao must have nonempty k and panel axes.")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer.")
    if not 0 <= index < n_panel:
        raise ValueError(f"index={index} is outside panel size {n_panel}.")
    gram = np.einsum("km,krm->r", ao[:, index, :].conj(), ao, optimize=True)
    return (np.abs(gram) ** 2 / n_kpts).astype(np.complex128)


def candidate_panel_indices(diag, rank, *, panel_factor=4, ramp_scale=1e-12):
    """Deterministically combine high-score and spatially stratified candidates."""
    diag = np.asarray(diag, dtype=np.float64)
    if diag.ndim != 1 or diag.size == 0 or not np.all(np.isfinite(diag)):
        raise ValueError("diag must be a nonempty 1-D array.")
    if np.any(diag < 0) or np.max(diag) <= 0:
        raise ValueError("diag must be nonnegative with a positive maximum.")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer.")
    if isinstance(panel_factor, bool) or not isinstance(panel_factor, (int, np.integer)):
        raise ValueError("panel_factor must be an integer.")
    if rank <= 0 or panel_factor < 2:
        raise ValueError("rank must be positive and panel_factor must be at least 2.")
    n_grid = diag.size
    panel_size = min(n_grid, panel_factor * rank)
    score = diag + ramp_scale * np.arange(n_grid) * float(np.max(diag))
    order = np.argsort(score, kind="stable")[::-1]
    high_count = min(panel_size // 2, n_grid)
    selected = list(order[:high_count])
    bins = np.array_split(np.arange(n_grid), panel_size - high_count)
    for indices in bins:
        selected.append(indices[np.argmax(score[indices])])
    unique = []
    seen = set()
    for index in selected + list(order):
        index = int(index)
        if index not in seen:
            unique.append(index)
            seen.add(index)
        if len(unique) == panel_size:
            break
    return np.asarray(unique, dtype=np.int64)


def full_grid_candidate_identity(n_grid):
    """Return a compact identity for the complete grid candidate set."""
    if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)) or n_grid <= 0:
        raise ValueError("n_grid must be a positive integer.")
    return {"kind": "range", "start": 0, "stop": int(n_grid), "step": 1}


def explicit_candidate_identity(indices):
    """Return the persisted identity for a bounded explicit candidate panel."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array.")
    return {"kind": "explicit_indices", "indices": indices.tolist()}


def build_cached_periodic_pivot_oracle(cell, kpts, grid_coords, block_size, *, stats=None):
    """Experimental full-cache oracle with the same metric as the streamed path."""
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_grid = len(grid_coords)
    n_kpts = len(kpts_np)
    cache = None
    for g0, g1, ao_block in stream_ao_blocks(
        cell, kpts_np, grid_coords, block_size, stats=stats,
    ):
        if cache is None:
            cache = np.empty((n_kpts, n_grid, ao_block.shape[2]), dtype=np.complex128)
        cache[:, g0:g1] = ao_block
    pooled = np.sum(np.abs(cache) ** 2, axis=(0, 2))
    diag = pooled ** 2 / n_kpts
    return diag, lambda j: periodic_metric_column_from_ao(cache, j), cache


# ---------------------------------------------------------------------------
# Staging-policy layer (design doc §6): predicted-byte-model-driven selection
# among ram/memmap/recompute eta-store policies, plus the mechanics.
# Byte model is PREDICTED ONLY; observed fields exist in the schema as
# None/"unmeasured" so a later calibration pass backfills without a schema
# change. Real resource queries live only in query_host_resources.


def predicted_byte_model(n_kpts, n_ip, n_grid, n_ao, block_size, *, itemsize=16):
    """Closed-form predicted byte counts for one build (c128, itemsize=16);
    nothing here is measured. Returns a dict of the per-component byte
    terms plus total_predicted_bytes (eta_store + selection_traffic).
    See design doc §6."""
    for name, value in (
        ("n_kpts", n_kpts), ("n_ip", n_ip), ("n_grid", n_grid),
        ("n_ao", n_ao), ("block_size", block_size),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    ao_grid_block_bytes = n_kpts * block_size * n_ao * itemsize
    eta_store_bytes = n_kpts * n_ip * n_grid * itemsize
    double_buffer_bytes = 2 * n_ip * block_size * itemsize
    fft_workspace_bytes = 2 * n_ip * n_grid * itemsize
    pi_v_w_workspace_bytes = (n_kpts * n_ip * n_ip + n_ip * n_ip) * itemsize
    selection_traffic_bytes = n_ip * n_kpts * n_grid * n_ao * itemsize
    total_predicted_bytes = eta_store_bytes + selection_traffic_bytes

    return {
        "ao_grid_block_bytes": ao_grid_block_bytes,
        "eta_store_bytes": eta_store_bytes,
        "double_buffer_bytes": double_buffer_bytes,
        "fft_workspace_bytes": fft_workspace_bytes,
        "pi_v_w_workspace_bytes": pi_v_w_workspace_bytes,
        "selection_traffic_bytes": selection_traffic_bytes,
        "total_predicted_bytes": total_predicted_bytes,
    }


def choose_staging_policy(byte_model, *, available_host_bytes, available_disk_bytes,
                           ram_headroom_fraction=0.5, disk_headroom_fraction=0.9):
    """Select the eta-store staging policy from a predicted byte model and
    caller-supplied resource numbers (never queried internally, so this
    stays synthetic-input testable).

    Rule: "ram" if eta_store_bytes <= ram_headroom_fraction *
    available_host_bytes; else "memmap" if it fits the disk headroom;
    else "recompute".

    Returns a provenance dict; observed_peak_host_bytes/observed_status
    are always None/"unmeasured" here so a later calibration pass
    backfills the SAME schema.
    """
    eta_bytes = byte_model["eta_store_bytes"]
    if eta_bytes < 0:
        raise ValueError(f"byte_model['eta_store_bytes'] must be non-negative, got {eta_bytes}.")
    for name, value in (
        ("available_host_bytes", available_host_bytes),
        ("available_disk_bytes", available_disk_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    for name, value in (
        ("ram_headroom_fraction", ram_headroom_fraction),
        ("disk_headroom_fraction", disk_headroom_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < value <= 1.0):
            raise ValueError(f"{name} must be in (0,1], got {value!r}.")

    ram_headroom_bytes = int(ram_headroom_fraction * available_host_bytes)
    disk_headroom_bytes = int(disk_headroom_fraction * available_disk_bytes)

    if eta_bytes <= ram_headroom_bytes:
        policy = "ram"
    elif eta_bytes <= disk_headroom_bytes:
        policy = "memmap"
    else:
        policy = "recompute"

    return {
        "policy": policy,
        "eta_store_bytes": eta_bytes,
        "ram_headroom_bytes": ram_headroom_bytes,
        "disk_headroom_bytes": disk_headroom_bytes,
        "ram_headroom_fraction": ram_headroom_fraction,
        "disk_headroom_fraction": disk_headroom_fraction,
        "available_host_bytes": available_host_bytes,
        "available_disk_bytes": available_disk_bytes,
        "observed_peak_host_bytes": None,
        "observed_status": "unmeasured",
    }


def query_host_resources(scratch_dir="."):
    """The one place this module reads real host/disk resource numbers
    (psutil / shutil.disk_usage); pass the result to choose_staging_policy.

    Returns:
        (available_host_bytes, available_disk_bytes): both int.
    """
    import shutil

    import psutil

    available_host_bytes = int(psutil.virtual_memory().available)
    available_disk_bytes = int(shutil.disk_usage(scratch_dir).free)
    return available_host_bytes, available_disk_bytes


def jit_memory_analysis_smoke(jitted_fn, *args):
    """Compile jitted_fn against args (shapes/dtypes only; not executed)
    and return its jax CompiledMemoryStats as a dict, labeled by the
    actual backend (CPU stats are a structural smoke, not HBM data).
    See design doc §6."""
    backend = jax.default_backend()
    label = (
        "cpu_backend_structural_smoke_not_hbm"
        if backend == "cpu"
        else "gpu_backend_compiled_memory_stats"
    )
    stats = jitted_fn.lower(*args).compile().memory_analysis()
    return {
        "backend": backend,
        "label": label,
        "generated_code_size_in_bytes": stats.generated_code_size_in_bytes,
        "argument_size_in_bytes": stats.argument_size_in_bytes,
        "output_size_in_bytes": stats.output_size_in_bytes,
        "alias_size_in_bytes": stats.alias_size_in_bytes,
        "temp_size_in_bytes": stats.temp_size_in_bytes,
        "host_generated_code_size_in_bytes": stats.host_generated_code_size_in_bytes,
        "host_argument_size_in_bytes": stats.host_argument_size_in_bytes,
        "host_output_size_in_bytes": stats.host_output_size_in_bytes,
        "host_alias_size_in_bytes": stats.host_alias_size_in_bytes,
        "host_temp_size_in_bytes": stats.host_temp_size_in_bytes,
    }


def stage_eta_memmap(eta_chunks_iter, shape, memmap_path):
    """memmap staging (policy 2): write streamed eta chunks into an
    np.memmap at memmap_path without holding the full (Nk,Nip,Ng) eta in
    RAM.

    Args:
        eta_chunks_iter: iterable of (g0, g1, chunk), chunk
            (Nk,Nip,g1-g0) complex128, covering [0,Ng) in order.
        shape: (Nk,Nip,Ng).

    Returns:
        np.memmap, dtype complex128, flushed to disk.
    """
    shape_t = tuple(int(x) for x in shape)
    if len(shape_t) != 3 or any(s <= 0 for s in shape_t):
        raise ValueError(f"shape must be 3 positive ints, got {shape_t}.")

    mm = np.memmap(memmap_path, dtype=np.complex128, mode="w+", shape=shape_t)
    for g0, g1, chunk in eta_chunks_iter:
        mm[:, :, g0:g1] = chunk
    mm.flush()
    return mm


def stage_eta_recompute_tile(X, ao_block_source, phase, neg, q_slice=None):
    """recompute staging (policy 3): rebuild eta on demand via build_pi_eta,
    with ao_block_source() returning a FRESH iterable of (Nk,blk,Nao)
    blocks on every call. q_slice is applied to the leading (Nk) axis
    AFTER the full build (the complete pass still runs each call).

    Returns:
        (Pi, eta): same as build_pi_eta, optionally sliced by q_slice.
    """
    Pi, eta = build_pi_eta(X, ao_block_source(), phase, neg)
    if q_slice is not None:
        return Pi[q_slice], eta[q_slice]
    return Pi, eta
