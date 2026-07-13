"""Periodic ISDF fit machinery (task #21, #proj-isdf-periodic, design
v2.1 sections 3-5): a generic, matrix-free Hermitian-PSD pivoted
Cholesky selector, the Pi^q/eta^q metric/RHS builders, and a minimal
plain-NumPy raw-kernel-apply-and-solve function needed to close the V2
reference-replay gate at the CPU/NumPy oracle level (Flinn's ruling,
task #21 thread, 2026-07-13: correctness oracles must close before
device work starts, so B1 owns this; the formal KernelProvider class
protocol -- device/jit path, pluggable ibp slot, provenance dict -- is
Phase C's job and will formalize/wrap this function, not replace it).

Design decision (Alice's option (a), design v2.1 section 3): the
existing pytc.df.pivots molecular pair core
(_pivoted_cholesky_pair_pivots_core) is one JIT with semantics far
richer than a (diag, col_eval) skeleton -- dual Cholesky states,
normalized/legacy tie-break ramps, a latched effective-rank prefix, and
dtype/rtol-coupled safety thresholds. Extract-and-rewrap cannot be
byte-identical in behavior or execution placement, so it is not
attempted. This module is a genuinely SEPARATE, simpler generic
primitive; pytc/df/pivots.py is not imported and not modified. The
argmax tie-break RULE (a tiny monotonically increasing ramp added to
the score before argmax, biasing ties toward the higher index) is
copied by inspection from pytc.df.pivots's own selector, not shared by
import -- see pivoted_cholesky_hermitian's docstring.
"""

from __future__ import annotations

import dataclasses
import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def pivoted_cholesky_hermitian(diag, col_eval, rank, *, rcond=1e-12, ramp_scale=1e-12):
    """Matrix-free pivoted (partial) Cholesky for an implicit N x N
    Hermitian PSD matrix M, given only its diagonal and an on-demand
    column oracle col_eval(j) -> M[:, j] (shape (N,), complex128). M
    itself is never materialized.

    Standard greedy pivoted-Cholesky / low-rank PSD approximation
    algorithm: at each step, select the largest remaining Schur-
    complement diagonal entry as the next pivot, fetch that column of
    the ORIGINAL matrix via col_eval, subtract off the already-selected
    pivots' contribution, and normalize by sqrt(pivot diagonal) to get
    the next column of the Cholesky factor L. The Schur-complement
    diagonal is updated by subtracting |l_t|^2 after each step and
    guarded to stay >= 0 (real, per the design spec) despite roundoff.

    Tie-break rule (copied from pytc.df.pivots's molecular selector, by
    inspection -- not shared code, not an import): a tiny monotonically
    increasing ramp `ramp_scale * arange(n) * max(diag)` is added to the
    score used for argmax, biasing an exact numerical tie toward the
    HIGHER index -- deterministic and reproducible, matching the
    molecular selector's own convention, rather than depending on
    argmax's otherwise implementation-defined first-max behavior.

    Args:
        diag: (n,) real, non-negative (guarded) diagonal of M.
        col_eval: callable, col_eval(j) -> (n,) complex128 array, the
            j-th column of the ORIGINAL M (not the Schur complement).
        rank: requested number of pivots (upper bound; may return fewer
            if the Schur-complement diagonal is numerically exhausted
            first).
        rcond: relative threshold (vs max(diag)) below which a
            candidate pivot's Schur-complement diagonal is treated as
            numerically zero -- selection stops there.
        ramp_scale: tie-break ramp coefficient (see above).

    Returns:
        (pivots, L, n_selected):
            pivots: (n_selected,) int64 array of selected column indices.
            L: (n, n_selected) complex128 array, the partial Cholesky
                factor restricted to selected columns (M[pivots,pivots]
                block satisfies L[pivots,:] @ L[pivots,:].conj().T ==
                M[pivots,pivots] to numerical precision; full
                reconstruction is L @ L.conj().T approx M when rank is
                sufficient).
            n_selected: int, <= rank.

    Raises:
        ValueError: rank > n, diag has a materially negative entry
            (M is not PSD within rcond), or col_eval returns a
            malformed shape/dtype.
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


def build_pi_eta(X, ao_blocks, kmesh, *, imag_tol=1e-10):
    """Build the per-q metric Pi^q and RHS eta^q (design v2.1 section 4/5):

        Pi^q  = pair_convolve(X, X, kmesh)[q]         (Nip, Nip)
        eta^q = pair_convolve(X, AO, kmesh)[q]         (Nip, Ng)

    eta is accumulated over grid blocks by calling pair_convolve once
    per block and concatenating along the grid axis -- this bounds the
    memory of any single pair_convolve call to one block's worth of AO
    data, at the cost of re-walking X's own per-q GEMM/FFT machinery
    once per block (the same reference-first, optimize-later posture as
    this module's other primitives; a genuinely fused/tiled device
    pipeline is Phase C's job, not this CPU oracle's).

    Args:
        X: (Nk, Nip, Nao) complex128 -- the interpolation-point factor
            (e.g. AO or MO values at the selected pivot points) across
            the canonical k-mesh.
        ao_blocks: a single (Nk, Ng, Nao) complex128 array, or an
            iterable of (Nk, blk_i, Nao) complex128 arrays (AO values at
            successive grid blocks) across the SAME canonical k-mesh.
        kmesh: (3,) positive ints, the canonical k/q-mesh shape (see
            pytc.pbc.df.kpts.KptsMesh.kmesh).
        imag_tol: forwarded to pair_convolve's imaginary-part gate.

    Returns:
        (Pi, eta): Pi is (Nk, Nip, Nip) complex128; eta is
        (Nk, Nip, Ng) complex128, Ng = sum of the ao_blocks' grid sizes.

    Raises:
        ValueError: forwarded from pair_convolve for malformed shapes,
            or if X/ao_blocks are not a valid time-reversal-symmetric
            pair (the imag_tol gate).
    """
    # Local import: pytc.pbc.df.isdf depends on pytc.pbc.df.kpts (both
    # are pbc/df/ peers), never the reverse -- kpts.py stays a leaf.
    from pytc.pbc.df.kpts import pair_convolve

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (Nk, Nip, Nao), got shape {X.shape}.")

    Pi = pair_convolve(X, X, kmesh, imag_tol=imag_tol)

    if isinstance(ao_blocks, np.ndarray):
        ao_blocks = [ao_blocks]
    else:
        ao_blocks = list(ao_blocks)
    if not ao_blocks:
        raise ValueError("ao_blocks must be nonempty.")

    eta_chunks = [
        pair_convolve(X, np.asarray(block), kmesh, imag_tol=imag_tol)
        for block in ao_blocks
    ]
    eta = np.concatenate(eta_chunks, axis=2)
    return Pi, eta


def apply_raw_kernel_and_solve(Pi_q, eta_q, *, cell, q_kpt, grid_coords, grid_mesh, rtol=1e-8):
    """Apply the "raw" (bare 4pi/G^2, exx=False) periodic Coulomb kernel
    to eta^q over the SPATIAL grid, contract back into a Nip x Nip
    kernel matrix, and solve the Hermitian sandwich for W^q (design
    v2.1 sections 5+7). Plain NumPy, single q at a time -- a minimal
    function proving the V0-V6 "raw" provider semantics at the CPU/
    NumPy oracle level; the formal KernelProvider class/device path is
    Phase C's job (see module docstring).

        lq   = eta_q * exp(-1j * grid_coords @ q_kpt)   # Bloch-phase
                                                          # correction:
            eta^q as built by pair_convolve carries only the k-INDEX
            Alg-1 phase machinery, not the grid-coordinate-dependent
            Bloch phase a per-q spatial-grid FFT needs -- this factor
            supplies it.
        wq   = FFT(lq, grid_mesh)                        # spatial FFT,
                                                          # one row per
                                                          # interpolation
                                                          # point.
        vq   = cell.get_coulG(q_kpt, exx=False, mesh=grid_mesh)
               * cell.vol / Ng                            # bare kernel,
                                                          # G=0 handled
                                                          # by pyscf's
                                                          # own exx=False
                                                          # convention.
        rq   = conj(IFFT(wq * vq, grid_mesh))             # spatial
                                                          # IFFT, then
                                                          # conjugate
                                                          # (matches the
                                                          # bare-kernel
                                                          # convention
                                                          # verified
                                                          # against a
                                                          # real
                                                          # reference
                                                          # dump -- see
                                                          # the V2
                                                          # reference-
                                                          # replay
                                                          # test).
        kern_q = lq @ rq.T / sqrt(Ng)                     # (Nip, Nip).
        W_q  = sqrt(Ng) * hermitian_sandwich_solve(Pi_q, kern_q)[0]
                                                          # the final
                                                          # sqrt(Ng)
                                                          # rescale
                                                          # exactly
                                                          # cancels
                                                          # kern_q's own
                                                          # 1/sqrt(Ng),
                                                          # per the
                                                          # paper's Eq.
                                                          # 10 factor
                                                          # list
                                                          # (section 4:
                                                          # "vol/Ng,
                                                          # 1/sqrt(Ng),
                                                          # sqrt(Ng)
                                                          # factor
                                                          # placements")
                                                          # -- empirically
                                                          # confirmed
                                                          # against a
                                                          # real fftisdf
                                                          # run on he2
                                                          # [1,1,3]: this
                                                          # exact
                                                          # rescale
                                                          # closes W_q to
                                                          # ~1e-9
                                                          # relative
                                                          # error (see
                                                          # the V2
                                                          # reference-
                                                          # replay
                                                          # test).

    exxdiv is NEVER applied here (per section 4/7: exxdiv ownership
    belongs to a later get_k post-processing step, never inside a
    kernel provider) -- this function only ever produces the bare
    kernel.

    Args:
        Pi_q: (Nip, Nip) complex128, this q's metric (e.g. from
            build_pi_eta).
        eta_q: (Nip, Ng) complex128, this q's RHS (e.g. from
            build_pi_eta), Ng matching grid_coords/grid_mesh.
        cell: pyscf.pbc.gto.Cell (needed for cell.vol and
            cell.get_coulG).
        q_kpt: (3,) absolute k-vector for this q (e.g.
            KptsMesh.canonical_kpts[q]).
        grid_coords: (Ng, 3) real-space grid point coordinates, in the
            SAME flattened order as eta_q's grid axis.
        grid_mesh: (3,) positive ints, the REAL-SPACE integration mesh
            shape (a DIFFERENT mesh from the k-point mesh) -- prod
            must equal Ng.
        rtol: forwarded to hermitian_sandwich_solve.

    Returns:
        (W_q, kern_q, solve_info): W_q is (Nip, Nip) complex128 (the
        solved kernel matrix); kern_q is the raw (Nip, Nip) contracted
        kernel before the sandwich solve; solve_info is
        hermitian_sandwich_solve's own info dict.

    Raises:
        ValueError: malformed shapes, or grid_mesh does not match Ng.
    """
    # Local imports: pytc.pbc.df.isdf depends on pytc.df.solvers (a
    # pytc/df/ peer, per the design's dependency direction) and pyscf's
    # own reciprocal-lattice tool, never the reverse.
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

    W_q_unscaled, solve_info = hermitian_sandwich_solve(Pi_q, kern_q, rtol=rtol)
    # Final sqrt(Ng) rescale, exactly cancelling kern_q's own 1/sqrt(Ng) --
    # see the docstring's Eq. 10 factor-placement note; empirically
    # confirmed against a real fftisdf run (V2 reference-replay test).
    W_q = np.sqrt(n_grid) * W_q_unscaled
    return W_q, kern_q, solve_info


# ---------------------------------------------------------------------------
# C1 (task #24, design v2.1 section 7): device path, KernelProvider protocol.
#
# Flinn's seam ruling (task #24 thread, msg 6916c47c) on top of the
# apply_raw_kernel_and_solve oracle above: KernelProvider.apply(q_index, lq)
# is a LINEAR q-momentum kernel operator applied to a PRE-PHASED (Nip, Ng)
# slab, returning v_q BEFORE the final conjugate -- the Bloch-phase multiply
# (eta_q -> lq) and the outer conjugate (v_q -> rq) are pipeline glue that
# does not vary between providers, not part of the provider contract. The
# vol/Ng normalization stays INSIDE the provider (it is part of "kernel on
# the canonical mesh"); exxdiv stays OUTSIDE, unchanged, owned by a later
# get_k post-processing step. "Raw" implements the operator in 3 Fourier
# passes; the contract itself does not fix a step count (a future IBP
# provider may need several FFT/IFFT passes for gradient/vector-valued
# G-space components inside one apply() call).
#
# q<->-q dagger law AT THIS SEAM (derived here, not copied from the original
# eta_q-based law in the design doc text, since that law was written for the
# phase-included signature): given
#     l_q[neg[q]] = conj(l_q[q])
# (true because eta[neg[q]] = conj(eta[q]) from pair_convolve's own q<->-q
# closure, and phase(neg[q]) = exp(-1j r.(-q)) = exp(+1j r.q) = conj(phase(q))
# for phase(k) = exp(-1j r.k)), the raw kernel-apply operator satisfies
#     apply(neg[q], conj(l_q)) == conj(apply(q, l_q))
# Proof: (1) coulG(-q)[G] = coulG(q)[-G] -- both equal 4pi/|G+-q|^2, and this
# value is REAL, so it equals its own conjugate; (2) the standard DFT
# conjugate-reversal identity FFT(conj(f))[G] = conj(FFT(f)[-G]); composing
# (1)+(2) through the coulG multiply, then one more IFFT (which turns a
# G-reversal + conjugate back into a plain conjugate in real/grid space, by
# the same identity applied in reverse), closes the law with no leftover
# phase. Verified numerically in test_raw_kernel_apply_dagger_law.


@partial(jax.jit, static_argnames=("grid_mesh",))
def _raw_kernel_apply_core(lq, coulG_scaled, grid_mesh):
    """Jitted, fixed-shape core of the "raw" KernelProvider: v_q =
    IFFT(coulG(q)*vol/Ng * FFT(lq)), device-resident throughout. See
    raw_kernel_apply's docstring for the full contract; this function
    does no validation (that lives in the host-side wrapper, since
    validation involves cell/pyscf calls that are not jittable) and
    performs no phase multiply and no outer conjugate -- both are
    pipeline glue applied by the caller, not this seam.
    """
    n_ip = lq.shape[0]
    lq_mesh = lq.reshape((n_ip,) + grid_mesh)
    wq_mesh = jnp.fft.fftn(lq_mesh, axes=(1, 2, 3))
    vq_mesh = jnp.asarray(coulG_scaled, dtype=lq.dtype).reshape(grid_mesh)
    vq_mesh = wq_mesh * vq_mesh[None, :, :, :]
    rq_mesh = jnp.fft.ifftn(vq_mesh, axes=(1, 2, 3))
    return rq_mesh.reshape(n_ip, -1)


def raw_kernel_apply(lq, *, cell, q_kpt, grid_mesh):
    """Host-side wrapper: validate inputs, compute coulG(q)*vol/Ng on the
    host via pyscf (not jittable -- cell.get_Gv/get_coulG are plain
    Python/NumPy pyscf calls), then dispatch to the jitted device core.

    Args:
        lq: (Nip, Ng) complex128, ALREADY Bloch-phase-corrected (the
            pipeline's job, not this function's -- see module-level
            comment above).
        cell: pyscf.pbc.gto.Cell.
        q_kpt: (3,) absolute k-vector for this q.
        grid_mesh: (3,) positive ints, the real-space integration mesh
            (prod must equal Ng).

    Returns:
        v_q: (Nip, Ng) complex128 jax array, BEFORE the outer conjugate
        (the pipeline applies conj(v_q) -> rq itself).

    Raises:
        ValueError: malformed shapes, or grid_mesh does not match Ng.
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


@dataclasses.dataclass(frozen=True)
class RawKernelProvider:
    """The "raw" (bare 4pi/G^2, exx=False) KernelProvider (design v2.1
    section 7): wraps raw_kernel_apply behind the KernelProvider seam
    (apply(q_index, lq) -> v_q) plus a provenance() accessor. A provider
    is a per-(cell, canonical k-mesh, grid) object -- q_index looks up
    the absolute k-vector from canonical_kpts internally, so callers
    never pass raw k-vectors across the provider boundary.

    Args:
        cell: pyscf.pbc.gto.Cell.
        canonical_kpts: (Nk, 3) float64, e.g. KptsMesh.canonical_kpts --
            canonical_kpts[q_index] is the absolute k-vector for that q.
        grid_mesh: (3,) positive ints, the real-space integration mesh.
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

    def apply(self, q_index, lq):
        n_kpts = self.canonical_kpts.shape[0]
        if not (0 <= q_index < n_kpts):
            raise ValueError(f"q_index={q_index} out of range for {n_kpts} k-points.")
        return raw_kernel_apply(
            lq, cell=self.cell, q_kpt=self.canonical_kpts[q_index], grid_mesh=self.grid_mesh
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


def apply_kernel_and_solve_device(provider, q_index, Pi_q, eta_q, *, grid_coords, rtol=1e-8):
    """S4 pipeline glue (design v2.1 section 6), device-resident,
    provider-agnostic: Bloch-phase multiply -> provider.apply(q_index,
    lq) -> outer conjugate -> ZGEMM contract to (Nip,Nip) -> device
    Hermitian sandwich solve. Reproduces apply_raw_kernel_and_solve's
    exact math when provider is a RawKernelProvider (validated to
    <=1e-12 in test_apply_kernel_and_solve_device_matches_numpy_oracle),
    but is written against the provider SEAM so a future ibp provider
    plugs in here unchanged.

    Args:
        provider: object exposing .apply(q_index, lq) -> v_q (Nip, Ng),
            e.g. RawKernelProvider.
        q_index: int, canonical index of this q (looked up by the
            provider itself for any provider-internal k-vector needs).
        Pi_q: (Nip, Nip) complex128, this q's metric.
        eta_q: (Nip, Ng) complex128, this q's RHS, NOT yet phase-
            corrected (the pipeline's job, matching build_pi_eta's raw
            output).
        grid_coords: (Ng, 3) real-space grid point coordinates, same
            flattened order as eta_q's grid axis.
        rtol: forwarded to the device sandwich solve.

    Returns:
        (W_q, kern_q, solve_info): W_q is (Nip, Nip) complex128 jax
        array (the solved kernel matrix); kern_q is the raw (Nip, Nip)
        contracted kernel before the sandwich solve; solve_info is the
        device sandwich solve's own info dict.
    """
    from pytc.df.solvers import hermitian_sandwich_solve_device

    q_kpt = provider.canonical_kpts[q_index]
    eta_q_np = np.asarray(eta_q)
    n_ip, n_grid = eta_q_np.shape
    grid_coords_np = np.asarray(grid_coords, dtype=np.float64)
    if grid_coords_np.shape != (n_grid, 3):
        raise ValueError(
            f"grid_coords must have shape ({n_grid},3) matching eta_q's grid axis, "
            f"got {grid_coords_np.shape}."
        )

    phase = jnp.exp(-1j * (jnp.asarray(grid_coords_np) @ jnp.asarray(q_kpt)))
    lq = jnp.asarray(eta_q_np, dtype=jnp.complex128) * phase[None, :]

    v_q = provider.apply(q_index, lq)
    rq = jnp.conj(v_q)

    kern_q = (lq @ rq.T) / jnp.sqrt(n_grid)

    Pi_q_jnp = jnp.asarray(Pi_q, dtype=jnp.complex128)
    W_q_unscaled, solve_info = hermitian_sandwich_solve_device(Pi_q_jnp, kern_q, rtol=rtol)
    W_q = jnp.sqrt(n_grid) * W_q_unscaled
    return W_q, kern_q, solve_info
