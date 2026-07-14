"""Periodic ISDF fit machinery (design v2.1 sections 3-5): a generic,
matrix-free Hermitian-PSD pivoted Cholesky selector, the Pi^q/eta^q
metric/RHS builders, and a minimal plain-NumPy raw-kernel-apply-and-solve
function needed to close the V2 reference-replay gate at the CPU/NumPy
oracle level (correctness oracles close before device work starts; the
formal KernelProvider class protocol -- device/jit path, pluggable ibp
slot, provenance dict -- formalizes/wraps this function below, not
replacing it).

Design decision (design v2.1 section 3): the existing pytc.df.pivots
molecular pair core
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


def build_pi_eta(X, ao_blocks, phase, *, imag_tol=1e-10):
    """Build the per-q metric Pi^q and RHS eta^q (design v2.1 section 4/5):

        Pi^q  = pair_convolve(X, X, phase)[q]         (Nip, Nip)
        eta^q = pair_convolve(X, AO, phase)[q]         (Nip, Ng)

    eta is accumulated over grid blocks by calling pair_convolve once
    per block and concatenating along the grid axis -- this bounds the
    memory of any single pair_convolve call to one block's worth of AO
    data, at the cost of re-walking X's own per-q GEMM/transform
    machinery once per block (the same reference-first, optimize-later
    posture as this module's other primitives; a genuinely fused/tiled
    device pipeline is Phase C's job, not this CPU oracle's).

    Args:
        X: (Nk, Nip, Nao) complex128 -- the interpolation-point factor
            (e.g. AO or MO values at the selected pivot points) across
            the canonical k-mesh.
        ao_blocks: a single (Nk, Ng, Nao) complex128 array, or an
            iterable of (Nk, blk_i, Nao) complex128 arrays (AO values at
            successive grid blocks) across the SAME canonical k-mesh.
        phase: (Nk, Nk) complex128 unitary k<->supercell-image transform
            matrix (see pytc.pbc.df.kpts.KptsMesh.phase).
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

    Pi = pair_convolve(X, X, phase, imag_tol=imag_tol)

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
    eta = np.concatenate(eta_chunks, axis=2)
    return Pi, eta


def apply_raw_kernel_and_solve(Pi_q, eta_q, *, cell, q_kpt, grid_coords, grid_mesh, rtol=1e-4):
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
# Device path, KernelProvider protocol (design v2.1 section 7).
#
# On top of the apply_raw_kernel_and_solve oracle above: KernelProvider.apply
# (q_index, lq) is a LINEAR q-momentum kernel operator applied to a
# PRE-PHASED (Nip, Ng) slab, returning v_q BEFORE the final conjugate -- the
# Bloch-phase multiply
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


def apply_kernel_and_solve_device(
    provider, q_index, Pi_q, eta_q, *, grid_coords, rtol=1e-4,
    retained_solve_residual_gate=1e-10,
):
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
        retained_solve_residual_gate: HARD host-side gate (design v2.1
            section 5: "machine-tier, HARD gate <=1e-10 at c128") on
            solve_info["retained_solve_residual"], checked AFTER the
            jitted solve returns. Also hard-fails if n_retained == 0.
            hermitian_sandwich_solve_device cannot raise from inside its
            own jax.jit graph on a traced value (n_retained==0 there
            silently degrades to W=0), so THIS host wrapper is
            responsible for turning that degradation into a precise
            error at the source -- not a confusing physics mismatch
            surfacing downstream in a consumer-level parity gate.

    Returns:
        (W_q, kern_q, solve_info): W_q is (Nip, Nip) complex128 jax
        array (the solved kernel matrix); kern_q is the raw (Nip, Nip)
        contracted kernel before the sandwich solve; solve_info is the
        device sandwich solve's own info dict.

    Raises:
        ValueError: malformed shapes, or (post-solve) n_retained == 0 or
            solve_info["retained_solve_residual"] exceeds
            retained_solve_residual_gate -- both include q_index in the
            message.
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

    # Host-side gate: the jitted solve cannot raise on a traced value,
    # so degeneracy is turned into a precise, q-indexed error HERE
    # rather than surfacing as a silent W_q=0 that would only be caught
    # downstream at a consumer-level parity gate.
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
                           retained_solve_residual_gate=1e-10):
    """S4 orchestration (design v2.1 section 6): build the full coul_kpt
    array (Nk, Nip, Nip) by calling apply_kernel_and_solve_device once
    per UNIQUE {q, neg[q]} pair, exploiting the q<->-q conjugate closure
    to halve the FFT/coulG/solve work -- "S4 streams per-q (and its
    neg[q] partner, processed together to exploit the conjugate
    relation)".

    For self-paired q (neg[q] == q -- Gamma, and any other BZ-edge
    self-paired point), the pipeline runs directly. For a genuine pair
    {q, neg[q]} with neg[q] != q, only the LOWER-indexed member runs
    through the pipeline; the other is set by direct conjugation:
        W[neg[q]]    = conj(W[q])
        kern[neg[q]] = conj(kern[q])
    This is an EXACT identity, not an approximation -- it follows from
    Pi[neg[q]] = conj(Pi[q]) and eta[neg[q]] = conj(eta[q])
    (pair_convolve's own q<->-q closure) propagating through: lq[neg[q]]
    = conj(lq[q]) (module-level comment above), kern[neg[q]] =
    conj(kern[q]) (same derivation extended one ZGEMM further), and
    hermitian_sandwich_solve's W = Pi^+ V Pi^+ conjugating cleanly
    because eigh(conj(M)) has the same eigenvalues with conjugated
    eigenvectors. Verified against INDEPENDENTLY computed neg[q] builds
    (not just self-consistency) in
    test_build_coul_kpt_device_conjugate_shortcut_matches_independent_build.

    Args:
        provider: KernelProvider (e.g. RawKernelProvider), already
            constructed against mesh_obj.canonical_kpts/grid_mesh.
        Pi: (Nk, Nip, Nip) complex128, e.g. from build_pi_eta.
        eta: (Nk, Nip, Ng) complex128, e.g. from build_pi_eta.
        grid_coords: (Ng, 3) real-space grid point coordinates.
        mesh_obj: pytc.pbc.df.kpts.KptsMesh (uses .neg, .n_kpts).
        rtol: forwarded to the device sandwich solve.
        retained_solve_residual_gate: forwarded to
            apply_kernel_and_solve_device.

    Returns:
        (coul_kpt, kern_kpt, infos, n_pipeline_calls): coul_kpt/kern_kpt
        are (Nk, Nip, Nip) jax arrays; infos is a length-Nk list of solve
        info dicts (a conjugated q shares its pair partner's dict object
        -- no independent solve ran for it, so there is no separate info
        to report); n_pipeline_calls is the number of q's that actually
        ran the FFT/coulG/solve pipeline (<=Nk, the measured efficiency
        win from the conjugate shortcut, recorded for provenance).

    Raises:
        ValueError: malformed shapes, or forwarded from
            apply_kernel_and_solve_device for any q that runs the
            pipeline directly.
    """
    n_kpts = mesh_obj.n_kpts
    Pi = np.asarray(Pi)
    eta = np.asarray(eta)
    if Pi.shape[0] != n_kpts:
        raise ValueError(f"Pi.shape[0]={Pi.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")
    if eta.shape[0] != n_kpts:
        raise ValueError(f"eta.shape[0]={eta.shape[0]} must equal mesh_obj.n_kpts={n_kpts}.")

    neg = mesh_obj.neg
    coul_kpt = [None] * n_kpts
    kern_kpt = [None] * n_kpts
    infos = [None] * n_kpts
    done = [False] * n_kpts
    n_pipeline_calls = 0

    for q in range(n_kpts):
        if done[q]:
            continue
        W_q, kern_q, info_q = apply_kernel_and_solve_device(
            provider, q, Pi[q], eta[q], grid_coords=grid_coords, rtol=rtol,
            retained_solve_residual_gate=retained_solve_residual_gate,
        )
        coul_kpt[q] = W_q
        kern_kpt[q] = kern_q
        infos[q] = info_q
        done[q] = True
        n_pipeline_calls += 1

        nq = int(neg[q])
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


def stream_ao_blocks(cell, kpts, grid_coords, block_size):
    """S1: stream AO values at kpts over grid_coords in blocks of
    block_size grid points, evaluating cell.pbc_eval_gto once per block --
    bounds host memory to one block's worth of AO data regardless of Ng.
    Blocks compose directly with build_pi_eta's own ao_blocks iterable
    support (pass a generator expression dropping the (g0,g1) bounds).

    Args:
        cell: pyscf.pbc.gto.Cell.
        kpts: (Nk,3) absolute k-points.
        grid_coords: (Ng,3) real-space grid point coordinates.
        block_size: positive int, grid points per block.

    Yields:
        (g0, g1, ao_block): g0/g1 are the grid-index bounds [g0,g1) this
        block covers; ao_block is (Nk, g1-g0, Nao) complex128.

    Raises:
        ValueError: malformed grid_coords or non-positive block_size.
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
        yield g0, g1, ao_block


def build_periodic_pivot_oracle(cell, kpts, grid_coords, block_size):
    """S2 periodic pivot-selection metric oracle (design v2.1 section 3):
    a (diag, col_eval) pair for the reference-cell (R=0 supercell image)
    pair-density Gram matrix

        M[r,r'] = | sum_{k,mu} conj(AO_k(r,mu)) AO_k(r',mu) |^2 / Nk

    fed to pivoted_cholesky_hermitian for interpolation-point selection.
    M is never materialized. diag(M) costs ONE streamed AO-grid sweep
    (a single pass suffices since M[r,r] only needs AO at r itself); each
    col_eval(j) call costs its OWN full streamed AO-grid sweep (a single
    extra point evaluation at r_j, then re-sweeping the whole grid against
    it) -- selecting `rank` pivots therefore costs `rank` full AO-grid
    sweeps, matching the design's own "measured cost honesty" accounting
    (this traffic is NOT a cheap column fetch and should be recorded by
    the caller, not hidden).

    Args:
        cell: pyscf.pbc.gto.Cell.
        kpts: (Nk,3) absolute k-points.
        grid_coords: (Ng,3) real-space grid point coordinates.
        block_size: forwarded to stream_ao_blocks.

    Returns:
        (diag, col_eval): diag is (Ng,) float64; col_eval(j) -> (Ng,)
        complex128 (M's j-th column; M is real-valued, complex128 dtype
        only to match pivoted_cholesky_hermitian's contract).

    Raises:
        ValueError: forwarded from stream_ao_blocks for malformed inputs.
    """
    grid_coords = np.asarray(grid_coords, dtype=np.float64)
    n_grid = grid_coords.shape[0]
    kpts_np = np.asarray(kpts, dtype=np.float64)
    n_kpts = kpts_np.shape[0]

    diag = np.empty(n_grid, dtype=np.float64)
    for g0, g1, ao_block in stream_ao_blocks(cell, kpts_np, grid_coords, block_size):
        pooled = np.sum(np.abs(ao_block) ** 2, axis=(0, 2))  # (blk,), sum_{k,mu} |AO_k(r,mu)|^2
        diag[g0:g1] = pooled ** 2 / n_kpts

    def col_eval(j):
        ao_j_block = np.asarray(
            cell.pbc_eval_gto("GTOval", grid_coords[j:j + 1], kpts=list(kpts_np)),
            dtype=np.complex128,
        )
        ao_j = ao_j_block[:, 0, :]  # (Nk, Nao)

        col = np.empty(n_grid, dtype=np.complex128)
        for g0, g1, ao_block in stream_ao_blocks(cell, kpts_np, grid_coords, block_size):
            gram = np.einsum("km,krm->r", ao_j.conj(), ao_block, optimize=True)
            col[g0:g1] = (np.abs(gram) ** 2 / n_kpts).astype(np.complex128)
        return col

    return diag, col_eval


# ---------------------------------------------------------------------------
# Staging-policy layer (design v2.1 section 6): predicted-byte-model-driven
# selection among three eta-store staging policies (ram/memmap/recompute),
# plus the memmap/recompute mechanics themselves.
#
# Byte model here is PREDICTED ONLY. Observed HBM/XLA figures are a later
# calibration-gate deliverable; this layer's provenance schema carries the
# observed fields from day one, explicitly None/"unmeasured", so a later
# pass backfills real numbers without any schema change. Real host/disk
# resource queries live in exactly one function (query_host_resources) so
# the policy-selection decision itself takes plain injected numbers and
# stays fully testable with synthetic inputs, including forced-demotion
# cases, with no real memory/disk pressure required.


def predicted_byte_model(n_kpts, n_ip, n_grid, n_ao, block_size, *, itemsize=16):
    """Predicted byte counts for one build (design v2.1 section 6), c128
    (itemsize=16) throughout. Every quantity is a closed-form prediction
    from the problem's own shape parameters -- nothing here is measured.

    Returns:
        dict: ao_grid_block_bytes (one streamed AO block), eta_store_bytes
        (the full (Nk,Nip,Ng) eta array a staging policy must place
        somewhere), double_buffer_bytes (per-block device scratch, 2x),
        fft_workspace_bytes (per-q-slab-pair scratch, 2x),
        pi_v_w_workspace_bytes (O(Nk*Nip^2) Pi/V/W storage +
        O(Nip^2) eigh scratch), selection_traffic_bytes (Nip full
        AO-grid sweeps -- the S2 pivot-selection cost), and
        total_predicted_bytes (eta_store + selection_traffic, the two
        terms choose_staging_policy actually gates on).

    Raises:
        ValueError: any shape parameter is not a positive integer.
    """
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
    """Select a staging policy for the eta store from a PREDICTED byte
    model (predicted_byte_model) and caller-supplied available-resource
    numbers -- never queried internally here, so this function stays
    synthetic-input testable (see query_host_resources for the one place
    real numbers are read).

    Rule: "ram" if eta_store_bytes fits within ram_headroom_fraction of
    available_host_bytes; else "memmap" if it fits within
    disk_headroom_fraction of available_disk_bytes; else "recompute".

    Args:
        byte_model: dict from predicted_byte_model.
        available_host_bytes: free host RAM, as measured or synthesized.
        available_disk_bytes: free scratch-disk space, as measured or
            synthesized.
        ram_headroom_fraction: overridable threshold (default 0.5,
            matching "<=50% of free RAM else demote policy").
        disk_headroom_fraction: overridable threshold for the memmap
            fallback (default 0.9).

    Returns:
        dict: policy ("ram"|"memmap"|"recompute"), eta_store_bytes,
        ram_headroom_bytes, disk_headroom_bytes, ram_headroom_fraction,
        disk_headroom_fraction, available_host_bytes,
        available_disk_bytes, observed_peak_host_bytes (always None
        here), observed_status (always "unmeasured" here) -- the last
        two fields exist so a later calibration pass backfills real
        values into this SAME schema, never a different one.

    Raises:
        ValueError: negative resource numbers, or a fraction outside
            (0,1].
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
    """The one place this module reads REAL host/disk resource numbers
    (psutil / shutil.disk_usage) -- keeps choose_staging_policy itself
    free of any hidden system call, so its decision logic stays
    synthetic-input testable. Callers wanting a real-system staging
    decision call this, then pass the result into choose_staging_policy;
    tests construct available_host_bytes/available_disk_bytes directly.

    Returns:
        (available_host_bytes, available_disk_bytes): both int.
    """
    import shutil

    import psutil

    available_host_bytes = int(psutil.virtual_memory().available)
    available_disk_bytes = int(shutil.disk_usage(scratch_dir).free)
    return available_host_bytes, available_disk_bytes


def jit_memory_analysis_smoke(jitted_fn, *args):
    """Compiled-program memory recording (design v2.1 section 6's "XLA
    temporaries bounded by the jitted-program's compiled memory report"
    note): compiles jitted_fn against args and returns its jax
    CompiledMemoryStats. The label reflects the ACTUAL backend this ran
    on -- "cpu_backend_structural_smoke_not_hbm" only when
    jax.default_backend() is "cpu" (this machine, today); on a real GPU
    backend the label instead reads "gpu_backend_compiled_memory_stats"
    since compiled memory stats on an actual GPU ARE genuine device
    memory data, not a smoke test -- mislabeling a real GPU run as
    "not_hbm" would be exactly the false/omitted-observed-number failure
    this layer's provenance schema is designed to avoid.

    Args:
        jitted_fn: a jax.jit-wrapped function.
        *args: example arguments determining the compiled program's
            shapes (values are only used for shape/dtype; not executed).

    Returns:
        dict: backend (jax.default_backend()), label (see above), and
        every field of jax's CompiledMemoryStats
        (generated_code_size_in_bytes, argument_size_in_bytes,
        output_size_in_bytes, alias_size_in_bytes, temp_size_in_bytes,
        plus the host_* counterparts).
    """
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
    """memmap staging mechanics (policy 2): write streamed eta chunks
    into a q-major np.memmap on disk at memmap_path, without ever
    holding the full (Nk,Nip,Ng) eta array in host RAM at once.

    Args:
        eta_chunks_iter: iterable of (g0, g1, chunk), chunk shape
            (Nk,Nip,g1-g0) complex128, covering [0,Ng) contiguously and
            in order (matches stream_ao_blocks' own (g0,g1,block) shape,
            after routing each streamed AO block through build_pi_eta's
            per-block pair_convolve call).
        shape: (Nk,Nip,Ng), the full logical eta array shape.
        memmap_path: filesystem path for the backing file.

    Returns:
        np.memmap of shape `shape`, dtype complex128, flushed to disk.

    Raises:
        ValueError: shape is not a 3-tuple of positive ints.
    """
    shape_t = tuple(int(x) for x in shape)
    if len(shape_t) != 3 or any(s <= 0 for s in shape_t):
        raise ValueError(f"shape must be 3 positive ints, got {shape_t}.")

    mm = np.memmap(memmap_path, dtype=np.complex128, mode="w+", shape=shape_t)
    for g0, g1, chunk in eta_chunks_iter:
        mm[:, :, g0:g1] = chunk
    mm.flush()
    return mm


def stage_eta_recompute_tile(X, ao_block_source, phase, q_slice=None):
    """recompute staging mechanics (policy 3): no staged array at all --
    re-exposes build_pi_eta's own streaming contract as the "recompute
    per-q-tile" entry point, so a caller under memory/disk pressure
    rebuilds eta ON DEMAND by re-streaming grid blocks through
    ao_block_source, at the cost of one full rebuild per call. The
    mechanics ARE build_pi_eta's existing streaming support; the only
    thing this adds is the CONTRACT that ao_block_source is called fresh
    every time (recompute implies re-streaming from scratch), plus an
    optional q-tile slice applied after the build.

    Args:
        X: (Nk,Nip,Nao) complex128, same as build_pi_eta's X.
        ao_block_source: callable, ao_block_source() -> a FRESH iterable
            of (Nk,blk,Nao) blocks each call (e.g. a lambda wrapping
            stream_ao_blocks(...)).
        phase: forwarded to build_pi_eta.
        q_slice: optional slice/index applied to Pi/eta's leading (Nk)
            axis AFTER the full build (this function still runs the
            complete Alg-1 pair-convolve pass every call; restricting
            grid streaming itself to a q-tile is a further optimization
            not implemented here).

    Returns:
        (Pi, eta): same as build_pi_eta, optionally sliced by q_slice.
    """
    Pi, eta = build_pi_eta(X, ao_block_source(), phase)
    if q_slice is not None:
        return Pi[q_slice], eta[q_slice]
    return Pi, eta
