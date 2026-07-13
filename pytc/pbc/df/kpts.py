"""Periodic k-point canonicalization and the Algorithm-1 pair-convolution
primitive (task #21, #proj-isdf-periodic, design v2.1 section 4).

This module is a leaf: it has zero dependency on pytc.df.{pivots,solvers}
or anything else in pytc, and takes a pyscf Cell object only as an input
argument (never imports pyscf.pbc.df or fftisdf). It does not evaluate
AOs itself -- check_time_reversal_residual takes already-evaluated AO
values as a plain array argument, so this module stays decoupled from
the actual pbc_eval_gto adapter wiring (a later, Phase-C concern).

k-point canonicalization never compares kpts arrays by exact equality
(`==`) -- task #20's fftisdf baseline found 8/9 of that repo's own pytest
suite failing on exactly this mistake: kpts get reprocessed through
PySCF's own pipeline and come back numerically equivalent but not
bit-identical (diff ~1e-16), so a strict `==` pre-check fails even
though the physics is correct. All matching here uses a tolerance
(ktol) and minimum-image distance on the fractional-coordinate torus.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from pyscf.pbc.tools import k2gamma

_DEFAULT_KTOL = 1e-8


def _validate_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be positive, got {ivalue}.")
    return ivalue


def _readonly_copy(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _fold_fractional(scaled, ktol):
    """Fold fractional (reciprocal-lattice-vector-unit) coordinates mod 1
    into [0,1), snapping values within ktol of the 0/1 boundary to exactly
    0.0 -- this is what makes wrap_around=True and wrap_around=False
    inputs (and any other gauge) canonicalize to the identical
    representation: e.g. a BZ-edge fractional coordinate of -0.5 and +0.5
    are literally the same physical point (mod 1, both fold to 0.5), and
    roundoff noise near an integer boundary (e.g. -1e-16) folds to exactly
    0.0 instead of ~1.0 - 1e-16."""
    folded = np.mod(scaled, 1.0)
    folded = np.where(folded > 1.0 - ktol, 0.0, folded)
    folded = np.where(folded < ktol, 0.0, folded)
    return folded


def _match_fractional_points(query, reference, ktol):
    """Return permutation such that permutation[i] is the unique index j
    into reference whose minimum-image distance (on the [0,1)^3 torus) to
    query[i] is <= ktol. Raises if any point has no unique match within
    tolerance -- never silently picks the nearest-but-too-far point."""
    n = query.shape[0]
    if reference.shape[0] != n:
        raise ValueError("query and reference must have the same number of points.")
    permutation = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        diff = reference - query[i]
        diff -= np.round(diff)  # minimum image on the torus, wraps to [-0.5, 0.5)
        distances = np.linalg.norm(diff, axis=1)
        distances = np.where(used, np.inf, distances)
        j = int(np.argmin(distances))
        if distances[j] > ktol:
            raise ValueError(
                f"Could not uniquely match point {i} to the reference mesh within "
                f"ktol={ktol} (closest distance {distances[j]:.3e})."
            )
        permutation[i] = j
        used[j] = True
    return permutation


@dataclasses.dataclass(frozen=True)
class KptsMesh:
    """A canonicalized k-point mesh.

    kpts: the ORIGINAL input k-points (absolute, cell reciprocal units),
        in the caller's original order.
    canonical_kpts: a fixed-gauge (wrap_around=False) reference mesh this
        module generates itself via cell.get_kpts -- the canonical
        ordering everything else (permutation, neg) is expressed against.
    permutation: permutation[i] = index into canonical_kpts that kpts[i]
        matches (within ktol, minimum-image distance in fractional
        coordinates) -- i.e. canonical_kpts[permutation[i]] is
        (numerically) the same physical k-point as kpts[i].
    neg: neg[c] = index into canonical_kpts of -canonical_kpts[c] mod G,
        an involution (neg[neg[c]] == c for all c).
    """
    kpts: object
    kmesh: tuple
    n_kpts: int
    canonical_kpts: object
    permutation: object
    neg: object
    ktol: float

    def __post_init__(self):
        kpts = np.asarray(self.kpts, dtype=np.float64)
        canonical_kpts = np.asarray(self.canonical_kpts, dtype=np.float64)
        permutation = np.asarray(self.permutation)
        neg = np.asarray(self.neg)

        if kpts.ndim != 2 or kpts.shape[1] != 3:
            raise ValueError(f"kpts must have shape (n_kpts,3), got {kpts.shape}.")
        n_kpts = _validate_positive_int("n_kpts", self.n_kpts)
        if kpts.shape[0] != n_kpts:
            raise ValueError(f"kpts.shape[0]={kpts.shape[0]} != n_kpts={n_kpts}.")
        if canonical_kpts.shape != (n_kpts, 3):
            raise ValueError(
                f"canonical_kpts must have shape ({n_kpts},3), got {canonical_kpts.shape}."
            )
        if not np.all(np.isfinite(kpts)) or not np.all(np.isfinite(canonical_kpts)):
            raise ValueError("kpts and canonical_kpts must be finite.")

        kmesh = tuple(int(x) for x in self.kmesh)
        if len(kmesh) != 3 or any(m <= 0 for m in kmesh):
            raise ValueError(f"kmesh must be 3 positive ints, got {kmesh}.")
        if int(np.prod(kmesh)) != n_kpts:
            raise ValueError(f"prod(kmesh)={int(np.prod(kmesh))} != n_kpts={n_kpts}.")

        if permutation.shape != (n_kpts,) or not np.issubdtype(permutation.dtype, np.integer):
            raise ValueError("permutation must be a 1-D integer array of length n_kpts.")
        if sorted(permutation.tolist()) != list(range(n_kpts)):
            raise ValueError("permutation must be a bijection on range(n_kpts).")

        if neg.shape != (n_kpts,) or not np.issubdtype(neg.dtype, np.integer):
            raise ValueError("neg must be a 1-D integer array of length n_kpts.")
        if sorted(neg.tolist()) != list(range(n_kpts)):
            raise ValueError("neg must be a bijection on range(n_kpts).")
        neg_list = neg.tolist()
        for k in range(n_kpts):
            if neg_list[neg_list[k]] != k:
                raise ValueError(f"neg must be an involution: neg[neg[{k}]] != {k}.")

        if isinstance(self.ktol, bool) or not isinstance(self.ktol, (int, float)):
            raise ValueError(f"ktol must be a finite positive float, got {self.ktol!r}.")
        if not math.isfinite(self.ktol) or self.ktol <= 0.0:
            raise ValueError(f"ktol must be a finite positive float, got {self.ktol!r}.")

        # Gamma must be present, at canonical index 0 -- cell.get_kpts always
        # places it first for a wrap_around=False mesh containing it; verified
        # directly here rather than assumed from the builder's own call.
        if not np.allclose(canonical_kpts[0], 0.0, atol=max(self.ktol * 10, 1e-10)):
            raise ValueError(
                "canonical_kpts[0] must be the Gamma point (0,0,0); this mesh does not "
                "contain Gamma at canonical index 0."
            )

        object.__setattr__(self, "kpts", _readonly_copy(kpts))
        object.__setattr__(self, "canonical_kpts", _readonly_copy(canonical_kpts))
        object.__setattr__(self, "permutation", _readonly_copy(permutation.astype(np.int64)))
        object.__setattr__(self, "neg", _readonly_copy(neg.astype(np.int64)))
        object.__setattr__(self, "kmesh", kmesh)
        object.__setattr__(self, "n_kpts", n_kpts)
        object.__setattr__(self, "ktol", float(self.ktol))


def canonicalize_kpts(cell, kpts, *, ktol=_DEFAULT_KTOL):
    """Canonicalize an arbitrary-order, arbitrary-gauge k-point array
    against a fixed-gauge reference mesh this module generates itself.

    Args:
        cell: pyscf.pbc.gto.Cell (or gto.Cell-compatible object exposing
            get_scaled_kpts/get_abs_kpts/get_kpts).
        kpts: (n_kpts,3) absolute k-points, any order, any wrap_around
            gauge -- must form a complete uniform Monkhorst-Pack mesh
            (checked).
        ktol: fractional-coordinate matching tolerance (default 1e-8).

    Returns:
        KptsMesh.

    Raises:
        ValueError: kpts do not form a complete uniform mesh, or any
            point cannot be uniquely matched to the canonical mesh
            within ktol.
    """
    kpts_np = np.asarray(kpts, dtype=np.float64)
    if kpts_np.ndim != 2 or kpts_np.shape[1] != 3:
        raise ValueError(f"kpts must have shape (n_kpts,3), got {kpts_np.shape}.")
    n_kpts = kpts_np.shape[0]
    if n_kpts == 0:
        raise ValueError("kpts must be nonempty.")

    kmesh_arr = k2gamma.kpts_to_kmesh(cell, kpts_np - kpts_np[0])
    kmesh = tuple(int(x) for x in kmesh_arr)
    if int(np.prod(kmesh)) != n_kpts:
        raise ValueError(
            f"Inferred kmesh {kmesh} (prod={int(np.prod(kmesh))}) does not match "
            f"n_kpts={n_kpts} -- kpts do not form a complete uniform mesh."
        )

    canonical_kpts = cell.get_kpts(list(kmesh), wrap_around=False)
    scaled_input = cell.get_scaled_kpts(kpts_np)
    scaled_canonical = cell.get_scaled_kpts(canonical_kpts)
    folded_input = _fold_fractional(scaled_input, ktol)
    folded_canonical = _fold_fractional(scaled_canonical, ktol)

    permutation = _match_fractional_points(folded_input, folded_canonical, ktol)
    neg_frac = _fold_fractional(-folded_canonical, ktol)
    neg = _match_fractional_points(neg_frac, folded_canonical, ktol)

    return KptsMesh(
        kpts=kpts_np, kmesh=kmesh, n_kpts=n_kpts, canonical_kpts=canonical_kpts,
        permutation=permutation, neg=neg, ktol=float(ktol),
    )


def check_time_reversal_residual(ao_at_kpts, neg, *, tol=1e-10):
    """Gate max_k ||AO[neg[k]] - conj(AO[k])|| / ||AO[k]|| <= tol,
    BEFORE any downstream .real is taken on quantities built from these
    AO values -- the relation conj(phi^k) = phi^{-k} is a physical
    identity that must be VERIFIED for a specific realized AO evaluation,
    never assumed by declaration (a numerical AO evaluator could
    legitimately fail to preserve it, e.g. through basis-function phase
    conventions or grid asymmetry).

    Args:
        ao_at_kpts: (n_kpts, ..., n_ao) complex128 AO (or AO-derived)
            values sampled at each canonical k-point, the SAME grid
            points for every k.
        neg: (n_kpts,) int array (KptsMesh.neg).
        tol: hard gate on the relative residual.

    Returns:
        max_relative_residual: float.

    Raises:
        ValueError: the gate fails.
    """
    ao = np.asarray(ao_at_kpts)
    neg_np = np.asarray(neg)
    n_kpts = ao.shape[0]
    if neg_np.shape != (n_kpts,):
        raise ValueError(f"neg must have shape ({n_kpts},), got {neg_np.shape}.")

    max_residual = 0.0
    for k in range(n_kpts):
        lhs = ao[int(neg_np[k])]
        rhs = ao[k].conj()
        denom = max(float(np.linalg.norm(rhs)), 1e-300)
        residual = float(np.linalg.norm(lhs - rhs)) / denom
        max_residual = max(max_residual, residual)

    if max_residual > tol:
        raise ValueError(
            f"check_time_reversal_residual: max relative residual {max_residual:.3e} "
            f"exceeds tol={tol:.1e} -- realized AO values do not satisfy "
            f"conj(AO[k]) = AO[neg[k]] to the required precision; do not take .real of "
            f"any downstream quantity built from these AO values until this is resolved."
        )
    return max_residual


def pair_convolve(X, Y, kmesh, *, imag_tol=1e-10):
    """Paper Algorithm 1: assemble Z[q] without an O(Nk^2) direct sum, via
    per-k GEMM, IFFT over the k-index, elementwise square in real
    (supercell-image) space, then FFT back to momentum space.

        T[k]   = X[k] @ conj(Y[k]).T   -- conj(Y[k]) stands for Y^{-k} by
                                           the time-reversal identity
                                           conj(phi^k) = phi^{-k}, not by
                                           array reindexing.
        T_R    = IFFT_k(T)             -- over the k index reshaped to
                                           (n1,n2,n3); NumPy default norm
                                           ("backward") carries the 1/Nk.
        assert ||Im(T_R)|| / ||T_R|| <= imag_tol, BEFORE discarding Im(T_R)
        Z_R    = Re(T_R) ** 2          -- elementwise square, real space.
        Z[q]   = FFT_k(Z_R)            -- no normalization (NumPy default).

    Args:
        X: (Nk, Nip, Nao) complex128.
        Y: (Nk, F, Nao) complex128.
        kmesh: (3,) positive ints, prod(kmesh) == Nk, the SAME canonical
            k/q-mesh ordering X/Y's axis 0 is indexed by (see KptsMesh).
        imag_tol: gate on ||Im(T_R)|| / ||T_R|| before discarding Im(T_R).

    Returns:
        Z: (Nk, Nip, F) complex128, in the same canonical k/q ordering.

    Raises:
        ValueError: malformed shapes/kmesh, or the imag_tol gate fails
            (X/Y are not a valid time-reversal-symmetric pair).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError("X and Y must be 3-D (Nk, ., Nao).")
    n_k = X.shape[0]
    if Y.shape[0] != n_k:
        raise ValueError(f"X and Y must share Nk (axis 0): {X.shape[0]} != {Y.shape[0]}.")
    if X.shape[2] != Y.shape[2]:
        raise ValueError(f"X and Y must share Nao (axis 2): {X.shape[2]} != {Y.shape[2]}.")
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError("X and Y must have at least one row (Nip/F > 0).")

    kmesh_t = tuple(int(x) for x in kmesh)
    if len(kmesh_t) != 3 or any(m <= 0 for m in kmesh_t):
        raise ValueError(f"kmesh must be 3 positive ints, got {kmesh_t}.")
    if int(np.prod(kmesh_t)) != n_k:
        raise ValueError(f"prod(kmesh)={int(np.prod(kmesh_t))} != Nk={n_k}.")

    n_ip = X.shape[1]
    n_f = Y.shape[1]

    T = np.einsum("kIu,kfu->kIf", X, Y.conj(), optimize=True)  # (Nk, Nip, F)

    T_kmesh = T.reshape(kmesh_t + (n_ip, n_f))
    T_R = np.fft.ifftn(T_kmesh, axes=(0, 1, 2), norm="backward")

    norm_im = float(np.linalg.norm(T_R.imag))
    norm_total = float(np.linalg.norm(T_R))
    imchk = norm_im / norm_total if norm_total > 0.0 else norm_im
    if imchk > imag_tol:
        raise ValueError(
            f"pair_convolve: ||Im(T_R)||/||T_R||={imchk:.3e} exceeds imag_tol={imag_tol:.1e} "
            f"-- the intermediate real-space quantity should be real by time-reversal "
            f"symmetry; this indicates X/Y are not a valid time-reversal-symmetric pair "
            f"(conj(X[neg[k]]) should equal X[k], and likewise for Y)."
        )
    T_R = T_R.real

    Z_R = T_R * T_R

    Z_kmesh = np.fft.fftn(Z_R, axes=(0, 1, 2), norm="backward")
    return Z_kmesh.reshape(n_k, n_ip, n_f).astype(np.complex128)
