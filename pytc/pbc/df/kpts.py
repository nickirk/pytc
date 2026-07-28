"""Periodic k-point canonicalization and the Algorithm-1 pair-convolution
primitive. See design doc §4.

k-points are never compared by exact equality: matching uses a tolerance
(ktol) and minimum-image distance on the fractional-coordinate torus.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from pyscf.pbc.tools import k2gamma
from scipy.linalg.blas import zgemm

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
    """Fold fractional coordinates mod 1 into [0,1), snapping values within
    ktol of the 0/1 boundary to exactly 0.0 so all wrap_around gauges
    canonicalize identically."""
    folded = np.mod(scaled, 1.0)
    folded = np.where(folded > 1.0 - ktol, 0.0, folded)
    folded = np.where(folded < ktol, 0.0, folded)
    return folded


def _match_fractional_points(query, reference, ktol):
    """Return permutation with permutation[i] = unique index j into reference
    within minimum-image distance ktol of query[i]; raises if no unique
    match."""
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

    kpts: original input k-points (absolute), caller's order.
    canonical_kpts: fixed-gauge (wrap_around=False) reference mesh;
        permutation and neg are expressed against it.
    permutation: permutation[i] = canonical index matching kpts[i].
    neg: neg[c] = canonical index of -canonical_kpts[c] mod G; involution.
    phase: (n_kpts, n_kpts) complex128 unitary k<->supercell-image matrix,
        phase[R,k] = exp(i R.canonical_kpts[k]) / sqrt(n_kpts), with R from
        k2gamma.translation_vectors_for_kmesh(cell, kmesh, wrap_around=False).
    """
    kpts: object
    kmesh: tuple
    n_kpts: int
    canonical_kpts: object
    permutation: object
    neg: object
    ktol: float
    phase: object

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

        # Gamma must be present at canonical index 0; verified, not assumed.
        if not np.allclose(canonical_kpts[0], 0.0, atol=max(self.ktol * 10, 1e-10)):
            raise ValueError(
                "canonical_kpts[0] must be the Gamma point (0,0,0); this mesh does not "
                "contain Gamma at canonical index 0."
            )

        phase = np.asarray(self.phase, dtype=np.complex128)
        if phase.shape != (n_kpts, n_kpts):
            raise ValueError(f"phase must have shape ({n_kpts},{n_kpts}), got {phase.shape}.")
        if not np.all(np.isfinite(phase)):
            raise ValueError("phase must be finite.")
        unitary_residual = float(
            np.linalg.norm(phase.conj().T @ phase - np.eye(n_kpts))
        )
        if unitary_residual > max(self.ktol * 1e4, 1e-8):
            raise ValueError(
                f"phase is not unitary (||phase^dagger @ phase - I||={unitary_residual:.3e}) -- "
                f"this indicates canonical_kpts/R vectors do not form a valid Fourier-conjugate "
                f"pair for this mesh."
            )

        object.__setattr__(self, "kpts", _readonly_copy(kpts))
        object.__setattr__(self, "canonical_kpts", _readonly_copy(canonical_kpts))
        object.__setattr__(self, "permutation", _readonly_copy(permutation.astype(np.int64)))
        object.__setattr__(self, "neg", _readonly_copy(neg.astype(np.int64)))
        object.__setattr__(self, "kmesh", kmesh)
        object.__setattr__(self, "n_kpts", n_kpts)
        object.__setattr__(self, "ktol", float(self.ktol))
        object.__setattr__(self, "phase", _readonly_copy(phase))


def canonicalize_kpts(cell, kpts, *, ktol=_DEFAULT_KTOL):
    """Canonicalize a (n_kpts,3) absolute k-point array (any order/gauge)
    into a KptsMesh; kpts must form a complete uniform Monkhorst-Pack mesh
    (checked). Raises ValueError otherwise."""
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

    R_vec_abs = k2gamma.translation_vectors_for_kmesh(cell, list(kmesh), wrap_around=False)
    phase = np.exp(1j * (R_vec_abs @ canonical_kpts.T)) / np.sqrt(n_kpts)

    return KptsMesh(
        kpts=kpts_np, kmesh=kmesh, n_kpts=n_kpts, canonical_kpts=canonical_kpts,
        permutation=permutation, neg=neg, ktol=float(ktol), phase=phase,
    )


def build_kconserv(cell, canonical_kpts):
    """Momentum-conservation table (n_kpts, n_kpts, n_kpts) int64:
    kconserv[k1,k2,k3] = k4 with k1-k2+k3-k4 a reciprocal lattice vector
    (pyscf get_kconserv convention). Delegates to pyscf."""
    from pyscf.pbc.lib.kpts_helper import get_kconserv

    canonical_kpts = np.asarray(canonical_kpts, dtype=np.float64)
    return np.asarray(get_kconserv(cell, canonical_kpts), dtype=np.int64)


def check_time_reversal_residual(ao_at_kpts, neg, *, tol=1e-10):
    """Gate max_k ||AO[neg[k]] - conj(AO[k])|| / ||AO[k]|| <= tol; must pass
    before any downstream .real is taken on quantities built from these AOs.

    Args:
        ao_at_kpts: (n_kpts, ..., n_ao) complex128, same grid for every k.
        neg: (n_kpts,) int array (KptsMesh.neg).

    Returns the max relative residual; raises ValueError if the gate fails.
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


def kpt_to_spc(m_kpt, phase, *, imag_tol=1e-10):
    """Unitary transform k-space -> supercell-image space:
    m_spc = (phase @ m_kpt).real. NOT a plain np.fft.ifftn reshape --
    the flat k-index does not map onto FFT frequency positions in general;
    phase must be the explicit (Nk,Nk) unitary matrix (KptsMesh.phase).
    The imaginary part is gated (imag_tol) before being discarded.

    Args:
        m_kpt: (Nk, ...) complex128, time-reversal-symmetric across k
            (m_kpt[neg[k]] = conj(m_kpt[k])) so the image is real.

    Returns (Nk, ...) real64; raises ValueError on bad shapes or gate fail.
    """
    m_kpt = np.asarray(m_kpt)
    if m_kpt.ndim < 1:
        raise ValueError("m_kpt must have at least 1 dimension (the k axis).")
    n_k = m_kpt.shape[0]
    phase = np.asarray(phase, dtype=np.complex128)
    if phase.ndim != 2 or phase.shape != (n_k, n_k):
        raise ValueError(f"phase must have shape ({n_k},{n_k}), got {phase.shape}.")

    trailing_shape = m_kpt.shape[1:]
    m_spc_complex = (phase @ m_kpt.reshape(n_k, -1)).reshape((n_k,) + trailing_shape)

    norm_im = float(np.linalg.norm(m_spc_complex.imag))
    norm_total = float(np.linalg.norm(m_spc_complex))
    imchk = norm_im / norm_total if norm_total > 0.0 else norm_im
    if imchk > imag_tol:
        raise ValueError(
            f"kpt_to_spc: ||Im(m_spc)||/||m_spc||={imchk:.3e} exceeds imag_tol={imag_tol:.1e} "
            f"-- m_kpt does not appear to be a valid time-reversal-symmetric collection "
            f"(m_kpt[neg[k]] should equal conj(m_kpt[k]))."
        )
    return m_spc_complex.real


def spc_to_kpt(m_spc, phase):
    """Inverse of kpt_to_spc: m_kpt = phase^dagger @ m_spc.
    Returns (Nk, ...) complex128; this direction never discards anything."""
    m_spc = np.asarray(m_spc)
    if m_spc.ndim < 1:
        raise ValueError("m_spc must have at least 1 dimension (the supercell-image axis).")
    n_k = m_spc.shape[0]
    phase = np.asarray(phase, dtype=np.complex128)
    if phase.ndim != 2 or phase.shape != (n_k, n_k):
        raise ValueError(f"phase must have shape ({n_k},{n_k}), got {phase.shape}.")

    trailing_shape = m_spc.shape[1:]
    m_kpt = (phase.conj().T @ m_spc.reshape(n_k, -1)).reshape((n_k,) + trailing_shape)
    return m_kpt.astype(np.complex128)


@partial(jax.jit, static_argnames=("imag_tol",))
def _pair_convolve_core(X, Y, phase, imag_tol):
    """Jitted device core of pair_convolve. No validation -- the host wrapper
    owns that -- but the time-reversal gate is kept, returned as a traced
    scalar the wrapper raises on. Dropping it on device would make the device
    path the only one that can silently accept a non-TR-symmetric pair.
    """
    T = jnp.einsum("kpa,kfa->kpf", X, jnp.conj(Y))
    flat = T.reshape(T.shape[0], -1)
    spc = (phase @ flat).reshape(T.shape)
    denom = jnp.linalg.norm(spc)
    imag_ratio = jnp.linalg.norm(spc.imag) / jnp.where(denom > 0, denom, 1.0)
    spc_real = spc.real
    Z_R = spc_real * spc_real
    Z = (phase.conj().T @ Z_R.reshape(Z_R.shape[0], -1)).reshape(Z_R.shape)
    return Z.astype(jnp.complex128), imag_ratio


def pair_convolve_device(X, Y, phase, *, imag_tol=1e-10):
    """pair_convolve on the device. Same contract, same gate, same result.

    INVARIANT: bit-equivalence with the numpy path is not claimed -- device
    reductions reorder summations -- so callers gate on the documented
    tolerance, not on identity.
    """
    X = jnp.asarray(X, dtype=jnp.complex128)
    Y = jnp.asarray(Y, dtype=jnp.complex128)
    if X.dtype != jnp.complex128 or Y.dtype != jnp.complex128:
        # Fail closed at the boundary: with jax_enable_x64 off, JAX silently
        # truncates the complex128 cast and the whole convolve runs in single
        # precision -- measured 5.9e-3 relative error, which would surface far
        # downstream as an unexplained accuracy loss. Same guard, same reason,
        # as hermitian_sandwich_solve_device.
        raise ValueError(
            f"pair_convolve_device: resolved dtype is {X.dtype}, not complex128 "
            f"-- jax_enable_x64 is off, so JAX silently downcast the cast and "
            f"this would run in single precision. Call "
            f"jax.config.update('jax_enable_x64', True) before building."
        )
    n_k = X.shape[0]
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError("X and Y must be 3-D (Nk, ., Nao).")
    if Y.shape[0] != n_k:
        raise ValueError(f"X and Y must share axis 0; got {X.shape[0]} and {Y.shape[0]}.")
    phase = jnp.asarray(phase, dtype=jnp.complex128)
    if phase.ndim != 2 or phase.shape != (n_k, n_k):
        raise ValueError(f"phase must have shape ({n_k},{n_k}), got {phase.shape}.")
    Z, imag_ratio = _pair_convolve_core(X, Y, phase, float(imag_tol))
    ratio = float(imag_ratio)
    if ratio > float(imag_tol):
        raise ValueError(
            f"pair_convolve_device: ||Im(m_spc)||/||m_spc||={ratio:.3e} exceeds "
            f"imag_tol={imag_tol:.1e} -- X/Y are not a valid time-reversal-"
            f"symmetric pair (conj(X[neg[k]]) should equal X[k], likewise Y)."
        )
    return np.asarray(Z)


def pair_convolve(X, Y, phase, *, imag_tol=1e-10):
    """Paper Algorithm 1: Z[q] without the O(Nk^2) direct sum. Per-k
    T[k] = X[k] @ conj(Y[k]).T (conj stands for Y^{-k} via time reversal),
    transform to supercell-image space (gated), square elementwise,
    transform back. See design doc §4.

    Args:
        X: (Nk, Nip, Nao) complex128.
        Y: (Nk, F, Nao) complex128.
        phase: (Nk, Nk) unitary matrix in the SAME canonical k/q ordering
            as X/Y's axis 0.

    Returns:
        Z: (Nk, Nip, F) complex128, same canonical k/q ordering.
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

    phase = np.asarray(phase, dtype=np.complex128)
    if phase.ndim != 2 or phase.shape != (n_k, n_k):
        raise ValueError(f"phase must have shape ({n_k},{n_k}), got {phase.shape}.")

    # BLAS, not einsum: einsum never dispatches this contraction and runs a
    # serial loop (24x slower at campaign shapes, pi_zhu JID 59244732).
    # Among BLAS spellings, taking the conjugation as zgemm's op flag beats a
    # batched matmul by 1.5x on OpenBLAS (JID 59244734) because conj(Y) is
    # never materialized -- at 444 that copy is 205 MiB per call. The ranking
    # is BLAS-dependent (the two are at parity on Apple Accelerate), so this
    # is a target-hardware optimization, not a portable truth.
    if X.dtype == np.complex128 and Y.dtype == np.complex128:
        T = np.stack([zgemm(1.0, X[k], Y[k], trans_b=2) for k in range(n_k)])
    else:
        T = np.matmul(X, Y.conj().transpose(0, 2, 1))  # (Nk, Nip, F)

    try:
        T_R = kpt_to_spc(T, phase, imag_tol=imag_tol)
    except ValueError as exc:
        raise ValueError(
            f"pair_convolve: {exc} -- this indicates X/Y are not a valid "
            f"time-reversal-symmetric pair (conj(X[neg[k]]) should equal X[k], and "
            f"likewise for Y)."
        ) from exc

    Z_R = T_R * T_R
    return spc_to_kpt(Z_R, phase)
