"""Phase-B robust DF/THC algebra oracle.

This module is deliberately standalone.  It models the *ordinary Coulomb*
virtual-virtual DF contribution only; except for
``extract_metric_applied_vv_df_factor`` -- consumed by
``isdf_xtc_ccsd.RCCSD`` when building its factorized state -- it has no call
site in the production CCSD path and does not change any default, solver
route, or chemistry tolerance.  Its source model is real molecular PySCF DF:
``with_df.loop()`` yields the already metric-applied ``cderi`` factor, not
raw three-centre integrals, so ``B B.T`` reconstructs the molecular Coulomb
pair metric.

PyTC's source factor is ``B[a, c, Q] = (a c | Q)``.  A scalar ISDF
collocation matrix is formed in exactly the same flattened virtual-pair order,
``C[(a,c), mu] = P[a,mu] P[c,mu]``.  Least squares gives ``B_tilde = C W``.
The robust metric is

``B_tilde B.T + B B_tilde.T - B_tilde B_tilde.T``.

Consequently, with ``DeltaB = B - B_tilde``, the signed residual is
``exact - robust = DeltaB DeltaB.T``.  The helpers below expose both pair
metrics and the VVVV--T2 sandwiches without materialising a VVVV tensor.
They are FP64 numerical-oracle utilities, not a production factor-direct
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Float64Array = NDArray[np.float64]
DEFAULT_LSTHC_RCOND = 1.0e-12


@dataclass(frozen=True)
class ErrorNorm:
    """Absolute and relative Frobenius error against an exact FP64 value."""

    absolute_frobenius: float
    relative_frobenius: float


@dataclass(frozen=True)
class LSTHCFit:
    """Auditable least-squares fit of the metric-applied DF factor B."""

    weights: Float64Array
    b_tilde: Float64Array
    rcond: float
    effective_rank: int
    singular_values: Float64Array
    singular_max: float
    singular_min_kept: float
    condition_number: float


@dataclass(frozen=True)
class RobustMetricSpectrum:
    """PSD diagnostic for a robust metric, which is not PSD by construction."""

    minimum_eigenvalue: float
    negative_spectral_weight: float


def _as_fp64(name: str, value: object, ndim: int) -> Float64Array:
    """Validate an FP64 real array without silently down-casting input."""

    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {array.shape}")
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be real float64; got {array.dtype}")
    return array


def _validate_rcond(rcond: float) -> float:
    """Require an explicit finite least-squares truncation threshold."""

    value = float(rcond)
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"rcond must be finite and in (0, 1]; got {rcond!r}")
    return value


def frobenius_error(reference: object, approximation: object) -> ErrorNorm:
    """Return FP64 absolute/relative Frobenius error without a silent dtype cast."""

    exact = np.asarray(reference)
    approx = np.asarray(approximation)
    if exact.dtype != np.dtype(np.float64) or approx.dtype != np.dtype(np.float64):
        raise ValueError(f"reference and approximation must be float64; got {exact.dtype} and {approx.dtype}")
    if exact.shape != approx.shape:
        raise ValueError(f"reference and approximation shapes differ: {exact.shape} vs {approx.shape}")
    absolute = float(np.linalg.norm(exact - approx))
    denominator = float(np.linalg.norm(exact))
    relative = absolute / denominator if denominator else (0.0 if absolute == 0.0 else np.inf)
    return ErrorNorm(absolute_frobenius=absolute, relative_frobenius=relative)


def virtual_pair_df_matrix(l_vv: object) -> Float64Array:
    """Flatten PyTC's ``B[a,c,Q]`` in C-order pair convention.

    The returned row ``a * nvir + c`` is precisely the virtual pair ``(a,c)``.
    This is the order produced after PyTC's ``lib.unpack_tril(eris.vvL[:],
    axis=0)`` at the DF call sites.
    """

    b = _as_fp64("l_vv", l_vv, 3)
    nvir_a, nvir_c, _ = b.shape
    if nvir_a != nvir_c:
        raise ValueError(f"l_vv virtual axes must be square; got {b.shape}")
    return b.reshape(nvir_a * nvir_c, b.shape[2])


def extract_metric_applied_vv_df_factor(
    with_df: object,
    mo_coeff: object,
    nocc: int,
) -> Float64Array:
    """Extract PySCF's metric-applied virtual DF factor in PyTC order.

    Consumed by :class:`pytc.solver.isdf_xtc_ccsd.RCCSD` when it builds its
    factorized state.  The extraction mirrors PyTC's DF construction exactly:
    ``with_df.loop()`` supplies PySCF's metric-applied cderi blocks,
    ``_ao2mo.nr_e2`` transforms them to the MO basis, the virtual block is
    packed into the same ``vvL[(ac),Q]`` layout, and
    ``lib.unpack_tril(..., axis=0)`` restores ``B[a,c,Q]``.

    ``nocc`` is explicit because the orbital-space partition must not be
    guessed from an occupation threshold.
    """

    raw_mo = np.asarray(mo_coeff)
    if np.iscomplexobj(raw_mo):
        raise ValueError("mo_coeff must be real; complex coefficients are not supported")

    try:
        from pyscf import lib
        from pyscf.ao2mo import _ao2mo
    except ImportError as exc:  # pragma: no cover - exercised by physical card
        raise RuntimeError("PySCF is required for physical DF-factor extraction") from exc

    mo = np.asarray(raw_mo, dtype=np.float64, order="F")
    if mo.ndim != 2 or mo.shape[0] < 1 or mo.shape[1] < 2:
        raise ValueError(f"mo_coeff must be a nonempty AO-by-MO matrix; got {mo.shape}")
    nmo = mo.shape[1]
    nocc = int(nocc)
    if not 0 < nocc < nmo:
        raise ValueError(f"nocc must leave a nonempty virtual space; got {nocc} for nmo={nmo}")
    if not hasattr(with_df, "loop"):
        raise ValueError("with_df must expose the PySCF metric-applied loop() API")

    vv_l_blocks: list[Float64Array] = []
    ijslice = (0, nmo, 0, nmo)
    l_pq = None
    for eri1 in with_df.loop():
        l_pq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym="s2", mosym="s1", out=l_pq)
        l_pq = np.asarray(l_pq, dtype=np.float64).reshape(-1, nmo, nmo)
        vv_l_blocks.append(lib.pack_tril(l_pq[:, nocc:, nocc:]))
    if not vv_l_blocks:
        raise ValueError("with_df.loop() yielded no metric-applied cderi blocks")

    vv_l = np.concatenate(vv_l_blocks, axis=0).T
    return np.asarray(lib.unpack_tril(vv_l, axis=0), dtype=np.float64)


def scalar_pair_collocation(p_virtual: object) -> Float64Array:
    """Return ``C[(a,c),mu] = P[a,mu] P[c,mu]`` in B's pair order."""

    p = _as_fp64("p_virtual", p_virtual, 2)
    nvir, rank = p.shape
    if rank < 1:
        raise ValueError("p_virtual must contain at least one selected ISDF column")
    return np.einsum("am,cm->acm", p, p, optimize=True).reshape(nvir * nvir, rank)


def fit_lsthc_pair_factor(
    b_pair: object,
    collocation: object,
    *,
    rcond: float = DEFAULT_LSTHC_RCOND,
) -> LSTHCFit:
    """Fit ``B_tilde = C W`` and retain every rank/conditioning diagnostic."""

    b = _as_fp64("b_pair", b_pair, 2)
    c = _as_fp64("collocation", collocation, 2)
    if b.shape[0] != c.shape[0]:
        raise ValueError(
            "b_pair and collocation must share the flattened virtual-pair axis; "
            f"got {b.shape} and {c.shape}"
        )
    resolved_rcond = _validate_rcond(rcond)
    weights, _, effective_rank, singular_values = np.linalg.lstsq(
        c, b, rcond=resolved_rcond
    )
    singular_values = np.asarray(singular_values, dtype=np.float64)
    singular_max = float(singular_values[0])
    if effective_rank:
        singular_min_kept = float(singular_values[effective_rank - 1])
        condition_number = singular_max / singular_min_kept
    else:
        singular_min_kept = 0.0
        condition_number = np.inf
    return LSTHCFit(
        weights=weights,
        b_tilde=c @ weights,
        rcond=resolved_rcond,
        effective_rank=int(effective_rank),
        singular_values=singular_values,
        singular_max=singular_max,
        singular_min_kept=singular_min_kept,
        condition_number=float(condition_number),
    )


@dataclass(frozen=True)
class RobustDFTHCModel:
    """Exact, LS-THC, and robust pair-space representations for one B matrix."""

    b_pair: Float64Array
    collocation: Float64Array
    b_tilde: Float64Array
    weights: Float64Array
    rcond: float
    effective_rank: int
    singular_values: Float64Array
    singular_max: float
    singular_min_kept: float
    condition_number: float
    delta_b: Float64Array
    exact_metric: Float64Array
    lsthc_metric: Float64Array
    robust_metric: Float64Array

    @property
    def exact_minus_robust(self) -> Float64Array:
        """The signed robust residual, ``B B.T - robust``."""

        return self.exact_metric - self.robust_metric

    @property
    def lsthc_metric_error(self) -> ErrorNorm:
        """Error of the full LS-THC pair metric against exact DF."""

        return frobenius_error(self.exact_metric, self.lsthc_metric)

    @property
    def robust_metric_error(self) -> ErrorNorm:
        """Error of the robust pair metric against exact DF."""

        return frobenius_error(self.exact_metric, self.robust_metric)

    @property
    def robust_metric_spectrum(self) -> RobustMetricSpectrum:
        """Return the required indefinite-metric diagnostic for robust DF."""

        symmetric_metric = 0.5 * (self.robust_metric + self.robust_metric.T)
        eigenvalues = np.linalg.eigvalsh(symmetric_metric)
        negative = eigenvalues[eigenvalues < 0.0]
        return RobustMetricSpectrum(
            minimum_eigenvalue=float(eigenvalues[0]),
            negative_spectral_weight=float(np.abs(negative).sum()),
        )


def build_robust_df_thc_model(
    l_vv: object,
    p_virtual: object,
    *,
    rcond: float = DEFAULT_LSTHC_RCOND,
) -> RobustDFTHCModel:
    """Build the Phase-B oracle matrices from B[a,c,Q] and P[a,mu]."""

    b_pair = virtual_pair_df_matrix(l_vv)
    collocation = scalar_pair_collocation(p_virtual)
    if collocation.shape[0] != b_pair.shape[0]:
        raise ValueError(
            "P's virtual dimension does not match L_vv; "
            f"got C {collocation.shape} and B {b_pair.shape}"
        )
    fit = fit_lsthc_pair_factor(b_pair, collocation, rcond=rcond)
    delta_b = b_pair - fit.b_tilde
    exact_metric = b_pair @ b_pair.T
    lsthc_metric = fit.b_tilde @ fit.b_tilde.T
    robust_metric = fit.b_tilde @ b_pair.T + b_pair @ fit.b_tilde.T - lsthc_metric
    return RobustDFTHCModel(
        b_pair=b_pair,
        collocation=collocation,
        b_tilde=fit.b_tilde,
        weights=fit.weights,
        rcond=fit.rcond,
        effective_rank=fit.effective_rank,
        singular_values=fit.singular_values,
        singular_max=fit.singular_max,
        singular_min_kept=fit.singular_min_kept,
        condition_number=fit.condition_number,
        delta_b=delta_b,
        exact_metric=exact_metric,
        lsthc_metric=lsthc_metric,
        robust_metric=robust_metric,
    )


def direct_df_vvvv_t2_sandwich(
    left_b: object,
    right_b: object,
    t2: object,
) -> Float64Array:
    """Evaluate ``sum_cdQ left[a,c,Q] right[b,d,Q] t2[ij,c,d]``.

    This is the direct PyTC VVVV--T2 ordering ``(a,c,b,d)``.  It uses only
    a rank-bearing sandwich intermediate and never forms ``V[a,c,b,d]``.
    """

    left = _as_fp64("left_b", left_b, 3)
    right = _as_fp64("right_b", right_b, 3)
    amplitudes = _as_fp64("t2", t2, 4)
    nvir = amplitudes.shape[2]
    if amplitudes.shape[2] != amplitudes.shape[3]:
        raise ValueError(f"t2 virtual axes must be square; got {amplitudes.shape}")
    expected = (nvir, nvir)
    if left.shape[:2] != expected or right.shape[:2] != expected:
        raise ValueError(
            "B virtual axes must match t2; "
            f"got left {left.shape}, right {right.shape}, t2 {amplitudes.shape}"
        )
    if left.shape[2] != right.shape[2]:
        raise ValueError(f"B auxiliary axes must agree; got {left.shape} and {right.shape}")
    # S[i,j,c,b,Q] = sum_d t2[i,j,c,d] right[b,d,Q]
    sandwich = np.einsum("ijcd,bdq->ijcbq", amplitudes, right, optimize=True)
    return np.einsum("acq,ijcbq->ijab", left, sandwich, optimize=True)


@dataclass(frozen=True)
class DirectDFSandwiches:
    """Exact, full LS-THC, and robust direct DF VVVV--T2 contractions."""

    exact: Float64Array
    lsthc: Float64Array
    robust: Float64Array
    delta_delta: Float64Array

    @property
    def exact_minus_robust(self) -> Float64Array:
        """The signed direct residual, equal to ``delta_delta``."""

        return self.exact - self.robust

    @property
    def lsthc_error(self) -> ErrorNorm:
        """Full LS-THC direct-sandwich error against the exact DF sandwich."""

        return frobenius_error(self.exact, self.lsthc)

    @property
    def robust_error(self) -> ErrorNorm:
        """Robust direct-sandwich error against the exact DF sandwich."""

        return frobenius_error(self.exact, self.robust)


def direct_df_sandwiches(model: RobustDFTHCModel, t2: object) -> DirectDFSandwiches:
    """Evaluate all three Phase-B direct-DF VVVV--T2 oracle sandwiches."""

    b_shape = model.b_pair.shape
    n_pair, naux = b_shape
    nvir = int(round(n_pair**0.5))
    if nvir * nvir != n_pair:
        raise ValueError(f"model pair axis is not square: {b_shape}")
    b = model.b_pair.reshape(nvir, nvir, naux)
    b_tilde = model.b_tilde.reshape(nvir, nvir, naux)
    delta_b = model.delta_b.reshape(nvir, nvir, naux)
    exact = direct_df_vvvv_t2_sandwich(b, b, t2)
    lsthc = direct_df_vvvv_t2_sandwich(b_tilde, b_tilde, t2)
    robust = (
        direct_df_vvvv_t2_sandwich(b_tilde, b, t2)
        + direct_df_vvvv_t2_sandwich(b, b_tilde, t2)
        - lsthc
    )
    delta_delta = direct_df_vvvv_t2_sandwich(delta_b, delta_b, t2)
    return DirectDFSandwiches(
        exact=exact,
        lsthc=lsthc,
        robust=robust,
        delta_delta=delta_delta,
    )
