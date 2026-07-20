"""Phase-B robust DF/THC algebra oracle.

This module is deliberately standalone.  It models the *ordinary Coulomb*
virtual-virtual DF contribution only; except for
``extract_metric_applied_vv_df_factor`` -- consumed by
``isdf_xtc_ccsd.RCCSD`` when building its factorized state -- it has no call
site in the production CCSD path and does not change any default, solver
route, or chemistry tolerance.

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


def _as_fp64(name: str, value: object, ndim: int) -> Float64Array:
    """Validate an FP64 real array without silently down-casting input."""

    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {array.shape}")
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be real float64; got {array.dtype}")
    return array


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


def scalar_pair_collocation(p_virtual: object) -> Float64Array:
    """Return ``C[(a,c),mu] = P[a,mu] P[c,mu]`` in B's pair order."""

    p = _as_fp64("p_virtual", p_virtual, 2)
    nvir, rank = p.shape
    return np.einsum("am,cm->acm", p, p, optimize=True).reshape(nvir * nvir, rank)


def fit_lsthc_pair_factor(
    b_pair: object,
    collocation: object,
    *,
    rcond: float | None = None,
) -> Float64Array:
    """Fit ``B_tilde = C W`` columnwise by ordinary FP64 least squares."""

    b = _as_fp64("b_pair", b_pair, 2)
    c = _as_fp64("collocation", collocation, 2)
    if b.shape[0] != c.shape[0]:
        raise ValueError(
            "b_pair and collocation must share the flattened virtual-pair axis; "
            f"got {b.shape} and {c.shape}"
        )
    weights, _, _, _ = np.linalg.lstsq(c, b, rcond=rcond)
    return c @ weights


@dataclass(frozen=True)
class RobustDFTHCModel:
    """Exact, LS-THC, and robust pair-space representations for one B matrix."""

    b_pair: Float64Array
    collocation: Float64Array
    b_tilde: Float64Array
    delta_b: Float64Array
    exact_metric: Float64Array
    lsthc_metric: Float64Array
    robust_metric: Float64Array

    @property
    def exact_minus_robust(self) -> Float64Array:
        """The signed robust residual, ``B B.T - robust``."""

        return self.exact_metric - self.robust_metric


def build_robust_df_thc_model(
    l_vv: object,
    p_virtual: object,
    *,
    rcond: float | None = None,
) -> RobustDFTHCModel:
    """Build the Phase-B oracle matrices from B[a,c,Q] and P[a,mu]."""

    b_pair = virtual_pair_df_matrix(l_vv)
    collocation = scalar_pair_collocation(p_virtual)
    if collocation.shape[0] != b_pair.shape[0]:
        raise ValueError(
            "P's virtual dimension does not match L_vv; "
            f"got C {collocation.shape} and B {b_pair.shape}"
        )
    b_tilde = fit_lsthc_pair_factor(b_pair, collocation, rcond=rcond)
    delta_b = b_pair - b_tilde
    exact_metric = b_pair @ b_pair.T
    lsthc_metric = b_tilde @ b_tilde.T
    robust_metric = b_tilde @ b_pair.T + b_pair @ b_tilde.T - lsthc_metric
    return RobustDFTHCModel(
        b_pair=b_pair,
        collocation=collocation,
        b_tilde=b_tilde,
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
