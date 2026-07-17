"""Test-only panelled robust DF/THC algebra for the Phase-C oracle.

The Phase-B oracle intentionally materializes dense pair-space arrays to make
the algebra easy to audit.  This module is a separate *test artifact* for the
next correctness gate: it keeps the metric-applied DF source ``B[a,c,Q]`` and
the ISDF factor ``P[a,mu]`` in their native three-/two-index forms and never
constructs ``C[(ac),mu]``, ``B_tilde[(ac),Q]``, or a four-virtual tensor.
It is not imported from any production call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Float64Array = NDArray[np.float64]
NORMAL_EQUATION_RESOLUTION_RCOND = float(np.sqrt(np.finfo(np.float64).eps))


def _as_fp64(name: str, value: object, ndim: int) -> Float64Array:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be a {ndim}D real float64 array; got {array.shape} {array.dtype}")
    return array


def _positive_panel(name: str, value: int, upper: int) -> int:
    size = int(value)
    if not 1 <= size <= upper:
        raise ValueError(f"{name} must be in [1, {upper}]; got {value}")
    return size


@dataclass(frozen=True)
class ScalableLSTHCFit:
    """Implicit LS-THC factor ``B_tilde[a,c,Q]=P[a,m]P[c,m]Y[m,Q]``."""

    p_virtual: Float64Array
    y: Float64Array
    gram: Float64Array
    cross: Float64Array
    # Requested dense-C/SVD threshold; the implicit normal-equation floor is
    # retained separately for rank/conditioning provenance.
    rcond: float
    resolved_rcond: float
    normal_equation_resolution_rcond: float
    effective_rank: int
    singular_values: Float64Array
    singular_max: float
    singular_min_kept: float
    condition_number: float


@dataclass(frozen=True)
class ScalableDirectSandwiches:
    """Panelled exact, cross, full-THC, and robust VVVV--T2 sandwiches."""

    exact: Float64Array
    fit_left_df_right: Float64Array
    df_left_fit_right: Float64Array
    full_thc: Float64Array
    robust: Float64Array


def fit_panelled_lsthc(
    p_virtual: object,
    b: object,
    *,
    rcond: float,
    virtual_panel: int,
) -> ScalableLSTHCFit:
    """Fit the implicit LS-THC factor without forming ``C`` or ``B_tilde``.

    The normal equations follow directly from scalar collocation:

    ``C.T @ C = (P.T @ P) ∘ (P.T @ P)`` and
    ``(C.T @ B)[m,Q] = sum_ac P[a,m] P[c,m] B[a,c,Q]``.

    The latter is accumulated over the first virtual axis, so only a native
    ``B[a_panel,c,Q]`` source panel is consumed at a time.  Normal equations
    square the singular spectrum, so FP64 ``G`` cannot honestly resolve a
    dense-C/SVD threshold below ``sqrt(eps)``.  The requested threshold and
    ``resolved_rcond=max(requested, sqrt(eps))`` are both reported.
    """

    p = _as_fp64("p_virtual", p_virtual, 2)
    df_factor = _as_fp64("b", b, 3)
    nvir, rank = p.shape
    if df_factor.shape[:2] != (nvir, nvir):
        raise ValueError(f"B and P virtual dimensions differ: {df_factor.shape} vs {p.shape}")
    if not np.isfinite(rcond) or not 0.0 < float(rcond) <= 1.0:
        raise ValueError(f"rcond must be in (0, 1]; got {rcond!r}")
    a_panel = _positive_panel("virtual_panel", virtual_panel, nvir)

    p_overlap = p.T @ p
    gram = p_overlap * p_overlap
    cross = np.zeros((rank, df_factor.shape[2]), dtype=np.float64)
    for a0 in range(0, nvir, a_panel):
        a1 = min(a0 + a_panel, nvir)
        cross += np.einsum(
            "am,cm,acq->mq", p[a0:a1], p, df_factor[a0:a1], optimize=True
        )

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    singular_max = float(singular_values[0]) if rank else 0.0
    resolved_rcond = max(float(rcond), NORMAL_EQUATION_RESOLUTION_RCOND)
    keep = singular_values > resolved_rcond * singular_max
    effective_rank = int(np.count_nonzero(keep))
    if effective_rank:
        kept_vectors = eigenvectors[:, keep]
        y = kept_vectors @ (
            (kept_vectors.T @ cross) / eigenvalues[keep, None]
        )
        singular_min_kept = float(singular_values[effective_rank - 1])
        condition_number = singular_max / singular_min_kept
    else:
        y = np.zeros_like(cross)
        singular_min_kept = 0.0
        condition_number = np.inf
    return ScalableLSTHCFit(
        p_virtual=p,
        y=y,
        gram=gram,
        cross=cross,
        rcond=float(rcond),
        resolved_rcond=resolved_rcond,
        normal_equation_resolution_rcond=NORMAL_EQUATION_RESOLUTION_RCOND,
        effective_rank=effective_rank,
        singular_values=singular_values,
        singular_max=singular_max,
        singular_min_kept=singular_min_kept,
        condition_number=float(condition_number),
    )


def _validate_contraction_inputs(
    b: object,
    fit: ScalableLSTHCFit,
    t2: object,
    *,
    rank_panel: int,
    aux_panel: int,
) -> tuple[Float64Array, Float64Array, int, int]:
    df_factor = _as_fp64("b", b, 3)
    amplitudes = _as_fp64("t2", t2, 4)
    nvir = amplitudes.shape[2]
    if amplitudes.shape[2:] != (nvir, nvir):
        raise ValueError(f"t2 virtual axes must be square; got {amplitudes.shape}")
    if df_factor.shape[:2] != (nvir, nvir):
        raise ValueError(f"B virtual dimensions do not match t2: {df_factor.shape} vs {amplitudes.shape}")
    if fit.p_virtual.shape[0] != nvir or fit.y.shape != (fit.p_virtual.shape[1], df_factor.shape[2]):
        raise ValueError("implicit fit dimensions do not match B and t2")
    return (
        df_factor,
        amplitudes,
        _positive_panel("rank_panel", rank_panel, fit.p_virtual.shape[1]),
        _positive_panel("aux_panel", aux_panel, df_factor.shape[2]),
    )


def _exact_df_panelled(
    b: Float64Array,
    t2: Float64Array,
    aux_panel: int,
) -> Float64Array:
    """Exact current-DF sandwich, panelled over the auxiliary source axis."""

    out = np.zeros_like(t2)
    for q0 in range(0, b.shape[2], aux_panel):
        q1 = min(q0 + aux_panel, b.shape[2])
        b_panel = b[:, :, q0:q1]
        right = np.einsum("ijcd,bdq->ijcbq", t2, b_panel, optimize=True)
        out += np.einsum("acq,ijcbq->ijab", b_panel, right, optimize=True)
    return out


def _partial_thc_crosses_panelled(
    b: Float64Array,
    fit: ScalableLSTHCFit,
    t2: Float64Array,
    rank_panel: int,
    aux_panel: int,
) -> tuple[Float64Array, Float64Array]:
    """Return both partial-THC crosses with a precontracted DF endpoint.

    For a rank panel, ``D[b,d,m] = sum_Q B[b,d,Q] Y[m,Q]`` is accumulated
    before the occupied-pair contractions.  Thus neither pair order ever
    retains an ``(ij,m,b,Qp)`` or ``(ij,a,m,Qp)`` intermediate; the endpoint
    is only ``D[v,v,m_panel]`` and is shared by both source pair orders.
    """

    p = fit.p_virtual
    fit_left_df_right = np.zeros_like(t2)
    df_left_fit_right = np.zeros_like(t2)
    for m0 in range(0, p.shape[1], rank_panel):
        m1 = min(m0 + rank_panel, p.shape[1])
        p_panel = p[:, m0:m1]
        endpoint = np.zeros((b.shape[0], b.shape[1], m1 - m0), dtype=np.float64)
        for q0 in range(0, b.shape[2], aux_panel):
            q1 = min(q0 + aux_panel, b.shape[2])
            endpoint += np.einsum(
                "bdq,mq->bdm", b[:, :, q0:q1], fit.y[m0:m1, q0:q1], optimize=True
            )
        tau_left = np.einsum("ijcd,cm->ijmd", t2, p_panel, optimize=True)
        right = np.einsum("ijmd,bdm->ijmb", tau_left, endpoint, optimize=True)
        fit_left_df_right += np.einsum("am,ijmb->ijab", p_panel, right, optimize=True)

        tau_right = np.einsum("ijcd,dm->ijcm", t2, p_panel, optimize=True)
        left = np.einsum("acm,ijcm->ijam", endpoint, tau_right, optimize=True)
        df_left_fit_right += np.einsum("bm,ijam->ijab", p_panel, left, optimize=True)
    return fit_left_df_right, df_left_fit_right


def _full_thc_panelled(
    fit: ScalableLSTHCFit,
    t2: Float64Array,
    rank_panel: int,
    aux_panel: int,
) -> Float64Array:
    """``B_tilde[a,c,Q] B_tilde[b,d,Q] t2[ij,c,d]`` without B_tilde."""

    p = fit.p_virtual
    out = np.zeros_like(t2)
    for m0 in range(0, p.shape[1], rank_panel):
        m1 = min(m0 + rank_panel, p.shape[1])
        p_m = p[:, m0:m1]
        tau_m = np.einsum("ijcd,cm->ijmd", t2, p_m, optimize=True)
        for n0 in range(0, p.shape[1], rank_panel):
            n1 = min(n0 + rank_panel, p.shape[1])
            p_n = p[:, n0:n1]
            tau_mn = np.einsum("ijmd,dn->ijmn", tau_m, p_n, optimize=True)
            rank_metric = np.zeros((m1 - m0, n1 - n0), dtype=np.float64)
            for q0 in range(0, fit.y.shape[1], aux_panel):
                q1 = min(q0 + aux_panel, fit.y.shape[1])
                rank_metric += fit.y[m0:m1, q0:q1] @ fit.y[n0:n1, q0:q1].T
            weighted = tau_mn * rank_metric[None, None]
            right = np.einsum("ijmn,bn->ijmb", weighted, p_n, optimize=True)
            out += np.einsum("am,ijmb->ijab", p_m, right, optimize=True)
    return out


def direct_df_sandwiches_panelled(
    b: object,
    fit: ScalableLSTHCFit,
    t2: object,
    *,
    rank_panel: int,
    aux_panel: int,
) -> ScalableDirectSandwiches:
    """Evaluate exact, both robust cross terms, and full-THC subtraction.

    All four terms preserve the Phase-B ``(a,c,b,d)`` source ordering.  The
    robust result is explicitly ``fit-left/DF-right + DF-left/fit-right -
    full-THC``; retaining the two cross terms separately makes signs and RCCSD
    pair swaps independently auditable in the random-FP64 gate.
    """

    df_factor, amplitudes, resolved_rank_panel, resolved_aux_panel = _validate_contraction_inputs(
        b, fit, t2, rank_panel=rank_panel, aux_panel=aux_panel
    )
    exact = _exact_df_panelled(df_factor, amplitudes, resolved_aux_panel)
    fit_left_df_right, df_left_fit_right = _partial_thc_crosses_panelled(
        df_factor, fit, amplitudes, resolved_rank_panel, resolved_aux_panel
    )
    full_thc = _full_thc_panelled(
        fit, amplitudes, resolved_rank_panel, resolved_aux_panel
    )
    return ScalableDirectSandwiches(
        exact=exact,
        fit_left_df_right=fit_left_df_right,
        df_left_fit_right=df_left_fit_right,
        full_thc=full_thc,
        robust=fit_left_df_right + df_left_fit_right - full_thc,
    )


def phase_c_shape_flop_memory_ledger(
    *,
    nocc: int,
    nvir: int,
    naux: int,
    n_fused: int,
    rank_panel: int,
    aux_panel: int,
) -> dict[str, object]:
    """Return a declared-shape FLOP and live-data ledger for Phase C.

    Counts are FP64 elements/bytes.  Contraction FLOPs follow the actual
    panel contraction order, with a multiply-add counted as two FLOPs.  Peak
    entries are conservative live-data estimates, not allocator measurements;
    they deliberately distinguish an in-core native DF source from the
    temporary source panel a later streaming reader may expose.  Panel sizes
    are explicit inputs, never an auto-tuned policy.
    """

    o, v, q, r = (int(nocc), int(nvir), int(naux), int(n_fused))
    if min(o, v, q, r) < 1:
        raise ValueError("nocc, nvir, naux, and n_fused must all be positive")
    m = _positive_panel("rank_panel", rank_panel, r)
    q_panel = _positive_panel("aux_panel", aux_panel, q)
    fp64 = 8
    elements = {
        "p_virtual": v * r,
        "gram": r * r,
        "cross_or_y": r * q,
        "t2_output": o * o * v * v,
        "df_source_full_if_in_core": v * v * q,
        "df_source_panel": v * v * q_panel,
        "fit_cross_panel_live": m * q_panel,
        "exact_aux_panel_live": o * o * v * v * q_panel,
        "partial_t2_rank_projection_live": o * o * m * v,
        "partial_df_y_endpoint_live": v * v * m,
        "partial_rank_to_virtual_live": o * o * m * v,
        "full_thc_rank_pair_live": o * o * m * m,
        "full_thc_rank_to_virtual_live": o * o * m * v,
    }
    fit_flops = {
        "p_transpose_p": 2 * v * r * r,
        "cross_c_transpose_b": 2 * v * v * r * q,
        "eigensolve_leading_order": r * r * r,
        "eigenbasis_cross_project_and_backproject": 4 * r * r * q,
    }
    contraction_flops = {
        # Per Q: B_Q @ t2 @ B_Q.T, evaluated as two v-by-v products.
        "exact_df": 4 * o * o * v * v * v * q,
        # Build D[b,d,m] = sum_Q B[b,d,Q] Y[m,Q] once, then use it for
        # both pair orders.  No occupied-pair intermediate carries Q.
        "both_partial_thc_cross_terms": 2 * v * v * r * q
        + 12 * o * o * r * v * v,
        "full_thc_subtraction": 2 * o * o * v * v * r
        + 2 * o * o * v * r * r
        + 2 * r * r * q
        + o * o * r * r
        + 2 * o * o * r * r * v
        + 2 * o * o * r * v * v,
    }
    peak_elements = {
        # ``P.T @ P``, Gram, eigensystem workspace, cross, and Y; B is an
        # immutable source input and is reported separately below.
        "fit_excluding_source_input": 4 * r * r + 2 * r * q,
        "fit_with_in_core_source": 4 * r * r + 2 * r * q + v * v * q,
        "fit_with_streamed_source_panel": 4 * r * r + 2 * r * q + v * v * q_panel,
        "exact_contraction": o * o * v * v + o * o * v * v * q_panel,
        "one_partial_cross_term": (
            o * o * v * v
            + o * o * m * v
            + v * v * m
        ),
        "full_thc_subtraction": (
            o * o * v * v
            + 2 * o * o * m * v
            + 2 * o * o * m * m
            + m * m
        ),
        # The test-only audit API retains exact, both cross terms, full THC,
        # and robust outputs simultaneously.  A production contraction need
        # not retain those diagnostics.
        "audit_api_all_returned_outputs_plus_largest_term": (
            5 * o * o * v * v
            + max(
                o * o * v * v * q_panel,
                o * o * m * v + v * v * m,
                2 * o * o * m * v + 2 * o * o * m * m + m * m,
            )
        ),
    }
    return {
        "dimensions": {"nocc": o, "nvir": v, "naux": q, "n_fused": r},
        "panels": {"rank_panel": m, "aux_panel": q_panel},
        "fp64_element_shapes": elements,
        "fp64_bytes": {name: count * fp64 for name, count in elements.items()},
        "peak_fp64_elements_estimate": peak_elements,
        "peak_fp64_bytes_estimate": {
            name: count * fp64 for name, count in peak_elements.items()
        },
        "fit_leading_flops": fit_flops,
        "contraction_leading_flops": contraction_flops,
        "forbidden_dense_shapes": {
            "scalar_collocation_c": [v * v, r],
            "fitted_df_factor_b_tilde": [v * v, q],
            "vvvv_tensor": [v, v, v, v],
        },
    }
