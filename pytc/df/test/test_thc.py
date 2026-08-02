"""FP64 numerical gates for the robust DF/THC oracle."""

from __future__ import annotations

import hashlib
import inspect
import unittest
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pytc.df import thc
from pytc.df.thc import df_sandwiches_jax, fit_lsthc_jax


class TestRobustDFTHCOracle(unittest.TestCase):
    """The algebra is explicit because this is not yet a production path."""

    def setUp(self):
        rng = np.random.default_rng(20260715)
        self.nocc = 2
        self.nvir = 5
        self.naux = 9
        self.rank = 4
        self.b = rng.normal(size=(self.nvir, self.nvir, self.naux)).astype(np.float64)
        self.p = rng.normal(size=(self.nvir, self.rank)).astype(np.float64)
        t2_raw = rng.normal(
            size=(self.nocc, self.nocc, self.nvir, self.nvir)
        ).astype(np.float64)
        self.t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))
        self.model = thc.build_robust_df_thc_model(self.b, self.p)

    def test_virtual_pair_and_scalar_collocation_orders_are_ac(self):
        b_pair = thc.virtual_pair_df_matrix(self.b)
        collocation = thc.scalar_pair_collocation(self.p)
        for a, c in ((0, 0), (1, 3), (4, 2)):
            row = a * self.nvir + c
            np.testing.assert_array_equal(b_pair[row], self.b[a, c])
            np.testing.assert_allclose(collocation[row], self.p[a] * self.p[c])

    def test_exact_lsthc_and_robust_pair_metrics(self):
        expected_exact = self.model.b_pair @ self.model.b_pair.T
        expected_lsthc = self.model.b_tilde @ self.model.b_tilde.T
        expected_robust = (
            self.model.b_tilde @ self.model.b_pair.T
            + self.model.b_pair @ self.model.b_tilde.T
            - expected_lsthc
        )
        np.testing.assert_allclose(self.model.exact_metric, expected_exact, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(self.model.lsthc_metric, expected_lsthc, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(self.model.robust_metric, expected_robust, rtol=1e-13, atol=1e-13)

    def test_robust_pair_residual_has_the_signed_delta_delta_order(self):
        expected = self.model.delta_b @ self.model.delta_b.T
        np.testing.assert_allclose(
            self.model.exact_minus_robust, expected, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(self.model.exact_minus_robust, expected.T, rtol=1e-13, atol=1e-13)

    def test_direct_df_vvvv_t2_sandwiches_match_dense_oracle_and_delta_residual(self):
        b_tilde = self.model.b_tilde.reshape(self.nvir, self.nvir, self.naux)
        delta_b = self.model.delta_b.reshape(self.nvir, self.nvir, self.naux)
        dense_exact = np.einsum("acq,bdq,ijcd->ijab", self.b, self.b, self.t2, optimize=True)
        dense_lsthc = np.einsum(
            "acq,bdq,ijcd->ijab", b_tilde, b_tilde, self.t2, optimize=True
        )
        dense_robust = (
            np.einsum("acq,bdq,ijcd->ijab", b_tilde, self.b, self.t2, optimize=True)
            + np.einsum("acq,bdq,ijcd->ijab", self.b, b_tilde, self.t2, optimize=True)
            - dense_lsthc
        )
        expected_residual = np.einsum(
            "acq,bdq,ijcd->ijab", delta_b, delta_b, self.t2, optimize=True
        )
        actual = thc.direct_df_sandwiches(self.model, self.t2)
        np.testing.assert_allclose(actual.exact, dense_exact, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.lsthc, dense_lsthc, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.robust, dense_robust, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.exact_minus_robust, expected_residual, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.delta_delta, expected_residual, rtol=1e-12, atol=1e-12)

    def test_rejects_non_fp64_inputs(self):
        with self.assertRaisesRegex(ValueError, "float64"):
            thc.build_robust_df_thc_model(self.b.astype(np.float32), self.p)

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

    All four terms preserve the ``(a,c,b,d)`` source ordering.  The
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
    """Return a declared-shape FLOP and live-data ledger.

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

def canonical_array_fingerprint(value: object) -> dict[str, object]:
    """Fingerprint a real numeric array in a stable, explicitly typed form.

    Floating inputs canonicalize to contiguous ``float64`` bytes; integral
    inputs (the ISDF pivots) canonicalize to contiguous ``int64`` bytes.
    The reported norm is calculated from the canonical representation in
    float64 so the record is comparable across numeric backends.  This helper
    is test-only provenance, not a cache key or production solver route.
    """

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError("canonical H10 fingerprints require real arrays")
    if np.issubdtype(array.dtype, np.floating):
        canonical_dtype = np.dtype(np.float64)
    elif np.issubdtype(array.dtype, np.integer):
        canonical_dtype = np.dtype(np.int64)
    else:
        raise ValueError(
            "canonical H10 fingerprints require floating or integral arrays; "
            f"got {array.dtype}"
        )

    canonical = np.ascontiguousarray(array, dtype=canonical_dtype)
    return {
        "shape": list(canonical.shape),
        "canonical_dtype": canonical_dtype.name,
        "frobenius_norm": float(
            np.linalg.norm(np.asarray(canonical, dtype=np.float64))
        ),
        "sha256_c_contiguous_canonical_bytes": hashlib.sha256(
            canonical.tobytes(order="C")
        ).hexdigest(),
    }

class TestJaxFitAndSandwich(unittest.TestCase):

    def test_jax_fit_and_sandwich_match_panelled_oracle(self):
        rng = np.random.default_rng(4)
        p = rng.normal(size=(4, 3))
        b = rng.normal(size=(4, 4, 5))
        t2 = rng.normal(size=(2, 2, 4, 4))
        oracle_fit = fit_panelled_lsthc(p, b, rcond=1e-12, virtual_panel=2)
        jax_fit = fit_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
        self.assertLess(
            np.max(np.abs(oracle_fit.y - np.asarray(jax_fit.y))), 1e-12)
        oracle = direct_df_sandwiches_panelled(
            b, oracle_fit, t2, rank_panel=2, aux_panel=3)
        actual = df_sandwiches_jax(
            b, jax_fit, t2, rank_panel=2, aux_panel=3)
        self.assertLess(
            np.max(np.abs(oracle.robust - np.asarray(actual.robust))), 1e-12)

    def test_jax_fit_rejects_invalid_rcond(self):
        p = np.eye(2)
        b = np.ones((2, 2, 1))
        for rcond in (0.0, -1.0, 1.1, np.nan, np.inf):
            with self.subTest(rcond=rcond):
                with self.assertRaisesRegex(ValueError, "rcond"):
                    fit_lsthc_jax(
                        p, b, rcond=rcond, virtual_panel=1)

    def test_jax_fit_keeps_b_host_resident(self):
        import pytc.df.thc as mod

        uploads = []
        real = mod._as_fp64_jax

        def recording(name, value, ndim):
            uploads.append((name, np.shape(value)))
            return real(name, value, ndim)

        mod._as_fp64_jax = recording
        try:
            rng = np.random.default_rng(5)
            p = rng.normal(size=(5, 3))
            b = rng.normal(size=(5, 5, 4))
            fit_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
        finally:
            mod._as_fp64_jax = real

        self.assertNotIn("b", [name for name, _ in uploads])
        panel_shapes = [shape for name, shape in uploads if name == "b_panel"]
        self.assertEqual(panel_shapes, [(2, 5, 4), (2, 5, 4), (1, 5, 4)])

    def test_sandwich_keeps_b_host_resident(self):
        # The full 3-index B block must never be cast onto the device
        # wholesale -- the panel loops read it one aux slice at a time
        # from the host.
        import pytc.df.thc as mod

        uploaded = []
        real = mod._as_fp64_jax

        def recording(name, value, ndim):
            uploaded.append(name)
            return real(name, value, ndim)

        mod._as_fp64_jax = recording
        try:
            rng = np.random.default_rng(7)
            p = rng.normal(size=(4, 3))
            b = rng.normal(size=(4, 4, 5))
            t2 = rng.normal(size=(2, 2, 4, 4))
            jax_fit = fit_lsthc_jax(
                p, b, rcond=1e-12, virtual_panel=2)
            uploaded.clear()
            df_sandwiches_jax(
                b, jax_fit, t2, rank_panel=2, aux_panel=3)
        finally:
            mod._as_fp64_jax = real
        self.assertNotIn("b", uploaded)
        self.assertTrue({"t2", "p_virtual", "y"} <= set(uploaded))

    def test_exact_panel_cap_preserves_result(self):
        # PYTC_EXACT_PANEL_CAP_GB clamps the (nocc^2, nvir, nvir, q)
        # scratch; forcing q_step=1 must reproduce the wider-panel result.
        import os
        rng = np.random.default_rng(11)
        p = rng.normal(size=(4, 3))
        b = rng.normal(size=(4, 4, 5))
        t2 = rng.normal(size=(2, 2, 4, 4))
        jax_fit = fit_lsthc_jax(p, b, rcond=1e-12, virtual_panel=2)
        wide = df_sandwiches_jax(
            b, jax_fit, t2, rank_panel=2, aux_panel=3)
        os.environ["PYTC_EXACT_PANEL_CAP_GB"] = str(512 / 1024 ** 3)
        try:
            clamped = df_sandwiches_jax(
                b, jax_fit, t2, rank_panel=2, aux_panel=3)
        finally:
            del os.environ["PYTC_EXACT_PANEL_CAP_GB"]
        self.assertLess(
            np.max(np.abs(np.asarray(wide.robust)
                          - np.asarray(clamped.robust))), 1e-12)

class TestPanelledRobustDFTHCOracle(unittest.TestCase):
    """Keep the algebra paired to the accepted dense FP64 oracle."""

    def setUp(self):
        rng = np.random.default_rng(20260716)
        self.nocc = 2
        self.nvir = 5
        self.naux = 9
        self.rank = 4
        b_raw = rng.normal(size=(self.nvir, self.nvir, self.naux)).astype(np.float64)
        self.b = 0.5 * (b_raw + b_raw.swapaxes(0, 1))
        self.p = rng.normal(size=(self.nvir, self.rank)).astype(np.float64)
        t2_raw = rng.normal(
            size=(self.nocc, self.nocc, self.nvir, self.nvir)
        ).astype(np.float64)
        self.t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))
        self.rcond = 1.0e-12
        self.dense_model = thc.build_robust_df_thc_model(
            self.b, self.p, rcond=self.rcond
        )
        self.fit = fit_panelled_lsthc(
            self.p, self.b, rcond=self.rcond, virtual_panel=2
        )
        self.panelled = direct_df_sandwiches_panelled(
            self.b, self.fit, self.t2, rank_panel=2, aux_panel=3
        )

    def test_panelled_fit_matches_dense_weights_rank_and_conditioning(self):
        self.assertEqual(self.fit.effective_rank, self.dense_model.effective_rank)
        self.assertEqual(self.fit.rcond, self.dense_model.rcond)
        self.assertEqual(
            self.fit.resolved_rcond,
            max(self.rcond, NORMAL_EQUATION_RESOLUTION_RCOND),
        )
        np.testing.assert_allclose(
            self.fit.y, self.dense_model.weights, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            self.fit.singular_values,
            self.dense_model.singular_values,
            rtol=1e-11,
            atol=1e-11,
        )
        self.assertAlmostEqual(
            self.fit.condition_number,
            self.dense_model.condition_number,
            delta=1e-11 * self.dense_model.condition_number,
        )

    def test_panelled_exact_cross_full_and_robust_terms_match_dense_oracle(self):
        b_tilde = self.dense_model.b_tilde.reshape(
            self.nvir, self.nvir, self.naux
        )
        expected_exact = thc.direct_df_vvvv_t2_sandwich(
            self.b, self.b, self.t2
        )
        expected_fit_left = thc.direct_df_vvvv_t2_sandwich(
            b_tilde, self.b, self.t2
        )
        expected_df_left = thc.direct_df_vvvv_t2_sandwich(
            self.b, b_tilde, self.t2
        )
        expected_full = thc.direct_df_vvvv_t2_sandwich(
            b_tilde, b_tilde, self.t2
        )
        expected_robust = expected_fit_left + expected_df_left - expected_full
        expected_delta_delta = thc.direct_df_vvvv_t2_sandwich(
            self.dense_model.delta_b.reshape(self.nvir, self.nvir, self.naux),
            self.dense_model.delta_b.reshape(self.nvir, self.nvir, self.naux),
            self.t2,
        )

        for actual, expected in (
            (self.panelled.exact, expected_exact),
            (self.panelled.fit_left_df_right, expected_fit_left),
            (self.panelled.df_left_fit_right, expected_df_left),
            (self.panelled.full_thc, expected_full),
            (self.panelled.robust, expected_robust),
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

        # The sign is the required robust identity, not an error magnitude.
        np.testing.assert_allclose(
            self.panelled.exact - self.panelled.robust,
            expected_delta_delta,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_panelled_cross_terms_and_results_preserve_rccsd_pair_swaps(self):
        pair_swap = (1, 0, 3, 2)
        np.testing.assert_allclose(
            self.panelled.fit_left_df_right,
            self.panelled.df_left_fit_right.transpose(pair_swap),
            rtol=1e-11,
            atol=1e-11,
        )
        for value in (
            self.panelled.exact,
            self.panelled.full_thc,
            self.panelled.robust,
        ):
            np.testing.assert_allclose(
                value, value.transpose(pair_swap), rtol=1e-11, atol=1e-11
            )

    def test_irregular_rank_and_auxiliary_panels_are_invariant(self):
        """Both choices deliberately leave tails on rank and auxiliary axes."""

        first = direct_df_sandwiches_panelled(
            self.b, self.fit, self.t2, rank_panel=3, aux_panel=4
        )
        second = direct_df_sandwiches_panelled(
            self.b, self.fit, self.t2, rank_panel=3, aux_panel=5
        )
        for name in (
            "exact",
            "fit_left_df_right",
            "df_left_fit_right",
            "full_thc",
            "robust",
        ):
            np.testing.assert_allclose(
                getattr(first, name), getattr(second, name), rtol=1e-11, atol=1e-11
            )

    def test_overcomplete_source_uses_reported_normal_equation_resolution_floor(self):
        """Rank provenance must not retain Gram-roundoff modes as LS modes."""

        rng = np.random.default_rng(20260717)
        nvir, rank, naux, nocc = 3, 8, 5, 2
        p = rng.normal(size=(nvir, rank)).astype(np.float64)
        b_raw = rng.normal(size=(nvir, nvir, naux)).astype(np.float64)
        b = 0.5 * (b_raw + b_raw.swapaxes(0, 1))
        t2_raw = rng.normal(size=(nocc, nocc, nvir, nvir)).astype(np.float64)
        t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))

        fit = fit_panelled_lsthc(p, b, rcond=1.0e-12, virtual_panel=2)
        dense = thc.build_robust_df_thc_model(
            b, p, rcond=fit.resolved_rcond
        )
        represented_b_tilde = np.einsum(
            "am,cm,mq->acq", p, p, fit.y, optimize=True
        )
        panelled = direct_df_sandwiches_panelled(
            b, fit, t2, rank_panel=3, aux_panel=2
        )
        dense_sandwiches = thc.direct_df_sandwiches(dense, t2)

        self.assertEqual(fit.rcond, 1.0e-12)
        self.assertEqual(fit.normal_equation_resolution_rcond, NORMAL_EQUATION_RESOLUTION_RCOND)
        self.assertEqual(fit.resolved_rcond, NORMAL_EQUATION_RESOLUTION_RCOND)
        self.assertEqual(fit.effective_rank, dense.effective_rank)
        self.assertEqual(fit.effective_rank, nvir * (nvir + 1) // 2)
        np.testing.assert_allclose(
            represented_b_tilde,
            dense.b_tilde.reshape(nvir, nvir, naux),
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            panelled.exact, dense_sandwiches.exact, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            panelled.full_thc, dense_sandwiches.lsthc, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            panelled.robust, dense_sandwiches.robust, rtol=1e-11, atol=1e-11
        )

    def test_seeded_random_inputs_have_canonical_fp64_fingerprints(self):
        fingerprints = {
            "b": canonical_array_fingerprint(self.b),
            "p": canonical_array_fingerprint(self.p),
            "t2": canonical_array_fingerprint(self.t2),
        }
        self.assertEqual(fingerprints["b"]["shape"], [5, 5, 9])
        self.assertEqual(fingerprints["p"]["shape"], [5, 4])
        self.assertEqual(fingerprints["t2"]["shape"], [2, 2, 5, 5])
        self.assertEqual(
            {entry["canonical_dtype"] for entry in fingerprints.values()}, {"float64"}
        )
        self.assertEqual(
            fingerprints["b"]["sha256_c_contiguous_canonical_bytes"],
            "9ac7258759d51b56bae6b56bac36665c9ce73cfabcbcb443cf743abf80eb0fad",
        )
        self.assertEqual(
            fingerprints["p"]["sha256_c_contiguous_canonical_bytes"],
            "1d5892cb89b32a6d86032548f2b9cea3cb4a1e515a7e29b0d2efeb56fcaa939c",
        )
        self.assertEqual(
            fingerprints["t2"]["sha256_c_contiguous_canonical_bytes"],
            "9f51a8cad9c90436913833311146565e5123a00e8f152be9e06acb76c1875286",
        )

    def test_panelled_source_does_not_call_phase_b_pair_or_b_tilde_builders(self):
        """The prototype may use B/P panels, never dense helpers."""

        source = inspect.getsource(fit_panelled_lsthc) + inspect.getsource(
            direct_df_sandwiches_panelled
        )
        self.assertNotIn("scalar_pair_collocation(", source)
        self.assertNotIn("virtual_pair_df_matrix(", source)
        self.assertNotIn("build_robust_df_thc_model(", source)
        self.assertNotIn("ijmbq", source)
        self.assertNotIn("ijanq", source)


class TestPhaseCShapeFlopMemoryLedger(unittest.TestCase):
    def test_h10_fixed_rank_ledger_keeps_forbidden_dense_shapes_as_metadata_only(self):
        ledger = phase_c_shape_flop_memory_ledger(
            nocc=5,
            nvir=5,
            naux=180,
            n_fused=240,
            rank_panel=17,
            aux_panel=31,
        )
        self.assertEqual(
            ledger["dimensions"],
            {"nocc": 5, "nvir": 5, "naux": 180, "n_fused": 240},
        )
        self.assertEqual(ledger["forbidden_dense_shapes"]["scalar_collocation_c"], [25, 240])
        self.assertEqual(ledger["forbidden_dense_shapes"]["fitted_df_factor_b_tilde"], [25, 180])
        self.assertEqual(ledger["forbidden_dense_shapes"]["vvvv_tensor"], [5, 5, 5, 5])
        self.assertEqual(ledger["fp64_bytes"]["df_source_panel"], 5 * 5 * 31 * 8)
        self.assertEqual(
            ledger["fp64_element_shapes"]["partial_df_y_endpoint_live"], 5 * 5 * 17
        )
        self.assertNotIn("partial_cross_panel_live", ledger["fp64_element_shapes"])
        self.assertGreater(
            ledger["peak_fp64_bytes_estimate"]["audit_api_all_returned_outputs_plus_largest_term"],
            0,
        )
        self.assertEqual(
            ledger["contraction_leading_flops"]["exact_df"],
            4 * 5 * 5 * 5 * 5 * 5 * 180,
        )
        self.assertEqual(
            ledger["contraction_leading_flops"]["both_partial_thc_cross_terms"],
            2 * 5 * 5 * 240 * 180 + 12 * 5 * 5 * 240 * 5 * 5,
        )
