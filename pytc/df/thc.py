"""DF/THC algebra: NumPy oracle, JAX production path, and X-store layouts.

The NumPy section is the reference oracle (and provides
``extract_vv_df_factor`` for the solver's factorized state).
The JAX section is the production implementation of the same panelled
algebra.  The X-store section converts ISDF X factors between the
rank-innermost store layout ``(nmo, nmo, rank)`` and the rank-major layout
``(rank, nmo, nmo)`` (panel-contiguous), in place or into a new store.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# ---------------- NumPy oracle ----------------


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


def extract_vv_df_factor(
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
    """Build the oracle matrices from B[a,c,Q] and P[a,mu]."""

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
    """Evaluate all three direct-DF VVVV--T2 oracle sandwiches."""

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

# ---------------- JAX production path ----------------


# JAX is float32 by default. Enabling x64 must happen before any array is created;
# doing it at import time is deliberate, and require_float64() verifies it actually
# took effect rather than assuming the flag was honoured.
jax.config.update("jax_enable_x64", True)


class Float64NotEnabled(RuntimeError):
    """Raised when x64 is off -- a float32 run would look like ~1e-7 parity."""


def require_float64():
    """Fail closed if JAX is not in x64 mode.

    Without this the computation runs in float32: parity comes out around
    1e-7 instead of 1e-12.  A wrong dtype is not a small error here; it is a
    different calculation.
    """
    probe = jnp.zeros(1, dtype=jnp.float64)
    if probe.dtype != jnp.float64:
        raise Float64NotEnabled(
            "jax_enable_x64 is not in effect (probe dtype "
            f"{probe.dtype}). The sandwich is an FP64 calculation; running it in "
            "float32 yields ~1e-7 agreement that can be mistaken for parity. "
            "Set JAX_ENABLE_X64=1 before importing jax, or call "
            "jax.config.update('jax_enable_x64', True) earlier.")
    return True


def _as_fp64_jax(name: str, value: object, ndim: int):
    """Mirror of the oracle's _as_fp64, but landing on device as float64."""
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array; got {array.ndim}D "
                         f"{array.shape}")
    return array


def _as_fp64_host(name: str, value: object, ndim: int):
    """FP64 as a HOST array -- for operands the panel loops only ever slice.

    A wholesale device cast of the full 3-index B block costs its entire
    size on device even though every consumer reads it one aux panel at a
    time.  Kept on the host, each panel slice is uploaded by the jitted
    kernel that receives it, so peak device cost is one panel.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array; got {array.ndim}D "
                         f"{array.shape}")
    return array


# ---------------------------------------------------------------- kernels ---
# Each jitted body is ONE panel's arithmetic, matching the oracle's einsum
# subscripts exactly. Subscripts are copied verbatim from the NumPy source: they
# encode the (a,c,b,d) source ordering that the RCCSD pair swaps depend on, and
# "simplifying" them is how a port silently breaks pair-swap symmetry.

@partial(jax.jit, donate_argnums=(0,))
def _exact_accumulate(out, b_panel, t2):
    """out + one aux panel's exact sandwich, donating the accumulator.

    Donating the accumulator avoids keeping three full-size buffers
    live per panel (out, panel result, sum).
    """
    right = jnp.einsum("ijcd,bdq->ijcbq", t2, b_panel)
    return out + jnp.einsum("acq,ijcbq->ijab", b_panel, right)


@partial(jax.jit, donate_argnums=(0,))
def _acc_into(acc, block):
    """acc + block with the accumulator's buffer donated (see above)."""

    return acc + block


def exact_df_panelled(b, t2, aux_panel: int):
    """Exact current-DF sandwich, panelled over the auxiliary axis.

    The ``(nocc^2, nvir, nvir, q)`` intermediate inside
    :func:`_exact_accumulate` costs ``nocc^2 * nvir^2 * q * 8`` bytes --
    ~4.9 GiB per aux column at large systems, so the caller's
    ``aux_panel=32`` would demand ~157 GiB.  The
    panel step is therefore clamped so the intermediate stays under
    ``PYTC_EXACT_PANEL_CAP_GB`` (default 8 GiB, read at call time); at q=1
    the GEMM shapes are unchanged (the batch is the occupied-pair axis, not
    q), so the clamp costs no efficiency.  Small cases are unaffected.
    """

    per_q = (t2.shape[0] * t2.shape[1] * t2.shape[2] * t2.shape[3]
             * np.dtype(np.float64).itemsize)
    cap = int(float(os.environ.get("PYTC_EXACT_PANEL_CAP_GB", "8")) * 1024 ** 3)
    q_step = max(1, min(int(aux_panel), cap // max(per_q, 1)))
    out = jnp.zeros_like(t2)
    for q0 in range(0, b.shape[2], q_step):
        q1 = min(q0 + q_step, b.shape[2])
        out = _exact_accumulate(out, b[:, :, q0:q1], t2)
    return out


@jax.jit
def _endpoint_panel(b_q, y_mq):
    # (m, b, d): rank-leading so the cross kernels get batch-leading GEMMs.
    return jnp.einsum("mq,bdq->mbd", y_mq, b_q)


@partial(jax.jit, static_argnames=("occupied_pair_batch_size",))
def _t2_left_contract_pairs_jit(t2_pairs, factor_panel, *,
                                occupied_pair_batch_size):
    """tau_t[m, n, d] = sum_c t2_pairs[n, c, d] F[c, m], pair-blocked.

    Whole-t2 middle-axis contractions make XLA materialize a physical
    transpose of the full t2 at large shapes.  Pair-blocking keeps every
    temporary at one pair block, and rank-leading output makes every
    downstream GEMM batch-leading.  ``t2_pairs`` is pair-flattened,
    pair-padded t2.
    """

    n_padded_pairs, nvir, _ = t2_pairs.shape
    n_rank = factor_panel.shape[1]
    n_pair_blocks = n_padded_pairs // occupied_pair_batch_size

    def pair_body(pair_block, acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir))
        block = jnp.einsum("cm,ncd->mnd", factor_panel, tau)
        return jax.lax.dynamic_update_slice(acc, block, (0, pair0, 0))

    return jax.lax.fori_loop(
        0, n_pair_blocks, pair_body,
        jnp.zeros((n_rank, n_padded_pairs, nvir), dtype=t2_pairs.dtype))


@partial(jax.jit, static_argnames=("occupied_pair_batch_size",))
def _t2_right_contract_pairs_jit(t2_pairs, factor_panel, *,
                                 occupied_pair_batch_size):
    """tau_t[m, n, c] = sum_d t2_pairs[n, c, d] F[d, m]; see the left twin."""

    n_padded_pairs, nvir, _ = t2_pairs.shape
    n_rank = factor_panel.shape[1]
    n_pair_blocks = n_padded_pairs // occupied_pair_batch_size

    def pair_body(pair_block, acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir))
        block = jnp.einsum("dm,ncd->mnc", factor_panel, tau)
        return jax.lax.dynamic_update_slice(acc, block, (0, pair0, 0))

    return jax.lax.fori_loop(
        0, n_pair_blocks, pair_body,
        jnp.zeros((n_rank, n_padded_pairs, nvir), dtype=t2_pairs.dtype))


@jax.jit
def _cross_panel(p_panel, endpoint_t, tau_left_t, tau_right_t):
    # All rank axes (m) leading: every dot is batch-leading both sides.
    # tau_left_t[m,i j,d] / endpoint_t[m,b,d] -> right_t[m,ij,b]
    right_t = jnp.einsum("mnd,mbd->mnb", tau_left_t, endpoint_t)
    fit_left = jnp.einsum("am,mnb->nab", p_panel, right_t)
    # endpoint_t[m,a,c] / tau_right_t[m,ij,c] -> left_t[m,ij,a]
    left_t = jnp.einsum("mac,mnc->mna", endpoint_t, tau_right_t)
    df_left = jnp.einsum("bm,mna->nab", p_panel, left_t)
    return fit_left, df_left


def partial_thc_crosses_panelled(b, p_virtual, y, t2, rank_panel: int,
                                 aux_panel: int):
    """Both partial-THC crosses with a precontracted DF endpoint.

    The endpoint D[m,b,d] is accumulated over aux panels BEFORE the
    occupied-pair contractions, so no (ij,m,b,Qp) intermediate is ever
    retained -- that property is the point of the panelled formulation and
    is preserved here.  t2 enters only through pair-blocked slices (see
    :func:`_t2_left_contract_pairs_jit` for why).
    """
    nocc_i, nocc_j, nvir, _ = t2.shape
    n_pairs = nocc_i * nocc_j
    occupied_pair_batch_size = 8
    n_pair_blocks = (n_pairs + occupied_pair_batch_size - 1) // occupied_pair_batch_size
    padded_pairs = n_pair_blocks * occupied_pair_batch_size
    t2_pairs = jnp.pad(
        jnp.asarray(t2).reshape(n_pairs, nvir, nvir),
        ((0, padded_pairs - n_pairs), (0, 0), (0, 0)))
    fit_left_df_right = jnp.zeros((padded_pairs, nvir, nvir), dtype=jnp.float64)
    df_left_fit_right = jnp.zeros((padded_pairs, nvir, nvir), dtype=jnp.float64)
    for m0 in range(0, p_virtual.shape[1], rank_panel):
        m1 = min(m0 + rank_panel, p_virtual.shape[1])
        p_panel = p_virtual[:, m0:m1]
        endpoint_t = jnp.zeros((m1 - m0, b.shape[0], b.shape[1]),
                               dtype=jnp.float64)
        for q0 in range(0, b.shape[2], aux_panel):
            q1 = min(q0 + aux_panel, b.shape[2])
            endpoint_t = endpoint_t + _endpoint_panel(
                b[:, :, q0:q1], y[m0:m1, q0:q1])
        tau_left_t = _t2_left_contract_pairs_jit(
            t2_pairs, p_panel,
            occupied_pair_batch_size=occupied_pair_batch_size)
        tau_right_t = _t2_right_contract_pairs_jit(
            t2_pairs, p_panel,
            occupied_pair_batch_size=occupied_pair_batch_size)
        fit_left, df_left = _cross_panel(
            p_panel, endpoint_t, tau_left_t, tau_right_t)
        fit_left_df_right = _acc_into(fit_left_df_right, fit_left)
        df_left_fit_right = _acc_into(df_left_fit_right, df_left)
    return (fit_left_df_right[:n_pairs].reshape(t2.shape),
            df_left_fit_right[:n_pairs].reshape(t2.shape))


@jax.jit
def _full_thc_block(p_m, p_n, tau_m_t, rank_metric):
    # tau_m_t[m, ij, d]; rank axis leading throughout (see the crosses).
    tau_mn = jnp.einsum("mjd,dn->mjn", tau_m_t, p_n)
    weighted = tau_mn * rank_metric[:, None, :]
    right_t = jnp.einsum("mjn,bn->mjb", weighted, p_n)
    return jnp.einsum("am,mjb->jab", p_m, right_t)


def full_thc_panelled(p_virtual, y, t2, rank_panel: int, aux_panel: int):
    """B_tilde[a,c,Q] B_tilde[b,d,Q] t2[ij,c,d] without ever forming B_tilde."""
    nocc_i, nocc_j, nvir, _ = t2.shape
    n_pairs = nocc_i * nocc_j
    occupied_pair_batch_size = 8
    n_pair_blocks = (n_pairs + occupied_pair_batch_size - 1) // occupied_pair_batch_size
    padded_pairs = n_pair_blocks * occupied_pair_batch_size
    t2_pairs = jnp.pad(
        jnp.asarray(t2).reshape(n_pairs, nvir, nvir),
        ((0, padded_pairs - n_pairs), (0, 0), (0, 0)))
    out = jnp.zeros((padded_pairs, nvir, nvir), dtype=jnp.float64)
    n_rank = p_virtual.shape[1]
    for m0 in range(0, n_rank, rank_panel):
        m1 = min(m0 + rank_panel, n_rank)
        p_m = p_virtual[:, m0:m1]
        tau_m_t = _t2_left_contract_pairs_jit(
            t2_pairs, p_m,
            occupied_pair_batch_size=occupied_pair_batch_size)
        for n0 in range(0, n_rank, rank_panel):
            n1 = min(n0 + rank_panel, n_rank)
            p_n = p_virtual[:, n0:n1]
            rank_metric = jnp.zeros((m1 - m0, n1 - n0), dtype=jnp.float64)
            for q0 in range(0, y.shape[1], aux_panel):
                q1 = min(q0 + aux_panel, y.shape[1])
                rank_metric = rank_metric + y[m0:m1, q0:q1] @ y[n0:n1, q0:q1].T
            out = _acc_into(out, _full_thc_block(p_m, p_n, tau_m_t, rank_metric))
    return out[:n_pairs].reshape(t2.shape)


# ------------------------------------------------------------ entry point ---
class ScalableDirectSandwichesJax:
    """Mirrors the oracle's result container, including the robust combination."""

    __slots__ = ("exact", "fit_left_df_right", "df_left_fit_right", "full_thc",
                 "robust")

    def __init__(self, exact, fit_left_df_right, df_left_fit_right, full_thc):
        self.exact = exact
        self.fit_left_df_right = fit_left_df_right
        self.df_left_fit_right = df_left_fit_right
        self.full_thc = full_thc
        # Explicitly fit-left/DF-right + DF-left/fit-right - full-THC, retaining
        # the two crosses separately so signs and RCCSD pair swaps stay auditable.
        self.robust = fit_left_df_right + df_left_fit_right - full_thc


def df_sandwiches_jax(b, fit, t2, *, rank_panel: int,
                                      aux_panel: int):
    """JAX port of the oracle's entry point. Same signature, same panel semantics.

    `fit` is the oracle's ScalableLSTHCFit (or anything exposing .p_virtual/.y),
    so the caller does not have to know which backend produced it.

    `b` is deliberately kept HOST-resident (see :func:`_as_fp64_host`): the
    panel loops below only read aux-axis slices, so uploading the whole
    block would cost its full size in device memory for no benefit.
    """
    require_float64()
    b_h = _as_fp64_host("b", b, 3)
    t2_j = _as_fp64_jax("t2", t2, 4)
    p_j = _as_fp64_jax("p_virtual", fit.p_virtual, 2)
    y_j = _as_fp64_jax("y", fit.y, 2)

    nvir = t2_j.shape[2]
    if t2_j.shape[2:] != (nvir, nvir):
        raise ValueError(f"t2 virtual axes must be square; got {t2_j.shape}")
    if b_h.shape[:2] != (nvir, nvir):
        raise ValueError(f"B virtual dimensions do not match t2: {b_h.shape} vs "
                         f"{t2_j.shape}")
    if p_j.shape[0] != nvir or y_j.shape != (p_j.shape[1], b_h.shape[2]):
        raise ValueError("implicit fit dimensions do not match B and t2")
    # Panel bounds mirror the oracle's _positive_panel exactly.
    for name, size, upper in (("rank_panel", rank_panel, p_j.shape[1]),
                              ("aux_panel", aux_panel, b_h.shape[2])):
        if not 1 <= int(size) <= upper:
            raise ValueError(f"{name} must be in [1, {upper}]; got {size}")

    exact = exact_df_panelled(b_h, t2_j, int(aux_panel))
    fit_left, df_left = partial_thc_crosses_panelled(
        b_h, p_j, y_j, t2_j, int(rank_panel), int(aux_panel))
    full = full_thc_panelled(p_j, y_j, t2_j, int(rank_panel), int(aux_panel))
    return ScalableDirectSandwichesJax(exact, fit_left, df_left, full)


def to_numpy(result):
    """Materialise a JAX result as NumPy float64 for comparison/accumulation."""
    return {name: np.asarray(getattr(result, name), dtype=np.float64)
            for name in ("exact", "fit_left_df_right", "df_left_fit_right",
                         "full_thc", "robust")}


def fit_lsthc_jax(p_virtual, b, *, rcond: float, virtual_panel: int):
    """JAX FP64 normal-equation LS-THC fit with host-resident DF factors."""
    require_float64()
    if not np.isfinite(rcond) or not 0.0 < float(rcond) <= 1.0:
        raise ValueError(f"rcond must be in (0, 1]; got {rcond!r}")
    p = _as_fp64_jax("p_virtual", p_virtual, 2)
    b_h = _as_fp64_host("b", b, 3)
    if b_h.shape[:2] != (p.shape[0], p.shape[0]):
        raise ValueError("B and P virtual dimensions differ")
    if not 1 <= int(virtual_panel) <= p.shape[0]:
        raise ValueError("virtual_panel is out of bounds")
    overlap = p.T @ p
    gram = overlap * overlap
    cross = jnp.zeros((p.shape[1], b_h.shape[2]), dtype=jnp.float64)
    for a0 in range(0, p.shape[0], int(virtual_panel)):
        a1 = min(a0 + int(virtual_panel), p.shape[0])
        b_panel = _as_fp64_jax("b_panel", b_h[a0:a1], 3)
        cross = cross + jnp.einsum(
            "am,cm,acq->mq", p[a0:a1], p, b_panel)
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (gram + gram.T))
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = jnp.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]
    singular_values = jnp.sqrt(eigenvalues)
    # Match the existing panelled oracle exactly.  This is a requested dense-C
    # threshold, not a dimension-dependent policy decision for this dispatch.
    floor = float(np.sqrt(np.finfo(np.float64).eps))
    threshold = max(float(rcond), floor) * singular_values[0]
    keep = singular_values > threshold
    inv = jnp.where(keep, 1.0 / jnp.where(eigenvalues > 0, eigenvalues, 1.0), 0.0)
    y = eigenvectors @ (inv[:, None] * (eigenvectors.T @ cross))
    return SimpleNamespace(p_virtual=p, y=y, gram=gram, cross=cross,
                           rcond=float(rcond), resolved_rcond=max(float(rcond), floor))

# ---------------- X-store layouts ----------------



def convert_x_to_rank_major(src, dst, *, dataset="X", row_block=8):
    """Convert one X dataset to rank-major layout, chunked over an nmo axis.

    ``src``/``dst`` are paths or open ``h5py.File`` objects.  The loop
    materializes one ``(nmo, row_block, rank)`` slab at a time, so peak
    host memory is ``nmo * row_block * rank * 8`` bytes (~1.6 GiB at the
    default block).  Returns the destination path or
    file unchanged.  The destination dataset is written contiguous (no
    HDF5 chunking) so panel reads stay single sequential extents.
    """

    close_src = not isinstance(src, h5py.File)
    close_dst = not isinstance(dst, h5py.File)
    src_f = h5py.File(src, "r") if close_src else src
    dst_f = h5py.File(dst, "w") if close_dst else dst
    try:
        _convert_x_dataset(src_f, dst_f, dataset, row_block)
        return dst
    finally:
        if close_src:
            src_f.close()
        if close_dst:
            dst_f.close()


def _convert_x_dataset(src_f, dst_f, dataset, row_block, out_name=None):
    if int(row_block) < 1:
        raise ValueError(f"row_block must be >= 1; got {row_block}")
    x_in = src_f[dataset]
    if x_in.ndim != 3 or x_in.shape[0] != x_in.shape[1]:
        raise ValueError(
            f"source dataset {dataset!r} must be (nmo, nmo, rank); "
            f"got {x_in.shape}")
    nmo, _, rank = x_in.shape
    x_out = dst_f.create_dataset(out_name or dataset,
                                 shape=(rank, nmo, nmo), dtype=np.float64)
    for key, val in x_in.attrs.items():
        x_out.attrs[key] = val
    x_out.attrs["x_layout"] = "rank_major"
    for j0 in range(0, nmo, int(row_block)):
        j1 = min(j0 + int(row_block), nmo)
        slab = np.asarray(x_in[j0:j1, :, :], dtype=np.float64)
        x_out[:, j0:j1, :] = np.ascontiguousarray(
            slab.transpose(2, 0, 1))


def convert_store_to_rank_major(src, dst, *, x_dataset="X", x_rm_dataset="X_rm",
                                row_block=8):
    """Whole-store conversion: every dataset copied; X kept AND a rank-major
    twin appended.

    The ISDF store carries more than X (K1/K3/D kernels, metadata); a
    store the driver can actually load needs all of it.  Top-level
    datasets and file attributes are copied verbatim -- including
    ``x_dataset`` itself, since legacy consumers read it as
    ``(nmo, nmo, rank)`` -- and a rank-major twin ``x_rm_dataset`` is
    added alongside (with the ``x_layout`` attribute and a provenance
    stamp of the source X).

    Writes go to ``<dst>.tmp`` and are atomically renamed into place, so
    an interrupted conversion never leaves a partial file at the
    destination.
    """

    tmp = f"{dst}.tmp"
    with h5py.File(src, "r") as src_f, h5py.File(tmp, "w") as dst_f:
        for key, val in src_f.attrs.items():
            dst_f.attrs[key] = val
        for key in src_f:
            src_f.copy(key, dst_f)
        _convert_x_dataset(src_f, dst_f, x_dataset, row_block,
                           out_name=x_rm_dataset)
        dst_f[x_rm_dataset].attrs["x_source_stamp"] = (
            _x_source_stamp(src_f[x_dataset]))
    os.replace(tmp, dst)
    return dst


def _x_source_stamp(x):
    """Provenance stamp for an X dataset: sha256 over shape plus EVERY row
    slab, so any changed entry changes the stamp (no sampling shortcuts)."""

    import hashlib
    h = hashlib.sha256(str(tuple(x.shape)).encode())
    nmo = x.shape[0]
    step = max(1, nmo // 64)
    for i0 in range(0, nmo, step):
        h.update(np.asarray(x[i0:i0 + step], dtype=np.float64).tobytes())
    return h.hexdigest()


def add_rank_major(store, *, x_dataset="X", out_dataset="X_rm", row_block=8):
    """Append a rank-major copy of X to an existing store, in place.

    The store keeps ``x_dataset`` (innermost) untouched for legacy
    consumers (eris build, fingerprint manifest); the factorized
    contraction reads ``out_dataset`` when present.  An existing
    ``out_dataset`` is kept only when it is shape/attr-valid AND its
    provenance stamp matches the current ``x_dataset``; a stale twin
    (X rewritten after the twin was built) is rebuilt.
    """

    with h5py.File(store, "r+") as fh:
        x_in = fh[x_dataset]
        if x_in.ndim != 3 or x_in.shape[0] != x_in.shape[1]:
            raise ValueError(
                f"{x_dataset!r} must be (nmo, nmo, rank); got {x_in.shape}")
        nmo, _, rank = x_in.shape
        if out_dataset in fh:
            x_rm = fh[out_dataset]
            valid = (tuple(x_rm.shape) == (rank, nmo, nmo)
                     and x_rm.attrs.get("x_layout") == "rank_major")
            if not valid:
                raise ValueError(
                    f"{out_dataset!r} exists but is not a valid rank-major X: "
                    f"shape {x_rm.shape}, attrs {dict(x_rm.attrs)}")
            if x_rm.attrs.get("x_source_stamp") == _x_source_stamp(x_in):
                return store
            del fh[out_dataset]
        # Write to a temp dataset and rename into place: an interrupted
        # conversion must never leave a correctly-shaped but partially
        # written X_rm behind.
        tmp_name = out_dataset + ".tmp"
        if tmp_name in fh:
            del fh[tmp_name]
        _convert_x_dataset(fh, fh, x_dataset, row_block, out_name=tmp_name)
        fh[tmp_name].attrs["x_source_stamp"] = _x_source_stamp(x_in)
        fh.move(tmp_name, out_dataset)
    return store


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", help="source store path (rank-innermost X)")
    parser.add_argument("dst", nargs="?", default=None,
                        help="destination path (not used by --add-rank-major)")
    parser.add_argument("--dataset", default="X",
                        help="dataset name in both files (default: X)")
    parser.add_argument("--row-block", type=int, default=8,
                        help="nmo rows converted per slab (default: 8)")
    parser.add_argument("--whole-store", action="store_true",
                        help="copy all datasets/attrs, converting only "
                             "--dataset (a driver-loadable store)")
    parser.add_argument("--add-rank-major", action="store_true",
                        help="append X_rm (rank-major) to the src store IN "
                             "PLACE, leaving X innermost untouched")
    args = parser.parse_args(argv)
    if args.add_rank_major:
        add_rank_major(args.src, x_dataset=args.dataset,
                       row_block=args.row_block)
    else:
        if args.dst is None:
            parser.error("dst is required unless --add-rank-major is given")
        if args.whole_store:
            convert_store_to_rank_major(
                args.src, args.dst, x_dataset=args.dataset,
                row_block=args.row_block)
        else:
            convert_x_to_rank_major(
                args.src, args.dst, dataset=args.dataset,
                row_block=args.row_block)


if __name__ == "__main__":
    main()
