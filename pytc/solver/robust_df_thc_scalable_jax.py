"""JAX/GPU port of the panelled robust-DF/THC Coulomb sandwich (task #46, Phase E).

WHY: at QZ the NumPy original is 783 s = 98.1% of arm-2's contraction cost
(JID 18874875), which is why factor-direct is 16.84x faster per cycle yet 2.31x
slower end-to-end. Porting it is the lever that converts the contraction win into
an end-to-end win.

WHAT THIS IS NOT: a rewrite. The NumPy module is the immovable parity oracle and
stays untouched (Felix, msg bd7e6ea3). This file mirrors it kernel-for-kernel and
panel-for-panel.

PANELLING IS PRESERVED DELIBERATELY (Felix ruling, msg 435b7557). #46 isolates
exactly ONE variable: NumPy -> JAX. Panel size is a blocking parameter and blocking
is a fairness axis (gate 3), so a port that also changed effective panelling would
make the rerun measure two things at once -- the same trap class as matched-blk in
task #44. "Does different panelling go faster on 181 GB HBM" is a legitimate
SEPARATE question with its own ledger; it is not this file's job.

Consequences of that choice, stated so nobody reads them as oversights:
  - the Python-level panel loops are retained rather than fused into one big
    einsum. They are the structure under test.
  - each panel's einsum is jitted; the loop itself is not unrolled into a scan,
    because a scan would change the blocking the ledger reports.
  - float64 is enforced. JAX defaults to float32 and would silently produce a
    ~1e-7 "parity" result that looks plausible and is wrong -- exactly the
    silent-wrong class this campaign keeps hitting. See require_float64().

Parity target: ~1e-12 vs the NumPy oracle on CPU AND GPU, including the task-35
panel-boundary adversarial cases, BEFORE any timing claim.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

# JAX is float32 by default. Enabling x64 must happen before any array is created;
# doing it at import time is deliberate, and require_float64() verifies it actually
# took effect rather than assuming the flag was honoured.
jax.config.update("jax_enable_x64", True)


class Float64NotEnabled(RuntimeError):
    """Raised when x64 is off -- a float32 run would look like ~1e-7 parity."""


def require_float64():
    """Fail closed if JAX is not in x64 mode.

    Without this the port silently computes in float32, parity comes out around
    1e-7, and a reviewer comparing against a ~1e-12 target sees 'close enough'.
    A wrong dtype is not a small error here; it is a different calculation.
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


# ---------------------------------------------------------------- kernels ---
# Each jitted body is ONE panel's arithmetic, matching the oracle's einsum
# subscripts exactly. Subscripts are copied verbatim from the NumPy source: they
# encode the (a,c,b,d) source ordering that the RCCSD pair swaps depend on, and
# "simplifying" them is how a port silently breaks pair-swap symmetry.

@jax.jit
def _exact_panel(b_panel, t2):
    right = jnp.einsum("ijcd,bdq->ijcbq", t2, b_panel)
    return jnp.einsum("acq,ijcbq->ijab", b_panel, right)


def exact_df_panelled(b, t2, aux_panel: int):
    """Exact current-DF sandwich, panelled over the auxiliary axis."""
    out = jnp.zeros_like(t2)
    for q0 in range(0, b.shape[2], aux_panel):
        q1 = min(q0 + aux_panel, b.shape[2])
        out = out + _exact_panel(b[:, :, q0:q1], t2)
    return out


@jax.jit
def _endpoint_panel(b_q, y_mq):
    return jnp.einsum("bdq,mq->bdm", b_q, y_mq)


@jax.jit
def _cross_panel(p_panel, endpoint, t2):
    tau_left = jnp.einsum("ijcd,cm->ijmd", t2, p_panel)
    right = jnp.einsum("ijmd,bdm->ijmb", tau_left, endpoint)
    fit_left = jnp.einsum("am,ijmb->ijab", p_panel, right)

    tau_right = jnp.einsum("ijcd,dm->ijcm", t2, p_panel)
    left = jnp.einsum("acm,ijcm->ijam", endpoint, tau_right)
    df_left = jnp.einsum("bm,ijam->ijab", p_panel, left)
    return fit_left, df_left


def partial_thc_crosses_panelled(b, p_virtual, y, t2, rank_panel: int,
                                 aux_panel: int):
    """Both partial-THC crosses with a precontracted DF endpoint.

    The endpoint D[b,d,m] is accumulated over aux panels BEFORE the occupied-pair
    contractions, so no (ij,m,b,Qp) intermediate is ever retained -- that property
    is the point of the panelled formulation and is preserved here.
    """
    fit_left_df_right = jnp.zeros_like(t2)
    df_left_fit_right = jnp.zeros_like(t2)
    for m0 in range(0, p_virtual.shape[1], rank_panel):
        m1 = min(m0 + rank_panel, p_virtual.shape[1])
        p_panel = p_virtual[:, m0:m1]
        endpoint = jnp.zeros((b.shape[0], b.shape[1], m1 - m0), dtype=jnp.float64)
        for q0 in range(0, b.shape[2], aux_panel):
            q1 = min(q0 + aux_panel, b.shape[2])
            endpoint = endpoint + _endpoint_panel(b[:, :, q0:q1], y[m0:m1, q0:q1])
        fit_left, df_left = _cross_panel(p_panel, endpoint, t2)
        fit_left_df_right = fit_left_df_right + fit_left
        df_left_fit_right = df_left_fit_right + df_left
    return fit_left_df_right, df_left_fit_right


@jax.jit
def _full_thc_block(p_m, p_n, tau_m, rank_metric):
    tau_mn = jnp.einsum("ijmd,dn->ijmn", tau_m, p_n)
    weighted = tau_mn * rank_metric[None, None]
    right = jnp.einsum("ijmn,bn->ijmb", weighted, p_n)
    return jnp.einsum("am,ijmb->ijab", p_m, right)


def full_thc_panelled(p_virtual, y, t2, rank_panel: int, aux_panel: int):
    """B_tilde[a,c,Q] B_tilde[b,d,Q] t2[ij,c,d] without ever forming B_tilde."""
    out = jnp.zeros_like(t2)
    n_rank = p_virtual.shape[1]
    for m0 in range(0, n_rank, rank_panel):
        m1 = min(m0 + rank_panel, n_rank)
        p_m = p_virtual[:, m0:m1]
        tau_m = jnp.einsum("ijcd,cm->ijmd", t2, p_m)
        for n0 in range(0, n_rank, rank_panel):
            n1 = min(n0 + rank_panel, n_rank)
            p_n = p_virtual[:, n0:n1]
            rank_metric = jnp.zeros((m1 - m0, n1 - n0), dtype=jnp.float64)
            for q0 in range(0, y.shape[1], aux_panel):
                q1 = min(q0 + aux_panel, y.shape[1])
                rank_metric = rank_metric + y[m0:m1, q0:q1] @ y[n0:n1, q0:q1].T
            out = out + _full_thc_block(p_m, p_n, tau_m, rank_metric)
    return out


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


def direct_df_sandwiches_panelled_jax(b, fit, t2, *, rank_panel: int,
                                      aux_panel: int):
    """JAX port of the oracle's entry point. Same signature, same panel semantics.

    `fit` is the oracle's ScalableLSTHCFit (or anything exposing .p_virtual/.y),
    so the caller does not have to know which backend produced it.
    """
    require_float64()
    b_j = _as_fp64_jax("b", b, 3)
    t2_j = _as_fp64_jax("t2", t2, 4)
    p_j = _as_fp64_jax("p_virtual", fit.p_virtual, 2)
    y_j = _as_fp64_jax("y", fit.y, 2)

    nvir = t2_j.shape[2]
    if t2_j.shape[2:] != (nvir, nvir):
        raise ValueError(f"t2 virtual axes must be square; got {t2_j.shape}")
    if b_j.shape[:2] != (nvir, nvir):
        raise ValueError(f"B virtual dimensions do not match t2: {b_j.shape} vs "
                         f"{t2_j.shape}")
    if p_j.shape[0] != nvir or y_j.shape != (p_j.shape[1], b_j.shape[2]):
        raise ValueError("implicit fit dimensions do not match B and t2")
    # Panel bounds mirror the oracle's _positive_panel exactly.
    for name, size, upper in (("rank_panel", rank_panel, p_j.shape[1]),
                              ("aux_panel", aux_panel, b_j.shape[2])):
        if not 1 <= int(size) <= upper:
            raise ValueError(f"{name} must be in [1, {upper}]; got {size}")

    exact = exact_df_panelled(b_j, t2_j, int(aux_panel))
    fit_left, df_left = partial_thc_crosses_panelled(
        b_j, p_j, y_j, t2_j, int(rank_panel), int(aux_panel))
    full = full_thc_panelled(p_j, y_j, t2_j, int(rank_panel), int(aux_panel))
    return ScalableDirectSandwichesJax(exact, fit_left, df_left, full)


def to_numpy(result):
    """Materialise a JAX result as NumPy float64 for comparison/accumulation."""
    return {name: np.asarray(getattr(result, name), dtype=np.float64)
            for name in ("exact", "fit_left_df_right", "df_left_fit_right",
                         "full_thc", "robust")}


def fit_panelled_lsthc_jax(p_virtual, b, *, rcond: float, virtual_panel: int):
    """JAX FP64 normal-equation LS-THC fit, retaining the oracle's panels."""
    require_float64()
    p = _as_fp64_jax("p_virtual", p_virtual, 2)
    b = _as_fp64_jax("b", b, 3)
    if b.shape[:2] != (p.shape[0], p.shape[0]):
        raise ValueError("B and P virtual dimensions differ")
    if not 1 <= int(virtual_panel) <= p.shape[0]:
        raise ValueError("virtual_panel is out of bounds")
    overlap = p.T @ p
    gram = overlap * overlap
    cross = jnp.zeros((p.shape[1], b.shape[2]), dtype=jnp.float64)
    for a0 in range(0, p.shape[0], int(virtual_panel)):
        a1 = min(a0 + int(virtual_panel), p.shape[0])
        cross = cross + jnp.einsum("am,cm,acq->mq", p[a0:a1], p, b[a0:a1])
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (gram + gram.T))
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = jnp.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]
    singular_values = jnp.sqrt(eigenvalues)
    floor = float(np.sqrt(np.finfo(np.float64).eps * p.shape[1]))
    threshold = max(float(rcond), floor) * singular_values[0]
    keep = singular_values > threshold
    inv = jnp.where(keep, 1.0 / jnp.where(eigenvalues > 0, eigenvalues, 1.0), 0.0)
    y = eigenvectors @ (inv[:, None] * (eigenvectors.T @ cross))
    return SimpleNamespace(p_virtual=p, y=y, gram=gram, cross=cross,
                           rcond=float(rcond), resolved_rcond=max(float(rcond), floor))
