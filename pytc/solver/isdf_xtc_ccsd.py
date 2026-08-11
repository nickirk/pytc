"""Factorized ISDF xTC-CCSD: RCCSD dispatch + factor-direct VVVV--T2 kernels.

The :class:`RCCSD` class inherits the CCSD machinery of
:mod:`pytc.solver.jax_xtc_ccsd` and replaces only the VVVV--T2 leg through
that module's instance hook.  The rest of this module is the factor-direct
contraction machinery it dispatches to: the X factor is never materialized
in full -- it stays on its host/HDF5 backing and is streamed one rank panel
at a time (three residency tiers: device lift / host-resident / store
stream), which is what makes large systems fittable on one GPU.

The raw ERI-like tile order in PyTC is ``(a, c, b, d)``; the public
functions contract that tile directly with a dense RCCSD ``t2`` in
``(i, j, c, d)`` order, without creating a ``(v, v, v, v)`` tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
import os
import time
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from pytc.df import thc
from pytc.df.thc import (
    df_sandwiches_jax,
    fit_lsthc_jax,
)
from pytc.solver import jax_xtc_ccsd
from pytc.utils import tile_timers as _tile_timers
from pytc.utils.prefetch import async_read, await_read

_logger = logging.getLogger(__name__)




class RCCSD(jax_xtc_ccsd.RCCSD):
    """JAX RCCSD whose default on-the-fly VVVV path is factorized."""

    factorized_rank_panel = 128
    factorized_aux_panel = 32
    factorized_virtual_panel = 32
    factorized_rcond = 1.0e-12

    # The contraction reads X panel-wise from its store backing, so the
    # materialized path's whole-X host preload (xtc_ccsd._make_xtc_eris) is
    # pure waste on this class: it is hundreds of GB of
    # host RAM for a copy nothing reads.  The preload is suppressed by
    # construction; the parent's materialized default is unchanged.
    _preload_x_for_eris = False

    def _factorized_state(self):
        state = getattr(self, "_isdf_factorized_state", None)
        if state is not None:
            return state
        base = self.xtc_obj
        nocc = self.nocc
        kernels = base.isdf_kernels
        required = ("K1_kernel", "K3_kernel", "D", "X")
        missing = [name for name in required if name not in kernels]
        if missing:
            raise RuntimeError(f"ISDF kernels missing for factorized RCCSD: {missing}")
        tc = {
            "p": np.asarray(base.phi_isdf[nocc:], dtype=np.float64),
            "grad_p": np.asarray(base.grad_phi_isdf[nocc:], dtype=np.float64),
            "u1": np.asarray(kernels["K1_kernel"], dtype=np.float64),
            "u3": np.asarray(kernels["K3_kernel"], dtype=np.float64),
            "d": np.asarray(kernels["D"], dtype=np.float64),
        }
        # X stays on its backing (a NumPy array or an HDF5 dataset): the
        # contraction streams it one rank panel at a time, so the full block
        # is never materialized on host or device.  Device X peak is one
        # panel.  When
        # the store carries a rank-major twin (X_rm, panel-contiguous
        # reads), the contraction uses it; legacy consumers keep X.
        x_backing = kernels.get("X_rm", kernels["X"])
        with_df = getattr(self, "with_df", None) or self._scf.with_df
        b = thc.extract_vv_df_factor(
            with_df, self.mo_coeff, nocc)
        nvir = tc["p"].shape[0]
        fit = fit_lsthc_jax(
            tc["p"], b, rcond=self.factorized_rcond,
            virtual_panel=min(self.factorized_virtual_panel, nvir))
        state = (tc, b, fit, x_backing)
        self._isdf_factorized_state = state
        return state

    def _contract_vvvv_t2(self, cc, t2_jax, eris, t2new_host):
        """Instance hook called by the inherited JAX update path; no VVVV tile."""
        del cc
        if (
            os.environ.get("PYTC_XTC_DROP_X") == "1"
            or os.environ.get("PYTC_XTC_DROP_X_RESIDUAL") == "1"
        ):
            raise RuntimeError(
                "PYTC_XTC_DROP_X and PYTC_XTC_DROP_X_RESIDUAL are not "
                "implemented for the factorized VVVV solver; use "
                "jax_xtc_ccsd.RCCSD for the normal-order/residual-X "
                "partition study"
            )
        if eris.vvvv is not None:
            raise RuntimeError(
                "factorized RCCSD refuses a materialized VVVV store; select "
                "jax_xtc_ccsd.RCCSD for the legacy materialized route")
        tc, b, fit, x_backing = self._factorized_state()
        rank_panel = min(self.factorized_rank_panel, fit.p_virtual.shape[1])
        aux_panel = min(self.factorized_aux_panel, b.shape[2])
        with _tile_timers.term("fd_isdf_terms") as _tt:
            terms = contract_terms_t2_auto(
                t2_jax, **tc, x_backing=x_backing, nocc=self.nocc,
                occupied_pair_batch_size=min(8, self.nocc * self.nocc),
                rank_panel_size=rank_panel)
            final = terms["final"]
            # Only "final" is consumed below; the other term tensors are
            # diagnostics and are dropped before the sandwich.
            del terms
            _tt.sync(final)
        with _tile_timers.term("fd_coulomb_sandwich") as _tt:
            coulomb = df_sandwiches_jax(
                b, fit, t2_jax, rank_panel=rank_panel, aux_panel=aux_panel)
            _tt.sync(coulomb.robust)
        t2new_host += np.asarray(final + coulomb.robust, dtype=np.float64)


# ============================================================================
# Factor-direct VVVV--T2 contraction machinery
# ============================================================================


Array = jax.Array


@dataclass(frozen=True)
class FactorDirectProfile:
    """One steady-state contraction measurement.

    ``schedule_intermediate_estimate_bytes`` counts only the named panels in
    the factor-direct algebra.  It excludes compiler-generated scratch and
    whole-array padding copies, so it is explicitly *not* an allocator bound.

    The ``compiled_xla_*`` fields come from ``Compiled.memory_analysis()`` for
    the executable that evaluates this branch on the active backend.  They are
    not a claim about process-wide allocator high-water mark.
    """

    wall_seconds: float
    schedule_intermediate_estimate_bytes: int
    compiled_xla_temporary_bytes: int
    compiled_xla_argument_bytes: int
    compiled_xla_output_bytes: int
    compiled_xla_alias_bytes: int
    compiled_xla_total_bytes: int
    rank_panel_size: int
    occupied_pair_batch_size: int
    materializes_v4: bool = False


@dataclass(frozen=True)
class CompiledXLAMemory:
    """Backend-specific buffer accounting reported by a JAX executable."""

    temporary_bytes: int
    argument_bytes: int
    output_bytes: int
    alias_bytes: int

    @property
    def total_bytes(self) -> int:
        """XLA's argument + output + temporary accounting, less aliases."""

        return self.argument_bytes + self.output_bytes + self.temporary_bytes - self.alias_bytes


def _dtype_itemsize(*arrays: Array) -> int:
    return max(jnp.dtype(array.dtype).itemsize for array in arrays)


def full_thc_schedule_intermediate_estimate_bytes(
    *,
    nvir: int,
    rank: int,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
    itemsize: int,
) -> int:
    """Estimate only the named live algebraic panels for a full-THC branch.

    The schedule has a ``t2`` slab, an output slab, and ``S``, ``T``, and
    ``Y``.  No term has four virtual indices, so this is
    O(B_ij v^2 + B_ij v B_r + B_ij r B_r), not O(v^4). It omits XLA scratch
    and whole-array ``jnp.pad`` copies; use ``compiled_full_thc_memory`` for
    executable memory accounting.
    """

    b_ij = occupied_pair_batch_size
    b_r = min(rank_panel_size, rank)
    elements = b_ij * (
        2 * nvir * nvir + 2 * nvir * b_r + rank * b_r
    )
    return int(elements * itemsize)


def partial_x_schedule_intermediate_estimate_bytes(
    *,
    nvir: int,
    rank: int,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
    itemsize: int,
) -> int:
    """Estimate only the named live algebraic panels for an X branch.

    The estimate deliberately excludes the full ``x`` / ``x_padded`` arrays;
    it covers the padded-panel schedule only.
    """

    del rank  # The implementation only holds a panel of the rank axis.
    b_ij = occupied_pair_batch_size
    b_r = rank_panel_size
    # tau slab + output slab + S + Y.
    elements = b_ij * (2 * nvir * nvir + 2 * nvir * b_r)
    return int(elements * itemsize)


def _validate_t2(t2: Array) -> None:
    if t2.ndim != 4:
        raise ValueError(f"t2 must have shape (nocc, nocc, nvir, nvir); got {t2.shape}")
    if t2.shape[2] != t2.shape[3]:
        raise ValueError(f"t2 virtual axes must be square; got {t2.shape}")


def _validate_full_thc(
    t2: Array, a: Array, b: Array, z: Array, c: Array, e: Array
) -> None:
    _validate_t2(t2)
    nvir = t2.shape[2]
    rank = z.shape[0]
    expected = (nvir, rank)
    for name, value in (("a", a), ("b", b), ("c", c), ("e", e)):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}; got {value.shape}")
    if z.shape != (rank, rank):
        raise ValueError(f"z must be square; got {z.shape}")


def _compiled_memory_from_executable(executable) -> CompiledXLAMemory:
    """Return stable integer fields from JAX's executable memory analysis."""

    if not hasattr(executable, "memory_analysis"):
        raise RuntimeError(
            "the active JAX backend does not expose Compiled.memory_analysis(); "
            "the GPU card requires executable memory accounting"
        )
    analysis = executable.memory_analysis()
    return CompiledXLAMemory(
        temporary_bytes=int(analysis.temp_size_in_bytes),
        argument_bytes=int(analysis.argument_size_in_bytes),
        output_bytes=int(analysis.output_size_in_bytes),
        alias_bytes=int(analysis.alias_size_in_bytes),
    )


def _contract_full_thc_t2_impl(
    t2: Array,
    a: Array,
    b: Array,
    z: Array,
    c: Array,
    e: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> Array:
    """JIT implementation of the blocked full-THC VVVV--T2 contraction."""

    nocc_i, nocc_j, nvir, _ = t2.shape
    rank = z.shape[0]
    n_pairs = nocc_i * nocc_j
    n_pair_blocks = (n_pairs + occupied_pair_batch_size - 1) // occupied_pair_batch_size
    n_rank_blocks = (rank + rank_panel_size - 1) // rank_panel_size
    padded_pairs = n_pair_blocks * occupied_pair_batch_size
    padded_rank = n_rank_blocks * rank_panel_size
    t2_pairs = jnp.pad(
        t2.reshape(n_pairs, nvir, nvir),
        ((0, padded_pairs - n_pairs), (0, 0), (0, 0)),
    )
    b_padded = jnp.pad(b, ((0, 0), (0, padded_rank - rank)))
    e_padded = jnp.pad(e, ((0, 0), (0, padded_rank - rank)))
    z_padded = jnp.pad(z, ((0, 0), (0, padded_rank - rank)))
    result = jnp.zeros((padded_pairs, nvir, nvir), dtype=t2.dtype)

    # ``fori_loop`` stages a single rank-panel body instead of unrolling one
    # JAX program per panel.  Every live rank-dependent temporary remains a
    # B_ij x r x B_r or B_ij x v x B_r panel.
    def pair_body(pair_block, result_acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau_block = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir),
        )

        def rank_body(rank_block, out_acc):
            rank0 = rank_block * rank_panel_size
            b_panel = jax.lax.dynamic_slice(
                b_padded, (0, rank0), (nvir, rank_panel_size),
            )
            e_panel = jax.lax.dynamic_slice(
                e_padded, (0, rank0), (nvir, rank_panel_size),
            )
            z_panel = jax.lax.dynamic_slice(
                z_padded, (0, rank0), (rank, rank_panel_size),
            )
            # S[n,c,nu] = sum_d tau[n,c,d] E[d,nu]
            s = jnp.einsum("ncd,dq->ncq", tau_block, e_panel)
            # T[n,mu,nu] = sum_c C[c,mu] S[n,c,nu]
            t = jnp.einsum("cm,ncq->nmq", c, s)
            t = t * z_panel[None, :, :]
            # Y[n,a,nu] = sum_mu A[a,mu] T[n,mu,nu]
            y = jnp.einsum("am,nmq->naq", a, t)
            return out_acc + jnp.einsum("naq,bq->nab", y, b_panel)

        out_block = jax.lax.fori_loop(
            0, n_rank_blocks, rank_body,
            jnp.zeros((occupied_pair_batch_size, nvir, nvir), dtype=t2.dtype),
        )
        return jax.lax.dynamic_update_slice(result_acc, out_block, (pair0, 0, 0))

    result = jax.lax.fori_loop(0, n_pair_blocks, pair_body, result)
    return result[:n_pairs].reshape(nocc_i, nocc_j, nvir, nvir)


_contract_full_thc_t2_jit = jax.jit(
    _contract_full_thc_t2_impl,
    static_argnames=("occupied_pair_batch_size", "rank_panel_size"),
)


def contract_full_thc_t2(
    t2: Array,
    a: Array,
    b: Array,
    z: Array,
    c: Array,
    e: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> Array:
    """Contract a full-THC ``(a,c,b,d)`` term with a dense RCCSD ``t2``.

    ``a``/``c`` are the left endpoint factors and ``b``/``e`` the right
    endpoint factors.  The central kernel ``z`` has shape ``(rank, rank)``.
    The returned residual has the standard ``(i,j,a,b)`` layout.
    """

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    arrays = tuple(map(jnp.asarray, (t2, a, b, z, c, e)))
    _validate_full_thc(*arrays)
    return _contract_full_thc_t2_jit(
        *arrays,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )


def compiled_full_thc_memory(
    t2: Array,
    a: Array,
    b: Array,
    z: Array,
    c: Array,
    e: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> CompiledXLAMemory:
    """Return XLA accounting for the compiled full-THC branch executable."""

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    arrays = tuple(map(jnp.asarray, (t2, a, b, z, c, e)))
    _validate_full_thc(*arrays)
    executable = _contract_full_thc_t2_jit.lower(
        *arrays,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    ).compile()
    return _compiled_memory_from_executable(executable)


@partial(jax.jit, static_argnames=("occupied_pair_batch_size", "rank_panel_size"))
def _contract_k1_direct_t2_jit(
    t2: Array,
    p: Array,
    grad_p: Array,
    u1: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> Array:
    """One executable for the three-component direct K1 term."""

    result = jnp.zeros_like(t2)
    for gamma in range(3):
        result = result + _contract_full_thc_t2_impl(
            t2, grad_p[:, :, gamma], p, u1[:, :, gamma], p, p,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        )
    return result


@partial(jax.jit, static_argnames=("occupied_pair_batch_size", "rank_panel_size"))
def _contract_k1_pair_t2_jit(
    t2: Array,
    p: Array,
    grad_p: Array,
    u1: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> Array:
    """One executable for the three-component pair-swapped K1 term."""

    result = jnp.zeros_like(t2)
    for gamma in range(3):
        result = result + _contract_full_thc_t2_impl(
            t2, p, grad_p[:, :, gamma], u1[:, :, gamma].T, p, p,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        )
    return result


@partial(jax.jit, static_argnames=("occupied_pair_batch_size", "rank_panel_size"))
def _contract_k2_direct_t2_jit(
    t2: Array,
    p: Array,
    grad_p: Array,
    u1: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> Array:
    """One executable for the three-component direct K2 term."""

    result = jnp.zeros_like(t2)
    for gamma in range(3):
        result = result + _contract_full_thc_t2_impl(
            t2, p, p, u1[:, :, gamma], grad_p[:, :, gamma], p,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        )
    return result


@partial(jax.jit, static_argnames=("occupied_pair_batch_size", "rank_panel_size"))
def _contract_k2_pair_t2_jit(
    t2: Array,
    p: Array,
    grad_p: Array,
    u1: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> Array:
    """One executable for the three-component pair-swapped K2 term."""

    result = jnp.zeros_like(t2)
    for gamma in range(3):
        result = result + _contract_full_thc_t2_impl(
            t2, p, p, u1[:, :, gamma].T, p, grad_p[:, :, gamma],
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        )
    return result


def _compiled_k12_memory(executable, t2, p, grad_p, u1, *,
                          occupied_pair_batch_size: int, rank_panel_size: int) -> CompiledXLAMemory:
    """Compile the exact three-component K1/K2 branch for its XLA accounting."""

    return _compiled_memory_from_executable(
        executable.lower(
            t2, p, grad_p, u1,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ).compile()
    )


def contract_full_thc_pair_swapped_t2(
    t2: Array,
    a: Array,
    b: Array,
    z: Array,
    c: Array,
    e: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> Array:
    """Contract ``V[b,d,a,c]`` for a full-THC ``V[a,c,b,d]`` term.

    The swapped contribution changes both endpoint order and the central
    kernel orientation.  Keeping this explicit is essential: it avoids an
    unverified symmetry assumption for K1/K2 and their gradient endpoint.
    """

    return contract_full_thc_t2(
        t2, b, a, jnp.asarray(z).T, e, c,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )


def _validate_x_factors(
    t2: Array, out_factor: Array, inner_factor: Array
) -> int:
    _validate_t2(t2)
    nvir = t2.shape[2]
    if out_factor.ndim != 2 or inner_factor.shape != out_factor.shape:
        raise ValueError("X endpoint factors must both have shape (nvir, rank)")
    if out_factor.shape[0] != nvir:
        raise ValueError(f"X factors have nvir={out_factor.shape[0]}, expected {nvir}")
    return nvir


def _validate_x(t2: Array, out_factor: Array, inner_factor: Array, x: Array) -> None:
    nvir = _validate_x_factors(t2, out_factor, inner_factor)
    if x.shape != (nvir, nvir, out_factor.shape[1]):
        raise ValueError(
            "x must have shape (nvir, nvir, rank); "
            f"got {x.shape}, expected {(nvir, nvir, out_factor.shape[1])}"
        )


def _contract_partial_x(
    kernel,
    t2: Array,
    out_factor: Array,
    inner_factor: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> Array:
    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    arrays = tuple(map(jnp.asarray, (t2, out_factor, inner_factor, x)))
    _validate_x(*arrays)
    return kernel(
        *arrays,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )


def _compiled_partial_x_memory(
    kernel,
    t2: Array,
    out_factor: Array,
    inner_factor: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> CompiledXLAMemory:
    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    arrays = tuple(map(jnp.asarray, (t2, out_factor, inner_factor, x)))
    _validate_x(*arrays)
    executable = kernel.lower(
        *arrays,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    ).compile()
    return _compiled_memory_from_executable(executable)


@partial(jax.jit, static_argnames=("occupied_pair_batch_size", "rank_panel_size"))
def _contract_x_left_t2_jit(
    t2: Array,
    left_out: Array,
    left_inner: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> Array:
    """JIT implementation for P[a,mu] P[c,mu] X[b,d,mu]."""

    nocc_i, nocc_j, nvir, _ = t2.shape
    rank = left_out.shape[1]
    n_pairs = nocc_i * nocc_j
    n_pair_blocks = (n_pairs + occupied_pair_batch_size - 1) // occupied_pair_batch_size
    n_rank_blocks = (rank + rank_panel_size - 1) // rank_panel_size
    padded_pairs = n_pair_blocks * occupied_pair_batch_size
    padded_rank = n_rank_blocks * rank_panel_size
    t2_pairs = jnp.pad(
        t2.reshape(n_pairs, nvir, nvir),
        ((0, padded_pairs - n_pairs), (0, 0), (0, 0)),
    )
    inner_padded = jnp.pad(left_inner, ((0, 0), (0, padded_rank - rank)))
    out_padded = jnp.pad(left_out, ((0, 0), (0, padded_rank - rank)))
    x_padded = jnp.pad(x, ((0, 0), (0, 0), (0, padded_rank - rank)))
    result = jnp.zeros((padded_pairs, nvir, nvir), dtype=t2.dtype)

    def pair_body(pair_block, result_acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau_block = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir),
        )

        def rank_body(rank_block, out_acc):
            rank0 = rank_block * rank_panel_size
            inner_panel = jax.lax.dynamic_slice(
                inner_padded, (0, rank0), (nvir, rank_panel_size),
            )
            out_panel = jax.lax.dynamic_slice(
                out_padded, (0, rank0), (nvir, rank_panel_size),
            )
            x_panel = jax.lax.dynamic_slice(
                x_padded, (0, 0, rank0), (nvir, nvir, rank_panel_size),
            )
            # S[n,d,mu] = sum_c tau[n,c,d] P[c,mu]
            s = jnp.einsum("ncd,cm->ndm", tau_block, inner_panel)
            # Y[n,b,mu] = sum_d S[n,d,mu] X[b,d,mu]
            y = jnp.einsum("ndm,bdm->nbm", s, x_panel)
            return out_acc + jnp.einsum("am,nbm->nab", out_panel, y)

        out_block = jax.lax.fori_loop(
            0, n_rank_blocks, rank_body,
            jnp.zeros((occupied_pair_batch_size, nvir, nvir), dtype=t2.dtype),
        )
        return jax.lax.dynamic_update_slice(result_acc, out_block, (pair0, 0, 0))

    result = jax.lax.fori_loop(0, n_pair_blocks, pair_body, result)
    return result[:n_pairs].reshape(nocc_i, nocc_j, nvir, nvir)


def contract_x_left_t2(
    t2: Array,
    left_out: Array,
    left_inner: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> Array:
    """Contract ``P[a,m] P[c,m] X[b,d,m]`` without a V^4 tile."""

    return _contract_partial_x(
        _contract_x_left_t2_jit, t2, left_out, left_inner, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )


def compiled_partial_x_left_memory(
    t2: Array,
    left_out: Array,
    left_inner: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> CompiledXLAMemory:
    """Return XLA accounting for ``P[a,m] P[c,m] X[b,d,m]``."""

    return _compiled_partial_x_memory(
        _contract_x_left_t2_jit, t2, left_out, left_inner, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )


@partial(jax.jit, static_argnames=("occupied_pair_batch_size", "rank_panel_size"))
def _contract_x_right_t2_jit(
    t2: Array,
    right_out: Array,
    right_inner: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int,
    rank_panel_size: int,
) -> Array:
    """JIT implementation for X[a,c,mu] P[b,mu] P[d,mu]."""

    nocc_i, nocc_j, nvir, _ = t2.shape
    rank = right_out.shape[1]
    n_pairs = nocc_i * nocc_j
    n_pair_blocks = (n_pairs + occupied_pair_batch_size - 1) // occupied_pair_batch_size
    n_rank_blocks = (rank + rank_panel_size - 1) // rank_panel_size
    padded_pairs = n_pair_blocks * occupied_pair_batch_size
    padded_rank = n_rank_blocks * rank_panel_size
    t2_pairs = jnp.pad(
        t2.reshape(n_pairs, nvir, nvir),
        ((0, padded_pairs - n_pairs), (0, 0), (0, 0)),
    )
    inner_padded = jnp.pad(right_inner, ((0, 0), (0, padded_rank - rank)))
    out_padded = jnp.pad(right_out, ((0, 0), (0, padded_rank - rank)))
    x_padded = jnp.pad(x, ((0, 0), (0, 0), (0, padded_rank - rank)))
    result = jnp.zeros((padded_pairs, nvir, nvir), dtype=t2.dtype)

    def pair_body(pair_block, result_acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau_block = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir),
        )

        def rank_body(rank_block, out_acc):
            rank0 = rank_block * rank_panel_size
            inner_panel = jax.lax.dynamic_slice(
                inner_padded, (0, rank0), (nvir, rank_panel_size),
            )
            out_panel = jax.lax.dynamic_slice(
                out_padded, (0, rank0), (nvir, rank_panel_size),
            )
            x_panel = jax.lax.dynamic_slice(
                x_padded, (0, 0, rank0), (nvir, nvir, rank_panel_size),
            )
            # S[n,c,mu] = sum_d tau[n,c,d] P[d,mu]
            s = jnp.einsum("ncd,dm->ncm", tau_block, inner_panel)
            # Y[n,a,mu] = sum_c X[a,c,mu] S[n,c,mu]
            y = jnp.einsum("acm,ncm->nam", x_panel, s)
            return out_acc + jnp.einsum("nam,bm->nab", y, out_panel)

        out_block = jax.lax.fori_loop(
            0, n_rank_blocks, rank_body,
            jnp.zeros((occupied_pair_batch_size, nvir, nvir), dtype=t2.dtype),
        )
        return jax.lax.dynamic_update_slice(result_acc, out_block, (pair0, 0, 0))

    result = jax.lax.fori_loop(0, n_pair_blocks, pair_body, result)
    return result[:n_pairs].reshape(nocc_i, nocc_j, nvir, nvir)


def contract_x_right_t2(
    t2: Array,
    right_out: Array,
    right_inner: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> Array:
    """Contract ``X[a,c,m] P[b,m] P[d,m]`` without a V^4 tile."""

    return _contract_partial_x(
        _contract_x_right_t2_jit, t2, right_out, right_inner, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )


def compiled_partial_x_right_memory(
    t2: Array,
    right_out: Array,
    right_inner: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> CompiledXLAMemory:
    """Return XLA accounting for ``X[a,c,m] P[b,m] P[d,m]``."""

    return _compiled_partial_x_memory(
        _contract_x_right_t2_jit, t2, right_out, right_inner, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )


# ---------------------------------------------------------------------------
# Streamed-X contraction (bounded device memory)
#
# ``contract_x_*_t2`` device-lift the whole X factor before the panel
# loop, which is impossible beyond a few hundred virtual orbitals.  The
# streamed variants below keep X on its host/HDF5 backing and device_put one
# rank panel at a time; the per-panel kernels reproduce the full-lift math
# term for term, with the rank-panel loop hoisted to the host.  Peak device X
# is one panel, never the full block.  Panels are transposed rank-leading
# ``(panel, nvir, nvir)`` on the host before upload so the panel kernels
# lower to cuBLAS-native batched GEMMs with no device-side panel transpose.


def _read_x_rank_panel(x_backing, nocc, m0, m1, panel_size):
    """Read one X rank panel from any backing (ndarray view or HDF5 dataset).

    Returns a fresh ``(nvir, nvir, panel_size)`` float64 host array, zero-
    padded on the rank axis when the tail panel is short so every panel
    shares one compiled shape.  Only the requested panel is materialized --
    slicing an HDF5 dataset reads just that selection.  INNERMOST-layout
    backings only; rank-major backings use :func:`_read_x_rank_panel_major`.
    """
    panel = np.asarray(x_backing[nocc:, nocc:, m0:m1], dtype=np.float64)
    short = panel_size - panel.shape[2]
    if short:
        panel = np.pad(panel, ((0, 0), (0, 0), (0, short)))
    return panel


def _read_x_rank_panel_major(x_backing, nocc, m0, m1, panel_size):
    """Rank-major twin of :func:`_read_x_rank_panel`, returning rank-leading
    ``(panel_size, nvir, nvir)`` panels -- already the layout the panel
    kernels consume, and one CONTIGUOUS disk read per panel instead of
    ~nmo^2 strided chunks.  Zero-pads the rank tail on axis 0.
    """
    panel = np.ascontiguousarray(x_backing[m0:m1, nocc:, nocc:], dtype=np.float64)
    short = panel_size - panel.shape[0]
    if short:
        panel = np.pad(panel, ((0, short), (0, 0), (0, 0)))
    return panel


def _x_backing_layout(x_backing, nocc, nvir, rank):
    """Detect an X backing's axis layout: ``"innermost"`` or ``"rank_major"``.

    ``"innermost"`` is the historical store layout ``(nmo, nmo, rank)``;
    ``"rank_major"`` is ``(rank, nmo, nmo)``, where a rank panel is one
    contiguous block instead of ~nmo^2 strided chunks.  An HDF5 backing may
    declare the layout explicitly via an ``x_layout`` attribute (the
    converter writes it); otherwise the two are told apart by which axis
    pair is square.  Shapes with nmo == rank are genuinely ambiguous and
    fall back to ``"innermost"`` -- rank-major stores with nmo == rank
    MUST carry the attribute.
    """

    attrs = getattr(x_backing, "attrs", None)
    declared = attrs.get("x_layout") if attrs is not None else None
    if declared is not None:
        declared = declared.decode() if isinstance(declared, bytes) else str(declared)
        if declared not in ("innermost", "rank_major"):
            raise ValueError(f"unrecognized x_layout attribute: {declared!r}")
        return declared
    shape = tuple(getattr(x_backing, "shape", ()))
    nmo = nvir + int(nocc)
    if len(shape) != 3:
        raise ValueError(
            "x_backing must be a 3-D array or HDF5 dataset, either "
            f"(nmo, nmo, rank) or (rank, nmo, nmo); got shape {shape}")
    if shape == (nmo, nmo, rank):
        return "innermost"
    if shape == (rank, nmo, nmo):
        return "rank_major"
    raise ValueError(
        f"x_backing must have shape ({nmo}, {nmo}, {rank}) or "
        f"({rank}, {nmo}, {nmo}); got {shape}")


def _validate_x_stream(t2, left_out, left_inner, x_backing, nocc):
    nvir = _validate_x_factors(t2, left_out, left_inner)
    rank = left_out.shape[1]
    layout = _x_backing_layout(x_backing, nocc, nvir, rank)
    if int(nocc) < 1 or nvir < 1:
        raise ValueError(f"nocc must leave a nonempty virtual space; got {nocc}")
    return layout


@partial(jax.jit, static_argnames=("occupied_pair_batch_size",))
def _xstream_left_panel_jit(t2_pairs, inner_panel, out_panel, x_panel, *,
                            occupied_pair_batch_size):
    """One rank panel's contribution to ``P[a,m] P[c,m] X[b,d,m]``.

    ``x_panel`` arrives rank-LEADING, ``(panel, nvir, nvir)`` (transposed on
    the host inside the prefetch/read path, where the copy is free-ish and
    overlapped).  With the rank axis leading, every einsum below lowers to a
    cuBLAS-native (strided-)batched GEMM; a rank-last panel layout makes
    XLA materialize a physical whole-panel transpose on device, whose
    autotuning buffers can exhaust the allocator at large shapes.
    """

    n_padded_pairs, nvir, _ = t2_pairs.shape
    n_pair_blocks = n_padded_pairs // occupied_pair_batch_size

    def pair_body(pair_block, out_acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau_block = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir))
        # S[mu,n,d] = sum_c P[c,mu] tau[n,c,d]
        s = jnp.einsum("cm,ncd->mnd", inner_panel, tau_block)
        # Y[mu,n,b] = sum_d S[mu,n,d] X[mu,b,d]
        y = jnp.einsum("mnd,mbd->mnb", s, x_panel)
        out_block = jnp.einsum("am,mnb->nab", out_panel, y)
        return jax.lax.dynamic_update_slice(out_acc, out_block, (pair0, 0, 0))

    return jax.lax.fori_loop(
        0, n_pair_blocks, pair_body, jnp.zeros_like(t2_pairs))


@partial(jax.jit, static_argnames=("occupied_pair_batch_size",))
def _xstream_right_panel_jit(t2_pairs, inner_panel, out_panel, x_panel, *,
                             occupied_pair_batch_size):
    """One rank panel's contribution to ``X[a,c,m] P[b,m] P[d,m]``.

    ``x_panel`` is rank-leading ``(panel, nvir, nvir)``; see
    :func:`_xstream_left_panel_jit` for why.
    """

    n_padded_pairs, nvir, _ = t2_pairs.shape
    n_pair_blocks = n_padded_pairs // occupied_pair_batch_size

    def pair_body(pair_block, out_acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau_block = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir))
        # S[mu,n,c] = sum_d P[d,mu] tau[n,c,d]
        s = jnp.einsum("dm,ncd->mnc", inner_panel, tau_block)
        # Y[mu,n,a] = sum_c X[mu,a,c] S[mu,n,c]
        y = jnp.einsum("mac,mnc->mna", x_panel, s)
        out_block = jnp.einsum("bm,mna->nab", out_panel, y)
        return jax.lax.dynamic_update_slice(out_acc, out_block, (pair0, 0, 0))

    return jax.lax.fori_loop(
        0, n_pair_blocks, pair_body, jnp.zeros_like(t2_pairs))


def _stream_partial_x(panel_kernel, t2, left_out, left_inner, x_backing, nocc,
                      *, occupied_pair_batch_size, rank_panel_size):
    """Host-loop rank-panel streaming shared by the left and right X terms.

    ``left_out``/``left_inner`` are the small endpoint factors (device-
    resident); ``x_backing`` is the ``(nmo, nmo, rank)`` X factor on any
    backing -- a NumPy array (view slicing) or an HDF5 dataset (partial
    reads) -- and only one panel is on the device at a time.  Panels are
    transposed rank-leading ``(panel, nvir, nvir)`` on the host before
    upload; the panel kernels require that layout (see
    :func:`_xstream_left_panel_jit`).
    """

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    nocc = int(nocc)
    layout = _validate_x_stream(t2, left_out, left_inner, x_backing, nocc)

    nocc_i, nocc_j, nvir, _ = t2.shape
    rank = left_out.shape[1]
    n_pairs = nocc_i * nocc_j
    n_pair_blocks = (n_pairs + occupied_pair_batch_size - 1) // occupied_pair_batch_size
    padded_pairs = n_pair_blocks * occupied_pair_batch_size
    n_rank_blocks = (rank + rank_panel_size - 1) // rank_panel_size
    padded_rank = n_rank_blocks * rank_panel_size

    t2_pairs = jnp.pad(
        jnp.asarray(t2).reshape(n_pairs, nvir, nvir),
        ((0, padded_pairs - n_pairs), (0, 0), (0, 0)))
    inner_padded = jnp.pad(jnp.asarray(left_inner), ((0, 0), (0, padded_rank - rank)))
    out_padded = jnp.pad(jnp.asarray(left_out), ((0, 0), (0, padded_rank - rank)))

    if layout == "rank_major":
        def read_panel(m0, m1):
            return _read_x_rank_panel_major(x_backing, nocc, m0, m1, rank_panel_size)
    else:
        def read_panel(m0, m1):
            return np.ascontiguousarray(
                _read_x_rank_panel(x_backing, nocc, m0, m1, rank_panel_size)
                .transpose(2, 0, 1))

    total = jnp.zeros((padded_pairs, nvir, nvir), dtype=t2_pairs.dtype)
    for rank_block in range(n_rank_blocks):
        m0 = rank_block * rank_panel_size
        m1 = min(m0 + rank_panel_size, rank)
        x_panel = jax.device_put(read_panel(m0, m1))
        inner_panel = inner_padded[:, m0:m0 + rank_panel_size]
        out_panel = out_padded[:, m0:m0 + rank_panel_size]
        total = total + panel_kernel(
            t2_pairs, inner_panel, out_panel, x_panel,
            occupied_pair_batch_size=occupied_pair_batch_size)
    return total[:n_pairs].reshape(nocc_i, nocc_j, nvir, nvir)


def contract_x_left_t2_streamed(t2, left_out, left_inner, x_backing, nocc,
                                        *, occupied_pair_batch_size=8,
                                        rank_panel_size=128):
    """Streamed ``P[a,m] P[c,m] X[b,d,m]``: one X rank panel on device at a time."""

    return _stream_partial_x(
        _xstream_left_panel_jit, t2, left_out, left_inner, x_backing, nocc,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size)


def contract_x_right_t2_streamed(t2, right_out, right_inner, x_backing, nocc,
                                         *, occupied_pair_batch_size=8,
                                         rank_panel_size=128):
    """Streamed ``X[a,c,m] P[b,m] P[d,m]``: one X rank panel on device at a time."""

    return _stream_partial_x(
        _xstream_right_panel_jit, t2, right_out, right_inner, x_backing, nocc,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size)


# ---------------------------------------------------------------------------
# Pipelined X panel streaming (tiers 2/3)
#
# At large system sizes the design variable is launch count, not transfer
# volume: small fixed-width panels pay heavy per-launch dispatch.  The
# pipelined loop below sizes each panel to the measured device working set
# and double-buffers the next panel's host read + H2D transfer behind the
# current panel's kernel.
#
# NOTE on parity: the panel kernels reduce over the rank axis inside each
# panel and the host loop accumulates across panels, so widening the panels
# regroups the reduction -- pipelined results are mathematically identical
# to the 128-panel streamed path but NOT bitwise identical; expect FP64
# reassociation-level agreement (relative L2 <= 1e-12), not 0.0.


def _measure_free_device_bytes():
    """Measured free bytes on the first local device, or None when unmeasurable.

    GPU backends report ``bytes_available``/``bytes_limit`` via
    ``Device.memory_stats()``; the CPU backend returns None there, in which
    case the tier gates treat device capacity as unknown.  Older jaxlib
    builds omit ``bytes_available``, so fall back to
    ``bytes_limit - bytes_in_use``; when no combination yields a value the
    raw stats are logged so the next run shows exactly what the device
    reported.
    """

    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception as exc:
        _logger.info("device memory_stats() raised %r; treating as unmeasurable", exc)
        return None
    if not stats:
        return None
    available = stats.get("bytes_available")
    if available is not None:
        return int(available)
    limit, in_use = stats.get("bytes_limit"), stats.get("bytes_in_use")
    if limit is not None and in_use is not None:
        return int(limit) - int(in_use)
    _logger.info("device memory_stats() lacks usable keys: %s", dict(stats))
    return None


def _cgroup_memory_available_bytes(root="/sys/fs/cgroup",
                                   proc_cgroup="/proc/self/cgroup"):
    """Available bytes under the most restrictive enclosing cgroup limit, or None.

    A SLURM job's ``--mem`` cap is enforced by the cgroup OOM killer, so
    node-wide RAM (psutil) overstates what a tier-2 host lift may use.
    The limit lives on the job's OWN cgroup (e.g.
    ``.../slurmstepd.scope/job_<id>/``), not at the hierarchy root.
    Discovers the process's cgroup(s) from ``/proc/self/cgroup``
    and walks each path UP to the root, taking the minimum of
    ``limit - current`` over every level that sets one (cgroup v2
    ``memory.max``/``memory.current``; v1
    ``memory.limit_in_bytes``/``memory.usage_in_bytes``).  Returns None
    when no limited level is found (e.g. macOS, non-cgroup hosts) and
    logs every level it inspected so the tier line is auditable.
    ``root``/``proc_cgroup`` exist for tests.
    """

    def _read(path):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            return None

    try:
        with open(proc_cgroup) as fh:
            entries = fh.read()
    except OSError:
        entries = ""

    candidates = []
    for line in entries.splitlines():
        parts = line.split(":")
        if len(parts) != 3:
            continue
        _, controllers, rel = parts
        rel = rel.lstrip("/")
        if controllers == "":  # cgroup v2 unified hierarchy
            candidates.append(os.path.join(root, rel) if rel else root)
        elif "memory" in controllers.split(","):  # v1 memory controller
            base = os.path.join(root, "memory")
            candidates.append(os.path.join(base, rel) if rel else base)
    if not candidates:  # unparseable / missing: fall back to the roots
        candidates = [root, os.path.join(root, "memory")]

    best = None
    for cand in candidates:
        path = os.path.normpath(cand)
        while True:
            max_v = _read(os.path.join(path, "memory.max"))
            cur_v = _read(os.path.join(path, "memory.current"))
            if max_v is None:  # try the v1 file names at this level
                max_v = _read(os.path.join(path, "memory.limit_in_bytes"))
                cur_v = _read(os.path.join(path, "memory.usage_in_bytes"))
            if max_v is not None and cur_v is not None and max_v != "max":
                try:
                    limit, current = int(max_v), int(cur_v)
                except ValueError:
                    limit = None
                if limit is not None and limit < 1 << 60:  # v1 "unlimited" is ~2^63
                    remaining = limit - current
                    _logger.info("cgroup memory limit at %s: %d bytes remaining", path, remaining)
                    best = remaining if best is None else min(best, remaining)
            if path == os.path.normpath(root) or path == os.path.dirname(path):
                break
            path = os.path.dirname(path)
    if best is None:
        _logger.info("no cgroup memory limit found; host measurement is node-wide")
    return best


def _measure_free_host_bytes():
    """Measured available host RAM in bytes, capped by any cgroup limit.

    psutil is imported lazily and treated as OPTIONAL: when it is absent
    the cgroup measurement alone answers, and when neither source exists
    the gate falls through safely (never admits on an unknown capacity).
    """

    try:
        import psutil
    except ImportError:
        available = None
    else:
        available = int(psutil.virtual_memory().available)
    cgroup = _cgroup_memory_available_bytes()
    if cgroup is not None:
        available = cgroup if available is None else min(available, cgroup)
    return available


def _pipelined_x_panel_size(nvir, rank, working_set_bytes, *,
                            rank_panel_size, panel_budget_bytes=None):
    """Rank width of one pipelined X panel, a multiple of ``rank_panel_size``.

    The per-panel budget is one third of the measured free device memory
    after the t2/accumulator working set and a 4 GiB reserve: resident
    panel, in-flight prefetch, and slack for kernel temporaries and BFC
    fragmentation.  ``PYTC_X_PANEL_BUDGET_GB`` (read at call time)
    overrides the computed budget.  When device memory is unmeasurable and
    no override is given, the width falls back to ``rank_panel_size``.
    """

    override_gb = os.environ.get("PYTC_X_PANEL_BUDGET_GB")
    if override_gb is not None:
        budget = int(float(override_gb) * 1024 ** 3)
    elif panel_budget_bytes is not None:
        budget = panel_budget_bytes
    else:
        free_device = _measure_free_device_bytes()
        if free_device is None:
            return rank_panel_size
        reserve = working_set_bytes + 4 * 1024 ** 3
        budget = max(0, int((0.9 * free_device - reserve) // 3))
    width = budget // (nvir * nvir * 8)
    if width < rank_panel_size:
        _logger.warning(
            "X panel budget %d bytes fits fewer than %d rank columns; "
            "falling back to rank_panel_size=%d",
            budget, rank_panel_size, rank_panel_size)
        return rank_panel_size
    padded_rank = ((rank + rank_panel_size - 1) // rank_panel_size) * rank_panel_size
    width = min(width, padded_rank)
    return (width // rank_panel_size) * rank_panel_size


def _x_backing_panel_source(x_backing, nocc, layout="innermost"):
    """Panel source that reads straight from the X backing (tier 3).

    Returns rank-leading ``(panel, nvir, nvir)`` panels -- the layout the
    panel kernels consume.  Innermost-layout backings are transposed inside
    the prefetch thread (the copy overlaps the current kernel); rank-major
    backings already store panels in this exact shape, so a panel read is
    one contiguous block with no transpose at all.
    """

    if layout == "rank_major":
        def read_panel(m0, m1, panel_size):
            return _read_x_rank_panel_major(x_backing, nocc, m0, m1, panel_size)
    else:
        def read_panel(m0, m1, panel_size):
            return np.ascontiguousarray(
                _read_x_rank_panel(x_backing, nocc, m0, m1, panel_size)
                .transpose(2, 0, 1))

    return read_panel


def _x_host_panel_source(x_host, layout="innermost"):
    """Panel source that slices a host-resident X (tier 2).

    ``x_host`` is ``(nvir, nvir, rank)`` for innermost layout or
    ``(rank, nvir, nvir)`` for rank-major.  Either way each panel is copied
    contiguous rank-leading ``(panel, nvir, nvir)`` in the prefetch thread;
    the rank tail is zero-padded exactly like :func:`_read_x_rank_panel`.
    """

    if layout == "rank_major":
        def read_panel(m0, m1, panel_size):
            panel = np.ascontiguousarray(x_host[m0:m1])
            short = panel_size - panel.shape[0]
            if short:
                panel = np.pad(panel, ((0, short), (0, 0), (0, 0)))
            return panel
    else:
        def read_panel(m0, m1, panel_size):
            panel = np.ascontiguousarray(
                x_host[:, :, m0:m1].transpose(2, 0, 1))
            short = panel_size - panel.shape[0]
            if short:
                panel = np.pad(panel, ((0, short), (0, 0), (0, 0)))
            return panel

    return read_panel


def _stream_partial_x_pipelined(panel_kernel, t2, left_out, left_inner,
                                panel_source, nocc, *,
                                occupied_pair_batch_size, rank_panel_size,
                                panel_budget_bytes=None):
    """Working-set X panel loop with double-buffered async prefetch.

    Same math, padding, and accumulation semantics as
    :func:`_stream_partial_x`, but the panel width is sized from the measured
    device working set and the next panel's host read is issued (via
    :func:`async_read`) before the current panel's kernel, so its H2D
    ``device_put`` overlaps compute (JAX async dispatch).

    ``panel_source(m0, m1, panel_size)`` returns one contiguous rank-leading
    ``(panel_size, nvir, nvir)`` float64 host panel, zero-padded on the rank
    tail exactly like :func:`_read_x_rank_panel`; every panel shares one
    shape, so each term compiles exactly once.  ``nocc`` is already baked
    into ``panel_source`` and is accepted for symmetry with
    :func:`_stream_partial_x`; callers validate with
    :func:`_validate_x_stream`.

    If a panel ``device_put`` or kernel dispatch raises a device
    out-of-memory error anyway (the free-memory measurement cannot see BFC
    fragmentation), the whole loop retries from scratch at half the panel
    width, halving again on each failure down to ``rank_panel_size``.
    """

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    del nocc
    _validate_t2(t2)

    nocc_i, nocc_j, nvir, _ = t2.shape
    rank = left_out.shape[1]
    n_pairs = nocc_i * nocc_j
    n_pair_blocks = (n_pairs + occupied_pair_batch_size - 1) // occupied_pair_batch_size
    padded_pairs = n_pair_blocks * occupied_pair_batch_size

    working_set = 3 * padded_pairs * nvir * nvir * 8
    panel_size = _pipelined_x_panel_size(
        nvir, rank, working_set,
        rank_panel_size=rank_panel_size, panel_budget_bytes=panel_budget_bytes)

    t2_pairs = jnp.pad(
        jnp.asarray(t2).reshape(n_pairs, nvir, nvir),
        ((0, padded_pairs - n_pairs), (0, 0), (0, 0)))

    while True:
        n_blocks = (rank + panel_size - 1) // panel_size
        panel_padded_rank = n_blocks * panel_size
        _logger.info(
            "X stream panels: panel_size=%d n_blocks=%d rank=%d "
            "(panel budget measured at call time)",
            panel_size, n_blocks, rank)

        inner_padded = jnp.pad(jnp.asarray(left_inner), ((0, 0), (0, panel_padded_rank - rank)))
        out_padded = jnp.pad(jnp.asarray(left_out), ((0, 0), (0, panel_padded_rank - rank)))

        def panel_read(k):
            m0 = k * panel_size
            return async_read(panel_source, m0, min(m0 + panel_size, rank), panel_size)

        try:
            total = jnp.zeros((padded_pairs, nvir, nvir), dtype=t2_pairs.dtype)
            future = panel_read(0)
            x_dev = jax.device_put(await_read(future))
            if n_blocks > 1:
                future = panel_read(1)
            for k in range(n_blocks):
                m0 = k * panel_size
                total = total + panel_kernel(
                    t2_pairs,
                    inner_padded[:, m0:m0 + panel_size],
                    out_padded[:, m0:m0 + panel_size],
                    x_dev,
                    occupied_pair_batch_size=occupied_pair_batch_size)
                if k + 1 < n_blocks:
                    # Awaiting the prefetch after dispatching kernel k lets
                    # the H2D transfer overlap the kernel's async execution.
                    x_next = jax.device_put(await_read(future))
                    if k + 2 < n_blocks:
                        future = panel_read(k + 2)
                    x_dev = x_next
            # Dispatch is asynchronous: an execution-time OOM surfaces at a
            # readiness barrier, so the barrier must live inside the try.
            total = jax.block_until_ready(total)
        except jax.errors.JaxRuntimeError as exc:
            smaller = (panel_size // 2 // rank_panel_size) * rank_panel_size
            msg = str(exc)
            if (("RESOURCE_EXHAUSTED" not in msg and "Out of memory" not in msg)
                    or smaller < rank_panel_size or smaller == panel_size):
                raise
            _logger.warning(
                "X stream device OOM at panel_size=%d; retrying at %d",
                panel_size, smaller)
            panel_size = smaller
            continue
        return total[:n_pairs].reshape(nocc_i, nocc_j, nvir, nvir)


def _contract_x_t2_pipelined(
    panel_kernel, t2, out_factor, inner_factor, x_backing, nocc, *,
    occupied_pair_batch_size, rank_panel_size, panel_budget_bytes,
):
    layout = _validate_x_stream(t2, out_factor, inner_factor, x_backing, nocc)
    return _stream_partial_x_pipelined(
        panel_kernel, t2, out_factor, inner_factor,
        _x_backing_panel_source(x_backing, nocc, layout), nocc,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
        panel_budget_bytes=panel_budget_bytes)


def contract_x_left_t2_pipelined(t2, left_out, left_inner, x_backing, nocc,
                                 *, occupied_pair_batch_size=8,
                                 rank_panel_size=128,
                                 panel_budget_bytes=None):
    """Pipelined ``P[a,m] P[c,m] X[b,d,m]``: working-set panels, prefetched."""

    return _contract_x_t2_pipelined(
        _xstream_left_panel_jit, t2, left_out, left_inner, x_backing, nocc,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
        panel_budget_bytes=panel_budget_bytes)


def contract_x_right_t2_pipelined(t2, right_out, right_inner, x_backing, nocc,
                                  *, occupied_pair_batch_size=8,
                                  rank_panel_size=128,
                                  panel_budget_bytes=None):
    """Pipelined ``X[a,c,m] P[b,m] P[d,m]``: working-set panels, prefetched."""

    return _contract_x_t2_pipelined(
        _xstream_right_panel_jit, t2, right_out, right_inner, x_backing, nocc,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
        panel_budget_bytes=panel_budget_bytes)


# Default device-residency cap for the full-lift X path (~24 GiB).  The
# streamed path exists for systems whose X_vv cannot live on the GPU; when X
# fits on device the full-lift path is faster (one compiled rank scan
# instead of many host-driven panel kernels).
_X_FULL_LIFT_CAP_BYTES = int(
    float(os.environ.get("PYTC_X_FULL_LIFT_CAP_GB", "24")) * 1024 ** 3)


def _validate_term_factors(p, grad_p, u1, u3, d, *extra):
    arrays = tuple(map(jnp.asarray, (p, grad_p, u1, u3, d, *extra)))
    p, grad_p, u1 = arrays[:3]
    if grad_p.shape != (p.shape[0], p.shape[1], 3):
        raise ValueError(
            "grad_p must have shape (nvir, rank, 3); "
            f"got {grad_p.shape} for p={p.shape}"
        )
    if u1.shape != (p.shape[1], p.shape[1], 3):
        raise ValueError(f"u1 must have shape (rank, rank, 3); got {u1.shape}")
    return arrays


def _assemble_terms(terms: Mapping[str, Array]) -> Mapping[str, Array]:
    k1_direct = terms["k1_direct"]
    k1_pair = terms["k1_pair"]
    k2_direct = terms["k2_direct"]
    k2_pair = terms["k2_pair"]
    k3_direct = terms["k3_direct"]
    k3_pair = terms["k3_pair"]
    d_direct = terms["d_direct"]
    d_pair = terms["d_pair"]
    x_direct = terms["x_direct"]
    x_pair = terms["x_pair"]
    tc_direct = 0.5 * (k1_direct - k2_direct + k3_direct)
    tc_pair = 0.5 * (k1_pair - k2_pair + k3_pair)
    delta_direct = d_direct - x_direct
    delta_pair = d_pair - x_pair
    tc = -(tc_direct + tc_pair)
    delta_u = -(delta_direct + delta_pair)
    return {
        **terms,
        "tc_direct": tc_direct,
        "tc_pair": tc_pair,
        "delta_direct": delta_direct,
        "delta_pair": delta_pair,
        "tc": tc,
        "delta_u": delta_u,
        "final": tc + delta_u,
    }


def contract_terms_t2_auto(
    t2: Array,
    p: Array,
    grad_p: Array,
    u1: Array,
    u3: Array,
    d: Array,
    x_backing,
    nocc: int,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
    cap_bytes: int = _X_FULL_LIFT_CAP_BYTES,
) -> Mapping[str, Array]:
    """Three-tier X path, selected on measured free memory.

    * Tier 1 (``fd_x_tier1_full_lift``): X_vv fits the device-residency cap
      AND at most half the measured free device memory -- the full-lift path
      (whole block device-lifted, one compiled rank scan), the fast path
      whenever the block fits.
    * Tier 2 (``fd_x_tier2_host_resident``): X_vv fails the device gate but
      fits in half the measured free host RAM -- X_vv is lifted to host RAM
      once, then contracted by the pipelined panel loop with panel reads as
      host-array slices (the block fits a node's RAM).
    * Tier 3 (``fd_x_tier3_stream``): otherwise -- the same pipelined loop
      with panel reads from the backing (HDF5 dataset or ndarray).

    Tiers 2/3 share the working-set panel loop: the panel width comes from
    the measured device working set and the next panel's read + H2D transfer
    is prefetched behind the current kernel.  The gate uses only measured
    free memory plus the cap; ``PYTC_X_FORCE_TIER`` = 1|2|3 (read at call
    time) pins the tier for tests and benchmarking, and
    ``PYTC_X_PANEL_BUDGET_GB`` pins the per-panel budget.  Which tier fired
    is recorded in the tile_timers counters so receipts show it.
    """
    nocc = int(nocc)
    nvir_guess = t2.shape[2]
    rank_guess = p.shape[1]
    layout = _x_backing_layout(x_backing, nocc, nvir_guess, rank_guess)
    if layout == "rank_major":
        rank = x_backing.shape[0]
        nvir = x_backing.shape[1] - nocc
    else:
        nvir = x_backing.shape[0] - nocc
        rank = x_backing.shape[2]
    x_bytes = nvir * nvir * rank * 8

    force = os.environ.get("PYTC_X_FORCE_TIER")
    if force in ("1", "2", "3"):
        # Forced tiers need no probes (and must not require psutil).
        tier = int(force)
        free_device = free_host = None
    else:
        free_device = _measure_free_device_bytes()
        free_host = _measure_free_host_bytes()
        # Fail closed: a tier is only chosen on a MEASURED capacity.
        if (x_bytes <= cap_bytes and free_device is not None
                and x_bytes <= 0.5 * free_device):
            tier = 1
        elif free_host is not None and x_bytes <= 0.5 * free_host:
            tier = 2
        else:
            tier = 3

    panel_size = None
    if tier != 1:
        n_pairs = t2.shape[0] * t2.shape[1]
        n_pair_blocks = (
            (n_pairs + occupied_pair_batch_size - 1) // occupied_pair_batch_size)
        working_set = 3 * n_pair_blocks * occupied_pair_batch_size * nvir * nvir * 8
        panel_size = _pipelined_x_panel_size(
            nvir, rank, working_set, rank_panel_size=rank_panel_size)
    _logger.info(
        "X tier selection: tier=%d x_bytes=%d free_device_bytes=%s "
        "free_host_bytes=%s panel_size=%s layout=%s",
        tier, x_bytes, free_device, free_host, panel_size, layout)

    if tier == 1:
        _tile_timers.incr("fd_x_tier1_full_lift")
        if layout == "rank_major":
            x_full = np.ascontiguousarray(
                np.asarray(x_backing[:, nocc:, nocc:], dtype=np.float64)
                .transpose(1, 2, 0))
        else:
            x_full = np.asarray(x_backing[nocc:, nocc:, :], dtype=np.float64)
        return contract_terms_t2(
            t2, p, grad_p, u1, u3, d, x_full,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size)
    if tier == 2:
        _tile_timers.incr("fd_x_tier2_host_resident")
        if layout == "rank_major":
            x_host = np.ascontiguousarray(
                np.asarray(x_backing[:, nocc:, nocc:], dtype=np.float64))
        else:
            x_host = np.ascontiguousarray(
                np.asarray(x_backing[nocc:, nocc:, :], dtype=np.float64))
        panel_source = _x_host_panel_source(x_host, layout)
    else:
        _tile_timers.incr("fd_x_tier3_stream")
        panel_source = _x_backing_panel_source(x_backing, nocc, layout)
    return contract_terms_t2_xstream(
        t2, p, grad_p, u1, u3, d, x_backing, nocc,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
        panel_source=panel_source)


def contract_terms_t2_xstream(
    t2: Array,
    p: Array,
    grad_p: Array,
    u1: Array,
    u3: Array,
    d: Array,
    x_backing,
    nocc: int,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
    panel_source=None,
    panel_budget_bytes=None,
) -> Mapping[str, Array]:
    """Factor-direct terms with X streamed panel-wise from its backing.

    Identical terms and signs to
    :func:`contract_terms_t2`; only the two X-consuming
    terms change how X reaches the device.  Every other input is small and
    device-lifted exactly as in the full-block path.

    With ``panel_source=None`` the X terms use the legacy 128-wide panel
    loop; with a panel source (see :func:`_x_host_panel_source` /
    :func:`_x_backing_panel_source`) they use the pipelined working-set
    panel loop -- tiers 2/3 of :func:`contract_terms_t2_auto`.
    """

    p, grad_p, u1, u3, d = _validate_term_factors(p, grad_p, u1, u3, d)

    def _timed(name, fn, *args, **kwargs):
        with _tile_timers.term(name) as _tt:
            out = fn(*args, **kwargs)
            _tt.sync(out)
        return out

    _kw = dict(occupied_pair_batch_size=occupied_pair_batch_size,
               rank_panel_size=rank_panel_size)
    k1_direct = _timed("fd_k1_direct", _contract_k1_direct_t2_jit,
                       t2, p, grad_p, u1, **_kw)
    k1_pair = _timed("fd_k1_pair", _contract_k1_pair_t2_jit,
                     t2, p, grad_p, u1, **_kw)
    k2_direct = _timed("fd_k2_direct", _contract_k2_direct_t2_jit,
                       t2, p, grad_p, u1, **_kw)
    k2_pair = _timed("fd_k2_pair", _contract_k2_pair_t2_jit,
                     t2, p, grad_p, u1, **_kw)
    k3_direct = _timed("fd_k3_direct", contract_full_thc_t2,
                       t2, p, p, u3, p, p, **_kw)
    k3_pair = _timed("fd_k3_pair", contract_full_thc_pair_swapped_t2,
                     t2, p, p, u3, p, p, **_kw)
    d_direct = _timed("fd_d_direct", contract_full_thc_t2,
                      t2, p, p, d, p, p, **_kw)
    d_pair = _timed("fd_d_pair", contract_full_thc_pair_swapped_t2,
                    t2, p, p, d, p, p, **_kw)
    if panel_source is None:
        x_direct = _timed("fd_x_left", contract_x_left_t2_streamed,
                          t2, p, p, x_backing, nocc, **_kw)
        x_pair = _timed("fd_x_right", contract_x_right_t2_streamed,
                        t2, p, p, x_backing, nocc, **_kw)
    else:
        _validate_x_stream(t2, p, p, x_backing, nocc)
        x_direct = _timed("fd_x_left", _stream_partial_x_pipelined,
                          _xstream_left_panel_jit, t2, p, p, panel_source, nocc,
                          panel_budget_bytes=panel_budget_bytes, **_kw)
        x_pair = _timed("fd_x_right", _stream_partial_x_pipelined,
                        _xstream_right_panel_jit, t2, p, p, panel_source, nocc,
                        panel_budget_bytes=panel_budget_bytes, **_kw)

    return _assemble_terms({
        "k1_direct": k1_direct,
        "k1_pair": k1_pair,
        "k2_direct": k2_direct,
        "k2_pair": k2_pair,
        "k3_direct": k3_direct,
        "k3_pair": k3_pair,
        "d_direct": d_direct,
        "d_pair": d_pair,
        "x_direct": x_direct,
        "x_pair": x_pair,
    })


def contract_terms_t2(
    t2: Array,
    p: Array,
    grad_p: Array,
    u1: Array,
    u3: Array,
    d: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> Mapping[str, Array]:
    """Return each exact-current-ISDF VVVV--T2 branch and final residual.

    Keys ending in ``_direct`` and ``_pair`` are intentionally retained for
    the dense-reference numerical gate.  The final values reproduce the current
    ``_assemble_tc_tile`` / ``_assemble_delta_u_tile`` signs:

    * ``tc = -0.5 * ((K1 - K2 + K3) + pair_swap(...))``
    * ``delta_u = -((D - X) + pair_swap(D - X))``

    There is no ordinary DF Coulomb term here: it is the separately-gated
    crossed-DF scope and is deliberately not changed here.
    """

    p, grad_p, u1, u3, d, x = _validate_term_factors(
        p, grad_p, u1, u3, d, x
    )

    # Each Cartesian K1/K2 sum is deliberately one compiled executable. This
    # makes its XLA memory accounting a true per-term record rather than a
    # gamma-0 proxy for three separately dispatched components.
    k1_direct = _contract_k1_direct_t2_jit(
        t2, p, grad_p, u1,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    k1_pair = _contract_k1_pair_t2_jit(
        t2, p, grad_p, u1,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    k2_direct = _contract_k2_direct_t2_jit(
        t2, p, grad_p, u1,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    k2_pair = _contract_k2_pair_t2_jit(
        t2, p, grad_p, u1,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )

    k3_direct = contract_full_thc_t2(
        t2, p, p, u3, p, p,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    k3_pair = contract_full_thc_pair_swapped_t2(
        t2, p, p, u3, p, p,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    d_direct = contract_full_thc_t2(
        t2, p, p, d, p, p,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    d_pair = contract_full_thc_pair_swapped_t2(
        t2, p, p, d, p, p,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    x_direct = contract_x_left_t2(
        t2, p, p, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    x_pair = contract_x_right_t2(
        t2, p, p, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )

    return _assemble_terms({
        "k1_direct": k1_direct,
        "k1_pair": k1_pair,
        "k2_direct": k2_direct,
        "k2_pair": k2_pair,
        "k3_direct": k3_direct,
        "k3_pair": k3_pair,
        "d_direct": d_direct,
        "d_pair": d_pair,
        "x_direct": x_direct,
        "x_pair": x_pair,
    })


def _profile_call(
    call: Callable[[], Array],
    *,
    schedule_intermediate_estimate_bytes: int,
    compiled_memory: CompiledXLAMemory,
    rank_panel_size: int,
    occupied_pair_batch_size: int,
) -> tuple[Array, FactorDirectProfile]:
    """Compile once, then record one synchronized steady-state wall time."""

    warm = call()
    jax.block_until_ready(warm)
    started = time.perf_counter()
    result = call()
    jax.block_until_ready(result)
    return result, FactorDirectProfile(
        wall_seconds=time.perf_counter() - started,
        schedule_intermediate_estimate_bytes=schedule_intermediate_estimate_bytes,
        compiled_xla_temporary_bytes=compiled_memory.temporary_bytes,
        compiled_xla_argument_bytes=compiled_memory.argument_bytes,
        compiled_xla_output_bytes=compiled_memory.output_bytes,
        compiled_xla_alias_bytes=compiled_memory.alias_bytes,
        compiled_xla_total_bytes=compiled_memory.total_bytes,
        rank_panel_size=rank_panel_size,
        occupied_pair_batch_size=occupied_pair_batch_size,
    )


def profile_isdf_factor_direct_terms_t2(
    t2: Array,
    p: Array,
    grad_p: Array,
    u1: Array,
    u3: Array,
    d: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> tuple[Mapping[str, Array], Mapping[str, FactorDirectProfile]]:
    """Measure every branch and return the exact factor-direct terms.

    Profiling is explicitly opt-in.  Each profile distinguishes the
    schedule-panel estimate from XLA's executable memory analysis.  Neither
    is a process-wide allocator peak.
    """

    terms = contract_terms_t2(
        t2, p, grad_p, u1, u3, d, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    # Force the precomputed aggregate before timing individual branches so
    # callers receive a complete, ready-to-inspect residual dictionary.
    jax.block_until_ready(tuple(terms.values()))
    nvir, rank = p.shape
    itemsize = _dtype_itemsize(t2, p, u1, u3, d, x)
    full_estimate = full_thc_schedule_intermediate_estimate_bytes(
        nvir=nvir, rank=rank,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
        itemsize=itemsize,
    )
    x_estimate = partial_x_schedule_intermediate_estimate_bytes(
        nvir=nvir, rank=rank,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
        itemsize=itemsize,
    )

    full_memories = {
        "k1_direct": _compiled_k12_memory(
            _contract_k1_direct_t2_jit, t2, p, grad_p, u1,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
        "k1_pair": _compiled_k12_memory(
            _contract_k1_pair_t2_jit, t2, p, grad_p, u1,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
        "k2_direct": _compiled_k12_memory(
            _contract_k2_direct_t2_jit, t2, p, grad_p, u1,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
        "k2_pair": _compiled_k12_memory(
            _contract_k2_pair_t2_jit, t2, p, grad_p, u1,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
        "k3_direct": compiled_full_thc_memory(
            t2, p, p, u3, p, p,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
        "k3_pair": compiled_full_thc_memory(
            t2, p, p, u3.T, p, p,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
        "d_direct": compiled_full_thc_memory(
            t2, p, p, d, p, p,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
        "d_pair": compiled_full_thc_memory(
            t2, p, p, d.T, p, p,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
    }
    x_memories = {
        "x_direct": compiled_partial_x_left_memory(
            t2, p, p, x,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
        "x_pair": compiled_partial_x_right_memory(
            t2, p, p, x,
            occupied_pair_batch_size=occupied_pair_batch_size,
            rank_panel_size=rank_panel_size,
        ),
    }
    calls: dict[str, tuple[Callable[[], Array], int, CompiledXLAMemory]] = {
        "k1_direct": (
            lambda: _contract_k1_direct_t2_jit(
                t2, p, grad_p, u1,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            full_estimate,
            full_memories["k1_direct"],
        ),
        "k1_pair": (
            lambda: _contract_k1_pair_t2_jit(
                t2, p, grad_p, u1,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            full_estimate,
            full_memories["k1_pair"],
        ),
        "k2_direct": (
            lambda: _contract_k2_direct_t2_jit(
                t2, p, grad_p, u1,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            full_estimate,
            full_memories["k2_direct"],
        ),
        "k2_pair": (
            lambda: _contract_k2_pair_t2_jit(
                t2, p, grad_p, u1,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            full_estimate,
            full_memories["k2_pair"],
        ),
        "k3_direct": (
            lambda: contract_full_thc_t2(
                t2, p, p, u3, p, p,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            full_estimate,
            full_memories["k3_direct"],
        ),
        "k3_pair": (
            lambda: contract_full_thc_pair_swapped_t2(
                t2, p, p, u3, p, p,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            full_estimate,
            full_memories["k3_pair"],
        ),
        "d_direct": (
            lambda: contract_full_thc_t2(
                t2, p, p, d, p, p,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            full_estimate,
            full_memories["d_direct"],
        ),
        "d_pair": (
            lambda: contract_full_thc_pair_swapped_t2(
                t2, p, p, d, p, p,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            full_estimate,
            full_memories["d_pair"],
        ),
        "x_direct": (
            lambda: contract_x_left_t2(
                t2, p, p, x,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            x_estimate,
            x_memories["x_direct"],
        ),
        "x_pair": (
            lambda: contract_x_right_t2(
                t2, p, p, x,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            x_estimate,
            x_memories["x_pair"],
        ),
    }
    profiles: dict[str, FactorDirectProfile] = {}
    for name, (call, schedule_estimate, compiled_memory) in calls.items():
        _, profiles[name] = _profile_call(
            call,
            schedule_intermediate_estimate_bytes=schedule_estimate,
            compiled_memory=compiled_memory,
            rank_panel_size=rank_panel_size,
            occupied_pair_batch_size=occupied_pair_batch_size,
        )
    return terms, profiles
