"""Standalone factor-direct ISDF VVVV--T2 prototype.

This module intentionally has no call site in :mod:`pytc.solver.jax_xtc_ccsd`.
It is a Phase-A validation harness for the existing ISDF K1/K2/K3 and
Delta-U (D/X) factors.  In particular, it does not change the production
``vvvv`` route and does not attempt to factorize the ordinary crossed DF
Coulomb contribution.

The raw ERI-like tile order in PyTC is ``(a, c, b, d)``.  The public
functions below contract that tile directly with a dense RCCSD ``t2`` in
``(i, j, c, d)`` order, without creating a ``(v, v, v, v)`` tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import time
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from pytc.utils import tile_timers as _tile_timers


Array = jax.Array


@dataclass(frozen=True)
class FactorDirectProfile:
    """One steady-state contraction measurement.

    ``schedule_intermediate_estimate_bytes`` counts only the named panels in
    the factor-direct algebra.  It excludes compiler-generated scratch and
    whole-array padding copies, so it is explicitly *not* an allocator bound.

    The ``compiled_xla_*`` fields come from ``Compiled.memory_analysis()`` for
    the executable that evaluates this branch on the active backend.  They are
    the reviewable XLA buffer accounting for the actual executable, but still
    are not a claim about process-wide allocator high-water mark.
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
    and the whole-array ``jnp.pad`` copies used by this small-deck prototype;
    use ``compiled_full_thc_memory`` for executable memory accounting.
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

    The estimate deliberately excludes the full ``x`` / ``x_padded`` arrays.
    The current prototype pads X as a whole device array and is therefore
    small-deck only; it is not the production host/disk-streamed X schedule.
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
            "the Phase-A GPU card requires executable memory accounting"
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


def _validate_x(t2: Array, left_out: Array, left_inner: Array, x: Array) -> None:
    _validate_t2(t2)
    nvir = t2.shape[2]
    if left_out.ndim != 2 or left_inner.shape != left_out.shape:
        raise ValueError("left_out and left_inner must both have shape (nvir, rank)")
    if left_out.shape[0] != nvir:
        raise ValueError(f"X factors have nvir={left_out.shape[0]}, expected {nvir}")
    if x.shape != (nvir, nvir, left_out.shape[1]):
        raise ValueError(
            "x must have shape (nvir, nvir, rank); "
            f"got {x.shape}, expected {(nvir, nvir, left_out.shape[1])}"
        )


@partial(jax.jit, static_argnames=("occupied_pair_batch_size", "rank_panel_size"))
def _contract_partial_x_left_t2_jit(
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


def contract_partial_x_left_t2(
    t2: Array,
    left_out: Array,
    left_inner: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> Array:
    """Contract ``P[a,m] P[c,m] X[b,d,m]`` without a V^4 tile."""

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    arrays = tuple(map(jnp.asarray, (t2, left_out, left_inner, x)))
    _validate_x(*arrays)
    return _contract_partial_x_left_t2_jit(
        *arrays,
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

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    arrays = tuple(map(jnp.asarray, (t2, left_out, left_inner, x)))
    _validate_x(*arrays)
    executable = _contract_partial_x_left_t2_jit.lower(
        *arrays,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    ).compile()
    return _compiled_memory_from_executable(executable)


@partial(jax.jit, static_argnames=("occupied_pair_batch_size", "rank_panel_size"))
def _contract_partial_x_right_t2_jit(
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


def contract_partial_x_right_t2(
    t2: Array,
    right_out: Array,
    right_inner: Array,
    x: Array,
    *,
    occupied_pair_batch_size: int = 8,
    rank_panel_size: int = 128,
) -> Array:
    """Contract ``X[a,c,m] P[b,m] P[d,m]`` without a V^4 tile."""

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    arrays = tuple(map(jnp.asarray, (t2, right_out, right_inner, x)))
    _validate_x(*arrays)
    return _contract_partial_x_right_t2_jit(
        *arrays,
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

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    arrays = tuple(map(jnp.asarray, (t2, right_out, right_inner, x)))
    _validate_x(*arrays)
    executable = _contract_partial_x_right_t2_jit.lower(
        *arrays,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    ).compile()
    return _compiled_memory_from_executable(executable)


# ---------------------------------------------------------------------------
# Streamed-X contraction (bounded device memory)
#
# ``contract_partial_x_*_t2`` device-lift the whole X factor before the panel
# loop, which is impossible beyond a few hundred virtual orbitals.  The
# streamed variants below keep X on its host/HDF5 backing and device_put one
# rank panel at a time; the per-panel kernels reproduce the full-lift math
# term for term, with the rank-panel loop hoisted to the host.  Peak device X
# is one panel ``(nvir, nvir, rank_panel_size)``, never the full block.


def _read_x_rank_panel(x_backing, nocc, m0, m1, panel_size):
    """Read one X rank panel from any backing (ndarray view or HDF5 dataset).

    Returns a fresh ``(nvir, nvir, panel_size)`` float64 host array, zero-
    padded on the rank axis when the tail panel is short so every panel
    shares one compiled shape.  Only the requested panel is materialized --
    slicing an HDF5 dataset reads just that selection.
    """
    panel = np.asarray(x_backing[nocc:, nocc:, m0:m1], dtype=np.float64)
    short = panel_size - panel.shape[2]
    if short:
        panel = np.pad(panel, ((0, 0), (0, 0), (0, short)))
    return panel


def _validate_x_stream(t2, left_out, left_inner, x_backing, nocc):
    _validate_t2(t2)
    nvir = t2.shape[2]
    if left_out.ndim != 2 or left_inner.shape != left_out.shape:
        raise ValueError("left_out and left_inner must both have shape (nvir, rank)")
    if left_out.shape[0] != nvir:
        raise ValueError(f"X factors have nvir={left_out.shape[0]}, expected {nvir}")
    rank = left_out.shape[1]
    if getattr(x_backing, "ndim", None) != 3:
        raise ValueError(
            "x_backing must be a 3-D (nmo, nmo, rank) array or HDF5 dataset; "
            f"got shape {getattr(x_backing, 'shape', None)}")
    expected = (nvir + int(nocc), nvir + int(nocc), rank)
    if tuple(x_backing.shape) != expected:
        raise ValueError(f"x_backing must have shape {expected}; got {x_backing.shape}")
    if not 0 < int(nocc) < x_backing.shape[0]:
        raise ValueError(f"nocc must leave a nonempty virtual space; got {nocc}")


@partial(jax.jit, static_argnames=("occupied_pair_batch_size",))
def _xstream_left_panel_jit(t2_pairs, inner_panel, out_panel, x_panel, *,
                            occupied_pair_batch_size):
    """One rank panel's contribution to ``P[a,m] P[c,m] X[b,d,m]``."""

    n_padded_pairs, nvir, _ = t2_pairs.shape
    n_pair_blocks = n_padded_pairs // occupied_pair_batch_size

    def pair_body(pair_block, out_acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau_block = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir))
        # S[n,d,mu] = sum_c tau[n,c,d] P[c,mu]
        s = jnp.einsum("ncd,cm->ndm", tau_block, inner_panel)
        # Y[n,b,mu] = sum_d S[n,d,mu] X[b,d,mu]
        y = jnp.einsum("ndm,bdm->nbm", s, x_panel)
        out_block = jnp.einsum("am,nbm->nab", out_panel, y)
        return jax.lax.dynamic_update_slice(out_acc, out_block, (pair0, 0, 0))

    return jax.lax.fori_loop(
        0, n_pair_blocks, pair_body, jnp.zeros_like(t2_pairs))


@partial(jax.jit, static_argnames=("occupied_pair_batch_size",))
def _xstream_right_panel_jit(t2_pairs, inner_panel, out_panel, x_panel, *,
                             occupied_pair_batch_size):
    """One rank panel's contribution to ``X[a,c,m] P[b,m] P[d,m]``."""

    n_padded_pairs, nvir, _ = t2_pairs.shape
    n_pair_blocks = n_padded_pairs // occupied_pair_batch_size

    def pair_body(pair_block, out_acc):
        pair0 = pair_block * occupied_pair_batch_size
        tau_block = jax.lax.dynamic_slice(
            t2_pairs, (pair0, 0, 0),
            (occupied_pair_batch_size, nvir, nvir))
        # S[n,c,mu] = sum_d tau[n,c,d] P[d,mu]
        s = jnp.einsum("ncd,dm->ncm", tau_block, inner_panel)
        # Y[n,a,mu] = sum_c X[a,c,mu] S[n,c,mu]
        y = jnp.einsum("acm,ncm->nam", x_panel, s)
        out_block = jnp.einsum("nam,bm->nab", y, out_panel)
        return jax.lax.dynamic_update_slice(out_acc, out_block, (pair0, 0, 0))

    return jax.lax.fori_loop(
        0, n_pair_blocks, pair_body, jnp.zeros_like(t2_pairs))


def _stream_partial_x(panel_kernel, t2, left_out, left_inner, x_backing, nocc,
                      *, occupied_pair_batch_size, rank_panel_size):
    """Host-loop rank-panel streaming shared by the left and right X terms.

    ``left_out``/``left_inner`` are the small endpoint factors (device-
    resident); ``x_backing`` is the ``(nmo, nmo, rank)`` X factor on any
    backing -- a NumPy array (view slicing) or an HDF5 dataset (partial
    reads) -- and only one panel is on the device at a time.
    """

    if occupied_pair_batch_size < 1 or rank_panel_size < 1:
        raise ValueError("occupied_pair_batch_size and rank_panel_size must be positive")
    nocc = int(nocc)
    _validate_x_stream(t2, left_out, left_inner, x_backing, nocc)

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

    total = jnp.zeros((padded_pairs, nvir, nvir), dtype=t2_pairs.dtype)
    for rank_block in range(n_rank_blocks):
        m0 = rank_block * rank_panel_size
        m1 = min(m0 + rank_panel_size, rank)
        x_panel = jax.device_put(
            _read_x_rank_panel(x_backing, nocc, m0, m1, rank_panel_size))
        inner_panel = inner_padded[:, m0:m0 + rank_panel_size]
        out_panel = out_padded[:, m0:m0 + rank_panel_size]
        total = total + panel_kernel(
            t2_pairs, inner_panel, out_panel, x_panel,
            occupied_pair_batch_size=occupied_pair_batch_size)
    return total[:n_pairs].reshape(nocc_i, nocc_j, nvir, nvir)


def contract_partial_x_left_t2_streamed(t2, left_out, left_inner, x_backing, nocc,
                                        *, occupied_pair_batch_size=8,
                                        rank_panel_size=128):
    """Streamed ``P[a,m] P[c,m] X[b,d,m]``: one X rank panel on device at a time."""

    return _stream_partial_x(
        _xstream_left_panel_jit, t2, left_out, left_inner, x_backing, nocc,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size)


def contract_partial_x_right_t2_streamed(t2, right_out, right_inner, x_backing, nocc,
                                         *, occupied_pair_batch_size=8,
                                         rank_panel_size=128):
    """Streamed ``X[a,c,m] P[b,m] P[d,m]``: one X rank panel on device at a time."""

    return _stream_partial_x(
        _xstream_right_panel_jit, t2, right_out, right_inner, x_backing, nocc,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size)


def contract_isdf_factor_direct_terms_t2_xstream(
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
) -> Mapping[str, Array]:
    """Factor-direct terms with X streamed panel-wise from its backing.

    Identical terms and signs to
    :func:`contract_isdf_factor_direct_terms_t2`; only the two X-consuming
    terms change how X reaches the device.  Every other input is small and
    device-lifted exactly as in the full-block path.
    """

    p, grad_p, u1, u3, d = map(jnp.asarray, (p, grad_p, u1, u3, d))
    if grad_p.shape != (p.shape[0], p.shape[1], 3):
        raise ValueError(
            "grad_p must have shape (nvir, rank, 3); "
            f"got {grad_p.shape} for p={p.shape}"
        )
    if u1.shape != (p.shape[1], p.shape[1], 3):
        raise ValueError(f"u1 must have shape (rank, rank, 3); got {u1.shape}")

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
    x_direct = _timed("fd_x_left", contract_partial_x_left_t2_streamed,
                      t2, p, p, x_backing, nocc, **_kw)
    x_pair = _timed("fd_x_right", contract_partial_x_right_t2_streamed,
                    t2, p, p, x_backing, nocc, **_kw)

    # Same sign assembly as contract_isdf_factor_direct_terms_t2.
    tc_direct = 0.5 * (k1_direct - k2_direct + k3_direct)
    tc_pair = 0.5 * (k1_pair - k2_pair + k3_pair)
    delta_direct = d_direct - x_direct
    delta_pair = d_pair - x_pair
    tc = -(tc_direct + tc_pair)
    delta_u = -(delta_direct + delta_pair)
    final = tc + delta_u
    return {
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
        "tc_direct": tc_direct,
        "tc_pair": tc_pair,
        "delta_direct": delta_direct,
        "delta_pair": delta_pair,
        "tc": tc,
        "delta_u": delta_u,
        "final": final,
    }


def contract_isdf_factor_direct_terms_t2(
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
    the Phase-A numerical gate.  The final values reproduce the current
    ``_assemble_tc_tile`` / ``_assemble_delta_u_tile`` signs:

    * ``tc = -0.5 * ((K1 - K2 + K3) + pair_swap(...))``
    * ``delta_u = -((D - X) + pair_swap(D - X))``

    There is no ordinary DF Coulomb term here: it is the separately-gated
    Phase-B scope and is deliberately not changed by this prototype.
    """

    p, grad_p, u1, u3, d, x = map(jnp.asarray, (p, grad_p, u1, u3, d, x))
    if grad_p.shape != (p.shape[0], p.shape[1], 3):
        raise ValueError(
            "grad_p must have shape (nvir, rank, 3); "
            f"got {grad_p.shape} for p={p.shape}"
        )
    if u1.shape != (p.shape[1], p.shape[1], 3):
        raise ValueError(f"u1 must have shape (rank, rank, 3); got {u1.shape}")

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
    x_direct = contract_partial_x_left_t2(
        t2, p, p, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )
    x_pair = contract_partial_x_right_t2(
        t2, p, p, x,
        occupied_pair_batch_size=occupied_pair_batch_size,
        rank_panel_size=rank_panel_size,
    )

    tc_direct = 0.5 * (k1_direct - k2_direct + k3_direct)
    tc_pair = 0.5 * (k1_pair - k2_pair + k3_pair)
    delta_direct = d_direct - x_direct
    delta_pair = d_pair - x_pair
    tc = -(tc_direct + tc_pair)
    delta_u = -(delta_direct + delta_pair)
    final = tc + delta_u
    return {
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
        "tc_direct": tc_direct,
        "tc_pair": tc_pair,
        "delta_direct": delta_direct,
        "delta_pair": delta_pair,
        "tc": tc,
        "delta_u": delta_u,
        "final": final,
    }


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
    """Measure every Phase-A branch and return the exact factor-direct terms.

    Profiling is explicitly opt-in and only used by the scratch prototype and
    its H10 run card. Each profile distinguishes the schedule-panel estimate
    from XLA's executable memory analysis. Neither is a process-wide allocator
    peak; the card labels any ``nvidia-smi`` sample accordingly.
    """

    terms = contract_isdf_factor_direct_terms_t2(
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
            lambda: contract_partial_x_left_t2(
                t2, p, p, x,
                occupied_pair_batch_size=occupied_pair_batch_size,
                rank_panel_size=rank_panel_size,
            ),
            x_estimate,
            x_memories["x_direct"],
        ),
        "x_pair": (
            lambda: contract_partial_x_right_t2(
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
