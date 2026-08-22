"""JAX implementation of X transcorrelated methods."""

import contextlib
from functools import partial, reduce
import numpy as np
import os
import gc
import logging
import threading
import time
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
import h5py
from flax import struct
from collections import OrderedDict
from .tc import (
    TC,
    ISDFTC,
    _laux_gradient_mode,
    _normalize_panel_layout,
    _transpose_panel_layout,
    _pad_axis,
    trim_panel,
)
from .utils.tile_memory import isdf_tile_peak_bytes as _isdf_tile_peak_bytes
from .utils.tile_memory import find_max_blksize as _find_max_blksize
from . import tc_helper
from . import kmat as kmat_jax
from .utils import sharding_core

logger = logging.getLogger(__name__)


def _contract_tucker_x_residual(phi_p, phi_q, u_r, u_s, z):
    """Contract an orbital-leg Tucker X core without reconstructing X.

    ``z`` represents ``X[r,s,c] = U[r,a] Z[a,b,c] U[s,b]``.  The returned
    quantity is the X contribution to the unsymmetrized Delta-U tile,
    ``-sum_c phi_p[p,c] phi_q[q,c] X[r,s,c]``.
    """
    phi_p = jnp.asarray(phi_p)
    phi_q = jnp.asarray(phi_q)
    u_r = jnp.asarray(u_r)
    u_s = jnp.asarray(u_s)
    z = jnp.asarray(z)
    c_pq = phi_p[:, None, :] * phi_q[None, :, :]
    core_pq = jnp.einsum("pqc,abc->pqab", c_pq, z, optimize=True)
    return -jnp.einsum("pqab,ra,sb->pqrs", core_pq, u_r, u_s,
                       optimize=True)


def _get_tucker_x_factors(kernels):
    """Return a validated ``(U, Z)`` Tucker-X representation, if present.

    A factorized X deliberately uses a different key from the dense ``X``
    dataset.  That prevents an approximate calculation from accidentally
    falling back to, or preloading, the full tensor.  The representation is
    ``X[r,s,c] ~= U[r,a] Z[a,b,c] U[s,b]``.
    """
    factors = kernels.get("X_tucker")
    if factors is None:
        return None
    if not isinstance(factors, dict) or set(("U", "Z")) - set(factors):
        raise ValueError("X_tucker must be a mapping containing U and Z")

    u = factors["U"]
    z = factors["Z"]
    if getattr(u, "ndim", None) != 2 or getattr(z, "ndim", None) != 3:
        raise ValueError("X_tucker factors must have U[orbital,factor] and Z[factor,factor,rank]")
    if z.shape[:2] != (u.shape[1], u.shape[1]):
        raise ValueError(
            "X_tucker dimensions disagree: "
            f"U has {u.shape[1]} factors but Z has shape {z.shape}"
        )
    return u, z


def _tucker_x_normal_order_intermediates(u, z, dm1, phi_tilde, gb):
    """Return the X intermediates used by reference normal ordering.

    This is algebraically equivalent to contracting a dense X tensor, but
    retains the separated orbital legs throughout.
    """
    u = jnp.asarray(u)
    z = jnp.asarray(z)
    dm1 = jnp.asarray(dm1)
    phi_tilde = jnp.asarray(phi_tilde)
    gb = jnp.asarray(gb)

    density_core = jnp.einsum("ra,rs,sb->ab", u, dm1, u, optimize=True)
    wc = jnp.einsum("abc,ab->c", z, density_core, optimize=True)
    phi_core = jnp.einsum("ra,rc->ac", u, phi_tilde, optimize=True)
    y_all = jnp.einsum("qb,abc,ac->qc", u, z, phi_core, optimize=True)
    j_x_sym = -jnp.einsum("pa,qb,abc,c->pq", u, u, z, gb, optimize=True)
    return wc, y_all, j_x_sym


# ------------------------------------------------------------------
# Opt-in issue-stage decomposition for ``_assemble_2b_tile``.
#
# The ``_run_tiled_block_pipeline`` scheduler (in ``xtc_ccsd.py``)
# reports ``issue_s`` = total wall time spent inside ``issue_tile``.
# For medium blocks that call is ``compute_2b_tile`` → a chain of
# ``_get_tc_direct_tile`` + ``_get_delta_u_direct_tile`` + a final
# ``tc_tile + delta_u_tile`` add.  To attribute ``issue_s`` to the
# three stages we maintain a process-wide float-accumulator dict
# that the assembler updates from within its own ``@jax.jit``
# boundary.
#
# The mechanism is strictly opt-in — if ``_ISSUE_STAGE_STATS["current"]``
# is ``None`` (the common case, including the stand-alone
# ``get_2b`` / test-fake paths), the assembler takes no cost at all.
# Callers that want timing (currently only the
# ``_run_tiled_block_pipeline``'s ``issue`` closure) open
# ``issue_stage_stats_scope()`` before the call and read the dict after.
#
# IMPORTANT: this is process-wide rather than ``threading.local()`` so
# that the issue worker threads spawned by ``_round_robin_pipeline``
# (each running on its own thread, post-parallel-issue refactor) can
# all see and update the parent pipeline's stats dict.  The lock around
# accumulation is held only briefly (one ``dict.get`` + add).
#
# We intentionally do NOT change the signature of ``_assemble_2b_tile``
# / ``_assemble_tc_tile`` / ``_assemble_delta_u_tile``: several test
# doubles implement those methods (e.g. the ``_FakeMOXTC`` fixture in
# ``test_vvvv_paneling.py``) and would silently ignore a new ``**kwargs``
# slot, giving the false impression that instrumentation is active.
# ------------------------------------------------------------------
_ISSUE_STAGE_STATS = {
    "current": None,
    "lock": threading.Lock(),
    "scope_lock": threading.Lock(),
    "owner_thread": None,
}


def _accum_issue_stage(name, dt):
    """Add ``dt`` seconds to ``name`` if a pipeline stats scope is active.

    Process-wide; called concurrently from multiple issue worker threads
    in ``_round_robin_pipeline``'s parallel-issue mode.  The lock window
    is a single dict ``get`` + add, so contention is negligible.
    """
    stats = _ISSUE_STAGE_STATS["current"]
    if stats is not None:
        with _ISSUE_STAGE_STATS["lock"]:
            stats[name] = stats.get(name, 0.0) + dt


@contextlib.contextmanager
def issue_stage_stats_scope():
    """Establish a per-pipeline accumulator for ``_accum_issue_stage``.

    Yields a dict that subsequent ``_accum_issue_stage`` calls will
    mutate.  The accumulator is process-wide (visible from all threads)
    so that the per-device issue worker threads spawned by
    ``_round_robin_pipeline`` can read it; ``threading.local`` would
    isolate the parent's scope from the workers and lose every timer.

    Concurrency contract
    --------------------
    Only ONE pipeline-level scope may be active at a time.  Two
    pipelines running concurrently on different threads would both
    overwrite ``_ISSUE_STAGE_STATS["current"]`` and silently mix /
    lose timer attribution.  The scheduler is single-pipeline today
    so this never happens in practice, but the contract is enforced
    explicitly here to fail fast if a future caller violates it.

    Nested scopes from the SAME thread are allowed — the outer scope
    is restored on ``__exit__`` — so it's safe for the scheduler to
    enter a scope even if a callee under the same thread might also
    enter one.
    """
    cur_thread = threading.get_ident()
    with _ISSUE_STAGE_STATS["scope_lock"]:
        owner = _ISSUE_STAGE_STATS["owner_thread"]
        if owner is not None and owner != cur_thread:
            raise RuntimeError(
                "issue_stage_stats_scope() does not support concurrent "
                f"pipelines from different threads — already owned by "
                f"thread {owner}, attempted entry from thread "
                f"{cur_thread}.  If you need per-pipeline timing, "
                "either serialise the pipelines or replace this "
                "process-wide accumulator with a per-pipeline dict "
                "passed explicitly through the call chain."
            )
        prev = _ISSUE_STAGE_STATS["current"]
        prev_owner = owner
        fresh: dict = {}
        _ISSUE_STAGE_STATS["current"] = fresh
        _ISSUE_STAGE_STATS["owner_thread"] = cur_thread
    try:
        yield fresh
    finally:
        with _ISSUE_STAGE_STATS["scope_lock"]:
            _ISSUE_STAGE_STATS["current"] = prev
            _ISSUE_STAGE_STATS["owner_thread"] = prev_owner

# ---------------------------------------------------------------------------
# Host-side cache for X orbital slices read from HDF5.
#
# Within each CCSD phase (ovvv / vovv / vvvv), every block iteration calls
# _contract_delta_U_kernels with the *same* (slice_r, slice_s) — only the
# (p, q) orbital indices vary.  Caching the last-read X slice avoids
# re-reading tens of GB from HDF5 per block.
#
# The cache holds at most two slices.
# ---------------------------------------------------------------------------
_X_HDF5_CACHE = OrderedDict()
# Keys added when the auto-shrink warning has been emitted for a device so
# we log once per device per run, not once per tile on the CCSD hot path.
_DELTA_U_AUTOSHRINK_WARNED = set()


def _read_X_slice(X, slice_r, slice_s):
    """Read ``X[slice_r, slice_s]``, reusing a host-side cache when possible.

    If *X* is an HDF5 dataset and the requested slice matches a
    recently read slice, the cached numpy array is returned directly — no I/O.
    """
    if isinstance(X, h5py.Dataset):
        key = (id(X), _slice_key(slice_r), _slice_key(slice_s))
        if key in _X_HDF5_CACHE:
            logger.debug("X slice cache HIT  (%s, %s)", slice_r, slice_s)
            data = _X_HDF5_CACHE.pop(key)
            _X_HDF5_CACHE[key] = data  # Move to end (most recently used)
            return data
            
        logger.debug("X slice cache MISS (%s, %s) — reading from HDF5", slice_r, slice_s)
        data = X[slice_r, slice_s]
        
        if len(_X_HDF5_CACHE) >= 2:
            _X_HDF5_CACHE.popitem(last=False)  # Remove least recently used
            
        _X_HDF5_CACHE[key] = data
        return data
        
    return X[slice_r, slice_s]


def invalidate_X_cache():
    """Explicitly free the cached X slice (e.g., at end of CCSD iteration)."""
    _X_HDF5_CACHE.clear()


def _slice_key(sl):
    """Hashable representation of a slice or index array."""
    if isinstance(sl, slice):
        return ("slice", sl.start, sl.stop, sl.step)
    return ("idx", tuple(np.asarray(sl).ravel()))


def _chunk_selector(idx, start, stop):
    """Convert a contiguous subrange of indices into a slice when possible."""
    sub = np.asarray(idx[start:stop])
    if sub.size == 0:
        return slice(0, 0, 1)
    if sub.size == 1:
        i = int(sub[0])
        return slice(i, i + 1, 1)

    steps = np.diff(sub)
    if np.all(steps == steps[0]):
        step = int(steps[0])
        # step==0 would make slice(start, stop, 0) raise; fall back to the
        # explicit index array (degenerate but legal for fancy indexing).
        if step == 0:
            return sub
        return slice(int(sub[0]), int(sub[-1] + step), step)
    return sub


def _estimate_delta_u_contraction_bytes(Np, Nq, Nr, Ns, N_rank):
    """Estimate device memory for the legacy scan-based Delta U kernel."""
    x_sliced_size_bytes = int(Nr * Ns * N_rank * 8)
    d_size_bytes = int(N_rank * N_rank * 8)
    scan_carry_bytes = int(Np * Nq * Nr * Ns * 8)
    total_needed_bytes = x_sliced_size_bytes * 2 + d_size_bytes + 3 * scan_carry_bytes
    return x_sliced_size_bytes, d_size_bytes, scan_carry_bytes, total_needed_bytes


def _estimate_delta_u_direct_tile_bytes(Np, Nq, Nr, Ns, N_rank, *, include_d=True):
    """Estimate device memory for the balanced direct Delta U tile kernel.

    Delegates to the canonical formula in
    :func:`pytc.utils.tile_memory.isdf_tile_peak_bytes` so that the runtime
    memory guard and the build-phase estimators in ``gpu_memory.py`` always
    agree.

    ``include_d`` should be False when D is already a persistent on-device
    resident (cached by the per-device cache); otherwise D gets counted both
    in ``in_use`` (via the free-bytes probe) and again in the total peak —
    effectively doubling its contribution and triggering spurious "tile
    exceeds device memory" refusals.
    """
    B = 8
    d_size_bytes   = int(N_rank * N_rank * B)
    x_size_bytes   = int(Nr * Ns * N_rank * B)
    cpq_size_bytes = int(Np * Nq * N_rank * B)
    out_size_bytes = int(Np * Nq * Nr * Ns * B)
    total = _isdf_tile_peak_bytes(Np, Nq, Nr, Ns, N_rank, include_d=include_d)
    return {
        "D":   d_size_bytes,
        "X":   x_size_bytes,
        "Cpq": cpq_size_bytes,
        "Crs": x_size_bytes,    # same shape as X
        "out": out_size_bytes,
        "total": total,
    }


def _get_device_free_bytes(device=None):
    """Return currently free bytes for a specific local device when possible."""
    if device is not None:
        if getattr(device, "platform", None) == "cpu":
            import psutil
            available = int(psutil.virtual_memory().available)
            if available <= 0:
                raise RuntimeError("host does not report usable available memory")
            return available

        stats = device.memory_stats()
        if not stats or "bytes_limit" not in stats:
            raise RuntimeError(
                f"device {device!r} does not report a usable memory limit"
            )
        pool_limit = int(stats['bytes_limit'])
        in_use = int(stats.get('bytes_in_use', 0))
        return max(pool_limit - in_use, 0)

    from pytc.utils.gpu_memory import _get_gpu_free_bytes
    return _get_gpu_free_bytes()


def compute_2b_tile(xtc_obj, jastrow_params, ranges, device=None, panel_size=None,
                    panel_layout="pr"):
    """Execute one XTC 2-body tile using the internal tile API only."""
    if not hasattr(xtc_obj, "_assemble_2b_tile"):
        raise TypeError(
            f"{type(xtc_obj).__name__} does not implement the internal XTC tile API"
        )

    kernels = getattr(xtc_obj, "isdf_kernels", None)
    required = ("K1_kernel", "K3_kernel", "D")
    missing = [key for key in required if kernels is None or key not in kernels]
    if kernels is None or ("X" not in kernels and "X_tucker" not in kernels):
        missing.append("X or X_tucker")
    if missing:
        raise RuntimeError(
            "XTC tile execution requires precomputed ISDF kernels. "
            f"Missing: {missing}"
        )

    return xtc_obj._assemble_2b_tile(
        jastrow_params, kernels, ranges, device=device,
        panel_size=panel_size, panel_layout=panel_layout
    )


@struct.dataclass
class XTC(TC):
    """JAX implementation of extended transcorrelated methods using flax dataclass.
    
    Attributes:
        mo_occ: Molecular orbital occupation numbers (N_orb,)
        energy_nuc: Nuclear repulsion energy (static)
    """
    mo_occ: jnp.ndarray = struct.field(default=None)
    energy_nuc: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def from_pyscf(cls, mf, jastrow_factor, mo_coeff=None, grid_lvl=2, grid_chunk_size=None):
        """Initialize XTC object from PySCF mean-field object."""
        tc_obj = super().from_pyscf(mf, jastrow_factor, mo_coeff, grid_lvl, grid_chunk_size)
        
        mo_occ = jnp.asarray(mf.mo_occ)
        energy_nuc = mf.energy_nuc()
        
        return cls(
            grid_points=tc_obj.grid_points,
            weights=tc_obj.weights,
            phi=tc_obj.phi,
            grad_phi=tc_obj.grad_phi,
            n_orb=tc_obj.n_orb,
            grid_lvl=tc_obj.grid_lvl,
            jastrow_factor=tc_obj.jastrow_factor,
            mo_coeff=tc_obj.mo_coeff,
            nocc=tc_obj.nocc,
            mo_occ=mo_occ,
            energy_nuc=energy_nuc
        )
    
    @property
    def n_grid(self):
        """Number of grid points."""
        return len(self.grid_points)
    
    def _calc_v_block(self, r1_batch, phi, weights, jastrow_params, slice_rows, slice_cols, batch_size=1000):
        """Calculate V_qt(r₁) for a batch of r1 points and specific row/col slices.
        
        Args:
            r1_batch: (batch_size, 3)
            phi: (Nb, N_grid)
            weights: (N_grid,)
            jastrow_params: Jastrow parameters
            slice_rows: slice object for row indices (q)
            slice_cols: slice object for col indices (t)
            batch_size: Inner batch size for r2 scan
            
        Returns:
            V_batch: (N_rows, N_cols, batch_size, 3)
        """
        n_orb, n_grid = phi.shape
        
        phi_rows = phi[slice_rows]  # (N_rows, N_grid)
        phi_cols = phi[slice_cols]  # (N_cols, N_grid)
        
        n_rows = phi_rows.shape[0]
        n_cols = phi_cols.shape[0]
        
        padded_size = ((n_grid + batch_size - 1) // batch_size) * batch_size
        padded_grid = jnp.pad(self.grid_points, ((0, padded_size - n_grid), (0, 0)))
        padded_weights = jnp.pad(weights, (0, padded_size - n_grid))
        padded_phi_rows = jnp.pad(phi_rows, ((0, 0), (0, padded_size - n_grid)))
        padded_phi_cols = jnp.pad(phi_cols, ((0, 0), (0, padded_size - n_grid)))
        
        r2_batches = padded_grid.reshape(-1, batch_size, 3)
        weights_batches = padded_weights.reshape(-1, batch_size)
        phi_rows_batches = padded_phi_rows.reshape(n_rows, -1, batch_size)
        phi_cols_batches = padded_phi_cols.reshape(n_cols, -1, batch_size)
        
        def scan_body(carry, args):
            r2_batch, w_batch, phi_row_batch, phi_col_batch = args
            
            # Compute phi_paired for this r2 batch: phi_q(r2) * phi_t(r2)
            # (N_rows, batch) * (N_cols, batch) -> (N_rows, N_cols, batch)
            phi_paired_r2 = jnp.einsum('ib,jb->ijb', phi_row_batch, phi_col_batch)
            weighted_phi_r2 = phi_paired_r2 * w_batch[None, None, :]
            
            # Compute gradients: grad_J(r1, r2) -> (batch_r1, batch_r2, 3)
            grads = self.jastrow_factor.grad_r_batch(r1_batch, r2_batch, jastrow_params)
            
            # Contract: sum_{r2} phi(r2) * grad(r1, r2)
            # weighted_phi_r2: (N_rows, N_cols, batch_r2)
            # grads: (batch_r1, batch_r2, 3)
            # Result: (N_rows, N_cols, batch_r1, 3)
            term = jnp.einsum('ijb,obd->ijod', weighted_phi_r2, grads)
            
            return carry + term, None

        init_val = jnp.zeros((n_rows, n_cols, len(r1_batch), 3))
        final_val, _ = jax.lax.scan(scan_body, init_val, 
                                   (r2_batches, weights_batches, 
                                    phi_rows_batches.transpose(1, 0, 2), 
                                    phi_cols_batches.transpose(1, 0, 2)))
        
        return final_val



    def get_delta_U(self, jastrow_params, dm1=None, ranges=None, batch_size=1000):
        """Get delta_U matrix with memory-efficient batching and multi-GPU support.
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (must be diagonal if provided)
            ranges: Tuple of slices (p, q, r, s) for block calculation
            batch_size: Batch size for grid integration
            
        Returns:
            delta_U: The correction term.
                     If ranges provided: (Np, Nr, Nq, Ns)
                     Otherwise: (N, N, N, N)
        """
        start_time = time.perf_counter()
        logger.debug("Starting XTC.get_delta_U")
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.n_grid
        
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        
        n_occ_vec = jnp.diagonal(dm1)
        
        remainder = n_grid % n_devices
        if remainder != 0:
            padding = n_devices - remainder
            padded_grid_points = np.pad(np.asarray(self.grid_points), ((0, padding), (0, 0)))
            padded_weights = np.pad(np.asarray(self.weights), ((0, padding),))
            padded_phi = np.pad(np.asarray(self.phi), ((0, 0), (0, padding)))
        else:
            padded_grid_points = np.asarray(self.grid_points)
            padded_weights = np.asarray(self.weights)
            padded_phi = np.asarray(self.phi)

        mesh = sharding_core.create_1d_mesh(devices=devices, axis_name='devices')
        rep_sharding = sharding_core.get_replicated_sharding(mesh)
        grid_sharding = NamedSharding(mesh, P('devices', None))
        weights_sharding = NamedSharding(mesh, P('devices'))
        phi_sharding = NamedSharding(mesh, P(None, 'devices'))

        sharded_grid_r1 = jax.device_put(padded_grid_points, grid_sharding)
        sharded_weights_r1 = jax.device_put(padded_weights, weights_sharding)
        sharded_phi_r1 = jax.device_put(padded_phi, phi_sharding)
        params_rep = jax.tree_util.tree_map(
            lambda x: jax.device_put(np.asarray(x), rep_sharding), jastrow_params
        )
        
        if ranges is None:
            full_slice = slice(None)
            ranges = (full_slice, full_slice, full_slice, full_slice)
            
        slice_p, slice_q, slice_r, slice_s = ranges
        slice_occ = slice(0, self.nocc) if self.nocc is not None else slice(None)
        
        n_occ_vec_active = n_occ_vec[slice_occ]
        
        def get_size(s, size):
            start, stop, step = s.indices(size)
            return (stop - start + (step - 1)) // step
            
        Np = get_size(slice_p, self.n_orb)
        Nq = get_size(slice_q, self.n_orb)
        Nr = get_size(slice_r, self.n_orb)
        Ns = get_size(slice_s, self.n_orb)
        Nocc = get_size(slice_occ, self.n_orb)

        def compute_on_device(grid_r1, weights_r1, phi_r1, jastrow_params):
            n_local = grid_r1.shape[0]
            local_remainder = n_local % batch_size
            if local_remainder != 0:
                local_padding = batch_size - local_remainder
                grid_r1_batched = jnp.pad(grid_r1, ((0, local_padding), (0, 0)))
                weights_r1_batched = jnp.pad(weights_r1, ((0, local_padding),))
                phi_r1_batched = jnp.pad(phi_r1, ((0, 0), (0, local_padding)))
            else:
                grid_r1_batched = grid_r1
                weights_r1_batched = weights_r1
                phi_r1_batched = phi_r1
                
            r1_batches = grid_r1_batched.reshape(-1, batch_size, 3)
            weights_batches = weights_r1_batched.reshape(-1, batch_size)
            phi_batches = phi_r1_batched.reshape(self.n_orb, -1, batch_size)
            
            @jax.checkpoint
            def scan_body(carry, args):
                r1_batch, w_batch, phi_batch = args
                
                curr_batch_size = r1_batch.shape[0]
                
                # --- Compute V blocks ---
                # We need V blocks for:
                # (p, r), (q, s)
                # (occ, p), (occ, r), (occ, q), (occ, s)
                # (occ, occ) for W
                
                # Optimization: Compute unique blocks only
                # V is symmetric in orbital indices, so V_ij = V_ji
                # But _calc_v_block returns (rows, cols, batch, 3)
                # So V_ji = V_ij.swapaxes(0, 1)
                
                V_occ_occ = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                              slice_occ, slice_occ, batch_size)
                
                V_pq = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                         slice_p, slice_q, batch_size)
                
                V_rs = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                         slice_r, slice_s, batch_size)
                
                # 4. V_occ_blocks
                # We need V_{k,p}, V_{k,q}, V_{k,r}, V_{k,s} where k in occ
                # We can compute V_{occ, p} etc.
                V_occ_p = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                            slice_occ, slice_p, batch_size)
                V_occ_q = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                            slice_occ, slice_q, batch_size)
                V_occ_r = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                            slice_occ, slice_r, batch_size)
                V_occ_s = self._calc_v_block(r1_batch, self.phi, self.weights, jastrow_params, 
                                            slice_occ, slice_s, batch_size)
                
                # --- Compute Intermediates (Diagonal dm1) ---
                
                # W = 2 * sum_k V_{kk} n_k
                # V_occ_occ: (Nocc, Nocc, batch, 3)
                # Diagonal V_{kk}: (Nocc, batch, 3)
                V_kk = jnp.einsum('ii...->i...', V_occ_occ)
                # n_occ_vec: (Nocc,)
                # W: (batch, 3)
                W = 2 * jnp.einsum('i,ibd->bd', n_occ_vec_active, V_kk)
                
                # Wbar = 2 * sum_k phi_{kk} n_k
                # phi_batch: (N_orb, batch)
                phi_occ = phi_batch[slice_occ] # (Nocc, batch)
                phi_kk = phi_occ * phi_occ
                # Wbar: (batch,)
                Wbar = 2 * jnp.einsum('i,ib->b', n_occ_vec_active, phi_kk)
                
                # --- Block (p, q) Terms ---
                # V_{pk} = V_{kp} = V_occ_p (Nocc, Np, batch, 3)
                # V_{qk} = V_{kq} = V_occ_q (Nocc, Nq, batch, 3)
                # Zbar_{pq}: (Np, Nq, batch)
                Zbar_pq = jnp.einsum('i,ipbd,iqbd->pqb', n_occ_vec_active, V_occ_p, V_occ_q)
                
                # G_{pq} = sum_k (phi_{kp} V_{qk} + phi_{kq} V_{pk}) n_k
                # phi_{kp} = phi_k * phi_p
                phi_p = phi_batch[slice_p] # (Np, batch)
                phi_q = phi_batch[slice_q] # (Nq, batch)
                # phi_{kp}: (Nocc, Np, batch)
                phi_kp = jnp.einsum('ib,pb->ipb', phi_occ, phi_p)
                phi_kq = jnp.einsum('ib,qb->iqb', phi_occ, phi_q)
                
                # G_{pq}: (Np, Nq, batch, 3)
                G_pq = jnp.einsum('i,ipb,iqbd->pqbd', n_occ_vec_active, phi_kp, V_occ_q) + \
                       jnp.einsum('i,iqb,ipbd->pqbd', n_occ_vec_active, phi_kq, V_occ_p)
                       
                # Vbar_{pq} = sum_d W_d * V_{pq,d}
                Vbar_pq = jnp.einsum('bd,pqbd->pqb', W, V_pq)
                A_pq = Vbar_pq - Zbar_pq # (Np, Nq, batch)
                
                # B_{pq} = 0.5 * Wbar * V_{pq} - G_{pq}
                # B_{pq}: (Np, Nq, batch, 3)
                B_pq = 0.5 * Wbar[None, None, :, None] * V_pq - G_pq
                
                # --- Block (r, s) Terms ---
                # Symmetric to (p, q)
                
                Zbar_rs = jnp.einsum('i,irbd,isbd->rsb', n_occ_vec_active, V_occ_r, V_occ_s)
                
                phi_r = phi_batch[slice_r]
                phi_s = phi_batch[slice_s]
                phi_kr = jnp.einsum('ib,rb->irb', phi_occ, phi_r)
                phi_ks = jnp.einsum('ib,sb->isb', phi_occ, phi_s)
                
                G_rs = jnp.einsum('i,irb,isbd->rsbd', n_occ_vec_active, phi_kr, V_occ_s) + \
                       jnp.einsum('i,isb,irbd->rsbd', n_occ_vec_active, phi_ks, V_occ_r)
                       
                Vbar_rs = jnp.einsum('bd,rsbd->rsb', W, V_rs)
                A_rs = Vbar_rs - Zbar_rs
                
                B_rs = 0.5 * Wbar[None, None, :, None] * V_rs - G_rs
                
                # --- Combine Terms ---
                
                # term1 = phi_{pq} * A_{rs}
                # phi_{pq} = phi_p * phi_q
                phi_pq = jnp.einsum('pb,qb->pqb', phi_p, phi_q)
                phi_pq_w = phi_pq * w_batch[None, None, :]
                
                # term1: (Np, Nq, Nr, Ns)
                # einsum: pqb, rsb -> pqrs (sum over b)
                term1 = jnp.einsum('pqb,rsb->pqrs', phi_pq_w, A_rs)
                
                # term2 = V_{pq} * B_{rs}
                # V_{pq}: (Np, Nq, batch, 3)
                # B_{rs}: (Nr, Ns, batch, 3)
                # term2: (Np, Nq, Nr, Ns)
                V_pq_w = V_pq * w_batch[None, None, :, None]
                term2 = jnp.einsum('pqbd,rsbd->pqrs', V_pq_w, B_rs)
                
                # term1_sym = phi_{rs} * A_{pq}
                phi_rs = jnp.einsum('rb,sb->rsb', phi_r, phi_s)
                phi_rs_w = phi_rs * w_batch[None, None, :]
                term1_sym = jnp.einsum('rsb,pqb->pqrs', phi_rs_w, A_pq)
                
                # term2_sym = V_{rs} * B_{pq}
                V_rs_w = V_rs * w_batch[None, None, :, None]
                term2_sym = jnp.einsum('rsbd,pqbd->pqrs', V_rs_w, B_pq)
                
                contrib = term1 + term2 + term1_sym + term2_sym
                
                return carry + contrib, None

            init_val = jnp.zeros((Np, Nq, Nr, Ns))
            
            local_delta_U, _ = jax.lax.scan(scan_body, init_val, (r1_batches, weights_batches, phi_batches.transpose(1, 0, 2)))
            
            # Sum results across devices
            total_delta_U = jax.lax.psum(local_delta_U, axis_name='devices')
            return total_delta_U

        @shard_map(
            mesh=mesh,
            in_specs=(P('devices', None), P('devices'), P(None, 'devices'), P()),
            out_specs=P(),
            check_vma=False,
        )
        def sharded_compute(grid_r1, weights_r1, phi_r1, params):
            return compute_on_device(grid_r1, weights_r1, phi_r1, params)

        total_delta_U = sharded_compute(sharded_grid_r1, sharded_weights_r1, sharded_phi_r1, params_rep)
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"XTC.get_delta_U completed in {total_time:.4f} s")
        return -total_delta_U

    def get_delta_h(self, jastrow_params, dm1=None, block_str=None, ranges=None, orb_block_size=None, batch_size=1000):
        """Get or compute delta_h with memory optimization."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
            
        if ranges is None:
            ranges = (slice(None), slice(None), slice(None), slice(None))
            
        slice_p, slice_q, slice_r, slice_s = ranges
        
        slice_occ = slice(0, self.nocc)
        
        # (pq|rs) term
        ranges_pqrs = (slice_p, slice_q, slice_occ, slice_occ)
        delta_U_pqrs = self.get_delta_U(jastrow_params, dm1, ranges=ranges_pqrs, batch_size=batch_size)
        # delta_U_pqrs shape: (Np, Nq, Nocc, Nocc)
        
        dm1_diag = jnp.diagonal(dm1)[slice_occ] # (Nocc,)
        
        # (pq|rs) * dm_rs -> (pq|oo) * dm_oo -> (pq)
        term1 = 2 * jnp.einsum('pqoo,o->pq', delta_U_pqrs, dm1_diag)
        
        # (ps|rq) term
        ranges_psrq = (slice_p, slice_occ, slice_occ, slice_q)
        delta_U_psrq = self.get_delta_U(jastrow_params, dm1, ranges=ranges_psrq, batch_size=batch_size)
        # delta_U_psrq shape: (Np, Nocc, Nocc, Nq)
        
        # (ps|rq) * dm_sr -> (po|oq) * dm_oo -> (pq)
        term2 = jnp.einsum('pooq,o->pq', delta_U_psrq, dm1_diag)
        
        delta_h = -0.5 * (term1 - term2)
        return delta_h

    def get_1b(self, jastrow_params, dm1=None, block_str=None, ranges=None, orb_block_size=None, batch_size=1000):
        """Get one-body operator correction.

        ``orb_block_size=None`` lets :meth:`get_delta_h` pick adaptively based on
        available GPU memory.
        """
        return self.get_delta_h(jastrow_params, dm1, block_str, ranges, orb_block_size, batch_size)

    def get_2b(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        """Compute two-body integrals correction."""
        start_time = time.perf_counter()
        logger.debug("Starting XTC.get_2b")
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
        
        # Accumulate on host to avoid holding two output-sized GPU tensors.
        # ISDFTC.get_2b already returns via host internally.
        tc_result = super().get_2b(jastrow_params, ranges=ranges)
        result_np = np.array(tc_result)  # writable host copy
        del tc_result
        
        delta_U = self.get_delta_U(jastrow_params, dm1, ranges=ranges, batch_size=batch_size)
        result_np += np.asarray(delta_U)
        del delta_U
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"XTC.get_2b completed in {time.perf_counter() - start_time:.4f} s")
        return jnp.asarray(result_np)

    def get_const(self, jastrow_params, dm1=None, delta_h=None):
        """Compute constant contribution."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
        if delta_h is None:
            delta_h = self.get_delta_h(jastrow_params, dm1)
        logger.debug("Starting XTC.get_const")
        start_time = time.perf_counter()
        const = -2/3 * jnp.einsum('qp,pq->', delta_h, dm1)
        const += self.energy_nuc
        logger.debug(f"XTC.get_const completed in {time.perf_counter() - start_time:.4f} s")
        return const
    
    def _calc_delta_h(self, delta_U, dm1=None):
        """Calculate δh using δU and density matrix."""
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        term1 = 2*jnp.einsum('qpsr,rs->qp', delta_U, dm1)
        term2 = jnp.einsum('spqr,rs->qp', delta_U, dm1)
        delta_h = -0.5 * (term1 - term2)
        return delta_h

    def _get_mf_dm(self):
        """Get mean-field 1-body density matrix for closed shell system."""
        dm1 = jnp.diag(self.mo_occ)/2
        return dm1

    def get_3b(self):
        """Get three-body extended correlation."""
        raise NotImplementedError("JAX implementation pending")
        
    def make_eris(self, mf, jastrow_params):
        """Create ChemistsERIs object for CCSD calculation.
        
        Args:
            mf: PySCF mean-field object (required for initializing RCCSD)
            jastrow_params: Parameters for the Jastrow factor
        """
        from pyscf.cc import rccsd
        mycc = rccsd.RCCSD(mf)
        nocc = np.sum(mf.mo_occ > 0)
        
        eris = rccsd._ChemistsERIs(mycc)
        
        eri_std = tc_helper.get_eri(mf, self.mo_coeff)
        h1e_std = tc_helper.get_hcore(mf, self.mo_coeff)
        
        # Force concrete value computation
        const = np.asarray(self.get_const(jastrow_params))
        h1e_corr = np.asarray(self.get_1b(jastrow_params))
        h2e_corr = np.asarray(self.get_2b(jastrow_params))
        
        h1e = h1e_std + h1e_corr
        h2e = eri_std + h2e_corr
        
        eris.e_core = np.float64(const)
        eris.fock = h1e.copy()
        
        fock_modification = (2 * np.einsum('pqii->pq', h2e[:,:,:nocc,:nocc]) - 
                           np.einsum('piiq->pq', h2e[:,:nocc,:nocc,:]))
        eris.fock += fock_modification
        eris.mo_energy = np.diag(eris.fock).copy()
        
        eris.oooo = h2e[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = h2e[:nocc,nocc:,:nocc,:nocc].copy()
        eris.ooov = h2e[:nocc,:nocc,:nocc,nocc:].copy()
        eris.vooo = h2e[nocc:,:nocc,:nocc,:nocc].copy()
        eris.ovov = h2e[:nocc,nocc:,:nocc,nocc:].copy()
        eris.vovo = h2e[nocc:,:nocc,nocc:,:nocc].copy()
        eris.ovvo = h2e[:nocc,nocc:,nocc:,:nocc].copy()
        eris.voov = h2e[nocc:,:nocc,:nocc,nocc:].copy()
        eris.oovv = h2e[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvv = h2e[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vovv = h2e[nocc:,:nocc,nocc:,nocc:].copy()
        eris.vvov = h2e[nocc:,nocc:,:nocc,nocc:].copy()
        eris.vvvv = h2e[nocc:,nocc:,nocc:,nocc:].copy()

        return eris


@partial(jax.jit, static_argnums=(6, 7))
def _contract_delta_U_kernels_jit(D, X_sliced, phi_p, phi_q, phi_r, phi_s,
                                   rank_block_size=128, include_x=True):
    """JITted version of Delta U contraction.
    
    Args:
        rank_block_size: Block size for scanning the ISDF rank dimension.
            This is a static argument — JAX recompiles if it changes.
    """
    
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]
    N_rank_D = D.shape[0] 
    N_rank_X = X_sliced.shape[2]
    
    # Term 1 & 4: T_D = sum_{a,d} (phi_p*phi_q)_a * D[a,d] * (phi_r*phi_s)_d
    # Scan over blocks of d (second index of D)
    
    padded_rank_D = ((N_rank_D + rank_block_size - 1) // rank_block_size) * rank_block_size
    pad_width_D = padded_rank_D - N_rank_D
    D_padded = jnp.pad(D, ((0, 0), (0, pad_width_D)))
    
    phi_r_padded = jnp.pad(phi_r, ((0, 0), (0, pad_width_D)))
    phi_s_padded = jnp.pad(phi_s, ((0, 0), (0, pad_width_D)))
    
    n_blocks_D = padded_rank_D // rank_block_size
    
    # Reshape for scan
    # D: (N_rank, n_blocks, block) -> (n_blocks, N_rank, block)
    D_scannable = D_padded.reshape(N_rank_D, n_blocks_D, rank_block_size).transpose(1, 0, 2)
    # phi_r/s: (N, n_blocks, block) -> (n_blocks, N, block)
    phi_r_scannable = phi_r_padded.reshape(Nr, n_blocks_D, rank_block_size).transpose(1, 0, 2)
    phi_s_scannable = phi_s_padded.reshape(Ns, n_blocks_D, rank_block_size).transpose(1, 0, 2)

    def scan_d_block(carry, args):
        D_block, phi_r_block, phi_s_block = args
        # D_block: (N_rank_D, block)
        
        # intermediate V[p,q,d_local] = sum_a (phi_p[p,a] * phi_q[q,a]) * D_block[a, d_local]
        # V[p,q,d'] = sum_a phi_p[p,a] * W[a, d', q]
        # W[a, d', q] = phi_q[q,a] * D_block[a, d']
        W = D_block[:, :, None] * phi_q.T[:, None, :] # (a, d', 1) * (a, 1, q) -> (a, d', q)
        W_flat = W.reshape(N_rank_D, rank_block_size * Nq)
        
        V_flat = jnp.matmul(phi_p, W_flat) # (p, a) @ (a, d'q) -> (p, d'q)
        V_block = V_flat.reshape(Np, rank_block_size, Nq)
        V_block = jnp.transpose(V_block, (0, 2, 1)) # (p, q, d')
        
        # C_rs[r, s, d_local]
        C_rs = phi_r_block[:, None, :] * phi_s_block[None, :, :]
        
        contribution = jnp.einsum('pqd,rsd->pqrs', V_block, C_rs)
        return carry + contribution, None

    term_d_init = jnp.zeros((Np, Nq, Nr, Ns))
    term_d, _ = jax.lax.scan(scan_d_block, term_d_init, (D_scannable, phi_r_scannable, phi_s_scannable))

    if not include_x:
        return term_d
    
    # Term 2 & 3: T_X = - sum_c (phi_p*phi_q)_c * X[r,s,c]
    # Scan over blocks of c (rank index of X)
    
    padded_rank_X = ((N_rank_X + rank_block_size - 1) // rank_block_size) * rank_block_size
    pad_width_X = padded_rank_X - N_rank_X
    # X is (Nr, Ns, c)
    X_padded = jnp.pad(X_sliced, ((0,0), (0,0), (0, pad_width_X)))
    
    phi_p_padded = jnp.pad(phi_p, ((0, 0), (0, pad_width_X)))
    phi_q_padded = jnp.pad(phi_q, ((0, 0), (0, pad_width_X)))
    
    n_blocks_X = padded_rank_X // rank_block_size
    
    # Reshape
    # X: (Nr, Ns, n_blocks, block) -> (n_blocks, Nr, Ns, block)
    X_scannable = X_padded.reshape(Nr, Ns, n_blocks_X, rank_block_size).transpose(2, 0, 1, 3)
    phi_p_scannable = phi_p_padded.reshape(Np, n_blocks_X, rank_block_size).transpose(1, 0, 2)
    phi_q_scannable = phi_q_padded.reshape(Nq, n_blocks_X, rank_block_size).transpose(1, 0, 2)
    
    def scan_c_block(carry, args):
        X_block, phi_p_block, phi_q_block = args
        # X_block: (Nr, Ns, block)
        
        # C_pq[p, q, c_local]
        C_pq = phi_p_block[:, None, :] * phi_q_block[None, :, :] # (Np, Nq, block)
        
        # Contract: - sum_c C_pq * X_block
        contribution = -jnp.einsum('pqc,rsc->pqrs', C_pq, X_block)
        return carry + contribution, None

    term_x_init = jnp.zeros((Np, Nq, Nr, Ns))
    term_x, _ = jax.lax.scan(scan_c_block, term_x_init, (X_scannable, phi_p_scannable, phi_q_scannable))

    return term_d + term_x


@jax.jit
def _contract_delta_u_direct_tile_jit(D, X_sliced, phi_p, phi_q, phi_r, phi_s):
    """Balanced direct Delta U tile contraction using fixed-shape matmuls."""
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]

    cpq = (phi_p[:, None, :] * phi_q[None, :, :]).reshape(Np * Nq, D.shape[0])
    crs = (phi_r[:, None, :] * phi_s[None, :, :]).reshape(Nr * Ns, D.shape[1])
    d_term = jnp.matmul(jnp.matmul(cpq, D), crs.T)
    x_flat = X_sliced.reshape(Nr * Ns, X_sliced.shape[2])
    x_term = jnp.matmul(cpq, x_flat.T)
    return (d_term - x_term).reshape(Np, Nq, Nr, Ns)


@jax.jit
def _contract_delta_u_tucker_direct_tile_jit(D, z, phi_p, phi_q, phi_r,
                                             phi_s, u_r, u_s):
    """Balanced direct Delta-U tile with separated X orbital legs.

    This is the solver-facing path for ``X_tucker``.  It never reconstructs
    an ``(r,s,c)`` X panel: the only X-side intermediate is the
    ``(p,q,a,b)`` contracted core.  Consequently the persisted and
    transferred exchange object is ``Z`` rather than a dense X tensor.
    """
    Np, Nq = phi_p.shape[0], phi_q.shape[0]
    Nr, Ns = phi_r.shape[0], phi_s.shape[0]

    cpq = (phi_p[:, None, :] * phi_q[None, :, :]).reshape(Np * Nq, D.shape[0])
    crs = (phi_r[:, None, :] * phi_s[None, :, :]).reshape(Nr * Ns, D.shape[1])
    d_term = jnp.matmul(jnp.matmul(cpq, D), crs.T).reshape(Np, Nq, Nr, Ns)
    return d_term + _contract_tucker_x_residual(phi_p, phi_q, u_r, u_s, z)


@jax.jit
def _delta_h_jk_terms(D, Gb, P_phi, phi_p, phi_q, Y_p, Y_q, wc,
                       J_D_total, J_X, J_X_sym):
    """JIT-compiled J/K algebra for get_delta_h (avoids re-tracing each call)."""
    J_total = J_D_total + J_X + J_X_sym

    DP = D * P_phi
    DP_sym = DP + DP.T
    K_D_total = jnp.linalg.multi_dot([phi_p, DP_sym, phi_q.T])
    K_X_1 = -jnp.dot(phi_p, Y_q.T)
    K_X_2 = -jnp.dot(Y_p, phi_q.T)
    K_total = K_D_total + K_X_1 + K_X_2

    return J_total - 0.5 * K_total


@struct.dataclass
class ISDFXTC(XTC, ISDFTC):
    """JAX implementation of extended transcorrelated methods using ISDF.
    
    Attributes:
        xi_phi: ISDF coefficients for density (N_fused, N_grid)
        xi_grad: ISDF coefficients for gradients (N_fused, N_grid, 3)
        pivots: ISDF pivot indices (N_fused,)
        phi_isdf: ISDF basis for density (Nb, N_fused)
        grad_phi_isdf: ISDF basis for gradients (Nb, N_fused, 3)
    """
    # Fields are inherited from ISDFTC

    @classmethod
    def from_xtc(cls, xtc_obj, n_rank=None, is_incore=False, save_path=None,
                 ls_grid_batch_size=16384, fixed_pivots=None, batch_size=1,
                 candidate_oversampling=2, n_topup=0):
        """Initialize ISDFXTC object from XTC object.

        Args:
            xtc_obj: XTC object
            n_rank: Number of ISDF ranks
            is_incore: Whether to perform in-core decomposition
            save_path: Path to save ISDF kernels
            ls_grid_batch_size: Batch size for grid evaluation in linear solver in ISDF decomposition (default: 16384)
            batch_size: Exact columns retained per blocked pivot round.  One
                preserves exact greedy pivot selection.
            candidate_oversampling: Candidate-pool multiplier for exact
                within-pool re-pivoting.
            n_topup: Final exact-greedy singleton pivots after blocked rounds.
        """
        from . import df
        from .utils import cache_state

        if n_rank is None:
            n_rank = xtc_obj.grid_points.shape[0] // 4

        # Fail-closed guard: if the caller built xtc_obj from a fresh mf whose
        # mo_coeff has a different gauge than the cache, mixing them silently
        # corrupts transcorrelated integrals (SCF gauge non-determinism).
        if save_path is not None and cache_state.cache_has_mf_state(save_path):
            if not cache_state.check_mo_coeff_matches_cache(
                xtc_obj.mo_coeff, save_path
            ):
                raise ValueError(
                    "mo_coeff in the ISDF cache at %s does not match the "
                    "current mo_coeff. This is usually SCF gauge "
                    "non-determinism (LAPACK eigvec sign/subspace mixing). "
                    "Call pytc.utils.cache_state.sync_mf_from_cache(mf, "
                    "save_path) BEFORE XTC.from_pyscf to adopt the cached "
                    "gauge; otherwise expect ~mHa-scale errors in "
                    "transcorrelated results." % save_path
                )

        # Remember whether the cache already contained ISDF kernels *before*
        # we call isdf_decompose — that call will write xi_phi etc. if the
        # cache is empty, so after the call we can no longer distinguish
        # "kernels pre-existed (legacy cache)" from "we just wrote them
        # (fresh compute)".  We need the distinction to decide whether it
        # is safe to persist the current xtc_obj.mo_coeff (below).
        _legacy_kernels_present = (
            save_path is not None
            and cache_state.cache_has_isdf_kernels(save_path)
            and not cache_state.cache_has_mf_state(save_path)
        )

        # Perform ISDF decomposition
        logger.info("ISDFXTC.from_xtc: building ISDF decomposition")
        phi_isdf, xi_phi, grad_phi_isdf, xi_grad, pivots, actual_save_path = df.isdf_decompose(
            xtc_obj.phi, xtc_obj.grad_phi, n_rank, n_rank, weights=xtc_obj.weights,
            is_incore=is_incore, save_path=save_path, grid_batch_size=ls_grid_batch_size,
            fixed_pivots=fixed_pivots, batch_size=batch_size,
            candidate_oversampling=candidate_oversampling, n_topup=n_topup,
        )

        # Persist xtc_obj's mo_coeff / mo_occ so subsequent runs that reuse
        # this cache can lock the orbital gauge via
        # ``pytc.utils.cache_state.sync_mf_from_cache(mf, save_path)``.
        #
        # Only do this on a *fresh* compute.  If the cache already held ISDF
        # kernels but no mo_coeff (legacy cache written before this feature
        # existed), those kernels were built from some *other* mo_coeff
        # gauge; writing the current mo_coeff would silently lock later
        # reloads to the wrong orbitals and, worse, suppress the
        # `check_mo_coeff_matches_cache` warning.  In that case refuse to
        # write and advise the user to regenerate the cache.
        if (
            actual_save_path is not None
            and not cache_state.cache_has_mf_state(actual_save_path)
        ):
            if _legacy_kernels_present:
                logger.warning(
                    "Legacy ISDF cache at %s already has kernels but no "
                    "cached mo_coeff. Not persisting the current mo_coeff "
                    "because those kernels were built from a possibly "
                    "different orbital gauge; auto-saving now would lock "
                    "future reloads to the wrong gauge. Delete the cache "
                    "file to regenerate it with gauge-safe mo_coeff "
                    "persistence enabled.",
                    actual_save_path,
                )
            else:
                try:
                    cache_state.save_orbital_state_to_cache(
                        actual_save_path,
                        mo_coeff=xtc_obj.mo_coeff,
                        mo_occ=xtc_obj.mo_occ,
                    )
                except Exception as exc:  # pragma: no cover — non-fatal diagnostic
                    logger.warning(
                        "Could not persist orbital state to %s: %r",
                        actual_save_path, exc,
                    )

        return cls(
            grid_points=xtc_obj.grid_points,
            weights=xtc_obj.weights,
            phi=xtc_obj.phi,
            grad_phi=xtc_obj.grad_phi,
            n_orb=xtc_obj.n_orb,
            grid_lvl=xtc_obj.grid_lvl,
            jastrow_factor=xtc_obj.jastrow_factor,
            mo_coeff=xtc_obj.mo_coeff,
            mo_occ=xtc_obj.mo_occ,
            nocc=xtc_obj.nocc,
            energy_nuc=xtc_obj.energy_nuc,
            xi_phi=xi_phi,
            xi_grad=xi_grad,
            pivots=pivots,
            phi_isdf=phi_isdf,
            grad_phi_isdf=grad_phi_isdf,
            isdf_kernels=None,
            is_incore=is_incore,
            save_path=actual_save_path
        )

    def isdf(
        self,
        jastrow_params,
        save_path=None,
        batch_size=1000,
        orb_block_size=128,
        host_grid_block_size=None,
        x_s_panel_blocks=1,
        d_reduce_group_blocks=1,
        r2_tile_size=None,
        gpu_budget_bytes=None,
        reuse_aux_kernels=None,
        use_laux_fast_grad=False,
    ):
        """Compute ISDF intermediates and store them.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor.
            save_path: Optional path to save intermediates to HDF5.
            batch_size: Batch size for computation.
            orb_block_size: Block size for orbital batching of X kernel.
            host_grid_block_size: Block size for grid batching on host.
            x_s_panel_blocks: Number of contiguous `s` orbital blocks processed
                together per X-kernel grid pass.
            d_reduce_group_blocks: Group size multiplier for D-kernel grid
                blocking (`D` effective host block = group * host_grid_block_size).
            r2_tile_size: Optional r2-grid tile size for kernel assembly.
            gpu_budget_bytes: Optional device-memory budget in bytes for kernel
                assembly.
            reuse_aux_kernels: Exact K1/K3 recovery from ``L_aux`` and its
                squared-gradient companion.  The parent enables it by default
                whenever the in-core state or a persistent output path supports
                the auxiliary contraction.
            use_laux_fast_grad: Opt into the Jastrow's L_aux-only fast
                derivative path.  The parent validates this mode against any
                reusable base cache.
        """
        logger.info("Computing ISDF intermediates (XTC)...")
        start_time = time.perf_counter()
        
        out_path = save_path if save_path else self.save_path
        
        isdf_tc = super().isdf(jastrow_params, save_path=out_path, batch_size=batch_size,
                               host_grid_block_size=host_grid_block_size,
                               r2_tile_size=r2_tile_size,
                               gpu_budget_bytes=gpu_budget_bytes,
                               reuse_aux_kernels=reuse_aux_kernels,
                               use_laux_fast_grad=use_laux_fast_grad)
        kernels = isdf_tc.isdf_kernels
        requested_laux_mode = _laux_gradient_mode(use_laux_fast_grad)
        
        if out_path and os.path.exists(out_path):
            f = None
            keep_open = False
            try:
                f = h5py.File(out_path, 'r')
                if 'D' in f and 'X' in f:
                    cached_x_mode = f.attrs.get("pytc_xtc_x_mode", "full")
                    if isinstance(cached_x_mode, bytes):
                        cached_x_mode = cached_x_mode.decode()
                    cached_laux_mode = f.attrs.get(
                        "pytc_xtc_laux_gradient_mode", "direct"
                    )
                    if isinstance(cached_laux_mode, bytes):
                        cached_laux_mode = cached_laux_mode.decode()
                    if cached_x_mode != "full":
                        logger.info(
                            "  Delta-U cache X mode is %s, not full; "
                            "recomputing D/X kernels.",
                            cached_x_mode,
                        )
                    elif cached_laux_mode != requested_laux_mode:
                        logger.info(
                            "  Delta-U cache L_aux mode is %s, but this run "
                            "requests %s; recomputing D/X kernels.",
                            cached_laux_mode,
                            requested_laux_mode,
                        )
                    else:
                        logger.info(f"  Found existing D and X in {out_path}. Reading from file...")
                        logger.info(f"  Loading D with shape: {f['D'].shape} on host RAM")
                        kernels['D'] = f['D'][:]
                        if self.is_incore:
                            logger.debug(
                                "incore mode: Loading X with shape: %s on host RAM",
                                f['X'].shape,
                            )
                            kernels['X'] = f['X'][:]
                        else:
                            # Keep the file open only when its datasets escape for
                            # out-of-core streaming.
                            logger.debug(f"out-of-core mode: Streaming X from file. X shape: {f['X'].shape}")
                            kernels['X'] = f['X']
                            # Rank-major twin (panel-contiguous) when the store
                            # carries it; the factorized contraction prefers it.
                            if 'X_rm' in f:
                                logger.debug(f"  rank-major X_rm found, shape: {f['X_rm'].shape}")
                                kernels['X_rm'] = f['X_rm']
                        logger.debug(f"ISDF intermediates (Delta U) loaded from file in {time.perf_counter() - start_time:.4f} s")
                        result = self.replace(
                            isdf_kernels=kernels,
                            save_path=out_path,
                            kmat_kernel_mode=isdf_tc.kmat_kernel_mode,
                        )
                        keep_open = not self.is_incore
                        return result
            except (IOError, KeyError) as e:
                logger.warning(f"  Error reading Delta U kernels from {out_path}: {e}. Recomputing...")
            finally:
                if f is not None and not keep_open:
                    f.close()

        # Pass L_aux to avoid redundant calculation
        delta_u_kernels = self.compute_delta_u_kernels(
            jastrow_params, batch_size, L_aux=kernels.get('L_aux'),
            orb_block_size=orb_block_size,
            save_path=out_path,
            host_grid_block_size=host_grid_block_size,
            x_s_panel_blocks=x_s_panel_blocks,
            d_reduce_group_blocks=d_reduce_group_blocks,
        )
        kernels.update(delta_u_kernels)
        
        # 3. Discard L_aux from ISDFXTC kernels to save RAM and avoid JAX types error
        # L_aux is used to compute D and X, but not needed for get_2b or get_delta_U
        if 'L_aux' in kernels:
            del kernels['L_aux']
        
        if out_path:
            with h5py.File(out_path, 'a') as f:
                f.attrs['pytc_xtc_laux_gradient_mode'] = requested_laux_mode
                if 'phi_isdf' not in f: f.create_dataset('phi_isdf', data=np.array(self.phi_isdf))
                if 'grad_phi_isdf' not in f: f.create_dataset('grad_phi_isdf', data=np.array(self.grad_phi_isdf))
                if 'pivots' not in f: f.create_dataset('pivots', data=np.array(self.pivots))
                
        logger.info(f"ISDF intermediates (Delta U) computed in {time.perf_counter() - start_time:.4f} s")
        
        return self.replace(
            isdf_kernels=kernels,
            save_path=out_path,
            kmat_kernel_mode=isdf_tc.kmat_kernel_mode,
        )

    def compute_delta_u_kernels(
        self,
        jastrow_params,
        batch_size=1000,
        L_aux=None,
        orb_block_size=128,
        save_path=None,
        host_grid_block_size=None,
        x_s_panel_blocks=1,
        d_reduce_group_blocks=1,
    ):
        """Compute D, X kernels for Delta U with orbital and grid batching."""
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)
            
        n_orb = self.n_orb
        n_rank = self.phi_isdf.shape[1]
        dm1 = self._get_mf_dm()
        
        # Precompute Gb and L_Q (low-rank factor of Q) to avoid redundant work
        Gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        
        # Q = phi_dm.T @ phi_isdf has rank <= n_orb
        # Factor as Q = L_Q @ L_Q.T where L_Q has shape (N_rank, n_orb)
        dm1_diag = jnp.diagonal(dm1)
        sqrt_dm1 = jnp.sqrt(jnp.maximum(dm1_diag, 0.0))  # Ensure non-negative
        L_Q = self.phi_isdf.T * sqrt_dm1[None, :]  # (N_rank, n_orb)
        
        logger.info("Computing D kernel...")
        d_start_time = time.perf_counter()
        D = self._compute_D_kernel(
            jastrow_params,
            batch_size,
            L_aux,
            Gb=Gb,
            host_grid_block_size=host_grid_block_size,
            d_reduce_group_blocks=d_reduce_group_blocks,
        )
        logger.info(
            "  compute_delta_u_kernels: D build completed in %.3f s",
            time.perf_counter() - d_start_time,
        )

        logger.info("Computing X kernel...")
        
        if save_path:
            # If L_aux is a dataset from the same file, we must close the read-only handle 
            # and reopen in 'a' mode to write D and X, while keeping L_aux streaming.
            f = None
            if isinstance(L_aux, h5py.Dataset):
                # Check if it's the same file. Use realpath to be safe.
                try:
                    l_aux_path = os.path.abspath(L_aux.file.filename)
                    target_path = os.path.abspath(save_path)
                    if l_aux_path == target_path:
                        logger.info("  L_aux is from target file. Switching handle to read-write for streaming...")
                        ds_name = L_aux.name
                        if L_aux.file: L_aux.file.close()
                        f = h5py.File(save_path, 'a')
                        L_aux = f[ds_name]
                except Exception as e:
                    logger.warning(f"  Could not check L_aux file path: {e}")
            
            if f is None:
                f = h5py.File(save_path, 'a')
            if 'D' in f: del f['D']
            f.create_dataset('D', data=np.array(D))
            if 'X' in f: del f['X']
            if 'X_rm' in f: del f['X_rm']  # stale twin once X is rewritten
            X = f.create_dataset('X', (n_orb, n_orb, n_rank), dtype='f8')
            f.attrs['pytc_xtc_x_mode'] = 'full'
        else:
            X = np.zeros((n_orb, n_orb, n_rank), dtype='f8')
            
        x_s_panel_blocks = max(1, int(x_s_panel_blocks))
        s_panel_span = max(1, orb_block_size) * x_s_panel_blocks

        # Exploit symmetry: X[r,s,a] = X[s,r,a], only compute upper triangle blocks
        x_start_time = time.perf_counter()
        for r0 in range(0, n_orb, orb_block_size):
            r1 = min(r0 + orb_block_size, n_orb)
            logger.info(f"  compute_delta_u_kernels: Computing X blocks for r-range [{r0}:{r1}]...")
            for s_panel0 in range(r0, n_orb, s_panel_span):
                s_panel1 = min(s_panel0 + s_panel_span, n_orb)
                logger.debug(
                    "  compute_delta_u_kernels: X panel r=[%d:%d], s=[%d:%d], panel_blocks=%d",
                    r0, r1, s_panel0, s_panel1, x_s_panel_blocks,
                )
                ranges = (
                    slice(None),
                    slice(None),
                    slice(r0, r1),
                    slice(s_panel0, s_panel1),
                )
                X_panel = self._compute_X_kernel(
                    jastrow_params,
                    ranges,
                    batch_size,
                    L_aux,
                    Gb=Gb,
                    L_Q=L_Q,
                    host_grid_block_size=host_grid_block_size,
                )
                X_panel_np = np.asarray(X_panel)

                for s0 in range(s_panel0, s_panel1, orb_block_size):
                    s1 = min(s0 + orb_block_size, s_panel1)
                    off0 = s0 - s_panel0
                    off1 = s1 - s_panel0
                    X_block_np = X_panel_np[:, off0:off1, :]
                    X[r0:r1, s0:s1, :] = X_block_np
                    if r0 != s0:
                        X[s0:s1, r0:r1, :] = X_block_np.transpose(1, 0, 2)

                del X_panel, X_panel_np
                gc.collect()

        logger.info(
            "  compute_delta_u_kernels: X build completed in %.3f s",
            time.perf_counter() - x_start_time,
        )
                
        if save_path:
            # Return dataset object for X to allow streaming
            return {'D': f['D'][:], 'X': X}
        else:
            return {'D': D, 'X': X}

    def select_tucker_x_orbital_basis(
        self,
        jastrow_params,
        n_factor,
        *,
        oversampling=8,
        seed=0,
        batch_size=1000,
        L_aux=None,
        orb_block_size=128,
        host_grid_block_size=None,
    ):
        """Select an orbital Tucker basis from a streamed X sketch.

        This avoids constructing or writing a global ``X[norb,norb,R]``
        tensor.  A Khatri--Rao random test matrix is applied to each X panel
        as it is produced, yielding only an ``norb x (n_factor+oversampling)``
        sketch.  QR of that sketch supplies the common orbital basis U.

        The approximation is controlled by ``n_factor``.  Requesting the
        complete orbital dimension returns a full orthogonal basis, for which
        a subsequently built Tucker core is an exact representation of X.
        """
        n_orb = int(self.n_orb)
        n_factor = int(n_factor)
        oversampling = int(oversampling)
        n_probe = min(n_orb, n_factor + oversampling)
        orb_block_size = int(orb_block_size)
        seed = int(seed)

        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)

        dm1 = self._get_mf_dm()
        gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        sqrt_dm1 = jnp.sqrt(jnp.maximum(jnp.diagonal(dm1), 0.0))
        l_q = self.phi_isdf.T * sqrt_dm1[None, :]
        n_rank = int(self.phi_isdf.shape[1])

        rng = np.random.default_rng(seed)
        # A separable random map over (s,c) avoids an norb*R*n_probe
        # allocation while still sampling both right-hand X indices.
        omega_s = rng.standard_normal((n_orb, n_probe)) / np.sqrt(n_probe)
        omega_c = rng.standard_normal((n_rank, n_probe))
        sketch = np.zeros((n_orb, n_probe), dtype=np.float64)

        logger.info(
            "Selecting Tucker-X orbital basis: factors=%d probes=%d; "
            "streaming fused r panels of %d rows",
            n_factor, n_probe, orb_block_size,
        )
        for r0 in range(0, n_orb, orb_block_size):
            r1 = min(r0 + orb_block_size, n_orb)
            # Apply the random map while the X contributions are still in
            # the grid kernel.  Materializing a (r, s, rank) panel first
            # would repeat virtually all of a dense-X build merely to reduce
            # it to the r-by-probe sketch on the host.
            panel_sketch = self._compute_X_sketch(
                jastrow_params,
                (slice(None), slice(None), slice(r0, r1), slice(None)),
                batch_size,
                L_aux,
                Gb=gb,
                L_Q=l_q,
                omega_s=omega_s,
                omega_c=omega_c,
                host_grid_block_size=host_grid_block_size,
            )
            panel_sketch = np.asarray(panel_sketch, dtype=np.float64)
            expected_shape = (r1 - r0, n_probe)
            if panel_sketch.shape != expected_shape:
                raise ValueError(
                    "X sketch panel has the wrong shape: "
                    f"expected {expected_shape}, got {panel_sketch.shape}"
                )
            sketch[r0:r1] = panel_sketch
            del panel_sketch
            gc.collect()

        basis, _ = np.linalg.qr(sketch, mode='reduced')
        return basis[:, :n_factor]

    def compute_tucker_x_core(
        self,
        jastrow_params,
        orbital_basis,
        *,
        batch_size=1000,
        L_aux=None,
        host_grid_block_size=None,
    ):
        """Build ``Z = U.T X U`` directly, without materializing dense X.

        ``orbital_basis`` must have orthonormal columns in the original MO
        row space.  The returned core has shape ``(M, M, R)`` and pairs with
        U through ``X[r,s,c] ~= U[r,a] Z[a,b,c] U[s,b]``.
        """
        u = np.asarray(orbital_basis, dtype=np.float64)
        n_orb = int(self.n_orb)
        if u.ndim != 2 or u.shape[0] != n_orb or not u.shape[1]:
            raise ValueError(
                f"orbital_basis must have shape ({n_orb}, M), got {u.shape}"
            )
        gram_error = np.max(np.abs(u.T @ u - np.eye(u.shape[1])))
        if gram_error > 1e-10:
            raise ValueError(
                "orbital_basis columns must be orthonormal; "
                f"maximum Gram-matrix error is {gram_error:.3e}"
            )
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)

        dm1 = self._get_mf_dm()
        gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        sqrt_dm1 = jnp.sqrt(jnp.maximum(jnp.diagonal(dm1), 0.0))
        l_q = self.phi_isdf.T * sqrt_dm1[None, :]
        projected_rows = jnp.matmul(jnp.asarray(u.T), self.phi_isdf)
        n_factor = u.shape[1]
        core = self._compute_X_kernel(
            jastrow_params,
            (slice(None), slice(None), slice(0, n_factor), slice(0, n_factor)),
            batch_size,
            L_aux,
            Gb=gb,
            L_Q=l_q,
            host_grid_block_size=host_grid_block_size,
            orbital_rows=projected_rows,
        )
        core = np.asarray(core, dtype=np.float64)
        expected_shape = (n_factor, n_factor, int(self.phi_isdf.shape[1]))
        if core.shape != expected_shape:
            raise ValueError(
                "Tucker-X core has the wrong shape: "
                f"expected {expected_shape}, got {core.shape}"
            )
        return {'U': u, 'Z': core}

    def build_tucker_x_kernels_direct(
        self,
        jastrow_params,
        n_factor,
        *,
        oversampling=8,
        seed=0,
        batch_size=1000,
        orb_block_size=128,
        host_grid_block_size=None,
        save_path=None,
        d_reduce_group_blocks=1,
        r2_tile_size=None,
        gpu_budget_bytes=None,
        reuse_aux_kernels=None,
        use_laux_fast_grad=False,
    ):
        """Build a factor-only X view without materializing dense ``X``.

        This path first prepares only the shared TC intermediates
        ``K1_kernel``, ``K3_kernel``, and ``L_aux``.  It then builds ``D`` and
        the Tucker orbital basis/core directly from ``L_aux``.  The returned
        object carries ``D`` and ``X_tucker`` but deliberately has no dense
        ``X`` key, making it suitable for source-free construction studies and
        production factor-only calculations.

        When ``save_path`` is supplied, the reusable base intermediates may be
        cached there by :class:`ISDFTC`; no dense exchange dataset is created.
        The accepted L_aux construction controls are forwarded unchanged.
        """
        if save_path and os.path.exists(save_path):
            with h5py.File(save_path, 'r') as handle:
                dense_keys = sorted({'X', 'X_rm'}.intersection(handle.keys()))
            if dense_keys:
                raise ValueError(
                    "Factor-only Tucker construction refuses a cache containing "
                    f"dense exchange data: {dense_keys}"
                )
        base = ISDFTC.isdf(
            self,
            jastrow_params,
            save_path=save_path,
            batch_size=batch_size,
            host_grid_block_size=host_grid_block_size,
            r2_tile_size=r2_tile_size,
            gpu_budget_bytes=gpu_budget_bytes,
            reuse_aux_kernels=reuse_aux_kernels,
            use_laux_fast_grad=use_laux_fast_grad,
        )
        kernels = dict(base.isdf_kernels)
        l_aux = kernels.pop("L_aux")
        d_kernel = base._compute_D_kernel(
            jastrow_params,
            batch_size,
            L_aux=l_aux,
            host_grid_block_size=host_grid_block_size,
            d_reduce_group_blocks=d_reduce_group_blocks,
        )
        orbital_basis = base.select_tucker_x_orbital_basis(
            jastrow_params,
            n_factor,
            oversampling=oversampling,
            seed=seed,
            batch_size=batch_size,
            L_aux=l_aux,
            orb_block_size=orb_block_size,
            host_grid_block_size=host_grid_block_size,
        )
        factors = base.compute_tucker_x_core(
            jastrow_params,
            orbital_basis,
            batch_size=batch_size,
            L_aux=l_aux,
            host_grid_block_size=host_grid_block_size,
        )
        kernels["D"] = np.asarray(d_kernel)
        kernels["X_tucker"] = factors
        return base.replace(isdf_kernels=kernels)

    def _iter_sharded_delta_u_blocks(
        self,
        L_aux,
        xi_phi_source,
        host_grid_block_size,
        block_padded,
        n_rank,
        devices,
        grid_sharding,
        weights_sharding,
        g_sharding,
        xi_sharding,
    ):
        """Yield prefetch-staged and sharded grid blocks for D/X kernels."""
        from pytc.utils.prefetch import PrefetchIterator, safe_hdf5_read

        n_devices = len(devices)
        n_grid = self.grid_points.shape[0]
        n_per_dev = block_padded // n_devices
        block_keys = [
            (g0, min(g0 + host_grid_block_size, n_grid))
            for g0 in range(0, n_grid, host_grid_block_size)
        ]

        def _load_block(key):
            g0_loc, g1_loc = key
            t_host_start = time.perf_counter()

            gb = np.asarray(self.grid_points[g0_loc:g1_loc])
            wb = np.asarray(self.weights[g0_loc:g1_loc])
            G_block = -safe_hdf5_read(L_aux, (slice(None), slice(g0_loc, g1_loc), slice(None)))
            xi_block = safe_hdf5_read(
                xi_phi_source,
                (slice(None), slice(g0_loc, g1_loc)),
            )

            cur_len = g1_loc - g0_loc
            if cur_len < block_padded:
                pad = block_padded - cur_len
                gb = np.pad(gb, ((0, pad), (0, 0)))
                wb = np.pad(wb, ((0, pad),))
                G_block = np.pad(G_block, ((0, 0), (0, pad), (0, 0)))
                xi_block = np.pad(xi_block, ((0, 0), (0, pad)))

            t_host = time.perf_counter() - t_host_start
            return g0_loc, g1_loc, gb, wb, G_block, xi_block, t_host

        with PrefetchIterator(block_keys, _load_block, prefetch_depth=1) as block_iter:
            for _, loaded in block_iter:
                g0_loc, g1_loc, gb, wb, G_block, xi_block, t_host = loaded
                t_h2d_start = time.perf_counter()
                grid_parts = []
                weight_parts = []
                g_parts = []
                xi_parts = []
                for d in range(n_devices):
                    s = d * n_per_dev
                    e = (d + 1) * n_per_dev
                    grid_parts.append(jax.device_put(gb[s:e], devices[d]))
                    weight_parts.append(jax.device_put(wb[s:e], devices[d]))
                    g_parts.append(jax.device_put(G_block[:, s:e, :], devices[d]))
                    xi_parts.append(jax.device_put(xi_block[:, s:e], devices[d]))

                s_grid = jax.make_array_from_single_device_arrays(
                    (block_padded, 3), grid_sharding, grid_parts
                )
                s_weights = jax.make_array_from_single_device_arrays(
                    (block_padded,), weights_sharding, weight_parts
                )
                s_G = jax.make_array_from_single_device_arrays(
                    (n_rank, block_padded, 3), g_sharding, g_parts
                )
                s_xi = jax.make_array_from_single_device_arrays(
                    (n_rank, block_padded), xi_sharding, xi_parts
                )
                t_h2d = time.perf_counter() - t_h2d_start
                yield g0_loc, g1_loc, s_grid, s_weights, s_G, s_xi, t_host, t_h2d

    def _compute_D_kernel(
        self,
        jastrow_params,
        batch_size=1024,
        L_aux=None,
        Gb=None,
        host_grid_block_size=None,
        d_reduce_group_blocks=1,
    ):
        """Compute D kernel for Delta U with grid-blocking to save host RAM."""
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)
            
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        dm1 = self._get_mf_dm()
        
        if host_grid_block_size is None:
            host_grid_block_size = n_grid
        d_reduce_group_blocks = max(1, int(d_reduce_group_blocks))
        d_host_grid_block_size = host_grid_block_size * d_reduce_group_blocks
        if d_reduce_group_blocks > 1:
            logger.debug(
                "_compute_D_kernel: using grouped D block size=%d (base=%d, group=%d)",
                d_host_grid_block_size,
                host_grid_block_size,
                d_reduce_group_blocks,
            )
            
        if Gb is None:
            Gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        phi_isdf = self.phi_isdf
        n_orb = self.n_orb
        
        D = np.zeros((n_rank, n_rank))
        
        xi_phi_ds = None
        f_xi = None
        if self.xi_phi is None and self.save_path:
            f_xi = h5py.File(self.save_path, 'r')
            xi_phi_ds = f_xi['xi_phi']

        mesh = sharding_core.create_1d_mesh(devices=devices, axis_name='devices')
        rep_sharding = sharding_core.get_replicated_sharding(mesh)
        grid_sharding = NamedSharding(mesh, P('devices', None))
        weights_sharding = NamedSharding(mesh, P('devices'))
        xi_sharding = NamedSharding(mesh, P(None, 'devices'))
        g_sharding = NamedSharding(mesh, P(None, 'devices', None))

        @shard_map(
            mesh=mesh,
            in_specs=(P('devices', None), P('devices'), P(None, 'devices'), P(None, 'devices', None), P()),
            out_specs=P(),
            check_vma=False,
        )
        def sharded_D(grid_shard, weights_shard, xi_shard, G_shard, params):
            d_local = self._calc_D_shard(
                params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, phi_isdf, None, n_orb, batch_size
            )
            return jax.lax.psum(d_local, 'devices')

        params_rep = jax.tree_util.tree_map(lambda x: jax.device_put(np.asarray(x), rep_sharding), jastrow_params)
        block_padded = ((d_host_grid_block_size + n_devices - 1) // n_devices) * n_devices

        n_blocks = (n_grid + d_host_grid_block_size - 1) // d_host_grid_block_size
        block_input_bytes = (
            block_padded * 3 * 8 +             # grid
            block_padded * 8 +                 # weights
            n_rank * block_padded * 3 * 8 +    # G
            n_rank * block_padded * 8          # xi
        )
        d_reduce_bytes = n_rank * n_rank * 8
        t_prepare_host = 0.0
        t_h2d = 0.0
        t_shard_compute = 0.0
        t_host_accumulate = 0.0
        t_kernel_start = time.perf_counter()

        try:
            logger.debug(
                f"  _compute_D_kernel: Starting shard_map "
                f"(n_rank={n_rank}, n_grid={n_grid}, n_devices={n_devices})..."
            )
            xi_phi_source = self.xi_phi if self.xi_phi is not None else xi_phi_ds
            for g0, g1, sharded_grid, sharded_weights, sharded_G, sharded_xi_phi, t_host_blk, t_h2d_blk in self._iter_sharded_delta_u_blocks(
                L_aux=L_aux,
                xi_phi_source=xi_phi_source,
                host_grid_block_size=d_host_grid_block_size,
                block_padded=block_padded,
                n_rank=n_rank,
                devices=devices,
                grid_sharding=grid_sharding,
                weights_sharding=weights_sharding,
                g_sharding=g_sharding,
                xi_sharding=xi_sharding,
            ):
                logger.debug(f"_compute_D_kernel: Processing grid block [{g0}:{g1}]...")
                t_prepare_host += t_host_blk
                t_h2d += t_h2d_blk
                t_compute_start = time.perf_counter()
                D_rep = sharded_D(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, params_rep)
                t_shard_compute += time.perf_counter() - t_compute_start
                t_accum_start = time.perf_counter()
                D += np.asarray(D_rep)
                t_host_accumulate += time.perf_counter() - t_accum_start
                
                del sharded_G, sharded_xi_phi, sharded_grid, sharded_weights, D_rep
                gc.collect()
                
        finally:
            if f_xi: f_xi.close()
        t_total = time.perf_counter() - t_kernel_start
        logger.debug(
            "  _compute_D_kernel profile: blocks=%d, input_block=%.2f MiB, reduce_block=%.2f MiB, "
            "prepare_host=%.4f s, host_to_device=%.4f s, shard_compute=%.4f s, host_accumulate=%.4f s, total=%.4f s",
            n_blocks,
            block_input_bytes / (1024.0 ** 2),
            d_reduce_bytes / (1024.0 ** 2),
            t_prepare_host,
            t_h2d,
            t_shard_compute,
            t_host_accumulate,
            t_total,
        )
            
        return D

    def _compute_X_sketch(self, jastrow_params, ranges, batch_size=1024,
                          L_aux=None, Gb=None, L_Q=None,
                          omega_s=None, omega_c=None,
                          host_grid_block_size=None):
        """Apply a separable random map to X without materializing X.

        Given ``omega_s[s,k]`` and ``omega_c[c,k]``, return

        ``S[r,k] = sum_{s,c} X[r,s,c] omega_s[s,k] omega_c[c,k]``.

        This is the same Khatri--Rao range-finder sketch used by
        :meth:`select_tucker_x_orbital_basis`, but the s/c projection is
        associated into the grid kernel.  In particular, no ``(r,s,rank)``
        exchange panel is allocated, copied to the host, or contracted after
        the fact.
        """
        if omega_s is None or omega_c is None:
            raise ValueError("omega_s and omega_c are required for an X sketch")
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)

        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        n_orb = self.phi_isdf.shape[0]
        omega_s = np.asarray(omega_s, dtype=np.float64)
        omega_c = np.asarray(omega_c, dtype=np.float64)
        if omega_s.ndim != 2 or omega_c.ndim != 2:
            raise ValueError("omega_s and omega_c must both be rank-2")
        if omega_s.shape[0] != n_orb or omega_c.shape[0] != n_rank:
            raise ValueError(
                "X sketch probe dimensions disagree with orbital/rank dimensions: "
                f"omega_s={omega_s.shape}, omega_c={omega_c.shape}, "
                f"expected ({n_orb}, K) and ({n_rank}, K)"
            )
        if omega_s.shape[1] != omega_c.shape[1]:
            raise ValueError("omega_s and omega_c must be rank-2 with equal probe counts")

        dm1 = self._get_mf_dm()
        if Gb is None:
            Gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        if L_Q is None:
            sqrt_dm1 = jnp.sqrt(jnp.maximum(jnp.diagonal(dm1), 0.0))
            L_Q = self.phi_isdf.T * sqrt_dm1[None, :]

        slice_p, slice_q, slice_r, slice_s = ranges
        del slice_p, slice_q
        nr = self.phi_isdf[slice_r].shape[0]
        omega_s_panel = omega_s[slice_s]
        n_probe = omega_s.shape[1]

        xi_phi_ds = None
        f_xi = None
        if self.xi_phi is None and self.save_path:
            f_xi = h5py.File(self.save_path, 'r')
            xi_phi_ds = f_xi['xi_phi']

        mesh = sharding_core.create_1d_mesh(devices=devices, axis_name='devices')
        rep_sharding = sharding_core.get_replicated_sharding(mesh)
        grid_sharding = NamedSharding(mesh, P('devices', None))
        weights_sharding = NamedSharding(mesh, P('devices'))
        xi_sharding = NamedSharding(mesh, P(None, 'devices'))
        g_sharding = NamedSharding(mesh, P(None, 'devices', None))

        @shard_map(
            mesh=mesh,
            in_specs=(P('devices', None), P('devices'), P(None, 'devices'),
                      P(None, 'devices', None), P(), P(), P(), P()),
            out_specs=P(),
            check_vma=False,
        )
        def sharded_X_sketch(grid_shard, weights_shard, xi_shard, G_shard,
                             params, omega_s_rep, omega_c_rep, l_q_rep):
            x_local = self._calc_X_sketch_shard(
                params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, self.phi_isdf, ranges, n_orb, batch_size, l_q_rep,
                omega_s_rep, omega_c_rep,
            )
            return jax.lax.psum(x_local, 'devices')

        params_rep = jax.tree_util.tree_map(
            lambda x: jax.device_put(np.asarray(x), rep_sharding), jastrow_params
        )
        omega_s_rep = jax.device_put(omega_s_panel, rep_sharding)
        omega_c_rep = jax.device_put(omega_c, rep_sharding)
        l_q_rep = jax.device_put(np.asarray(L_Q), rep_sharding)
        if host_grid_block_size is None:
            host_grid_block_size = n_grid
        block_padded = (
            (host_grid_block_size + n_devices - 1) // n_devices
        ) * n_devices

        sketch = np.zeros((nr, n_probe), dtype=np.float64)
        t_prepare_host = t_h2d = t_shard_compute = t_host_accumulate = 0.0
        t_kernel_start = time.perf_counter()
        try:
            xi_phi_source = self.xi_phi if self.xi_phi is not None else xi_phi_ds
            for _g0, _g1, sharded_grid, sharded_weights, sharded_G, sharded_xi_phi, t_host_blk, t_h2d_blk in self._iter_sharded_delta_u_blocks(
                L_aux=L_aux,
                xi_phi_source=xi_phi_source,
                host_grid_block_size=host_grid_block_size,
                block_padded=block_padded,
                n_rank=n_rank,
                devices=devices,
                grid_sharding=grid_sharding,
                weights_sharding=weights_sharding,
                g_sharding=g_sharding,
                xi_sharding=xi_sharding,
            ):
                t_prepare_host += t_host_blk
                t_h2d += t_h2d_blk
                t_compute_start = time.perf_counter()
                sketch_rep = sharded_X_sketch(
                    sharded_grid, sharded_weights, sharded_xi_phi, sharded_G,
                    params_rep, omega_s_rep, omega_c_rep, l_q_rep,
                )
                t_shard_compute += time.perf_counter() - t_compute_start
                t_accum_start = time.perf_counter()
                sketch += np.asarray(sketch_rep)
                t_host_accumulate += time.perf_counter() - t_accum_start
                del sharded_G, sharded_xi_phi, sharded_grid, sharded_weights, sketch_rep
                gc.collect()
        finally:
            if f_xi:
                f_xi.close()

        logger.debug(
            "  _compute_X_sketch profile: prepare_host=%.4f s, host_to_device=%.4f s, "
            "shard_compute=%.4f s, host_accumulate=%.4f s, total=%.4f s",
            t_prepare_host, t_h2d, t_shard_compute, t_host_accumulate,
            time.perf_counter() - t_kernel_start,
        )
        return sketch

    def _compute_X_kernel(self, jastrow_params, ranges, batch_size=1024,
                          L_aux=None, Gb=None, L_Q=None,
                          host_grid_block_size=None, orbital_rows=None):
        """Compute X kernel for Delta U for a specific orbital range with grid-blocking.
        
        Uses low-rank factorization: Q = L_Q @ L_Q.T where L_Q has shape (N_rank, n_orb).
        This reduces the expensive O(batch × Nr × N_rank²) matmuls to O(batch × Nr × N_rank × n_orb).
        """
        if L_aux is None:
            L_aux = self._compute_L_aux(jastrow_params, batch_size)
            
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        dm1 = self._get_mf_dm()
        
        if host_grid_block_size is None:
            host_grid_block_size = n_grid
            
        if Gb is None:
            Gb = jnp.einsum('ub,sb,us->b', self.phi_isdf, self.phi_isdf, dm1)
        if L_Q is None:
            # Compute L_Q if not provided
            dm1 = self._get_mf_dm()
            dm1_diag = jnp.diagonal(dm1)
            sqrt_dm1 = jnp.sqrt(jnp.maximum(dm1_diag, 0.0))
            L_Q = self.phi_isdf.T * sqrt_dm1[None, :]
            
        # ``orbital_rows`` lets a caller form U.T @ X @ U directly.  The
        # grid/kernel algebra remains unchanged because X is bilinear in the
        # two orbital rows; only the rows supplied to _calc_X_shard change.
        # The density factor L_Q intentionally remains in the original MO
        # basis, since it represents the reference density rather than an X
        # output index.
        phi_isdf = self.phi_isdf if orbital_rows is None else orbital_rows
        n_orb = phi_isdf.shape[0]
        
        slice_p, slice_q, slice_r, slice_s = ranges
        Nr = phi_isdf[slice_r].shape[0]
        Ns = phi_isdf[slice_s].shape[0]
        X = np.zeros((Nr, Ns, n_rank))
        
        xi_phi_ds = None
        f_xi = None
        if self.xi_phi is None and self.save_path:
            f_xi = h5py.File(self.save_path, 'r')
            xi_phi_ds = f_xi['xi_phi']

        mesh = sharding_core.create_1d_mesh(devices=devices, axis_name='devices')
        rep_sharding = sharding_core.get_replicated_sharding(mesh)
        grid_sharding = NamedSharding(mesh, P('devices', None))
        weights_sharding = NamedSharding(mesh, P('devices'))
        xi_sharding = NamedSharding(mesh, P(None, 'devices'))
        g_sharding = NamedSharding(mesh, P(None, 'devices', None))

        @shard_map(
            mesh=mesh,
            in_specs=(P('devices', None), P('devices'), P(None, 'devices'), P(None, 'devices', None), P()),
            out_specs=P(),
            check_vma=False,
        )
        def sharded_X(grid_shard, weights_shard, xi_shard, G_shard, params):
            x_local = self._calc_X_shard(
                params, dm1, grid_shard, weights_shard, xi_shard, G_shard,
                Gb, phi_isdf, ranges, n_orb, batch_size, L_Q
            )
            return jax.lax.psum(x_local, 'devices')

        params_rep = jax.tree_util.tree_map(lambda x: jax.device_put(np.asarray(x), rep_sharding), jastrow_params)
        block_padded = ((host_grid_block_size + n_devices - 1) // n_devices) * n_devices

        n_blocks = (n_grid + host_grid_block_size - 1) // host_grid_block_size
        block_input_bytes = (
            block_padded * 3 * 8 +             # grid
            block_padded * 8 +                 # weights
            n_rank * block_padded * 3 * 8 +    # G
            n_rank * block_padded * 8          # xi
        )
        x_reduce_bytes = Nr * Ns * n_rank * 8
        t_prepare_host = 0.0
        t_h2d = 0.0
        t_shard_compute = 0.0
        t_host_accumulate = 0.0
        t_kernel_start = time.perf_counter()

        try:
            logger.debug(
                f"  _compute_X_kernel: Starting shard_map "
                f"(n_rank={n_rank}, n_grid={n_grid}, n_devices={n_devices})..."
            )
            xi_phi_source = self.xi_phi if self.xi_phi is not None else xi_phi_ds
            for g0, g1, sharded_grid, sharded_weights, sharded_G, sharded_xi_phi, t_host_blk, t_h2d_blk in self._iter_sharded_delta_u_blocks(
                L_aux=L_aux,
                xi_phi_source=xi_phi_source,
                host_grid_block_size=host_grid_block_size,
                block_padded=block_padded,
                n_rank=n_rank,
                devices=devices,
                grid_sharding=grid_sharding,
                weights_sharding=weights_sharding,
                g_sharding=g_sharding,
                xi_sharding=xi_sharding,
            ):
                logger.debug(f"_compute_X_kernel: Processing grid block [{g0}:{g1}]...")
                t_prepare_host += t_host_blk
                t_h2d += t_h2d_blk
                t_compute_start = time.perf_counter()
                X_rep = sharded_X(sharded_grid, sharded_weights, sharded_xi_phi, sharded_G, params_rep)
                t_shard_compute += time.perf_counter() - t_compute_start
                t_accum_start = time.perf_counter()
                X += np.asarray(X_rep)
                t_host_accumulate += time.perf_counter() - t_accum_start
                
                del sharded_G, sharded_xi_phi, sharded_grid, sharded_weights, X_rep
                gc.collect()
                
        finally:
            if f_xi: f_xi.close()
        t_total = time.perf_counter() - t_kernel_start
        logger.debug(
            "  _compute_X_kernel profile: blocks=%d, input_block=%.2f MiB, reduce_block=%.2f MiB, "
            "prepare_host=%.4f s, host_to_device=%.4f s, shard_compute=%.4f s, host_accumulate=%.4f s, total=%.4f s",
            n_blocks,
            block_input_bytes / (1024.0 ** 2),
            x_reduce_bytes / (1024.0 ** 2),
            t_prepare_host,
            t_h2d,
            t_shard_compute,
            t_host_accumulate,
            t_total,
        )
            
        return X


    def _calc_D_shard(self, jastrow_params, dm1, grid_points, weights, xi_phi, G_shard,
                      Gb, phi, ranges, n_orb, batch_size=1024):
        """Calculate D kernel for a shard."""
        N_rank = phi.shape[1]
        N_shard = grid_points.shape[0]
        
        padded_size = ((N_shard + batch_size - 1) // batch_size) * batch_size
        weights_padded = jnp.pad(weights, (0, padded_size - N_shard))
        xi_padded = jnp.pad(xi_phi, ((0, 0), (0, padded_size - N_shard)))
        G_padded = jnp.pad(G_shard, ((0, 0), (0, padded_size - N_shard), (0, 0)))
        
        n_batches = padded_size // batch_size

        def scan_D(D_acc, i_batch):
            w_batch = jax.lax.dynamic_slice(weights_padded, (i_batch * batch_size,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(xi_padded, (0, i_batch * batch_size), (N_rank, batch_size))
            G_batch = jax.lax.dynamic_slice(G_padded, (0, i_batch * batch_size, 0), (N_rank, batch_size, 3))
            
            G_flat = G_batch.reshape(N_rank, -1)
            
            H = jnp.einsum('b,bik->ik', Gb, G_batch)
            V = jnp.einsum('ik,dik->di', H, G_batch)
            # D1_update = einsum('i,ai,di->ad') but use matmul to avoid large intermediate
            # (xi * w).T @ V.T = (N_rank, batch) @ (batch, N_rank) -> (N_rank, N_rank)
            xi_w = xi_batch * w_batch[None, :]  # (N_rank, batch)
            D1_update = jnp.matmul(xi_w, V.T)
            
            w_tilde = w_batch * jnp.einsum('b,bi->i', Gb, xi_batch)
            G_weighted = G_batch * w_tilde[None, :, None]
            G_weighted_flat = G_weighted.reshape(N_rank, -1)
            D4_update = jnp.dot(G_weighted_flat, G_flat.T)
            
            return D_acc + 2 * D1_update + D4_update, None
        
        D_final, _ = jax.lax.scan(scan_D, jnp.zeros((N_rank, N_rank)), jnp.arange(n_batches))
        return D_final

    def _calc_X_shard(self, jastrow_params, dm1, grid_points, weights, xi_phi, G_shard,
                      Gb, phi, ranges, n_orb, batch_size=1024, L_Q=None):
        """Calculate X kernel for a shard (merged X2, X3_1, X3_2).
        
        Uses low-rank factorization: Q = L_Q @ L_Q.T where L_Q has shape (N_rank, n_orb).
        Instead of: einsum('bra,ac->brc', X, Q) which is O(batch × Nr × N_rank²)
        We compute: (X @ L_Q) @ L_Q.T which is O(batch × Nr × N_rank × n_orb) - ~9x faster!
        """
        N_rank = phi.shape[1]
        N_shard = grid_points.shape[0]
        
        slice_p, slice_q, slice_r, slice_s = ranges
        phi_r = phi[slice_r]
        phi_s = phi[slice_s]
        Nr = phi_r.shape[0]
        Ns = phi_s.shape[0]
        
        padded_size = ((N_shard + batch_size - 1) // batch_size) * batch_size
        weights_padded = jnp.pad(weights, (0, padded_size - N_shard))
        xi_padded = jnp.pad(xi_phi, ((0, 0), (0, padded_size - N_shard)))
        G_padded = jnp.pad(G_shard, ((0, 0), (0, padded_size - N_shard), (0, 0)))
        
        n_batches = padded_size // batch_size
        
        # Helper: compute P = (phi * vec_T) @ L_Q using chunking over Rank to avoid OOM
        def compute_proj(phi, vec_T, L_Q, chunk_size=2048):
            """Compute P[b,r,o] = sum_a phi[r,a] * vec_T[b,a] * L_Q[a,o]"""
            Ns = phi.shape[0]
            Nb = vec_T.shape[0]
            No = L_Q.shape[1]
            Na = L_Q.shape[0]
            
            num_chunks = (Na + chunk_size - 1) // chunk_size
            
            pad_len = num_chunks * chunk_size - Na
            if pad_len > 0:
                phi_p = jnp.pad(phi, ((0,0), (0, pad_len)))
                vec_p = jnp.pad(vec_T, ((0,0), (0, pad_len)))
                lq_p = jnp.pad(L_Q, ((0, pad_len), (0,0)))
            else:
                phi_p, vec_p, lq_p = phi, vec_T, L_Q
                
            def body_fn(carry, i):
                start = i * chunk_size
                p_c = jax.lax.dynamic_slice(phi_p, (0, start), (Ns, chunk_size))
                v_c = jax.lax.dynamic_slice(vec_p, (0, start), (Nb, chunk_size))
                l_c = jax.lax.dynamic_slice(lq_p, (start, 0), (chunk_size, No))
                
                # (batch, Nr, chunk) * (chunk, n_orb) -> (batch, Nr, n_orb)
                # v_c[:, None, :] broadcasts to (batch, 1, chunk)
                # p_c[None, :, :] broadcasts to (1, Nr, chunk)
                # product is (batch, Nr, chunk)
                term = jnp.matmul(p_c[None, :, :] * v_c[:, None, :], l_c)
                return carry + term, None
            
            res, _ = jax.lax.scan(body_fn, jnp.zeros((Nb, Ns, No)), jnp.arange(num_chunks))
            return res

        def scan_X(X_acc, i_batch):
            w_batch = jax.lax.dynamic_slice(weights_padded, (i_batch * batch_size,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(xi_padded, (0, i_batch * batch_size), (N_rank, batch_size))
            G_batch = jax.lax.dynamic_slice(G_padded, (0, i_batch * batch_size, 0), (N_rank, batch_size, 3))
            
            xi_T = xi_batch.T  # (batch, N_rank)
            
            # Precompute projections for xi (phi_s_xi and phi_r_xi terms)
            # P_s_xi = (phi_s * xi) @ L_Q -> (batch, Ns, N_orb)
            P_s_xi = compute_proj(phi_s, xi_T, L_Q)
            P_r_xi = compute_proj(phi_r, xi_T, L_Q)
            
            # Loop over k (x,y,z components)
            for k in range(3):
                G_k_T = G_batch[:, :, k].T  # (batch, N_rank)
                
                P_r_G = compute_proj(phi_r, G_k_T, L_Q)
                P_s_G = compute_proj(phi_s, G_k_T, L_Q)
                
                # Reconstruct X contributions using low-rank outer products
                # Original: YZ_k = (phi_r_G_Q) @ (phi_s_G).T = (P_r_G @ L_Q.T) @ (P_s_G @ L_Q.T).T
                # where tmp_r_G_Q = (phi_r * G) @ L_Q @ L_Q.T = P_r_G @ L_Q.T
                # and phi_s_G = phi_s * G
                # So YZ_k = (P_r_G @ L_Q.T) @ (phi_s * G).T
                #         = P_r_G @ ( (phi_s * G) @ L_Q ).T
                #         = P_r_G @ P_s_G.T
                
                # X2: YZ_k = P_r_G @ P_s_G.T
                # einsum('bro,bso->brs', P_r_G, P_s_G)
                YZ_k = jnp.matmul(P_r_G, P_s_G.transpose(0, 2, 1))
                X_acc = X_acc + jnp.matmul((YZ_k * w_batch[:, None, None]).transpose(1, 2, 0), xi_T)
                
                # X3_1: M_k = (P_r_G @ L_Q.T) @ (phi_s * xi).T
                #           = P_r_G @ P_s_xi.T
                M_k = jnp.matmul(P_r_G, P_s_xi.transpose(0, 2, 1))
                X_acc = X_acc + jnp.matmul((M_k * w_batch[:, None, None]).transpose(1, 2, 0), G_k_T)
                
                # X3_2: N_k = (P_r_xi @ L_Q.T) @ (phi_s * G).T
                #           = P_r_xi @ P_s_G.T
                N_k = jnp.matmul(P_r_xi, P_s_G.transpose(0, 2, 1))
                X_acc = X_acc + jnp.matmul((N_k * w_batch[:, None, None]).transpose(1, 2, 0), G_k_T)
            
            return X_acc, None
        
        X_final, _ = jax.lax.scan(scan_X, jnp.zeros((Nr, Ns, N_rank)), jnp.arange(n_batches))
        return X_final

    def _calc_X_sketch_shard(self, jastrow_params, dm1, grid_points, weights,
                             xi_phi, G_shard, Gb, phi, ranges, n_orb,
                             batch_size=1024, L_Q=None, omega_s=None,
                             omega_c=None):
        """Return the in-kernel Khatri--Rao X sketch for one grid shard.

        This is the associative form of ``_calc_X_shard`` followed by
        ``einsum('rsc,sk,ck->rk', X, omega_s, omega_c)``.  Contracting the
        s and ISDF-rank legs first avoids the dense X accumulator that the
        basis selector does not otherwise need.
        """
        del jastrow_params, dm1, Gb, n_orb
        N_rank = phi.shape[1]
        N_shard = grid_points.shape[0]
        _slice_p, _slice_q, slice_r, slice_s = ranges
        phi_r = phi[slice_r]
        phi_s = phi[slice_s]
        Nr = phi_r.shape[0]
        Ns = phi_s.shape[0]
        n_probe = omega_s.shape[1]

        if omega_s.shape[0] != Ns or omega_c.shape != (N_rank, n_probe):
            raise ValueError(
                "X sketch shard probe dimensions disagree with its orbital/rank panels"
            )

        padded_size = ((N_shard + batch_size - 1) // batch_size) * batch_size
        weights_padded = jnp.pad(weights, (0, padded_size - N_shard))
        xi_padded = jnp.pad(xi_phi, ((0, 0), (0, padded_size - N_shard)))
        G_padded = jnp.pad(G_shard, ((0, 0), (0, padded_size - N_shard), (0, 0)))
        n_batches = padded_size // batch_size

        def compute_proj(phi_rows, vec_T, l_q, chunk_size=2048):
            """Compute ``(phi_rows * vec_T) @ l_q`` in rank chunks."""
            n_rows = phi_rows.shape[0]
            n_batch = vec_T.shape[0]
            n_density = l_q.shape[1]
            n_rank = l_q.shape[0]
            num_chunks = (n_rank + chunk_size - 1) // chunk_size
            pad_len = num_chunks * chunk_size - n_rank
            if pad_len > 0:
                phi_p = jnp.pad(phi_rows, ((0, 0), (0, pad_len)))
                vec_p = jnp.pad(vec_T, ((0, 0), (0, pad_len)))
                lq_p = jnp.pad(l_q, ((0, pad_len), (0, 0)))
            else:
                phi_p, vec_p, lq_p = phi_rows, vec_T, l_q

            def body_fn(carry, i):
                start = i * chunk_size
                p_c = jax.lax.dynamic_slice(phi_p, (0, start), (n_rows, chunk_size))
                v_c = jax.lax.dynamic_slice(vec_p, (0, start), (n_batch, chunk_size))
                l_c = jax.lax.dynamic_slice(lq_p, (start, 0), (chunk_size, n_density))
                term = jnp.matmul(p_c[None, :, :] * v_c[:, None, :], l_c)
                return carry + term, None

            result, _ = jax.lax.scan(
                body_fn, jnp.zeros((n_batch, n_rows, n_density)),
                jnp.arange(num_chunks),
            )
            return result

        def project_pair(p_r, p_s, rank_probe, w_batch):
            # p_s is first reduced over its orbital leg.  The remaining
            # contraction is only (batch, r, probe), rather than a dense
            # (batch, r, s) pair object followed by a rank-sized output.
            p_s_probe = jnp.einsum('bso,sk->bok', p_s, omega_s,
                                    optimize=True)
            return jnp.einsum('bro,bok,bk,b->rk', p_r, p_s_probe,
                              rank_probe, w_batch, optimize=True)

        def scan_X_sketch(sketch_acc, i_batch):
            start = i_batch * batch_size
            w_batch = jax.lax.dynamic_slice(weights_padded, (start,), (batch_size,))
            xi_batch = jax.lax.dynamic_slice(
                xi_padded, (0, start), (N_rank, batch_size)
            )
            G_batch = jax.lax.dynamic_slice(
                G_padded, (0, start, 0), (N_rank, batch_size, 3)
            )
            xi_T = xi_batch.T
            xi_probe = jnp.matmul(xi_T, omega_c)
            p_s_xi = compute_proj(phi_s, xi_T, L_Q)
            p_r_xi = compute_proj(phi_r, xi_T, L_Q)

            for component in range(3):
                g_T = G_batch[:, :, component].T
                g_probe = jnp.matmul(g_T, omega_c)
                p_r_g = compute_proj(phi_r, g_T, L_Q)
                p_s_g = compute_proj(phi_s, g_T, L_Q)
                sketch_acc = sketch_acc + project_pair(
                    p_r_g, p_s_g, xi_probe, w_batch
                )
                sketch_acc = sketch_acc + project_pair(
                    p_r_g, p_s_xi, g_probe, w_batch
                )
                sketch_acc = sketch_acc + project_pair(
                    p_r_xi, p_s_g, g_probe, w_batch
                )
            return sketch_acc, None

        sketch_final, _ = jax.lax.scan(
            scan_X_sketch, jnp.zeros((Nr, n_probe)), jnp.arange(n_batches)
        )
        return sketch_final



    def get_delta_U(self, jastrow_params, dm1=None, block_str=None, ranges=None, batch_size=1000):
        """Get delta_U matrix using ISDF with shard_map multi-device support."""
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
            
        if ranges is None:
            full_slice = slice(None)
            ranges = (full_slice, full_slice, full_slice, full_slice)
            
        start_time = time.perf_counter()
        logger.debug("Starting ISDFXTC.get_delta_U")
        
        # Check if kernels are available
        if self.isdf_kernels is None:
             # Compute kernels on the fly if not available
             logger.warning("ISDF kernels missing in get_delta_U. Computing on-the-fly with orbital batching. "
                             "This might be slow. Consider calling .isdf() first.")
             kernels = self.compute_delta_u_kernels(jastrow_params, batch_size)
        else:
            kernels = self.isdf_kernels

        result = self._contract_delta_U_kernels(kernels, ranges)

        # Public get_delta_U is a generic block API, not the solver tile path.
        # Keep its original chunk-aware symmetrization so large pq/rs blocks do
        # not route through the fixed-size tile executor.
        slice_p, slice_q, slice_r, slice_s = ranges
        if slice_p == slice_r and slice_q == slice_s:
            result = -(result + result.transpose(2, 3, 0, 1))
        else:
            result_np = -np.asarray(result)
            del result
            nmo = self.phi_isdf.shape[0]
            r_start = slice_r.start if slice_r.start is not None else 0
            r_stop = slice_r.stop if slice_r.stop is not None else nmo
            r_len = r_stop - r_start
            n_sub = 2
            chunk_size = max(1, (r_len + n_sub - 1) // n_sub)
            for i0 in range(0, r_len, chunk_size):
                i1 = min(i0 + chunk_size, r_len)
                sub_ranges = (slice(r_start + i0, r_start + i1),
                              slice_s, slice_p, slice_q)
                tmp = self._contract_delta_U_kernels(kernels, sub_ranges)
                chunk_np = np.asarray(tmp.transpose(2, 3, 0, 1))
                del tmp
                result_np[:, :, i0:i1, :] -= chunk_np
                del chunk_np
            result = jnp.asarray(result_np)

        total_time = time.perf_counter() - start_time
        logger.debug(f"ISDFXTC.get_delta_U completed in {total_time:.4f} s")
        return result


    def get_delta_h(self, jastrow_params, dm1=None,
                    block_str=None, ranges=None,
                    orb_block_size=None,
                    batch_size=1000):
        r"""Get or compute delta_h using ISDF kernels efficiently.
        
        Evaluates $\delta h_{pq} = \sum_{rs} (2 \Delta U_{pqrs} - \Delta U_{psrq}) \gamma_{rs}$
        directly from ISDF kernels D and X.
        
        Note on Symmetry:
        $\Delta U_{pqrs} = - (R_{pqrs} + R_{rspq})$ where $R_{pqrs} = (\phi_p \phi_q | \text{kernel} | \phi_r \phi_s)$.
        Term 1 (J-like): $2 \sum \Delta U_{pqrs} \gamma_{rs} = -2 (J + J_{sym})$.
        Term 2 (K-like): $\sum \Delta U_{psrq} \gamma_{rs} = - (K + K_{sym})$.
        $\delta h = -0.5 * (Term 1 - Term 2) = (J + J_{sym}) - 0.5 (K + K_{sym})$.
        
        $J$ involves $R_{pqrs}$, $J_{sym}$ involves $R_{rspq}$.
        $K$ involves $R_{psrq}$, $K_{sym}$ involves $R_{rqps}$.
        """
        logger.debug("Starting ISDFXTC.get_delta_h")
        start_time = time.perf_counter()
        
        if dm1 is None:
            dm1 = self._get_mf_dm()
            
        # Ensure kernels are available
        if self.isdf_kernels is None:
             logger.warning("ISDF kernels missing in get_delta_h. Computing on-the-fly.")
             kernels = self.compute_delta_u_kernels(jastrow_params, batch_size)
        else:
             kernels = self.isdf_kernels
             
        D = kernels['D']
        tucker_x = _get_tucker_x_factors(kernels)
        if tucker_x is None:
            X = kernels['X']
        else:
            # Do not touch a dense X backing when a Tucker view was supplied:
            # the normal-order path below contracts U/Z directly.
            X = None
        phi = self.phi_isdf

        slice_p = slice(None)
        slice_q = slice(None)
        if ranges is not None:
             slice_p, slice_q = ranges[0], ranges[1]

        # Adaptive orb_block_size: each HDF5 chunk is
        # (orb_block_size, n_orb, n_fused) * 8 bytes. For cc-pCV5Z-class
        # n_fused (~25k) the historical default 128/256 asks for ~30-60 GiB
        # per chunk and OOMs any single-device allocation.
        if orb_block_size is None:
            from pytc.utils.gpu_memory import choose_orb_block_size
            orb_block_size = choose_orb_block_size(
                n_orb=phi.shape[0], n_fused=phi.shape[1],
            )
            logger.debug(
                "get_delta_h: auto orb_block_size=%d (n_orb=%d, n_fused=%d)",
                orb_block_size, phi.shape[0], phi.shape[1],
            )

        Gb = jnp.einsum('rb,sb,rs->b', phi, phi, dm1)
        P_phi = jnp.linalg.multi_dot([phi.T, dm1, phi])
        phi_tilde = jnp.dot(dm1, phi)

        is_hdf5 = isinstance(X, (h5py.Dataset, h5py.File))

        wc = jnp.zeros((phi.shape[1],)) # (N_rank,)
        Y_all = jnp.zeros((self.n_orb, phi.shape[1])) # (N_orb, N_rank)

        tucker_j_x_sym = None
        if tucker_x is not None:
            wc, Y_all, tucker_j_x_sym = _tucker_x_normal_order_intermediates(
                tucker_x[0], tucker_x[1], dm1, phi_tilde, Gb,
            )
        elif is_hdf5:
            # Process strictly in chunks to respect memory
            logger.debug("Streaming X in chunks from HDF5")
            chunk_size = orb_block_size
            from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read

            pending_h = None
            for i in range(0, self.n_orb, chunk_size):
                start = i
                stop = min(i + chunk_size, self.n_orb)
                sl = slice(start, stop)
                logger.debug(f"Processing slice {start}-{stop}")

                if pending_h is not None:
                    X_chunk = await_read(pending_h)
                    pending_h = None
                else:
                    X_chunk = safe_hdf5_read(X, sl)

                # Prefetch next chunk while einsum runs
                next_start = stop
                if next_start < self.n_orb:
                    next_stop = min(next_start + chunk_size, self.n_orb)
                    pending_h = async_read(lambda _s=slice(next_start, next_stop): safe_hdf5_read(X, _s))
                
                wc += jnp.einsum('rsc,rs->c', X_chunk, dm1[sl])
                Y_all += jnp.einsum('rqc,rc->qc', X_chunk, phi_tilde[sl])
                
        else:
            # In-memory array
            wc = jnp.einsum('rsc,rs->c', X, dm1)
            Y_all = jnp.einsum('rqc,rc->qc', X, phi_tilde)
        
        phi_p = phi[slice_p]
        phi_q = phi[slice_q]
        Y_p = Y_all[slice_p]
        Y_q = Y_all[slice_q]
        
        D_sym = D + D.T
        tmp_a = jnp.dot(D_sym, Gb)
        J_D_total = jnp.dot(phi_p * tmp_a[None, :], phi_q.T)
        
        # J_X: - sum phi_p phi_q w_c
        J_X = - jnp.dot(phi_p * wc[None, :], phi_q.T)

        # J_X_sym: - sum X_pq G_c
        if tucker_j_x_sym is not None:
            J_X_sym = tucker_j_x_sym[slice_p, slice_q]
        elif is_hdf5:
            start_p, stop_p, step_p = slice_p.indices(self.n_orb)
            start_q, stop_q, step_q = slice_q.indices(self.n_orb)
            
            Np = (stop_p - start_p + step_p - 1) // step_p
            Nq = (stop_q - start_q + step_q - 1) // step_q
            
            J_X_sym_blocks = []
            
            from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read
            pending_jx = None
            for i in range(0, Np, orb_block_size):
                i_end = min(i + orb_block_size, Np)
                p_abs_start = start_p + i * step_p
                p_abs_stop = start_p + i_end * step_p
                p_abs_slice = slice(p_abs_start, p_abs_stop, step_p)
                
                if pending_jx is not None:
                    X_chunk = await_read(pending_jx)
                    pending_jx = None
                else:
                    X_chunk = safe_hdf5_read(X, (p_abs_slice, slice_q))

                block_res = - jnp.einsum('pqc,c->pq', X_chunk, Gb)

                # Prefetch next X block while einsum runs
                next_i = i + orb_block_size
                if next_i < Np:
                    ni_end = min(next_i + orb_block_size, Np)
                    np_start = start_p + next_i * step_p
                    np_stop = start_p + ni_end * step_p
                    n_slice = slice(np_start, np_stop, step_p)
                    pending_jx = async_read(lambda _sl=n_slice: safe_hdf5_read(X, (_sl, slice_q)))

                J_X_sym_blocks.append(block_res)
                
            J_X_sym = jnp.concatenate(J_X_sym_blocks, axis=0)

        else:
            X_pq = X[slice_p, slice_q]
            J_X_sym = - jnp.einsum('pqc,c->pq', X_pq, Gb)
        
        delta_h = _delta_h_jk_terms(D, Gb, P_phi, phi_p, phi_q,
                                    Y_p, Y_q, wc, J_D_total, J_X, J_X_sym)
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"ISDFXTC.get_delta_h completed in {total_time:.4f} s")
        return delta_h

    def _contract_delta_U_kernels(self, kernels, ranges):
        """Contract precomputed kernels to get Delta U block."""
        D = kernels['D']
        tucker_x = _get_tucker_x_factors(kernels)
        
        slice_p, slice_q, slice_r, slice_s = ranges

        if tucker_x is not None:
            # This generic public-block API is intentionally kept separate
            # from the solver tile path below.  It forms the existing
            # scan-based D contribution, then adds the factor-direct X term;
            # neither branch materialises dense X[r,s,c].
            n_rank = D.shape[0]
            phi_p = self.phi_isdf[slice_p]
            phi_q = self.phi_isdf[slice_q]
            phi_r = self.phi_isdf[slice_r]
            phi_s = self.phi_isdf[slice_s]
            rbs = self._get_fixed_rank_block_size()
            if rbs is None:
                from pytc.utils.gpu_memory import adaptive_rank_block_size
                rbs = adaptive_rank_block_size(
                    phi_p.shape[0], phi_q.shape[0], n_rank,
                    gpu_max_memory_mb=getattr(self, 'gpu_max_memory', None),
                )
            d_only = _contract_delta_U_kernels_jit(
                jnp.asarray(D), jnp.zeros((1, 1, 1), dtype=jnp.float64),
                jnp.asarray(phi_p), jnp.asarray(phi_q), jnp.asarray(phi_r),
                jnp.asarray(phi_s), rbs, False,
            )
            u, z = tucker_x
            return d_only + _contract_tucker_x_residual(
                phi_p, phi_q, u[slice_r], u[slice_s], z,
            )

        X = kernels['X']
        
        def get_info(sl, total):
            if isinstance(sl, slice):
                idx = np.arange(*sl.indices(total))
            else:
                idx = np.array(sl)
            return len(idx), idx

        Np, _ = get_info(slice_p, self.n_orb)
        from pytc.utils.gpu_memory import adaptive_rank_block_size, _get_gpu_free_bytes
        Nq, _ = get_info(slice_q, self.n_orb)
        Nr, r_idx = get_info(slice_r, self.n_orb)
        Ns, s_idx = get_info(slice_s, self.n_orb)
        N_rank = X.shape[2]
        phi_p = self.phi_isdf[slice_p]
        phi_q = self.phi_isdf[slice_q]

        # Use fixed rank_block_size (worst-case over all phases) to avoid
        # JIT recompilation when (Np, Nq) changes across CCSD blocks.
        _rbs = self._get_fixed_rank_block_size()
        if _rbs is None:
            _rbs = adaptive_rank_block_size(
                Np, Nq, N_rank,
                gpu_max_memory_mb=getattr(self, 'gpu_max_memory', None))

        gpu_free_bytes = _get_gpu_free_bytes()
        
        # Estimate total GPU memory needed for delta_U calculation.
        # _contract_delta_U_kernels_jit runs TWO sequential lax.scans:
        #   1. D-scan: accumulates term_d (Np, Nq, Nr, Ns)
        #   2. X-scan: accumulates term_x, with term_d still alive
        # Peak during X-scan:
        #   X_sliced (JIT input, stays resident) + X_padded (~14% larger copy)
        #   + D (JIT input) + term_d + carry + contribution
        #   ≈ X_sliced * 2 + D + 3 × carry
        # Peak during D-scan:
        #   X_sliced (alive for later) + D + 2 × carry + W + C_rs
        x_sliced_size_bytes, d_size_bytes, scan_carry_bytes, total_needed_bytes = (
            _estimate_delta_u_contraction_bytes(Np, Nq, Nr, Ns, N_rank)
        )
        
        # Threshold: use half of actually free GPU memory (not budget).
        # This is more accurate than the budget-based estimate since it
        # accounts for pre-allocated tensors (phi_isdf, etc.).
        threshold_bytes = int(gpu_free_bytes * 0.5)
        
        logger.debug(
            "delta_U memory estimate: X_sliced=%.2f GB, scan_carry=%.2f GB, total=%.2f GB "
            "(threshold=%.2f GB, dims: Np=%d, Nq=%d, Nr=%d, Ns=%d)",
            x_sliced_size_bytes / (1024.0**3),
            scan_carry_bytes / (1024.0**3),
            total_needed_bytes / (1024.0**3),
            threshold_bytes / (1024.0**3),
            Np, Nq, Nr, Ns,
        )
        
        if total_needed_bytes < threshold_bytes:
            phi_r = self.phi_isdf[slice_r]
            phi_s = self.phi_isdf[slice_s]
            X_full = _read_X_slice(X, slice_r, slice_s)
            return _contract_delta_U_kernels_jit(
                D,
                jnp.asarray(X_full),
                jnp.asarray(phi_p),
                jnp.asarray(phi_q),
                jnp.asarray(phi_r),
                jnp.asarray(phi_s),
                _rbs,
                True,
            )

        # Chunking strategy to avoid VRAM exhaustion. Stream only the X panel
        # needed for each chunk instead of first materializing the full X slice.
        logger.warning(
            "  delta_U memory estimate (%.2f GB) exceeds %.2f GB limit. Chunking orbital indices.",
            total_needed_bytes / (1024.0**3),
            threshold_bytes / (1024.0**3),
        )

        # Keep 1-D index arrays on host (cheap); avoid eagerly copying the full
        # phi_isdf matrix — only the rows needed per chunk are materialised
        # inside _prepare_*_chunk below.
        phi_isdf_src = self.phi_isdf  # may be a JAX array or np.ndarray
        if isinstance(r_idx, np.ndarray):
            r_idx_np = r_idx
            s_idx_np = s_idx
        else:
            r_idx_np = np.asarray(r_idx)
            s_idx_np = np.asarray(s_idx)
        
        result = np.zeros((Np, Nq, Nr, Ns), dtype=np.float64)
        
        # Choose chunk size to keep TOTAL memory (X_chunk + 3× carry) under budget.
        # For r-chunking (reducing Nr to Nr_chunk):
        #   X_chunk = (Nr_chunk, Ns, N_rank)
        #   carry_chunk = (Np, Nq, Nr_chunk, Ns)
        #   Total per Nr_chunk_unit = Ns * N_rank * 8 * 2 (input+padded)
        #                            + 3 * Np * Nq * Ns * 8
        # For s-chunking (reducing Ns to Ns_chunk):
        #   X_chunk = (Nr, Ns_chunk, N_rank)
        #   carry_chunk = (Np, Nq, Nr, Ns_chunk)
        #   Total per Ns_chunk_unit = Nr * N_rank * 8 * 2 (input+padded)
        #                            + 3 * Np * Nq * Nr * 8
        # Target: total + D fits in threshold.
        target_bytes = max(threshold_bytes - d_size_bytes, 1)
        
        if Nr >= Ns:
            per_r_unit_bytes = int(2 * Ns * N_rank * 8 + 3 * Np * Nq * Ns * 8)
            max_Nr_chunk = max(1, int(target_bytes / max(per_r_unit_bytes, 1))) if per_r_unit_bytes > 0 else Nr
            orb_chunk_size = min(max_Nr_chunk, Nr)
            
            chunk_total_bytes = orb_chunk_size * per_r_unit_bytes + d_size_bytes
            logger.debug(
                "Chunking over 'r' index. Chunk size: %d (est. per chunk: %.2f GB)",
                orb_chunk_size, chunk_total_bytes / (1024.0**3),
            )
            logger.debug("  Chunking with streaming r panels. Total Nr=%d, Ns=%d.", Nr, Ns)
            
            phi_s = self.phi_isdf[slice_s]

            def _prepare_r_chunk(i_start):
                """Prepare phi_r_chunk and X_chunk for a given r-index range (host side)."""
                ie = min(i_start + orb_chunk_size, Nr)
                alen = ie - i_start
                r_sel = _chunk_selector(r_idx_np, i_start, ie)
                pr_np = np.asarray(phi_isdf_src[r_sel])
                xc_np = np.asarray(_read_X_slice(X, r_sel, slice_s))
                if alen < orb_chunk_size:
                    pad = orb_chunk_size - alen
                    pr_np = np.pad(pr_np, ((0, pad), (0, 0)))
                    xc_np = np.pad(xc_np, ((0, pad), (0, 0), (0, 0)))
                return pr_np, xc_np, alen

            phi_r_chunk, X_chunk, actual_len = _prepare_r_chunk(0)
            chunk_timer = time.perf_counter()

            for i in range(0, Nr, orb_chunk_size):
                chunk_id = i // orb_chunk_size
                logger.debug(
                    f"  Starting delta_U chunk r[{i}:{i+orb_chunk_size}] "
                    f"(idx={chunk_id}, prepped_len={actual_len}, Np={Np}, Nq={Nq}, Ns={Ns})"
                )
                cur_phi_r = jnp.asarray(phi_r_chunk)
                cur_X = jnp.asarray(X_chunk)
                cur_actual = actual_len

                res_chunk = _contract_delta_U_kernels_jit(
                    D, cur_X, phi_p, phi_q, cur_phi_r, phi_s, _rbs, True)

                next_i = i + orb_chunk_size
                if next_i < Nr:
                    # Prepare next chunk on host while current chunk is being reduced.
                    phi_r_chunk, X_chunk, actual_len = _prepare_r_chunk(next_i)

                result[:, :, i:i+cur_actual, :] = np.asarray(res_chunk)[:, :, :cur_actual, :]
                del res_chunk

                elapsed = time.perf_counter() - chunk_timer
                logger.debug(
                    f"  Finished delta_U chunk r[{i}:{i+cur_actual}] in {elapsed:.3f} s"
                )
                chunk_timer = time.perf_counter()

                gc.collect()
        else:
            per_s_unit_bytes = int(2 * Nr * N_rank * 8 + 3 * Np * Nq * Nr * 8)
            max_Ns_chunk = max(1, int(target_bytes / max(per_s_unit_bytes, 1))) if per_s_unit_bytes > 0 else Ns
            orb_chunk_size = min(max_Ns_chunk, Ns)
            
            chunk_total_bytes = orb_chunk_size * per_s_unit_bytes + d_size_bytes
            logger.debug(
                "Chunking over 's' index. Chunk size: %d (est. per chunk: %.2f GB)",
                orb_chunk_size, chunk_total_bytes / (1024.0**3),
            )
            logger.debug("  Chunking with streaming s panels. Total Nr=%d, Ns=%d.", Nr, Ns)

            phi_r = self.phi_isdf[slice_r]

            def _prepare_s_chunk(i_start):
                """Prepare phi_s_chunk and X_chunk for a given s-index range (host side)."""
                ie = min(i_start + orb_chunk_size, Ns)
                alen = ie - i_start
                s_sel = _chunk_selector(s_idx_np, i_start, ie)
                ps_np = np.asarray(phi_isdf_src[s_sel])
                xc_np = np.asarray(_read_X_slice(X, slice_r, s_sel))
                if alen < orb_chunk_size:
                    pad = orb_chunk_size - alen
                    ps_np = np.pad(ps_np, ((0, pad), (0, 0)))
                    xc_np = np.pad(xc_np, ((0, 0), (0, pad), (0, 0)))
                return ps_np, xc_np, alen

            phi_s_chunk, X_chunk, actual_len = _prepare_s_chunk(0)
            chunk_timer = time.perf_counter()

            for i in range(0, Ns, orb_chunk_size):
                chunk_id = i // orb_chunk_size
                logger.debug(
                    f"  Starting delta_U chunk s[{i}:{i+orb_chunk_size}] "
                    f"(idx={chunk_id}, prepped_len={actual_len}, Np={Np}, Nq={Nq}, Nr={Nr})"
                )
                cur_phi_s = phi_s_chunk
                cur_X = X_chunk
                cur_actual = actual_len

                cur_phi_s = jnp.asarray(cur_phi_s)
                cur_X = jnp.asarray(cur_X)
                res_chunk = _contract_delta_U_kernels_jit(
                    D, cur_X, phi_p, phi_q, phi_r, cur_phi_s, _rbs, True)

                next_i = i + orb_chunk_size
                if next_i < Ns:
                    # Prepare next chunk on host while current chunk is being reduced.
                    phi_s_chunk, X_chunk, actual_len = _prepare_s_chunk(next_i)

                result[:, :, :, i:i+cur_actual] = np.asarray(res_chunk)[:, :, :, :cur_actual]
                del res_chunk

                elapsed = time.perf_counter() - chunk_timer
                logger.debug(
                    f"  Finished delta_U chunk s[{i}:{i+cur_actual}] in {elapsed:.3f} s"
                )
                chunk_timer = time.perf_counter()

                gc.collect()
                
        return jnp.asarray(result)

    def _get_delta_u_direct_tile(self, kernels, ranges, device=None, panel_size=None,
                                 panel_layout="pr"):
        """Compute one unsymmetrized Delta U tile from prepared kernel panels."""
        D = kernels['D']
        tucker_x = _get_tucker_x_factors(kernels)
        X = None if tucker_x is not None else kernels['X']
        slice_p, slice_q, slice_r, slice_s = ranges
        device_key = getattr(device, "id", "host")
        panel_layout = _normalize_panel_layout(panel_layout)
        cache = self._get_isdf_device_cache(
            kernels, device=device, include_grad=False, include_delta_u=True
        )
        phi_src = cache["phi_isdf"] if cache is not None else self.phi_isdf
        D_resident = cache.get("D") if cache is not None else None

        phi_p = phi_src[slice_p]
        phi_q = phi_src[slice_q]
        phi_r = phi_src[slice_r]
        phi_s = phi_src[slice_s]

        p_len = phi_p.shape[0]
        Nq = phi_q.shape[0]
        r_len = phi_r.shape[0]
        Ns = phi_s.shape[0]
        N_rank = D.shape[0]
        Np = panel_size if panel_size is not None and "p" in panel_layout else p_len
        Nq_eff = panel_size if panel_size is not None and "q" in panel_layout else Nq
        Nr = panel_size if panel_size is not None and "r" in panel_layout else r_len
        Ns_eff = panel_size if panel_size is not None and "s" in panel_layout else Ns

        # D is already counted in ``in_use`` when it's resident in the cache,
        # so don't add it again to the peak estimate — that would double its
        # contribution and wrongly refuse tiles that actually fit.
        mem = _estimate_delta_u_direct_tile_bytes(
            Np, Nq_eff, Nr, Ns_eff, N_rank,
            include_d=(D_resident is None),
        )
        total_needed_bytes = mem["total"]
        # Safety factor on top of the analytical peak estimate. 0.7 leaves
        # ~43 % head-room for XLA workspace / BFC fragmentation on top of
        # the estimate (which already counts D, X_sliced, C_pq × 2, C_rs,
        # and 2 × out).
        threshold_bytes = int(_get_device_free_bytes(device) * 0.7)
        if total_needed_bytes >= threshold_bytes:
            raise RuntimeError(
                "Delta U direct tile exceeds available device memory: "
                f"need ~{total_needed_bytes / (1024.0 ** 3):.2f} GiB for "
                f"tile ({Np}, {Nq_eff}, {Nr}, {Ns_eff}), have "
                f"~{threshold_bytes / (1024.0 ** 3):.2f} GiB usable "
                f"(D_resident={D_resident is not None}). "
                "Reduce the solver tile panel size."
            )

        if tucker_x is not None:
            # Keep the existing conservative tile guard until a measured
            # factor-direct peak model is available.  It uses the historical
            # dense-X bound, so it may choose a smaller panel than necessary
            # but it cannot overcommit a GPU while we validate this path.
            u, z = tucker_x
            u_r = u[slice_r]
            u_s = u[slice_s]
            if panel_size is not None:
                phi_p = _pad_axis(phi_p, 0, Np) if Np != p_len else jnp.asarray(phi_p)
                phi_q = _pad_axis(phi_q, 0, Nq_eff) if Nq_eff != Nq else jnp.asarray(phi_q)
                phi_r = _pad_axis(phi_r, 0, Nr) if Nr != r_len else jnp.asarray(phi_r)
                phi_s = _pad_axis(phi_s, 0, Ns_eff) if Ns_eff != Ns else jnp.asarray(phi_s)
                u_r = _pad_axis(u_r, 0, Nr) if Nr != r_len else jnp.asarray(u_r)
                u_s = _pad_axis(u_s, 0, Ns_eff) if Ns_eff != Ns else jnp.asarray(u_s)
            else:
                phi_p = jnp.asarray(phi_p)
                phi_q = jnp.asarray(phi_q)
                phi_r = jnp.asarray(phi_r)
                phi_s = jnp.asarray(phi_s)

            if device is not None:
                D = D_resident if D_resident is not None else jax.device_put(np.asarray(D), device)
                # The per-device ISDF cache is keyed by this object.  Retain
                # the factor backing itself as the identity guard so a new
                # approximation cannot inherit stale device buffers.
                factor_cache = cache.get("X_tucker") if cache is not None else None
                if factor_cache is None or factor_cache[0] is not u or factor_cache[1] is not z:
                    factor_cache = (
                        u, z,
                        jax.device_put(np.asarray(u), device),
                        jax.device_put(np.asarray(z), device),
                    )
                    if cache is not None:
                        cache["X_tucker"] = factor_cache
                u_device, z_device = factor_cache[2:]
                u_r = u_device[slice_r]
                u_s = u_device[slice_s]
                if panel_size is not None:
                    u_r = _pad_axis(u_r, 0, Nr) if Nr != r_len else u_r
                    u_s = _pad_axis(u_s, 0, Ns_eff) if Ns_eff != Ns else u_s
                if cache is None:
                    phi_p = jax.device_put(phi_p, device)
                    phi_q = jax.device_put(phi_q, device)
                    phi_r = jax.device_put(phi_r, device)
                    phi_s = jax.device_put(phi_s, device)
            else:
                D = jnp.asarray(D)
                u_r = jnp.asarray(u_r)
                u_s = jnp.asarray(u_s)
                z_device = jnp.asarray(z)

            device_ctx = (
                jax.default_device(device)
                if device is not None
                else contextlib.nullcontext()
            )
            with device_ctx:
                return _contract_delta_u_tucker_direct_tile_jit(
                    D, z_device, phi_p, phi_q, phi_r, phi_s, u_r, u_s,
                )

        X_sliced = _read_X_slice(X, slice_r, slice_s)
        if panel_size is not None:
            phi_p = _pad_axis(phi_p, 0, Np) if Np != p_len else jnp.asarray(phi_p)
            phi_q = _pad_axis(phi_q, 0, Nq_eff) if Nq_eff != Nq else jnp.asarray(phi_q)
            phi_r = _pad_axis(phi_r, 0, Nr) if Nr != r_len else jnp.asarray(phi_r)
            phi_s = _pad_axis(phi_s, 0, Ns_eff) if Ns_eff != Ns else jnp.asarray(phi_s)
            # Pad ``X_sliced`` on host (NumPy) so the multi-GB slab does
            # NOT get materialised on the JAX default device (typically
            # GPU 0) before being copied to the actual target device.
            # Keeping ``X_sliced`` as a
            # NumPy array until the explicit ``jax.device_put`` below
            # gives a single, correctly-targeted host→device copy.
            if isinstance(X_sliced, np.ndarray):
                if Nr != r_len or Ns_eff != Ns:
                    pad_cfg = [(0, 0)] * X_sliced.ndim
                    if Nr != r_len:
                        pad_cfg[0] = (0, Nr - X_sliced.shape[0])
                    if Ns_eff != Ns:
                        pad_cfg[1] = (0, Ns_eff - X_sliced.shape[1])
                    X_sliced = np.pad(X_sliced, pad_cfg)
            else:
                # ``X_sliced`` is already a JAX array (e.g. from a per-
                # device cache).  Pad with the JAX helper, which keeps
                # it on its current device.
                if Nr != r_len:
                    X_sliced = _pad_axis(X_sliced, 0, Nr)
                if Ns_eff != Ns:
                    X_sliced = _pad_axis(X_sliced, 1, Ns_eff)
        else:
            phi_p = jnp.asarray(phi_p)
            phi_q = jnp.asarray(phi_q)
            phi_r = jnp.asarray(phi_r)
            phi_s = jnp.asarray(phi_s)
            # Don't ``jnp.asarray(X_sliced)`` here — that would land the
            # multi-GB slab on the default device (GPU 0) before the
            # explicit ``device_put`` copies it to the target device.
            # Leaving it as NumPy keeps the host→device transfer
            # single-step.
        if device is not None:
            D = D_resident if D_resident is not None else jax.device_put(np.asarray(D), device)
            X_sliced = jax.device_put(X_sliced, device)
            if cache is None:
                phi_p = jax.device_put(phi_p, device)
                phi_q = jax.device_put(phi_q, device)
                phi_r = jax.device_put(phi_r, device)
                phi_s = jax.device_put(phi_s, device)
        else:
            D = jnp.asarray(D)
            X_sliced = jnp.asarray(X_sliced)
        device_ctx = (
            jax.default_device(device)
            if device is not None
            else contextlib.nullcontext()
        )
        with device_ctx:
            return _contract_delta_u_direct_tile_jit(
                D, X_sliced, phi_p, phi_q, phi_r, phi_s,
            )

    def _assemble_delta_u_tile(self, kernels, ranges, device=None, panel_size=None,
                               panel_layout="pr"):
        """Assemble and symmetrize one finished Delta U tile."""
        global _DELTA_U_AUTOSHRINK_WARNED
        panel_layout = _normalize_panel_layout(panel_layout)

        # Auto-shrink panel_size to fit current free memory before dispatching
        # either direct-tile call.  The size is resolved once here so both
        # calls below use identical dimensions — required for the symmetrize
        # step (direct + tmp.transpose(2,3,0,1)) to produce matching shapes.
        if panel_size is not None:
            device_key = getattr(device, "id", "host")
            nmo = self.phi_isdf.shape[0]
            slice_p, slice_q, slice_r, slice_s = ranges
            p_len = (slice_p.stop or nmo) - (slice_p.start or 0)
            q_len = (slice_q.stop or nmo) - (slice_q.start or 0)
            r_len = (slice_r.stop or nmo) - (slice_r.start or 0)
            s_len = (slice_s.stop or nmo) - (slice_s.start or 0)
            N_rank = kernels['D'].shape[0]
            # Mirror _get_delta_u_direct_tile: D is already excluded from
            # free_bytes when it's resident in the device cache, so only
            # add it to the estimate when it's NOT resident — otherwise the
            # double-count can over-shrink panel_size or falsely trigger the
            # genuine-OOM guard on later tiles.
            _cache = self._get_isdf_device_cache(
                kernels, device=device,
                include_grad=False, include_delta_u=True
            )
            _D_resident = _cache.get("D") if _cache is not None else None
            free_bytes = _get_device_free_bytes(device)
            threshold_bytes = int(free_bytes * 0.7)

            def _tile_bytes(ps, _incl_d=(_D_resident is None)):
                Np = ps if "p" in panel_layout else p_len
                Nq = ps if "q" in panel_layout else q_len
                Nr = ps if "r" in panel_layout else r_len
                Ns = ps if "s" in panel_layout else s_len
                return _isdf_tile_peak_bytes(Np, Nq, Nr, Ns, N_rank, include_d=_incl_d)

            safe_ps = _find_max_blksize(_tile_bytes, lo=1, hi=panel_size,
                                        gpu_target=threshold_bytes)
            # _pad_axis raises ValueError when target < current length (pad up only).
            # The genuine-OOM guard must therefore check only the PADDED axes —
            # axes not in panel_layout are passed at their full slice length and
            # never padded, so safe_ps < that length is fine.
            min_padded = max(
                p_len if "p" in panel_layout else 0,
                q_len if "q" in panel_layout else 0,
                r_len if "r" in panel_layout else 0,
                s_len if "s" in panel_layout else 0,
            )
            if safe_ps < min_padded:
                # Even the minimum padded tile doesn't fit — genuine OOM.
                # (The setup-time estimate should have prevented this; it can
                # happen if another process consumed GPU memory between setup
                # and dispatch.)
                raise RuntimeError(
                    f"_assemble_delta_u_tile: GPU memory too low for current tile "
                    f"(layout={panel_layout!r}, min_padded={min_padded}): "
                    f"need at least {_tile_bytes(min_padded) / 2**30:.2f} GiB "
                    f"but only {threshold_bytes / 2**30:.2f} GiB available "
                    f"({free_bytes / 2**30:.2f} GiB free, device={device_key}). "
                    "Reduce n_fused/nkeep or use a larger GPU."
                )
            if safe_ps < panel_size:
                if device_key not in _DELTA_U_AUTOSHRINK_WARNED:
                    _DELTA_U_AUTOSHRINK_WARNED.add(device_key)
                    logger.warning(
                        "_assemble_delta_u_tile: dispatch-time shrink panel_size "
                        "%d → %d (%.2f GiB free after tc_tile, device=%s). "
                        "Set PYTC_SOLVER_BLK=%d to pin this size and avoid JAX "
                        "recompiles.",
                        panel_size, safe_ps,
                        free_bytes / 2**30,
                        device_key,
                        safe_ps,
                    )
                panel_size = safe_ps

        direct = self._get_delta_u_direct_tile(
            kernels, ranges, device=device, panel_size=panel_size,
            panel_layout=panel_layout)

        slice_p, slice_q, slice_r, slice_s = ranges
        if panel_size is not None:
            # The ``direct + direct.transpose(2, 3, 0, 1)`` shortcut is
            # only valid when the panel padding is invariant under the
            # ``(p↔r, q↔s)`` axis swap — i.e. when ``panel_layout == "pr"``
            # (pads axes 0 and 2 symmetrically, leaves 1 and 3 alone).
            # ``"qr"`` and ``"ps"`` pad asymmetrically so ``direct`` and
            # ``direct.transpose(2, 3, 0, 1)`` come out with different
            # shapes that cannot broadcast — e.g. an ovov single-tile
            # under ``"qr"`` would try to add ``(nocc, ps, ps, nvir)``
            # to ``(ps, nvir, nocc, ps)``.  For the asymmetric layouts
            # we fall through to the explicit-``tmp`` branch, which uses
            # ``_transpose_panel_layout()`` to build a partner tile whose
            # shape matches ``direct`` after the (2, 3, 0, 1) transpose.
            # See the parallel guard in ``_assemble_tc_tile``.
            if (slice_p == slice_r and slice_q == slice_s
                    and panel_layout == "pr"):
                return -(direct + direct.transpose(2, 3, 0, 1))

            ranges_T = (slice_r, slice_s, slice_p, slice_q)
            tmp = self._get_delta_u_direct_tile(
                kernels, ranges_T, device=device, panel_size=panel_size,
                panel_layout=_transpose_panel_layout(panel_layout))
            return -(direct + tmp.transpose(2, 3, 0, 1))

        if slice_p == slice_r and slice_q == slice_s:
            return -(direct + direct.transpose(2, 3, 0, 1))

        result_np = -np.asarray(direct)
        del direct
        nmo = self.phi_isdf.shape[0]
        r_start = slice_r.start if slice_r.start is not None else 0
        r_stop = slice_r.stop if slice_r.stop is not None else nmo
        r_len = r_stop - r_start
        n_sub = 2
        chunk_size = max(1, (r_len + n_sub - 1) // n_sub)
        for i0 in range(0, r_len, chunk_size):
            i1 = min(i0 + chunk_size, r_len)
            sub_ranges = (slice(r_start + i0, r_start + i1),
                          slice_s, slice_p, slice_q)
            tmp = self._get_delta_u_direct_tile(kernels, sub_ranges, device=device)
            chunk_np = np.asarray(tmp.transpose(2, 3, 0, 1))
            del tmp
            result_np[:, :, i0:i1, :] -= chunk_np
            del chunk_np
        return jnp.asarray(result_np)

    def _assemble_2b_tile(self, jastrow_params, kernels, ranges, device=None,
                          panel_size=None, panel_layout="pr"):
        """Assemble a finished ISDF-XTC 2-body tile from TC and Delta U parts."""
        del jastrow_params
        panel_layout = _normalize_panel_layout(panel_layout)
        # Per-tile stage timing (only active when a pipeline has opened an
        # ``issue_stage_stats_scope`` — see the comment near the top of
        # this file).  We measure pure Python-return time here, *without*
        # forcing ``block_until_ready``, because the goal is to pin down
        # which sub-call is blocking the main dispatch thread under normal
        # async JAX semantics — adding an explicit barrier would
        # manufacture the very stall we're trying to detect.
        _stage_timing = _ISSUE_STAGE_STATS["current"] is not None
        if _stage_timing:
            _t_stage_tc0 = time.perf_counter()
        tc_tile = super()._assemble_tc_tile(
            kernels, ranges, device=device, panel_size=panel_size,
            panel_layout=panel_layout)
        if _stage_timing:
            _t_stage_tc1 = time.perf_counter()
            _accum_issue_stage("tc_assemble_s", _t_stage_tc1 - _t_stage_tc0)
        if _stage_timing:
            _t_stage_du0 = time.perf_counter()
        delta_u_tile = self._assemble_delta_u_tile(
            kernels, ranges, device=device, panel_size=panel_size,
            panel_layout=panel_layout)
        if _stage_timing:
            _t_stage_du1 = time.perf_counter()
            _accum_issue_stage("delta_u_assemble_s", _t_stage_du1 - _t_stage_du0)
        if panel_size is not None:
            if _stage_timing:
                _t_stage_sum0 = time.perf_counter()
            result = tc_tile + delta_u_tile
            if _stage_timing:
                _t_stage_sum1 = time.perf_counter()
                _accum_issue_stage("final_sum_s", _t_stage_sum1 - _t_stage_sum0)
                _accum_issue_stage("n_tiles", 1)
            return result

        tc_tile = np.array(tc_tile)
        tc_tile += np.array(delta_u_tile)
        return jnp.asarray(tc_tile)
    
