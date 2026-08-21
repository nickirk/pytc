"""JAX implementation of Transcorrelated method."""

import contextlib
import threading
import weakref
from typing import Any
import numpy as np
import os
import logging
import time
import gc
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
import h5py
from flax import struct
from pyscf import dft
from . import kmat as kmat_jax
from .utils import sharding_core

logger = logging.getLogger(__name__)

# Module-level cache for the fixed rank_block_size computed once per
# (n_orb, N_fused) pair.  Using the worst-case orbital dimensions
# (n_orb, n_orb) produces the most conservative (smallest) power-of-2
# block size, which is safe for every (Np, Nq) slice encountered
# during CCSD iterations.  This eliminates JIT recompilation from
# changing static_argnums values across ovvv / vovv / vvvv phases.
_FIXED_RBS_CACHE: dict = {}


def _laux_gradient_mode(use_fast):
    """Return the cache fingerprint for one L_aux construction mode."""
    return "fast" if use_fast else "direct"


def _panel_blk_overrides():
    """Read optional process-wide panel/block-size overrides from the env.

    Returns ``(panel_blk, gpu_max_memory_mb)`` where each is ``None`` when the
    corresponding env var is unset (the default), so the autotuned behaviour is
    unchanged unless a caller opts in. These are expert escape hatches for
    shrinking the direct-tile / K-stream panel on memory-constrained systems;
    the permanent fix is to make the delta_U direct tile honour
    ``SAFE_TILE_BYTES_CEILING`` like the CCSD vvvv path (separate change).

      PYTC_PANEL_BLK          cap on the K-stream panel_size AND the
                              rank_block_size (must be ≥ 1; ignored otherwise).
      PYTC_GPU_MAX_MEMORY_MB  authoritative total-GPU budget (MiB) fed into
                              ``adaptive_rank_block_size``'s budget path.
      PYTC_SOLVER_BLK         cap on the CCSD vvvv tile panel_size in
                              ``resolve_vvvv_panel_block_sizes`` AND the v3o
                              (large-blocks) tile in
                              ``resolve_v3o_panel_block_size`` (must be ≥ 1).
                              Use when the delta_U direct-tile pre-flight is
                              borderline; reduces tile size O(blk²).
    """
    try:
        pb = os.environ.get("PYTC_PANEL_BLK")
        panel_blk = int(pb) if pb is not None and pb.strip() else None
    except (TypeError, ValueError):
        panel_blk = None
    try:
        gm = os.environ.get("PYTC_GPU_MAX_MEMORY_MB")
        gpu_max_memory_mb = float(gm) if gm is not None and gm.strip() else None
    except (TypeError, ValueError):
        gpu_max_memory_mb = None
    if gpu_max_memory_mb is not None and not (gpu_max_memory_mb > 0 and
                                           gpu_max_memory_mb != float('inf')):
        # Reject non-positive / non-finite (nan, inf, -inf) so they degrade to no-op.
        gpu_max_memory_mb = None
    if panel_blk is not None and panel_blk < 1:
        panel_blk = None
    return panel_blk, gpu_max_memory_mb

# ----------------------------------------------------------------------
# Per-(ISDFTC instance, device) cache of device-resident phi_isdf /
# grad_phi_isdf / TC kernels / D so the per-tile dispatch path doesn't
# re-upload them.
#
# The cache is process-wide (rather than per-instance) so that multiple
# code paths sharing the same ISDFTC instance can both benefit from a
# single upload.  Keys are ``(id(self), device_id)`` for fast lookup.
#
# Lifecycle
# ------------------------
# Without explicit cleanup the cache leaks both ways:
#   1. Memory: an ISDFTC instance that is GC'd leaves its multi-GB
#      device-resident phi/kernels / D entries permanently in the
#      cache, so VRAM monotonically grows across runs in a long-lived
#      process (notebook, REPL, sweep script).
#   2. Correctness: ``id()`` is reused after object destruction, so a
#      *new* ISDFTC that happens to receive a recycled id() would
#      silently inherit the *old* instance's stale device buffers.
#
# Both are addressed by registering a ``weakref.finalize`` the first
# time we cache anything for an instance: the finaliser drops every
# ``(id(self), *)`` entry the moment ``self`` is GC'd, which both frees
# VRAM and prevents id-reuse hits.  ``invalidate_isdf_device_cache``
# remains available for callers that want explicit eviction without
# relying on GC.
# ----------------------------------------------------------------------
_ISDF_DEVICE_CACHE: dict = {}
_ISDF_DEVICE_CACHE_LOCK = threading.Lock()
_ISDF_DEVICE_CACHE_FINALISED: set = set()  # ids we've already attached a finaliser to


def _evict_isdf_device_cache_for_id(self_id):
    """Drop all cache entries for an ISDFTC instance (called from GC)."""
    with _ISDF_DEVICE_CACHE_LOCK:
        keys_to_drop = [k for k in _ISDF_DEVICE_CACHE if k[0] == self_id]
        for k in keys_to_drop:
            _ISDF_DEVICE_CACHE.pop(k, None)
        _ISDF_DEVICE_CACHE_FINALISED.discard(self_id)


def invalidate_isdf_device_cache(instance=None):
    """Manually evict ISDFTC device-cache entries.

    Without arguments, clears the entire cache (every instance, every
    device).  With an ISDFTC instance, evicts only that instance's
    entries — useful for releasing a known-stale set of buffers ahead
    of GC.

    Equivalent to the automatic ``weakref.finalize`` path, but
    available for callers that want explicit lifecycle control.
    """
    if instance is None:
        with _ISDF_DEVICE_CACHE_LOCK:
            _ISDF_DEVICE_CACHE.clear()
            _ISDF_DEVICE_CACHE_FINALISED.clear()
    else:
        _evict_isdf_device_cache_for_id(id(instance))


_TC_DIRECT_TILE_PROFILED: set = set()  # log first tile's phase breakdown once per device/layout


def _array_nbytes(arr):
    """Return the byte size of an array-like object without copying."""
    shape = getattr(arr, "shape", None)
    dtype = getattr(arr, "dtype", None)
    if shape is None or dtype is None:
        arr_np = np.asarray(arr)
        return int(arr_np.size * arr_np.dtype.itemsize)
    return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)


def _get_local_device_free_bytes(device):
    """Return free bytes for one local device when available."""
    try:
        stats = device.memory_stats()
        pool_limit = int(stats["bytes_limit"])
        in_use = int(stats.get("bytes_in_use", 0))
        return max(pool_limit - in_use, 0)
    except Exception:
        return 1 << 60


def _cache_d_on_device(device, arr, *, fraction=0.35):
    """Whether a persistent device cache should keep ``arr`` resident."""
    return _array_nbytes(arr) <= int(_get_local_device_free_bytes(device) * fraction)


def _cache_tc_kernels_on_device(device, u1, u3, *, fraction=0.30):
    """Whether persistent TC kernels should be cached on one device.

    Kept for API compatibility with call-sites that only want a bool.
    """
    return _choose_tc_kernel_strategy(device, u1, u3, fraction=fraction)[0]


def _choose_tc_kernel_strategy(device, u1, u3, *, fraction=0.15):
    """Decide how TC kernels (K1 + K3) should be kept for tile consumption.

    Returns ``(resident, panel_size)``:
      * ``(True, None)``   — K1 + K3 fit comfortably on ``device``; keep them
        resident. Consumers call the streaming wrapper with ``panel_size=None``
        which delegates to the existing JIT unchanged.
      * ``(False, int)``   — K1 + K3 would claim too large a share of the
        device budget, leaving no room for tile transients. Caller should keep
        the kernels on host; panel_size is the axis-1 slab width the consumer
        should stream in per call.

    ``fraction`` is the per-device share (of probed free memory) above which
    we switch to streaming. 0.15 leaves >= 80 % of the budget for tile
    transients (K1_transient / scratch / D / phi), which matches the scratch
    profile observed empirically on A100-80G at n_fused ~25k.
    """
    total = _array_nbytes(u1) + _array_nbytes(u3)
    free_bytes = _get_local_device_free_bytes(device)
    if total <= int(free_bytes * fraction):
        return True, None

    # Streaming: pick panel_size so each K1 panel claims at most ~0.08 of
    # the budget (generous head-room for device_put + padded/scannable
    # transients inside the JIT, which bloat ~3x per panel).
    n_fused = u1.shape[1]
    panel_bytes_cap = int(free_bytes * 0.08)
    # K1 panel element stride is n_fused * 3 * 8 bytes per axis-1 column;
    # K3 panel is smaller (no c component) so K1 dominates.
    bytes_per_axis1 = u1.shape[0] * 3 * 8
    panel_size = max(1024, panel_bytes_cap // max(bytes_per_axis1, 1))
    panel_size = min(int(panel_size), n_fused)
    panel_blk = _panel_blk_overrides()[0]
    if panel_blk is not None:
        panel_size = min(panel_size, max(1, int(panel_blk)))
        logger.info(
            "PYTC_PANEL_BLK=%d capping K-stream panel_size=%d (autotuned=%d)",
            panel_blk, panel_size,
            min(int(max(1024, panel_bytes_cap // max(bytes_per_axis1, 1))),
                n_fused),
        )
    return False, panel_size


def _pad_axis(arr, axis, target):
    """Pad one axis of an array with zeros up to ``target``."""
    cur = arr.shape[axis]
    if cur == target:
        return jnp.asarray(arr)
    if cur > target:
        raise ValueError(f"cannot pad axis-{axis} from {cur} down to {target}")
    pad_cfg = [(0, 0)] * arr.ndim
    pad_cfg[axis] = (0, target - cur)
    return jnp.pad(jnp.asarray(arr), pad_cfg)


def _normalize_panel_layout(panel_layout):
    """Normalize the internal tiled-axis layout selector."""
    layout = "pr" if panel_layout is None else str(panel_layout)
    if layout not in ("pr", "qr", "ps"):
        raise ValueError(f"unsupported panel_layout={layout!r}")
    return layout


def _transpose_panel_layout(panel_layout):
    """Return the layout required for the transpose partner tile."""
    layout = _normalize_panel_layout(panel_layout)
    if layout == "qr":
        return "ps"
    if layout == "ps":
        return "qr"
    return layout


def trim_panel(tc, panel_layout, actual_a, actual_b):
    """Strip JIT padding from the two variable axes of a tile.

    When ``panel_size`` is used to stabilise XLA shapes, the raw kernel
    output may be larger than the true data on the padded axes.  This
    function returns a sliced view with those axes trimmed back to their
    actual extents.

    Parameters
    ----------
    tc : array-like
        Raw tile of shape ``(Np_pad, Nq_pad, Nr_pad, Ns_pad)`` — only the
        axes listed in *panel_layout* may exceed the actual data.
    panel_layout : str
        Which pair of axes was JIT-padded:

        * ``"pr"`` — axes 0 (p) and 2 (r)
        * ``"qr"`` — axes 1 (q) and 2 (r)
        * ``"ps"`` — axes 0 (p) and 3 (s)
    actual_a : int
        True extent of the *first* padded axis (p for ``"pr"``/``"ps"``,
        q for ``"qr"``).
    actual_b : int
        True extent of the *second* padded axis (r for ``"pr"``/``"qr"``,
        s for ``"ps"``).

    Returns
    -------
    ndarray
        View of *tc* with the padded axes sliced to *actual_a* and
        *actual_b* respectively; all other axes are untouched.
    """
    layout = _normalize_panel_layout(panel_layout)
    if layout == "pr":
        return tc[:actual_a, :, :actual_b, :]
    if layout == "qr":
        return tc[:, :actual_a, :actual_b, :]
    # layout == "ps"
    return tc[:actual_a, :, :, :actual_b]


def _compute_2b_shard(phi, grad_phi, grid, weights, jastrow_params, jastrow_factor, ranges, batch_size):
    """Compute K terms for one device shard."""
    def get_size(r, size):
        start, stop, step = slice(*r).indices(size)
        return (stop - start + (step - 1)) // step
    
    n_orb = phi.shape[0]
    t_p, t_q, t_r, t_s = ranges
    
    Np = get_size(t_p, n_orb)
    Nq = get_size(t_q, n_orb)
    Nr = get_size(t_r, n_orb)
    Ns = get_size(t_s, n_orb)
    
    # Compute K1 (nabla on p)
    slices = tuple(slice(*r) for r in ranges)
    
    k1_raw = kmat_jax.calc_K1(
        phi, grad_phi,
        jastrow_factor, jastrow_params,
        grid, weights,
        ranges=slices,
        batch_size=batch_size
    )
    k1 = k1_raw.reshape(Np, Nq, Nr, Ns)
    
    # Compute K2 (nabla on q)
    if t_p == t_q:
        # If p and q ranges are identical, K2 is just K1 with p,q swapped
        k2 = k1.transpose(1, 0, 2, 3)
    else:
        ranges_k2 = (slices[1], slices[0], slices[2], slices[3])
        k2_raw = kmat_jax.calc_K1(
            phi, grad_phi,
            jastrow_factor, jastrow_params,
            grid, weights,
            ranges=ranges_k2,
            batch_size=batch_size
        )
        # Result is (Nq, Np, Nr, Ns), transpose to (Np, Nq, Nr, Ns)
        k2 = k2_raw.reshape(Nq, Np, Nr, Ns).transpose(1, 0, 2, 3)
        
    k3_raw = kmat_jax.calc_K3(
        phi, jastrow_factor, jastrow_params,
        grid, weights,
        ranges=slices,
        batch_size=batch_size
    )
    k3 = k3_raw.reshape(Np, Nq, Nr, Ns)
    
    result_local = 0.5 * (k1 - k2 + k3)
    
    result_sum = jax.lax.psum(result_local, axis_name='devices')
    
    return result_sum

@struct.dataclass
class TC:
    """JAX implementation of Transcorrelated method using flax dataclass.
    
    Attributes:
        grid_points: Grid points for numerical integration (N_grid, 3)
        weights: Grid weights (N_grid,)
        phi: Basis functions evaluated on grid (N_orb, N_grid)
        grad_phi: Basis function gradients on grid (N_orb, N_grid, 3)
        n_orb: Number of orbitals (static)
        grid_lvl: Grid level (static)
        jastrow_factor: Jastrow factor instance (PyTree)
        mo_coeff: Molecular orbital coefficients (N_ao, N_orb)
    """
    grid_points: jnp.ndarray
    weights: jnp.ndarray
    phi: jnp.ndarray
    grad_phi: jnp.ndarray
    n_orb: int = struct.field(pytree_node=False)
    grid_lvl: int = struct.field(pytree_node=False)
    jastrow_factor: Any = struct.field(pytree_node=True)
    mo_coeff: jnp.ndarray = struct.field(default=None)
    nocc: int = struct.field(pytree_node=False, default=None)

    @classmethod
    def from_pyscf(cls, mf, jastrow_factor, mo_coeff=None, grid_lvl=2, grid_chunk_size=None):
        """Initialize TC object from PySCF mean-field object.

        Args:
            mf: PySCF mean-field object
            jastrow_factor: JAX Jastrow factor instance
            mo_coeff: Optional molecular orbital coefficients
            grid_lvl: Grid level for numerical integration
            grid_chunk_size: Number of grid points to transform to MO basis
                at a time (see below). None (default) auto-sizes from a
                fixed ~2GiB per-chunk gradient-buffer target.

        Returns:
            TC: Initialized TC object
        """
        mol = mf.mol
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        n_orb = mo_coeff.shape[1]
        nocc = int(np.sum(mf.mo_occ > 0))

        logger.info(f"TC: Initializing grid with level {grid_lvl}")
        start_time = time.perf_counter()
        grids = dft.gen_grid.Grids(mol)
        grids.level = grid_lvl
        grids.build()
        logger.debug(f"TC: Grid initialized in {time.perf_counter() - start_time:.3f} seconds")

        grid_points = jnp.asarray(grids.coords)
        weights = jnp.asarray(grids.weights)

        logger.info(f"TC: Evaluating basis on grid")
        start_time = time.perf_counter()
        ao = dft.numint.eval_ao(mol, grids.coords, deriv=1)
        ao_values = ao[0].T  # (N_ao, N_grid)
        ao_gradients = ao[1:4].transpose(2, 1, 0)  # (N_ao, N_grid, 3)
        logger.debug(f"TC: AO basis evaluated in {time.perf_counter() - start_time:.3f} seconds")

        # Chunk the grid transform so the full AO-gradient intermediate is
        # never held on-device (it's AO-basis-sized and unsharded).
        logger.info(f"TC: Transforming to MO basis (GPU, chunked)")
        start_time = time.perf_counter()

        mo_coeff_jax = jnp.asarray(mo_coeff)
        n_mo = mo_coeff.shape[1]
        n_ao = mo_coeff.shape[0]
        n_grid = grid_points.shape[0]

        if grid_chunk_size is None:
            # Auto-size from a fixed ~2GiB per-chunk target.
            target_chunk_bytes = 2 * 1024 ** 3
            bytes_per_grid_point = n_ao * 3 * ao_gradients.dtype.itemsize
            grid_chunk_size = max(1, target_chunk_bytes // bytes_per_grid_point)
        else:
            grid_chunk_size = int(grid_chunk_size)
            if grid_chunk_size < 1:
                raise ValueError(
                    f"grid_chunk_size must be a positive integer, got {grid_chunk_size!r}"
                )
        grid_chunk_size = min(grid_chunk_size, n_grid)

        phi_chunks = []
        grad_phi_chunks = []
        for g0 in range(0, n_grid, grid_chunk_size):
            g1 = min(g0 + grid_chunk_size, n_grid)

            ao_values_chunk = jnp.asarray(ao_values[:, g0:g1])
            phi_chunks.append(jnp.matmul(mo_coeff_jax.T, ao_values_chunk))

            ao_gradients_chunk = jnp.asarray(ao_gradients[:, g0:g1, :])
            n_chunk = g1 - g0
            ao_grad_reshaped = ao_gradients_chunk.reshape(n_ao, -1)
            grad_phi_reshaped = jnp.matmul(mo_coeff_jax.T, ao_grad_reshaped)
            grad_phi_chunks.append(grad_phi_reshaped.reshape(n_mo, n_chunk, 3))

        phi = jnp.concatenate(phi_chunks, axis=1)
        grad_phi = jnp.concatenate(grad_phi_chunks, axis=1)

        logger.debug(f"TC: MO basis transformed in {time.perf_counter() - start_time:.3f} seconds")
        
        return cls(
            grid_points=grid_points,
            weights=weights,
            phi=phi,
            grad_phi=grad_phi,
            n_orb=n_orb,
            grid_lvl=grid_lvl,
            jastrow_factor=jastrow_factor,
            mo_coeff=jnp.asarray(mo_coeff),
            nocc=nocc
        )
    
    def _get_block_ranges(self, block_str):
        """Parse block string into slice ranges.
        
        block_str is expected to be in chemists' notation (p, q, r, s),
        where p, q share coordinate 1 and r, s share coordinate 2.
        
        Returns ranges in the order (p, q, r, s) expected by calc_K1.
        """
        if self.nocc is None:
            raise ValueError("nocc must be set to use block_str")
        
        ranges_list = []
        for char in block_str:
            if char == 'o':
                ranges_list.append(slice(0, self.nocc))
            elif char == 'v':
                ranges_list.append(slice(self.nocc, self.n_orb))
            elif char == 'g':
                ranges_list.append(slice(0, self.n_orb))
            else:
                raise ValueError(f"Invalid block character: {char}")
        
        # block_str indices: 0->p, 1->q, 2->r, 3->s
        # calc_K1 expects: (p, q, r, s)
        if len(ranges_list) == 4:
            p = ranges_list[0]
            q = ranges_list[1]
            r = ranges_list[2]
            s = ranges_list[3]
        elif len(ranges_list) == 2:
            p = ranges_list[0]
            q = ranges_list[1]
            r = ranges_list[0]
            s = ranges_list[1]
        else:
            raise ValueError("block_str must have 2 or 4 characters")
        
        return (p, q, r, s)

    def get_2b(self, jastrow_params, block_str=None, ranges=None, batch_size=1000):
        """Calculate TC correction terms (K1 + K2 + K3) with multi-GPU support.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor
            block_str: Optional string specifying the block (e.g. 'oovv')
            ranges: Optional tuple of slices (slice_p, slice_q, slice_r, slice_s)
            
        Returns:
            jnp.ndarray: The TC correction term.
                         If block_str/ranges is provided, returns the raw block (Np, Nr, Nq, Ns).
                         Otherwise, returns the full symmetrized correction (N, N, N, N).
        """
        start_time = time.perf_counter()
        logger.debug("Starting TC.get_2b")
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.grid_points.shape[0]

        remainder = n_grid % n_devices
        padding = (n_devices - remainder) if remainder != 0 else 0
        if padding > 0:
            padded_grid_points = np.pad(np.asarray(self.grid_points), ((0, padding), (0, 0)))
            padded_weights = np.pad(np.asarray(self.weights), ((0, padding),))
            padded_phi = np.pad(np.asarray(self.phi), ((0, 0), (0, padding)))
            padded_grad_phi = np.pad(np.asarray(self.grad_phi), ((0, 0), (0, padding), (0, 0)))
        else:
            padded_grid_points = np.asarray(self.grid_points)
            padded_weights = np.asarray(self.weights)
            padded_phi = np.asarray(self.phi)
            padded_grad_phi = np.asarray(self.grad_phi)

        mesh = sharding_core.create_1d_mesh(devices=devices, axis_name='devices')
        rep_sharding = sharding_core.get_replicated_sharding(mesh)
        grid_sharding = NamedSharding(mesh, P('devices', None))
        weights_sharding = NamedSharding(mesh, P('devices'))
        phi_sharding = NamedSharding(mesh, P(None, 'devices'))
        grad_sharding = NamedSharding(mesh, P(None, 'devices', None))

        sharded_grid = jax.device_put(padded_grid_points, grid_sharding)
        sharded_weights = jax.device_put(padded_weights, weights_sharding)
        sharded_phi = jax.device_put(padded_phi, phi_sharding)
        sharded_grad_phi = jax.device_put(padded_grad_phi, grad_sharding)
        params_rep = jax.tree_util.tree_map(
            lambda x: jax.device_put(np.asarray(x), rep_sharding), jastrow_params
        )

        def _run_sharded(ranges_local):
            @shard_map(
                mesh=mesh,
                in_specs=(P(None, 'devices'), P(None, 'devices', None), P('devices', None), P('devices'), P()),
                out_specs=P(),
                check_vma=False,
            )
            def _compute(phi_shard, grad_shard, grid_shard, weights_shard, params):
                return _compute_2b_shard(
                    phi_shard, grad_shard, grid_shard, weights_shard,
                    params, self.jastrow_factor, ranges_local, batch_size
                )

            return _compute(sharded_phi, sharded_grad_phi, sharded_grid, sharded_weights, params_rep)
        
        if ranges is None:
            full_slice = slice(None)
            ranges = (full_slice, full_slice, full_slice, full_slice)
            
        # Convert slices to hashable tuples for static closure args
        ranges_tuple = tuple((s.start, s.stop, s.step) for s in ranges)

        result = _run_sharded(ranges_tuple)
        
        slice_p, slice_q, slice_r, slice_s = ranges
        
        if slice_p == slice_r and slice_q == slice_s:
            result += result.transpose(2, 3, 0, 1)
        else:
            ranges_T = (slice_r, slice_s, slice_p, slice_q)
            ranges_T_tuple = tuple((s.start, s.stop, s.step) for s in ranges_T)
            
            result_T = _run_sharded(ranges_T_tuple)
            
            result += jax.lax.transpose(result_T, (2, 3, 0, 1))
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"TC.get_2b completed in {total_time:.4f} s")
        return -result

    def get_1b_fock(self, jastrow_params, dm1=None):
        """Get one-body Fock matrix correction (AO basis)."""
        return jnp.zeros((self.n_orb, self.n_orb))

    def get_2b_fock(self, jastrow_params, dm1, T=None):
        """Get 2-body Fock matrix correction.
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix
            T: Optional cached 2-body tensor (N, N, N, N).
        """
        if T is None:
            k_2b = self.get_2b(jastrow_params)
            T = k_2b
            
        # Coulomb-like contribution
        # \sum_{r,s} T_{pqrs} P_{rs}
        J_mat = jnp.einsum('pqrs,rs->pq', T, dm1)
        # Exchange-like contribution
        # \sum_{r,s} T_{psrq} P_{rs}
        K_mat = jnp.einsum('psrq,rs->pq', T, dm1)
        
        return J_mat - 0.5 * K_mat


    def get_3b_fock(self, jastrow_params, dm1):
        """Get 3-body Fock matrix correction (on-the-fly).
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis)
            
        Returns:
            Fock matrix contribution (N, N)
        """
        density_g = jnp.einsum('mg,ng,mn->g', self.phi, self.phi, dm1)
        
        N_grid = self.grid_points.shape[0]
        batch_size = 1000
        
        def compute_W_batch(r_batch):
            def inner_scan(carry, chunk_idx):
                start = chunk_idx * batch_size
                end = jnp.minimum(start + batch_size, N_grid)
                
                slice_len = batch_size
                r2_chunk = jax.lax.dynamic_slice(self.grid_points, (start, 0), (slice_len, 3))
                w_chunk = jax.lax.dynamic_slice(self.weights, (start,), (slice_len,))
                density_chunk = jax.lax.dynamic_slice(density_g, (start,), (slice_len,))
                
                # grad(r_batch, r2_chunk) -> (B, B_inner, 3)
                grads = self.jastrow_factor.grad_r_batch(r_batch, r2_chunk, jastrow_params)
                
                # sum_j w_j density_j grad_ij
                weighted_grads = grads * (w_chunk * density_chunk)[None, :, None]
                chunk_sum = jnp.sum(weighted_grads, axis=1)
                
                return carry + chunk_sum, None
                
            n_chunks = (N_grid + batch_size - 1) // batch_size
            W_batch, _ = jax.lax.scan(inner_scan, jnp.zeros((r_batch.shape[0], 3)), jnp.arange(n_chunks))
            return W_batch

        n_batches = (N_grid + batch_size - 1) // batch_size
        
        # Pad grid arrays to be divisible by batch_size to avoid slicing issues
        padded_size = n_batches * batch_size
        padding = padded_size - N_grid
        if padding > 0:
            grid_padded = jnp.pad(self.grid_points, ((0, padding), (0, 0)))
            weights_padded = jnp.pad(self.weights, ((0, padding),))
            density_g_padded = jnp.pad(density_g, ((0, padding),))
        else:
            grid_padded = self.grid_points
            weights_padded = self.weights
            density_g_padded = density_g
            
        def compute_W_batch_padded(r_batch, grid_p, weights_p, density_p):
            def inner_scan_p(carry, chunk_idx):
                start = chunk_idx * batch_size
                r2_chunk = jax.lax.dynamic_slice(grid_p, (start, 0), (batch_size, 3))
                w_chunk = jax.lax.dynamic_slice(weights_p, (start,), (batch_size,))
                density_chunk = jax.lax.dynamic_slice(density_p, (start,), (batch_size,))
                
                grads = self.jastrow_factor.grad_r_batch(r_batch, r2_chunk, jastrow_params)
                weighted_grads = grads * (w_chunk * density_chunk)[None, :, None]
                chunk_sum = jnp.sum(weighted_grads, axis=1)
                return carry + chunk_sum, None
            
            n_chunks = (grid_p.shape[0]) // batch_size
            W_batch, _ = jax.lax.scan(inner_scan_p, jnp.zeros((r_batch.shape[0], 3), dtype=jnp.complex128), jnp.arange(n_chunks))
            return W_batch

        def outer_scan(carry, batch_idx):
            start = batch_idx * batch_size
            r_batch = jax.lax.dynamic_slice(grid_padded, (start, 0), (batch_size, 3))
            W_batch = compute_W_batch_padded(r_batch, grid_padded, weights_padded, density_g_padded)
            return carry, W_batch 
            
        _, W_all = jax.lax.scan(outer_scan, None, jnp.arange(n_batches))
        W_all = W_all.reshape(-1, 3)[:N_grid]
        
        # 3. Compute V_3b_direct(r) = |W(r)|^2
        V_3b_g = jnp.sum(W_all**2, axis=1) # (N_grid,)
        
        # 4. Integrate to get Fock matrix elements
        # F_mn = \int \phi_m(r) \phi_n(r) V_3b(r) dr
        #      = \sum_g w_g \phi_m(g) \phi_n(g) V_3b(g)
        
        # (N_orb, N_grid) * (N_grid,) -> (N_orb, N_grid)
        weighted_phi = self.phi * (self.weights * V_3b_g)[None, :]
        F_3b = jnp.dot(weighted_phi, self.phi.T)
        
        return F_3b

    def get_3b_fock_full(self, jastrow_params, dm1):
        """Get 3-body Fock matrix correction (full tensor calculation).
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis)
            
        Returns:
            Fock matrix contribution (N, N)
        """
        density_g = jnp.einsum('mg,ng,mn->g', self.phi, self.phi, dm1)
        
        # 2. Compute W(r) on grid using full broadcasting
        # W(r_i) = \sum_j w_j rho(r_j) \nabla_i u(r_i, r_j)
        
        # grads: (N_grid, N_grid, 3)
        # This might be memory intensive for large grids!
        grads = self.jastrow_factor.grad_r_batch(self.grid_points, self.grid_points, jastrow_params)
        
        # Weighted density: (N_grid,)
        w_density = self.weights * density_g
        
        # Contract: (N_i, N_j, 3) * (N_j,) -> (N_i, 3)
        W_all = jnp.einsum('ijc,j->ic', grads, w_density)
        
        # 3. Compute V_3b_direct(r) = |W(r)|^2
        V_3b_g = jnp.sum(W_all**2, axis=1) # (N_grid,)
        
        weighted_phi = self.phi * (self.weights * V_3b_g)[None, :]
        F_3b = jnp.dot(weighted_phi, self.phi.T)
        
        return F_3b




@struct.dataclass
class ISDFTC(TC):
    """JAX implementation of Transcorrelated method using ISDF.
    
    Attributes:
        xi_rho: ISDF coefficients for density (N_fused, N_grid)
        xi_grad: ISDF coefficients for gradients (N_fused, N_grid, 3)
        pivots: ISDF pivot indices (N_fused,)
        phi: ISDF basis for density (Nb, N_fused)
        grad_phi: ISDF basis for gradients (Nb, N_fused, 3)
        isdf_kernels: Dictionary storing precomputed kernels (U1, U3)
        kmat_kernel_mode: Resolved K1/K3 construction route (``direct`` or
            ``aux-recovery``), excluded from the numeric pytree.
    """
    xi_phi: jnp.ndarray = struct.field(default=None)
    xi_grad: jnp.ndarray = struct.field(default=None)
    pivots: jnp.ndarray = struct.field(default=None)
    phi_isdf: jnp.ndarray = struct.field(default=None)
    grad_phi_isdf: jnp.ndarray = struct.field(default=None)
    isdf_kernels: dict = struct.field(default=None, pytree_node=True)
    is_incore: bool = struct.field(default=False, pytree_node=False)
    save_path: str = struct.field(default=None, pytree_node=False)
    kmat_kernel_mode: str = struct.field(default=None, pytree_node=False)

    def _get_fixed_rank_block_size(self):
        """Return a fixed rank_block_size that is safe for all orbital slices.

        Uses worst-case dimensions ``(n_orb, n_orb)`` so the chosen
        power-of-2 is the smallest (most conservative) over the CCSD phases,
        which avoids JIT recompilation when ``(Np, Nq)`` changes across
        tiles. Resident-state-aware: accounts for K1/K3 being resident or
        streamed via the same ``_choose_tc_kernel_strategy`` used by the
        per-device TC cache, plus D + phi + tile accumulator.

        Cache key includes the streaming decision so a change in K1/K3
        residency (e.g. resident on small n_fused, streaming on large
        n_fused) picks the appropriate rbs without stale-cache issues.

        Returns ``None`` when ``phi_isdf`` is not yet available (pre-ISDF).
        """
        if self.phi_isdf is None:
            return None
        N_fused = self.phi_isdf.shape[1]

        from pytc.utils.gpu_memory import (
            adaptive_rank_block_size, estimate_tc_contract_resident_bytes,
            _get_gpu_free_bytes,
        )

        streaming = False
        k_stream_panel = None
        kernels = getattr(self, "isdf_kernels", None)
        if kernels is not None and "K1_kernel" in kernels and "K3_kernel" in kernels:
            try:
                devices = jax.local_devices()
                # Pick the device with the smallest free budget so the rbs
                # is safe on every card.
                def _free(d):
                    stats = d.memory_stats()
                    return int(stats.get('bytes_limit', 0)) - int(stats.get('bytes_in_use', 0))
                constrained = min(devices, key=_free) if devices else None
                if constrained is not None:
                    resident_flag, k_stream_panel = _choose_tc_kernel_strategy(
                        constrained,
                        kernels["K1_kernel"], kernels["K3_kernel"],
                    )
                    streaming = not resident_flag
            except Exception:
                # If probing fails, assume streaming (conservative: larger
                # resident estimate → smaller, safer rbs).
                streaming = True
                k_stream_panel = None

        panel_blk, gpu_max_memory_mb = _panel_blk_overrides()
        # Env overrides are part of the cache key so a same-process env change
        # is honoured without a stale hit. Build the full key once for both
        # lookup and store; repeated same-env calls then hit the cache.
        key = (int(self.n_orb), int(N_fused), bool(streaming),
               int(k_stream_panel or 0), panel_blk, gpu_max_memory_mb)
        if key not in _FIXED_RBS_CACHE:
            resident_bytes = estimate_tc_contract_resident_bytes(
                n_orb=self.n_orb, n_fused=N_fused,
                k_stream_panel=k_stream_panel if streaming else None,
            )
            rbs = adaptive_rank_block_size(
                self.n_orb, self.n_orb, N_fused,
                resident_bytes=resident_bytes,
                gpu_max_memory_mb=gpu_max_memory_mb,
            )
            if panel_blk is not None:
                rbs = min(rbs, max(1, int(panel_blk)))
            try:
                budget_gib = _get_gpu_free_bytes() / (1024 ** 3)
            except Exception:
                budget_gib = float('nan')
            logger.info(
                "  Fixed rank_block_size = %d "
                "(n_orb=%d, N_fused=%d, streaming=%s, panel=%s, "
                "resident=%.1f GiB, free=%.1f GiB%s%s)",
                rbs, self.n_orb, N_fused, streaming,
                k_stream_panel if streaming else "n/a",
                resident_bytes / (1024 ** 3),
                budget_gib,
                f", PYTC_PANEL_BLK={panel_blk}" if panel_blk is not None else "",
                f", PYTC_GPU_MAX_MEMORY_MB={gpu_max_memory_mb}"
                if gpu_max_memory_mb is not None else "",
            )
            _FIXED_RBS_CACHE[key] = rbs
        return _FIXED_RBS_CACHE[key]

    def _get_isdf_device_cache(self, kernels=None, device=None, *,
                               include_grad=False,
                               include_delta_u=False,
                               include_tc=False):
        """Return persistent ISDF operands resident on one device.

        The first call for a given ``(self, device)`` materialises
        ``phi_isdf`` (and optionally grad/TC/D) on ``device`` and stores
        them in the process-wide ``_ISDF_DEVICE_CACHE``.  A
        ``weakref.finalize`` is attached to ``self`` the first time we
        cache anything for it, so the moment ``self`` is GC'd the
        finaliser drops every ``(id(self), *)`` cache entry — releasing
        the device-resident buffers and preventing a recycled ``id()``
        from silently inheriting the old instance's cache.  See the
        ``_ISDF_DEVICE_CACHE`` docstring at module top.
        """
        if device is None:
            return None

        self_id = id(self)
        key = (self_id, getattr(device, "id", repr(device)))

        with _ISDF_DEVICE_CACHE_LOCK:
            cache = _ISDF_DEVICE_CACHE.get(key)
            if cache is None:
                cache = {
                    "phi_isdf": jax.device_put(np.asarray(self.phi_isdf), device),
                }
                _ISDF_DEVICE_CACHE[key] = cache
                # Register the finaliser exactly once per instance,
                # not per (instance, device) pair.
                if self_id not in _ISDF_DEVICE_CACHE_FINALISED:
                    _ISDF_DEVICE_CACHE_FINALISED.add(self_id)
                    weakref.finalize(
                        self, _evict_isdf_device_cache_for_id, self_id,
                    )

        if include_grad and "grad_phi_isdf" not in cache:
            cache["grad_phi_isdf"] = jax.device_put(np.asarray(self.grad_phi_isdf), device)

        if include_tc and kernels is not None:
            need_u1 = "K1_kernel" in kernels and "K1_kernel" not in cache
            need_u3 = "K3_kernel" in kernels and "K3_kernel" not in cache
            if need_u1 or need_u3:
                resident, panel_size = _choose_tc_kernel_strategy(
                    device, kernels["K1_kernel"], kernels["K3_kernel"],
                )
                if resident:
                    logger.debug(
                        "Caching TC kernels on device %s (K1=%.2f GiB, K3=%.2f GiB, resident)",
                        getattr(device, "id", "host"),
                        _array_nbytes(kernels["K1_kernel"]) / (1024.0 ** 3),
                        _array_nbytes(kernels["K3_kernel"]) / (1024.0 ** 3),
                    )
                    cache["K1_kernel"] = jax.device_put(np.asarray(kernels["K1_kernel"]), device)
                    cache["K3_kernel"] = jax.device_put(np.asarray(kernels["K3_kernel"]), device)
                    cache["K_stream_panel"] = None
                else:
                    logger.info(
                        "Streaming TC kernels from host on device %s "
                        "(K1=%.2f GiB, K3=%.2f GiB, panel_size=%d)",
                        getattr(device, "id", "host"),
                        _array_nbytes(kernels["K1_kernel"]) / (1024.0 ** 3),
                        _array_nbytes(kernels["K3_kernel"]) / (1024.0 ** 3),
                        panel_size,
                    )
                    cache["K1_kernel"] = np.asarray(kernels["K1_kernel"])
                    cache["K3_kernel"] = np.asarray(kernels["K3_kernel"])
                    cache["K_stream_panel"] = int(panel_size)

        if include_delta_u and kernels is not None and "D" in kernels and "D" not in cache:
            if _cache_d_on_device(device, kernels["D"]):
                logger.debug(
                    "Caching Delta U D kernel on device %s (%.2f GiB)",
                    getattr(device, "id", "host"),
                    _array_nbytes(kernels["D"]) / (1024.0 ** 3),
                )
                cache["D"] = jax.device_put(np.asarray(kernels["D"]), device)
            else:
                cache["D"] = None

        return cache

    @classmethod
    def from_tc(
        cls,
        tc_obj,
        n_rank=None,
        is_incore=False,
        save_path=None,
        ls_grid_batch_size=16384,
        fixed_pivots=None,
        batch_size=1,
        candidate_oversampling=2,
        n_topup=0,
    ):
        """Initialize ISDFTC object from TC object.
        
        Args:
            tc_obj: TC object
            n_rank: Rank for ISDF decomposition (default: N_grid // 4)
            is_incore: Whether to perform in-core decomposition
            save_path: Path to save ISDF kernels
            ls_grid_batch_size: Batch size for grid decomposition in isdf_decompose
            fixed_pivots: Optional fixed pivot sequence.
            batch_size: Exact columns retained per blocked pivot round. One
                preserves exact greedy pivot selection.
            candidate_oversampling: Candidate-pool multiplier for exact
                within-pool re-pivoting.
            n_topup: Final exact-greedy singleton pivots after blocked rounds.
            
        Returns:
            ISDFTC: Initialized ISDFTC object
        """
        from . import df
        
        if n_rank is None:
            n_rank = tc_obj.grid_points.shape[0] // 4
            
        logger.info("ISDFTC.from_tc: building ISDF decomposition")
        phi_isdf, xi_phi, grad_phi_isdf, xi_grad, pivots, actual_save_path = df.isdf_decompose(
            tc_obj.phi, tc_obj.grad_phi, n_rank, n_rank, weights=tc_obj.weights,
            is_incore=is_incore, save_path=save_path,
            grid_batch_size=ls_grid_batch_size, fixed_pivots=fixed_pivots,
            batch_size=batch_size,
            candidate_oversampling=candidate_oversampling, n_topup=n_topup,
        )
        
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
            xi_phi=xi_phi,
            xi_grad=xi_grad,
            pivots=pivots,
            phi_isdf=phi_isdf,
            grad_phi_isdf=grad_phi_isdf,
            isdf_kernels=None,
            is_incore=is_incore,
            save_path=actual_save_path
        )

    def compute_kmat_kernels(
        self,
        jastrow_params,
        batch_size=1024,
        host_grid_block_size=None,
        r2_tile_size=None,
        mesh_shape=None,
        gpu_budget_bytes=None,
    ):
        """Compute K1 and K3 kernels with a 2D (k, g)-mesh + double-host-tiled
        algorithm.

        Uses a mesh of shape ``(m_k, m_g)`` with ``m_k * m_g == n_devices``:
        the n_fused (k) axis of the r1 xi inputs and of the K1/K3 outputs is
        partitioned across ``m_k`` devices; the r2 grid axis of ``xi_phi_r2``
        (and grid_r2 / weights_r2) is partitioned across ``m_g`` devices and
        summed via ``psum`` on the ``g_ax``.  The double host loop iterates
        over r1 host blocks and r2 tiles so that the per-device working set
        fits an explicit memory budget.

        Per-device memory model (approx., float64):
            K1 + scratch + psum buffer: ~ alpha * 3 * n_rank^2 / m_k * 8
            xi_phi_r2 tile (g-shard):        n_rank * r2_tile_size / m_g * 8
            r1 block (phi + 3*grad, k-shard): 4 * n_rank * host_grid_block_size / m_k * 8

        Parameters
        ----------
        mesh_shape : (m_k, m_g) or None
            Device factorization.  If ``None``, auto-chosen to minimise per-device
            K1 memory, then preferring the smallest ``m_g`` (fewer psum comms).
        host_grid_block_size : int or None
            r1 host-loop tile size.  If ``None``, auto-chosen from the budget.
        r2_tile_size : int or None
            r2 host-loop tile size.  If ``None``, auto-chosen from the budget.
            Rounded up to a multiple of ``m_g``.
        gpu_budget_bytes : int or None
            Per-device memory budget used in auto-selection.  If ``None``,
            probed via ``pytc.utils.gpu_memory._get_gpu_free_bytes()``.

        Returns
        -------
        dict
            ``{'K1_kernel': jnp.ndarray (n_rank, n_rank, 3),
               'K3_kernel': jnp.ndarray (n_rank, n_rank)}``.
        """
        from pytc.utils import gpu_memory
        from pytc.utils.prefetch import safe_hdf5_read

        n_devices = jax.local_device_count()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        devices = jax.local_devices()

        # --- Budget ----------------------------------------------------------
        if gpu_budget_bytes is None:
            try:
                gpu_budget_bytes = int(gpu_memory._get_gpu_free_bytes())
            except Exception:
                gpu_budget_bytes = 50 * (1024 ** 3)
        gpu_budget_bytes = max(int(gpu_budget_bytes), 1)

        # --- Per-device peak-memory model (bytes, float64) -------------------
        # Each line item maps to a specific tensor or transient inside the
        # shard_map'd K-kernel scan. Summed they give the "fixed" footprint
        # (independent of tile sizes); r1/r2 tile sizes add on top per-element.
        #
        # Peak occurs inside the shard_map while the scan is running, because
        # the on-device K1_accum / K3_accum stay resident across r1 blocks.
        # The post-shard_map accumulation (K1_accum = K1_accum + K1_contrib)
        # has a smaller peak because the scan-internal buffers are freed.
        def _k_kernel_fixed_bytes(m_k, m_g):
            k1_carry      = 3.0 * n_rank * n_rank / m_k * 8.0  # (n/m_k, n, 3)
            k1_transient  = k1_carry                           # .at.add transient inside scan
            k1_c_interm   = n_rank * n_rank / m_k * 8.0        # (n/m_k, n) matmul output per c
            k3_carry      = n_rank * n_rank / m_k * 8.0
            k3_transient  = k3_carry
            # Persistent on-device accumulators carried across r1 blocks:
            k1_accum      = 3.0 * n_rank * n_rank / m_k * 8.0
            k3_accum      = n_rank * n_rank / m_k * 8.0
            psum_buf      = k1_carry if m_g > 1 else 0.0
            return (k1_accum + k3_accum
                    + k1_carry + k1_transient + k1_c_interm
                    + k3_carry + k3_transient + psum_buf)

        # Per-unit elemental costs (bytes per grid point):
        #   r1 side: xi_phi_r1 + 3 xi_grad_r1 components, k-sharded.
        #   r2 side: xi_phi_r2 shard, g-sharded on axis 1.
        def _r1_bytes_per_unit(m_k):
            return 4.0 * n_rank / m_k * 8.0

        def _r2_bytes_per_unit(m_g):
            return n_rank / m_g * 8.0

        # --- Auto-pick mesh_shape --------------------------------------------
        if mesh_shape is None:
            usable = gpu_budget_bytes * (1.0 - gpu_memory.K_KERNEL_SAFETY_FRACTION)
            divisors = [d for d in range(1, n_devices + 1) if n_devices % d == 0]
            best = None  # (m_k, m_g, fixed_cost)
            for m_k_try in divisors:
                m_g_try = n_devices // m_k_try
                fixed = _k_kernel_fixed_bytes(m_k_try, m_g_try)
                # Require the fixed cost alone to leave room for at least
                # some r1/r2 tiles within the safety-adjusted budget.
                if fixed >= usable * 0.85:
                    continue
                # Rank by (m_g ascending, m_k ascending); m_g=1 avoids a
                # psum and an extra K1-sized buffer.
                key = (m_g_try, m_k_try)
                if best is None or key < (best[1], best[0]):
                    best = (m_k_try, m_g_try, fixed)
            if best is None:
                fixed_max = _k_kernel_fixed_bytes(n_devices, 1)
                raise RuntimeError(
                    f"compute_kmat_kernels: fixed K1/K3 footprint "
                    f"({fixed_max / 1024 ** 3:.1f} GiB at m_k={n_devices}) "
                    f"exceeds safety-adjusted budget "
                    f"({usable / 1024 ** 3:.1f} GiB = "
                    f"{gpu_budget_bytes / 1024 ** 3:.1f} GiB × "
                    f"(1 - {gpu_memory.K_KERNEL_SAFETY_FRACTION:.0%})). "
                    f"Use more devices, or lower K_KERNEL_SAFETY_FRACTION."
                )
            m_k, m_g, fixed_cost = best
        else:
            m_k, m_g = mesh_shape
            assert m_k * m_g == n_devices, (m_k, m_g, n_devices)
            fixed_cost = _k_kernel_fixed_bytes(m_k, m_g)

        # --- Solve for host tile sizes --------------------------------------
        # xi_phi_r2 is by far the largest single r2-side input; enforce the
        # K_KERNEL_SINGLE_INPUT_MAX_FRACTION cap on it so one replicated input
        # cannot eat the whole budget (forces host tiling when replication
        # becomes pathological, independently of the specific system size).
        tile_solve = gpu_memory.solve_tile_sizes(
            budget_bytes=gpu_budget_bytes,
            fixed_bytes=int(fixed_cost),
            r1_elem_bytes_per_unit=_r1_bytes_per_unit(m_k),
            r2_elem_bytes_per_unit=_r2_bytes_per_unit(m_g),
            r2_single_input_bytes_per_unit=_r2_bytes_per_unit(m_g),
            r1_upper=n_grid,
            r2_upper=n_grid,
        )
        if host_grid_block_size is None:
            host_grid_block_size = tile_solve["B_r1"]
        host_grid_block_size = min(host_grid_block_size, n_grid)

        if r2_tile_size is None:
            r2_tile_size = tile_solve["B_r2"]
        r2_tile_size = min(r2_tile_size, n_grid)
        # r2_tile_size must be divisible by m_g (round up).
        if r2_tile_size % m_g != 0:
            r2_tile_size = ((r2_tile_size + m_g - 1) // m_g) * m_g

        predicted_peak = (
            fixed_cost
            + host_grid_block_size * _r1_bytes_per_unit(m_k)
            + r2_tile_size * _r2_bytes_per_unit(m_g)
        )

        # --- Mesh, shardings -------------------------------------------------
        mesh = sharding_core.create_2d_mesh(
            devices, (m_k, m_g), axis_names=('k_ax', 'g_ax')
        )
        rep_sharding = NamedSharding(mesh, P())
        device_grid = np.asarray(mesh.devices)

        n_rank_padded = ((n_rank + m_k - 1) // m_k) * m_k
        pad_k = n_rank_padded - n_rank

        logger.info(
            "  compute_kmat_kernels: mesh=(m_k=%d, m_g=%d), n_fused=%d, "
            "n_rank_padded=%d, n_grid=%d, host_grid_block_size=%d, "
            "r2_tile_size=%d, budget=%.1f GiB, fixed=%.1f GiB, "
            "predicted_peak=%.1f GiB (safety=%.0f%%, "
            "single_input_max=%.0f%%)",
            m_k, m_g, n_rank, n_rank_padded, n_grid,
            host_grid_block_size, r2_tile_size,
            gpu_budget_bytes / (1024 ** 3),
            fixed_cost / (1024 ** 3),
            predicted_peak / (1024 ** 3),
            gpu_memory.K_KERNEL_SAFETY_FRACTION * 100,
            gpu_memory.K_KERNEL_SINGLE_INPUT_MAX_FRACTION * 100,
        )

        # --- Preload ISDF intermediates into host RAM once ------------------
        # The inner r1 / r2 loops each want a slice of xi_phi and xi_grad. If
        # those live in HDF5 on network storage, each loop iteration stalled
        # the GPU ~30-45 s reading from disk. Host RAM on typical nodes
        # (>= 400 GiB) easily absorbs the full (n_fused, n_grid) and
        # (n_fused, n_grid, 3) arrays (here ~20 GiB + ~60 GiB), so we pay the
        # I/O once and slice from RAM thereafter.
        if self.xi_phi is not None:
            xi_phi_host = np.asarray(self.xi_phi)
            xi_grad_host = np.asarray(self.xi_grad)
        elif self.save_path:
            t_io = time.perf_counter()
            with h5py.File(self.save_path, 'r') as f_xi:
                xi_phi_bytes = int(np.prod(f_xi['xi_phi'].shape)) * 8
                xi_grad_bytes = int(np.prod(f_xi['xi_grad'].shape)) * 8
                logger.info(
                    "  compute_kmat_kernels: preloading xi_phi (%.1f GiB) "
                    "and xi_grad (%.1f GiB) from %s into host RAM",
                    xi_phi_bytes / (1024 ** 3),
                    xi_grad_bytes / (1024 ** 3),
                    self.save_path,
                )
                xi_phi_host = np.asarray(f_xi['xi_phi'][:])
                xi_grad_host = np.asarray(f_xi['xi_grad'][:])
            logger.info(
                "  compute_kmat_kernels: preload done in %.1fs",
                time.perf_counter() - t_io,
            )
        else:
            raise RuntimeError(
                "compute_kmat_kernels: xi_phi/xi_grad not in-core and no save_path set"
            )

        # --- Helpers for loading + sharding ---------------------------------
        def _load_r1_block(g0, g1):
            cur_len = g1 - g0
            grid_block = np.asarray(self.grid_points[g0:g1])
            weights_block = np.asarray(self.weights[g0:g1])
            xi_phi_block = xi_phi_host[:, g0:g1]
            xi_grad_block = xi_grad_host[:, g0:g1, :]

            if cur_len < host_grid_block_size:
                pad = host_grid_block_size - cur_len
                grid_block = np.pad(grid_block, ((0, pad), (0, 0)))
                weights_block = np.pad(weights_block, ((0, pad),))
                xi_phi_block = np.pad(xi_phi_block, ((0, 0), (0, pad)))
                xi_grad_block = np.pad(xi_grad_block, ((0, 0), (0, pad), (0, 0)))

            if pad_k > 0:
                xi_phi_block = np.pad(xi_phi_block, ((0, pad_k), (0, 0)))
                xi_grad_block = np.pad(xi_grad_block, ((0, pad_k), (0, 0), (0, 0)))
            return grid_block, weights_block, xi_phi_block, xi_grad_block

        def _load_r2_tile(g0, g1):
            cur_len = g1 - g0
            grid_tile = np.asarray(self.grid_points[g0:g1])
            weights_tile = np.asarray(self.weights[g0:g1])
            xi_phi_tile = xi_phi_host[:, g0:g1]

            if cur_len < r2_tile_size:
                pad = r2_tile_size - cur_len
                grid_tile = np.pad(grid_tile, ((0, pad), (0, 0)))
                weights_tile = np.pad(weights_tile, ((0, pad),))
                xi_phi_tile = np.pad(xi_phi_tile, ((0, 0), (0, pad)))
            return grid_tile, weights_tile, xi_phi_tile

        def _build_k_sharded(np_arr):
            """Shard along axis 0 into m_k, replicate along g_ax."""
            chunks = np.split(np_arr, m_k, axis=0)
            single_dev_arrays = []
            for i_k in range(m_k):
                for i_g in range(m_g):
                    single_dev_arrays.append(
                        jax.device_put(chunks[i_k], device_grid[i_k, i_g])
                    )
            spec = P('k_ax', *([None] * (np_arr.ndim - 1)))
            sharding = NamedSharding(mesh, spec)
            return jax.make_array_from_single_device_arrays(
                np_arr.shape, sharding, single_dev_arrays
            )

        def _build_g_sharded(np_arr, axis):
            """Shard along ``axis`` into m_g, replicate along k_ax."""
            chunks = np.split(np_arr, m_g, axis=axis)
            single_dev_arrays = []
            for i_k in range(m_k):
                for i_g in range(m_g):
                    single_dev_arrays.append(
                        jax.device_put(chunks[i_g], device_grid[i_k, i_g])
                    )
            spec_tuple = [None] * np_arr.ndim
            spec_tuple[axis] = 'g_ax'
            sharding = NamedSharding(mesh, P(*spec_tuple))
            return jax.make_array_from_single_device_arrays(
                np_arr.shape, sharding, single_dev_arrays
            )

        # --- shard_map body --------------------------------------------------
        jastrow_factor = self.jastrow_factor

        @shard_map(
            mesh=mesh,
            in_specs=(
                P(), P(),                       # grid_r1, weights_r1 (replicated)
                P('k_ax', None),                # xi_phi_r1 (k-sharded, g-replicated)
                P('k_ax', None, None),          # xi_grad_r1
                P('g_ax', None),                # grid_r2 (g-sharded on axis 0)
                P('g_ax'),                      # weights_r2
                P(None, 'g_ax'),                # xi_phi_r2 (g-sharded on axis 1)
                P(),                            # params
            ),
            out_specs=(P('k_ax', None, None), P('k_ax', None)),
            check_vma=False,
        )
        def sharded_compute(
            grid_r1_block, weights_r1_block, xi_phi_r1_block, xi_grad_r1_block,
            grid_r2_tile, weights_r2_tile, xi_phi_r2_tile, params
        ):
            k1_local = kmat_jax.calc_K1_kernel(
                xi_grad_r1_block, xi_phi_r2_tile,
                weights_r1_block, weights_r2_tile,
                jastrow_factor, params,
                grid_r1_block, grid_r2_tile, batch_size,
            )
            k3_local = kmat_jax.calc_K3_kernel(
                xi_phi_r1_block, xi_phi_r2_tile,
                weights_r1_block, weights_r2_tile,
                jastrow_factor, params,
                grid_r1_block, grid_r2_tile, batch_size,
            )
            # psum over g_ax only; k_ax stays sharded. When m_g == 1 this is a
            # no-op.
            return (
                jax.lax.psum(k1_local, 'g_ax'),
                jax.lax.psum(k3_local, 'g_ax'),
            )

        params_rep = jax.tree_util.tree_map(
            lambda x: jax.device_put(np.asarray(x), rep_sharding), jastrow_params
        )

        # --- On-device k-sharded accumulators -------------------------------
        # Kept resident across all (r1_block, r2_tile) iterations. Per-block
        # contributions are added on-device via JAX; nothing is pulled to host
        # until the very end. This eliminates ~4 GB + ~1.3 GB of D2H traffic
        # per r1 block and the associated host numpy += pass (which was the
        # dominant idle-gap phase on the nvidia-smi timeline).
        def _make_k_sharded_zeros(shape):
            per_dev_shape = (shape[0] // m_k,) + tuple(shape[1:])
            single_dev_arrays = []
            for i_k in range(m_k):
                for i_g in range(m_g):
                    single_dev_arrays.append(
                        jax.device_put(
                            np.zeros(per_dev_shape), device_grid[i_k, i_g]
                        )
                    )
            spec = P('k_ax', *([None] * (len(shape) - 1)))
            sharding = NamedSharding(mesh, spec)
            return jax.make_array_from_single_device_arrays(
                shape, sharding, single_dev_arrays
            )

        K1_accum = _make_k_sharded_zeros((n_rank_padded, n_rank, 3))
        K3_accum = _make_k_sharded_zeros((n_rank_padded, n_rank))

        n_r2_tiles = (n_grid + r2_tile_size - 1) // r2_tile_size
        n_r1_blocks = (n_grid + host_grid_block_size - 1) // host_grid_block_size

        for j, g_r2_0 in enumerate(range(0, n_grid, r2_tile_size)):
            g_r2_1 = min(g_r2_0 + r2_tile_size, n_grid)
            logger.info(
                "    r2 tile %d/%d [%d:%d]",
                j + 1, n_r2_tiles, g_r2_0, g_r2_1
            )
            grid_r2_np, weights_r2_np, xi_phi_r2_np = _load_r2_tile(g_r2_0, g_r2_1)
            sh_grid_r2 = _build_g_sharded(grid_r2_np, axis=0)
            sh_weights_r2 = _build_g_sharded(weights_r2_np, axis=0)
            sh_xi_phi_r2 = _build_g_sharded(xi_phi_r2_np, axis=1)

            for i, g_r1_0 in enumerate(range(0, n_grid, host_grid_block_size)):
                g_r1_1 = min(g_r1_0 + host_grid_block_size, n_grid)
                logger.debug(
                    "      r1 block %d/%d [%d:%d]",
                    i + 1, n_r1_blocks, g_r1_0, g_r1_1
                )
                grid_r1_np, weights_r1_np, xi_phi_r1_np, xi_grad_r1_np = \
                    _load_r1_block(g_r1_0, g_r1_1)
                sh_grid_r1 = jax.device_put(grid_r1_np, rep_sharding)
                sh_weights_r1 = jax.device_put(weights_r1_np, rep_sharding)
                sh_xi_phi_r1 = _build_k_sharded(xi_phi_r1_np)
                sh_xi_grad_r1 = _build_k_sharded(xi_grad_r1_np)

                K1_contrib, K3_contrib = sharded_compute(
                    sh_grid_r1, sh_weights_r1,
                    sh_xi_phi_r1, sh_xi_grad_r1,
                    sh_grid_r2, sh_weights_r2, sh_xi_phi_r2,
                    params_rep,
                )
                K1_accum = K1_accum + K1_contrib
                K3_accum = K3_accum + K3_contrib
                del K1_contrib, K3_contrib
                del sh_grid_r1, sh_weights_r1, sh_xi_phi_r1, sh_xi_grad_r1

            del sh_grid_r2, sh_weights_r2, sh_xi_phi_r2

        # Final D2H gather — single materialisation at the end of the loop.
        K1_kernel_padded = np.asarray(K1_accum)
        K3_kernel_padded = np.asarray(K3_accum)
        del K1_accum, K3_accum

        # Strip k-axis padding (zero by construction).
        K1_kernel = K1_kernel_padded[:n_rank]
        K3_kernel = K3_kernel_padded[:n_rank]
        return {'K1_kernel': jnp.asarray(K1_kernel),
                'K3_kernel': jnp.asarray(K3_kernel)}

    def _compute_L_aux(self, jastrow_params, batch_size=1024, save_path=None,
                       host_grid_block_size=None, include_h_aux=False,
                       use_laux_fast_grad=False):
        """Compute L_aux, and optionally its exact squared-gradient companion.

        When ``include_h_aux`` is true this also forms
        ``H_aux[a,g] = sum_h xi_phi[a,h] w_h |grad u(g,h)|^2`` in the same
        double-grid pass.  ``H_aux`` lets the K3 ISDF kernel be recovered by
        one-grid contraction; paired with ``L_aux`` it likewise recovers K1.
        The default is deliberately unchanged for existing callers.

        ``use_laux_fast_grad`` selects a Jastrow-provided derivative fast path
        for this construction only.  It is opt-in so existing caches and all
        non-L_aux derivative callers retain the ordinary implementation.

        """
        n_devices = jax.local_device_count()
        devices = jax.local_devices()
        n_grid = self.grid_points.shape[0]
        n_rank = self.phi_isdf.shape[1]
        
        if host_grid_block_size is None:
            host_grid_block_size = n_grid
            
        L_aux_out = None
        H_aux_out = None
        f_out = None
        if save_path:
            f_out = h5py.File(save_path, 'a')
            if 'L_aux' in f_out: del f_out['L_aux']
            L_aux_out = f_out.create_dataset('L_aux', (n_rank, n_grid, 3), dtype='f8')
            if include_h_aux:
                if 'H_aux' in f_out: del f_out['H_aux']
                H_aux_out = f_out.create_dataset('H_aux', (n_rank, n_grid), dtype='f8')
        else:
            L_aux_out = np.zeros((n_rank, n_grid, 3))
            if include_h_aux:
                H_aux_out = np.zeros((n_rank, n_grid))
            
        xi_phi_ds = None
        f_xi = None
        if self.xi_phi is None and self.save_path:
            f_xi = h5py.File(self.save_path, 'r')
            xi_phi_ds = f_xi['xi_phi']

        mesh = sharding_core.create_1d_mesh(devices=devices, axis_name='devices')
        rep_sharding = sharding_core.get_replicated_sharding(mesh)
        eval_sharding = NamedSharding(mesh, P('devices', None))
        params_rep = jax.tree_util.tree_map(
            lambda x: jax.device_put(np.asarray(x), rep_sharding), jastrow_params
        )
            
        def compute_block_on_device(grid_eval_shard, jastrow_params, grid_int, weights_int, xi_phi_int):
            def scan_body(carry, i):
                if include_h_aux:
                    l_aux_carry, h_aux_carry = carry
                r_eval = grid_eval_shard
                g_batch = jax.lax.dynamic_slice(grid_int, (i * batch_size, 0), (batch_size, 3))
                w_batch = jax.lax.dynamic_slice(weights_int, (i * batch_size,), (batch_size,))
                xi_batch = jax.lax.dynamic_slice(xi_phi_int, (0, i * batch_size), (n_rank, batch_size))
                
                if use_laux_fast_grad:
                    u_grad = self.jastrow_factor.grad_r_batch_laux(
                        r_eval, g_batch, jastrow_params
                    )
                else:
                    u_grad = self.jastrow_factor.grad_r_batch(
                        r_eval, g_batch, jastrow_params
                    )
                xi_weighted = xi_batch * w_batch[None, :]
                update = jnp.einsum('ab,ibk->aik', xi_weighted, u_grad)
                if include_h_aux:
                    grad_sq = jnp.sum(u_grad**2, axis=-1)
                    h_update = jnp.einsum('ab,ib->ai', xi_weighted, grad_sq)
                    return (l_aux_carry + update, h_aux_carry + h_update), None
                return carry + update, None

            n_int = grid_int.shape[0]
            n_batches = (n_int + batch_size - 1) // batch_size
            
            pad_int = n_batches * batch_size - n_int
            if pad_int > 0:
                grid_int = jnp.pad(grid_int, ((0, pad_int), (0, 0)))
                weights_int = jnp.pad(weights_int, (0, pad_int))
                xi_phi_int = jnp.pad(xi_phi_int, ((0, 0), (0, pad_int)))
                
            l_aux_init = jnp.zeros((n_rank, grid_eval_shard.shape[0], 3))
            if include_h_aux:
                h_aux_init = jnp.zeros((n_rank, grid_eval_shard.shape[0]))
                (l_aux_res, h_aux_res), _ = jax.lax.scan(
                    scan_body, (l_aux_init, h_aux_init), jnp.arange(n_batches)
                )
                return l_aux_res, h_aux_res
            l_aux_res, _ = jax.lax.scan(scan_body, l_aux_init, jnp.arange(n_batches))
            return l_aux_res

        if include_h_aux:
            @shard_map(
                mesh=mesh,
                in_specs=(P('devices', None), P(), P(), P(), P()),
                out_specs=(P(None, 'devices', None), P(None, 'devices')),
                check_vma=False,
            )
            def sharded_compute(grid_eval_shard, params, grid_int, weights_int, xi_phi_int):
                return compute_block_on_device(grid_eval_shard, params, grid_int, weights_int, xi_phi_int)
        else:
            @shard_map(
                mesh=mesh,
                in_specs=(P('devices', None), P(), P(), P(), P()),
                out_specs=P(None, 'devices', None),
                check_vma=False,
            )
            def sharded_compute(grid_eval_shard, params, grid_int, weights_int, xi_phi_int):
                return compute_block_on_device(grid_eval_shard, params, grid_int, weights_int, xi_phi_int)

        try:
            for r0 in range(0, n_grid, host_grid_block_size):
                r1 = min(r0 + host_grid_block_size, n_grid)
                n_eval = r1 - r0
                logger.info(f"		_compute_L_aux: Processing block [{r0}:{r1}]...")
                
                remainder = n_eval % n_devices
                padding = (n_devices - remainder) if remainder != 0 else 0
                n_eval_padded = n_eval + padding
                
                grid_eval_block = np.asarray(self.grid_points[r0:r1])
                if padding > 0:
                    grid_eval_block = np.pad(grid_eval_block, ((0, padding), (0, 0)))
                sharded_grid_eval = jax.device_put(grid_eval_block, eval_sharding)
                
                res_rep_accum = None
                h_aux_rep_accum = None
                
                # Also controlled by host_grid_block_size to limit peak memory of inputs
                from pytc.utils.prefetch import async_read, await_read, safe_hdf5_read

                def _prepare_Laux_int_block(g0_loc):
                    """Load integration chunk to device (background-thread safe)."""
                    g1_loc = min(g0_loc + host_grid_block_size, n_grid)
                    g_chunk = jax.device_put(np.asarray(self.grid_points[g0_loc:g1_loc]), rep_sharding)
                    w_chunk = jax.device_put(np.asarray(self.weights[g0_loc:g1_loc]), rep_sharding)
                    if self.xi_phi is not None:
                        xi_chunk = jax.device_put(
                            safe_hdf5_read(self.xi_phi, (slice(None), slice(g0_loc, g1_loc))),
                            rep_sharding,
                        )
                    else:
                        xi_chunk = jax.device_put(
                            safe_hdf5_read(xi_phi_ds, (slice(None), slice(g0_loc, g1_loc))),
                            rep_sharding,
                        )
                    return g_chunk, w_chunk, xi_chunk

                pending_int = None
                for g0 in range(0, n_grid, host_grid_block_size):
                    if pending_int is not None:
                        grid_int_chunk, weights_int_chunk, xi_phi_chunk = await_read(pending_int)
                        pending_int = None
                    else:
                        grid_int_chunk, weights_int_chunk, xi_phi_chunk = _prepare_Laux_int_block(g0)
                            
                    res_partial = sharded_compute(
                        sharded_grid_eval, params_rep, grid_int_chunk, weights_int_chunk, xi_phi_chunk
                    )

                    if include_h_aux:
                        res_partial, h_partial = res_partial

                    # While shard_map runs, prefetch next integration block.
                    next_g0 = g0 + host_grid_block_size
                    if next_g0 < n_grid:
                        pending_int = async_read(lambda _g=next_g0: _prepare_Laux_int_block(_g))
                    
                    if res_rep_accum is None:
                        res_rep_accum = res_partial
                        if include_h_aux:
                            h_aux_rep_accum = h_partial
                    else:
                        res_rep_accum += res_partial
                        if include_h_aux:
                            h_aux_rep_accum += h_partial
                    
                    del grid_int_chunk, weights_int_chunk, xi_phi_chunk, res_partial
                    if include_h_aux:
                        del h_partial
                
                res_block = res_rep_accum[:, :n_eval, :]
                L_aux_out[:, r0:r1, :] = np.asarray(res_block)
                if include_h_aux:
                    h_aux_block = h_aux_rep_accum[:, :n_eval]
                    H_aux_out[:, r0:r1] = np.asarray(h_aux_block)
                    del h_aux_block
                
                del sharded_grid_eval, res_rep_accum, res_block
                if include_h_aux:
                    del h_aux_rep_accum
                gc.collect()
                
        finally:
            if f_xi: f_xi.close()
            # If we are returning a dataset, we MUST NOT close f_out here.
            # The caller or the dataset object itself will manage the lifecycle.
            # However, h5py datasets require the file to remain open.
            # To be safe and consistent with other parts, we'll close it if we're not returning it.
            if f_out and not isinstance(L_aux_out, h5py.Dataset):
                f_out.close()
            
        if include_h_aux:
            return L_aux_out, H_aux_out
        return L_aux_out
	

    def isdf(self, jastrow_params, save_path=None, batch_size=1000, host_grid_block_size=None,
             r2_tile_size=None, gpu_budget_bytes=None, reuse_aux_kernels=None,
             use_laux_fast_grad=False):
        """Compute ISDF intermediates and store them.
        
        Computes K1_kernel, K3_kernel, and L_aux.
        
        Args:
            jastrow_params: Parameters for the Jastrow factor.
            save_path: Optional path to save intermediates to HDF5.
            batch_size: Batch size for computation.
            host_grid_block_size: Block size for grid batching on host.
            r2_tile_size: Optional r2-grid tile size for kernel assembly.
            gpu_budget_bytes: Optional device-memory budget in bytes for kernel
                assembly.
            reuse_aux_kernels: Exact path that builds ``L_aux`` and ``H_aux``
                together, then recovers K1/K3 from one-grid contractions.  By
                default it is enabled for in-core calculations and out-of-core
                calculations with a persistent output path.  Out-of-core
                calculations without a path retain direct K1/K3 construction.
                Passing ``True`` explicitly requires a persistent path for the
                L_aux/H_aux datasets.
            use_laux_fast_grad: Opt into an algebraically equivalent,
                Jastrow-provided fast derivative for the L_aux double-grid
                contraction.  The choice is persisted on out-of-core caches.
        """
        logger.info("Computing ISDF intermediates (TC)...")
        start_time = time.perf_counter()
        
        out_path = save_path if save_path else self.save_path
        reuse_aux_kernels_defaulted = reuse_aux_kernels is None
        if reuse_aux_kernels_defaulted:
            reuse_aux_kernels = self.is_incore or bool(out_path)
        if reuse_aux_kernels and not self.is_incore and not out_path:
            raise ValueError(
                "out-of-core reuse_aux_kernels requires save_path so L_aux "
                "and H_aux can be streamed rather than materialized"
            )
        laux_gradient_mode = _laux_gradient_mode(use_laux_fast_grad)
        kernels = {}
        # An explicit auxiliary-reuse request must execute that path, not
        # silently accept a pre-existing direct K1/K3 cache.  The default may
        # accept an exact warm cache instead of rebuilding equivalent kernels.
        if (
            out_path
            and os.path.exists(out_path)
            and (not reuse_aux_kernels or reuse_aux_kernels_defaulted)
        ):
            try:
                f = h5py.File(out_path, 'r')
                if 'K1_kernel' in f and 'K3_kernel' in f and 'L_aux' in f:
                    cached_laux_mode = f.attrs.get(
                        "pytc_laux_gradient_mode", "direct"
                    )
                    if isinstance(cached_laux_mode, bytes):
                        cached_laux_mode = cached_laux_mode.decode()
                    if cached_laux_mode != laux_gradient_mode:
                        logger.info(
                            "  L_aux cache gradient mode is %s, but this run "
                            "requests %s; recomputing base intermediates.",
                            cached_laux_mode, laux_gradient_mode,
                        )
                        f.close()
                        f = None
                    else:
                        logger.info(f"  Found existing K1, K3, and L_aux in {out_path}. Reading from file...")
                        logger.info(f"  Loading K1 with shape: {f['K1_kernel'].shape} on host RAM.")
                        kernels['K1_kernel'] = f['K1_kernel'][:]
                        logger.info(f"  Loading K3 with shape: {f['K3_kernel'].shape} on host RAM")
                        kernels['K3_kernel'] = f['K3_kernel'][:]
                        cached_kmat_mode = f.attrs.get(
                            "pytc_kmat_kernel_mode", "legacy-cache-unknown"
                        )
                        if isinstance(cached_kmat_mode, bytes):
                            cached_kmat_mode = cached_kmat_mode.decode()
                        if self.is_incore:
                            logger.debug(f"incore mode: Loading L_aux with shape: {f['L_aux'].shape} on host RAM")
                            kernels['L_aux'] = f['L_aux'][:]
                            f.close()
                        else:
                            logger.debug(f"out-of-core mode: Streaming L_aux with shape: {f['L_aux'].shape} from {out_path}")
                            kernels['L_aux'] = f['L_aux']

                        logger.info(f"ISDF intermediates loaded from file in {time.perf_counter() - start_time:.4f} s")
                        return self.replace(
                            isdf_kernels=kernels,
                            kmat_kernel_mode=cached_kmat_mode,
                        )
                
                # If we are here, keys are missing. Close the file!
                if f is not None:
                    f.close()
            except (IOError, KeyError) as e:
                logger.warning(f"  Error reading kernels from {out_path}: {e}. Recomputing...")

        if reuse_aux_kernels:
            logger.info("  Computing L_aux and H_aux for exact K1/K3 recovery...")
            aux_build_start = time.perf_counter()
            aux_result = self._compute_L_aux(
                jastrow_params,
                batch_size,
                save_path=out_path if not self.is_incore else None,
                host_grid_block_size=host_grid_block_size,
                include_h_aux=True,
                use_laux_fast_grad=use_laux_fast_grad,
            )
            L_aux, H_aux = aux_result
            logger.info(
                "  L_aux/H_aux construction completed in %.4f s",
                time.perf_counter() - aux_build_start,
            )
            recovery_start = time.perf_counter()
            if self.is_incore:
                kernels = kmat_jax.calc_kmat_kernels_from_aux(
                    self.xi_phi, self.xi_grad, self.weights, L_aux, H_aux
                )
            else:
                # The grid panel follows the L_aux construction panel; rows
                # of the left K rank are deliberately small so no full R^2
                # temporary is ever materialized on a GPU.
                stream_grid_block = host_grid_block_size or min(
                    self.grid_points.shape[0], 4096
                )
                stream_rank_block = min(self.phi_isdf.shape[1], 128)
                logger.info(
                    "  Recovering K1/K3 from streamed L_aux/H_aux "
                    "(grid_block=%d, rank_block=%d)...",
                    stream_grid_block,
                    stream_rank_block,
                )
                # A genuine out-of-core ISDF object intentionally carries
                # ``None`` for xi_phi/xi_grad.  Reuse the already-open
                # L_aux HDF5 file instead of reloading either collocation
                # tensor into host memory.
                xi_source_file = L_aux.file
                xi_phi_source = (
                    self.xi_phi if self.xi_phi is not None
                    else xi_source_file['xi_phi']
                )
                xi_grad_source = (
                    self.xi_grad if self.xi_grad is not None
                    else xi_source_file['xi_grad']
                )
                kernels = kmat_jax.calc_kmat_kernels_from_aux_streamed(
                    xi_phi_source,
                    xi_grad_source,
                    self.weights,
                    L_aux,
                    H_aux,
                    grid_block_size=stream_grid_block,
                    rank_block_size=stream_rank_block,
                )
                # H_aux is a transient companion to L_aux, not a cache
                # format change.  Drop it immediately after exact K3
                # recovery rather than retaining an extra R-by-G dataset.
                if isinstance(H_aux, h5py.Dataset):
                    h_aux_file = H_aux.file
                    del h_aux_file['H_aux']
                    h_aux_file.flush()
            logger.info(
                "  K1/K3 auxiliary recovery completed in %.4f s",
                time.perf_counter() - recovery_start,
            )
        else:
            logger.info("  Computing K1 and K3 kernels...")
            direct_k_start = time.perf_counter()
            kernels = self.compute_kmat_kernels(
                jastrow_params,
                batch_size,
                host_grid_block_size=host_grid_block_size,
                r2_tile_size=r2_tile_size,
                gpu_budget_bytes=gpu_budget_bytes,
            )
            logger.info(
                "  K1/K3 direct construction completed in %.4f s",
                time.perf_counter() - direct_k_start,
            )
            logger.info("  Computing L_aux...")
            laux_start = time.perf_counter()
            L_aux = self._compute_L_aux(
                jastrow_params,
                batch_size,
                save_path=out_path if not self.is_incore else None,
                host_grid_block_size=host_grid_block_size,
                use_laux_fast_grad=use_laux_fast_grad,
            )
            logger.info(
                "  L_aux construction completed in %.4f s",
                time.perf_counter() - laux_start,
            )
        logger.info(f"   K1 kernel on device size: {kernels['K1_kernel'].size * 8 / 1024**3:.2f} GB")
        logger.info(f"   K3 kernel on device size: {kernels['K3_kernel'].size * 8 / 1024**3:.2f} GB")
        
        # Move L_aux to CPU RAM to avoid GPU OOM (it can be very large)
        # If it's an HDF5 dataset, we keep it as is.
        if not isinstance(L_aux, (np.ndarray, jnp.ndarray)):
             logger.info("  L_aux is streaming from HDF5")
             kernels['L_aux'] = L_aux
        else:
            logger.info("  Moving L_aux to CPU")
            logger.info(f"    L_aux shape: {L_aux.shape}")
            logger.info(f"    L_aux size: {L_aux.size * 8 / 1024**3:.2f} GB")
            cpu_device = jax.devices("cpu")[0]
            L_aux = jax.device_put(L_aux, cpu_device)
            kernels['L_aux'] = L_aux
        
        if out_path and not self.is_incore:
            with h5py.File(out_path, 'a') as f:
                # This provenance is deliberately on the base cache because
                # timing cards must not mistake a direct K cache for an
                # auxiliary-recovered one (or vice versa) after a restart.
                f.attrs['pytc_kmat_kernel_mode'] = (
                    'aux-recovery' if reuse_aux_kernels else 'direct'
                )
                f.attrs['pytc_laux_gradient_mode'] = laux_gradient_mode
                for k, v in kernels.items():
                    if k == 'L_aux': continue
                    if k in f: del f[k]
                    f.create_dataset(k, data=np.array(v))
                if 'phi_isdf' not in f: f.create_dataset('phi_isdf', data=np.array(self.phi_isdf))
                if 'grad_phi_isdf' not in f: f.create_dataset('grad_phi_isdf', data=np.array(self.grad_phi_isdf))
                if 'pivots' not in f: f.create_dataset('pivots', data=np.array(self.pivots))
                
        logger.info(f"ISDF intermediates computed in {time.perf_counter() - start_time:.4f} s")
        
        return self.replace(
            isdf_kernels=kernels,
            save_path=out_path,
            kmat_kernel_mode=(
                "aux-recovery" if reuse_aux_kernels else "direct"
            ),
        )

    def _accumulate_transpose_block(self, result_np, U1, U3, ranges_T,
                                    scale, n_sub=2, device=None):
        """Compute transpose block in sub-chunks on GPU, accumulate on host.

        Each sub-chunk is computed on GPU, transferred to host via np.asarray(),
        and accumulated into the NumPy array ``result_np``.  GPU only ever holds
        one sub-chunk at a time → peak GPU ≈ S/n_sub (not S).

        Parameters
        ----------
        result_np : numpy array (Np, Nq, Nr, Ns) — host-side accumulator.
        U1, U3 : ISDF kernels (on GPU).
        ranges_T : (slice_r, slice_s, slice_p, slice_q) for the transpose block.
        scale : float multiplier.
        n_sub : int — number of sub-chunks (default 2).
        """
        slice_r_T, slice_s_T, slice_p_T, slice_q_T = ranges_T

        nmo = self.phi_isdf.shape[0]
        r_start = slice_r_T.start if slice_r_T.start is not None else 0
        r_stop = slice_r_T.stop if slice_r_T.stop is not None else nmo
        r_len = r_stop - r_start

        # Use fixed rank_block_size to avoid JIT recompilation.
        rbs = self._get_fixed_rank_block_size()

        chunk_size = max(1, (r_len + n_sub - 1) // n_sub)
        cache_getter = getattr(self, "_get_isdf_device_cache", None)
        cache = (
            cache_getter(device=device, include_grad=True)
            if callable(cache_getter) else None
        )
        phi_full = cache["phi_isdf"] if cache is not None else self.phi_isdf
        grad_full = cache["grad_phi_isdf"] if cache is not None else self.grad_phi_isdf
        u1 = jax.device_put(U1, device) if device is not None else U1
        u3 = jax.device_put(U3, device) if device is not None else U3
        device_ctx = jax.default_device(device) if device is not None else contextlib.nullcontext()
        for i0 in range(0, r_len, chunk_size):
            i1 = min(i0 + chunk_size, r_len)
            sub_slice_r = slice(r_start + i0, r_start + i1)
            sub_ranges = (sub_slice_r, slice_s_T, slice_p_T, slice_q_T)

            with device_ctx:
                if sub_slice_r == slice_s_T:
                    tmp = kmat_jax.contract_K1_isdf(
                        phi_full, grad_full, u1, sub_ranges, rank_block_size=rbs)
                    tmp = tmp - tmp.transpose(1, 0, 2, 3)
                else:
                    tmp = kmat_jax.contract_K1_minus_K2_isdf(
                        phi_full, grad_full, u1, sub_ranges, rank_block_size=rbs)

                tmp = tmp + kmat_jax.contract_K3_isdf(
                    phi_full, u3, sub_ranges, rank_block_size=rbs)

            # Transfer to host and accumulate — frees GPU memory for next chunk
            chunk_np = np.asarray(tmp.transpose(2, 3, 0, 1))
            del tmp
            result_np[:, :, i0:i1, :] += chunk_np * scale
            del chunk_np

    def _get_tc_direct_tile(self, kernels, ranges, device=None, panel_size=None,
                            panel_layout="pr"):
        """Compute the unsymmetrized direct TC tile 0.5*(K1-K2+K3)."""
        global _TC_DIRECT_TILE_PROFILED
        _device_key = getattr(device, "id", "host")
        panel_layout = _normalize_panel_layout(panel_layout)
        _profile_key = (_device_key, panel_layout)
        _profile = panel_size is not None and _profile_key not in _TC_DIRECT_TILE_PROFILED
        if _profile:
            _TC_DIRECT_TILE_PROFILED.add(_profile_key)
            _t0 = time.perf_counter()

        U1 = kernels['K1_kernel']
        U3 = kernels['K3_kernel']
        slice_p, slice_q, slice_r, slice_s = ranges

        rbs = self._get_fixed_rank_block_size()
        cache_getter = getattr(self, "_get_isdf_device_cache", None)
        cache = (
            cache_getter(kernels, device=device, include_grad=True, include_tc=True)
            if callable(cache_getter) else None
        )
        # ``u1`` / ``u3`` may be device jax.Arrays (resident) OR host numpy
        # arrays (streaming); the downstream contract_* wrappers handle both.
        # ``k_panel`` is None in the resident case (single JIT call) or an
        # int in the streaming case (axis-1 panel width).
        u1 = cache.get("K1_kernel") if cache is not None else None
        u3 = cache.get("K3_kernel") if cache is not None else None
        k_panel = cache.get("K_stream_panel") if cache is not None else None
        if u1 is None:
            u1 = jax.device_put(U1, device) if device is not None else U1
        if u3 is None:
            u3 = jax.device_put(U3, device) if device is not None else U3

        if _profile:
            jax.block_until_ready((u1, u3))
            _t_put = time.perf_counter()
            logger.debug(
                "_get_tc_direct_tile first-tile profile: K1+K3 device_put %.3fs "
                "(K1=%.1fMB, K3=%.1fMB, device=%s)",
                _t_put - _t0,
                getattr(U1, 'nbytes', 0) / 1e6,
                getattr(U3, 'nbytes', 0) / 1e6,
                _device_key,
            )

        device_ctx = jax.default_device(device) if device is not None else contextlib.nullcontext()

        phi_src = cache["phi_isdf"] if cache is not None else self.phi_isdf
        grad_src = cache["grad_phi_isdf"] if cache is not None else self.grad_phi_isdf

        phi_p = phi_src[slice_p]
        phi_q = phi_src[slice_q]
        phi_r = phi_src[slice_r]
        phi_s = phi_src[slice_s]
        grad_phi_p = grad_src[slice_p]
        grad_phi_q = grad_src[slice_q]

        if panel_size is not None:
            if "p" in panel_layout:
                phi_p = _pad_axis(phi_p, 0, panel_size)
                grad_phi_p = _pad_axis(grad_phi_p, 0, panel_size)
            else:
                phi_p = jnp.asarray(phi_p)
                grad_phi_p = jnp.asarray(grad_phi_p)
            if "q" in panel_layout:
                phi_q = _pad_axis(phi_q, 0, panel_size)
                grad_phi_q = _pad_axis(grad_phi_q, 0, panel_size)
            else:
                phi_q = jnp.asarray(phi_q)
                grad_phi_q = jnp.asarray(grad_phi_q)
            if "r" in panel_layout:
                phi_r = _pad_axis(phi_r, 0, panel_size)
            else:
                phi_r = jnp.asarray(phi_r)
            if "s" in panel_layout:
                phi_s = _pad_axis(phi_s, 0, panel_size)
            else:
                phi_s = jnp.asarray(phi_s)
        else:
            phi_p = jnp.asarray(phi_p)
            phi_q = jnp.asarray(phi_q)
            phi_r = jnp.asarray(phi_r)
            phi_s = jnp.asarray(phi_s)
            grad_phi_p = jnp.asarray(grad_phi_p)
            grad_phi_q = jnp.asarray(grad_phi_q)
        if device is not None and cache is None:
            phi_p = jax.device_put(phi_p, device)
            phi_q = jax.device_put(phi_q, device)
            phi_r = jax.device_put(phi_r, device)
            phi_s = jax.device_put(phi_s, device)
            grad_phi_p = jax.device_put(grad_phi_p, device)
            grad_phi_q = jax.device_put(grad_phi_q, device)

        # Caller-expected p/q extents (before any symmetrization re-padding).
        p_len_out = phi_p.shape[0]
        q_len_out = phi_q.shape[0]

        with device_ctx:
            if slice_p == slice_q:
                # The antisymmetrization ``k12 - k12.T(1,0,2,3)`` requires
                # ``phi_p.shape[0] == phi_q.shape[0]``.  Panel padding can
                # violate that when ``nocc < panel_blk`` and the layout
                # pads only one of p/q (e.g. "pr" for oovv): phi_p → panel_size,
                # phi_q stays at nocc.  Match the shorter side to the longer
                # before the contract; we slice axes 0/1 back to ``p_len_out``
                # / ``q_len_out`` after combining with K3 so the returned tile
                # still follows the panel_layout convention.
                match_len = max(phi_p.shape[0], phi_q.shape[0])
                if phi_p.shape[0] < match_len:
                    phi_p      = _pad_axis(phi_p, 0, match_len)
                    grad_phi_p = _pad_axis(grad_phi_p, 0, match_len)
                if phi_q.shape[0] < match_len:
                    phi_q      = _pad_axis(phi_q, 0, match_len)
                # In-kernel antisymmetrisation in (p,q) — avoids
                # materialising k12 and its transposed copy (~36 s/tile at 5z).
                k12 = kmat_jax.contract_K1_antisym_pq_isdf_streaming(
                    phi_p, phi_r, phi_s, grad_phi_p, u1, rbs,
                    panel_size=k_panel)
            else:
                k12 = kmat_jax.contract_K1_minus_K2_isdf_streaming(
                    phi_p, phi_q, phi_r, phi_s, grad_phi_p, grad_phi_q, u1, rbs,
                    panel_size=k_panel)

            if panel_size is not None:
                if _profile:
                    jax.block_until_ready(k12)
                    _t_k1 = time.perf_counter()
                    logger.debug("_get_tc_direct_tile first-tile profile: K1 compute %.3fs",
                                 _t_k1 - _t_put)
                k3 = kmat_jax.contract_K3_isdf_streaming(
                    phi_p, phi_q, phi_r, phi_s, u3, rbs, panel_size=k_panel)
                if _profile:
                    jax.block_until_ready(k3)
                    _t_k3 = time.perf_counter()
                    logger.debug("_get_tc_direct_tile first-tile profile: K3 compute %.3fs, "
                                 "total tile %.3fs", _t_k3 - _t_k1, _t_k3 - _t0)
                result = 0.5 * (k12 + k3)
                # If the slice_p==slice_q branch re-padded phi_p/phi_q to
                # equalise them, slice axes 0/1 back to the caller-expected
                # extents so the tile matches the panel_layout convention.
                if result.shape[0] != p_len_out:
                    result = result[:p_len_out, :, :, :]
                if result.shape[1] != q_len_out:
                    result = result[:, :q_len_out, :, :]
                return result

            result_np = np.array(k12)
            del k12
            k3 = kmat_jax.contract_K3_isdf_streaming(
                phi_p, phi_q, phi_r, phi_s, u3, rbs, panel_size=k_panel)
            result_np += np.asarray(k3)
            del k3
            result_np *= 0.5
            # Match caller-expected shape when symmetrization re-padding
            # widened axes 0/1 beyond the original slice extents.
            if result_np.shape[0] != p_len_out or result_np.shape[1] != q_len_out:
                result_np = result_np[:p_len_out, :q_len_out, :, :]
            return jnp.asarray(result_np)

    def _assemble_tc_tile(self, kernels, ranges, device=None, panel_size=None,
                          panel_layout="pr"):
        """Assemble and symmetrize one finished TC tile."""
        panel_layout = _normalize_panel_layout(panel_layout)
        direct = self._get_tc_direct_tile(
            kernels, ranges, device=device, panel_size=panel_size,
            panel_layout=panel_layout)

        if panel_size is not None:
            slice_p, slice_q, slice_r, slice_s = ranges
            # The ``direct + direct.transpose(2, 3, 0, 1)`` shortcut fuses
            # the (p↔r, q↔s) symmetrisation into a single tile compute.
            # It is only valid when the shape of ``direct`` is invariant
            # under that axis swap — i.e. when the panel-padded axis set
            # is itself symmetric under (0↔2, 1↔3).  Of the three
            # panel_layouts only ``"pr"`` satisfies this (pads axes 0 and
            # 2 identically, leaves 1 and 3 untouched).  ``"qr"`` pads
            # axis 2 but not 0, and ``"ps"`` pads axis 0 but not 2 —
            # so ``direct.transpose(2, 3, 0, 1)`` comes out with a
            # different shape from ``direct`` and the broadcast add
            # crashes (e.g. ``(nocc, ps, ps, nvir) + (ps, nvir, nocc, ps)``
            # on an ovov single-tile).  For the asymmetric layouts we
            # fall through to the explicit-``tmp`` branch, which uses
            # ``_transpose_panel_layout()`` to produce a partner tile
            # whose shape matches ``direct`` after the (2, 3, 0, 1)
            # transpose.
            if (slice_p == slice_r and slice_q == slice_s
                    and panel_layout == "pr"):
                return -(direct + direct.transpose(2, 3, 0, 1))

            ranges_T = (slice_r, slice_s, slice_p, slice_q)
            tmp = self._get_tc_direct_tile(
                kernels, ranges_T, device=device, panel_size=panel_size,
                panel_layout=_transpose_panel_layout(panel_layout))
            return -(direct + tmp.transpose(2, 3, 0, 1))

        result_np = np.array(direct)
        del direct

        slice_p, slice_q, slice_r, slice_s = ranges
        if slice_p == slice_r and slice_q == slice_s:
            result_np += result_np.transpose(2, 3, 0, 1)
        else:
            ranges_T = (slice_r, slice_s, slice_p, slice_q)
            self._accumulate_transpose_block(
                result_np, kernels['K1_kernel'], kernels['K3_kernel'],
                ranges_T, scale=0.5, n_sub=2, device=device)

        return jnp.asarray(-result_np)

    def get_2b(self, jastrow_params, block_str=None, ranges=None, batch_size=1000):
        """Calculate TC correction terms using ISDF with multi-GPU support.

        Memory-optimized: computes each piece on GPU, immediately transfers
        to host, and accumulates on host (NumPy).  GPU only ever holds one
        contraction output at a time → peak GPU ≈ 1S (one output block).
        Returns a JAX array.
        """
        start_time = time.perf_counter()
        logger.debug("Starting ISDFTC.get_2b")
        if ranges is None and block_str is not None:
            ranges = self._get_block_ranges(block_str)
        # Full-tensor fallback: when no block/ranges requested, build the
        # entire (n_orb, n_orb, n_orb, n_orb) TC correction.
        if ranges is None:
            full = slice(None)
            ranges = (full, full, full, full)

        if self.isdf_kernels is None:
            kernels = self.compute_kmat_kernels(jastrow_params, batch_size)
        else:
            kernels = self.isdf_kernels

        result = self._assemble_tc_tile(kernels, ranges)
        total_time = time.perf_counter() - start_time
        logger.debug(f"ISDFTC.get_2b completed in {total_time:.4f} s")
        return result

    def get_3b_fock(self, jastrow_params, dm1, L_aux=None):
        """Get 3-body Fock matrix correction using ISDF.
        
        Scaling: O(N_aux * N_grid) per SCF step (after precomputation).
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix
            L_aux: Optional precomputed auxiliary potential (N_aux, N_grid, 3).
                   If None, it will be computed on the fly.
        """
        if L_aux is None:
            if self.isdf_kernels is not None and 'L_aux' in self.isdf_kernels:
                L_aux = self.isdf_kernels['L_aux']
            else:
                L_aux = self._compute_L_aux(jastrow_params)
        
        # 2. Compute density at pivots
        # rho(mu) = sum_pq D_pq phi_p(mu) phi_q(mu)
        # phi_pivots = self.phi_isdf (N_orb, N_aux)
        rho_pivots = jnp.einsum('ma,na,mn->a', self.phi_isdf, self.phi_isdf, dm1)
        
        # 3. Compute W on full grid using L_aux
        # W(r) = sum_mu rho(mu) L_mu(r)
        # L_aux: (N_aux, N_grid, 3)
        # rho_pivots: (N_aux,)
        W_g = jnp.einsum('a,agc->gc', rho_pivots, L_aux)
        
        V_3b_g = jnp.sum(W_g**2, axis=1)
        
        # 5. Integrate Fock matrix using ISDF
        # F_mn = sum_mu (sum_g w_g xi_mu(g) V_3b(g)) phi_m(mu) phi_n(mu)
        weighted_V = self.weights * V_3b_g
        
        if self.xi_phi is not None:
            V_proj = jnp.dot(self.xi_phi, weighted_V) # (N_aux,)
        else:
            with h5py.File(self.save_path, 'r') as f:
                xi_phi = f['xi_phi'][:]
                V_proj = jnp.dot(xi_phi, weighted_V)
        
        phi_pivots = self.phi_isdf # (N_orb, N_aux)
        weighted_phi = phi_pivots * V_proj[None, :]
        F_3b = jnp.dot(weighted_phi, phi_pivots.T)
        
        return F_3b

    def get_2b_fock(self, jastrow_params, dm1, T=None):
        """Get 2-body Fock matrix correction using ISDF kernels.
        
        This implementation avoids forming the full O(N^4) tensor by contracting
        the density matrix directly with the ISDF kernels. It computes:
        v_2b = Fock(result) = J(result) - 0.5 * K(result)
        where result = 0.5 * (K1 - K2 + K3) + transpose(2,3,0,1)
        
        Args:
            jastrow_params: Jastrow parameters
            dm1: Density matrix (AO basis)
            T: Optional precomputed 2-body tensor. If provided, uses base class implementation.
            
        Returns:
            Fock matrix contribution (N, N)
        """
        if T is not None:
            return super().get_2b_fock(jastrow_params, dm1, T)
            
        if self.isdf_kernels is None:
             kernels = self.compute_kmat_kernels(jastrow_params)
        else:
            kernels = self.isdf_kernels
            
        U1 = kernels['K1_kernel'] # (k, l, c)
        U3 = kernels['K3_kernel'] # (k, l)
        
        phi = self.phi_isdf # (n_orb, n_fused)
        grad = self.grad_phi_isdf # (n_orb, n_fused, 3)
        dm1 = jnp.array(dm1)
        
        # --- Direct Part Intermediates ---
        # rho[l] = sum_rs D_rs phi_r(l) phi_s(l)
        rho = jnp.einsum('rl,sl,rs->l', phi, phi, dm1)
        # m[k,l] = sum_rs phi_s(k) D_rs phi_r(l)
        m = jnp.einsum('sk,rl,rs->kl', phi, phi, dm1)
        # m_grad[k,l,c] = sum_rs grad_s(k,c) D_rs phi_r(l)
        m_grad = jnp.einsum('skc,rl,rs->klc', grad, phi, dm1)
        
        # --- Direct Part Terms ---
        # K3: phi phi U3 phi phi
        J3 = jnp.einsum('pk,qk,kl,l->pq', phi, phi, U3, rho)
        K3 = jnp.einsum('pk,kl,ql,lk->pq', phi, U3, phi, m)
        
        # K1: grad phi U1 phi phi
        U1_rho = jnp.einsum('klc,l->kc', U1, rho)
        J1 = jnp.einsum('pkc,qk,kc->pq', grad, phi, U1_rho)
        K1 = jnp.einsum('pkc,klc,ql,lk->pq', grad, U1, phi, m)
        
        # K2: phi grad U1 phi phi
        J2 = jnp.einsum('pk,qkc,kc->pq', phi, grad, U1_rho)
        K2 = jnp.einsum('pk,klc,ql,klc->pq', phi, U1, phi, m_grad)
        
        J_dir = 0.5 * (J1 - J2 + J3)
        K_dir = 0.5 * (K1 - K2 + K3)
        
        # --- Transpose Part Intermediates ---
        # rho_grad[k,c] = sum_rs D_rs grad_r(k,c) phi_s(k)
        rho_grad = jnp.einsum('rkc,sk,rs->kc', grad, phi, dm1)
        # rho_grad_rev[k,c] = sum_rs D_rs phi_r(k) grad_s(k,c)
        rho_grad_rev = jnp.einsum('rk,skc,rs->kc', phi, grad, dm1)
        # m_grad_rev[k,l,c] = sum_rs phi_s(k) D_rs grad_r(l,c)
        m_grad_rev = jnp.einsum('sk,rlc,rs->klc', phi, grad, dm1)
        
        U1T = U1.transpose(1, 0, 2)
        U3T = U3.T
        
        # --- Transpose Part Terms ---
        # K3T: phi phi U3T phi phi
        J3T = jnp.einsum('pk,qk,kl,l->pq', phi, phi, U3T, rho)
        K3T = jnp.einsum('pk,kl,ql,lk->pq', phi, U3T, phi, m)
        
        # K1T: phi phi U1T grad phi
        U1T_rho1 = jnp.einsum('klc,lc->k', U1T, rho_grad)
        J1T = jnp.einsum('pk,qk,k->pq', phi, phi, U1T_rho1)
        K1T = jnp.einsum('pk,klc,ql,klc->pq', phi, U1T, phi, m_grad_rev)
        
        # K2T: phi phi U1T phi grad
        U1T_rho2 = jnp.einsum('klc,lc->k', U1T, rho_grad_rev)
        J2T = jnp.einsum('pk,qk,k->pq', phi, phi, U1T_rho2)
        K2T = jnp.einsum('pk,klc,qlc,lk->pq', phi, U1T, grad, m)
        
        J_trans = 0.5 * (J1T - J2T + J3T)
        K_trans = 0.5 * (K1T - K2T + K3T)
        
        J_tot = J_dir + J_trans
        K_tot = K_dir + K_trans
        
        return -(J_tot - 0.5 * K_tot)
