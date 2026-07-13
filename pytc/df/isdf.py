"""ISDF decomposition of orbitals and their gradients -- shared numerical
machinery (pytc/df/ package reorganization, task #8, isdf-coulomb-cuda,
2026-07-12): "orbital values + derivative values as two factor channels"
is a generic ISDF operation used by TC and xTC alike, not TC-specific in
the sense of belonging to a peer integral-model module."""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import os
import logging
import time
import h5py
import uuid
import gc

from .pivots import pivoted_cholesky_pair_pivots
from .solvers import prepare_normal_equations_solver, solve_normal_equations_batch_prepared

logger = logging.getLogger(__name__)


def _pivoted_cholesky_phi(phi_weighted, n_rank, shift):
    """Specialized pivoted Cholesky for phi decomposition.

    Thin wrapper: the "same factor on both sides" case of
    ``pivoted_cholesky_pair_pivots`` (see its docstring). Kept as a
    distinct name at TC's existing call site rather than inlining, so
    that site's intent stays self-documenting.

    Discards the ``effective_rank`` half of the shared primitive's
    ``(pivots, effective_rank)`` return to preserve this wrapper's
    pre-existing single-array-return contract with ``isdf_decompose``.
    """
    pivots, _effective_rank = pivoted_cholesky_pair_pivots(phi_weighted, phi_weighted, n_rank, shift)
    return pivots


def _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank, shift):
    """Specialized pivoted Cholesky for gradient decomposition.

    Thin wrapper: the "two different factors" case of
    ``pivoted_cholesky_pair_pivots`` (see its docstring), with
    grad_phi_weighted's (n_orb, n_grid, 3) shape flattened to the 2-D
    (n_orb*3, n_grid) form the shared primitive expects -- the
    transpose-then-reshape collapses (orb, xyz) into one feature axis
    so the dot-product reduction still sums over exactly what the
    original per-component ``for c in range(3)`` loop summed over.

    Discards the ``effective_rank`` half of the shared primitive's
    ``(pivots, effective_rank)`` return to preserve this wrapper's
    pre-existing single-array-return contract with ``isdf_decompose``.
    """
    n_orb, n_grid, _ = grad_phi_weighted.shape
    grad_flat = grad_phi_weighted.transpose(0, 2, 1).reshape(n_orb * 3, n_grid)
    pivots, _effective_rank = pivoted_cholesky_pair_pivots(phi_weighted, grad_flat, n_rank, shift)
    return pivots


def isdf_decompose(phi, grad_phi, n_rank_phi, n_rank_grad, weights=None,
                   grid_batch_size=4096, rcond=1e-14,
                   is_incore=False, save_path=None, fixed_pivots=None):
    """Perform ISDF decomposition of orbitals and their gradients.

    Memory-efficient implementation using SVD-based solver to avoid
    materializing large C matrices (n_orb² × n_fused).

    Args:
        phi: Orbitals on grid (n_orb, n_grid)
        grad_phi: Orbital gradients on grid (n_orb, n_grid, 3)
        n_rank_phi: Rank for phi decomposition
        n_rank_grad: Rank for gradient decomposition
        weights: Optional (n_grid,) array of integration weights.
                 If provided, pivot selection is weighted by these weights.
        grid_batch_size: Number of grid points to process in each batch
        rcond: Relative condition number cutoff for SVD pseudoinverse (default 1e-14).
               Smaller values retain more singular values (more accurate but less stable).

    Returns:
        phi_piv: (N_orb, N_fused)
        xi_phi: (N_fused, N_grid)
        grad_phi_piv: (N_orb, N_fused, 3)
        xi_grad: (N_fused, N_grid, 3)
        pivots: (N_fused,)
    """
    if save_path is not None and os.path.exists(save_path) and fixed_pivots is None:
        try:
            with h5py.File(save_path, 'r') as f:
                if all(k in f for k in ['xi_phi', 'xi_grad', 'pivots', 'phi_isdf', 'grad_phi_isdf']):
                    logger.info(f"Loading ISDF decomposition from {save_path}")
                    pivots = jnp.array(f['pivots'][:])
                    phi_piv = jnp.array(f['phi_isdf'][:])
                    grad_phi_piv = jnp.array(f['grad_phi_isdf'][:])

                    if is_incore:
                        cpu_device = jax.devices("cpu")[0]
                        xi_phi = jax.device_put(f['xi_phi'][:], cpu_device)
                        xi_grad = jax.device_put(f['xi_grad'][:], cpu_device)
                    else:
                        xi_phi = None
                        xi_grad = None

                    return phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots, save_path
        except Exception as e:
            logger.warning(f"Failed to load ISDF from {save_path}: {e}. Recomputing...")

    n_orb, n_grid = phi.shape

    if weights is None:
        w_sqrt = jnp.ones(n_grid)
    else:
        w_sqrt = jnp.sqrt(jnp.abs(weights))  # Use abs to avoid NaN

    start_time = time.perf_counter()
    logger.info(f"Starting ISDF decomposition with n_orb={n_orb}, n_grid={n_grid}, n_rank_phi={n_rank_phi}, n_rank_grad={n_rank_grad}")
    if weights is not None:
        logger.info(f"  Using integration weights (min={jnp.min(weights):.3e}, max={jnp.max(weights):.3e})")

    # --- 1. Phi Decomposition ---
    t0 = time.perf_counter()

    # Apply weights to orbitals for pivot selection
    # The Gram matrix is (phi^T W phi)(phi^T W phi) where W = diag(weights)
    # Equivalently: (sqrt(W) phi)^T (sqrt(W) phi) squared
    phi_weighted = phi * w_sqrt  # (n_orb, n_grid)

    # Pre-compute diagonal for phi to calculate shift
    orb_sq = jnp.sum(phi_weighted**2, axis=0)  # Weighted orbital norms
    diag_phi = orb_sq**2
    shift_phi = 1e-12 * jnp.max(jnp.abs(diag_phi))

    pivots_phi = _pivoted_cholesky_phi(phi_weighted, n_rank_phi, shift_phi)
    t1 = time.perf_counter()
    logger.debug(f"Phi decomposition completed in {t1 - t0:.4f} s")

    # --- 2. Gradient Decomposition ---
    t0 = time.perf_counter()

    # Apply weights to gradients for pivot selection
    grad_phi_weighted = grad_phi * w_sqrt[:, None]  # (n_orb, n_grid, 3)

    # Pre-compute diagonal for grad to calculate shift
    A_diag = jnp.sum(phi_weighted**2, axis=0)
    B_diag = jnp.sum(jnp.sum(grad_phi_weighted**2, axis=2), axis=0)
    diag_grad = A_diag * B_diag
    shift_grad = 1e-12 * jnp.max(jnp.abs(diag_grad))

    pivots_grad = _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank_grad, shift_grad)
    t1 = time.perf_counter()
    logger.debug(f"Grad decomposition completed in {t1 - t0:.4f} s")

    # --- 3. Fuse pivots ---
    t0 = time.perf_counter()

    # Use numpy for unique to avoid JAX dynamic shape overhead
    pivots_all = np.concatenate([np.array(pivots_phi), np.array(pivots_grad)])
    pivots = jnp.array(np.unique(pivots_all))
    n_fused = pivots.shape[0]
    t1 = time.perf_counter()
    logger.info(f"Pivots fused: {pivots_phi.shape[0]} + {pivots_grad.shape[0]} -> {n_fused} in {t1 - t0:.4f} s")

    # Experiment hook (Task B precision investigation): override the device-selected
    # pivots with an externally-supplied fused-pivot set. Used to force CPU-selected
    # pivots onto the GPU interpolation so we can isolate whether the GPU/CPU
    # isdf_dU_err gap comes from pivot SELECTION (gap collapses) or downstream
    # numerics (gap persists). No effect on the default path (fixed_pivots=None).
    if fixed_pivots is not None:
        fp = np.asarray(fixed_pivots)
        if fp.ndim != 1 or not np.issubdtype(fp.dtype, np.integer):
            raise ValueError("fixed_pivots must be a 1-D integer array of grid indices")
        if np.unique(fp).shape[0] != fp.shape[0]:
            raise ValueError("fixed_pivots must be unique")
        if fp.size == 0 or fp.min() < 0 or fp.max() >= n_grid:
            raise ValueError(f"fixed_pivots out of range [0, {n_grid})")
        pivots = jnp.asarray(fp, dtype=pivots.dtype)
        n_fused = int(pivots.shape[0])
        logger.info(f"isdf_decompose: overriding with {n_fused} externally-supplied fixed pivots")

    # --- 4. Extract pivot values ---
    t0 = time.perf_counter()

    phi_piv = phi[:, pivots]  # (n_orb, n_fused)
    grad_phi_piv = grad_phi[:, pivots, :]  # (n_orb, n_fused, 3)

    t1 = time.perf_counter()
    logger.debug(f"Pivot values extracted in {t1 - t0:.4f} s")

    # --- 5. Solve for xi_phi and xi_grad using fast normal equations solver ---
    t0 = time.perf_counter()
    logger.info("Using fast normal equations solver")

    # Solve for xi_phi and xi_grad
    cpu_device = jax.devices("cpu")[0]
    grid_batch_size = min(grid_batch_size, n_grid)
    n_batches = (n_grid + grid_batch_size - 1) // grid_batch_size if grid_batch_size > 0 else 0

    # Pre-factor normal-equation matrices once and reuse for all grid batches.
    # This avoids rebuilding/re-factorizing ATA in every batch.
    phi_chol, phi_lower = prepare_normal_equations_solver(phi_piv, phi_piv, rcond=rcond)
    grad_chol = []
    grad_lower = []
    for c in range(3):
        chol_c, lower_c = prepare_normal_equations_solver(grad_phi_piv[:, :, c], phi_piv, rcond=rcond)
        grad_chol.append(chol_c)
        grad_lower.append(lower_c)

    # Multi-device: shard the grid axis of each batch across local devices;
    # replicate factors. Use local_devices() so this is safe under multi-process
    # JAX (arrays can only be placed on devices visible to this process).
    local_devices = jax.local_devices()
    n_devices = len(local_devices)
    use_sharding = n_devices > 1
    if use_sharding:
        mesh = Mesh(np.array(local_devices), ('g',))
        grid_shard = NamedSharding(mesh, P(None, 'g'))
        repl = NamedSharding(mesh, P())
        phi_chol = jax.device_put(phi_chol, repl)
        phi_piv_d = jax.device_put(phi_piv, repl)
        grad_chol = [jax.device_put(c, repl) for c in grad_chol]
        grad_phi_piv_d = jax.device_put(grad_phi_piv, repl)
        logger.info(f"  Multi-device sharding enabled across {n_devices} devices (grid axis)")
    else:
        phi_piv_d = phi_piv
        grad_phi_piv_d = grad_phi_piv

    # Setup storage
    h5_file = None
    if is_incore:
        logger.info(f"  Processing {n_batches} batches of size {grid_batch_size} (In-core)")
        xi_phi_storage = np.zeros((n_fused, n_grid), dtype=phi.dtype)
        xi_grad_storage = np.zeros((n_fused, n_grid, 3), dtype=phi.dtype)
    else:
        if save_path is None:
            save_path = f"isdf_temp_{uuid.uuid4().hex[:8]}.h5"
            logger.info(f"  No save_path provided, creating temporary HDF5: {save_path}")

        h5_file = h5py.File(save_path, 'a')
        logger.info(f"  Processing {n_batches} batches of size {grid_batch_size} (HDF5: {save_path})")

        # Create/Reset datasets
        for name, shape in [('xi_phi', (n_fused, n_grid)), ('xi_grad', (n_fused, n_grid, 3))]:
            if name in h5_file: del h5_file[name]
            h5_file.create_dataset(name, shape=shape, dtype=phi.dtype)

        for name, data in [('pivots', pivots), ('phi_isdf', phi_piv), ('grad_phi_isdf', grad_phi_piv)]:
            if name in h5_file: del h5_file[name]
            h5_file.create_dataset(name, data=np.array(data))

        xi_phi_storage = h5_file['xi_phi']
        xi_grad_storage = h5_file['xi_grad']

    # Warm up JIT so the sharded-program compile cost doesn't dominate short loops
    # (on a 7-batch benzene-5Z run the sharded compile otherwise ate the steady-state
    # speedup from parallel GPUs).
    if n_batches > 1:
        t_warm = time.perf_counter()
        warm_batch = jnp.zeros((n_orb, grid_batch_size), dtype=phi.dtype)
        if use_sharding:
            warm_batch = jax.device_put(warm_batch, grid_shard)
        warm = solve_normal_equations_batch_prepared(
            phi_chol, phi_lower, phi_piv_d, phi_piv_d, warm_batch, warm_batch
        )
        jax.block_until_ready(warm)
        for c in range(3):
            warm = solve_normal_equations_batch_prepared(
                grad_chol[c], grad_lower[c], grad_phi_piv_d[:, :, c], phi_piv_d,
                warm_batch, warm_batch
            )
            jax.block_until_ready(warm)
        del warm, warm_batch
        logger.debug(f"  Solve warmup (JIT compile) took {time.perf_counter() - t_warm:.2f} s")

    try:
        t_batch_start = time.perf_counter()
        for batch_idx in range(n_batches):
            g_start = batch_idx * grid_batch_size
            g_end = min(g_start + grid_batch_size, n_grid)
            bs = g_end - g_start

            # Pad batch width to a multiple of n_devices so the grid axis shards evenly.
            pad = (-bs) % n_devices if use_sharding else 0

            # 1. Xi_phi
            phi_batch = phi[:, g_start:g_end]
            if pad:
                phi_batch = jnp.pad(phi_batch, ((0, 0), (0, pad)))
            if use_sharding:
                phi_batch = jax.device_put(phi_batch, grid_shard)
            xi_phi_batch = solve_normal_equations_batch_prepared(
                phi_chol, phi_lower, phi_piv_d, phi_piv_d, phi_batch, phi_batch
            )
            if pad:
                xi_phi_batch = xi_phi_batch[:, :bs]
            xi_phi_storage[:, g_start:g_end] = np.array(xi_phi_batch)

            # 2. Xi_grad
            for c in range(3):
                grad_phi_batch_c = grad_phi[:, g_start:g_end, c]
                if pad:
                    grad_phi_batch_c = jnp.pad(grad_phi_batch_c, ((0, 0), (0, pad)))
                if use_sharding:
                    grad_phi_batch_c = jax.device_put(grad_phi_batch_c, grid_shard)
                xi_grad_batch = solve_normal_equations_batch_prepared(
                    grad_chol[c], grad_lower[c], grad_phi_piv_d[:, :, c], phi_piv_d,
                    grad_phi_batch_c, phi_batch
                )
                if pad:
                    xi_grad_batch = xi_grad_batch[:, :bs]
                xi_grad_storage[:, g_start:g_end, c] = np.array(xi_grad_batch)

            if batch_idx % 4 == 0 and batch_idx > 0:
                elapsed = time.perf_counter() - t_batch_start
                rate = batch_idx / elapsed
                eta = (n_batches - batch_idx) / rate if rate > 0 else 0
                logger.debug(f"Batch {batch_idx}/{n_batches} ({rate:.1f} batch/s, ETA: {eta:.1f}s)")

        # Load into JAX CPU RAM if requested
        if is_incore:
            xi_phi = jax.device_put(xi_phi_storage[:], cpu_device)
            xi_grad = jax.device_put(xi_grad_storage[:], cpu_device)
        else:
            xi_phi = None
            xi_grad = None

        # Explicitly delete storage to save RAM
        if is_incore:
            del xi_phi_storage, xi_grad_storage
            gc.collect()

    finally:
        if h5_file is not None:
            h5_file.close()
        gc.collect()

    total_time = time.perf_counter() - start_time
    logger.debug(f"Total fused ranks = {n_fused}")
    logger.info(f"ISDF decomposition total time: {total_time:.4f} s")

    return phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots, save_path
