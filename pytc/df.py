"""JAX implementation of Density Fitting / ISDF."""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from functools import partial
import os
import logging
import time
import h5py
import uuid
import gc

logger = logging.getLogger(__name__)


@jax.jit
def solve_normal_equations_batch(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                   phi_p_batch: jnp.ndarray, phi_q_batch: jnp.ndarray,
                                   rcond: float = 1e-14) -> jnp.ndarray:
    """Fast solver using LU decomposition for structured least-squares.
    
    Solves min ||C*X - B||² where:
      C[pq, m] = phi_piv_p[p, m] * phi_piv_q[q, m]
      B[pq, g] = phi_p_batch[p, g] * phi_q_batch[q, g]
    
    This exploits the separable structure to avoid O(N^2) intermediates:
    (C^T B)[m, g] = (sum_p phi_piv_p[p,m]*phi_p_batch[p,g]) * (sum_q phi_piv_q[q,m]*phi_q_batch[q,g])
    
    Args:
        phi_piv_p: (n_orb, n_fused) first factor of pivots
        phi_piv_q: (n_orb, n_fused) second factor of pivots
        phi_p_batch: (n_orb, batch_size) first factor of target
        phi_q_batch: (n_orb, batch_size) second factor of target
        rcond: Relative regularization strength (default 1e-14)
        
    Returns:
        X: (n_fused, batch_size) solutions
    """
    # Compute A^T A efficiently using the Kronecker-like structure
    gram_p = phi_piv_p.T @ phi_piv_p  # (n_fused, n_fused)
    gram_q = phi_piv_q.T @ phi_piv_q  # (n_fused, n_fused)
    ATA = gram_p * gram_q  # Element-wise product
    
    # Compute A^T B efficiently using separable structure
    term_p = jnp.matmul(phi_piv_p.T, phi_p_batch)  # (n_fused, batch_size)
    term_q = jnp.matmul(phi_piv_q.T, phi_q_batch)  # (n_fused, batch_size)
    ATB = term_p * term_q  # (n_fused, batch_size)
    
    # Use LU solve (jnp.linalg.solve) with Tikhonov regularization
    diag_mean = jnp.mean(jnp.diag(ATA))
    jitter = diag_mean * rcond
    ATA_reg = ATA + jitter * jnp.eye(ATA.shape[0])
    X = jnp.linalg.solve(ATA_reg, ATB)
    
    return X


solve_normal_equations_batch = jax.jit(solve_normal_equations_batch, static_argnames=['rcond'])


@jax.jit
def _build_normal_matrix(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray) -> jnp.ndarray:
    """Build unregularized normal-equation matrix for structured LS."""
    gram_p = phi_piv_p.T @ phi_piv_p
    gram_q = phi_piv_q.T @ phi_piv_q
    return gram_p * gram_q


def prepare_normal_equations_solver(phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                    rcond: float = 1e-14,
                                    max_jitter_tries: int = 8,
                                    jitter_growth: float = 10.0):
    """Prepare robust Cholesky factor for repeated batched solves.

    Uses adaptive jitter escalation to guarantee numerically SPD matrices.
    """
    ata = _build_normal_matrix(phi_piv_p, phi_piv_q)
    ata = 0.5 * (ata + ata.T)
    diag_mean = float(jnp.mean(jnp.diag(ata)))
    eps_scale = float(jnp.finfo(ata.dtype).eps) * max(diag_mean, 1.0)
    base_jitter = max(diag_mean * rcond, eps_scale)
    eye = jnp.eye(ata.shape[0], dtype=ata.dtype)

    last_chol = None
    for attempt in range(max_jitter_tries):
        jitter = base_jitter * (jitter_growth ** attempt)
        chol, lower = jsp_linalg.cho_factor(ata + jitter * eye, lower=True)
        if bool(jnp.all(jnp.isfinite(chol))):
            if attempt > 0:
                logger.warning(
                    "Cholesky jitter escalated: base=%.3e final=%.3e tries=%d",
                    base_jitter, jitter, attempt + 1
                )
            return chol, bool(lower)
        last_chol = chol

    raise np.linalg.LinAlgError(
        f"Adaptive Cholesky failed after {max_jitter_tries} tries; "
        f"base_jitter={base_jitter:.3e}, last_nonfinite={bool(jnp.any(jnp.isnan(last_chol)))}"
    )


@partial(jax.jit, static_argnames=('lower',))
def solve_normal_equations_batch_prepared(chol: jnp.ndarray, lower: bool,
                                          phi_piv_p: jnp.ndarray, phi_piv_q: jnp.ndarray,
                                          phi_p_batch: jnp.ndarray, phi_q_batch: jnp.ndarray) -> jnp.ndarray:
    """Solve batched normal equations using precomputed Cholesky factor."""
    term_p = jnp.matmul(phi_piv_p.T, phi_p_batch)
    term_q = jnp.matmul(phi_piv_q.T, phi_q_batch)
    atb = term_p * term_q
    return jsp_linalg.cho_solve((chol, lower), atb)


@partial(jax.jit, static_argnames=('n_rank',))
@partial(jax.jit, static_argnames=('n_rank',))
def pivoted_cholesky_pair_pivots(factor_p_weighted, factor_q_weighted, n_rank, shift):
    """Matrix-free pivoted-Cholesky selection of grid/interpolation points
    that best span the pair-product space ``A[g,(p,q)] = factor_p[p,g] *
    factor_q[q,g]``, without ever forming ``A`` (task #5,
    isdf-coulomb-cuda: jastrow-independent shared primitive both the TC
    pipeline and the Coulomb path import).

    Exploits the same algebraic identity the original TC-specific
    ``_pivoted_cholesky_phi``/``_pivoted_cholesky_grad`` functions each
    hard-coded separately: the Gram matrix of the pair-product matrix
    factors as ``(A A^T)[g,g'] = (sum_p factor_p[p,g] factor_p[p,g']) *
    (sum_q factor_q[q,g] factor_q[q,g'])`` -- a product of the two
    factors' own (n_grid, n_grid) Gram matrices, each computable directly
    from the (n_feature, n_grid) inputs with no O(n_grid * n_p * n_q)
    intermediate. Passing the SAME array for both factors reproduces
    ``_pivoted_cholesky_phi``'s original single-Gram-squared case (the
    "oo"/"vv" sector case, same MO subset on both sides of the pair);
    passing two DIFFERENT arrays reproduces ``_pivoted_cholesky_grad``'s
    original two-different-Grams case (the "ov" sector case, or TC's
    phi/grad_phi pairing when grad_phi is pre-flattened to 2-D by the
    caller -- e.g. ``grad_phi_weighted.transpose(0,2,1).reshape(n_orb*3,
    n_grid)`` collapses the (orb, xyz) axes together, so the dot-product
    reduction still sums over exactly what the original per-component
    ``for c in range(3)`` loop summed over).

    Args:
        factor_p_weighted: (n_p, n_grid) weighted values (e.g. occupied
            MO values on a grid, sqrt(weight)-scaled).
        factor_q_weighted: (n_q, n_grid) weighted values for the pair's
            other factor. Pass the same array as factor_p_weighted for a
            same-set pair space (TC's phi decomposition; the Coulomb
            path's "oo"/"vv" sectors); pass a different array for a
            mixed pair space (the Coulomb path's "ov" sector; TC's
            phi/grad_phi pairing, factor_q pre-flattened to 2-D).
        n_rank: Number of interpolation points (pivots) to select.
        shift: Tikhonov-style regularization added to the diagonal
            before each pivot selection (numerically pins near-zero/
            near-degenerate residuals to exactly zero rather than
            leaving them as noise-dominated candidates).

    Returns:
        pivots: (n_rank,) selected grid-point indices, in selection order.
    """
    n_grid = factor_p_weighted.shape[1]

    # Deterministic tie-break ramp for argmax (GPU/CPU pivot selection).
    # GPU tree-reduction in sum(factor**2,axis=0) produces diag_err values
    # differing ~eps·V (≈1e-16) device-to-device, flipping near-tie argmax
    # results. Adding a ramp = 1e-12 * arange(n_grid) * max(|diag_err|)
    # overrides reduction noise (~4 orders above eps, ~4 orders below
    # typical candidate gaps) so the highest index wins ties
    # deterministically on both devices.
    A_diag = jnp.sum(factor_p_weighted**2, axis=0)
    B_diag = jnp.sum(factor_q_weighted**2, axis=0)
    diag_err = A_diag * B_diag + shift
    diag_err = diag_err + 1e-12 * jnp.arange(n_grid, dtype=diag_err.dtype) * jnp.max(jnp.abs(diag_err))

    # Storage for L factor (N_grid, n_rank)
    L = jnp.zeros((n_grid, n_rank))
    pivots = jnp.zeros(n_rank, dtype=int)

    def body_fn(step, state):
        diag_err, L, pivots = state
        pivot = jnp.argmax(diag_err)
        pivots = pivots.at[step].set(pivot)
        pivot_val = diag_err[pivot]

        A_col = jnp.dot(factor_p_weighted.T, factor_p_weighted[:, pivot])
        B_col = jnp.dot(factor_q_weighted.T, factor_q_weighted[:, pivot])
        S_col = A_col * B_col
        S_col = S_col.at[pivot].add(shift)

        dot_prod = jnp.dot(L, L[pivot])
        is_small = pivot_val < 1e-12
        safe_pivot = jnp.where(is_small, 1.0, pivot_val)
        inv_sqrt_pivot = jax.lax.rsqrt(safe_pivot)

        l_col = (S_col - dot_prod) * inv_sqrt_pivot
        l_col = jnp.where(is_small, 0.0, l_col)
        L = L.at[:, step].set(l_col)
        diag_err = jnp.maximum(diag_err - l_col**2, 0.0)
        diag_err = jnp.where(is_small, diag_err.at[pivot].set(0.0), diag_err)

        return diag_err, L, pivots

    _, _, final_pivots = jax.lax.fori_loop(0, n_rank, body_fn, (diag_err, L, pivots))
    return final_pivots


def _pivoted_cholesky_phi(phi_weighted, n_rank, shift):
    """Specialized pivoted Cholesky for phi decomposition.

    Thin wrapper: the "same factor on both sides" case of
    ``pivoted_cholesky_pair_pivots`` (see its docstring). Kept as a
    distinct name at TC's existing call site rather than inlining, so
    that site's intent stays self-documenting.
    """
    return pivoted_cholesky_pair_pivots(phi_weighted, phi_weighted, n_rank, shift)


def _pivoted_cholesky_grad(phi_weighted, grad_phi_weighted, n_rank, shift):
    """Specialized pivoted Cholesky for gradient decomposition.

    Thin wrapper: the "two different factors" case of
    ``pivoted_cholesky_pair_pivots`` (see its docstring), with
    grad_phi_weighted's (n_orb, n_grid, 3) shape flattened to the 2-D
    (n_orb*3, n_grid) form the shared primitive expects -- the
    transpose-then-reshape collapses (orb, xyz) into one feature axis
    so the dot-product reduction still sums over exactly what the
    original per-component ``for c in range(3)`` loop summed over.
    """
    n_orb, n_grid, _ = grad_phi_weighted.shape
    grad_flat = grad_phi_weighted.transpose(0, 2, 1).reshape(n_orb * 3, n_grid)
    return pivoted_cholesky_pair_pivots(phi_weighted, grad_flat, n_rank, shift)






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
