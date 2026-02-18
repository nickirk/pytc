"""Optimizer implementations for VMC."""

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as spla
import optax
import folx
from jax.tree_util import tree_map
from jax.lax import stop_gradient
import jax.flatten_util

class NewtonOptimizer:
    """Newton Optimizer (formerly Matrix-Free Optimizer).
    
    Supports:
    - Stochastic Reconfiguration (SR) / Natural Gradient for Energy Minimization
      (curvature="fisher")
    - Gauss-Newton for Variance Minimization
      (curvature="gauss_newton")
      
    Solvers:
    - "cg": Conjugate Gradient (iterative, matrix-free)
    - "exact" or "cholesky": Exact matrix inversion
    """
    def __init__(self, value_and_grad_func, learning_rate, damping=1e-3, maxiter=100, curvature_type="fisher", max_vmap_batch_size=0, solver="exact", solve_kwargs=None, jacobian_sample_size=0, clip_multiplier=5.0):
        self.value_and_grad_func = value_and_grad_func
        self.learning_rate = learning_rate
        self.damping = damping
        self.maxiter = maxiter
        self.curvature_type = curvature_type
        self.max_vmap_batch_size = max_vmap_batch_size
        self.solver = solver
        self.solve_kwargs = solve_kwargs if solve_kwargs is not None else {}
        self.jacobian_sample_size = jacobian_sample_size
        self.clip_multiplier = clip_multiplier

    def _get_vmap(self):
        """Return the appropriate vmap implementation.

        Automatically detects multi-GPU environments via get_vmap_fn.
        """
        from .sharding import get_vmap_fn
        return get_vmap_fn(max_vmap_batch_size=self.max_vmap_batch_size)

    def init(self, params, rng, batch):
        return 0  # step count

    def step(self, params, state, rng, batch, global_step_int=None):
        walkers, ansatz = batch
        
        # 2. Define MVP or Solve Exact
        if self.solver == "exact" or self.solver == "cholesky":
            # Exact inversion: (M + lambda I) delta = -g
            
            if self.curvature_type == "fisher":
                # SR: S = Cov(grad_log_psi)
                # Need loss + grads from value_and_grad_func, plus Jacobian separately.
                (loss, aux_data), grads = self.value_and_grad_func(params, batch)
                
                # J_i = d(log_psi(w_i))/dp
                def single_log_psi_grad(w, p):
                    return jax.grad(lambda pp: ansatz(w, pp)[0][1])(p)
                
                # Compute Jacobian for all walkers: Shape (N, P)
                vmap_fn = self._get_vmap()
                jac = vmap_fn(single_log_psi_grad, in_axes=(0, None))(walkers, params)
                
                # Flatten params structure for linear algebra
                jac_flat, params_treedef = jax.tree_util.tree_flatten(jac)
                jac_mat = jnp.concatenate([jnp.reshape(leaf, (walkers.shape[0], -1)) for leaf in jac_flat], axis=1)
                
                # Center the Jacobian (Covariance)
                jac_centered = jac_mat - jnp.mean(jac_mat, axis=0, keepdims=True)
                
                # S = 1/N * J.T @ J
                n_walkers = walkers.shape[0]
                curvature_mat = (jac_centered.T @ jac_centered) / n_walkers
                
            elif self.curvature_type == "gauss_newton":
                # GN: G = 2/M * J_centered.T @ J_centered
                # J_i = d(E_L(w_i))/dp
                #
                # Optimization: compute Jacobian and local energies in one pass,
                # then derive the variance loss and gradient analytically:
                #   variance = sum((E - mean(E))^2) / (M - 1)
                #   grad_variance = 2/(M-1) * J^T @ (E - mean(E))
                #
                # When jacobian_sample_size > 0, a random subset of walkers is
                # used for the Jacobian (gradient + curvature), reducing cost
                # from O(N) to O(M) local-energy differentiations per step.
                
                def single_local_energy_and_grad(w, p):
                    """Compute both E_L(w) and grad_p E_L(w) in one pass."""
                    return jax.value_and_grad(lambda pp: ansatz.local_energy(w, pp)[0])(p)
                
                n_walkers_total = walkers.shape[0]
                
                # Sub-sample walkers for the Jacobian if requested
                if self.jacobian_sample_size > 0 and self.jacobian_sample_size < n_walkers_total:
                    sample_size = self.jacobian_sample_size
                    # In multi-device mode, shard_map requires the sharded axis
                    # length to be divisible by device count.
                    from .sharding import is_multi_gpu, n_devices
                    if is_multi_gpu():
                        ndev = n_devices()
                        if sample_size % ndev != 0:
                            sample_size = max(ndev, (sample_size // ndev) * ndev)
                        sample_size = min(sample_size, n_walkers_total)
                    # Use rng to select a random subset of walker indices
                    indices = jax.random.choice(rng, n_walkers_total,
                                                shape=(sample_size,), replace=False)
                    indices = jnp.sort(indices).astype(jnp.int32)  # sort for deterministic gather
                    sub_walkers = jax.tree_util.tree_map(lambda x: x[indices], walkers)
                else:
                    sample_size = n_walkers_total
                    sub_walkers = walkers
                
                # Compute energies and Jacobian for (sub-sampled) walkers
                vmap_fn = self._get_vmap()
                energies, jac = vmap_fn(
                    single_local_energy_and_grad, 
                    in_axes=(0, None)
                )(sub_walkers, params)
                
                n_walkers = sample_size
                
                # Clip energies to suppress outliers.  The gradient is
                # 2/(M-1) * J^T @ (E - mean(E)), so clipping energies
                # naturally limits the influence of extreme walkers.
                if self.clip_multiplier > 0:
                    e_mean_raw = jnp.mean(energies)
                    e_std_raw = jnp.mean(jnp.abs(energies - e_mean_raw))
                    energies = jnp.clip(
                        energies,
                        e_mean_raw - self.clip_multiplier * e_std_raw,
                        e_mean_raw + self.clip_multiplier * e_std_raw,
                    )
                
                # Flatten Jacobian params structure to (M, P_total) matrix
                jac_flat, params_treedef = jax.tree_util.tree_flatten(jac)
                jac_mat = jnp.concatenate([jnp.reshape(leaf, (n_walkers, -1)) for leaf in jac_flat], axis=1)
                
                # Compute variance loss and auxiliary data analytically from energies
                e_mean = jnp.mean(energies)
                e_std = jnp.std(energies)
                energy_diff = energies - e_mean
                loss = jnp.sum(energy_diff**2) / (n_walkers - 1)
                aux_data = (e_mean, e_std)
                
                # Compute variance gradient analytically: 
                # grad_variance = 2/(M-1) * J^T @ (E - mean(E))
                grads_vec = (2.0 / (n_walkers - 1)) * (jac_mat.T @ energy_diff)
                
                # Center the Jacobian for curvature matrix
                jac_centered = jac_mat - jnp.mean(jac_mat, axis=0, keepdims=True)
                curvature_mat = (2.0 / n_walkers) * (jac_centered.T @ jac_centered)
            
            else:
                raise ValueError(f"Unknown curvature type: {self.curvature_type}")
            
            # Add damping
            curvature_mat = curvature_mat + self.damping * jnp.eye(curvature_mat.shape[0])
            
            # Flatten gradients to match matrix (for fisher, grads are pytree; for gauss_newton, already flat)
            if self.curvature_type == "fisher":
                grads_vec, unravel_fn = jax.flatten_util.ravel_pytree(grads)
            else:
                # gauss_newton: grads_vec is already flat, need unravel_fn
                _, unravel_fn = jax.flatten_util.ravel_pytree(params)
            
            # Solve linear system
            # (M + lambda I) delta = -g
            # Use provided solve_kwargs or default to assume_a='pos'
            solve_kwargs = self.solve_kwargs.copy()
            if "assume_a" not in solve_kwargs:
                solve_kwargs["assume_a"] = "pos"
                
            delta_vec = jax.scipy.linalg.solve(curvature_mat, -grads_vec, **solve_kwargs)
            
            # Unflatten delta to match params structure
            delta = unravel_fn(delta_vec)
            
            # Update
            new_params = jax.tree_util.tree_map(lambda p, d: p + self.learning_rate * d, params, delta)
            
            return new_params, state + 1, {"loss": loss, "aux": aux_data}

        # CG Solver: need loss + grads from value_and_grad_func
        (loss, aux_data), grads = self.value_and_grad_func(params, batch)
        
        if self.curvature_type == "fisher":
            # SR: S = Cov(grad_log_psi)
            
            # Helper for single walker log_psi
            def single_log_psi(w, p):
                return ansatz(w, p)[0][1]

            def mvp(v):
                # Forward: w = J v
                # Compute JVP per walker: d(log_psi)/dp * v
                def compute_jvp(w):
                    _, tangent = jax.jvp(lambda p: single_log_psi(w, p), (params,), (v,))
                    return tangent

                vmap_fn = self._get_vmap()
                w = vmap_fn(compute_jvp)(walkers)
                
                w_centered = w - jnp.mean(w)
                
                # Backward: J.T w_centered
                # Compute VJP per walker: (d(log_psi)/dp)^T * w_i
                def compute_vjp(w_el, w_val):
                    _, vjp_fun = jax.vjp(lambda p: single_log_psi(w_el, p), params)
                    return vjp_fun(w_val)[0]
                
                per_walker_grads = vmap_fn(compute_vjp)(walkers, w_centered)
                
                # Sum over walkers
                u = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), per_walker_grads)
                
                # S = 1/N * J.T @ (J @ v centered)
                n_walkers = walkers.shape[0]
                return jax.tree_util.tree_map(lambda x: x / n_walkers, u)

        elif self.curvature_type == "gauss_newton":
            # GN: G = 2/N * J.T @ J
            
            # Helper for single walker local energy
            def single_local_energy(w, p):
                return ansatz.local_energy(w, p)[0]

            def mvp(v):
                # Forward: w = J v
                def compute_jvp(w):
                    _, tangent = jax.jvp(lambda p: single_local_energy(w, p), (params,), (v,))
                    return tangent

                vmap_fn = self._get_vmap()
                w = vmap_fn(compute_jvp)(walkers)
                
                w_centered = w - jnp.mean(w)
                
                # Backward: J.T w
                def compute_vjp(w_el, w_val):
                    _, vjp_fun = jax.vjp(lambda p: single_local_energy(w_el, p), params)
                    return vjp_fun(w_val)[0]
                
                per_walker_grads = vmap_fn(compute_vjp)(walkers, w_centered)
                
                # Sum over walkers
                u = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), per_walker_grads)
                
                n_walkers = walkers.shape[0]
                return jax.tree_util.tree_map(lambda x: 2.0 * x / n_walkers, u)
        
        else:
            raise ValueError(f"Unknown curvature type: {self.curvature_type}")

        # Add damping
        def damped_mvp(v):
            mvp_val = mvp(v)
            return jax.tree_util.tree_map(lambda x, y: x + self.damping * y, mvp_val, v)

        # 3. Solve (S + lambda I) delta = -g
        # RHS is -grads
        rhs = jax.tree_util.tree_map(lambda x: -x, grads)
        
        delta, info = spla.cg(
            damped_mvp, 
            rhs, 
            maxiter=self.maxiter
        )
        
        # 4. Update
        new_params = jax.tree_util.tree_map(lambda p, d: p + self.learning_rate * d, params, delta)
        
        return new_params, state + 1, {"loss": loss, "aux": aux_data}

def create_optimizer(optimizer_type, learning_rate, opt_kwargs=None):
    """Create an optimizer based on specified type and parameters."""
    if opt_kwargs is None:
        opt_kwargs = {}
    
    base_kwargs = {}
    merged_kwargs = {**base_kwargs, **opt_kwargs}
    def schedule_lr(step):
        return learning_rate / (1.0 + step/100)
    
    if optimizer_type.lower() == "adam":
        #return optax.adamw(learning_rate=schedule_lr)
        return optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "sgd":
        return optax.chain(
            optax.sgd(learning_rate=learning_rate),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "rmsprop":
        return optax.chain(
            optax.scale_by_rms(decay=merged_kwargs.get("decay", 0.9), eps=merged_kwargs.get("eps", 1e-8)),
            optax.scale_by_learning_rate(schedule_lr),
        )
    elif optimizer_type.lower() == "lion":
        return optax.lion(learning_rate=learning_rate, b1=merged_kwargs.get("b1", 0.9), b2=merged_kwargs.get("b2", 0.99))
    elif optimizer_type.lower() == "newton":
        if "value_and_grad_func" not in merged_kwargs:
            raise ValueError("Newton optimizer requires value_and_grad_func in opt_kwargs")
            
        return NewtonOptimizer(
            value_and_grad_func=merged_kwargs["value_and_grad_func"],
            learning_rate=learning_rate,
            damping=merged_kwargs.get("damping", 1e-3),
            maxiter=merged_kwargs.get("maxiter", 100),
            curvature_type=merged_kwargs.get("curvature", "fisher"),
            max_vmap_batch_size=merged_kwargs.get("max_vmap_batch_size", 0),
            solver=merged_kwargs.get("solver", "exact"),
            solve_kwargs=merged_kwargs.get("solve_kwargs", None),
            jacobian_sample_size=merged_kwargs.get("jacobian_sample_size", 0),
            clip_multiplier=merged_kwargs.get("clip_multiplier", 5.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_gradient_mask(ansatz, params, frozen_params):
    """Create a gradient mask PyTree for the combined params structure.
    
    Assumes params = [jastrow_params, linear_coeffs]. The mask is applied
    only to the jastrow_params part based on frozen_params identifiers.
    The linear_coeffs part of the mask is always True (not frozen).

    Args:
        ansatz: The wavefunction ansatz object.
        params: The combined parameters PyTree [jastrow_params, linear_coeffs].
        frozen_params: A list of identifiers (int index or str name/type)
                       for Jastrow factors whose parameters should be frozen.

    Returns:
        A PyTree with the same structure as params, where frozen parameters
        are wrapped with `jax.lax.stop_gradient`.
    """
    if not frozen_params:
        return params  # No freezing requested, return params unchanged

    if not isinstance(params, (list, tuple)) or len(params) != 2:
        raise ValueError("`params` must be a list or tuple: [jastrow_params, linear_coeffs]")

    jastrow_params = params[0]
    linear_coeffs = params[1]

    print(f"Creating gradient mask for frozen Jastrow parameters: {frozen_params}")
    jastrows = ansatz.jastrow.jastrows
    if not isinstance(jastrow_params, (list, tuple)) or len(jastrow_params) != len(jastrows):
        raise TypeError(f"Jastrow params structure (length {len(jastrow_params)}) does not match jastrows (length {len(jastrows)})")

    # Deep copy the parameters to avoid modifying the input
    masked_jastrow_params = []
    for i, (param_pytree, jastrow) in enumerate(zip(jastrow_params, jastrows)):
        should_freeze = False
        for fp in frozen_params:
            if isinstance(fp, int) and fp == i:
                should_freeze = True
                break
            elif isinstance(fp, str):
                if fp == jastrow.__class__.__name__ or fp == getattr(jastrow, 'name', None):
                    should_freeze = True
                    break
        
        if should_freeze:
            # Apply stop_gradient to all leaves in the frozen parameter PyTree
            param_pytree = tree_map(stop_gradient, param_pytree)
            print(f"  Freezing Jastrow {i}: type={jastrow.__class__.__name__}, name={getattr(jastrow, 'name', None)}")
        masked_jastrow_params.append(param_pytree)

    # Return the masked parameters
    return [masked_jastrow_params, linear_coeffs]

def apply_gradient_mask(grads, mask):
    """Apply gradient mask to gradients to freeze parameters.
    
    Args:
        grads: The gradient PyTree
        mask: The mask PyTree created by create_gradient_mask
        
    Returns:
        A PyTree with the same structure as grads, where gradients for frozen
        parameters are set to zero.
    """
    if mask is None:
        return grads
        
    def _apply_mask(g, m):
        # If m is already stop_gradient'd, zero out the gradient
        if isinstance(m, type(stop_gradient(m))):
            return jnp.zeros_like(g)
        return g
        
    return tree_map(_apply_mask, grads, mask)
