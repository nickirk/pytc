"""Optimizer implementations for VMC."""

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as spla
import optax
import kfac_jax
import folx
from typing import Dict, Any, Optional
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
    def __init__(self, value_and_grad_func, learning_rate, damping=1e-3, maxiter=100, curvature_type="fisher", max_vmap_batch_size=0, solver="exact", solve_kwargs=None):
        self.value_and_grad_func = value_and_grad_func
        self.learning_rate = learning_rate
        self.damping = damping
        self.maxiter = maxiter
        self.curvature_type = curvature_type
        self.max_vmap_batch_size = max_vmap_batch_size
        self.solver = solver
        self.solve_kwargs = solve_kwargs if solve_kwargs is not None else {}

    def init(self, params, rng, batch):
        return 0  # step count

    def step(self, params, state, rng, batch, global_step_int=None):
        walkers, ansatz = batch
        
        # 1. Compute Gradients
        # We pass batch directly to value_and_grad_func.
        # It should handle (walkers, ansatz) or walkers.
        (loss, aux_data), grads = self.value_and_grad_func(params, batch)
        
        # 2. Define MVP or Solve Exact
        if self.solver == "exact" or self.solver == "cholesky":
            # Exact inversion: (M + lambda I) delta = -g
            
            if self.curvature_type == "fisher":
                # SR: S = Cov(grad_log_psi)
                # J_i = d(log_psi(w_i))/dp
                
                def single_log_psi_grad(w, p):
                    return jax.grad(lambda pp: ansatz(w, pp)[0][1])(p)
                
                # Compute Jacobian for all walkers: Shape (N, P)
                # Use batched_vmap if requested to avoid OOM
                if self.max_vmap_batch_size > 0:
                    jac = folx.batched_vmap(single_log_psi_grad, max_batch_size=self.max_vmap_batch_size, in_axes=(0, None))(walkers, params)
                else:
                    jac = jax.vmap(single_log_psi_grad, in_axes=(0, None))(walkers, params)
                
                # Flatten params structure for linear algebra
                jac_flat, params_treedef = jax.tree_util.tree_flatten(jac)
                # Concatenate all parameter gradients into a single matrix (N, P_total)
                # Note: This assumes all leaves are arrays. We need to handle this carefully.
                # A safer way is to flatten each leaf and concatenate.
                jac_mat = jnp.concatenate([jnp.reshape(leaf, (walkers.shape[0], -1)) for leaf in jac_flat], axis=1)
                
                # Center the Jacobian (Covariance)
                jac_centered = jac_mat - jnp.mean(jac_mat, axis=0, keepdims=True)
                
                # S = 1/N * J.T @ J
                n_walkers = walkers.shape[0]
                curvature_mat = (jac_centered.T @ jac_centered) / n_walkers
                
            elif self.curvature_type == "gauss_newton":
                # GN: G = 2/N * J.T @ J
                # J_i = d(E_L(w_i))/dp
                
                def single_local_energy_grad(w, p):
                    return jax.grad(lambda pp: ansatz.local_energy(w, pp)[0])(p)
                
                # Compute Jacobian for all walkers: Shape (N, P)
                if self.max_vmap_batch_size > 0:
                    jac = folx.batched_vmap(single_local_energy_grad, max_batch_size=self.max_vmap_batch_size, in_axes=(0, None))(walkers, params)
                else:
                    jac = jax.vmap(single_local_energy_grad, in_axes=(0, None))(walkers, params)
                
                # Flatten params structure
                jac_flat, params_treedef = jax.tree_util.tree_flatten(jac)
                jac_mat = jnp.concatenate([jnp.reshape(leaf, (walkers.shape[0], -1)) for leaf in jac_flat], axis=1)
                
                # Center the Jacobian to match the iterative solver and correctly minimize variance
                # The iterative solver centers the JVP output w = Jv - mean(Jv), which is equivalent
                # to using a centered Jacobian matrix in the quadratic form.
                jac_centered = jac_mat - jnp.mean(jac_mat, axis=0, keepdims=True)
                
                n_walkers = walkers.shape[0]
                curvature_mat = (2.0 / n_walkers) * (jac_centered.T @ jac_centered)
            
            else:
                raise ValueError(f"Unknown curvature type: {self.curvature_type}")
            
            # Add damping
            curvature_mat = curvature_mat + self.damping * jnp.eye(curvature_mat.shape[0])
            
            # Flatten gradients to match matrix
            grads_vec, unravel_fn = jax.flatten_util.ravel_pytree(grads)
            
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

        # 2. Define MVP (CG Solver)
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

                if self.max_vmap_batch_size > 0:
                    w = folx.batched_vmap(compute_jvp, max_batch_size=self.max_vmap_batch_size)(walkers)
                else:
                    w = jax.vmap(compute_jvp)(walkers)
                
                w_centered = w - jnp.mean(w)
                
                # Backward: J.T w_centered
                # Compute VJP per walker: (d(log_psi)/dp)^T * w_i
                def compute_vjp(w_el, w_val):
                    _, vjp_fun = jax.vjp(lambda p: single_log_psi(w_el, p), params)
                    return vjp_fun(w_val)[0]
                
                if self.max_vmap_batch_size > 0:
                    per_walker_grads = folx.batched_vmap(compute_vjp, max_batch_size=self.max_vmap_batch_size)(walkers, w_centered)
                else:
                    per_walker_grads = jax.vmap(compute_vjp)(walkers, w_centered)
                
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

                if self.max_vmap_batch_size > 0:
                    w = folx.batched_vmap(compute_jvp, max_batch_size=self.max_vmap_batch_size)(walkers)
                else:
                    w = jax.vmap(compute_jvp)(walkers)
                
                w_centered = w - jnp.mean(w)
                
                # Backward: J.T w
                def compute_vjp(w_el, w_val):
                    _, vjp_fun = jax.vjp(lambda p: single_local_energy(w_el, p), params)
                    return vjp_fun(w_val)[0]
                
                if self.max_vmap_batch_size > 0:
                    per_walker_grads = folx.batched_vmap(compute_vjp, max_batch_size=self.max_vmap_batch_size)(walkers, w_centered)
                else:
                    per_walker_grads = jax.vmap(compute_vjp)(walkers, w_centered)
                
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
    
    if optimizer_type.lower() == "kfac":
        # K-FAC requires a value_and_grad_func, which should be provided in opt_kwargs
        if "value_and_grad_func" not in merged_kwargs:
            raise ValueError("KFAC optimizer requires value_and_grad_func in opt_kwargs")
        
        return kfac_jax.Optimizer(
            value_and_grad_func=merged_kwargs["value_and_grad_func"],
            l2_reg=merged_kwargs.get("l2_reg", 0.0),
            value_func_has_aux=merged_kwargs.get("value_func_has_aux", False),
            value_func_has_state=merged_kwargs.get("value_func_has_state", False),
            value_func_has_rng=merged_kwargs.get("value_func_has_rng", False),
            learning_rate_schedule=merged_kwargs.get("learning_rate_schedule", None),
            use_adaptive_learning_rate=merged_kwargs.get("use_adaptive_learning_rate", True),
            use_adaptive_momentum=merged_kwargs.get("use_adaptive_momentum", True),
            use_adaptive_damping=merged_kwargs.get("use_adaptive_damping", True),
            initial_damping=merged_kwargs.get("initial_damping", 1.0),
            num_burnin_steps=merged_kwargs.get("num_burnin_steps", 0),  # Set to 0 by default to avoid requiring data_iterator
            multi_device=merged_kwargs.get("multi_device", False),
        )
    elif optimizer_type.lower() == "adam":
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
            solver=merged_kwargs.get("solver", "cg"),
            solve_kwargs=merged_kwargs.get("solve_kwargs", None)
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
            print("Warning: Freezing parameters does not work for KFAC yet.")
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
