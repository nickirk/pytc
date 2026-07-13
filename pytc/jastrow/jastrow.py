"""JAX-based autodiff implementations for pytc."""

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from typing import Optional
import folx
from flax import struct
from functools import partial

@struct.dataclass
class Jastrow:
    """Abstract base class for JAX-based Jastrow factors.
    
    This class defines the interface for Jastrow factors. Unlike the previous implementation,
    parameters are not stored in the instance but passed directly to methods that need them.
    This aligns better with JAX's philosophy for parameter handling and computational graph tracing.
    """
    # name field is removed from base to avoid dataclass inheritance issues with defaults.
    # Subclasses should define 'name' field if needed.
    
    def _compute(self, r1, r2, params):
        """Core computation of Jastrow exponent u.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
            
        Returns:
            Jastrow exponent value u
        """
        raise NotImplementedError

    def __call__(self, r1, r2, params):
        """Evaluate Jastrow factor J = exp(u) for a single pair.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
            
        Returns:
            Jastrow factor value J = exp(u)
        """
        return jnp.exp(self._compute(r1, r2, params))
    
    def grad_r(self, r1, r2, params):
        """Compute gradient of u w.r.t r1 coordinates.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
            
        Returns:
            Gradient array of shape (3,)
        """
        def scalar_fn(x):
            return self._compute(x, r2, params).reshape(-1)[0]
        return jax.grad(scalar_fn)(r1)

    def grad_r_batch(self, r1_batch, r2_batch, params):
        """Compute gradients for a batch of r1 and r2 points.
        
        Args:
            r1_batch: (batch_size_out, 3)
            r2_batch: (batch_size_in, 3)
            params: Jastrow parameters
            
        Returns:
            Gradients of shape (batch_size_out, batch_size_in, 3)
        """
        # Default implementation using vmap over grad_r
        @partial(jax.vmap, in_axes=(None, 0))
        def grad_fn(r1, r2):
            return self.grad_r(r1, r2, params)
        
        return jax.vmap(grad_fn, in_axes=(0, None))(r1_batch, r2_batch)
    
    def laplacian_r(self, r1, r2, params):
        """Compute Laplacian of u w.r.t r1 coordinates.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
            
        Returns:
            Laplacian value (scalar)
        """
        def scalar_fn(x):
            return self._compute(x, r2, params).reshape(-1)[0]
            
        # Use folx for efficient forward-mode Laplacian
        return folx.forward_laplacian(scalar_fn)(r1).laplacian
    
    
    def grad_params(self, r1, r2, params):
        """Compute gradient of u w.r.t parameters.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
            
        Returns:
            Gradient array with same shape as params
        """
        return jax.grad(lambda p: jnp.sum(self._compute(r1, r2, p)))(params)
    
    def get_log_grads_r1(self, r1, r2, params):
        """Compute ∇u and ∇²u w.r.t first electron coordinates.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
            
        Returns:
            tuple (grad_u, lapl_u) containing:
                grad_u: gradient of u w.r.t r1, shape (3,)
                lapl_u: laplacian of u w.r.t r1 (scalar)
        """
        def scalar_fn(x):
            return self._compute(x, r2, params).reshape(-1)[0]
            
        # Use folx for efficient forward-mode gradient and Laplacian
        fwd_lapl = folx.forward_laplacian(scalar_fn)(r1)
        grad_u = fwd_lapl.jacobian.dense_array
        lapl_u = fwd_lapl.laplacian
            
        return grad_u, lapl_u
    
    def get_log_grads_r2(self, r1, r2, params):
        """Compute ∇u and ∇²u w.r.t second electron coordinates.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: Jastrow parameters
            
        Returns:
            tuple (grad_u, lapl_u) containing:
                grad_u: gradient of u w.r.t r2, shape (3,)
                lapl_u: laplacian of u w.r.t r2 (scalar)
        """
        def scalar_fn(x):
            return self._compute(r1, x, params).reshape(-1)[0]
            
        # Use folx for efficient forward-mode gradient and Laplacian
        fwd_lapl = folx.forward_laplacian(scalar_fn)(r2)
        grad_u = fwd_lapl.jacobian.dense_array
        lapl_u = fwd_lapl.laplacian
            
        return grad_u, lapl_u
    
    def init_params(self, **kwargs):
        """Initialize parameters. Subclasses should implement this."""
        pass

    def get_pair_grid_grad_lap(self, elec_coords, params):
        """Compute grad_1/lap_1 of u(r_i, r_j) wrt r_i for ALL (i, j) pairs
        at once.

        Default implementation: a per-pair vmap grid built directly from
        ``get_log_grads_r1`` -- correct for any subclass, but recomputes
        each electron's per-pair quantities independently for every one of
        the N pairs it appears in (O(N^2*M) redundant work for Jastrows
        with atom-dependent structure). Subclasses with a cheaper
        whole-electron-set formulation (e.g. ``BoysHandyAnalytical``,
        which precomputes per-electron tables once and assembles the pair
        grid via a scan) should override this method; callers never branch
        on which implementation is in use -- polymorphism handles it.

        Args:
            elec_coords: (N, 3) all electron positions.
            params: Jastrow parameters.

        Returns:
            (grad_pair, lap_pair): grad_pair has shape (N, N, 3) and
            lap_pair has shape (N, N), where [i, j] holds grad_1/lap_1 of
            u(r_i, r_j) wrt r_i. Diagonal (i == j) entries are meaningless
            and are masked out by the caller.
        """
        def compute_pair(i, r_i, j, r_j):
            # Displace the i == j diagonal before evaluating: the caller
            # masks it out, but a NaN produced at r_i == r_j (0/0 in
            # autodiff'd pair norms) survives a multiplicative mask
            # (NaN * 0 = NaN). The displaced value is discarded, so its
            # magnitude is irrelevant; index-based so genuinely coincident
            # DISTINCT electrons still propagate their true value.
            r_j_safe = jnp.where(i == j, r_j + 1.0, r_j)
            return self.get_log_grads_r1(r_i, r_j_safe, params)

        idx = jnp.arange(elec_coords.shape[0])
        inner = jax.vmap(compute_pair, in_axes=(None, None, 0, 0))
        outer = jax.vmap(inner, in_axes=(0, 0, None, None))
        return outer(idx, elec_coords, idx, elec_coords)
