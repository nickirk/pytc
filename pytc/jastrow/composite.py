from pytc.jastrow import Jastrow
import jax.numpy as jnp
import jax.tree_util as tree_util
import h5py
import pickle
import numpy as np # Needed for saving/loading attributes
import os
from flax import struct
from typing import List

@struct.dataclass
class CompositeJastrow(Jastrow):
    """Combines multiple Jastrow factors by adding their exponents."""
    jastrows: List[Jastrow]
    jastrow_types: List[str] = struct.field(pytree_node=False)
    jastrow_names: List[str] = struct.field(pytree_node=False)
    name: str = struct.field(pytree_node=False, default=None)
    
    @classmethod
    def create(cls, jastrows, name=None):
        jastrow_types = [j.__class__.__name__ for j in jastrows]
        jastrow_names = [j.name for j in jastrows]
        return cls(name=name, jastrows=jastrows, jastrow_types=jastrow_types, jastrow_names=jastrow_names)
        
    def _compute(self, r1, r2, params):
        """Compute sum of Jastrow exponents.
        
        Args:
            r1, r2: Electron positions
            params: List of parameter sets, one per Jastrow factor
        """
        total = 0.0
        idx = 0
        for jastrow in self.jastrows:
            total += jastrow._compute(r1, r2, params[idx])
            idx += 1
        return total
    
    def get_log_grads_r1(self, r1, r2, params):
        """Sum gradients and laplacians, handling NCusp normalization."""
        grad_total = jnp.zeros(3)
        lap_total = 0.0
        idx = 0
        
        for jastrow in self.jastrows:
            grad_u, lap_u = jastrow.get_log_grads_r1(r1, r2, params[idx])
            grad_total += grad_u
            lap_total += lap_u
            idx += 1
            
        return grad_total, lap_total
    
    def get_log_grads_r2(self, r1, r2, params):
        """Sum gradients and laplacians, handling NCusp normalization."""
        grad_total = jnp.zeros(3)
        lap_total = 0.0
        idx = 0
        
        for jastrow in self.jastrows:
            grad_u, lap_u = jastrow.get_log_grads_r2(r1, r2, params[idx])
            grad_total += grad_u
            lap_total += lap_u
            idx += 1
            
        return grad_total, lap_total

    def grad_r_batch_laux(self, r1_batch, r2_batch, params):
        """Accumulate component-specific L_aux derivative fast paths."""
        grad_total = jnp.zeros(
            (r1_batch.shape[0], r2_batch.shape[0], 3), dtype=r1_batch.dtype
        )
        for jastrow, jastrow_params in zip(self.jastrows, params):
            grad_total += jastrow.grad_r_batch_laux(
                r1_batch, r2_batch, jastrow_params
            )
        return grad_total

    def grad_r_batch_laux_residual(self, r1_batch, r2_batch, params):
        """Accumulate only the pair-dependent L_aux gradient pieces."""
        grad_total = jnp.zeros(
            (r1_batch.shape[0], r2_batch.shape[0], 3), dtype=r1_batch.dtype
        )
        for jastrow, jastrow_params in zip(self.jastrows, params):
            grad_total += jastrow.grad_r_batch_laux_residual(
                r1_batch, r2_batch, jastrow_params
            )
        return grad_total

    def grad_r_batch_residual(self, r1_batch, r2_batch, params):
        """Accumulate ordinary pair-gradient residuals for exact splitting."""
        grad_total = jnp.zeros(
            (r1_batch.shape[0], r2_batch.shape[0], 3), dtype=r1_batch.dtype
        )
        for jastrow, jastrow_params in zip(self.jastrows, params):
            grad_total += jastrow.grad_r_batch_residual(
                r1_batch, r2_batch, jastrow_params
            )
        return grad_total

    def laux_one_grid_gradient(self, r_batch, params):
        """Sum exact integration-coordinate-independent component gradients."""
        grad_total = None
        for jastrow, jastrow_params in zip(self.jastrows, params):
            contribution = jastrow.laux_one_grid_gradient(r_batch, jastrow_params)
            if contribution is not None:
                grad_total = (
                    contribution if grad_total is None else grad_total + contribution
                )
        return grad_total

    def grad_params(self, r1, r2, params):
        """Compute gradient of u w.r.t parameters.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: List of parameter sets, one per Jastrow factor
            
        Returns:
            List of gradients, matching the structure of params
        """
        grads = []
        for i, jastrow in enumerate(self.jastrows):
            grad = jastrow.grad_params(r1, r2, params[i])
            grads.append(grad)
        return grads

    def init_params(self):
        """Initialize parameters for all Jastrow factors."""
        return [j.init_params() for j in self.jastrows]

    def save_params(self, params, filename='jastrow_params.hdf5'):
        """Save parameters for all Jastrow factors into an HDF5 file."""
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'

        print(f"Saving composite Jastrow parameters to {filename}...")
        with h5py.File(filename, 'w') as f:
            f.attrs['num_jastrows'] = len(self.jastrows)
            
            for i, param_dict in enumerate(params):
                group = f.create_group(f'jastrow_{i}')
                for key, value in param_dict.items():
                    # For nested structures like net_vars, use tree_flatten
                    if isinstance(value, dict):
                        nested_group = group.create_group(key)
                        leaves, treedef = tree_util.tree_flatten(value)
                        nested_group.attrs['treedef'] = np.void(pickle.dumps(treedef))
                        nested_group.attrs['num_leaves'] = len(leaves)
                        for j, leaf in enumerate(leaves):
                            nested_group.create_dataset(f'leaf_{j}', data=np.asarray(leaf))
                    else:
                        group.create_dataset(key, data=np.asarray(value))

    def read_params(self, filename='jastrow_params.hdf5'):
        """Read parameters for all Jastrow factors from an HDF5 file."""
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'

        if not os.path.exists(filename):
            print(f"Parameter file {filename} not found. Returning None.")
            return None

        print(f"Reading composite Jastrow parameters from {filename}...")
        loaded_params = []
        with h5py.File(filename, 'r') as f:
            num_jastrows = f.attrs['num_jastrows']
            for i in range(num_jastrows):
                group = f[f'jastrow_{i}']
                param_dict = {}
                
                for key in group.keys():
                    if isinstance(group[key], h5py.Group):  # Nested structure (e.g., net_vars)
                        nested_group = group[key]
                        treedef = pickle.loads(nested_group.attrs['treedef'].tobytes())
                        leaves = []
                        for j in range(nested_group.attrs['num_leaves']):
                            leaves.append(jnp.array(nested_group[f'leaf_{j}'][()]))
                        param_dict[key] = tree_util.tree_unflatten(treedef, leaves)
                    else:
                        param_dict[key] = jnp.array(group[key][()])
                
                loaded_params.append(param_dict)

        print("Read complete.")
        return loaded_params
    
    def grad_r(self, r1, r2, params):
        """Compute gradient of u w.r.t r1 coordinates.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: List of parameter sets, one per Jastrow factor
            
        Returns:
            Gradient array of shape (3,)
        """
        grad_total = jnp.zeros(3)
        idx = 0
        
        for jastrow in self.jastrows:
            grad_u = jastrow.grad_r(r1, r2, params[idx])
            grad_total += grad_u
            idx += 1
            
        return grad_total
    
    def laplacian_r(self, r1, r2, params):
        """Compute Laplacian of u w.r.t r1 coordinates.
        
        Args:
            r1: Array of shape (3,) for first electron position
            r2: Array of shape (3,) for second electron position
            params: List of parameter sets, one per Jastrow factor
            
        Returns:
            Laplacian value (scalar)
        """
        lap_total = 0.0
        idx = 0
        
        for jastrow in self.jastrows:
            lap_u = jastrow.laplacian_r(r1, r2, params[idx])
            lap_total += lap_u
            idx += 1
            
        return lap_total
