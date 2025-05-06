from pytc.autodiff.jastrow import Jastrow
import jax.numpy as jnp
import jax.tree_util as tree_util
import h5py
import pickle
import numpy as np # Needed for saving/loading attributes
import os

class CompositeJastrow(Jastrow):
    """Combines multiple Jastrow factors by adding their exponents."""
    
    def __init__(self, jastrows):
        """Initialize with list of Jastrow factors.
        
        Args:
            jastrows: List of Jastrow instances to combine
        """
        super().__init__()
        self.jastrows = jastrows
        # Track jastrow identifiers for filtering
        self.jastrow_types = [j.__class__.__name__ for j in jastrows]
        self.jastrow_names = [j.name for j in jastrows]
        
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
        """Save parameters for all Jastrow factors into an HDF5 file.

        Args:
            params: List of parameter PyTrees, one per Jastrow factor.
            filename: Name of the HDF5 file to save to.
        """
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'

        print(f"Saving composite Jastrow parameters to {filename}...")
        with h5py.File(filename, 'w') as f:
            f.attrs['num_jastrows'] = len(self.jastrows)
            for i, param_pytree in enumerate(params):
                group = f.create_group(f'jastrow_{i}')
                leaves, treedef = tree_util.tree_flatten(param_pytree)
                # Store the treedef serialization as a byte string attribute
                group.attrs['treedef'] = np.void(pickle.dumps(treedef))
                group.attrs['num_leaves'] = len(leaves)
                # Store each leaf array as a dataset
                for j, leaf in enumerate(leaves):
                    group.create_dataset(f'leaf_{j}', data=np.asarray(leaf))
        print("Save complete.")

    def read_params(self, filename='jastrow_params.hdf5'):
        """Read parameters for all Jastrow factors from an HDF5 file.

        Args:
            filename: Name of the HDF5 file to read from.

        Returns:
            List of parameter PyTrees, matching the structure expected by the composite Jastrow.
            Returns None if the file does not exist.
        """
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'

        if not os.path.exists(filename):
            print(f"Parameter file {filename} not found. Returning None.")
            return None

        print(f"Reading composite Jastrow parameters from {filename}...")
        loaded_params = []
        with h5py.File(filename, 'r') as f:
            num_jastrows_expected = len(self.jastrows)
            num_jastrows_file = f.attrs.get('num_jastrows', -1)

            if num_jastrows_file != num_jastrows_expected:
                print(f"Warning: Mismatch in number of Jastrow factors. Expected {num_jastrows_expected}, found {num_jastrows_file} in file.")
                # Decide how to handle mismatch, here we proceed but only load up to expected number
                num_to_load = min(num_jastrows_expected, num_jastrows_file)
            else:
                num_to_load = num_jastrows_expected

            for i in range(num_to_load):
                group_name = f'jastrow_{i}'
                if group_name not in f:
                    raise IOError(f"Group {group_name} not found in {filename}.")
                group = f[group_name]

                # Load the treedef from the attribute
                treedef_bytes = group.attrs.get('treedef')
                if treedef_bytes is None:
                     raise IOError(f"'treedef' attribute not found in group {group_name} in {filename}.")
                treedef = pickle.loads(treedef_bytes.tobytes())

                # Load the leaves
                num_leaves = group.attrs.get('num_leaves', -1)
                if num_leaves == -1:
                     raise IOError(f"'num_leaves' attribute not found in group {group_name} in {filename}.")

                leaves = []
                for j in range(num_leaves):
                    dataset_name = f'leaf_{j}'
                    if dataset_name not in group:
                        raise IOError(f"Dataset {dataset_name} not found in group {group_name} in {filename}.")
                    # Load dataset and convert back to jnp array
                    leaves.append(jnp.array(group[dataset_name][()]))

                # Reconstruct the PyTree
                param_pytree = tree_util.tree_unflatten(treedef, leaves)
                loaded_params.append(param_pytree)

            # If file had fewer jastrows than expected, potentially initialize remaining ones
            if num_to_load < num_jastrows_expected:
                 print(f"Initializing default parameters for remaining {num_jastrows_expected - num_to_load} Jastrow factors.")
                 for i in range(num_to_load, num_jastrows_expected):
                      loaded_params.append(self.jastrows[i].init_params())

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