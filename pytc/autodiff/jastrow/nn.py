import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Sequence

from pytc.autodiff.jastrow import Jastrow 

class MLP(nn.Module):
    """Multi-layer perceptron network using Flax with residual connections."""
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        
        for i, feat in enumerate(self.features[:-1]):
            # Store layer input for residual connection
            layer_input = x
            # Dense layer + activation
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
            # Add residual connection if shapes match
            if layer_input.shape[-1] == feat:
                x = x + layer_input
            
        # Final layer without residual connection
        x = nn.Dense(self.features[-1])(x)
        return x


class NeuralBase(Jastrow):
    """Base class for neural network-based Jastrow factors."""
    def __init__(self, mol, layer_widths=[16, 16], epsilon=1e-8, name=None, **kwargs):
        super().__init__(name=name)
        self.mol = mol
        self.nelectron = mol.nelectron
        self.nuclear_pos = jnp.array(mol.atom_coords())
        self.nuclear_charges = jnp.array(mol.atom_charges())
        self.features = [*layer_widths, 1]
        self.epsilon = epsilon
        self.net = MLP(features=self.features)
        # Params are now initialized in subclasses
        # key = kwargs.get('key', random.PRNGKey(0))
        # self.params = self.init_params(key=key)

    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)

    def _flatten_params(self, nested_net_params):
        """Flatten nested network parameter dictionary into 1D array."""
        flat_params = []
        # Assumes nested_net_params is the dict under the 'params' key from Flax
        for layer in nested_net_params.values():
            flat_params.extend([layer['kernel'].ravel(), layer['bias'].ravel()])
        return jnp.concatenate(flat_params)

    def _unflatten_params(self, flat_net_params, input_size):
        """Reconstruct nested network parameter dictionary from 1D array"""
        params = {}
        idx = 0
        prev_width = input_size
        
        for i, width in enumerate(self.features):
            kernel_size = prev_width * width
            bias_size = width
            
            # Ensure flat_net_params is not empty and idx is within bounds
            if idx + kernel_size + bias_size > flat_net_params.shape[0]:
                 raise ValueError(f"Insufficient elements in flat_net_params. "
                                  f"Needed: {idx + kernel_size + bias_size}, "
                                  f"Available: {flat_net_params.shape[0]}")

            kernel = flat_net_params[idx:idx + kernel_size].reshape((prev_width, width))
            idx += kernel_size
            bias = flat_net_params[idx:idx + bias_size]
            idx += bias_size
            
            params[f'Dense_{i}'] = {
                'kernel': kernel,
                'bias': bias
            }
            prev_width = width
            
        # Check if all parameters were used
        if idx != flat_net_params.shape[0]:
            print(f"Warning: Not all elements used in _unflatten_params. "
                  f"Used: {idx}, Total: {flat_net_params.shape[0]}")

        return {'params': params} # Return structure expected by net.apply

    def get_param_count(self, input_size):
        """Return parameter count for a single network"""
        total = 0
        prev_width = input_size
        for width in self.features:
            total += prev_width * width + width
            prev_width = width
        return total


class NeuralEN(NeuralBase):
    """Neural network for electron-nuclear correlations."""
    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)

    def init_params(self, **kwargs):
        key = kwargs.get('key', random.PRNGKey(0))
        dummy_x = jnp.zeros((1, len(self.nuclear_charges)))
        variables = self.net.init(key, dummy_x)
        # Initialize raw parameter for rc_en such that softplus(raw) ~ 0.1
        initial_rc_en_raw = 0.5
        return {
            'rc_en_raw': initial_rc_en_raw, 
            'net_params': self._flatten_params(variables['params']) * 0.0001
        }

    def _compute(self, r1, r2, params):
        # Extract raw decay parameter and network weights
        rc_en_raw = params['rc_en_raw']
        flat_net_params = params['net_params']
        
        # Ensure rc_en is positive using softplus
        rc_en = nn.softplus(rc_en_raw)

        r1n_dist = self._safe_norm(r1[None, :] - self.nuclear_pos)
        # Apply decay parameter
        r1n_feat = r1n_dist

        features = r1n_feat.reshape(1, -1)
        # Reconstruct network variables dictionary from flattened weights
        vars_dict = self._unflatten_params(flat_net_params, len(self.nuclear_charges))
        return self.net.apply(vars_dict, features)[0, 0] / (self.nelectron - 1)
    
    def get_log_grads_r2(self, r1, r2, params):
        return self.get_log_grads_r1(r2, r1, params)


class NeuralEE(NeuralBase):
    """Neural network for electron-electron correlations."""
    def init_params(self, **kwargs):
        key = kwargs.get('key', random.PRNGKey(0))
        dummy_x = jnp.zeros((1, 1))
        variables = self.net.init(key, dummy_x)
        # Initialize raw parameter for rc_ee such that softplus(raw) ~ 0.1
        initial_rc_ee_raw = 0.5
        return {
            'rc_ee_raw': initial_rc_ee_raw, 
            'net_params': self._flatten_params(variables['params']) * 0.0001
        }

    def _compute(self, r1, r2, params):
        # Extract raw decay parameter and network weights
        rc_ee_raw = params['rc_ee_raw']
        flat_net_params = params['net_params']

        # Ensure rc_ee is positive using softplus
        rc_ee = nn.softplus(rc_ee_raw)

        r12_dist = self._safe_norm(r1 - r2)
        # Apply decay parameter
        # r12_feat = r12_dist * jnp.exp(-0.5 * r12_dist)
        r12_feat = r12_dist / (1 + r12_dist) # Normalize to [0, 1]

        features = r12_feat.reshape(1, -1)
        # Reconstruct network variables dictionary from flattened weights
        vars_dict = self._unflatten_params(flat_net_params, 1)
        return self.net.apply(vars_dict, features)[0, 0]


class NeuralEEN(NeuralBase):
    """Neural network for electron-electron-nuclear correlations."""
    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)

    def init_params(self, **kwargs):
        key = kwargs.get('key', random.PRNGKey(0))
        input_size = 1+2*len(self.nuclear_charges)
        dummy_x = jnp.zeros((1, input_size))
        variables = self.net.init(key, dummy_x)
        # Initialize raw decay parameters
        initial_rc_raw = 0.5 # approx -2.25
        return {
            'rc_ee_raw': initial_rc_raw,
            'rc_en_raw': initial_rc_raw,
            'net_params': self._flatten_params(variables['params']) * 0.0001
        }

    def _compute(self, r1, r2, params):
        # Extract raw decay parameters and network weights
        rc_ee_raw = params['rc_ee_raw']
        rc_en_raw = params['rc_en_raw']
        flat_net_params = params['net_params']

        # Ensure decay parameters are positive using softplus
        rc_ee = nn.softplus(rc_ee_raw)
        rc_en = nn.softplus(rc_en_raw)

        # Calculate distances
        r12_dist = self._safe_norm(r1 - r2)
        r1n_dist = self._safe_norm(r1[None, :] - self.nuclear_pos)
        r2n_dist = self._safe_norm(r2[None, :] - self.nuclear_pos)

        # Apply decay parameters
        r12_feat = r12_dist  / (1+r12_dist) 
        r1n_feat = r1n_dist  / (1+r1n_dist) 
        r2n_feat = r2n_dist  / (1+r2n_dist) 
        #feat = r12_feat * r1n_feat * r2n_feat * jnp.exp(-0.5 * r1n_dist) * jnp.exp(-0.5 * r2n_dist) * jnp.exp(-0.5 * r12_dist)

        features = jnp.concatenate([
            jnp.asarray([r12_feat]), # Ensure it's an array
            r1n_feat,
            r2n_feat,
        ]).reshape(1, -1)
        
        # Reconstruct network variables dictionary from flattened weights
        input_size = 1+2*len(self.nuclear_charges)
        vars_dict = self._unflatten_params(flat_net_params, input_size)
        return self.net.apply(vars_dict, features)[0, 0]
