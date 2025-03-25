from pytc.autodiff import jastrow
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from functools import partial
from typing import Sequence

class MLP(nn.Module):
    """Multi-layer perceptron network using Flax with residual connections."""
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        # Store input for final residual connection
        input_x = x
        
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

class NeuralJastrow(jastrow.Jastrow):
    def __init__(self, nuclear_pos, nuclear_charges, layer_widths=[16, 16], 
                 epsilon=1e-8, key=random.PRNGKey(0)):
        super().__init__()
        self.nuclear_pos = jnp.array(nuclear_pos)
        self.nuclear_charges = jnp.array(nuclear_charges)
        self.epsilon = epsilon

        # Configure separate MLPs with same architecture
        n_nuclei = len(nuclear_charges)
        self.features = [*layer_widths, 1]
        
        # Initialize three networks
        key1, key2, key3 = random.split(key, 3)
        self.net_en = MLP(features=self.features)   # Input: 2*n_nuclei distances
        self.net_ee = MLP(features=self.features)   # Input: 1 distance
        self.net_een = MLP(features=self.features)  # Input: 1 + 2*n_nuclei distances
        
        self.params = self.init_params(key1, key2, key3)

    def init_params(self, key1, key2, key3):
        """Initialize network parameters using Flax."""
        # Create dummy inputs for initialization
        dummy_x_en = jnp.zeros((1, len(self.nuclear_charges) * 2))
        dummy_x_ee = jnp.zeros((1, 1))
        dummy_x_een = jnp.zeros((1, 1 + 2*len(self.nuclear_charges)))
        
        variables_en = self.net_en.init(key1, dummy_x_en)
        variables_ee = self.net_ee.init(key2, dummy_x_ee)
        variables_een = self.net_een.init(key3, dummy_x_een)
        
        # Flatten parameters for compatibility with existing interface
        flat_params_en = self._flatten_params(variables_en['params'])
        flat_params_ee = self._flatten_params(variables_ee['params'])
        flat_params_een = self._flatten_params(variables_een['params'])
        
        return jnp.concatenate([flat_params_en, flat_params_ee, flat_params_een]) * 0.01

    def _flatten_params(self, nested_params):
        """Flatten nested parameter dictionary into 1D array."""
        flat_params = []
        for layer in nested_params.values():
            flat_params.extend([layer['kernel'].ravel(), layer['bias'].ravel()])
        return jnp.concatenate(flat_params)

    def _unflatten_params(self, flat_params, input_size):
        """Reconstruct nested parameter dictionary from 1D array"""
        params = {}
        idx = 0
        prev_width = input_size
        
        for i, width in enumerate(self.features):
            kernel_size = prev_width * width
            bias_size = width
            
            kernel = flat_params[idx:idx + kernel_size].reshape((prev_width, width))
            idx += kernel_size
            bias = flat_params[idx:idx + bias_size]
            idx += bias_size
            
            params[f'Dense_{i}'] = {
                'kernel': kernel,
                'bias': bias
            }
            prev_width = width
            
        return {'params': params}
        
    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
    def _construct_features(self, r1, r2):
        """Construct separate feature sets for each network"""
        # Basic distances using broadcasting
        r12 = self._safe_norm(r1 - r2)
        r1n = self._safe_norm(r1[None, :] - self.nuclear_pos)  # (n_nuclei,)
        r2n = self._safe_norm(r2[None, :] - self.nuclear_pos)  # (n_nuclei,)
        
        # Separate features for each network
        en_features = jnp.concatenate([r1n, r2n]).reshape(1, -1)  # e-n distances
        ee_features = r12.reshape(1, -1)                          # e-e distance
        een_features = jnp.concatenate([                          # all distances
            jnp.array([r12]),
            r1n,
            r2n,
        ]).reshape(1, -1)
        
        return en_features, ee_features, een_features

    def _compute(self, r1, r2, params):
        """Combine outputs from three networks"""
        # Get features
        en_features, ee_features, een_features = self._construct_features(r1, r2)
        
        # Split parameters for each network
        n_params_en = self.get_param_count_single(2 * len(self.nuclear_charges))
        n_params_ee = self.get_param_count_single(1)
        
        params_en = params[:n_params_en]
        params_ee = params[n_params_en:n_params_en + n_params_ee]
        params_een = params[n_params_en + n_params_ee:]
        
        # Apply each network
        vars_en = self._unflatten_params(params_en, 2 * len(self.nuclear_charges))
        vars_ee = self._unflatten_params(params_ee, 1)
        vars_een = self._unflatten_params(params_een, 1 + 2 * len(self.nuclear_charges))
        
        en_out = self.net_en.apply(vars_en, en_features)[0, 0]
        ee_out = self.net_ee.apply(vars_ee, ee_features)[0, 0]
        een_out = self.net_een.apply(vars_een, een_features)[0, 0]
        
        return (en_out + ee_out + een_out)

    def get_param_count_single(self, input_size):
        """Return parameter count for a single network"""
        total = 0
        prev_width = input_size
        for width in self.features:
            total += prev_width * width + width
            prev_width = width
        return total

    def get_param_count(self):
        """Return total number of parameters needed for all networks"""
        n_nuclei = len(self.nuclear_charges)
        return (self.get_param_count_single(2 * n_nuclei) +  # en network
                self.get_param_count_single(1) +             # ee network
                self.get_param_count_single(1 + 2 * n_nuclei))  # een network
