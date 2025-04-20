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


class NeuralBase(Jastrow):
    """Base class for neural network-based Jastrow factors."""
    def __init__(self, layer_widths=[16, 16], epsilon=1e-8, **kwargs):
        super().__init__()
        self.features = [*layer_widths, 1]
        self.epsilon = epsilon
        self.net = MLP(features=self.features)
        key = kwargs.get('key', random.PRNGKey(0))
        self.params = self.init_params(key=key)

    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)

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
    def __init__(self, nuclear_pos, nuclear_charges, **kwargs):
        self.nuclear_pos = jnp.array(nuclear_pos)
        self.nuclear_charges = jnp.array(nuclear_charges)
        super().__init__(**kwargs)

    def init_params(self, **kwargs):
        key = kwargs.get('key', random.PRNGKey(0))
        dummy_x = jnp.zeros((1, len(self.nuclear_charges)))
        variables = self.net.init(key, dummy_x)
        return self._flatten_params(variables['params']) * 0.01

    def _compute(self, r1, r2, params):
        r1n = self._safe_norm(r1[None, :] - self.nuclear_pos)
        features = r1n.reshape(1, -1)
        vars_dict = self._unflatten_params(params, len(self.nuclear_charges))
        return self.net.apply(vars_dict, features)[0, 0]


class NeuralEE(NeuralBase):
    """Neural network for electron-electron correlations."""
    def init_params(self, **kwargs):
        key = kwargs.get('key', random.PRNGKey(0))
        dummy_x = jnp.zeros((1, 1))
        variables = self.net.init(key, dummy_x)
        return self._flatten_params(variables['params']) * 0.01

    def _compute(self, r1, r2, params):
        r12 = self._safe_norm(r1 - r2)
        features = r12.reshape(1, -1)
        vars_dict = self._unflatten_params(params, 1)
        return self.net.apply(vars_dict, features)[0, 0]


class NeuralEEN(NeuralBase):
    """Neural network for electron-electron-nuclear correlations."""
    def __init__(self, nuclear_pos, nuclear_charges, **kwargs):
        self.nuclear_pos = jnp.array(nuclear_pos)
        self.nuclear_charges = jnp.array(nuclear_charges)
        super().__init__(**kwargs)

    def init_params(self, **kwargs):
        key = kwargs.get('key', random.PRNGKey(0))
        input_size = 1 + 2 * len(self.nuclear_charges)
        dummy_x = jnp.zeros((1, input_size))
        variables = self.net.init(key, dummy_x)
        return self._flatten_params(variables['params']) * 0.01

    def _compute(self, r1, r2, params):
        r12 = self._safe_norm(r1 - r2)
        r1n = self._safe_norm(r1[None, :] - self.nuclear_pos)
        r2n = self._safe_norm(r2[None, :] - self.nuclear_pos)
        
        features = jnp.concatenate([
            jnp.array([r12]),
            r1n,
            r2n,
        ]).reshape(1, -1)
        
        vars_dict = self._unflatten_params(params, 1 + 2 * len(self.nuclear_charges))
        return self.net.apply(vars_dict, features)[0, 0]
