from pytc.autodiff import jastrow
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from functools import partial
from typing import Sequence

class MLP(nn.Module):
    """Multi-layer perceptron network using Flax."""
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        return x

class NeuralJastrow(jastrow.Jastrow):
    def __init__(self, nuclear_pos, nuclear_charges, layer_widths=[16, 16], 
                 epsilon=1e-8, key=random.PRNGKey(0)):
        """
        Args:
            nuclear_pos: Array of shape (n_nuclei, 3) for nuclear positions
            nuclear_charges: Array of shape (n_nuclei,) for nuclear charges
            layer_widths: List of integers specifying width of each hidden layer
            epsilon: Small number to prevent division by zero
            key: JAX random key for parameter initialization
        """
        super().__init__()
        self.nuclear_pos = jnp.array(nuclear_pos)
        self.nuclear_charges = jnp.array(nuclear_charges)
        self.epsilon = epsilon

        # Configure MLP architecture
        n_nuclei = len(nuclear_charges)
        input_size = 1 + 2 * n_nuclei  # r12 + r1N + r2N distances
        self.features = [*layer_widths, 1]  # Add output layer with 1 unit

        # Initialize neural network
        self.net = MLP(features=self.features)
        self.params = self.init_params(key)

    def init_params(self, key=random.PRNGKey(0)):
        """Initialize network parameters using Flax."""
        # Create dummy input for initialization
        dummy_x = jnp.zeros((1, 1 + 2*len(self.nuclear_charges)))
        variables = self.net.init(key, dummy_x)
        # Flatten parameters for compatibility with existing interface
        flat_params = self._flatten_params(variables['params'])
        return flat_params

    def _flatten_params(self, nested_params):
        """Flatten nested parameter dictionary into 1D array."""
        flat_params = []
        for layer in nested_params.values():
            flat_params.extend([layer['kernel'].ravel(), layer['bias'].ravel()])
        return jnp.concatenate(flat_params)

    def _unflatten_params(self, flat_params):
        """Reconstruct nested parameter dictionary from 1D array."""
        params = {}
        idx = 0
        for i, feat_out in enumerate(self.features):
            feat_in = (1 + 2*len(self.nuclear_charges)) if i == 0 else self.features[i-1]
            kernel_size = feat_in * feat_out
            bias_size = feat_out
            
            kernel = flat_params[idx:idx + kernel_size].reshape((feat_in, feat_out))
            idx += kernel_size
            bias = flat_params[idx:idx + bias_size]
            idx += bias_size
            
            params[f'Dense_{i}'] = {
                'kernel': kernel,
                'bias': bias
            }
        return {'params': params}
        
    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
    def _construct_features(self, r1, r2):
        """Construct input features and cusp terms."""
        # Basic distances using broadcasting
        r12 = self._safe_norm(r1 - r2)
        r12 = r12/(1.0 + r12)  # Normalize to avoid large values
        r1n = self._safe_norm(r1[None, :] - self.nuclear_pos)  # (n_nuclei,)
        r2n = self._safe_norm(r2[None, :] - self.nuclear_pos)  # (n_nuclei,)
        r1n = r1n/(1.0 + r1n) * self.nuclear_charges  # Normalize to avoid large values
        r2n = r2n/(1.0 + r2n) * self.nuclear_charges  # Normalize to avoid large values
        # Cusp terms
        # e-e cusp: u ~ 0.5*r12 at small r12
        cusp_r12 = 0.5 * r12 / (1.0 + r12)
        
        # e-n cusp: u ~ -Z*r at small r
        # The 1/(1+r) factor ensures smooth decay at large r
        cusp_r1n = r1n * 0.001 
        cusp_r2n = r2n * 0.001
        
        # Network input features (raw distances)
        net_features = jnp.concatenate([
            jnp.array([r12]),
            r1n,
            r2n,
        ]).reshape(1, -1)
        
        # Cusp features for direct combination
        cusp_features = (cusp_r12, cusp_r1n, cusp_r2n)
        
        return net_features, cusp_features

    def _compute(self, r1, r2, params):
        """Compute Jastrow exponent using neural network and cusp terms."""
        # Get both feature types
        net_features, (cusp_r12, cusp_r1n, cusp_r2n) = self._construct_features(r1, r2)
        
        # Neural network part
        variables = self._unflatten_params(params)
        net_output = self.net.apply(variables, net_features)[0, 0]
        
        # Combine with cusp terms linearly
        cusp_sum = (cusp_r12 +           # electron-electron cusp
                   (jnp.sum(cusp_r1n) +   # electron-nuclear cusps for e1
                   jnp.sum(cusp_r2n))*0.)    # electron-nuclear cusps for e2
        
        return net_output + cusp_sum

    def get_param_count(self):
        """Return total number of parameters needed."""
        total = 0
        input_size = 1 + 2*len(self.nuclear_charges)
        prev_width = input_size
        for width in self.features:
            total += prev_width * width + width  # weights + biases
            prev_width = width
        return total
