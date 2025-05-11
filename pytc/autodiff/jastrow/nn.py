import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Sequence
import kfac_jax
from pytc.autodiff.jastrow import Jastrow 

class KFACDense(nn.Module):
    """Dense layer that registers with KFAC."""
    features: int
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.lecun_normal()
        bias_init = nn.initializers.zeros

        kernel = self.param('kernel', kernel_init, (x.shape[-1], self.features))
        bias = self.param('bias', bias_init, (self.features,)) if self.use_bias else None
        
        y = x @ kernel
        if bias is not None:
            y += bias
            
        # Register with KFAC using raw parameters
        kfac_jax.register_dense(x, y, kernel, bias)
        return y

class MLP(nn.Module):
    """Multi-layer perceptron network using Flax with residual connections."""
    features: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features[:-1]):
            layer_input = x
            # Use KFAC-aware Dense layer
            x = KFACDense(feat)(x)
            x = nn.tanh(x)
            if layer_input.shape[-1] == feat:
                x = x + layer_input
        
        # Final layer using KFAC-aware Dense
        x = KFACDense(self.features[-1])(x)
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

    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
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
        # Use standard Flax variable structure without flattening
        variables = self.net.init(key, dummy_x)
        # Initialize raw parameter for rc_en such that softplus(raw) ~ 0.1
        initial_rc_en_raw = 0.5
        return {
            'rc_en_raw': initial_rc_en_raw, 
            'net_vars': variables  # Store the entire variables dictionary
        }

    def _compute(self, r1, r2, params):
        # Extract raw decay parameter and network variables
        rc_en_raw = params['rc_en_raw']
        net_vars = params['net_vars']
        
        # Ensure rc_en is positive using softplus
        rc_en = nn.softplus(rc_en_raw)

        r1n_dist = self._safe_norm(r1[None, :] - self.nuclear_pos)
        # Apply decay parameter
        r1n_feat = r1n_dist

        features = r1n_feat.reshape(1, -1)
        # Use the standard Flax variable structure directly
        return self.net.apply(net_vars, features)[0, 0] / (self.nelectron - 1)
    
    def get_log_grads_r2(self, r1, r2, params):
        return self.get_log_grads_r1(r2, r1, params)


class NeuralEE(NeuralBase):
    """Neural network for electron-electron correlations."""
    def init_params(self, **kwargs):
        key = kwargs.get('key', random.PRNGKey(0))
        dummy_x = jnp.zeros((1, 1))
        # Use standard Flax variable structure
        variables = self.net.init(key, dummy_x)
        # Initialize raw parameter for rc_ee such that softplus(raw) ~ 0.1
        initial_rc_ee_raw = 0.5
        return {
            'rc_ee_raw': initial_rc_ee_raw, 
            'net_vars': variables  # Store the entire variables dictionary
        }

    def _compute(self, r1, r2, params):
        # Extract raw decay parameter and network variables
        rc_ee_raw = params['rc_ee_raw']
        net_vars = params['net_vars']

        # Ensure rc_ee is positive using softplus
        rc_ee = nn.softplus(rc_ee_raw)

        r12_dist = self._safe_norm(r1 - r2)
        # Apply decay parameter
        r12_feat = r12_dist

        features = r12_feat.reshape(1, -1)
        # Use the standard Flax variable structure directly
        return self.net.apply(net_vars, features)[0, 0]


class NeuralEEN(NeuralBase):
    """Neural network for electron-electron-nuclear correlations."""
    def __init__(self, mol, **kwargs):
        super().__init__(mol, **kwargs)

    def init_params(self, **kwargs):
        key = kwargs.get('key', random.PRNGKey(0))
        input_size = 1+2*len(self.nuclear_charges)
        dummy_x = jnp.zeros((1, input_size))
        # Use standard Flax variable structure
        variables = self.net.init(key, dummy_x)
        # Initialize raw decay parameters
        initial_rc_raw = 0.5 # approx -2.25
        return {
            'rc_ee_raw': initial_rc_raw,
            'rc_en_raw': initial_rc_raw,
            'net_vars': variables  # Store the entire variables dictionary
        }

    def _compute(self, r1, r2, params):
        # Extract raw decay parameters and network variables
        rc_ee_raw = params['rc_ee_raw']
        rc_en_raw = params['rc_en_raw']
        net_vars = params['net_vars']

        # Ensure decay parameters are positive using softplus
        rc_ee = nn.softplus(rc_ee_raw)
        rc_en = nn.softplus(rc_en_raw)

        # Calculate distances
        r12_dist = self._safe_norm(r1 - r2)
        r1n_dist = self._safe_norm(r1[None, :] - self.nuclear_pos)
        r2n_dist = self._safe_norm(r2[None, :] - self.nuclear_pos)

        # Apply decay parameters
        r12_feat = r12_dist
        r1n_feat = r1n_dist
        r2n_feat = r2n_dist
        features = jnp.concatenate([
            jnp.asarray([r12_feat]), # Ensure it's an array
            r1n_feat,
            r2n_feat,
        ], axis=-1).reshape(1, -1)
        
        # Use the standard Flax variable structure directly
        return self.net.apply(net_vars, features)[0, 0]
