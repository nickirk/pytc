import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Sequence, List
try:
    import kfac_jax
    _HAS_KFAC = True
except ImportError:
    _HAS_KFAC = False

from pytc.autodiff.jastrow import Jastrow 
from flax import struct
import jax

class KFACDense(nn.Module):
    """Dense layer that registers with KFAC."""
    features: int
    use_bias: bool = True
    is_een_first_layer: bool = False  # Flag for EEN first layer
    num_nuclei: int = None  # Number of nuclei for EEN averaging
    
    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.lecun_normal()
        bias_init = nn.initializers.zeros

        kernel = self.param('kernel', kernel_init, (x.shape[-1], self.features))
        bias = self.param('bias', bias_init, (self.features,)) if self.use_bias else None
        
        if self.is_een_first_layer and self.num_nuclei is not None:
            # Split kernel into r12, r1n, r2n parts
            w_r12 = kernel[0:1]  # First row for r12
            w_r1n = kernel[1:self.num_nuclei+1]  # Rows for r1n
            w_r2n = kernel[self.num_nuclei+1:]  # Rows for r2n
            
            # Average the r1n and r2n weights
            w_rn_avg = (w_r1n + w_r2n) / 2
            
            # Reconstruct kernel with averaged weights
            kernel = jnp.concatenate([w_r12, w_rn_avg, w_rn_avg])
            
        y = x @ kernel
        if bias is not None:
            y += bias
            
        # Register with KFAC using raw parameters
        if _HAS_KFAC:
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

@struct.dataclass
class NeuralBase(Jastrow):
    """Base class for neural network-based Jastrow factors."""
    nuclear_pos: jax.Array
    nuclear_charges: jax.Array
    net: nn.Module = struct.field(pytree_node=False)
    features: Sequence[int] = struct.field(pytree_node=False)
    nelectron: int = struct.field(pytree_node=False)
    epsilon: float = struct.field(pytree_node=False, default=1e-8)
    name: str = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(cls, mol, layer_widths=[16, 16], epsilon=1e-8, name=None, **kwargs):
        nuclear_pos = jnp.array(mol.atom_coords())
        nuclear_charges = jnp.array(mol.atom_charges())
        features = list(layer_widths) + [1]
        net = MLP(features=features)
        
        return cls(
            name=name,
            nuclear_pos=nuclear_pos,
            nuclear_charges=nuclear_charges,
            net=net,
            features=features,
            nelectron=mol.nelectron,
            epsilon=epsilon
        )

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

@struct.dataclass
class NeuralEN(NeuralBase):
    """Neural network for electron-nuclear correlations."""
    
    @classmethod
    def create(cls, mol, **kwargs):
        # Reuse base create but ensure correct class
        base = NeuralBase.create(mol, **kwargs)
        return cls(
            name=base.name,
            nuclear_pos=base.nuclear_pos,
            nuclear_charges=base.nuclear_charges,
            net=base.net,
            features=base.features,
            nelectron=base.nelectron,
            epsilon=base.epsilon
        )

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
        return self.net.apply(net_vars, features)[0, 0]/(self.nelectron - 1)
    
    def grad_r(self, r1, r2, params):
        return super().grad_r(r1, r2, params) * (self.nelectron - 1)/self.nelectron/2.

    def get_log_grads_r2(self, r1, r2, params):
        return self.get_log_grads_r1(r2, r1, params)

@struct.dataclass
class NeuralEE(NeuralBase):
    """Neural network for electron-electron correlations."""
    
    @classmethod
    def create(cls, mol, **kwargs):
        base = NeuralBase.create(mol, **kwargs)
        return cls(
            name=base.name,
            nuclear_pos=base.nuclear_pos,
            nuclear_charges=base.nuclear_charges,
            net=base.net,
            features=base.features,
            nelectron=base.nelectron,
            epsilon=base.epsilon
        )

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


class EENMLP(nn.Module):
    """MLP specifically for EEN with equivariant first layer."""
    features: Sequence[int]
    num_nuclei: int
    
    @nn.compact
    def __call__(self, x):
        # First layer is equivariant
        x = KFACDense(
            self.features[0], 
            is_een_first_layer=True, 
            num_nuclei=self.num_nuclei
        )(x)
        x = nn.tanh(x)
        
        # Remaining layers are standard
        for feat in self.features[1:-1]:
            layer_input = x
            x = KFACDense(feat)(x)
            x = nn.tanh(x)
            if layer_input.shape[-1] == feat:
                x = x + layer_input
        
        x = KFACDense(self.features[-1])(x)
        return x

@struct.dataclass
class NeuralEEN(NeuralBase):
    """Neural network for electron-electron-nuclear correlations."""
    num_nuclei: int = struct.field(pytree_node=False, default=0)
    
    @classmethod
    def create(cls, mol, layer_widths=[16, 16], epsilon=1e-8, name=None, **kwargs):
        nuclear_pos = jnp.array(mol.atom_coords())
        nuclear_charges = jnp.array(mol.atom_charges())
        num_nuclei = len(nuclear_charges)
        features = list(layer_widths) + [1]
        net = EENMLP(features=features, num_nuclei=num_nuclei)
        
        return cls(
            name=name,
            nuclear_pos=nuclear_pos,
            nuclear_charges=nuclear_charges,
            net=net,
            features=features,
            nelectron=mol.nelectron,
            epsilon=epsilon,
            num_nuclei=num_nuclei
        )
    
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
        net_vars = params['net_vars']

        # Calculate distances
        r12_dist = self._safe_norm(r1 - r2)[None]  # Add singleton dimension
        r1n_dist = self._safe_norm(r1[None, :] - self.nuclear_pos)  # Shape: (N,)
        r2n_dist = self._safe_norm(r2[None, :] - self.nuclear_pos)  # Shape: (N,)

        
        # Concatenate features with consistent dimensions
        features = jnp.concatenate([
            r12_dist,  # Shape: (1,)
            r1n_dist,  # Shape: (N,)
            r2n_dist,  # Shape: (N,)
        ], axis=0).reshape(1, -1)
        
        return self.net.apply(net_vars, features)[0, 0]
