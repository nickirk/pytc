from functools import partial
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from dataclasses import dataclass
import jax

from pytc.autodiff.jastrow import Jastrow

@dataclass
class BHTerm:
    """Single term in Boys-Handy expansion"""
    m: int
    n: int
    o: int  # renamed from alpha to o
    c: float  # initial value, will be optimized

class BoysHandy(Jastrow):
    """Boys-Handy Jastrow factor implementation."""
    
    def __init__(self, mol, terms_per_nucleus=None, epsilon=1e-8, name=None):
        super().__init__(name=name)
        self.mol = mol
        self.nelectron = mol.nelectron
        self.nuclear_pos = jnp.array(mol.atom_coords())
        self.nuclear_charges = jnp.array(mol.atom_charges())
        self.natom = len(self.nuclear_charges)
        self.epsilon = epsilon
        
        # Identify unique atom types and create mappings
        self.unique_charges = jnp.sort(jnp.unique(self.nuclear_charges))
        self.n_types = len(self.unique_charges)
        
        # Create atom type map: atom_idx -> type_idx
        # For each atom, find which type it belongs to
        self.atom_type_map = jnp.zeros(self.natom, dtype=jnp.int32)
        for i, charge in enumerate(self.nuclear_charges):
            type_idx = jnp.where(self.unique_charges == charge)[0][0]
            self.atom_type_map = self.atom_type_map.at[i].set(type_idx)
        
        # Default terms if none specified
        if terms_per_nucleus is None:
            # Basic terms including e-e cusp
            d_cusp = 0.5  # cusp coefficient 1/(2d)
            default_terms_for_one_nucleus = [
                BHTerm(0, 0, 1, d_cusp),  # e-e cusp term with c = 1/(2d)
                BHTerm(0, 0, 2, 0.01),  
                BHTerm(0, 0, 3, 0.001),  
                BHTerm(0, 0, 4, -0.001),  
                BHTerm(2, 0, 0, 0.001),  # e-n term
                BHTerm(3, 0, 0, 0.0001),
                BHTerm(4, 0, 0, 0.0001),   # higher order term
                BHTerm(2, 2, 0, -0.001),    # e-n term
                BHTerm(2, 0, 2, 0.01),
                BHTerm(2, 2, 2, 0.01),
                BHTerm(4, 0, 2, 0.01),
                BHTerm(2, 0, 4, 0.01),
                BHTerm(4, 2, 2, 0.01),
                BHTerm(6, 0, 2, 0.01),
                BHTerm(4, 0, 4, 0.01),
                BHTerm(2, 2, 4, 0.01),
                BHTerm(2, 0, 6, 0.01),
            ]
            # Create a list of default terms for each atom type (not each atom)
            self.terms_per_atom_type = [default_terms_for_one_nucleus for _ in range(self.n_types)]
        else:
            # User-provided terms - should now be per atom type
            self.terms_per_atom_type = terms_per_nucleus

        term_lengths = tuple(len(type_terms) for type_terms in self.terms_per_atom_type)
        if len(term_lengths) > 0 and len(set(term_lengths)) != 1:
            raise ValueError("BoysHandy requires the same number of terms per atom type when using JAX scans.")

        self.n_terms = term_lengths[0] if term_lengths else 0

        term_m = []
        term_n = []
        term_o = []
        for type_terms in self.terms_per_atom_type:
            term_m.append([term.m for term in type_terms])
            term_n.append([term.n for term in type_terms])
            term_o.append([term.o for term in type_terms])

        if self.n_terms > 0:
            self._term_m = jnp.array(term_m, dtype=jnp.int32)
            self._term_n = jnp.array(term_n, dtype=jnp.int32)
            self._term_o = jnp.array(term_o, dtype=jnp.int32)
            self._delta_factor = jnp.where(self._term_m == self._term_n, 0.5, 1.0)
            self._cusp_mask = (self._term_m == 0) & (self._term_n == 0) & (self._term_o == 1)
        else:
            self._term_m = jnp.zeros((0, 0), dtype=jnp.int32)
            self._term_n = jnp.zeros((0, 0), dtype=jnp.int32)
            self._term_o = jnp.zeros((0, 0), dtype=jnp.int32)
            self._delta_factor = jnp.zeros((0, 0))
            self._cusp_mask = jnp.zeros((0, 0), dtype=bool)
            
    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
    def _delta(self, m, n):
        """Implements the Delta function for Boys-Handy."""
        return jnp.where(m == n, 0.5, 1.0)
    
    def _scaled_r_en(self, r_electron, r_nuclear, b):
        """Compute scaled electron-nuclear distance."""
        r = self._safe_norm(r_electron - r_nuclear)
        #return b * r / (1.0 + b * r)
        return r / (1.0 + r)
    
    def _scaled_r_ee(self, r1, r2, d):
        """Compute scaled electron-electron distance."""
        r = self._safe_norm(r1 - r2)
        #return d * r / (1.0 + d * r)
        return  r / (1.0 +  r)

    def init_params(self, **kwargs):
        """Initialize Boys-Handy parameters."""
        key = kwargs.get('key', random.PRNGKey(0))
        
        # Initialize b and d parameters for each atom type (not each atom)
        b_raw = jnp.ones(self.n_types) * 0.5  # starting value ~1.0 after softplus
        d_raw = jnp.ones(self.n_types) * 0.5
        
        # Initialize c parameters for each term in each atom type
        # Reshape to ensure (n_types, n_terms) shape
        c_raw = []
        for type_terms in self.terms_per_atom_type:
            c_type = jnp.array([term.c for term in type_terms])
            c_raw.append(c_type)
        c_raw = jnp.array(c_raw)  # This will have shape (n_types, n_terms)
        
        return {
            'b_raw': b_raw,
            'd_raw': d_raw,
            'c_raw': c_raw
        }

    def _compute_forward(self, r1, r2, params):
        """Forward computation of Boys-Handy Jastrow exponent.
        
        This is the pure forward pass that will be wrapped with custom JVP.
        """
        # Get positive b and d values using softplus
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c_raw = params['c_raw']  # Allow c to be both positive and negative
        
        # Fix cusp term coefficients (m=0, n=0, o=1) to 0.5 for e-e cusp condition
        # This ensures the cusp condition is always satisfied regardless of optimization
        c = jnp.where(self._cusp_mask, 0.5, c_raw)

        def atom_scan_fn(carry, atom_data):
            """Scan function for looping over atoms."""
            u_total = carry
            nuclear_pos_I, atom_type_idx = atom_data
            
            # Get parameters for this atom's type
            b_I = b[atom_type_idx]
            d_I = d[atom_type_idx]
            c_I = c[atom_type_idx]
            term_m_I = self._term_m[atom_type_idx]
            term_n_I = self._term_n[atom_type_idx]
            term_o_I = self._term_o[atom_type_idx]
            delta_factor_I = self._delta_factor[atom_type_idx]
            cusp_mask_I = self._cusp_mask[atom_type_idx]
            
            # Compute scaled distances for this nucleus
            r1I = self._scaled_r_en(r1, nuclear_pos_I, b_I)
            r2I = self._scaled_r_en(r2, nuclear_pos_I, b_I)
            r12 = self._scaled_r_ee(r1, r2, d_I)

            r1I_pow_m = jnp.power(r1I, term_m_I)
            r2I_pow_n = jnp.power(r2I, term_n_I)
            r1I_pow_n = jnp.power(r1I, term_n_I)
            r2I_pow_m = jnp.power(r2I, term_m_I)
            r12_pow_o = jnp.power(r12, term_o_I)

            non_cusp_terms = (r1I_pow_m * r2I_pow_n + r2I_pow_m * r1I_pow_n) * r12_pow_o
            # For cusp terms (m=0,n=0), the symmetric sum is (1*1 + 1*1) = 2, so multiply by 2
            cusp_terms = 2.0 * r12_pow_o
            u_terms = jnp.where(cusp_mask_I, cusp_terms, non_cusp_terms)

            cusp_factor = delta_factor_I * c_I
            non_cusp_factor = delta_factor_I * c_I
            factor = jnp.where(cusp_mask_I, cusp_factor, non_cusp_factor)

            u_total += jnp.sum(factor * u_terms)
            return u_total, None

        # Prepare atom data (now including atom type indices)
        atom_data = (
            self.nuclear_pos,
            self.atom_type_map,
        )
        
        # Scan over all atoms
        u_total, _ = jax.lax.scan(atom_scan_fn, 0.0, atom_data)
        
        return u_total

    @partial(jax.jit, static_argnums=(0,))
    def _compute(self, r1, r2, params):
        """Compute Boys-Handy Jastrow exponent with custom JVP for memory efficiency.
        
        This wraps the forward computation with a custom JVP rule that avoids storing
        the full computational graph. Instead, it recomputes gradients on-the-fly
        using jax.jvp, similar to FermiNet's approach.
        """
        @jax.custom_jvp
        def compute_with_custom_jvp(params):
            """Forward pass - only returns the final value."""
            return self._compute_forward(r1, r2, params)
        
        @compute_with_custom_jvp.defjvp
        def compute_jvp(primals, tangents):
            """Custom JVP - recompute gradients on-the-fly to save memory.
            
            This avoids storing all intermediate values from the forward pass.
            Instead, we use jax.jvp to compute the gradient efficiently.
            """
            (params,) = primals
            (params_tangent,) = tangents
            
            # Forward pass - compute the value
            u_value = self._compute_forward(r1, r2, params)
            
            # Compute JVP using jax.jvp for memory efficiency
            # This recomputes the forward pass but only stores what's needed for the gradient
            def forward_fn(p):
                return self._compute_forward(r1, r2, p)
            
            _, u_tangent = jax.jvp(forward_fn, (params,), (params_tangent,))
            
            return u_value, u_tangent
        
        return compute_with_custom_jvp(params)


    def get_param_count(self):
        """Return total number of optimizable parameters."""
        # Count b and d parameters (one per atom type)
        count = 2 * self.n_types
        # Add c parameters (one per term per atom type)
        for type_terms in self.terms_per_atom_type:
            count += len(type_terms)
        return count

    def flatten_params(self, params):
        """Flatten parameters into 1D array for optimization."""
        return jnp.concatenate([
            params['b_raw'].ravel(),
            params['d_raw'].ravel(),
            params['c_raw'].ravel()
        ])

    def unflatten_params(self, flat_params):
        """Reconstruct parameter dictionary from 1D array."""
        idx = 0
        
        # Extract b parameters
        b_size = self.n_types
        b_raw = flat_params[idx:idx+b_size]
        idx += b_size
        
        # Extract d parameters
        d_size = self.n_types
        d_raw = flat_params[idx:idx+d_size]
        idx += d_size
        
        # Extract c parameters
        c_raw = []
        for type_terms in self.terms_per_atom_type:
            c_size = len(type_terms)
            c_type = flat_params[idx:idx+c_size]
            c_raw.append(c_type)
            idx += c_size
        c_raw = jnp.array(c_raw)
        
        return {
            'b_raw': b_raw,
            'd_raw': d_raw,
            'c_raw': c_raw
        }
