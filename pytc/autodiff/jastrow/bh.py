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
            # Create a list of default terms for each nucleus
            self.terms_per_nucleus = [default_terms_for_one_nucleus for _ in range(self.natom)]
        else:
            self.terms_per_nucleus = terms_per_nucleus
            
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
        
        # Initialize b and d parameters for each nucleus
        b_raw = jnp.ones(self.natom) * 0.5  # starting value ~1.0 after softplus
        d_raw = jnp.ones(self.natom) * 0.5
        
        # Initialize c parameters for each term in each nucleus
        # Reshape to ensure (n_nuclei, n_terms) shape
        c_raw = []
        for nucleus_terms in self.terms_per_nucleus:
            c_nucleus = jnp.array([term.c for term in nucleus_terms])
            c_raw.append(c_nucleus)
        c_raw = jnp.array(c_raw)  # This will have shape (n_nuclei, n_terms)
        
        return {
            'b_raw': b_raw,
            'd_raw': d_raw,
            'c_raw': c_raw
        }

    @partial(jax.jit, static_argnums=(0,))
    def _compute(self, r1, r2, params):
        """Compute Boys-Handy Jastrow exponent."""
        
        # Get positive b and d values using softplus
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c = params['c_raw']  # Allow c to be both positive and negative

        def nucleus_scan_fn(carry, nucleus_data):
            """Scan function for looping over nuclei."""
            u_total = carry
            I, nuclear_pos_I, b_I, d_I, c_I = nucleus_data
            
            # Compute scaled distances for this nucleus
            r1I = self._scaled_r_en(r1, nuclear_pos_I, b_I)
            r2I = self._scaled_r_en(r2, nuclear_pos_I, b_I)
            r12 = self._scaled_r_ee(r1, r2, d_I)
            
            def term_scan_fn(carry_inner, term_data):
                """Scan function for looping over terms within a nucleus."""
                u_nucleus = carry_inner
                k, term_m, term_n, term_o = term_data
                
                # Check if this is a cusp term
                is_cusp = (term_m == 0) & (term_n == 0) & (term_o == 1)
                
                # Compute factor
                delta_factor = self._delta(term_m, term_n)
                factor = jnp.where(is_cusp, 
                                 delta_factor * 0.5,
                                 delta_factor * c_I[k])
                
                # Compute u_term
                u_term = jnp.where(is_cusp,
                                 r12**term_o,
                                 (r1I**term_m * r2I**term_n + 
                                  r2I**term_m * r1I**term_n) * r12**term_o)
                
                u_nucleus += factor * u_term
                return u_nucleus, None
            
            # Prepare term data for this nucleus
            nucleus_terms = self.terms_per_nucleus[I]
            term_indices = jnp.arange(len(nucleus_terms))
            term_m = jnp.array([term.m for term in nucleus_terms])
            term_n = jnp.array([term.n for term in nucleus_terms])
            term_o = jnp.array([term.o for term in nucleus_terms])
            term_data = (term_indices, term_m, term_n, term_o)
            
            # Scan over terms for this nucleus
            u_nucleus, _ = jax.lax.scan(term_scan_fn, 0.0, term_data)
            u_total += u_nucleus
            
            return u_total, None

        # Prepare nucleus data
        nucleus_indices = jnp.arange(self.natom)
        nucleus_data = (nucleus_indices, self.nuclear_pos, b, d, c)
        
        # Scan over nuclei
        u_total, _ = jax.lax.scan(nucleus_scan_fn, 0.0, nucleus_data)
        
        return u_total


    def get_param_count(self):
        """Return total number of optimizable parameters."""
        # Count b and d parameters (one per nucleus)
        count = 2 * self.natom
        # Add c parameters (one per term per nucleus)
        for nucleus_terms in self.terms_per_nucleus:
            count += len(nucleus_terms)
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
        b_size = self.natom
        b_raw = flat_params[idx:idx+b_size]
        idx += b_size
        
        # Extract d parameters
        d_size = self.natom
        d_raw = flat_params[idx:idx+d_size]
        idx += d_size
        
        # Extract c parameters
        c_raw = []
        for nucleus_terms in self.terms_per_nucleus:
            c_size = len(nucleus_terms)
            c_nucleus = flat_params[idx:idx+c_size]
            c_raw.append(c_nucleus)
            idx += c_size
        c_raw = jnp.array(c_raw)
        
        return {
            'b_raw': b_raw,
            'd_raw': d_raw,
            'c_raw': c_raw
        }
