import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Sequence, List, Tuple
from dataclasses import dataclass
import jax
from jax import lax

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
                BHTerm(2, 0, 2, 0.1),
                BHTerm(2, 2, 2, 0.1),
                BHTerm(4, 0, 2, 0.1),
                BHTerm(2, 0, 4, 0.1),
                BHTerm(4, 2, 2, 0.1),
                BHTerm(6, 0, 2, 0.1),
                BHTerm(4, 0, 4, 0.1),
                BHTerm(2, 2, 4, 0.1),
                BHTerm(2, 0, 6, 0.1),
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

    def _compute(self, r1, r2, params):
        """Compute Boys-Handy Jastrow exponent."""

        # Function to execute if r1 and r2 are not close (original computation)
        # Get positive b and d values using softplus
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c = params['c_raw']  # Allow c to be both positive and negative

        u_total = 0.0

        # Loop over nuclei
        for I in range(self.natom):
            # Compute scaled distances
            r1I = self._scaled_r_en(r1, self.nuclear_pos[I], b[I])
            r2I = self._scaled_r_en(r2, self.nuclear_pos[I], b[I])
            r12 = self._scaled_r_ee(r1, r2, d[I])

            # Sum over terms for this nucleus
            for k, term in enumerate(self.terms_per_nucleus[I]):
                if term.m == 0 and term.n == 0 and term.o == 1:
                    # Cusp term
                    factor = self._delta(term.m, term.n) * 0.5
                    u_term = r12**term.o
                else:
                    factor = self._delta(term.m, term.n) * c[I, k]
                    # Symmetric combination of r1I and r2I terms
                    u_term = (r1I**term.m * r2I**term.n +
                             r2I**term.m * r1I**term.n) * r12**term.o
                u_total += factor * u_term

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
