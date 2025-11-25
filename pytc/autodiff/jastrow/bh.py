from functools import partial
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from dataclasses import dataclass
import jax
from flax import struct
from typing import List, Any

from pytc.autodiff.jastrow import Jastrow

@dataclass
class BHTerm:
    """Single term in Boys-Handy expansion"""
    m: int
    n: int
    o: int  # renamed from alpha to o
    c: float  # initial value, will be optimized

@struct.dataclass
class BoysHandy(Jastrow):
    """Boys-Handy Jastrow factor implementation."""
    nuclear_pos: jax.Array
    nuclear_charges: jax.Array
    atom_type_map: jax.Array
    unique_charges: jax.Array
    _term_m: jax.Array
    _term_n: jax.Array
    _term_o: jax.Array
    _delta_factor: jax.Array
    _cusp_mask: jax.Array
    
    nelectron: int = struct.field(pytree_node=False)
    natom: int = struct.field(pytree_node=False)
    n_types: int = struct.field(pytree_node=False)
    n_terms: int = struct.field(pytree_node=False)
    epsilon: float = struct.field(pytree_node=False, default=1e-8)
    terms_per_atom_type: List[List[BHTerm]] = struct.field(pytree_node=False, default=None)
    nuclei_by_type: List[jax.Array] = struct.field(default=None)
    name: str = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(cls, mol, terms_per_nucleus=None, epsilon=1e-8, name=None):
        nelectron = mol.nelectron
        nuclear_pos = jnp.array(mol.atom_coords())
        nuclear_charges = jnp.array(mol.atom_charges())
        natom = len(nuclear_charges)
        
        # Identify unique atom types and create mappings
        unique_charges = jnp.sort(jnp.unique(nuclear_charges))
        n_types = len(unique_charges)
        
        # Create atom type map: atom_idx -> type_idx
        atom_type_map = jnp.zeros(natom, dtype=jnp.int32)
        for i, charge in enumerate(nuclear_charges):
            type_idx = jnp.where(unique_charges == charge)[0][0]
            atom_type_map = atom_type_map.at[i].set(type_idx)
        
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
            terms_per_atom_type = [default_terms_for_one_nucleus for _ in range(n_types)]
        else:
            terms_per_atom_type = terms_per_nucleus

        term_lengths = tuple(len(type_terms) for type_terms in terms_per_atom_type)
        if len(term_lengths) > 0 and len(set(term_lengths)) != 1:
            raise ValueError("BoysHandy requires the same number of terms per atom type when using JAX scans.")

        n_terms = term_lengths[0] if term_lengths else 0

        term_m = []
        term_n = []
        term_o = []
        for type_terms in terms_per_atom_type:
            term_m.append([term.m for term in type_terms])
            term_n.append([term.n for term in type_terms])
            term_o.append([term.o for term in type_terms])

        if n_terms > 0:
            _term_m = jnp.array(term_m, dtype=jnp.int32)
            _term_n = jnp.array(term_n, dtype=jnp.int32)
            _term_o = jnp.array(term_o, dtype=jnp.int32)
            _delta_factor = jnp.where(_term_m == _term_n, 0.5, 1.0)
            _cusp_mask = (_term_m == 0) & (_term_n == 0) & (_term_o == 1)
        else:
            _term_m = jnp.zeros((0, 0), dtype=jnp.int32)
            _term_n = jnp.zeros((0, 0), dtype=jnp.int32)
            _term_o = jnp.zeros((0, 0), dtype=jnp.int32)
            _delta_factor = jnp.zeros((0, 0))
            _cusp_mask = jnp.zeros((0, 0), dtype=bool)
            
        # Group nuclei by type for efficient computation
        nuclei_by_type = []
        for i in range(n_types):
            type_mask = (atom_type_map == i)
            # We can't use boolean indexing on JAX arrays during creation if we want static shapes?
            # Actually, create is called eagerly (not JITted usually), so we can use numpy or boolean indexing.
            # atom_type_map is a JAX array, but we can convert to numpy if needed or just use JAX.
            # Since create returns the class instance which is a Pytree, the arrays in nuclei_by_type
            # will be leaves. They must have fixed shapes.
            # Using boolean indexing on JAX array returns a concrete array if not traced.
            # If traced, it might be dynamic. Ideally create is called outside JIT.
            
            # Use numpy for grouping to ensure static shapes
            mask_np = jnp.array(atom_type_map) == i
            nuclei_group = nuclear_pos[jnp.array(mask_np)]
            nuclei_by_type.append(nuclei_group)

        return cls(
            nuclear_pos=nuclear_pos,
            nuclear_charges=nuclear_charges,
            atom_type_map=atom_type_map,
            unique_charges=unique_charges,
            _term_m=_term_m,
            _term_n=_term_n,
            _term_o=_term_o,
            _delta_factor=_delta_factor,
            _cusp_mask=_cusp_mask,
            nelectron=nelectron,
            natom=natom,
            n_types=n_types,
            n_terms=n_terms,
            epsilon=epsilon,
            terms_per_atom_type=terms_per_atom_type,
            nuclei_by_type=nuclei_by_type,
            name=name
        )
            
    def _safe_norm(self, x):
        """Compute norm with a small epsilon to prevent division by zero."""
        return jnp.sqrt(jnp.sum(x*x, axis=-1) + self.epsilon)
    
    def _delta(self, m, n):
        """Implements the Delta function for Boys-Handy."""
        return jnp.where(m == n, 0.5, 1.0)
    
    def _scaled_r_en(self, r_electron, r_nuclear, b):
        """Compute scaled electron-nuclear distance."""
        r = self._safe_norm(r_electron - r_nuclear)
        return r / (1.0 + r)
    
    def _scaled_r_ee(self, r1, r2, d):
        """Compute scaled electron-electron distance."""
        r = self._safe_norm(r1 - r2)
        return  r / (1.0 +  r)

    def init_params(self, **kwargs):
        """Initialize Boys-Handy parameters."""
        # Initialize b and d parameters for each atom type
        b_raw = jnp.ones(self.n_types) * 0.5  
        d_raw = jnp.ones(self.n_types) * 0.5
        
        # Initialize c parameters
        c_raw = []
        for type_terms in self.terms_per_atom_type:
            c_type = jnp.array([term.c for term in type_terms])
            c_raw.append(c_type)
        c_raw = jnp.array(c_raw)
        
        return {
            'b_raw': b_raw,
            'd_raw': d_raw,
            'c_raw': c_raw
        }

    def _compute_forward(self, r1, r2, params):
        """Forward computation of Boys-Handy Jastrow exponent."""
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c_raw = params['c_raw']
        
        c = jnp.where(self._cusp_mask, 0.5, c_raw)

        u_total = 0.0
        
        # Iterate over atom types (static loop)
        for i in range(self.n_types):
            # Get parameters for this type
            b_I = b[i]
            d_I = d[i]
            c_I = c[i] # Shape (n_terms,)
            
            term_m = self._term_m[i] # Shape (n_terms,)
            term_n = self._term_n[i]
            term_o = self._term_o[i]
            delta = self._delta_factor[i]
            mask = self._cusp_mask[i]
            
            # Get nuclei of this type
            nuclei_group = self.nuclei_by_type[i]
            
            # Skip if no nuclei of this type
            if nuclei_group.shape[0] == 0:
                continue
                
            # Vectorized computation over nuclei of this type
            def compute_for_nucleus(nuc_pos):
                r1I = self._scaled_r_en(r1, nuc_pos, b_I)
                r2I = self._scaled_r_en(r2, nuc_pos, b_I)
                r12 = self._scaled_r_ee(r1, r2, d_I)
                
                r1I_pow_m = jnp.power(r1I, term_m)
                r2I_pow_n = jnp.power(r2I, term_n)
                r1I_pow_n = jnp.power(r1I, term_n)
                r2I_pow_m = jnp.power(r2I, term_m)
                r12_pow_o = jnp.power(r12, term_o)
                
                non_cusp = (r1I_pow_m * r2I_pow_n + r2I_pow_m * r1I_pow_n) * r12_pow_o
                cusp = 2.0 * r12_pow_o
                terms = jnp.where(mask, cusp, non_cusp)
                
                return jnp.sum(delta * c_I * terms)

            # Sum over nuclei of this type
            contributions = jax.vmap(compute_for_nucleus)(nuclei_group)
            u_total += jnp.sum(contributions)
            
        return u_total

    def _compute(self, r1, r2, params):
        return self._compute_forward(r1, r2, params)

    def get_param_count(self):
        count = 2 * self.n_types
        for type_terms in self.terms_per_atom_type:
            count += len(type_terms)
        return count

    def flatten_params(self, params):
        return jnp.concatenate([
            params['b_raw'].ravel(),
            params['d_raw'].ravel(),
            params['c_raw'].ravel()
        ])

    def unflatten_params(self, flat_params):
        idx = 0
        b_size = self.n_types
        b_raw = flat_params[idx:idx+b_size]
        idx += b_size
        
        d_size = self.n_types
        d_raw = flat_params[idx:idx+d_size]
        idx += d_size
        
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
