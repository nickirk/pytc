import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Mapping, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class BHTerm:
    """Single term in Boys-Handy expansion"""
    m: int
    n: int
    o: int
    c: float

def make_bh_jastrow(mol, terms_per_nucleus=None, epsilon=1e-8):
    """
    Creates a Boys-Handy Jastrow factor.
    
    Args:
        mol: Molecule object (pytc compatible)
        terms_per_nucleus: List of BHTerm objects or list of lists.
        epsilon: Small constant for numerical stability.
        
    Returns:
        init: Function returning initial parameters
        apply: Function evaluating the Jastrow factor
    """
    nelectron = mol.nelectron
    nuclear_pos = jnp.array(mol.atom_coords())
    nuclear_charges = jnp.array(mol.atom_charges())
    natom = len(nuclear_charges)
    
    unique_charges = jnp.sort(jnp.unique(nuclear_charges))
    n_types = len(unique_charges)
    
    atom_type_map = jnp.zeros(natom, dtype=jnp.int32)
    for i, charge in enumerate(nuclear_charges):
        type_idx = jnp.where(unique_charges == charge)[0][0]
        atom_type_map = atom_type_map.at[i].set(type_idx)
    
    if terms_per_nucleus is None:
        d_cusp = 0.5
        default_terms_for_one_nucleus = [
            BHTerm(0, 0, 1, d_cusp),
            BHTerm(0, 0, 2, 0.01),  
            BHTerm(0, 0, 3, 0.001),  
            BHTerm(0, 0, 4, -0.001),
            BHTerm(2, 0, 0, 0.001),
            BHTerm(3, 0, 0, 0.0001),
            BHTerm(4, 0, 0, 0.0001),
            BHTerm(2, 2, 0, -0.001),
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
        # Assuming terms_per_nucleus is a list of lists if n_types > 1, or list if n_types=1?
        # For simplicity, if it's a list of BHTerm, replicate for all types.
        if isinstance(terms_per_nucleus[0], BHTerm):
             terms_per_atom_type = [terms_per_nucleus for _ in range(n_types)]
        else:
             terms_per_atom_type = terms_per_nucleus

    # Prepare term arrays for JAX
    term_m_list = []
    term_n_list = []
    term_o_list = []
    max_degree = 0
    
    for type_terms in terms_per_atom_type:
        ms = [term.m for term in type_terms]
        ns = [term.n for term in type_terms]
        os = [term.o for term in type_terms]
        term_m_list.append(ms)
        term_n_list.append(ns)
        term_o_list.append(os)
        
        curr_max = max(max(ms), max(ns), max(os)) if ms else 0
        max_degree = max(max_degree, curr_max)
        
    term_m = jnp.array(term_m_list, dtype=jnp.int32)
    term_n = jnp.array(term_n_list, dtype=jnp.int32)
    term_o = jnp.array(term_o_list, dtype=jnp.int32)
    
    delta_factor = jnp.where(term_m == term_n, 0.5, 1.0)
    cusp_mask = (term_m == 0) & (term_n == 0) & (term_o == 1)

    def init() -> Mapping[str, Any]:
        b_raw = jnp.ones(n_types) * 0.5  
        d_raw = jnp.ones(n_types) * 0.5
        
        c_raw = []
        for type_terms in terms_per_atom_type:
            c_type = jnp.array([term.c for term in type_terms])
            c_raw.append(c_type)
        c_raw = jnp.array(c_raw)
        
        return {
            'b_raw': b_raw,
            'd_raw': d_raw,
            'c_raw': c_raw
        }

    def apply(r_ee, params, nspins=None, r_ae=None):
        """
        Evaluates Boys-Handy Jastrow.
        
        Args:
            r_ee: Electron-electron distances (nelec, nelec, 1) or (nelec, nelec)
            params: Parameters dictionary
            nspins: Unused
            r_ae: Electron-atom distances (nelec, natom)
        """
        if r_ae is None:
            raise ValueError("Boys-Handy Jastrow requires r_ae (electron-atom distances).")
            
        # Ensure r_ee is (nelec, nelec)
        if r_ee.ndim == 3:
            r_ee = r_ee.squeeze(-1)
            
        b = nn.softplus(params['b_raw'])
        d = nn.softplus(params['d_raw'])
        c_raw = params['c_raw']
        
        # Apply cusp mask to c parameters (fix cusp coeff to 0.5 if masked)
        # Note: In original code, it was 0.5. But usually cusp depends on spin.
        # Here we assume spin-independent or average cusp?
        # Original code: c = jnp.where(self._cusp_mask, 0.5, c_raw)
        c = jnp.where(cusp_mask, 0.5, c_raw)
        
        # Helper to compute powers
        def get_powers(x, degree):
            # x: (N, ...)
            # returns: (N, ..., degree+1)
            exponents = jnp.arange(degree + 1)
            return jnp.power(x[..., None], exponents)

        # Vectorized computation over atoms
        def compute_atom_contribution(atom_idx):
            type_idx = atom_type_map[atom_idx]
            
            b_I = b[type_idx]
            d_I = d[type_idx]
            c_I = c[type_idx]
            
            m_inds = term_m[type_idx]
            n_inds = term_n[type_idx]
            o_inds = term_o[type_idx]
            delta = delta_factor[type_idx]
            mask = cusp_mask[type_idx]
            
            # Distances
            r_iI = r_ae[:, atom_idx] # (N,)
            
            # Use safe r_ee for gradient stability at diagonal (r=0)
            # We mask the diagonal later, so the value here doesn't matter as long as it's non-zero.
            n_elec = r_ee.shape[0]
            r_ij = r_ee + jnp.eye(n_elec) # (N, N)
            
            # Scaled distances
            # Restore scaling using b and d parameters
            # Using form x = b*r / (1 + b*r) which maps [0, inf) to [0, 1)
            r_iI_bar = r_iI / (1.0 + r_iI)
            r_ij_bar = r_ij / (1.0 + r_ij)
            
            # Powers
            p_r_iI = get_powers(r_iI_bar, max_degree) # (N, deg+1)
            p_r_ij = get_powers(r_ij_bar, max_degree) # (N, N, deg+1)
            
            # Terms: sum_{i<j} [ r_iI^m r_jI^n + r_iI^n r_jI^m ] r_ij^o
            # = sum_{i!=j} r_iI^m r_jI^n r_ij^o
            
            # We can compute this for all terms at once
            # v_r_iI_m: (N, n_terms)
            v_r_iI_m = p_r_iI[:, m_inds]
            v_r_iI_n = p_r_iI[:, n_inds]
            
            # v_r_ij_o: (N, N, n_terms)
            v_r_ij_o = p_r_ij[:, :, o_inds]
            
            # Term values for each pair (i, j) and term k
            # (N, 1, K) * (1, N, K) * (N, N, K) -> (N, N, K)
            term_vals = v_r_iI_m[:, None, :] * v_r_iI_n[None, :, :] * v_r_ij_o
            
            # Mask diagonal (i=j)
            n = r_ee.shape[0]
            mask_diag = 1.0 - jnp.eye(n)
            term_vals = term_vals * mask_diag[..., None]
            
            # Sum over pairs (i, j)
            # Result: (K,)
            term_sums = jnp.sum(term_vals, axis=(0, 1))
            
            weighted_sum = jnp.sum(delta * c_I * term_sums)
            return weighted_sum

        # Sum over all atoms
        total_val = jnp.sum(jax.vmap(compute_atom_contribution)(jnp.arange(natom)))
        
        return total_val

    return init, apply
