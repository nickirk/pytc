"""
JAX-compatible spherical GTO implementation.

Spherical (i.e. real solid harmonic × radial Gaussian) basis-function values
are computed by evaluating Cartesian monomials x^{lx} y^{ly} z^{lz} for
lx+ly+lz = l and contracting with PySCF's Cartesian→spherical transformation
matrix ``pyscf.gto.mole.cart2sph(l)``.  This route handles arbitrary l
(s, p, d, f, g, h, ...) by construction: previous hand-coded spherical
harmonic formulas only covered l ≤ 3 and silently returned zeros for l ≥ 4,
producing wrong AO values for any basis set containing g or higher shells
(e.g. cc-pVQZ on TM atoms, cc-pV5Z on first-row atoms).

The contraction is

    χ_{l, m}(r) = (radial GTO) · Σ_{lx+ly+lz=l} c2s[lx,ly,lz; m] · x^{lx} y^{ly} z^{lz}

where ``r = (x,y,z)`` is electron–center displacement.  Because the
Cartesian monomials already carry the r^l factor implicitly, no separate
r^l multiplication is needed (the previous code had to apply it).

PySCF's cart2sph matrix is precomputed once per l at MolGTO_Spherical
create time, stored on the dataclass, and reused inside the JIT graph as
a constant array.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from pyscf import gto
import folx
from flax import struct


def _cart_monomial_indices(l: int) -> np.ndarray:
    """(lx, ly, lz) tuples for Cartesian shell of angular momentum l, in
    PySCF order.  Same iteration as ``pytc.ansatz.gto.angular_momentum_xyz``.
    """
    out = []
    for lx in reversed(range(l + 1)):
        for ly in reversed(range(l + 1 - lx)):
            lz = l - lx - ly
            out.append((lx, ly, lz))
    return np.array(out, dtype=np.int32)   # shape (ncart, 3)


def real_solid_harmonics_via_cart(
    l: int,
    x: jax.Array, y: jax.Array, z: jax.Array,
    cart_ijk: jax.Array,    # (ncart, 3) int monomial exponents
    c2s: jax.Array,         # (ncart, 2l+1) PySCF cart→sph matrix
) -> jax.Array:
    """Real solid harmonics S_{l,m}(r) = r^l · Y_{l,m}(r̂), m = -l..+l.

    Returns shape (..., 2l+1).
    """
    # xyz_pow[..., k, d] = (x,y,z)[d] ** cart_ijk[k, d]; we want product over d.
    # Broadcasting: r has shape (..., 1, 3), exponents have shape (ncart, 3).
    r = jnp.stack([x, y, z], axis=-1)                    # (..., 3)
    r_exp = r[..., None, :]                              # (..., 1, 3)
    ijk = cart_ijk[None, :, :]                           # (1, ncart, 3) after broadcasting
    pow_per_axis = r_exp ** ijk                          # (..., ncart, 3)
    cart = jnp.prod(pow_per_axis, axis=-1)               # (..., ncart)
    sph = cart @ c2s                                     # (..., 2l+1)
    return sph

def _eval_shell_group(
    xyz: jax.Array,
    l: int,
    centers: jax.Array,
    expts: jax.Array,
    coeffs: jax.Array,
    cart_ijk: jax.Array,
    c2s: jax.Array,
) -> jax.Array:
    """Evaluate a group of shells with the same angular momentum l.

    Uses Cartesian monomials + PySCF's cart→sph transformation, which works
    for any l (see module docstring).
    """
    r_vecs = xyz[None, :] - centers
    r2 = jnp.sum(r_vecs * r_vecs, axis=-1, keepdims=True)
    gauss = jnp.exp(-expts * r2)
    radial = jnp.sum(coeffs * gauss, axis=-1)   # (n_shells,) — no r^l here.

    x, y, z = r_vecs[:, 0], r_vecs[:, 1], r_vecs[:, 2]
    sph = real_solid_harmonics_via_cart(l, x, y, z, cart_ijk, c2s)  # (n_shells, 2l+1)

    return radial[..., None] * sph

@struct.dataclass
class MolGTO_Spherical:
    """
    JAX-compatible spherical GTO evaluator optimized by grouping shells.
    """
    params_by_l: Dict[int, Dict[str, jax.Array]]
    nao: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, mol):
        params_by_l = cls._extract_params(mol)
        return cls(params_by_l, nao=mol.nao_nr())
    
    @staticmethod
    def _cart_data(l):
        """Return JAX (cart_ijk, cart2sph) constants for angular momentum l."""
        ijk = jnp.asarray(_cart_monomial_indices(l), dtype=jnp.int32)
        c2s = jnp.asarray(gto.mole.cart2sph(l), dtype=jnp.float64)
        return ijk, c2s

    @staticmethod
    def _extract_params(mol):
        centers = mol.atom_coords()
        data_by_l = {}
        ao_loc = mol.ao_loc_nr()
        
        for i in range(mol.nbas):
            ell = mol.bas_angular(i)
            atom_idx = mol.bas_atom(i)
            atom_center = centers[atom_idx]
            
            es = mol.bas_exp(i)
            cs = mol.bas_ctr_coeff(i)
            n_contractions = cs.shape[1]
            base_ao_idx = ao_loc[i]
            
            # Normalize coefficients
            # PySCF's gto_norm gives normalization for r^l * exp(-alpha * r^2)
            norms = np.array([gto.gto_norm(ell, e) for e in es])
            
            n_prim = len(es)
            
            for c_idx in range(n_contractions):
                c_vec = cs[:, c_idx]
                c_vec_normalized = c_vec * norms
                
                start = base_ao_idx + c_idx * (2*ell + 1)
                end = start + (2*ell + 1)
                # Indices for this contraction
                # Each contraction produces 2l+1 functions
                indices = np.arange(start, end)
                
                if ell not in data_by_l:
                    data_by_l[ell] = {
                        'centers': [],
                        'expts': [],
                        'coeffs': [],
                        'indices': []
                    }
                
                data_by_l[ell]['centers'].append(atom_center)
                data_by_l[ell]['expts'].append(es)
                data_by_l[ell]['coeffs'].append(c_vec_normalized)
                data_by_l[ell]['indices'].append(indices)
        
        params_by_l = {}
        for ell, data in data_by_l.items():
            centers_arr = jnp.array(data['centers'])
            indices_arr = jnp.array(data['indices'])

            es_list = data['expts']
            cs_list = data['coeffs']
            max_prim = max(len(e) for e in es_list)

            expts_padded = []
            coeffs_padded = []
            for e, c in zip(es_list, cs_list):
                pad_len = max_prim - len(e)
                expts_padded.append(jnp.pad(e, (0, pad_len), constant_values=0.0))
                coeffs_padded.append(jnp.pad(c, (0, pad_len), constant_values=0.0))

            cart_ijk, c2s = MolGTO_Spherical._cart_data(ell)

            params_by_l[ell] = {
                'centers': centers_arr,
                'expts': jnp.array(expts_padded),
                'coeffs': jnp.array(coeffs_padded),
                'indices': indices_arr,
                'cart_ijk': cart_ijk,
                'c2s': c2s,
            }

        return params_by_l

def eval_gto_spherical(mol_gto: MolGTO_Spherical, xyz: jax.Array) -> jax.Array:
    """Evaluate spherical GTOs at a single point."""
    results = jnp.zeros((mol_gto.nao,), dtype=xyz.dtype)
    
    # Iterate over l groups. Sorted keys ensure deterministic order.
    # Note: dict keys (integers) in params_by_l are static structure of the PyTree if passed properly.
    # JAX vmap might unroll this if dict structure is preserved.
    for ell in sorted(mol_gto.params_by_l.keys()):
        params = mol_gto.params_by_l[ell]
        centers = params['centers']
        expts = params['expts']
        coeffs = params['coeffs']
        indices = params['indices']
        
        vals = _eval_shell_group(
            xyz, ell, centers, expts, coeffs,
            params['cart_ijk'], params['c2s'],
        )
        
        vals_flat = vals.ravel()
        indices_flat = indices.ravel()
        
        results = results.at[indices_flat].add(vals_flat)
        
    return results

def eval_ao_spherical(mol_gto: MolGTO_Spherical, pos: jax.Array, deriv=0):
    """
    JAX-compatible eval_ao for spherical basis sets.
    """
    batch_shape = pos.shape[:-1]
    pos_flat = pos.reshape(-1, 3)
    
    if deriv == 0:
        vmap_eval = jax.vmap(lambda x: eval_gto_spherical(mol_gto, x))
        vals = vmap_eval(pos_flat)
        return vals.reshape(batch_shape + (-1,))
    
    elif deriv == 1:
        def value_and_grad_single(xyz):
            val = eval_gto_spherical(mol_gto, xyz)
            grad = jax.jacfwd(lambda x: eval_gto_spherical(mol_gto, x))(xyz)
            return val, grad
        
        vmap_val_grad = jax.vmap(value_and_grad_single)
        vals, grads = vmap_val_grad(pos_flat)
        return (vals.reshape(batch_shape + (-1,)), 
                grads.reshape(batch_shape + (-1, 3)))
    
    elif deriv == 2:
        fwd_lap = folx.forward_laplacian(lambda x: eval_gto_spherical(mol_gto, x))
        vmap_fwd_lap = jax.vmap(fwd_lap)
        
        res = vmap_fwd_lap(pos_flat)
        
        vals = res.x
        grads = res.jacobian.data 
        # Transpose from (batch, 3, nao) to (batch, nao, 3)
        grads = jnp.transpose(grads, (0, 2, 1))
        
        laps = res.laplacian
        
        return (vals.reshape(batch_shape + (-1,)),
                grads.reshape(batch_shape + (-1, 3)),
                laps.reshape(batch_shape + (-1,)))
    
    else:
        raise ValueError(f"Unsupported derivative order: {deriv}")
