"""
JAX-compatible spherical GTO implementation.

This implements a pure JAX version of eval_ao for spherical basis sets,
following the structure of PySCF's internal implementation but optimized for JAX
by grouping basis functions by angular momentum.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import List, Tuple, Dict, Any
from pyscf import gto
import folx
from flax import struct

def real_spherical_harmonics_all(l: int, x: jax.Array, y: jax.Array, z: jax.Array, r: jax.Array) -> jax.Array:
    """
    Compute all real spherical harmonics Y_lm(x,y,z) for a given l.
    Returns components in PySCF spherical order.
    """
    safe_r = jnp.where(r == 0, 1.0, r)
    x_norm = x / safe_r
    y_norm = y / safe_r
    z_norm = z / safe_r
    
    if l == 0:
        val = jnp.full_like(x, jnp.sqrt(1.0 / (4 * jnp.pi)))
        return val[..., None]
        
    elif l == 1:
        c = jnp.sqrt(3.0 / (4 * jnp.pi))
        px = c * x_norm
        py = c * y_norm
        pz = c * z_norm
        return jnp.stack([px, py, pz], axis=-1)
        
    elif l == 2:
        c5 = jnp.sqrt(5.0 / (16 * jnp.pi))
        c15 = jnp.sqrt(15.0 / (4 * jnp.pi))
        c15_16 = jnp.sqrt(15.0 / (16 * jnp.pi))
        
        dz2 = c5 * (3 * z_norm**2 - 1)
        dxz = c15 * x_norm * z_norm
        dyz = c15 * y_norm * z_norm
        dx2y2 = c15_16 * (x_norm**2 - y_norm**2)
        dxy = c15 * x_norm * y_norm
        
        return jnp.stack([dxy, dyz, dz2, dxz, dx2y2], axis=-1)
        
    elif l == 3:
        c7 = jnp.sqrt(7.0 / (16 * jnp.pi))
        c21 = jnp.sqrt(21.0 / (32 * jnp.pi))
        c105_16 = jnp.sqrt(105.0 / (16 * jnp.pi))
        c105_4 = jnp.sqrt(105.0 / (4 * jnp.pi))
        c35 = jnp.sqrt(35.0 / (32 * jnp.pi))
        
        fz3 = c7 * (5 * z_norm**3 - 3 * z_norm)
        fxz2 = c21 * (5 * x_norm * z_norm**2 - x_norm)
        fyz2 = c21 * (5 * y_norm * z_norm**2 - y_norm)
        fzx2zy2 = c105_16 * z_norm * (x_norm**2 - y_norm**2)
        fxyz = c105_4 * x_norm * y_norm * z_norm
        fx33xy2 = c35 * (x_norm**3 - 3 * x_norm * y_norm**2)
        f3yx2y3 = c35 * (3 * x_norm**2 * y_norm - y_norm**3)
        
        return jnp.stack([f3yx2y3, fxyz, fyz2, fz3, fxz2, fzx2zy2, fx33xy2], axis=-1)
        
    elif l == 4:
        return jnp.zeros(x.shape + (9,))

    else:
        return jnp.zeros(x.shape + (2*l + 1,))

def _eval_shell_group(
    xyz: jax.Array,
    l: int,
    centers: jax.Array,
    expts: jax.Array,
    coeffs: jax.Array,
) -> jax.Array:
    """
    Evaluate a group of shells with same angular momentum l.
    """
    r_vecs = xyz[None, :] - centers
    r = jnp.linalg.norm(r_vecs, axis=-1)
    x, y, z = r_vecs[:, 0], r_vecs[:, 1], r_vecs[:, 2]
    
    r2 = r[..., None] ** 2
    gauss = jnp.exp(-expts * r2)
    radial = jnp.sum(coeffs * gauss, axis=-1) * (r ** l)
    angular = real_spherical_harmonics_all(l, x, y, z, r)
    
    return radial[..., None] * angular

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
            
            params_by_l[ell] = {
                'centers': centers_arr,
                'expts': jnp.array(expts_padded),
                'coeffs': jnp.array(coeffs_padded),
                'indices': indices_arr
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
        
        vals = _eval_shell_group(xyz, ell, centers, expts, coeffs)
        
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
