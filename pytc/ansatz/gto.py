import jax
import jax.numpy as jnp
import numpy as np
import folx
from typing import Generator, Tuple
from pyscf import gto
from flax import struct

def angular_momentum_xyz(ell: int) -> Generator[Tuple[int, int, int], None, None]:
    """
    Generate (lx, ly, lz) tuples for a given total angular momentum ell.
    Follows the order used in PySCF for Cartesian orbitals.
    """
    for lx in reversed(range(ell + 1)):
        for ly in reversed(range(ell + 1 - lx)):
            lz = ell - lx - ly
            yield (lx, ly, lz)

def _cartesian_gto(
    centers: jax.Array,
    ijk: jax.Array,
    expts: jax.Array,
    coeffs: jax.Array,
    images: jax.Array,
    xyz: jax.Array,
) -> jax.Array:
    """
    Evaluate Cartesian Gaussian-type orbitals (GTOs).
    """
    centers2d = jnp.atleast_2d(centers)
    ctr_xyz_first = xyz[jnp.newaxis, :] - centers2d  # (N, 3)
    ctr_xyz = ctr_xyz_first[:, jnp.newaxis, :] + images  # (N, nimages, 3)
    
    # Cartesian monomials: x^i y^j z^k
    xyz_pow = ctr_xyz ** ijk[:, jnp.newaxis, :]
    xyz_ijk = jnp.prod(xyz_pow, axis=-1) # (N, nimages)
    xyz_ijk = xyz_ijk[:, :, jnp.newaxis] # (N, nimages, 1)
    
    # Radial part: exp(-alpha * r^2)
    r2 = jnp.sum(ctr_xyz**2, axis=-1) # (N, nimages)
    gauss = jnp.exp(-expts[:, jnp.newaxis, :] * r2[:, :, jnp.newaxis]) # (N, nimages, M)
    
    # Combine
    all_prod = coeffs[:, jnp.newaxis, :] * gauss * xyz_ijk # (N, nimages, M)
    
    # Sum over images then primitives
    term_sum = jnp.sum(all_prod, axis=(1, 2)) # (N,)
    return term_sum

@struct.dataclass
class MolGTO:
    """
    JAX-compatible GTO evaluator (Cartesian).
    Stores parameters as JAX arrays (PyTree leaves).
    """
    centers: jax.Array
    ijk: jax.Array
    expts: jax.Array
    coeffs: jax.Array
    images: jax.Array
    cart: bool = struct.field(pytree_node=False, default=True)

    @classmethod
    def create(cls, mol):
        if not mol.cart:
            raise ValueError("JAX GTO evaluator currently only supports Cartesian basis sets.")
        
        centers, ijk, expts, coeffs, images = cls._extract_params(mol)
        return cls(centers, ijk, expts, coeffs, images, cart=mol.cart)
    
    @staticmethod
    def _extract_params(mol):
        centers = mol.atom_coords()
        natom = mol.natm
        atom_symbols = [mol.atom_symbol(i) for i in range(natom)]
        
        centers_aos = []
        ijks = []
        expts = []
        coeffs = []
        
        def double_factorial(n):
            if n <= 0: return 1
            return n * double_factorial(n - 2)
        
        # PySCF uses gto_norm(ell, alpha) for all Cartesian components of angular momentum ell
        # even though this means individual Cartesian components are not normalized to 1.
        def cartesian_norm(l, m, n, alpha):
            ell = l + m + n
            return gto.gto_norm(ell, alpha)

        for i, sym in enumerate(atom_symbols):
            atom_basis = mol._basis[mol.atom_pure_symbol(i)]
            atom_center = centers[i]
            
            for shell in atom_basis:
                ell = shell[0]
                primitives = np.array(shell[1:])
                es = primitives[:, 0]
                cs = primitives[:, 1:]
                n_prim = len(es)
                n_contractions = cs.shape[1]
                
                for c_idx in range(n_contractions):
                    c_vec_raw = cs[:, c_idx]
                    
                    for ijk in angular_momentum_xyz(ell):
                        lx, ly, lz = ijk
                        norms = np.array([cartesian_norm(lx, ly, lz, a) for a in es])
                        c_prim_normalized = c_vec_raw * norms
                        
                        norm_sq = 0.0
                        for p1 in range(n_prim):
                            for p2 in range(n_prim):
                                a1 = es[p1]
                                a2 = es[p2]
                                def overlap_integral(k, alpha):
                                    return double_factorial(k-1) / ((2*alpha)**(k/2.0)) * np.sqrt(np.pi/alpha) if k>0 else np.sqrt(np.pi/alpha)
                                beta = a1 + a2
                                Ix = overlap_integral(2*lx, beta)
                                Iy = overlap_integral(2*ly, beta)
                                Iz = overlap_integral(2*lz, beta)
                                S_12 = Ix * Iy * Iz
                                norm_sq += c_prim_normalized[p1] * c_prim_normalized[p2] * S_12
                        
                        contraction_norm = 1.0 / np.sqrt(norm_sq)
                        final_coeffs = c_prim_normalized * contraction_norm
                        
                        centers_aos.append(atom_center)
                        ijks.append(ijk)
                        expts.append(es)
                        coeffs.append(final_coeffs)

        max_len = max(len(e) for e in expts)
        expts_padded = []
        coeffs_padded = []
        for e, c in zip(expts, coeffs):
            pad_len = max_len - len(e)
            expts_padded.append(np.pad(e, (0, pad_len), constant_values=1.0))
            coeffs_padded.append(np.pad(c, (0, pad_len), constant_values=0.0))
            
        return (
            jnp.array(centers_aos),
            jnp.array(ijks),
            jnp.array(expts_padded),
            jnp.array(coeffs_padded),
            jnp.array([[0.0, 0.0, 0.0]]),
        )

# Standalone evaluation functions
def eval_gto(mol_gto: MolGTO, xyz: jax.Array) -> jax.Array:
    """Evaluate basis functions at a single point."""
    return _cartesian_gto(
        mol_gto.centers,
        mol_gto.ijk,
        mol_gto.expts,
        mol_gto.coeffs,
        mol_gto.images,
        xyz
    )

def eval_gto_grad(mol_gto: MolGTO, xyz: jax.Array) -> jax.Array:
    """Evaluate gradients."""
    return jax.jacfwd(lambda x: eval_gto(mol_gto, x))(xyz)

def eval_gto_lap(mol_gto: MolGTO, xyz: jax.Array) -> jax.Array:
    """Evaluate laplacian via folx forward-mode (no full Hessian)."""
    return folx.forward_laplacian(lambda x: eval_gto(mol_gto, x))(xyz).laplacian

def eval_gto_value_and_grad(mol_gto: MolGTO, xyz: jax.Array):
    """Evaluate value and gradient."""
    val = eval_gto(mol_gto, xyz)
    grad = eval_gto_grad(mol_gto, xyz)
    return val, grad

def eval_gto_all(mol_gto: MolGTO, xyz: jax.Array):
    """Evaluate value, gradient, and laplacian in one folx forward pass."""
    result = folx.forward_laplacian(lambda x: eval_gto(mol_gto, x))(xyz)
    # jacobian.data shape: (3, nao) → transpose to (nao, 3) to match jacfwd convention
    return result.x, jnp.transpose(result.jacobian.data), result.laplacian

def eval_ao(mol_gto: MolGTO, pos: jax.Array, deriv=0):
    """
    JAX-compatible eval_ao.
    pos: (batch..., 3)
    """
    batch_shape = pos.shape[:-1]
    pos_flat = pos.reshape(-1, 3)
    
    if deriv == 0:
        vmap_eval = jax.vmap(lambda x: eval_gto(mol_gto, x))
        vals = vmap_eval(pos_flat)
        return vals.reshape(batch_shape + (-1,))
        
    elif deriv == 1:
        vmap_eval_grad = jax.vmap(lambda x: eval_gto_value_and_grad(mol_gto, x))
        vals, grads = vmap_eval_grad(pos_flat)
        return vals.reshape(batch_shape + (-1,)), grads.reshape(batch_shape + (-1, 3))
        
    elif deriv == 2:
        fwd_lap = folx.forward_laplacian(lambda x: eval_gto(mol_gto, x))
        res = jax.vmap(fwd_lap)(pos_flat)
        # jacobian.data: (batch, 3, nao) → transpose to (batch, nao, 3)
        grads = jnp.transpose(res.jacobian.data, (0, 2, 1))
        return res.x.reshape(batch_shape + (-1,)), grads.reshape(batch_shape + (-1, 3)), res.laplacian.reshape(batch_shape + (-1,))
        
    else:
        raise ValueError("Unsupported derivative order")
