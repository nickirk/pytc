import jax
import jax.numpy as jnp
import numpy as np
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
    """Evaluate Cartesian Gaussian-type orbitals (GTOs), summed over images.

    The image sum is performed with :func:`jax.lax.scan` rather than a
    vmap-broadcast — this drops the intermediate tensor footprint from
    ``(n_ao, n_image, n_prim)`` to ``(n_ao, n_prim)`` per scan iteration.

    Molecular usage has ``images == [[0,0,0]]`` (a single zero-vector
    image), so the scan is a 1-step no-op and the previous behaviour is
    preserved. PBC usage with a few hundred lattice images is what made
    the broadcast pattern OOM at production walker counts; this version
    fixes that without changing the math.

    TODO (future efficiency): analytical Laplacian to avoid
    ``jacfwd ∘ jacfwd`` Hessian intermediates; spline interpolation
    à la CASINO/QMCPACK for the cleanest production scale.
    """
    centers2d = jnp.atleast_2d(centers)
    ctr_xyz_first = xyz[jnp.newaxis, :] - centers2d                       # (n_ao, 3)

    def per_image(carry, image):
        # carry: (n_ao,) accumulator over the image sum.
        d = ctr_xyz_first + image                                          # (n_ao, 3)
        xyz_ijk = jnp.prod(d ** ijk, axis=-1)                              # (n_ao,)
        r2 = jnp.sum(d * d, axis=-1)                                       # (n_ao,)
        gauss = jnp.exp(-expts * r2[:, jnp.newaxis])                       # (n_ao, n_prim)
        contrib = jnp.sum(coeffs * gauss, axis=-1) * xyz_ijk               # (n_ao,)
        return carry + contrib, None

    n_ao = ijk.shape[0]
    init = jnp.zeros((n_ao,), dtype=jnp.result_type(centers, ijk, expts, coeffs, xyz))
    out, _ = jax.lax.scan(per_image, init, images)
    return out                                                            # (n_ao,)

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
    """Evaluate laplacian."""
    hess = jax.jacfwd(lambda x: eval_gto_grad(mol_gto, x))(xyz)
    return jnp.trace(hess, axis1=1, axis2=2)

def eval_gto_value_and_grad(mol_gto: MolGTO, xyz: jax.Array):
    """Evaluate value and gradient."""
    val = eval_gto(mol_gto, xyz)
    grad = eval_gto_grad(mol_gto, xyz)
    return val, grad

def eval_gto_all(mol_gto: MolGTO, xyz: jax.Array):
    """Evaluate value, gradient, and laplacian."""
    val = eval_gto(mol_gto, xyz)
    grad = eval_gto_grad(mol_gto, xyz)
    lap = eval_gto_lap(mol_gto, xyz)
    return val, grad, lap

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
        vmap_eval_all = jax.vmap(lambda x: eval_gto_all(mol_gto, x))
        vals, grads, laps = vmap_eval_all(pos_flat)
        return vals.reshape(batch_shape + (-1,)), grads.reshape(batch_shape + (-1, 3)), laps.reshape(batch_shape + (-1,))
        
    else:
        raise ValueError("Unsupported derivative order")
