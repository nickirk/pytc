import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Generator, Tuple, List, Any
import itertools
from pyscf import gto

def angular_momentum_xyz(ell: int) -> Generator[Tuple[int, int, int], None, None]:
    """
    Generate (lx, ly, lz) tuples for a given total angular momentum ell.
    Follows the order used in PySCF for Cartesian orbitals.
    """
    # PySCF loop order for cartesian:
    # for lx in reversed(range(l+1)):
    #   for ly in reversed(range(l+1-lx)):
    #     lz = l - lx - ly
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
    
    Args:
        centers: (N, 3) centers of Gaussian mixtures
        ijk: (N, 3) angular momentum powers
        expts: (N, M) exponents
        coeffs: (N, M) coefficients
        images: (N, nimages, 3) PBC images (use [[0,0,0]] for open boundary)
        xyz: (3,) evaluation point
        
    Returns:
        (N,) values of basis functions
    """
    centers2d = jnp.atleast_2d(centers)
    ctr_xyz_first = xyz[jnp.newaxis, :] - centers2d  # (N, 3)
    ctr_xyz = ctr_xyz_first[:, jnp.newaxis, :] + images  # (N, nimages, 3)
    
    # Cartesian monomials: x^i y^j z^k
    # (N, nimages, 3) ** (N, 1, 3) -> (N, nimages, 3)
    xyz_pow = ctr_xyz ** ijk[:, jnp.newaxis, :]
    xyz_ijk = jnp.prod(xyz_pow, axis=-1) # (N, nimages)
    xyz_ijk = xyz_ijk[:, :, jnp.newaxis] # (N, nimages, 1)
    
    # Radial part: exp(-alpha * r^2)
    r2 = jnp.sum(ctr_xyz**2, axis=-1) # (N, nimages)
    gauss = jnp.exp(-expts[:, jnp.newaxis, :] * r2[:, :, jnp.newaxis]) # (N, nimages, M)
    
    # Combine
    # coeffs: (N, M)
    # gauss: (N, nimages, M)
    # xyz_ijk: (N, nimages, 1)
    all_prod = coeffs[:, jnp.newaxis, :] * gauss * xyz_ijk # (N, nimages, M)
    
    # Sum over images then primitives
    term_sum = jnp.sum(all_prod, axis=(1, 2)) # (N,)
    return term_sum

class MolGTO:
    def __init__(self, mol):
        """
        JAX-compatible GTO evaluator.
        
        Note: This implementation currently only supports Cartesian basis sets.
        For spherical basis sets, use the PySCF eval_ao callback.
        """
        if not mol.cart:
            raise ValueError("JAX GTO evaluator currently only supports Cartesian basis sets. Please build mole with cart=True or use the PySCF eval_ao callback for spherical basis sets.")
        
        self.mol = mol
        self.params = self._extract_params(mol)
    
    def _extract_params(self, mol):
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
        
        def cartesian_norm(l, m, n, alpha):
            # Use PySCF's normalization for the shell, then apply Cartesian scaling
            # PySCF's gto_norm(l, alpha) returns normalization for spherical harmonic
            # For Cartesian components, we need to scale by sqrt((2l-1)!! / (2l_x-1)!!(2l_y-1)!!(2l_z-1)!!)
            # where l_x, l_y, l_z are the Cartesian powers
            
            # Get PySCF's normalization for the shell
            pyscf_norm = gto.gto_norm(l, alpha)
            
            # Calculate the Cartesian scaling factor
            # For a Cartesian Gaussian x^l_x y^l_y z^l_z exp(-alpha r^2)
            # The normalization relative to spherical is:
            # sqrt( (2l-1)!! / ((2l_x-1)!!(2l_y-1)!!(2l_z-1)!!) )
            total_l = l + m + n
            if total_l != l:
                # This is a Cartesian component, apply scaling
                spherical_factor = double_factorial(2*l - 1)
                cartesian_factor = double_factorial(2*l - 1) / (
                    double_factorial(2*l - 1) * double_factorial(2*m - 1) * double_factorial(2*n - 1)
                )
                # Actually, the correct formula is more complex
                # Let's use the empirical ratios we found from debugging
                # For d-orbitals: xx needs ~1.585 scaling, xy needs ~0.915 scaling
                # These correspond to sqrt(3) and 1/sqrt(3) respectively
                # For general: scaling = sqrt( (2l-1)!! / prod_i (2l_i-1)!! )
                scaling = np.sqrt(double_factorial(2*l - 1) / 
                                 (double_factorial(2*l - 1) * double_factorial(2*m - 1) * double_factorial(2*n - 1)))
                return pyscf_norm * scaling
            else:
                # s-orbital case
                return pyscf_norm

        for i, sym in enumerate(atom_symbols):
            atom_basis = mol._basis[mol.atom_pure_symbol(i)]
            atom_center = centers[i]
            
            for shell in atom_basis:
                ell = shell[0]
                primitives = np.array(shell[1:])
                es = primitives[:, 0] # Exponents
                cs = primitives[:, 1:] # Coefficients
                n_prim = len(es)
                n_contractions = cs.shape[1]
                
                # Normalize primitives first
                # We need to compute normalization for each primitive Gaussian x^i y^j z^k exp(-alpha r^2)
                
                for c_idx in range(n_contractions):
                    c_vec_raw = cs[:, c_idx]
                    
                    for ijk in angular_momentum_xyz(ell):
                        lx, ly, lz = ijk
                        
                        # Compute normalization constants for each primitive
                        norms = np.array([cartesian_norm(lx, ly, lz, a) for a in es])
                        
                        # PySCF contraction coefficients are usually for normalized primitives.
                        # The stored coefficients in `mol._basis` are `c_vec`.
                        # However, `gto.Mole` logic might involve additional normalization of the contracted function.
                        # `mol.gto_norm` gives normalization for the contracted function assuming spherical?
                        # Let's check `gto.gto_norm(l, exp)`.
                        
                        # Actually, PySCF `_basis` coefficients are for normalized primitives if `gto.gto_norm` was used during basis parsing?
                        # No, `_basis` contains raw coefficients from basis set file.
                        # Standard basis sets (like STO-3G) assume normalized primitives.
                        # But we must apply the normalization factor `norms` to the primitives.
                        
                        # Additionally, the contracted function itself might need normalization.
                        # Phi = Sum_k c_k * (Norm_k * Prim_k)
                        # <Phi|Phi> = Sum_k,l c_k c_l Norm_k Norm_l <Prim_k|Prim_l>
                        # We need to scale c_k such that <Phi|Phi> = 1.
                        
                        # Let's use PySCF to get the contraction normalization if possible.
                        # But PySCF usually handles this internally.
                        # Since we have the `mol` object, we can ask it?
                        # Not easily for individual Cartesian components.
                        
                        # Let's manually compute the contraction normalization.
                        # We have `c_vec_raw`. We define:
                        # Phi_un-normalized = Sum_p c_raw[p] * Norm_prim[p] * Gaussian_prim[p]
                        # We want Phi_final = N_contract * Phi_un-normalized.
                        
                        # First, normalize primitives:
                        c_prim_normalized = c_vec_raw * norms
                        
                        # Now compute overlap of this contracted function with itself.
                        # Overlap of two Cartesian Gaussians with exp a and b:
                        # <G_a | G_b> = I(2l, a+b) * I(2m, a+b) * I(2n, a+b)
                        # where G_a = x^l y^m z^n exp(-a r^2) (without normalization constant)
                        # But we are using normalized primitives here.
                        
                        norm_sq = 0.0
                        for p1 in range(n_prim):
                            for p2 in range(n_prim):
                                a1 = es[p1]
                                a2 = es[p2]
                                # Overlap of un-normalized primitives
                                ovlp_prim = (
                                    (1.0 / np.sqrt(cartesian_norm(lx, ly, lz, a1)**2 * cartesian_norm(lx, ly, lz, a2)**2)) # Wait, this is 1/ (N1*N2) ?
                                    # No. Overlap integral S_12.
                                    # S_12 = Integral (x^2l ... exp(-(a1+a2)r^2))
                                )
                                # Let's use the formula directly:
                                def overlap_integral(k, alpha):
                                    # Integral x^k exp(-alpha x^2)
                                    return double_factorial(k-1) / ((2*alpha)**(k/2.0)) * np.sqrt(np.pi/alpha) if k>0 else np.sqrt(np.pi/alpha)
                                
                                # Correct formula for I(2n, beta):
                                # (2n-1)!! / (2beta)^n * sqrt(pi/beta)
                                
                                beta = a1 + a2
                                Ix = overlap_integral(2*lx, beta)
                                Iy = overlap_integral(2*ly, beta)
                                Iz = overlap_integral(2*lz, beta)
                                S_12 = Ix * Iy * Iz
                                
                                term = c_prim_normalized[p1] * c_prim_normalized[p2] * S_12
                                # Wait, c_prim_normalized[p] includes norms[p]. 
                                # Phi = Sum c_prim_normalized[p] * G_prim_unnormalized[p]
                                # <Phi|Phi> = Sum c_pn[p1] * c_pn[p2] * <G_un[p1] | G_un[p2]>
                                # <G_un[p1] | G_un[p2]> is S_12 calculated above.
                                
                                # Wait, `c_prim_normalized` = `c_vec_raw` * `norms`.
                                # And `norms` was defined as 1/sqrt(Integral G^2).
                                # So G_normalized = norms * G_unnormalized.
                                # Phi = Sum c_vec_raw * G_normalized = Sum (c_vec_raw * norms) * G_unnormalized.
                                # So coeffs for unnormalized primitives are `c_prim_normalized`.
                                
                                norm_sq += c_prim_normalized[p1] * c_prim_normalized[p2] * S_12
                        
                        contraction_norm = 1.0 / np.sqrt(norm_sq)
                        
                        final_coeffs = c_prim_normalized * contraction_norm
                        
                        centers_aos.append(atom_center)
                        ijks.append(ijk)
                        expts.append(es)
                        coeffs.append(final_coeffs)

        # Padding to make rectangular arrays
        max_len = max(len(e) for e in expts)
        expts_padded = []
        coeffs_padded = []
        for e, c in zip(expts, coeffs):
            pad_len = max_len - len(e)
            expts_padded.append(np.pad(e, (0, pad_len), constant_values=1.0)) # Pad with dummy exponent? 1.0 is safe, coeff 0 will kill it.
            coeffs_padded.append(np.pad(c, (0, pad_len), constant_values=0.0))
            
        return (
            jnp.array(centers_aos),
            jnp.array(ijks),
            jnp.array(expts_padded),
            jnp.array(coeffs_padded),
            jnp.array([[0.0, 0.0, 0.0]]), # No PBC images
        )

    @partial(jax.jit, static_argnums=(0,))
    def eval(self, xyz):
        """Evaluate basis functions at a single point."""
        return _cartesian_gto(*self.params, xyz)

    @partial(jax.jit, static_argnums=(0,))
    def eval_grad(self, xyz):
        """Evaluate gradients."""
        return jax.jacfwd(self.eval)(xyz)

    @partial(jax.jit, static_argnums=(0,))
    def eval_lap(self, xyz):
        """Evaluate laplacian."""
        hess = jax.jacfwd(self.eval_grad)(xyz)
        return jnp.trace(hess, axis1=1, axis2=2)
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_value_and_grad(self, xyz):
        """Evaluate value and gradient."""
        # More efficient to compute together? 
        # For autodiff, maybe separate calls or value_and_grad wrapper
        val = self.eval(xyz)
        grad = self.eval_grad(xyz)
        return val, grad

    @partial(jax.jit, static_argnums=(0,))
    def eval_all(self, xyz):
        """Evaluate value, gradient, and laplacian."""
        val = self.eval(xyz)
        grad = self.eval_grad(xyz)
        lap = self.eval_lap(xyz)
        return val, grad, lap

def eval_ao(mol_gto: MolGTO, pos: jax.Array, deriv=0):
    """
    JAX-compatible eval_ao.
    pos: (batch..., 3)
    """
    # Vectorize over position
    # pos shape can be anything ending in 3.
    
    # Use jax.vmap to handle batching.
    # The basic functions work on single (3,) input.
    
    batch_shape = pos.shape[:-1]
    pos_flat = pos.reshape(-1, 3)
    
    if deriv == 0:
        vmap_eval = jax.vmap(mol_gto.eval)
        vals = vmap_eval(pos_flat)
        return vals.reshape(batch_shape + (-1,))
        
    elif deriv == 1:
        vmap_eval_grad = jax.vmap(mol_gto.eval_value_and_grad)
        vals, grads = vmap_eval_grad(pos_flat)
        return vals.reshape(batch_shape + (-1,)), grads.reshape(batch_shape + (-1, 3))
        
    elif deriv == 2:
        vmap_eval_all = jax.vmap(mol_gto.eval_all)
        vals, grads, laps = vmap_eval_all(pos_flat)
        return vals.reshape(batch_shape + (-1,)), grads.reshape(batch_shape + (-1, 3)), laps.reshape(batch_shape + (-1,))
        
    else:
        raise ValueError("Unsupported derivative order")
