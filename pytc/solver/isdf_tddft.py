"""
Time-Dependent Density Functional Theory (tddft) for excitation energy.
Both restricted and unrestricted cases are supported.
tddft can be solved with (energy-specific) Davidson algorithm or fully diagonalization.

References, adapted from the BSE code from:
    Hillenbrand, Christopher, Jiachen Li, and Tianyu Zhu. arXiv:2410.24168 (2024).
    J. Comput. Chem. 38, 383 (2017).
    Ghosh, S. K and  Chattaraj, P. K. (Eds.). (2013).
    SIAM J. Matrix Anal. Appl. 39, 683 (2018).
"""

import time
import os
import numpy as np
import scipy
import scipy.linalg as sla
import jax
import jax.numpy as jnp
from jax import lax
import gc
from pytc.df import isdf_decompose
from pytc.df_outcore import isdf_decompose_outcore

jax.config.update("jax_enable_x64", True)
from jax.scipy import special as jsp
import h5py

from pyscf import lib, dft
from pyscf.data import nist
from pyscf.tools import mo_mapping

from scipy.spatial import KDTree
import pyscf.gto

# from fxc import einsum, isdf_lda_mvp, isdf_gga_mvp

DEFAULT_EINSUM_BACKEND = 'pytblis'
HARTREE2EV = nist.HARTREE2EV

import jax
import psutil
import os


def einsum(script, *tensors, out=None, alpha=1.0, beta=0.0, einsum_backend = DEFAULT_EINSUM_BACKEND):
    '''Wrapper for einsum supporting pytblis, pyscf.lib.einsum, or numpy.einsum backends.'''
    if einsum_backend == 'pytblis':
        try:
            import pytblis
        except ImportError:
            import numpy as np
            einsum_backend = 'numpy'
    elif einsum_backend == 'numpy':
        import numpy as np
    elif einsum_backend == 'pyscf':
        from pyscf import lib
    else:
        raise ValueError(f"Unknown einsum_backend: {einsum_backend}")


    if einsum_backend == 'pytblis':
        if out is None:
            result = pytblis.contract(script, *tensors)
        else:
            pytblis.contract(script, *tensors, out=out, alpha=alpha, beta=beta)
            return out
    elif einsum_backend == 'pyscf':
        result = lib.einsum(script, *tensors, optimize='optimal')
    else:
        result = np.einsum(script, *tensors, optimize='optimal')

    if out is None:
        if alpha != 1.0:
            result = alpha * result
        return result
    else:
        if beta == 0.0:
            out[:] = alpha * result
        else:
            out[:] = alpha * result + beta * out
        return out



# --- ISDF Extentions ---
def compute_J_munu(xi_phi, weights, coords):
    """
    Computes J_{mu, nu} = sum_i sum_j w_i w_j xi_mu(r_i) (1/|r_i - r_j|) xi_nu(r_j)
    Done in batches to avoid OOM.
    """
    print("\nBuilding ISDF kernel J_{mu, nu}...")
    t0 = time.time()
    
    w_xi = xi_phi * weights[None, :]
    n_fused, n_grid = w_xi.shape
    
    J_munu = jnp.zeros((n_fused, n_fused))
    
    # Compute in batches
    batch_size = 500
    for i in range(0, n_grid, batch_size):
        end_i = min(i + batch_size, n_grid)
        r_i = coords[i:end_i] # (B, 3)
        w_xi_i = w_xi[:, i:end_i] # (n_fused, B)
        
        # Distance matrix B x n_grid
        diff = r_i[:, None, :] - coords[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)
        
        # Add small epsilon to diagonal equivalent (when dist is 0) to avoid div by zero
        # In rigorous numerical integration, self-interaction on grid is handled differently,
        # but for demonstration we just avoid NaN.
        inv_dist = 1.0 / jnp.where(dist < 1e-10, 1e10, dist)
        
        # Contraction: sum_j (1/|r_i - r_j|) * w_xi[nu, j]
        v_j = inv_dist @ w_xi.T # shape (B, n_fused)
        
        # Second Contraction: sum_i w_xi[mu, i] * v_j[i, nu]
        J_munu_batch = w_xi_i @ v_j # shape (n_fused, n_fused)
        
        J_munu += J_munu_batch

    t1 = time.time()
    print(f"Building J_munu took: {t1 - t0:.2f} s")
    return J_munu

def compute_J_munu_lr(xi_phi, weights, coords, omega):
    """
    Computes J_{mu, nu}^{lr} = sum_i sum_j w_i w_j xi_mu(r_i) (erf(omega * |r_i - r_j|) / |r_i - r_j|) xi_nu(r_j)
    Done in batches to avoid OOM.
    """
    print(f"\nBuilding ISDF range-separated kernel J_{{mu, nu}}^{{lr}} with omega={omega}...")
    t0 = time.time()
    
    w_xi = xi_phi * weights[None, :]
    n_fused, n_grid = w_xi.shape
    
    J_munu_lr = jnp.zeros((n_fused, n_fused))
    
    # Compute in batches
    batch_size = 500
    for i in range(0, n_grid, batch_size):
        end_i = min(i + batch_size, n_grid)
        r_i = coords[i:end_i] # (B, 3)
        w_xi_i = w_xi[:, i:end_i] # (n_fused, B)
        
        # Distance matrix B x n_grid
        diff = r_i[:, None, :] - coords[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)
        
        safe_dist = jnp.where(dist < 1e-10, 1e10, dist)
        kernel_val = jsp.erf(omega * dist) / safe_dist
        
        # Fix diagonal elements explicitly where dist == 0
        # limit of erf(omega * r) / r as r -> 0 is 2 * omega / sqrt(pi)
        kernel_val = jnp.where(dist < 1e-10, 2.0 * omega / jnp.sqrt(np.pi), kernel_val)
        
        v_j = kernel_val @ w_xi.T # shape (B, n_fused)
        J_munu_batch = w_xi_i @ v_j # shape (n_fused, n_fused)
        
        J_munu_lr += J_munu_batch

    t1 = time.time()
    print(f"Building J_munu_lr took: {t1 - t0:.2f} s")
    return J_munu_lr

def compute_J_munu_numpy(xi_phi, weights, coords):
    """NumPy equivalent for exact J_{mu, nu}."""
    print("\nBuilding ISDF kernel J_{mu, nu} [NUMPY]...")
    t0 = time.time()
    
    w_xi = xi_phi * weights[None, :]
    n_fused, n_grid = w_xi.shape
    
    J_munu = np.zeros((n_fused, n_fused))
    
    batch_size = 500
    for i in range(0, n_grid, batch_size):
        end_i = min(i + batch_size, n_grid)
        r_i = coords[i:end_i]
        w_xi_i = w_xi[:, i:end_i]
        
        diff = r_i[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        
        inv_dist = 1.0 / np.where(dist < 1e-10, 1e10, dist)
        
        v_j = inv_dist @ w_xi.T
        J_munu_batch = w_xi_i @ v_j
        
        J_munu += J_munu_batch

    t1 = time.time()
    print(f"Building J_munu [NUMPY] took: {t1 - t0:.2f} s")
    return J_munu

def compute_J_munu_lr_numpy(xi_phi, weights, coords, omega):
    """NumPy equivalent for exact J_{mu, nu}^{lr}."""
    print(f"\nBuilding ISDF range-separated kernel J_{{mu, nu}}^{{lr}} [NUMPY] with omega={omega}...")
    t0 = time.time()
    
    import scipy.special
    
    w_xi = xi_phi * weights[None, :]
    n_fused, n_grid = w_xi.shape
    
    J_munu_lr = np.zeros((n_fused, n_fused))
    
    batch_size = 500
    for i in range(0, n_grid, batch_size):
        end_i = min(i + batch_size, n_grid)
        r_i = coords[i:end_i]
        w_xi_i = w_xi[:, i:end_i]
        
        diff = r_i[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        
        safe_dist = np.where(dist < 1e-10, 1e10, dist)
        kernel_val = scipy.special.erf(omega * dist) / safe_dist
        
        kernel_val = np.where(dist < 1e-10, 2.0 * omega / np.sqrt(np.pi), kernel_val)
        
        v_j = kernel_val @ w_xi.T
        J_munu_batch = w_xi_i @ v_j
        
        J_munu_lr += J_munu_batch

    t1 = time.time()
    print(f"Building J_munu_lr range separated [NUMPY] took: {t1 - t0:.2f} s")
    return J_munu_lr

def compute_dynamic_alphas(pivots, gammas=[0.25, 0.5]):
    """
    Determines optimal Gaussian exponents based on local pivot density.
    
    Args:
        pivots: (Naux, 3) array of pivot coordinates.
        gammas: Coverage factors as a list. Higher = narrower Gaussians.
        
    Returns:
        alphas: (Naux, len(gammas)) array of exponents.
    """
    # 1. Build a KDTree for fast neighbor lookup
    tree = KDTree(pivots)
    
    # 2. Query the distance to the 2nd nearest neighbor 
    # (The 1st nearest neighbor is always the point itself, dist=0)
    dists, _ = tree.query(pivots, k=2)
    h_i = dists[:, 1]
    
    # 3. Handle potential duplicate points to avoid division by zero
    # Replace zeros with a tiny epsilon or a mean distance
    h_i = np.where(h_i < 1e-8, np.mean(h_i), h_i)
    
    # 4. Apply the scaling law
    gammas_arr = np.array(gammas)
    alphas = gammas_arr[None, :] / (h_i[:, None]**2)
    
    return alphas

def build_floating_basis(pivots, alphas, unit='Bohr'):
    """
    Builds a PySCF Mole object with floating s-type Gaussians at ISDF pivots.
    
    Args:
        pivots: (Naux, 3) numpy array of spatial coordinates.
        alphas: (Naux,) numpy array of Gaussian exponents (or a single float).
        unit: 'Bohr' or 'Angstrom' corresponding to your pivot coordinates.
    """
    Naux = len(pivots)
    
    # If a single alpha is provided, broadcast it to all pivots
    if isinstance(alphas, (float, int)):
        alphas = np.full((Naux, 1), alphas)
    elif alphas.ndim == 1:
        alphas = alphas[:, None]
        
    # 1. Define Ghost Atoms at the pivot coordinates
    # We name them X0, X1, X2... so we can assign a unique alpha to each if needed
    ghost_atoms = [(f'X{i}', coord) for i, coord in enumerate(pivots)]
    
    # 2. Define the Custom Basis Dictionary
    # PySCF basis format: { 'AtomSymbol': [[ angular_momentum, (exponent, contraction_coeff) ]] }
    # l=0 is an s-type function. We use an uncontracted coefficient of 1.0.
    custom_basis = {}
    for i in range(Naux):
        funcs = []
        for a in alphas[i]:
            funcs.append([0, (a, 1.0)])
        custom_basis[f'X{i}'] = funcs
    
    # 3. Build the Auxiliary PySCF Object
    aux_mol = pyscf.gto.M(
        atom=ghost_atoms,
        basis=custom_basis,
        charge=0,
        spin=0,
        unit=unit
    )
    
    return aux_mol

@jax.jit
def _compute_J_jax_core(aux_eval, weights, xi_phi, J_PQ, rcond):
    """
    Pure JAX implementation of the projection and contraction.
    Runs entirely on GPU.
    """
    # Step 3: Contractions
    # w_aux_eval is (Naux, Ngrid)
    w_aux_eval = (aux_eval * weights[:, None]).T
    
    # S_PQ: (Naux, Naux) | V_Pmu: (Naux, Nmunu)
    S_PQ = jnp.matmul(w_aux_eval, aux_eval)
    V_Pmu = jnp.matmul(w_aux_eval, xi_phi.T)
    
    # Step 4: GPU-accelerated SVD
    # JAX uses cuSOLVER on NVIDIA GPUs for this
    U, s, Vh = jnp.linalg.svd(S_PQ, full_matrices=False)
    
    # Robust pseudo-inverse
    mask = s > s[0] * rcond
    s_inv = jnp.where(mask, 1.0 / s, 0.0)
    
    # Calculate d coefficients: (Naux, Nmunu)
    # Equivalent to S_PQ_inv @ V_Pmu
    d = jnp.matmul(Vh.T, (s_inv[:, None] * jnp.matmul(U.T, V_Pmu)))
    
    # Step 5: Final Double Contraction
    # Result = d.T @ J_PQ @ d
    # Using jnp.einsum or chain matmuls for speed
    J_munu = jnp.matmul(d.T, jnp.matmul(J_PQ, d))
    
    return J_munu


def compute_ISDF_J_kernels_DF_incore(xi_phi, weights, coords, pivots, gammas=[0.25, 0.5], omega=0, rcond=1e-12):
    import time
    from pyscf import gto
    
    # --- CPU PREPROCESSING (PySCF) ---
    t_start = time.time()
    alphas = compute_dynamic_alphas(pivots, gammas=gammas)
    aux_mol = build_floating_basis(pivots, alphas)
    
    
    if omega > 0:
        aux_mol.set_range_coulomb(omega)
        
    # Generate the static matrices on CPU
    R_cpu = aux_mol.eval_gto('GTOval', coords)
    J_PQ_cpu = aux_mol.intor('int2c2e')
    print(f"PySCF Preprocessing: {time.time() - t_start:.4f}s")
    
    
    # --- DEVICE TRANSFER ---
    # Move everything to GPU memory
    t_transfer = time.time()
    R_incore = jax.device_put(jnp.array(R_cpu))
    J_PQ_incore = jax.device_put(jnp.array(J_PQ_cpu))
    xi_phi_incore = jax.device_put(jnp.array(xi_phi))
    weights_incore = jax.device_put(jnp.array(weights))
    print(f"Host-to-Device Transfer: {time.time() - t_transfer:.4f}s")
    

    # --- JAX KERNEL EXECUTION ---
    t_jax = time.time()
    J_munu = _compute_J_jax_core(R_incore, weights_incore, xi_phi_incore, J_PQ_incore, rcond)

    # Block until finished to get accurate timing (JAX is asynchronous)
    # J_munu.block_until_ready()
    print(f"JAX GPU Kernel: {time.time() - t_jax:.4f}s")
    
    
    return J_munu

def compute_ISDF_J_kernels_DF(xi_phi, weights, coords, pivots, gammas=[0.25, 0.5], omega=0, rcond=1e-12):
    """
    Computes a single J_munu (either standard Coulomb or range-separated) by 
    projecting ISDF interpolants onto a floating Gaussian auxiliary basis and using 
    analytical integrals.
    
    Uses an SVD-based pseudo-inverse for robust projection.
    """
    from scipy import linalg
    import time
    
    label = "Standard Coulomb" if omega == 0 else f"Range-Separated (omega={omega})"
    print(f"\nBuilding Analytical DF J-Kernel ({label}) with floating basis (gammas={gammas})...")
    t0 = time.time()
    
    # 1. Create Auxiliary Molecule
    t_start = time.time()
    alphas = compute_dynamic_alphas(pivots, gammas=gammas)
    aux_mol = build_floating_basis(pivots, alphas)
    if omega > 0:
        aux_mol.set_range_coulomb(omega)
    print(f"  Step 1: Create Aux Mol took: {time.time() - t_start:.4f} s")
    
    # 2. Evaluate Auxiliary Basis on the ISDF Grid
    t_start = time.time()
    aux_eval = aux_mol.eval_gto('GTOval', coords)
    print(f"  Step 2: Eval Aux GTOs on grid took: {time.time() - t_start:.4f} s")
    
    # 3. Project ISDF onto Auxiliary Basis
    t_start = time.time()
    w_aux_eval = (aux_eval * weights[:, None]).T
    S_PQ = w_aux_eval @ aux_eval         
    V_Pmu = w_aux_eval @ xi_phi.T        
    print(f"  Step 3: S_PQ and V_Pmu contractions took: {time.time() - t_start:.4f} s")
    
    # 4. SVD-based pseudo-inverse
    t_start = time.time()
    U, s, Vh = linalg.svd(S_PQ, full_matrices=False)
    mask = s > s[0] * rcond
    s_inv = np.zeros_like(s)
    s_inv[mask] = 1.0 / s[mask]
    d = (Vh.T * s_inv) @ (U.T @ V_Pmu)
    print(f"  Step 4: SVD and d coefficient solve took: {time.time() - t_start:.4f} s")
    
    # 5. Compute J via Analytical Exact Integrals (P|Q)
    t_start = time.time()
    J_PQ = aux_mol.intor('int2c2e')      
    J_munu = d.T @ J_PQ @ d
    print(f"  Step 5: J_PQ and J_munu contraction took: {time.time() - t_start:.4f} s")
    
    t1 = time.time()
    print(f"Analytical J-kernel build total took: {t1 - t0:.2f} s")
    return J_munu


def isdf_lda_mvp(C_o, C_v, V_xc, z):
    """
    Contract a trial vector z with the ISDF-compressed LDA kernel.
    
    C_o: (naux, nocc) - Occupied orbital values at ISDF pivots
    C_v: (naux, nvir) - Virtual orbital values at ISDF pivots
    V_xc: (naux, naux) - Compressed fxc kernel V^{nu mu}
    z: (nocc, nvir) - Trial vector
    """
    # 1. Form density response in ISDF space (T^mu)
    tmp = einsum('ma,ia->mi', C_v, z)
    T_aux = einsum('mi,mi->m', C_o, tmp)
    
    # 2. Apply compressed fxc kernel
    U_aux = einsum('nm,m->n', V_xc, T_aux)
    
    # 3. Project back to MO space
    tmp_o = einsum('n,ni->ni', U_aux, C_o)
    mvp = einsum('ni,na->ia', tmp_o, C_v)
    
    return mvp

def mask_grid(x, threshold):
    return np.max(x, axis=0) > threshold

def isdf_gga_mvp(C_o, C_v, V_fxc, z):
    """
    Contract a trial vector z with the ISDF-compressed GGA kernel.
    
    C_o: (4, naux, nocc) - Occupied orbitals and gradients at ISDF pivots
    C_v: (4, naux, nvir) - Virtual orbitals and gradients at ISDF pivots
    V_fxc: (4, 4, naux, naux) - Compressed fxc kernel V_{yx}^{nu mu}
    z: (nocc, nvir) - Trial vector
    
    No einsum calls — uses only matmul and Hadamard to avoid pytblis type issues.
    """
    naux = C_o.shape[1]
    
    # Step 1: T_x^m — density response at pivots (product rule)
    #   T_0^m = sum_i C_o_i^m * (sum_a C_v_a^m * z_{ia})
    #   T_x^m = sum_i C_o_x_i^m * (sum_a C_v_a^m * z_{ia})
    #         + sum_i C_o_i^m * (sum_a C_v_x_a^m * z_{ia})     for x=1,2,3
    zv = C_v[0] @ z.T                                # (naux, nocc) — DGEMM
    T_aux = np.zeros((4, naux), dtype=np.float64)
    for x in range(4):
        T_aux[x] = np.sum(C_o[x] * zv, axis=1)       # Hadamard + reduce
    for x in range(1, 4):
        zv_grad = C_v[x] @ z.T                        # (naux, nocc) — DGEMM
        T_aux[x] += np.sum(C_o[0] * zv_grad, axis=1)  # Hadamard + reduce
    
    # Step 2: U_y^n = sum_{x} V_fxc[y,x] @ T_aux[x]  (16 DGEMV calls)
    U_aux = np.zeros((4, naux), dtype=np.float64)
    for y in range(4):
        for x in range(4):
            U_aux[y] += V_fxc[y, x] @ T_aux[x]        # DGEMV
    
    # Step 3: back-project to MO space (reverse product rule)
    #   mvp_{ia} = sum_y sum_n U[y,n] * C_o[y,n,i] * C_v[0,n,a]
    #            + sum_{y>0} sum_n U[y,n] * C_o[0,n,i] * C_v[y,n,a]
    mvp = np.zeros((C_o.shape[2], C_v.shape[2]), dtype=np.float64)
    for y in range(4):
        tmp = C_o[y] * U_aux[y, :, None]               # (naux, nocc) — Hadamard
        mvp += tmp.T @ C_v[0]                           # (nocc, nvir) — DGEMM
    for y in range(1, 4):
        tmp = C_v[y] * U_aux[y, :, None]               # (naux, nvir) — Hadamard
        mvp += C_o[0].T @ tmp                           # (nocc, nvir) — DGEMM
    
    return mvp

@jax.jit
def compress_isdf_lda_kernel(xi, wfxc):
    """
    GPU Accelerated LDA compression.
    xi: (naux, ngrid)
    wfxc: (ngrid,)
    """
    # Using jnp.matmul with broadcasting for the weight
    # (naux, ngrid) * (1, ngrid) @ (ngrid, naux)
    return (xi * wfxc[None, :]) @ xi.T

@jax.jit
def compress_isdf_gga_kernel(xi_phi, xi_grad, wfxc):
    """
    JAX version of GGA compression.
    Maintains your loop structure and logic exactly.
    """
    naux, ngrid = xi_phi.shape
    
    # 1. Initialize and Pack Interpolators
    # We use .at[].set() to mirror your NumPy assignments
    xi_full = jnp.zeros((4, naux, ngrid))
    xi_full = xi_full.at[0].set(xi_phi)
    xi_full = xi_full.at[1:4].set(jnp.transpose(xi_grad, (2, 0, 1)))
    
    # 2. Contract components
    V_fxc = jnp.zeros((4, 4, naux, naux))
    for y in range(4):
        for x in range(4):
            # Hadamard product over grid, then DGEMM
            tmp = xi_full[y] * wfxc[y, x][None, :]
            # Maintain the nested loop structure as requested
            V_fxc = V_fxc.at[y, x].set(tmp @ xi_full[x].T)
            
    return V_fxc

def tddft_full_diagonalization(multi, nocc, mo_energy, C_o, C_v, J, J_rsh = None, TDA=False, ni_fn = None, hyb_coeff = 1.0, k_rsh = None, subset_by_value = None, hybrid=True):
    """Full diagonalization of tddft equation.
    tddft equation is defined as equation 1 in doi.org/10.1002/jcc.24688.
    Spin-adapted formalism can be found in chapter 18.3.2 in "Concepts and methods in modern theoretical chemistry.
    Electronic structure (2013, CRC) Ghosh S.K., Chattaraj P.K. (eds.)"
    The working equation is rewritten as equation 15 in doi.org/10.1063/1.477483.

    Args:
        tddft (fcdmft.gw.mol.tddft.tddft): tddft object.
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.
        nocc (int) : number of occ orbs
        Lpq (ndarray) : density fitted eri
        TDA (boolean): Tamm-Dancoff approximation
        ni_fn (function) : add the real space integrated fxc kernel to apb
        hyb_coeff (float) : coefficient of HF exchange
        k_rsh (float) : coefficient of SR RSH exchange, includes percent SR excha
        subset_by_value ([a, b]), if provided, only return eigvals and vecs within this range: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html

    Returns:
        exci (double array): excitation energy.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
    """
    nspin = len(C_o)
    nmo = C_o[0].shape[1] + C_v[0].shape[1]

    # determine dimension
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]

    scale = 4.0 / nspin
    if TDA:
        scale /= 2.0

    # Build apb and amb strictly by chaining the ISDF MVPs over identity vectors
    tri_vec = np.eye(full_dim, dtype=np.double)
    
    apb = np.zeros(shape=[full_dim, full_dim], dtype=np.double)
    if not TDA:
        amb = np.zeros_like(apb)
    else:
        amb = None
        
    work_done = 0
    apb, amb, work_done = _isdf_contractions(
        C_o, C_v, J, apb, amb, tri_vec, work_done, nocc, scale, hyb_coeff, hybrid, TDA, multi
    )
    if k_rsh is not None and k_rsh > 0:
        apb, amb, work_done = _isdf_contractions(
            C_o, C_v, J_rsh, apb, amb, tri_vec, work_done, nocc, scale, k_rsh, True, TDA, 't'
        )

    # orbital energy contribution to A+B and A-B matrix
    for s in range(nspin):
        orb_diff = np.asarray(mo_energy[s][None, nocc[s] :] - mo_energy[s][: nocc[s], None]).reshape(-1)
        # Add diagonal to correct off-set slice
        diag_idx = np.arange(dim[s]) + s * dim[0]
        apb[diag_idx, diag_idx] += orb_diff
        if not TDA:
            amb[diag_idx, diag_idx] += orb_diff

    # TDDFT fxc       
    # del Lpq / Cleanup removed

    if ni_fn is not None:
        # d2E = 
        # with h5py.File('d2E.h5', 'w') as f:
        #     f.create_dataset('d2E', data=d2E.copy())
        for i in range(nspin):
            apb[i * dim[0] : i * dim[0] + dim[i], i * dim[0] : i * dim[0] + dim[i]] = ni_fn(apb, scale)

    if TDA:
        # Diagonalizing A is numerically more stable than
        # diagonalizing A^2. Solve standard hermitian eigenvalue problem

        # B = 0, so A = apb
        print('beginning scipy.linalg.eigh')
        exci, xpy = scipy.linalg.eigh(apb, subset_by_value = subset_by_value)
        X_vec = xpy.T
        Y_vec = np.zeros_like(xpy)

    else:
        # equation 15 in doi/10.1063/1.477483, solved by LAPACK function dsygvd
        if subset_by_value is not None:
            subset_by_value_sq = [s**2 for s in subset_by_value]
        else:
            subset_by_value_sq = None
        # TODO: Replace Lpq contractions with ISDF contractions
        exci_sqr, xpy_w = scipy.linalg.eigh(apb, amb, type=3, subset_by_value = subset_by_value_sq)
        exci = np.sqrt(exci_sqr)

        # dsygvd normalizes xpy_w such that
        # xpy_w @ xpy_w.T = A - B
        # Using the fact that A - B = (X+Y) @ diag(w) @ (X+Y).T,
        # we calculate X+Y = xpy_w @ diag(1/sqrt(w)).
        xpy = xpy_w / np.sqrt(exci)[None, :]

        # (A+B) |X+Y> = w |X-Y>, so
        # |X-Y> = w^-1 (A+B) |X+Y>
        xmy = (apb @ xpy) / exci[None, :]

        # Rows of X_vec and Y_vec are the eigenvectors, hence the transpose.
        X_vec = (xpy + xmy).T / 2.0
        Y_vec = (xpy - xmy).T / 2.0

    # reshape X and Y eigenvector
    if nspin == 1:
        X_vec = [X_vec.reshape(-1, nocc[0], nvir[0])]
        Y_vec = [Y_vec.reshape(-1, nocc[0], nvir[0])]
    else:
        X_vec_a, X_vec_b, Y_vec_a, Y_vec_b = [], [], [], []
        for r in range(len(exci)):
            X_vec_a.append(X_vec[r][: dim[0]].reshape(nocc[0], nvir[0]))
            X_vec_b.append(X_vec[r][dim[0] :].reshape(nocc[1], nvir[1]))
            Y_vec_a.append(Y_vec[r][: dim[0]].reshape(nocc[0], nvir[0]))
            Y_vec_b.append(Y_vec[r][dim[0] :].reshape(nocc[1], nvir[1]))
        X_vec = [np.asarray(X_vec_a), np.asarray(X_vec_b)]
        Y_vec = [np.asarray(Y_vec_a), np.asarray(Y_vec_b)]

    #tddft.exci = exci
    #tddft.X_vec = X_vec
    #tddft.Y_vec = Y_vec

    return exci, X_vec, Y_vec


def get_excitation_from_eigenvector(multi, nocc, mo_energy, C_o, C_v, J, X_vec, Y_vec, J_rsh=None, k_rsh=0.0, TDA=False):
    """Get excitation energies from corresponding eigenvectors.
    exci = (X+Y)^T (A+B) (X+Y)
    Equation.18 in doi.org/10.1063/1.477483

    Args:
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.
        nocc (int array): the number of occupied orbitals.
        mo_energy (double ndarray): orbital energy.
        C_o (double ndarray): Occupied orbitals at ISDF pivots
        C_v (double ndarray): Virtual orbitals at ISDF pivots
        J (double ndarray): Coulomb matrix at ISDF pivots
        X_vec (list of double ndarray): X block of eigenvector (excitation).
        Y_vec (list of double ndarray): Y block of eigenvector (de-excitation).
        TDA (bool, optional): use TDA approximation. Defaults to False.

    Returns:
        exci (double array): excitation energies.
    """
    nspin = len(C_o)
    naux = C_o[0].shape[0]
    nmo = C_o[0].shape[1] + C_v[0].shape[1]

    # determine dimension
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]


    nroot = X_vec[0].shape[0]
    if nspin == 1:
        xpy = (X_vec[0] + Y_vec[0]).reshape(nroot, -1)
    else:
        X_vec_a, X_vec_b = X_vec[0].reshape(nroot, -1), X_vec[1].reshape(nroot, -1)
        Y_vec_a, Y_vec_b = Y_vec[0].reshape(nroot, -1), Y_vec[1].reshape(nroot, -1)
        X_vec_ab = np.concatenate((X_vec_a, X_vec_b), axis=1)
        Y_vec_ab = np.concatenate((Y_vec_a, Y_vec_b), axis=1)
        xpy = X_vec_ab + Y_vec_ab

    # scale Coulomb matrix
    scale = 4.0 / nspin
    if TDA is True:
        scale /= 2.0

    apb_prod = np.zeros(shape=[nroot, full_dim], dtype=np.double)

    # Call the exact same _isdf_contractions we use throughout the solver!
    work_done = 0
    # Note: hybrid true is assumed, rsh can also be applied here similarly if passed in
    apb_prod, amb_prod, work_done = _isdf_contractions(
        C_o, C_v, J, apb_prod, None, xpy, work_done, nocc, scale, 1.0, True, TDA, multi
    )
    if J_rsh is not None and k_rsh > 0:
         apb_prod, amb_prod, work_done = _isdf_contractions(
            C_o, C_v, J_rsh, apb_prod, None, xpy, work_done, nocc, scale, k_rsh, True, TDA, 't'
        )

    # contraction: orbital energy difference
    for ivec in range(nroot):
        for s in range(nspin):
            orb_diff = np.asarray(mo_energy[s][None, nocc[s] :] - mo_energy[s][: nocc[s], None]).reshape(-1)
            oz = orb_diff * xpy[ivec][s * dim[0] : s * dim[0] + dim[s]]
            apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += oz

    # As shown in doi.org/10.1063/1.477483
    # equation 18 (A+B)|X+Y>=w|X-Y> and orthogonality in equation 20 <X+Y|X-Y> = delta
    # so <X+Y|(A+B) = <X+Y|w|X-Y> = w
    exci = np.zeros(shape=[nroot], dtype=np.double)
    for i in range(nroot):
        exci[i] = np.matmul(xpy[i].T, apb_prod[i])

    return exci

def tddft_davidson(
    tddft,
    multi,
    e_min=0.0,
    delta=0.0,
    core_orbs=None,
    expand_only_core=False,
    precond_exact_diag=False
):
    """Davidson algorithm for tddft.
    The Davidson algorithm follows doi.org/10.1063/1.477483.
    tddft equation is defined as equation 1 in doi.org/10.1002/jcc.24688.
    Spin-adapted formalism can be found in chapter 18.3.2 in "Concepts and methods in modern theoretical chemistry.
    Electronic structure (2013, CRC) Ghosh S.K., Chattaraj P.K. (eds.)"

    Args:
        tddft (fcdmft.gw.mol.tddft.tddft): tddft object.
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.
        e_min (float, optional): minimum desired excitation energy. Defaults to 0.0.
        delta (float, optional): energy shift for trial vector generation, typically <=0.0.
        core_orbs (optional) : filter function or AO labels or AO index, for generating trial vectors from core orbitals.
                    If this is provided, then e_min and delta are not used to generate trial vectors.

    Returns:
        exci (double array): excitation energy.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
    """
    # load matrix
    nspin = tddft.nspin
    nmo = tddft.nmo
    nocc = tddft.nocc
    mo_energy = tddft.mo_energy
    # load parameter
    TDA = tddft.TDA
    max_vec = tddft.max_vec
    nroot = tddft.nroot
    max_iter = tddft.max_iter
    max_expand = tddft.max_expand
    init_ntri = max(2, tddft.init_ntri)
    residue_thresh = tddft.residue_thresh

    # determine dimension
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]

    # initialize trial vector
    tri_vec = np.zeros(shape=[max_vec, full_dim], dtype=np.double)
    ntri = min(init_ntri, full_dim)  # initial guess size should be larger than nroot
    if tddft.trial == 'identity':
        ntri_found, tri_vec_found = get_davidson_trial_vector(
            tddft, ntri=ntri, nocc=nocc, mo_energy=mo_energy, e_min=e_min, delta=delta, core_orbs=core_orbs
        )
    elif tddft.trial == 'subspace':
        ntri_found, tri_vec_found = get_davidson_trial_vector_diag(
            ntri, multi, nocc, mo_energy, tddft.C_o, tddft.C_v, tddft.J, 
            J_rsh=getattr(tddft, 'J_rsh', None),
            nocc_sub=tddft.nocc_sub, nvir_sub=tddft.nvir_sub, e_min=e_min, delta=delta,
            TDA=TDA, hyb_coeff = tddft.hyb, k_rsh = tddft.k_rsh
        )
    else:
        raise ValueError

    if ntri_found < ntri:
        print(f'only {ntri_found} trial vectors are generated rather than {ntri}.')
        ntri = ntri_found
    if ntri_found < init_ntri:
        raise ValueError('cannot find enough trial vectors; lower e_min or add more trial vectors')
    tri_vec[:ntri, :] = tri_vec_found
    del tri_vec_found

    # initialize Davidson matrix
    apb_prod = np.zeros_like(tri_vec)
    if not TDA:
        amb_prod = np.zeros_like(tri_vec)
    else:
        amb_prod = None
    
    # Removed Lpq slicing block since C_o, C_v replaces Lia, Lii, Laa
    # These assignments are no longer needed



    if precond_exact_diag:
        assert TDA
        # Diagonal elements of the ISDF Coulomb and Exchange responses:
        # v_iaia = (ia|ia) = sum_{mu, nu} C_{o,i}^mu C_{v,a}^mu J_{mu nu} C_{o,i}^nu C_{v,a}^nu
        scale = 4.0 / nspin if not TDA else 2.0 / nspin
        
        v_iaia = []
        Kiiaa = []
        Kiaia = []
        
        for s in range(nspin):
            C_o_s = tddft.C_o[s]  # (naux, nocc)
            C_v_s = tddft.C_v[s]  # (naux, nvir)
            J = tddft.J
            
            # Form C_ov[mu, ia]
            C_ov = einsum('mi,ma->mia', C_o_s, C_v_s).reshape(J.shape[0], -1) 
            # Diagonal of C_ov^T J C_ov
            v_s = np.sum(C_ov * (J @ C_ov), axis=0) * scale
            v_iaia.append(v_s)
            
            # Exact Exchange Diagonals
            C_oo_diag = einsum('mi,mi->mi', C_o_s, C_o_s) # (naux, nocc)
            C_vv_diag = einsum('ma,ma->ma', C_v_s, C_v_s) # (naux, nvir)
            
            part_Kiiaa = einsum('mi,mn,na->ia', C_oo_diag, J, C_vv_diag).reshape(-1)
            Kiiaa.append(part_Kiiaa)
            Kiaia.append(np.sum(C_ov * (J @ C_ov), axis=0)) # Same as v_iaia unscaled

        if TDA:
            apb_diag = [v_iaia[s] - Kiiaa[s] - Kiaia[s] for s in range(nspin)]
            amb_diag = apb_diag
        else:
            apb_diag = [2 * v_iaia[s] - Kiiaa[s] - Kiaia[s] for s in range(nspin)]
            amb_diag = [Kiaia[s] - Kiiaa[s] for s in range(nspin)]
    
    if tddft.nspin == 1:
        tddft.load_fxc_intermediates()
    else:
        # TODO: fxc handling for UKS
        from pyscf.dft.libxc import xc_type
        tddft.xctype = xc_type(tddft.mf.xc)
        pass

    # Lpq is completely deleted as it isn't expected in ISDF mode
    if getattr(tddft, 'delete_lpq', False):
        pass

    iter = 0
    nprod = 0  # the number of contracted vectors
    total_contract_work = 0
    total_linalg_work = 0

    Mm = None
    Mp = None

    if tddft.ni is not None and tddft.xctype != 'HF':
        if tddft.nspin == 1 and tddft.multi.lower() != 't': 
            tddft.rho0, tddft.vxc, tddft.fxc = tddft.ni.cache_xc_kernel(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.mo_coeff[0], tddft.mo_occ[0], 0)
            tddft.rho0 = [tddft.rho0]
            tddft.vxc = [tddft.vxc]
            tddft.fxc = [tddft.fxc]
            tddft.dm0 = [tddft.dm0]
                
            
        elif tddft.nspin == 2:
            tddft.rho0, tddft.vxc, tddft.fxc = tddft.ni.cache_xc_kernel(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.mf.mo_coeff, tddft.mf.mo_occ, 1)
            
        elif tddft.multi.lower() == 't':
            tddft.rho0, tddft.vxc, tddft.fxc = tddft.ni.cache_xc_kernel(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.mf.mo_coeff, tddft.mf.mo_occ, 1)
            tddft.rho0 = [tddft.rho0]
            tddft.vxc = [tddft.vxc]
            tddft.fxc = [tddft.fxc]
            tddft.dm0 = [tddft.dm0]
                
        def numint_fn_old(tri_vec, dim, nocc, nvir):
                 
            if tddft.nspin == 1 :
                s = 0
                hermi = 0
                zs = tri_vec[s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])
                # in AO basis    
                dm_zs  = einsum('ov,pv,qo->pq', zs, tddft.mo_coeff[s][:,nocc[s]:], tddft.mo_coeff[s][:,:nocc[s]])
                if tddft.multi.lower() != 't':
                    t0 = time.time()
                    v = tddft.ni.nr_rks_fxc(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.dm0[s], dm_zs, 0, hermi, tddft.rho0[s], tddft.vxc[s], tddft.fxc[s], max_memory=tddft.max_memory_fxc)   
                    print('dm contracted with fxc time', time.time() - t0, flush = True)
                else:
                    v = 0.5*tddft.ni.nr_rks_fxc_st(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.dm0, dm_zs, 0, False,
                                          tddft.rho0[s], tddft.vxc[s], tddft.fxc[s], max_memory=tddft.max_memory_fxc)
                # return in MO basis
                v = einsum('pq,qo,pv->ov', v, tddft.mo_coeff[s][:,:nocc[s]], tddft.mo_coeff[s][:,nocc[s]:]).reshape(-1)[np.newaxis, ...]
                return v
            
            else:
                hermi = 0
                dm_z = []
                for s in range(tddft.nspin):
                    zs = tri_vec[s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])
                    dm_z.append(lib.einsum('ov,pv,qo->pq', zs, tddft.mo_coeff[s][:,nocc[s]:], tddft.mo_coeff[s][:,:nocc[s]]))
                
                dm_z = np.stack(dm_z)
                dm_z = (dm_z + dm_z.transpose(0,2,1))*0.5
                
                vdft_zs = tddft.ni.nr_uks_fxc(tddft.mol, tddft.mf.grids, tddft.mf.xc, tddft.dm0, dm_z, 0, hermi, tddft.rho0, tddft.vxc, tddft.fxc, max_memory=tddft.max_memory_fxc)
                vdft_z = []
                for s in range(tddft.nspin):
                    vdft_z.append(lib.einsum('pq,qo,pv->ov', vdft_zs[s], tddft.mo_coeff[s][:,:nocc[s]], tddft.mo_coeff[s][:,nocc[s]:]).reshape(-1))

                return vdft_z

        def numint_fn(tri_vec, dim, nocc, nvir):
            if tddft.nspin == 1:
                s = 0
                hermi = 0
                zs = tri_vec[s * dim[0] : s * dim[0] + dim[s]].reshape(nocc[s], nvir[s])

                if tddft.xctype == 'LDA':
                    if getattr(tddft, 'wfxc', None) is not None:
                        v_new = isdf_lda_mvp(tddft.C_o[0], tddft.C_v[0], tddft.wfxc, zs).reshape(-1)[np.newaxis, ...]                    
                    else:
                        raise NotImplementedError("ISDF fxc compression must be precomputed via load_fxc_intermediates")
                    return v_new
                elif tddft.xctype == 'GGA': 
                    if getattr(tddft, 'wfxc', None) is not None:
                        # USING COMPRESSED ISDF KERNEL
                        v_new = isdf_gga_mvp(tddft.C_o_gga[0], tddft.C_v_gga[0], tddft.wfxc, zs).reshape(-1)[np.newaxis, ...]
                    else:
                        raise NotImplementedError("ISDF fxc compression must be precomputed via load_fxc_intermediates")
                    if tddft.multi.lower() == 't':
                        v_new *= 0.5
                    # v = numint_fn_old(tri_vec, dim, nocc, nvir)
                    # print(v.shape, v_new.shape, np.linalg.norm(v - v_new))
                    return v_new
                else: 
                    raise NotImplementedError
            else:
                raise NotImplementedError

        if tddft.nspin == 2:
            # TODO: better fxc_handling for UKS
            numint_fn = numint_fn_old # fall back to PYSCF 
    else:
        numint_fn = None
    
    

    while iter < max_iter:
        print('\ntddft Davidson #%d iteration, ntri= %d , nprod= %d .', iter + 1, ntri, nprod)
        if not TDA:
            # TODO: Replace Lpq contractions inside _tddft_contraction with ISDF contractions
            apb_prod[nprod:ntri, :], amb_prod[nprod:ntri, :], contract_work_this_iter = _tddft_contraction(
                multi=multi,
                nocc=nocc,
                mo_energy=mo_energy,
                C_o=tddft.C_o,
                C_v=tddft.C_v,
                J=tddft.J,
                J_rsh=tddft.J_rsh,
                tri_vec=tri_vec[nprod:ntri, :],
                TDA=False,
                hybrid = tddft.hybrid,
                rsh = tddft.rsh,
                hyb_coeff = tddft.hyb,
                ni_fn = numint_fn,
                k_rsh = tddft.k_rsh
            )
        else:
            apb_prod[nprod:ntri, :], _, contract_work_this_iter = _tddft_contraction(
                multi=multi,
                nocc=nocc,
                mo_energy=mo_energy,
                C_o=tddft.C_o,
                C_v=tddft.C_v,
                J=tddft.J,
                J_rsh=tddft.J_rsh,
            
                tri_vec=tri_vec[nprod:ntri, :],
                TDA=True,
                hybrid = tddft.hybrid,
                hyb_coeff = tddft.hyb,
                ni_fn = numint_fn,
                k_rsh = tddft.k_rsh
            )
        total_contract_work += contract_work_this_iter
        print(f'work for iter {iter+1}: {float(contract_work_this_iter):.2E}')

        Mp, Mm, mmwork = update_mp_mm(Mp, Mm, tri_vec, apb_prod, amb_prod, ntri, nprod)
        Mp_sym = (Mp + Mp.T) / 2.0
        if not TDA:
            Mm_sym = (Mm + Mm.T) / 2.0
        total_linalg_work += mmwork
        nprod_prev, nprod = nprod, ntri

        nroot_current = min(nroot, ntri)
        # equation 15 in doi/10.1063/1.477483, solved by LAPACK function dsygvd

        # Save current NumPy error handling settings
        nperrhandling = np.geterr()['invalid']
        if not TDA:
            exci_sqr, xpy_w = scipy.linalg.eigh(Mp_sym.T, Mm_sym.T, type=3)
            np.seterr(invalid='raise')
            e_tri = np.sqrt(exci_sqr)
        else:
            np.seterr(invalid='raise')
            e_tri, xpy_w = scipy.linalg.eigh(Mp_sym.T, driver='evd')

        if not TDA:
            # dsygvd normalizes xpy_w such that
            # xpy_w @ xpy_w.T = A - B
            # Using the fact that A - B = (X+Y) @ diag(w) @ (X+Y).T,
            # we calculate X+Y = xpy_w @ diag(1/sqrt(w)).
            xpy = xpy_w / np.sqrt(e_tri)[None, :]

            # (A+B) |X+Y> = w |X-Y>, so
            # |X-Y> = w^-1 (A+B) |X+Y>
            xmy = (Mp_sym @ xpy) / e_tri[None, :]

            # Thanks to the use of the generalized eigensolver,
            # xpy and xmy already form a biorthonormal system.

        else:
            # TDA is easy
            xpy = xpy_w

        total_linalg_work += ntri**3

        found_roots = np.flatnonzero(e_tri >= e_min)
        nrootfound = min(nroot, found_roots.size)
        lib.logger.debug(tddft, 'lowest %d exci above minimum: \n%s', nrootfound, e_tri[found_roots[:nrootfound]])
        emin_index = np.searchsorted(e_tri, e_min, side='left')
        if emin_index + nroot_current > ntri:
            emin_index = ntri - nroot_current
            if ntri >= nroot:
                print('fewer than nroot exci found above e_min.')

        if core_orbs is not None and nspin == 1 and expand_only_core:
            if not hasattr(tddft, 'mol'):
                raise ValueError('mol object is required if core_orbs is given.')
            # Select those occupied orbitals with a significant contribution from given core orbitals.
            occ_we_want = np.flatnonzero(
                mo_mapping.mo_comps(core_orbs, tddft.mol, tddft.mo_coeff[0][:, : nocc[0]]) >= 0.3
            )
            core_roots = []

            for idx in range(emin_index, ntri):
                if not TDA:
                    Xvec = (0.5 * (xpy[:, idx].T + xmy[:, idx].T)) @ tri_vec[:ntri, :]
                else:
                    Xvec = xpy[:, idx].T @ tri_vec[:ntri, :]
                Xvec = Xvec.reshape(nocc[0], nvir[0])
                Xvecsqr = np.linalg.norm(Xvec, axis=1)
                X_core_component = np.linalg.norm(Xvecsqr[occ_we_want])
                if X_core_component > 0.3:
                    core_roots.append(idx)
                if len(core_roots) >= nroot_current:
                    break
            exci_candidate_indices = np.asarray(core_roots, dtype=int)
            lib.logger.debug(
                tddft,
                'lowest %d core excitations above minimum: \n%s',
                exci_candidate_indices.size,
                e_tri[exci_candidate_indices],
            )

        else:
            exci_candidate_indices = np.s_[emin_index : emin_index + nroot_current]

        ntri_old = ntri

        exci = e_tri[exci_candidate_indices]
        # print('e_tri: ', e_tri)
        # get left and right eigenvector in the full space, equation 25 and 26 in doi.org/10.1063/1.477483

        right_vec_tri = xpy.T[exci_candidate_indices, :]
        right_vec = np.matmul(right_vec_tri, tri_vec[:ntri, :])
        total_linalg_work += nroot_current * ntri * full_dim

        if not TDA:
            left_vec_tri = xmy.T[exci_candidate_indices, :]
            left_vec = np.matmul(left_vec_tri, tri_vec[:ntri, :])
            total_linalg_work += nroot_current * ntri * full_dim

        if not TDA:
            right_res = -exci[:, None] * left_vec
            left_res = -exci[:, None] * right_vec
            right_res += np.matmul(right_vec_tri, apb_prod[:ntri, :])
            left_res += np.matmul(left_vec_tri, amb_prod[:ntri, :])

            # check convergence
            res_norms_left = np.linalg.norm(left_res, axis=1) ** 2
            res_norms_right = np.linalg.norm(right_res, axis=1) ** 2
            res_norms = np.maximum(res_norms_left, res_norms_right)

        else:  # TDA
            right_res = -exci[:, None] * right_vec
            right_res += np.matmul(right_vec_tri, apb_prod[:ntri, :])
            res_norms = np.linalg.norm(right_res, axis=1) ** 2

        max_res_norm = np.max(res_norms)
        conv_vec = res_norms < residue_thresh
        print(f'max residue norm = {max_res_norm:0.4e}')
        if conv_vec.size >= nroot:
            if np.all(conv_vec[:nroot]):
                conv = True
                break

        not_converged = np.flatnonzero(~conv_vec)
        errs_not_converged = res_norms[not_converged]
        assert np.max(errs_not_converged) == max_res_norm
        srt_errs = np.argsort(errs_not_converged)[::-1]
        nexpand = min(max_expand, nroot_current, not_converged.size, full_dim - ntri)
        candidates_to_expand = not_converged[srt_errs[:nexpand]]

        # Gather both left and right residues
        if not TDA:
            all_res = np.empty(shape=(2 * nexpand, full_dim), dtype=np.double)
        else:
            all_res = np.empty(shape=(nexpand, full_dim), dtype=np.double)

        # preconditioning the residues, equation 29 in doi.org/10.1063/1.477483.
        for s in range(nspin):
            q_vec = exci[candidates_to_expand, None, None] - (
                mo_energy[s][None, None, nocc[s] :] - mo_energy[s][None, : nocc[s], None]
            )
            q_vec = q_vec.reshape(-1, nocc[s] * nvir[s])
            if precond_exact_diag:
                q_vec -= apb_diag[s].reshape(-1, nocc[s] * nvir[s])
            all_res[:nexpand, s * dim[0] : s * dim[0] + dim[s]] = (
                right_res[candidates_to_expand, s * dim[0] : s * dim[0] + dim[s]] / q_vec
            )
            if not TDA:
                all_res[nexpand:, s * dim[0] : s * dim[0] + dim[s]] = (
                    left_res[candidates_to_expand, s * dim[0] : s * dim[0] + dim[s]] / q_vec
                )

        # The rows of all_res are now the preconditioned left residues
        # followed by the preconditioned right residues.

        # Orthogonalize residues against current trial vectors
        all_res -= (all_res @ tri_vec[:ntri, :].T) @ tri_vec[:ntri, :]
        # Orthogonalize residues amongst themselves
        Q, R, _ = scipy.linalg.qr(all_res.T, mode='economic', pivoting=True)

        # Don't care about the small residues
        orth_res = Q.T[np.abs(np.diag(R)) > 1e-10]
        # But we should take at least one new vector.
        if orth_res.size == 0:
            orth_res = Q.T[:1]

        # Make sure the residues are orthogonal to the trial vectors
        # and normalize them.
        orth_res -= (orth_res @ tri_vec[:ntri, :].T) @ tri_vec[:ntri, :]
        orth_res /= np.linalg.norm(orth_res, axis=1)[:, None]

        n_new_vec = min(orth_res.shape[0], full_dim - ntri)
        if n_new_vec > 0:
            if ntri + n_new_vec > tri_vec.shape[0]:
                raise ValueError('Exceeded max_vec. Davidson algorithm for tddft is not converged!')
            tri_vec[ntri : ntri + n_new_vec] = orth_res[:n_new_vec]
            ntri += n_new_vec
            print(f'add {n_new_vec} new trial vectors.')
        else:
            raise ValueError('No new vectors, but Davidson has not converged')
        conv = False

        iter += 1
        if conv is True:
            break

    assert conv is True, 'Davidson algorithm for tddft is not converged!'



    print(f'tddft converged in {iter} iterations, final subspace size = {nprod}')
    print(f'total work for contraction: {float(total_contract_work):.2E}')
    print(f'total work for linalg: {float(total_linalg_work):.2E}')
    print(f'Mp condition number: {np.linalg.cond(Mp_sym)}')
    if Mm is not None:
        print(f'Mm condition number: {np.linalg.cond(Mm_sym)}')

    found_roots = np.flatnonzero((exci >= e_min) & conv_vec)
    nrootfound = found_roots.size
    lib.logger.debug(tddft, 'Finished with %d converged roots: \n%s', nrootfound, exci[found_roots])

    # transfer left and right eigenvector to X and Y

    if not TDA:
        X_vec = (left_vec[found_roots] + right_vec[found_roots]) * 0.5
        Y_vec = (-left_vec[found_roots] + right_vec[found_roots]) * 0.5
    else:
        X_vec = right_vec[found_roots]
        Y_vec = np.zeros_like(X_vec)

    # reshape X and Y eigenvector
    if nspin == 1:
        X_vec = [X_vec.reshape(nrootfound, nocc[0], nvir[0])]
        Y_vec = [Y_vec.reshape(nrootfound, nocc[0], nvir[0])]
    else:
        X_vec_a, X_vec_b, Y_vec_a, Y_vec_b = [], [], [], []
        for r in range(nrootfound):
            X_vec_a.append(X_vec[r][: dim[0]].reshape(nocc[0], nvir[0]))
            X_vec_b.append(X_vec[r][dim[0] :].reshape(nocc[1], nvir[1]))
            Y_vec_a.append(Y_vec[r][: dim[0]].reshape(nocc[0], nvir[0]))
            Y_vec_b.append(Y_vec[r][dim[0] :].reshape(nocc[1], nvir[1]))
        X_vec = [np.asarray(X_vec_a), np.asarray(X_vec_b)]
        Y_vec = [np.asarray(Y_vec_a), np.asarray(Y_vec_b)]

    tddft.exci = exci[found_roots]
    tddft.X_vec = X_vec
    tddft.Y_vec = Y_vec

    return exci[found_roots], X_vec, Y_vec


def update_mp_mm(Mp, Mm, tri_vec, apb_prod, amb_prod, ntri, nprod):
    """Update Mp and Mm to reflect the new trial vectors.

    Parameters
    ----------
    Mp : ndarray
        The matrix <tri_vec|A+B|tri_vec>
    Mm : ndarray or None
        The matrix <tri_vec|A-B|tri_vec>
    tri_vec : ndarray
        Trial vectors (stored as rows).
    apb_prod : ndarray
        The vectors (A+B)|tri_vec> (stored as rows).
    amb_prod : ndarray or None
        The vectors (A-B)|tri_vec> (stored as rows).
    ntri : int
        Number of valid trial vectors in tri_vec.
    nprod : int
        Number of valid trial vectors when Mm and Mp were last updated.

    Returns
    -------
    (ndarray, ndarray, int)
        Mm, Mp, work; where work is a rough estimate of the FLOP count.
    """
    full_dim = tri_vec.shape[1]
    work = 0
    if Mp is None or Mm is None:
        # A+B and A-B in subspace, step 3 in doi.org/10.1063/1.477483
        if apb_prod is not None:
            Mp = np.matmul(tri_vec[:ntri, :], apb_prod[:ntri, :].T)
            work += ntri**2 * full_dim

        if amb_prod is not None:
            Mm = np.matmul(tri_vec[:ntri, :], amb_prod[:ntri, :].T)
            work += ntri**2 * full_dim

    else:
        if apb_prod is not None:
            Mp_new = np.zeros(shape=[ntri, ntri], dtype=np.double)
            Mp_new[:nprod, :nprod] = Mp[:nprod, :nprod]
            Mp_new[nprod:ntri, :ntri] = tri_vec[nprod:ntri, :] @ apb_prod[:ntri, :].T
            Mp_new[:ntri, nprod:ntri] = Mp_new[nprod:ntri, :ntri].T
            Mp_new[nprod:ntri, nprod:ntri] = tri_vec[nprod:ntri, :] @ apb_prod[nprod:ntri, :].T
            Mp = Mp_new
            work += (ntri**2 - nprod**2) * full_dim

        if amb_prod is not None:
            Mm_new = np.zeros(shape=[ntri, ntri], dtype=np.double)
            Mm_new[:nprod, :nprod] = Mm[:nprod, :nprod]
            Mm_new[nprod:ntri, :ntri] = tri_vec[nprod:ntri, :] @ amb_prod[:ntri, :].T
            Mm_new[:ntri, nprod:ntri] = Mm_new[nprod:ntri, :ntri].T
            Mm_new[nprod:ntri, nprod:ntri] = tri_vec[nprod:ntri, :] @ amb_prod[nprod:ntri, :].T
            Mm = Mm_new
            work += (ntri**2 - nprod**2) * full_dim

    return Mp, Mm, work

def get_davidson_trial_vector(tddft, ntri, nocc, mo_energy, e_min=0.0, delta=0.0, core_orbs=None):
    """Generate initial trial vectors for particle-hole excitations.
    The order is determined by the occ-vir pair orbital energy difference.
    The initial trial vectors are diagonal. They are generated by taking
    occ-vir pairs with an energy difference of >= e_min + delta.

    Args:
        ntri (int): the number of desired initial trial vectors.
        nocc (int array): the number of occupied orbitals.
        mo_energy (double ndarray): orbital energy.
        e_min (float, optional): minimum desired excitation energy. Defaults to 0.0.
        delta (float, optional): energy shift for trial vector generation, typically <=0.0.

    Returns:
        ntri, int: the number of actual trial vectors generated
        tri_vec, double ndarray: initial trial vectors
    """
    nspin, nmo = mo_energy.shape
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]

    if core_orbs is not None:
        if not hasattr(tddft, 'mol'):
            raise ValueError('mol object is required for generating trial vectors for core excitations.')
        # Select those occupied orbitals with a significant contribution from given core orbitals.
        occ_to_take = [
            np.flatnonzero(mo_mapping.mo_comps(core_orbs, tddft.mol, tddft.mo_coeff[s]) >= 0.3) for s in range(nspin)
        ]
    else:
        occ_to_take = [np.arange(nocc[s], dtype=int) for s in range(nspin)]

    e_diffs = []
    e_diffs_shp = []

    for s in range(nspin):
        # The shape of e_diffs_s is (nocc[s], nvir[s])
        # e_diffs_s[i, a] = mo_energy[s][a] - mo_energy[s][i]
        e_diffs_s = mo_energy[s][None, nocc[s] :] - mo_energy[s][occ_to_take[s], None]
        e_diffs_shp.append(e_diffs_s.shape)
        # Flatten e_diffs[s] into a 1D array.
        e_diffs_s = e_diffs_s.reshape(-1)
        e_diffs.append(e_diffs_s)

    # At this point, the structure of e_diffs is as follows:
    # e_diffs[spin, ia] = mo_energy[spin][a] - mo_energy[spin][i]
    # where ia = a + nvir[spin] * i

    # Glue the e_diffs together into a 1D array.
    all_ediffs = np.concatenate(e_diffs, axis=0)

    # Compute the sizes of the occ-vir blocks for each spin.
    e_diffs_sizes = [0] + [nocc[s] * nvir[s] for s in range(nspin)]
    # Compute the starting index of each spin's occ-vir block.
    # This indicates where e_diffs[s] resides in all_ediffs, for each s.
    e_diffs_starts = np.cumsum(e_diffs_sizes)

    # Find the indices which sort all_ediffs.
    sort_index = np.argsort(all_ediffs)

    # Take the lowest ntri pairs with energy difference greater than e_min + delta.
    e_min_index = np.searchsorted(all_ediffs, e_min + delta, side='left', sorter=sort_index)
    if e_min_index + ntri > all_ediffs.size:
        # cannot find enough pairs for trial vectors; lower e_min
        ntri = all_ediffs.size - e_min_index
    exci_to_take = sort_index[e_min_index : e_min_index + ntri]

    # exci_to_take is an index into all_ediffs.
    # We need to convert it back to orbital indices.

    tri_vec = np.zeros(shape=[ntri, full_dim], dtype=np.double)
    cur_trivec = 0
    for s in range(nspin):
        # Figure out which excitation indices are in this spin block.
        exci_this_spin = np.extract(
            (exci_to_take >= e_diffs_starts[s]) & (exci_to_take < e_diffs_starts[s + 1]), exci_to_take
        )
        # Subtract the starting index of this spin's occ-vir block.
        # They are now in the form ia = i * nvir[s] + a.
        # That is, they are indices into e_diffs[s].reshape(-1).
        exci_this_spin -= e_diffs_starts[s]
        # Convert the indices from 1D form (i * nvir[s] + a) to 2D form (i, a).
        ex_occ, ex_vir = np.unravel_index(exci_this_spin, e_diffs_shp[s])
        ex_occ = occ_to_take[s][ex_occ]
        n_exci = exci_this_spin.size

        # The following is shorthand for
        # for i, a in zip(ex_occ, ex_vir):
        #     tri_vec[cur_trivec, s * dim[s] + i * nvir[s] + a] = 1.
        #     cur_trivec += 1
        tri_vec[range(cur_trivec, cur_trivec + n_exci), s * dim[s] + ex_occ * nvir[s] + ex_vir] = 1.0
        cur_trivec += n_exci

    return ntri, tri_vec

def get_davidson_trial_vector_diag(
    ntri, multi, nocc, mo_energy, C_o, C_v, J, J_rsh = None, hyb_coeff = 1.0, k_rsh = None, nocc_sub=50, nvir_sub=150, e_min=0.0, delta=0.0, TDA=False
):
    """Get trial vectors from subspace diagonalization.

    Parameters
    ----------
    ntri : int
        number of trial vectors
    multi : str
        multiplicity
    nocc : list
        number of occupied orbitals
    mo_energy : ndarray
        orbital energy
    C_o : ndarray
        ISDF occupied orbitals
    C_v : ndarray
        ISDF virtual orbitals
    J : ndarray
        ISDF Coulomb kernel
    J_rsh:
        ISDF intermediate for RSH
    k_rsh
    nocc_sub : int, optional
        number of subspace occupied orbitals, by default 50
    nvir_sub : int, optional
        number of subspace virtual orbitals, by default 150
    e_min : float, optional
        minimum desired excitation energy, by default 0.0
    delta : float, optional
        energy shift for trial vector generation, typically <=0.0, by default 0.0
    TDA : bool, optional
        use Tamm-Dancoff approximation, by default False

    Returns
    -------
    ntri : int
        the number of actual trial vectors generated
    tri_vec : double ndarray
        initial trial vectors
    """
    nspin, nmo = mo_energy.shape
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]

    # adjust active space if necessary
    nocc_sub = int(min(nocc[0], nocc_sub))
    nvir_sub = int(min(nvir[0], nvir_sub))

    if nspin == 1:
        nocc_sub = [nocc_sub]
        nvir_sub = [nvir_sub]
    else:
        # numbers of beta orbitals are determined by alpha
        spin = nocc[0] - nocc[1]
        nocc_sub = [nocc_sub, nocc_sub - spin]
        nvir_sub = [nvir_sub, nvir_sub + spin]

    # get active-space tddft input
    start = [(nocc[s] - nocc_sub[s]) for s in range(nspin)]
    end = [(nocc[s] + nvir_sub[s]) for s in range(nspin)]
    mo_energy_sub = np.asarray([mo_energy[s, start[s] : end[s]] for s in range(nspin)])
    
    # Slice the ISDF interpolants into the reduced active space
    C_o_sub = [C_o[s][:, :, start[s] : nocc[s]] for s in range(nspin)]
    C_v_sub = [C_v[s][:, :, : nvir_sub[s]] for s in range(nspin)]
    
    exci, X_vec, Y_vec = tddft_full_diagonalization(
        multi=multi, nocc=nocc_sub, mo_energy=mo_energy_sub, 
        C_o=C_o_sub, C_v=C_v_sub, J=J, J_rsh=J_rsh,
        hyb_coeff=hyb_coeff, k_rsh=k_rsh, TDA=TDA
    )

    for i in range(len(exci)):
        if exci[i] > (e_min + delta):
            first_state = i
            break

    ntri = min(ntri, len(exci) - first_state)
    tri_vec = []
    for s in range(nspin):
        tri_vec.append(np.zeros(shape=[ntri, nocc[s], nvir[s]], dtype=np.double))
        X_vec_tri = X_vec[s][first_state : first_state + ntri].reshape(ntri, nocc_sub[s], nvir_sub[s])
        tri_vec[s][:, nocc[s] - nocc_sub[s] :, :nvir_sub[s]] = X_vec_tri
        tri_vec[s] = tri_vec[s].reshape(ntri, dim[s])
    tri_vec = np.concatenate(tri_vec, axis=1)

    return ntri, tri_vec


def _isdf_contractions(C_o, C_v, J, apb_prod, amb_prod, tri_vec, work_done, nocc, scale, hyb_coeff, hybrid, TDA, multi):
    """Contraction for J and K with trial vectors using ISDF intermediates.

    Verified against exact ERIs and Lpq density-fitting contractions.
    
    ISDF approximation of 4-center ERIs:
        (pq|rs) ≈ Σ_{μν} C_p^μ C_q^μ J_{μν} C_r^ν C_s^ν

    Coulomb (J): (ia|jb) z_{jb}
        T^μ = Σ_{jb} C_o_j^μ C_v_b^μ z_{jb}     (density response in aux space)
        U^μ = Σ_ν J_{μν} T^ν                      (Coulomb contraction)
        (Jz)_{ia} = Σ_μ C_o_i^μ C_v_a^μ U^μ       (project back to MO space)

    Exchange K_A: -(ji|ab) z_{jb}
        Uses Hadamard product: K_{μν} = Σ_j C_o_j^μ (Σ_b C_v_b^ν z_{jb})
        Equivalent to: Σ_{μν} C_o_j^μ C_o_i^μ J_{μν} C_v_a^ν C_v_b^ν z_{jb}
    
    Exchange K_B: -(ib|ja) z_{jb}  [uses K_trans^T]
        Equivalent to: Σ_{μν} C_o_i^μ C_v_b^μ J_{μν} C_o_j^ν C_v_a^ν z_{jb}

    Args:
        C_o: list of occupied MOs at interpolation points, C_o[s] shape (naux, nocc_s)
        C_v: list of virtual MOs at interpolation points, C_v[s] shape (naux, nvir_s)
        J: Coulomb kernel in ISDF space, shape (naux, naux)
        apb_prod: output (A+B) @ z
        amb_prod: output (A-B) @ z
        tri_vec: trial vectors
        work_done: flop counter
        nocc: list of occupied orbital counts per spin
        scale: Coulomb scaling (4/nspin for singlet, 2/nspin for TDA)
        hyb_coeff: hybrid exchange coefficient
        hybrid: whether to include exact exchange
        TDA: Tamm-Dancoff approximation
        multi: 's'=singlet, 't'=triplet, 'u'=unrestricted
    """
    ntri = apb_prod.shape[0]
    nspin = len(C_o)
    nmo = C_o[0].shape[1] + C_v[0].shape[1]
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    naux = C_o[0].shape[0]
    nocc_list = [C_o[i].shape[1] for i in range(nspin)]
    nvir_list = [C_v[i].shape[1] for i in range(nspin)]
    
    # ---- Coulomb (J) contraction: (ia|jb) z_{jb} ----
    # Skipped for triplet since Coulomb vanishes
    if multi.lower() != 't':
        for ivec in range(ntri):
            T_aux = np.empty(shape=[nspin, naux], dtype=np.double)
            for s in range(nspin):
                z = tri_vec[ivec][s * dim[0] : s * dim[0] + dim[s]].reshape(nocc_list[s], nvir_list[s])
                # T^μ = Σ_{jb} C_o_j^μ C_v_b^μ z_{jb}
                # T_aux[s] = einsum('nj,nb,jb->n', C_o[s], C_v[s], z)
                tmp = C_v[s] @ z.T                         # (naux, nocc) — DGEMM
                T_aux[s] = np.sum(C_o[s] * tmp, axis=1)    # (naux,) — Hadamard + sum

            for s in range(nspin):
                for t in range(nspin):
                    # U^μ = Σ_ν J_{μν} T^ν
                    U_aux = J @ T_aux[t]
                    # (Jz)_{ia} = Σ_μ C_o_i^μ C_v_a^μ U^μ
                    # vz = einsum('n,ni,na->ia', U_aux, C_o[s], C_v[s]).reshape(-1) * scale
                    tmp = C_o[s] * U_aux[:, None]              # (naux, nocc) — Hadamard broadcast
                    vz = (tmp.T @ C_v[s]).reshape(-1) * scale  # (nocc, nvir) — DGEMM
                    apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += vz

    # ---- Exchange (K) contraction ----
    # K_A: -(ji|ab) z_{jb} via Hadamard product J ⊙ K_trans
    # K_B: -(ib|ja) z_{jb} via Hadamard product J ⊙ K_trans^T
    if hybrid:
        for ivec in range(ntri):
            for s in range(nspin):
                z = tri_vec[ivec][s * dim[0] : s * dim[0] + dim[s]].reshape(nocc_list[s], nvir_list[s])
    
                # Step 1: Z_v^{ν,j} = Σ_b C_v_b^ν z_{jb}
                Z_v = einsum('nb,jb->nj', C_v[s], z)
                
                # Step 2: K_{μν} = Σ_j C_o_j^μ Z_v^{ν,j}
                K_trans = einsum('mi,ni->mn', C_o[s], Z_v)
                
                # Step 3: K_A = -Σ_{μν} C_o_i^μ (J_{μν} K_{μν}) C_v_a^ν
                K_A_tilde = J * K_trans
                U_A = einsum('mn,na->ma', K_A_tilde, C_v[s])
                kaz = -einsum('mi,ma->ia', C_o[s], U_A).reshape(-1) * hyb_coeff
                
                if not TDA:
                    # Step 4: K_B uses J ⊙ K_trans^T
                    K_B_tilde = J * K_trans.T
                    U_B = einsum('mn,na->ma', K_B_tilde, C_v[s])
                    kbz = -einsum('mi,ma->ia', C_o[s], U_B).reshape(-1) * hyb_coeff
                    
                    apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += kaz + kbz
                    amb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += kaz - kbz
                else:
                    apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += kaz
    
    # ---- fxc contraction ----
    # TODO: Implement ISDF-based fxc contractions for LDA and GGA.
    # For TDHF (xc='hf'), this term is zero.
    # For DFT functionals, handled separately via ni_fn in _tddft_contraction.

    return apb_prod, amb_prod, work_done


def _tddft_contraction(multi, nocc, mo_energy, C_o, C_v, J, tri_vec, TDA=False, ni_fn = None,
                      hyb_coeff = 1.0, hybrid = True, rsh = False, J_rsh = None, k_rsh = 0.0):
    """Contraction for TDDFT matrix and trial vectors using ISDF.

    Args:
        multi (str): multiplicity, 's'=singlet, 't'=triplet, 'u'=unrestricted.
        nocc (int array): the number of occupied orbitals.
        mo_energy (double ndarray): orbital energy.
        C_o: Occupied MOs interpolants.
        C_v: Virtual MOs interpolants.
        J: Coulomb J kernel in ISDF space.
        tri_vec (double ndarray): trial vector.
        TDA (bool, optional): use TDA approximation. Defaults to False. If True, only apb_prod is returned.

    Returns:
        apb_prod, double ndarray: A+B matrix and trial vector contracted vectors.
        amb_prod, double ndarray: A-B matrix and trial vector contracted vectors.
    """
    nspin = 1
    nmo = C_o[0].shape[-1] + C_v[0].shape[-1]
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    
    dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    
    full_dim = dim[0] + dim[1] if nspin == 2 else dim[0]

    ntri = tri_vec.shape[0]
    work_done = 0

    scale = 4.0 / nspin
    if TDA is True:
        scale /= 2.0

    apb_prod = np.zeros(shape=[ntri, full_dim], dtype=np.double)
    if TDA:
        amb_prod = None
    else:
        amb_prod = np.zeros(shape=[ntri, full_dim], dtype=np.double)



    apb_prod, amb_prod, work_done = _isdf_contractions(C_o, C_v, J, apb_prod, amb_prod, tri_vec, work_done, nocc, scale, hyb_coeff, hybrid, TDA, multi)
    if k_rsh > 0:
        apb_prod, amb_prod, work_done = _isdf_contractions(C_o, C_v, J_rsh, apb_prod, amb_prod, tri_vec, work_done, nocc, scale, k_rsh, True, TDA, 't')
            
    # contraction: orbital energy difference
    for s in range(nspin):
        orb_diff = np.asarray(mo_energy[s][None, nocc[s] :] - mo_energy[s][: nocc[s], None]).reshape(-1)
        for ivec in range(ntri):
            oz = orb_diff * tri_vec[ivec][s * dim[0] : s * dim[0] + dim[s]]
            apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += oz
            if not TDA:
                amb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += oz
            work_done += 2 * oz.size


    # TDDFT fxc                 
    if ni_fn is not None:
        for ivec in range(ntri):
            vdft_z = ni_fn(tri_vec[ivec], dim, nocc, nvir)
            for s in range(nspin):
                apb_prod[ivec][s * dim[0] : s * dim[0] + dim[s]] += vdft_z[s] * scale

    return apb_prod, amb_prod, work_done

# ---------------------------------------------------------------------------
# Streaming helpers: build J_munu and wfxc without holding full xi_phi/xi_grad
# ---------------------------------------------------------------------------

@jax.jit
def update_df_kernels(aux_b, xi_b, w_b):
    w_aux_b = (aux_b * w_b[:, None]).T
    S = w_aux_b @ aux_b
    V = w_aux_b @ xi_b.T
    return S, V

def compute_ISDF_J_kernels_DF_streaming(
    h5_path, weights, coords, pivots, gammas=[0.25, 0.5], omega=0, rcond=1e-12,
    batch_size=4096, backend='jax'
):
    """
    Streaming version of compute_ISDF_J_kernels_DF.

    Reads xi_phi from `h5_path` in grid batches and accumulates S_PQ and
    V_Pmu.  When backend='jax' (default) each batch is pushed to the JAX
    default device (GPU if available).  When backend='numpy' everything
    runs on CPU with pure NumPy — no JAX required.

    Args:
        h5_path   : str  — HDF5 file with dataset 'xi_phi'  (naux, ngrid)
        weights   : (ngrid,) integration weights
        coords    : (ngrid, 3) grid coordinates
        pivots    : (naux, 3) ISDF pivot coordinates
        gammas, omega, rcond : same as compute_ISDF_J_kernels_DF
        batch_size : number of grid points processed per batch
        backend   : 'jax' (default, GPU-accelerated) or 'numpy' (CPU-only)

    Returns:
        J_munu : (naux, naux) ndarray  (on CPU)
    """
    label = "Standard Coulomb" if omega == 0 else f"Range-Separated (omega={omega})"
    print(f"\nBuilding Streaming J-Kernel ({label}) [{backend.upper()}] with floating basis (gammas={gammas})...")
    t0 = time.time()

    # Build auxiliary molecule (CPU / PySCF, done once)
    alphas  = compute_dynamic_alphas(pivots, gammas=gammas)
    aux_mol = build_floating_basis(pivots, alphas)
    if omega > 0:
        aux_mol.set_range_coulomb(omega)

    n_grid   = coords.shape[0]
    n_aux_df = aux_mol.nao  # n_aux_df = naux * len(gammas)

    from concurrent.futures import ThreadPoolExecutor

    def fetch_batch_j(start_idx, end_idx):
        return np.array(f['xi_phi'][:, start_idx:end_idx]), weights[start_idx:end_idx], coords[start_idx:end_idx]

    with h5py.File(h5_path, 'r') as f, ThreadPoolExecutor(max_workers=1) as executor:
        n_fused = f['xi_phi'].shape[0]

        S_PQ  = np.zeros((n_aux_df, n_aux_df), dtype=np.float64)
        V_Pmu = np.zeros((n_aux_df, n_fused),  dtype=np.float64)

        next_future = None
        if n_grid > 0:
            first_end = min(batch_size, n_grid)
            next_future = executor.submit(fetch_batch_j, 0, first_end)

        for g_start in range(0, n_grid, batch_size):
            g_end     = min(g_start + batch_size, n_grid)
            
            xi_b_cpu, weights_b, coords_b = next_future.result()

            next_start = g_start + batch_size
            if next_start < n_grid:
                next_end = min(next_start + batch_size, n_grid)
                next_future = executor.submit(fetch_batch_j, next_start, next_end)

            # aux_eval is always a PySCF CPU call
            aux_b_np = aux_mol.eval_gto('GTOval', coords_b)  # (B, naux_df)

            if backend == 'numpy':
                # Pure NumPy path — no device transfers
                w_aux_b = (aux_b_np * weights_b[:, None]).T   # (naux_df, B)
                S_PQ  += np.matmul(w_aux_b, aux_b_np)         # (naux_df, naux_df)
                V_Pmu += np.matmul(w_aux_b, xi_b_cpu.T)        # (naux_df, naux)
                del w_aux_b, xi_b_cpu
            else:
                # JAX path — push batch to device (GPU if available)
                aux_b  = jax.device_put(jnp.array(aux_b_np))
                xi_b   = jax.device_put(jnp.array(xi_b_cpu))
                w_b    = jax.device_put(jnp.array(weights_b))
                spq_batch, vpm_batch = update_df_kernels(aux_b, xi_b, w_b)
                S_PQ  += np.array(spq_batch)
                V_Pmu += np.array(vpm_batch)
                del aux_b, xi_b, w_b, spq_batch, vpm_batch, xi_b_cpu

    gc.collect()
    if backend == 'jax':
        jax.clear_caches()

    # 1. Solve symmetric eigenproblem (always NumPy — already on CPU)
    s, U = np.linalg.eigh(S_PQ)

    # 2. Sort descending (eigh returns ascending)
    s = s[::-1]
    U = U[:, ::-1]

    # 3. Apply rcond mask to find the effective rank (k)
    k = int(np.sum(s > (s[0] * rcond)))  # Number of 'important' dimensions
    print(f'k / n_aux_df for eigh(S_PQ): {k} / {n_aux_df}')
    print(f'minimum eigval(S_PQ):', s.min())

    # 4. Project to DF auxiliary space
    U_s = U[:, :k] * (1.0 / s[:k])
    d = U[:, :k].T @ V_Pmu  # (k, naux)
    d = U_s @ d             # (n_aux_df, naux)

    del S_PQ, V_Pmu, U, U_s
    gc.collect()

    # Analytical 2-centre integrals (PySCF CPU)
    J_PQ   = aux_mol.intor('int2c2e')  # (naux_df, naux_df)
    J_munu = d.T @ J_PQ @ d           # (naux, naux)

    print(f"Streaming J-kernel build took: {time.time() - t0:.2f} s")
    return J_munu


def compress_isdf_lda_kernel_streaming(h5_path, wfxc_real, batch_size=4096, backend='jax'):
    """
    Streaming LDA fxc compression.

    Computes  wfxc[mu, nu] = sum_g  xi_phi[mu,g] * wfxc_real[g] * xi_phi[nu,g]
    by reading xi_phi in batches from HDF5.
    When backend='jax' (default) each batch is pushed to the JAX device.
    When backend='numpy' everything runs as pure NumPy on CPU.

    Args:
        h5_path   : str — HDF5 file with dataset 'xi_phi'  (naux, ngrid)
        wfxc_real : (ngrid,) ndarray of  fxc * weight  on the grid
        batch_size : grid batch size
        backend   : 'jax' (default) or 'numpy'

    Returns:
        wfxc : (naux, naux) ndarray  (on CPU)
    """
    print(f"Streaming LDA fxc compression [{backend.upper()}]...")
    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        n_fused, n_grid = f['xi_phi'].shape
        wfxc = np.zeros((n_fused, n_fused), dtype=np.float64)

        for g_start in range(0, n_grid, batch_size):
            g_end = min(g_start + batch_size, n_grid)
            if backend == 'numpy':
                xi_b = np.array(f['xi_phi'][:, g_start:g_end])  # (naux, B)
                w_b  = np.asarray(wfxc_real[g_start:g_end])     # (B,)
                wfxc += np.matmul(xi_b * w_b, xi_b.T)           # (naux, naux)
                del xi_b, w_b
            else:
                xi_b = jax.device_put(jnp.array(f['xi_phi'][:, g_start:g_end]))  # (naux, B) GPU
                w_b  = jax.device_put(jnp.array(wfxc_real[g_start:g_end]))       # (B,)      GPU
                wfxc += np.array(jnp.matmul(xi_b * w_b, xi_b.T)) # (naux, naux)
                del xi_b, w_b

    print(f"  => wfxc shape {wfxc.shape}, took {time.time()-t0:.2f} s")
    return wfxc

@jax.jit
def update_df_gga_kernels(xi_phi, xi_grad, w_batch):
    xi_full = jnp.concatenate([
        xi_phi[None, :, :],
        xi_grad.transpose(2, 0, 1)
    ], axis=0)
    tmp = jnp.einsum('ymg,yxg->yxmg', xi_full, w_batch)   # (Y,X,M,G)
    return jnp.einsum('yxmg,xng->yxmn', tmp, xi_full)

def compress_isdf_gga_kernel_streaming(h5_path, wfxc_real, batch_size=4096, backend='jax'):
    """
    Streaming GGA fxc compression.

    Computes  wfxc[y,x,mu,nu] = sum_g  xi_full[y,mu,g] * wfxc_real[y,x,g] * xi_full[x,nu,g]
    where xi_full = [xi_phi, xi_grad_x, xi_grad_y, xi_grad_z]  (shape 4, naux, ngrid).
    When backend='jax' (default) each batch is pushed to the JAX device.
    When backend='numpy' everything runs as pure NumPy on CPU.

    Args:
        h5_path   : str — HDF5 file with datasets 'xi_phi' (naux, ngrid) and
                          'xi_grad' (naux, ngrid, 3)
        wfxc_real : (4, 4, ngrid) ndarray of fxc (transposed to (y, x, r) convention)
        batch_size : grid batch size
        backend   : 'jax' (default) or 'numpy'

    Returns:
        wfxc_cpu : (4, 4, naux, naux) ndarray  (on CPU)
    """
    print(f"Streaming GGA fxc compression [{backend.upper()}]...")
    t0 = time.time()
    _tt = {'h5': 0.0, 'xi': 0.0, 'w': 0.0, 'einsum': 0.0, 'copy': 0.0}

    from concurrent.futures import ThreadPoolExecutor

    def fetch_batch_gga(start_idx, end_idx):
        xi_phi_b  = np.array(f['xi_phi'][:, start_idx:end_idx])
        xi_grad_b = np.array(f['xi_grad'][:, start_idx:end_idx, :])
        w_batch   = np.array(wfxc_real[:, :, start_idx:end_idx])
        return xi_phi_b, xi_grad_b, w_batch

    with h5py.File(h5_path, 'r') as f, ThreadPoolExecutor(max_workers=1) as executor:
        n_aux, n_grid = f['xi_phi'].shape
        wfxc_cpu = np.zeros((4, 4, n_aux, n_aux), dtype=np.float64)

        next_future = None
        if n_grid > 0:
            first_end = min(batch_size, n_grid)
            next_future = executor.submit(fetch_batch_gga, 0, first_end)

        for g_start in range(0, n_grid, batch_size):
            g_end = min(g_start + batch_size, n_grid)

            xi_phi_b_cpu, xi_grad_b_cpu, w_batch_cpu = next_future.result()

            next_start = g_start + batch_size
            if next_start < n_grid:
                next_end = min(next_start + batch_size, n_grid)
                next_future = executor.submit(fetch_batch_gga, next_start, next_end)

            if backend == 'numpy':
                # Load intermediates into numpy
                xi_phi_b  = xi_phi_b_cpu
                xi_grad_b = xi_grad_b_cpu

                # Shape: (4, naux, B)
                xi_full = np.concatenate([
                    xi_phi_b[np.newaxis, :, :],
                    xi_grad_b.transpose(2, 0, 1)
                ], axis=0)

                w_batch = w_batch_cpu

                # Vectorized: no loops over y/x.
                # weighted_xi[y,x,m,g] = xi_full[y,m,g] * w_batch[y,x,g]
                weighted_xi = xi_full[:, np.newaxis, :, :] * w_batch[:, :, np.newaxis, :]  # (4,4,n_aux,B)
                # Batched matmul: (4,4,n_aux,B) @ (1,4,B,n_aux) -> (4,4,n_aux,n_aux)
                # np.matmul broadcasts (4,4) vs (1,4) batch dims correctly.
                update = weighted_xi @ xi_full.transpose(0, 2, 1)[np.newaxis]  # (4,4,n_aux,n_aux)
                wfxc_cpu += update
                del xi_phi_b, xi_grad_b, xi_full, w_batch, weighted_xi, update
            else:
                # JAX path — detailed timing
                tb = time.time()

                _t = time.time()
                xi_phi_b  = jax.device_put(jnp.array(xi_phi_b_cpu))
                xi_grad_b = jax.device_put(jnp.array(xi_grad_b_cpu))
                # host-to-device transfer time (disk read is hidden)
                t_h5 = time.time() - _t; _tt['h5'] += t_h5

                _t = time.time()
                # Concatenation is now inside the JIT compiled function
                t_xi = time.time() - _t; _tt['xi'] += 0.0 # kept 0.0 to not break print output

                _t = time.time()
                w_batch = jax.device_put(jnp.array(w_batch_cpu))
                t_w = time.time() - _t; _tt['w'] += t_w

                _t = time.time()
                update = update_df_gga_kernels(xi_phi_b, xi_grad_b, w_batch)
                update.block_until_ready()  # UNCOMMENTED: So einsum takes the timing
                t_einsum = time.time() - _t; _tt['einsum'] += t_einsum

                _t = time.time()
                # Fast Device-to-Host transfer and in-place accumulate
                # Note: `np.add` is already perfectly optimized for straight layout (ijkl -> ijkl).
                # No transposition is needed.
                np.add(wfxc_cpu, jax.device_get(update), out=wfxc_cpu)
                t_copy = time.time() - _t; _tt['copy'] += t_copy

                t_batch = time.time() - tb
                print(f"  batch {g_start:6d}-{g_end:6d}: "
                      f"h5/D2H={t_h5:.3f}s  xi={t_xi:.3f}s  w={t_w:.3f}s  "
                      f"einsum={t_einsum:.3f}s  copy={t_copy:.3f}s  | total={t_batch:.3f}s")

                del xi_phi_b, xi_grad_b, w_batch, update, xi_phi_b_cpu, xi_grad_b_cpu, w_batch_cpu
                jax.clear_caches()

    t_total = time.time() - t0
    print(f"  => wfxc shape {wfxc_cpu.shape}, took {t_total:.2f} s")
    if backend == 'jax':
        print(f"  [timing totals]  h5={_tt['h5']:.2f}s  xi={_tt['xi']:.2f}s  "
              f"w={_tt['w']:.2f}s  einsum={_tt['einsum']:.2f}s  copy={_tt['copy']:.2f}s")
    return wfxc_cpu


class TDDFT(lib.StreamObject):
    def __init__(
        self,
        # initialize with a GW object
        mf=None,
        # initialize with nocc, mo_energy, C_o, C_v, J
        nocc=None,
        isdf_rcond=1e-6,
        isdf_grid_batch_size=2048,
        isdf_grid_level=3,
        isdf_naux_factor=8,
        isdf_gammas=[0.1, 0.4],
        isdf_stream_path=None,
        isdf_stream_batch_size=4096,
        isdf_exact_J=False,
        isdf_grid_rho_cutoff = 0,
        isdf_backend = 'numpy',
        isdf_cd_sample_factor = None,
        isdf_cd_seed = 42,
        isdf_skip_grad_pivots = False,
        verbose=5,
        # options
        TDA=False,
        nroot=10,
        max_vec=None,
        max_iter=100,
        max_expand=None,
        init_ntri=None,
        residue_thresh=1.0e-8,
        delete_lpq=False,
        max_memory_fxc = 2000,
        trial_method = 'identity'
    ):
        """Initialize tddft object.
        The tddft object can be initialized by restricted or unrestricted GW object.

        Args:
            gw (fcdmft.gw.mol, optional): GW object.
            nocc (int/int array, optional): number of occupied orbitals.
            mo_energy(double ndarray, optional): quasiparticle energy.
            C_o, C_v, J (double ndarray, optional): ISDF compressed matrices.
            verbose (int, optional): print level.
            TDA (bool, optional): use TDA approximation to ignore B matrix. Defaults to False.
            nroot (int, optional): the number of desired roots. Defaults to 10.
            max_vec (int, optional): max allowed subspace size. Defaults to 200.
            max_iter (int, optional): max Davidson iteration. Defaults to 100.
            max_expand(int, optional): max number of trial vectors to expand. Defaults to nroot.
            init_ntri (int, optional): initial number of trial vectors. Default is 4 * nroot.
            residue_thresh (double, optional): threshold if the residue needs to be added as a new trial vector.
            Defaults to 1.0e-8.
            delete_lpq (bool, optional): delete Lpq during the calculation to save memory. Defaults to False.
        """
        # initialize matrix
    
        if mf is not None:
            self.nspin = 1 if np.asarray(mf.mo_energy).ndim == 1 else 2
            self.verbose = mf.mol.verbose
            self.mol = mf.mol
            self.mf = mf
            self.nocc = [mf.mol.nelectron // 2] if self.nspin == 1 else mf.nelec
            
            self.mo_energy = np.asarray(mf.mo_energy)
            if self.mo_energy.ndim == 1:
                self.mo_energy = self.mo_energy[np.newaxis, ...]
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = mf.mo_occ
            if self.mo_coeff.ndim == 2:
                self.mo_coeff = self.mo_coeff[np.newaxis, ...]
                self.mo_occ = self.mo_occ[np.newaxis, ...]
            self.nmo = self.mo_energy.shape[-1]

        # options
        self.TDA = TDA  # use TDA approximation to ignore B matrix
        
        # ISDF parameters
        self.isdf_rcond = isdf_rcond
        self.isdf_grid_level = isdf_grid_level
        self.isdf_naux_factor = isdf_naux_factor
        self.isdf_gammas = isdf_gammas
        self.isdf_grid_batch_size = isdf_grid_batch_size
        self.isdf_grid_rho_cutoff = isdf_grid_rho_cutoff  # if True, use O(Ngrid^2) compute_J_munu directly
        # Streaming: if set, xi_phi/xi_grad are written to this HDF5 path and
        # streamed during J-kernel build and fxc compression instead of being
        # loaded fully into RAM.  After all compressed objects are built the
        # file is deleted automatically.
        self.isdf_stream_path = isdf_stream_path
        self.isdf_stream_batch_size = isdf_stream_batch_size
        self.isdf_exact_J = isdf_exact_J  # if True, use O(Ngrid^2) compute_J_munu directly
        self._isdf_h5_path = None  # internal: set during _build_isdf_intermediates
        self.isdf_backend = isdf_backend  # 'jax' or 'numpy' for isdf_decompose_outcore
        self.isdf_cd_sample_factor = isdf_cd_sample_factor
        self.isdf_cd_seed = isdf_cd_seed
        self.isdf_skip_grad_pivots = isdf_skip_grad_pivots
        self.mf.grids.level = self.isdf_grid_level
        self.mf.grids.build(with_non0tab=False)

        # Davidson algorithm
        self.multi = None  # multiplicity
        self.nroot = nroot  # the number of desired roots
        self.trial = trial_method  # mode to initialize trial vector
        self.nocc_sub = 50  # number of occpuied orbitals in the trial vector subspace
        self.nvir_sub = 150  # number of virtual orbitals in the trial vector subspace
        self.max_vec = 12 * nroot if max_vec is None else max_vec  # max allowed subspace size
        self.max_iter = max_iter  # max Davidson iteration
        # max number of trial vectors to expand per iteration
        self.max_expand = min(100, nroot) if max_expand is None else max_expand
        self.residue_thresh = residue_thresh  # threshold if the residue needs to be added as a new trial vector
        self.init_ntri = min(100, nroot) if init_ntri is None else init_ntri

        # results
        self.exci = None  # excitation energy
        self.X_vec = None  # X block of eigenvector (excitation)
        self.Y_vec = None  # Y block of eigenvector (de-excitation)

        # TDDFT part, for fxc
        self.max_memory_fxc = max_memory_fxc
        if mf is None or self.mf.xc.lower() == 'hf':
            self.ni = None
            self.omega, self.alpha, self.hyb = 0, 1.0, 1.0
            # self.omega, self.alpha, self.hyb = self.ni.rsh_and_hybrid_coeff("hf", self.mol.spin)
            self.hybrid = True
            self.rsh = False
            self.k_rsh = 0
            self.dm0 = self.mf.make_rdm1()
        else:
            self.ni = self.mf._numint
            self.ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            if mf.do_nlc():
                log.warn(mf, 'NLC functional found in DFT object.  Its second '
                            'derivative is not available. Its contribution is '
                            'not included in the response function.')
            self.omega, self.alpha, self.hyb = self.ni.rsh_and_hybrid_coeff(self.mf.xc, self.mol.spin)
            self.hybrid = self.ni.libxc.is_hybrid_xc(self.mf.xc)
            self.dm0 = self.mf.make_rdm1()
            if self.omega > 0:
                self.k_rsh = self.alpha - self.hyb
                self.rsh = True
        
            else:
                
                self.rsh = False
                self.k_rsh = 0

        self.J_rsh = None
        self.J = None

        self._build_isdf_intermediates()


    def _build_isdf_intermediates(self):
        """Automatically construct the ISDF interpolants and exact J matrices from the PySCF molecule object."""

        print('\n--- Starting Auto ISDF Decomposition ---')
        grids = self.mf.grids
        ni = getattr(self.mf, '_numint', dft.numint.NumInt())
        
        # Build grid data for all spin channels individually
        self.C_o = []
        self.C_v = []
        self.C_o_gga = []
        self.C_v_gga = []
        
        # Determine Ranks — isdf_naux_factor can be a scalar (same for phi and grad)
        # or a list/tuple [factor_phi, factor_grad]; factor_grad=0 skips grad pivots.
        _nf = self.isdf_naux_factor
        if hasattr(_nf, '__len__'):
            _nf_phi, _nf_grad = _nf[0], _nf[1]
        else:
            _nf_phi, _nf_grad = _nf, _nf
        n_rank_phi = int(_nf_phi * self.nmo)
        n_rank_grad = int(_nf_grad * self.nmo)
        _skip_grad = (n_rank_grad == 0) or self.isdf_skip_grad_pivots


        # We need a unified J kernel across spins (it's solely spatial)
        # We will decompose the spatial orbitals if Restricted, or alpha/beta if Unrestricted
        # For simplicity in this auto-builder, we process the spin channels independently 
        # but in a rigorous implementation, spin-unrestricted usually shares a single spatial ISDF basis.

        for s in range(self.nspin):
            
            nocc_s = self.nocc[s] if isinstance(self.nocc, list) else self.nocc
            orbs_s = self.mo_coeff[s]
            
            n_grid_total = grids.coords.shape[0]
            phi = np.zeros((self.nmo, n_grid_total))
            grad_phi = np.zeros((self.nmo, n_grid_total, 3))
            weights = np.zeros(n_grid_total)
            grid_coords = np.zeros((n_grid_total,3))
            
            nstart, nstop = 0, 0
            for ao, mask, weight, coords in ni.block_loop(self.mol, grids, self.mol.nao, 1, self.mf.max_memory):
                nstop += ao.shape[1]
                phi[:, nstart:nstop] = (ao[0] @ orbs_s).T
                grad_phi[:, nstart:nstop, :] = (ao[1:4] @ orbs_s).transpose(2, 1, 0)
                weights[nstart:nstop] = weight
                grid_coords[nstart:nstop] = coords
                nstart += ao.shape[1]
            
            del ao, mask, weight, coords
            if self.isdf_grid_rho_cutoff > 0:
                # TODO: needs to be integrated with wfxc in fxc porition
                raise NotImplementedError

                # self.grid_mask = mask_grid(phi, self.isdf_grid_rho_cutoff)
                # self.grid_mask = self.grid_mask | mask_grid(np.linalg.norm(grad_phi, axis = -1), self.isdf_grid_rho_cutoff)
                # phi, grad_phi = phi[:,self.grid_mask], grad_phi[:,self.grid_mask,:]
                # weights, grid_coords = weights[self.grid_mask], grid_coords[self.grid_mask]            
            
            t0 = time.time()

            # --- Decide streaming vs in-core BEFORE the decompose call (spin 0 only) ---
            # In streaming mode, xi_phi/xi_grad are written directly to HDF5 and
            # returned as None, so they never occupy RAM beyond isdf_decompose.
            if s == 0 and self.isdf_stream_path is not None:
                import h5py
                # Resolve HDF5 path (True → auto temp under /tmp)
                if self.isdf_stream_path is True:
                    import uuid as _uuid
                    stream_h5 = f"/tmp/isdf_stream_{_uuid.uuid4().hex[:8]}.h5"
                else:
                    stream_h5 = self.isdf_stream_path
                
                with h5py.File(stream_h5, 'w') as f:
                    f.create_dataset('phi', data=phi)
                    f.create_dataset('grad_phi', data=grad_phi)
                self.isdf_output_path = stream_h5.replace('.h5', '_out.h5')
                _gc = np.array(grid_coords) if self.isdf_backend == 'numpy' else jnp.array(grid_coords)
                _wt = np.array(weights)      if self.isdf_backend == 'numpy' else jnp.array(weights)
                pivots, phi_piv, grad_phi_piv = isdf_decompose_outcore(
                    stream_h5, self.isdf_output_path, n_rank_phi, n_rank_grad,
                    _gc, _wt, grid_batch_size=self.isdf_grid_batch_size,
                    rcond=self.isdf_rcond, backend=self.isdf_backend,
                    cd_sample_factor=self.isdf_cd_sample_factor,
                    cd_seed=self.isdf_cd_seed,
                    skip_grad_pivots=_skip_grad
                )
            else:
                stream_h5 = None  # sentinel: non-streaming

                phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots, _ = isdf_decompose(
                    jnp.array(phi), jnp.array(grad_phi), n_rank_phi, n_rank_grad,
                    jnp.array(weights), grid_batch_size=self.isdf_grid_batch_size,
                    is_incore=True, rcond=self.isdf_rcond
                )
            del phi, grad_phi

            gc.collect()
            t1 = time.time()
            print(f"ISDF decomposition for spin {s} took: {t1 - t0:.2f} s")
            pivot_coords = grid_coords[np.array(pivots)]

            C_val = np.array(phi_piv).T
            C_grad = np.array(grad_phi_piv).transpose(2, 1, 0)
            
            C_full = np.zeros((4, C_val.shape[0], self.nmo))
            C_full[0] = C_val
            C_full[1:4] = C_grad
            
            C_o_s = C_val[:, :nocc_s]
            C_v_s = C_val[:, nocc_s:]
            self.C_o.append(C_o_s)
            self.C_v.append(C_v_s)
            
            C_o_gga_s = C_full[:, :, :nocc_s]
            C_v_gga_s = C_full[:, :, nocc_s:]
            self.C_o_gga.append(C_o_gga_s)
            self.C_v_gga.append(C_v_gga_s)
            
            
            if s == 0:
                t0 = time.time()
                if stream_h5 is not None:
                    import h5py
                    # --- Streaming mode ---
                    # stream_h5 was resolved and isdf_decompose used is_incore=False,
                    # so xi_phi/xi_grad are None (written to HDF5).
                    # Remember the path so load_fxc_intermediates can stream from it.
                    self._isdf_h5_path = stream_h5
                    # del pivots, phi_piv, grad_phi_piv

                    # Build J-kernel: exact O(Ngrid^2) or floating-basis DF
                    if self.isdf_exact_J:
                        with h5py.File(self.isdf_output_path, 'r') as _f:
                            xi_phi_arr = _f['xi_phi'][:]
                        
                        if self.isdf_backend == 'jax':
                            print('Exact J for Streaming ISDF in JAX...')
                            _xi_jax = jnp.array(xi_phi_arr)
                            self.J = np.array(compute_J_munu(
                                _xi_jax, jnp.array(weights), jnp.array(grid_coords)
                            ))
                            if getattr(self, 'omega', 0.0) > 0:
                                self.J_rsh = np.array(compute_J_munu_lr(
                                    _xi_jax, jnp.array(weights), jnp.array(grid_coords),
                                    omega=self.omega
                                ))
                            else:
                                self.J_rsh = None
                            del _xi_jax
                        else:
                            print('Exact J for Streaming ISDF in NUMPY...')
                            self.J = compute_J_munu_numpy(
                                xi_phi_arr, np.array(weights), np.array(grid_coords)
                            )
                            if getattr(self, 'omega', 0.0) > 0:
                                self.J_rsh = compute_J_munu_lr_numpy(
                                    xi_phi_arr, np.array(weights), np.array(grid_coords),
                                    omega=self.omega
                                )
                            else:
                                self.J_rsh = None
                        del xi_phi_arr
                    else:
                        self.J = compute_ISDF_J_kernels_DF_streaming(
                            self.isdf_output_path, weights, grid_coords, pivot_coords,
                            gammas=self.isdf_gammas, batch_size=self.isdf_stream_batch_size,
                            backend=self.isdf_backend
                        )
                        if getattr(self, 'omega', 0.0) > 0:
                            self.J_rsh = compute_ISDF_J_kernels_DF_streaming(
                                self.isdf_output_path, weights, grid_coords, pivot_coords,
                                gammas=self.isdf_gammas, omega=self.omega,
                                batch_size=self.isdf_stream_batch_size,
                                backend=self.isdf_backend
                            )
                        else:
                            self.J_rsh = None
                        

                    # If there is no fxc to compress (pure HF), we can delete the
                    # HDF5 immediately since load_fxc_intermediates will never be called.
                    if getattr(self, 'ni', None) is None:
                        if os.path.exists(stream_h5):
                            os.remove(stream_h5)
                        self._isdf_h5_path = None
                else:
                    # --- Non-streaming (original) mode ---
                    # Save xi_phi and xi_grad for later fxc kernel compression (force float64)
                    self.xi_phi = np.array(xi_phi, dtype=np.float64)
                    self.xi_grad = np.array(xi_grad, dtype=np.float64)
                    del xi_phi, xi_grad, grad_phi_piv, pivots
                    

                    # Build J-kernel: exact O(Ngrid^2) or floating-basis DF
                    if self.isdf_exact_J:
                        if self.isdf_backend == 'jax':
                            _xi_jax = jnp.array(self.xi_phi)
                            self.J = np.array(compute_J_munu(
                                _xi_jax, jnp.array(weights), jnp.array(grid_coords)
                            ))
                            if getattr(self, 'omega', 0.0) > 0:
                                self.J_rsh = np.array(compute_J_munu_lr(
                                    _xi_jax, jnp.array(weights), jnp.array(grid_coords),
                                    omega=self.omega
                                ))
                            else:
                                self.J_rsh = None
                            del _xi_jax
                        else:
                            self.J = compute_J_munu_numpy(
                                self.xi_phi, np.array(weights), np.array(grid_coords)
                            )
                            if getattr(self, 'omega', 0.0) > 0:
                                self.J_rsh = compute_J_munu_lr_numpy(
                                    self.xi_phi, np.array(weights), np.array(grid_coords),
                                    omega=self.omega
                                )
                            else:
                                self.J_rsh = None
                    else:
                        self.J = np.array(compute_ISDF_J_kernels_DF_incore(
                            self.xi_phi, weights, grid_coords, pivot_coords, gammas=self.isdf_gammas
                        ))
                        if getattr(self, 'omega', 0.0) > 0:
                            self.J_rsh = np.array(compute_ISDF_J_kernels_DF_incore(
                                self.xi_phi, weights, grid_coords, pivot_coords,
                                gammas=self.isdf_gammas, omega=self.omega
                            ))
                        else:
                            self.J_rsh = None

                
                t1 = time.time()
                self.isdf_naux = self.J.shape[0]
                print(f"Analytical ISDF J-kernel build(s) took: {t1 - t0:.2f} s, final isdf_naux: {self.isdf_naux}")
            else:
                raise NotImplementedError
                 
        print('--- Finished Auto ISDF Decomposition ---\n')
        
    def dump_flags(self):
        log = lib.logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        nvir = [(self.nmo - self.nocc[i]) for i in range(self.nspin)]
        dim = [(self.nocc[i] * nvir[i]) for i in range(self.nspin)]
        log.info('multiplicity = %s', self.multi)
        log.info('nmo = %s', self.nmo)
        log.info('nocc = %s', self.nocc[0] if self.nspin == 1 else self.nocc)
        log.info('nvir = %s', nvir[0] if self.nspin == 1 else nvir)
        log.info('occ-vir dimension = %s', dim[0] if self.nspin == 1 else dim)
        if self.nspin == 2:
            log.info('tddft full dimension = %s', dim[0] + dim[1])
        log.info('Tamm-Dancoff approximation = %s', self.TDA)
        log.info('number of roots = %d', self.nroot)
        log.info('trial vector = %s', self.trial)
        if self.trial == 'subspace':
            log.info('subspace nocc = %d nvir = %d', self.nocc_sub, self.nvir_sub)
        log.info('max subspace size = %d', self.max_vec)
        log.info('max iteration = %s', self.max_iter)
        log.info('convergence tolerance = %s', self.residue_thresh)
        log.info(f'omega, alpha, hyb: {self.omega}, {self.alpha}, {self.hyb}')
        log.info('')
        return

    def check_memory(self):
        pass
        return

    def load_fxc_intermediates(self):
        """From pyscf.tdscf.rks
        """
        # self.mf.grids.atom_grid = (10, 50)  # (radial, angular), e.g. (10, 50) is very coarse
        self.mf.grids.build(with_non0tab=False)  # sometimes helps avoid excess caching
        self.ni = self.mf._numint
        self.xctype = self.ni._xc_type(self.mf.xc)

        dm0 = self.mf.make_rdm1(self.mo_coeff[0], self.mo_occ[0])
        make_rho = self.ni._gen_rho_evaluator(self.mf.mol, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.mf.max_memory*.4-mem_now)
        orbv = self.mo_coeff[0][:,self.nocc[0]:]
        orbo = self.mo_coeff[0][:,:self.nocc[0]]
        
        if self.xctype == 'LDA':
            t0 = time.time()
            ao_deriv = 0
            n_grid_lda = self.mf.grids.coords.shape[0]
            wfxc = np.zeros(n_grid_lda)       
            nstart, nstop = 0, 0
            for ao, mask, weight, coords \
                    in self.ni.block_loop(self.mf.mol, self.mf.grids, self.mol.nao, ao_deriv, max_memory):
                nstop += ao.shape[0]
                rho = make_rho(0, ao, mask, self.xctype)
                fxc = self.mf._numint.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=self.xctype)[2]
                wfxc[nstart:nstop] = fxc[0,0] * weight

                nstart += ao.shape[0]
            t1 = time.time()
            print(f'LDA fxc kernel calculated in real-space, took: {t1 - t0:.2f} s"')
            # Compress wfxc into ISDF interpolant space: (ngrid,) -> (naux, naux)
            if self._isdf_h5_path is not None:
                # Streaming: read xi_phi from HDF5 in batches
                self.wfxc = compress_isdf_lda_kernel_streaming(
                    self.isdf_output_path, wfxc, batch_size=self.isdf_stream_batch_size,
                    backend=self.isdf_backend
                )
                # HDF5 no longer needed — delete it
                if os.path.exists(self._isdf_h5_path):
                    os.remove(self._isdf_h5_path)
                self._isdf_h5_path = None
            else:
                self.wfxc = compress_isdf_lda_kernel(self.xi_phi, wfxc)
                del self.xi_phi, self.xi_grad
            print(f'ISDF LDA fxc kernel compressed: {wfxc.shape} -> {self.wfxc.shape}')
            return

        elif self.xctype == 'GGA':
            t0 = time.time()
            ao_deriv = 1
            wfxc = np.zeros((4, 4, self.mf.grids.coords.shape[0]))       
            nstart, nstop = 0, 0
            for ao, mask, weight, coords \
                    in self.ni.block_loop(self.mf.mol, self.mf.grids, self.mol.nao, ao_deriv, max_memory):
                nstop += ao.shape[1]
                rho = make_rho(0, ao, mask, self.xctype)
                if self.multi == 't':
                    rho *= 0.5
                    rho = np.repeat(rho[np.newaxis], 2, axis=0)
                    fxc = self.ni.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=self.xctype)[2] 
                    fxc = fxc[0,:,0] - fxc[0,:,1]
                    wfxc[:, :, nstart:nstop] = fxc*weight
                
                else:
                    fxc = self.ni.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=self.xctype)[2] 
                    wfxc[:, :, nstart:nstop] = fxc*weight
                del rho
                nstart += ao.shape[1]
                del ao, mask, weight, coords
            t1 = time.time()
            print(f'GGA fxc kernel calculated in real-space, took: {t1 - t0:.2f} s"')
            # Compress wfxc into ISDF interpolant space: (4, 4, ngrid) -> (4, 4, naux, naux)
            # wfxc is (x, y, r) from eval_xc_eff; compress_isdf_gga_kernel expects (y, x, r)
            t0 = time.time()
            # Transpose to (y, x, r) convention before compression
            wfxc_yx = wfxc.transpose(1, 0, 2)  # (4, 4, ngrid)
            if self._isdf_h5_path is not None:
                # Streaming: read xi_phi/xi_grad from HDF5 in batches
                self.wfxc = compress_isdf_gga_kernel_streaming(
                    self.isdf_output_path, wfxc_yx, batch_size=self.isdf_stream_batch_size,
                    backend=self.isdf_backend
                )
                # HDF5 no longer needed — delete it
                if os.path.exists(self._isdf_h5_path):
                    os.remove(self._isdf_h5_path)
                self._isdf_h5_path = None
            else:
                self.wfxc = compress_isdf_gga_kernel(self.xi_phi, self.xi_grad, wfxc_yx)
                del self.xi_phi, self.xi_grad
            t1 = time.time()
            print(f'ISDF GGA fxc kernel compressed: {wfxc.shape} -> {self.wfxc.shape}, took: {t1 - t0:.2f} s"')
            return
                
        elif self.xctype == 'HF':
            print('\n : no fxc needed for Hartree-Fock%s')

        elif self.xctype == 'NLC':
            raise NotImplementedError

        elif self.xctype == 'MGGA':
            raise NotImplementedError

    def kernel(self, multi, e_min=0.0, delta=0.0, **kwargs):
        # check spin and multiplicity
        assert isinstance(multi, str)
        multi = multi[0].lower()
        assert (self.nspin == 1 and (multi == 's' or multi == 't')) or (self.nspin == 2 and multi == 'u')
        self.multi = multi

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.check_memory()
        exci, X, Y = tddft_davidson(tddft=self, multi=multi, e_min=e_min, delta=delta, **kwargs)
        lib.logger.timer(self, 'tddft', *cput0)
        return exci, X, Y

    def full_diagonalization(self, multi, subset_by_value = None):
        cput0 = (time.process_time(), time.perf_counter())
        print('\ntddft full diagonalization: %s', multi)
        self.multi = multi

        # set nroot as full dimension for analysis
        nvir = [(self.nmo - self.nocc[i]) for i in range(self.nspin)]
        dim = [(self.nocc[i] * nvir[i]) for i in range(self.nspin)]
        self.nroot = dim[0] + dim[1] if self.nspin == 2 else dim[0]

        # A+B, A-B, X+Y, X-Y
        mem = (self.nroot * self.nroot * 4) * 8
        print('tddft needs at least %.1f GB memory.', mem / 1.0e9)

        if self.ni is not None:
            
            if self.nspin == 1 and self.multi.lower() != 't': 
                pass

            elif self.nspin == 2:
                raise NotImplementedError
                
            elif self.multi.lower() == 't':
                raise NotImplementedError

            def numint_fn(fxc_iajb, scale):
                """From pyscf.tdscf.rks
                """
                # self.mf.grids.atom_grid = (10, 50)  # (radial, angular), e.g. (10, 50) is very coarse
                self.mf.grids.build(with_non0tab=False)  # sometimes helps avoid excess caching
                self.ni = self.mf._numint
                xctype = self.ni._xc_type(self.mf.xc)
                dm0 = self.mf.make_rdm1(self.mo_coeff[0], self.mo_occ[0])
                make_rho = self.ni._gen_rho_evaluator(self.mf.mol, dm0, hermi=1, with_lapl=False)[0]
                mem_now = lib.current_memory()[0]
                max_memory = max(2000, self.mf.max_memory*.4-mem_now)
                print('full diagonalization max memory for fxc_iajb: ', max_memory)
                orbv = self.mo_coeff[0][:,self.nocc[0]:]
                orbo = self.mo_coeff[0][:,:self.nocc[0]]
                
                if self.xctype == 'LDA':
                    if getattr(self, 'wfxc', None) is not None:
                        # Build dense fxc_iajb = C_ov^T @ V_xc @ C_ov
                        # where C_ov[mu, ia] = C_o_i^mu * C_v_a^mu
                        C_o_s, C_v_s = self.C_o[0], self.C_v[0]
                        C_ov = (C_o_s[:, :, None] * C_v_s[:, None, :]).reshape(C_o_s.shape[0], -1)  # (naux, nocc*nvir)
                        dense_fxc = C_ov.T @ self.wfxc @ C_ov  # (ndim, ndim)
                        fxc_iajb += dense_fxc * scale
                    else:
                        raise NotImplementedError("ISDF fxc compression must be precomputed via load_fxc_intermediates")

                elif self.xctype == 'GGA':
                    if getattr(self, 'wfxc', None) is not None:
                        # Build dense fxc_iajb from ISDF compressed GGA kernel
                        # rho_ov_x^{mu,ia} follows the product rule:
                        #   x=0: C_o_i^mu * C_v_a^mu
                        #   x>0: C_o_x_i^mu * C_v_a^mu + C_o_i^mu * C_v_x_a^mu
                        C_o_s = self.C_o_gga[0]   # (4, naux, nocc)
                        C_v_s = self.C_v_gga[0]   # (4, naux, nvir)
                        nocc_s, nvir_s = C_o_s.shape[2], C_v_s.shape[2]
                        naux = C_o_s.shape[1]
                        
                        # Build rho_ov_x: (4, naux, nocc*nvir)
                        rho_ov = np.zeros((4, naux, nocc_s * nvir_s))
                        # x=0 term
                        rho_ov[0] = (C_o_s[0, :, :, None] * C_v_s[0, :, None, :]).reshape(naux, -1)
                        # x>0 terms (product rule)
                        for x in range(1, 4):
                            rho_ov[x] = (C_o_s[x, :, :, None] * C_v_s[0, :, None, :]).reshape(naux, -1) \
                                      + (C_o_s[0, :, :, None] * C_v_s[x, :, None, :]).reshape(naux, -1)
                        
                        # fxc_iajb = sum_{x,y} rho_ov_y^T @ V_fxc[y,x] @ rho_ov_x
                        dense_fxc = np.zeros((nocc_s * nvir_s, nocc_s * nvir_s))
                        for y in range(4):
                            for x in range(4):
                                dense_fxc += rho_ov[y].T @ self.wfxc[y, x] @ rho_ov[x]
                        

                        if self.multi.lower() == 't':
                            dense_fxc *= 0.5
                        fxc_iajb += dense_fxc * scale
                    else:
                        raise NotImplementedError("ISDF fxc compression must be precomputed via load_fxc_intermediates")

                elif self.xctype == 'HF':
                    pass

                elif self.xctype == 'NLC':
                    pass # Processed later

                elif xctype == 'MGGA':
                    ao_deriv = 1
                    for ao, mask, weight, coords \
                            in self.ni.block_loop(self.mf.mol, self.mf.grids, self.mol.nao, ao_deriv, max_memory):
                        rho = make_rho(0, ao, mask, xctype)
                        fxc = self.ni.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=xctype)[2]
                        wfxc = fxc * weight
                        rho_o = einsum('xrp,pi->xri', ao, orbo)
                        rho_v = einsum('xrp,pi->xri', ao, orbv)
                        rho_ov = einsum('xri,ra->xria', rho_o, rho_v[0])
                        rho_ov[1:4] += einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                        tau_ov = einsum('xri,xra->ria', rho_o[1:4], rho_v[1:4]) * .5
                        rho_ov = np.vstack([rho_ov, tau_ov[np.newaxis]])
                        w_ov = einsum('xyr,xria->yria', wfxc, rho_ov)
                        iajb = einsum('xria,xrjb->iajb', w_ov, rho_ov)
                        fxc_iajb += iajb.reshape((self.nocc[0]*nvir[0],self.nocc[0]*nvir[0]))*scale
                return fxc_iajb

                            
        else:
            numint_fn = None

        self.exci, self.X_vec, self.Y_vec = tddft_full_diagonalization(
            multi=multi, nocc=self.nocc, mo_energy=self.mo_energy, 
            C_o=self.C_o, C_v=self.C_v, J=self.J,
            J_rsh=self.J_rsh, k_rsh=self.k_rsh, hyb_coeff=self.hyb, 
            TDA=self.TDA, ni_fn=numint_fn, subset_by_value=subset_by_value
        )
        lib.logger.timer(self, 'tddft full diagonalization', *cput0)
        return self.exci, self.X_vec, self.Y_vec

if __name__ == '__main__':
    from pyscf import gto, dft, scf
    
    # Testing cholesky decomposition of fxc
    mol = gto.Mole()
    mol.atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161'''
    mol.basis = 'def2svpd'
    mol.build()
    
    mf = scf.RKS(mol)
    mf.xc = 'WB97XD'
    mf.kernel()
    
    pyscf_td = mf.TDDFT()
    pyscf_td.singlet = True
    pyscf_td.nstates = 10
    pyscf_ref = np.sort(pyscf_td.kernel()[0])
    
    print('---values in eV---')
    print('pyscf exci:', pyscf_ref*HARTREE2EV)

    mytd = TDDFT(mf = mf, nroot = 10, max_vec = 150, residue_thresh = 1.0e-8, isdf_rcond = 1e-14, isdf_grid_level = 2, isdf_naux_factor = [8, 4], isdf_gammas = [0.1, 0.4], isdf_stream_path = './my_isdf_tmp.h5', isdf_exact_J=False, isdf_backend = 'jax', isdf_grid_batch_size = 8192, isdf_cd_sample_factor = 500)
    exci_new = np.sort(mytd.kernel(multi = 's')[0])
    
    print('Davidson exci:', exci_new*HARTREE2EV)
    print('Davidson mae vs PySCF:', HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))

    # Run full diagonalization as a sanity check
    # from pyscf.dft.libxc import xc_type
    # mytd.xctype = xc_type(mf.xc)
    # exci_full = np.sort(mytd.full_diagonalization(multi = 's')[0])

    
    # print('Full diag S1-S10:', exci_full[:10]*HARTREE2EV)
    # print('Full diag mae vs PySCF:', HARTREE2EV*np.mean(np.abs(exci_full[:10]-pyscf_ref)))

    