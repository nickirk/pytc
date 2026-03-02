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

def compute_dynamic_alphas(pivots, gamma=0.8):
    """
    Determines optimal Gaussian exponents based on local pivot density.
    
    Args:
        pivots: (Naux, 3) array of pivot coordinates.
        gamma: Coverage factor. Higher = narrower Gaussians.
        
    Returns:
        alphas: (Naux,) array of exponents.
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
    alphas = gamma / (h_i**2)
    
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
        alphas = np.full(Naux, alphas)
        
    # 1. Define Ghost Atoms at the pivot coordinates
    # We name them X0, X1, X2... so we can assign a unique alpha to each if needed
    ghost_atoms = [(f'X{i}', coord) for i, coord in enumerate(pivots)]
    
    # 2. Define the Custom Basis Dictionary
    # PySCF basis format: { 'AtomSymbol': [[ angular_momentum, (exponent, contraction_coeff) ]] }
    # l=0 is an s-type function. We use an uncontracted coefficient of 1.0.
    custom_basis = {
        f'X{i}': [[0, (alphas[i], 1.0)]] for i in range(Naux)
    }
    
    # 3. Build the Auxiliary PySCF Object
    aux_mol = pyscf.gto.M(
        atom=ghost_atoms,
        basis=custom_basis,
        charge=0,
        spin=0,
        unit=unit
    )
    
    return aux_mol

def compute_ISDF_J_kernels_DF(xi_phi, weights, coords, pivots, gamma=0.8, omega=0, rcond=1e-12):
    """
    Computes a single J_munu (either standard Coulomb or range-separated) by 
    projecting ISDF interpolants onto a floating Gaussian auxiliary basis and using 
    analytical integrals.
    
    Uses an SVD-based pseudo-inverse for robust projection.
    """
    from scipy import linalg
    import time
    
    label = "Standard Coulomb" if omega == 0 else f"Range-Separated (omega={omega})"
    print(f"\nBuilding Analytical DF J-Kernel ({label}) with floating basis (gamma={gamma})...")
    t0 = time.time()
    
    # 1. Create Auxiliary Molecule
    t_start = time.time()
    alphas = compute_dynamic_alphas(pivots, gamma=gamma)
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

def compress_isdf_lda_kernel(xi, wfxc):
    """
    Compress the N_grid x N_grid LDA kernel into N_aux x N_aux.
    
    xi: (naux, ngrid) - ISDF interpolation functions $\zeta_{\mu}(\mathbf{r})$
    wfxc: (ngrid,) - Grid integrated fxc kernel
    
    Returns:
    V_xc: (naux, naux)
    """
    # V^{\nu \mu} = \sum_{r} \zeta_{\nu}(r) f(r) \zeta_{\mu}(r)
    # return einsum('nr,r,mr->nm', xi, wfxc, xi)
    return (xi * wfxc[None, :]) @ xi.T  # Hadamard + DGEMM


def compress_isdf_gga_kernel(xi_phi, xi_grad, wfxc):
    """
    Compress the GGA fxc kernel into the ISDF auxiliary basis using both density and gradient interpolators.
    xi_phi: (naux, ngrid)
    xi_grad: (naux, ngrid, 3) 
    wfxc: (4, 4, ngrid) containing w(r) * f_{xy}(r)
    """
    naux, ngrid = xi_phi.shape
    
    # Pack interpolators into a single array (4, naux, ngrid)
    xi_full = np.zeros((4, naux, ngrid))
    xi_full[0] = xi_phi
    xi_full[1:4] = np.transpose(xi_grad, (2, 0, 1))
    
    # Contract: V_{yx}^{nu, mu} = sum_r xi_{nu, y}(r) * wfxc_{yx}(r) * xi_{mu, x}(r)
    V_fxc = np.zeros((4, 4, naux, naux))
    for y in range(4):
        for x in range(4):
            # Hadamard product over grid, then DGEMM
            tmp = xi_full[y] * wfxc[y, x][None, :]
            V_fxc[y, x] = tmp @ xi_full[x].T
            
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

def _get_oscillator_strength(multi, exci, X_vec, Y_vec, mo_coeff, nocc, mol):
    """Get transition dipoles and oscillator strengths.

    Args:
        multi (char): multiplicity. "s"=singlet, "t"=triplet, "u"=unrestricted.
        exci (double array): excitation energy.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
        mo_coeff (double ndarray): coefficient from AO to MO.
        nocc (int array): number of occupied orbitals.
        mol (pyscf.gto.mole.Mole): Mole object for generating dipole matrix.

    Returns:
        dipole (double ndarray): transition dipoles of all excitations.
        oscillator_strength (double array): oscillator strengths of all excitations.
    """
    nspin, nao, nmo = mo_coeff.shape
    nroot = X_vec[0].shape[0]

    dipole = np.zeros(shape=[3, nroot], dtype=np.double, order='F')
    oscillator_strength = np.zeros(shape=[nroot], dtype=np.double)

    # tddft is blind to triplet oscillator strength
    if multi == 't':
        return dipole, oscillator_strength

    with mol.with_common_orig((0, 0, 0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)

    # Transform AO dipole integrals to MO basis
    mo_dip = [mo_coeff[s][:, : nocc[s]].T @ ao_dip @ mo_coeff[s][:, nocc[s] :] for s in range(nspin)]

    for j in range(nroot):
        for s in range(nspin):
            dipole[:, j] += einsum('ia,xia->x', X_vec[s][j], mo_dip[s]) + einsum(
                'ia,xia->x', Y_vec[s][j], mo_dip[s]
            )

    if nspin == 1:
        dipole *= np.sqrt(2)

    oscillator_strength = (2 / 3) * exci * np.sum(dipole**2, axis=0)

    return dipole, oscillator_strength


def _get_spin_square(nocc, X_vec, Y_vec, mo_coeff, ovlp):
    """Get <S2> expectation value.

    Args:
        nocc (int array): number of occupied orbitals.
        X_vec (double ndarray): X block of eigenvector (excitation).
        Y_vec (double ndarray): Y block of eigenvector (de-excitation).
        mo_coeff (double ndarray): coefficient from AO to MO.
        ovlp (double ndarray): overlap matrix.

    Returns:
        s2 (double array): <S2> expectation value of excitations.
    """
    nroot = X_vec[0].shape[0]
    ab_ovlp = mo_coeff[0].T @ ovlp @ mo_coeff[1]
    s2 = np.zeros(shape=[nroot], dtype=np.double)
    s2[:] = nocc[0] - (nocc[0] - nocc[1]) / 2.0 + ((nocc[0] - nocc[1]) / 2.0) ** 2
    for iroot in range(nroot):
        # alpha excitation ket
        # a alpha and j beta exchange: alpha excitation bra
        s2[iroot] -= einsum(
            'ia,ib,aj,bj->',
            X_vec[0][iroot] + Y_vec[0][iroot],
            X_vec[0][iroot] - Y_vec[0][iroot],
            ab_ovlp[nocc[0] :, : nocc[1]],
            ab_ovlp[nocc[0] :, : nocc[1]],
        )
        # a alpha and j beta exchange: beta excitation bra
        s2[iroot] -= einsum(
            'ia,jb,ij,ab->',
            X_vec[0][iroot] + Y_vec[0][iroot],
            X_vec[1][iroot] - Y_vec[1][iroot],
            ab_ovlp[: nocc[0], : nocc[1]],
            ab_ovlp[nocc[0] :, nocc[1] :],
        )
        # i alpha and j beta exchange: same alpha excitation bra
        s2[iroot] -= einsum(
            'ia,ia,jk->',
            X_vec[0][iroot] + Y_vec[0][iroot],
            X_vec[0][iroot] - Y_vec[0][iroot],
            ab_ovlp[: nocc[0], : nocc[1]] ** 2,
        )
        s2[iroot] += einsum(
            'ia,ia,ik->',
            X_vec[0][iroot] + Y_vec[0][iroot],
            X_vec[0][iroot] - Y_vec[0][iroot],
            ab_ovlp[: nocc[0], : nocc[1]] ** 2,
        )
        # beta excitation ket
        # i alpha and b beta exchange: beta excitation bra
        s2[iroot] -= einsum(
            'ia,ib,ja,jb->',
            X_vec[1][iroot] + Y_vec[1][iroot],
            X_vec[1][iroot] - Y_vec[1][iroot],
            ab_ovlp[: nocc[0], nocc[1] :],
            ab_ovlp[: nocc[0], nocc[1] :],
        )
        # i alpha and b beta exchange: alpha excitation bra
        s2[iroot] -= einsum(
            'ia,jb,ji,ba->',
            X_vec[1][iroot] + Y_vec[1][iroot],
            X_vec[0][iroot] - Y_vec[0][iroot],
            ab_ovlp[: nocc[0], : nocc[1]],
            ab_ovlp[nocc[0] :, nocc[1] :],
        )
        # i alpha and j beta exchange: same alpha excitation bra
        s2[iroot] -= einsum(
            'ia,ia,jk->',
            X_vec[1][iroot] + Y_vec[1][iroot],
            X_vec[1][iroot] - Y_vec[1][iroot],
            ab_ovlp[: nocc[0], : nocc[1]] ** 2,
        )
        s2[iroot] += einsum(
            'ia,ia,ji->',
            X_vec[1][iroot] + Y_vec[1][iroot],
            X_vec[1][iroot] - Y_vec[1][iroot],
            ab_ovlp[: nocc[0], : nocc[1]] ** 2,
        )

    return s2


class TDDFT(lib.StreamObject):
    def __init__(
        self,
        # initialize with a GW object
        mf=None,
        # initialize with nocc, mo_energy, C_o, C_v, J
        nocc=None,
        isdf_rcond=1e-6,
        isdf_grid_level=3,
        isdf_naux_factor=8,
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
        import time
        from pyscf import dft
        from pytc.df import isdf_decompose as isdf_decompose_jax
        # from isdf_coulomb_exchange import compute_J_munu, compute_J_munu_lr
        import jax.numpy as jnp

        print('\n--- Starting Auto ISDF Decomposition ---')
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = getattr(self, 'isdf_grid_level', 3)
        grids.build()
        
        ni = getattr(self.mf, '_numint', dft.numint.NumInt())
        
        # Build grid data for all spin channels individually
        self.C_o = []
        self.C_v = []
        self.C_o_gga = []
        self.C_v_gga = []
        
        # Determine Ranks (Using default conservative heuristics, customize as needed)
        n_rank_phi = self.isdf_naux_factor * self.nmo
        n_rank_grad = self.isdf_naux_factor * self.nmo

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
            
            nstart, nstop = 0, 0
            for ao, mask, weight, coords in ni.block_loop(self.mol, grids, self.mol.nao, 1, self.mf.max_memory):
                nstop += ao.shape[1]
                phi[:, nstart:nstop] = (ao[0] @ orbs_s).T
                grad_phi[:, nstart:nstop, :] = (ao[1:4] @ orbs_s).transpose(2, 1, 0)
                weights[nstart:nstop] = weight
                nstart += ao.shape[1]
            
            t0 = time.time()
            phi_piv, xi_phi, grad_phi_piv, xi_grad, pivots, _ = isdf_decompose_jax(
                jnp.array(phi), jnp.array(grad_phi), n_rank_phi, n_rank_grad,
                jnp.array(weights), grid_batch_size=8192, is_incore=True, rcond=self.isdf_rcond
            )
            t1 = time.time()
            print(f"ISDF decomposition for spin {s} took: {t1 - t0:.2f} s")
            
            pivot_coords = grids.coords[np.array(pivots)]
            
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
                # Save xi_phi and xi_grad for later fxc kernel compression (force float64)
                self.xi_phi = np.array(xi_phi, dtype=np.float64)
                self.xi_grad = np.array(xi_grad, dtype=np.float64)
                
                # Build the Analytical J-Kernels (DF)
                t0 = time.time()
                self.J = np.array(compute_ISDF_J_kernels_DF(
                    self.xi_phi, weights, grids.coords, pivot_coords, gamma=0.6
                ))
                
                if getattr(self, 'omega', 0.0) > 0:
                    self.J_rsh = np.array(compute_ISDF_J_kernels_DF(
                        self.xi_phi, weights, grids.coords, pivot_coords, gamma=0.4, omega=self.omega
                    ))
                else:
                    self.J_rsh = None
                
                t1 = time.time()
                print(f"Analytical ISDF J-kernel build(s) took: {t1 - t0:.2f} s")
                 
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
            ao_deriv = 0
            wfxc = np.zeros((self.mf.grids.coords.shape[0]))       
            nstart, nstop = 0, 0
            for ao, mask, weight, coords \
                    in self.ni.block_loop(self.mf.mol, self.mf.grids, self.mol.nao, ao_deriv, max_memory):
                nstop += ao.shape[0]
                rho = make_rho(0, ao, mask, self.xctype)
                fxc = self.mf._numint.eval_xc_eff(self.mf.xc, rho, deriv=2, xctype=self.xctype)[2]
                wfxc[nstart:nstop] = fxc[0,0] * weight

                nstart += ao.shape[0]

            # Compress wfxc into ISDF interpolant space: (ngrid,) -> (naux, naux)
            self.wfxc = compress_isdf_lda_kernel(self.xi_phi, wfxc)
            print(f'ISDF LDA fxc kernel compressed: {wfxc.shape} -> {self.wfxc.shape}')
            return

        elif self.xctype == 'GGA':
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

            # Compress wfxc into ISDF interpolant space: (4, 4, ngrid) -> (4, 4, naux, naux)
            # wfxc is (x, y, r) from eval_xc_eff; compress_isdf_gga_kernel expects (y, x, r)
            self.wfxc = compress_isdf_gga_kernel(self.xi_phi, self.xi_grad, wfxc.transpose(1, 0, 2))
            print(f'ISDF GGA fxc kernel compressed: {wfxc.shape} -> {self.wfxc.shape}')
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

    def analyze(self, thresh=0.1, oscillator=True, s2=True, e_min=0.0):
        """Analyze excitations.

        Args:
            thresh (float, optional): threshold to print dominant component. Defaults to 0.1.
            oscillator (bool, optional): calculate oscillator strength. Defaults to True.
            s2 (bool, optional): calculate <S2> expectation value. Defaults to True.
            e_min (float, optional): minimum excitation energy to analyze. Defaults to 0.0.

        Returns:
            all_data (list): list of dictionaries containing results of analysis.
        """
        multi = self.multi
        nspin = self.nspin
        nmo = self.nmo
        nocc = self.nocc

        emin_index = np.searchsorted(self.exci, e_min, side='left')
        exci = self.exci[emin_index:]

        X_vec = [X_vec_s[emin_index:] for X_vec_s in self.X_vec]
        Y_vec = [Y_vec_s[emin_index:] for Y_vec_s in self.Y_vec]
        nvir = [(nmo - nocc[i]) for i in range(nspin)]

        if oscillator is True:
            dipole, oscillator_strength = _get_oscillator_strength(
                multi=multi, exci=exci, X_vec=X_vec, Y_vec=Y_vec, mo_coeff=self.mo_coeff, nocc=nocc, mol=self.mol
            )

        if s2 is True and nspin == 2:
            s2 = _get_spin_square(nocc=nocc, X_vec=X_vec, Y_vec=Y_vec, mo_coeff=self.mo_coeff, ovlp=self.mf.get_ovlp())

        all_data = []

        print('-' * 55)
        if multi == 's':
            print('restricted singlet tddft')
        elif multi == 't':
            print('restricted triplet tddft')
        elif multi == 'u':
            print('unrestricted tddft')
        for r in range(exci.size):
            this_datum = {
                'excited_state': r + 1,
                'excitation_energy': float(exci[r]),
                'excitation_energy_ev': float(exci[r] * HARTREE2EV),
            }
            print('-' * 55)
            print('excited state: %-d' % (r + 1))
            print('excitation energy:   %15.8f   AU   %15.8f   eV' % (exci[r], exci[r] * HARTREE2EV))
            if multi == 's':
                if oscillator is True:
                    print('spin allowed, oscillator strength:   %15.8f   AU' % oscillator_strength[r])
                    print(
                        'transition dipole: x =  %15.6f  , y =  %15.6f  , z =  %15.6f'
                        % (dipole[0][r], dipole[1][r], dipole[2][r])
                    )
                    this_datum['oscillator_strength'] = float(oscillator_strength[r])
                    this_datum['transition_dipole'] = (float(dipole[0][r]), float(dipole[1][r]), float(dipole[2][r]))
            elif multi == 't':
                if oscillator is True:
                    print('spin forbidden, oscillator strength and transition dipoles are not defined')
            elif multi == 'u':
                if s2 is True:
                    print('<S^2> =    %.6f', s2[r])
                if oscillator is True:
                    print('oscillator strength:   %15.8f   AU' % oscillator_strength[r])
                    print(
                        'transition dipole: x =  %15.6f  , y =  %15.6f  , z =  %15.6f'
                        % (dipole[0][r], dipole[1][r], dipole[2][r])
                    )
                    this_datum['s2'] = s2[r]
                    this_datum['oscillator_strength'] = float(oscillator_strength[r])
                    this_datum['transition_dipole'] = (float(dipole[0][r]), float(dipole[1][r]), float(dipole[2][r]))

            def print_component(comp, with_spin=False):
                if not with_spin:
                    print(f"{comp['i']:5} -> {comp['a']:5}, {comp['weight']:15f}, {comp['type']}")
                else:
                    print(
                        f"{comp['i']:5} -> {comp['a']:5}, spin {comp['spin']}, {comp['weight']:15f}, {comp['type']}"
                    )

            this_datum_components = []
            print('dominant component')
            if nspin == 1:
                for i in range(nocc[0]):
                    for a in range(nvir[0]):
                        if abs(X_vec[0][r][i][a]) > thresh:
                            comp = {
                                    'i': i + 1,
                                    'a': int(a + nocc[0] + 1),
                                    'spin': 0,
                                    'weight': float(X_vec[0][r][i][a]),
                                    'type': 'X',
                            }
                            this_datum_components.append(comp)
                            print_component(comp)
                        if abs(Y_vec[0][r][i][a]) > thresh:
                            comp = {
                                    'i': i + 1,
                                    'a': int(a + nocc[0] + 1),
                                    'spin': 0,
                                    'weight': float(Y_vec[0][r][i][a]),
                                    'type': 'Y',
                            }
                            this_datum_components.append(comp)
                            print_component(comp)
            else:
                for s in range(nspin):
                    spin = 'a' if s == 0 else 'b'
                    for i in range(nocc[s]):
                        for a in range(nvir[s]):
                            if abs(X_vec[s][r][i][a]) > thresh:
                                comp = {
                                        'i': i + 1,
                                        'a': int(a + nocc[s] + 1),
                                        'spin': s,
                                        'weight': float(X_vec[s][r][i][a]),
                                        'type': 'X',
                                }
                                this_datum_components.append(comp)
                                print_component(comp, with_spin=True)
                            if abs(Y_vec[s][r][i][a]) > thresh:
                                comp = {
                                        'i': i + 1,
                                        'a': int(a + nocc[s] + 1),
                                        'spin': s,
                                        'weight': float(X_vec[s][r][i][a]),
                                        'type': 'Y',
                                }
                                this_datum_components.append(comp)
                                print_component(comp, with_spin=True)
            this_datum['components'] = this_datum_components
            all_data.append(this_datum)
        return all_data

    def get_oscillator_strength(self):
        """Get transition dipoles and oscillator strengths.

        Returns:
            dipole (double ndarray): transition dipoles.
            oscillator_strength (double array): oscillator strengths.
        """
        assert self.exci is not None and self.X_vec is not None and self.Y_vec is not None
        assert self.mo_coeff is not None and self.mol is not None
        dipole, oscillator_strength = _get_oscillator_strength(
            multi=self.multi,
            exci=self.exci,
            X_vec=self.X_vec,
            Y_vec=self.Y_vec,
            mo_coeff=self.mo_coeff,
            nocc=self.nocc,
            mol=self.mol,
        )

        return dipole, oscillator_strength

    def _contract_multipole(self, ints, hermi=True, xy=None):
        '''ints is the integral tensor of a spin-independent operator'''
        if xy is None: xy = self.xy
        nstates = len(xy)
        pol_shape = ints.shape[:-2]
        nao = ints.shape[-1]

        if not self.multi.lower() == 's':
            return np.zeros((nstates,) + pol_shape)

        mo_coeff = self.mo_coeff[0]
        mo_occ = self.mo_occ[0]
        orbo = mo_coeff[:,mo_occ==2]
        orbv = mo_coeff[:,mo_occ==0]

        #Incompatible to old numpy version
        #ints = numpy.einsum('...pq,pi,qj->...ij', ints, orbo, orbv.conj())
        ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo, orbv.conj())
        pol = np.array([np.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
        if isinstance(xy[0][1], np.ndarray):
            if hermi:
                pol += [np.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
            else:  # anti-Hermitian
                pol -= [np.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        pol = pol.reshape((nstates,)+pol_shape)
        return pol
    


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
    mytd = TDDFT(mf = mf, nroot = 10, max_vec = 150, residue_thresh = 1.0e-6, isdf_rcond = 1e-6)
    
    # Cholesky decomposing fxc is not worth it unless you want a large number of roots
    # mytd.load_fxc_intermediates()
    # mytd.Lia_fxc = cholesky_fit_gga(mytd.rho_o, mytd.rho_v, mytd.wfxc, tol = 1e-4, max_rank = 2000)
    pyscf_td = mf.TDDFT()
    pyscf_td.singlet = True
    pyscf_td.nstates = 10
    pyscf_ref = np.sort(pyscf_td.kernel()[0])
    
    print('---values in eV---')
    print('pyscf exci:', pyscf_ref*HARTREE2EV)

    exci_new = np.sort(mytd.kernel(multi = 's')[0])
    
    print('Davidson exci:', exci_new*HARTREE2EV)
    print('Davidson mae vs PySCF:', HARTREE2EV*np.mean(np.abs(exci_new-pyscf_ref)))

    # Run full diagonalization as a sanity check
    # from pyscf.dft.libxc import xc_type
    # mytd.xctype = xc_type(mf.xc)
    # exci_full = np.sort(mytd.full_diagonalization(multi = 's')[0])

    
    # print('Full diag S1-S10:', exci_full[:10]*HARTREE2EV)
    # print('Full diag mae vs PySCF:', HARTREE2EV*np.mean(np.abs(exci_full[:10]-pyscf_ref)))

    