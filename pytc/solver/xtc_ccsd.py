import logging
import time
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.cc import rccsd
from pyscf.cc import rintermediates as imd
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

logger = logging.getLogger(__name__)

class RCCSD(rccsd.RCCSD):
    """Restricted CCSD with ISDF-XTC integrals."""
    def __init__(self, mf, xtc_obj=None, jastrow_params=None, **kwargs):
        self.gpu_max_memory = kwargs.pop('gpu_max_memory', 4000)
        max_memory = kwargs.pop('max_memory', None)
        rccsd.RCCSD.__init__(self, mf, **kwargs)
        self.xtc_obj = xtc_obj
        self.jastrow_params = jastrow_params
        
        if max_memory is not None:
            self.max_memory = max_memory
        if getattr(self, 'max_memory', None) is None:
            self.max_memory = getattr(mf, 'max_memory', 4000)
            
        self._keys = self._keys.union(['xtc_obj', 'jastrow_params', 'gpu_max_memory'])

    def ao2mo(self, mo_coeff=None):
        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        eris = _make_xtc_eris(self, mo_coeff)
        self.e_hf = self.get_e_hf(eris)
        return eris
        
    def update_amps(self, t1, t2, eris):
        return _update_amps(self, t1, t2, eris)

    def energy(self, t1=None, t2=None, eris=None):
        return _energy(self, t1, t2, eris)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=None):
        if eris is None:
            eris = self.ao2mo()
        self.e_hf = self.get_e_hf(eris)
        return super().ccsd(t1, t2, eris, mbpt2)

    def energy_tot(self, t1=None, t2=None, eris=None):
        return self.get_e_hf(eris) + self.energy(t1, t2, eris)

    def get_e_hf(self, eris=None):
        if eris is None:
             if getattr(self, 'e_hf', None) is not None:
                 return self.e_hf
             return self._scf.e_tot
        
        no = self.nocc
        fock = eris.fock
        # E_hf = 2*sum_i F_ii - 2*sum_ij (ii|jj) + sum_ij (ij|ji) + E_core
        e_hf = 2*np.einsum('ii->', fock[:no,:no])
        if hasattr(eris, 'oooo'):
            oooo = np.asarray(eris.oooo)
            e_hf -= 2*np.einsum('iijj ->', oooo)
            e_hf += np.einsum('ijji ->', oooo)
        
        e_hf += getattr(eris, 'e_core', 0)
        
        return e_hf.real

    def _finalize(self):
        if self.converged:
            lib.logger.info(self, '%s converged', self.__class__.__name__)
        else:
            lib.logger.note(self, '%s not converged', self.__class__.__name__)
        
        lib.logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        return self

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr
    
    def density_fit(self, auxbasis=None, with_df=None, n_rank_xtc=None, with_isdf_xtc=None, **kwargs):
        '''
        Update local RCCSD object to use density fitting for standard Coulomb integrals
        and/or ISDF for XTC integrals.

        Args:
            auxbasis (str): Auxiliary basis for standard Coulomb DF.
            with_df (pyscf.df.DF): Existing DF object for standard integrals.
            n_rank_xtc (int): Rank for ISDF-XTC decomposition. used if with_isdf_xtc is None.
            with_isdf_xtc (ISDFXTC): Existing ISDF-XTC object to use.
            **kwargs: Additional arguments for ISDFXTC conversion (e.g., save_path) if creating new ISDFXTC.
        
        Returns:
            A new RCCSD object with DF enabled.
        '''
        new_cc = self.copy()
        
        # 1. Setup Standard Coulomb DF
        if with_df is not None:
             new_cc.with_df = with_df
        elif getattr(self._scf, 'with_df', None):
             new_cc.with_df = self._scf.with_df.copy()
        else:
             from pyscf import df
             new_cc.with_df = df.DF(self.mol)
             new_cc.with_df.auxbasis = auxbasis or 'weigend'
        
        if auxbasis is not None and new_cc.with_df.auxbasis != auxbasis:
             new_cc.with_df = new_cc.with_df.copy()
             new_cc.with_df.auxbasis = auxbasis
             
        # 2. Setup ISDF-XTC
        from pytc.autodiff.xtc import ISDFXTC
        if with_isdf_xtc is not None:
            new_cc.xtc_obj = with_isdf_xtc
        elif n_rank_xtc is not None:
            if not isinstance(self.xtc_obj, ISDFXTC):
                 # Convert to ISDFXTC
                 new_cc.xtc_obj = ISDFXTC.from_xtc(self.xtc_obj, n_rank=n_rank_xtc, **kwargs)
                 # Ensure ISDF kernels are computed
                 if self.jastrow_params is not None:
                     new_cc.xtc_obj = new_cc.xtc_obj.isdf(self.jastrow_params)
            else:
                 logger.warning("xtc_obj is already ISDFXTC, but n_rank_xtc provided. Ignoring n_rank_xtc re-decomposition for now to avoid complexity.")
        
        return new_cc



class _ChemistsERIs(rccsd._ChemistsERIs):
    """Custom ERIs holding XTC data."""
    def __init__(self, mol=None):
        rccsd._ChemistsERIs.__init__(self, mol)
        self.xtc_obj = None
        self.jastrow_params = None
        self.max_memory = 4000
        self.gpu_max_memory = 4000
        if not hasattr(self, '_keys'):
            self._keys = set()
        self._keys = self._keys.union(['xtc_obj', 'jastrow_params', 'max_memory', 'gpu_max_memory'])

def _make_xtc_eris(cc, mo_coeff=None):
    if mo_coeff is None:
        mo_coeff = cc.mo_coeff
        
    xtc_obj = cc.xtc_obj
    jastrow_params = cc.jastrow_params
    
    # Use standard make_eris if it's not an ISDF object to avoid recomputation
    # BUT if density fitting is requested (with_df), we must use the DF path 
    # even for standard XTC objects (to use DF for standard integrals).
    with_df = getattr(cc, 'with_df', None)
    if with_df is None and getattr(cc._scf, 'with_df', None):
        with_df = cc._scf.with_df

    from pytc.autodiff.xtc import XTC, ISDFXTC
    if isinstance(xtc_obj, XTC) and not isinstance(xtc_obj, ISDFXTC) and with_df is None:
        logger.info("Using XTC.make_eris for standard XTC object")
        return xtc_obj.make_eris(cc._scf, jastrow_params)

    eris = _ChemistsERIs(cc.mol)
    eris._common_init_(cc, mo_coeff)
    eris.xtc_obj = xtc_obj
    eris.jastrow_params = jastrow_params
    eris.max_memory = cc.max_memory
    eris.gpu_max_memory = cc.gpu_max_memory
    
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    # mo_o removed as unused
    
    # 1. Fock matrix construction 
    # Start from MF Fock matrix (diagonal in MO basis if mo_coeff are mf.mo_coeff)
    # We use get_fock to ensure standard ERI part is exactly consistent with mf
    dm_std = cc._scf.make_rdm1(mo_coeff=mo_coeff, mo_occ=cc._scf.mo_occ)
    fock_std = cc._scf.get_fock(dm=dm_std)
    fock_std = reduce(np.dot, (mo_coeff.T, fock_std, mo_coeff))

    
    h1e_corr = np.asarray(xtc_obj.get_1b(jastrow_params))
    eris.e_core = np.asarray(xtc_obj.get_const(jastrow_params))
    # Corrections to Fock from TC 2-body part: (pq|ii) and (pi|iq) corrections only
    h2e_pqii_corr = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=(slice(None), slice(None), slice(0, nocc), slice(0, nocc))))
    h2e_piiq_corr = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=(slice(None), slice(0, nocc), slice(0, nocc), slice(None))))
    
    fock_corr = h1e_corr + 2 * np.einsum('pqii->pq', h2e_pqii_corr) - np.einsum('piiq->pq', h2e_piiq_corr)
    eris.fock = fock_std + fock_corr
    eris.fvo = eris.fock[nocc:, :nocc].copy()
    eris.mo_energy = np.diag(eris.fock)

    # 2. Materialize required blocks (except vvvv) using ISDF efficiently
    
    # Check for density fitting
    with_df = getattr(cc, 'with_df', None)
    if with_df is None and getattr(cc._scf, 'with_df', None):
        with_df = cc._scf.with_df

    if with_df is not None:
        # --- Density Fitting Path ---
        logger.info("Using Density Fitting for standard Coulomb integrals in XTC-CCSD")
        
        # Prepare 3-index tensors L_pq = (L|pq)
        naux = with_df.get_naoaux()
        Loo, Lov = _init_df_eris(eris, with_df, nvir, naux, nocc, nmo, mo_coeff)
        
        def get_block_df(block_str):
            logger.debug(f"    Computing block {block_str}")
            tc_part = np.asarray(xtc_obj.get_2b(jastrow_params, block_str=block_str))
            
            if block_str == 'oooo':
                std = lib.ddot(Loo.T, Loo).reshape(nocc, nocc, nocc, nocc)
            elif block_str == 'ovoo':
                std = lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc)
            elif block_str == 'ooov':
                std = lib.ddot(Loo.T, Lov).reshape(nocc, nocc, nocc, nvir)
            elif block_str == 'ovov':
                std = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir)
            elif block_str == 'ovvo':
                # (kc|al) -> (k, c, a, l)
                tmp = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir)
                std = tmp.transpose(0, 1, 3, 2)
            elif block_str == 'oovv':
                # (kl|cd). Loo (kl, L). Lvv (cd, L).
                Lvv_flat = L_vv_full.reshape(nvir*nvir, naux).T
                std = lib.ddot(Loo.T, Lvv_flat).reshape(nocc, nocc, nvir, nvir)
            elif block_str == 'vvoo':
                # (cd|kl). Lvv (cd, L). Loo (kl, L).
                Lvv_flat = L_vv_full.reshape(nvir*nvir, naux).T
                std = lib.ddot(Lvv_flat.T, Loo).reshape(nvir, nvir, nocc, nocc)
            elif block_str == 'vooo':
                # (ck|li). Lvo? Lov is (L, kc).
                tmp = lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc)
                std = tmp.transpose(1, 0, 2, 3)
            elif block_str == 'vovo':
                # (ak|cl) -> (a, k, c, l). From ovov (kacl) transpose to (a, k, c, l)
                tmp = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir)
                std = tmp.transpose(1, 0, 3, 2)
            else:
                raise NotImplementedError(f"Block {block_str} not supported in get_block_df")
                
            return std + tc_part

        # Unpack Lvv to RAM if possible (approx 5-10GB for 800 orbitals)
        L_vv_full = lib.unpack_tril(eris.vvL[:], axis=0) # (nvir, nvir, naux)
        Lov_reshaped = Lov.reshape(naux, nocc, nvir)
        
        # Create HDF5 datasets for large blocks
        eris_blocks = {
            'ovvv': (nocc, nvir, nvir, nvir),
            'vovv': (nvir, nocc, nvir, nvir),
        }
        for name, shape in eris_blocks.items():
            if name in eris.feri:
                del eris.feri[name]
            setattr(eris, name, eris.feri.create_dataset(name, shape, 'f8'))

        logger.info("Computing large blocks...")
        _compute_large_blocks(eris, eris_blocks, xtc_obj, jastrow_params, Lov_reshaped, L_vv_full, nocc, nvir, nmo)
        
        # Medium blocks (keep in memory as per user request < 3 virtuals)
        eris.oovv = get_block_df('oovv') # 2 vir (11 GB)
        eris.vvoo = get_block_df('vvoo') # 2 vir (11 GB)
        
        # Free L_vv_full after we are done with blocks needing it
        del L_vv_full
        
        eris.ovvo = get_block_df('ovvo') # 2 vir
        eris.ovov = get_block_df('ovov') # 2 vir
        
        # Small blocks
        eris.oooo = get_block_df('oooo')
        eris.ovoo = get_block_df('ovoo')
        eris.ooov = get_block_df('ooov')
        eris.vooo = get_block_df('vooo')
        
        # Handle vovo, voov if needed.
        eris.vovo = get_block_df('vovo') 
        # eris.voov = get_block_df('voov') # Unused?
        
        eris.vvvv = None
        
        del Loo, Lov, Lov_reshaped

        # Keep eris.vvL for vvvv contraction
        
        return eris

    else:
        # --- Standard Path (ao2mo) --- 
        eri_std_full = ao2mo.kernel(cc.mol, mo_coeff, compact=False, aosym='s1', intor='int2e')
        eri_std_full = eri_std_full.reshape(nmo, nmo, nmo, nmo)
        
        def get_block(block_str):
            tc_part = np.asarray(xtc_obj.get_2b(jastrow_params, block_str=block_str))
            slices = [slice(0, nocc) if c == 'o' else slice(nocc, nmo) for c in block_str]
            return eri_std_full[tuple(slices)] + tc_part
    
        eris.oooo = get_block('oooo')
        eris.ovoo = get_block('ovoo')
        eris.ooov = get_block('ooov')
        eris.vooo = get_block('vooo')
        eris.ovov = get_block('ovov')
        eris.vovo = get_block('vovo')
        eris.ovvo = get_block('ovvo')
        eris.voov = get_block('voov')
        eris.oovv = get_block('oovv')
        eris.vvoo = get_block('vvoo')
        eris.ovvv = get_block('ovvv')
        eris.vvov = get_block('vvov')
        eris.vovv = get_block('vovv')
        eris.vvvv = None
        
        return eris



def _contract_vvvv_t2(cc, t2, eris, out=None):
    """Contraction of (vv|vv) with t2. Handles both materialized and block-wise cases."""
    if eris.vvvv is not None:
        # Materialized case: Transpose to (a, c, b, d) and contract
        # Standard PySCF index for Wvvvv is (ab|cd) contracted with t2(ij|cd) gives (ij|ab)
        # Here we follow PySCF's rintermediates.cc_Wvvvv logic if materialized
        vvvv = np.asarray(eris.vvvv)
        return lib.einsum('abcd,ijcd->ijab', vvvv.transpose(0, 2, 1, 3), t2)

    if out is None:
        out = np.zeros_like(t2)

    
    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc
    xtc_obj = cc.xtc_obj
    jastrow_params = cc.jastrow_params

    # Memory-efficient block size
    mem_host = cc.max_memory * 1e6
    mem_gpu = cc.gpu_max_memory * 1e6
    # vvvv_block shape: (blksize, nvir, nvir, nvir)
    blksize = max(1, int(min(mem_host, mem_gpu) / (nvir**3 * 8)))
    blksize = min(nvir, blksize)
    
    # Pre-unpack L_vv_full if using density fitting to avoid repeated IO/unpacking
    with_df = getattr(cc, 'with_df', None)
    if with_df is None and getattr(cc._scf, 'with_df', None):
         with_df = cc._scf.with_df
    
    L_vv_full = None
    if with_df is not None:
         naux = eris.vvL.shape[1]
         # This might be large (approx 5-10GB for 800 orbitals) but necessary for performance
         L_vv_full = lib.unpack_tril(eris.vvL[:], axis=0) # (nvir, nvir, naux)

    for p0 in range(0, nvir, blksize):
        p1 = min(p0 + blksize, nvir)
        ranges = (slice(nocc + p0, nocc + p1), slice(nocc, nmo), slice(nocc, nmo), slice(nocc, nmo))
        
        vvvv_block = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=ranges))
        
        logger.debug("    Contraction block %d:%d", p0, p1)
        t0 = time.perf_counter()
        if with_df is not None:
             # L_vv_full is (nvir, nvir, naux)
             # Slicing along axis 0 corresponds to 'a' index in L_ab (L|ab) -> unpacked to (a, b, L)
             L_ab_sub = L_vv_full[p0:p1]
             
             # Contract (a_blk, d, L) with (c, d, L) on L (axis 2) -> (a_blk, d, c, d)
             # std_block shape is (blk, nvir, nvir, nvir)
             std_block = np.tensordot(L_ab_sub, L_vv_full, axes=((2), (2)))
             
             vvvv_block = vvvv_block + std_block
             
        else:
             mo_v = cc.mo_coeff[:, nocc:]
             std_block = ao2mo.general(cc.mol, (mo_v[:, p0:p1], mo_v, mo_v, mo_v), compact=False)
             vvvv_block = vvvv_block + std_block.reshape(p1-p0, nvir, nvir, nvir)
        
        # Transpose to (a, c, b, d) and contract
        vvvv_trans = vvvv_block.transpose(0, 2, 1, 3)
        out[:, :, p0:p1, :] += lib.einsum('abcd,ijcd->ijab', vvvv_trans, t2)
        logger.debug("    Block %d:%d done in %.3f s", p0, p1, time.perf_counter()-t0)
    
    if L_vv_full is not None:
        del L_vv_full
        
    return out

def _update_amps(cc, t1, t2, eris):
    """
    Restricted CCSD amplitude update with efficient Wvvvv contraction.
    
    Modified from PySCF rccsd.py and ccsd.py:
    Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    logger.debug("Starting _update_amps")
    t_start = time.perf_counter()
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc,nocc:].copy()

    # Small intermediates using PySCF logic (safe for RAM)
    t0 = time.perf_counter()
    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)
    logger.debug("    imd Foo, Fvv, Fov done in %.3f s", time.perf_counter()-t0)

    Foo[np.diag_indices(nocc)] -= mo_e_o
    # Keep an unshifted copy of Fvv for Lvv initialization
    Fvv_unshifted = Fvv.copy()
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation - terms not involving ovvv
    t1new  =-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += eris.fock[nocc:, :nocc].T
    logger.debug("    T1 initial terms done in %.3f s", time.perf_counter()-t_start)
    
    eris_ovvo = np.asarray(eris.ovvo)
    eris_oovv = np.asarray(eris.oovv)
    
    t1new += 2*np.einsum('kcai,kc->ia', eris_ovvo, t1)
    t1new +=  -np.einsum('kiac,kc->ia', eris_oovv, t1)
    
    eris_ovoo = np.asarray(eris.ovoo)
    t1new +=-2*lib.einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new +=   lib.einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new +=-2*lib.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new +=   lib.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)

    # Pre-allocate large intermediates that fit in RAM (11GB)
    # Wvoov: (a, k, i, c) -> (nvir, nocc, nocc, nvir)
    Wvoov = np.zeros((nvir, nocc, nocc, nvir))
    # Wvovo: (a, k, c, i) -> (nvir, nocc, nvir, nocc)
    Wvovo = np.zeros((nvir, nocc, nvir, nocc))
    
    # Lvv: (a, c) -> (nvir, nvir)
    # Initialize with non-ovvv terms (using unshifted Fvv)
    Lvv = Fvv_unshifted - np.einsum('kc,ka->ac', fov, t1)
    
    # tmp_a: (k, a, i, j) -> (nocc, nvir, nocc, nocc)
    tmp_a = np.zeros((nocc, nvir, nocc, nocc))
    # tmp_b: (k, b, i, j) -> (nocc, nvir, nocc, nocc)
    tmp_b = np.zeros((nocc, nvir, nocc, nocc))
    
    # Prepare Tau for tmp_a/b
    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    
    # Add non-ovvv contributions to Wvoov/Wvovo
    # Wvoov += eris.ovvo.transpose(...) etc.
    # Logic copied from rintermediates.py but using arrays
    # Wvoov (akic)
    Wvoov += eris_ovvo.transpose(2,0,3,1)
    Wvoov -= lib.einsum('kcli,la->akic', eris_ovoo, t1)
    eris_ovov = np.asarray(eris.ovov)
    Wvoov -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wvoov -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wvoov -= lib.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wvoov += lib.einsum('ldkc,ilad->akic', eris_ovov, t2)
    
    # Wvovo (akci)
    Wvovo += eris_oovv.transpose(2,0,3,1)
    Wvovo -= lib.einsum('lcki,la->akci', eris_ovoo, t1)
    Wvovo -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wvovo -= lib.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    logger.debug("    Wvoov, Wvovo basic terms done in %.3f s", time.perf_counter()-t_start)



    mem_host = cc.max_memory * 1e6
    # Handle ovvv contributions - branch based on array type
    if isinstance(eris.ovvv, np.ndarray):
        # In-memory path: compute t1new ovvv terms and tmp_a/tmp_b
        # Lvv, Wvoov, Wvovo are computed by imd.* calls later
        eris_ovvv = np.asarray(eris.ovvv)
        t1new += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
        t1new +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
        t1new += 2*lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
        t1new +=  -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
        
        # tmp_a, tmp_b for t2new (still needed, not in imd)
        tmp_a = lib.einsum('kdac,ijcd->kaij', eris_ovvv, tau)
        tmp_b = lib.einsum('kcbd,ijcd->kbij', eris_ovvv, tau)
    else:
        # HDF5 blocked loop path
        blksize = max(4, int(mem_host / (nocc*nvir*nvir*8)))
        blksize = min(nvir, blksize)
        logger.debug("    Starting ovvv loop (blksize=%d)", blksize)
        t_loop = time.perf_counter()
        for p0 in range(0, nvir, blksize):
            p1 = min(p0 + blksize, nvir)
            _process_ovvv_block(eris, t1, t2, tau, t1new, Lvv, Wvoov, Wvovo, tmp_a, tmp_b, p0, p1)
        logger.debug("    ovvv loop done in %.3f s", time.perf_counter()-t_loop)

    t2new = np.zeros_like(t2)
    
    # Handle vovv contributions - branch based on array type
    if isinstance(eris.vovv, np.ndarray):
        # In-memory path (matches working code)
        tmp2  = lib.einsum('kibc,ka->abic', eris_oovv, -t1)
        tmp2 += np.asarray(eris.vovv).transpose(0, 2, 1, 3)
        tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
        t2new = tmp + tmp.transpose(1,0,3,2)
    else:
        # HDF5 blocked loop path
        mem_host = cc.max_memory * 1e6
        blksize_t2 = max(4, int(mem_host / (nvir*nocc*nvir*8)))
        blksize_t2 = min(nvir, blksize_t2)
        logger.debug("    Starting vovv loop (blksize=%d)", blksize_t2)
        t_loop = time.perf_counter()
        for p0 in range(0, nvir, blksize_t2):
            p1 = min(p0 + blksize_t2, nvir)
            _process_vovv_block(eris, eris_oovv, t1, t2, t2new, p0, p1)
        # Symmetrize the accumulated t2new from vovv blocks
        t2new = t2new + t2new.transpose(1, 0, 3, 2)    
        logger.debug("    vovv loop done in %.3f s", time.perf_counter()-t_loop)


    tmp2  = lib.einsum('kcai,jc->akij', eris_ovvo, t1)
    tmp2 += np.asarray(eris.vooo).transpose(0, 2, 1, 3)
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)

    t2new -= tmp + tmp.transpose(1,0,3,2)
    t2new += np.asarray(eris.vovo).transpose(1,3,0,2)
    logger.debug("    t2new basic terms done in %.3f s", time.perf_counter()-t_start)

    # Add W loops
    Loo = imd.Loo(t1, t2, eris)
    Loo[np.diag_indices(nocc)] -= mo_e_o
    
    # Lvv, Wvoov, Wvovo depend on ovvv - use imd for in-memory, or already-computed for blocked
    if isinstance(eris.ovvv, np.ndarray):
        # In-memory: use imd functions (which internally handle ovvv)
        Lvv = imd.Lvv(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
    # else: Lvv, Wvoov, Wvovo were already computed incrementally in the blocked loop
    
    Lvv[np.diag_indices(nvir)] -= mo_e_v
    Woooo = imd.cc_Woooo(t1, t2, eris)

    t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
    t_vvvv = time.perf_counter()
    t2new += _contract_vvvv_t2(cc, tau, eris)
    logger.debug("    _contract_vvvv_t2 done in %.3f s", time.perf_counter()-t_vvvv)

    # Use precomputed tmp_a, tmp_b
    t2new -= lib.einsum('kb,kaij->ijab', t1, tmp_a)
    t2new -= lib.einsum('ka,kbij->ijab', t1, tmp_b)

    tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)
    tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    logger.debug("    Final t1/t2 processing done in %.3f s", time.perf_counter()-t_start)

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new

def _energy(cc, t1, t2, eris):
    """CCSD correlation energy for non-Hermitian case."""
    nocc, nvir = t1.shape
    fock = eris.fock
    # e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
    # Non-Hermitian: uses fov?
    fov = fock[:nocc, nocc:]
    e = 2*np.einsum('ia,ia', fov, t1)
    tau = np.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
    e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    return e.real


def _get_slice(dset, sl, axis=0):
    """Helper to slice HDF5 dataset or numpy array safely."""
    if hasattr(dset, 'shape'): 
        idx = [slice(None)] * dset.ndim
        idx[axis] = sl
        return np.asarray(dset[tuple(idx)])
    return dset

def _process_ovvv_block(eris, t1, t2, tau, t1new, Lvv, Wvoov, Wvovo, tmp_a, tmp_b, p0, p1):
    """Process a chunk of ovvv block for amplitude updates.
    
    ovvv indexing: (k, d, a, c) with shape (nocc, nvir, nvir, nvir).
    We slice along axis 2 ('a') to get ovvv_blk with shape (nocc, nvir, blk, nvir).
    
    Key PySCF formulas from rintermediates.py:
    - t1new: 2*einsum('kdac,ikcd->ia') - einsum('kcad,ikcd->ia')
    - Lvv:   2*einsum('kdac,kd->ac') - einsum('kcad,kd->ac')  
    - Wvoov: einsum('kcad,id->akic')
    - Wvovo: einsum('kdac,id->akci')
    - tmp_a: einsum('kdac,ijcd->kaij')  
    - tmp_b: einsum('kcbd,ijcd->kbij')
    """
    # ovvv_blk: (k, d, a_blk, c) - sliced along 'a' (axis 2)
    logger.debug("    _process_ovvv_block chunk %d:%d", p0, p1)
    t0 = time.perf_counter()
    ovvv_blk = _get_slice(eris.ovvv, slice(p0, p1), axis=2)  # (nocc, nvir, blk, nvir)
    
    # 1. Update t1new (ia)
    # PySCF: 2*einsum('kdac,ikcd->ia') - einsum('kcad,ikcd->ia')
    # First term: contract k,d (axes 0,1) with t2's i,k,c,d -> output (i, a_blk)
    t1new[:, p0:p1] += 2*lib.einsum('kdac,ikcd->ia', ovvv_blk, t2)
    # Second term: 'kcad' means axis1='c', axis3='d', contract k,d with t2's k,d 
    t1new[:, p0:p1] +=  -lib.einsum('kcad,ikcd->ia', ovvv_blk, t2)

    t1new[:, p0:p1] += 2*lib.einsum('kdac,kd,ic->ia', ovvv_blk, t1, t1)
    t1new[:, p0:p1] +=  -lib.einsum('kcad,kd,ic->ia', ovvv_blk, t1, t1)

    # 2. Update Lvv (ac)
    # PySCF: 2*einsum('kdac,kd->ac') - einsum('kcad,kd->ac')
    Lvv[p0:p1, :] += 2*lib.einsum('kdac,kd->ac', ovvv_blk, t1)
    Lvv[p0:p1, :] -=   lib.einsum('kcad,kd->ac', ovvv_blk, t1)

    # 3. Update Wvoov (akic)
    # PySCF: einsum('kcad,id->akic')
    Wvoov[p0:p1, :, :, :] += lib.einsum('kcad,id->akic', ovvv_blk, t1)

    # 4. Update Wvovo (akci)
    # PySCF: einsum('kdac,id->akci')
    Wvovo[p0:p1, :, :, :] += lib.einsum('kdac,id->akci', ovvv_blk, t1)
    
    # 5. Update tmp_a (kaij) for t2new
    # PySCF: einsum('kdac,ijcd->kaij')
    tmp_a[:, p0:p1, :, :] += lib.einsum('kdac,ijcd->kaij', ovvv_blk, tau) 
    
    # 6. Update tmp_b (kbij) for t2new
    # PySCF: einsum('kcbd,ijcd->kbij') - iterates over 'b', not 'a'
    # When we slice ovvv along 'a', we get contributions to tmp_b[:, a_blk, :, :]
    # But tmp_b is indexed by 'b'. The mapping is: 'kcbd' with ovvv (k,d,a,c) means
    # k=axis0, c=axis1, b=axis2 (which is our 'a'), d=axis3 (which is our 'c')
    # So we're computing contributions to tmp_b[:, p0:p1, :, :] 
    tmp_b[:, p0:p1, :, :] += lib.einsum('kcbd,ijcd->kbij', ovvv_blk, tau)
    logger.debug("    chunk %d:%d done in %.3f s", p0, p1, time.perf_counter()-t0)

def _process_vovv_block(eris, eris_oovv, t1, t2, t2new, p0, p1):
    """Process a chunk of vovv block for t2 updates."""
    # vovv shape is (a, i, b, c) - slice along axis 0 (a)
    logger.debug("    _process_vovv_block chunk %d:%d", p0, p1)
    t0 = time.perf_counter()
    vovv_slice = _get_slice(eris.vovv, slice(p0, p1), axis=0)  # (a_blk, i, b, c)
    
    # For tmp2 = -oovv.ka + vovv, we need the contribution for a in [p0:p1]
    # oovv is (k, i, b, c), t1 is (k, a)
    # einsum('kibc,ka->abic') with t1[:, p0:p1] gives (a_blk, b, i, c)
    t1_slice = t1[:, p0:p1]  # (k, a_blk)
    tmp2_blk = lib.einsum('kibc,ka->abic', eris_oovv, -t1_slice)
    
    # vovv_slice is (a_blk, i, b, c), transpose to (a_blk, b, i, c)
    tmp2_blk += vovv_slice.transpose(0, 2, 1, 3)
    
    # Contract: tmp2(a_blk, b, i, c) * t1(j, c) -> (a_blk, b, i, j) -> transpose to (i, j, a_blk, b)
    term = lib.einsum('abic,jc->ijab', tmp2_blk, t1)
    
    # Only add to the a_blk slice, symmetrization will be handled by the caller
    t2new[:, :, p0:p1, :] += term

    logger.debug("    chunk %d:%d done in %.3f s", p0, p1, time.perf_counter()-t0)

def _init_df_eris(eris, with_df, nvir, naux, nocc, nmo, mo_coeff):
    """Initialize DF tensors and HDF5 file."""
    if isinstance(with_df._cderi, str):
        import h5py
        eris.feri = h5py.File(with_df._cderi, 'a')
    elif isinstance(getattr(with_df, '_cderi_to_save', None), str):
        import h5py
        eris.feri = h5py.File(with_df._cderi_to_save, 'a')
    else:
        eris.feri = lib.H5TmpFile()
        
    nvir_pair = nvir * (nvir+1) // 2
    
    Loo = np.empty((naux, nocc, nocc))
    Lov = np.empty((naux, nocc, nvir))
    
    chunks = (min(nvir_pair, int(4e8/with_df.blockdim)), min(naux, with_df.blockdim))
    eris.vvL = eris.feri.create_dataset('vvL', (nvir_pair, naux), 'f8', chunks=chunks)
    
    mo = np.asarray(mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    
    for k, eri1 in enumerate(with_df.loop()):
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1-p0, nmo, nmo)
        
        Loo[p0:p1] = Lpq[:, :nocc, :nocc]
        Lov[p0:p1] = Lpq[:, :nocc, nocc:]
        
        Lvv_tril = lib.pack_tril(Lpq[:, nocc:, nocc:])
        eris.vvL[:, p0:p1] = Lvv_tril.T
        
    Loo = Loo.reshape(naux, nocc*nocc)
    Lov = Lov.reshape(naux, nocc*nvir)
    return Loo, Lov

def _compute_large_blocks(eris, eris_blocks, xtc_obj, jastrow_params, Lov_reshaped, L_vv_full, nocc, nvir, nmo):
    """Compute and write ovvv and vovv blocks to HDF5."""
    for name, shape in eris_blocks.items():
        ds = getattr(eris, name)
        
        if name == 'ovvv': # (k, c, a, d) - iterate 'a' (idx 2)
             mem_host = eris.max_memory * 1e6
             blksize = min(nvir, max(4, int(mem_host/((nocc*nvir)*8))))
             for p0, p1 in lib.prange(0, nvir, blksize):
                 L_vv_slice = L_vv_full[p0:p1] 
                 # (L, k, c) x (a, d, L) -> (k, c, a, d) tensor dot
                 # (L, k, c) x (a, d, L) -> (k, c, a, d) tensor dot
                 std_blk = np.tensordot(Lov_reshaped, L_vv_slice, axes=((0), (2)))
                 # std_blk already (k, c, a, d)
                 
                 ranges = (slice(0, nocc), slice(nocc, nmo), slice(nocc+p0, nocc+p1), slice(nocc, nmo))
                 tc_blk = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=ranges))
                 ds[:, :, p0:p1, :] = std_blk + tc_blk

        elif name == 'vovv': # (c, k, a, d) - iterate 'c' (idx 0)
             mem_host = eris.max_memory * 1e6
             blksize = min(nvir, max(4, int(mem_host/((nocc*nvir)*8))))
             for p0, p1 in lib.prange(0, nvir, blksize):
                Lov_slice = Lov_reshaped[:, :, p0:p1] # (L, k, c_blk)
                std_blk = np.tensordot(Lov_slice, L_vv_full, axes=((0), (2)))
                std_blk = std_blk.transpose(1, 0, 2, 3) 
                ranges = (slice(nocc+p0, nocc+p1), slice(0, nocc), slice(nocc, nmo), slice(nocc, nmo))
                tc_blk = np.asarray(xtc_obj.get_2b(jastrow_params, ranges=ranges))
                ds[p0:p1, :, :, :] = std_blk + tc_blk